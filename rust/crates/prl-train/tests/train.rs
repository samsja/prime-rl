//! Deterministic tests ported from `tests/unit/orchestrator/test_{batch,train_sink,algorithms,advantage,metrics,rollout_staleness}.py`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use indexmap::IndexMap;
use prl_config::*;
use prl_train::algo::{build_algorithm, stamp_loss_routing, AlgorithmDeps};
use prl_train::packer::{prepare_batch, prepare_sample};
use prl_train::sink::{has_loss, prune_zero_advantages};
use prl_train::staleness::rollout_staleness;
use prl_train::*;
use prl_types::{CancelReason, DispatchFailure, ErrorInfo, GroupCancellation, Kind, Scorer, Subset};
use prl_wire::*;

// ---------------------------------------------------------------- builders

fn node(parent: Option<usize>, role: &str, tokens: &[i32], sampled: usize) -> MessageNode {
    let mut message = Map::new();
    message.insert("role".into(), role.into());
    let n = tokens.len();
    MessageNode {
        parent,
        message: Value::Map(message),
        sampled: sampled > 0,
        token_ids: tokens.to_vec(),
        mask: (0..n).map(|i| i >= n - sampled).collect(),
        logprobs: vec![-0.5; sampled],
        ..Default::default()
    }
}

fn trace(id: &str, advantage: Option<f32>, reward: f64) -> Trace {
    let mut answer = node(Some(1), "assistant", &[3, 4], 2);
    answer.advantages = advantage.map(|a| vec![a, a]);
    Trace {
        version: 1,
        id: id.into(),
        verifiers: Value::Nil,
        task: TraceTask { type_: "Task".into(), data: Arc::new(Map::new()), key: None, hash: None },
        agent: AgentInfo { config: Arc::new(Value::Nil), runtime: Value::Nil, name: "agent".into(), trainable: true },
        tools: vec![],
        nodes: vec![node(None, "user", &[1], 0), node(Some(0), "tool", &[2], 0), answer],
        calls: vec![],
        mm_token_type_id_map: IndexMap::new(),
        request_rewrites: vec![],
        response_rewrites: vec![],
        rewards: IndexMap::from([("correct".to_string(), Some(Reward { score: reward, weight: 1.0 }))]),
        metrics: IndexMap::new(),
        info: Map::new(),
        root_reply: None,
        extra_usage: vec![],
        is_completed: true,
        ok: true,
        stop_condition: None,
        is_timeout: false,
        errors: vec![],
        dropped: None,
        timing: Timing::default(),
    }
}

fn episode(group: &str, traces: Vec<Trace>, version: i64) -> Episode {
    Episode {
        id: format!("ep-{group}-{}", traces.first().map_or("", |t| t.id.as_str())),
        env: EnvInfo { id: "test".into(), name: Some("test".into()) },
        task: traces.first().map_or(TraceTask { type_: "Task".into(), data: Arc::new(Map::new()), key: None, hash: None }, |t| t.task.clone()),
        group: Some(GroupInfo { id: group.into() }),
        run: Some(RunInfo::Train(TrainRunInfo {
            id: "run".into(),
            name: None,
            work: WorkInfo::Train { step: 1, policy: Some(PolicySpan::new(version, version).unwrap()) },
        })),
        ok: true,
        is_timeout: false,
        errors: vec![],
        traces,
    }
}

struct FixedScorer;

#[async_trait]
impl Scorer for FixedScorer {
    async fn score(&self, token_ids: Vec<i32>) -> anyhow::Result<Vec<f32>> {
        Ok(token_ids.iter().map(|&t| -(t as f32) / 8.0).collect())
    }
}

fn deps() -> AlgorithmDeps {
    AlgorithmDeps {
        policy: Arc::new(FixedScorer),
        policy_model_name: "policy".into(),
        connect_frozen: Arc::new(|_| Box::pin(async { Ok(Arc::new(FixedScorer) as Arc<dyn Scorer>) })),
        bridge: None,
    }
}

fn algo(params: AlgoParams, action_loss_type: ActionLossType) -> Box<dyn prl_train::algo::Algorithm> {
    let config = AlgoConfig { sampling: AlgoSamplingConfig { source: ModelReference::Policy }, action_loss_type, params };
    build_algorithm(&config, &deps()).unwrap()
}

struct AdmitAll;

#[async_trait]
impl Admission for AdmitAll {
    async fn admit(&mut self, _group: &[Episode]) -> anyhow::Result<bool> {
        Ok(true)
    }
}

/// A sink over one env (group size 1) whose algorithm keeps the advantages already on the traces.
fn sink(batch_size: usize, constant: bool, released: Arc<Mutex<Vec<String>>>) -> TrainSink {
    let env = TrainEnvSpec {
        group_size: 1,
        temperature: 1.0,
        requires_sampling_masks: false,
        requires_top_logprobs: false,
        algorithm: algo(AlgoParams::Sft, ActionLossType::Rl),
    };
    let cfg = SinkConfig { batch_size, max_staleness: 1, constant_trainer_batch_size: constant };
    TrainSink::new(cfg, IndexMap::from([("test".to_string(), env)]), PayloadReleaser::new(move |urls| released.lock().unwrap().extend(urls)))
}

fn metric(window: &RolloutWindow, key: &str) -> f64 {
    window.metrics.to_dict()[key].as_f64().unwrap()
}

// ---------------------------------------------------------------- transport

#[test]
fn micro_batch_encoding_is_byte_identical_to_msgspec() {
    // `msgspec.msgpack.encode([MicroBatch(...)], enc_hook=encode_array)` of the same batch.
    const PYTHON: &str = "91dc0017c40c01000000020000002c010000c403000101c40c000000000000003f0000003fc40c00000000000080be000000bfc40c000000000100000000000000920201c40c0000803f0000803f0000003f92a161a162920201c40c000080bf000080bf000080bfc0919193a1750001c40c010000000000000000000000c40c0000803f0000803f00000000c40c00000000000000000000803fc0c092a174a09200ffc001c406ffff000002009296aa687474703a2f2f683a31a3742d300291ae726f757465645f65787065727473920204a575696e7438c0";
    let mb = MicroBatch {
        input_ids: vec![1, 2, 300],
        loss_mask: vec![false, true, true],
        advantages: vec![0.0, 0.5, 0.5],
        inference_logprobs: vec![0.0, -0.25, -0.5],
        position_ids: vec![0, 1, 0],
        sequence_lengths: vec![2, 1],
        temperatures: vec![1.0, 1.0, 0.5],
        env_names: vec!["a".into(), "b".into()],
        seq_lens: vec![2, 1],
        ref_logprobs: Some(vec![-1.0; 3]),
        mm_refs: Some(MMRefs { images: vec![MMImageRef { url: "u".into(), offset: 0, length: 1 }] }),
        mm_token_type_ids: Some(vec![1, 0, 0]),
        rl_weights: Some(vec![1.0, 1.0, 0.0]),
        ce_weights: Some(vec![0.0, 0.0, 1.0]),
        ref_kl_weights: None,
        trace_ids: Some(vec!["t".into(), String::new()]),
        branch_indices: Some(vec![0, -1]),
        padding: 1,
        staleness: Some(vec![-1, 0, 2]),
        payloads: Some(vec![
            Some(PayloadSlice {
                url: "http://h:1".into(),
                id: "t-0".into(),
                num_tokens: 2,
                fields: vec!["routed_experts".into()],
                routed_shape: Some(vec![2, 4]),
                routed_dtype: Some("uint8".into()),
            }),
            None,
        ]),
    };
    let hex: String = encode_micro_batches(&[mb]).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, PYTHON);
}

#[tokio::test]
async fn zmq_sender_binds_a_hostname() {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let transport = TransportConfig::Zmq(ZMQTransportConfig { host: "localhost".into(), port });
    BatchSender::new("/tmp".into(), 1, 0, &transport).await.unwrap();
}

// ---------------------------------------------------------------- packer

fn sized(length: usize) -> TrainingSample {
    TrainingSample {
        token_ids: vec![1; length],
        mask: (0..length).map(|i| i == length - 1).collect(),
        logprobs: vec![0.0; length],
        temperatures: vec![1.0; length],
        env_name: "env".into(),
        advantages: Some((0..length).map(|i| if i == length - 1 { 1.0 } else { 0.0 }).collect()),
        ..Default::default()
    }
}

fn tokens_table() -> FlopTable {
    FlopTable::new((0..256).collect())
}

fn flop_table() -> FlopTable {
    FlopTable::new((0..256).map(|n| 64 * n + n * n).collect())
}

#[test]
fn packing_balances_tokens_across_workers() {
    let grid = prepare_batch([100, 90, 80, 70].map(sized).to_vec(), 100, 2, &tokens_table(), 1).unwrap();
    let sums: Vec<usize> = grid.iter().map(|w| w.iter().map(|mb| mb.input_ids.len()).sum()).collect();
    assert_eq!(sums, [170, 170]);
}

#[test]
fn flop_aware_balancing_pairs_long_and_short_sequences() {
    let grid = prepare_batch([32, 32, 16, 16, 16, 16].map(sized).to_vec(), 32, 2, &flop_table(), 1).unwrap();
    for worker in grid {
        let mut lens: Vec<Vec<usize>> = worker.iter().map(|mb| mb.sequence_lengths.clone()).collect();
        lens.sort();
        assert_eq!(lens, [vec![16, 16], vec![32]]);
    }
}

#[test]
fn packed_streams_backfill_and_pad() {
    let mut longer = sized(6);
    longer.ce_weights = Some(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    longer.rl_weights = Some(vec![0.0; 6]);
    longer.staleness = Some(vec![-1, -1, 0, 1, 3, 3]);
    let shorter = sized(4);
    let grid = prepare_batch(vec![longer, shorter], 16, 1, &tokens_table(), 4).unwrap();
    let [mb] = grid[0].as_slice() else { panic!("one micro batch") };
    assert_eq!(mb.sequence_lengths, [6, 6]);
    assert_eq!(mb.padding, 2);
    assert_eq!(mb.rl_weights.as_deref(), Some(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0][..]));
    assert_eq!(mb.ce_weights.as_deref(), Some(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][..]));
    assert_eq!(mb.staleness.as_deref(), Some(&[-1, -1, 0, 1, 3, 3, -1, -1, -1, -1, -1, -1][..]));
    assert_eq!(mb.position_ids, [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1]);
    assert_eq!(&mb.input_ids[10..], &[1, 1]);
}

#[test]
fn dummy_micro_batches_carry_no_loss_and_no_identity() {
    let samples = (0..3)
        .map(|i| TrainingSample { trace_id: Some(format!("t{i}")), branch_index: Some(0), ..sized(4) })
        .collect();
    let grid = prepare_batch(samples, 4, 2, &tokens_table(), 1).unwrap();
    assert!(grid.iter().all(|w| w.len() == 2));
    let dummies: Vec<&MicroBatch> = grid.iter().flatten().filter(|mb| !mb.loss_mask.iter().any(|&m| m)).collect();
    assert_eq!(dummies.len(), 1);
    assert!(dummies[0].trace_ids.is_none() && dummies[0].branch_indices.is_none());
    assert!(dummies[0].advantages.iter().all(|&a| a == 0.0));
}

#[test]
fn truncation_never_splits_an_image() {
    let sample = TrainingSample {
        mm_token_type_ids: Some(vec![0, 1, 1, 0, 1, 1, 0]),
        mm_refs: Some(MMRefs {
            images: vec![
                MMImageRef { url: "first".into(), offset: 1, length: 2 },
                MMImageRef { url: "second".into(), offset: 4, length: 2 },
            ],
        }),
        ..sized(7)
    };
    let mb = prepare_sample(sample, 5).unwrap();
    assert_eq!(mb.input_ids.len(), 4);
    assert_eq!(mb.mm_refs.unwrap().images.iter().map(|i| i.url.as_str()).collect::<Vec<_>>(), ["first"]);
}

// ---------------------------------------------------------------- algorithms

#[test]
fn stamp_loss_routing_moves_action_tokens() {
    let mut s = TrainingSample { ce_weights: Some(vec![0.25, 0.0, 0.0, 0.0]), ..sized(4) };
    stamp_loss_routing(&mut s.clone(), ActionLossType::Rl);
    stamp_loss_routing(&mut s, ActionLossType::Ce);
    assert_eq!(s.rl_weights, Some(vec![0.0; 4]));
    assert_eq!(s.ce_weights, Some(vec![0.25, 0.0, 0.0, 1.0]));
    let mut s = sized(4);
    stamp_loss_routing(&mut s, ActionLossType::RefKl);
    assert_eq!(s.ref_kl_weights, Some(vec![0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn zero_advantage_pruning() {
    let mut s = sized(4);
    assert!(prune_zero_advantages(&mut s).unwrap());
    assert!(s.rl_weights.is_none());
    s.advantages = Some(vec![0.0; 4]);
    assert!(!prune_zero_advantages(&mut s).unwrap());
    let mut s = TrainingSample { advantages: Some(vec![0.0; 4]), ce_weights: Some(vec![0.0, 0.5, 0.0, 0.0]), ..sized(4) };
    assert!(prune_zero_advantages(&mut s).unwrap() && has_loss(&s));
}

fn advantages(episodes: &[Episode]) -> Vec<f32> {
    episodes.iter().map(|e| e.traces[0].nodes[2].advantages.as_ref().unwrap()[0]).collect()
}

#[tokio::test]
async fn grpo_and_max_rl_center_on_the_group_mean() {
    let group = || (0..4).map(|i| episode("g", vec![trace(&format!("t{i}"), None, [1.0, 0.0, 0.0, 1.0][i])], 0)).collect::<Vec<_>>();
    let grpo = AlgoParams::Grpo(GRPOAlgoConfig { length_penalty: None, length_weighted_baseline: false, loss_aggregation: LossAggregation::Token });
    let mut episodes = group();
    algo(grpo.clone(), ActionLossType::Rl).score_group(&mut episodes).await.unwrap();
    assert_eq!(advantages(&episodes), [0.5, -0.5, -0.5, 0.5]);
    let mut episodes = group();
    algo(AlgoParams::MaxRl, ActionLossType::Rl).score_group(&mut episodes).await.unwrap();
    assert_eq!(advantages(&episodes), [1.0, -1.0, -1.0, 1.0]);
    let mut single = vec![episode("g", vec![trace("t", None, 1.0)], 0)];
    algo(grpo, ActionLossType::Rl).score_group(&mut single).await.unwrap();
    assert_eq!(advantages(&single), [0.0]);
}

#[tokio::test]
async fn opd_scores_sampled_tokens_under_the_teacher() {
    let teacher = FrozenModelConfig {
        client: ClientConfig {
            base_url: "http://teacher/v1".into(),
            api_key_var: "KEY".into(),
            headers: IndexMap::new(),
            headers_from_env: IndexMap::new(),
            timeout: ClientTimeoutConfig { connect: None, read: None, write: None, pool: None },
            wait_for_ready_timeout: 0,
            skip_model_check: true,
            admin_base_url: None,
        },
        name: "teacher".into(),
    };
    let mut opd = algo(AlgoParams::Opd(OPDAlgoConfig { teacher }), ActionLossType::RefKl);
    opd.setup().await.unwrap();
    let mut ep = episode("g", vec![trace("t", None, 1.0)], 0);
    opd.finalize_episode(&mut ep).await.unwrap();
    // Branch tokens [1, 2, 3, 4]; the assistant node's sampled tokens are 3 and 4.
    assert_eq!(ep.traces[0].nodes[2].reference_logprobs, Some(vec![-0.375, -0.5]));
    assert!(ep.traces[0].nodes[0].reference_logprobs.is_none());
}

// ---------------------------------------------------------------- staleness

#[test]
fn rollout_staleness_is_max_token_staleness() {
    let mut t = trace("t", None, 1.0);
    t.nodes = vec![node(None, "user", &[0, 0], 1), node(Some(0), "assistant", &[0; 13], 12)];
    t.nodes[1].weight_versions = Some(vec![
        WeightVersionSpan { version: "default".into(), start: 0, end: 4 },
        WeightVersionSpan { version: "7".into(), start: 4, end: 10 },
        WeightVersionSpan { version: "9".into(), start: 10, end: 12 },
    ]);
    let mut ep = episode("g", vec![t], 6);
    ep.run = Some(RunInfo::Train(TrainRunInfo { id: "run".into(), name: None, work: WorkInfo::Train { step: 7, policy: Some(PolicySpan::new(6, 9).unwrap()) } }));
    // Batch 11 trains on v10. Unlabelled tokens take the dispatch version v6.
    let s = rollout_staleness(&ep, ep.traces.iter(), 11).unwrap().unwrap();
    assert_eq!((s.rollout, s.in_flight, s.in_queue, s.token_sum, s.token_count), (4, 3, 1, 40, 13));
}

// ---------------------------------------------------------------- sink

#[tokio::test]
async fn partially_stale_trace_ships_fresh_tokens() {
    let mut s = sink(1, true, Arc::default());
    let mut t = trace("t", Some(1.0), 1.0);
    t.nodes[2].weight_versions = Some(vec![
        WeightVersionSpan { version: "default".into(), start: 0, end: 1 },
        WeightVersionSpan { version: "3".into(), start: 1, end: 2 },
    ]);
    let batch = s.add(episode("g", vec![t], 1), 4, &mut AdmitAll).await.unwrap().unwrap();
    assert_eq!(batch.samples[0].mask, [false, false, false, true]);
    // Batch 4 trains on v3; the unlabelled token took the dispatch version v1.
    assert_eq!(batch.samples[0].staleness, Some(vec![-1, -1, 2, 0]));
    assert_eq!(metric(&s.take_rollout_window().unwrap(), "staleness/token/masked"), 0.5);
}

#[tokio::test]
async fn discarded_siblings_are_counted_and_their_payloads_freed() {
    let released = Arc::new(Mutex::new(Vec::new()));
    let mut s = sink(100, true, released.clone());
    let zero = {
        let mut t = trace("zero", Some(0.0), 0.0);
        let mut payload = Map::new();
        payload.insert("url".into(), "http://env:9".into());
        let mut branch = Map::new();
        branch.insert("id".into(), "zero-0".into());
        branch.insert("fields".into(), Value::Array(vec!["sampling_mask".into()]));
        payload.insert("branches".into(), Value::Map(Map::from_iter([("0".to_string(), Value::Map(branch))])));
        t.info.insert("prime_rl_payload".into(), Value::Map(payload));
        t
    };
    let failed = Trace { ok: false, errors: vec![Error { type_: "ProviderError".into(), message: "x".into(), status_code: None, traceback: None }], ..trace("failed", Some(1.0), 1.0) };
    let untrainable = { let mut t = trace("judge", Some(1.0), 1.0); t.agent.trainable = false; t };
    let ep = episode("g", vec![trace("kept", Some(1.0), 1.0), zero, failed, untrainable], 0);
    assert!(s.add(ep, 1, &mut AdmitAll).await.unwrap().is_none());
    assert_eq!(s.batch_progress(), (1, 100));
    assert_eq!(*released.lock().unwrap(), ["http://env:9/zero-0"]);
    let window = s.take_rollout_window().unwrap();
    assert_eq!((window.traces, window.attempts), (4, 1));
    assert_eq!(metric(&window, "rollout/traces/discarded"), 3.0);
    assert_eq!(metric(&window, "train/agg/all/agent/has_error/mean"), 0.25);
}

#[tokio::test]
async fn stale_queue_drops_traces_without_recounting_arrivals() {
    let mut s = sink(4, true, Arc::default());
    s.add(episode("g", vec![trace("t", Some(1.0), 1.0)], 0), 1, &mut AdmitAll).await.unwrap();
    assert_eq!(s.take_rollout_window().unwrap().attempts, 1);
    assert!(s.take_batch(4).await.unwrap().is_none());
    let window = s.take_rollout_window().unwrap();
    assert_eq!((window.attempts, window.traces), (0, 0));
    assert_eq!(metric(&window, "staleness/rollout/dropped"), 1.0);
    assert_eq!(s.batch_progress().0, 0);
}

#[tokio::test]
async fn deferred_pruning_keeps_episode_boundaries() {
    let mut s = sink(2, false, Arc::default());
    let traces = vec![trace("zero", Some(0.0), 0.0), trace("kept", Some(1.0), 1.0), trace("sibling", Some(1.0), 1.0), trace("another", Some(1.0), 1.0)];
    let batch = s.add(episode("g", traces, 0), 1, &mut AdmitAll).await.unwrap().unwrap();
    assert_eq!(batch.samples.iter().map(|s| s.trace_id.clone().unwrap()).collect::<Vec<_>>(), ["kept"]);
    assert_eq!((batch.cohort.len(), batch.cohort.num_traces()), (1, 1));
    let batch = s.take_batch(2).await.unwrap().unwrap();
    assert_eq!(batch.samples.len(), 2);
    assert_eq!((batch.cohort.len(), batch.cohort.num_traces()), (1, 2));
    assert_eq!(metric(&s.take_rollout_window().unwrap(), "rollout/queued/pruned"), 1.0);
}

#[tokio::test]
async fn failures_and_cancellations_accumulate() {
    let mut s = sink(4, true, Arc::default());
    for i in 0..8 {
        let failure = DispatchFailure {
            kind: Kind::Train,
            env_name: "test".into(),
            group_id: format!("f{i}"),
            step: 1,
            policy_version: 0,
            task_type: "Task".into(),
            task_key: "k".into(),
            task_hash: "h".into(),
            error: ErrorInfo { type_: "TransportError".into(), message: "boom".into(), traceback: None, status_code: None },
        };
        assert!(s.fail(failure, 1, &mut AdmitAll).await.unwrap().is_none());
    }
    for i in 0..4 {
        let cancellation = GroupCancellation { kind: Kind::Train, env_name: "test".into(), group_id: format!("c{i}"), step: 1, count: 1, reason: CancelReason::Overload };
        s.cancel(cancellation, 1, &mut AdmitAll).await.unwrap();
    }
    let window = s.take_rollout_window().unwrap();
    assert_eq!((window.attempts, window.discarded, window.cancelled, window.errored), (12, 12, 4, 8));
    assert_eq!(metric(&window, "train/agg/all/dispatch_failure/mean"), 1.0);
    assert_eq!(metric(&window, "train/agg/all/dispatch_failure/TransportError"), 8.0);
}

// ---------------------------------------------------------------- metrics

#[test]
fn stat_and_dispatch_failure_metrics() {
    let s = Stat::new(vec![1.0, 2.0, 3.0]);
    assert_eq!((s.mean(), s.max(), s.min()), (2.0, 3.0, 1.0));
    assert!((s.percentile(10.0) - 1.2).abs() < 1e-12 && (s.percentile(90.0) - 2.8).abs() < 1e-12);
    assert_eq!(Stat::default().percentile(90.0), 0.0);
    let failure = |ty: &str| DispatchFailure {
        kind: Kind::Eval,
        env_name: "e".into(),
        group_id: "g".into(),
        step: 1,
        policy_version: 0,
        task_type: "T".into(),
        task_key: "k".into(),
        task_hash: "h".into(),
        error: ErrorInfo { type_: ty.into(), message: "m".into(), traceback: None, status_code: None },
    };
    let got = dispatch_failure_metrics(&[failure("B"), failure("A"), failure("B")], "eval/e/all", 4);
    let got: Vec<(&str, f64)> = got.iter().map(|(k, v)| (k.as_str(), v.as_f64().unwrap())).collect();
    assert_eq!(got, [("eval/e/all/dispatch_failure/mean", 0.75), ("eval/e/all/dispatch_failure/A", 1.0), ("eval/e/all/dispatch_failure/B", 2.0)]);
}

#[test]
fn eval_metrics_average_and_pass_at_k() {
    let episodes = (0..4).map(|i| Arc::new(episode(&format!("g{}", i / 2), vec![trace(&format!("t{i}"), None, [1.0, 0.0, 0.0, 0.0][i])], 0))).collect();
    let metrics = EvalEpisodes::new(episodes, 2).effective().to_wandb("eval/e", Subset::Effective);
    let get = |k: &str| metrics[k].as_f64().unwrap();
    assert_eq!(get("eval/e/effective/agent/avg@2"), 0.25);
    assert_eq!(get("eval/e/effective/agent/pass@1"), 0.25);
    assert_eq!(get("eval/e/effective/agent/pass@2"), 0.5);
    assert_eq!(get("eval/e/effective/agent/pass^2"), 0.0);
}
