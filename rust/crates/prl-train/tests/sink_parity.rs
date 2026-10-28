//! TrainSink parity with Python: random native episodes, failures and cancellations replayed
//! through both sinks (fixtures from `tests/fixtures/gen_sink.py`).

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use indexmap::IndexMap;
use prl_config::*;
use prl_train::algo::{build_algorithm, AlgorithmDeps};
use prl_train::staleness::rollout_staleness;
use prl_train::{Admission, PayloadReleaser, RolloutWindow, SinkConfig, TrainBatch, TrainEnvSpec, TrainSink, TrainingSample};
use prl_types::{CancelReason, DispatchFailure, ErrorInfo, GroupCancellation, Kind, Metrics, Subset};
use prl_wire::Episode;
use serde_json::Value;

fn load() -> Value {
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(&include_bytes!("fixtures/sink_cases.json.gz")[..]).read_to_end(&mut raw).unwrap();
    serde_json::from_slice(&raw).unwrap()
}

fn grpo(length_penalty: Option<LengthPenaltyConfig>, length_weighted_baseline: bool, loss_aggregation: LossAggregation) -> AlgoParams {
    AlgoParams::Grpo(GRPOAlgoConfig { length_penalty, length_weighted_baseline, loss_aggregation })
}

fn algo(name: &str) -> AlgoConfig {
    let (params, action_loss_type) = match name {
        "grpo" => (grpo(None, false, LossAggregation::Token), ActionLossType::Rl),
        "grpo_linear" => (
            grpo(
                Some(LengthPenaltyConfig::Linear(LinearLengthPenaltyConfig {
                    num_output_tokens_weight: 0.3,
                    num_input_tokens_weight: 0.1,
                    num_turns_weight: 0.1,
                })),
                true,
                LossAggregation::Token,
            ),
            ActionLossType::Rl,
        ),
        "grpo_subagent_prompt" => (
            grpo(
                Some(LengthPenaltyConfig::LinearSubagent(LinearSubagentLengthPenaltyConfig {
                    num_output_tokens_weight: 0.25,
                    num_input_tokens_weight: 0.1,
                    num_turns_weight: 0.1,
                    subagent_token_weight: 0.5,
                })),
                false,
                LossAggregation::Prompt,
            ),
            ActionLossType::Rl,
        ),
        "max_rl" => (AlgoParams::MaxRl, ActionLossType::Rl),
        "rae" => (AlgoParams::Rae(RAEAlgoConfig { decay: 0.9 }), ActionLossType::Rl),
        "hierarchical_grpo" => (
            AlgoParams::HierarchicalGrpo(HierarchicalGRPOAlgoConfig { episode_agents: vec!["solver".into()] }),
            ActionLossType::Rl,
        ),
        "debug" => (AlgoParams::Debug(DebugAlgoConfig { advantage: 0.5 }), ActionLossType::Rl),
        "sft" => (AlgoParams::Sft, ActionLossType::Ce),
        "echo" => (
            AlgoParams::Echo(EchoAlgoConfig {
                length_penalty: None,
                length_weighted_baseline: false,
                loss_aggregation: LossAggregation::Token,
                roles: EchoRolesConfig {
                    system: None,
                    user: Some(EchoRoleConfig { alpha: 0.5 }),
                    assistant: None,
                    tool: Some(EchoRoleConfig { alpha: 0.25 }),
                },
                filter: None,
            }),
            ActionLossType::Rl,
        ),
        other => panic!("unknown algo {other}"),
    };
    AlgoConfig { sampling: AlgoSamplingConfig { source: ModelReference::Policy }, action_loss_type, params }
}

struct NoScorer;

#[async_trait]
impl prl_types::Scorer for NoScorer {
    async fn score(&self, _token_ids: Vec<i32>) -> anyhow::Result<Vec<f32>> {
        unreachable!("no algorithm in these fixtures scores")
    }
}

fn deps() -> AlgorithmDeps {
    AlgorithmDeps {
        policy: Arc::new(NoScorer),
        policy_model_name: "policy".into(),
        connect_frozen: Arc::new(|_| Box::pin(async { anyhow::bail!("no frozen models in these fixtures") })),
        bridge: None,
    }
}

struct Verdicts(HashMap<String, bool>);

#[async_trait]
impl Admission for Verdicts {
    async fn admit(&mut self, group: &[Episode]) -> anyhow::Result<bool> {
        let gid = &group[0].group.as_ref().unwrap().id;
        Ok(*self.0.get(gid).unwrap_or_else(|| panic!("no verdict for {gid}")))
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

fn check_metrics(ctx: &str, got: &Metrics, want: &Value) {
    let want = want.as_object().unwrap();
    let got_keys: HashSet<&str> = got.keys().map(String::as_str).collect();
    let want_keys: HashSet<&str> = want.keys().map(String::as_str).collect();
    let missing: Vec<_> = want_keys.difference(&got_keys).collect();
    let extra: Vec<_> = got_keys.difference(&want_keys).collect();
    assert!(missing.is_empty() && extra.is_empty(), "{ctx}: missing {missing:?} extra {extra:?}");
    for (key, value) in want {
        let g = got[key.as_str()].as_f64().unwrap();
        let w = value.as_f64().unwrap();
        assert!(close(g, w, 1e-6), "{ctx}: {key}: got {g} want {w}");
    }
}

fn check_window(ctx: &str, got: Option<RolloutWindow>, want: &Value) {
    if want.is_null() {
        assert!(got.is_none(), "{ctx}: unexpected window");
        return;
    }
    let got = got.unwrap_or_else(|| panic!("{ctx}: missing window"));
    let counters = [got.tokens, got.traces as i64, got.groups as i64, got.attempts as i64, got.discarded as i64, got.stale as i64, got.errored as i64, got.cancelled as i64];
    let want_counters: Vec<i64> = want["counters"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    assert_eq!(counters.to_vec(), want_counters, "{ctx}: window counters");
    check_metrics(&format!("{ctx} window"), &got.metrics.to_dict(), &want["metrics"]);
}

fn floats(v: &Value) -> Option<Vec<f64>> {
    v.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap()).collect())
}

fn check_floats(ctx: &str, name: &str, got: Option<Vec<f64>>, want: &Value) {
    let want = floats(want);
    assert_eq!(got.is_some(), want.is_some(), "{ctx}: {name} presence");
    if let (Some(g), Some(w)) = (got, want) {
        assert_eq!(g.len(), w.len(), "{ctx}: {name} length");
        for (i, (a, b)) in g.iter().zip(&w).enumerate() {
            assert!(close(*a, *b, 1e-6), "{ctx}: {name}[{i}]: got {a} want {b}");
        }
    }
}

fn f64s(v: &Option<Vec<f32>>) -> Option<Vec<f64>> {
    v.as_ref().map(|v| v.iter().map(|&x| f64::from(x)).collect())
}

fn check_sample(ctx: &str, got: &TrainingSample, want: &Value) {
    let ints = |v: &Value| v.as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap()).collect::<Vec<_>>());
    assert_eq!(Some(got.token_ids.iter().map(|&x| x as i64).collect()), ints(&want["token_ids"]), "{ctx}: token_ids");
    let mask: Vec<bool> = want["mask"].as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect();
    assert_eq!(got.mask, mask, "{ctx}: mask");
    assert_eq!(got.env_name, want["env_name"].as_str().unwrap());
    assert_eq!(got.trace_id.as_deref(), want["trace_id"].as_str(), "{ctx}: trace_id");
    assert_eq!(got.branch_index, want["branch_index"].as_i64(), "{ctx}: branch_index");
    assert_eq!(got.staleness.as_ref().map(|s| s.iter().map(|&x| x as i64).collect()), ints(&want["staleness"]), "{ctx}: staleness");
    for (name, value) in [
        ("logprobs", Some(got.logprobs.clone())),
        ("temperatures", Some(got.temperatures.clone())),
        ("ref_logprobs", got.ref_logprobs.clone()),
        ("rl_weights", got.rl_weights.clone()),
        ("ce_weights", got.ce_weights.clone()),
        ("ref_kl_weights", got.ref_kl_weights.clone()),
        ("advantages", got.advantages.clone()),
    ] {
        check_floats(ctx, name, f64s(&value), &want[name]);
    }
    match (&got.payload, want["payload"].as_array()) {
        (None, None) => {}
        (Some(p), Some(w)) => {
            assert_eq!(p.url, w[0].as_str().unwrap());
            assert_eq!(p.id, w[1].as_str().unwrap());
            assert_eq!(p.num_tokens as i64, w[2].as_i64().unwrap());
            let fields: Vec<&str> = w[3].as_array().unwrap().iter().map(|f| f.as_str().unwrap()).collect();
            assert_eq!(p.fields, fields);
        }
        (g, w) => panic!("{ctx}: payload mismatch {g:?} vs {w:?}"),
    }
}

fn check_batch(ctx: &str, batch: &TrainBatch, step: u64, want: &Value) {
    let want_samples = want["samples"].as_array().unwrap();
    assert_eq!(batch.samples.len(), want_samples.len(), "{ctx}: sample count");
    for (i, (g, w)) in batch.samples.iter().zip(want_samples).enumerate() {
        check_sample(&format!("{ctx} sample {i}"), g, w);
    }
    let mut episodes: Vec<String> = batch.cohort.episodes.iter().map(|e| e.id().to_owned()).collect();
    episodes.sort();
    let want_episodes: Vec<String> = serde_json::from_value(want["cohort_episodes"].clone()).unwrap();
    assert_eq!(episodes, want_episodes, "{ctx}: cohort episodes");
    let mut traces: Vec<String> = batch.cohort.episodes.iter().flat_map(|e| e.traces().map(|t| t.id.clone())).collect();
    traces.sort();
    let want_traces: Vec<String> = serde_json::from_value(want["cohort_traces"].clone()).unwrap();
    assert_eq!(traces, want_traces, "{ctx}: cohort traces");

    let effective = batch.cohort.effective();
    let mut metrics = effective.to_wandb("train/agg", Subset::Effective);
    for (env, pool) in effective.by_env() {
        metrics.extend(pool.to_wandb(&format!("train/{env}"), Subset::Effective));
    }
    check_metrics(&format!("{ctx} effective"), &metrics, &want["effective_metrics"]);

    let mut staleness: Vec<Vec<i64>> = effective
        .selected_episodes()
        .iter()
        .filter_map(|e| rollout_staleness(&e.episode, e.traces(), step as i64).unwrap())
        .map(|s| vec![s.rollout, s.in_flight, s.in_queue, s.token_sum, s.token_count])
        .collect();
    staleness.sort();
    let want_staleness: Vec<Vec<i64>> = serde_json::from_value(want["staleness"].clone()).unwrap();
    assert_eq!(staleness, want_staleness, "{ctx}: staleness");

    let advantages: HashMap<String, Option<f64>> = effective
        .vf_episodes()
        .iter()
        .flat_map(|e| e.traces.iter())
        .map(|t| (t.id.clone(), t.info.get("advantage").and_then(|v| v.as_f64())))
        .collect();
    let want_adv = want["advantages"].as_object().unwrap();
    assert_eq!(advantages.len(), want_adv.len(), "{ctx}: advantage traces");
    for (id, w) in want_adv {
        let g = advantages[id];
        match (g, w.as_f64()) {
            (None, None) => {}
            (Some(a), Some(b)) => assert!(close(a, b, 1e-5), "{ctx}: advantage {id}: {a} vs {b}"),
            other => panic!("{ctx}: advantage {id}: {other:?}"),
        }
    }
    let means = effective.with_metrics(|m| {
        vec![m.reward().mean(), m.num_turns().mean(), m.num_branches().mean(), m.is_truncated().mean(), m.is_timeout().mean(), m.has_error().mean()]
    });
    check_floats(ctx, "means", Some(means), &want["means"]);
}

async fn run_scenario(scenario: &Value) {
    let seed = scenario["seed"].as_i64().unwrap();
    let cfg = SinkConfig {
        batch_size: scenario["batch_size"].as_u64().unwrap() as usize,
        max_staleness: scenario["max_staleness"].as_i64().unwrap(),
        constant_trainer_batch_size: scenario["constant"].as_bool().unwrap(),
    };
    let mut envs = IndexMap::new();
    for (name, spec) in scenario["envs"].as_object().unwrap() {
        envs.insert(
            name.clone(),
            TrainEnvSpec {
                group_size: spec["group_size"].as_u64().unwrap() as usize,
                temperature: spec["temperature"].as_f64().unwrap() as f32,
                requires_sampling_masks: false,
                requires_top_logprobs: false,
                algorithm: build_algorithm(&algo(spec["algo"].as_str().unwrap()), &deps()).unwrap(),
            },
        );
    }
    let released = Arc::new(Mutex::new(Vec::<String>::new()));
    let recorder = released.clone();
    let mut sink = TrainSink::new(cfg, envs, PayloadReleaser::new(move |urls| recorder.lock().unwrap().extend(urls)));
    let mut admission = Verdicts(serde_json::from_value(scenario["verdicts"].clone()).unwrap());
    let mut step = 1u64;
    let results = scenario["results"].as_array().unwrap();
    for (index, (event, want)) in scenario["events"].as_array().unwrap().iter().zip(results).enumerate() {
        let ctx = format!("seed {seed} event {index} ({})", event["type"].as_str().unwrap());
        released.lock().unwrap().clear();
        let mut batch = match event["type"].as_str().unwrap() {
            "episode" => {
                let episode = Episode::from_record_json(&serde_json::to_vec(&event["record"]).unwrap()).unwrap();
                sink.add(episode, step, &mut admission).await.unwrap()
            }
            "fail" => {
                let failure = DispatchFailure {
                    kind: Kind::Train,
                    env_name: event["env_name"].as_str().unwrap().into(),
                    group_id: event["group_id"].as_str().unwrap().into(),
                    step,
                    policy_version: 0,
                    task_type: "Task".into(),
                    task_key: "k".into(),
                    task_hash: "h".into(),
                    error: ErrorInfo { type_: event["error_type"].as_str().unwrap().into(), message: "boom".into(), traceback: None, status_code: None },
                };
                sink.fail(failure, step, &mut admission).await.unwrap()
            }
            _ => {
                let reason = match event["reason"].as_str().unwrap() {
                    "shutdown" => CancelReason::Shutdown,
                    _ => CancelReason::Overload,
                };
                let cancellation = GroupCancellation {
                    kind: Kind::Train,
                    env_name: event["env_name"].as_str().unwrap().into(),
                    group_id: event["group_id"].as_str().unwrap().into(),
                    step,
                    count: event["count"].as_u64().unwrap() as usize,
                    reason,
                };
                sink.cancel(cancellation, step, &mut admission).await.unwrap()
            }
        };
        let want_batches = want["batches"].as_array().unwrap();
        let mut seen = 0;
        while let Some(b) = batch {
            let wb = want_batches.get(seen).unwrap_or_else(|| panic!("{ctx}: unexpected batch {seen}"));
            assert_eq!(step, wb["step"].as_u64().unwrap(), "{ctx}: batch step");
            check_window(&format!("{ctx} batch {seen}"), sink.take_rollout_window(), &wb["window"]);
            check_batch(&format!("{ctx} batch {seen}"), &b, step, &wb["batch"]);
            seen += 1;
            step += 1;
            batch = sink.take_batch(step, ).await.unwrap();
        }
        assert_eq!(seen, want_batches.len(), "{ctx}: batch count");
        if event.get("advance_after").and_then(Value::as_bool).unwrap_or(false) {
            step += 1;
        }
        let mut got_released = released.lock().unwrap().clone();
        got_released.sort();
        let want_released: Vec<String> = serde_json::from_value(want["released"].clone()).unwrap();
        assert_eq!(got_released, want_released, "{ctx}: released payloads");
        let (pending, size) = sink.batch_progress();
        assert_eq!(vec![pending as u64, size as u64], serde_json::from_value::<Vec<u64>>(want["progress"].clone()).unwrap(), "{ctx}: batch progress");
        assert_eq!(sink.buffered_count() as u64, want["buffered"].as_u64().unwrap(), "{ctx}: buffered");
    }
    released.lock().unwrap().clear();
    sink.discard_queued();
    let mut got_released = released.lock().unwrap().clone();
    got_released.sort();
    let want_released: Vec<String> = serde_json::from_value(scenario["final"]["released"].clone()).unwrap();
    assert_eq!(got_released, want_released, "seed {seed}: final released");
    check_window(&format!("seed {seed} final"), sink.take_rollout_window(), &scenario["final"]["window"]);
}

#[tokio::test]
async fn sink_matches_python() {
    let fixture = load();
    for scenario in fixture["scenarios"].as_array().unwrap() {
        run_scenario(scenario).await;
    }
}

#[test]
fn dispatch_failure_metrics_match_python() {
    let fixture = load();
    let failures: Vec<DispatchFailure> = ["B", "A", "B"]
        .iter()
        .map(|t| DispatchFailure {
            kind: Kind::Eval,
            env_name: "e".into(),
            group_id: "g".into(),
            step: 1,
            policy_version: 0,
            task_type: "T".into(),
            task_key: "k".into(),
            task_hash: "h".into(),
            error: ErrorInfo { type_: (*t).into(), message: "m".into(), traceback: None, status_code: None },
        })
        .collect();
    let got = prl_train::dispatch_failure_metrics(&failures, "eval/e/all", 7);
    check_metrics("dispatch_failure_metrics", &got, &fixture["dispatch_failure_metrics"]);
    let order: Vec<&str> = got.keys().map(String::as_str).collect();
    assert_eq!(order, ["eval/e/all/dispatch_failure/mean", "eval/e/all/dispatch_failure/A", "eval/e/all/dispatch_failure/B"]);
}
