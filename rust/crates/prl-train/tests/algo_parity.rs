//! Algorithms that call out (opd teacher, opsd demo-conditioned policy, echo user filter) match
//! Python against deterministic fakes (fixtures from `tests/fixtures/gen_algos.py`), plus unit
//! tests ported from `tests/unit/orchestrator/test_algorithms.py` / `test_train_sink.py`.

use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;

use async_trait::async_trait;
use indexmap::IndexMap;
use prl_config::*;
use prl_train::algo::{build_algorithm, format_template, stamp_loss_routing, AlgorithmDeps};
use prl_train::sink::{has_loss, prune_zero_advantages};
use prl_train::TrainingSample;
use prl_types::{Bridge, Kind, LoadedTasks, Scorer, TaskRecord};
use prl_wire::Episode;
use serde_json::Value;

struct FakeScorer;

#[async_trait]
impl Scorer for FakeScorer {
    async fn score(&self, token_ids: Vec<i32>) -> anyhow::Result<Vec<f32>> {
        Ok(token_ids.iter().map(|t| -((t % 97) as f32) / 64.0).collect())
    }
}

struct FakeBridge {
    masks: HashMap<String, Vec<Vec<bool>>>,
}

#[async_trait]
impl Bridge for FakeBridge {
    async fn load_tasks(&self, _: Kind, _: &str) -> anyhow::Result<LoadedTasks> {
        unimplemented!()
    }
    async fn next_tasks(&self, _: Kind, _: &str, _: usize) -> anyhow::Result<Vec<TaskRecord>> {
        unimplemented!()
    }
    async fn flop_cost_table(&self, _: usize) -> anyhow::Result<Vec<i128>> {
        unimplemented!()
    }
    async fn render_ids(&self, _model: &str, _renderer: Value, system_content: &str) -> anyhow::Result<Vec<i32>> {
        Ok(vec![9000 + (system_content.chars().count() % 7) as i32, 9001, 9002])
    }
    async fn echo_filter(&self, import_path: &str, _kwargs: Value, trace: Value) -> anyhow::Result<Vec<Vec<bool>>> {
        assert_eq!(import_path, "filters.keep_even");
        Ok(self.masks[trace["id"].as_str().unwrap()].clone())
    }
}

fn client() -> ClientConfig {
    ClientConfig {
        base_url: "http://teacher:1/v1".into(),
        api_key_var: "KEY".into(),
        headers: IndexMap::new(),
        headers_from_env: IndexMap::new(),
        timeout: ClientTimeoutConfig { connect: None, read: None, write: None, pool: None },
        wait_for_ready_timeout: 0,
        skip_model_check: true,
        admin_base_url: None,
    }
}

fn deps(bridge: Arc<dyn Bridge>) -> AlgorithmDeps {
    AlgorithmDeps {
        policy: Arc::new(FakeScorer),
        policy_model_name: "policy".into(),
        connect_frozen: Arc::new(|_| Box::pin(async { Ok(Arc::new(FakeScorer) as Arc<dyn Scorer>) })),
        bridge: Some(bridge),
    }
}

fn config(params: AlgoParams, action_loss_type: ActionLossType) -> AlgoConfig {
    AlgoConfig { sampling: AlgoSamplingConfig { source: ModelReference::Policy }, action_loss_type, params }
}

fn episodes(records: &Value) -> Vec<Episode> {
    records.as_array().unwrap().iter().map(|r| Episode::from_record_json(&serde_json::to_vec(r).unwrap()).unwrap()).collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * (1.0 + a.abs().max(b.abs()))
}

fn check_stream(ctx: &str, got: Option<&Vec<f32>>, want: &Value) {
    match (got, want.as_array()) {
        (None, None) => {}
        (Some(g), Some(w)) => {
            assert_eq!(g.len(), w.len(), "{ctx}: length");
            for (a, b) in g.iter().zip(w) {
                assert!(close(f64::from(*a), b.as_f64().unwrap()), "{ctx}: {a} vs {b}");
            }
        }
        (g, w) => panic!("{ctx}: presence {:?} vs {:?}", g.is_some(), w.is_some()),
    }
}

fn check_episodes(ctx: &str, got: &[Episode], want: &Value) {
    for (e, (episode, record)) in got.iter().zip(want.as_array().unwrap()).enumerate() {
        for (t, (trace, trace_record)) in episode.traces.iter().zip(record["traces"].as_array().unwrap()).enumerate() {
            for (n, (node, node_record)) in trace.nodes.iter().zip(trace_record["nodes"].as_array().unwrap()).enumerate() {
                let at = format!("{ctx} e{e} t{t} n{n}");
                check_stream(&format!("{at} advantages"), node.advantages.as_ref(), &node_record["advantages"]);
                check_stream(&format!("{at} reference_logprobs"), node.reference_logprobs.as_ref(), &node_record["reference_logprobs"]);
                let want_weights = node_record["loss_weights"].as_object();
                assert_eq!(node.loss_weights.is_some(), want_weights.is_some(), "{at} loss_weights presence");
                if let (Some(g), Some(w)) = (&node.loss_weights, want_weights) {
                    assert_eq!(g.keys().collect::<Vec<_>>(), w.keys().collect::<Vec<_>>(), "{at} loss_weights keys");
                    for (name, stream) in w {
                        check_stream(&format!("{at} loss_weights.{name}"), g.get(name), stream);
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn opd_opsd_echo_match_python() {
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(&include_bytes!("fixtures/algo_cases.json.gz")[..]).read_to_end(&mut raw).unwrap();
    let cases: Vec<Value> = serde_json::from_slice(&raw).unwrap();
    for (i, case) in cases.iter().enumerate() {
        let masks: HashMap<String, Vec<Vec<bool>>> = serde_json::from_value(case["echo_masks"].clone()).unwrap();
        let deps = deps(Arc::new(FakeBridge { masks }));

        let mut opd = build_algorithm(
            &config(AlgoParams::Opd(OPDAlgoConfig { teacher: FrozenModelConfig { client: client(), name: "teacher".into() } }), ActionLossType::RefKl),
            &deps,
        )
        .unwrap();
        opd.setup().await.unwrap();
        let mut eps = episodes(&case["before"]);
        for e in &mut eps {
            opd.finalize_episode(e).await.unwrap();
        }
        check_episodes(&format!("case {i} opd"), &eps, &case["opd"]);

        let opsd_config = OPSDAlgoConfig {
            demo_key: "demonstration".into(),
            template: "Demo:\n{demonstration}\n{{end}}".into(),
            renderer: Value::Null,
        };
        let mut opsd = build_algorithm(&config(AlgoParams::Opsd(opsd_config), ActionLossType::RefKl), &deps).unwrap();
        let mut eps = episodes(&case["before"]);
        for e in &mut eps {
            opsd.finalize_episode(e).await.unwrap();
        }
        check_episodes(&format!("case {i} opsd"), &eps, &case["opsd"]);

        let echo_config = EchoAlgoConfig {
            length_penalty: None,
            length_weighted_baseline: false,
            loss_aggregation: LossAggregation::Token,
            roles: EchoRolesConfig { system: None, user: Some(EchoRoleConfig { alpha: 0.5 }), assistant: None, tool: Some(EchoRoleConfig { alpha: 0.25 }) },
            filter: Some(EchoFilterConfig { import_path: "filters.keep_even".into(), kwargs: Value::Null }),
        };
        let mut echo = build_algorithm(&config(AlgoParams::Echo(echo_config), ActionLossType::Rl), &deps).unwrap();
        let mut eps = episodes(&case["before"]);
        for e in &mut eps {
            echo.finalize_episode(e).await.unwrap();
        }
        echo.score_group(&mut eps).await.unwrap();
        check_episodes(&format!("case {i} echo"), &eps, &case["echo"]);
    }
}

#[test]
fn opsd_template_formats_like_str_format() {
    assert_eq!(format_template("A {demonstration} {{x}} }}", "d").unwrap(), "A d {x} }");
    assert!(format_template("{other}", "d").is_err());
}

fn sample(ce: Option<Vec<f32>>, rl: Option<Vec<f32>>) -> TrainingSample {
    TrainingSample {
        token_ids: vec![1, 2, 3, 4],
        mask: vec![false, false, true, true],
        logprobs: vec![0.0, 0.0, -0.1, -0.2],
        temperatures: vec![1.0; 4],
        env_name: "e".into(),
        advantages: Some(vec![0.0, 0.0, 1.0, 1.0]),
        ce_weights: ce,
        rl_weights: rl,
        ..Default::default()
    }
}

#[test]
fn stamp_loss_routing_uniform_rl() {
    let mut s = sample(None, None);
    stamp_loss_routing(&mut s, ActionLossType::Rl);
    assert!(s.rl_weights.is_none() && s.ce_weights.is_none() && s.ref_kl_weights.is_none());
}

#[test]
fn stamp_loss_routing_ref_kl_action() {
    let mut s = sample(None, None);
    stamp_loss_routing(&mut s, ActionLossType::RefKl);
    assert_eq!(s.rl_weights, Some(vec![0.0; 4]));
    assert_eq!(s.ref_kl_weights, Some(vec![0.0, 0.0, 1.0, 1.0]));
    assert!(s.ce_weights.is_none());
}

#[test]
fn stamp_loss_routing_ce_action_merges_written_ce_stream() {
    let mut s = sample(Some(vec![0.25, 0.0, 0.0, 0.0]), None);
    stamp_loss_routing(&mut s, ActionLossType::Ce);
    assert_eq!(s.rl_weights, Some(vec![0.0; 4]));
    assert_eq!(s.ce_weights, Some(vec![0.25, 0.0, 1.0, 1.0]));
}

#[test]
fn prune_zero_advantages_keeps_or_voids() {
    let mut s = sample(None, None);
    assert!(prune_zero_advantages(&mut s).unwrap());
    assert!(s.rl_weights.is_none(), "nothing pruned keeps the stream absent");

    let mut s = sample(None, None);
    s.advantages = Some(vec![0.0; 4]);
    assert!(!prune_zero_advantages(&mut s).unwrap());
    assert_eq!(s.rl_weights, Some(vec![0.0; 4]));

    let mut s = sample(Some(vec![0.0, 0.5, 0.0, 0.0]), None);
    s.advantages = Some(vec![0.0; 4]);
    assert!(prune_zero_advantages(&mut s).unwrap(), "a ce component keeps the sample");
    assert!(has_loss(&s));
}
