//! Extract every orchestrator config under `configs/` and `examples/` (plus the variant
//! TOMLs in `tools/extra/`) from the real pydantic objects. A renamed or removed pydantic
//! field, or a new union variant, fails extraction here.

use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use prl_config::*;

fn init_python() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // SAFETY: set once, before the interpreter starts and before other threads read the env.
        unsafe { std::env::set_var("PYTHONHOME", env!("PRL_TEST_PYTHONHOME")) };
        pyo3::prepare_freethreaded_python();
    });
}

const SETUP: &str = r#"
import site, sys
from pathlib import Path
root = Path(ROOT)
site.addsitedir(str(next((root / ".venv" / "lib").glob("python3*/site-packages"))))
sys.path.insert(0, str(root / "rust" / "crates" / "prl-config" / "tools"))
"#;

type Loaded<'py> = Vec<(String, Bound<'py, PyAny>)>;

/// `(path, OrchestratorConfig object)` for every loadable config, via `tools/load_configs.py`.
fn load_all(py: Python<'_>) -> Loaded<'_> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let setup = CString::new(format!("ROOT = {root:?}\n{SETUP}")).unwrap();
    py.run(&setup, None, None).unwrap();
    let loader = PyModule::import(py, "load_configs").unwrap();
    let (loaded, skipped): (Loaded<'_>, Vec<(String, String)>) =
        loader.call_method0("load_all").unwrap().extract().unwrap();
    for (name, error) in &skipped {
        eprintln!("skipped {name}: {error}");
    }
    loaded
}

fn by_name<'a>(configs: &'a [(String, OrchestratorConfig)], name: &str) -> &'a OrchestratorConfig {
    &configs
        .iter()
        .find(|(path, _)| path.ends_with(name))
        .unwrap_or_else(|| panic!("{name} not loaded"))
        .1
}

#[test]
fn extracts_every_config() {
    init_python();
    Python::with_gil(|py| {
        let loaded = load_all(py);
        assert!(loaded.len() >= 40, "only {} configs loaded", loaded.len());
        let configs: Vec<(String, OrchestratorConfig)> = loaded
            .iter()
            .map(|(name, obj)| {
                let config =
                    OrchestratorConfig::extract(obj).unwrap_or_else(|e| panic!("{name}: {e}"));
                (name.clone(), config)
            })
            .collect();

        for (name, config) in &configs {
            assert_eq!(
                config.dump["batch_size"].as_u64(),
                Some(config.batch_size),
                "{name}"
            );
            let env_names: Vec<&str> = config
                .env_addresses
                .keys()
                .map(|(_, name)| name.as_str())
                .collect();
            let mut expected: Vec<&str> = config
                .train
                .source
                .iter()
                .map(|s| s.resolved_name.as_str())
                .collect();
            if let Some(eval) = &config.eval {
                expected.extend(eval.source.iter().map(|s| s.resolved_name.as_str()));
            }
            assert_eq!(env_names, expected, "{name}");
        }

        let a = by_name(&configs, "tools/extra/variants_a.toml");
        assert_eq!(
            a.inference_metrics_roles,
            Some(vec![InferenceRole::Prefill, InferenceRole::Decode])
        );
        assert_eq!(a.weight_broadcast.tag(), "nccl");
        assert_eq!(a.weight_broadcast.timeout(), 600);
        assert_eq!(a.rollout_transport, TransportConfig::Filesystem);
        assert_eq!(a.gc, None);
        assert_eq!(
            a.resume,
            Some(ResumeConfig {
                step: Some(3),
                dir: None,
                dir_step: None
            })
        );
        assert_eq!(
            a.model.client.admin_base_url.as_ref().map(Vec::len),
            Some(2)
        );
        assert_eq!(
            a.model
                .client
                .dynamo
                .as_ref()
                .unwrap()
                .discovery_url
                .as_deref(),
            Some("http://dynamo:8000")
        );
        assert_eq!(a.monitors.file.as_ref().unwrap().float_decimals, None);
        assert_eq!(a.renderer["name"], "default");
        let rt_a = &a.train.source[0];
        assert!(rt_a.sampling.truncates_distribution);
        assert_eq!(rt_a.sampling.extra_body["extra_args"]["prl_replay"], true);
        assert_eq!(rt_a.algo.action_loss_type, ActionLossType::Rl);
        match &rt_a.algo.params {
            AlgoParams::Grpo(grpo) => {
                assert_eq!(grpo.loss_aggregation, LossAggregation::Prompt);
                assert!(matches!(
                    grpo.length_penalty,
                    Some(LengthPenaltyConfig::LinearSubagent(_))
                ));
            }
            other => panic!("expected grpo, got {other:?}"),
        }
        let curriculum = rt_a.curriculum.as_ref().unwrap();
        match &curriculum.sampler {
            TaskSamplerConfig::DifficultyPool(pools) => {
                assert_eq!(
                    pools.pools.keys().collect::<Vec<_>>(),
                    ["hard", "normal", "easy"]
                );
            }
            other => panic!("expected difficulty_pool, got {other:?}"),
        }
        assert!(matches!(
            curriculum.gates["adv"],
            AdmissionGateConfig::AdvantageRange(_)
        ));
        assert_eq!(
            a.env_addresses[&(Split::Train, "rt-a".to_string())].as_deref(),
            Some("tcp://127.0.0.1:5000")
        );
        assert_eq!(a.env_addresses[&(Split::Eval, "rt-eval".to_string())], None);
        assert!(
            matches!(&a.train.source[1].algo.params, AlgoParams::Echo(echo) if echo.filter.is_some())
        );
        let eval = a.eval.as_ref().unwrap();
        assert_eq!(eval.intervals.get("rt-eval"), Some(&2));
        assert_eq!(eval.intervals.get("rt-eval-2"), Some(&5));
        assert_eq!(eval.source[0].sampling["top_k"], 20);
        assert_eq!(eval.source[1].sampling["top_k"], -1);

        let b = by_name(&configs, "tools/extra/variants_b.toml");
        assert_eq!(b.weight_broadcast.tag(), "nixl");
        assert_eq!(b.concurrency.max_inflight, None);
        assert_eq!(b.monitors.file, None);
        assert_eq!(b.monitors.prime, None);
        let resume = b.resume.as_ref().unwrap();
        assert_eq!(resume.dir_step, Some(50));
        let tags: Vec<&str> = b.train.source.iter().map(|s| s.algo.params.tag()).collect();
        assert_eq!(
            tags,
            [
                "sft",
                "debug",
                "rae",
                "max_rl",
                "opd",
                "opsd",
                "hierarchical_grpo"
            ]
        );
        let losses: Vec<ActionLossType> = b
            .train
            .source
            .iter()
            .map(|s| s.algo.action_loss_type)
            .collect();
        use ActionLossType::*;
        assert_eq!(losses, [Ce, Rl, Rl, Rl, RefKl, RefKl, Rl]);
        match &b.train.source[0].algo.sampling.source {
            ModelReference::Frozen(frozen) => {
                assert_eq!(frozen.name, "teacher");
                assert_eq!(frozen.api_key_var, "T_KEY");
            }
            ModelReference::Policy => panic!("sft samples from a frozen model"),
        }
        assert!(
            !b.any_policy_sourced
                || b.train
                    .source
                    .iter()
                    .any(|s| s.algo.sampling.source == ModelReference::Policy)
        );
    });
}
