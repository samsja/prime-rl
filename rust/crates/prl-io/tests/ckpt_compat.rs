//! Checkpoint and RNG compatibility with the Python implementation (fixtures from tools/gen_fixtures.py).

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use prl_io::ckpt::{CheckpointManager, OrchestratorState, PyRandom, PyRandomState, SamplerState};
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn rng_json(rng: &PyRandomState) -> Value {
    serde_json::json!({"version": 3, "key": rng.key.to_vec(), "pos": rng.pos, "gauss_next": rng.gauss_next})
}

fn state_json(state: &OrchestratorState) -> Value {
    let p = &state.progress;
    let envs: serde_json::Map<String, Value> = state
        .train_source
        .envs
        .iter()
        .map(|(name, c)| {
            let sampler = match &c.sampler {
                SamplerState::Standard { cursor } => serde_json::json!({"cursor": cursor}),
                SamplerState::DifficultyPool { rng, task_rewards } => {
                    serde_json::json!({"rng": rng_json(rng), "task_rewards": task_rewards})
                }
            };
            let gates: IndexMap<&str, Value> = c.gates.keys().map(|k| (k.as_str(), serde_json::json!({}))).collect();
            (name.clone(), serde_json::json!({"sampler": sampler, "gates": gates}))
        })
        .collect();
    serde_json::json!({
        "progress": {"step": p.step, "total_tokens": p.total_tokens, "total_samples": p.total_samples,
                     "total_problems": p.total_problems, "rollout_window": p.rollout_window},
        "train_source": {"rng": rng_json(&state.train_source.rng), "envs": envs},
    })
}

fn expected() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("ckpt_expected.json")).unwrap()).unwrap()
}

#[test]
fn loads_python_checkpoint() {
    let manager = CheckpointManager::new(&fixtures().join("ckpt_run"));
    let state = manager.load(16, None).unwrap();
    assert_eq!(state_json(&state), expected());
    let configured = vec!["code".to_string(), "math".to_string()];
    state.train_source.validate_envs(&configured).unwrap();
    assert!(state.train_source.validate_envs(&configured[..1]).is_err());
}

#[test]
fn rust_checkpoint_round_trips() {
    let state = CheckpointManager::new(&fixtures().join("ckpt_run")).load(16, None).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let manager = CheckpointManager::new(dir.path());
    manager.save(&state, 16).unwrap();
    let files: Vec<_> = std::fs::read_dir(manager.ckpt_path(16)).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(files, vec![std::ffi::OsString::from("progress.pt")]);
    assert_eq!(manager.load(16, None).unwrap(), state);
    assert!(manager.load(3, None).unwrap_err().to_string().starts_with("Orchestrator checkpoint not found at"));
}

/// Python's `CheckpointManager.load` reads a Rust-written checkpoint. Needs the project venv:
/// `PRL_IO_PYTHON=1 cargo test -p prl-io --test ckpt_compat` from `rust/`.
#[test]
fn python_loads_rust_checkpoint() {
    if std::env::var_os("PRL_IO_PYTHON").is_none() {
        eprintln!("skipped: set PRL_IO_PYTHON=1 to run against the Python implementation");
        return;
    }
    let state = CheckpointManager::new(&fixtures().join("ckpt_run")).load(16, None).unwrap();
    let dir = tempfile::tempdir().unwrap();
    CheckpointManager::new(dir.path()).save(&state, 16).unwrap();
    let script = format!(
        r#"
import json, sys, torch
sys.path.insert(0, "{tools}")
from gen_fixtures import state_json
state = torch.load("{file}", weights_only=False)
print(json.dumps(state_json(state)))
"#,
        tools = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools").display(),
        file = dir.path().join("checkpoints/step_16/orchestrator/progress.pt").display(),
    );
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let output = std::process::Command::new("uv")
        .args(["run", "--no-sync", "python", "-c", &script])
        .current_dir(repo)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let loaded: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(loaded, expected());
}

#[test]
fn pyrandom_matches_cpython() {
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(fixtures().join("pyrandom.json")).unwrap()).unwrap();
    for case in cases {
        let seed = case["seed"].as_u64().unwrap();
        let mut rng = PyRandom::new(seed);
        let draws: Vec<f64> = (0..5).map(|_| rng.random()).collect();
        let expected_draws: Vec<f64> = serde_json::from_value(case["draws"].clone()).unwrap();
        assert_eq!(draws, expected_draws, "seed {seed}");

        let weights: Vec<f64> = serde_json::from_value(case["weights"].clone()).unwrap();
        let mut rng = PyRandom::new(seed);
        let picks: Vec<usize> = (0..50).map(|_| rng.choices_index(&weights).unwrap()).collect();
        let expected_picks: Vec<usize> = serde_json::from_value(case["picks"].clone()).unwrap();
        assert_eq!(picks, expected_picks, "seed {seed}");

        let state = rng.state();
        let key: Vec<u32> = serde_json::from_value(case["state_after"]["key"].clone()).unwrap();
        assert_eq!(state.key.to_vec(), key, "seed {seed}");
        assert_eq!(state.pos as u64, case["state_after"]["pos"].as_u64().unwrap());
        let mut restored = PyRandom::from_state(&state);
        assert_eq!(restored.random(), rng.random());
    }
    assert!(PyRandom::new(1).choices_index(&[0.0, 0.0]).is_err());
}
