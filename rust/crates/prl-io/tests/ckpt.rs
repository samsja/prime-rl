//! A checkpoint written by the Python orchestrator (`torch.save`) loads, and a Rust save
//! round-trips. The fixture holds a standard and a difficulty-pool curriculum.

use std::path::Path;

use prl_io::ckpt::{CheckpointManager, PyRandom, SamplerState};

fn python_checkpoint() -> CheckpointManager {
    CheckpointManager::new(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ckpt_run"))
}

#[test]
fn loads_python_checkpoint() {
    let state = python_checkpoint().load(16, None).unwrap();
    let p = &state.progress;
    assert_eq!((p.step, p.total_tokens, p.total_samples, p.total_problems, p.rollout_window), (17, (1 << 40) + 3, 4096, 512, 3));

    // random.Random(42) after 7 draws
    let mut rng = PyRandom::new(42);
    (0..7).for_each(|_| {
        rng.random();
    });
    assert_eq!(state.train_source.rng, rng.state());

    let envs: Vec<&str> = state.train_source.envs.keys().map(String::as_str).collect();
    assert_eq!(envs, ["math", "code"]);
    assert_eq!(state.train_source.envs["math"].sampler, SamplerState::Standard { cursor: 12345 });
    let SamplerState::DifficultyPool { rng, task_rewards } = &state.train_source.envs["code"].sampler else { panic!() };
    assert_eq!(rng.gauss_next, Some(-0.22555725575068641));
    assert_eq!(task_rewards.iter().map(|(k, v)| (k.as_str(), *v)).collect::<Vec<_>>(), [("t-1", 0.5), ("t-2", 1.0), ("t-3", 0.0)]);
    assert_eq!(state.train_source.envs["code"].gates.keys().collect::<Vec<_>>(), ["adv"]);
}

#[test]
fn rust_checkpoint_round_trips() {
    let state = python_checkpoint().load(16, None).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let manager = CheckpointManager::new(dir.path());
    manager.save(&state, 16).unwrap();
    assert_eq!(manager.load(16, None).unwrap(), state);
    assert!(manager.load(3, None).unwrap_err().to_string().starts_with("Orchestrator checkpoint not found at"));
}
