//! Resuming from a checkpoint written by the Python orchestrator.

use prl_core::curriculum::{CurriculumConfig, Tasks};
use prl_core::train_source::{TrainEnvTasks, TrainSource};
use prl_io::ckpt::{CheckpointManager, PyRandom};
use prl_types::TaskRecord;

fn tasks(n: usize) -> Vec<TaskRecord> {
    (0..n)
        .map(|i| TaskRecord { key: format!("key-{i}"), hash: format!("hash-{i}"), data: serde_json::json!({"idx": i}) })
        .collect()
}

#[tokio::test]
async fn resumes_python_checkpoint() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/py_ckpt");
    let state = CheckpointManager::new(std::path::Path::new(dir)).load(20, None).unwrap();
    assert_eq!(state.progress.step, 20);
    assert_eq!(state.progress.total_samples, 2576);
    // python: r.setstate(state["train_source"]["rng"]); [r.random() for _ in range(3)]
    let mut rng = PyRandom::from_state(&state.train_source.rng);
    assert_eq!(rng.random(), 0.07085734988865344);
    assert_eq!(rng.random(), 0.23800463436899522);

    let mut source = TrainSource::new(vec![TrainEnvTasks {
        name: "reverse-text".into(),
        ratio: 1.0,
        curriculum: CurriculumConfig::default(),
        tasks: Tasks::Finite(tasks(1000)),
    }])
    .unwrap();
    let resumed = source.load(&state.train_source).await.unwrap();
    assert_eq!(resumed, vec!["reverse-text".to_string()]);
    // The saved cursor is 165: the next task continues the cycle there.
    assert_eq!(source.next_task(21).await.unwrap().task.key, "key-165");
}
