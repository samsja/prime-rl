//! Training task selection across envs and the per-env curricula (`train_source.py`).

use anyhow::{ensure, Result};
use indexmap::IndexMap;
use prl_io::ckpt::{PyRandom, TrainSourceState};
use prl_types::{Metrics, TaskRecord};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::curriculum::{Curriculum, CurriculumConfig, GroupOutcome, Tasks};
use crate::eval_source::TaskRequest;

/// One train env as the source sees it.
pub struct TrainEnvTasks {
    pub name: String,
    pub ratio: f64,
    pub curriculum: CurriculumConfig,
    pub tasks: Tasks,
}

/// Mix train envs and host one curriculum per env.
pub struct TrainSource {
    rng: PyRandom,
    env_names: Vec<String>,
    weights: Vec<f64>,
    pub curricula: IndexMap<String, Curriculum>,
    admitted: IndexMap<String, u64>,
    rejected: IndexMap<String, u64>,
}

impl TrainSource {
    pub fn new(envs: Vec<TrainEnvTasks>) -> Result<Self> {
        ensure!(!envs.is_empty(), "TrainSource needs at least one train env");
        let mut curricula = IndexMap::new();
        let mut env_names = Vec::new();
        let mut weights = Vec::new();
        for env in envs {
            env_names.push(env.name.clone());
            weights.push(env.ratio);
            curricula.insert(env.name, Curriculum::new(env.curriculum, env.tasks)?);
        }
        Ok(Self {
            rng: PyRandom::new(42),
            env_names,
            weights,
            curricula,
            admitted: IndexMap::new(),
            rejected: IndexMap::new(),
        })
    }

    pub async fn next_task(&mut self, step: u64) -> Result<TaskRequest> {
        let env_name = self.env_names[self.rng.choices_index(&self.weights)?].clone();
        let task: TaskRecord = self.curricula[&env_name].next_task().await?;
        Ok(TaskRequest { env_name, task, step, rollouts: None, group_id: None })
    }

    /// Report a finalized group of `env_name`; returns whether it should train.
    pub fn on_result(&mut self, env_name: &str, outcome: &GroupOutcome) -> Result<bool> {
        let curriculum = self
            .curricula
            .get_mut(env_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown train env {env_name:?}"))?;
        let admitted = curriculum.on_result(outcome)?;
        let counter = if admitted { &mut self.admitted } else { &mut self.rejected };
        *counter.entry(env_name.to_owned()).or_default() += 1;
        Ok(admitted)
    }

    /// Per-env admission rate since the last call plus curriculum metrics.
    pub fn metrics(&mut self) -> Metrics {
        let mut m = Metrics::new();
        for (env_name, curriculum) in &self.curricula {
            let admitted = self.admitted.shift_remove(env_name).unwrap_or(0);
            let rejected = self.rejected.shift_remove(env_name).unwrap_or(0);
            let total = admitted + rejected;
            if total > 0 {
                m.insert(format!("curriculum/{env_name}/admission_rate"), (admitted as f64 / total as f64).into());
            }
            for (name, value) in curriculum.metrics() {
                m.insert(format!("curriculum/{env_name}/{name}"), value.into());
            }
        }
        m
    }

    pub fn state(&self) -> TrainSourceState {
        TrainSourceState {
            rng: self.rng.state(),
            envs: self.curricula.iter().map(|(name, c)| (name.clone(), c.state())).collect(),
        }
    }

    /// Restore a checkpoint; returns the env names whose curriculum was resumed.
    pub async fn load(&mut self, state: &TrainSourceState) -> Result<Vec<String>> {
        let configured: Vec<String> = self.curricula.keys().cloned().collect();
        state.validate_envs(&configured)?;
        self.rng = PyRandom::from_state(&state.rng);
        for (name, curriculum) in &mut self.curricula {
            curriculum.load(&state.envs[name]).await?;
        }
        Ok(state.envs.keys().filter(|name| self.curricula.contains_key(*name)).cloned().collect())
    }
}

// ── actor ──────────────────────────────────────────────────────────────────

enum Command {
    NextTask { step: u64, reply: oneshot::Sender<Result<TaskRequest>> },
    OnResult { env_name: String, outcome: GroupOutcome, reply: oneshot::Sender<Result<bool>> },
    Metrics { reply: oneshot::Sender<Metrics> },
    State { reply: oneshot::Sender<TrainSourceState> },
}

/// Handle onto the train source actor: the dispatcher draws tasks, the train pipeline
/// reports finalized groups, reads metrics at ship time and snapshots it for checkpoints.
#[derive(Clone)]
pub struct TrainSourceHandle {
    tx: mpsc::UnboundedSender<Command>,
}

fn stopped() -> anyhow::Error {
    anyhow::anyhow!("train source stopped")
}

impl TrainSourceHandle {
    pub async fn next_task(&self, step: u64) -> Result<TaskRequest> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Command::NextTask { step, reply }).map_err(|_| stopped())?;
        rx.await.map_err(|_| stopped())?
    }

    pub async fn on_result(&self, env_name: &str, outcome: GroupOutcome) -> Result<bool> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Command::OnResult { env_name: env_name.to_owned(), outcome, reply }).map_err(|_| stopped())?;
        rx.await.map_err(|_| stopped())?
    }

    pub async fn metrics(&self) -> Result<Metrics> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Command::Metrics { reply }).map_err(|_| stopped())?;
        rx.await.map_err(|_| stopped())
    }

    pub async fn state(&self) -> Result<TrainSourceState> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Command::State { reply }).map_err(|_| stopped())?;
        rx.await.map_err(|_| stopped())
    }
}

impl TrainSource {
    /// Run as an actor until every handle is dropped.
    pub fn spawn(mut self) -> (TrainSourceHandle, JoinHandle<()>) {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    Command::NextTask { step, reply } => {
                        let _ = reply.send(self.next_task(step).await);
                    }
                    Command::OnResult { env_name, outcome, reply } => {
                        let _ = reply.send(self.on_result(&env_name, &outcome));
                    }
                    Command::Metrics { reply } => {
                        let _ = reply.send(self.metrics());
                    }
                    Command::State { reply } => {
                        let _ = reply.send(self.state());
                    }
                }
            }
        });
        (TrainSourceHandle { tx }, task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curriculum::{AdvRangeGate, DifficultyPool, SamplerConfig};

    fn tasks(n: usize) -> Vec<TaskRecord> {
        (0..n)
            .map(|i| TaskRecord { key: format!("key-{i}"), hash: format!("hash-{i}"), data: serde_json::json!({"idx": i}) })
            .collect()
    }

    /// One trainable episode of `key` with `reward` and node `advantages`.
    fn rollout(key: &str, reward: f64, advantages: &[f32]) -> GroupOutcome {
        GroupOutcome {
            task_keys: vec![Some(key.into())],
            rewards: vec![reward],
            advantages: advantages.to_vec(),
            any_trainable: true,
        }
    }

    fn pools(weights: [f64; 3], seed: u64) -> CurriculumConfig {
        let pool = |threshold, weight| DifficultyPool { threshold, weight };
        CurriculumConfig {
            sampler: SamplerConfig::DifficultyPool {
                pools: [("hard", pool(0.25, weights[0])), ("normal", pool(0.75, weights[1])), ("easy", pool(1.0, weights[2]))]
                    .into_iter()
                    .map(|(name, p)| (name.to_string(), p))
                    .collect(),
                seed,
            },
            gates: [("zero_advantage".to_string(), AdvRangeGate { reject_min: 0.0, reject_max: 0.0 })].into_iter().collect(),
        }
    }

    #[tokio::test]
    async fn standard_sampler_resumes_at_its_cursor() {
        let mut finite = Curriculum::new(CurriculumConfig::default(), Tasks::Finite(tasks(5))).unwrap();
        for _ in 0..3 {
            finite.next_task().await.unwrap();
        }
        let mut restored = Curriculum::new(CurriculumConfig::default(), Tasks::Finite(tasks(5))).unwrap();
        restored.load(&finite.state()).await.unwrap();
        for _ in 0..4 {
            assert_eq!(restored.next_task().await.unwrap().key, finite.next_task().await.unwrap().key);
        }
        let duplicate = Curriculum::new(CurriculumConfig::default(), Tasks::Finite(vec![tasks(1)[0].clone(); 2]));
        assert!(duplicate.err().unwrap().to_string().contains("Task keys must be unique"));
    }

    #[tokio::test]
    async fn difficulty_pools_stack_with_the_advantage_gate() {
        let mut curriculum = Curriculum::new(pools([0.0, 1.0, 0.0], 7), Tasks::Finite(tasks(3))).unwrap();
        let decisions: Vec<bool> = [(0.1, 0.0), (0.5, 1.0), (0.9, 1.0)]
            .iter()
            .enumerate()
            .map(|(i, (reward, adv))| curriculum.on_result(&rollout(&format!("key-{i}"), *reward, &[*adv])).unwrap())
            .collect();
        assert_eq!(decisions, vec![false, true, true]);
        let metrics: Vec<(String, f64)> = curriculum.metrics().into_iter().collect();
        assert_eq!(metrics[0], ("sampler/pool/unseen".to_string(), 0.0));
        // Only the normal pool has weight: every draw is key-1.
        for _ in 0..5 {
            assert_eq!(curriculum.next_task().await.unwrap().key, "key-1");
        }
        let gate = AdvRangeGate { reject_min: -0.1, reject_max: 0.1 };
        assert!(!gate.admit(&rollout("k", 0.0, &[-0.05, 0.0, 0.05])));
        assert!(gate.admit(&rollout("k", 0.0, &[0.2])));
        assert!(gate.admit(&rollout("k", 0.0, &[])));
    }

    #[tokio::test]
    async fn sampling_matches_cpython_random() {
        // python: r = random.Random(42); [r.choices(['a','b'], weights=[1.0,3.0])[0] for _ in range(20)]
        let env = |name: &str, ratio| TrainEnvTasks {
            name: name.into(),
            ratio,
            curriculum: CurriculumConfig::default(),
            tasks: Tasks::Finite(tasks(2)),
        };
        let mut source = TrainSource::new(vec![env("a", 1.0), env("b", 3.0)]).unwrap();
        let mut picked = String::new();
        for _ in 0..20 {
            picked.push_str(&source.next_task(1).await.unwrap().env_name);
        }
        assert_eq!(picked, "bababbbabaabaabbabba");
        // python: r = random.Random(42); [r.choices(range(3), weights=[0.2,1.0,0.2])[0] for _ in range(20)]
        let mut curriculum = Curriculum::new(pools([0.2, 1.0, 0.2], 42), Tasks::Finite(tasks(3))).unwrap();
        for (i, reward) in [0.1, 0.5, 0.9].into_iter().enumerate() {
            curriculum.on_result(&rollout(&format!("key-{i}"), reward, &[1.0])).unwrap();
        }
        for idx in [1, 0, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0] {
            assert_eq!(curriculum.next_task().await.unwrap().key, format!("key-{idx}"));
        }
        assert_eq!(source.metrics().len(), 0);
    }
}
