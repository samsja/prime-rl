//! Task selection and finalized-group admission for training envs (`curriculum/*`).

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use async_trait::async_trait;
use indexmap::IndexMap;
use prl_io::ckpt::{CurriculumState, PyRandom, PyValue, SamplerState};
use prl_types::{Bridge, Kind, TaskRecord};

/// What the curriculum reads off a finalized group (computed by the train pipeline).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroupOutcome {
    /// `episode.task.key` per episode.
    pub task_keys: Vec<Option<String>>,
    /// Rewards of trainable traces of ok episodes without errors and with turns (sampler signal).
    pub rewards: Vec<f64>,
    /// Advantages of every node of trainable traces of ok episodes without errors (gate signal).
    pub advantages: Vec<f32>,
    /// `any(iter_trainable_traces(group))`.
    pub any_trainable: bool,
}

/// Pulls more tasks of an unbounded taskset.
#[async_trait]
pub trait TaskFeed: Send {
    /// Up to `n` more tasks; fewer means the stream ended.
    async fn next_batch(&mut self, n: usize) -> Result<Vec<TaskRecord>>;
}

/// Feed over the Python bridge's `next_tasks`.
pub struct BridgeFeed {
    pub bridge: Arc<dyn Bridge>,
    pub kind: Kind,
    pub env_name: String,
}

#[async_trait]
impl TaskFeed for BridgeFeed {
    async fn next_batch(&mut self, n: usize) -> Result<Vec<TaskRecord>> {
        self.bridge.next_tasks(self.kind, &self.env_name, n).await
    }
}

const FEED_BATCH: usize = 64;

/// An env's selected tasks: materialized (bounded) or streamed (unbounded).
pub enum Tasks {
    Finite(Vec<TaskRecord>),
    Stream(Box<dyn TaskFeed>),
}

struct Stream {
    feed: Box<dyn TaskFeed>,
    buffer: VecDeque<TaskRecord>,
}

impl Stream {
    async fn next(&mut self) -> Result<TaskRecord> {
        if self.buffer.is_empty() {
            self.buffer.extend(self.feed.next_batch(FEED_BATCH).await?);
        }
        match self.buffer.pop_front() {
            Some(task) => Ok(task),
            None => bail!("The task stream ended"),
        }
    }
}

fn check_unique_keys(tasks: &[TaskRecord]) -> Result<()> {
    let mut seen = HashSet::new();
    let mut duplicates: Vec<&str> = tasks.iter().filter(|t| !seen.insert(&t.key)).map(|t| t.key.as_str()).collect();
    duplicates.sort_unstable();
    duplicates.dedup();
    ensure!(duplicates.is_empty(), "Task keys must be unique within a taskset: {duplicates:?}");
    Ok(())
}

/// `StandardSampler`: cycle a finite taskset in source order, or advance the stream.
pub struct StandardSampler {
    finite: Option<Vec<TaskRecord>>,
    stream: Option<Stream>,
    cursor: u64,
}

impl StandardSampler {
    pub fn new(tasks: Tasks) -> Result<Self> {
        match tasks {
            Tasks::Finite(tasks) => {
                ensure!(!tasks.is_empty(), "A finite curriculum needs at least one task");
                check_unique_keys(&tasks)?;
                Ok(Self { finite: Some(tasks), stream: None, cursor: 0 })
            }
            Tasks::Stream(feed) => {
                Ok(Self { finite: None, stream: Some(Stream { feed, buffer: VecDeque::new() }), cursor: 0 })
            }
        }
    }

    async fn next(&mut self) -> Result<TaskRecord> {
        let task = match (&self.finite, &mut self.stream) {
            (Some(tasks), _) => tasks[(self.cursor % tasks.len() as u64) as usize].clone(),
            (None, Some(stream)) => stream.next().await?,
            (None, None) => unreachable!(),
        };
        self.cursor += 1;
        Ok(task)
    }

    async fn load(&mut self, cursor: u64) -> Result<()> {
        if let Some(stream) = &mut self.stream {
            for _ in 0..cursor {
                stream.next().await?;
            }
        }
        self.cursor = cursor;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DifficultyPool {
    pub threshold: f64,
    pub weight: f64,
}

/// `DifficultyPoolSampler`: weight finite tasks by the pool of their latest group mean.
pub struct DifficultyPoolSampler {
    tasks: Vec<TaskRecord>,
    rng: PyRandom,
    pools: IndexMap<String, DifficultyPool>,
    ordered: Vec<(String, f64)>,
    task_rewards: IndexMap<String, f64>,
}

impl DifficultyPoolSampler {
    pub fn new(pools: IndexMap<String, DifficultyPool>, seed: u64, tasks: Tasks) -> Result<Self> {
        let Tasks::Finite(tasks) = tasks else { bail!("DifficultyPoolSampler requires a finite taskset") };
        ensure!(!tasks.is_empty(), "DifficultyPoolSampler requires at least one task");
        check_unique_keys(&tasks)?;
        let mut ordered: Vec<(String, f64)> = pools.iter().map(|(n, p)| (n.clone(), p.threshold)).collect();
        ordered.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(Self { tasks, rng: PyRandom::new(seed), pools, ordered, task_rewards: IndexMap::new() })
    }

    /// The task's current pool, or None until it has a score.
    pub fn task_pool(&self, task_key: &str) -> Option<&str> {
        let score = *self.task_rewards.get(task_key)?;
        let pool = self.ordered.iter().find(|(_, threshold)| score <= *threshold).unwrap_or(self.ordered.last()?);
        Some(pool.0.as_str())
    }

    fn next(&mut self) -> Result<TaskRecord> {
        let weights: Vec<f64> = self
            .tasks
            .iter()
            .map(|t| self.task_pool(&t.key).map_or(1.0, |pool| self.pools[pool].weight))
            .collect();
        ensure!(
            weights.iter().any(|w| *w != 0.0),
            "DifficultyPoolSampler has no tasks in a pool with positive weight"
        );
        Ok(self.tasks[self.rng.choices_index(&weights)?].clone())
    }

    fn observe(&mut self, outcome: &GroupOutcome) -> Result<()> {
        if outcome.rewards.is_empty() {
            return Ok(());
        }
        let Some(Some(key)) = outcome.task_keys.first() else { bail!("A finalized group is missing Task.key") };
        let mean = outcome.rewards.iter().sum::<f64>() / outcome.rewards.len() as f64;
        self.task_rewards.insert(key.clone(), mean);
        Ok(())
    }

    fn metrics(&self) -> IndexMap<String, f64> {
        let mut occupancy: IndexMap<String, f64> = self.pools.keys().map(|k| (k.clone(), 0.0)).collect();
        for key in self.task_rewards.keys() {
            if let Some(pool) = self.task_pool(key) {
                occupancy[pool] += 1.0;
            }
        }
        let mut m = IndexMap::new();
        m.insert("pool/unseen".to_string(), (self.tasks.len() - self.task_rewards.len()) as f64);
        for (name, count) in occupancy {
            m.insert(format!("pool/{name}"), count);
        }
        m
    }
}

pub enum Sampler {
    Standard(StandardSampler),
    DifficultyPool(DifficultyPoolSampler),
}

/// `AdvRangeGate`: reject groups whose trainable advantages all fall inside a range.
#[derive(Debug, Clone, PartialEq)]
pub struct AdvRangeGate {
    pub reject_min: f64,
    pub reject_max: f64,
}

impl AdvRangeGate {
    pub fn admit(&self, outcome: &GroupOutcome) -> bool {
        if outcome.advantages.is_empty() {
            return true;
        }
        // numpy compares the float32 stream against the Python float as float32 (NEP 50).
        let (lo, hi) = (self.reject_min as f32, self.reject_max as f32);
        !outcome.advantages.iter().all(|a| *a >= lo && *a <= hi)
    }
}

/// Sampler + gate configuration of one env (`CurriculumConfig`).
#[derive(Debug, Clone, PartialEq)]
pub enum SamplerConfig {
    Standard,
    DifficultyPool { pools: IndexMap<String, DifficultyPool>, seed: u64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CurriculumConfig {
    pub sampler: SamplerConfig,
    pub gates: IndexMap<String, AdvRangeGate>,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self { sampler: SamplerConfig::Standard, gates: IndexMap::new() }
    }
}

/// One task sampler composed with zero or more admission gates.
pub struct Curriculum {
    pub sampler: Sampler,
    pub gates: IndexMap<String, AdvRangeGate>,
}

impl Curriculum {
    pub fn new(config: CurriculumConfig, tasks: Tasks) -> Result<Self> {
        let sampler = match config.sampler {
            SamplerConfig::Standard => Sampler::Standard(StandardSampler::new(tasks)?),
            SamplerConfig::DifficultyPool { pools, seed } => {
                Sampler::DifficultyPool(DifficultyPoolSampler::new(pools, seed, tasks)?)
            }
        };
        Ok(Self { sampler, gates: config.gates })
    }

    pub async fn next_task(&mut self) -> Result<TaskRecord> {
        match &mut self.sampler {
            Sampler::Standard(s) => s.next().await,
            Sampler::DifficultyPool(s) => s.next(),
        }
    }

    /// Observe every result; admit groups with a trainable trace that pass every gate.
    pub fn on_result(&mut self, outcome: &GroupOutcome) -> Result<bool> {
        ensure!(!outcome.task_keys.is_empty(), "Cannot report an empty rollout group");
        ensure!(outcome.task_keys.iter().all(Option::is_some), "A finalized group is missing Task.key");
        let keys: HashSet<&str> = outcome.task_keys.iter().flatten().map(String::as_str).collect();
        if keys.len() != 1 {
            let mut keys: Vec<&str> = keys.into_iter().collect();
            keys.sort_unstable();
            bail!("A finalized group contains multiple task keys: {keys:?}");
        }
        if let Sampler::DifficultyPool(s) = &mut self.sampler {
            s.observe(outcome)?;
        }
        let all_admit = self.gates.values().map(|g| g.admit(outcome)).fold(true, |acc, d| acc && d);
        Ok(all_admit && outcome.any_trainable)
    }

    pub fn state(&self) -> CurriculumState {
        let sampler = match &self.sampler {
            Sampler::Standard(s) => SamplerState::Standard { cursor: s.cursor },
            Sampler::DifficultyPool(s) => {
                SamplerState::DifficultyPool { rng: Box::new(s.rng.state()), task_rewards: s.task_rewards.clone() }
            }
        };
        let gates = self.gates.keys().map(|k| (k.clone(), PyValue::dict::<String>([]))).collect();
        CurriculumState { sampler, gates }
    }

    pub async fn load(&mut self, state: &CurriculumState) -> Result<()> {
        let mut saved: Vec<&str> = state.gates.keys().map(String::as_str).collect();
        let mut configured: Vec<&str> = self.gates.keys().map(String::as_str).collect();
        saved.sort_unstable();
        configured.sort_unstable();
        ensure!(
            saved == configured,
            "Curriculum checkpoint gates {saved:?} do not match configured gates {configured:?}"
        );
        match (&mut self.sampler, &state.sampler) {
            (Sampler::Standard(s), SamplerState::Standard { cursor }) => s.load(*cursor).await,
            (Sampler::DifficultyPool(s), SamplerState::DifficultyPool { rng, task_rewards }) => {
                s.rng = PyRandom::from_state(rng);
                s.task_rewards = task_rewards.clone();
                Ok(())
            }
            _ => bail!("Curriculum checkpoint sampler does not match the configured sampler"),
        }
    }

    /// `sampler/...` and `gate/{name}/...` metrics (the built-in gates report none).
    pub fn metrics(&self) -> IndexMap<String, f64> {
        match &self.sampler {
            Sampler::Standard(_) => IndexMap::new(),
            Sampler::DifficultyPool(s) => s.metrics().into_iter().map(|(k, v)| (format!("sampler/{k}"), v)).collect(),
        }
    }
}
