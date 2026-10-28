//! Shared orchestrator runtime types (port of `orchestrator/types.py` plus the
//! event vocabulary of the actor design).

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Ordered metric dict (Python dicts keep insertion order; monitors rely on it).
pub type Metrics = IndexMap<String, f64>;

/// `WorkKind` in Python.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Train,
    Eval,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Train => "train",
            Kind::Eval => "eval",
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Trace subset a monitors log call targets (`"all"` on arrival, `"effective"` at ship/epoch end).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Subset {
    All,
    Effective,
}

impl Subset {
    pub fn as_str(self) -> &'static str {
        match self {
            Subset::All => "all",
            Subset::Effective => "effective",
        }
    }
}

/// The policy inference currently serves. Single writer (the weight watcher),
/// published through `tokio::sync::watch`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub version: u64,
    pub model_name: String,
}

/// Persistent counters; `step` is the trainer-aligned step (1-indexed).
/// Field names match the Python dataclass (checkpointed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Progress {
    pub step: u64,
    pub total_tokens: u64,
    pub total_samples: u64,
    pub total_problems: u64,
    pub rollout_window: u64,
}

impl Default for Progress {
    fn default() -> Self {
        Self { step: 1, total_tokens: 0, total_samples: 0, total_problems: 0, rollout_window: 0 }
    }
}

/// Run identity stamped on every episode (`vf.TrainRunInfo.id/name`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunIdentity {
    /// `$PRL_RUN_ID`, or a fresh uuid4 hex for standalone runs.
    pub id: String,
    /// `$PRL_RUN_NAME`.
    pub name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CancelReason {
    Overload,
    Superseded,
    Shutdown,
}

impl CancelReason {
    pub fn as_str(self) -> &'static str {
        match self {
            CancelReason::Overload => "overload",
            CancelReason::Superseded => "superseded",
            CancelReason::Shutdown => "shutdown",
        }
    }
}

/// Terminal marker for a dropped group: one message covering every episode the
/// group still owed the sink (in-flight and never-dispatched).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupCancellation {
    pub kind: Kind,
    pub env_name: String,
    pub group_id: String,
    pub step: u64,
    pub count: usize,
    pub reason: CancelReason,
}

/// `vf.Error` as carried by a `DispatchFailure`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorInfo {
    #[serde(rename = "type")]
    pub type_: String,
    pub message: String,
    #[serde(default)]
    pub traceback: Option<String>,
    #[serde(default)]
    pub status_code: Option<i64>,
}

/// An environment request that failed before producing an episode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchFailure {
    pub kind: Kind,
    pub env_name: String,
    pub group_id: String,
    pub step: u64,
    pub policy_version: u64,
    pub task_type: String,
    pub task_key: String,
    pub task_hash: String,
    pub error: ErrorInfo,
}

/// Everything the dispatcher emits per dispatched attempt. Generic over the
/// episode type so this crate does not depend on the wire crate.
#[derive(Debug, Clone)]
pub enum DispatchResult<E> {
    Episode(E),
    Failure(DispatchFailure),
    Cancelled(GroupCancellation),
}

/// Broadcast pipeline facts (`tokio::sync::broadcast`). Components subscribe;
/// nobody references the supervisor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineEvent {
    /// Inference serves weights of `version` (written after the swap).
    PolicyAdvanced { version: u64 },
    /// The batch of `step` was sent to the trainer; `progress.step` is now `step + 1`.
    BatchShipped { step: u64 },
    /// Eval epochs fired for `envs` at `step`.
    EvalTriggered { step: u64, envs: Vec<String> },
    /// The final step shipped (or the budget was exceeded): no more train work.
    Draining { reason: String },
}

/// One task: `data` is shipped verbatim as the run request's `task_data`; `key`/`hash` are the
/// identity checked against the returned episode's `task`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskRecord {
    pub key: String,
    pub hash: String,
    pub data: serde_json::Value,
}

/// A bounded taskset comes back whole (`num_tasks = Some(len)`); an unbounded one has
/// `num_tasks = None`, no tasks, and is pulled with `Bridge::next_tasks`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadedTasks {
    pub num_tasks: Option<usize>,
    pub tasks: Vec<TaskRecord>,
}

/// Calls back into the Python process (`prime_rl.orchestrator.rust.PyBridge`), implemented by
/// prl-py. Only for work with no sane Rust equivalent.
#[async_trait::async_trait]
pub trait Bridge: Send + Sync + 'static {
    async fn load_tasks(&self, kind: Kind, env_name: &str) -> anyhow::Result<LoadedTasks>;
    /// Up to `n` more tasks of an unbounded taskset; fewer means the stream ended.
    async fn next_tasks(&self, kind: Kind, env_name: &str, n: usize) -> anyhow::Result<Vec<TaskRecord>>;
    /// `[bin_cost([n]) for n in 0..=max_len]` from the trainer model's FlopCounter.
    async fn flop_cost_table(&self, max_len: usize) -> anyhow::Result<Vec<i128>>;
    /// opsd: renderer token ids of a single system message (no generation prompt).
    async fn render_ids(&self, model_name: &str, renderer_config: serde_json::Value, system_content: &str)
        -> anyhow::Result<Vec<i32>>;
    /// echo: user filter masks for one trace (JSON-encoded trace).
    async fn echo_filter(&self, import_path: &str, kwargs: serde_json::Value, trace: serde_json::Value)
        -> anyhow::Result<Vec<Vec<bool>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_names_match_python() {
        let c = GroupCancellation {
            kind: Kind::Train,
            env_name: "e".into(),
            group_id: "g".into(),
            step: 3,
            count: 2,
            reason: CancelReason::Overload,
        };
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["kind"], "train");
        assert_eq!(v["reason"], "overload");
        let e = ErrorInfo { type_: "ValueError".into(), message: "m".into(), traceback: None, status_code: None };
        assert_eq!(serde_json::to_value(&e).unwrap()["type"], "ValueError");
        assert_eq!(Progress::default().step, 1);
    }
}
