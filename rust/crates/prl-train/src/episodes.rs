//! Episode views and provenance helpers (`orchestrator/utils.py`).

use std::sync::Arc;

use anyhow::{anyhow, Result};
use prl_wire::{Episode, PolicySpan, RunInfo, Trace, TrainRunInfo, WorkInfo};

/// An episode restricted to some of its traces — the Rust counterpart of Python's
/// `episode.model_copy(update={"traces": [...]})`, without copying the traces.
#[derive(Debug, Clone)]
pub struct EpisodeRef {
    pub episode: Arc<Episode>,
    trace_indices: Vec<usize>,
}

impl EpisodeRef {
    pub fn whole(episode: Arc<Episode>) -> Self {
        let trace_indices = (0..episode.traces.len()).collect();
        Self { episode, trace_indices }
    }

    pub fn subset(episode: Arc<Episode>, trace_indices: Vec<usize>) -> Self {
        Self { episode, trace_indices }
    }

    pub fn id(&self) -> &str {
        &self.episode.id
    }

    pub fn trace_indices(&self) -> &[usize] {
        &self.trace_indices
    }

    pub fn traces(&self) -> impl Iterator<Item = &Trace> + '_ {
        self.trace_indices.iter().map(|&i| &self.episode.traces[i])
    }

    pub fn num_traces(&self) -> usize {
        self.trace_indices.len()
    }

    pub fn first_trace(&self) -> &Trace {
        &self.episode.traces[self.trace_indices[0]]
    }

    /// An owned episode holding exactly the view's traces.
    pub fn to_episode(&self) -> Episode {
        self.with_traces(self.traces().cloned().collect())
    }

    /// An owned copy of the episode carrying `traces` instead of its own.
    pub fn with_traces(&self, traces: Vec<Trace>) -> Episode {
        let e = &self.episode;
        Episode {
            id: e.id.clone(),
            env: e.env.clone(),
            task: e.task.clone(),
            group: e.group.clone(),
            run: e.run.clone(),
            ok: e.ok,
            is_timeout: e.is_timeout,
            errors: e.errors.clone(),
            traces,
        }
    }
}

pub fn episode_env_name(episode: &Episode) -> Result<&str> {
    episode.env.name.as_deref().ok_or_else(|| anyhow!("Orchestrated episode is missing its environment name"))
}

pub fn episode_group_id(episode: &Episode) -> Result<&str> {
    episode.group.as_ref().map(|g| g.id.as_str()).ok_or_else(|| anyhow!("Orchestrated episode is missing its rollout group"))
}

/// The live policy span a train episode was generated under; `None` for frozen-sourced work.
pub fn train_policy(episode: &Episode) -> Result<Option<&PolicySpan>> {
    match &episode.run {
        Some(RunInfo::Train(TrainRunInfo { work: WorkInfo::Train { policy, .. }, .. })) => Ok(policy.as_ref()),
        _ => Err(anyhow!("Train episode is missing training-work provenance")),
    }
}
