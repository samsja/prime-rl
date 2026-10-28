//! Evaluation-side epoch assembly. Port of `eval_sink.py`; generic over the episode type so
//! the eval runner and the orchestrator share it.

use indexmap::IndexMap;
use prl_types::{DispatchFailure, GroupCancellation, Kind};

/// One env's completed eval epoch: every returned episode (errored included), the requests
/// that returned no episode, and the cancelled count.
#[derive(Debug)]
pub struct EvalEpoch<E> {
    pub env_name: String,
    pub step: u64,
    pub episodes: Vec<E>,
    pub failures: Vec<DispatchFailure>,
    pub cancelled: usize,
}

#[derive(Debug)]
struct Pending<E> {
    episodes: Vec<E>,
    failures: Vec<DispatchFailure>,
    cancelled: usize,
}

impl<E> Default for Pending<E> {
    fn default() -> Self {
        Self { episodes: Vec::new(), failures: Vec::new(), cancelled: 0 }
    }
}

impl<E> Pending<E> {
    fn size(&self) -> usize {
        self.episodes.len() + self.failures.len() + self.cancelled
    }
}

#[derive(Debug)]
pub struct EvalSink<E> {
    /// Rollouts per epoch by env: examples x group size.
    epoch_sizes: IndexMap<String, usize>,
    pending: IndexMap<(String, u64), Pending<E>>,
}

impl<E> EvalSink<E> {
    pub fn new(epoch_sizes: IndexMap<String, usize>) -> Self {
        Self { epoch_sizes, pending: IndexMap::new() }
    }

    pub fn add(&mut self, env_name: &str, step: u64, episode: E) -> Option<EvalEpoch<E>> {
        let key = (env_name.to_owned(), step);
        self.pending.entry(key.clone()).or_default().episodes.push(episode);
        self.complete(key)
    }

    pub fn fail(&mut self, failure: DispatchFailure) -> anyhow::Result<Option<EvalEpoch<E>>> {
        anyhow::ensure!(failure.kind == Kind::Eval, "EvalSink cannot process a {} dispatch failure", failure.kind);
        let key = (failure.env_name.clone(), failure.step);
        self.pending.entry(key.clone()).or_default().failures.push(failure);
        Ok(self.complete(key))
    }

    pub fn cancel(&mut self, cancellation: &GroupCancellation) -> anyhow::Result<Option<EvalEpoch<E>>> {
        anyhow::ensure!(
            cancellation.kind == Kind::Eval,
            "EvalSink cannot process a {} group cancellation",
            cancellation.kind
        );
        let key = (cancellation.env_name.clone(), cancellation.step);
        self.pending.entry(key.clone()).or_default().cancelled += cancellation.count;
        Ok(self.complete(key))
    }

    fn complete(&mut self, key: (String, u64)) -> Option<EvalEpoch<E>> {
        let size = self.pending.get(&key).map_or(0, Pending::size);
        if size < self.epoch_sizes[&key.0] {
            return None;
        }
        let pending = self.pending.shift_remove(&key).unwrap_or_default();
        Some(EvalEpoch {
            env_name: key.0,
            step: key.1,
            episodes: pending.episodes,
            failures: pending.failures,
            cancelled: pending.cancelled,
        })
    }

    /// `(env, step, arrived, expected)` per epoch in progress.
    pub fn batch_progress(&self) -> Vec<(String, u64, usize, usize)> {
        self.pending
            .iter()
            .map(|((env, step), p)| (env.clone(), *step, p.size(), self.epoch_sizes[env]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prl_types::{CancelReason, ErrorInfo};

    #[test]
    fn epoch_completes_on_episodes_failures_and_cancellations() {
        let mut sink: EvalSink<u32> = EvalSink::new([("a".to_string(), 4)].into_iter().collect());
        assert!(sink.add("a", 2, 1).is_none());
        let failure = DispatchFailure {
            kind: Kind::Eval,
            env_name: "a".into(),
            group_id: "g".into(),
            step: 2,
            policy_version: 1,
            task_type: "Task".into(),
            task_key: "k".into(),
            task_hash: "h".into(),
            error: ErrorInfo { type_: "E".into(), message: "m".into(), traceback: None, status_code: None },
        };
        assert!(sink.fail(failure).unwrap().is_none());
        assert!(sink.add("a", 4, 9).is_none());
        assert_eq!(sink.batch_progress(), vec![("a".into(), 2, 2, 4), ("a".into(), 4, 1, 4)]);
        let c = GroupCancellation {
            kind: Kind::Eval,
            env_name: "a".into(),
            group_id: "g".into(),
            step: 2,
            count: 2,
            reason: CancelReason::Superseded,
        };
        let epoch = sink.cancel(&c).unwrap().unwrap();
        assert_eq!((epoch.step, epoch.episodes, epoch.failures.len(), epoch.cancelled), (2, vec![1], 1, 2));
        assert_eq!(sink.batch_progress().len(), 1);
        let train = GroupCancellation { kind: Kind::Train, ..c };
        assert!(sink.cancel(&train).is_err());
    }
}
