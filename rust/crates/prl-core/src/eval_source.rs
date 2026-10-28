//! Trigger-driven, finite-per-epoch pull of eval tasks. Port of `eval_source.py`.

use std::collections::{HashMap, HashSet, VecDeque};

use indexmap::IndexMap;
use prl_types::TaskRecord;
use prl_wire::{Episode, GroupInfo};

/// A task picked by a source with its pinned run step.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskRequest {
    pub env_name: String,
    pub task: TaskRecord,
    pub step: u64,
    /// Rollouts this request asks for; None is the env's group size.
    pub rollouts: Option<usize>,
    /// The group these rollouts join; None mints one.
    pub group_id: Option<String>,
}

/// One eval env as the source sees it.
#[derive(Debug, Clone)]
pub struct EvalEnvTasks {
    pub name: String,
    pub tasks: Vec<TaskRecord>,
    pub group_size: usize,
    /// Step interval of online evals; standalone evals use 1.
    pub interval: u64,
}

#[derive(Debug)]
pub struct EvalSource {
    skip_first_step: bool,
    envs: IndexMap<String, EvalEnvTasks>,
    queue: VecDeque<TaskRequest>,
    first_trigger: bool,
}

/// `verifiers.v1.utils.eval.plan_rollouts`: assign distinct saved episodes to task
/// occurrences by content hash; each gets up to `num_rollouts` and the count still missing.
fn plan_rollouts(
    tasks: &[TaskRecord],
    num_rollouts: usize,
    completed: Vec<Episode>,
) -> Vec<(TaskRecord, Vec<Episode>, usize)> {
    let mut saved: HashMap<String, VecDeque<Episode>> = HashMap::new();
    let mut seen: HashSet<String> = HashSet::new();
    for episode in completed {
        if !seen.insert(episode.id.clone()) {
            continue;
        }
        let hash = episode.task.hash.clone().unwrap_or_else(|| prl_wire::task_hash(&episode.task.data));
        saved.entry(hash).or_default().push_back(episode);
    }
    tasks
        .iter()
        .map(|task| {
            let episodes = saved.entry(task.hash.clone()).or_default();
            let kept: Vec<Episode> = episodes.drain(..num_rollouts.min(episodes.len())).collect();
            let missing = num_rollouts - kept.len();
            (task.clone(), kept, missing)
        })
        .collect()
}

impl EvalSource {
    pub fn new(envs: Vec<EvalEnvTasks>, skip_first_step: bool, is_resumed: bool) -> Self {
        Self {
            skip_first_step,
            envs: envs.into_iter().map(|e| (e.name.clone(), e)).collect(),
            queue: VecDeque::new(),
            // A fresh run evaluates the base policy; resumed runs apply interval rules.
            first_trigger: !is_resumed,
        }
    }

    /// Fire eligible envs; returns their names and the saved episodes reused (with fresh
    /// group ids). `force` fires every env regardless of interval.
    pub fn trigger(&mut self, step: u64, force: bool, completed: Vec<Episode>) -> (Vec<String>, Vec<Episode>) {
        let is_first = std::mem::replace(&mut self.first_trigger, false);
        if is_first && self.skip_first_step {
            return (Vec::new(), Vec::new());
        }
        let fired: Vec<String> = self
            .envs
            .values()
            .filter(|e| (is_first || force || step.is_multiple_of(e.interval)) && !e.tasks.is_empty())
            .map(|e| e.name.clone())
            .collect();
        let mut saved: HashMap<String, Vec<Episode>> = HashMap::new();
        for episode in completed {
            let env = episode.env.name.clone().unwrap_or_else(|| episode.env.id.clone());
            saved.entry(env).or_default().push(episode);
        }
        let mut plans: Vec<_> = fired
            .iter()
            .map(|name| {
                let env = &self.envs[name];
                plan_rollouts(&env.tasks, env.group_size, saved.remove(name).unwrap_or_default()).into_iter()
            })
            .collect();
        // Round-robin across fired envs (A1, B1, A2, ...).
        let mut restored = Vec::new();
        loop {
            let mut any = false;
            for (env_name, plan) in fired.iter().zip(plans.iter_mut()) {
                let Some((task, mut kept, rollouts)) = plan.next() else { continue };
                any = true;
                let group_id = uuid::Uuid::new_v4().to_string();
                for episode in &mut kept {
                    episode.group = Some(GroupInfo { id: group_id.clone() });
                }
                restored.extend(kept);
                if rollouts > 0 {
                    self.queue.push_back(TaskRequest {
                        env_name: env_name.clone(),
                        task,
                        step,
                        rollouts: Some(rollouts),
                        group_id: Some(group_id),
                    });
                }
            }
            if !any {
                break;
            }
        }
        (fired, restored)
    }

    pub fn next_task(&mut self) -> Option<TaskRequest> {
        self.queue.pop_front()
    }

    /// Remove and return queued requests of a superseded eval step.
    pub fn cancel_step(&mut self, step: u64) -> Vec<TaskRequest> {
        let (cancelled, kept): (Vec<_>, Vec<_>) = self.queue.drain(..).partition(|r| r.step == step);
        self.queue = kept.into();
        cancelled
    }

    pub fn queued(&self) -> impl Iterator<Item = &TaskRequest> {
        self.queue.iter()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(key: &str) -> TaskRecord {
        TaskRecord { key: key.into(), hash: format!("h-{key}"), data: serde_json::json!({"idx": 0}) }
    }

    fn source(skip_first: bool, resumed: bool) -> EvalSource {
        let env = |name: &str, tasks: Vec<TaskRecord>, group_size, interval| EvalEnvTasks {
            name: name.into(),
            tasks,
            group_size,
            interval,
        };
        EvalSource::new(
            vec![env("a", vec![task("a1"), task("a2")], 2, 2), env("b", vec![task("b1")], 4, 3), env("empty", vec![], 1, 1)],
            skip_first,
            resumed,
        )
    }

    #[test]
    fn triggers_follow_intervals_and_round_robin() {
        let mut s = source(false, false);
        assert_eq!(s.trigger(0, false, vec![]).0, vec!["a", "b"]);
        let queued: Vec<_> = std::iter::from_fn(|| s.next_task()).map(|r| (r.task.key, r.rollouts)).collect();
        assert_eq!(queued, vec![("a1".into(), Some(2)), ("b1".into(), Some(4)), ("a2".into(), Some(2))]);
        assert_eq!(s.trigger(3, false, vec![]).0, vec!["b"]);
        assert!(s.trigger(5, false, vec![]).0.is_empty());
        assert_eq!(s.trigger(5, true, vec![]).0, vec!["a", "b"]);
        assert_eq!(s.cancel_step(5).len(), 3);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn skip_first_step_and_resume_skip_the_base_eval() {
        let mut s = source(true, false);
        assert!(s.trigger(0, false, vec![]).0.is_empty());
        assert_eq!(s.trigger(6, false, vec![]).0, vec!["a", "b"]);
        assert!(source(false, true).trigger(1, false, vec![]).0.is_empty());
    }
}
