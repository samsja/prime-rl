//! Trigger-driven, finite-per-epoch pull of eval tasks. Port of `eval_source.py`.

use std::collections::{HashMap, HashSet, VecDeque};

use indexmap::IndexMap;
use prl_types::TaskRecord;

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

/// A previously completed episode a resumed eval may reuse (`plan_rollouts`).
pub trait SavedEpisode {
    fn id(&self) -> &str;
    /// The env this episode ran in (`env.name or env.id`).
    fn env_key(&self) -> &str;
    /// `task.hash`, or the hash of its task data when unset.
    fn task_hash(&self) -> String;
    fn set_group(&mut self, group_id: &str);
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
pub fn plan_rollouts<E: SavedEpisode>(
    tasks: &[TaskRecord],
    num_rollouts: usize,
    completed: Vec<E>,
) -> Vec<(TaskRecord, Vec<E>, usize)> {
    let mut saved: HashMap<String, VecDeque<E>> = HashMap::new();
    let mut seen: HashSet<String> = HashSet::new();
    for episode in completed {
        if !seen.insert(episode.id().to_owned()) {
            continue;
        }
        saved.entry(episode.task_hash()).or_default().push_back(episode);
    }
    tasks
        .iter()
        .map(|task| {
            let episodes = saved.entry(task.hash.clone()).or_default();
            let n = num_rollouts.min(episodes.len());
            let kept: Vec<E> = episodes.drain(..n).collect();
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

    pub fn group_size(&self, env_name: &str) -> usize {
        self.envs[env_name].group_size
    }

    /// Rollouts of one full epoch of `env_name`.
    pub fn epoch_size(&self, env_name: &str) -> usize {
        let env = &self.envs[env_name];
        env.tasks.len() * env.group_size
    }

    /// Fire eligible envs; returns their names and the saved episodes reused (with fresh
    /// group ids). `force` fires every env regardless of interval.
    pub fn trigger<E: SavedEpisode>(&mut self, step: u64, force: bool, completed: Vec<E>) -> (Vec<String>, Vec<E>) {
        let is_first = std::mem::replace(&mut self.first_trigger, false);
        if is_first && self.skip_first_step {
            return (Vec::new(), Vec::new());
        }
        let fired: Vec<String> = self
            .envs
            .values()
            .filter(|e| (is_first || force || step % e.interval == 0) && !e.tasks.is_empty())
            .map(|e| e.name.clone())
            .collect();
        let mut saved: HashMap<String, Vec<E>> = HashMap::new();
        for episode in completed {
            saved.entry(episode.env_key().to_owned()).or_default().push(episode);
        }
        let mut plans: Vec<std::vec::IntoIter<(TaskRecord, Vec<E>, usize)>> = fired
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
                    episode.set_group(&group_id);
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

    #[derive(Debug)]
    struct Saved {
        id: String,
        env: String,
        hash: String,
        group: Option<String>,
    }

    impl SavedEpisode for Saved {
        fn id(&self) -> &str {
            &self.id
        }
        fn env_key(&self) -> &str {
            &self.env
        }
        fn task_hash(&self) -> String {
            self.hash.clone()
        }
        fn set_group(&mut self, group_id: &str) {
            self.group = Some(group_id.into());
        }
    }

    fn source(skip_first: bool, resumed: bool) -> EvalSource {
        EvalSource::new(
            vec![
                EvalEnvTasks { name: "a".into(), tasks: vec![task("a1"), task("a2"), task("a3")], group_size: 2, interval: 2 },
                EvalEnvTasks { name: "b".into(), tasks: vec![task("b1")], group_size: 4, interval: 3 },
                EvalEnvTasks { name: "empty".into(), tasks: vec![], group_size: 1, interval: 1 },
            ],
            skip_first,
            resumed,
        )
    }

    fn drain(s: &mut EvalSource) -> Vec<(String, String, Option<usize>)> {
        std::iter::from_fn(|| s.next_task()).map(|r| (r.env_name, r.task.key, r.rollouts)).collect()
    }

    #[test]
    fn first_trigger_fires_all_round_robin() {
        let mut s = source(false, false);
        let (fired, _) = s.trigger::<Saved>(0, false, vec![]);
        assert_eq!(fired, vec!["a", "b"]);
        assert_eq!(
            drain(&mut s),
            vec![
                ("a".into(), "a1".into(), Some(2)),
                ("b".into(), "b1".into(), Some(4)),
                ("a".into(), "a2".into(), Some(2)),
                ("a".into(), "a3".into(), Some(2)),
            ]
        );
        assert_eq!(s.trigger::<Saved>(3, false, vec![]).0, vec!["b"]);
        assert_eq!(s.trigger::<Saved>(4, false, vec![]).0, vec!["a"]);
        assert!(s.trigger::<Saved>(5, false, vec![]).0.is_empty());
        assert_eq!(s.trigger::<Saved>(5, true, vec![]).0, vec!["a", "b"]);
    }

    #[test]
    fn skip_first_step_and_resume() {
        let mut s = source(true, false);
        assert!(s.trigger::<Saved>(0, false, vec![]).0.is_empty());
        assert!(s.is_empty());
        assert_eq!(s.trigger::<Saved>(6, false, vec![]).0, vec!["a", "b"]);
        let mut s = source(false, true);
        assert!(s.trigger::<Saved>(1, false, vec![]).0.is_empty());
    }

    #[test]
    fn saved_episodes_fill_groups_and_cancel_step() {
        let mut s = source(false, false);
        let saved = vec![
            Saved { id: "1".into(), env: "a".into(), hash: "h-a1".into(), group: None },
            Saved { id: "2".into(), env: "a".into(), hash: "h-a1".into(), group: None },
            Saved { id: "2".into(), env: "a".into(), hash: "h-a1".into(), group: None },
            Saved { id: "3".into(), env: "b".into(), hash: "h-b1".into(), group: None },
        ];
        let (_, restored) = s.trigger(0, false, saved);
        assert_eq!(restored.len(), 3);
        assert_eq!(restored[0].group, restored[1].group);
        let reqs: Vec<_> = std::iter::from_fn(|| s.next_task()).collect();
        // a1 fully restored -> no request; b1 needs 3 more in the restored group
        assert_eq!(reqs.iter().map(|r| r.task.key.as_str()).collect::<Vec<_>>(), vec!["b1", "a2", "a3"]);
        assert_eq!(reqs[0].rollouts, Some(3));
        assert_eq!(reqs[0].group_id, restored[2].group);

        s.trigger::<Saved>(4, false, vec![]);
        s.trigger::<Saved>(6, false, vec![]);
        let cancelled = s.cancel_step(4);
        assert_eq!(cancelled.len(), 3);
        assert_eq!(s.len(), 4);
    }
}
