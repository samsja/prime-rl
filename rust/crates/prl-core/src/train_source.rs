//! Training task selection across envs and the per-env curricula (`train_source.py`).

use anyhow::{ensure, Result};
use indexmap::IndexMap;
use prl_io::ckpt::{PyRandom, TrainSourceState};
use prl_types::{Metrics, TaskRecord};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curriculum::{AdvRangeGate, DifficultyPool, SamplerConfig, TaskFeed};
    use async_trait::async_trait;

    fn task(idx: usize) -> TaskRecord {
        TaskRecord { key: format!("key-{idx}"), hash: format!("hash-{idx}"), data: serde_json::json!({"idx": idx}) }
    }

    fn tasks(n: usize) -> Vec<TaskRecord> {
        (0..n).map(task).collect()
    }

    struct Counting(usize);

    #[async_trait]
    impl TaskFeed for Counting {
        async fn next_batch(&mut self, n: usize) -> Result<Vec<TaskRecord>> {
            let out = (self.0..(self.0 + n).min(10)).map(task).collect::<Vec<_>>();
            self.0 += out.len();
            Ok(out)
        }
    }

    /// Outcome of one `make_rollout(task, reward, advantages)` episode of the Python test.
    fn rollout(key: &str, reward: f64, advantages: Option<&[f32]>) -> GroupOutcome {
        GroupOutcome {
            task_keys: vec![Some(key.into())],
            rewards: vec![reward],
            advantages: advantages.map(<[f32]>::to_vec).unwrap_or_default(),
            any_trainable: advantages.is_some(),
        }
    }

    #[tokio::test]
    async fn default_curriculum_resumes_finite_and_infinite_tasksets() {
        let mut finite = Curriculum::new(CurriculumConfig::default(), Tasks::Finite(tasks(5))).unwrap();
        for _ in 0..3 {
            finite.next_task().await.unwrap();
        }
        let state = finite.state();
        let mut expected = Vec::new();
        for _ in 0..4 {
            expected.push(finite.next_task().await.unwrap().key);
        }
        let mut restored = Curriculum::new(CurriculumConfig::default(), Tasks::Finite(tasks(5))).unwrap();
        restored.load(&state).await.unwrap();
        for key in expected {
            assert_eq!(restored.next_task().await.unwrap().key, key);
        }

        let mut infinite = Curriculum::new(CurriculumConfig::default(), Tasks::Stream(Box::new(Counting(0)))).unwrap();
        infinite.next_task().await.unwrap();
        infinite.next_task().await.unwrap();
        let mut restored = Curriculum::new(CurriculumConfig::default(), Tasks::Stream(Box::new(Counting(0)))).unwrap();
        restored.load(&infinite.state()).await.unwrap();
        assert_eq!(restored.next_task().await.unwrap().key, "key-2");
    }

    #[test]
    fn curriculum_requires_unique_finite_task_keys() {
        let err = Curriculum::new(CurriculumConfig::default(), Tasks::Finite(vec![task(0), task(0)])).err().unwrap();
        assert!(err.to_string().contains("Task keys must be unique"));
    }

    #[tokio::test]
    async fn train_source_composes_sampler_and_all_gates_with_state_and_metrics() {
        let config = || CurriculumConfig {
            sampler: SamplerConfig::DifficultyPool {
                pools: [("all".to_string(), DifficultyPool { threshold: 1.0, weight: 1.0 })].into_iter().collect(),
                seed: 42,
            },
            gates: [
                ("reject".to_string(), AdvRangeGate { reject_min: 0.0, reject_max: 0.0 }),
                ("admit".to_string(), AdvRangeGate { reject_min: 0.5, reject_max: 0.5 }),
            ]
            .into_iter()
            .collect(),
        };
        let env = || TrainEnvTasks { name: "test".into(), ratio: 1.0, curriculum: config(), tasks: Tasks::Finite(tasks(3)) };
        let mut source = TrainSource::new(vec![env()]).unwrap();
        let sampled = source.next_task(1).await.unwrap().task.key;
        // group (reward 0.25, adv [0]) + failed episode (excluded) + empty (no turns, excluded)
        let mixed = GroupOutcome {
            task_keys: vec![Some(sampled.clone()); 3],
            rewards: vec![0.25],
            advantages: vec![0.0],
            any_trainable: true,
        };
        assert!(!source.on_result("test", &mixed).unwrap());
        let m = source.metrics();
        assert_eq!(
            m.into_iter().collect::<Vec<_>>(),
            vec![
                ("curriculum/test/admission_rate".to_string(), 0.0.into()),
                ("curriculum/test/sampler/pool/unseen".to_string(), 2.0.into()),
                ("curriculum/test/sampler/pool/all".to_string(), 1.0.into()),
            ]
        );
        let state = source.state();
        let mut restored = TrainSource::new(vec![env()]).unwrap();
        restored.load(&state).await.unwrap();
        let restored_state = restored.curricula["test"].state();
        assert!(matches!(
            restored_state.sampler,
            prl_io::ckpt::SamplerState::DifficultyPool { ref task_rewards, .. } if task_rewards[&sampled] == 0.25
        ));
        assert_eq!(restored_state.gates.keys().collect::<Vec<_>>(), vec!["reject", "admit"]);

        let empty = GroupOutcome { task_keys: vec![Some(sampled.clone())], ..Default::default() };
        assert!(!source.on_result("test", &empty).unwrap());
        assert_eq!(source.metrics()["curriculum/test/admission_rate"], 0.0.into());
        assert!(!source.on_result("test", &rollout(&sampled, 0.5, None)).unwrap());
        source.metrics();
        assert!(source.on_result("test", &rollout(&sampled, 0.75, Some(&[1.0]))).unwrap());
        assert_eq!(source.metrics()["curriculum/test/admission_rate"], 1.0.into());
    }

    #[tokio::test]
    async fn difficulty_pools_stack_with_advantage_gate_and_resume_sampling() {
        let config = || CurriculumConfig {
            sampler: SamplerConfig::DifficultyPool {
                pools: [
                    ("hard".to_string(), DifficultyPool { threshold: 0.25, weight: 0.0 }),
                    ("normal".to_string(), DifficultyPool { threshold: 0.75, weight: 1.0 }),
                    ("easy".to_string(), DifficultyPool { threshold: 1.0, weight: 0.0 }),
                ]
                .into_iter()
                .collect(),
                seed: 7,
            },
            gates: [("zero_advantage".to_string(), AdvRangeGate { reject_min: 0.0, reject_max: 0.0 })]
                .into_iter()
                .collect(),
        };
        let mut curriculum = Curriculum::new(config(), Tasks::Finite(tasks(3))).unwrap();
        let rewards = [0.1, 0.5, 0.9];
        let decisions: Vec<bool> = (0..3)
            .map(|i| {
                let adv = [if i > 0 { 1.0 } else { 0.0 }];
                curriculum.on_result(&rollout(&format!("key-{i}"), rewards[i], Some(&adv))).unwrap()
            })
            .collect();
        assert_eq!(decisions, vec![false, true, true]);
        assert_eq!(
            curriculum.metrics().into_iter().collect::<Vec<_>>(),
            vec![
                ("sampler/pool/unseen".to_string(), 0.0),
                ("sampler/pool/hard".to_string(), 1.0),
                ("sampler/pool/normal".to_string(), 1.0),
                ("sampler/pool/easy".to_string(), 1.0),
            ]
        );
        let state = curriculum.state();
        let mut expected = Vec::new();
        for _ in 0..10 {
            expected.push(curriculum.next_task().await.unwrap().key);
        }
        assert!(expected.iter().all(|k| k == "key-1"));
        let mut restored = Curriculum::new(config(), Tasks::Finite(tasks(3))).unwrap();
        restored.load(&state).await.unwrap();
        for key in expected {
            assert_eq!(restored.next_task().await.unwrap().key, key);
        }
    }

    #[test]
    fn advantage_range_gate_generalizes_zero_advantage_rejection() {
        let zero = AdvRangeGate { reject_min: 0.0, reject_max: 0.0 };
        assert!(!zero.admit(&rollout("k", 0.0, Some(&[0.0, 0.0]))));
        assert!(zero.admit(&rollout("k", 0.0, Some(&[0.0, 0.2]))));
        assert!(zero.admit(&rollout("k", 0.0, None)));
        let tolerance = AdvRangeGate { reject_min: -0.1, reject_max: 0.1 };
        assert!(!tolerance.admit(&rollout("k", 0.0, Some(&[-0.05, 0.0, 0.05]))));
        let positive = AdvRangeGate { reject_min: 0.5, reject_max: 0.5 };
        assert!(!positive.admit(&rollout("k", 0.0, Some(&[0.5]))));
    }

    #[tokio::test]
    async fn env_mix_and_pool_sampling_match_cpython_sequences() {
        // python: r = random.Random(42); [r.choices(['a','b'], weights=[1.0,3.0])[0] for _ in range(20)]
        let expected = "bababbbabaabaabbabba";
        let env = |name: &str, ratio| TrainEnvTasks {
            name: name.into(),
            ratio,
            curriculum: CurriculumConfig::default(),
            tasks: Tasks::Finite(tasks(2)),
        };
        let mut source = TrainSource::new(vec![env("a", 1.0), env("b", 3.0)]).unwrap();
        let mut got = String::new();
        for _ in 0..20 {
            got.push_str(&source.next_task(1).await.unwrap().env_name);
        }
        assert_eq!(got, expected);
        // python: r = random.Random(42); [r.choices(range(3), weights=[0.2,1.0,0.2])[0] for _ in range(20)]
        let expected = [1, 0, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0];
        let mut sampler = Curriculum::new(
            CurriculumConfig {
                sampler: SamplerConfig::DifficultyPool {
                    pools: [
                        ("hard".to_string(), DifficultyPool { threshold: 0.25, weight: 0.2 }),
                        ("normal".to_string(), DifficultyPool { threshold: 0.75, weight: 1.0 }),
                        ("easy".to_string(), DifficultyPool { threshold: 1.0, weight: 0.2 }),
                    ]
                    .into_iter()
                    .collect(),
                    seed: 42,
                },
                gates: IndexMap::new(),
            },
            Tasks::Finite(tasks(3)),
        )
        .unwrap();
        for (i, r) in [0.1, 0.5, 0.9].into_iter().enumerate() {
            sampler.on_result(&rollout(&format!("key-{i}"), r, Some(&[1.0]))).unwrap();
        }
        for idx in expected {
            assert_eq!(sampler.next_task().await.unwrap().key, format!("key-{idx}"));
        }
    }
}
