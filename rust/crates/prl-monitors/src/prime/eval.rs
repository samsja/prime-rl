//! `PrimeEvalMonitor`: each eval epoch streamed to the platform as one evaluation per env and
//! step — opened when the epoch is counted (`log_eval_plan`), fed every episode as it lands, and
//! finished with the epoch's aggregates (`log_eval_epoch`). Under a managed launch (`$RUN_ID`)
//! the epoch metrics also go to the parent training run.

use std::path::PathBuf;

use anyhow::{bail, Result};
use async_trait::async_trait;
use indexmap::IndexMap;
use prl_types::{Kind, Subset};
use serde_json::{json, Map, Value as Json};

use super::backend::{RunKind, RunSpec, RunStatus};
use super::run::{Mode, Run, RunOptions};
use super::{base_url_override, metrics_json, prime_mode, FINISH_TIMEOUT};
use crate::pyjson::{py_sum, Row};
use crate::records::{
    platform_run_path, publish_link, register_links, update_record, PRIME_EVAL_LINKS,
};
use crate::{EpisodeRef, Monitor};

pub const EVAL_ID_VAR: &str = "PRIME_RUNS_EVAL_ID";

/// One eval source as the platform evaluation is registered with it.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalSourceSpec {
    pub resolved_name: String,
    pub taskset_id: String,
    pub group_size: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrimeEvalSetup {
    /// `PrimeEvalMonitorConfig.name`.
    pub name: Option<String>,
    /// The evaluated model (`"unknown"` without a config).
    pub model: String,
    pub sources: Vec<EvalSourceSpec>,
}

/// Run-level aggregates over episodes (`prime_runs.metrics.from_episodes`): rewards and metrics
/// over the trainable traces (all traces when none is trainable), the share of failed episodes.
pub fn summary_from_episodes(episodes: &[EpisodeRef]) -> Map<String, Json> {
    let traces: Vec<&prl_wire::Trace> = episodes.iter().flat_map(|e| e.traces.iter()).collect();
    let trainable: Vec<&prl_wire::Trace> = traces
        .iter()
        .copied()
        .filter(|t| t.agent.trainable)
        .collect();
    let scored = if trainable.is_empty() {
        traces
    } else {
        trainable
    };
    let mut sums: IndexMap<&str, (f64, u64)> = IndexMap::new();
    for trace in &scored {
        let mut values: IndexMap<&str, f64> = IndexMap::new();
        for (name, reward) in &trace.rewards {
            if let Some(reward) = reward {
                values.insert(name, reward.score);
            }
        }
        for (name, value) in &trace.metrics {
            if let Some(value) = value {
                values.insert(name, *value);
            }
        }
        for (name, value) in values {
            let entry = sums.entry(name).or_insert((0.0, 0));
            entry.0 += value;
            entry.1 += 1;
        }
    }
    let n = scored.len();
    let avg_error = if episodes.is_empty() {
        0.0
    } else {
        episodes.iter().filter(|e| !e.ok).count() as f64 / episodes.len() as f64
    };
    // Trace.reward and the mean are Python sums
    let reward = |t: &prl_wire::Trace| py_sum(t.rewards.values().flatten().map(|r| r.value()));
    let avg_reward = if n > 0 {
        py_sum(scored.iter().map(|t| reward(t))) / n as f64
    } else {
        0.0
    };
    let avg_metrics: Map<String, Json> = sums
        .into_iter()
        .map(|(name, (sum, count))| (name.to_string(), json!(sum / count as f64)))
        .collect();
    json!({"avg_reward": avg_reward, "avg_metrics": avg_metrics, "avg_error": avg_error})
        .as_object()
        .expect("object")
        .clone()
}

pub struct PrimeEvalMonitor {
    mode: Mode,
    name: Option<String>,
    model: String,
    sources: IndexMap<String, EvalSourceSpec>,
    run_id: Option<String>,
    output_dir: Option<PathBuf>,
    /// `(env, step)` -> its open evaluation; None once opening it failed.
    runs: IndexMap<(String, u64), Option<Run>>,
    parent: Option<Run>,
    evaluation_id: Option<String>,
}

impl PrimeEvalMonitor {
    pub async fn init(setup: PrimeEvalSetup, output_dir: Option<PathBuf>) -> Result<Self> {
        let mode = prime_mode()?;
        let sources: IndexMap<String, EvalSourceSpec> = setup
            .sources
            .into_iter()
            .map(|s| (s.resolved_name.clone(), s))
            .collect();
        let evaluation_id = std::env::var(EVAL_ID_VAR).ok().filter(|v| !v.is_empty());
        if evaluation_id.is_some() && sources.len() != 1 {
            bail!(
                "${EVAL_ID_VAR} names one platform evaluation, so the run needs exactly one eval source (got {})",
                sources.len()
            );
        }
        let run_id = std::env::var("PRL_RUN_ID").ok();
        let mut monitor = PrimeEvalMonitor {
            mode,
            name: setup.name,
            model: setup.model,
            sources,
            run_id: run_id.clone(),
            output_dir: output_dir.clone(),
            runs: IndexMap::new(),
            parent: None,
            evaluation_id,
        };
        if mode == Mode::Online {
            tracing::info!("Streaming eval epochs to the Prime platform");
            monitor.parent = monitor.attach_parent().await?;
            if let Some(output_dir) = &output_dir {
                let path = platform_run_path(output_dir);
                let run_id = run_id.clone();
                tokio::task::spawn_blocking(move || {
                    update_record(&path, |mut record| {
                        if record.get("kind").and_then(Json::as_str) == Some("train") {
                            record.entry("evaluations").or_insert(json!({}));
                            record.insert("run_id".into(), run_id.into());
                            record
                        } else {
                            json!({"kind": "eval", "run_id": run_id, "evaluations": {}})
                                .as_object()
                                .expect("object")
                                .clone()
                        }
                    })
                })
                .await?;
            }
        } else {
            tracing::info!("Prime evaluations disabled");
        }
        if let Some(output_dir) = output_dir {
            let expected = if mode == Mode::Online {
                monitor.sources.len() as u64
            } else {
                0
            };
            tokio::task::spawn_blocking(move || {
                register_links(&output_dir, PRIME_EVAL_LINKS, expected)
            })
            .await?;
        }
        Ok(monitor)
    }

    /// The parent training run, for metrics only. Its lifecycle belongs to the trainer: this
    /// process never reports its status (no crash hook).
    async fn attach_parent(&self) -> Result<Option<Run>> {
        let Some(run_id) = std::env::var("RUN_ID").ok().filter(|v| !v.is_empty()) else {
            return Ok(None);
        };
        let run = Run::init(RunOptions {
            kind: RunKind::Train,
            mode: self.mode,
            base_url: base_url_override(),
            finish_timeout: FINISH_TIMEOUT,
            id: Some(run_id),
            spec: RunSpec::default(),
        })
        .await?;
        tracing::info!("Logging eval metrics to parent training run {}", run.id);
        Ok(Some(run))
    }

    async fn open(&self, env_name: &str, step: u64, expected: Option<u64>) -> Result<Run> {
        let options = |id: Option<String>, spec: RunSpec| RunOptions {
            kind: RunKind::Eval,
            mode: self.mode,
            base_url: base_url_override(),
            finish_timeout: FINISH_TIMEOUT,
            id,
            spec,
        };
        if let Some(id) = &self.evaluation_id {
            if !self.runs.is_empty() {
                bail!("${EVAL_ID_VAR} holds one epoch, and it already took one");
            }
            return Ok(Run::init(options(Some(id.clone()), RunSpec::default())).await?);
        }
        let source = self
            .sources
            .get(env_name)
            .ok_or_else(|| anyhow::anyhow!("KeyError: '{env_name}'"))?;
        let base = self.name.clone().unwrap_or_else(|| "None".into());
        let name = if self.sources.len() == 1 {
            self.name.clone()
        } else {
            Some(format!("{base}--{env_name}"))
        };
        let num_examples = expected.filter(|e| *e > 0).map(|e| e / source.group_size);
        let config = json!({
            "model": self.model,
            "step": step,
            "run_id": self.run_id,
            "num_examples": num_examples,
            "rollouts_per_example": source.group_size,
        });
        let spec = RunSpec {
            name,
            environments: vec![source.taskset_id.clone()],
            model: Some(self.model.clone()),
            framework: Some("prime-rl".into()),
            config: config.as_object().expect("object").clone(),
            ..Default::default()
        };
        Ok(Run::init(options(None, spec)).await?)
    }

    /// The epoch's evaluation, opened on first use.
    async fn run_for(
        &mut self,
        env_name: &str,
        step: u64,
        expected: Option<u64>,
    ) -> Option<&mut Run> {
        let key = (env_name.to_string(), step);
        if !self.runs.contains_key(&key) {
            let opened = self.open(env_name, step, expected).await;
            match opened {
                Err(e) => {
                    tracing::warn!("Failed to open the {env_name} (Step {step}) evaluation: {e:#}");
                    self.runs.insert(key.clone(), None);
                    if let Some(output_dir) = self.output_dir.clone() {
                        let env = env_name.to_string();
                        let _ = tokio::task::spawn_blocking(move || {
                            publish_link(&output_dir, PRIME_EVAL_LINKS, &env, None)
                        })
                        .await;
                    }
                }
                Ok(run) => {
                    if let Some(url) = &run.url {
                        let attached = if run.attached {
                            format!(" (attached via ${EVAL_ID_VAR})")
                        } else {
                            String::new()
                        };
                        tracing::info!(
                            "Streaming {env_name} (Step {step}) evaluation - {url}{attached}"
                        );
                    }
                    if let Some(output_dir) = self.output_dir.clone() {
                        let (env, id, url, run_id) = (
                            env_name.to_string(),
                            run.id.clone(),
                            run.url.clone(),
                            self.run_id.clone(),
                        );
                        let _ = tokio::task::spawn_blocking(move || {
                            if let Some(url) = &url {
                                update_record(&platform_run_path(&output_dir), |mut record| {
                                    record.entry("kind").or_insert("eval".into());
                                    record.entry("run_id").or_insert(run_id.into());
                                    let evaluations =
                                        record.entry("evaluations").or_insert(json!({}));
                                    if let Some(evaluations) = evaluations.as_object_mut() {
                                        evaluations.insert(
                                            env.clone(),
                                            json!({"step": step, "id": id, "url": url}),
                                        );
                                    }
                                    record
                                });
                            }
                            publish_link(&output_dir, PRIME_EVAL_LINKS, &env, url.as_deref());
                        })
                        .await;
                    }
                    self.runs.insert(key.clone(), Some(run));
                }
            }
        }
        self.runs.get_mut(&key).and_then(Option::as_mut)
    }
}

#[async_trait]
impl Monitor for PrimeEvalMonitor {
    fn name(&self) -> &'static str {
        "PrimeEvalMonitor"
    }

    /// An epoch's aggregates onto the parent training run, if any.
    async fn log_metrics(&mut self, metrics: &Row, step: Option<u64>) -> Result<()> {
        if let Some(parent) = &mut self.parent {
            parent.log_metrics(metrics_json(metrics), step).await;
        }
        Ok(())
    }

    async fn log_eval_plan(&mut self, env_name: &str, step: u64, expected: u64) -> Result<()> {
        self.run_for(env_name, step, Some(expected)).await;
        Ok(())
    }

    /// Every eval episode as it lands (the `all` subset).
    async fn log_episodes(
        &mut self,
        episodes: &[EpisodeRef],
        step: u64,
        kind: Kind,
        subset: Subset,
    ) -> Result<()> {
        if kind != Kind::Eval || subset != Subset::All {
            return Ok(());
        }
        let mut by_env: IndexMap<String, Vec<EpisodeRef>> = IndexMap::new();
        for episode in episodes {
            if let Some(name) = &episode.env.name {
                by_env
                    .entry(name.clone())
                    .or_default()
                    .push(episode.clone());
            }
        }
        for (env_name, batch) in by_env {
            if let Some(run) = self.run_for(&env_name, step, None).await {
                run.log_episodes(batch).await;
            }
        }
        Ok(())
    }

    async fn log_eval_epoch(
        &mut self,
        env_name: &str,
        step: u64,
        episodes: &[EpisodeRef],
    ) -> Result<()> {
        let episodes = episodes.to_vec();
        let summary = tokio::task::spawn_blocking(move || summary_from_episodes(&episodes)).await?;
        let Some(run) = self.run_for(env_name, step, None).await else {
            return Ok(());
        };
        run.finish(Some(summary), RunStatus::Completed, None).await;
        tracing::info!(
            "Uploaded {env_name} (Step {step}) evaluation - {}",
            run.url.as_deref().unwrap_or("None")
        );
        Ok(())
    }

    /// An epoch the run did not finish leaves its evaluation open: close it as cancelled.
    async fn finalize(&mut self, _paused: bool) -> Result<()> {
        for run in self.runs.values_mut().flatten() {
            if !run.finished {
                run.finish(None, RunStatus::Cancelled, Some("interrupted".into()))
                    .await;
            }
        }
        if let Some(parent) = &self.parent {
            // Drain only: the parent run belongs to the trainer and the platform.
            parent.flush(FINISH_TIMEOUT).await;
        }
        Ok(())
    }

    /// The SDK's atexit hook on each open evaluation (the parent has none).
    async fn crash(&mut self) -> Result<()> {
        for run in self.runs.values_mut().flatten() {
            if !run.finished {
                tracing::warn!("Run {} was never finished; reporting it as crashed", run.id);
                run.finish(
                    None,
                    RunStatus::Crashed,
                    Some("process exited without finishing the run".into()),
                )
                .await;
            }
        }
        if let Some(parent) = &self.parent {
            parent.flush(FINISH_TIMEOUT).await;
        }
        Ok(())
    }
}
