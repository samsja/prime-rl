//! `PrimeEvalMonitor`: each eval epoch streamed to the platform as one evaluation per env and
//! step, opened when the epoch is counted (`log_eval_plan`), fed every episode as it lands and
//! finished with the epoch's aggregates (`log_eval_epoch`). Under a managed launch (`$RUN_ID`)
//! the epoch metrics also go to the parent training run.

use std::path::PathBuf;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use indexmap::IndexMap;
use prl_types::{Kind, Subset};
use serde_json::{json, Map, Value as Json};

use super::backend::{RunKind, RunSpec, RunStatus};
use super::run::{Run, FINISH_TIMEOUT};
use super::{metrics_json, online};
use crate::file::index::py_sum;
use crate::records::{
    platform_run_path, publish_link, register_links, update_record, PRIME_EVAL_LINKS,
};
use crate::{EpisodeRef, Monitor, Row};

const EVAL_ID_VAR: &str = "PRIME_RUNS_EVAL_ID";

/// One eval source as its platform evaluation is registered.
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
    /// The evaluated model.
    pub model: String,
    pub sources: Vec<EvalSourceSpec>,
}

/// Run-level aggregates over episodes (`prime_runs.metrics.from_episodes`): rewards and metrics
/// over the trainable traces (all traces when none is), and the share of failed episodes.
pub fn summary_from_episodes(episodes: &[EpisodeRef]) -> Map<String, Json> {
    let traces: Vec<&prl_wire::Trace> = episodes.iter().flat_map(|e| &e.traces).collect();
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
        let scores = trace
            .rewards
            .iter()
            .filter_map(|(name, r)| Some((name.as_str(), r.as_ref()?.score)));
        let metrics = trace
            .metrics
            .iter()
            .filter_map(|(name, v)| Some((name.as_str(), (*v)?)));
        // `{**scores, **metrics}`: a metric named like a reward replaces it
        let values: IndexMap<&str, f64> = scores.chain(metrics).collect();
        for (name, value) in values {
            let entry = sums.entry(name).or_default();
            *entry = (entry.0 + value, entry.1 + 1);
        }
    }
    let mean = |total: f64, n: usize| if n > 0 { total / n as f64 } else { 0.0 };
    let reward = |t: &&prl_wire::Trace| py_sum(t.rewards.values().flatten().map(|r| r.value()));
    let avg_metrics: Map<String, Json> = sums
        .into_iter()
        .map(|(name, (sum, n))| (name.to_string(), json!(sum / n as f64)))
        .collect();
    let failed = episodes.iter().filter(|e| !e.ok).count();
    let summary = json!({
        "avg_reward": mean(py_sum(scored.iter().map(reward)), scored.len()),
        "avg_metrics": avg_metrics,
        "avg_error": mean(failed as f64, episodes.len()),
    });
    summary.as_object().expect("object").clone()
}

pub struct PrimeEvalMonitor {
    online: bool,
    name: Option<String>,
    model: String,
    sources: IndexMap<String, EvalSourceSpec>,
    run_id: Option<String>,
    output_dir: PathBuf,
    /// `(env, step)` -> its evaluation; None once opening it failed (no retry per episode).
    runs: IndexMap<(String, u64), Option<Run>>,
    /// The parent training run, for metrics only: its lifecycle belongs to the trainer.
    parent: Option<Run>,
    evaluation_id: Option<String>,
}

impl PrimeEvalMonitor {
    pub async fn init(setup: PrimeEvalSetup, output_dir: PathBuf) -> Result<Self> {
        let online = online()?;
        let sources: IndexMap<String, EvalSourceSpec> = setup
            .sources
            .into_iter()
            .map(|s| (s.resolved_name.clone(), s))
            .collect();
        let evaluation_id = std::env::var(EVAL_ID_VAR).ok().filter(|v| !v.is_empty());
        if evaluation_id.is_some() && sources.len() != 1 {
            bail!("${EVAL_ID_VAR} names one platform evaluation, so the run needs exactly one eval source (got {})", sources.len());
        }
        let run_id = std::env::var("PRL_RUN_ID").ok();
        let mut parent = None;
        if online {
            tracing::info!("Streaming eval epochs to the Prime platform");
            if let Some(id) = std::env::var("RUN_ID").ok().filter(|v| !v.is_empty()) {
                let run = Run::init(RunKind::Train, true, Some(id), RunSpec::default())
                    .await
                    .map_err(|e| anyhow!(e.describe()))?;
                tracing::info!("Logging eval metrics to parent training run {}", run.id);
                parent = Some(run);
            }
            // a train record (an eval process sharing the trainer's run dir) keeps its link
            let (path, run_id) = (platform_run_path(&output_dir), run_id.clone());
            tokio::task::spawn_blocking(move || {
                update_record(&path, |mut record| {
                    if record.get("kind").and_then(Json::as_str) != Some("train") {
                        return json!({"kind": "eval", "run_id": run_id, "evaluations": {}});
                    }
                    record
                        .as_object_mut()
                        .expect("records are objects")
                        .entry("evaluations")
                        .or_insert(json!({}));
                    record["run_id"] = run_id.into();
                    record
                })
            })
            .await?;
        } else {
            tracing::info!("Prime evaluations disabled");
        }
        let (dir, expected) = (
            output_dir.clone(),
            if online { sources.len() as u64 } else { 0 },
        );
        tokio::task::spawn_blocking(move || register_links(&dir, PRIME_EVAL_LINKS, expected))
            .await?;
        Ok(PrimeEvalMonitor {
            online,
            name: setup.name,
            model: setup.model,
            sources,
            run_id,
            output_dir,
            runs: IndexMap::new(),
            parent,
            evaluation_id,
        })
    }

    async fn open(&self, env_name: &str, step: u64, expected: Option<u64>) -> Result<Run> {
        if let Some(id) = &self.evaluation_id {
            // a hosted launch pre-created the evaluation
            if !self.runs.is_empty() {
                bail!("${EVAL_ID_VAR} holds one epoch, and it already took one");
            }
            return Run::init(
                RunKind::Eval,
                self.online,
                Some(id.clone()),
                RunSpec::default(),
            )
            .await
            .map_err(|e| anyhow!(e.describe()));
        }
        let source = self
            .sources
            .get(env_name)
            .ok_or_else(|| anyhow!("KeyError: '{env_name}'"))?;
        let name = if self.sources.len() == 1 {
            self.name.clone()
        } else {
            Some(format!(
                "{}--{env_name}",
                self.name.as_deref().unwrap_or("None")
            ))
        };
        let config = json!({
            "model": self.model,
            "step": step,
            "run_id": self.run_id,
            "num_examples": expected.filter(|e| *e > 0).map(|e| e / source.group_size),
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
        Run::init(RunKind::Eval, self.online, None, spec)
            .await
            .map_err(|e| anyhow!(e.describe()))
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
            let run = match self.open(env_name, step, expected).await {
                Ok(run) => Some(run),
                Err(e) => {
                    tracing::warn!("Failed to open the {env_name} (Step {step}) evaluation: {e:#}");
                    None
                }
            };
            let url = run.as_ref().and_then(|r| r.url.clone());
            if let (Some(run), Some(url)) = (&run, &url) {
                let attached = if run.attached {
                    format!(" (attached via ${EVAL_ID_VAR})")
                } else {
                    String::new()
                };
                tracing::info!("Streaming {env_name} (Step {step}) evaluation - {url}{attached}");
            }
            let (dir, env, run_id) = (
                self.output_dir.clone(),
                env_name.to_string(),
                self.run_id.clone(),
            );
            let id = run.as_ref().map(|r| r.id.clone());
            tokio::task::spawn_blocking(move || {
                if let (Some(id), Some(url)) = (id, &url) {
                    update_record(&platform_run_path(&dir), |mut record| {
                        let fields = record.as_object_mut().expect("records are objects");
                        fields.entry("kind").or_insert("eval".into());
                        fields.entry("run_id").or_insert(run_id.into());
                        let evaluations = fields.entry("evaluations").or_insert(json!({}));
                        evaluations[&env] = json!({"step": step, "id": id, "url": url});
                        record
                    });
                }
                publish_link(&dir, PRIME_EVAL_LINKS, &env, url.as_deref());
            })
            .await
            .expect("record update panicked");
            self.runs.insert(key.clone(), run);
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
        let mut by_env: IndexMap<&str, Vec<EpisodeRef>> = IndexMap::new();
        for episode in episodes {
            if let Some(name) = &episode.env.name {
                by_env.entry(name).or_default().push(episode.clone());
            }
        }
        for (env_name, batch) in by_env {
            if let Some(run) = self.run_for(env_name, step, None).await {
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
        run.finish(summary, RunStatus::Completed, None).await;
        tracing::info!(
            "Uploaded {env_name} (Step {step}) evaluation - {}",
            run.url.as_deref().unwrap_or("None")
        );
        Ok(())
    }

    /// An epoch the run did not finish leaves its evaluation open: close it as cancelled. The
    /// parent run is only drained: its status is never this process's to report.
    async fn finalize(&mut self, _paused: bool) -> Result<()> {
        for run in self.runs.values_mut().flatten() {
            run.finish(Map::new(), RunStatus::Cancelled, Some("interrupted".into()))
                .await;
        }
        if let Some(parent) = &self.parent {
            parent.flush(FINISH_TIMEOUT).await;
        }
        Ok(())
    }

    async fn crash(&mut self) -> Result<()> {
        for run in self.runs.values_mut().flatten() {
            run.crash().await;
        }
        if let Some(parent) = &self.parent {
            parent.flush(FINISH_TIMEOUT).await;
        }
        Ok(())
    }
}
