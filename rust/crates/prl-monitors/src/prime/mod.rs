//! The Prime platform monitors (`monitors/prime.py`) over a native port of the `prime_runs`
//! SDK paths they use.

pub mod backend;
pub mod client;
pub mod eval;
pub mod run;
pub mod sinks;
pub mod worker;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use prl_types::{Kind, Subset};
use serde_json::{json, Map, Value as Json};

use crate::pyjson::{sanitize, scalar_json, Row};
use crate::records::{publish_link, register_links, update_platform_record, PRIME_LINKS};
use crate::{EpisodeRef, Monitor};
use backend::{RunKind, RunSpec, RunStatus, TrainingSpec};
use client::Settings;
use run::{Mode, Run, RunOptions};

pub const BASE_URL_VAR: &str = "PRIME_API_BASE";
pub const MODE_ENV: &str = "PRIME_RUNS_MODE";
/// How long finish lets queued uploads drain.
pub const FINISH_TIMEOUT: Duration = Duration::from_secs(60);
const OFF_HINT: &str = "turn the monitor off with --no-monitors.prime";

/// `$PRIME_RUNS_MODE` when set, else online — which needs an API key: a configured monitor
/// without credentials fails the run instead of leaving it untracked.
pub fn prime_mode() -> Result<Mode> {
    if let Some(mode) = std::env::var(MODE_ENV).ok().filter(|m| !m.is_empty()) {
        return Ok(Mode::parse(&mode)?);
    }
    // `os.getenv("PRIME_API_KEY") or PrimeConfig().api_key`: the profile is only read without the env var
    if std::env::var("PRIME_API_KEY").map_or(true, |k| k.is_empty())
        && Settings::load_with_context()?.api_key().is_empty()
    {
        return Err(anyhow!(
            "The prime monitor found no Prime API key. Authenticate with `prime login` (or set PRIME_API_KEY), or {OFF_HINT}"
        ));
    }
    Ok(Mode::Online)
}

/// `$PRIME_API_BASE` historically points at the RFT API root; the SDK takes the platform base.
pub fn base_url_override() -> Option<String> {
    std::env::var(BASE_URL_VAR)
        .ok()
        .filter(|b| !b.is_empty())
        .map(|b| {
            let b = b.trim_end_matches('/');
            b.strip_suffix("/rft").unwrap_or(b).to_string()
        })
}

/// A row as the SDK's metrics JSON, after naming dropped non-finite values.
pub(crate) fn metrics_json(metrics: &Row) -> Map<String, Json> {
    let (metrics, dropped) = sanitize(metrics);
    if !dropped.is_empty() {
        tracing::warn!(
            "Dropping {} non-finite metric value(s): {}",
            dropped.len(),
            dropped
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    metrics
        .iter()
        .map(|(k, v)| (k.clone(), scalar_json(v)))
        .collect()
}

/// What the training run registers with (from the orchestrator config).
#[derive(Debug, Clone, PartialEq)]
pub struct PrimeTrainSetup {
    pub name: Option<String>,
    pub model: String,
    /// Train sources' env ids.
    pub environments: Vec<String>,
    pub training: TrainingSpec,
    /// `config.model_dump(exclude_none=True, mode="json")`.
    pub run_config: Map<String, Json>,
}

/// Logs metrics and episodes to a Prime platform training run.
pub struct PrimeTrainMonitor {
    run: Run,
}

impl PrimeTrainMonitor {
    pub async fn init(
        setup: Option<PrimeTrainSetup>,
        name: Option<String>,
        output_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let attach = std::env::var("RUN_ID").ok().filter(|id| !id.is_empty());
        let spec = match (&attach, setup) {
            // A managed launch pre-created the platform run: attach, register nothing.
            (Some(_), _) => RunSpec::default(),
            (None, Some(setup)) => RunSpec {
                name: setup.name,
                environments: setup.environments,
                model: Some(setup.model),
                training: Some(setup.training),
                config: setup.run_config,
                ..Default::default()
            },
            // The RFT API requires a base model; "unknown" is what a config-less init sends.
            (None, None) => RunSpec {
                name,
                model: Some("unknown".into()),
                ..Default::default()
            },
        };
        let mode = prime_mode()?;
        let options = RunOptions {
            kind: RunKind::Train,
            mode,
            base_url: base_url_override(),
            finish_timeout: FINISH_TIMEOUT,
            id: attach.clone(),
            spec,
        };
        let run = match Run::init(options).await {
            Ok(run) => run,
            Err(e) if attach.is_some() => return Err(e.into()),
            Err(e) => {
                return Err(anyhow!(
                    "Failed to register the Prime run ({}). Select a team with `prime switch` (or set PRIME_TEAM_ID), or {OFF_HINT}",
                    e.describe()
                ))
            }
        };
        match &run.url {
            Some(url) => {
                let attached = if run.attached {
                    " (attached via $RUN_ID)"
                } else {
                    ""
                };
                tracing::info!(
                    "Logging metrics and episodes to Prime run {} ({url}){attached}",
                    run.id
                );
            }
            None => tracing::info!("Prime run disabled"),
        }
        if let Some(output_dir) = output_dir {
            let (id, url) = (run.id.clone(), run.url.clone());
            tokio::task::spawn_blocking(move || {
                register_links(&output_dir, PRIME_LINKS, u64::from(url.is_some()));
                if let Some(url) = url {
                    let update = json!({"kind": "train", "id": id, "url": url});
                    update_platform_record(
                        &output_dir,
                        update.as_object().expect("object").clone(),
                    );
                    publish_link(&output_dir, PRIME_LINKS, "run", Some(&url));
                }
            })
            .await?;
        }
        Ok(PrimeTrainMonitor { run })
    }
}

#[async_trait]
impl Monitor for PrimeTrainMonitor {
    fn name(&self) -> &'static str {
        "PrimeTrainMonitor"
    }

    async fn log_metrics(&mut self, metrics: &Row, step: Option<u64>) -> Result<()> {
        self.run.log_metrics(metrics_json(metrics), step).await;
        Ok(())
    }

    /// Every episode ships whole as it arrives (the `all` subset).
    async fn log_episodes(
        &mut self,
        episodes: &[EpisodeRef],
        _step: u64,
        _kind: Kind,
        subset: Subset,
    ) -> Result<()> {
        if subset == Subset::All && !episodes.is_empty() {
            self.run.log_episodes(episodes.to_vec()).await;
        }
        Ok(())
    }

    async fn finalize(&mut self, paused: bool) -> Result<()> {
        if paused {
            self.run.flush(Duration::from_secs(30)).await;
        } else {
            self.run.finish(None, RunStatus::Completed, None).await;
        }
        Ok(())
    }

    /// The SDK's atexit hook: a run never finished reports crashed.
    async fn crash(&mut self) -> Result<()> {
        if !self.run.finished {
            tracing::warn!(
                "Run {} was never finished; reporting it as crashed",
                self.run.id
            );
            self.run
                .finish(
                    None,
                    RunStatus::Crashed,
                    Some("process exited without finishing the run".into()),
                )
                .await;
        }
        Ok(())
    }
}
