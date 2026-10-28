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

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use prl_types::{Kind, Subset};
use serde_json::{json, Map, Value as Json};

use crate::records::{publish_link, register_links, update_platform_record, PRIME_LINKS};
use crate::{sanitize, EpisodeRef, Monitor, Row};
use backend::{RunKind, RunSpec, RunStatus, TrainingSpec};
use client::Settings;
use run::Run;

const OFF_HINT: &str = "turn the monitor off with --no-monitors.prime";

/// Whether the run is tracked online: `$PRIME_RUNS_MODE` when set, else online, which needs an
/// API key (a configured monitor without credentials fails the run instead of leaving it
/// untracked).
fn online() -> Result<bool> {
    if let Some(mode) = std::env::var("PRIME_RUNS_MODE")
        .ok()
        .filter(|m| !m.is_empty())
    {
        return match mode.trim().to_lowercase().as_str() {
            "online" => Ok(true),
            "disabled" => Ok(false),
            other => {
                bail!("ConfigurationError: mode={other:?} is not one of 'online' or 'disabled'")
            }
        };
    }
    // `os.getenv("PRIME_API_KEY") or PrimeConfig().api_key`: the CLI profile only without the env var
    if std::env::var("PRIME_API_KEY").map_or(true, |k| k.is_empty())
        && Settings::load_with_context()?.api_key().is_empty()
    {
        bail!("The prime monitor found no Prime API key. Authenticate with `prime login` (or set PRIME_API_KEY), or {OFF_HINT}");
    }
    Ok(true)
}

/// A row as the SDK's metrics JSON, after naming the dropped non-finite values.
fn metrics_json(metrics: &Row) -> Map<String, Json> {
    let (metrics, dropped) = sanitize(metrics);
    if !dropped.is_empty() {
        let shown = dropped[..dropped.len().min(5)].join(", ");
        tracing::warn!(
            "Dropping {} non-finite metric value(s): {shown}",
            dropped.len()
        );
    }
    match serde_json::to_value(metrics) {
        Ok(Json::Object(map)) => map,
        _ => unreachable!("a row serializes to an object"),
    }
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
    pub async fn init(setup: PrimeTrainSetup, output_dir: PathBuf) -> Result<Self> {
        // a managed launch pre-created the platform run: attach, register nothing
        let attach = std::env::var("RUN_ID").ok().filter(|id| !id.is_empty());
        let spec = if attach.is_some() {
            RunSpec::default()
        } else {
            RunSpec {
                name: setup.name,
                environments: setup.environments,
                model: Some(setup.model),
                training: setup.training,
                config: setup.run_config,
                ..Default::default()
            }
        };
        let run = match Run::init(RunKind::Train, online()?, attach.clone(), spec).await {
            Ok(run) => run,
            Err(e) if attach.is_some() => return Err(anyhow!(e.describe())),
            Err(e) => bail!(
                "Failed to register the Prime run ({}). Select a team with `prime switch` (or set PRIME_TEAM_ID), or {OFF_HINT}",
                e.describe()
            ),
        };
        match &run.url {
            Some(url) if run.attached => tracing::info!(
                "Logging metrics and episodes to Prime run {} ({url}) (attached via $RUN_ID)",
                run.id
            ),
            Some(url) => tracing::info!(
                "Logging metrics and episodes to Prime run {} ({url})",
                run.id
            ),
            None => tracing::info!("Prime run disabled"),
        }
        let (id, url) = (run.id.clone(), run.url.clone());
        tokio::task::spawn_blocking(move || {
            register_links(&output_dir, PRIME_LINKS, u64::from(url.is_some()));
            if let Some(url) = url {
                update_platform_record(&output_dir, json!({"kind": "train", "id": id, "url": url}));
                publish_link(&output_dir, PRIME_LINKS, "run", Some(&url));
            }
        })
        .await?;
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
        if subset == Subset::All {
            self.run.log_episodes(episodes.to_vec()).await;
        }
        Ok(())
    }

    /// A paused run only drains and stays open: it continues on resume.
    async fn finalize(&mut self, paused: bool) -> Result<()> {
        if paused {
            self.run.flush(Duration::from_secs(30)).await;
        } else {
            self.run
                .finish(Map::new(), RunStatus::Completed, None)
                .await;
        }
        Ok(())
    }

    async fn crash(&mut self) -> Result<()> {
        self.run.crash().await;
        Ok(())
    }
}
