//! The run handle (`prime_runs.Run` / `prime_runs.init`): an id, a URL, two uploaders
//! (samples, metrics) and the lifecycle that closes the run out.

use std::time::Duration;

use serde_json::{Map, Value as Json};
use tokio::time::Instant;

use super::backend::{Backend, RunKind, RunSpec, RunStatus};
use super::client::{ApiError, PlatformClient, Settings};
use super::sinks::{RftMetricsSink, TracesSink};
use super::worker::UploadWorker;
use crate::EpisodeRef;

/// How long finishing lets queued uploads drain.
pub const FINISH_TIMEOUT: Duration = Duration::from_secs(60);
/// Training runs last days: a sink that struck out on transient failures is tried again after this.
const TRAIN_RETIRE_COOLDOWN: Duration = Duration::from_secs(300);

/// `$PRIME_API_BASE` historically points at the RFT API root; the SDK takes the platform base.
fn base_url_override() -> Option<String> {
    let base = std::env::var("PRIME_API_BASE")
        .ok()
        .filter(|b| !b.is_empty())?;
    let base = base.trim_end_matches('/');
    Some(base.strip_suffix("/rft").unwrap_or(base).to_string())
}

pub struct Run {
    pub id: String,
    pub url: Option<String>,
    pub attached: bool,
    kind: RunKind,
    /// None in disabled mode.
    backend: Option<Backend>,
    samples: UploadWorker<EpisodeRef>,
    metrics: UploadWorker<Json>,
    config: Map<String, Json>,
    summary: Map<String, Json>,
    pub finished: bool,
}

impl Run {
    /// Open a run: attach to `id` (a run a launcher created) or register `spec`. A disabled
    /// run (`online = false`) has a local id and uploads nothing.
    pub async fn init(
        kind: RunKind,
        online: bool,
        id: Option<String>,
        mut spec: RunSpec,
    ) -> Result<Run, ApiError> {
        let settings = Settings::load();
        spec.team_id = spec.team_id.or_else(|| settings.team_id());
        let mut run = Run {
            id: id.clone().unwrap_or_else(|| {
                format!(
                    "disabled-{}",
                    &uuid::Uuid::new_v4().simple().to_string()[..16]
                )
            }),
            url: None,
            attached: id.is_some(),
            kind,
            backend: None,
            samples: UploadWorker::disabled(),
            metrics: UploadWorker::disabled(),
            config: spec.config.clone(),
            summary: Map::new(),
            finished: false,
        };
        if !online {
            return Ok(run);
        }
        // resolved before any platform call: a bad value fails here, not after the run exists
        let opted_out = settings.traces_opt_out()?;
        let api_key = settings.api_key();
        if api_key.is_empty() {
            return Err(ApiError::Configuration(
                "mode=\"online\" needs an API key. Set PRIME_API_KEY, run `prime login`, or pass mode=\"disabled\".".into(),
            ));
        }
        let client = PlatformClient::new(
            &api_key,
            &base_url_override().unwrap_or_else(|| settings.base_url()),
        )?;
        let backend = Backend::new(
            client.clone(),
            kind,
            settings.frontend_url(),
            spec.team_id.clone(),
        );
        let handle = match &id {
            Some(id) => backend.attach(id),
            None => backend.create(&spec).await?,
        };
        let mut context = vec![
            ("source", "prime-runs".to_string()),
            ("run_type", kind.as_str().to_string()),
        ];
        context.extend(spec.framework.clone().map(|f| ("framework", f)));
        context.extend(spec.model.clone().map(|m| ("model", m)));
        let mut traces = TracesSink::new(
            &api_key,
            spec.team_id.as_deref(),
            &settings.traces_url(),
            &handle.id,
            kind.as_str(),
            &context,
        )?;
        if opted_out {
            traces.opt_out();
        }
        let cooldown = (kind == RunKind::Train).then_some(TRAIN_RETIRE_COOLDOWN);
        run.samples = UploadWorker::spawn(Box::new(traces), cooldown);
        if kind == RunKind::Train {
            let metrics = RftMetricsSink::new(client, &handle.id);
            run.metrics = UploadWorker::spawn(Box::new(metrics), cooldown);
        }
        run.id = handle.id;
        run.url = Some(handle.url);
        run.backend = Some(backend);
        Ok(run)
    }

    /// Queue episodes for Prime Traces.
    pub async fn log_episodes(&mut self, episodes: Vec<EpisodeRef>) {
        if !episodes.is_empty() {
            self.samples.submit(episodes).await;
        }
    }

    /// Queue one row of metrics (training runs): `step` and `_timestamp` added unless set.
    pub async fn log_metrics(&mut self, mut row: Map<String, Json>, step: Option<u64>) {
        if let Some(step) = step {
            row.entry("step").or_insert(step.into());
        }
        row.entry("_timestamp")
            .or_insert(crate::file::unix_time().into());
        self.metrics.submit(vec![row.into()]).await;
    }

    /// Wait until queued records have been written.
    pub async fn flush(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        self.samples.flush(deadline).await;
        self.metrics.flush(deadline).await;
    }

    /// Drain the uploads and close the run out. Idempotent.
    pub async fn finish(
        &mut self,
        summary: Map<String, Json>,
        status: RunStatus,
        error: Option<String>,
    ) {
        if std::mem::replace(&mut self.finished, true) {
            return;
        }
        self.summary.extend(summary);
        let deadline = Instant::now() + FINISH_TIMEOUT;
        // records first, so the dashboard never sees a finished run with samples landing
        let drained = self.samples.flush(deadline).await & self.metrics.flush(deadline).await;
        if !drained {
            tracing::warn!(
                "Run {}: uploads did not drain within {}s; finalizing anyway. Some records may be missing from this run.",
                self.id,
                FINISH_TIMEOUT.as_secs()
            );
        }
        let reports: Vec<_> = [
            self.samples.close(deadline).await,
            self.metrics.close(deadline).await,
        ]
        .into_iter()
        .flatten()
        .collect();
        let mut errors: Vec<String> = reports
            .iter()
            .flat_map(|r| r.errors.iter().cloned())
            .collect();
        if let Some(backend) = self.backend.take() {
            let written: Vec<usize> = reports.iter().filter_map(|r| r.episodes_written).collect();
            if self.kind == RunKind::Eval && !written.is_empty() {
                let entry = self
                    .summary
                    .entry("prime_runs")
                    .or_insert_with(|| serde_json::json!({}));
                if !entry.is_object() {
                    *entry = serde_json::json!({});
                }
                entry["traces_episodes_written"] = written.iter().sum::<usize>().into();
            }
            // an attached run's config is the launcher's
            let config = if self.attached {
                Map::new()
            } else {
                self.config.clone()
            };
            if let Err(e) = backend.update(&self.id, &config, &self.summary).await {
                errors.push(self.note("updating the run", &e));
            }
            if self.attached && status != RunStatus::Completed {
                tracing::info!(
                    "Run {}: {}, but the platform owns the status of an attached run; leaving it to the launcher",
                    self.id,
                    status.as_str()
                );
            } else {
                let error = error.or_else(|| errors.first().cloned());
                if let Err(e) = backend
                    .finalize(
                        &self.id,
                        status,
                        &self.summary,
                        error.as_deref(),
                        &self.config,
                    )
                    .await
                {
                    self.note("finalizing the run", &e);
                }
            }
        }
        let dropped = self.samples.dropped + self.metrics.dropped;
        if dropped > 0 {
            tracing::warn!("Run {} finished with {dropped} record(s) that reached no sink; the producer outran the uploader.", self.id);
        }
        for (sink, count) in reports.iter().filter_map(|r| r.failed_records) {
            tracing::warn!(
                "Run {}: the {sink} sink could not store {count} record(s)",
                self.id
            );
        }
    }

    /// The SDK's atexit hook: a run never finished reports crashed.
    pub async fn crash(&mut self) {
        if !self.finished {
            tracing::warn!(
                "Run {} was never finished; reporting it as crashed",
                self.id
            );
            self.finish(
                Map::new(),
                RunStatus::Crashed,
                Some("process exited without finishing the run".into()),
            )
            .await;
        }
    }

    fn note(&self, what: &str, e: &ApiError) -> String {
        let message = format!("{what} failed: {}", e.describe());
        tracing::warn!("Run {}: {message}", self.id);
        message
    }
}
