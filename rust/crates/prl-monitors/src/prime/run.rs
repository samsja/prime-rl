//! The run handle (`prime_runs.Run` / `prime_runs.init`): an id, a URL, two uploaders
//! (samples, metrics) and the lifecycle that closes the run out.

use std::time::Duration;

use serde_json::{Map, Value as Json};
use tokio::time::Instant;

use super::backend::{Backend, RunHandle, RunKind, RunSpec, RunStatus};
use super::client::{ApiError, PlatformClient, Settings};
use super::sinks::{RftMetricsSink, TracesSink};
use super::worker::{Sink, UploadWorker, WorkerReport};
use crate::EpisodeRef;

/// Training runs last days: a sink that struck out on transient failures is tried again after this.
pub const TRAIN_RETIRE_COOLDOWN: Duration = Duration::from_secs(300);
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Online,
    Disabled,
}

impl Mode {
    pub fn parse(mode: &str) -> Result<Mode, ApiError> {
        match mode.trim().to_lowercase().as_str() {
            "online" => Ok(Mode::Online),
            "disabled" => Ok(Mode::Disabled),
            other => Err(ApiError::Configuration(format!(
                "mode={other:?} is not one of 'online' or 'disabled'"
            ))),
        }
    }
}

pub struct RunOptions {
    pub kind: RunKind,
    pub mode: Mode,
    pub base_url: Option<String>,
    pub finish_timeout: Duration,
    /// Attach to a run a launcher already created instead of creating one.
    pub id: Option<String>,
    pub spec: RunSpec,
}

pub struct Run {
    pub id: String,
    pub url: Option<String>,
    pub attached: bool,
    pub kind: RunKind,
    backend: Option<Backend>,
    samples: UploadWorker<EpisodeRef>,
    metrics: UploadWorker<Json>,
    config: Map<String, Json>,
    summary: Map<String, Json>,
    errors: Vec<String>,
    finish_timeout: Duration,
    pub finished: bool,
}

fn disabled_run_id() -> String {
    format!(
        "disabled-{}",
        &uuid::Uuid::new_v4().simple().to_string()[..16]
    )
}

impl Run {
    pub async fn init(options: RunOptions) -> Result<Run, ApiError> {
        let settings = Settings::load();
        let RunOptions {
            kind,
            mode,
            base_url,
            finish_timeout,
            id,
            mut spec,
        } = options;
        spec.team_id = spec.team_id.or_else(|| settings.team_id());
        let base_url = base_url.unwrap_or_else(|| settings.base_url());
        // Resolved before any platform call: a bad value fails here, not after the run exists.
        let opted_out = mode == Mode::Online && settings.traces_opt_out()?;
        let mut run = Run {
            id: String::new(),
            url: None,
            attached: false,
            kind,
            backend: None,
            samples: UploadWorker::spawn(vec![], None),
            metrics: UploadWorker::spawn(vec![], None),
            config: spec.config.clone(),
            summary: Map::new(),
            errors: Vec::new(),
            finish_timeout,
            finished: false,
        };
        if mode == Mode::Disabled {
            run.id = id.unwrap_or_else(disabled_run_id);
            return Ok(run);
        }
        let api_key = settings.api_key();
        if api_key.is_empty() {
            return Err(ApiError::Configuration(
                "mode=\"online\" needs an API key. Set PRIME_API_KEY, run `prime login`, or pass mode=\"disabled\".".into(),
            ));
        }
        let client = PlatformClient::new(&api_key, &base_url, DEFAULT_TIMEOUT)?;
        let backend = Backend::new(
            client.clone(),
            kind,
            &settings.frontend_url(),
            spec.team_id.clone(),
        );
        let handle: RunHandle = match &id {
            Some(id) => {
                run.attached = true;
                backend.attach(id)
            }
            None => backend.create(&spec).await?,
        };
        let mut context = vec![
            ("source", "prime-runs".to_string()),
            ("run_type", kind.as_str().to_string()),
        ];
        if let Some(framework) = &spec.framework {
            context.push(("framework", framework.clone()));
        }
        if let Some(model) = &spec.model {
            context.push(("model", model.clone()));
        }
        let mut traces = TracesSink::new(
            &api_key,
            spec.team_id.clone(),
            &settings.traces_url(),
            &handle.id,
            kind.as_str(),
            &context,
        )?;
        if opted_out {
            traces.opt_out();
        }
        let cooldown = (kind == RunKind::Train).then_some(TRAIN_RETIRE_COOLDOWN);
        run.samples = UploadWorker::spawn(
            vec![Box::new(traces) as Box<dyn Sink<EpisodeRef>>],
            cooldown,
        );
        if kind == RunKind::Train {
            run.metrics = UploadWorker::spawn(
                vec![Box::new(RftMetricsSink::new(client, &handle.id)) as Box<dyn Sink<Json>>],
                cooldown,
            );
        }
        run.id = handle.id;
        run.url = handle.url;
        run.backend = Some(backend);
        if let Some(url) = &run.url {
            tracing::debug!("Run {}: {url}", run.id);
        }
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

    /// Block until queued records have been written.
    pub async fn flush(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let samples = self.samples.flush(deadline).await;
        self.metrics.flush(deadline).await && samples
    }

    /// Drain the uploads and close the run out. Idempotent.
    pub async fn finish(
        &mut self,
        summary: Option<Map<String, Json>>,
        status: RunStatus,
        error: Option<String>,
    ) {
        if self.finished {
            return;
        }
        self.finished = true;
        if let Some(summary) = summary {
            self.summary.extend(summary);
        }
        let deadline = Instant::now() + self.finish_timeout;
        // Records first, so the dashboard never sees a finished run with samples landing.
        let drained = self.samples.flush(deadline).await;
        let drained = self.metrics.flush(deadline).await && drained;
        if !drained {
            tracing::warn!(
                "Run {}: uploads did not drain within {}s; finalizing anyway. Some records may be missing from this run.",
                self.id,
                self.finish_timeout.as_secs_f64()
            );
        }
        let reports: Vec<WorkerReport> = [
            self.samples.close(deadline).await,
            self.metrics.close(deadline).await,
        ]
        .into_iter()
        .flatten()
        .collect();
        for report in &reports {
            self.errors.extend(report.errors.iter().cloned());
        }
        if let Some(backend) = self.backend.take() {
            let written: Vec<usize> = reports
                .iter()
                .flat_map(|r| r.episodes_written.iter().copied())
                .collect();
            if self.kind == RunKind::Eval && !written.is_empty() {
                let mut prime_runs = match self.summary.get("prime_runs") {
                    Some(Json::Object(m)) => m.clone(),
                    _ => Map::new(),
                };
                prime_runs.insert(
                    "traces_episodes_written".into(),
                    written.iter().sum::<usize>().into(),
                );
                self.summary.insert("prime_runs".into(), prime_runs.into());
            }
            let config = (!self.attached && !self.config.is_empty()).then_some(&self.config);
            let summary = (!self.summary.is_empty()).then_some(&self.summary);
            if let Err(e) = backend.update(&self.id, config, summary).await {
                self.note("updating the run", &e);
            }
            if self.attached && status != RunStatus::Completed {
                tracing::info!(
                    "Run {}: {}, but the platform owns the status of an attached run; leaving it to the launcher",
                    self.id,
                    status.as_str()
                );
            } else {
                let error = error.or_else(|| self.errors.first().cloned());
                let config = (!self.config.is_empty()).then_some(&self.config);
                let summary = (!self.summary.is_empty()).then_some(&self.summary);
                if let Err(e) = backend
                    .finalize(&self.id, status, summary, error.as_deref(), config)
                    .await
                {
                    self.note("finalizing the run", &e);
                }
            }
        }
        let dropped = self.samples.dropped + self.metrics.dropped;
        if dropped > 0 {
            tracing::warn!(
                "Run {} finished with {dropped} record(s) that reached no sink; the producer outran the uploader.",
                self.id
            );
        }
        for report in &reports {
            for (sink, count) in &report.failed_records {
                tracing::warn!(
                    "Run {}: the {sink} sink could not store {count} record(s)",
                    self.id
                );
            }
        }
    }

    fn note(&mut self, what: &str, e: &ApiError) {
        let message = format!("{what} failed: {}", e.describe());
        tracing::warn!("Run {}: {message}", self.id);
        self.errors.push(message);
    }
}
