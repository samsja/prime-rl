//! `uv run eval`: one epoch of every configured eval source against the weights the inference
//! server currently serves (port of `eval/eval.py`). An interrupted run resumes with `--resume`
//! from its trace stream: the landed episodes rejoin the epoch and only the rollouts still owed run.

use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prl_config::{EvalConfig, EvalMonitorsConfig};
use prl_monitors::prime::eval::{EvalSourceSpec, PrimeEvalSetup};
use prl_monitors::{Monitors, MonitorsSetup, PrimeSetup, WandbBridge};
use prl_types::Bridge;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::resume;
use crate::runner::EvalRunner;

/// The monitors of an eval process (`monitors.setup(producer=..., overview_flavor=...)`).
pub(crate) struct EvalMonitors<'a> {
    pub config: &'a EvalMonitorsConfig,
    /// Python's `str(config.monitors)`, for the init line.
    pub repr: &'a str,
    pub output_dir: &'a Path,
    pub producer: &'static str,
    pub overview_flavor: &'static str,
    pub model: &'a str,
    pub sources: Vec<EvalSourceSpec>,
    pub wandb: Option<Box<dyn WandbBridge>>,
}

impl EvalMonitors<'_> {
    /// Run `body` with the monitors; they are finalized only on a clean exit — a crashed run must
    /// not mark itself completed.
    pub async fn run<Fut: Future<Output = Result<()>>>(self, body: impl FnOnce(Monitors) -> Fut) -> Result<()> {
        info!("Initializing monitors ({})", self.repr);
        let eval_env_names = self.sources.iter().map(|s| s.resolved_name.clone()).collect();
        let prime = self.config.prime.as_ref().map(|prime| {
            PrimeSetup::Eval(PrimeEvalSetup { name: prime.name.clone(), model: self.model.to_owned(), sources: self.sources })
        });
        let setup = MonitorsSetup {
            output_dir: self.output_dir.to_path_buf(),
            producer: Some(self.producer.to_owned()),
            prime,
            wandb: self.wandb,
            file: self.config.file.clone(),
            train_env_names: Vec::new(),
            eval_env_names,
            overview_flavor: self.overview_flavor.to_owned(),
        };
        let cancel = CancellationToken::new();
        let (monitors, task) = prl_monitors::spawn(setup, cancel.clone()).await?;
        let result = body(monitors.clone()).await;
        if result.is_ok() {
            monitors.finalize(false).await;
        }
        cancel.cancel();
        let _ = task.await;
        result
    }
}

/// `fut`, or an error once `shutdown` fires (the future is dropped).
pub(crate) async fn until_shutdown<T>(shutdown: &CancellationToken, fut: impl Future<Output = Result<T>>) -> Result<T> {
    tokio::select! {
        result = fut => result,
        () = shutdown.cancelled() => anyhow::bail!("interrupted"),
    }
}

/// Run `uv run eval`. `shutdown` (a signal) stops it without finalizing the monitors.
pub async fn run_eval(
    config: EvalConfig,
    bridge: Arc<dyn Bridge>,
    wandb: Option<Box<dyn WandbBridge>>,
    monitors_repr: &str,
    shutdown: CancellationToken,
) -> Result<()> {
    // Read and set aside before the monitors start: the resumed attempt writes a fresh stream.
    let landed = match &config.resume {
        Some(r) => resume::take_landed(&config.run_dir, &config.dump, r.skip_checks)?,
        None => Vec::new(),
    };
    let (config, shutdown) = (&config, &shutdown);
    let sources = config.served.source.iter();
    let monitors = EvalMonitors {
        config: &config.monitors,
        repr: monitors_repr,
        output_dir: &config.run_dir,
        producer: "eval",
        overview_flavor: "eval",
        model: &config.model,
        sources: sources
            .map(|s| EvalSourceSpec { resolved_name: s.resolved_name.clone(), taskset_id: s.env_id.clone(), group_size: s.group_size })
            .collect(),
        wandb,
    };
    monitors
        .run(|monitors| async move {
            resume::stamp_config(&config.run_dir, &config.dump)?;
            let setup =
                EvalRunner::setup(&config.served, &config.model, &config.run_dir, config.log.interval, None, monitors, bridge);
            let mut runner = until_shutdown(shutdown, setup).await?;
            let body = async {
                let outcome = runner.trigger(0, false, landed).await?;
                if config.resume.is_some() {
                    info!(
                        "Resuming from the trace stream: {} episodes restored, {} rollouts owed",
                        outcome.restored.len(),
                        outcome.owed
                    );
                }
                runner.start()?;
                runner.run_epoch(outcome, 0, None).await?;
                runner.drain().await;
                Ok(())
            };
            let result = until_shutdown(shutdown, body).await;
            runner.stop().await;
            result
        })
        .await
}
