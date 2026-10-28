//! `uv run eval`: one epoch of every configured eval source against the weights the inference
//! server currently serves (port of `eval/eval.py`).
//!
//! Every episode streams through the monitors as it arrives; a finished epoch also goes to the
//! platform when `monitors.prime` is set. An interrupted run resumes with `--resume` from its trace
//! stream: the landed episodes rejoin the epoch and only the rollouts still owed run.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prl_config::{EvalConfig, EvalMonitorsConfig};
use prl_monitors::prime::eval::{EvalSourceSpec, PrimeEvalSetup};
use prl_monitors::{FileMonitorConfig, Monitors, MonitorsSetup, PrimeSetup, WandbBridge};
use prl_types::Bridge;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::resume;
use crate::runner::EvalRunner;

/// `monitors.setup(...)` of an eval process.
pub(crate) async fn setup_monitors(
    monitors: &EvalMonitorsConfig,
    output_dir: &Path,
    producer: &str,
    overview_flavor: &str,
    model: &str,
    sources: Vec<EvalSourceSpec>,
    wandb: Option<Box<dyn WandbBridge>>,
    cancel: CancellationToken,
) -> Result<(Monitors, JoinHandle<Result<()>>)> {
    info!("Initializing monitors ({})", monitors_repr(monitors));
    let eval_env_names = sources.iter().map(|s| s.resolved_name.clone()).collect();
    let prime = monitors.prime.as_ref().map(|prime| {
        PrimeSetup::Eval(PrimeEvalSetup { name: prime.name.clone(), model: model.to_owned(), sources })
    });
    let setup = MonitorsSetup {
        output_dir: output_dir.to_path_buf(),
        producer: Some(producer.to_owned()),
        prime,
        wandb,
        file: monitors.file.as_ref().map(FileMonitorConfig::from),
        train_env_names: Vec::new(),
        eval_env_names,
        overview_flavor: overview_flavor.to_owned(),
    };
    prl_monitors::spawn(setup, cancel).await
}

fn monitors_repr(monitors: &EvalMonitorsConfig) -> String {
    let on = |set: bool, name: &str| set.then(|| name.to_owned());
    let names: Vec<String> = [
        on(monitors.wandb.is_some(), "wandb"),
        on(monitors.file.is_some(), "file"),
        on(monitors.prime.is_some(), "prime"),
    ]
    .into_iter()
    .flatten()
    .collect();
    names.join(", ")
}

pub async fn run_eval(config: EvalConfig, bridge: Arc<dyn Bridge>, wandb: Option<Box<dyn WandbBridge>>) -> Result<()> {
    let cancel = CancellationToken::new();
    // Read and set aside before the monitors start: the resumed attempt writes a fresh stream.
    let landed = match &config.resume {
        Some(r) => resume::take_landed_episodes(&config.run_dir, &config.dump, r.skip_checks)?,
        None => Vec::new(),
    };
    let sources = config
        .served
        .source
        .iter()
        .map(|s| EvalSourceSpec { resolved_name: s.resolved_name.clone(), taskset_id: s.env_id.clone(), group_size: s.group_size })
        .collect();
    let (monitors, monitors_task) = setup_monitors(
        &config.monitors,
        &config.run_dir,
        "eval",
        "eval",
        &config.model,
        sources,
        wandb,
        cancel.child_token(),
    )
    .await?;
    let result = run(&config, landed, monitors.clone(), bridge).await;
    if result.is_ok() {
        // Finalize only on a clean exit — a crashed run must not mark itself completed.
        monitors.finalize(false).await;
    }
    drop(monitors);
    cancel.cancel();
    let _ = monitors_task.await;
    result
}

async fn run(config: &EvalConfig, landed: Vec<prl_wire::Episode>, monitors: Monitors, bridge: Arc<dyn Bridge>) -> Result<()> {
    resume::stamp_config(&config.run_dir, &config.dump)?;
    let mut runner =
        EvalRunner::setup(&config.served, &config.model, &config.run_dir, config.log.interval, None, monitors, bridge)
            .await?;
    let result = async {
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
    }
    .await;
    runner.stop().await;
    result
}
