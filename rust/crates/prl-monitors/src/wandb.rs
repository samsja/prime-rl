//! The W&B monitor. There is no Rust W&B SDK: the Python `WandbMonitor` runs behind a narrow
//! bridge (`prime_rl.monitors.wandb.bridge.WandbBridge`) and only scalar rows cross it.

use anyhow::{Context, Result};
use async_trait::async_trait;
use prl_types::{Kind, Subset};

use crate::{EpisodeRef, Monitor, Row};

/// The Python side of the W&B monitor. Calls block (they take the GIL): the monitor makes
/// them on the blocking pool.
pub trait WandbBridge: Send + 'static {
    /// `WandbMonitor.init(output_dir, config, train_env_names, eval_env_names, overview_flavor)`.
    fn init(
        &mut self,
        train_env_names: &[String],
        eval_env_names: &[String],
        overview_flavor: &str,
    ) -> Result<()>;
    /// `WandbMonitor.log_metrics(json.loads(row_json), step)`; the row may hold `NaN`/`Infinity`.
    fn log(&mut self, row_json: &str, step: Option<u64>) -> Result<()>;
    /// `WandbMonitor.finalize(paused)`.
    fn finish(&mut self, paused: bool) -> Result<()>;
}

pub struct WandbMonitor(Option<Box<dyn WandbBridge>>);

impl WandbMonitor {
    pub async fn init(
        bridge: Box<dyn WandbBridge>,
        train: Vec<String>,
        eval: Vec<String>,
        flavor: String,
    ) -> Result<Self> {
        let mut monitor = WandbMonitor(Some(bridge));
        monitor
            .call(move |b| b.init(&train, &eval, &flavor))
            .await?;
        Ok(monitor)
    }

    async fn call(
        &mut self,
        f: impl FnOnce(&mut dyn WandbBridge) -> Result<()> + Send + 'static,
    ) -> Result<()> {
        let mut bridge = self.0.take().context("W&B bridge lost after a panic")?;
        let (bridge, result) = tokio::task::spawn_blocking(move || {
            let result = f(bridge.as_mut());
            (bridge, result)
        })
        .await?;
        self.0 = Some(bridge);
        result
    }
}

#[async_trait]
impl Monitor for WandbMonitor {
    fn name(&self) -> &'static str {
        "WandbMonitor"
    }

    async fn log_metrics(&mut self, metrics: &Row, step: Option<u64>) -> Result<()> {
        let row = prl_wire::json::to_python_json(&prl_wire::value::to_value(metrics)?, false);
        self.call(move |b| b.log(&row, step)).await
    }

    async fn log_episodes(&mut self, _: &[EpisodeRef], _: u64, _: Kind, _: Subset) -> Result<()> {
        Ok(())
    }

    async fn finalize(&mut self, paused: bool) -> Result<()> {
        self.call(move |b| b.finish(paused)).await
    }
}
