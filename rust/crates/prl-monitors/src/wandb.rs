//! The W&B monitor. There is no Rust W&B SDK: the Python `WandbMonitor` runs behind a narrow
//! bridge (`prime_rl.orchestrator.rust_monitors.WandbBridge`) and only scalar rows cross it.

use anyhow::{Context, Result};
use async_trait::async_trait;
use prl_types::{Kind, Subset};

use crate::pyjson::{dumps_row, Row};
use crate::{EpisodeRef, Monitor};

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

pub struct WandbMonitor {
    bridge: Option<Box<dyn WandbBridge>>,
}

impl WandbMonitor {
    pub async fn init(
        bridge: Box<dyn WandbBridge>,
        train_env_names: Vec<String>,
        eval_env_names: Vec<String>,
        overview_flavor: String,
    ) -> Result<Self> {
        let mut monitor = WandbMonitor {
            bridge: Some(bridge),
        };
        monitor
            .call(move |b| b.init(&train_env_names, &eval_env_names, &overview_flavor))
            .await?;
        Ok(monitor)
    }

    async fn call(
        &mut self,
        f: impl FnOnce(&mut dyn WandbBridge) -> Result<()> + Send + 'static,
    ) -> Result<()> {
        let mut bridge = self
            .bridge
            .take()
            .context("W&B bridge lost after a panic")?;
        let (bridge, result) = tokio::task::spawn_blocking(move || {
            let result = f(bridge.as_mut());
            (bridge, result)
        })
        .await?;
        self.bridge = Some(bridge);
        result
    }
}

#[async_trait]
impl Monitor for WandbMonitor {
    fn name(&self) -> &'static str {
        "WandbMonitor"
    }

    async fn log_metrics(&mut self, metrics: &Row, step: Option<u64>) -> Result<()> {
        let row = dumps_row(metrics);
        self.call(move |b| b.log(&row, step)).await
    }

    async fn log_episodes(
        &mut self,
        _episodes: &[EpisodeRef],
        _step: u64,
        _kind: Kind,
        _subset: Subset,
    ) -> Result<()> {
        Ok(())
    }

    async fn finalize(&mut self, paused: bool) -> Result<()> {
        self.call(move |b| b.finish(paused)).await
    }
}
