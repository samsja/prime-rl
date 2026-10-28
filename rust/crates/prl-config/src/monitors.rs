//! `prime_rl.configs.monitors`.

use std::path::PathBuf;

use pyo3::prelude::*;

/// `WandbMonitorConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct WandbMonitorConfig {
    pub project: String,
    pub entity: Option<String>,
    pub name: Option<String>,
    pub group: Option<String>,
    pub tags: Option<Vec<String>>,
    pub offline: bool,
}

/// `FileMonitorConfig`: metrics JSONL and episode trace streams under the output dir.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct FileMonitorConfig {
    /// Metrics file, relative to `monitors/file/` under the output dir (absolute wins).
    pub path: PathBuf,
    /// Size at which a trace stream rolls to a new chunk file.
    pub chunk_bytes: u64,
    /// Seal full chunks with seekable zstd.
    pub compress: bool,
    /// Decimals kept for per-token float streams; None keeps every digit.
    pub float_decimals: Option<u32>,
}

/// `PrimeTrainMonitorConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct PrimeTrainMonitorConfig {
    /// Run name on the platform (defaulted to the W&B run name by the config validator).
    pub name: Option<String>,
}

/// `TrainMonitorsConfig` (`[orchestrator.monitors]`); None disables a monitor.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct TrainMonitorsConfig {
    pub wandb: Option<WandbMonitorConfig>,
    pub file: Option<FileMonitorConfig>,
    pub prime: Option<PrimeTrainMonitorConfig>,
}
