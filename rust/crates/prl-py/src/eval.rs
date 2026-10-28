//! `prime_rl_orch_rs.run_eval` / `run_online_eval`: `uv run eval` and the SFT online-eval
//! process on the Rust core (`prl_eval`). Python parses and validates the pydantic config and
//! launches the env servers; Rust runs the eval with the GIL released.

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::bridge::PyBridge;
use crate::nixl::PyNixl;
use crate::wandb::PyWandb;

/// Run `fut` on a fresh multi-thread runtime with the GIL released. SIGINT/SIGTERM (Python's
/// handlers cannot run while Rust holds the thread) drop the run and raise.
fn block_on(py: Python<'_>, fut: impl Future<Output = anyhow::Result<()>> + Send) -> PyResult<()> {
    py.allow_threads(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().thread_name("prl-eval").build()?;
        runtime.block_on(async {
            let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
            tokio::select! {
                result = fut => result,
                _ = tokio::signal::ctrl_c() => {
                    tracing::warn!("Received SIGINT, stopping the eval");
                    anyhow::bail!("interrupted by SIGINT")
                }
                _ = sigterm.recv() => {
                    tracing::warn!("Received SIGTERM, stopping the eval");
                    anyhow::bail!("interrupted by SIGTERM")
                }
            }
        })
    })
    .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
}

/// Run `uv run eval` until it exits. `config` is the validated `EvalConfig`; Rust logs to the
/// console (at `console_level` and above, when set) and to `log_file`.
#[pyfunction]
#[pyo3(signature = (config, bridge, log_file=None, console_level=None))]
pub fn run_eval(
    py: Python<'_>,
    config: &Bound<'_, PyAny>,
    bridge: Py<PyAny>,
    log_file: Option<PathBuf>,
    console_level: Option<String>,
) -> PyResult<()> {
    let config = prl_config::EvalConfig::extract(config)?;
    prl_io::logging::setup_logger(&config.log.level, config.log.json_logging, log_file.as_deref(), console_level.as_deref())
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    let wandb = PyWandb::from_bridge(py, &bridge)?;
    let bridge = Arc::new(PyBridge::new(bridge));
    block_on(py, prl_eval::run_eval(config, bridge, wandb))
}

/// Run the SFT online-eval process until it exits. `config` is the validated `SFTOnlineEvalConfig`.
#[pyfunction]
pub fn run_online_eval(py: Python<'_>, config: &Bound<'_, PyAny>, bridge: Py<PyAny>) -> PyResult<()> {
    let config = prl_config::SFTOnlineEvalConfig::extract(config)?;
    prl_io::logging::setup_logger(&config.log.level, config.log.json_logging, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    let wandb = PyWandb::from_bridge(py, &bridge)?;
    let nixl = match &config.weight_broadcast {
        Some(prl_config::WeightBroadcastConfig::Nixl(_)) => {
            Some(PyNixl::from_bridge(py, &bridge).map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?)
        }
        _ => None,
    };
    let bridge = Arc::new(PyBridge::new(bridge));
    block_on(py, prl_eval::run_online_eval(config, bridge, wandb, nixl))
}
