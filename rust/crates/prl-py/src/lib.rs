//! Python extension module `prime_rl_orch_rs`: entry point of the Rust orchestrator.

pub mod bridge;
pub mod wandb;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::bridge::PyBridge;
use crate::wandb::PyWandb;

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_env("PRIME_RL_ORCH_RS_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

// Stub until prl-core lands: `prl_core::run(config, Arc::new(bridge), wandb)`.
async fn run_orchestrator(
    config: prl_config::OrchestratorConfig,
    bridge: PyBridge,
    wandb: Option<Box<dyn prl_monitors::WandbBridge>>,
) -> anyhow::Result<()> {
    tracing::info!(batch_size = config.batch_size, wandb = wandb.is_some(), "rust orchestrator started (stub)");
    let _ = bridge;
    Ok(())
}

/// Run the orchestrator until it exits. `config` is the validated `OrchestratorConfig`.
/// Releases the GIL for the whole run; Rust calls back into `bridge` (a
/// `prime_rl.orchestrator.rust.PyBridge`) from blocking threads that re-acquire the GIL.
#[pyfunction]
fn run(py: Python<'_>, config: &Bound<'_, PyAny>, bridge: Py<PyAny>) -> PyResult<()> {
    init_tracing();
    let config: prl_config::OrchestratorConfig = config.extract()?;
    let wandb = PyWandb::from_bridge(py, &bridge)?;
    py.allow_threads(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("prl-orch")
            .build()?;
        runtime.block_on(run_orchestrator(config, PyBridge::new(bridge), wandb))
    })
    .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
}

#[pymodule]
fn prime_rl_orch_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
