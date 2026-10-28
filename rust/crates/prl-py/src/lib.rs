//! Python extension module `prime_rl_orch_rs`: entry point of the Rust orchestrator.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_env("PRIME_RL_ORCH_RS_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

async fn run_orchestrator(config: serde_json::Value, _bridge: Py<PyAny>) -> anyhow::Result<()> {
    let keys: Vec<&String> = config.as_object().map(|o| o.keys().collect()).unwrap_or_default();
    tracing::info!(?keys, "rust orchestrator started (stub)");
    Ok(())
}

/// Run the orchestrator until it exits. Releases the GIL for the whole run; Rust calls back into
/// `bridge` (a `prime_rl.orchestrator.rust.PyBridge`) from blocking threads that re-acquire the GIL.
#[pyfunction]
fn run(py: Python<'_>, config_json: &str, bridge: Py<PyAny>) -> PyResult<()> {
    init_tracing();
    let config: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid orchestrator config JSON: {e}")))?;
    py.allow_threads(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("prl-orch")
            .build()?;
        runtime.block_on(run_orchestrator(config, bridge))
    })
    .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
}

#[pymodule]
fn prime_rl_orch_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
