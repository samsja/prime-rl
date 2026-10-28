//! Python extension module `prime_rl_orch_rs`: entry point of the Rust orchestrator.

pub mod bridge;
pub mod eval;
pub mod nixl;
pub mod wandb;

use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio_util::sync::CancellationToken;

use crate::bridge::PyBridge;
use crate::nixl::PyNixl;
use crate::wandb::PyWandb;

/// How often the main thread runs Python's signal handlers while the orchestrator runs.
const SIGNAL_POLL: Duration = Duration::from_millis(100);

/// Run the orchestrator until it exits. `config` is the validated `OrchestratorConfig`.
/// Releases the GIL for the whole run; Rust calls back into `bridge` (a
/// `prime_rl.orchestrator.rust.PyBridge`) from blocking threads that re-acquire the GIL.
///
/// Python only runs signal handlers on the main thread between bytecodes, so the main
/// thread polls them: a handler that raises (Ctrl-C's `KeyboardInterrupt`) shuts the
/// orchestrator down the way cancelling the Python orchestrator's main task did, and the
/// exception is re-raised once it has stopped.
#[pyfunction]
fn run(py: Python<'_>, config: &Bound<'_, PyAny>, bridge: Py<PyAny>) -> PyResult<()> {
    let config: prl_config::OrchestratorConfig = config.extract()?;
    prl_io::logging::setup_logger(&config.log.level, config.log.json_logging, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    let wandb = PyWandb::from_bridge(py, &bridge)?;
    let nixl = match config.weight_broadcast {
        prl_config::WeightBroadcastConfig::Nixl(_) => {
            Some(PyNixl::from_bridge(py, &bridge).map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?)
        }
        _ => None,
    };
    let bridge: Arc<dyn prl_types::Bridge> = Arc::new(PyBridge::new(bridge));
    let (result, interrupt) = py.allow_threads(|| -> anyhow::Result<_> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("prl-orch")
            .build()?;
        Ok(runtime.block_on(async {
            let shutdown = CancellationToken::new();
            let orchestrator = prl_core::run(config, bridge, wandb, nixl, shutdown.clone());
            tokio::pin!(orchestrator);
            let mut interrupt = None;
            let mut poll = tokio::time::interval(SIGNAL_POLL);
            let result = loop {
                tokio::select! {
                    result = &mut orchestrator => break result,
                    _ = poll.tick(), if interrupt.is_none() => {
                        if let Err(err) = Python::with_gil(|py| py.check_signals()) {
                            tracing::warn!(target: "prl_py", "Received {err}, shutting down the orchestrator");
                            shutdown.cancel();
                            interrupt = Some(err);
                        }
                    }
                }
            };
            (result, interrupt)
        }))
    })
    .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    if let Some(interrupt) = interrupt {
        return Err(interrupt);
    }
    result.map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
}

#[pymodule]
fn prime_rl_orch_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(eval::run_eval, m)?)?;
    m.add_function(wrap_pyfunction!(eval::run_online_eval, m)?)?;
    Ok(())
}
