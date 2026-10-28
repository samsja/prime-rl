//! Python extension module `prime_rl_orch_rs`: entry point of the Rust orchestrator.

pub mod bridge;
pub mod eval;
pub mod nixl;
pub mod wandb;

use std::future::Future;
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
/// Signals: see `run_until_signal`.
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
    run_until_signal(py, "orchestrator", |shutdown| prl_core::run(config, bridge, wandb, nixl, shutdown))
}

/// Run `run(shutdown)` to completion on a fresh multi-thread tokio runtime, with the GIL released.
///
/// Python only runs signal handlers on the main thread between bytecodes, so the main thread polls
/// them (the `block_on` future runs on the calling thread, which is Python's main thread). A
/// handler that raises (Ctrl-C's `KeyboardInterrupt`, a launcher's SIGTERM handler) cancels
/// `shutdown`, the way cancelling the Python main task did; the exception is re-raised once `run`
/// has stopped. An `Err` from `run` is raised as `RuntimeError`.
pub fn run_until_signal<F, Fut>(py: Python<'_>, name: &str, run: F) -> PyResult<()>
where
    F: FnOnce(CancellationToken) -> Fut + Send,
    Fut: Future<Output = anyhow::Result<()>>,
{
    let (result, interrupt) = py
        .allow_threads(|| -> anyhow::Result<_> {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("prl-orch")
                .build()?;
            Ok(runtime.block_on(async {
                let shutdown = CancellationToken::new();
                let task = run(shutdown.clone());
                tokio::pin!(task);
                let mut interrupt = None;
                let mut poll = tokio::time::interval(SIGNAL_POLL);
                let result = loop {
                    tokio::select! {
                        result = &mut task => break result,
                        _ = poll.tick(), if interrupt.is_none() => {
                            if let Err(err) = Python::with_gil(|py| py.check_signals()) {
                                tracing::warn!(target: "prl_py", "Received {err}, shutting down the {name}");
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

/// Start an embedded interpreter for tests: the build interpreter's home and the project venv's
/// site-packages, so tests can import `prime_rl` and its dependencies. Call before any `with_gil`.
#[doc(hidden)]
pub fn init_test_python() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // SAFETY: set once, before the interpreter starts and before other threads read the env.
        unsafe {
            std::env::set_var("PYTHONHOME", env!("PRL_TEST_PYTHONHOME"));
            std::env::set_var("PYTHONPATH", env!("PRL_TEST_PYTHONPATH"));
        }
        pyo3::prepare_freethreaded_python();
    });
}

#[pymodule]
fn prime_rl_orch_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(eval::run_eval, m)?)?;
    m.add_function(wrap_pyfunction!(eval::run_online_eval, m)?)?;
    Ok(())
}
