//! Python extension module `prime_rl_orch_rs`: runs the Rust orchestrator (`run`), `uv run eval`
//! (`run_eval`) and the SFT online evals (`run_online_eval`). Python parses and validates the
//! pydantic config and calls back into `prime_rl.rust_bridge.PyBridge` for what stays in Python.

mod bridge;

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use prl_config::WeightBroadcastConfig;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio_util::sync::CancellationToken;

use crate::bridge::{PyBridge, PyNixl, PyWandb};

/// How often the main thread runs Python's signal handlers.
const SIGNAL_POLL: Duration = Duration::from_millis(100);

fn runtime_error(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err:#}"))
}

fn nixl(
    py: Python<'_>,
    bridge: &Py<PyAny>,
    config: Option<&WeightBroadcastConfig>,
) -> PyResult<Option<Box<dyn prl_io::weights::NixlBridge>>> {
    match config {
        Some(WeightBroadcastConfig::Nixl(_)) => PyNixl::from_bridge(py, bridge)
            .map(Some)
            .map_err(runtime_error),
        _ => Ok(None),
    }
}

/// Run `run(shutdown)` on a multi-thread tokio runtime with the GIL released.
///
/// Python runs signal handlers only on the main thread, so the `block_on` future (which runs on the
/// calling thread) polls them. A handler that raises (Ctrl-C's `KeyboardInterrupt`, a launcher's
/// SIGTERM handler) cancels `shutdown`; the exception is re-raised once `run` has stopped.
fn run_until_signal<Fut>(
    py: Python<'_>,
    name: &str,
    run: impl FnOnce(CancellationToken) -> Fut + Send,
) -> PyResult<()>
where
    Fut: Future<Output = anyhow::Result<()>>,
{
    let (result, interrupt) = py
        .allow_threads(|| -> anyhow::Result<_> {
            let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().thread_name("prl-orch").build()?;
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
                            if let Some((exception, err)) = Python::with_gil(|py| py.check_signals().err().map(|err| (err.get_type(py).to_string(), err))) {
                                tracing::warn!(target: "prl_py", "Received {exception}, shutting down the {name}");
                                shutdown.cancel();
                                interrupt = Some(err);
                            }
                        }
                    }
                };
                (result, interrupt)
            }))
        })
        .map_err(runtime_error)?;
    match interrupt {
        Some(err) => Err(err),
        None => result.map_err(runtime_error),
    }
}

/// Run the orchestrator until it exits. `config` is the validated `OrchestratorConfig`.
#[pyfunction]
fn run(py: Python<'_>, config: &Bound<'_, PyAny>, bridge: Py<PyAny>) -> PyResult<()> {
    let config: prl_config::OrchestratorConfig = config.extract()?;
    prl_io::logging::setup_logger(&config.log.level, config.log.json_logging, None, None)
        .map_err(runtime_error)?;
    let wandb = PyWandb::from_bridge(py, &bridge)?;
    let nixl = nixl(py, &bridge, Some(&config.weight_broadcast))?;
    let bridge = Arc::new(PyBridge::new(bridge));
    run_until_signal(py, "orchestrator", |shutdown| {
        prl_core::run(config, bridge, wandb, nixl, shutdown)
    })
}

/// Run `uv run eval` until it exits. `config` is the validated `EvalConfig`; Rust logs to the
/// console (at `console_level` and above, when set) and to `log_file`.
#[pyfunction]
#[pyo3(signature = (config, bridge, log_file=None, console_level=None))]
fn run_eval(
    py: Python<'_>,
    config: &Bound<'_, PyAny>,
    bridge: Py<PyAny>,
    log_file: Option<PathBuf>,
    console_level: Option<String>,
) -> PyResult<()> {
    let monitors_repr = config.getattr("monitors")?.str()?.to_string();
    let config: prl_config::EvalConfig = config.extract()?;
    prl_io::logging::setup_logger(
        &config.log.level,
        config.log.json_logging,
        log_file.as_deref(),
        console_level.as_deref(),
    )
    .map_err(runtime_error)?;
    let wandb = PyWandb::from_bridge(py, &bridge)?;
    let bridge = Arc::new(PyBridge::new(bridge));
    run_until_signal(py, "eval", move |shutdown| async move {
        prl_eval::run_eval(config, bridge, wandb, &monitors_repr, shutdown).await
    })
}

/// Run the SFT online-eval process until it exits. `config` is the validated `SFTOnlineEvalConfig`.
#[pyfunction]
fn run_online_eval(py: Python<'_>, config: &Bound<'_, PyAny>, bridge: Py<PyAny>) -> PyResult<()> {
    let monitors_repr = config.getattr("monitors")?.str()?.to_string();
    let config: prl_config::SFTOnlineEvalConfig = config.extract()?;
    prl_io::logging::setup_logger(&config.log.level, config.log.json_logging, None, None)
        .map_err(runtime_error)?;
    let wandb = PyWandb::from_bridge(py, &bridge)?;
    let nixl = nixl(py, &bridge, config.weight_broadcast.as_ref())?;
    let bridge = Arc::new(PyBridge::new(bridge));
    run_until_signal(py, "online evals", move |shutdown| async move {
        prl_eval::run_online_eval(config, bridge, wandb, nixl, &monitors_repr, shutdown).await
    })
}

#[pymodule]
fn prime_rl_orch_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_eval, m)?)?;
    m.add_function(wrap_pyfunction!(run_online_eval, m)?)?;
    Ok(())
}
