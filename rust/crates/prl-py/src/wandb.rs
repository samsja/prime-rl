//! `prl_monitors::WandbBridge` over the Python `WandbBridge` (`PyBridge.wandb`).
//! prl-monitors calls it from a blocking thread; each call takes the GIL.

use anyhow::{Context, Result};
use pyo3::prelude::*;

use crate::bridge::python_error;

pub struct PyWandb(Py<PyAny>);

impl PyWandb {
    /// `bridge.wandb` as a W&B bridge, or `None` when `monitors.wandb` is not configured.
    pub fn from_bridge(py: Python<'_>, bridge: &Py<PyAny>) -> PyResult<Option<Box<dyn prl_monitors::WandbBridge>>> {
        let wandb = bridge.getattr(py, "wandb")?;
        Ok((!wandb.is_none(py)).then(|| Box::new(PyWandb(wandb)) as Box<dyn prl_monitors::WandbBridge>))
    }

    fn call<'py, A: pyo3::call::PyCallArgs<'py>>(&self, py: Python<'py>, method: &str, args: A) -> Result<()> {
        self.0
            .bind(py)
            .call_method1(method, args)
            .map_err(|e| python_error(py, e))
            .with_context(|| format!("WandbBridge.{method} failed"))?;
        Ok(())
    }
}

impl prl_monitors::WandbBridge for PyWandb {
    fn init(&mut self, train_env_names: &[String], eval_env_names: &[String], overview_flavor: &str) -> Result<()> {
        Python::with_gil(|py| self.call(py, "init", (train_env_names.to_vec(), eval_env_names.to_vec(), overview_flavor)))
    }

    fn log(&mut self, row_json: &str, step: Option<u64>) -> Result<()> {
        Python::with_gil(|py| self.call(py, "log", (row_json, step)))
    }

    fn finish(&mut self, paused: bool) -> Result<()> {
        Python::with_gil(|py| self.call(py, "finish", (paused,)))
    }
}
