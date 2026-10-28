//! Rust handle onto the Python `PyBridge` (`prime_rl.orchestrator.rust.PyBridge`).
//!
//! Every call runs on tokio's blocking pool and takes the GIL there, so reactor threads never
//! block on Python. Bridge methods return JSON-encoded `bytes`, decoded here with serde.

use std::sync::Arc;

use anyhow::{Context, Result};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use prl_types::{Bridge, Kind, LoadedTasks, TaskRecord};
use serde::de::DeserializeOwned;

#[derive(Clone)]
pub struct PyBridge {
    obj: Arc<Py<PyAny>>,
}

impl PyBridge {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj: Arc::new(obj) }
    }

    /// Call `method(*args)` on the Python bridge and decode its JSON `bytes` result.
    pub async fn call_json<A, T>(&self, method: &'static str, args: A) -> Result<T>
    where
        A: for<'py> pyo3::call::PyCallArgs<'py> + Send + 'static,
        T: DeserializeOwned + Send + 'static,
    {
        let obj = Arc::clone(&self.obj);
        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> Result<T> {
                let result = obj
                    .bind(py)
                    .call_method1(method, args)
                    .map_err(|e| python_error(py, e))
                    .with_context(|| format!("PyBridge.{method} failed"))?;
                let bytes = result
                    .downcast::<PyBytes>()
                    .map_err(|e| anyhow::anyhow!("PyBridge.{method} must return bytes: {e}"))?;
                serde_json::from_slice(bytes.as_bytes())
                    .with_context(|| format!("PyBridge.{method} returned malformed JSON"))
            })
        })
        .await?
    }
}

#[async_trait::async_trait]
impl Bridge for PyBridge {
    async fn load_tasks(&self, kind: Kind, env_name: &str) -> Result<LoadedTasks> {
        self.call_json("load_tasks", (kind.as_str(), env_name.to_owned())).await
    }

    async fn next_tasks(&self, kind: Kind, env_name: &str, n: usize) -> Result<Vec<TaskRecord>> {
        self.call_json("next_tasks", (kind.as_str(), env_name.to_owned(), n)).await
    }

    async fn flop_cost_table(&self, max_len: usize) -> Result<Vec<i128>> {
        let table: Vec<String> = self.call_json("flop_cost_table", (max_len,)).await?;
        table
            .iter()
            .map(|cost| cost.parse().with_context(|| format!("invalid flop cost {cost:?}")))
            .collect()
    }

    async fn render_ids(
        &self,
        model_name: &str,
        renderer_config: serde_json::Value,
        system_content: &str,
    ) -> Result<Vec<i32>> {
        let args = (model_name.to_owned(), renderer_config.to_string(), system_content.to_owned());
        self.call_json("render_ids", args).await
    }

    async fn echo_filter(
        &self,
        import_path: &str,
        kwargs: serde_json::Value,
        trace: serde_json::Value,
    ) -> Result<Vec<Vec<bool>>> {
        let args = (import_path.to_owned(), kwargs.to_string(), PyBytesArg(serde_json::to_vec(&trace)?));
        self.call_json("echo_filter", args).await
    }
}

/// Passes a byte buffer to Python as `bytes` (a bare `Vec<u8>` converts to `list[int]`).
struct PyBytesArg(Vec<u8>);

impl<'py> IntoPyObject<'py> for PyBytesArg {
    type Target = PyBytes;
    type Output = Bound<'py, PyBytes>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> std::result::Result<Self::Output, Self::Error> {
        Ok(PyBytes::new(py, &self.0))
    }
}

/// A Python exception as an error carrying its formatted traceback.
fn python_error(py: Python<'_>, err: PyErr) -> anyhow::Error {
    let traceback = err
        .traceback(py)
        .and_then(|tb| tb.format().ok())
        .unwrap_or_default();
    anyhow::anyhow!("{traceback}{err}")
}
