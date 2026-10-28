//! Rust handles onto the Python objects of `prime_rl.rust_bridge`: the `PyBridge` itself, its
//! W&B monitor and the NIXL receiver. Python exceptions become errors carrying their traceback.

use std::sync::Arc;

use anyhow::{Context, Result};
use prl_types::{Bridge, Kind, LoadedTasks, TaskRecord};
use pyo3::call::PyCallArgs;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::de::DeserializeOwned;

/// `obj.method(*args)` with the GIL held.
fn call<'py>(
    obj: &Bound<'py, PyAny>,
    method: &str,
    args: impl PyCallArgs<'py>,
) -> Result<Bound<'py, PyAny>> {
    obj.call_method1(method, args).map_err(|err| {
        let traceback = err
            .traceback(obj.py())
            .and_then(|tb| tb.format().ok())
            .unwrap_or_default();
        let class = obj
            .get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_default();
        anyhow::anyhow!("{traceback}{err}").context(format!("{class}.{method} failed"))
    })
}

/// `call` on tokio's blocking pool, so reactor threads never wait for the GIL.
async fn call_blocking<A, T>(
    obj: &Arc<Py<PyAny>>,
    method: &'static str,
    args: A,
    result: impl FnOnce(&Bound<'_, PyAny>) -> Result<T> + Send + 'static,
) -> Result<T>
where
    A: for<'py> PyCallArgs<'py> + Send + 'static,
    T: Send + 'static,
{
    let obj = Arc::clone(obj);
    tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| result(&call(obj.bind(py), method, args)?))
    })
    .await?
}

/// `prime_rl.rust_bridge.PyBridge`: its methods return JSON-encoded `bytes`.
pub struct PyBridge(Arc<Py<PyAny>>);

impl PyBridge {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self(Arc::new(obj))
    }

    async fn call_json<A, T>(&self, method: &'static str, args: A) -> Result<T>
    where
        A: for<'py> PyCallArgs<'py> + Send + 'static,
        T: DeserializeOwned + Send + 'static,
    {
        call_blocking(&self.0, method, args, move |result| {
            let bytes = result
                .downcast::<PyBytes>()
                .map_err(|e| anyhow::anyhow!("PyBridge.{method}: {e}"))?;
            serde_json::from_slice(bytes.as_bytes())
                .with_context(|| format!("PyBridge.{method} returned malformed JSON"))
        })
        .await
    }
}

#[async_trait::async_trait]
impl Bridge for PyBridge {
    async fn load_tasks(&self, kind: Kind, env_name: &str) -> Result<LoadedTasks> {
        self.call_json("load_tasks", (kind.as_str(), env_name.to_owned()))
            .await
    }

    async fn next_tasks(&self, kind: Kind, env_name: &str, n: usize) -> Result<Vec<TaskRecord>> {
        self.call_json("next_tasks", (kind.as_str(), env_name.to_owned(), n))
            .await
    }

    async fn flop_cost_table(&self, max_len: usize) -> Result<Vec<i128>> {
        // Decimal strings: the costs can exceed 64 bits.
        let table: Vec<String> = self.call_json("flop_cost_table", (max_len,)).await?;
        table
            .iter()
            .map(|cost| {
                cost.parse()
                    .with_context(|| format!("invalid flop cost {cost:?}"))
            })
            .collect()
    }

    async fn render_ids(
        &self,
        model_name: &str,
        renderer_config: serde_json::Value,
        system_content: &str,
    ) -> Result<Vec<i32>> {
        self.call_json(
            "render_ids",
            (
                model_name.to_owned(),
                renderer_config.to_string(),
                system_content.to_owned(),
            ),
        )
        .await
    }

    async fn echo_filter(
        &self,
        import_path: &str,
        kwargs: serde_json::Value,
        trace: serde_json::Value,
    ) -> Result<Vec<Vec<bool>>> {
        self.call_json(
            "echo_filter",
            (
                import_path.to_owned(),
                kwargs.to_string(),
                trace.to_string(),
            ),
        )
        .await
    }
}

/// `PyBridge.wandb`, a `prime_rl.monitors.wandb.bridge.WandbBridge`. prl-monitors calls it from a
/// blocking thread.
pub struct PyWandb(Py<PyAny>);

impl PyWandb {
    /// `None` when `monitors.wandb` is not configured.
    pub fn from_bridge(
        py: Python<'_>,
        bridge: &Py<PyAny>,
    ) -> PyResult<Option<Box<dyn prl_monitors::WandbBridge>>> {
        let wandb = bridge.getattr(py, "wandb")?;
        Ok((!wandb.is_none(py))
            .then(|| Box::new(PyWandb(wandb)) as Box<dyn prl_monitors::WandbBridge>))
    }
}

impl prl_monitors::WandbBridge for PyWandb {
    fn init(
        &mut self,
        train_env_names: &[String],
        eval_env_names: &[String],
        overview_flavor: &str,
    ) -> Result<()> {
        Python::with_gil(|py| {
            call(
                self.0.bind(py),
                "init",
                (train_env_names, eval_env_names, overview_flavor),
            )
            .map(drop)
        })
    }

    fn log(&mut self, row_json: &str, step: Option<u64>) -> Result<()> {
        Python::with_gil(|py| call(self.0.bind(py), "log", (row_json, step)).map(drop))
    }

    fn finish(&mut self, paused: bool) -> Result<()> {
        Python::with_gil(|py| call(self.0.bind(py), "finish", (paused,)).map(drop))
    }
}

/// `PyBridge.nixl_receiver()`, a `prime_rl.transports.weights.nixl.bridge.NixlReceiverBridge`.
pub struct PyNixl(Arc<Py<PyAny>>);

impl PyNixl {
    pub fn from_bridge(
        py: Python<'_>,
        bridge: &Py<PyAny>,
    ) -> Result<Box<dyn prl_io::weights::NixlBridge>> {
        let receiver = call(bridge.bind(py), "nixl_receiver", ())?;
        Ok(Box::new(PyNixl(Arc::new(receiver.unbind()))))
    }
}

#[async_trait::async_trait]
impl prl_io::weights::NixlBridge for PyNixl {
    async fn publish(&mut self) -> Result<()> {
        call_blocking(&self.0, "publish", (), |_| Ok(())).await
    }

    async fn wait_ready(&mut self, step: u64, timeout_s: u64) -> Result<()> {
        call_blocking(&self.0, "wait_ready", (step, timeout_s as f64), |_| Ok(())).await
    }

    async fn send_complete(&mut self, step: u64) -> Result<()> {
        call_blocking(&self.0, "send_complete", (step,), |_| Ok(())).await
    }
}
