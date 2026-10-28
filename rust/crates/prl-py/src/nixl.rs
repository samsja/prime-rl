//! `prl_io::weights::NixlBridge` over the Python `NixlReceiverBridge` (`PyBridge.nixl_receiver()`):
//! the consumer's NIXL agent and ModelExpress session. Calls run on the blocking pool and take
//! the GIL there; `wait_ready` blocks in Python (sleeping, so the GIL is released between polls).

use std::sync::Arc;

use anyhow::{Context, Result};
use pyo3::prelude::*;

use crate::bridge::python_error;

pub struct PyNixl(Arc<Py<PyAny>>);

impl PyNixl {
    /// Build the Python NIXL receiver bridge (`bridge.nixl_receiver()`). Only for `nixl` weight broadcasts.
    pub fn from_bridge(py: Python<'_>, bridge: &Py<PyAny>) -> Result<Box<dyn prl_io::weights::NixlBridge>> {
        let receiver = bridge
            .call_method0(py, "nixl_receiver")
            .map_err(|e| python_error(py, e))
            .context("PyBridge.nixl_receiver failed")?;
        Ok(Box::new(PyNixl(Arc::new(receiver))))
    }

    async fn call<A>(&self, method: &'static str, args: A) -> Result<()>
    where
        A: for<'py> pyo3::call::PyCallArgs<'py> + Send + 'static,
    {
        let obj = Arc::clone(&self.0);
        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                obj.bind(py)
                    .call_method1(method, args)
                    .map_err(|e| python_error(py, e))
                    .with_context(|| format!("NixlReceiverBridge.{method} failed"))?;
                Ok(())
            })
        })
        .await?
    }
}

#[async_trait::async_trait]
impl prl_io::weights::NixlBridge for PyNixl {
    async fn publish(&mut self) -> Result<()> {
        self.call("publish", ()).await
    }

    async fn wait_ready(&mut self, step: u64, timeout_s: u64) -> Result<()> {
        self.call("wait_ready", (step, timeout_s as f64)).await
    }

    async fn send_complete(&mut self, step: u64) -> Result<()> {
        self.call("send_complete", (step,)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// The Rust wrapper drives a Python receiver bridge with the right methods and arguments.
    #[tokio::test]
    async fn drives_python_bridge() {
        let (bridge, calls) = Python::with_gil(|py| {
            let code = CString::new(
                r#"
calls = []
class Receiver:
    def publish(self): calls.append(("publish",))
    def wait_ready(self, step, timeout): calls.append(("wait_ready", step, timeout))
    def send_complete(self, step): calls.append(("send_complete", step))
class Bridge:
    def nixl_receiver(self): return Receiver()
bridge = Bridge()
"#,
            )
            .unwrap();
            let module = PyModule::from_code(py, &code, c"nixl_test.py", c"nixl_test").unwrap();
            (module.getattr("bridge").unwrap().unbind(), module.getattr("calls").unwrap().unbind())
        });
        let mut nixl = Python::with_gil(|py| PyNixl::from_bridge(py, &bridge)).unwrap();
        nixl.publish().await.unwrap();
        nixl.wait_ready(3, 60).await.unwrap();
        nixl.send_complete(3).await.unwrap();
        let recorded = Python::with_gil(|py| calls.bind(py).repr().unwrap().to_string());
        assert_eq!(recorded, "[('publish',), ('wait_ready', 3, 60.0), ('send_complete', 3)]");
    }

    /// Rust drives the real Python `NixlReceiverBridge` against a trainer-side NIXL agent over
    /// UCX/TCP (ModelExpress faked). Needs the project venv on PYTHONPATH, libpython on
    /// LD_LIBRARY_PATH and `PRL_NIXL_LIVE=1`, e.g. `UCX_NET_DEVICES=eth0 UCX_TLS=tcp,self,sm`.
    #[tokio::test]
    async fn real_nixl_handshake() {
        if std::env::var_os("PRL_NIXL_LIVE").is_none() {
            eprintln!("skipped: set PRL_NIXL_LIVE=1");
            return;
        }
        let (bridge, trainer) = Python::with_gil(|py| {
            let code = CString::new(
                r#"
import threading
from types import SimpleNamespace
from prime_rl.configs.shared import NIXLWeightBroadcastConfig
from prime_rl.transports.weights.nixl import bridge as bridge_module
from prime_rl.transports.weights.nixl.agent import NixlAgent, policy_notification
from prime_rl.transports.weights.nixl.trainer_tensor_table import TrainerAgent, TrainerTensorTable

trainer = NixlAgent("trainer-live-test-r0")
table = TrainerTensorTable(agents=[TrainerAgent(name="trainer-live-test-r0", metadata=trainer.get_metadata(), device_id=0)], staging_buffer_count=1, groups=[])
published = {}

class FakeSession:
    def __init__(self, **kwargs): pass
    def publish(self, *, nixl_metadata=b""):
        published["orchestrator"] = nixl_metadata
        threading.Thread(target=trainer_side, daemon=True).start()
    def wait_for(self, role, *, count, timeout): return ["trainer-ref"]
    def fetch(self, ref): return SimpleNamespace(nixl_metadata=table.encode())

result = {}
def trainer_side():
    peer = trainer.add_remote_agent(published["orchestrator"])
    trainer.make_connection(peer)
    trainer.send_notification(peer, policy_notification(7, "ready"))
    trainer.wait_for_notification([peer], policy_notification(7, "complete"), timeout=30)
    result["complete"] = True

bridge_module.ModelExpressSession = FakeSession
class Bridge:
    config = SimpleNamespace(weight_broadcast=NIXLWeightBroadcastConfig(host="localhost", port=8001))
    def nixl_receiver(self): return bridge_module.NixlReceiverBridge(self.config.weight_broadcast)
bridge = Bridge()
"#,
            )
            .unwrap();
            let module = PyModule::from_code(py, &code, c"nixl_live.py", c"nixl_live").unwrap();
            (module.getattr("bridge").unwrap().unbind(), module.getattr("result").unwrap().unbind())
        });
        let mut nixl = Python::with_gil(|py| PyNixl::from_bridge(py, &bridge)).unwrap();
        nixl.publish().await.unwrap();
        nixl.wait_ready(7, 30).await.unwrap();
        nixl.send_complete(7).await.unwrap();
        for _ in 0..300 {
            if Python::with_gil(|py| trainer.bind(py).contains("complete").unwrap()) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        panic!("the trainer never received the complete notification");
    }
}
