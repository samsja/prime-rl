//! `PyBridge` calls into a Python object from tokio's blocking pool.

use std::ffi::CString;

use prime_rl_orch_rs::bridge::PyBridge;
use prl_types::{Bridge, Kind};
use pyo3::prelude::*;
use serde_json::json;

const FAKE_BRIDGE: &str = r#"
import json

class FakeBridge:
    def load_tasks(self, split, env_name):
        if env_name == "missing":
            raise KeyError(f"No {split} env named {env_name!r}")
        tasks = [{"key": f"k{i}", "hash": f"h{i}", "data": {"idx": i, "split": split}} for i in range(3)]
        return json.dumps({"num_tasks": 3, "tasks": tasks}).encode()

    def next_tasks(self, split, env_name, n):
        return json.dumps([{"key": "s", "hash": "s", "data": {"n": n}}]).encode()

    def flop_cost_table(self, max_len):
        return json.dumps([str(n * 2**70) for n in range(max_len + 1)]).encode()

    def render_ids(self, model_name, renderer_config, system_content):
        assert json.loads(renderer_config) == {"name": "auto"}
        return json.dumps([len(model_name), len(system_content)]).encode()

    def echo_filter(self, import_path, kwargs, trace):
        assert isinstance(trace, bytes)
        n = len(json.loads(trace)["nodes"])
        return json.dumps([[i < json.loads(kwargs)["keep"] for i in range(n)]]).encode()
"#;

fn fake_bridge() -> PyBridge {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // SAFETY: set once, before the interpreter starts and before other threads read the env.
        unsafe { std::env::set_var("PYTHONHOME", env!("PRL_TEST_PYTHONHOME")) };
        pyo3::prepare_freethreaded_python();
    });
    Python::with_gil(|py| {
        let code = CString::new(FAKE_BRIDGE).unwrap();
        let module = PyModule::from_code(py, &code, c"fake_bridge.py", c"fake_bridge").unwrap();
        PyBridge::new(module.getattr("FakeBridge").unwrap().call0().unwrap().unbind())
    })
}

/// Run `f` on a multi-thread runtime with the GIL released, as `prime_rl_orch_rs.run` does.
fn run_released<T: Send>(f: impl std::future::Future<Output = T> + Send) -> T {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
            runtime.block_on(f)
        })
    })
}

#[test]
fn tasks_round_trip() {
    let bridge = fake_bridge();
    let (loaded, next) = run_released(async {
        let (a, b) = tokio::join!(bridge.load_tasks(Kind::Train, "rt"), bridge.next_tasks(Kind::Eval, "rt", 7));
        (a.unwrap(), b.unwrap())
    });
    assert_eq!(loaded.num_tasks, Some(3));
    assert_eq!(loaded.tasks[2].key, "k2");
    assert_eq!(loaded.tasks[0].data["split"], "train");
    assert_eq!(next[0].data["n"], 7);
}

#[test]
fn python_exception_becomes_error_with_traceback() {
    let bridge = fake_bridge();
    let err = run_released(async { bridge.load_tasks(Kind::Eval, "missing").await.unwrap_err() });
    let err = format!("{err:#}");
    assert!(err.contains("PyBridge.load_tasks failed"), "{err}");
    assert!(err.contains("KeyError"), "{err}");
    assert!(err.contains("fake_bridge.py"), "{err}");
}

#[test]
fn train_helpers() {
    let bridge = fake_bridge();
    let (flops, ids, masks) = run_released(async {
        (
            bridge.flop_cost_table(2).await.unwrap(),
            bridge.render_ids("model", json!({"name": "auto"}), "hint").await.unwrap(),
            bridge.echo_filter("m.f", json!({"keep": 1}), json!({"nodes": [{}, {}]})).await.unwrap(),
        )
    });
    assert_eq!(flops, vec![0, 1i128 << 70, 2i128 << 70]);
    assert_eq!(ids, vec![5, 4]);
    assert_eq!(masks, vec![vec![true, false]]);
}
