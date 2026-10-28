//! Extract every orchestrator and eval config under `configs/` and `examples/` from the real
//! pydantic objects, so a renamed or removed pydantic field fails here.

use std::collections::HashMap;
use std::ffi::CString;

use prl_config::{EvalConfig, OrchestratorConfig, SFTOnlineEvalConfig};
use pyo3::prelude::*;

/// Loads each TOML as the first config class that validates it. Taskset packages resolve
/// their config classes at validation time and are not necessarily installed, so their
/// source roots go on the path.
const LOAD: &str = r#"
import contextlib, io, os, site, sys, tomllib
from pathlib import Path

root = Path(ROOT).resolve()
site.addsitedir(str(next((root / ".venv" / "lib").glob("python3*/site-packages"))))
for env_dir in [*root.glob("deps/verifiers/environments/*"), *root.glob("deps/prime-envs/environments/*/*")]:
    sys.path.append(str(env_dir))
os.environ.pop("PRL_OUTPUT_DIR", None)
os.chdir(root)

from prime_rl.configs.eval import EvalConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.utils.config import cli

def load(cls, path):
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            return cli(cls, args=["@", str(path), "--output-dir", "outputs"])
    except (SystemExit, Exception):
        return None

loaded = {"orchestrator": [], "eval": [], "sft_online_eval": []}
for path in sorted([*root.glob("configs/**/*.toml"), *root.glob("examples/**/*.toml")]):
    if "orchestrator" in tomllib.loads(path.read_text()):
        kinds = [("orchestrator", RLConfig, lambda c: c.orchestrator)]
    else:
        kinds = [
            ("orchestrator", OrchestratorConfig, lambda c: c),
            ("eval", EvalConfig, lambda c: c),
            ("sft_online_eval", SFTConfig, lambda c: c.eval),
        ]
    for kind, cls, pick in kinds:
        if (config := load(cls, path)) is not None and pick(config) is not None:
            loaded[kind].append((str(path.relative_to(root)), pick(config)))
            break
"#;

fn extract_all<'py, T: FromPyObject<'py>>(configs: &[(String, Bound<'py, PyAny>)]) -> Vec<T> {
    configs
        .iter()
        .map(|(name, obj)| obj.extract().unwrap_or_else(|e| panic!("{name}: {e}")))
        .collect()
}

#[test]
fn extracts_every_config() {
    // SAFETY: set before the interpreter starts, while this is the only thread reading the env.
    unsafe { std::env::set_var("PYTHONHOME", env!("PRL_TEST_PYTHONHOME")) };
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
        let code = CString::new(format!("ROOT = {root:?}\n{LOAD}")).unwrap();
        let globals = pyo3::types::PyDict::new(py);
        py.run(&code, Some(&globals), None).unwrap();
        let loaded: HashMap<String, Vec<(String, Bound<'_, PyAny>)>> = globals
            .get_item("loaded")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        let orchestrator: Vec<OrchestratorConfig> = extract_all(&loaded["orchestrator"]);
        assert!(
            orchestrator.len() >= 40,
            "only {} orchestrator configs",
            orchestrator.len()
        );
        for config in &orchestrator {
            let mut names: Vec<&str> = config
                .train
                .source
                .iter()
                .map(|s| s.resolved_name.as_str())
                .collect();
            names.extend(
                config
                    .eval
                    .iter()
                    .flat_map(|e| e.source.iter().map(|s| s.resolved_name.as_str())),
            );
            let addressed: Vec<&str> = config
                .env_addresses
                .keys()
                .map(|(_, name)| name.as_str())
                .collect();
            assert_eq!(addressed, names);
        }
        let eval: Vec<EvalConfig> = extract_all(&loaded["eval"]);
        assert!(eval.len() >= 10, "only {} eval configs", eval.len());
        let online: Vec<SFTOnlineEvalConfig> = extract_all(&loaded["sft_online_eval"]);
        assert!(!online.is_empty(), "no sft online eval config");
    });
}
