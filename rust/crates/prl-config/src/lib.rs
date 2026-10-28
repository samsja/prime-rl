//! The orchestrator and eval configs, extracted from the validated pydantic objects
//! (`OrchestratorConfig`, `EvalConfig`, `SFTOnlineEvalConfig`).
//!
//! Pydantic stays the only config schema and does all validation. The Python entrypoint
//! passes the validated `OrchestratorConfig` object to Rust, which extracts it once at
//! startup with [`OrchestratorConfig::extract`] (pyo3 `FromPyObject`, attribute access).
//! After that Rust holds plain values and never touches the GIL to read config.
//!
//! Only the fields the orchestrator reads are declared. Values the Python code derives
//! (`resolved_name`, `env_id`, `env_addresses`, `truncates_distribution()`,
//! `resume.dir_step`, `eval.intervals`, `any_policy_sourced`, algorithm `action_loss_type`)
//! are read off the Python object, never recomputed here. Sub-configs Rust only forwards
//! (renderer, eval sampling, the full config for monitors) are `model_dump(mode="json")`ed
//! into [`serde_json::Value`]. Every struct has public fields, so other crates can build
//! configs in plain Rust for tests.

mod algorithm;
mod eval;
mod monitors;
mod orchestrator;
mod py;
mod shared;

pub use algorithm::*;
pub use eval::*;
pub use monitors::*;
pub use orchestrator::*;
pub use shared::*;
