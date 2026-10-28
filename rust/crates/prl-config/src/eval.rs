//! `prime_rl.configs.eval`: `uv run eval` (`EvalConfig`) and the sft online-eval
//! process (`SFTOnlineEvalConfig`).

use std::path::PathBuf;

use indexmap::IndexMap;
use pyo3::prelude::*;
use serde_json::Value;

use crate::monitors::{FileMonitorConfig, WandbMonitorConfig};
use crate::orchestrator::{ConcurrencyConfig, OnlineEvalSourceConfig, Split};
use crate::py::{model_dump, model_dump_exclude_none};
use crate::shared::{
    ClientConfig, ControlConfig, HeartbeatConfig, LogConfig, WeightBroadcastConfig,
};

/// `EvalSourceConfig`: one eval environment of `uv run eval`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct EvalSourceConfig {
    /// `EnvConfig.resolved_name`.
    pub resolved_name: String,
    /// `EnvConfig.env_id`.
    pub env_id: String,
    /// `verifiers.v1.SamplingConfig` dump, sent to the env server as is.
    #[pyo3(from_py_with = model_dump)]
    pub sampling: Value,
    pub group_size: u64,
}

/// `ServedEvalConfig`: eval sources against a live inference server. These fields are
/// shared by [`EvalConfig`] (sources [`EvalSourceConfig`]) and [`SFTOnlineEvalConfig`]
/// (sources [`OnlineEvalSourceConfig`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ServedEvalConfig<S> {
    pub source: Vec<S>,
    /// Client of the inference server (`ClientConfig` or `VLLMClientConfig`).
    pub client: ClientConfig,
    pub concurrency: ConcurrencyConfig,
    pub dispatch_per_minute: Option<u64>,
    pub control: Option<ControlConfig>,
    pub heartbeat: Option<HeartbeatConfig>,
    /// `ServedEvalConfig.env_addresses`: `("eval", resolved_name)` -> `serve.address`, or
    /// None when the launcher spawns the server.
    pub env_addresses: IndexMap<(Split, String), Option<String>>,
}

impl<'py, S: FromPyObject<'py>> FromPyObject<'py> for ServedEvalConfig<S> {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let attr = |name: &str| obj.getattr(name);
        Ok(Self {
            source: attr("source")?.extract()?,
            client: attr("client")?.extract()?,
            concurrency: attr("concurrency")?.extract()?,
            dispatch_per_minute: attr("dispatch_per_minute")?.extract()?,
            control: attr("control")?.extract()?,
            heartbeat: attr("heartbeat")?.extract()?,
            env_addresses: attr("env_addresses")?.extract()?,
        })
    }
}

/// `PrimeEvalMonitorConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct PrimeEvalMonitorConfig {
    /// Evaluation name on the platform (defaulted to `run.name` by `EvalConfig`).
    pub name: Option<String>,
}

/// `EvalMonitorsConfig`; None disables a monitor.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct EvalMonitorsConfig {
    pub wandb: Option<WandbMonitorConfig>,
    pub file: Option<FileMonitorConfig>,
    pub prime: Option<PrimeEvalMonitorConfig>,
}

/// `RunConfig`. Both are set by `EvalConfig`'s validator.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct RunConfig {
    pub name: Option<String>,
    /// Run directory name under `output_dir`.
    pub dir: Option<String>,
}

/// `EvalResumeConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct EvalResumeConfig {
    /// Reuse saved episodes even when resolved configs differ or snapshots are missing.
    pub skip_checks: bool,
}

/// `EvalConfig` (`uv run eval`), extracted once at startup from the validated object.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalConfig {
    pub served: ServedEvalConfig<EvalSourceConfig>,
    /// The `model` of every eval request.
    pub model: String,
    pub run: RunConfig,
    pub output_dir: PathBuf,
    /// `EvalConfig.run_dir`: `output_dir / run.dir`.
    pub run_dir: PathBuf,
    pub resume: Option<EvalResumeConfig>,
    pub log: LogConfig,
    pub monitors: EvalMonitorsConfig,
    /// The whole config as `model_dump(mode="json")`: the run config monitors upload.
    pub dump: Value,
    /// `model_dump(exclude_none=True, mode="json")`: the run config the prime monitor registers.
    pub prime_run_config: Value,
}

impl EvalConfig {
    /// Extract from a validated `prime_rl.configs.eval.EvalConfig`.
    pub fn extract(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        config.extract()
    }
}

impl<'py> FromPyObject<'py> for EvalConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let attr = |name: &str| obj.getattr(name);
        Ok(Self {
            served: obj.extract()?,
            model: attr("model")?.extract()?,
            run: attr("run")?.extract()?,
            output_dir: attr("output_dir")?.extract()?,
            run_dir: attr("run_dir")?.extract()?,
            resume: attr("resume")?.extract()?,
            log: attr("log")?.extract()?,
            monitors: attr("monitors")?.extract()?,
            dump: model_dump(obj)?,
            prime_run_config: model_dump_exclude_none(obj)?,
        })
    }
}

/// `SFTOnlineEvalConfig`: the online-eval process the sft launcher spawns. It evaluates
/// interval-driven sources on each trainer weight broadcast.
#[derive(Debug, Clone, PartialEq)]
pub struct SFTOnlineEvalConfig {
    pub served: ServedEvalConfig<OnlineEvalSourceConfig>,
    /// `ScheduledEvalConfig.intervals`: step interval per eval env, by resolved name.
    pub intervals: IndexMap<String, u64>,
    pub skip_first_step: bool,
    pub retrigger_on_resume: bool,
    /// Cancel unfinished episodes when a newer trainer checkpoint is ready.
    pub cancel_on_new_checkpoint: bool,
    /// Name the inference server serves the model under.
    pub model: Option<String>,
    /// None reloads weights from the filesystem broadcasts.
    pub weight_broadcast: Option<WeightBroadcastConfig>,
    /// `step_{n}` broadcasts directory (defaulted to `<output_dir>/broadcasts`).
    pub broadcasts_dir: Option<PathBuf>,
    pub max_steps: Option<u64>,
    /// Trainer step the run resumed from.
    pub resume_step: Option<u64>,
    pub output_dir: PathBuf,
    pub log: LogConfig,
    pub monitors: EvalMonitorsConfig,
    /// The whole config as `model_dump(mode="json")`.
    pub dump: Value,
    /// `model_dump(exclude_none=True, mode="json")`: the run config the prime monitor registers.
    pub prime_run_config: Value,
}

impl SFTOnlineEvalConfig {
    /// Extract from a validated `prime_rl.configs.eval.SFTOnlineEvalConfig`.
    pub fn extract(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        config.extract()
    }
}

impl<'py> FromPyObject<'py> for SFTOnlineEvalConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let attr = |name: &str| obj.getattr(name);
        Ok(Self {
            served: obj.extract()?,
            intervals: attr("intervals")?.extract()?,
            skip_first_step: attr("skip_first_step")?.extract()?,
            retrigger_on_resume: attr("retrigger_on_resume")?.extract()?,
            cancel_on_new_checkpoint: attr("cancel_on_new_checkpoint")?.extract()?,
            model: attr("model")?.extract()?,
            weight_broadcast: attr("weight_broadcast")?.extract()?,
            broadcasts_dir: attr("broadcasts_dir")?.extract()?,
            max_steps: attr("max_steps")?.extract()?,
            resume_step: attr("resume_step")?.extract()?,
            output_dir: attr("output_dir")?.extract()?,
            log: attr("log")?.extract()?,
            monitors: attr("monitors")?.extract()?,
            dump: model_dump(obj)?,
            prime_run_config: model_dump_exclude_none(obj)?,
        })
    }
}
