//! `prime_rl.configs.orchestrator`: the top-level config, sources and curriculum.

use std::path::PathBuf;

use indexmap::IndexMap;
use pyo3::prelude::*;
use serde_json::Value;

use crate::algorithm::AlgoConfig;
use crate::monitors::TrainMonitorsConfig;
use crate::py::{model_dump, model_dump_exclude_none, str_enum, tag, to_json, unknown_tag};
use crate::shared::{
    CheckpointConfig, ControlConfig, GCConfig, HeartbeatConfig, LogConfig, ModelConfig,
    ResumeConfig, TransportConfig, WeightBroadcastConfig,
};

str_enum!(
    /// A source's split: the first element of `env_addresses` keys.
    Split { Train = "train", Eval = "eval" }
);

str_enum!(
    /// `inference_metrics_roles` item: the P/D role of a policy admin client.
    InferenceRole { Prefill = "prefill", Decode = "decode" }
);

/// `TrainSamplingConfig`: sampling for train rollouts (group defaults already merged in).
#[derive(Debug, Clone, PartialEq)]
pub struct TrainSamplingConfig {
    pub temperature: f64,
    pub top_p: f64,
    /// Set by the config validator when truncation needs a bounded sampling mask.
    pub top_k: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    /// Top sampler probabilities to record per token.
    pub logprobs: Option<u64>,
    /// Extra request body; policy sources carry `extra_args.prl_replay` here.
    pub extra_body: Value,
    /// `TrainSamplingConfig.truncates_distribution()`.
    pub truncates_distribution: bool,
}

impl<'py> FromPyObject<'py> for TrainSamplingConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            temperature: obj.getattr("temperature")?.extract()?,
            top_p: obj.getattr("top_p")?.extract()?,
            top_k: obj.getattr("top_k")?.extract()?,
            max_completion_tokens: obj.getattr("max_completion_tokens")?.extract()?,
            logprobs: obj.getattr("logprobs")?.extract()?,
            extra_body: to_json(&obj.getattr("extra_body")?)?,
            truncates_distribution: obj.call_method0("truncates_distribution")?.extract()?,
        })
    }
}

/// `DifficultyPoolConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct DifficultyPoolConfig {
    /// Inclusive maximum reward assigned to this pool.
    pub threshold: f64,
    /// Relative per-task sampling weight.
    pub weight: f64,
}

/// `DifficultyPoolSamplerConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct DifficultyPoolSamplerConfig {
    /// Named pools, in config order.
    pub pools: IndexMap<String, DifficultyPoolConfig>,
    pub seed: i64,
}

/// `TaskSamplerConfig`, discriminated on `type`.
#[derive(Debug, Clone, PartialEq)]
pub enum TaskSamplerConfig {
    Standard,
    DifficultyPool(DifficultyPoolSamplerConfig),
}

impl<'py> FromPyObject<'py> for TaskSamplerConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        match tag(obj, "type")?.as_str() {
            "standard" => Ok(Self::Standard),
            "difficulty_pool" => obj.extract().map(Self::DifficultyPool),
            other => Err(unknown_tag("TaskSamplerConfig", other)),
        }
    }
}

/// `AdvRangeGateConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct AdvRangeGateConfig {
    pub reject_min: f64,
    pub reject_max: f64,
}

/// `AdmissionGateConfig`, discriminated on `type`.
#[derive(Debug, Clone, PartialEq)]
pub enum AdmissionGateConfig {
    AdvantageRange(AdvRangeGateConfig),
}

impl<'py> FromPyObject<'py> for AdmissionGateConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        match tag(obj, "type")?.as_str() {
            "advantage_range" => obj.extract().map(Self::AdvantageRange),
            other => Err(unknown_tag("AdmissionGateConfig", other)),
        }
    }
}

/// `CurriculumConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct CurriculumConfig {
    pub sampler: TaskSamplerConfig,
    /// Named admission gates, in config order.
    pub gates: IndexMap<String, AdmissionGateConfig>,
}

/// `TrainSourceConfig`: one training environment.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct TrainSourceConfig {
    /// `EnvConfig.resolved_name`: display name in logs, metrics and buffer keys.
    pub resolved_name: String,
    /// `EnvConfig.env_id`: the taskset id, prefixed by the paired env id.
    pub env_id: String,
    pub sampling: TrainSamplingConfig,
    /// Relative sampling weight of this env.
    pub ratio: f64,
    pub group_size: u64,
    pub algo: AlgoConfig,
    pub curriculum: Option<CurriculumConfig>,
}

/// `TrainConfig` (`[orchestrator.train]`). The group defaults are already merged into
/// each source by pydantic, so only the sources are read.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct TrainConfig {
    pub source: Vec<TrainSourceConfig>,
}

/// `OnlineEvalSourceConfig`: one eval environment of a training run.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct OnlineEvalSourceConfig {
    /// `EnvConfig.resolved_name`.
    pub resolved_name: String,
    /// `EnvConfig.env_id`.
    pub env_id: String,
    /// `verifiers.v1.SamplingConfig` dump, sent to the env server as is.
    #[pyo3(from_py_with = model_dump)]
    pub sampling: Value,
    pub group_size: u64,
    pub interval: u64,
}

/// `RLOnlineEvalConfig` (`[orchestrator.eval]`).
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct RLOnlineEvalConfig {
    pub source: Vec<OnlineEvalSourceConfig>,
    pub skip_first_step: bool,
    pub retrigger_on_resume: bool,
    /// `ScheduledEvalConfig.intervals`: step interval per eval env, by resolved name.
    pub intervals: IndexMap<String, u64>,
}

/// `ConcurrencyConfig`: bounds and seed of the adaptive in-flight episode cap.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct ConcurrencyConfig {
    pub initial_inflight: Option<u64>,
    pub min_inflight: u64,
    /// None removes the ceiling.
    pub max_inflight: Option<u64>,
}

/// `OrchestratorConfig`, extracted once at startup from the validated pydantic object.
#[derive(Debug, Clone, PartialEq)]
pub struct OrchestratorConfig {
    pub model: ModelConfig,
    pub train: TrainConfig,
    /// `renderers.RendererConfig` dump (`name` plus the renderer's options), forwarded to
    /// the env server with the policy's renderer client.
    pub renderer: Value,
    pub eval: Option<RLOnlineEvalConfig>,
    pub log: LogConfig,
    pub monitors: TrainMonitorsConfig,
    pub collect_inference_metrics: bool,
    pub inference_metrics_roles: Option<Vec<InferenceRole>>,
    pub ckpt: Option<CheckpointConfig>,
    pub resume: Option<ResumeConfig>,
    pub weight_broadcast: WeightBroadcastConfig,
    pub rollout_transport: TransportConfig,
    /// Replicas of each launcher-managed env server; None means one, at `<name>.address`.
    pub env_server_replicas: Option<u64>,
    pub output_dir: PathBuf,
    pub dispatch_per_minute: Option<u64>,
    pub batch_size: u64,
    pub constant_trainer_batch_size: bool,
    pub concurrency: ConcurrencyConfig,
    pub gc: Option<GCConfig>,
    pub seq_len: u64,
    pub num_train_workers: u64,
    pub pad_to_multiple_of: u64,
    pub pad_to_seq_len: bool,
    pub max_steps: Option<u64>,
    pub max_staleness: u64,
    pub heartbeat: Option<HeartbeatConfig>,
    pub control: Option<ControlConfig>,
    /// `OrchestratorConfig.env_addresses`: `(split, resolved_name)` -> the source's own
    /// `serve.address`, or None when the launcher spawns the server (which publishes its
    /// address file). Train sources first, then eval.
    pub env_addresses: IndexMap<(Split, String), Option<String>>,
    /// The whole config as `model_dump(mode="json")`: the run config monitors upload.
    pub dump: Value,
    /// The whole config as `model_dump(exclude_none=True, mode="json")`: the run config the
    /// prime monitor registers.
    pub prime_run_config: Value,
    /// `str()` of sub-configs the orchestrator prints in its setup log lines.
    pub reprs: ConfigReprs,
}

/// Python `str()` (pydantic field repr) of sub-configs, for log lines like
/// `Initializing weight broadcast (timeout=3600 type='filesystem')`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConfigReprs {
    pub model: String,
    pub rollout_transport: String,
    pub weight_broadcast: String,
}

impl OrchestratorConfig {
    /// Extract from a validated `prime_rl.configs.orchestrator.OrchestratorConfig`.
    pub fn extract(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        config.extract()
    }
}

impl<'py> FromPyObject<'py> for OrchestratorConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let attr = |name: &str| obj.getattr(name);
        Ok(Self {
            model: attr("model")?.extract()?,
            train: attr("train")?.extract()?,
            renderer: model_dump(&attr("renderer")?)?,
            eval: attr("eval")?.extract()?,
            log: attr("log")?.extract()?,
            monitors: attr("monitors")?.extract()?,
            collect_inference_metrics: attr("collect_inference_metrics")?.extract()?,
            inference_metrics_roles: attr("inference_metrics_roles")?.extract()?,
            ckpt: attr("ckpt")?.extract()?,
            resume: attr("resume")?.extract()?,
            weight_broadcast: attr("weight_broadcast")?.extract()?,
            rollout_transport: attr("rollout_transport")?.extract()?,
            env_server_replicas: attr("env_server_replicas")?.extract()?,
            output_dir: attr("output_dir")?.extract()?,
            dispatch_per_minute: attr("dispatch_per_minute")?.extract()?,
            batch_size: attr("batch_size")?.extract()?,
            constant_trainer_batch_size: attr("constant_trainer_batch_size")?.extract()?,
            concurrency: attr("concurrency")?.extract()?,
            gc: attr("gc")?.extract()?,
            seq_len: attr("seq_len")?.extract()?,
            num_train_workers: attr("num_train_workers")?.extract()?,
            pad_to_multiple_of: attr("pad_to_multiple_of")?.extract()?,
            pad_to_seq_len: attr("pad_to_seq_len")?.extract()?,
            max_steps: attr("max_steps")?.extract()?,
            max_staleness: attr("max_staleness")?.extract()?,
            heartbeat: attr("heartbeat")?.extract()?,
            control: attr("control")?.extract()?,
            env_addresses: attr("env_addresses")?.extract()?,
            dump: model_dump(obj)?,
            prime_run_config: model_dump_exclude_none(obj)?,
            reprs: ConfigReprs {
                model: attr("model")?.str()?.to_string(),
                rollout_transport: attr("rollout_transport")?.str()?.to_string(),
                weight_broadcast: attr("weight_broadcast")?.str()?.to_string(),
            },
        })
    }
}
