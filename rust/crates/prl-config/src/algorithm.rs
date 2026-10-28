//! `prime_rl.configs.algorithm`: the training algorithm of a train source.

use std::ops::Deref;

use pyo3::prelude::*;
use pyo3::types::PyString;
use serde_json::Value;

use crate::py::{model_dump, str_enum, tag, to_json, unknown_tag};
use crate::shared::ClientConfig;

/// `FrozenModelConfig`: an externally hosted model (never launched or updated by prime-rl).
#[derive(Debug, Clone, PartialEq)]
pub struct FrozenModelConfig {
    pub client: ClientConfig,
    /// Served model name, sent as `model` on every request.
    pub name: String,
}

impl Deref for FrozenModelConfig {
    type Target = ClientConfig;

    fn deref(&self) -> &ClientConfig {
        &self.client
    }
}

impl<'py> FromPyObject<'py> for FrozenModelConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            client: obj.extract()?,
            name: obj.getattr("name")?.extract()?,
        })
    }
}

/// `ModelReference`: `"policy"` (the live policy) or an inline frozen model.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelReference {
    Policy,
    Frozen(Box<FrozenModelConfig>),
}

impl<'py> FromPyObject<'py> for ModelReference {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<PyString>() {
            let value: String = obj.extract()?;
            return match value.as_str() {
                "policy" => Ok(Self::Policy),
                other => Err(unknown_tag("ModelReference", other, &["policy"])),
            };
        }
        Ok(Self::Frozen(Box::new(obj.extract()?)))
    }
}

/// `algorithm.SamplingConfig`: which model generates train rollouts.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct AlgoSamplingConfig {
    pub source: ModelReference,
}

str_enum!(
    /// `ActionLossType`: the loss component that consumes the action tokens.
    ActionLossType { Rl = "rl", Ce = "ce", RefKl = "ref_kl" }
);

str_enum!(
    /// `GRPOAlgoConfig.loss_aggregation`.
    LossAggregation { Token = "token", Prompt = "prompt" }
);

/// `LinearLengthPenaltyConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct LinearLengthPenaltyConfig {
    pub num_output_tokens_weight: f64,
    pub num_input_tokens_weight: f64,
    pub num_turns_weight: f64,
}

/// `LinearSubagentLengthPenaltyConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct LinearSubagentLengthPenaltyConfig {
    pub num_output_tokens_weight: f64,
    pub num_input_tokens_weight: f64,
    pub num_turns_weight: f64,
    /// Multiplier on subagent input and output tokens before group normalization.
    pub subagent_token_weight: f64,
}

/// `LengthPenaltyConfig`, discriminated on `type`.
#[derive(Debug, Clone, PartialEq)]
pub enum LengthPenaltyConfig {
    Linear(LinearLengthPenaltyConfig),
    LinearSubagent(LinearSubagentLengthPenaltyConfig),
}

impl<'py> FromPyObject<'py> for LengthPenaltyConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        match tag(obj, "type")?.as_str() {
            "linear" => obj.extract().map(Self::Linear),
            "linear-subagent" => obj.extract().map(Self::LinearSubagent),
            other => Err(unknown_tag(
                "LengthPenaltyConfig",
                other,
                &["linear", "linear-subagent"],
            )),
        }
    }
}

/// `GRPOAlgoConfig` parameters.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct GRPOAlgoConfig {
    pub length_penalty: Option<LengthPenaltyConfig>,
    pub length_weighted_baseline: bool,
    pub loss_aggregation: LossAggregation,
}

/// `EchoRoleConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct EchoRoleConfig {
    /// Per-token ce weight for this role's env-provided tokens.
    pub alpha: f64,
}

/// `EchoRolesConfig`: which env-provided message roles train (None = disabled).
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct EchoRolesConfig {
    pub system: Option<EchoRoleConfig>,
    pub user: Option<EchoRoleConfig>,
    pub assistant: Option<EchoRoleConfig>,
    pub tool: Option<EchoRoleConfig>,
}

/// `EchoFilterConfig`: a user filter `filter_fn(rollout, **kwargs) -> list[list[bool]]`.
/// It is a Python callable, so it runs in Python.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct EchoFilterConfig {
    pub import_path: String,
    #[pyo3(from_py_with = to_json)]
    pub kwargs: Value,
}

/// `EchoAlgoConfig` parameters (GRPO's plus the echo role table).
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct EchoAlgoConfig {
    pub length_penalty: Option<LengthPenaltyConfig>,
    pub length_weighted_baseline: bool,
    pub loss_aggregation: LossAggregation,
    pub roles: EchoRolesConfig,
    pub filter: Option<EchoFilterConfig>,
}

/// `RAEAlgoConfig` parameters.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct RAEAlgoConfig {
    /// EMA decay of the per-agent baselines.
    pub decay: f64,
}

/// `HierarchicalGRPOAlgoConfig` parameters.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct HierarchicalGRPOAlgoConfig {
    /// Roles compared within one proposed problem.
    pub episode_agents: Vec<String>,
}

/// `OPDAlgoConfig` parameters.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct OPDAlgoConfig {
    pub teacher: FrozenModelConfig,
}

/// `OPSDAlgoConfig` parameters.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct OPSDAlgoConfig {
    pub demo_key: String,
    /// System message content; receives `{demonstration}`.
    pub template: String,
    /// `renderers.RendererConfig` dump, forwarded to the renderer.
    #[pyo3(from_py_with = model_dump)]
    pub renderer: Value,
}

/// `DebugAlgoConfig` parameters.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct DebugAlgoConfig {
    /// Constant per-token advantage (nonzero).
    pub advantage: f64,
}

/// The algorithm-specific part of `AlgoConfig`, by its `type`.
#[derive(Debug, Clone, PartialEq)]
pub enum AlgoParams {
    Grpo(GRPOAlgoConfig),
    Echo(EchoAlgoConfig),
    MaxRl,
    Rae(RAEAlgoConfig),
    HierarchicalGrpo(HierarchicalGRPOAlgoConfig),
    Opd(OPDAlgoConfig),
    Opsd(OPSDAlgoConfig),
    Sft,
    Debug(DebugAlgoConfig),
}

impl AlgoParams {
    pub const TAGS: &'static [&'static str] = &[
        "grpo",
        "echo",
        "max_rl",
        "rae",
        "hierarchical_grpo",
        "opd",
        "opsd",
        "sft",
        "debug",
    ];

    /// The algorithm's `type`.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Grpo(_) => "grpo",
            Self::Echo(_) => "echo",
            Self::MaxRl => "max_rl",
            Self::Rae(_) => "rae",
            Self::HierarchicalGrpo(_) => "hierarchical_grpo",
            Self::Opd(_) => "opd",
            Self::Opsd(_) => "opsd",
            Self::Sft => "sft",
            Self::Debug(_) => "debug",
        }
    }
}

impl<'py> FromPyObject<'py> for AlgoParams {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        match tag(obj, "type")?.as_str() {
            "grpo" => obj.extract().map(Self::Grpo),
            "echo" => obj.extract().map(Self::Echo),
            "max_rl" => Ok(Self::MaxRl),
            "rae" => obj.extract().map(Self::Rae),
            "hierarchical_grpo" => obj.extract().map(Self::HierarchicalGrpo),
            "opd" => obj.extract().map(Self::Opd),
            "opsd" => obj.extract().map(Self::Opsd),
            "sft" => Ok(Self::Sft),
            "debug" => obj.extract().map(Self::Debug),
            other => Err(unknown_tag("AlgoConfig", other, Self::TAGS)),
        }
    }
}

/// `AlgoConfig`: a source's training algorithm. `sampling` and `action_loss_type` are on
/// every algorithm; `params` holds the rest, by `type`.
#[derive(Debug, Clone, PartialEq)]
pub struct AlgoConfig {
    pub sampling: AlgoSamplingConfig,
    /// The algorithm class's `action_loss_type` (a class variable, read off the object).
    pub action_loss_type: ActionLossType,
    pub params: AlgoParams,
}

impl<'py> FromPyObject<'py> for AlgoConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            sampling: obj.getattr("sampling")?.extract()?,
            action_loss_type: obj.getattr("action_loss_type")?.extract()?,
            params: obj.extract()?,
        })
    }
}
