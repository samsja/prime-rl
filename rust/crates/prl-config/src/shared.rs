//! `prime_rl.configs.shared` (and the verifiers client config it builds on).

use std::ops::Deref;
use std::path::PathBuf;

use indexmap::IndexMap;
use pyo3::prelude::*;

use crate::py::{tag, unknown_tag};

/// `verifiers.v1.configs.client.ClientTimeoutConfig`: per-phase HTTP timeouts in seconds
/// (None waits indefinitely).
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct ClientTimeoutConfig {
    pub connect: Option<f64>,
    pub read: Option<f64>,
    pub write: Option<f64>,
    pub pool: Option<f64>,
}

/// `ClientConfig` (`verifiers` `BaseClientConfig` plus prime-rl's readiness/admin
/// fields): an OpenAI-compatible endpoint. Extracted from any of its subclasses
/// (`PolicyClientConfig`, `FrozenModelConfig`).
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct ClientConfig {
    pub base_url: String,
    /// Env var holding the API key.
    pub api_key_var: String,
    pub headers: IndexMap<String, String>,
    /// Header name -> env var, resolved when building a client.
    pub headers_from_env: IndexMap<String, String>,
    pub timeout: ClientTimeoutConfig,
    /// Seconds to wait at startup for the inference pool to become ready.
    pub wait_for_ready_timeout: u64,
    pub skip_model_check: bool,
    /// Per-engine admin URLs that bypass the router; None uses `base_url`.
    pub admin_base_url: Option<Vec<String>>,
}

/// `DynamoConfig`: Dynamo worker discovery for the inference admin plane.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct DynamoConfig {
    pub enabled: bool,
    pub discovery_url: Option<String>,
}

/// `PolicyClientConfig`: the client of the policy deployment (`[orchestrator.model.client]`).
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyClientConfig {
    pub client: ClientConfig,
    pub dynamo: Option<DynamoConfig>,
}

impl Deref for PolicyClientConfig {
    type Target = ClientConfig;

    fn deref(&self) -> &ClientConfig {
        &self.client
    }
}

impl<'py> FromPyObject<'py> for PolicyClientConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            client: obj.extract()?,
            dynamo: obj.getattr("dynamo")?.extract()?,
        })
    }
}

/// `ModelConfig` (`[orchestrator.model]`): the trained model and its live deployment.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct ModelConfig {
    pub name: String,
    pub client: PolicyClientConfig,
}

/// `LogConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct LogConfig {
    pub level: String,
    pub json_logging: bool,
    /// Seconds between periodic logs.
    pub interval: f64,
}

/// `GCConfig`: run a full garbage collection every `interval` steps.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct GCConfig {
    pub interval: u64,
}

/// `HeartbeatConfig`: BetterStack heartbeat.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct HeartbeatConfig {
    pub url: String,
    /// Minimum seconds between pings; surplus beats are dropped.
    pub min_interval: f64,
}

/// `OrchestratorControlConfig`: where the control API listens.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct ControlConfig {
    pub host: String,
    pub port: u16,
}

/// `CheckpointConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct CheckpointConfig {
    /// Step interval at which to save the orchestrator checkpoint.
    pub interval: Option<u64>,
}

/// `ResumeConfig`: `step` and `dir` are mutually exclusive; both None resumes from the
/// latest checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeConfig {
    pub step: Option<u64>,
    /// External checkpoint step directory (`.../step_<N>`).
    pub dir: Option<PathBuf>,
    /// `ResumeConfig.dir_step`: the step in `dir`'s name; set exactly when `dir` is.
    pub dir_step: Option<u64>,
}

impl<'py> FromPyObject<'py> for ResumeConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dir: Option<PathBuf> = obj.getattr("dir")?.extract()?;
        let dir_step = match dir {
            Some(_) => Some(obj.getattr("dir_step")?.extract()?),
            None => None,
        };
        Ok(Self {
            step: obj.getattr("step")?.extract()?,
            dir,
            dir_step,
        })
    }
}

/// `FileSystemWeightBroadcastConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct FileSystemWeightBroadcastConfig {
    /// Seconds for the broadcast handshake and transfer.
    pub timeout: u64,
}

/// `NCCLWeightBroadcastConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct NCCLWeightBroadcastConfig {
    pub timeout: u64,
    pub host: String,
    pub port: u16,
    pub inference_world_size: u64,
}

/// `NIXLWeightBroadcastConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct NIXLWeightBroadcastConfig {
    pub timeout: u64,
    pub host: String,
    pub port: u16,
    pub inference_world_size: u64,
    pub session_id: String,
    pub overlap_transfer_and_replay: bool,
}

/// `WeightBroadcastConfig`, discriminated on `type`.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightBroadcastConfig {
    Filesystem(FileSystemWeightBroadcastConfig),
    Nccl(NCCLWeightBroadcastConfig),
    Nixl(NIXLWeightBroadcastConfig),
}

impl WeightBroadcastConfig {
    /// The `type` tag.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Filesystem(_) => "filesystem",
            Self::Nccl(_) => "nccl",
            Self::Nixl(_) => "nixl",
        }
    }

    /// The `timeout` every variant has.
    pub fn timeout(&self) -> u64 {
        match self {
            Self::Filesystem(c) => c.timeout,
            Self::Nccl(c) => c.timeout,
            Self::Nixl(c) => c.timeout,
        }
    }
}

impl<'py> FromPyObject<'py> for WeightBroadcastConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        match tag(obj, "type")?.as_str() {
            "filesystem" => obj.extract().map(Self::Filesystem),
            "nccl" => obj.extract().map(Self::Nccl),
            "nixl" => obj.extract().map(Self::Nixl),
            other => Err(unknown_tag("WeightBroadcastConfig", other)),
        }
    }
}

/// `ZMQTransportConfig`.
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub struct ZMQTransportConfig {
    pub host: String,
    pub port: u16,
}

/// `TransportConfig` (rollouts to the trainer), discriminated on `type`.
#[derive(Debug, Clone, PartialEq)]
pub enum TransportConfig {
    Filesystem,
    Zmq(ZMQTransportConfig),
}

impl<'py> FromPyObject<'py> for TransportConfig {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        match tag(obj, "type")?.as_str() {
            "filesystem" => Ok(Self::Filesystem),
            "zmq" => obj.extract().map(Self::Zmq),
            other => Err(unknown_tag("TransportConfig", other)),
        }
    }
}
