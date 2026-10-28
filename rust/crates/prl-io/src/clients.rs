//! Inference clients and the admin plane of the policy deployment (`orchestrator/clients.py`).

use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use indexmap::IndexMap;
use prl_config::{ClientConfig, FrozenModelConfig, InferenceRole, PolicyClientConfig};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::dynamo::DynamoAdminPlane;
use crate::http::{
    admin_post, admin_post_once, header_map, post_all, resolve_api_key, resolve_headers, router_admin_client,
    setup_admin_clients, AdminClient, AdminRequest, RetryPolicy, UPDATE_WEIGHTS_TIMEOUT_S,
};

/// Weight transport of an update (`update_weights(transport=...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    Filesystem,
    Nccl,
    Nixl,
}

/// Per-engine times of the last update_weights phases (`AdminPlane.update_times`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UpdateTimes {
    pub pause: Vec<f64>,
    pub transfer: Vec<f64>,
    pub resume: Vec<f64>,
}

impl UpdateTimes {
    /// `(phase, times)` in Python's dict order.
    pub fn phases(&self) -> [(&'static str, &[f64]); 3] {
        [("pause", &self.pause), ("transfer", &self.transfer), ("resume", &self.resume)]
    }
}

// ── v1 client configs sent to env servers ─────────────────────────────────────

/// Generation client type (`train_client_type` / `eval_client_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientType {
    /// Token-in/out through the renderer (`TrainClientConfig`).
    Renderer,
    /// Plain chat completions (`EvalClientConfig`).
    ChatCompletions,
}

/// `setup_client`: the v1 client config (`EvalClientConfig` / `TrainClientConfig`
/// `model_dump(mode="json")`) env servers use to reach this endpoint.
fn setup_client(config: &ClientConfig, client_type: ClientType, renderer: Option<&Value>, renderer_model_name: Option<&str>) -> Value {
    let t = &config.timeout;
    let mut client = json!({
        "base_url": config.base_url,
        "api_key_var": config.api_key_var,
        "headers": config.headers,
        "headers_from_env": config.headers_from_env,
        "timeout": {"connect": t.connect, "read": t.read, "write": t.write, "pool": t.pool},
        "type": "eval",
    });
    if client_type == ClientType::Renderer {
        client["type"] = "train".into();
        client["renderer"] = renderer.cloned().unwrap_or(Value::Null);
        client["renderer_model_name"] = renderer_model_name.into();
        client["multiplex"] = 256.into();
    }
    client
}

// ── data plane ────────────────────────────────────────────────────────────────

/// Prefill-scores token ids against an endpoint (`prefill_logprobs` through the OpenAI SDK:
/// bearer auth, resolved headers, no retries, the client config's timeouts).
#[derive(Debug, Clone)]
struct PrefillScorer {
    http: reqwest::Client,
    url: String,
}

impl PrefillScorer {
    fn new(config: &ClientConfig) -> Result<Self> {
        let mut headers = resolve_headers(&config.headers, &config.headers_from_env, &config.api_key_var, &config.base_url);
        headers.insert("Authorization".into(), format!("Bearer {}", resolve_api_key(&config.api_key_var, &config.base_url)));
        let mut builder = reqwest::Client::builder().default_headers(header_map(&headers)?);
        if let Some(connect) = config.timeout.connect {
            builder = builder.connect_timeout(Duration::from_secs_f64(connect));
        }
        if let Some(read) = config.timeout.read {
            builder = builder.read_timeout(Duration::from_secs_f64(read));
        }
        // `/inference/v1/generate` is mounted at the server root, not under `/v1`.
        let base = config.base_url.trim_end_matches('/');
        let base = base.strip_suffix("/v1").unwrap_or(base);
        Ok(Self { http: builder.build()?, url: format!("{base}/inference/v1/generate") })
    }

    async fn score(&self, model: &str, token_ids: &[i32]) -> Result<Vec<f64>> {
        let body = json!({
            "model": model,
            "token_ids": token_ids,
            "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "prompt_logprobs": 1},
        });
        let response = self.http.post(&self.url).json(&body).send().await?.error_for_status()?;
        parse_prompt_logprobs(&response.bytes().await?)
    }
}

#[derive(Deserialize)]
struct GenerateTokensResponse {
    #[serde(default)]
    prompt_logprobs: Option<Vec<Option<IndexMap<String, Logprob>>>>,
}

#[derive(Deserialize)]
struct Logprob {
    #[serde(default)]
    logprob: Option<f64>,
}

/// One float per `prompt_logprobs` entry: the first listed token's logprob, 0.0 for the
/// leading (context-free) token and missing values.
fn parse_prompt_logprobs(body: &[u8]) -> Result<Vec<f64>> {
    let response: GenerateTokensResponse = serde_json::from_slice(body).context("invalid /inference/v1/generate response")?;
    Ok(response
        .prompt_logprobs
        .unwrap_or_default()
        .into_iter()
        .map(|entry| entry.and_then(|e| e.into_values().next()).and_then(|lp| lp.logprob).unwrap_or(0.0))
        .collect())
}

/// Data-plane clients for one inference endpoint (the router of the policy deployment, or an
/// external API for frozen models).
#[derive(Debug, Clone)]
pub struct InferenceClient {
    /// v1 client configs for env-server requests.
    pub train_client: Value,
    pub eval_client: Value,
    pub model_name: String,
    scorer: PrefillScorer,
    /// Router admin client; only managed routed deployments (`admin_base_url` set) have one.
    session_client: Option<AdminClient>,
}

impl InferenceClient {
    pub fn new(
        config: &ClientConfig,
        model_name: &str,
        train_client_type: ClientType,
        eval_client_type: ClientType,
        renderer: Option<&Value>,
    ) -> Result<Self> {
        let renderer_model_name = (train_client_type == ClientType::Renderer).then_some(model_name);
        Ok(Self {
            train_client: setup_client(config, train_client_type, renderer, renderer_model_name),
            eval_client: setup_client(config, eval_client_type, None, None),
            model_name: model_name.to_owned(),
            scorer: PrefillScorer::new(config)?,
            session_client: if config.admin_base_url.is_some() { Some(router_admin_client(config)?) } else { None },
        })
    }

    /// Prefill-score `token_ids` under this endpoint's model (one logprob per token, 0.0 for
    /// the leading token).
    pub async fn score(&self, token_ids: &[i32]) -> Result<Vec<f64>> {
        self.scorer.score(&self.model_name, token_ids).await
    }

    /// Release completed sessions when the client-facing router supports it. Failures are
    /// logged at debug and dropped.
    pub async fn finish_sessions(&self, session_ids: &[String]) {
        let Some(client) = &self.session_client else { return };
        futures::future::join_all(session_ids.iter().map(|id| async move {
            let request = AdminRequest::query(vec![("session_id", id.clone())]);
            if let Err(error) = admin_post(client, "/finish_session", &request, 5.0).await {
                tracing::debug!("Failed to release inference session {id}: {error}");
            }
        }))
        .await;
    }
}

#[async_trait::async_trait]
impl prl_types::Scorer for InferenceClient {
    async fn score(&self, token_ids: Vec<i32>) -> Result<Vec<f32>> {
        Ok(InferenceClient::score(self, &token_ids).await?.into_iter().map(|v| v as f32).collect())
    }
}

/// Connect to an externally hosted frozen model and wait for it (`connect_frozen_client`).
pub async fn connect_frozen_client(config: &FrozenModelConfig, renderer: Option<&Value>) -> Result<InferenceClient> {
    tracing::info!("Initializing frozen model pool (model={}, base_url={})", config.name, config.base_url);
    let train_type = if renderer.is_some() { ClientType::Renderer } else { ClientType::ChatCompletions };
    let client = InferenceClient::new(config, &config.name, train_type, ClientType::ChatCompletions, renderer)?;
    check_inference_ready(config, &config.name).await?;
    Ok(client)
}

// ── readiness ─────────────────────────────────────────────────────────────────

/// Wait until every client's `/health` responds (`check_health`). With `quiet`, the periodic
/// waiting lines log at debug (engines fronted by a router).
pub(crate) async fn check_health(clients: &[AdminClient], interval: u64, log_interval: u64, timeout: u64, quiet: bool) -> Result<()> {
    futures::future::try_join_all(clients.iter().map(|client| check_health_one(client, interval, log_interval, timeout, quiet))).await?;
    Ok(())
}

async fn check_health_one(client: &AdminClient, interval: u64, log_interval: u64, timeout: u64, quiet: bool) -> Result<()> {
    let mut wait_time = 0u64;
    tracing::debug!("Pinging /health until the inference server is ready");
    while wait_time < timeout {
        let error = match client.get("/health").send().await {
            Ok(response) if response.status().as_u16() == 404 => {
                tracing::warn!("The route /health does not exist. Skipping health check.");
                return Ok(());
            }
            Ok(response) => match response.error_for_status() {
                Ok(_) => {
                    tracing::debug!("Inference pool is ready after {wait_time} seconds");
                    return Ok(());
                }
                Err(error) => error,
            },
            Err(error) => error,
        };
        if wait_time.is_multiple_of(log_interval) && wait_time > 0 {
            let message = format!("Waiting for inference server at {} to start up ({wait_time}s elapsed)", client.base_url());
            if quiet {
                tracing::debug!("{message}");
            } else {
                tracing::info!("{message}");
            }
            tracing::debug!("Inference server at {} not reachable: {error}", client.base_url());
        }
        tokio::time::sleep(Duration::from_secs(interval)).await;
        wait_time += interval;
    }
    let message = format!("Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting...");
    tracing::error!("{message}");
    bail!(message)
}

/// Check that `model_name` is served by every client (`maybe_check_has_model`).
pub(crate) async fn maybe_check_has_model(clients: &[AdminClient], model_name: &str, skip_model_check: bool) -> Result<()> {
    if skip_model_check {
        return Ok(());
    }
    tracing::debug!("Checking if model {model_name} is in the inference pool");
    let responses = futures::future::try_join_all(clients.iter().map(|client| async move {
        let response = client.get("/v1/models").send().await?;
        let status = response.status().as_u16();
        let is_json = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("application/json"));
        let text = response.text().await?;
        anyhow::Ok((status, is_json, text))
    }))
    .await?;
    for (client, (status, is_json, text)) in clients.iter().zip(responses) {
        let body: Value = if is_json { serde_json::from_str(&text)? } else { json!({}) };
        let Some(models) = body.get("data").filter(|_| status == 200) else {
            let head: String = text.chars().take(300).collect();
            bail!("Listing the models of {} failed with status {status}: {head}", client.base_url());
        };
        let served = models.as_array().into_iter().flatten().any(|model| {
            model.get("id").and_then(Value::as_str) == Some(model_name) || model.get("root").and_then(Value::as_str) == Some(model_name)
        });
        if !served {
            bail!("Model {model_name} was not found in the inference pool on {}", client.base_url());
        }
    }
    tracing::debug!("Model {model_name} was found in the inference pool");
    Ok(())
}

/// One-shot readiness check of an endpoint with transient clients (`check_inference_ready`).
async fn check_inference_ready(config: &ClientConfig, model_name: &str) -> Result<()> {
    StandardAdminPlane::new(config, None)?.wait_for_ready(model_name).await
}

// ── admin plane ───────────────────────────────────────────────────────────────

/// `AdminPlane`: one admin client per engine process. Client order is load-bearing: it must
/// match the GPU rank order of NCCL/NIXL initialization and the metrics role list.
#[derive(Debug, Clone)]
pub struct StandardAdminPlane {
    clients: Vec<AdminClient>,
    /// P/D role of each client; prefill engines pause before decode engines.
    roles: Option<Vec<InferenceRole>>,
    router_clients: Vec<AdminClient>,
    skip_model_check: bool,
    wait_for_ready_timeout: u64,
}

impl StandardAdminPlane {
    fn new(config: &ClientConfig, roles: Option<Vec<InferenceRole>>) -> Result<Self> {
        let clients = setup_admin_clients(config)?;
        // When admin URLs bypass a router, also health-check the client-facing endpoint.
        let router_clients = if config.admin_base_url.as_ref().is_some_and(|urls| !urls.is_empty()) {
            vec![router_admin_client(config)?]
        } else {
            Vec::new()
        };
        Ok(Self {
            clients,
            roles,
            router_clients,
            skip_model_check: config.skip_model_check,
            wait_for_ready_timeout: config.wait_for_ready_timeout,
        })
    }

    async fn wait_for_ready(&self, model_name: &str) -> Result<()> {
        let timeout = self.wait_for_ready_timeout;
        tokio::try_join!(
            check_health(&self.clients, 1, 30, timeout, !self.router_clients.is_empty()),
            check_health(&self.router_clients, 1, 30, timeout, false),
        )?;
        maybe_check_has_model(&self.clients, model_name, self.skip_model_check).await
    }

    async fn initialize_nccl(&self, host: &str, port: u16, timeout: u64, inference_world_size: u64) -> Result<()> {
        let n = self.clients.len() as u64;
        let gpus_per_server = inference_world_size / n;
        tracing::info!(
            "Initializing NCCL broadcast: {n} servers, inference_world_size={inference_world_size}, gpus_per_server={gpus_per_server}"
        );
        futures::future::try_join_all(self.clients.iter().enumerate().map(|(i, client)| async move {
            let body = json!({
                "host": host,
                "port": port,
                "rank_offset": i as u64 * gpus_per_server,
                "inference_world_size": inference_world_size,
                "timeout": timeout,
            });
            let response = client.post("/init_broadcaster").json(&body).send().await?;
            // Python only reports a missing route; other HTTP errors are not raised.
            if response.status().as_u16() == 404 {
                tracing::warn!("The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization.");
            }
            anyhow::Ok(())
        }))
        .await?;
        Ok(())
    }

    async fn update_weights(
        &self,
        weight_dir: Option<&Path>,
        step: u64,
        on_paused: impl FnOnce() -> Result<()>,
    ) -> Result<UpdateTimes> {
        update_engines(&self.clients, self.roles.as_deref(), weight_dir, step, on_paused).await
    }
}

/// `AdminPlane.update_weights`: pause (prefill first), `on_paused`, `/update_weights` on every
/// engine, and resume even when the transfer failed.
pub(crate) async fn update_engines(
    clients: &[AdminClient],
    roles: Option<&[InferenceRole]>,
    weight_dir: Option<&Path>,
    step: u64,
    on_paused: impl FnOnce() -> Result<()>,
) -> Result<UpdateTimes> {
    let pause = pause_engines(clients, step, roles).await?;
    let transfer = async {
        on_paused()?;
        let request = AdminRequest::json(json!({
            "weight_dir": weight_dir.map(|d| d.to_string_lossy().into_owned()),
            "weight_version": step,
        }));
        anyhow::Ok(post_all(clients, "/update_weights", &request, UPDATE_WEIGHTS_TIMEOUT_S).await?)
    }
    .await;
    let resume = resume_engines(clients).await;
    let transfer = transfer?;
    Ok(UpdateTimes { pause, transfer, resume: resume? })
}

/// `_pause_engines`: prefill engines first (a paused decode never pulls the KV blocks a
/// draining prefill still holds), then the rest; `POST /pause?mode=keep&clear_cache=false`.
async fn pause_engines(clients: &[AdminClient], step: u64, roles: Option<&[InferenceRole]>) -> Result<Vec<f64>> {
    tracing::debug!("Pausing inference engines to update weights to policy v{step}");
    if let Some(roles) = roles {
        anyhow::ensure!(roles.len() == clients.len(), "got {} roles for {} admin clients", roles.len(), clients.len());
    }
    let is_prefill = |i: usize| roles.is_some_and(|r| r[i] == InferenceRole::Prefill);
    let prefill: Vec<AdminClient> = (0..clients.len()).filter(|&i| is_prefill(i)).map(|i| clients[i].clone()).collect();
    let rest: Vec<AdminClient> = (0..clients.len()).filter(|&i| !is_prefill(i)).map(|i| clients[i].clone()).collect();
    let request = AdminRequest::query(vec![("mode", "keep".into()), ("clear_cache", "false".into())]);
    let mut times = Vec::with_capacity(clients.len());
    for group in [prefill, rest] {
        times.extend(post_all(&group, "/pause", &request, crate::http::ADMIN_TIMEOUT_S).await?);
    }
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    tracing::debug!("All inference engines paused in {min:.2}-{max:.2}s");
    Ok(times)
}

/// `_resume_engines`: idempotent, so transient failures are retried.
async fn resume_engines(clients: &[AdminClient]) -> Result<Vec<f64>> {
    let times = post_all(clients, "/resume", &AdminRequest::default(), crate::http::ADMIN_TIMEOUT_S).await?;
    tracing::debug!("All inference engines resumed");
    Ok(times)
}

/// The admin plane of the policy deployment (`setup_admin_plane`): Dynamo-discovered when
/// `client.dynamo` is enabled, else the configured engine URLs.
#[derive(Debug)]
pub enum AdminPlane {
    Standard(StandardAdminPlane),
    Dynamo(DynamoAdminPlane),
}

impl AdminPlane {
    pub fn standard(config: &ClientConfig, roles: Option<Vec<InferenceRole>>) -> Result<Self> {
        Ok(Self::Standard(StandardAdminPlane::new(config, roles)?))
    }

    pub fn setup(config: &PolicyClientConfig, model_name: &str, roles: Option<Vec<InferenceRole>>) -> Result<Self> {
        match &config.dynamo {
            Some(dynamo) if dynamo.enabled => Ok(Self::Dynamo(DynamoAdminPlane::new(config, model_name)?)),
            _ => Self::standard(config, roles),
        }
    }

    /// Engine admin clients (cheap clones), for the metrics collector, LoRA loads and NIXL init.
    pub fn clients(&self) -> &[AdminClient] {
        match self {
            Self::Standard(plane) => &plane.clients,
            Self::Dynamo(plane) => plane.clients(),
        }
    }

    pub async fn wait_for_ready(&mut self, model_name: &str) -> Result<()> {
        match self {
            Self::Standard(plane) => plane.wait_for_ready(model_name).await,
            Self::Dynamo(plane) => plane.wait_for_ready(model_name).await,
        }
    }

    pub async fn initialize_nccl(&mut self, host: &str, port: u16, timeout: u64, inference_world_size: u64) -> Result<()> {
        match self {
            Self::Standard(plane) => plane.initialize_nccl(host, port, timeout, inference_world_size).await,
            Self::Dynamo(plane) => plane.initialize_nccl(host, port, timeout, inference_world_size).await,
        }
    }

    pub async fn update_weights(
        &mut self,
        weight_dir: Option<&Path>,
        transport: Transport,
        step: u64,
        on_paused: impl FnOnce() -> Result<()>,
    ) -> Result<UpdateTimes> {
        match self {
            Self::Standard(plane) => plane.update_weights(weight_dir, step, on_paused).await,
            Self::Dynamo(plane) => plane.update_weights(weight_dir, transport, step, on_paused).await,
        }
    }
}

/// Per-attempt and total bounds for `/load_lora_adapter`.
const LORA_LOAD_READ_TIMEOUT_S: f64 = 30.0;
const LORA_LOAD_TOTAL_TIMEOUT_S: f64 = 120.0;

/// Load a LoRA adapter on every engine (`load_lora_adapter`), retrying 404/500 (NFS propagation)
/// and transport errors. The prefix cache is salted per weight version, so it is not reset.
pub(crate) async fn load_lora_adapter(clients: &[AdminClient], lora_name: &str, lora_path: &Path, step: u64) -> Result<()> {
    let policy = RetryPolicy { max_delay: Duration::from_secs_f64(LORA_LOAD_TOTAL_TIMEOUT_S), max_attempts: 10 };
    let path = lora_path.to_string_lossy().into_owned();
    futures::future::try_join_all(clients.iter().map(|client| {
        let request = AdminRequest {
            query: vec![("weight_version", step.to_string())],
            json: Some(json!({"lora_name": lora_name, "lora_path": path})),
        };
        async move {
            policy
                .run(crate::http::is_retryable_lora_error, || {
                    tracing::debug!("Sending request to load LoRA adapter {lora_name} from {}", lora_path.display());
                    admin_post_once(client, "/load_lora_adapter", &request, LORA_LOAD_READ_TIMEOUT_S)
                })
                .await
        }
    }))
    .await?;
    Ok(())
}

/// Configure every vLLM worker for NIXL + ModelExpress pulls (`init_nixl_broadcast`).
pub(crate) async fn init_nixl_broadcast(
    clients: &[AdminClient],
    host: &str,
    port: u16,
    timeout: u64,
    inference_world_size: u64,
    session_id: &str,
) -> Result<()> {
    if clients.is_empty() {
        return Err(anyhow!("no admin clients to initialize NIXL on"));
    }
    let workers_per_server = inference_world_size / clients.len() as u64;
    let timeout_s = crate::http::ADMIN_TIMEOUT_S.max(timeout as f64);
    futures::future::try_join_all(clients.iter().enumerate().map(|(i, client)| {
        let request = AdminRequest::json(json!({
            "host": host,
            "port": port,
            "rank_offset": i as u64 * workers_per_server,
            "inference_world_size": inference_world_size,
            "timeout": timeout,
            "session_id": session_id,
        }));
        async move { admin_post(client, "/init_broadcaster", &request, timeout_s).await }
    }))
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_client_matches_pydantic_dump() {
        let config = ClientConfig {
            base_url: "http://worker-a:8000/v1".into(),
            api_key_var: "VLLM_API_KEY".into(),
            headers: [("X-Test".to_string(), "test".to_string())].into_iter().collect(),
            headers_from_env: [("X-Token".to_string(), "PROVIDER_TOKEN".to_string())].into_iter().collect(),
            timeout: prl_config::ClientTimeoutConfig { connect: Some(30.0), read: Some(12.0), write: None, pool: None },
            wait_for_ready_timeout: 3600,
            skip_model_check: false,
            admin_base_url: None,
        };
        let renderer = json!({"name": "qwen3-vl", "thinking_retention": null, "add_vision_id": false, "image_cache_max": 256});
        // setup_client(config, "renderer", Qwen3VLRendererConfig(), "Qwen/Qwen3-VL-4B-Instruct").model_dump(mode="json")
        let expected = json!({
            "base_url": "http://worker-a:8000/v1", "api_key_var": "VLLM_API_KEY",
            "headers": {"X-Test": "test"}, "headers_from_env": {"X-Token": "PROVIDER_TOKEN"},
            "timeout": {"connect": 30.0, "read": 12.0, "write": null, "pool": null},
            "type": "train", "renderer": renderer, "renderer_model_name": "Qwen/Qwen3-VL-4B-Instruct", "multiplex": 256,
        });
        assert_eq!(setup_client(&config, ClientType::Renderer, Some(&renderer), Some("Qwen/Qwen3-VL-4B-Instruct")), expected);
        let chat = setup_client(&config, ClientType::ChatCompletions, None, None);
        assert_eq!((chat["type"].as_str(), chat.get("renderer")), (Some("eval"), None));
    }

    #[test]
    fn prompt_logprobs_take_the_first_token_and_zero_the_leading_one() {
        let body = br#"{"prompt_logprobs": [null, {"11": {"logprob": -0.7}}, {"12": {"logprob": -0.3}, "5": {"logprob": -9.0}}]}"#;
        assert_eq!(parse_prompt_logprobs(body).unwrap(), vec![0.0, -0.7, -0.3]);
    }
}
