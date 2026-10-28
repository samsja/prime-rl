//! Admin plane pinned to two identical snapshots of a trusted Dynamo discovery endpoint
//! (`inference/dynamo.py`).

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use indexmap::IndexMap;
use prl_config::PolicyClientConfig;
use serde_json::{json, Value};

use crate::clients::{check_health, maybe_check_has_model, update_engines, Transport, UpdateTimes};
use crate::http::{admin_post, header_map, router_admin_client, AdminClient, AdminRequest, ADMIN_TIMEOUT_S, UPDATE_WEIGHTS_TIMEOUT_S};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamoWorker {
    pub admin_base_url: String,
    pub instance_id: u64,
}

impl DynamoWorker {
    pub fn fingerprint(&self) -> (u64, String) {
        (self.instance_id, self.admin_base_url.clone())
    }
}

/// Why a discovery attempt failed: `Pending` and `Retryable` are retried by the readiness loop,
/// `Fatal` is raised.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    /// The endpoint is healthy but has not published a complete worker set.
    #[error("{0}")]
    Pending(String),
    /// 5xx, transport errors and timeouts.
    #[error("{0}")]
    Retryable(String),
    #[error(transparent)]
    Fatal(#[from] anyhow::Error),
}

fn invalid(message: impl Into<String>) -> DiscoveryError {
    DiscoveryError::Fatal(anyhow!(message.into()))
}

/// `parse_dynamo_worker`: the single bound worker serving `model_name`.
pub fn parse_dynamo_worker(payload: &Value, model_name: &str, expected_admin_host: Option<&str>) -> Result<DynamoWorker, DiscoveryError> {
    if payload.get("protocol_version") != Some(&json!(1)) {
        return Err(invalid("Dynamo snapshot must have protocol_version 1"));
    }
    let workers = payload
        .get("workers")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Dynamo snapshot must list workers"))?;
    let mut matching = Vec::new();
    for raw in workers {
        let raw = raw.as_object().ok_or_else(|| invalid("Dynamo workers must be objects"))?;
        if raw.get("model").and_then(Value::as_str) != Some(model_name) {
            continue;
        }
        if raw.get("error").is_some_and(|e| !e.is_null()) {
            return Err(DiscoveryError::Pending("Dynamo worker is not ready".into()));
        }
        let Some(admin) = raw.get("admin_base_url").filter(|v| !v.is_null()) else {
            return Err(DiscoveryError::Pending("Dynamo worker is missing admin_base_url".into()));
        };
        if raw.get("world_size").and_then(Value::as_u64) != Some(1) || raw.get("world_size").is_some_and(Value::is_boolean) {
            return Err(invalid("Dynamo RL currently supports exactly one inference rank"));
        }
        let admin_base_url = admin.as_str().ok_or_else(|| invalid("admin_base_url must be a string"))?.to_owned();
        let instance_id = raw
            .get("instance_id")
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid("instance_id must be a non-negative integer"))?;
        let url = url::Url::parse(&admin_base_url).map_err(|_| invalid("admin_base_url must be a valid URL"))?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none_or(str::is_empty)
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || url.path() != "/"
        {
            return Err(invalid("admin_base_url must be an http(s) origin"));
        }
        if let Some(expected) = expected_admin_host {
            let host = url.host_str().unwrap_or_default();
            if host != expected {
                return Err(invalid(format!(
                    "Dynamo worker admin host '{host}' does not match discovery host '{expected}'"
                )));
            }
        }
        matching.push(DynamoWorker { admin_base_url, instance_id });
    }
    match matching.len() {
        0 => Err(DiscoveryError::Pending(format!("Dynamo returned no bound workers for model '{model_name}'"))),
        1 => Ok(matching.remove(0)),
        _ => Err(invalid("Dynamo RL currently supports exactly one inference worker")),
    }
}

/// `resolve_dynamo_discovery_url`: explicit URL, else `base_url` with port + 1 and no path.
pub fn resolve_dynamo_discovery_url(config: &PolicyClientConfig) -> Result<String> {
    let dynamo = config.dynamo.as_ref().filter(|d| d.enabled).ok_or_else(|| anyhow!("Dynamo discovery is not enabled"))?;
    if let Some(url) = &dynamo.discovery_url {
        return Ok(url.clone());
    }
    let base = url::Url::parse(&config.base_url)
        .map_err(|_| anyhow!("Set dynamo.discovery_url when client.base_url is not a valid URL"))?;
    let port = base.port().filter(|&p| p != 65535).ok_or_else(|| anyhow!("Set dynamo.discovery_url when client.base_url has no incrementable port"))?;
    Ok(format!("{}://{}:{}", base.scheme(), base.host_str().unwrap_or_default(), port + 1))
}

/// Discovery headers: static + env headers, `Authorization: Bearer $key` when a key is set.
fn discovery_headers(config: &PolicyClientConfig) -> IndexMap<String, String> {
    let mut headers = config.headers.clone();
    for (name, var) in &config.headers_from_env {
        if let Ok(value) = std::env::var(var) {
            headers.insert(name.clone(), value);
        }
    }
    if let Ok(key) = std::env::var(&config.api_key_var) {
        if !key.is_empty() && key != "EMPTY" {
            headers.retain(|name, _| !name.eq_ignore_ascii_case("authorization"));
            headers.insert("Authorization".into(), format!("Bearer {key}"));
        }
    }
    headers
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclState {
    Uninitialized,
    Initializing,
    Ready,
    Terminal,
}

#[derive(Debug)]
pub struct DynamoAdminPlane {
    discovery_url: String,
    discovery_host: String,
    model_name: String,
    pub poll_interval: Duration,
    timeout: u64,
    skip_model_check: bool,
    headers: IndexMap<String, String>,
    frontend_clients: Vec<AdminClient>,
    clients: Vec<AdminClient>,
    fingerprint: Option<(u64, String)>,
    pub nccl_state: NcclState,
}

impl DynamoAdminPlane {
    pub fn new(config: &PolicyClientConfig, model_name: &str) -> Result<Self> {
        let discovery_url = resolve_dynamo_discovery_url(config)?;
        let parsed = url::Url::parse(&discovery_url).map_err(|_| anyhow!("dynamo.discovery_url must be a valid URL"))?;
        if !matches!(parsed.scheme(), "http" | "https")
            || parsed.host_str().is_none_or(str::is_empty)
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            bail!("dynamo.discovery_url must be an http(s) URL without credentials, query, or fragment");
        }
        Ok(Self {
            discovery_host: parsed.host_str().unwrap_or_default().to_owned(),
            discovery_url,
            model_name: model_name.to_owned(),
            poll_interval: Duration::from_secs(1),
            timeout: config.wait_for_ready_timeout,
            skip_model_check: config.skip_model_check,
            headers: discovery_headers(config),
            frontend_clients: vec![router_admin_client(config)?],
            clients: Vec::new(),
            fingerprint: None,
            nccl_state: NcclState::Uninitialized,
        })
    }

    pub fn discovery_url(&self) -> &str {
        &self.discovery_url
    }

    pub fn clients(&self) -> &[AdminClient] {
        &self.clients
    }

    fn worker_timeout(&self) -> Duration {
        Duration::from_secs_f64((self.timeout as f64).clamp(1.0, 30.0))
    }

    fn discovery_endpoint(&self) -> String {
        let url = url::Url::parse(&self.discovery_url).expect("validated discovery url");
        let path = url.path().trim_end_matches('/');
        let path = path.strip_suffix("/v1").unwrap_or(path);
        let mut url = url.clone();
        url.set_path(&format!("{path}/v1/rl/workers"));
        url.to_string()
    }

    /// One `GET /v1/rl/workers` snapshot.
    pub async fn discover(&self) -> Result<DynamoWorker, DiscoveryError> {
        let timeout = self.worker_timeout();
        let http = reqwest::Client::builder()
            .default_headers(header_map(&self.headers)?)
            .timeout(timeout)
            .no_proxy()
            .build()
            .map_err(anyhow::Error::from)?;
        let response = http.get(self.discovery_endpoint()).send().await.map_err(classify)?;
        let response = response.error_for_status().map_err(classify)?;
        let payload: Value = response.json().await.map_err(classify)?;
        parse_dynamo_worker(&payload, &self.model_name, Some(&self.discovery_host))
    }

    fn make_worker_client(&self, worker: &DynamoWorker) -> Result<AdminClient> {
        let builder = reqwest::Client::builder()
            .default_headers(header_map(&self.headers)?)
            .timeout(self.worker_timeout())
            .no_proxy();
        AdminClient::with_builder(&worker.admin_base_url, builder)
    }

    fn remaining(deadline: Instant) -> Result<Duration> {
        let now = Instant::now();
        if now >= deadline {
            bail!("Dynamo readiness deadline expired");
        }
        Ok(deadline - now)
    }

    pub async fn wait_for_ready(&mut self, model_name: &str) -> Result<()> {
        if model_name != self.model_name {
            bail!("Dynamo admin plane was configured for '{}', not '{model_name}'", self.model_name);
        }
        let deadline = Instant::now() + Duration::from_secs(self.timeout);
        let remaining = Self::remaining(deadline)?;
        let frontend = async {
            check_health(&self.frontend_clients, 1, 30, remaining.as_secs_f64().ceil() as u64, false).await?;
            maybe_check_has_model(&self.frontend_clients, model_name, self.skip_model_check).await
        };
        tokio::time::timeout(remaining, frontend)
            .await
            .map_err(|_| anyhow!("Dynamo frontend readiness exceeded {} seconds", self.timeout))??;

        let mut previous: Option<(u64, String)> = None;
        let mut last_error: Option<String> = None;
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) {
            let attempt = match tokio::time::timeout(remaining, self.discover()).await {
                Err(_) => Err(DiscoveryError::Retryable("Dynamo discovery timed out".into())),
                Ok(result) => result,
            };
            match attempt {
                Ok(worker) => {
                    let fingerprint = worker.fingerprint();
                    if previous.as_ref() == Some(&fingerprint) {
                        let client = self.make_worker_client(&worker)?;
                        let remaining = Self::remaining(deadline)?;
                        tokio::time::timeout(
                            remaining,
                            check_health(std::slice::from_ref(&client), 1, 30, remaining.as_secs_f64().ceil() as u64, true),
                        )
                        .await
                        .map_err(|_| anyhow!("Dynamo worker health check timed out"))??;
                        self.fingerprint = Some(fingerprint);
                        self.clients = vec![client];
                        return Ok(());
                    }
                    previous = Some(fingerprint);
                }
                Err(DiscoveryError::Fatal(error)) => return Err(error),
                Err(error) => {
                    previous = None;
                    last_error = Some(error.to_string());
                }
            }
            if let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
                tokio::time::sleep(self.poll_interval.min(remaining)).await;
            }
        }
        let error = anyhow!("Dynamo workers did not become ready before the discovery timeout");
        Err(match last_error {
            Some(cause) => anyhow!(cause).context(error.to_string()),
            None => error,
        })
    }

    /// Fail when the worker topology changed since it was pinned (confirmed by two snapshots).
    pub async fn ensure_topology_current(&self) -> Result<()> {
        if self.nccl_state == NcclState::Terminal {
            bail!("Dynamo administration is in a terminal state; restart is required");
        }
        let pinned = self.fingerprint.as_ref().ok_or_else(|| anyhow!("Dynamo topology has not been pinned"))?;
        let mut previous_changed: Option<(u64, String)> = None;
        let mut last_error: Option<String> = None;
        let deadline = Instant::now() + Duration::from_secs(self.timeout);
        for attempt in 0..3 {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) else { break };
            let result = match tokio::time::timeout(remaining, self.discover()).await {
                Err(_) => Err(DiscoveryError::Retryable("Dynamo discovery timed out".into())),
                Ok(result) => result,
            };
            match result {
                Ok(worker) => {
                    let fingerprint = worker.fingerprint();
                    if &fingerprint == pinned {
                        return Ok(());
                    }
                    if previous_changed.as_ref() == Some(&fingerprint) {
                        bail!("Dynamo worker topology changed after initialization");
                    }
                    previous_changed = Some(fingerprint);
                    last_error = None;
                }
                Err(DiscoveryError::Fatal(error)) => return Err(error),
                Err(error) => last_error = Some(error.to_string()),
            }
            if attempt < 2 {
                let Some(remaining) = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) else { break };
                tokio::time::sleep(self.poll_interval.min(remaining)).await;
            }
        }
        match last_error {
            Some(cause) => Err(anyhow!(cause).context("Could not verify the pinned Dynamo topology")),
            None => bail!("Dynamo topology changed but could not be confirmed"),
        }
    }

    async fn collective_rpc(&self, method: &str, timeout: f64, args: Value) -> Result<()> {
        let operation_timeout = timeout.max(1.0);
        let client = self.clients.first().ok_or_else(|| anyhow!("Dynamo worker is not bound"))?;
        let body = json!({"method": method, "timeout": operation_timeout, "args": args, "kwargs": {}});
        let call = async {
            let response = client
                .post("/collective_rpc")
                .timeout(Duration::from_secs_f64(operation_timeout))
                .json(&body)
                .send()
                .await?
                .error_for_status()?;
            anyhow::Ok(response.json::<Value>().await?)
        };
        let payload = tokio::time::timeout(Duration::from_secs_f64(operation_timeout + 15.0), call)
            .await
            .map_err(|_| anyhow!("Dynamo collective RPC {method} timed out"))??;
        if payload != json!({"results": [null]}) {
            bail!("Dynamo worker returned an invalid collective RPC response");
        }
        Ok(())
    }

    pub async fn update_weights(
        &mut self,
        weight_dir: Option<&Path>,
        transport: Transport,
        step: u64,
        on_paused: impl FnOnce() -> Result<()>,
    ) -> Result<UpdateTimes> {
        if transport != Transport::Nccl {
            self.ensure_topology_current().await?;
            return update_engines(&self.clients, None, weight_dir, step, on_paused).await;
        }
        let weight_dir = weight_dir.ok_or_else(|| anyhow!("NCCL weight updates require a broadcast directory"))?;
        if self.nccl_state != NcclState::Ready {
            bail!("Dynamo weight updates require a ready NCCL initialization");
        }
        self.ensure_topology_current().await?;
        self.nccl_state = NcclState::Terminal;
        let client = self.clients[0].clone();
        let pause_request = AdminRequest::query(vec![("mode", "keep".into()), ("clear_cache", "false".into())]);

        let start = Instant::now();
        admin_post(&client, "/pause", &pause_request, ADMIN_TIMEOUT_S)
            .await
            .context("Dynamo pause failed; worker state is unknown and restart is required")?;
        let pause = start.elapsed().as_secs_f64();
        on_paused().context("Dynamo pause callback failed; engines remain paused and restart is required")?;

        let start = Instant::now();
        let dir = weight_dir.to_string_lossy().into_owned();
        self.collective_rpc("update_weights_from_path", UPDATE_WEIGHTS_TIMEOUT_S, json!([dir]))
            .await
            .context("Dynamo nccl update failed; engines remain paused and restart is required")?;
        let transfer = start.elapsed().as_secs_f64();

        let start = Instant::now();
        admin_post(&client, "/resume", &AdminRequest::default(), ADMIN_TIMEOUT_S)
            .await
            .context("Dynamo resume failed; worker state is unknown and restart is required")?;
        let resume = start.elapsed().as_secs_f64();
        self.nccl_state = NcclState::Ready;
        tracing::info!("Applied NCCL weights for policy v{step} to the Dynamo worker");
        Ok(UpdateTimes { pause: vec![pause], transfer: vec![transfer], resume: vec![resume] })
    }

    pub async fn initialize_nccl(&mut self, host: &str, port: u16, timeout: u64, inference_world_size: u64) -> Result<()> {
        if self.nccl_state != NcclState::Uninitialized {
            bail!("Dynamo NCCL can only be initialized once; restart the inference workers before retrying");
        }
        if inference_world_size != 1 {
            bail!("Dynamo RL currently supports exactly one inference rank");
        }
        self.nccl_state = NcclState::Initializing;
        let result = async {
            self.ensure_topology_current().await?;
            tracing::info!("Initializing Dynamo NCCL broadcast for one inference rank");
            self.collective_rpc("init_broadcaster", timeout as f64, json!([host, port, 0, 1, timeout, "default"])).await
        }
        .await;
        match result {
            Ok(()) => {
                self.nccl_state = NcclState::Ready;
                Ok(())
            }
            Err(error) => {
                self.nccl_state = NcclState::Terminal;
                Err(error.context("Dynamo NCCL initialization failed; inference workers must restart before retrying"))
            }
        }
    }
}

fn classify(error: reqwest::Error) -> DiscoveryError {
    match error.status() {
        Some(status) if status.as_u16() < 500 => DiscoveryError::Fatal(error.into()),
        _ if error.is_decode() => DiscoveryError::Fatal(error.into()),
        _ => DiscoveryError::Retryable(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "Qwen/Qwen3-0.6B";

    fn worker(instance_id: u64, admin: Option<&str>, model: &str) -> Value {
        json!({
            "instance_id": instance_id,
            "admin_base_url": admin.map(str::to_owned).unwrap_or_else(|| format!("http://worker:{}", 8200 + instance_id)),
            "world_size": 1,
            "model": model,
        })
    }

    fn snapshot(workers: Vec<Value>) -> Value {
        json!({"protocol_version": 1, "workers": workers})
    }

    #[test]
    fn parse_filters_model_and_checks_admin_host() {
        let parsed = parse_dynamo_worker(&snapshot(vec![worker(3, None, MODEL), worker(1, None, "other")]), MODEL, Some("worker")).unwrap();
        assert_eq!(parsed.admin_base_url, "http://worker:8203");
        let error = parse_dynamo_worker(&snapshot(vec![worker(1, Some("http://other-worker:8201"), MODEL)]), MODEL, Some("worker")).unwrap_err();
        assert!(error.to_string().contains("does not match discovery host"));
    }

    #[test]
    fn parse_rejects_multiple_matching_workers() {
        let error = parse_dynamo_worker(&snapshot(vec![worker(1, None, MODEL), worker(2, None, MODEL)]), MODEL, Some("worker")).unwrap_err();
        assert!(matches!(error, DiscoveryError::Fatal(_)));
        assert!(error.to_string().contains("exactly one inference worker"));
    }

    #[test]
    fn parse_validates_required_singleton_metadata() {
        let mut missing = worker(1, None, MODEL);
        missing["admin_base_url"] = Value::Null;
        let error = parse_dynamo_worker(&snapshot(vec![missing]), MODEL, Some("worker")).unwrap_err();
        assert!(matches!(&error, DiscoveryError::Pending(m) if m.contains("admin_base_url")));
        let mut wide = worker(1, None, MODEL);
        wide["world_size"] = json!(2);
        let error = parse_dynamo_worker(&snapshot(vec![wide]), MODEL, Some("worker")).unwrap_err();
        assert!(matches!(&error, DiscoveryError::Fatal(e) if e.to_string().contains("exactly one inference rank")));
    }

    fn policy_config(base_url: &str, discovery_url: Option<&str>, enabled: bool) -> PolicyClientConfig {
        PolicyClientConfig {
            client: prl_config::ClientConfig {
                base_url: base_url.into(),
                api_key_var: "VLLM_API_KEY".into(),
                headers: IndexMap::new(),
                headers_from_env: IndexMap::new(),
                timeout: prl_config::ClientTimeoutConfig { connect: Some(30.0), read: None, write: None, pool: None },
                wait_for_ready_timeout: 2,
                skip_model_check: true,
                admin_base_url: None,
            },
            dynamo: Some(prl_config::DynamoConfig { enabled, discovery_url: discovery_url.map(str::to_owned) }),
        }
    }

    #[test]
    fn discovery_url_derivation() {
        let plane = crate::clients::AdminPlane::setup(&policy_config("http://worker:8000/v1", None, true), MODEL, None).unwrap();
        let crate::clients::AdminPlane::Dynamo(plane) = plane else { panic!("expected dynamo") };
        assert_eq!(plane.discovery_url(), "http://worker:8001");
        assert_eq!(plane.discovery_endpoint(), "http://worker:8001/v1/rl/workers");
        let disabled = crate::clients::AdminPlane::setup(&policy_config("http://worker:8000/v1", None, false), MODEL, None).unwrap();
        assert!(matches!(disabled, crate::clients::AdminPlane::Standard(_)));
        let error = DynamoAdminPlane::new(&policy_config("http://worker/v1", None, true), MODEL).unwrap_err();
        assert!(error.to_string().contains("Set dynamo.discovery_url"));
    }

    #[tokio::test]
    async fn nccl_init_rejects_multiple_ranks_without_state_change() {
        let mut plane = DynamoAdminPlane::new(&policy_config("http://worker:8000/v1", Some("http://worker:8001"), true), MODEL).unwrap();
        let error = plane.initialize_nccl("trainer", 29501, 10, 2).await.unwrap_err();
        assert!(error.to_string().contains("exactly one inference rank"));
        assert_eq!(plane.nccl_state, NcclState::Uninitialized);
        let error = plane.update_weights(Some(Path::new("/tmp/step_1")), Transport::Nccl, 1, || Ok(())).await.unwrap_err();
        assert!(error.to_string().contains("ready NCCL initialization"));
    }
}
