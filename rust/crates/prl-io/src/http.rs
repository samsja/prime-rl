//! HTTP plumbing shared by the inference clients: header/API-key resolution (verifiers'
//! `resolve_headers`/`resolve_api_key`), admin clients, and the tenacity retry policies.

use std::future::Future;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use prl_config::ClientConfig;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

const PRIME_INFERENCE_HOST: &str = "pinference.ai";
const PRIME_TEAM_ID_HEADER: &str = "X-Prime-Team-ID";

/// `~/.prime/config.json` (`prime login`), empty when absent or invalid.
fn load_prime_config() -> serde_json::Map<String, Value> {
    let Some(home) = std::env::var_os("HOME") else { return Default::default() };
    let path = std::path::Path::new(&home).join(".prime/config.json");
    let Ok(text) = std::fs::read_to_string(&path) else { return Default::default() };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(map)) => map,
        Ok(_) => {
            tracing::warn!("Invalid prime config: expected dict");
            Default::default()
        }
        Err(error) => {
            tracing::warn!("Failed to load prime config: {error}");
            Default::default()
        }
    }
}

fn host_of(url: &str) -> String {
    url::Url::parse(url).ok().and_then(|u| u.host_str().map(str::to_owned)).unwrap_or_default()
}

fn is_prime_inference(api_key_var: &str, base_url: &str) -> bool {
    let host = host_of(base_url);
    api_key_var == "PRIME_API_KEY" && (host == PRIME_INFERENCE_HOST || host.ends_with(&format!(".{PRIME_INFERENCE_HOST}")))
}

/// `resolve_api_key`: the env var, falling back to the Prime CLI config for a pinference
/// endpoint keyed by `PRIME_API_KEY`; `"EMPTY"` when unset.
pub(crate) fn resolve_api_key(api_key_var: &str, base_url: &str) -> String {
    let mut key = std::env::var(api_key_var).ok().filter(|k| !k.is_empty());
    if key.is_none() && is_prime_inference(api_key_var, base_url) {
        key = load_prime_config().get("api_key").and_then(Value::as_str).map(str::to_owned);
    }
    key.filter(|k| !k.is_empty()).unwrap_or_else(|| "EMPTY".into())
}

/// `resolve_headers`: static headers, then set `headers_from_env` values, merged
/// case-insensitively keeping the first spelling; plus the Prime team header for pinference.
pub(crate) fn resolve_headers(
    headers: &IndexMap<String, String>,
    headers_from_env: &IndexMap<String, String>,
    api_key_var: &str,
    base_url: &str,
) -> IndexMap<String, String> {
    let mut names: IndexMap<String, String> = IndexMap::new();
    let mut resolved: IndexMap<String, String> = IndexMap::new();
    let env_values = headers_from_env.iter().filter_map(|(name, var)| std::env::var(var).ok().map(|v| (name.clone(), v)));
    for (name, value) in headers.iter().map(|(n, v)| (n.clone(), v.clone())).chain(env_values) {
        let spelling = names.entry(name.to_lowercase()).or_insert(name).clone();
        resolved.insert(spelling, value);
    }
    if is_prime_inference(api_key_var, base_url) && !names.contains_key(&PRIME_TEAM_ID_HEADER.to_lowercase()) {
        let team_id = std::env::var("PRIME_TEAM_ID")
            .ok()
            .or_else(|| load_prime_config().get("team_id").and_then(Value::as_str).map(str::to_owned));
        if let Some(team_id) = team_id.filter(|t| !t.is_empty()) {
            resolved.insert(PRIME_TEAM_ID_HEADER.into(), team_id);
        }
    }
    resolved
}

pub(crate) fn header_map(headers: &IndexMap<String, String>) -> anyhow::Result<HeaderMap> {
    let mut map = HeaderMap::new();
    for (name, value) in headers {
        map.insert(HeaderName::from_bytes(name.as_bytes())?, HeaderValue::from_str(value)?);
    }
    Ok(map)
}

/// One engine's admin endpoint (`setup_admin_clients`): a dedicated small connection pool so
/// admin ops never queue behind streaming requests. No default timeout; calls bound themselves.
#[derive(Debug, Clone)]
pub struct AdminClient {
    /// Root URL without trailing slash or `/v1` (admin routes live at the root).
    base: String,
    http: reqwest::Client,
}

impl AdminClient {
    pub fn new(base_url: &str, headers: HeaderMap) -> anyhow::Result<Self> {
        Self::with_builder(base_url, reqwest::Client::builder().default_headers(headers))
    }

    /// Admin client over a custom reqwest builder (pool limits are applied here).
    pub(crate) fn with_builder(base_url: &str, builder: reqwest::ClientBuilder) -> anyhow::Result<Self> {
        let base = base_url.trim_end_matches('/');
        let base = base.strip_suffix("/v1").unwrap_or(base).trim_end_matches('/').to_owned();
        let http = builder.pool_max_idle_per_host(1).connect_timeout(Duration::from_secs(10)).build()?;
        Ok(Self { base, http })
    }

    /// `str(client.base_url)`: httpx enforces a trailing slash on base URLs that carry a path.
    pub fn base_url(&self) -> String {
        let has_path = url::Url::parse(&self.base).is_ok_and(|u| u.path() != "/");
        if has_path {
            format!("{}/", self.base)
        } else {
            self.base.clone()
        }
    }

    /// `str(client.base_url).rstrip("/")`.
    pub fn key(&self) -> &str {
        &self.base
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    pub fn get(&self, path: &str) -> reqwest::RequestBuilder {
        self.http.get(self.url(path))
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        self.http.post(self.url(path))
    }
}

/// `setup_admin_clients`: one admin client per `admin_base_url` (or `base_url`), carrying the
/// resolved headers plus `Authorization: Bearer <key>` unless the key is `EMPTY`.
pub(crate) fn setup_admin_clients(config: &ClientConfig) -> anyhow::Result<Vec<AdminClient>> {
    let urls = match &config.admin_base_url {
        Some(urls) if !urls.is_empty() => urls.clone(),
        _ => vec![config.base_url.clone()],
    };
    urls.iter().map(|url| admin_client_for(config, url)).collect()
}

/// The admin client of `config.base_url` itself (the client-facing router).
pub(crate) fn router_admin_client(config: &ClientConfig) -> anyhow::Result<AdminClient> {
    admin_client_for(config, &config.base_url)
}

fn admin_client_for(config: &ClientConfig, url: &str) -> anyhow::Result<AdminClient> {
    let mut headers = resolve_headers(&config.headers, &config.headers_from_env, &config.api_key_var, url);
    let api_key = resolve_api_key(&config.api_key_var, url);
    if api_key != "EMPTY" {
        headers.insert("Authorization".into(), format!("Bearer {api_key}"));
    }
    AdminClient::new(url, header_map(&headers)?)
}

/// A tenacity policy: `stop_after_delay(max_delay) | stop_after_attempt(max_attempts)`,
/// `wait_exponential(multiplier=1, min=1, max=10)`, `reraise=True`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RetryPolicy {
    pub(crate) max_delay: Duration,
    pub(crate) max_attempts: u32,
}

impl RetryPolicy {
    /// Seconds slept after the `attempt`-th failure (1-based).
    fn wait(attempt: u32) -> Duration {
        let exp = 2f64.powi(attempt as i32 - 1);
        Duration::from_secs_f64(exp.clamp(1.0, 10.0))
    }

    pub(crate) async fn run<T, E, F, Fut>(&self, retryable: impl Fn(&E) -> bool, mut call: F) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let start = Instant::now();
        let mut attempt = 0;
        loop {
            attempt += 1;
            match call().await {
                Ok(value) => return Ok(value),
                Err(error) => {
                    if !retryable(&error) || attempt >= self.max_attempts || start.elapsed() >= self.max_delay {
                        return Err(error);
                    }
                    tokio::time::sleep(Self::wait(attempt)).await;
                }
            }
        }
    }
}

/// Per-attempt read timeout for admin ops (sized for `/pause`, which drains in-flight requests).
pub(crate) const ADMIN_TIMEOUT_S: f64 = 300.0;
/// `/update_weights` runs a collective receive across all DP workers.
pub(crate) const UPDATE_WEIGHTS_TIMEOUT_S: f64 = 720.0;

/// Admin ops retry transient server errors (5xx) and transport failures, never 4xx.
pub(crate) fn is_retryable_admin_error(error: &reqwest::Error) -> bool {
    match error.status() {
        Some(status) => status.as_u16() >= 500,
        None => !error.is_builder(),
    }
}

/// LoRA loads also retry 404 (adapter files not yet visible on NFS) and 500.
pub(crate) fn is_retryable_lora_error(error: &reqwest::Error) -> bool {
    match error.status() {
        Some(status) => matches!(status.as_u16(), 404 | 500),
        None => !error.is_builder(),
    }
}

/// Query and JSON body of an admin POST.
#[derive(Debug, Clone, Default)]
pub(crate) struct AdminRequest {
    pub(crate) query: Vec<(&'static str, String)>,
    pub(crate) json: Option<Value>,
}

impl AdminRequest {
    pub(crate) fn json(json: Value) -> Self {
        Self { query: Vec::new(), json: Some(json) }
    }

    pub(crate) fn query(query: Vec<(&'static str, String)>) -> Self {
        Self { query, json: None }
    }

    fn apply(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if !self.query.is_empty() {
            builder = builder.query(&self.query);
        }
        if let Some(json) = &self.json {
            builder = builder.json(json);
        }
        builder
    }
}

/// One admin POST attempt with a bounded (per-attempt) timeout; non-2xx is an error.
pub(crate) async fn admin_post_once(client: &AdminClient, path: &str, request: &AdminRequest, timeout_s: f64) -> Result<(), reqwest::Error> {
    let builder = request.apply(client.post(path)).timeout(Duration::from_secs_f64(timeout_s));
    builder.send().await?.error_for_status()?;
    Ok(())
}

/// `_admin_post`: retried POST, total budget `2 * timeout_s` or 10 attempts.
pub(crate) async fn admin_post(client: &AdminClient, path: &str, request: &AdminRequest, timeout_s: f64) -> Result<(), reqwest::Error> {
    let policy = RetryPolicy { max_delay: Duration::from_secs_f64(2.0 * timeout_s), max_attempts: 10 };
    policy.run(is_retryable_admin_error, || admin_post_once(client, path, request, timeout_s)).await
}

/// `_post_all`: POST `path` to every engine concurrently; each engine's request time in seconds.
pub(crate) async fn post_all(clients: &[AdminClient], path: &str, request: &AdminRequest, timeout_s: f64) -> Result<Vec<f64>, reqwest::Error> {
    futures::future::try_join_all(clients.iter().map(|client| async move {
        let start = Instant::now();
        admin_post(client, path, request, timeout_s).await?;
        Ok(start.elapsed().as_secs_f64())
    }))
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_wait_matches_tenacity() {
        let waits: Vec<f64> = (1..=6).map(|a| RetryPolicy::wait(a).as_secs_f64()).collect();
        assert_eq!(waits, vec![1.0, 2.0, 4.0, 8.0, 10.0, 10.0]);
    }

    #[test]
    fn admin_base_strips_v1() {
        let client = AdminClient::new("http://worker-a:8000/v1/", HeaderMap::new()).unwrap();
        assert_eq!(client.base_url(), "http://worker-a:8000");
        assert_eq!(AdminClient::new("http://h/api", HeaderMap::new()).unwrap().base_url(), "http://h/api/");
        assert_eq!(client.key(), "http://worker-a:8000");
        assert_eq!(client.url("/health"), "http://worker-a:8000/health");
    }

    #[test]
    fn headers_merge_case_insensitively() {
        let headers: IndexMap<String, String> = [("X-Test".to_string(), "a".to_string())].into_iter().collect();
        std::env::set_var("PRL_IO_TEST_HEADER", "b");
        let from_env: IndexMap<String, String> =
            [("x-test".to_string(), "PRL_IO_TEST_HEADER".to_string()), ("X-Missing".to_string(), "PRL_IO_UNSET_VAR".to_string())]
                .into_iter()
                .collect();
        let resolved = resolve_headers(&headers, &from_env, "VLLM_API_KEY", "http://localhost:8000/v1");
        assert_eq!(resolved.into_iter().collect::<Vec<_>>(), vec![("X-Test".to_string(), "b".to_string())]);
    }
}
