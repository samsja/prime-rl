//! The platform HTTP layer of the Prime monitors: settings (`prime_traces.core.Config` and
//! `prime_runs.Config`), the error classification the SDK branches on, and the authenticated
//! platform client with its retry policy (`prime_runs._http`).

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Method, StatusCode};
use serde_json::Value as Json;

const DEFAULT_BASE_URL: &str = "https://api.primeintellect.ai";
pub const USER_AGENT: &str = "prime-runs/0.1.4 prime-rl-rust";
const MAX_ATTEMPTS: u32 = 5;

/// Env vars, then `~/.prime/config.json`.
#[derive(Default)]
pub struct Settings {
    file: serde_json::Map<String, Json>,
}

fn prime_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".prime")
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn strip_api_v1(url: &str) -> String {
    let url = url.trim_end_matches('/');
    url.strip_suffix("/api/v1").unwrap_or(url).to_string()
}

impl Settings {
    pub fn load() -> Self {
        let file = std::fs::read(prime_dir().join("config.json"))
            .ok()
            .and_then(|data| serde_json::from_slice(&data).ok())
            .unwrap_or_default();
        Settings { file }
    }

    /// `prime_sandboxes.Config`: the config file with the prime CLI's `$PRIME_CONTEXT` profile
    /// laid over it. Only the monitors' credential check reads it, as in Python.
    pub fn load_with_context() -> Result<Self, ApiError> {
        let mut settings = Settings::load();
        let Some(context) = env("PRIME_CONTEXT") else {
            return Ok(settings);
        };
        if context.to_lowercase() == "production" {
            settings
                .file
                .insert("base_url".into(), DEFAULT_BASE_URL.into());
            for key in ["team_id", "team_name", "team_role"] {
                settings.file.insert(key.into(), Json::Null);
            }
            return Ok(settings);
        }
        if !context
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(ApiError::Configuration(format!(
                "Invalid context name: {context:?}"
            )));
        }
        let path = prime_dir()
            .join("environments")
            .join(format!("{context}.json"));
        let data = std::fs::read(&path).map_err(|_| {
            ApiError::Configuration(format!("Context file not found: {}", path.display()))
        })?;
        match serde_json::from_slice(&data) {
            Ok(Json::Object(overlay)) => settings.file.extend(overlay),
            Ok(_) => {
                return Err(ApiError::Configuration(format!(
                    "Invalid context '{context}': expected a JSON object"
                )))
            }
            Err(e) => {
                return Err(ApiError::Configuration(format!(
                    "Failed to load context '{context}': {e}"
                )))
            }
        }
        Ok(settings)
    }

    fn file_str(&self, key: &str) -> Option<String> {
        self.file
            .get(key)
            .and_then(Json::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }

    pub fn api_key(&self) -> String {
        env("PRIME_API_KEY")
            .or_else(|| self.file_str("api_key"))
            .unwrap_or_default()
    }

    /// An explicitly empty `$PRIME_TEAM_ID` means personal scope.
    pub fn team_id(&self) -> Option<String> {
        match std::env::var("PRIME_TEAM_ID") {
            Ok(team) => Some(team).filter(|t| !t.is_empty()),
            Err(_) => self.file_str("team_id"),
        }
    }

    pub fn base_url(&self) -> String {
        let url = env("PRIME_API_BASE_URL")
            .or_else(|| env("PRIME_BASE_URL"))
            .or_else(|| self.file_str("base_url"));
        strip_api_v1(&url.unwrap_or_else(|| DEFAULT_BASE_URL.into()))
    }

    /// Never the base URL: only production serves Prime Traces on the API domain.
    pub fn traces_url(&self) -> String {
        strip_api_v1(
            &env("PRIME_TRACES_URL")
                .or_else(|| self.file_str("traces_url"))
                .unwrap_or_else(|| DEFAULT_BASE_URL.into()),
        )
    }

    pub fn frontend_url(&self) -> String {
        let url = env("PRIME_FRONTEND_URL").or_else(|| self.file_str("frontend_url"));
        url.unwrap_or_else(|| "https://app.primeintellect.ai".into())
            .trim_end_matches('/')
            .to_string()
    }

    /// `$PRIME_TRACES_OPT_OUT` > config `traces_opt_out` > off.
    pub fn traces_opt_out(&self) -> Result<bool, ApiError> {
        if let Some(value) = std::env::var("PRIME_TRACES_OPT_OUT")
            .ok()
            .filter(|v| !v.trim().is_empty())
        {
            return match value.trim().to_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                _ => Err(ApiError::Configuration(format!(
                    "PRIME_TRACES_OPT_OUT={value:?} is not one of true or false"
                ))),
            };
        }
        Ok(match self.file.get("traces_opt_out") {
            Some(Json::String(s)) => matches!(
                s.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ),
            Some(Json::Bool(b)) => *b,
            _ => false,
        })
    }
}

/// An error of the platform or traces APIs, or one raised before any request.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    Configuration(String),
    /// A non-2xx response; `code` is the service's error code.
    #[error("{message}")]
    Http {
        status: u16,
        code: Option<String>,
        message: String,
        retry_after: Option<f64>,
    },
    /// No response to interpret; `ambiguous` when the request may have reached the service.
    #[error("{message}")]
    Transport { message: String, ambiguous: bool },
    #[error("Line {line} is {size} bytes, exceeding the {limit} byte single-line limit. Oversized lines are rejected locally rather than split.")]
    TraceTooLarge {
        line: usize,
        size: usize,
        limit: usize,
    },
}

impl ApiError {
    /// The Python exception class, for messages.
    fn kind(&self) -> &'static str {
        match self {
            ApiError::Configuration(_) => "ConfigurationError",
            ApiError::Http { status: 400, .. } => "ValidationRejectedError",
            ApiError::Http { status: 401, .. } => "UnauthorizedError",
            ApiError::Http { status: 403, .. } => "ForbiddenError",
            ApiError::Http { status: 404, .. } => "NotFoundError",
            ApiError::Http { status: 409, .. } => "LineFormatConflictError",
            ApiError::Http {
                status: 429 | 502 | 503 | 504,
                ..
            } => "RetryableAPIError",
            ApiError::Http { .. } => "APIError",
            ApiError::Transport { .. } => "TransportError",
            ApiError::TraceTooLarge { .. } => "TraceTooLargeError",
        }
    }

    /// A failure about this moment (retry later) rather than this run (stop).
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            ApiError::Http {
                status: 429 | 502 | 503 | 504,
                ..
            } | ApiError::Transport { .. }
        )
    }

    /// A failure specific to the submitted batch: a later batch can still succeed.
    pub fn is_record_rejection(&self) -> bool {
        matches!(
            self,
            ApiError::Http {
                status: 400 | 409 | 413 | 422,
                ..
            } | ApiError::TraceTooLarge { .. }
        )
    }

    /// An error from the platform, as opposed to one raised before any request.
    pub fn is_api_error(&self) -> bool {
        matches!(self, ApiError::Http { .. } | ApiError::Transport { .. })
    }

    /// `f"{type(exc).__name__}: {exc}"`.
    pub fn describe(&self) -> String {
        format!("{}: {self}", self.kind())
    }

    pub fn transport(err: &reqwest::Error, what: &str) -> ApiError {
        // connection setup never reached the service; anything later may have
        ApiError::Transport {
            message: format!("{what} failed: {err}"),
            ambiguous: !err.is_connect() && !err.is_builder(),
        }
    }

    /// The error of a non-2xx response: the service envelope's code and message, FastAPI's
    /// `detail`, or the raw text.
    pub fn from_response(status: StatusCode, headers: &HeaderMap, body: &[u8]) -> Option<ApiError> {
        if status.is_success() {
            return None;
        }
        let status = status.as_u16();
        let text = String::from_utf8_lossy(body);
        let fallback = if text.is_empty() {
            format!("HTTP {status}")
        } else {
            text.into_owned()
        };
        let (code, message) = match serde_json::from_slice::<Json>(body) {
            Ok(Json::Object(obj)) => match (obj.get("error"), obj.get("detail")) {
                (Some(Json::Object(error)), _) => (
                    error.get("code").and_then(Json::as_str).map(str::to_string),
                    error
                        .get("message")
                        .and_then(Json::as_str)
                        .filter(|m| !m.is_empty())
                        .map_or_else(|| format!("HTTP {status}"), str::to_string),
                ),
                (_, Some(Json::String(detail))) if !detail.is_empty() => (None, detail.clone()),
                (_, Some(detail)) if !detail.is_null() => (None, detail.to_string()),
                _ => (None, fallback),
            },
            _ => (None, fallback),
        };
        let message = match status {
            401 if message != format!("HTTP {status}") => {
                format!("API key unauthorized ({message}). Check PRIME_API_KEY and its scopes.")
            }
            401 => "API key unauthorized. Check PRIME_API_KEY and its scopes.".into(),
            402 => "Payment required. Check billing status.".into(),
            400 | 403 | 404 | 409 | 429 | 502 | 503 | 504 => message,
            _ => format!("HTTP {status}: {message}"),
        };
        Some(ApiError::Http {
            status,
            code,
            message,
            retry_after: retry_after(headers),
        })
    }
}

/// Retry-After in either form: delta-seconds or an HTTP date.
fn retry_after(headers: &HeaderMap) -> Option<f64> {
    let value = headers.get("retry-after")?.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<f64>() {
        return Some(seconds.max(0.0));
    }
    let target = httpdate::parse_http_date(value).ok()?;
    Some(
        target
            .duration_since(SystemTime::now())
            .map_or(0.0, |d| d.as_secs_f64()),
    )
}

/// Seconds to wait before retrying `attempt` (0-based): Retry-After wins (capped at 60 s),
/// else jittered exponential backoff (1 s doubling, capped at 30 s).
pub fn retry_delay(error: &ApiError, attempt: u32) -> Duration {
    if let ApiError::Http {
        retry_after: Some(delay),
        ..
    } = error
    {
        return Duration::from_secs_f64(delay.min(60.0));
    }
    Duration::from_secs_f64(2f64.powi(attempt as i32).min(30.0) * (0.5 + rand::random::<f64>()))
}

/// `(origin, api_path)` for a base URL that is a platform origin or one of its API roots: the
/// internal RFT root a hosted run's launcher injects maps to `/api/internal`.
fn split_api_root(base_url: &str) -> (String, &'static str) {
    let base_url = base_url.trim_end_matches('/');
    for (suffix, api_path) in [
        ("/api/internal/rft", "/api/internal"),
        ("/api/internal", "/api/internal"),
        ("/api/v1/rft", "/api/v1"),
        ("/api/v1", "/api/v1"),
    ] {
        if let Some(origin) = base_url.strip_suffix(suffix) {
            return (origin.to_string(), api_path);
        }
    }
    (base_url.to_string(), "/api/v1")
}

pub fn http_client(
    headers: HeaderMap,
    read_timeout: Duration,
) -> Result<reqwest::Client, ApiError> {
    reqwest::Client::builder()
        .default_headers(headers)
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(read_timeout)
        .build()
        .map_err(|e| ApiError::Configuration(format!("http client: {e}")))
}

pub fn header(value: &str) -> Result<HeaderValue, ApiError> {
    HeaderValue::from_str(value)
        .map_err(|e| ApiError::Configuration(format!("invalid header value: {e}")))
}

/// Authenticated platform API client: connect errors and 429 are replayed for every call,
/// ambiguous failures (502/503/504, a lost response) only for idempotent ones.
#[derive(Clone)]
pub struct PlatformClient {
    http: reqwest::Client,
    api_prefix: String,
}

impl PlatformClient {
    pub fn new(api_key: &str, base_url: &str) -> Result<Self, ApiError> {
        let (origin, api_path) = split_api_root(base_url);
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            header(&format!("Bearer {api_key}"))?,
        );
        if api_path == "/api/internal" {
            // the internal router reads a hosted run's own token from this header
            headers.insert("x-api-key", header(api_key)?);
        }
        Ok(PlatformClient {
            http: http_client(headers, Duration::from_secs(60))?,
            api_prefix: format!("{origin}{api_path}"),
        })
    }

    /// Send one request, retrying transient failures; the JSON body (`{}` when empty).
    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&Json>,
        idempotent: bool,
    ) -> Result<Json, ApiError> {
        let url = format!("{}{path}", self.api_prefix);
        let what = format!("{method} {path}");
        // an empty mapping sends no body
        let payload = body
            .filter(|b| b.as_object().is_none_or(|m| !m.is_empty()))
            .map(|b| b.to_string());
        let mut attempt = 0;
        loop {
            let mut request = self.http.request(method.clone(), &url);
            if let Some(payload) = &payload {
                request = request
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(payload.clone());
            }
            let error = match request.send().await {
                Err(e) => ApiError::transport(&e, &what),
                Ok(response) => {
                    let (status, headers) = (response.status(), response.headers().clone());
                    let bytes = response
                        .bytes()
                        .await
                        .map_err(|e| ApiError::transport(&e, &what))?;
                    match ApiError::from_response(status, &headers, &bytes) {
                        None if bytes.is_empty() => return Ok(Json::Object(Default::default())),
                        None => {
                            return match serde_json::from_slice(&bytes) {
                                Ok(Json::Object(m)) => Ok(Json::Object(m)),
                                Ok(other) => Ok(serde_json::json!({ "data": other })),
                                Err(_) => Err(ApiError::Http {
                                    status: status.as_u16(),
                                    code: None,
                                    message: format!("{what} returned non-JSON ({status})"),
                                    retry_after: None,
                                }),
                            }
                        }
                        Some(error) if error.is_transient() => error,
                        Some(error) => return Err(error),
                    }
                }
            };
            let ambiguous = match &error {
                ApiError::Http { status, .. } => *status != 429,
                ApiError::Transport { ambiguous, .. } => *ambiguous,
                _ => true,
            };
            attempt += 1;
            if attempt == MAX_ATTEMPTS || (ambiguous && !idempotent) {
                return Err(error);
            }
            tokio::time::sleep(retry_delay(&error, attempt - 1)).await;
        }
    }

    pub async fn get(&self, path: &str) -> Result<Json, ApiError> {
        self.request(Method::GET, path, None, true).await
    }

    pub async fn post(&self, path: &str, body: &Json, idempotent: bool) -> Result<Json, ApiError> {
        self.request(Method::POST, path, Some(body), idempotent)
            .await
    }

    pub async fn put(&self, path: &str, body: &Json) -> Result<Json, ApiError> {
        self.request(Method::PUT, path, Some(body), true).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_roots() {
        assert_eq!(
            split_api_root("https://api.x.ai"),
            ("https://api.x.ai".into(), "/api/v1")
        );
        assert_eq!(
            split_api_root("https://api.x.ai/api/v1/rft/"),
            ("https://api.x.ai".into(), "/api/v1")
        );
        assert_eq!(
            split_api_root("https://x.ai/api/internal/rft"),
            ("https://x.ai".into(), "/api/internal")
        );
    }

    #[test]
    fn response_errors() {
        let mut headers = HeaderMap::new();
        let forbidden = br#"{"error":{"code":"service_not_enabled","message":"off"}}"#;
        let err = ApiError::from_response(StatusCode::FORBIDDEN, &headers, forbidden).unwrap();
        assert_eq!(err.describe(), "ForbiddenError: off");
        let err = ApiError::from_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            &headers,
            br#"{"detail":"bad"}"#,
        )
        .unwrap();
        assert_eq!(err.to_string(), "HTTP 422: bad");
        assert!(err.is_record_rejection());
        headers.insert("retry-after", HeaderValue::from_static("7"));
        let err = ApiError::from_response(StatusCode::TOO_MANY_REQUESTS, &headers, b"").unwrap();
        assert!(err.is_transient());
        assert_eq!(retry_delay(&err, 3), Duration::from_secs(7));
    }
}
