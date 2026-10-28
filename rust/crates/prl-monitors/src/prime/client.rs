//! The platform HTTP layer shared by the Prime monitors: settings (`prime_traces.core.Config`
//! and `prime_runs.Config`), error mapping (`prime_traces.core.client.raise_for_response`), and
//! the authenticated platform client with its per-call retry policy (`prime_runs._http`).

use std::path::PathBuf;
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Method, StatusCode};
use serde_json::Value as Json;

pub const DEFAULT_BASE_URL: &str = "https://api.primeintellect.ai";
pub const DEFAULT_TRACES_URL: &str = "https://api.primeintellect.ai";
pub const DEFAULT_FRONTEND_URL: &str = "https://app.primeintellect.ai";
pub const USER_AGENT: &str = "prime-runs/0.1.4 prime-rl-rust";
pub const DEFAULT_MAX_ATTEMPTS: u32 = 5;
pub const PUBLIC_API_PATH: &str = "/api/v1";
pub const INTERNAL_API_PATH: &str = "/api/internal";
const BACKOFF_BASE_S: f64 = 1.0;
const BACKOFF_CAP_S: f64 = 30.0;
const RETRY_AFTER_CAP_S: f64 = 60.0;

/// Env vars, then `~/.prime/config.json`.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    file: serde_json::Map<String, Json>,
}

fn strip_api_v1(url: &str) -> String {
    let url = url.trim_end_matches('/');
    url.strip_suffix("/api/v1").unwrap_or(url).to_string()
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

impl Settings {
    pub fn load() -> Self {
        let path = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(".prime/config.json");
        let file = std::fs::read(&path)
            .ok()
            .and_then(|data| serde_json::from_slice::<Json>(&data).ok())
            .and_then(|v| match v {
                Json::Object(m) => Some(m),
                _ => None,
            })
            .unwrap_or_default();
        Settings { file }
    }

    fn file_str(&self, key: &str) -> Option<String> {
        self.file
            .get(key)
            .and_then(Json::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }

    pub fn api_key(&self) -> String {
        env_nonempty("PRIME_API_KEY")
            .or_else(|| self.file_str("api_key"))
            .unwrap_or_default()
    }

    /// An explicitly empty `$PRIME_TEAM_ID` means personal scope.
    pub fn team_id(&self) -> Option<String> {
        match std::env::var("PRIME_TEAM_ID") {
            Ok(v) => Some(v).filter(|v| !v.is_empty()),
            Err(_) => self.file_str("team_id"),
        }
    }

    pub fn base_url(&self) -> String {
        let url = env_nonempty("PRIME_API_BASE_URL")
            .or_else(|| env_nonempty("PRIME_BASE_URL"))
            .or_else(|| self.file_str("base_url"))
            .unwrap_or_else(|| DEFAULT_BASE_URL.into());
        strip_api_v1(&url)
    }

    pub fn traces_url(&self) -> String {
        let url = env_nonempty("PRIME_TRACES_URL")
            .or_else(|| self.file_str("traces_url"))
            .unwrap_or_else(|| DEFAULT_TRACES_URL.into());
        strip_api_v1(&url)
    }

    pub fn frontend_url(&self) -> String {
        env_nonempty("PRIME_FRONTEND_URL")
            .or_else(|| self.file_str("frontend_url"))
            .unwrap_or_else(|| DEFAULT_FRONTEND_URL.into())
            .trim_end_matches('/')
            .to_string()
    }

    /// `$PRIME_TRACES_OPT_OUT` > config `traces_opt_out` > off.
    pub fn traces_opt_out(&self) -> Result<bool, ApiError> {
        if let Some(v) = std::env::var("PRIME_TRACES_OPT_OUT")
            .ok()
            .filter(|v| !v.trim().is_empty())
        {
            return match v.trim().to_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                _ => Err(ApiError::Configuration(format!(
                    "PRIME_TRACES_OPT_OUT={v:?} is not one of true or false"
                ))),
            };
        }
        Ok(match self.file.get("traces_opt_out") {
            Some(Json::String(s)) => matches!(
                s.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ),
            Some(Json::Bool(b)) => *b,
            Some(Json::Number(n)) => n.as_f64() != Some(0.0),
            _ => false,
        })
    }
}

/// Errors of the platform and traces APIs, classified the way the SDK branches on them.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    Configuration(String),
    #[error("{message}")]
    Unauthorized {
        message: String,
        code: Option<String>,
    },
    #[error("Payment required. Check billing status.")]
    PaymentRequired { code: Option<String> },
    #[error("{message}")]
    Forbidden {
        message: String,
        code: Option<String>,
    },
    #[error("{message}")]
    NotFound {
        message: String,
        code: Option<String>,
    },
    #[error("{message}")]
    ValidationRejected {
        message: String,
        code: Option<String>,
    },
    #[error("{message}")]
    Conflict {
        message: String,
        code: Option<String>,
    },
    #[error("{message}")]
    Retryable {
        status: u16,
        message: String,
        code: Option<String>,
        retry_after: Option<f64>,
    },
    #[error("{message}")]
    Api {
        status: Option<u16>,
        message: String,
        code: Option<String>,
    },
    #[error("{message}")]
    Timeout { message: String, ambiguous: bool },
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
    pub fn kind(&self) -> &'static str {
        match self {
            ApiError::Configuration(_) => "ConfigurationError",
            ApiError::Unauthorized { .. } => "UnauthorizedError",
            ApiError::PaymentRequired { .. } => "PaymentRequiredError",
            ApiError::Forbidden { .. } => "ForbiddenError",
            ApiError::NotFound { .. } => "NotFoundError",
            ApiError::ValidationRejected { .. } => "ValidationRejectedError",
            ApiError::Conflict { .. } => "LineFormatConflictError",
            ApiError::Retryable { .. } => "RetryableAPIError",
            ApiError::Api { .. } => "APIError",
            ApiError::Timeout { .. } => "APITimeoutError",
            ApiError::Transport { .. } => "TransportError",
            ApiError::TraceTooLarge { .. } => "TraceTooLargeError",
        }
    }

    pub fn status(&self) -> Option<u16> {
        match self {
            ApiError::Unauthorized { .. } => Some(401),
            ApiError::PaymentRequired { .. } => Some(402),
            ApiError::Forbidden { .. } => Some(403),
            ApiError::NotFound { .. } => Some(404),
            ApiError::ValidationRejected { .. } => Some(400),
            ApiError::Conflict { .. } => Some(409),
            ApiError::Retryable { status, .. } => Some(*status),
            ApiError::Api { status, .. } => *status,
            _ => None,
        }
    }

    pub fn code(&self) -> Option<&str> {
        match self {
            ApiError::Unauthorized { code, .. }
            | ApiError::PaymentRequired { code }
            | ApiError::Forbidden { code, .. }
            | ApiError::NotFound { code, .. }
            | ApiError::ValidationRejected { code, .. }
            | ApiError::Conflict { code, .. }
            | ApiError::Retryable { code, .. }
            | ApiError::Api { code, .. } => code.as_deref(),
            _ => None,
        }
    }

    /// A failure about this moment (retry later) rather than this run (stop).
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            ApiError::Retryable { .. } | ApiError::Timeout { .. } | ApiError::Transport { .. }
        )
    }

    /// A failure specific to the submitted batch: a later batch can still succeed.
    pub fn is_record_rejection(&self) -> bool {
        matches!(
            self,
            ApiError::TraceTooLarge { .. }
                | ApiError::ValidationRejected { .. }
                | ApiError::Conflict { .. }
        ) || matches!(
            self,
            ApiError::Api {
                status: Some(413 | 422),
                ..
            }
        )
    }

    /// An error from the platform (`prime_traces.exceptions.APIError` and subclasses), as
    /// opposed to one raised before any request.
    pub fn is_api_error(&self) -> bool {
        !matches!(
            self,
            ApiError::Configuration(_) | ApiError::TraceTooLarge { .. }
        )
    }

    /// `f"{type(exc).__name__}: {exc}"`.
    pub fn describe(&self) -> String {
        format!("{}: {self}", self.kind())
    }

    fn retry_after(&self) -> Option<f64> {
        match self {
            ApiError::Retryable { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    pub fn from_reqwest(err: &reqwest::Error, what: &str) -> ApiError {
        // connection setup never reached the service; anything later may have
        let ambiguous = !err.is_connect() && !err.is_builder();
        if err.is_timeout() {
            ApiError::Timeout {
                message: format!("{what} timed out: {err}"),
                ambiguous,
            }
        } else {
            ApiError::Transport {
                message: format!("{what} failed: {err}"),
                ambiguous,
            }
        }
    }
}

/// Seconds to wait before retrying `attempt` (0-based): Retry-After wins (capped), else
/// jittered exponential backoff.
pub fn retry_delay(error: &ApiError, attempt: u32) -> Duration {
    if let Some(delay) = error.retry_after() {
        return Duration::from_secs_f64(delay.min(RETRY_AFTER_CAP_S));
    }
    let base = (BACKOFF_BASE_S * 2f64.powi(attempt as i32)).min(BACKOFF_CAP_S);
    Duration::from_secs_f64(base * (0.5 + rand::random::<f64>()))
}

fn parse_retry_after(headers: &HeaderMap) -> Option<f64> {
    let value = headers.get("retry-after")?.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<f64>() {
        return Some(seconds.max(0.0));
    }
    let target = httpdate::parse_http_date(value).ok()?;
    Some(
        target
            .duration_since(std::time::SystemTime::now())
            .map_or(0.0, |d| d.as_secs_f64()),
    )
}

/// `(code, message)` from an error body: the service envelope, FastAPI's `detail`, or raw text.
fn extract_error(status: StatusCode, body: &[u8]) -> (Option<String>, String) {
    let text = String::from_utf8_lossy(body).into_owned();
    let fallback = if text.is_empty() {
        format!("HTTP {}", status.as_u16())
    } else {
        text.clone()
    };
    let Ok(Json::Object(obj)) = serde_json::from_slice::<Json>(body) else {
        return (None, fallback);
    };
    if let Some(Json::Object(error)) = obj.get("error") {
        let code = error.get("code").and_then(Json::as_str).map(str::to_string);
        let message = error
            .get("message")
            .and_then(Json::as_str)
            .filter(|m| !m.is_empty())
            .map_or_else(|| format!("HTTP {}", status.as_u16()), str::to_string);
        return (code, message);
    }
    match obj.get("detail") {
        Some(Json::String(s)) if !s.is_empty() => (None, s.clone()),
        Some(d) if !d.is_null() && d != &Json::Bool(false) => (None, d.to_string()),
        _ => (None, fallback),
    }
}

/// Map a non-2xx response to a typed error.
pub fn error_for_response(
    status: StatusCode,
    headers: &HeaderMap,
    body: &[u8],
) -> Option<ApiError> {
    if status.is_success() {
        return None;
    }
    let s = status.as_u16();
    let (code, message) = extract_error(status, body);
    Some(match s {
        401 => {
            let detail = if message != format!("HTTP {s}") {
                format!(" ({message})")
            } else {
                String::new()
            };
            ApiError::Unauthorized {
                message: format!(
                    "API key unauthorized{detail}. Check PRIME_API_KEY and its scopes."
                ),
                code,
            }
        }
        402 => ApiError::PaymentRequired { code },
        403 => ApiError::Forbidden { message, code },
        404 => ApiError::NotFound { message, code },
        400 => ApiError::ValidationRejected { message, code },
        409 => ApiError::Conflict { message, code },
        429 | 502 | 503 | 504 => ApiError::Retryable {
            status: s,
            message,
            code,
            retry_after: parse_retry_after(headers),
        },
        _ => ApiError::Api {
            status: Some(s),
            message: format!("HTTP {s}: {message}"),
            code,
        },
    })
}

/// `(origin, api_path)` for a base URL that is a platform origin or one of its API roots.
pub fn split_api_root(base_url: &str) -> (String, &'static str) {
    let (scheme_host, path) = match base_url.find("://") {
        Some(i) => match base_url[i + 3..].find('/') {
            Some(j) => base_url.split_at(i + 3 + j),
            None => (base_url, ""),
        },
        None => (base_url, ""),
    };
    let path = path
        .split(['?', '#'])
        .next()
        .unwrap_or("")
        .trim_end_matches('/');
    let mut api_path = PUBLIC_API_PATH;
    let mut origin_path = path;
    for (suffix, candidate) in [
        ("/api/internal/rft", INTERNAL_API_PATH),
        ("/api/internal", INTERNAL_API_PATH),
        ("/api/v1/rft", PUBLIC_API_PATH),
        ("/api/v1", PUBLIC_API_PATH),
    ] {
        if let Some(stripped) = path.strip_suffix(suffix) {
            origin_path = stripped;
            api_path = candidate;
            break;
        }
    }
    (
        format!("{scheme_host}{origin_path}")
            .trim_end_matches('/')
            .to_string(),
        api_path,
    )
}

/// Authenticated platform API client with the SDK's retry policy: connect errors and 429 are
/// replayed for every method; ambiguous failures (502/503/504, a lost response) only when the
/// call is idempotent.
#[derive(Clone)]
pub struct PlatformClient {
    http: reqwest::Client,
    pub api_prefix: String,
    pub internal: bool,
    max_attempts: u32,
}

impl PlatformClient {
    pub fn new(api_key: &str, base_url: &str, read_timeout: Duration) -> Result<Self, ApiError> {
        let (origin, api_path) = split_api_root(base_url);
        let internal = api_path == INTERNAL_API_PATH;
        let mut headers = HeaderMap::new();
        let bearer = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|e| ApiError::Configuration(format!("invalid API key header: {e}")))?;
        headers.insert(reqwest::header::AUTHORIZATION, bearer);
        if internal {
            headers.insert(
                "x-api-key",
                HeaderValue::from_str(api_key)
                    .map_err(|e| ApiError::Configuration(e.to_string()))?,
            );
        }
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent(USER_AGENT)
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(read_timeout)
            .build()
            .map_err(|e| ApiError::Configuration(format!("http client: {e}")))?;
        Ok(PlatformClient {
            http,
            api_prefix: format!("{origin}{api_path}"),
            internal,
            max_attempts: DEFAULT_MAX_ATTEMPTS,
        })
    }

    /// Send one request, retrying transient failures; returns the JSON body (`{}` when empty).
    /// `idempotent` defaults to `method != POST`.
    pub async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&Json>,
        idempotent: Option<bool>,
    ) -> Result<Json, ApiError> {
        let url = format!("{}{path}", self.api_prefix);
        let what = format!("{method} {path}");
        let replayable = idempotent.unwrap_or(method != Method::POST);
        let payload = body
            .filter(|b| !is_empty_json(b))
            .map(|b| serde_json::to_vec(b).expect("JSON body"));
        for attempt in 0..self.max_attempts {
            let mut req = self.http.request(method.clone(), &url);
            if let Some(payload) = &payload {
                req = req
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(payload.clone());
            }
            let (error, ambiguous) = match req.send().await {
                Err(e) => {
                    let error = ApiError::from_reqwest(&e, &what);
                    let ambiguous = matches!(
                        error,
                        ApiError::Timeout {
                            ambiguous: true,
                            ..
                        } | ApiError::Transport {
                            ambiguous: true,
                            ..
                        }
                    );
                    (error, ambiguous)
                }
                Ok(response) => {
                    let status = response.status();
                    let headers = response.headers().clone();
                    let bytes = response
                        .bytes()
                        .await
                        .map_err(|e| ApiError::from_reqwest(&e, &what))?;
                    match error_for_response(status, &headers, &bytes) {
                        None => return decode(&what, status, &bytes),
                        Some(error @ ApiError::Retryable { status, .. }) => (error, status != 429),
                        Some(error) => return Err(error),
                    }
                }
            };
            if attempt + 1 == self.max_attempts || (ambiguous && !replayable) {
                return Err(error);
            }
            tokio::time::sleep(retry_delay(&error, attempt)).await;
        }
        unreachable!("the loop returns on its last attempt")
    }

    pub async fn get(&self, path: &str) -> Result<Json, ApiError> {
        self.request(Method::GET, path, None, None).await
    }
    pub async fn post(&self, path: &str, body: &Json, idempotent: bool) -> Result<Json, ApiError> {
        self.request(Method::POST, path, Some(body), Some(idempotent))
            .await
    }
    pub async fn put(&self, path: &str, body: &Json) -> Result<Json, ApiError> {
        self.request(Method::PUT, path, Some(body), None).await
    }
}

/// `encode_json(json_body) if json_body else None`: an empty mapping sends no body.
fn is_empty_json(value: &Json) -> bool {
    matches!(value, Json::Object(m) if m.is_empty())
}

fn decode(what: &str, status: StatusCode, body: &[u8]) -> Result<Json, ApiError> {
    if body.is_empty() {
        return Ok(Json::Object(Default::default()));
    }
    match serde_json::from_slice::<Json>(body) {
        Ok(Json::Object(m)) => Ok(Json::Object(m)),
        Ok(other) => Ok(serde_json::json!({ "data": other })),
        Err(_) => Err(ApiError::Api {
            status: Some(status.as_u16()),
            message: format!(
                "{what} returned non-JSON ({}): {:?}",
                status.as_u16(),
                String::from_utf8_lossy(&body[..body.len().min(200)])
            ),
            code: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_roots() {
        assert_eq!(
            split_api_root("https://api.x.ai"),
            ("https://api.x.ai".into(), PUBLIC_API_PATH)
        );
        assert_eq!(
            split_api_root("https://api.x.ai/api/v1/rft/"),
            ("https://api.x.ai".into(), PUBLIC_API_PATH)
        );
        assert_eq!(
            split_api_root("https://x.ai/api/internal/rft"),
            ("https://x.ai".into(), INTERNAL_API_PATH)
        );
        assert_eq!(
            split_api_root("http://localhost:8000/p/api/internal"),
            ("http://localhost:8000/p".into(), INTERNAL_API_PATH)
        );
        assert_eq!(
            split_api_root("https://rft.host"),
            ("https://rft.host".into(), PUBLIC_API_PATH)
        );
    }

    #[test]
    fn response_mapping() {
        let h = HeaderMap::new();
        let err = error_for_response(
            StatusCode::FORBIDDEN,
            &h,
            br#"{"error":{"code":"service_not_enabled","message":"off"}}"#,
        )
        .unwrap();
        assert!(
            matches!(&err, ApiError::Forbidden { code: Some(c), message } if c == "service_not_enabled" && message == "off")
        );
        let err = error_for_response(StatusCode::UNPROCESSABLE_ENTITY, &h, br#"{"detail":"bad"}"#)
            .unwrap();
        assert_eq!(err.to_string(), "HTTP 422: bad");
        assert!(err.is_record_rejection());
        let mut h = HeaderMap::new();
        h.insert("retry-after", HeaderValue::from_static("7"));
        let err = error_for_response(StatusCode::TOO_MANY_REQUESTS, &h, b"").unwrap();
        assert!(err.is_transient());
        assert_eq!(retry_delay(&err, 3), Duration::from_secs(7));
        assert!(error_for_response(StatusCode::OK, &h, b"").is_none());
    }
}
