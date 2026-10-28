//! The eval's control API (port of `eval/control.py`): read and set its episode concurrency
//! while it runs. `GET /v1/concurrency`, `PUT /v1/concurrency {"target": int >= 1 | null}`.

use anyhow::{Context, Result};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use prl_core::concurrency::{ConcurrencyHandle, ConcurrencyState};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

pub fn router(concurrency: ConcurrencyHandle) -> Router {
    Router::new().route("/v1/concurrency", get(get_concurrency).put(set_concurrency)).with_state(concurrency)
}

/// Serve the control API on `host:port` until `cancel` fires.
pub async fn serve(concurrency: ConcurrencyHandle, host: &str, port: u16, cancel: CancellationToken) -> Result<()> {
    let listener = tokio::net::TcpListener::bind((host, port))
        .await
        .with_context(|| format!("bind the eval control API on {host}:{port}"))?;
    axum::serve(listener, router(concurrency)).with_graceful_shutdown(cancel.cancelled_owned()).await?;
    Ok(())
}

async fn get_concurrency(State(concurrency): State<ConcurrencyHandle>) -> Response {
    reply(concurrency.get().await)
}

async fn set_concurrency(State(concurrency): State<ConcurrencyHandle>, body: Bytes) -> Response {
    match parse_target(&body) {
        Ok(target) => reply(concurrency.set_target(target).await),
        Err(detail) => (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "detail": [detail] }))).into_response(),
    }
}

fn reply(state: Result<ConcurrencyState>) -> Response {
    match state {
        Ok(state) => Json(state).into_response(),
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ "detail": e.to_string() }))).into_response(),
    }
}

/// `ConcurrencyRequest.target: int | None = Field(ge=1)` — required, nullable, at least 1.
fn parse_target(body: &[u8]) -> Result<Option<usize>, Value> {
    let error = |kind: &str, loc: Value, msg: &str, input: &Value| json!({"type": kind, "loc": loc, "msg": msg, "input": input});
    let body: Value = serde_json::from_slice(body)
        .map_err(|e| error("json_invalid", json!(["body"]), "JSON decode error", &json!(e.to_string())))?;
    let Some(target) = body.as_object().and_then(|o| o.get("target")) else {
        return Err(error("missing", json!(["body", "target"]), "Field required", &body));
    };
    let loc = json!(["body", "target"]);
    let value = match target {
        Value::Null => return Ok(None),
        Value::Number(n) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => i,
            (None, Some(f)) if f.fract() == 0.0 => f as i64,
            _ => return Err(error("int_from_float", loc, "Input should be a valid integer, got a number with a fractional part", target)),
        },
        Value::String(s) => s
            .trim()
            .parse()
            .map_err(|_| error("int_parsing", loc.clone(), "Input should be a valid integer, unable to parse string as an integer", target))?,
        _ => return Err(error("int_type", loc, "Input should be a valid integer", target)),
    };
    if value < 1 {
        return Err(error("greater_than_equal", loc, "Input should be greater than or equal to 1", target));
    }
    Ok(Some(value as usize))
}
