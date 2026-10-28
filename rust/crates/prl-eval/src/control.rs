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
use serde::Deserialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;

pub fn router(concurrency: ConcurrencyHandle) -> Router {
    Router::new().route("/v1/concurrency", get(get_concurrency).put(set_concurrency)).with_state(concurrency)
}

pub async fn serve(concurrency: ConcurrencyHandle, host: &str, port: u16, cancel: CancellationToken) -> Result<()> {
    let listener = tokio::net::TcpListener::bind((host, port))
        .await
        .with_context(|| format!("bind the eval control API on {host}:{port}"))?;
    axum::serve(listener, router(concurrency)).with_graceful_shutdown(cancel.cancelled_owned()).await?;
    Ok(())
}

/// `ConcurrencyRequest`: `target` is required (null hands the cap back to the controller).
#[derive(Deserialize)]
struct ConcurrencyRequest {
    #[serde(deserialize_with = "Option::deserialize")]
    target: Option<usize>,
}

async fn get_concurrency(State(concurrency): State<ConcurrencyHandle>) -> Response {
    reply(concurrency.get().await)
}

async fn set_concurrency(State(concurrency): State<ConcurrencyHandle>, body: Bytes) -> Response {
    match serde_json::from_slice::<ConcurrencyRequest>(&body) {
        Ok(ConcurrencyRequest { target: Some(0) }) => invalid("target: Input should be greater than or equal to 1"),
        Ok(request) => reply(concurrency.set_target(request.target).await),
        Err(e) => invalid(&e.to_string()),
    }
}

fn invalid(detail: &str) -> Response {
    (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "detail": detail }))).into_response()
}

fn reply(state: Result<ConcurrencyState>) -> Response {
    match state {
        Ok(state) => Json(state).into_response(),
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ "detail": e.to_string() }))).into_response(),
    }
}
