//! Port of `tests/unit/eval/test_control.py`.

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use prl_core::concurrency::{ConcurrencyController, ConcurrencyHandle, ConcurrencyIo, ConcurrencyParams, EngineLoadSample};
use serde_json::{json, Value};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

const INFLIGHT: usize = 1800;

/// The actor plus every limit it published after the initial one.
struct Actor {
    handle: ConcurrencyHandle,
    limit: watch::Receiver<usize>,
    _inflight: watch::Sender<usize>,
    _samples: mpsc::Sender<Vec<EngineLoadSample>>,
    _cancel: CancellationToken,
}

fn spawn() -> Actor {
    let controller = ConcurrencyController::new(
        ConcurrencyParams { min_inflight: 1, max_inflight: Some(2000), initial_inflight: Some(2000) },
        8192,
    );
    let (samples_tx, samples) = mpsc::channel(4);
    let (inflight_tx, inflight) = watch::channel(INFLIGHT);
    let (limit, limit_rx) = watch::channel(0);
    let cancel = CancellationToken::new();
    let (handle, _task) = controller.spawn(ConcurrencyIo { samples, inflight, limit, overload: None }, cancel.clone());
    Actor { handle, limit: limit_rx, _inflight: inflight_tx, _samples: samples_tx, _cancel: cancel }
}

async fn call(handle: &ConcurrencyHandle, method: Method, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri("/v1/concurrency").header("content-type", "application/json");
    let body = body.map(|b| Body::from(b.to_string())).unwrap_or_else(Body::empty);
    let response = prl_eval::control::router(handle.clone()).oneshot(request.body(body).unwrap()).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn set_concurrency() {
    let cases: [(&[Option<usize>], Value, usize); 3] = [
        (&[Some(1500)], json!({"target": 1500, "cap": 1500, "inflight": INFLIGHT}), 1500),
        (&[Some(2500)], json!({"target": 2500, "cap": 2500, "inflight": INFLIGHT}), 2500),
        (&[Some(1500), None], json!({"target": null, "cap": 1500, "inflight": INFLIGHT}), 1500),
    ];
    for (targets, expected, expected_limit) in cases {
        let Actor { handle, limit, .. } = spawn();
        let mut response = None;
        for target in targets {
            response = Some(call(&handle, Method::PUT, Some(json!({ "target": target }))).await);
        }
        assert_eq!(response.unwrap(), (StatusCode::OK, expected));
        assert_eq!(*limit.borrow(), expected_limit);
    }
}

#[tokio::test]
async fn get_concurrency() {
    let actor = spawn();
    let handle = &actor.handle;
    assert_eq!(
        call(handle, Method::GET, None).await,
        (StatusCode::OK, json!({"target": null, "cap": 2000, "inflight": INFLIGHT}))
    );
}

#[tokio::test]
async fn set_concurrency_invalid() {
    let actor = spawn();
    let handle = &actor.handle;
    for body in [json!({"target": 0}), json!({})] {
        assert_eq!(call(handle, Method::PUT, Some(body)).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    }
    assert_eq!(call(handle, Method::GET, None).await.1["cap"], 2000);
}
