//! Port of `tests/unit/eval/test_control.py`.

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use prl_core::concurrency::{ConcurrencyController, ConcurrencyIo, ConcurrencyParams};
use serde_json::{json, Value};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

#[tokio::test]
async fn read_and_set_the_concurrency() {
    let controller = ConcurrencyController::new(
        ConcurrencyParams { min_inflight: 1, max_inflight: Some(2000), initial_inflight: Some(2000) },
        8192,
    );
    let (_samples, samples) = mpsc::channel(1);
    let (_inflight, inflight) = watch::channel(1800);
    let (limit, limit_rx) = watch::channel(0);
    let (handle, _task) = controller.spawn(ConcurrencyIo { samples, inflight, limit, overload: None }, CancellationToken::new());
    let call = |method: Method, body: Option<Value>| {
        let request = Request::builder().method(method).uri("/v1/concurrency").header("content-type", "application/json");
        let request = request.body(body.map_or_else(Body::empty, |b| Body::from(b.to_string()))).unwrap();
        let router = prl_eval::control::router(handle.clone());
        async move {
            let response = router.oneshot(request).await.unwrap();
            let status = response.status();
            let body = response.into_body().collect().await.unwrap().to_bytes();
            (status, serde_json::from_slice::<Value>(&body).unwrap())
        }
    };

    assert_eq!(call(Method::GET, None).await, (StatusCode::OK, json!({"target": null, "cap": 2000, "inflight": 1800})));
    // a pinned target may sit above the band
    let pinned = call(Method::PUT, Some(json!({"target": 2500}))).await;
    assert_eq!(pinned, (StatusCode::OK, json!({"target": 2500, "cap": 2500, "inflight": 1800})));
    assert_eq!(*limit_rx.borrow(), 2500);
    // null hands the cap back to the controller, which starts from the pinned value
    call(Method::PUT, Some(json!({"target": 1500}))).await;
    let adaptive = call(Method::PUT, Some(json!({"target": null}))).await;
    assert_eq!(adaptive, (StatusCode::OK, json!({"target": null, "cap": 1500, "inflight": 1800})));
    for body in [json!({"target": 0}), json!({})] {
        assert_eq!(call(Method::PUT, Some(body)).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    }
}
