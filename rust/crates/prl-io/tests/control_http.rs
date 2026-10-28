//! The control API's routes and payloads (FastAPI app in `control/app.py`).

use prl_io::control::{serve_on, Controller};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn control_api_routes() {
    let controller = Controller::new(true);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let cancel = CancellationToken::new();
    let server = tokio::spawn(serve_on(controller.clone(), listener, cancel.clone()));
    let http = reqwest::Client::new();

    let missing = http.post(format!("{base}/v1/actions")).json(&json!({"type": "pause"})).send().await.unwrap();
    assert_eq!(missing.status().as_u16(), 422);
    let body: Value = missing.json().await.unwrap();
    assert_eq!(body["detail"], "The orchestrator pauses at a step you name: pass the trainer's accepted 'step'.");

    let bad = http.post(format!("{base}/v1/actions")).json(&json!({"type": "stop"})).send().await.unwrap();
    assert_eq!(bad.status().as_u16(), 422);

    let submitted = controller.submitted();
    let accepted = http.post(format!("{base}/v1/actions")).json(&json!({"type": "pause", "step": 5})).send().await.unwrap();
    assert_eq!(accepted.status().as_u16(), 202);
    let action: Value = accepted.json().await.unwrap();
    assert_eq!((action["type"].clone(), action["state"].clone(), action["step"].clone()), (json!("pause"), json!("pending"), json!(5)));
    tokio::time::timeout(std::time::Duration::from_secs(1), submitted.notified()).await.unwrap();

    let conflict = http.post(format!("{base}/v1/actions")).json(&json!({"type": "pause", "step": 6})).send().await.unwrap();
    assert_eq!(conflict.status().as_u16(), 409);

    assert!(controller.update(5));
    controller.mark_saved(5);
    let id = action["id"].as_str().unwrap();
    let got: Value = http.get(format!("{base}/v1/actions/{id}")).send().await.unwrap().json().await.unwrap();
    assert_eq!(got, json!({"id": id, "type": "pause", "state": "committed", "step": 5}));

    let unknown = http.get(format!("{base}/v1/actions/nope")).send().await.unwrap();
    assert_eq!(unknown.status().as_u16(), 404);
    let body: Value = unknown.json().await.unwrap();
    assert_eq!(body["detail"], "Unknown action 'nope': the run may have restarted since it was submitted; submit it again.");

    cancel.cancel();
    server.await.unwrap().unwrap();
}
