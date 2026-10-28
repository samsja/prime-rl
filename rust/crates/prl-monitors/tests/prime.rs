//! The Prime train monitor against a mock platform: registration, metrics rows, Prime Traces
//! uploads and the run's close-out, through the monitors actor.

use std::io::Read;
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::{Multipart, Path, State};
use axum::http::HeaderMap;
use axum::routing::{post, put};
use axum::{Json, Router};
use prl_monitors::prime::backend::TrainingSpec;
use prl_monitors::{Monitors, MonitorsSetup, PrimeSetup, PrimeTrainSetup, Row, Scalar};
use prl_types::{Kind, Subset};
use prl_wire::Value;
use serde_json::{json, Value as Json_};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
struct Request {
    path: String,
    headers: HeaderMap,
    body: Json_,
}

type Log = Arc<Mutex<Vec<Request>>>;

async fn json_route(State(log): State<Log>, axum::extract::OriginalUri(uri): axum::extract::OriginalUri, headers: HeaderMap, body: Bytes) -> Json<Json_> {
    let path = uri.path().to_string();
    let body: Json_ = if body.is_empty() { Json_::Null } else { serde_json::from_slice(&body).unwrap() };
    log.lock().unwrap().push(Request { path: path.clone(), headers, body });
    if path.ends_with("/rft/external-runs") {
        Json(json!({"run": {"id": "r1", "name": "my-run"}}))
    } else {
        Json(json!({}))
    }
}

async fn status_route(State(log): State<Log>, Path(id): Path<String>, headers: HeaderMap, Json(body): Json<Json_>) -> Json<Json_> {
    log.lock().unwrap().push(Request { path: format!("status/{id}"), headers, body });
    Json(json!({}))
}

async fn traces_route(State(log): State<Log>, headers: HeaderMap, mut multipart: Multipart) -> Json<Json_> {
    let mut parts = serde_json::Map::new();
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        let file_name = field.file_name().map(str::to_string);
        let content_type = field.content_type().map(str::to_string);
        let encoding = field.headers().get("content-encoding").map(|v| v.to_str().unwrap().to_string());
        let data = field.bytes().await.unwrap();
        let text = if encoding.as_deref() == Some("gzip") {
            let mut out = String::new();
            flate2::read::GzDecoder::new(&data[..]).read_to_string(&mut out).unwrap();
            out
        } else {
            String::from_utf8(data.to_vec()).unwrap()
        };
        parts.insert(name, json!({"file_name": file_name, "content_type": content_type, "encoding": encoding, "text": text}));
    }
    log.lock().unwrap().push(Request { path: "/api/v1/traces".into(), headers, body: Json_::Object(parts) });
    Json(json!({"upload_id": "x", "status": "committed"}))
}

async fn serve() -> (String, Log) {
    let log: Log = Arc::default();
    let app = Router::new()
        .route("/api/v1/rft/external-runs", post(json_route))
        .route("/api/v1/rft/metrics", post(json_route))
        .route("/api/v1/rft/finalize", post(json_route))
        .route("/api/v1/rft/external-runs/{id}/status", put(status_route))
        .route("/api/v1/traces", post(traces_route))
        .with_state(log.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), log)
}

fn record(id: &str) -> Arc<Value> {
    let trace: Value = [("id", Value::from(format!("{id}-t"))), ("logprobs", Value::Array(vec![Value::F64(-0.5)]))].into_iter().collect();
    Arc::new([("id", Value::from(id)), ("traces", Value::Array(vec![trace]))].into_iter().collect())
}

fn setup(output_dir: &std::path::Path) -> MonitorsSetup {
    MonitorsSetup {
        output_dir: output_dir.to_path_buf(),
        producer: Some("orch".into()),
        prime: Some(PrimeSetup::Train {
            name: Some("my-run".into()),
            run: Some(PrimeTrainSetup {
                name: Some("my-run".into()),
                model: "Qwen/Qwen3-0.6B".into(),
                environments: vec!["gsm8k".into()],
                training: TrainingSpec { max_steps: 10, batch_size: Some(8), rollouts_per_example: Some(4), seq_len: Some(2048), wandb_project: None },
                run_config: json!({"max_steps": 10}).as_object().unwrap().clone(),
            }),
        }),
        wandb: None,
        file: None,
        train_env_names: vec!["gsm8k".into()],
        eval_env_names: vec![],
        overview_flavor: "rl".into(),
    }
}

fn paths(log: &Log) -> Vec<String> {
    log.lock().unwrap().iter().map(|r| r.path.clone()).collect()
}

async fn start(dir: &std::path::Path) -> (Monitors, tokio::task::JoinHandle<anyhow::Result<()>>, CancellationToken) {
    let cancel = CancellationToken::new();
    let (monitors, task) = prl_monitors::spawn(setup(dir), cancel.clone()).await.unwrap();
    (monitors, task, cancel)
}

#[tokio::test]
async fn prime_train_monitor_against_mock_platform() {
    let (url, log) = serve().await;
    let home = tempfile::tempdir().unwrap();
    // SAFETY: the only test in this binary; nothing else reads the environment concurrently.
    unsafe {
        for var in ["RUN_ID", "PRIME_RUNS_MODE", "PRIME_API_BASE", "PRIME_TRACES_OPT_OUT", "RANK", "DP_RANK", "PRL_ATTEMPT_LOG_DIR"] {
            std::env::remove_var(var);
        }
        std::env::set_var("HOME", home.path());
        std::env::set_var("PRIME_API_KEY", "key");
        std::env::set_var("PRIME_TEAM_ID", "team");
        std::env::set_var("PRIME_API_BASE_URL", &url);
        std::env::set_var("PRIME_TRACES_URL", &url);
        std::env::set_var("PRIME_FRONTEND_URL", "https://app.test");
    }

    // A clean run: register, log, finalize.
    let out = tempfile::tempdir().unwrap();
    let (monitors, task, _cancel) = start(out.path()).await;
    let mut row = Row::new();
    row.insert("loss/mean".into(), Scalar::Float(0.5));
    row.insert("bad".into(), Scalar::Float(f64::NAN));
    monitors.log_metrics(row, Some(3)).await;
    monitors.log_episodes(vec![record("e1"), record("e2")], 3, Kind::Train, Subset::All).await;
    monitors.log_episodes(vec![record("e3")], 3, Kind::Train, Subset::Effective).await;
    monitors.finalize(false).await;
    task.await.unwrap().unwrap();

    let requests = log.lock().unwrap().clone();
    assert_eq!(paths(&log), ["/api/v1/rft/external-runs", "/api/v1/rft/metrics", "/api/v1/traces", "/api/v1/rft/finalize"]);
    let create = &requests[0];
    assert_eq!(create.headers["authorization"], "Bearer key");
    assert_eq!(
        create.body,
        json!({"base_model": "Qwen/Qwen3-0.6B", "max_steps": 10, "environments": [{"id": "gsm8k"}], "team_id": "team",
               "name": "my-run", "batch_size": 8, "rollouts_per_example": 4, "seq_len": 2048, "run_config": {"max_steps": 10}})
    );
    let metrics = &requests[1].body;
    assert_eq!(metrics["run_id"], "r1");
    assert_eq!(metrics["metrics"]["loss/mean"], 0.5);
    assert_eq!(metrics["metrics"]["step"], 3);
    assert!(metrics["metrics"]["_timestamp"].is_f64());
    assert!(metrics["metrics"].get("bad").is_none());

    let upload = &requests[2];
    let text = upload.body["traces"]["text"].as_str().unwrap();
    assert_eq!(upload.headers["idempotency-key"], format!("sha256:{}", hex::encode(Sha256::digest(text.as_bytes()))));
    assert_eq!(upload.headers["x-prime-line-format"], "episode");
    assert_eq!(upload.headers["x-prime-team-id"], "team");
    assert_eq!(upload.body["traces"]["encoding"], "gzip");
    assert_eq!(upload.body["traces"]["file_name"], "traces.jsonl");
    assert_eq!(upload.body["traces"]["content_type"], "application/x-ndjson");
    assert_eq!(
        upload.body["metadata"]["text"],
        r#"{"schema_version": 1, "context": {"source": "prime-runs", "run_type": "train", "model": "Qwen/Qwen3-0.6B"}}"#
    );
    assert_eq!(
        text,
        concat!(
            r#"{"id":"e1","traces":[{"id":"e1-t","logprobs":[-0.5],"run":{"id":"r1","type":"train"}}],"run":{"id":"r1","type":"train"}}"#,
            "\n",
            r#"{"id":"e2","traces":[{"id":"e2-t","logprobs":[-0.5],"run":{"id":"r1","type":"train"}}],"run":{"id":"r1","type":"train"}}"#,
            "\n"
        )
    );
    assert_eq!(requests[3].body, json!({"run_id": "r1", "exit_code": 0}));

    let run_record: Json_ = serde_json::from_slice(&std::fs::read(out.path().join("monitors/prime/run.json")).unwrap()).unwrap();
    assert_eq!(run_record, json!({"kind": "train", "id": "r1", "url": "https://app.test/dashboard/training/r1"}));
    let links: Json_ = serde_json::from_slice(&std::fs::read(out.path().join("monitors/links.json")).unwrap()).unwrap();
    assert_eq!(links["prime"]["links"]["run"], "https://app.test/dashboard/training/r1");
    assert_eq!(links["prime"]["expected"], 1);

    // A run cancelled before finalize reports crashed; logs handed over before still go out.
    log.lock().unwrap().clear();
    let out = tempfile::tempdir().unwrap();
    let (monitors, task, cancel) = start(out.path()).await;
    monitors.log_metrics(Row::new(), Some(1)).await;
    cancel.cancel();
    task.await.unwrap().unwrap();
    assert_eq!(paths(&log), ["/api/v1/rft/external-runs", "/api/v1/rft/metrics", "status/r1"]);
    assert_eq!(
        log.lock().unwrap()[2].body,
        json!({"status": "failed", "error_message": "crashed: process exited without finishing the run"})
    );

    // An attached run ($RUN_ID) registers nothing and leaves a crash to the launcher.
    log.lock().unwrap().clear();
    unsafe { std::env::set_var("RUN_ID", "launched") };
    let (monitors, task, cancel) = start(out.path()).await;
    monitors.log_metrics(Row::new(), None).await;
    drop(monitors);
    cancel.cancel();
    task.await.unwrap().unwrap();
    assert_eq!(paths(&log), ["/api/v1/rft/metrics"]);
    assert_eq!(log.lock().unwrap()[0].body["run_id"], "launched");
}
