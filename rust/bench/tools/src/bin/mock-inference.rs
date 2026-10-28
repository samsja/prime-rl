//! A fast fake vLLM for orchestrator benchmarks: the token-in/out `/inference/v1/generate`
//! endpoint (prime-rl's train client), the admin routes the orchestrator calls around weight
//! updates, `/v1/models`, `/health` and a minimal `/metrics`. Each generate answers with one
//! of the canned completions (token ids ending in a stop token), chosen at random so rewards
//! vary, after an optional fixed latency.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use clap::Parser;
use rand::Rng;
use serde_json::{json, Value};

#[derive(Parser)]
struct Args {
    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Model name served at /v1/models.
    #[arg(long)]
    model: String,
    /// JSON file: a list of completions, each a list of token ids (ending in a stop token).
    #[arg(long)]
    completions: std::path::PathBuf,
    /// Fixed latency per generate request, in milliseconds.
    #[arg(long, default_value_t = 0)]
    latency_ms: u64,
    /// max_model_len reported at /v1/models.
    #[arg(long, default_value_t = 32768)]
    max_model_len: u64,
    /// Tokio worker threads.
    #[arg(long, default_value_t = 4)]
    threads: usize,
}

struct App {
    model: String,
    max_model_len: u64,
    completions: Vec<(Vec<i64>, String)>,
    latency: Duration,
    generated: AtomicU64,
    running: AtomicU64,
    weight_version: AtomicU64,
}

async fn generate(State(app): State<Arc<App>>, Json(body): Json<Value>) -> impl IntoResponse {
    app.running.fetch_add(1, Ordering::Relaxed);
    if !app.latency.is_zero() {
        tokio::time::sleep(app.latency).await;
    }
    let prompt_len = body.get("token_ids").and_then(Value::as_array).map_or(0, Vec::len);
    let (ids, logprobs) = &app.completions[rand::thread_rng().gen_range(0..app.completions.len())];
    let n = app.generated.fetch_add(1, Ordering::Relaxed);
    app.running.fetch_sub(1, Ordering::Relaxed);
    let version = app.weight_version.load(Ordering::Relaxed).to_string();
    Json(json!({
        "request_id": format!("gen-{n}"),
        "usage": {"prompt_tokens": prompt_len, "completion_tokens": ids.len(), "total_tokens": prompt_len + ids.len()},
        "choices": [{
            "index": 0,
            "token_ids": ids,
            "completion_logprobs": {"data": logprobs, "shape": [ids.len()], "dtype": "float32"},
            "finish_reason": "stop",
            "weight_versions": [{"version": version, "start": 0, "end": ids.len()}],
        }],
    }))
}

async fn models(State(app): State<Arc<App>>) -> impl IntoResponse {
    Json(json!({"object": "list", "data": [{"id": app.model, "object": "model", "root": app.model, "max_model_len": app.max_model_len}]}))
}

async fn update_weights(State(app): State<Arc<App>>, Json(body): Json<Value>) -> impl IntoResponse {
    if let Some(version) = body.get("weight_version").and_then(Value::as_u64) {
        app.weight_version.store(version, Ordering::Relaxed);
    }
    Json(json!({}))
}

async fn ok() -> impl IntoResponse {
    Json(json!({}))
}

async fn metrics(State(app): State<Arc<App>>) -> impl IntoResponse {
    let model = &app.model;
    format!(
        "# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{{model_name=\"{model}\"}} {}\n\
         # TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting{{model_name=\"{model}\"}} 0.0\n\
         # TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc{{model_name=\"{model}\"}} 0.0\n\
         # TYPE mock:generated_total counter\nmock:generated_total {}\n",
        app.running.load(Ordering::Relaxed),
        app.generated.load(Ordering::Relaxed),
    )
}

/// Generated completions so far (for measuring the mock's own throughput).
async fn stats(State(app): State<Arc<App>>) -> impl IntoResponse {
    Json(json!({"generated": app.generated.load(Ordering::Relaxed)}))
}

fn main() {
    let args = Args::parse();
    let completions: Vec<Vec<i64>> = serde_json::from_slice(&std::fs::read(&args.completions).unwrap()).unwrap();
    let completions = completions
        .into_iter()
        .map(|ids| {
            let logprobs: Vec<u8> = (0..ids.len()).flat_map(|i| (-0.1f32 - 0.01 * i as f32).to_le_bytes()).collect();
            (ids, base64::engine::general_purpose::STANDARD.encode(logprobs))
        })
        .collect();
    let app = Arc::new(App {
        model: args.model,
        max_model_len: args.max_model_len,
        completions,
        latency: Duration::from_millis(args.latency_ms),
        generated: AtomicU64::new(0),
        running: AtomicU64::new(0),
        weight_version: AtomicU64::new(0),
    });
    let router = Router::new()
        .route("/inference/v1/generate", post(generate))
        .route("/v1/models", get(models))
        .route("/health", get(ok))
        .route("/metrics", get(metrics))
        .route("/stats", get(stats))
        .route("/update_weights", post(update_weights))
        .route("/pause", post(ok))
        .route("/resume", post(ok))
        .route("/finish_session", post(ok))
        .with_state(app);
    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(args.threads).enable_all().build().unwrap();
    runtime.block_on(async {
        let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        eprintln!("mock inference listening on http://{addr}");
        axum::serve(listener, router).await.unwrap();
    });
}
