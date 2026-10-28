//! HTTP behavior of the inference clients, admin plane and weight receiver against a mock server
//! (ports `tests/unit/orchestrator/test_clients.py` and `test_prefill_logprobs.py`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{OriginalUri, State};
use axum::http::{Method, StatusCode};
use axum::response::IntoResponse;
use axum::Router;
use indexmap::IndexMap;
use prl_config::{
    ClientConfig, ClientTimeoutConfig, FileSystemWeightBroadcastConfig, InferenceRole, NIXLWeightBroadcastConfig, WeightBroadcastConfig,
};
use prl_io::clients::{
    check_health, load_lora_adapter, maybe_check_has_model, pause_engines, setup_client, AdminPlane, ClientType, InferenceClient,
};
use prl_io::http::{admin_post, AdminClient, AdminRequest};
use prl_io::weights::{NixlBridge, WeightReceiver};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
struct Call {
    name: String,
    method: Method,
    path: String,
    query: Option<String>,
    body: Value,
}

type Responses = HashMap<String, Vec<(u16, Value)>>;

#[derive(Clone, Default)]
struct Mock {
    name: String,
    calls: Arc<Mutex<Vec<Call>>>,
    /// path -> queued (status, body) responses; the last one repeats.
    responses: Arc<Mutex<Responses>>,
    delay: Duration,
}

async fn handle(State(mock): State<Mock>, method: Method, OriginalUri(uri): OriginalUri, body: Bytes) -> impl IntoResponse {
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let path = uri.path().to_owned();
    mock.calls.lock().unwrap().push(Call {
        name: format!("{} start", mock.name),
        method: method.clone(),
        path: path.clone(),
        query: uri.query().map(str::to_owned),
        body: body.clone(),
    });
    tokio::time::sleep(mock.delay).await;
    let (status, response) = {
        let mut responses = mock.responses.lock().unwrap();
        match responses.get_mut(&path) {
            Some(queue) if queue.len() > 1 => queue.remove(0),
            Some(queue) => queue[0].clone(),
            None => (200, json!({})),
        }
    };
    mock.calls.lock().unwrap().push(Call { name: format!("{} done", mock.name), method, path, query: None, body: Value::Null });
    (StatusCode::from_u16(status).unwrap(), axum::Json(response))
}

impl Mock {
    async fn serve(name: &str, delay: Duration, calls: Arc<Mutex<Vec<Call>>>) -> (Self, String) {
        let mock = Mock { name: name.into(), calls, responses: Arc::default(), delay };
        let app = Router::new().fallback(handle).with_state(mock.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (mock, url)
    }

    fn respond(&self, path: &str, responses: Vec<(u16, Value)>) {
        self.responses.lock().unwrap().insert(path.into(), responses);
    }

    fn requests(&self) -> Vec<Call> {
        self.calls.lock().unwrap().iter().filter(|c| c.name.ends_with("start")).cloned().collect()
    }
}

fn client_config(base_url: &str) -> ClientConfig {
    ClientConfig {
        base_url: base_url.into(),
        api_key_var: "PRL_IO_TEST_UNSET_KEY".into(),
        headers: IndexMap::new(),
        headers_from_env: IndexMap::new(),
        timeout: ClientTimeoutConfig { connect: Some(30.0), read: None, write: None, pool: None },
        wait_for_ready_timeout: 5,
        skip_model_check: false,
        admin_base_url: None,
    }
}

fn admin(url: &str) -> AdminClient {
    AdminClient::new(&format!("{url}/v1"), Default::default()).unwrap()
}

#[tokio::test]
async fn load_lora_adapter_request_shape_and_retry_on_404() {
    let (mock, url) = Mock::serve("engine", Duration::ZERO, Arc::default()).await;
    mock.respond("/load_lora_adapter", vec![(404, json!({})), (200, json!({}))]);
    load_lora_adapter(&[admin(&url)], "test-lora", Path::new("/test/path"), 3).await.unwrap();
    let calls = mock.requests();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].method, Method::POST);
    assert_eq!(calls[1].path, "/load_lora_adapter");
    assert_eq!(calls[1].query.as_deref(), Some("weight_version=3"));
    assert_eq!(calls[1].body, json!({"lora_name": "test-lora", "lora_path": "/test/path"}));
}

#[tokio::test]
async fn admin_post_retries_5xx_but_not_4xx() {
    let (mock, url) = Mock::serve("engine", Duration::ZERO, Arc::default()).await;
    mock.respond("/resume", vec![(503, json!({})), (200, json!({}))]);
    admin_post(&admin(&url), "/resume", &AdminRequest::default(), 5.0).await.unwrap();
    assert_eq!(mock.requests().len(), 2);

    mock.respond("/pause", vec![(400, json!({}))]);
    let error = admin_post(&admin(&url), "/pause", &AdminRequest::default(), 5.0).await.unwrap_err();
    assert_eq!(error.status().map(|s| s.as_u16()), Some(400));
    assert_eq!(mock.requests().iter().filter(|c| c.path == "/pause").count(), 1);
}

#[tokio::test]
async fn admin_plane_initializes_nccl() {
    let (mock, url) = Mock::serve("engine", Duration::ZERO, Arc::default()).await;
    let mut plane = AdminPlane::standard(&client_config(&format!("{url}/v1")), None).unwrap();
    plane.initialize_nccl("trainer", 29501, 1200, 1).await.unwrap();
    let calls = mock.requests();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].path, "/init_broadcaster");
    assert_eq!(calls[0].body, json!({"host": "trainer", "port": 29501, "rank_offset": 0, "inference_world_size": 1, "timeout": 1200}));
}

#[tokio::test]
async fn pause_engines_pauses_prefill_before_decode() {
    let events: Arc<Mutex<Vec<Call>>> = Arc::default();
    let (_decode, decode_url) = Mock::serve("decode", Duration::ZERO, events.clone()).await;
    let (_prefill, prefill_url) = Mock::serve("prefill", Duration::from_millis(20), events.clone()).await;
    let clients = [admin(&decode_url), admin(&prefill_url)];
    pause_engines(&clients, 1, Some(&[InferenceRole::Decode, InferenceRole::Prefill])).await.unwrap();
    let names: Vec<String> = events.lock().unwrap().iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, vec!["prefill start", "prefill done", "decode start", "decode done"]);
    let calls = events.lock().unwrap().clone();
    assert_eq!(calls[0].query.as_deref(), Some("mode=keep&clear_cache=false"));
}

#[tokio::test]
async fn check_health_retries_non_success_status() {
    let (mock, url) = Mock::serve("engine", Duration::ZERO, Arc::default()).await;
    mock.respond("/health", vec![(503, json!({})), (200, json!({}))]);
    check_health(&[admin(&url)], 1, 30, 2, false).await.unwrap();
    assert_eq!(mock.requests().len(), 2);

    mock.respond("/health", vec![(503, json!({}))]);
    let error = check_health(&[admin(&url)], 1, 30, 1, false).await.unwrap_err();
    assert_eq!(error.to_string(), "Inference server is not ready after 1 (>1) seconds. Aborting...");
}

#[tokio::test]
async fn model_check_accepts_lora_root_and_rejects_missing_models() {
    let (mock, url) = Mock::serve("engine", Duration::ZERO, Arc::default()).await;
    mock.respond("/v1/models", vec![(200, json!({"data": [{"id": "m-base", "root": "m"}]}))]);
    maybe_check_has_model(&[admin(&url)], "m", false).await.unwrap();
    let error = maybe_check_has_model(&[admin(&url)], "other", false).await.unwrap_err();
    assert_eq!(error.to_string(), format!("Model other was not found in the inference pool on {url}"));
    mock.respond("/v1/models", vec![(500, json!({"error": "boom"}))]);
    let error = maybe_check_has_model(&[admin(&url)], "m", false).await.unwrap_err();
    assert!(error.to_string().starts_with(&format!("Listing the models of {url} failed with status 500")));
}

#[tokio::test]
async fn update_weights_pauses_transfers_and_resumes() {
    let (mock, url) = Mock::serve("engine", Duration::ZERO, Arc::default()).await;
    let mut plane = AdminPlane::standard(&client_config(&format!("{url}/v1")), None).unwrap();
    let paused = Arc::new(Mutex::new(false));
    let flag = paused.clone();
    let times = plane
        .update_weights(Some(Path::new("/run/broadcasts/step_2")), prl_io::clients::Transport::Filesystem, 2, move || {
            *flag.lock().unwrap() = true;
            Ok(())
        })
        .await
        .unwrap();
    assert!(*paused.lock().unwrap());
    assert_eq!((times.pause.len(), times.transfer.len(), times.resume.len()), (1, 1, 1));
    let paths: Vec<String> = mock.requests().into_iter().map(|c| c.path).collect();
    assert_eq!(paths, vec!["/pause", "/update_weights", "/resume"]);
    assert_eq!(mock.requests()[1].body, json!({"weight_dir": "/run/broadcasts/step_2", "weight_version": 2}));

    // A failed transfer still resumes the engines.
    mock.respond("/update_weights", vec![(400, json!({}))]);
    assert!(plane.update_weights(None, prl_io::clients::Transport::Nixl, 3, || Ok(())).await.is_err());
    let paths: Vec<String> = mock.requests().into_iter().map(|c| c.path).collect();
    assert_eq!(paths[3..], ["/pause", "/update_weights", "/resume"]);
}

#[tokio::test]
async fn prefill_logprobs_uses_inference_generate() {
    let (mock, url) = Mock::serve("engine", Duration::ZERO, Arc::default()).await;
    mock.respond(
        "/inference/v1/generate",
        vec![(200, json!({"request_id": "gen-test", "choices": [], "kv_transfer_params": null,
                          "prompt_logprobs": [null, {"11": {"logprob": -0.7}}, {"12": {"logprob": -0.3}, "5": {"logprob": -9.0}}]}))],
    );
    let client = InferenceClient::new(&client_config(&format!("{url}/v1")), "ref-model", ClientType::ChatCompletions, ClientType::ChatCompletions, None).unwrap();
    assert_eq!(client.score(&[1, 2, 3]).await.unwrap(), vec![0.0, -0.7, -0.3]);
    let calls = mock.requests();
    assert_eq!(calls[0].path, "/inference/v1/generate");
    assert_eq!(
        calls[0].body,
        json!({"model": "ref-model", "token_ids": [1, 2, 3],
               "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "prompt_logprobs": 1}})
    );
}

#[test]
fn setup_client_matches_python_model_dump() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/client_configs.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut config = client_config("http://worker-a:8000/v1");
    config.api_key_var = "VLLM_API_KEY".into();
    config.headers.insert("X-Test".into(), "test".into());
    config.headers_from_env.insert("X-Token".into(), "PROVIDER_TOKEN".into());
    config.timeout.read = Some(12.0);
    let renderer = &fixture["renderer_config"];
    let cases = [
        ("renderer", setup_client(&config, ClientType::Renderer, Some(renderer), Some("Qwen/Qwen3-VL-4B-Instruct"))),
        ("renderer_auto", setup_client(&config, ClientType::Renderer, None, None)),
        ("chat", setup_client(&config, ClientType::ChatCompletions, None, None)),
    ];
    for (name, client) in cases {
        assert_eq!(serde_json::to_value(&client).unwrap(), fixture["cases"][name], "{name}");
    }
}

#[tokio::test]
async fn filesystem_receiver_acks_waits_for_finished_and_updates() {
    let (mock, url) = Mock::serve("engine", Duration::ZERO, Arc::default()).await;
    let dir = tempfile::tempdir().unwrap();
    let broadcasts = dir.path().join("broadcasts");
    let plane = AdminPlane::standard(&client_config(&format!("{url}/v1")), None).unwrap();
    let config = WeightBroadcastConfig::Filesystem(FileSystemWeightBroadcastConfig { timeout: 10 });
    let mut receiver = WeightReceiver::new(broadcasts.clone(), config, plane, "m", None).unwrap();

    assert_eq!(receiver.next_version(0).unwrap(), 0);
    let step_dir = receiver.step_dir(1);
    std::fs::create_dir_all(&step_dir).unwrap();
    std::fs::create_dir_all(receiver.step_dir(2)).unwrap(); // offered dirs without a marker don't count
    std::fs::write(step_dir.join(".sender_ready"), b"").unwrap();
    assert_eq!(receiver.next_version(0).unwrap(), 1);

    let marker_dir = step_dir.clone();
    let trainer = tokio::spawn(async move {
        while !marker_dir.join(".receiver_ready").exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        std::fs::write(marker_dir.join(".finished"), b"").unwrap();
    });
    let times = receiver.sync_startup(1, Duration::from_secs(5)).await.unwrap();
    trainer.await.unwrap();
    assert_eq!(times.transfer.len(), 1);
    let calls = mock.requests();
    assert_eq!(calls[1].body, json!({"weight_dir": step_dir.to_string_lossy(), "weight_version": 1}));

    let error = receiver.sync_startup(7, Duration::from_millis(200)).await.unwrap_err();
    assert!(error.to_string().contains("timed out"));
}

/// Records the NIXL bridge calls into the mock's call log, so their order relative to the HTTP
/// admin calls is visible.
struct FakeNixl {
    calls: Arc<Mutex<Vec<Call>>>,
    step_dir: std::path::PathBuf,
}

impl FakeNixl {
    fn record(&self, name: String) {
        self.calls.lock().unwrap().push(Call { name: format!("{name} start"), method: Method::GET, path: name, query: None, body: Value::Null });
    }
}

#[async_trait::async_trait]
impl NixlBridge for FakeNixl {
    async fn publish(&mut self) -> anyhow::Result<()> {
        self.record("nixl:publish".into());
        Ok(())
    }

    async fn wait_ready(&mut self, step: u64, timeout_s: u64) -> anyhow::Result<()> {
        // The trainer only sends `ready` after the receiver acknowledged.
        assert!(self.step_dir.join(".receiver_ready").exists());
        self.record(format!("nixl:wait_ready:{step}:{timeout_s}"));
        Ok(())
    }

    async fn send_complete(&mut self, step: u64) -> anyhow::Result<()> {
        self.record(format!("nixl:send_complete:{step}"));
        Ok(())
    }
}

#[tokio::test]
async fn nixl_receiver_initializes_and_receives_in_protocol_order() {
    let calls: Arc<Mutex<Vec<Call>>> = Arc::default();
    let (mock, url) = Mock::serve("engine", Duration::ZERO, calls.clone()).await;
    let dir = tempfile::tempdir().unwrap();
    let broadcasts = dir.path().join("broadcasts");
    let mut config = client_config(&format!("{url}/v1"));
    config.admin_base_url = Some(vec![format!("{url}/v1"), format!("{url}/v1")]);
    let plane = AdminPlane::standard(&config, None).unwrap();
    let nixl_config = WeightBroadcastConfig::Nixl(NIXLWeightBroadcastConfig {
        timeout: 600,
        host: "mx".into(),
        port: 8001,
        inference_world_size: 4,
        session_id: "s1".into(),
        overlap_transfer_and_replay: false,
    });
    assert!(WeightReceiver::new(broadcasts.clone(), nixl_config.clone(), AdminPlane::standard(&config, None).unwrap(), "m", None).is_err());
    let step_dir = broadcasts.join("step_5");
    let fake = FakeNixl { calls: calls.clone(), step_dir: step_dir.clone() };
    let mut receiver = WeightReceiver::new(broadcasts, nixl_config, plane, "m", Some(Box::new(fake))).unwrap();

    receiver.initialize().await.unwrap();
    std::fs::create_dir_all(&step_dir).unwrap();
    std::fs::write(step_dir.join(".sender_ready"), b"").unwrap();
    let times = receiver.receive(5).await.unwrap();
    assert_eq!(times.transfer.len(), 2);

    let sequence: Vec<String> = mock.requests().into_iter().map(|c| c.path).collect();
    assert_eq!(
        sequence,
        vec![
            "/init_broadcaster",
            "/init_broadcaster",
            "nixl:publish",
            "nixl:wait_ready:5:600",
            "/pause",
            "/pause",
            "/update_weights",
            "/update_weights",
            "/resume",
            "/resume",
            "nixl:send_complete:5",
        ]
    );
    let requests = mock.requests();
    let mut offsets: Vec<Value> = requests[..2].iter().map(|c| c.body["rank_offset"].clone()).collect();
    offsets.sort_by_key(|v| v.as_u64());
    assert_eq!(offsets, vec![json!(0), json!(2)]);
    assert_eq!(
        requests[0].body,
        json!({"host": "mx", "port": 8001, "rank_offset": requests[0].body["rank_offset"], "inference_world_size": 4, "timeout": 600, "session_id": "s1"})
    );
    assert_eq!(requests[6].body, json!({"weight_dir": null, "weight_version": 5}));
}
