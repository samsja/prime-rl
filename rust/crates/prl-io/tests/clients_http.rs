//! Admin plane and weight receivers against a mock vLLM engine.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{OriginalUri, State};
use axum::http::StatusCode;
use indexmap::IndexMap;
use prl_config::{
    ClientConfig, ClientTimeoutConfig, FileSystemWeightBroadcastConfig, InferenceRole, NIXLWeightBroadcastConfig, WeightBroadcastConfig,
};
use prl_io::clients::{AdminPlane, Transport};
use prl_io::weights::{NixlBridge, WeightReceiver};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
struct Call {
    engine: String,
    path: String,
    query: Option<String>,
    body: Value,
}

/// Records requests; `responses[path]` queues statuses, the last one repeats (default 200).
#[derive(Clone)]
struct Mock {
    engine: String,
    calls: Arc<Mutex<Vec<Call>>>,
    responses: Arc<Mutex<HashMap<String, Vec<u16>>>>,
    delay: Duration,
}

async fn handle(State(mock): State<Mock>, OriginalUri(uri): OriginalUri, body: Bytes) -> StatusCode {
    let call = Call {
        engine: mock.engine.clone(),
        path: uri.path().to_owned(),
        query: uri.query().map(str::to_owned),
        body: serde_json::from_slice(&body).unwrap_or(Value::Null),
    };
    mock.calls.lock().unwrap().push(call);
    tokio::time::sleep(mock.delay).await;
    let mut responses = mock.responses.lock().unwrap();
    let status = match responses.get_mut(uri.path()) {
        Some(queue) if queue.len() > 1 => queue.remove(0),
        Some(queue) => queue[0],
        None => 200,
    };
    StatusCode::from_u16(status).unwrap()
}

impl Mock {
    async fn serve(engine: &str, delay: Duration, calls: Arc<Mutex<Vec<Call>>>) -> (Self, String) {
        let mock = Mock { engine: engine.into(), calls, responses: Arc::default(), delay };
        let app = axum::Router::new().fallback(handle).with_state(mock.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (mock, url)
    }

    fn respond(&self, path: &str, statuses: Vec<u16>) {
        self.responses.lock().unwrap().insert(path.into(), statuses);
    }

    fn paths(&self) -> Vec<String> {
        self.calls.lock().unwrap().iter().map(|c| c.path.clone()).collect()
    }
}

fn client_config(urls: &[&str]) -> ClientConfig {
    ClientConfig {
        base_url: urls[0].into(),
        api_key_var: "PRL_IO_TEST_UNSET_KEY".into(),
        headers: IndexMap::new(),
        headers_from_env: IndexMap::new(),
        timeout: ClientTimeoutConfig { connect: Some(30.0), read: None, write: None, pool: None },
        wait_for_ready_timeout: 5,
        skip_model_check: false,
        admin_base_url: (urls.len() > 1).then(|| urls.iter().map(|u| u.to_string()).collect()),
    }
}

#[tokio::test]
async fn update_weights_pauses_prefill_first_retries_5xx_and_always_resumes() {
    let calls: Arc<Mutex<Vec<Call>>> = Arc::default();
    let (decode, decode_url) = Mock::serve("decode", Duration::ZERO, calls.clone()).await;
    let (_prefill, prefill_url) = Mock::serve("prefill", Duration::from_millis(20), calls.clone()).await;
    let roles = vec![InferenceRole::Decode, InferenceRole::Prefill];
    let mut plane = AdminPlane::standard(&client_config(&[&decode_url, &prefill_url]), Some(roles)).unwrap();
    decode.respond("/resume", vec![503, 200]);

    let times = plane.update_weights(Some(Path::new("/run/broadcasts/step_2")), Transport::Filesystem, 2, || Ok(())).await.unwrap();
    assert_eq!((times.pause.len(), times.transfer.len(), times.resume.len()), (2, 2, 2));
    let order: Vec<(String, String)> = calls.lock().unwrap().iter().map(|c| (c.engine.clone(), c.path.clone())).collect();
    // The prefill engine is paused (and finishes, 20 ms) before the decode engine.
    assert_eq!(order[..2], [("prefill".into(), "/pause".into()), ("decode".into(), "/pause".into())]);
    let first = calls.lock().unwrap()[0].clone();
    assert_eq!(first.query.as_deref(), Some("mode=keep&clear_cache=false"));
    let update = calls.lock().unwrap().iter().find(|c| c.path == "/update_weights").unwrap().clone();
    assert_eq!(update.body, json!({"weight_dir": "/run/broadcasts/step_2", "weight_version": 2}));
    let decode_resumes = calls.lock().unwrap().iter().filter(|c| c.engine == "decode" && c.path == "/resume").count();
    assert_eq!(decode_resumes, 2);

    // A 4xx transfer error is not retried, and the engines are still resumed.
    calls.lock().unwrap().clear();
    decode.respond("/update_weights", vec![400]);
    assert!(plane.update_weights(None, Transport::Nixl, 3, || Ok(())).await.is_err());
    let decode_calls: Vec<String> = calls.lock().unwrap().iter().filter(|c| c.engine == "decode").map(|c| c.path.clone()).collect();
    assert_eq!(decode_calls, ["/pause", "/update_weights", "/resume"]);
}

#[tokio::test]
async fn filesystem_receiver_handshake_full_update_and_lora_retry() {
    let (mock, url) = Mock::serve("engine", Duration::ZERO, Arc::default()).await;
    let dir = tempfile::tempdir().unwrap();
    let plane = AdminPlane::standard(&client_config(&[&url]), None).unwrap();
    let config = WeightBroadcastConfig::Filesystem(FileSystemWeightBroadcastConfig { timeout: 10 });
    let broadcasts = dir.path().join("broadcasts");
    let mut receiver = WeightReceiver::new(broadcasts.clone(), config, plane, "m", None).unwrap();

    let offer = |step: u64, adapter: bool| {
        let step_dir = broadcasts.join(format!("step_{step}"));
        std::fs::create_dir_all(&step_dir).unwrap();
        if adapter {
            std::fs::write(step_dir.join("adapter_config.json"), b"{}").unwrap();
        }
        std::fs::write(step_dir.join(".sender_ready"), b"").unwrap();
        // The trainer writes `.finished` once the receiver acknowledged.
        tokio::spawn(async move {
            while !step_dir.join(".receiver_ready").exists() {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            std::fs::write(step_dir.join(".finished"), b"").unwrap();
        })
    };

    assert_eq!(receiver.next_version(0).unwrap(), 0);
    let trainer = offer(1, false);
    std::fs::create_dir_all(receiver.step_dir(2)).unwrap(); // a dir without `.sender_ready` is not offered
    assert_eq!(receiver.next_version(0).unwrap(), 1);
    receiver.sync_startup(1, Duration::from_secs(5)).await.unwrap();
    trainer.await.unwrap();
    assert_eq!(mock.paths(), ["/pause", "/update_weights", "/resume"]);

    // An adapter is hot-swapped without pausing; 404 (files not yet visible) is retried.
    mock.calls.lock().unwrap().clear();
    mock.respond("/load_lora_adapter", vec![404, 200]);
    let trainer = offer(3, true);
    receiver.receive(3).await.unwrap();
    trainer.await.unwrap();
    let calls = mock.calls.lock().unwrap().clone();
    assert_eq!(calls.iter().map(|c| c.path.as_str()).collect::<Vec<_>>(), ["/load_lora_adapter", "/load_lora_adapter"]);
    assert_eq!(calls[1].query.as_deref(), Some("weight_version=3"));
    assert_eq!(calls[1].body, json!({"lora_name": "m", "lora_path": receiver.step_dir(3).to_string_lossy()}));

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
        self.calls.lock().unwrap().push(Call { engine: "nixl".into(), path: name, query: None, body: Value::Null });
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
    let config = client_config(&[&url, &url]);
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

    let sequence = mock.paths();
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
    let requests = mock.calls.lock().unwrap().clone();
    let mut offsets: Vec<Value> = requests[..2].iter().map(|c| c.body["rank_offset"].clone()).collect();
    offsets.sort_by_key(|v| v.as_u64());
    assert_eq!(offsets, vec![json!(0), json!(2)]);
    assert_eq!(
        requests[0].body,
        json!({"host": "mx", "port": 8001, "rank_offset": requests[0].body["rank_offset"], "inference_world_size": 4, "timeout": 600, "session_id": "s1"})
    );
    assert_eq!(requests[6].body, json!({"weight_dir": null, "weight_version": 5}));
}
