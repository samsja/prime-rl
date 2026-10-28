//! Live checks against a running prime-rl inference server (bare engine). Skipped unless
//! `PRL_IO_VLLM_URL` (e.g. `http://localhost:8731`) and `PRL_IO_VLLM_MODEL` are set;
//! `PRL_IO_VLLM_WEIGHTS` (an HF checkpoint dir of the served model) also exercises `/update_weights`.

use std::path::Path;
use std::time::Duration;

use indexmap::IndexMap;
use prl_config::{ClientConfig, ClientTimeoutConfig};
use prl_io::clients::{AdminPlane, ClientType, InferenceClient, Transport};
use prl_io::inference_metrics::InferenceMetricsCollector;

#[tokio::test]
async fn live_inference_server() {
    let (Ok(url), Ok(model)) = (std::env::var("PRL_IO_VLLM_URL"), std::env::var("PRL_IO_VLLM_MODEL")) else {
        eprintln!("skipped: set PRL_IO_VLLM_URL and PRL_IO_VLLM_MODEL");
        return;
    };
    let config = ClientConfig {
        base_url: format!("{url}/v1"),
        api_key_var: "VLLM_API_KEY".into(),
        headers: IndexMap::new(),
        headers_from_env: IndexMap::new(),
        timeout: ClientTimeoutConfig { connect: Some(30.0), read: None, write: None, pool: None },
        wait_for_ready_timeout: 600,
        skip_model_check: false,
        admin_base_url: None,
    };
    let mut plane = AdminPlane::standard(&config, None).unwrap();
    plane.wait_for_ready(&model).await.unwrap();

    let (load_tx, mut load_rx) = tokio::sync::mpsc::channel(4);
    let (metrics_tx, mut metrics_rx) = tokio::sync::mpsc::channel(4);
    let mut collector = InferenceMetricsCollector::new(plane.clients(), None, true, Some(load_tx), Some(metrics_tx)).unwrap();
    assert!(collector.probe(3, Duration::from_secs(2)).await);
    let load = load_rx.recv().await.unwrap();
    eprintln!("load: {load:?}");
    assert!(load[0].kv_capacity_tokens.is_some_and(|t| t > 0));
    assert!(load[0].max_model_len.is_some());
    let metrics = metrics_rx.recv().await.unwrap();
    eprintln!("{} metrics, e.g. {:?}", metrics.len(), metrics.iter().take(3).collect::<Vec<_>>());

    let client = InferenceClient::new(&config, &model, ClientType::Renderer, ClientType::ChatCompletions, None).unwrap();
    let logprobs = client.score(&[785, 6722, 315, 9625, 374]).await.unwrap();
    eprintln!("prefill logprobs: {logprobs:?}");
    assert_eq!(logprobs.len(), 5);
    assert_eq!(logprobs[0], 0.0);
    assert!(logprobs[1..].iter().all(|&lp| lp < 0.0));

    let weights = std::env::var("PRL_IO_VLLM_WEIGHTS").ok();
    let times = plane.update_weights(weights.as_deref().map(Path::new), Transport::Filesystem, 1, || Ok(())).await;
    match weights {
        Some(_) => eprintln!("update times: {:?}", times.unwrap()),
        None => eprintln!("update without weights: {times:?}"),
    }

    // Engines serve again after the update.
    let second = collector.collect_and_log().await;
    assert!(second.is_ok());
    assert_eq!(client.score(&[785, 6722, 315]).await.unwrap().len(), 3);
}
