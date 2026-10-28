//! Against a real `uv run env-server` (see `tests/e2e/run_env_server_e2e.py`, which starts
//! the server and a mock OpenAI endpoint, runs this test and checks what it wrote).

use std::time::Duration;

use prl_wire::{value, EnvClient, Map, Value};

fn read_json(path: &std::path::Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a running env server: tests/e2e/run_env_server_e2e.py"]
async fn runs_against_a_real_env_server() {
    let dir = std::path::PathBuf::from(std::env::var("PRL_WIRE_E2E_DIR").expect("PRL_WIRE_E2E_DIR"));
    let address = std::fs::read_to_string(dir.join("addr")).unwrap();
    let tasks = read_json(&dir.join("tasks.json"));
    let client_config = read_json(&dir.join("client.json"));
    let sampling = read_json(&dir.join("sampling.json"));
    let model = std::fs::read_to_string(dir.join("model")).unwrap();
    let train = client_config.get("type").and_then(Value::as_str) == Some("train");
    let client = EnvClient::connect(address.trim()).unwrap();
    client.wait_for_server_startup(Duration::from_secs(120)).await.unwrap();

    let mut runs = Vec::new();
    for task in tasks.as_array().unwrap().clone() {
        let (client, client_config, sampling, model) = (client.clone(), client_config.clone(), sampling.clone(), model.clone());
        runs.push(tokio::spawn(async move {
            let data: Map = task.get("data").unwrap().as_map().unwrap().clone();
            let deltas = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counter = deltas.clone();
            let episode = client
                .run(&data, &client_config, &model, &sampling, Some(Box::new(move |_: &Value| {
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                })))
                .await
                .unwrap();
            assert!(deltas.load(std::sync::atomic::Ordering::Relaxed) > 0);
            assert_eq!(episode.task.key.as_deref(), task.get("key").unwrap().as_str());
            assert_eq!(episode.task.hash.as_deref(), task.get("hash").unwrap().as_str());
            assert_eq!(episode.task.content_hash(), task.get("hash").unwrap().as_str().unwrap());
            episode
        }));
    }
    let mut payload_urls = Vec::new();
    for (i, run) in runs.into_iter().enumerate() {
        let episode = run.await.unwrap();
        assert!(episode.ok, "{:?}", episode.errors);
        assert_eq!(episode.traces.len(), 1);
        let trace = &episode.traces[0];
        assert!(trace.num_turns() >= 1);
        if train {
            // token-in/out: every branch carries its tokens, sampled logprobs and kept arrays
            let branch = &trace.branches()[0];
            assert!(!branch.token_ids().is_empty());
            assert_eq!(branch.logprobs().len(), branch.token_ids().len());
            assert!(branch.sampled_mask().iter().any(|&m| m));
            let urls = prl_wire::payload::trace_payload_urls(trace);
            assert!(!urls.is_empty(), "the payload hook kept no arrays: {:?}", trace.info);
            payload_urls.extend(urls);
        }
        std::fs::write(dir.join(format!("episode_{i}.msgpack")), value::to_msgpack(&episode.to_wire())).unwrap();
        std::fs::write(dir.join(format!("episode_{i}.json")), episode.to_json(Some(4))).unwrap();
    }

    for url in &payload_urls {
        assert_eq!(reqwest::get(url).await.unwrap().status(), 200, "{url}");
    }
    prl_wire::payload::delete_payloads(payload_urls.clone());
    prl_wire::payload::wait_for_deletes();
    for url in &payload_urls {
        assert_eq!(reqwest::get(url).await.unwrap().status(), 404, "{url}");
    }

    // a dropped run is cancelled server-side; the client keeps serving
    let data = tasks.as_array().unwrap()[0].get("data").unwrap().as_map().unwrap().clone();
    let dropped = tokio::time::timeout(Duration::from_millis(1), client.run(&data, &client_config, &model, &sampling, None)).await;
    assert!(dropped.is_err());
    assert!(client.health(Duration::from_secs(5)).await.unwrap());
    client.close().await;
}
