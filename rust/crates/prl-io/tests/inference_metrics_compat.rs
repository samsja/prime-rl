//! Inference metrics parity with the Python collector (fixture from tools/gen_fixtures.py).

use std::path::Path;

use prl_config::InferenceRole;
use prl_io::http::AdminClient;
use prl_io::inference_metrics::{parse_prometheus_text, EngineSample, InferenceMetricsCollector};
use serde_json::Value;

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

#[test]
fn collector_matches_python_on_pd_deployment() {
    check_fixture("inference_metrics.json");
}

#[test]
fn collector_matches_python_on_real_vllm_scrapes() {
    check_fixture("inference_metrics_vllm.json");
}

fn check_fixture(name: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let clients: Vec<AdminClient> = fixture["base_urls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|u| AdminClient::new(u.as_str().unwrap(), Default::default()).unwrap())
        .collect();
    let roles: Option<Vec<InferenceRole>> = fixture["roles"].as_array().map(|roles| {
        roles.iter().map(|r| if r == "prefill" { InferenceRole::Prefill } else { InferenceRole::Decode }).collect()
    });
    let mut collector = InferenceMetricsCollector::new(&clients, roles.as_deref(), true, None, None).unwrap();
    let last = clients.last().unwrap().key().to_owned();
    collector.set_max_model_len(&last, fixture["max_model_len"].as_u64().unwrap());

    for poll in fixture["polls"].as_array().unwrap() {
        let timestamp = poll["timestamp"].as_f64().unwrap();
        let mut samples = Vec::new();
        for (endpoint, text) in poll["texts"].as_array().unwrap().iter().enumerate() {
            for (engine_label, snapshot) in parse_prometheus_text(text.as_str().unwrap()).unwrap() {
                samples.push(EngineSample { endpoint, engine_label, timestamp, snapshot });
            }
        }
        let (metrics, load) = collector.observe(samples);

        let expected = poll["metrics"].as_array().unwrap();
        let keys: Vec<&str> = metrics.keys().map(String::as_str).collect();
        let expected_keys: Vec<&str> = expected.iter().map(|kv| kv[0].as_str().unwrap()).collect();
        assert_eq!(keys, expected_keys);
        for kv in expected {
            let key = kv[0].as_str().unwrap();
            let got = metrics[key].as_f64().unwrap();
            let want = kv[1].as_f64().unwrap();
            assert!(close(got, want), "{key}: {got} != {want}");
        }

        let expected_load = poll["load"].as_array().unwrap();
        assert_eq!(load.len(), expected_load.len());
        for (got, want) in load.iter().zip(expected_load) {
            assert_eq!(got.engine_id, want["engine_id"].as_str().unwrap());
            assert_eq!(got.role.map(|r| r.as_str()), want["role"].as_str());
            assert_eq!(got.kv_capacity_tokens, want["kv_capacity_tokens"].as_u64());
            assert_eq!(got.max_model_len, want["max_model_len"].as_u64());
            assert_eq!(got.kv_usage, want["kv_usage"].as_f64().unwrap());
            assert_eq!(got.running, want["running"].as_u64().unwrap());
            assert_eq!(got.waiting, want["waiting"].as_u64().unwrap());
            assert_eq!(got.waiting_capacity, want["waiting_capacity"].as_u64());
            assert_eq!(got.preemptions_delta, want["preemptions_delta"].as_u64().unwrap());
        }
    }
}
