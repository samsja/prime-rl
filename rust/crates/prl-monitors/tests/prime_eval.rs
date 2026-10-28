//! The Prime eval monitor against a mock platform: one evaluation per (env, step), streamed
//! episodes, the finished epoch's aggregates, metrics onto the parent run.

use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::{OriginalUri, State};
use axum::http::Method;
use axum::routing::any;
use axum::{Json, Router};
use prl_monitors::prime::eval::{summary_from_episodes, EvalSourceSpec, PrimeEvalSetup};
use prl_monitors::{MonitorsSetup, PrimeSetup, Row, Scalar};
use prl_types::{Kind, Subset};
use serde_json::{json, Value as Json_};
use tokio_util::sync::CancellationToken;

type Log = Arc<Mutex<Vec<(String, String, Json_)>>>;

async fn route(
    State(log): State<Log>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    body: Bytes,
) -> Json<Json_> {
    let path = uri.path().to_string();
    let body: Json_ = if path.ends_with("/traces") || body.is_empty() {
        Json_::Null
    } else {
        serde_json::from_slice(&body).unwrap()
    };
    log.lock()
        .unwrap()
        .push((method.to_string(), path.clone(), body));
    Json(match path.as_str() {
        "/api/v1/environmentshub/resolve" => json!({"data": {"id": "env-1"}}),
        "/api/v1/evaluations/" => json!({"evaluation_id": "ev1", "name": "my-eval"}),
        "/api/v1/traces" => json!({"upload_id": "x", "status": "committed"}),
        _ => json!({}),
    })
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}

fn episodes() -> Vec<Arc<prl_wire::Episode>> {
    fixture("file_monitor/records.jsonl")
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| Arc::new(prl_wire::Episode::from_record_json(l).unwrap()))
        .collect()
}

#[test]
fn summary_matches_prime_runs() {
    let want: Json_ = serde_json::from_slice(&fixture("eval_summary.json")).unwrap();
    assert_eq!(Json_::Object(summary_from_episodes(&episodes())), want);
}

#[tokio::test]
async fn prime_eval_monitor_against_mock_platform() {
    let log: Log = Arc::default();
    let app = Router::new().fallback(any(route)).with_state(log.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let home = tempfile::tempdir().unwrap();
    // SAFETY: the only test touching the environment in this binary.
    unsafe {
        for var in [
            "PRIME_RUNS_MODE",
            "PRIME_API_BASE",
            "PRIME_TRACES_OPT_OUT",
            "PRIME_RUNS_EVAL_ID",
            "RANK",
            "PRL_ATTEMPT_LOG_DIR",
        ] {
            std::env::remove_var(var);
        }
        std::env::set_var("HOME", home.path());
        std::env::set_var("PRIME_API_KEY", "key");
        std::env::set_var("PRIME_TEAM_ID", "team");
        std::env::set_var("PRIME_API_BASE_URL", &url);
        std::env::set_var("PRIME_TRACES_URL", &url);
        std::env::set_var("PRIME_FRONTEND_URL", "https://app.test");
        std::env::set_var("PRL_RUN_ID", "prl-run");
        std::env::set_var("RUN_ID", "parent");
    }
    let out = tempfile::tempdir().unwrap();
    let setup = MonitorsSetup {
        output_dir: out.path().to_path_buf(),
        producer: Some("eval".into()),
        prime: Some(PrimeSetup::Eval(PrimeEvalSetup {
            name: Some("my-eval".into()),
            model: "m".into(),
            sources: vec![EvalSourceSpec {
                resolved_name: "test".into(),
                taskset_id: "gsm8k".into(),
                group_size: 2,
            }],
        })),
        wandb: None,
        file: None,
        train_env_names: vec![],
        eval_env_names: vec!["test".into()],
        overview_flavor: "eval".into(),
    };
    let (monitors, task) = prl_monitors::spawn(setup, CancellationToken::new())
        .await
        .unwrap();
    let eps = episodes();
    monitors.log_eval_plan("test".into(), 5, 8).await;
    monitors
        .log_episodes(eps[..2].to_vec(), 5, Kind::Eval, Subset::All)
        .await;
    monitors
        .log_episodes(eps[..2].to_vec(), 5, Kind::Train, Subset::All)
        .await; // not an eval: ignored
    monitors.log_eval_epoch("test".into(), 5, eps.clone()).await;
    let mut row = Row::new();
    row.insert("eval/test/avg@2".into(), Scalar::Float(0.5));
    monitors.log_metrics(row, Some(5)).await;
    monitors.finalize(false).await;
    task.await.unwrap().unwrap();

    let requests = log.lock().unwrap().clone();
    let calls: Vec<(&str, &str)> = requests
        .iter()
        .map(|(m, p, _)| (m.as_str(), p.as_str()))
        .collect();
    assert_eq!(
        calls,
        [
            ("POST", "/api/v1/environmentshub/resolve"),
            ("POST", "/api/v1/evaluations/"),
            ("POST", "/api/v1/traces"),
            ("PUT", "/api/v1/evaluations/ev1"),
            ("POST", "/api/v1/evaluations/ev1/finalize"),
            ("POST", "/api/v1/rft/metrics"),
        ]
    );
    assert_eq!(requests[0].2, json!({"name": "gsm8k", "team_id": "team"}));
    assert_eq!(
        requests[1].2,
        json!({"name": "my-eval", "environments": [{"id": "env-1"}], "tags": [], "model_name": "m", "dataset": "gsm8k",
               "framework": "prime-rl",
               "metadata": {"model": "m", "step": 5, "run_id": "prl-run", "num_examples": 4, "rollouts_per_example": 2},
               "team_id": "team"})
    );
    let mut summary = json!(summary_from_episodes(&eps));
    summary["prime_runs"] = json!({"traces_episodes_written": 2});
    assert_eq!(requests[3].2["metrics"], summary);
    assert_eq!(requests[3].2["metadata"]["step"], 5);
    assert_eq!(requests[4].2, json!({"metrics": summary}));
    assert_eq!(requests[5].2["run_id"], "parent");
    assert_eq!(requests[5].2["metrics"]["eval/test/avg@2"], 0.5);

    let record: Json_ =
        serde_json::from_slice(&std::fs::read(out.path().join("monitors/prime/run.json")).unwrap())
            .unwrap();
    assert_eq!(
        record,
        json!({"kind": "eval", "run_id": "prl-run",
               "evaluations": {"test": {"step": 5, "id": "ev1", "url": "https://app.test/dashboard/evaluations/ev1"}}})
    );
    let links: Json_ =
        serde_json::from_slice(&std::fs::read(out.path().join("monitors/links.json")).unwrap())
            .unwrap();
    assert_eq!(
        links["prime/eval"]["links"]["test"],
        "https://app.test/dashboard/evaluations/ev1"
    );
}
