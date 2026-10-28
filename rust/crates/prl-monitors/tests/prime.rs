//! The Prime monitors against a mock platform: registration, metrics rows, Prime Traces uploads
//! and the close-out of training runs and evaluations, through the monitors actor.

use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::{FromRequest, OriginalUri, Request, State};
use axum::http::{HeaderMap, Method};
use axum::routing::any;
use axum::{Json, Router};
use prl_monitors::prime::backend::TrainingSpec;
use prl_monitors::prime::eval::{summary_from_episodes, EvalSourceSpec, PrimeEvalSetup};
use prl_monitors::{MonitorsSetup, PrimeSetup, PrimeTrainSetup, Row, Scalar};
use prl_types::{Kind, Subset};
use prl_wire::Episode;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

/// (method, path, headers, body); a Prime Traces upload's body holds its decoded parts.
type Log = Arc<Mutex<Vec<(String, String, HeaderMap, Value)>>>;

async fn route(
    State(log): State<Log>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    request: Request,
) -> Json<Value> {
    let headers = request.headers().clone();
    let path = uri.path().to_string();
    let body = if path == "/api/v1/traces" {
        let mut multipart = axum::extract::Multipart::from_request(request, &())
            .await
            .unwrap();
        let mut parts = serde_json::Map::new();
        while let Some(field) = multipart.next_field().await.unwrap() {
            let name = field.name().unwrap().to_string();
            let gzip = field
                .headers()
                .get("content-encoding")
                .is_some_and(|v| v == "gzip");
            let data = field.bytes().await.unwrap();
            let mut text = String::new();
            if gzip {
                flate2::read::GzDecoder::new(&data[..])
                    .read_to_string(&mut text)
                    .unwrap();
            } else {
                text = String::from_utf8(data.to_vec()).unwrap();
            }
            parts.insert(name, text.into());
        }
        Value::Object(parts)
    } else {
        let bytes = Bytes::from_request(request, &()).await.unwrap();
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    log.lock()
        .unwrap()
        .push((method.to_string(), path.clone(), headers, body));
    Json(match path.as_str() {
        "/api/v1/rft/external-runs" => json!({"run": {"id": "r1"}}),
        "/api/v1/environmentshub/resolve" => json!({"data": {"id": "env-1"}}),
        "/api/v1/evaluations/" => json!({"evaluation_id": "ev1"}),
        _ => json!({}),
    })
}

fn calls(log: &Log) -> Vec<String> {
    log.lock()
        .unwrap()
        .drain(..)
        .map(|(method, path, _, _)| format!("{method} {path}"))
        .collect()
}

fn episodes() -> Vec<Arc<Episode>> {
    let data = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/file_monitor/records.jsonl"),
    )
    .unwrap();
    data.split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| Arc::new(Episode::from_record_json(l).unwrap()))
        .collect()
}

fn setup(output_dir: &Path, prime: PrimeSetup) -> MonitorsSetup {
    MonitorsSetup {
        output_dir: output_dir.to_path_buf(),
        producer: None,
        prime: Some(prime),
        wandb: None,
        file: None,
        train_env_names: vec![],
        eval_env_names: vec![],
        overview_flavor: "rl".into(),
    }
}

fn train() -> PrimeSetup {
    PrimeSetup::Train(PrimeTrainSetup {
        name: Some("my-run".into()),
        model: "Qwen/Qwen3-0.6B".into(),
        environments: vec!["gsm8k".into()],
        training: TrainingSpec {
            max_steps: 10,
            batch_size: Some(8),
            rollouts_per_example: Some(4),
            seq_len: Some(2048),
            wandb_project: None,
        },
        run_config: json!({"max_steps": 10}).as_object().unwrap().clone(),
    })
}

fn read_json(path: std::path::PathBuf) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn eval_summary_matches_prime_runs() {
    // `prime_runs.metrics.from_episodes` over the same episodes
    let want = json!({"avg_reward": 0.6714285714285715, "avg_metrics": {"correct": 0.5714285714285714, "style": 0.5, "len": 7.0, "count": 4.0}, "avg_error": 0.0});
    assert_eq!(Value::Object(summary_from_episodes(&episodes())), want);
}

#[tokio::test]
async fn prime_monitors_against_mock_platform() {
    let log: Log = Arc::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().fallback(any(route)).with_state(log.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let home = tempfile::tempdir().unwrap();
    // SAFETY: the only test in this binary that touches the environment.
    unsafe {
        for var in [
            "RUN_ID",
            "PRIME_RUNS_MODE",
            "PRIME_API_BASE",
            "PRIME_TRACES_OPT_OUT",
            "PRIME_RUNS_EVAL_ID",
            "RANK",
            "PRL_ATTEMPT_LOG_DIR",
        ] {
            std::env::remove_var(var);
        }
        for (var, value) in [
            ("HOME", home.path().to_str().unwrap()),
            ("PRIME_API_KEY", "key"),
            ("PRIME_TEAM_ID", "team"),
            ("PRL_RUN_ID", "prl-run"),
        ] {
            std::env::set_var(var, value);
        }
        for var in ["PRIME_API_BASE_URL", "PRIME_TRACES_URL"] {
            std::env::set_var(var, &url);
        }
        std::env::set_var("PRIME_FRONTEND_URL", "https://app.test");
    }
    let eps = episodes();

    // a training run: register, log, finalize
    let out = tempfile::tempdir().unwrap();
    let (monitors, task) =
        prl_monitors::spawn(setup(out.path(), train()), CancellationToken::new())
            .await
            .unwrap();
    let row: Row = [
        ("loss/mean".to_string(), Scalar::Float(0.5)),
        ("bad".to_string(), Scalar::Float(f64::NAN)),
    ]
    .into_iter()
    .collect();
    monitors.log_metrics(row, Some(3)).await;
    monitors
        .log_episodes(eps[..2].to_vec(), 3, Kind::Train, Subset::All)
        .await;
    monitors
        .log_episodes(eps[2..3].to_vec(), 3, Kind::Train, Subset::Effective)
        .await;
    monitors.finalize(false).await;
    task.await.unwrap().unwrap();
    let requests = log.lock().unwrap().clone();
    assert_eq!(
        calls(&log),
        [
            "POST /api/v1/rft/external-runs",
            "POST /api/v1/rft/metrics",
            "POST /api/v1/traces",
            "POST /api/v1/rft/finalize"
        ]
    );
    let (_, _, headers, create) = &requests[0];
    assert_eq!(headers["authorization"], "Bearer key");
    assert_eq!(
        *create,
        json!({"base_model": "Qwen/Qwen3-0.6B", "max_steps": 10, "environments": [{"id": "gsm8k"}], "team_id": "team",
               "name": "my-run", "batch_size": 8, "rollouts_per_example": 4, "seq_len": 2048, "run_config": {"max_steps": 10}})
    );
    let metrics = &requests[1].3;
    assert_eq!(
        (
            &metrics["run_id"],
            &metrics["metrics"]["loss/mean"],
            &metrics["metrics"]["step"]
        ),
        (&json!("r1"), &json!(0.5), &json!(3))
    );
    assert!(metrics["metrics"]["_timestamp"].is_f64() && metrics["metrics"].get("bad").is_none());
    let (_, _, headers, upload) = &requests[2];
    let text = upload["traces"].as_str().unwrap();
    let digest = <sha2::Sha256 as sha2::Digest>::digest(text.as_bytes());
    assert_eq!(
        headers["idempotency-key"],
        format!("sha256:{}", hex::encode(digest))
    );
    assert_eq!(
        (&headers["x-prime-line-format"], &headers["x-prime-team-id"]),
        (&"episode".parse().unwrap(), &"team".parse().unwrap())
    );
    assert_eq!(
        upload["metadata"],
        r#"{"schema_version": 1, "context": {"source": "prime-runs", "run_type": "train", "model": "Qwen/Qwen3-0.6B"}}"#
    );
    let expected: String = eps[..2]
        .iter()
        .map(|ep| {
            let stamped =
                prl_monitors::prime::sinks::stamp_run(&ep.to_record(Some(4)), "r1", "train");
            String::from_utf8(prl_wire::json::to_vec(&stamped)).unwrap() + "\n"
        })
        .collect();
    assert_eq!(text, expected);
    assert_eq!(requests[3].3, json!({"run_id": "r1", "exit_code": 0}));
    assert_eq!(
        read_json(out.path().join("monitors/prime/run.json")),
        json!({"kind": "train", "id": "r1", "url": "https://app.test/dashboard/training/r1"})
    );
    let links = read_json(out.path().join("monitors/links.json"));
    assert_eq!(
        (&links["prime"]["links"]["run"], &links["prime"]["expected"]),
        (&json!("https://app.test/dashboard/training/r1"), &json!(1))
    );

    // cancelled before finalize: what was logged still goes out, then the run reports crashed
    let cancel = CancellationToken::new();
    let (monitors, task) = prl_monitors::spawn(setup(out.path(), train()), cancel.clone())
        .await
        .unwrap();
    monitors.log_metrics(Row::new(), Some(1)).await;
    cancel.cancel();
    task.await.unwrap().unwrap();
    let crashed = log.lock().unwrap()[2].3.clone();
    assert_eq!(
        calls(&log),
        [
            "POST /api/v1/rft/external-runs",
            "POST /api/v1/rft/metrics",
            "PUT /api/v1/rft/external-runs/r1/status"
        ]
    );
    assert_eq!(
        crashed,
        json!({"status": "failed", "error_message": "crashed: process exited without finishing the run"})
    );

    // opted out of Prime Traces: the run works, episodes never reach Prime Traces
    unsafe { std::env::set_var("PRIME_TRACES_OPT_OUT", "1") };
    let (monitors, task) =
        prl_monitors::spawn(setup(out.path(), train()), CancellationToken::new())
            .await
            .unwrap();
    monitors
        .log_episodes(eps[..1].to_vec(), 3, Kind::Train, Subset::All)
        .await;
    monitors.finalize(false).await;
    task.await.unwrap().unwrap();
    assert_eq!(
        calls(&log),
        [
            "POST /api/v1/rft/external-runs",
            "POST /api/v1/rft/finalize"
        ]
    );
    unsafe { std::env::remove_var("PRIME_TRACES_OPT_OUT") };

    // an evaluation per (env, step), metrics onto the attached parent training run
    unsafe { std::env::set_var("RUN_ID", "parent") };
    let out = tempfile::tempdir().unwrap();
    let sources = vec![EvalSourceSpec {
        resolved_name: "test".into(),
        taskset_id: "gsm8k".into(),
        group_size: 2,
    }];
    let eval = PrimeSetup::Eval(PrimeEvalSetup {
        name: Some("my-eval".into()),
        model: "m".into(),
        sources,
    });
    let (monitors, task) = prl_monitors::spawn(setup(out.path(), eval), CancellationToken::new())
        .await
        .unwrap();
    monitors.log_eval_plan("test".into(), 5, 8).await;
    monitors
        .log_episodes(eps[..2].to_vec(), 5, Kind::Eval, Subset::All)
        .await;
    monitors
        .log_episodes(eps[..2].to_vec(), 5, Kind::Train, Subset::All)
        .await; // not an eval: ignored
    monitors.log_eval_epoch("test".into(), 5, eps.clone()).await;
    monitors
        .log_metrics(
            [("eval/test/avg@2".to_string(), Scalar::Float(0.5))]
                .into_iter()
                .collect(),
            Some(5),
        )
        .await;
    monitors.finalize(false).await;
    task.await.unwrap().unwrap();
    let requests = log.lock().unwrap().clone();
    assert_eq!(
        calls(&log),
        [
            "POST /api/v1/environmentshub/resolve",
            "POST /api/v1/evaluations/",
            "POST /api/v1/traces",
            "PUT /api/v1/evaluations/ev1",
            "POST /api/v1/evaluations/ev1/finalize",
            "POST /api/v1/rft/metrics",
        ]
    );
    assert_eq!(requests[0].3, json!({"name": "gsm8k", "team_id": "team"}));
    assert_eq!(
        requests[1].3,
        json!({"name": "my-eval", "environments": [{"id": "env-1"}], "tags": [], "model_name": "m", "dataset": "gsm8k",
               "framework": "prime-rl", "team_id": "team",
               "metadata": {"model": "m", "step": 5, "run_id": "prl-run", "num_examples": 4, "rollouts_per_example": 2}})
    );
    let mut summary = Value::Object(summary_from_episodes(&eps));
    summary["prime_runs"] = json!({"traces_episodes_written": 2});
    assert_eq!(
        (&requests[3].3["metrics"], &requests[4].3),
        (&summary, &json!({"metrics": summary}))
    );
    assert_eq!(
        (
            &requests[5].3["run_id"],
            &requests[5].3["metrics"]["eval/test/avg@2"]
        ),
        (&json!("parent"), &json!(0.5))
    );
    assert_eq!(
        read_json(out.path().join("monitors/prime/run.json")),
        json!({"kind": "eval", "run_id": "prl-run", "evaluations": {"test": {"step": 5, "id": "ev1", "url": "https://app.test/dashboard/evaluations/ev1"}}})
    );
    assert_eq!(
        read_json(out.path().join("monitors/links.json"))["prime/eval"]["links"]["test"],
        "https://app.test/dashboard/evaluations/ev1"
    );
}
