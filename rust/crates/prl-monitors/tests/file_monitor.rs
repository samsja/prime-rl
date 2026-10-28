//! The Rust file monitor against the Python one: same inputs, byte-identical outputs
//! (fixtures from `tests/fixtures/gen_file_monitor.py`).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use prl_monitors::file::{FileMonitor, FileMonitorConfig};
use prl_monitors::{LiveEvent, Row, Scalar};
use prl_types::{Kind, Subset};
use prl_wire::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/file_monitor")
}

fn json_lines(name: &str) -> Vec<(Vec<u8>, Value)> {
    fs::read(fixtures().join(name))
        .unwrap()
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| (l.to_vec(), serde_json::from_slice(l).unwrap()))
        .collect()
}

fn live_events() -> Vec<LiveEvent> {
    let raw =
        prl_wire::value::from_msgpack(&fs::read(fixtures().join("live_events.msgpack")).unwrap())
            .unwrap();
    raw.into_array()
        .unwrap()
        .into_iter()
        .map(|event| {
            let s = |k: &str| event.get(k).and_then(Value::as_str).map(str::to_string);
            if let Some(id) = s("done") {
                LiveEvent::Done(id)
            } else if let Some(id) = s("dispatched") {
                LiveEvent::Dispatched(id)
            } else if let Some(id) = s("pending") {
                LiveEvent::Pending {
                    id,
                    dispatch: event.get("dispatch").unwrap().clone(),
                }
            } else {
                LiveEvent::Delta {
                    delta: event.get("delta").unwrap().clone(),
                    dispatch: event.get("dispatch").unwrap().clone(),
                }
            }
        })
        .collect()
}

/// Every file under `dir`, relative path -> bytes.
fn tree(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push((
                    path.strip_prefix(dir).unwrap().to_path_buf(),
                    fs::read(&path).unwrap(),
                ));
            }
        }
    }
    out.sort();
    out
}

/// metrics.jsonl with each row's `"time": <float>` blanked: wall time differs between runs.
fn without_time(data: &[u8]) -> String {
    String::from_utf8(data.to_vec())
        .unwrap()
        .lines()
        .map(|line| {
            let start = line.find("\"time\": ").unwrap() + 8;
            let end = start + line[start..].find(',').unwrap();
            format!("{}T{}\n", &line[..start], &line[end..])
        })
        .collect()
}

#[test]
fn records_round_trip_through_the_orjson_writer() {
    for (line, record) in json_lines("records.jsonl") {
        assert_eq!(prl_wire::json::to_vec(&record), line);
    }
}

#[test]
fn file_monitor_matches_python() {
    let records: Vec<Arc<Value>> = json_lines("records.jsonl")
        .into_iter()
        .map(|(_, r)| Arc::new(r))
        .collect();
    let updates: Vec<Value> = json_lines("updates.jsonl")
        .into_iter()
        .map(|(_, u)| u)
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    let config = FileMonitorConfig {
        chunk_bytes: 8000,
        ..Default::default()
    };
    let mut monitor = FileMonitor::init(config, &out, Some("orch".into())).unwrap();
    monitor
        .log_episodes(&records[..4], 2, Kind::Train, Subset::All)
        .unwrap();
    monitor
        .log_episodes(&records[..2], 2, Kind::Train, Subset::Effective)
        .unwrap();
    monitor
        .log_episodes(&records[4..], 3, Kind::Eval, Subset::All)
        .unwrap();
    monitor.log_annotations(&updates).unwrap();
    monitor.log_live(&live_events()).unwrap();
    monitor.log_eval_plan("gsm8k", 2, 64).unwrap();
    monitor.log_eval_plan("gsm8k", 4, 64).unwrap();
    monitor.log_eval_plan("aime", 4, 30).unwrap();
    let row: Row = [
        ("loss/mean", Scalar::Float(0.5)),
        ("progress/total_tokens", Scalar::Int(1234)),
        ("tiny", Scalar::Float(1e-5)),
        ("big", Scalar::Float(1e16)),
        ("nan", Scalar::Float(f64::NAN)),
        ("inf/x", Scalar::Float(f64::INFINITY)),
        ("flag", Scalar::Bool(true)),
        ("name", Scalar::Str("é😀".into())),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    monitor.log_metrics(&row, Some(2)).unwrap();
    monitor
        .log_metrics(
            &[("inference/agg/x".to_string(), Scalar::Float(3.0))]
                .into_iter()
                .collect(),
            None,
        )
        .unwrap();
    monitor.finalize().unwrap();

    let want = tree(&fixtures().join("run"));
    let got = tree(&out);
    let names = |t: &[(PathBuf, Vec<u8>)]| t.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>();
    assert_eq!(names(&got), names(&want));
    for ((path, got), (_, want)) in got.iter().zip(&want) {
        if path.starts_with("monitors/file/traces/live") && std::env::var("SKIP_LIVE").is_ok() {
            continue;
        }
        if path.ends_with("metrics.jsonl") {
            assert_eq!(without_time(got), without_time(want));
        } else {
            assert!(
                got == want,
                "{} differs:\n got: {}\nwant: {}",
                path.display(),
                String::from_utf8_lossy(got),
                String::from_utf8_lossy(want)
            );
        }
    }
}

#[test]
fn relaunch_continues_index_numbering() {
    let records: Vec<Arc<Value>> = json_lines("records.jsonl")
        .into_iter()
        .map(|(_, r)| Arc::new(r))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    for _ in 0..2 {
        let mut monitor =
            FileMonitor::init(FileMonitorConfig::default(), dir.path(), None).unwrap();
        monitor
            .log_episodes(&records[..2], 1, Kind::Train, Subset::All)
            .unwrap();
        monitor.finalize().unwrap();
    }
    let index =
        fs::read_to_string(dir.path().join("monitors/file/traces/stream.index.jsonl")).unwrap();
    let lines: Vec<i64> = index
        .lines()
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["line"]
                .as_i64()
                .unwrap()
        })
        .collect();
    assert_eq!(lines, [1, 2, 3, 4]);
}
