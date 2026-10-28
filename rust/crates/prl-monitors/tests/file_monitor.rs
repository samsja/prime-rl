//! The Rust file monitor against the Python one: the same inputs give byte-identical outputs.
//! `tests/fixtures/file_monitor` holds the inputs and the tree the Python `FileMonitor` wrote
//! for them (chunk size 8000, so the stream rolls and seals).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use prl_monitors::file::FileMonitor;
use prl_monitors::{FileMonitorConfig, LiveEvent, Row, Scalar};
use prl_types::Subset;
use prl_wire::{Episode, Value};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/file_monitor")
}

fn lines(name: &str) -> Vec<Vec<u8>> {
    fs::read(fixtures().join(name))
        .unwrap()
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(<[u8]>::to_vec)
        .collect()
}

fn episodes() -> Vec<Arc<Episode>> {
    lines("records.jsonl")
        .iter()
        .map(|l| Arc::new(Episode::from_record_json(l).unwrap()))
        .collect()
}

fn config(chunk_bytes: u64) -> FileMonitorConfig {
    FileMonitorConfig {
        path: "metrics.jsonl".into(),
        chunk_bytes,
        compress: true,
        float_decimals: Some(4),
    }
}

fn live_events() -> Vec<LiveEvent> {
    let events =
        prl_wire::value::from_msgpack(&fs::read(fixtures().join("live_events.msgpack")).unwrap())
            .unwrap();
    events
        .into_array()
        .unwrap()
        .into_iter()
        .map(|event| {
            let field = |k: &str| event.get(k).cloned().unwrap_or_default();
            let id = |k: &str| event.get(k).and_then(Value::as_str).map(str::to_string);
            if let Some(trace) = id("done") {
                LiveEvent::Done(trace)
            } else if let Some(dispatch_id) = id("dispatched") {
                LiveEvent::Dispatched(dispatch_id)
            } else if let Some(dispatch_id) = id("pending") {
                LiveEvent::Pending {
                    id: dispatch_id,
                    dispatch: field("dispatch"),
                }
            } else {
                LiveEvent::Delta {
                    delta: field("delta"),
                    dispatch: field("dispatch"),
                }
            }
        })
        .collect()
}

/// Every file under `dir`: relative path -> bytes, with each metrics row's wall time blanked.
fn tree(dir: &Path) -> Vec<(PathBuf, String)> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for path in fs::read_dir(&d).unwrap().map(|e| e.unwrap().path()) {
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let mut text = String::from_utf8_lossy(&fs::read(&path).unwrap()).into_owned();
            if path.ends_with("metrics.jsonl") {
                text = text
                    .lines()
                    .map(|line| {
                        let start = line.find("\"time\": ").unwrap() + 8;
                        format!(
                            "{}T{}\n",
                            &line[..start],
                            &line[start + line[start..].find(',').unwrap()..]
                        )
                    })
                    .collect();
            }
            files.push((path.strip_prefix(dir).unwrap().to_path_buf(), text));
        }
    }
    files.sort();
    files
}

#[test]
fn file_monitor_matches_python() {
    let episodes = episodes();
    let updates: Vec<Value> = lines("updates.jsonl")
        .iter()
        .map(|l| serde_json::from_slice(l).unwrap())
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    let mut monitor = FileMonitor::init(config(8000), &out, Some("orch".into())).unwrap();
    monitor.log_episodes(&episodes[..4], Subset::All).unwrap();
    monitor
        .log_episodes(&episodes[..2], Subset::Effective)
        .unwrap();
    monitor.log_episodes(&episodes[4..], Subset::All).unwrap();
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

    let (got, want) = (tree(&out), tree(&fixtures().join("run")));
    assert_eq!(
        got.iter().map(|(p, _)| p).collect::<Vec<_>>(),
        want.iter().map(|(p, _)| p).collect::<Vec<_>>()
    );
    for ((path, got), (_, want)) in got.iter().zip(&want) {
        assert!(
            got == want,
            "{} differs:\n got: {got}\nwant: {want}",
            path.display()
        );
    }
}

#[test]
fn stamp_batch_matches_python() {
    let episodes: Vec<Episode> = episodes().into_iter().map(Arc::unwrap_or_clone).collect();
    let updates = prl_monitors::annotations::stamp_batch(&episodes[..3], 2);
    let want = lines("updates.jsonl");
    assert_eq!(updates.len(), want.len());
    for (mut update, want) in updates.into_iter().zip(want) {
        // the ship time is wall time: take Python's
        let want_value: Value = serde_json::from_slice(&want).unwrap();
        let time = want_value
            .get("info")
            .unwrap()
            .get("ship")
            .unwrap()
            .get("time")
            .unwrap()
            .clone();
        update
            .get_mut("info")
            .unwrap()
            .get_mut("ship")
            .unwrap()
            .insert("time", time);
        assert_eq!(
            String::from_utf8(prl_wire::json::to_vec(&update)).unwrap(),
            String::from_utf8(want).unwrap()
        );
    }
}

#[test]
fn relaunch_continues_index_numbering() {
    let episodes = episodes();
    let dir = tempfile::tempdir().unwrap();
    for _ in 0..2 {
        let mut monitor = FileMonitor::init(config(1 << 20), dir.path(), None).unwrap();
        monitor.log_episodes(&episodes[..2], Subset::All).unwrap();
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
