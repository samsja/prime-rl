use std::fs;
use std::path::Path;

use prl_eval::resume::{archives, stamp_config, take_landed};
use serde_json::{json, Value};

fn record(id: &str, ok: bool) -> String {
    json!({
        "id": id,
        "env": {"id": "math", "name": "math"},
        "task": {"type": "Task", "data": {"idx": 0}, "key": id, "hash": id},
        "group": {"id": "g"},
        "ok": ok,
        "traces": [],
    })
    .to_string()
}

/// One attempt's file monitor dir: the stamped config and its stream chunks (`None` = sealed).
fn attempt(run: &Path, dir: &str, config: &Value, chunks: &[(&str, bool)]) {
    let monitors = run.join("monitors");
    stamp_config(run, config).unwrap();
    let stream = monitors.join("file/traces/stream");
    fs::create_dir_all(&stream).unwrap();
    for (number, (text, sealed)) in chunks.iter().enumerate() {
        let path = stream.join(format!("{number:05}.jsonl"));
        if *sealed {
            fs::write(path.with_extension("jsonl.zst"), zstd::encode_all(text.as_bytes(), 3).unwrap()).unwrap();
        } else {
            fs::write(path, text).unwrap();
        }
    }
    if dir != "file" {
        fs::rename(monitors.join("file"), monitors.join(dir)).unwrap();
    }
}

#[test]
fn landed_episodes_come_from_every_attempt() {
    let run = tempfile::tempdir().unwrap();
    let config = json!({"model": "m", "resume": null});
    let sealed = format!("{}\n{}\n", record("a0", true), record("a1", false));
    let no_traces = json!({"id": "a2", "env": {"id": "math"}, "task": {"type": "Task", "data": {}}, "ok": true}).to_string();
    attempt(run.path(), "file.attempt_1", &config, &[(&sealed, true), (&format!("{no_traces}\n"), false)]);
    // the live chunk of the interrupted attempt ends in a torn line
    let live = format!("{}\n{{\"id\": \"torn\", \"ok\": tr", record("b0", true));
    attempt(run.path(), "file", &json!({"model": "m", "resume": {"skip_checks": false}}), &[(&live, false)]);

    let landed = take_landed(run.path(), &config, false).unwrap();

    assert_eq!(landed.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["a0", "b0"]);
    assert!(!run.path().join("monitors/file").exists());
    let names: Vec<String> = archives(run.path()).unwrap().iter().map(|p| p.file_name().unwrap().to_string_lossy().into()).collect();
    assert_eq!(names, ["file.attempt_1", "file.attempt_2"]);
}

#[test]
fn changed_configs_need_skip_checks() {
    let run = tempfile::tempdir().unwrap();
    attempt(run.path(), "file", &json!({"model": "m", "group_size": 1, "log": {"level": "info"}}), &[]);
    let current = json!({"model": "m", "group_size": 2, "log": {"level": "debug"}, "resume": {}});

    let err = take_landed(run.path(), &current, false).unwrap_err().to_string();
    assert!(err.contains("monitors/file/eval.json in [group_size, log]"), "{err}");
    assert!(err.contains("--resume.skip-checks"));
    assert!(run.path().join("monitors/file").is_dir());

    assert!(take_landed(run.path(), &current, true).unwrap().is_empty());
    assert_eq!(archives(run.path()).unwrap().len(), 1);
}

#[test]
fn missing_saved_config_or_invalid_episode_leaves_the_run_untouched() {
    let run = tempfile::tempdir().unwrap();
    fs::create_dir_all(run.path().join("monitors/file")).unwrap();
    let err = take_landed(run.path(), &json!({}), false).unwrap_err().to_string();
    assert!(err.contains("no saved config"), "{err}");

    attempt(run.path(), "file", &json!({}), &[("{\"id\": \"bad\", \"traces\": null}\n", false)]);
    assert!(take_landed(run.path(), &json!({}), false).is_err());
    assert!(archives(run.path()).unwrap().is_empty());
}
