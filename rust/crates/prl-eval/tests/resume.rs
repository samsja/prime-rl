use std::fs;
use std::path::{Path, PathBuf};

use prl_eval::resume::{archives, read_records, stamp_config, take_landed, to_orjson_indent2};
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load(name: &str) -> Value {
    serde_json::from_slice(&fs::read(fixtures().join(name)).unwrap()).unwrap()
}

fn copy_dir(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let path = entry.unwrap().path();
        let target = dst.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_dir(&path, &target);
        } else {
            fs::copy(&path, &target).unwrap();
        }
    }
}

fn parse(line: &[u8]) -> anyhow::Result<(String, bool)> {
    let record: Value = serde_json::from_slice(line)?;
    Ok((record["id"].as_str().unwrap().to_owned(), record["ok"].as_bool().unwrap()))
}

#[test]
fn stamp_matches_orjson_indent2() {
    let expected = load("resume_expected.json");
    let stamp = expected["stamp"].as_str().unwrap();
    let config: Value = serde_json::from_str(stamp).unwrap();
    assert_eq!(String::from_utf8(to_orjson_indent2(&config).unwrap()).unwrap(), stamp);

    let dir = tempfile::tempdir().unwrap();
    stamp_config(dir.path(), &config).unwrap();
    assert_eq!(fs::read_to_string(dir.path().join("monitors/file/eval.json")).unwrap(), stamp);
}

#[test]
fn take_landed_reads_sealed_and_live_chunks_and_rotates() {
    let expected = load("resume_expected.json");
    let config = load("resume_config.json");
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("run");
    copy_dir(&fixtures().join("resume_run"), &run);

    let landed = take_landed(&run, &config, false, parse).unwrap();

    let ids: Vec<&str> = expected["landed"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(landed, ids);
    let names: Vec<String> =
        archives(&run).unwrap().iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(names, expected["archives"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect::<Vec<_>>());
    assert!(!run.join("monitors/file").exists());
}

#[test]
fn torn_line_ends_the_stream() {
    let stream = fixtures().join("resume_run/monitors/file/traces/stream");
    let ids: Vec<String> = read_records(&stream).unwrap().into_iter().map(|(_, r)| r["id"].as_str().unwrap().to_owned()).collect();
    assert_eq!(ids, ["b0", "b1"]);
}

#[test]
fn validation_error_leaves_the_run_untouched() {
    let config = load("resume_config.json");
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("run");
    copy_dir(&fixtures().join("resume_run"), &run);
    let err = take_landed(&run, &config, false, |_| -> anyhow::Result<((), bool)> { anyhow::bail!("invalid") });
    assert!(err.is_err());
    assert!(run.join("monitors/file").is_dir());
    assert_eq!(archives(&run).unwrap().len(), 1);
}

#[test]
fn config_checks_match_python() {
    for case in load("resume_config_cases.json").as_array().unwrap() {
        for skip_checks in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            stamp_config(dir.path(), &case["saved"]).unwrap();
            let result = take_landed(dir.path(), &case["current"], skip_checks, parse);
            match (&case["error"], skip_checks) {
                (Value::String(message), false) => {
                    let got = result.unwrap_err().to_string().replace(&dir.path().display().to_string(), "<RUN>");
                    assert_eq!(&got, message);
                    assert!(dir.path().join("monitors/file").is_dir());
                }
                _ => {
                    assert!(result.unwrap().is_empty());
                    assert!(!dir.path().join("monitors/file").exists());
                    assert_eq!(archives(dir.path()).unwrap().len(), 1);
                }
            }
        }
    }
}

#[test]
fn missing_saved_config_needs_skip_checks() {
    let config = load("resume_config.json");
    for skip_checks in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("monitors/file")).unwrap();
        let result = take_landed(dir.path(), &config, skip_checks, parse);
        if skip_checks {
            assert!(result.unwrap().is_empty());
            assert_eq!(archives(dir.path()).unwrap().len(), 1);
        } else {
            assert!(result.unwrap_err().to_string().contains("no saved config"));
            assert!(archives(dir.path()).unwrap().is_empty());
        }
    }
}
