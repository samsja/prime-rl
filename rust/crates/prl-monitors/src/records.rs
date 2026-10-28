//! Locked JSON records under `monitors/` (`prime_rl.utils.record` / `prime_rl.utils.links`):
//! the run-page links for the launcher's banner and the Prime run identity for the dashboard.
//! Several processes of a run merge into the same files.

use std::fs::{self, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use serde_json::{json, Value};

const RECORD_LOCK_TIMEOUT: Duration = Duration::from_secs(10);
pub const PRIME_LINKS: &str = "prime";
pub const PRIME_EVAL_LINKS: &str = "prime/eval";

pub fn platform_run_path(output_dir: &Path) -> PathBuf {
    output_dir.join("monitors/prime/run.json")
}

/// Locked read-modify-write: flock on a sibling `.lock` file (never unlinked: a stable inode
/// is the whole guarantee), bounded non-blocking retries, then an atomic replace through a
/// pid-suffixed tmp file, indented like `orjson.OPT_INDENT_2`.
fn merge_record(path: &Path, merge: impl FnOnce(Value) -> Value) -> Result<()> {
    fs::create_dir_all(path.parent().expect("records live in a directory"))?;
    let lock = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{}.lock", path.display()))?;
    let deadline = Instant::now() + RECORD_LOCK_TIMEOUT;
    // SAFETY: the descriptor stays open (owned by `lock`) for the whole call.
    while unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::WouldBlock {
            return Err(err.into());
        }
        if Instant::now() >= deadline {
            bail!(
                "record lock at {}.lock still held after {RECORD_LOCK_TIMEOUT:?}",
                path.display()
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let record = if path.is_file() {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        json!({})
    };
    let tmp = PathBuf::from(format!("{}.tmp.{}", path.display(), std::process::id()));
    let written = fs::write(&tmp, serde_json::to_vec_pretty(&merge(record))?)
        .and_then(|()| fs::rename(&tmp, path));
    if written.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    Ok(written?)
    // the lock goes with the descriptor
}

/// Best-effort: a lock or storage failure is logged and skipped, never fatal (the records only
/// back links).
pub fn update_record(path: &Path, merge: impl FnOnce(Value) -> Value) {
    if let Err(e) = merge_record(path, merge) {
        tracing::warn!(
            "Failed to persist {} (lock or storage error); the run's links may be missing: {e}",
            path.display()
        );
    }
}

/// Overlay `update` onto the platform record, preserving keys it does not set.
pub fn update_platform_record(output_dir: &Path, update: Value) {
    update_record(&platform_run_path(output_dir), |mut record| {
        if let (Some(record), Value::Object(update)) = (record.as_object_mut(), update) {
            record.extend(update);
        }
        record
    });
}

/// `monitor`'s entry in `links.json` for this attempt (`$PRL_ATTEMPT_LOG_DIR`), started over
/// when the record holds a previous attempt's.
fn update_links(output_dir: &Path, monitor: &str, expected: u64, update: impl FnOnce(&mut Value)) {
    update_record(&output_dir.join("monitors/links.json"), |mut record| {
        let attempt = std::env::var("PRL_ATTEMPT_LOG_DIR").ok();
        if record[monitor]["attempt"].as_str() != attempt.as_deref() || !record[monitor].is_object()
        {
            record[monitor] = json!({"attempt": attempt, "expected": expected, "links": {}});
        }
        update(&mut record[monitor]);
        record
    });
}

/// Announce that `monitor` publishes `expected` links in this attempt.
pub fn register_links(output_dir: &Path, monitor: &str, expected: u64) {
    update_links(output_dir, monitor, expected, |entry| {
        entry["expected"] = entry["expected"].as_u64().unwrap_or(0).max(expected).into();
    });
}

/// One of `monitor`'s links; `None` when its page did not open.
pub fn publish_link(output_dir: &Path, monitor: &str, label: &str, url: Option<&str>) {
    update_links(output_dir, monitor, 1, |entry| {
        entry["links"][label] = url.into()
    });
}
