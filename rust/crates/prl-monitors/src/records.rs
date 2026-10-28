//! Locked JSON records under `monitors/` (`prime_rl.utils.record` / `prime_rl.utils.links`):
//! the run-page links for the launcher's banner and the Prime run identity for the dashboard.

use std::fs::{self, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use serde_json::{json, Map, Value};

pub const RECORD_LOCK_TIMEOUT: Duration = Duration::from_secs(10);
const RECORD_LOCK_POLL: Duration = Duration::from_millis(50);

/// Exported by the launcher to every component: entries stamped with another attempt are a
/// previous launch's.
pub const ATTEMPT_VAR: &str = "PRL_ATTEMPT_LOG_DIR";
pub const PRIME_LINKS: &str = "prime";
pub const WANDB_LINKS: &str = "wandb";

pub fn links_path(output_dir: &Path) -> PathBuf {
    output_dir.join("monitors").join("links.json")
}

pub fn platform_run_path(output_dir: &Path) -> PathBuf {
    output_dir.join("monitors").join("prime").join("run.json")
}

pub fn read_record(path: &Path) -> Result<Option<Map<String, Value>>> {
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_slice(&fs::read(path)?)?))
}

/// Atomic replace through a pid-suffixed tmp file, indented like `orjson.OPT_INDENT_2`.
pub fn write_record(path: &Path, record: &Map<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = PathBuf::from(format!("{}.tmp.{}", path.display(), std::process::id()));
    let result =
        fs::write(&tmp, serde_json::to_vec_pretty(record)?).and_then(|()| fs::rename(&tmp, path));
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    Ok(result?)
}

/// flock on a sibling `.lock` file (never unlinked), acquired with bounded non-blocking retries.
fn with_record_lock<T>(path: &Path, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let lock_path = PathBuf::from(format!("{}.lock", path.display()));
    let lock_file = OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(&lock_path)?;
    let fd = lock_file.as_raw_fd();
    let deadline = Instant::now() + RECORD_LOCK_TIMEOUT;
    loop {
        // SAFETY: fd is a valid open descriptor owned by lock_file for this whole scope.
        if unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            break;
        }
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::WouldBlock {
            return Err(err.into());
        }
        if Instant::now() >= deadline {
            bail!(
                "record lock at {} still held after {:?}",
                lock_path.display(),
                RECORD_LOCK_TIMEOUT
            );
        }
        std::thread::sleep(RECORD_LOCK_POLL);
    }
    let result = f();
    // SAFETY: as above.
    unsafe { libc::flock(fd, libc::LOCK_UN) };
    result
}

/// Best-effort locked read-modify-write: a lock or storage failure is logged and skipped,
/// never fatal (the records only back links).
pub fn update_record(path: &Path, merge: impl FnOnce(Map<String, Value>) -> Map<String, Value>) {
    let result = with_record_lock(path, || {
        write_record(path, &merge(read_record(path)?.unwrap_or_default()))
    });
    if let Err(e) = result {
        tracing::warn!(
            "Failed to persist {} (lock or storage error); the run's links may be missing: {e}",
            path.display()
        );
    }
}

/// This attempt's entry for `monitor`, started over when the record holds a previous attempt's.
fn entry<'a>(
    record: &'a mut Map<String, Value>,
    monitor: &str,
    expected: u64,
) -> &'a mut Map<String, Value> {
    let attempt = std::env::var(ATTEMPT_VAR).ok();
    let stale = match record.get(monitor) {
        Some(Value::Object(e)) => e.get("attempt").and_then(Value::as_str) != attempt.as_deref(),
        _ => true,
    };
    if stale {
        record.insert(
            monitor.to_string(),
            json!({"attempt": attempt, "expected": expected, "links": {}}),
        );
    }
    record
        .get_mut(monitor)
        .and_then(Value::as_object_mut)
        .expect("entry is an object")
}

/// Announce that `monitor` publishes `expected` links in this attempt.
pub fn register_links(output_dir: &Path, monitor: &str, expected: u64) {
    update_record(&links_path(output_dir), |mut record| {
        let e = entry(&mut record, monitor, expected);
        let current = e.get("expected").and_then(Value::as_u64).unwrap_or(0);
        e.insert("expected".into(), current.max(expected).into());
        record
    });
}

/// One of `monitor`'s links; `None` when its page did not open.
pub fn publish_link(output_dir: &Path, monitor: &str, label: &str, url: Option<&str>) {
    update_record(&links_path(output_dir), |mut record| {
        let e = entry(&mut record, monitor, 1);
        if let Some(Value::Object(links)) = e.get_mut("links") {
            links.insert(label.to_string(), url.map_or(Value::Null, |u| u.into()));
        }
        record
    });
}

/// Overlay `update` onto the platform record, preserving keys it does not set.
pub fn update_platform_record(output_dir: &Path, update: Map<String, Value>) {
    update_record(&platform_run_path(output_dir), |mut record| {
        record.extend(update);
        record
    });
}
