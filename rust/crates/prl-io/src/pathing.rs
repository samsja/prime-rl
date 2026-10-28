//! Run-directory layout helpers (`utils/pathing.py`).

use std::path::{Path, PathBuf};
use std::time::Duration;

pub fn ckpt_dir(output_dir: &Path) -> PathBuf {
    output_dir.join("checkpoints")
}

pub fn broadcast_dir(output_dir: &Path) -> PathBuf {
    output_dir.join("broadcasts")
}

pub(crate) fn step_path(path: &Path, step: u64) -> PathBuf {
    path.join(format!("step_{step}"))
}

/// All `step_*` steps under `dir`, ascending. A missing dir has none.
pub fn all_ckpt_steps(dir: &Path) -> std::io::Result<Vec<u64>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut steps = Vec::new();
    for entry in entries {
        let name = entry?.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some(rest) = name.strip_prefix("step_") {
            // Python: int(name.split("_")[-1]); a non-integer suffix raises there.
            let last = name.rsplit('_').next().unwrap_or(rest);
            let step = last.parse::<u64>().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("invalid step dir {name:?} in {}", dir.display()))
            })?;
            steps.push(step);
        }
    }
    steps.sort_unstable();
    Ok(steps)
}

/// Latest checkpoint step, logging like `resolve_latest_ckpt_step`.
pub fn resolve_latest_ckpt_step(ckpt_dir: &Path) -> std::io::Result<Option<u64>> {
    let steps = all_ckpt_steps(ckpt_dir)?;
    match steps.last() {
        None => {
            tracing::warn!("No checkpoints found in {}. Starting from scratch.", ckpt_dir.display());
            Ok(None)
        }
        Some(&step) => {
            tracing::info!("Found latest checkpoint in {}: {step}", ckpt_dir.display());
            Ok(Some(step))
        }
    }
}

/// Poll every second until `path` exists (`wait_for_path`).
pub async fn wait_for_path(path: &Path) {
    let interval = 1u64;
    let log_interval = 10u64;
    let mut wait_time = 0u64;
    tracing::debug!("Waiting for path `{}`", path.display());
    loop {
        if path.exists() {
            tracing::debug!("Found path `{}`", path.display());
            return;
        }
        if wait_time.is_multiple_of(log_interval) && wait_time > 0 {
            tracing::debug!("Waiting for path `{}` for {wait_time} seconds", path.display());
        }
        tokio::time::sleep(Duration::from_secs(interval)).await;
        wait_time += interval;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_are_sorted_numerically() {
        let dir = tempfile::tempdir().unwrap();
        for step in [10, 2, 1] {
            std::fs::create_dir(step_path(dir.path(), step)).unwrap();
        }
        std::fs::create_dir(dir.path().join("other")).unwrap();
        assert_eq!(all_ckpt_steps(dir.path()).unwrap(), vec![1, 2, 10]);
        assert_eq!(all_ckpt_steps(&dir.path().join("missing")).unwrap(), Vec::<u64>::new());
        assert_eq!(resolve_latest_ckpt_step(dir.path()).unwrap(), Some(10));
    }
}
