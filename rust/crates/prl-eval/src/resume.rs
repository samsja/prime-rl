//! Resume an interrupted eval from its trace stream (port of `eval/resume.py`).
//!
//! Successful episodes rejoin the epoch; failed and cut-off ones are owed again. Each attempt's
//! saved resolved config must match the current one (except `resume`) unless
//! `--resume.skip-checks` is set.

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use prl_monitors::file::chunks::{chunk_numbers, chunk_path};
use prl_monitors::file::{file_monitor_dir, trace_stream};
use prl_wire::Episode;
use serde_json::{Map, Value};

/// The resolved config an attempt stamps beside the episodes it produces.
pub const CONFIG_NAME: &str = "eval.json";

pub fn stamp_config(run_dir: &Path, config: &Value) -> Result<()> {
    let directory = file_monitor_dir(run_dir);
    fs::create_dir_all(&directory)?;
    fs::write(directory.join(CONFIG_NAME), serde_json::to_vec_pretty(config)?)?;
    Ok(())
}

/// Every record of a trace stream, plain or sealed chunks in order. A torn line (the interrupted
/// process died mid-append) ends the stream.
pub fn read_records(stream: &Path) -> Result<Vec<prl_wire::Value>> {
    let mut records = Vec::new();
    for number in chunk_numbers(stream)?.into_keys() {
        let plain = chunk_path(stream, number);
        let reader: Box<dyn Read> = match fs::File::open(&plain) {
            Ok(file) => Box::new(file),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let sealed = PathBuf::from(format!("{}.zst", plain.display()));
                Box::new(zstd::Decoder::new(fs::File::open(&sealed).with_context(|| format!("open {}", sealed.display()))?)?)
            }
            Err(e) => return Err(e.into()),
        };
        for line in BufReader::new(reader).split(b'\n') {
            let Ok(record) = serde_json::from_slice(&line?) else { return Ok(records) };
            records.push(record);
        }
    }
    Ok(records)
}

/// File monitor directories of earlier attempts (`monitors/file.attempt_N`), oldest first.
pub fn archives(run_dir: &Path) -> Result<Vec<PathBuf>> {
    let file_dir = file_monitor_dir(run_dir);
    let monitors = file_dir.parent().expect("the file monitor dir has a parent");
    let mut found = Vec::new();
    if monitors.is_dir() {
        for entry in fs::read_dir(monitors)? {
            let path = entry?.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            if let Some(n) = name.strip_prefix("file.attempt_") {
                found.push((n.parse::<u64>().with_context(|| format!("archive name {name}"))?, path));
            }
        }
    }
    found.sort();
    Ok(found.into_iter().map(|(_, path)| path).collect())
}

fn without_resume(config: &Value) -> Result<Map<String, Value>> {
    let mut map = config.as_object().context("a resolved eval config must be a JSON object")?.clone();
    map.shift_remove("resume");
    Ok(map)
}

/// Successful episodes from every attempt (the rollout planner deduplicates them). The current
/// file monitor directory joins the archives so the resumed attempt writes a fresh stream.
/// `config` is the current `model_dump(mode="json")`.
pub fn take_landed(run_dir: &Path, config: &Value, skip_checks: bool) -> Result<Vec<Episode>> {
    let current = file_monitor_dir(run_dir);
    let stream = trace_stream(run_dir);
    let stream = stream.strip_prefix(&current).expect("the trace stream lives in the file monitor dir");
    let mut directories = archives(run_dir)?;
    if current.is_dir() || directories.is_empty() {
        directories.push(current.clone());
    }
    let expected = without_resume(config)?;
    if skip_checks {
        tracing::warn!("Skipping resume compatibility checks; saved episodes may use different experiment settings.");
    }
    let mut landed = Vec::new();
    for directory in &directories {
        if !skip_checks {
            let saved_path = directory.join(CONFIG_NAME);
            if !saved_path.is_file() {
                bail!(
                    "--resume: no saved config at {}. Set --resume.skip-checks to reuse episodes without a saved config.",
                    saved_path.display()
                );
            }
            let previous = without_resume(&serde_json::from_slice(&fs::read(&saved_path)?)?)?;
            let mut changed: Vec<&String> =
                expected.keys().chain(previous.keys()).filter(|k| expected.get(*k) != previous.get(*k)).collect();
            changed.sort();
            changed.dedup();
            if !changed.is_empty() {
                let changed: Vec<&str> = changed.into_iter().map(String::as_str).collect();
                bail!(
                    "--resume: config differs from {} in [{}]. Use the saved config, start a fresh run, or set \
                     --resume.skip-checks if the changes are intentional.",
                    saved_path.display(),
                    changed.join(", ")
                );
            }
        }
        if directory.join(stream).is_dir() {
            for record in read_records(&directory.join(stream))? {
                let has_traces = record.get("traces").is_some();
                let episode = Episode::from_wire(&record)
                    .map_err(|e| anyhow::anyhow!("invalid saved episode at {}: {}", e.path, e.message))?;
                if episode.ok && has_traces {
                    landed.push(episode);
                }
            }
        }
    }
    if current.is_dir() {
        fs::rename(&current, current.with_file_name(format!("file.attempt_{}", archives(run_dir)?.len() + 1)))?;
    }
    Ok(landed)
}
