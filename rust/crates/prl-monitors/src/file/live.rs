//! Live traces (`monitors/file/traces/live.py`): one file of deltas per in-flight trace under
//! `traces/live/`, plus a placeholder per dispatched episode under `traces/live/pending/`.

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use indexmap::IndexMap;
use prl_wire::{Map, Value};

pub const STAGES: [&str; 8] = [
    "pending", "boot", "setup", "running", "finalize", "scoring", "done", "error",
];

/// Node fields that stay on the wire but out of live trace files.
pub const LIVE_NODE_EXCLUDE: [&str; 3] = ["routed_experts", "sampling_mask", "top_logprobs"];

/// One event of the dispatcher's live stream.
#[derive(Debug, Clone)]
pub enum LiveEvent {
    /// A delta as it arrived, stamped with who dispatched its episode.
    Delta { delta: Value, dispatch: Value },
    /// The trace's episode finished: its live file goes away.
    Done(String),
    /// An episode was dispatched; a placeholder until its first trace streams.
    Pending { id: String, dispatch: Value },
    /// The episode's placeholder is no longer needed.
    Dispatched(String),
}

pub fn live_dir(output_dir: &Path) -> PathBuf {
    super::trace_dir(output_dir).join("live")
}

pub fn pending_dir(output_dir: &Path) -> PathBuf {
    live_dir(output_dir).join("pending")
}

/// The phase a trace is in, read off its timing spans the way the rollout sets them.
pub fn stage(trace: &Value) -> &'static str {
    if trace.get("errors").is_some_and(Value::truthy) {
        return "error";
    }
    let empty = Value::map();
    let timing = trace.get("timing").filter(|t| t.truthy()).unwrap_or(&empty);
    let span = |name: &str, field: &str| {
        timing
            .get(name)
            .filter(|s| s.truthy())
            .and_then(|s| s.get(field))
            .is_some_and(Value::truthy)
    };
    if span("scoring", "end") {
        return "done";
    }
    for (span_name, stage_name) in [
        ("scoring", "scoring"),
        ("finalize", "finalize"),
        ("agent", "running"),
        ("setup", "setup"),
        ("boot", "boot"),
    ] {
        if span(span_name, "start") {
            return stage_name;
        }
    }
    "pending"
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

/// A delta as its live file line: heavy node fields dropped, ndarrays decoded, and the
/// dispatch identity on the trace's first (`open`) line.
fn live_line(delta: &Map, dispatch: &Value) -> Result<Vec<u8>> {
    let mut line = delta.clone();
    if let Some(Value::Array(nodes)) = line.get("nodes") {
        let nodes: Vec<Value> = nodes
            .iter()
            .map(|node| -> Result<Value> {
                let mut out = Map::new();
                for (key, value) in node.as_map().into_iter().flatten() {
                    if !LIVE_NODE_EXCLUDE.contains(&key.as_str()) {
                        out.insert(key.clone(), prl_wire::ndarray::live_value(value)?);
                    }
                }
                Ok(Value::Map(out))
            })
            .collect::<Result<_>>()?;
        line.insert("nodes".into(), Value::Array(nodes));
    }
    if delta.contains_key("open") {
        line.insert("dispatch".into(), dispatch.clone());
    }
    let mut bytes = prl_wire::json::to_vec(&Value::Map(line));
    bytes.push(b'\n');
    Ok(bytes)
}

/// Applies live events to `traces/live/`.
pub struct LiveWriter {
    live_dir: PathBuf,
    pending_dir: PathBuf,
    cleared: bool,
}

impl LiveWriter {
    pub fn new(output_dir: &Path) -> Self {
        LiveWriter {
            live_dir: live_dir(output_dir),
            pending_dir: pending_dir(output_dir),
            cleared: false,
        }
    }

    pub fn write(&mut self, events: &[LiveEvent]) -> Result<()> {
        // A previous attempt's live traces are stale: nothing streams into them again.
        if !self.cleared {
            self.cleared = true;
            match fs::remove_dir_all(&self.live_dir) {
                Err(e) if e.kind() != ErrorKind::NotFound => {
                    tracing::debug!("clearing {}: {e}", self.live_dir.display())
                }
                _ => {}
            }
        }
        fs::create_dir_all(&self.pending_dir)?;
        let mut appends: IndexMap<PathBuf, Vec<Vec<u8>>> = IndexMap::new();
        for event in events {
            match event {
                LiveEvent::Done(trace_id) => {
                    let path = self.live_dir.join(format!("{trace_id}.jsonl"));
                    appends.shift_remove(&path);
                    remove_if_exists(&path)?;
                }
                LiveEvent::Pending { id, dispatch } => {
                    let path = self.pending_dir.join(format!("{id}.json"));
                    let tmp = self.pending_dir.join(format!("{id}.json.tmp"));
                    fs::write(&tmp, prl_wire::json::to_vec(dispatch))?;
                    fs::rename(&tmp, &path)?;
                }
                LiveEvent::Dispatched(id) => {
                    remove_if_exists(&self.pending_dir.join(format!("{id}.json")))?
                }
                LiveEvent::Delta { delta, dispatch } => {
                    let delta = delta
                        .as_map()
                        .ok_or_else(|| anyhow::anyhow!("live delta is not a map"))?;
                    let trace = match delta.get("trace") {
                        Some(Value::Str(s)) => s.clone(),
                        other => anyhow::bail!("live delta without a trace id: {other:?}"),
                    };
                    let path = self.live_dir.join(format!("{trace}.jsonl"));
                    if delta.get("discard").is_some_and(Value::truthy) {
                        appends.shift_remove(&path);
                        remove_if_exists(&path)?;
                        continue;
                    }
                    appends
                        .entry(path)
                        .or_default()
                        .push(live_line(delta, dispatch)?);
                }
            }
        }
        // one open per file per batch
        for (path, lines) in appends {
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?;
            let mut buf = Vec::with_capacity(lines.iter().map(|l| l.len() + 1).sum());
            for line in lines {
                buf.extend_from_slice(&line);
                buf.push(b'\n');
            }
            file.write_all(&buf)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(timing: Value) -> Value {
        [("timing", timing)].into_iter().collect()
    }

    fn span(fields: &[(&str, f64)]) -> Value {
        fields.iter().map(|(k, v)| (*k, Value::F64(*v))).collect()
    }

    #[test]
    fn stage_reads_timing_spans() {
        assert_eq!(stage(&Value::map()), "pending");
        assert_eq!(
            stage(&trace(
                [("boot", span(&[("start", 1.0)]))].into_iter().collect()
            )),
            "boot"
        );
        assert_eq!(
            stage(&trace(
                [
                    ("boot", span(&[("start", 1.0)])),
                    ("agent", span(&[("start", 2.0)]))
                ]
                .into_iter()
                .collect()
            )),
            "running"
        );
        assert_eq!(
            stage(&trace(
                [("scoring", span(&[("start", 1.0), ("end", 2.0)]))]
                    .into_iter()
                    .collect()
            )),
            "done"
        );
        let errored: Value = [("errors", Value::Array(vec![Value::map()]))]
            .into_iter()
            .collect();
        assert_eq!(stage(&errored), "error");
    }
}
