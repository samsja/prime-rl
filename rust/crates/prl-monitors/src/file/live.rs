//! Live traces (`monitors/file/traces/live.py`): one file of deltas per in-flight trace under
//! `traces/live/`, plus a placeholder per dispatched episode under `traces/live/pending/`.

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use indexmap::IndexMap;
use prl_wire::{Map, Value};

pub const STAGES: [&str; 8] = [
    "pending", "boot", "setup", "running", "finalize", "scoring", "done", "error",
];

/// Node fields that stay on the wire but out of live trace files.
const LIVE_NODE_EXCLUDE: [&str; 3] = ["routed_experts", "sampling_mask", "top_logprobs"];

/// One event of the dispatcher's live stream.
#[derive(Debug, Clone)]
pub enum LiveEvent {
    /// A delta as it arrived, stamped with who dispatched its episode.
    Delta { delta: Value, dispatch: Value },
    /// The trace's episode finished: its live file goes away.
    Done(String),
    /// An episode was dispatched: a placeholder until its first trace streams.
    Pending { id: String, dispatch: Value },
    /// The episode's placeholder is no longer needed.
    Dispatched(String),
}

/// The phase a trace is in, read off its timing spans the way the rollout sets them.
pub fn stage(trace: &Value) -> &'static str {
    if trace.get("errors").is_some_and(Value::truthy) {
        return "error";
    }
    let span = |name: &str, field: &str| {
        trace
            .get("timing")
            .and_then(|t| t.get(name))
            .and_then(|s| s.get(field))
            .is_some_and(Value::truthy)
    };
    if span("scoring", "end") {
        return "done";
    }
    let spans = [
        ("scoring", "scoring"),
        ("finalize", "finalize"),
        ("agent", "running"),
        ("setup", "setup"),
        ("boot", "boot"),
    ];
    spans
        .into_iter()
        .find(|(name, _)| span(name, "start"))
        .map_or("pending", |(_, stage)| stage)
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
    if let Some(Value::Array(nodes)) = delta.get("nodes") {
        let mut live_nodes = Vec::with_capacity(nodes.len());
        for node in nodes {
            let mut fields = Map::new();
            for (key, value) in node.as_map().into_iter().flatten() {
                if !LIVE_NODE_EXCLUDE.contains(&key.as_str()) {
                    fields.insert(key.clone(), prl_wire::ndarray::live_value(value)?);
                }
            }
            live_nodes.push(Value::Map(fields));
        }
        line.insert("nodes".into(), Value::Array(live_nodes));
    }
    if delta.contains_key("open") {
        line.insert("dispatch".into(), dispatch.clone());
    }
    let mut bytes = prl_wire::json::to_vec(&Value::Map(line));
    // Python's writer appends a newline per line on top of orjson's: kept for byte parity
    bytes.extend_from_slice(b"\n\n");
    Ok(bytes)
}

pub struct LiveWriter {
    live_dir: PathBuf,
    cleared: bool,
}

impl LiveWriter {
    pub fn new(live_dir: &Path) -> Self {
        LiveWriter {
            live_dir: live_dir.to_path_buf(),
            cleared: false,
        }
    }

    pub fn write(&mut self, events: &[LiveEvent]) -> Result<()> {
        // A previous attempt's live traces are stale: nothing streams into them again.
        if !std::mem::replace(&mut self.cleared, true) {
            let _ = fs::remove_dir_all(&self.live_dir);
        }
        let pending_dir = self.live_dir.join("pending");
        fs::create_dir_all(&pending_dir)?;
        let mut appends: IndexMap<PathBuf, Vec<u8>> = IndexMap::new();
        for event in events {
            match event {
                LiveEvent::Done(trace_id) => {
                    let path = self.live_dir.join(format!("{trace_id}.jsonl"));
                    appends.shift_remove(&path);
                    remove_if_exists(&path)?;
                }
                LiveEvent::Pending { id, dispatch } => {
                    let tmp = pending_dir.join(format!("{id}.json.tmp"));
                    fs::write(&tmp, prl_wire::json::to_vec(dispatch))?;
                    fs::rename(&tmp, pending_dir.join(format!("{id}.json")))?;
                }
                LiveEvent::Dispatched(id) => {
                    remove_if_exists(&pending_dir.join(format!("{id}.json")))?
                }
                LiveEvent::Delta {
                    delta: Value::Map(delta),
                    dispatch,
                } => {
                    let Some(Value::Str(trace)) = delta.get("trace") else {
                        bail!("live delta without a trace id")
                    };
                    let path = self.live_dir.join(format!("{trace}.jsonl"));
                    if delta.get("discard").is_some_and(Value::truthy) {
                        appends.shift_remove(&path);
                        remove_if_exists(&path)?;
                    } else {
                        appends
                            .entry(path)
                            .or_default()
                            .extend(live_line(delta, dispatch)?);
                    }
                }
                LiveEvent::Delta { delta, .. } => bail!("live delta is not a map: {delta:?}"),
            }
        }
        // one open per file per batch
        for (path, lines) in appends {
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?
                .write_all(&lines)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(spans: &[(&str, &str)]) -> Value {
        let timing: Value = spans
            .iter()
            .map(|(span, field)| {
                (
                    *span,
                    [(*field, Value::F64(1.0))].into_iter().collect::<Value>(),
                )
            })
            .collect();
        [("timing", timing)].into_iter().collect()
    }

    #[test]
    fn stage_reads_timing_spans() {
        assert_eq!(stage(&Value::map()), "pending");
        assert_eq!(stage(&trace(&[("boot", "start")])), "boot");
        assert_eq!(
            stage(&trace(&[("boot", "start"), ("agent", "start")])),
            "running"
        );
        assert_eq!(stage(&trace(&[("scoring", "end")])), "done");
        assert_eq!(
            stage(
                &[("errors", Value::Array(vec![Value::map()]))]
                    .into_iter()
                    .collect()
            ),
            "error"
        );
    }
}
