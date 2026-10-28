//! `EpisodeAssembly`: the client's growing picture of one served episode, built from the
//! env server's trace deltas (`verifiers/v1/serve/delta.py`).

use indexmap::IndexMap;

use crate::value::{Map, Value};

/// Append-only trace lists; each delta carries the items past the sent count.
pub const LIST_FIELDS: [&str; 6] = ["nodes", "calls", "errors", "extra_usage", "request_rewrites", "response_rewrites"];

/// A finished trace's identity and size, for the assembly check.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceSummary {
    pub id: String,
    pub nodes: usize,
    pub calls: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum AssemblyError {
    #[error("malformed delta: {0}")]
    Malformed(String),
    #[error("served episode is missing trace {0}")]
    MissingTrace(String),
    #[error("served trace {id} assembled {nodes} nodes / {calls} calls, server reports {expected_nodes} / {expected_calls}")]
    CountMismatch { id: String, nodes: usize, calls: usize, expected_nodes: usize, expected_calls: usize },
}

fn malformed<T>(message: impl Into<String>) -> Result<T, AssemblyError> {
    Err(AssemblyError::Malformed(message.into()))
}

/// Raw trace dicts in arrival order, each shaped like a dumped `Trace` plus its `pending`
/// preview.
#[derive(Clone, Debug, Default)]
pub struct EpisodeAssembly {
    pub traces: IndexMap<String, Map>,
}

impl EpisodeAssembly {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self, delta: &Value) -> Result<(), AssemblyError> {
        let Some(trace_id) = delta.get("trace").and_then(Value::as_str) else {
            return malformed("delta without a trace id");
        };
        if delta.get("discard").is_some_and(Value::truthy) {
            self.traces.shift_remove(trace_id);
            return Ok(());
        }
        if !self.traces.contains_key(trace_id) {
            let Some(Value::Map(open)) = delta.get("open") else {
                return malformed(format!("first delta of trace {trace_id} has no header"));
            };
            let mut trace = open.clone();
            for field in LIST_FIELDS {
                trace.insert(field.into(), Value::Array(Vec::new()));
            }
            trace.insert("pending".into(), Value::Array(Vec::new()));
            self.traces.insert(trace_id.to_owned(), trace);
        }
        let trace = self.traces.get_mut(trace_id).expect("trace inserted above");
        if let Some(links) = delta.get("links").filter(|l| !l.is_nil()) {
            let Some(links) = links.as_map() else { return malformed("links is not a dict") };
            for (index, new_links) in links {
                let Ok(index) = index.parse::<usize>() else { return malformed(format!("bad link index {index:?}")) };
                let Some(new_links) = new_links.as_array() else { return malformed("links entry is not a list") };
                let node = trace
                    .get_mut("nodes")
                    .and_then(Value::as_array_mut)
                    .and_then(|nodes| nodes.get_mut(index))
                    .ok_or_else(|| AssemblyError::Malformed(format!("link to unknown node {index}")))?;
                let Some(parents) = node.get_mut("semantic_parents").and_then(Value::as_array_mut) else {
                    return malformed(format!("node {index} has no semantic_parents list"));
                };
                parents.extend(new_links.iter().cloned());
            }
        }
        if let Some(nodes) = delta.get("nodes") {
            let Some(nodes) = nodes.as_array() else { return malformed("nodes is not a list") };
            let target = trace.get_mut("nodes").and_then(Value::as_array_mut).expect("nodes list");
            for node in nodes {
                let Value::Map(node) = node else { return malformed("node is not a dict") };
                let mut node = node.clone();
                let parents = node.get("semantic_parents").filter(|p| p.truthy()).cloned().unwrap_or(Value::Array(Vec::new()));
                node.insert("semantic_parents".into(), parents);
                target.push(Value::Map(node));
            }
        }
        for field in LIST_FIELDS {
            if field == "nodes" {
                continue;
            }
            if let Some(items) = delta.get(field) {
                let Some(items) = items.as_array() else { return malformed(format!("{field} is not a list")) };
                let target = trace.get_mut(field).and_then(Value::as_array_mut).expect("list field");
                target.extend(items.iter().cloned());
            }
        }
        if let Some(set) = delta.get("set").filter(|s| s.truthy()) {
            let Some(set) = set.as_map() else { return malformed("set is not a dict") };
            for (k, v) in set {
                trace.insert(k.clone(), v.clone());
            }
        }
        // a committed turn absorbs the preview; an explicit preview replaces it
        if delta.get("nodes").is_some() {
            trace.insert("pending".into(), Value::Array(Vec::new()));
        }
        if let Some(pending) = delta.get("pending") {
            trace.insert("pending".into(), pending.clone());
        }
        Ok(())
    }

    /// The full episode record: `head` (the episode without its traces) joined with the
    /// assembled traces in the server's order, without their previews.
    pub fn finish(&self, head: &Map, summaries: &[TraceSummary]) -> Result<Value, AssemblyError> {
        let mut traces = Vec::with_capacity(summaries.len());
        for summary in summaries {
            let Some(trace) = self.traces.get(&summary.id) else {
                return Err(AssemblyError::MissingTrace(summary.id.clone()));
            };
            let mut trace = trace.clone();
            trace.shift_remove("pending");
            let count = |field: &str| trace.get(field).and_then(Value::as_array).map_or(0, Vec::len);
            let (nodes, calls) = (count("nodes"), count("calls"));
            if nodes != summary.nodes || calls != summary.calls {
                return Err(AssemblyError::CountMismatch {
                    id: summary.id.clone(),
                    nodes,
                    calls,
                    expected_nodes: summary.nodes,
                    expected_calls: summary.calls,
                });
            }
            traces.push(Value::Map(trace));
        }
        let mut episode = head.clone();
        episode.insert("traces".into(), Value::Array(traces));
        Ok(Value::Map(episode))
    }
}
