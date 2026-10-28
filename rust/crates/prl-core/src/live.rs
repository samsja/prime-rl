//! The dispatcher's side of the live traces (`live.py`): the phase and turn count of each
//! in-flight trace, and the events relayed to the monitors.

use indexmap::IndexMap;
use prl_monitors::file::live::{stage, LiveEvent, STAGES};
use prl_types::Kind;
use prl_wire::{Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct LiveTrace {
    pub stage: &'static str,
    pub turns: usize,
}

impl Default for LiveTrace {
    fn default() -> Self {
        Self { stage: "pending", turns: 0 }
    }
}

/// Who an in-flight episode is, stamped on its live traces and pending placeholder.
#[derive(Debug, Clone)]
pub struct DispatchInfo {
    pub id: String,
    pub kind: Kind,
    pub env: String,
    pub group: String,
    /// `str(data.name)` if truthy, else `idx={data.idx}`.
    pub task: String,
    pub policy_version: u64,
    pub step: u64,
    /// Wall-clock time of dispatch.
    pub started: f64,
}

impl DispatchInfo {
    pub fn to_value(&self) -> Value {
        let mut m = Map::new();
        m.insert("id".into(), self.id.as_str().into());
        m.insert("kind".into(), self.kind.as_str().into());
        m.insert("env".into(), self.env.as_str().into());
        m.insert("group".into(), self.group.as_str().into());
        m.insert("task".into(), self.task.as_str().into());
        m.insert("policy_version".into(), self.policy_version.into());
        m.insert("step".into(), self.step.into());
        m.insert("started".into(), self.started.into());
        Value::Map(m)
    }
}

/// The live label of a task: its data's `name` when truthy, else `idx={idx}`.
pub fn task_label(data: &serde_json::Value) -> String {
    use serde_json::Value as J;
    let truthy = |v: &J| match v {
        J::Null => false,
        J::Bool(b) => *b,
        J::Number(n) => n.as_f64() != Some(0.0),
        J::String(s) => !s.is_empty(),
        J::Array(a) => !a.is_empty(),
        J::Object(o) => !o.is_empty(),
    };
    match data.get("name") {
        Some(J::String(s)) if !s.is_empty() => s.clone(),
        Some(v) if truthy(v) => v.to_string(),
        _ => match data.get("idx") {
            Some(J::Null) | None => "idx=None".to_string(),
            Some(v) => format!("idx={v}"),
        },
    }
}

pub fn pending_event(info: &DispatchInfo) -> LiveEvent {
    LiveEvent::Pending { id: info.id.clone(), dispatch: info.to_value() }
}

pub fn dispatched_event(info: &DispatchInfo) -> LiveEvent {
    LiveEvent::Dispatched(info.id.clone())
}

/// Fold one delta into an episode's phase/turn bookkeeping.
pub fn apply(live: &mut IndexMap<String, LiveTrace>, delta: &Value) {
    let Some(trace_id) = delta.get("trace").and_then(Value::as_str) else { return };
    if delta.get("discard").is_some_and(Value::truthy) {
        live.shift_remove(trace_id);
        return;
    }
    let trace = live.entry(trace_id.to_owned()).or_default();
    trace.turns += delta.get("calls").and_then(Value::as_array).map_or(0, Vec::len);
    let changed = delta.get("set");
    if delta.get("errors").is_some_and(Value::truthy) {
        trace.stage = "error";
    } else if let Some(timing) = changed.and_then(|c| c.get("timing")) {
        let mut m = Map::new();
        m.insert("timing".into(), timing.clone());
        trace.stage = stage(&Value::Map(m));
    }
}

/// `boot 1 · running 3`: in-flight traces by phase in lifecycle order; an episode with no
/// trace yet counts as pending.
pub fn stage_counts<'a>(episodes: impl IntoIterator<Item = &'a IndexMap<String, LiveTrace>>) -> String {
    let mut counts: IndexMap<&str, usize> = STAGES.iter().map(|s| (*s, 0)).collect();
    for live in episodes {
        if live.is_empty() {
            counts["pending"] += 1;
        }
        for trace in live.values() {
            *counts.entry(trace.stage).or_default() += 1;
        }
    }
    counts.iter().filter(|(_, n)| **n > 0).map(|(name, n)| format!("{name} {n}")).collect::<Vec<_>>().join(" · ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(pairs: Vec<(&str, Value)>) -> Value {
        Value::Map(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    #[test]
    fn apply_tracks_turns_stage_and_discards() {
        let mut live = IndexMap::new();
        apply(&mut live, &delta(vec![("trace", "t1".into()), ("calls", Value::Array(vec![Value::map(), Value::map()]))]));
        assert_eq!(live["t1"], LiveTrace { stage: "pending", turns: 2 });
        let mut timing = Map::new();
        let mut agent = Map::new();
        agent.insert("start".into(), 1.0.into());
        timing.insert("agent".into(), Value::Map(agent));
        let mut set = Map::new();
        set.insert("timing".into(), Value::Map(timing));
        apply(&mut live, &delta(vec![("trace", "t1".into()), ("set", Value::Map(set))]));
        assert_eq!(live["t1"].stage, "running");
        apply(&mut live, &delta(vec![("trace", "t2".into()), ("errors", Value::Array(vec![Value::map()]))]));
        assert_eq!(live["t2"].stage, "error");
        assert_eq!(stage_counts([&live, &IndexMap::new()]), "pending 1 · running 1 · error 1");
        apply(&mut live, &delta(vec![("trace", "t2".into()), ("discard", true.into())]));
        assert!(!live.contains_key("t2"));
    }

    #[test]
    fn task_label_prefers_name() {
        assert_eq!(task_label(&serde_json::json!({"name": "foo", "idx": 3})), "foo");
        assert_eq!(task_label(&serde_json::json!({"name": "", "idx": 3})), "idx=3");
        assert_eq!(task_label(&serde_json::json!({"idx": 3})), "idx=3");
    }
}
