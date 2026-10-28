//! The episode index (`monitors/file/traces/index.py`): one compact row per episode,
//! written as each episode lands, with the chunk and byte offset of its record.

use std::collections::HashSet;

use indexmap::IndexMap;
use prl_wire::{Map, Value};

/// The stop conditions `verifiers.v1.Trace.is_truncated` counts as truncation.
pub const TRUNCATING_STOPS: [&str; 6] = [
    "max_turns",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "max_turn_tokens",
    "compaction_failed",
];

/// A Python number: sums of ints stay ints, anything touching a float is a float.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn of(value: &Value) -> Option<Num> {
        match value {
            Value::Int(i) => Some(Num::Int(*i)),
            Value::UInt(u) => Some(Num::Float(*u as f64)),
            Value::F32(f) => Some(Num::Float(*f as f64)),
            Value::F64(f) => Some(Num::Float(*f)),
            _ => None,
        }
    }
    fn f(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
    fn mul(self, other: Num) -> Num {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) => Num::Int(a * b),
            (a, b) => Num::Float(a.f() * b.f()),
        }
    }
    fn value(self) -> Value {
        match self {
            Num::Int(i) => Value::Int(i),
            Num::Float(f) => Value::F64(f),
        }
    }
}

/// Python 3.12 `sum(values)` (start 0): ints add exactly until the first float, which adds
/// plainly; later floats add Neumaier-compensated, later ints plainly.
fn sum(values: &[Num]) -> Num {
    let mut iter = values.iter();
    let mut int_total = 0i64;
    let mut total = loop {
        match iter.next() {
            None => return Num::Int(int_total),
            Some(Num::Int(i)) => int_total += i,
            Some(Num::Float(f)) => break int_total as f64 + f,
        }
    };
    let mut c = 0.0;
    for value in iter {
        match value {
            Num::Float(x) => {
                let t = total + x;
                if total.abs() >= x.abs() {
                    c += (total - t) + x;
                } else {
                    c += (x - t) + total;
                }
                total = t;
            }
            Num::Int(i) => total += *i as f64,
        }
    }
    if c != 0.0 && c.is_finite() {
        total += c;
    }
    Num::Float(total)
}

fn mean(values: &[Num]) -> Value {
    if values.is_empty() {
        Value::Nil
    } else {
        Value::F64(sum(values).f() / values.len() as f64)
    }
}

/// `d.get(key) or {}` / `or []`: a falsy value reads as empty.
fn get<'a>(obj: &'a Value, key: &str) -> Option<&'a Value> {
    obj.get(key).filter(|v| v.truthy())
}

fn get_or_nil(obj: &Value, key: &str) -> Value {
    obj.get(key).cloned().unwrap_or(Value::Nil)
}

fn items<'a>(obj: &'a Value, key: &str) -> &'a [Value] {
    get(obj, key)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// Flatten a trace timing tree to phase -> seconds (same walk as the viewer).
pub fn walk_timing(obj: &Map, prefix: &str, out: &mut IndexMap<String, f64>) {
    let num = |key: &str| obj.get(key).and_then(Num::of);
    if let Some(duration) = num("duration") {
        *out.entry(prefix.to_string()).or_insert(0.0) += duration.f();
    } else if let (Some(start), Some(end)) = (num("start"), num("end")) {
        *out.entry(prefix.to_string()).or_insert(0.0) += end.f() - start.f();
    }
    for (key, value) in obj {
        if let Value::Map(child) = value {
            let path = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}/{key}")
            };
            walk_timing(child, &path, out);
        }
    }
}

/// The kind of work an episode did: stamped on its traces as it lands, else read off its run info.
pub fn episode_kind(rec: &Value) -> Value {
    for trace in items(rec, "traces") {
        if let Some(kind) = trace
            .get("info")
            .and_then(|i| i.get("kind"))
            .and_then(Value::as_str)
        {
            if kind == "train" || kind == "eval" {
                return kind.into();
            }
        }
    }
    let run = get(rec, "run");
    let work_type = run
        .and_then(|r| get(r, "work"))
        .and_then(|w| get(w, "type"));
    let run_type = run.and_then(|r| get(r, "type"));
    work_type
        .or(run_type)
        .cloned()
        .unwrap_or_else(|| "eval".into())
}

/// `Trace.is_truncated` for a dumped trace.
pub fn trace_truncated(trace: &Value) -> bool {
    if let Some(stop) = trace.get("stop_condition").and_then(Value::as_str) {
        if TRUNCATING_STOPS.contains(&stop) {
            return true;
        }
    }
    items(trace, "calls")
        .iter()
        .rev()
        .find(|call| call.get("error").is_none_or(Value::is_nil))
        .is_some_and(|call| call.get("finish_reason").and_then(Value::as_str) == Some("length"))
}

/// One index row (`summarize_episode`): the reward/metric/timing maps included.
pub fn summarize_episode(line: u64, rec: &Value, offset: Option<u64>) -> Map {
    let mut rewards: Vec<Num> = Vec::new();
    let mut advantages: Vec<Num> = Vec::new();
    let (mut turns, mut branches) = (0i64, 0i64);
    let mut stop_condition = Value::Nil;
    let mut truncated = false;
    let mut timeout = rec.get("is_timeout").is_some_and(Value::truthy);
    let mut reward_parts: IndexMap<String, Vec<Num>> = IndexMap::new();
    let mut metric_parts: IndexMap<String, Vec<Num>> = IndexMap::new();
    let mut timing: IndexMap<String, f64> = IndexMap::new();
    let mut costs: Vec<Num> = Vec::new();
    for trace in items(rec, "traces") {
        for call in items(trace, "calls") {
            if let Some(usage @ Value::Map(_)) = get(call, "usage") {
                costs.extend(usage.get("cost").and_then(Num::of));
            }
        }
        let nodes = items(trace, "nodes");
        let parents: HashSet<Vec<u8>> = nodes
            .iter()
            .filter_map(|n| n.get("parent"))
            .map(prl_wire::json::to_vec)
            .collect();
        branches += (nodes.len() as i64 - parents.len() as i64).max(0);
        if let Some(trace_rewards) = get(trace, "rewards").and_then(Value::as_map) {
            let products: Vec<Num> = trace_rewards
                .values()
                .filter(|r| matches!(r, Value::Map(_)))
                .map(|r| {
                    let score = get(r, "score").and_then(Num::of).unwrap_or(Num::Int(0));
                    let weight = r
                        .get("weight")
                        .filter(|w| !w.is_nil())
                        .and_then(Num::of)
                        .unwrap_or(Num::Int(1));
                    score.mul(weight)
                })
                .collect();
            rewards.push(sum(&products));
            for (name, r) in trace_rewards {
                if let Some(score) = r.get("score").and_then(Num::of) {
                    reward_parts.entry(name.clone()).or_default().push(score);
                }
            }
        }
        if let Some(metrics) = get(trace, "metrics").and_then(Value::as_map) {
            for (name, value) in metrics {
                if let Some(v) = Num::of(value) {
                    metric_parts.entry(name.clone()).or_default().push(v);
                }
            }
        }
        if let Some(Value::Map(t)) = trace.get("timing") {
            walk_timing(t, "", &mut timing);
        }
        if let Some(advantage) = get(trace, "info")
            .and_then(|i| i.get("advantage"))
            .filter(|a| !a.is_nil())
        {
            advantages.extend(Num::of(advantage));
        }
        turns += nodes
            .iter()
            .filter(|n| n.get("sampled").is_some_and(Value::truthy))
            .count() as i64;
        if let Some(stop) = trace.get("stop_condition") {
            stop_condition = stop.clone();
        }
        truncated = truncated || trace_truncated(trace);
        timeout = timeout || trace.get("is_timeout").is_some_and(Value::truthy);
    }
    let empty = Value::map();
    let first_info = items(rec, "traces").first().unwrap_or(&empty);
    let first_info = get(first_info, "info").unwrap_or(&empty);
    let stamp_time = |key: &str| get(first_info, key).map_or(Value::Nil, |s| get_or_nil(s, "time"));
    let env = get(rec, "env");
    let env_name = env
        .and_then(|e| get(e, "name"))
        .cloned()
        .or_else(|| env.map(|e| get_or_nil(e, "id")));

    let mut row = Map::new();
    let means = |parts: IndexMap<String, Vec<Num>>| -> Value {
        parts
            .into_iter()
            .map(|(k, v)| (k, mean(&v)))
            .collect::<Map>()
            .into()
    };
    row.insert("rewards".into(), means(reward_parts));
    row.insert("metrics".into(), means(metric_parts));
    row.insert(
        "timing".into(),
        timing
            .iter()
            .map(|(k, v)| (k.clone(), Value::F64(*v)))
            .collect::<Map>()
            .into(),
    );
    row.insert(
        "cost".into(),
        if costs.is_empty() {
            Value::Nil
        } else {
            sum(&costs).value()
        },
    );
    row.insert("line".into(), Value::Int(line as i64));
    row.insert(
        "offset".into(),
        offset.map_or(Value::Nil, |o| Value::Int(o as i64)),
    );
    row.insert("id".into(), get_or_nil(rec, "id"));
    row.insert("kind".into(), episode_kind(rec));
    let trace_ids: Vec<Value> = items(rec, "traces")
        .iter()
        .filter_map(|t| get(t, "id"))
        .cloned()
        .collect();
    row.insert("trace_ids".into(), trace_ids.into());
    row.insert("env".into(), env_name.unwrap_or(Value::Nil));
    row.insert(
        "group".into(),
        get(rec, "group").map_or(Value::Nil, |g| get_or_nil(g, "id")),
    );
    row.insert("ok".into(), get_or_nil(rec, "ok"));
    row.insert(
        "num_errors".into(),
        Value::Int(items(rec, "errors").len() as i64),
    );
    row.insert("reward".into(), mean(&rewards));
    row.insert("advantage".into(), mean(&advantages));
    row.insert("input_tokens".into(), get_or_nil(rec, "num_input_tokens"));
    row.insert("output_tokens".into(), get_or_nil(rec, "num_output_tokens"));
    row.insert("turns".into(), Value::Int(turns));
    row.insert("branches".into(), Value::Int(branches));
    row.insert("stop_condition".into(), stop_condition);
    row.insert("truncated".into(), Value::Bool(truncated));
    row.insert("timeout".into(), Value::Bool(timeout));
    row.insert("dispatch".into(), stamp_time("dispatch"));
    row.insert("arrival".into(), stamp_time("arrival"));
    row.insert("duration".into(), episode_duration(first_info, &timing));
    row
}

/// Wall-clock seconds from dispatch to arrival, or the summed top-level phases.
fn episode_duration(info: &Value, timing: &IndexMap<String, f64>) -> Value {
    let time = |key: &str| get(info, key).and_then(|s| s.get("time")).and_then(Num::of);
    if let (Some(dispatched), Some(arrived)) = (time("dispatch"), time("arrival")) {
        // max(0.0, x) keeps x's type when positive
        let delta = match (arrived, dispatched) {
            (Num::Int(a), Num::Int(d)) => Num::Int(a - d),
            (a, d) => Num::Float(a.f() - d.f()),
        };
        return if delta.f() > 0.0 {
            delta.value()
        } else {
            Value::F64(0.0)
        };
    }
    let phases: Vec<f64> = timing
        .iter()
        .filter(|(k, _)| !k.contains('/'))
        .map(|(_, v)| *v)
        .collect();
    if phases.is_empty() {
        Value::Nil
    } else {
        Value::F64(crate::pyjson::py_sum(phases))
    }
}

/// The row the file monitor writes: without the nested maps, plus the chunk.
pub fn index_row(line: u64, rec: &Value, chunk: u64, offset: u64) -> Map {
    let mut row = summarize_episode(line, rec, Some(offset));
    row.insert("chunk".into(), Value::Int(chunk as i64));
    for key in ["rewards", "metrics", "timing"] {
        row.shift_remove(key);
    }
    row
}
