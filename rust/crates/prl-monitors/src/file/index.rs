//! The episode index (`monitors/file/traces/index.py`): one compact row per episode, written
//! as each episode lands, with the chunk and byte offset of its record. Arithmetic follows
//! Python's (int/float typing, 3.12's compensated `sum`) so rows match byte for byte.

use std::collections::HashSet;

use indexmap::IndexMap;
use prl_wire::{Map, Value};

/// The stop conditions `verifiers.v1.Trace.is_truncated` counts as truncation.
const TRUNCATING_STOPS: [&str; 6] = [
    "max_turns",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "max_turn_tokens",
    "compaction_failed",
];

/// A Python number: sums of ints stay ints, anything touching a float is a float.
#[derive(Clone, Copy)]
enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn of(value: &Value) -> Option<Num> {
        match value {
            Value::Int(i) => Some(Num::Int(*i)),
            other => other.as_f64().map(Num::Float),
        }
    }

    fn f(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
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
        match *value {
            Num::Float(x) => {
                let t = total + x;
                c += if total.abs() >= x.abs() {
                    (total - t) + x
                } else {
                    (x - t) + total
                };
                total = t;
            }
            Num::Int(i) => total += i as f64,
        }
    }
    if c != 0.0 && c.is_finite() {
        total += c;
    }
    Num::Float(total)
}

/// Python's `sum()` over floats.
pub(crate) fn py_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    sum(&values.into_iter().map(Num::Float).collect::<Vec<_>>()).f()
}

fn mean(values: &[Num]) -> Value {
    if values.is_empty() {
        Value::Nil
    } else {
        Value::F64(sum(values).f() / values.len() as f64)
    }
}

/// `d.get(key) or {}`: a falsy value reads as missing.
fn get<'a>(obj: &'a Value, key: &str) -> Option<&'a Value> {
    obj.get(key).filter(|v| v.truthy())
}

fn get_or_nil(obj: &Value, key: &str) -> Value {
    obj.get(key).cloned().unwrap_or_default()
}

fn items<'a>(obj: &'a Value, key: &str) -> &'a [Value] {
    get(obj, key)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// Flatten a trace timing tree to phase -> seconds (same walk as the viewer).
fn walk_timing(obj: &Map, prefix: &str, out: &mut IndexMap<String, f64>) {
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

/// The kind of work an episode did: stamped on its traces as it lands, else read off its run.
fn episode_kind(rec: &Value) -> Value {
    for trace in items(rec, "traces") {
        let kind = trace
            .get("info")
            .and_then(|i| i.get("kind"))
            .and_then(Value::as_str);
        if let Some(kind @ ("train" | "eval")) = kind {
            return kind.into();
        }
    }
    let run = get(rec, "run");
    let work_type = run
        .and_then(|r| get(r, "work"))
        .and_then(|w| get(w, "type"));
    work_type
        .or_else(|| run.and_then(|r| get(r, "type")))
        .cloned()
        .unwrap_or_else(|| "eval".into())
}

/// `Trace.is_truncated` for a dumped trace.
fn trace_truncated(trace: &Value) -> bool {
    let stop = trace.get("stop_condition").and_then(Value::as_str);
    stop.is_some_and(|s| TRUNCATING_STOPS.contains(&s))
        || items(trace, "calls")
            .iter()
            .rev()
            .find(|call| call.get("error").is_none_or(Value::is_nil))
            .is_some_and(|call| call.get("finish_reason").and_then(Value::as_str) == Some("length"))
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
        Value::F64(py_sum(phases))
    }
}

/// The index row of the episode record `rec`, the `line`-th of its stream, written at
/// `offset` of chunk `chunk` (`index_row` in Python).
pub fn index_row(line: u64, rec: &Value, chunk: u64, offset: u64) -> Map {
    let mut rewards = Vec::new();
    let mut advantages = Vec::new();
    let (mut turns, mut branches) = (0i64, 0i64);
    let mut stop_condition = Value::Nil;
    let mut truncated = false;
    let mut timeout = rec.get("is_timeout").is_some_and(Value::truthy);
    let mut timing = IndexMap::new();
    let mut costs = Vec::new();
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
        // reward-less seats (e.g. a judge) stay out of the episode mean
        if let Some(trace_rewards) = get(trace, "rewards").and_then(Value::as_map) {
            let weighted: Vec<Num> = trace_rewards
                .values()
                .filter(|r| matches!(r, Value::Map(_)))
                .map(|r| {
                    let score = get(r, "score").and_then(Num::of).unwrap_or(Num::Int(0));
                    let weight = r.get("weight").and_then(Num::of).unwrap_or(Num::Int(1));
                    match (score, weight) {
                        (Num::Int(s), Num::Int(w)) => Num::Int(s * w),
                        (s, w) => Num::Float(s.f() * w.f()),
                    }
                })
                .collect();
            rewards.push(sum(&weighted));
        }
        if let Some(Value::Map(t)) = trace.get("timing") {
            walk_timing(t, "", &mut timing);
        }
        if let Some(advantage) = get(trace, "info").and_then(|i| i.get("advantage")) {
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
    let first_info = items(rec, "traces")
        .first()
        .and_then(|t| get(t, "info"))
        .unwrap_or(&empty);
    let stamp_time = |key: &str| get(first_info, key).map_or(Value::Nil, |s| get_or_nil(s, "time"));
    let env = get(rec, "env");
    let env_name = env
        .and_then(|e| get(e, "name"))
        .cloned()
        .or_else(|| env.map(|e| get_or_nil(e, "id")));
    let trace_ids: Vec<Value> = items(rec, "traces")
        .iter()
        .filter_map(|t| get(t, "id"))
        .cloned()
        .collect();
    let cost = if costs.is_empty() {
        Value::Nil
    } else {
        sum(&costs).value()
    };
    [
        ("cost", cost),
        ("line", Value::Int(line as i64)),
        ("offset", Value::Int(offset as i64)),
        ("id", get_or_nil(rec, "id")),
        ("kind", episode_kind(rec)),
        ("trace_ids", trace_ids.into()),
        ("env", env_name.unwrap_or_default()),
        (
            "group",
            get(rec, "group").map_or(Value::Nil, |g| get_or_nil(g, "id")),
        ),
        ("ok", get_or_nil(rec, "ok")),
        ("num_errors", Value::Int(items(rec, "errors").len() as i64)),
        ("reward", mean(&rewards)),
        ("advantage", mean(&advantages)),
        ("input_tokens", get_or_nil(rec, "num_input_tokens")),
        ("output_tokens", get_or_nil(rec, "num_output_tokens")),
        ("turns", Value::Int(turns)),
        ("branches", Value::Int(branches)),
        ("stop_condition", stop_condition),
        ("truncated", Value::Bool(truncated)),
        ("timeout", Value::Bool(timeout)),
        ("dispatch", stamp_time("dispatch")),
        ("arrival", stamp_time("arrival")),
        ("duration", episode_duration(first_info, &timing)),
        ("chunk", Value::Int(chunk as i64)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}
