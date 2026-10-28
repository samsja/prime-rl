//! Trace updates (`monitors/file/traces/update.py`): post-hoc facts about a logged trace,
//! appended to the producer's annotation stream.

use prl_wire::{Map, Value};

pub const UPDATE_VERSION: i64 = 1;

/// One update record. `branches` maps a branch index to its token streams.
pub fn make_update(trace_id: &str, info: Option<Map>, branches: Vec<(i64, Map)>) -> Value {
    let branches: Vec<Value> = branches
        .into_iter()
        .map(|(index, streams)| {
            let mut branch = Map::new();
            branch.insert("index".into(), Value::Int(index));
            branch.extend(streams);
            Value::Map(branch)
        })
        .collect();
    [
        ("version", Value::Int(UPDATE_VERSION)),
        ("trace_id", trace_id.into()),
        ("info", Value::Map(info.unwrap_or_default())),
        ("branches", branches.into()),
    ]
    .into_iter()
    .collect()
}

/// An update's scalars plus where its record sits.
pub fn update_index_row(update: &Value, chunk: u64, offset: u64) -> Value {
    let info = update
        .get("info")
        .filter(|i| i.truthy())
        .cloned()
        .unwrap_or_else(Value::map);
    [
        (
            "trace_id",
            update.get("trace_id").cloned().unwrap_or_default(),
        ),
        ("chunk", Value::Int(chunk as i64)),
        ("offset", Value::Int(offset as i64)),
        ("info", info),
    ]
    .into_iter()
    .collect()
}
