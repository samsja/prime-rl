//! Serve protocol messages (`verifiers/v1/serve/types.py`).

use crate::assembly::TraceSummary;
use crate::value::{Map, Value};

pub const METHOD_HEALTH: &str = "health";
pub const METHOD_RUN: &str = "run";
pub const METHOD_CANCEL: &str = "cancel";
pub const KIND_DELTA: &[u8] = b"delta";
pub const KIND_REPLY: &[u8] = b"reply";

/// `BaseResponse` fields every reply carries, plus the run reply's.
#[derive(Clone, Debug, Default)]
pub struct Response {
    pub success: bool,
    pub error: Option<String>,
    /// `RunResponse.head`: the episode without its traces.
    pub head: Option<Map>,
    /// `RunResponse.traces`.
    pub traces: Vec<TraceSummary>,
    /// `CancelResponse.cancelled`.
    pub cancelled: bool,
}

impl Response {
    pub fn from_value(value: &Value) -> Result<Response, String> {
        let Value::Map(map) = value else { return Err(format!("reply is not a dict: {value:?}")) };
        let success = match map.get("success") {
            None => true,
            Some(Value::Bool(b)) => *b,
            Some(other) => return Err(format!("success is not a bool: {other:?}")),
        };
        let error = match map.get("error") {
            None | Some(Value::Nil) => None,
            Some(Value::Str(s)) => Some(s.clone()),
            Some(other) => return Err(format!("error is not a str: {other:?}")),
        };
        let head = match map.get("head") {
            None | Some(Value::Nil) => None,
            Some(Value::Map(m)) => Some(m.clone()),
            Some(other) => return Err(format!("head is not a dict: {other:?}")),
        };
        let mut traces = Vec::new();
        for summary in map.get("traces").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default() {
            let id = summary.get("id").and_then(Value::as_str).ok_or("trace summary without id")?;
            let count = |k: &str| summary.get(k).and_then(Value::as_u64).ok_or(format!("trace summary without {k}"));
            traces.push(TraceSummary { id: id.to_owned(), nodes: count("nodes")? as usize, calls: count("calls")? as usize });
        }
        let cancelled = map.get("cancelled").and_then(Value::as_bool).unwrap_or(false);
        Ok(Response { success, error, head, traces, cancelled })
    }
}
