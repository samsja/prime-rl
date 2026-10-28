//! The typed wire data model: `WireEpisode` = `Episode[WireTaskData, State, WireAgentConfig]`.
//!
//! The fields the orchestrator reads or writes are typed; the rest stay `Value`s exactly as
//! the env server dumped them (python mode), so records re-dump them like pydantic does.
//! Field order everywhere is the pydantic declaration order.

use std::cell::RefCell;
use std::sync::Arc;

use indexmap::IndexMap;

use crate::ndarray::{self, NdError};
use crate::schema::{self, SCHEMAS};
use crate::value::{Map, Value};

const TRACE_VERSION: i64 = 1;

/// A wire value that does not validate into the model (pydantic `ValidationError`).
#[derive(Debug, thiserror::Error)]
#[error("{path}: {message}")]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl From<NdError> for ValidationError {
    fn from(e: NdError) -> Self {
        ValidationError { path: String::new(), message: e.to_string() }
    }
}

type VResult<T> = Result<T, ValidationError>;

fn invalid<T>(path: &str, message: impl Into<String>) -> VResult<T> {
    Err(ValidationError { path: path.to_owned(), message: message.into() })
}

// ── field readers ──────────────────────────────────────────────────────────

struct Fields<'a> {
    map: &'a Map,
    path: String,
}

impl<'a> Fields<'a> {
    fn of(value: &'a Value, path: impl Into<String>) -> VResult<Fields<'a>> {
        let path = path.into();
        match value {
            Value::Map(map) => Ok(Fields { map, path }),
            other => invalid(&path, format!("expected a dict, got {other:?}")),
        }
    }

    fn path(&self, key: &str) -> String {
        format!("{}.{key}", self.path)
    }

    /// The value of a field; `None` when absent or null.
    fn get(&self, key: &str) -> Option<&'a Value> {
        self.map.get(key).filter(|v| !v.is_nil())
    }

    fn req(&self, key: &str) -> VResult<&'a Value> {
        match self.map.get(key) {
            Some(v) => Ok(v),
            None => invalid(&self.path(key), "field required"),
        }
    }

    fn string(&self, key: &str) -> VResult<String> {
        match self.req(key)? {
            Value::Str(s) => Ok(s.clone()),
            other => invalid(&self.path(key), format!("expected str, got {other:?}")),
        }
    }

    fn string_or_else(&self, key: &str, default: impl FnOnce() -> String) -> VResult<String> {
        if self.map.contains_key(key) { self.string(key) } else { Ok(default()) }
    }

    fn string_or(&self, key: &str, default: &str) -> VResult<String> {
        if self.map.contains_key(key) { self.string(key) } else { Ok(default.to_owned()) }
    }

    fn opt_string(&self, key: &str) -> VResult<Option<String>> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Str(s)) => Ok(Some(s.clone())),
            Some(other) => invalid(&self.path(key), format!("expected str, got {other:?}")),
        }
    }

    fn int(&self, key: &str) -> VResult<i64> {
        let v = self.req(key)?;
        v.as_i64().map_or_else(|| invalid(&self.path(key), format!("expected int, got {v:?}")), Ok)
    }

    fn int_or(&self, key: &str, default: i64) -> VResult<i64> {
        if self.map.contains_key(key) { self.int(key) } else { Ok(default) }
    }

    fn opt_int(&self, key: &str) -> VResult<Option<i64>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v.as_i64().map(Some).map_or_else(|| invalid(&self.path(key), format!("expected int, got {v:?}")), Ok),
        }
    }

    fn float(&self, key: &str) -> VResult<f64> {
        let v = self.req(key)?;
        v.as_f64().map_or_else(|| invalid(&self.path(key), format!("expected float, got {v:?}")), Ok)
    }

    fn float_or(&self, key: &str, default: f64) -> VResult<f64> {
        if self.map.contains_key(key) { self.float(key) } else { Ok(default) }
    }

    fn opt_float(&self, key: &str) -> VResult<Option<f64>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v.as_f64().map(Some).map_or_else(|| invalid(&self.path(key), format!("expected float, got {v:?}")), Ok),
        }
    }

    fn boolean_or(&self, key: &str, default: bool) -> VResult<bool> {
        match self.map.get(key) {
            None => Ok(default),
            Some(Value::Bool(b)) => Ok(*b),
            Some(other) => invalid(&self.path(key), format!("expected bool, got {other:?}")),
        }
    }

    fn list(&self, key: &str) -> VResult<&'a [Value]> {
        match self.map.get(key) {
            None => Ok(&[]),
            Some(Value::Array(items)) => Ok(items),
            Some(other) => invalid(&self.path(key), format!("expected list, got {other:?}")),
        }
    }

    fn list_of<T>(&self, key: &str, parse: impl Fn(&Value, String) -> VResult<T>) -> VResult<Vec<T>> {
        self.list(key)?.iter().enumerate().map(|(i, v)| parse(v, format!("{}[{i}]", self.path(key)))).collect()
    }

    fn opt_of<T>(&self, key: &str, parse: impl Fn(&Value, String) -> VResult<T>) -> VResult<Option<T>> {
        self.get(key).map(|v| parse(v, self.path(key))).transpose()
    }

    fn dict(&self, key: &str) -> VResult<Map> {
        match self.map.get(key) {
            None => Ok(Map::new()),
            Some(Value::Map(m)) => Ok(m.clone()),
            Some(other) => invalid(&self.path(key), format!("expected dict, got {other:?}")),
        }
    }

    fn raw(&self, key: &str) -> Value {
        self.map.get(key).cloned().unwrap_or_default()
    }
}

/// `uuid.uuid4().hex`, the default id of episodes and traces.
pub(crate) fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

fn map_value(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(entries.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

fn list_value<T>(items: &[T], f: impl Fn(&T) -> Value) -> Value {
    Value::Array(items.iter().map(f).collect())
}

// ── shared values ──────────────────────────────────────────────────────────

thread_local! {
    static LAST_AGENT_CONFIG: RefCell<Option<Arc<Value>>> = const { RefCell::new(None) };
    static LAST_TASK_DATA: RefCell<Option<Arc<Map>>> = const { RefCell::new(None) };
}

/// `value` behind an `Arc`, reusing this thread's previous one when equal: the agent config
/// of every episode of an env, and the task data of a group's rollouts, are then stored once
/// instead of per episode (as Python's shallow `model_copy`s share them).
fn shared<T: PartialEq>(last: &'static std::thread::LocalKey<RefCell<Option<Arc<T>>>>, value: T) -> Arc<T> {
    last.with(|last| {
        let mut last = last.borrow_mut();
        match last.as_ref() {
            Some(previous) if **previous == value => previous.clone(),
            _ => {
                let value = Arc::new(value);
                *last = Some(value.clone());
                value
            }
        }
    })
}

type DumpCache<T> = RefCell<Vec<(std::sync::Weak<T>, bool, Value)>>;

thread_local! {
    static AGENT_CONFIG_DUMPS: DumpCache<Value> = const { RefCell::new(Vec::new()) };
    static TASK_DATA_DUMPS: DumpCache<Map> = const { RefCell::new(Vec::new()) };
}

/// The JSON-mode dump of a shared subtree, cached per `Arc` (and `exclude_none`) on this
/// thread: records of an env's episodes dump the same agent config, a group's the same task.
fn cached_dump<T>(cache: &'static std::thread::LocalKey<DumpCache<T>>, value: &Arc<T>, exclude_none: bool, dump: impl FnOnce(&T) -> Value) -> Value {
    const ENTRIES: usize = 16;
    cache.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((_, _, dumped)) =
            cache.iter().find(|(weak, flag, _)| *flag == exclude_none && weak.upgrade().is_some_and(|a| Arc::ptr_eq(&a, value)))
        {
            return dumped.clone();
        }
        let dumped = dump(value);
        if cache.len() == ENTRIES {
            cache.remove(0);
        }
        cache.push((Arc::downgrade(value), exclude_none, dumped.clone()));
        dumped
    })
}

/// Put `key` at `index` of a dumped map: in place when the dump kept a placeholder for it,
/// else inserted there (exclude_none dropped the placeholder).
fn put_at(map: &mut Map, index: usize, key: &str, value: Value) {
    if let Some(slot) = map.get_mut(key) {
        *slot = value;
    } else {
        map.shift_insert(index.min(map.len()), key.to_owned(), value);
    }
}

// ── small models ───────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Error {
    pub type_: String,
    pub message: String,
    pub status_code: Option<i64>,
    pub traceback: Option<String>,
}

impl Error {
    pub fn new(type_: impl Into<String>, message: impl Into<String>) -> Error {
        Error { type_: type_.into(), message: message.into(), status_code: None, traceback: None }
    }

    fn from_value(value: &Value, path: String) -> VResult<Error> {
        let f = Fields::of(value, path)?;
        Ok(Error {
            type_: f.string("type")?,
            message: f.string("message")?,
            status_code: f.opt_int("status_code")?,
            traceback: f.opt_string("traceback")?,
        })
    }

    fn to_python(&self) -> Value {
        map_value(vec![
            ("type", self.type_.as_str().into()),
            ("message", self.message.as_str().into()),
            ("status_code", self.status_code.into()),
            ("traceback", self.traceback.clone().into()),
        ])
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub cached_input_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
    pub cost: Option<f64>,
}

impl Usage {
    fn from_value(value: &Value, path: String) -> VResult<Usage> {
        let f = Fields::of(value, path)?;
        Ok(Usage {
            prompt_tokens: f.int("prompt_tokens")?,
            completion_tokens: f.int("completion_tokens")?,
            cached_input_tokens: f.opt_int("cached_input_tokens")?,
            reasoning_tokens: f.opt_int("reasoning_tokens")?,
            cost: f.opt_float("cost")?,
        })
    }

    fn to_python(&self) -> Value {
        map_value(vec![
            ("prompt_tokens", self.prompt_tokens.into()),
            ("completion_tokens", self.completion_tokens.into()),
            ("cached_input_tokens", self.cached_input_tokens.into()),
            ("reasoning_tokens", self.reasoning_tokens.into()),
            ("cost", self.cost.into()),
        ])
    }

    pub fn input_tokens(&self) -> i64 {
        self.prompt_tokens + self.cached_input_tokens.unwrap_or(0)
    }

    pub fn total_tokens(&self) -> i64 {
        self.input_tokens() + self.completion_tokens
    }

    /// `Usage.aggregate`: sums; optional fields sum the usages reporting them, `None` when
    /// none do; `None` for no usages.
    pub fn aggregate<'a>(usages: impl IntoIterator<Item = &'a Usage>) -> Option<Usage> {
        let values: Vec<&Usage> = usages.into_iter().collect();
        if values.is_empty() {
            return None;
        }
        // folds from zero like Python's sum (Rust's float sum of nothing is -0.0)
        fn sum_opt<T: std::ops::Add<Output = T> + Default + Copy>(items: impl Iterator<Item = Option<T>>) -> Option<T> {
            let present: Vec<T> = items.flatten().collect();
            (!present.is_empty()).then(|| present.into_iter().fold(T::default(), |a, b| a + b))
        }
        Some(Usage {
            prompt_tokens: values.iter().map(|u| u.prompt_tokens).sum(),
            completion_tokens: values.iter().map(|u| u.completion_tokens).sum(),
            cached_input_tokens: sum_opt(values.iter().map(|u| u.cached_input_tokens)),
            reasoning_tokens: sum_opt(values.iter().map(|u| u.reasoning_tokens)),
            cost: sum_opt(values.iter().map(|u| u.cost)),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimeSpan {
    pub start: f64,
    pub end: f64,
}

impl TimeSpan {
    fn from_value(value: &Value, path: String) -> VResult<TimeSpan> {
        let f = Fields::of(value, path)?;
        Ok(TimeSpan { start: f.float_or("start", 0.0)?, end: f.float_or("end", 0.0)? })
    }

    fn to_python(&self) -> Value {
        map_value(vec![("start", self.start.into()), ("end", self.end.into())])
    }

    pub fn duration(&self) -> f64 {
        if self.end != 0.0 { (self.end - self.start).max(0.0) } else { 0.0 }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimeSplit {
    pub duration: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AgentSpan {
    pub start: f64,
    pub end: f64,
    pub model: TimeSplit,
    pub harness: TimeSplit,
}

impl AgentSpan {
    pub fn duration(&self) -> f64 {
        TimeSpan { start: self.start, end: self.end }.duration()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timing {
    pub start: f64,
    pub boot: TimeSpan,
    pub setup: TimeSpan,
    pub agent: AgentSpan,
    pub finalize: TimeSpan,
    pub scoring: TimeSpan,
}

impl Timing {
    fn from_value(value: &Value, path: String) -> VResult<Timing> {
        let f = Fields::of(value, path)?;
        let span = |key: &str| -> VResult<TimeSpan> {
            f.opt_of(key, TimeSpan::from_value).map(Option::unwrap_or_default)
        };
        let agent = match f.get("agent") {
            None => AgentSpan::default(),
            Some(v) => {
                let a = Fields::of(v, f.path("agent"))?;
                let split = |key: &str| -> VResult<TimeSplit> {
                    Ok(match a.get(key) {
                        None => TimeSplit::default(),
                        Some(v) => TimeSplit { duration: Fields::of(v, a.path(key))?.float_or("duration", 0.0)? },
                    })
                };
                AgentSpan {
                    start: a.float_or("start", 0.0)?,
                    end: a.float_or("end", 0.0)?,
                    model: split("model")?,
                    harness: split("harness")?,
                }
            }
        };
        Ok(Timing {
            start: if f.map.contains_key("start") { f.float("start")? } else { now() },
            boot: span("boot")?,
            setup: span("setup")?,
            agent,
            finalize: span("finalize")?,
            scoring: span("scoring")?,
        })
    }

    fn to_python(&self) -> Value {
        let split = |s: &TimeSplit| map_value(vec![("duration", s.duration.into())]);
        map_value(vec![
            ("start", self.start.into()),
            ("boot", self.boot.to_python()),
            ("setup", self.setup.to_python()),
            (
                "agent",
                map_value(vec![
                    ("start", self.agent.start.into()),
                    ("end", self.agent.end.into()),
                    ("model", split(&self.agent.model)),
                    ("harness", split(&self.agent.harness)),
                ]),
            ),
            ("finalize", self.finalize.to_python()),
            ("scoring", self.scoring.to_python()),
        ])
    }

}

#[derive(Clone, Debug, PartialEq)]
pub struct Reward {
    pub score: f64,
    pub weight: f64,
}

impl Reward {
    pub fn value(&self) -> f64 {
        self.score * self.weight
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParentLink {
    pub node: usize,
    pub type_: String,
}

impl ParentLink {
    fn from_value(value: &Value, path: String) -> VResult<ParentLink> {
        let f = Fields::of(value, path)?;
        let node = f.int("node")?;
        if node < 0 {
            return invalid(&f.path("node"), "must be >= 0");
        }
        Ok(ParentLink { node: node as usize, type_: f.string("type")? })
    }

    fn to_python(&self) -> Value {
        map_value(vec![("node", self.node.into()), ("type", self.type_.as_str().into())])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightVersionSpan {
    pub version: String,
    pub start: i64,
    pub end: i64,
}

impl WeightVersionSpan {
    fn from_value(value: &Value, path: String) -> VResult<WeightVersionSpan> {
        let f = Fields::of(value, path)?;
        Ok(WeightVersionSpan { version: f.string("version")?, start: f.int("start")?, end: f.int("end")? })
    }

    fn to_python(&self) -> Value {
        map_value(vec![
            ("version", self.version.as_str().into()),
            ("start", self.start.into()),
            ("end", self.end.into()),
        ])
    }
}

// ── model calls ────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelCall {
    pub node: Option<usize>,
    pub model: Option<String>,
    /// `SamplingConfig | None`, as dumped.
    pub sampling: Value,
    pub endpoint: Option<String>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
    pub time: TimeSpan,
    pub error: Option<Error>,
    /// `PolicyEvent | None`, as dumped.
    pub policy: Value,
    /// `ACPInfo | None`, as dumped.
    pub acp: Value,
}

impl ModelCall {
    fn from_value(value: &Value, path: String) -> VResult<ModelCall> {
        let f = Fields::of(value, path)?;
        Ok(ModelCall {
            node: f.opt_int("node")?.map(|n| n as usize),
            model: f.opt_string("model")?,
            sampling: f.raw("sampling"),
            endpoint: f.opt_string("endpoint")?,
            finish_reason: f.opt_string("finish_reason")?,
            usage: f.opt_of("usage", Usage::from_value)?,
            time: f.opt_of("time", TimeSpan::from_value)?.unwrap_or_default(),
            error: f.opt_of("error", Error::from_value)?,
            policy: f.raw("policy"),
            acp: f.raw("acp"),
        })
    }

    fn to_python(&self) -> Value {
        map_value(vec![
            ("node", self.node.into()),
            ("model", self.model.clone().into()),
            ("sampling", self.sampling.clone()),
            ("endpoint", self.endpoint.clone().into()),
            ("finish_reason", self.finish_reason.clone().into()),
            ("usage", self.usage.as_ref().map_or(Value::Nil, Usage::to_python)),
            ("time", self.time.to_python()),
            ("error", self.error.as_ref().map_or(Value::Nil, Error::to_python)),
            ("policy", self.policy.clone()),
            ("acp", self.acp.clone()),
        ])
    }
}

/// Per-call lower bound on newly fed-in tokens (`trace.min_new_input_tokens`): each call's
/// prompt beyond the previous call's final sequence, clamped at zero; calls without usage
/// are skipped. Yields `(index into calls, increment)`.
pub fn min_new_input_tokens<'a>(calls: impl IntoIterator<Item = &'a ModelCall>) -> Vec<(usize, i64)> {
    let mut prev_total = 0;
    let mut out = Vec::new();
    for (i, call) in calls.into_iter().enumerate() {
        let Some(usage) = &call.usage else { continue };
        out.push((i, (usage.input_tokens() - prev_total).max(0)));
        prev_total = usage.total_tokens();
    }
    out
}

// ── message nodes ──────────────────────────────────────────────────────────

pub const RECORD_FLOAT_DECIMALS: u32 = 4;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MessageNode {
    pub parent: Option<usize>,
    /// `list[Tool]`, populated on root nodes only.
    pub tools: Vec<Value>,
    pub semantic_parents: Vec<ParentLink>,
    /// The message (`role` discriminated), as dumped.
    pub message: Value,
    pub sampled: bool,
    pub timestamp: f64,
    pub token_ids: Vec<i32>,
    pub mask: Vec<bool>,
    pub is_content: Vec<bool>,
    pub logprobs: Vec<f32>,
    pub advantages: Option<Vec<f32>>,
    pub reference_logprobs: Option<Vec<f32>>,
    pub trainer_logprobs: Option<Vec<f32>>,
    pub entropies: Option<Vec<f32>>,
    pub loss_weights: Option<IndexMap<String, Vec<f32>>>,
    /// `__nd__` uint8 `[tokens, layers, top_k]`, as received.
    pub routed_experts: Option<Value>,
    /// `{ids, counts}` of `__nd__` arrays, as received.
    pub sampling_mask: Option<Value>,
    pub weight_versions: Option<Vec<WeightVersionSpan>>,
    /// `{ids, logprobs, counts}` of `__nd__` arrays, as received.
    pub top_logprobs: Option<Value>,
}

fn opt_float_stream(f: &Fields, key: &str) -> VResult<Option<Vec<f32>>> {
    f.get(key).map(|v| ndarray::float_stream(v).map_err(|e| ValidationError { path: f.path(key), message: e.to_string() })).transpose()
}

impl MessageNode {
    fn from_value(value: &Value, path: String) -> VResult<MessageNode> {
        let f = Fields::of(value, path)?;
        let stream_err = |key: &str| {
            let path = f.path(key);
            move |e: NdError| ValidationError { path, message: e.to_string() }
        };
        let token_ids = match f.get("token_ids") {
            Some(v) => ndarray::int_stream(v).map_err(stream_err("token_ids"))?,
            None => Vec::new(),
        };
        let bools = |key: &str| -> VResult<Vec<bool>> {
            match f.get(key) {
                Some(v) => ndarray::bool_stream(v).map_err(stream_err(key)),
                None => Ok(Vec::new()),
            }
        };
        let loss_weights = match f.get("loss_weights") {
            None => None,
            Some(Value::Map(streams)) => Some(
                streams
                    .iter()
                    .map(|(name, v)| Ok((name.clone(), ndarray::float_stream(v).map_err(stream_err("loss_weights"))?)))
                    .collect::<VResult<IndexMap<_, _>>>()?,
            ),
            Some(other) => return invalid(&f.path("loss_weights"), format!("expected dict, got {other:?}")),
        };
        let message = f.req("message")?.clone();
        if !matches!(message.get("role").and_then(Value::as_str), Some("system" | "user" | "assistant" | "tool")) {
            return invalid(&f.path("message"), "message needs a role of system/user/assistant/tool");
        }
        Ok(MessageNode {
            parent: f.opt_int("parent")?.map(|p| p as usize),
            tools: f.list("tools")?.to_vec(),
            semantic_parents: f.list_of("semantic_parents", ParentLink::from_value)?,
            message,
            sampled: f.boolean_or("sampled", false)?,
            timestamp: f.float_or("timestamp", 0.0)?,
            token_ids,
            mask: bools("mask")?,
            is_content: bools("is_content")?,
            logprobs: opt_float_stream(&f, "logprobs")?.unwrap_or_default(),
            advantages: opt_float_stream(&f, "advantages")?,
            reference_logprobs: opt_float_stream(&f, "reference_logprobs")?,
            trainer_logprobs: opt_float_stream(&f, "trainer_logprobs")?,
            entropies: opt_float_stream(&f, "entropies")?,
            loss_weights,
            routed_experts: f.get("routed_experts").cloned(),
            sampling_mask: f.get("sampling_mask").cloned(),
            weight_versions: f
                .get("weight_versions")
                .map(|_| f.list_of("weight_versions", WeightVersionSpan::from_value))
                .transpose()?,
            top_logprobs: f.get("top_logprobs").cloned(),
        })
    }

    /// The node's fields other than its per-token streams and wire tensors, python mode.
    fn to_python_meta(&self) -> Map {
        let mut map = Map::new();
        map.insert("parent".into(), self.parent.into());
        if !self.tools.is_empty() {
            map.insert("tools".into(), Value::Array(self.tools.clone()));
        }
        map.insert("semantic_parents".into(), list_value(&self.semantic_parents, ParentLink::to_python));
        map.insert("message".into(), self.message.clone());
        map.insert("sampled".into(), self.sampled.into());
        map.insert("timestamp".into(), self.timestamp.into());
        map.insert(
            "weight_versions".into(),
            self.weight_versions.as_ref().map_or(Value::Nil, |w| list_value(w, WeightVersionSpan::to_python)),
        );
        map
    }

    /// `MessageNode.record_streams`: the per-token streams of a JSON record, appended after
    /// the node's other fields. Float streams in `ROUNDED_STREAMS` round to `float_decimals`.
    fn record_streams(&self, out: &mut Map, float_decimals: Option<u32>, exclude_none: bool) {
        let ints = Value::Array(self.token_ids.iter().map(|&t| Value::Int(t as i64)).collect());
        let bools = |s: &[bool]| Value::Array(s.iter().map(|&b| Value::Bool(b)).collect());
        let floats = |s: &[f32], decimals: Option<u32>| {
            Value::Array(s.iter().map(|&x| Value::F64(round_stream_value(x as f64, decimals))).collect())
        };
        out.insert("token_ids".into(), ints);
        out.insert("mask".into(), bools(&self.mask));
        out.insert("is_content".into(), bools(&self.is_content));
        let rounded: [(&str, Option<&Vec<f32>>); 5] = [
            ("logprobs", Some(&self.logprobs)),
            ("advantages", self.advantages.as_ref()),
            ("reference_logprobs", self.reference_logprobs.as_ref()),
            ("trainer_logprobs", self.trainer_logprobs.as_ref()),
            ("entropies", self.entropies.as_ref()),
        ];
        for (name, stream) in rounded {
            match stream {
                Some(s) => {
                    out.insert(name.into(), floats(s, float_decimals));
                }
                None if !exclude_none => {
                    out.insert(name.into(), Value::Nil);
                }
                None => {}
            }
        }
        match &self.loss_weights {
            Some(w) => {
                out.insert(
                    "loss_weights".into(),
                    Value::Map(w.iter().map(|(k, v)| (k.clone(), floats(v, None))).collect()),
                );
            }
            None if !exclude_none => {
                out.insert("loss_weights".into(), Value::Nil);
            }
            None => {}
        }
    }

    pub fn role(&self) -> &str {
        self.message.get("role").and_then(Value::as_str).unwrap_or_default()
    }
}

/// `verifiers.v1.task.task_key` of task wire data (`WireTaskData`, python-mode dump).
pub fn task_hash(data: &Map) -> String {
    use sha2::Digest;
    let dumped = schema::dump_model(data, SCHEMAS.model("WireTaskData"), true);
    let text = crate::json::to_python_json(&dumped, true);
    sha2::Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// `np.round(np.float64(x), decimals)`: scale, round half to even, unscale.
fn round_stream_value(x: f64, decimals: Option<u32>) -> f64 {
    match decimals {
        None => x,
        Some(d) => {
            let scale = 10f64.powi(d as i32);
            (x * scale).round_ties_even() / scale
        }
    }
}

// ── branches ───────────────────────────────────────────────────────────────

/// A root-to-leaf path of a trace's message graph; one training sample.
#[derive(Clone, Debug)]
pub struct Branch<'a> {
    pub index: usize,
    /// Indices into `Trace.nodes`, root first.
    pub node_ids: Vec<usize>,
    pub nodes: Vec<&'a MessageNode>,
    /// Indices into `Trace.calls` of the calls that committed the path's nodes.
    pub call_ids: Vec<usize>,
    pub calls: Vec<&'a ModelCall>,
    pub trainable: bool,
    pub mm_token_type_id_map: &'a IndexMap<i64, i64>,
}

impl<'a> Branch<'a> {
    pub fn num_tokens(&self) -> usize {
        self.nodes.iter().map(|n| n.token_ids.len()).sum()
    }

    pub fn token_ids(&self) -> Vec<i32> {
        self.nodes.iter().flat_map(|n| n.token_ids.iter().copied()).collect()
    }

    /// `Branch.spread`: a per-sampled-token node field widened onto the branch's tokens.
    fn spread(&self, values: impl Fn(&MessageNode) -> Option<&[f32]>) -> Vec<f32> {
        let mut out = vec![0f32; self.nodes.iter().map(|n| n.mask.len()).sum()];
        let mut offset = 0;
        for node in &self.nodes {
            if let Some(values) = values(node) {
                let positions = node.mask.iter().enumerate().filter(|(_, &m)| m).map(|(i, _)| i);
                for (position, &value) in positions.zip(values.iter()) {
                    out[offset + position] = value;
                }
            }
            offset += node.mask.len();
        }
        out
    }

    fn spread_optional(&self, values: impl Fn(&MessageNode) -> Option<&[f32]>) -> Option<Vec<f32>> {
        if self.nodes.iter().all(|n| values(n).is_none()) {
            return None;
        }
        Some(self.spread(values))
    }

    pub fn logprobs(&self) -> Vec<f32> {
        self.spread(|n| Some(&n.logprobs))
    }

    pub fn advantages(&self) -> Option<Vec<f32>> {
        self.spread_optional(|n| n.advantages.as_deref())
    }

    pub fn reference_logprobs(&self) -> Option<Vec<f32>> {
        self.spread_optional(|n| n.reference_logprobs.as_deref())
    }

    /// Per-token modality markers (0 text, 1 image, 2 video), `None` when none are present.
    pub fn mm_token_type_ids(&self) -> Option<Vec<i32>> {
        if self.mm_token_type_id_map.is_empty() {
            return None;
        }
        let types: Vec<i32> = self
            .token_ids()
            .iter()
            .map(|t| self.mm_token_type_id_map.get(&(*t as i64)).map_or(0, |&ty| ty as i32))
            .collect();
        types.iter().any(|&t| t != 0).then_some(types)
    }

    fn usage(&self) -> Option<Usage> {
        Usage::aggregate(self.calls.iter().filter_map(|c| c.usage.as_ref()))
    }

    fn last_usage(&self) -> Option<&Usage> {
        self.calls.iter().rev().find_map(|c| c.usage.as_ref())
    }

    pub fn num_total_tokens(&self) -> i64 {
        self.last_usage().map_or(0, Usage::total_tokens)
    }

    pub fn num_output_tokens(&self) -> i64 {
        self.usage().map_or(0, |u| u.completion_tokens)
    }

    pub fn num_input_tokens(&self) -> i64 {
        min_new_input_tokens(self.calls.iter().copied()).iter().map(|(_, inc)| inc).sum()
    }
}

// ── traces ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct TraceTask {
    /// The Task class name.
    pub type_: String,
    /// `WireTaskData` (declared fields + task-specific extras), as dumped. Shared: the
    /// rollouts of a group carry the same data.
    pub data: Arc<Map>,
    pub key: Option<String>,
    pub hash: Option<String>,
}

impl TraceTask {
    fn from_value(value: &Value, path: String) -> VResult<TraceTask> {
        let f = Fields::of(value, path)?;
        match f.req("data")? {
            Value::Map(_) => {}
            other => return invalid(&f.path("data"), format!("expected dict, got {other:?}")),
        }
        let data = shared(&LAST_TASK_DATA, f.dict("data")?);
        Ok(TraceTask { type_: f.string("type")?, data, key: f.opt_string("key")?, hash: f.opt_string("hash")? })
    }

    /// The JSON-mode dump of this task (`TraceTask[WireTaskData]`), its data dump cached
    /// per shared `Arc` (a group's rollouts share it).
    fn record_value(&self, exclude_none: bool) -> Value {
        let python = map_value(vec![
            ("type", self.type_.as_str().into()),
            ("data", Value::Nil),
            ("key", self.key.clone().into()),
            ("hash", self.hash.clone().into()),
        ]);
        let mut dumped = schema::dump_json(&python, &schema::Schema::Model { name: schema::TRACE_TASK.into() }, exclude_none);
        let data = cached_dump(&TASK_DATA_DUMPS, &self.data, exclude_none, |data| {
            schema::dump_model(data, SCHEMAS.model("WireTaskData"), exclude_none)
        });
        put_at(dumped.as_map_mut().expect("task record"), 1, "data", data);
        dumped
    }

    /// `task.data.idx`.
    pub fn idx(&self) -> Option<i64> {
        self.data.get("idx").and_then(Value::as_i64)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentInfo {
    /// `WireAgentConfig`, as dumped. Shared: every episode of an env carries the same config.
    pub config: Arc<Value>,
    /// `RuntimeInfo | None`, as dumped.
    pub runtime: Value,
    pub name: String,
    pub trainable: bool,
}

impl AgentInfo {
    fn from_value(value: &Value, path: String) -> VResult<AgentInfo> {
        let f = Fields::of(value, path)?;
        Ok(AgentInfo {
            config: shared(&LAST_AGENT_CONFIG, f.req("config")?.clone()),
            runtime: f.raw("runtime"),
            name: f.string_or("name", "agent")?,
            trainable: f.boolean_or("trainable", true)?,
        })
    }

    /// The JSON-mode dump of this agent info, its config dump cached per shared `Arc`.
    fn record_value(&self, exclude_none: bool) -> Value {
        let python = map_value(vec![
            ("config", Value::Nil),
            ("runtime", self.runtime.clone()),
            ("name", self.name.as_str().into()),
            ("trainable", self.trainable.into()),
        ]);
        let mut dumped = schema::dump_json(&python, &schema::Schema::Model { name: schema::AGENT_INFO.into() }, exclude_none);
        let config = cached_dump(&AGENT_CONFIG_DUMPS, &self.config, exclude_none, |config| {
            schema::dump_json(config, SCHEMAS.field(schema::AGENT_INFO, "config"), exclude_none)
        });
        put_at(dumped.as_map_mut().expect("agent record"), 0, "config", config);
        dumped
    }

}

#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    pub version: i64,
    pub id: String,
    /// `VersionInfo{version, commit}`, as dumped.
    pub verifiers: Value,
    pub task: TraceTask,
    pub agent: AgentInfo,
    pub tools: Vec<Value>,
    pub nodes: Vec<MessageNode>,
    pub calls: Vec<ModelCall>,
    pub mm_token_type_id_map: IndexMap<i64, i64>,
    pub request_rewrites: Vec<Value>,
    pub response_rewrites: Vec<Value>,
    pub rewards: IndexMap<String, Option<Reward>>,
    pub metrics: IndexMap<String, Option<f64>>,
    pub info: Map,
    pub root_reply: Option<String>,
    pub extra_usage: Vec<Usage>,
    pub is_completed: bool,
    pub ok: bool,
    pub stop_condition: Option<String>,
    pub is_timeout: bool,
    pub errors: Vec<Error>,
    pub dropped: Option<String>,
    pub timing: Timing,
}

/// Stop conditions `Trace.is_truncated` counts as truncation.
const TRUNCATION_STOP_CONDITIONS: [&str; 6] =
    ["max_turns", "max_input_tokens", "max_output_tokens", "max_total_tokens", "max_turn_tokens", "compaction_failed"];

impl Trace {
    pub(crate) fn from_value(value: &Value, path: String) -> VResult<Trace> {
        let f = Fields::of(value, path)?;
        let mut mm = IndexMap::new();
        for (k, v) in f.dict("mm_token_type_id_map")? {
            let Ok(token) = k.parse::<i64>() else {
                return invalid(&f.path("mm_token_type_id_map"), format!("key {k:?} is not an int"));
            };
            let Some(ty) = v.as_i64() else {
                return invalid(&f.path("mm_token_type_id_map"), format!("value {v:?} is not an int"));
            };
            mm.insert(token, ty);
        }
        let mut rewards = IndexMap::new();
        for (name, v) in f.dict("rewards")? {
            let reward = match v {
                Value::Nil => None,
                v => {
                    let r = Fields::of(&v, format!("{}.{name}", f.path("rewards")))?;
                    Some(Reward { score: r.float("score")?, weight: r.float_or("weight", 1.0)? })
                }
            };
            rewards.insert(name, reward);
        }
        let mut metrics = IndexMap::new();
        for (name, v) in f.dict("metrics")? {
            let metric = match v {
                Value::Nil => None,
                v => match v.as_f64() {
                    Some(x) => Some(x),
                    None => return invalid(&f.path("metrics"), format!("{name}: expected float, got {v:?}")),
                },
            };
            metrics.insert(name, metric);
        }
        Ok(Trace {
            version: f.int_or("version", TRACE_VERSION)?,
            id: f.string_or_else("id", new_id)?,
            verifiers: f.raw("verifiers"),
            task: TraceTask::from_value(f.req("task")?, f.path("task"))?,
            agent: AgentInfo::from_value(f.req("agent")?, f.path("agent"))?,
            tools: f.list("tools")?.to_vec(),
            nodes: f.list_of("nodes", MessageNode::from_value)?,
            calls: f.list_of("calls", ModelCall::from_value)?,
            mm_token_type_id_map: mm,
            request_rewrites: f.list("request_rewrites")?.to_vec(),
            response_rewrites: f.list("response_rewrites")?.to_vec(),
            rewards,
            metrics,
            info: f.dict("info")?,
            root_reply: f.opt_string("root_reply")?,
            extra_usage: f.list_of("extra_usage", Usage::from_value)?,
            is_completed: f.boolean_or("is_completed", false)?,
            ok: f.boolean_or("ok", false)?,
            stop_condition: f.opt_string("stop_condition")?,
            is_timeout: f.boolean_or("is_timeout", false)?,
            errors: f.list_of("errors", Error::from_value)?,
            dropped: f.opt_string("dropped")?,
            timing: match f.get("timing") {
                Some(v) => Timing::from_value(v, f.path("timing"))?,
                None => Timing { start: now(), ..Timing::default() },
            },
        })
    }

    /// The trace as a python-mode dump with null `task` and `agent` (the record dumps those
    /// shared subtrees apart) and `nodes` as `node` builds them.
    fn to_python(&self, node: impl Fn(&MessageNode) -> Value) -> Value {
        let mut map = Map::new();
        map.insert("version".into(), self.version.into());
        map.insert("id".into(), self.id.as_str().into());
        map.insert("verifiers".into(), self.verifiers.clone());
        map.insert("task".into(), Value::Nil);
        map.insert("agent".into(), Value::Nil);
        map.insert("tools".into(), Value::Array(self.tools.clone()));
        map.insert("nodes".into(), list_value(&self.nodes, node));
        map.insert("calls".into(), list_value(&self.calls, ModelCall::to_python));
        map.insert(
            "mm_token_type_id_map".into(),
            Value::Map(self.mm_token_type_id_map.iter().map(|(k, v)| (k.to_string(), Value::Int(*v))).collect()),
        );
        map.insert("request_rewrites".into(), Value::Array(self.request_rewrites.clone()));
        map.insert("response_rewrites".into(), Value::Array(self.response_rewrites.clone()));
        map.insert(
            "rewards".into(),
            Value::Map(
                self.rewards
                    .iter()
                    .map(|(k, r)| {
                        let v = r.as_ref().map_or(Value::Nil, |r| {
                            map_value(vec![("score", r.score.into()), ("weight", r.weight.into())])
                        });
                        (k.clone(), v)
                    })
                    .collect(),
            ),
        );
        map.insert("metrics".into(), Value::Map(self.metrics.iter().map(|(k, v)| (k.clone(), (*v).into())).collect()));
        map.insert("info".into(), Value::Map(self.info.clone()));
        map.insert("root_reply".into(), self.root_reply.clone().into());
        map.insert("extra_usage".into(), list_value(&self.extra_usage, Usage::to_python));
        map.insert("is_completed".into(), self.is_completed.into());
        map.insert("ok".into(), self.ok.into());
        map.insert("stop_condition".into(), self.stop_condition.clone().into());
        map.insert("is_timeout".into(), self.is_timeout.into());
        map.insert("errors".into(), list_value(&self.errors, Error::to_python));
        map.insert("dropped".into(), self.dropped.clone().into());
        map.insert("timing".into(), self.timing.to_python());
        map.insert("num_input_tokens".into(), self.num_input_tokens().into());
        map.insert("num_output_tokens".into(), self.num_output_tokens().into());
        map.insert("num_total_tokens".into(), self.num_total_tokens().into());
        Value::Map(map)
    }

    fn record_value(&self, float_decimals: Option<u32>, exclude_none: bool) -> Value {
        // the shared task and agent subtrees are dumped (and cached) on their own
        let python = self.to_python(|n| Value::Map(n.to_python_meta()));
        let mut record = schema::dump_json(&python, &schema::Schema::Model { name: schema::TRACE.into() }, exclude_none);
        let map = record.as_map_mut().expect("trace record");
        put_at(map, 3, "task", self.task.record_value(exclude_none));
        put_at(map, 4, "agent", self.agent.record_value(exclude_none));
        let nodes = record.get_mut("nodes").and_then(Value::as_array_mut).expect("dumped trace has nodes");
        for (node_record, node) in nodes.iter_mut().zip(&self.nodes) {
            node.record_streams(node_record.as_map_mut().expect("node record"), float_decimals, exclude_none);
        }
        record
    }

    /// `Trace.to_record(float_decimals)` (no `exclude_none`).
    pub fn to_record(&self, float_decimals: Option<u32>) -> Value {
        self.record_value(float_decimals, false)
    }

    pub fn reward(&self) -> f64 {
        self.rewards.values().flatten().map(Reward::value).fold(0.0, |a, b| a + b)
    }

    pub fn has_error(&self) -> bool {
        !self.ok
    }

    pub fn is_dropped(&self) -> bool {
        self.dropped.is_some()
    }

    pub fn last_error(&self) -> Option<&Error> {
        self.errors.last()
    }

    /// Node ids that are no node's parent (`graph.leaves`).
    fn leaves(&self) -> Vec<usize> {
        let mut has_child = vec![false; self.nodes.len()];
        for node in &self.nodes {
            if let Some(parent) = node.parent {
                if parent < has_child.len() {
                    has_child[parent] = true;
                }
            }
        }
        (0..self.nodes.len()).filter(|&i| !has_child[i]).collect()
    }

    pub fn num_branches(&self) -> usize {
        self.leaves().len()
    }

    /// One root-to-leaf view per leaf. A compaction attempt (a leaf with a
    /// `compaction_attempt` link) is trainable only when a `compaction` link points at it.
    pub fn branches(&self) -> Vec<Branch<'_>> {
        let mut by_node: IndexMap<usize, usize> = IndexMap::new();
        for (i, call) in self.calls.iter().enumerate() {
            if let Some(node) = call.node {
                by_node.insert(node, i);
            }
        }
        let accepted: std::collections::HashSet<usize> = self
            .nodes
            .iter()
            .flat_map(|n| n.semantic_parents.iter())
            .filter(|l| l.type_ == "compaction")
            .map(|l| l.node)
            .collect();
        self.leaves()
            .into_iter()
            .enumerate()
            .map(|(index, leaf)| {
                let mut path = Vec::new();
                let mut current = Some(leaf);
                while let Some(id) = current {
                    path.push(id);
                    current = self.nodes[id].parent;
                }
                path.reverse();
                let is_attempt = self.nodes[leaf].semantic_parents.iter().any(|l| l.type_ == "compaction_attempt");
                let call_ids: Vec<usize> = path.iter().filter_map(|n| by_node.get(n).copied()).collect();
                Branch {
                    index,
                    nodes: path.iter().map(|&n| &self.nodes[n]).collect(),
                    node_ids: path,
                    calls: call_ids.iter().map(|&c| &self.calls[c]).collect(),
                    call_ids,
                    trainable: !is_attempt || accepted.contains(&leaf),
                    mm_token_type_id_map: &self.mm_token_type_id_map,
                }
            })
            .collect()
    }

    pub fn num_turns(&self) -> usize {
        self.nodes.iter().filter(|n| n.sampled).count()
    }

    fn usage(&self) -> Option<Usage> {
        Usage::aggregate(self.calls.iter().filter_map(|c| c.usage.as_ref()))
    }

    /// Fed-in tokens counted once across branches (a call on a shared prefix counts once).
    pub fn num_input_tokens(&self) -> i64 {
        let mut seen = std::collections::HashSet::new();
        let mut total = 0;
        for branch in self.branches() {
            for (position, increment) in min_new_input_tokens(branch.calls.iter().copied()) {
                if seen.insert(branch.call_ids[position]) {
                    total += increment;
                }
            }
        }
        total
    }

    pub fn num_output_tokens(&self) -> i64 {
        self.branches().iter().map(Branch::num_output_tokens).sum()
    }

    pub fn num_total_tokens(&self) -> i64 {
        self.num_input_tokens() + self.usage().map_or(0, |u| u.completion_tokens)
    }

    pub fn is_truncated(&self) -> bool {
        if let Some(stop) = &self.stop_condition {
            if TRUNCATION_STOP_CONDITIONS.contains(&stop.as_str()) {
                return true;
            }
        }
        self.calls
            .iter()
            .rev()
            .find(|c| c.error.is_none())
            .is_some_and(|c| c.finish_reason.as_deref() == Some("length"))
    }
}

// ── episodes ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EnvInfo {
    pub id: String,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupInfo {
    pub id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicySpan {
    pub start: i64,
    pub end: i64,
}

impl PolicySpan {
    pub fn new(start: i64, end: i64) -> Result<PolicySpan, String> {
        if end < start {
            return Err("policy span end must be at least its start".into());
        }
        Ok(PolicySpan { start, end })
    }

    fn to_python(&self) -> Value {
        map_value(vec![("start", self.start.into()), ("end", self.end.into())])
    }

    fn from_value(value: &Value, path: String) -> VResult<PolicySpan> {
        let f = Fields::of(value, path)?;
        PolicySpan::new(f.int("start")?, f.int("end")?).map_err(|m| ValidationError { path: f.path.clone(), message: m })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum WorkInfo {
    Train { step: i64, policy: Option<PolicySpan> },
    Eval { step: i64, policy: Option<PolicySpan> },
}

impl WorkInfo {
    pub fn step(&self) -> i64 {
        match self {
            WorkInfo::Train { step, .. } | WorkInfo::Eval { step, .. } => *step,
        }
    }

    pub fn policy(&self) -> Option<&PolicySpan> {
        match self {
            WorkInfo::Train { policy, .. } | WorkInfo::Eval { policy, .. } => policy.as_ref(),
        }
    }

    fn to_python(&self) -> Value {
        let (ty, step, policy) = match self {
            WorkInfo::Train { step, policy } => ("train", step, policy),
            WorkInfo::Eval { step, policy } => ("eval", step, policy),
        };
        map_value(vec![
            ("type", ty.into()),
            ("step", (*step).into()),
            ("policy", policy.as_ref().map_or(Value::Nil, PolicySpan::to_python)),
        ])
    }

    fn from_value(value: &Value, path: String) -> VResult<WorkInfo> {
        let f = Fields::of(value, path)?;
        let step = f.int("step")?;
        let policy = f.opt_of("policy", PolicySpan::from_value)?;
        match f.string("type")?.as_str() {
            "train" => Ok(WorkInfo::Train { step, policy }),
            "eval" => Ok(WorkInfo::Eval { step, policy }),
            other => invalid(&f.path("type"), format!("unknown work type {other:?}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainRunInfo {
    pub id: String,
    pub name: Option<String>,
    pub work: WorkInfo,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvalRunInfo {
    pub id: String,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RunInfo {
    Train(TrainRunInfo),
    Eval(EvalRunInfo),
}

impl RunInfo {
    fn to_python(&self) -> Value {
        match self {
            RunInfo::Train(r) => map_value(vec![
                ("type", "train".into()),
                ("id", r.id.as_str().into()),
                ("name", r.name.clone().into()),
                ("work", r.work.to_python()),
            ]),
            RunInfo::Eval(r) => map_value(vec![
                ("type", "eval".into()),
                ("id", r.id.as_str().into()),
                ("name", r.name.clone().into()),
            ]),
        }
    }

    fn from_value(value: &Value, path: String) -> VResult<RunInfo> {
        let f = Fields::of(value, path)?;
        let id = f.string("id")?;
        let name = f.opt_string("name")?;
        match f.string("type")?.as_str() {
            "train" => Ok(RunInfo::Train(TrainRunInfo {
                id,
                name,
                work: WorkInfo::from_value(f.req("work")?, f.path("work"))?,
            })),
            "eval" => Ok(RunInfo::Eval(EvalRunInfo { id, name })),
            other => invalid(&f.path("type"), format!("unknown run type {other:?}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Episode {
    pub id: String,
    pub env: EnvInfo,
    pub task: TraceTask,
    pub group: Option<GroupInfo>,
    pub run: Option<RunInfo>,
    pub ok: bool,
    pub is_timeout: bool,
    pub errors: Vec<Error>,
    pub traces: Vec<Trace>,
}

impl Episode {
    /// `WireEpisode.model_validate(value)` for an assembled episode record.
    pub fn from_wire(value: &Value) -> VResult<Episode> {
        let f = Fields::of(value, "episode")?;
        let env = match f.get("env") {
            None => EnvInfo::default(),
            Some(v) => {
                let e = Fields::of(v, f.path("env"))?;
                EnvInfo { id: e.string_or("id", "")?, name: e.opt_string("name")? }
            }
        };
        Ok(Episode {
            id: f.string_or_else("id", new_id)?,
            env,
            task: TraceTask::from_value(f.req("task")?, f.path("task"))?,
            group: f.opt_of("group", |v, p| Ok(GroupInfo { id: Fields::of(v, p)?.string("id")? }))?,
            run: f.opt_of("run", RunInfo::from_value)?,
            ok: f.boolean_or("ok", false)?,
            is_timeout: f.boolean_or("is_timeout", false)?,
            errors: f.list_of("errors", Error::from_value)?,
            traces: f.list_of("traces", Trace::from_value)?,
        })
    }

    /// `WireEpisode.model_validate(orjson.loads(line))` for a JSON record (`to_json`).
    pub fn from_record_json(data: &[u8]) -> Result<Episode, ValidationError> {
        let value: Value = serde_json::from_slice(data)
            .map_err(|e| ValidationError { path: "episode".into(), message: format!("invalid JSON: {e}") })?;
        Episode::from_wire(&value)
    }

    /// The episode as a python-mode dump with a null `task` and no traces (the record dumps
    /// those apart).
    fn to_python(&self) -> Value {
        map_value(vec![
            ("id", self.id.as_str().into()),
            ("env", map_value(vec![("id", self.env.id.as_str().into()), ("name", self.env.name.clone().into())])),
            ("task", Value::Nil),
            ("group", self.group.as_ref().map_or(Value::Nil, |g| map_value(vec![("id", g.id.as_str().into())]))),
            ("run", self.run.as_ref().map_or(Value::Nil, RunInfo::to_python)),
            ("ok", self.ok.into()),
            ("is_timeout", self.is_timeout.into()),
            ("errors", list_value(&self.errors, Error::to_python)),
            ("traces", Value::Array(vec![Value::Nil; self.traces.len()])),
            ("num_input_tokens", self.num_input_tokens().into()),
            ("num_output_tokens", self.num_output_tokens().into()),
            ("num_total_tokens", self.num_total_tokens().into()),
        ])
    }

    /// `Episode.to_record(float_decimals)`: the JSON record (exclude_none) with per-token
    /// streams and without wire tensors.
    pub fn to_record(&self, float_decimals: Option<u32>) -> Value {
        let python = self.to_python();
        let mut record = schema::dump_model(python.as_map().unwrap(), SCHEMAS.model(schema::EPISODE), true);
        put_at(record.as_map_mut().expect("episode record"), 2, "task", self.task.record_value(true));
        let traces = record.get_mut("traces").and_then(Value::as_array_mut).expect("dumped episode has traces");
        for (slot, trace) in traces.iter_mut().zip(&self.traces) {
            *slot = trace.record_value(float_decimals, true);
        }
        record
    }

    /// `Episode.to_json(float_decimals)`: `orjson.dumps(to_record(...))`.
    pub fn to_json(&self, float_decimals: Option<u32>) -> Vec<u8> {
        crate::json::to_vec(&self.to_record(float_decimals))
    }

    pub fn record_run(&mut self, run: RunInfo) {
        self.run = Some(run);
    }

    pub fn last_error(&self) -> Option<&Error> {
        self.errors.last()
    }

    pub fn num_input_tokens(&self) -> i64 {
        self.traces.iter().map(Trace::num_input_tokens).sum()
    }

    pub fn num_output_tokens(&self) -> i64 {
        self.traces.iter().map(Trace::num_output_tokens).sum()
    }

    pub fn num_total_tokens(&self) -> i64 {
        self.traces.iter().map(Trace::num_total_tokens).sum()
    }
}
