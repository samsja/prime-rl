//! The verifiers `SamplingConfig` dicts the orchestrator sends with each run request.

use crate::value::{Map, Value};

/// The train sources' sampling knobs (`prime_rl.configs.orchestrator.TrainSamplingConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainSampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<i64>,
    pub max_completion_tokens: Option<i64>,
    pub logprobs: Option<i64>,
    pub extra_body: Map,
}

/// `vf.SamplingConfig(**sampling.model_dump(exclude_none=True, exclude={"logprobs"}),
/// logprobs=(sampling.logprobs or True) if live_policy else None).model_dump(mode="json")`
/// (`TrainEnv.__init__`): only policy rollouts ask for sampling logprobs.
pub fn train_sampling(sampling: &TrainSampling, live_policy: bool) -> Map {
    let mut data = Map::new();
    data.insert("temperature".into(), sampling.temperature.into());
    data.insert("top_p".into(), sampling.top_p.into());
    if let Some(top_k) = sampling.top_k {
        data.insert("top_k".into(), top_k.into());
    }
    if let Some(max) = sampling.max_completion_tokens {
        data.insert("max_completion_tokens".into(), max.into());
    }
    data.insert("extra_body".into(), Value::Map(sampling.extra_body.clone()));
    let logprobs = match (live_policy, sampling.logprobs) {
        (false, _) => Value::Nil,
        (true, Some(n)) => Value::from(n),
        (true, None) => Value::Bool(true),
    };
    data.insert("logprobs".into(), logprobs);
    sampling_config(&data)
}

/// `vf.SamplingConfig.model_validate(data).model_dump(mode="json")`: `extra_body` is
/// layered under the top-level keys, `max_completion_tokens` becomes `max_tokens`; the
/// declared fields come first (None when unset), then the provider keys in order.
fn sampling_config(data: &Map) -> Map {
    let mut normalized = Map::new();
    let extra_body = match data.get("extra_body") {
        Some(Value::Map(m)) => m.clone(),
        _ => Map::new(),
    };
    for layer in [extra_body, data.clone()] {
        let mut layer = layer;
        if let Some(max) = layer.shift_remove("max_completion_tokens") {
            if !layer.contains_key("max_tokens") {
                layer.insert("max_tokens".into(), max);
            }
        }
        layer.shift_remove("extra_body");
        for (k, v) in layer {
            normalized.insert(k, v);
        }
    }
    let mut out = Map::new();
    for field in ["temperature", "top_p", "reasoning_effort", "max_tokens"] {
        let value = normalized.get(field).cloned().unwrap_or_default();
        let value = match (field, value) {
            ("temperature" | "top_p", Value::Int(i)) => Value::F64(i as f64),
            (_, v) => v,
        };
        out.insert(field.into(), value);
    }
    for (k, v) in normalized {
        if !out.contains_key(&k) {
            out.insert(k, v);
        }
    }
    out
}

/// `sampling.model_copy(update={"cache_salt": salt})` (`Env._sampling`).
pub fn with_cache_salt(sampling: &Map, cache_salt: Option<&str>) -> Map {
    let mut sampling = sampling.clone();
    if let Some(salt) = cache_salt {
        sampling.insert("cache_salt".into(), salt.into());
    }
    sampling
}
