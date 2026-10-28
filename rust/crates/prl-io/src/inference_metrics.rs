//! vLLM Prometheus `/metrics` polling (`orchestrator/inference_metrics.py`): every `vllm:*`
//! family mirrored as `inference/{scope}/{metric}/{stat}` metrics, plus per-engine load samples
//! for the concurrency controller.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use indexmap::IndexMap;
use prl_config::InferenceRole;
use prl_types::Metrics;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::http::AdminClient;

pub const POLL_INTERVAL: Duration = Duration::from_secs(5);
pub const FETCH_TIMEOUT: Duration = Duration::from_secs(5);
const METRIC_PREFIX: &str = "vllm:";
const CACHE_CONFIG_FAMILY: &str = "vllm:cache_config_info";
const QUANTILES: [(&str, f64); 3] = [("p50", 0.5), ("p90", 0.9), ("p99", 0.99)];
/// Scope-level ratios from counter deltas; each operand lists legacy and OpenMetrics names.
const RATIO_METRICS: [(&str, [&str; 2], [&str; 2]); 1] = [(
    "prefix_cache_hit_rate",
    ["prefix_cache_hits", "prefix_cache_hits_total"],
    ["prefix_cache_queries", "prefix_cache_queries_total"],
)];

// ── prometheus text parsing (prometheus_client.parser semantics) ──────────────

#[derive(Debug, Clone, PartialEq)]
struct Sample {
    name: String,
    labels: Vec<(String, String)>,
    value: f64,
}

impl Sample {
    fn label(&self, name: &str) -> Option<&str> {
        self.labels.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct MetricFamily {
    name: String,
    type_: String,
    samples: Vec<Sample>,
}

/// Python `float()` of a sample value (`+Inf`, `NaN`, ...).
fn parse_value(value: &str) -> Result<f64> {
    match value.to_ascii_lowercase().trim_start_matches('+') {
        "inf" | "infinity" => Ok(f64::INFINITY),
        "-inf" | "-infinity" => Ok(f64::NEG_INFINITY),
        "nan" => Ok(f64::NAN),
        _ => value.parse().map_err(|_| anyhow::anyhow!("could not convert string to float: {value:?}")),
    }
}

/// `name{k="v",...} value [timestamp]`.
fn parse_sample(line: &str) -> Result<Sample> {
    let name_end = line.find(|c: char| c == '{' || c.is_whitespace()).unwrap_or(line.len());
    let name = line[..name_end].to_owned();
    let mut rest = &line[name_end..];
    let mut labels = Vec::new();
    if let Some(body) = rest.strip_prefix('{') {
        let mut chars = body.char_indices().peekable();
        let mut end = None;
        while let Some((i, c)) = chars.next() {
            if c == '}' {
                end = Some(i);
                break;
            }
            if c == ',' || c.is_whitespace() {
                continue;
            }
            let mut key = String::from(c);
            while let Some((_, c)) = chars.next_if(|(_, c)| *c != '=') {
                key.push(c);
            }
            chars.next(); // '='
            if chars.next().map(|(_, c)| c) != Some('"') {
                bail!("Invalid labels: {line}");
            }
            let mut value = String::new();
            loop {
                match chars.next().map(|(_, c)| c) {
                    Some('"') => break,
                    Some('\\') => match chars.next().map(|(_, c)| c) {
                        Some('n') => value.push('\n'),
                        Some(c) => value.push(c),
                        None => bail!("Invalid labels: {line}"),
                    },
                    Some(c) => value.push(c),
                    None => bail!("Invalid labels: {line}"),
                }
            }
            labels.push((key.trim().to_owned(), value));
        }
        let end = end.ok_or_else(|| anyhow::anyhow!("Invalid labels: {line}"))?;
        rest = &body[end + 1..];
    }
    let value = rest.split_whitespace().next().ok_or_else(|| anyhow::anyhow!("missing value: {line}"))?;
    Ok(Sample { name, labels, value: parse_value(value)? })
}

fn build_family(mut name: String, type_: String, mut samples: Vec<Sample>) -> MetricFamily {
    // Counters use the OpenMetrics representation: family without `_total`, samples with it.
    if type_ == "counter" {
        match name.strip_suffix("_total") {
            Some(stripped) => name = stripped.to_owned(),
            None => samples.iter_mut().for_each(|s| s.name.push_str("_total")),
        }
    }
    MetricFamily { name, type_, samples }
}

/// `text_string_to_metric_families`: `# HELP`/`# TYPE` open a family; a sample whose name the
/// family type does not allow becomes an untyped singleton family.
fn parse_metric_families(text: &str) -> Result<Vec<MetricFamily>> {
    let mut families = Vec::new();
    let mut name = String::new();
    let mut type_ = "untyped".to_string();
    let mut samples: Vec<Sample> = Vec::new();
    let mut allowed: Vec<String> = Vec::new();

    for line in text.lines().map(str::trim) {
        if let Some(comment) = line.strip_prefix('#') {
            let parts: Vec<&str> = comment.split_whitespace().take(3).collect();
            let (kind, candidate) = match parts.as_slice() {
                [kind @ ("HELP" | "TYPE"), candidate, ..] => (*kind, *candidate),
                _ => continue,
            };
            if candidate != name {
                if !name.is_empty() {
                    families.push(build_family(std::mem::take(&mut name), type_.clone(), std::mem::take(&mut samples)));
                }
                name = candidate.to_owned();
                samples.clear();
                if kind == "HELP" {
                    type_ = "untyped".into();
                    allowed = vec![name.clone()];
                }
            }
            if kind == "TYPE" {
                type_ = parts.get(2).map(|t| t.to_string()).ok_or_else(|| anyhow::anyhow!("invalid TYPE line: {line}"))?;
                let suffixes: &[&str] = match type_.as_str() {
                    "summary" => &["_count", "_sum", ""],
                    "histogram" => &["_count", "_sum", "_bucket"],
                    _ => &[""],
                };
                allowed = suffixes.iter().map(|s| format!("{name}{s}")).collect();
            }
        } else if !line.is_empty() {
            let sample = parse_sample(line)?;
            if allowed.contains(&sample.name) {
                samples.push(sample);
            } else {
                if !name.is_empty() {
                    families.push(build_family(std::mem::take(&mut name), type_.clone(), std::mem::take(&mut samples)));
                }
                type_ = "untyped".into();
                allowed.clear();
                families.push(build_family(sample.name.clone(), "untyped".into(), vec![sample]));
            }
        }
    }
    if !name.is_empty() {
        families.push(build_family(name, type_, samples));
    }
    Ok(families)
}

// ── per-engine snapshots ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq)]
struct HistogramSnapshot {
    sum: f64,
    count: f64,
    /// Upper bound (`le`) -> cumulative count, in exposition order.
    buckets: Vec<(f64, f64)>,
}

fn bucket_get(buckets: &[(f64, f64)], le: f64) -> Option<f64> {
    buckets.iter().find(|(b, _)| *b == le).map(|(_, c)| *c)
}

fn bucket_add(buckets: &mut Vec<(f64, f64)>, le: f64, count: f64) {
    match buckets.iter_mut().find(|(b, _)| *b == le) {
        Some(slot) => slot.1 += count,
        None => buckets.push((le, count)),
    }
}

/// All samples scraped for one engine, keyed by vLLM sample name (prefix stripped); samples with
/// extra labels are summed over them.
#[derive(Debug, Clone, Default, PartialEq)]
struct EngineSnapshot {
    gauges: IndexMap<String, f64>,
    counters: IndexMap<String, f64>,
    histograms: IndexMap<String, HistogramSnapshot>,
    /// Labels of `vllm:cache_config_info` (e.g. `num_gpu_blocks`, `block_size`).
    cache_config: HashMap<String, String>,
}

/// `parse_prometheus_text`: one snapshot per `engine` label (default `"0"`), sorted by label.
fn parse_prometheus_text(text: &str) -> Result<Vec<(String, EngineSnapshot)>> {
    let mut engines: IndexMap<String, EngineSnapshot> = IndexMap::new();
    for family in parse_metric_families(text)? {
        if !family.name.starts_with(METRIC_PREFIX) || !matches!(family.type_.as_str(), "counter" | "gauge" | "histogram") {
            continue;
        }
        for sample in &family.samples {
            if sample.name.ends_with("_created") || !sample.value.is_finite() {
                continue;
            }
            let engine = engines.entry(sample.label("engine").unwrap_or("0").to_owned()).or_default();
            if family.name == CACHE_CONFIG_FAMILY {
                engine.cache_config.extend(sample.labels.iter().cloned());
                continue;
            }
            let mut name = sample.name.strip_prefix(METRIC_PREFIX).unwrap_or(&sample.name).to_owned();
            if name == "num_requests_waiting_by_reason" {
                name = format!("num_requests_waiting_reason_{}", sample.label("reason").unwrap_or("unknown"));
            }
            match family.type_.as_str() {
                "gauge" => *engine.gauges.entry(name).or_insert(0.0) += sample.value,
                "counter" => *engine.counters.entry(name).or_insert(0.0) += sample.value,
                _ => {
                    let histogram = engine.histograms.entry(family.name[METRIC_PREFIX.len()..].to_owned()).or_default();
                    if sample.name.ends_with("_sum") {
                        histogram.sum += sample.value;
                    } else if sample.name.ends_with("_count") {
                        histogram.count += sample.value;
                    } else if sample.name.ends_with("_bucket") {
                        let le = parse_value(sample.label("le").ok_or_else(|| anyhow::anyhow!("bucket without le"))?)?;
                        bucket_add(&mut histogram.buckets, le, sample.value);
                    }
                }
            }
        }
    }
    let mut engines: Vec<(String, EngineSnapshot)> = engines.into_iter().collect();
    engines.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(engines)
}

// ── aggregation ───────────────────────────────────────────────────────────────

/// Python 3.12 `sum()` over floats (Neumaier-compensated).
fn py_sum(values: &[f64]) -> f64 {
    let mut result = 0.0f64;
    let mut c = 0.0f64;
    for &x in values {
        let t = result + x;
        if result.abs() >= x.abs() {
            c += (result - t) + x;
        } else {
            c += (x - t) + result;
        }
        result = t;
    }
    if c != 0.0 && c.is_finite() {
        result += c;
    }
    result
}

/// Correctly rounded sum (`math.fsum`, Shewchuk partials).
fn fsum(values: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for &value in values {
        let mut x = value;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    partials.iter().rev().fold(0.0, |acc, p| acc + p)
}

/// `statistics.mean` (exact sum, then divided).
fn py_mean(values: &[f64]) -> f64 {
    fsum(values) / values.len() as f64
}

/// `statistics.median`.
fn py_median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn aggregate(name: &str, values: &[f64]) -> f64 {
    match name {
        "min" => values.iter().cloned().fold(f64::INFINITY, f64::min),
        "max" => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        "sum" => py_sum(values),
        "mean" => py_mean(values),
        "median" => py_median(values),
        _ => unreachable!(),
    }
}

const AGGREGATIONS: [&str; 5] = ["min", "max", "sum", "mean", "median"];

/// Estimate a quantile from cumulative bucket deltas, Prometheus-style.
fn histogram_quantile(buckets: &[(f64, f64)], quantile: f64) -> Option<f64> {
    let total = buckets.iter().map(|(_, c)| *c).fold(0.0, f64::max);
    if total <= 0.0 {
        return None;
    }
    let target = quantile * total;
    let mut sorted = buckets.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("bucket bounds are numbers"));
    let (mut previous_le, mut previous_cumulative) = (0.0, 0.0);
    for (le, cumulative) in sorted {
        if cumulative >= target {
            if le.is_infinite() || cumulative == previous_cumulative {
                return Some(previous_le);
            }
            return Some(previous_le + (le - previous_le) * (target - previous_cumulative) / (cumulative - previous_cumulative));
        }
        previous_le = le;
        previous_cumulative = cumulative;
    }
    None
}

#[derive(Debug, Clone)]
struct TimedSnapshot {
    timestamp: f64,
    snapshot: EngineSnapshot,
}

/// Pass-through values plus interval rates (`:rate`), histogram means (`:mean`) and per-engine
/// ratios for one engine (`engine_values`).
fn engine_values(snapshot: &EngineSnapshot, timestamp: f64, previous: Option<&TimedSnapshot>) -> IndexMap<String, f64> {
    let mut values: IndexMap<String, f64> = snapshot.gauges.clone();
    for (name, value) in &snapshot.counters {
        values.insert(name.clone(), *value);
    }
    for (name, histogram) in &snapshot.histograms {
        values.insert(format!("{name}_sum"), histogram.sum);
        values.insert(format!("{name}_count"), histogram.count);
    }
    let Some(previous) = previous else { return values };
    let dt = timestamp - previous.timestamp;
    if dt <= 0.0 {
        return values;
    }
    for (name, value) in &snapshot.counters {
        let delta = value - previous.snapshot.counters.get(name).copied().unwrap_or(0.0);
        if delta >= 0.0 {
            values.insert(format!("{name}:rate"), delta / dt);
        }
    }
    let empty = HistogramSnapshot::default();
    for (name, histogram) in &snapshot.histograms {
        let prev = previous.snapshot.histograms.get(name).unwrap_or(&empty);
        let sum_delta = histogram.sum - prev.sum;
        let count_delta = histogram.count - prev.count;
        if count_delta >= 0.0 {
            values.insert(format!("{name}:rate"), count_delta / dt);
        }
        if count_delta > 0.0 && sum_delta >= 0.0 {
            values.insert(format!("{name}:mean"), sum_delta / count_delta);
        }
    }
    for (name, numerators, denominators) in RATIO_METRICS {
        let delta = |names: [&str; 2]| -> f64 {
            py_sum(
                &names
                    .iter()
                    .map(|c| snapshot.counters.get(*c).copied().unwrap_or(0.0) - previous.snapshot.counters.get(*c).copied().unwrap_or(0.0))
                    .collect::<Vec<_>>(),
            )
        };
        let (numerator, denominator) = (delta(numerators), delta(denominators));
        if numerator >= 0.0 && denominator > 0.0 {
            values.insert(name.to_owned(), numerator / denominator);
        }
    }
    values
}

/// Per-histogram cumulative bucket deltas since the previous scrape.
fn bucket_deltas(snapshot: &EngineSnapshot, previous: Option<&TimedSnapshot>) -> IndexMap<String, Vec<(f64, f64)>> {
    let mut deltas = IndexMap::new();
    let Some(previous) = previous else { return deltas };
    for (name, histogram) in &snapshot.histograms {
        let previous_buckets = previous.snapshot.histograms.get(name).map(|h| h.buckets.as_slice()).unwrap_or(&[]);
        let delta: Vec<(f64, f64)> =
            histogram.buckets.iter().map(|(le, count)| (*le, count - bucket_get(previous_buckets, *le).unwrap_or(0.0))).collect();
        if !delta.is_empty() && delta.iter().all(|(_, c)| *c >= 0.0) {
            deltas.insert(name.clone(), delta);
        }
    }
    deltas
}

/// Aggregate per-engine values into `inference/{scope}/{metric}/{agg}` series.
fn build_scope_metrics(
    scope: &str,
    values_per_engine: &[&IndexMap<String, f64>],
    bucket_deltas_per_engine: &[&IndexMap<String, Vec<(f64, f64)>>],
    counter_deltas: &IndexMap<String, f64>,
) -> Vec<(String, f64)> {
    let prefix = format!("inference/{scope}");
    let mut metrics = Vec::new();

    let mut names: Vec<&String> = values_per_engine.iter().flat_map(|v| v.keys()).collect();
    names.sort();
    names.dedup();
    for name in names {
        let values: Vec<f64> = values_per_engine.iter().filter_map(|v| v.get(name).copied()).collect();
        for aggregation in AGGREGATIONS {
            metrics.push((format!("{prefix}/{name}/{aggregation}"), aggregate(aggregation, &values)));
        }
    }

    let mut histogram_names: Vec<&String> = bucket_deltas_per_engine.iter().flat_map(|d| d.keys()).collect();
    histogram_names.sort();
    histogram_names.dedup();
    for name in histogram_names {
        let mut pooled: Vec<(f64, f64)> = Vec::new();
        for deltas in bucket_deltas_per_engine {
            for (le, count) in deltas.get(name).map(Vec::as_slice).unwrap_or(&[]) {
                bucket_add(&mut pooled, *le, *count);
            }
        }
        for (label, quantile) in QUANTILES {
            if let Some(value) = histogram_quantile(&pooled, quantile) {
                metrics.push((format!("{prefix}/{name}/{label}"), value));
            }
        }
    }

    for (name, numerators, denominators) in RATIO_METRICS {
        let numerator = py_sum(&numerators.iter().map(|c| counter_deltas.get(*c).copied().unwrap_or(0.0)).collect::<Vec<_>>());
        let denominator = py_sum(&denominators.iter().map(|c| counter_deltas.get(*c).copied().unwrap_or(0.0)).collect::<Vec<_>>());
        if denominator > 0.0 {
            metrics.push((format!("{prefix}/{name}/pooled"), numerator / denominator));
        }
    }
    metrics
}

// ── collector ─────────────────────────────────────────────────────────────────

/// Raw per-engine load facts for one poll; thresholds and verdicts live in the controller.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineLoadSample {
    pub engine_id: String,
    pub role: Option<InferenceRole>,
    pub kv_capacity_tokens: Option<u64>,
    pub max_model_len: Option<u64>,
    pub kv_usage: f64,
    pub running: u64,
    pub waiting: u64,
    /// Requests queued for KV capacity (None when the engine has no by-reason breakdown).
    pub waiting_capacity: Option<u64>,
    pub preemptions_delta: u64,
}

#[derive(Debug, Clone)]
struct MetricsEndpoint {
    client: AdminClient,
    role: Option<InferenceRole>,
    key: String,
    name: String,
}

/// Attach optional P/D roles and stable display names (`server0`, `prefill1`, ...).
fn build_metrics_endpoints(clients: &[AdminClient], roles: Option<&[InferenceRole]>) -> Result<Vec<MetricsEndpoint>> {
    if let Some(roles) = roles {
        if roles.len() != clients.len() {
            bail!("Got {} inference metric role(s) for {} admin client(s)", roles.len(), clients.len());
        }
    }
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    Ok(clients
        .iter()
        .enumerate()
        .map(|(i, client)| {
            let role = roles.map(|r| r[i]);
            let prefix = role.map(|r| r.as_str()).unwrap_or("server");
            let index = counts.entry(prefix).or_insert(0);
            let name = format!("{prefix}{index}");
            *index += 1;
            MetricsEndpoint { client: client.clone(), role, key: client.key().to_owned(), name }
        })
        .collect())
}

#[derive(Debug, Clone)]
struct EngineSample {
    endpoint: usize,
    engine_label: String,
    timestamp: f64,
    snapshot: EngineSnapshot,
}

/// Polls vLLM `/metrics` on every engine. Keys: per-engine `inference/{engine_id}/{metric}` and
/// cross-engine `inference/agg/{metric}/{stat}` (plus `inference/prefill|decode/...` for P/D).
#[derive(Debug)]
pub struct InferenceMetricsCollector {
    endpoints: Vec<MetricsEndpoint>,
    previous: HashMap<(String, String), TimedSnapshot>,
    max_model_len_by_endpoint: HashMap<String, u64>,
    has_pd_roles: bool,
    log_metrics: bool,
    clock: Instant,
    load_tx: Option<mpsc::Sender<Vec<EngineLoadSample>>>,
    metrics_tx: Option<mpsc::Sender<Metrics>>,
}

impl InferenceMetricsCollector {
    /// `load_tx` receives every poll's load samples; `metrics_tx` the time-keyed (step=None)
    /// metrics when `log_metrics`.
    pub fn new(
        clients: &[AdminClient],
        roles: Option<&[InferenceRole]>,
        log_metrics: bool,
        load_tx: Option<mpsc::Sender<Vec<EngineLoadSample>>>,
        metrics_tx: Option<mpsc::Sender<Metrics>>,
    ) -> Result<Self> {
        let endpoints = build_metrics_endpoints(clients, roles)?;
        let has_pd_roles = endpoints.iter().any(|e| e.role == Some(InferenceRole::Prefill))
            && endpoints.iter().any(|e| e.role == Some(InferenceRole::Decode));
        tracing::info!(
            "Collecting inference metrics from {}",
            endpoints.iter().map(|e| format!("{}={}", e.name, e.key)).collect::<Vec<_>>().join(", ")
        );
        Ok(Self {
            endpoints,
            previous: HashMap::new(),
            max_model_len_by_endpoint: HashMap::new(),
            has_pd_roles,
            log_metrics,
            clock: Instant::now(),
            load_tx,
            metrics_tx,
        })
    }

    /// Poll until cancelled; a failed poll is logged and retried on the next interval.
    pub async fn run(mut self, cancel: CancellationToken) {
        loop {
            if let Err(error) = self.collect_and_log().await {
                tracing::warn!("Inference metrics poll failed: {error:#}");
            }
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(POLL_INTERVAL) => {}
            }
        }
    }

    /// Scrape once outside the poll loop; whether any engine answered with metrics.
    pub async fn probe(&mut self, attempts: u32, interval: Duration) -> bool {
        for attempt in 0..attempts {
            if attempt > 0 {
                tokio::time::sleep(interval).await;
            }
            if let Err(error) = self.collect_and_log().await {
                tracing::warn!("Inference metrics probe failed: {error:#}");
            }
            if !self.previous.is_empty() {
                return true;
            }
        }
        false
    }

    async fn fetch(endpoint: &MetricsEndpoint) -> Option<String> {
        let result = async {
            let response = endpoint.client.get("/metrics").timeout(FETCH_TIMEOUT).send().await?.error_for_status()?;
            response.text().await
        }
        .await;
        match result {
            Ok(text) => Some(text),
            Err(error) => {
                tracing::debug!("Failed to fetch metrics from {}: {error}", endpoint.client.base_url());
                None
            }
        }
    }

    async fn collect_and_log(&mut self) -> Result<()> {
        let now = self.clock.elapsed().as_secs_f64();
        let texts = futures::future::join_all(self.endpoints.iter().map(Self::fetch)).await;
        let mut samples = Vec::new();
        for (endpoint, text) in texts.into_iter().enumerate() {
            let Some(text) = text else { continue };
            for (engine_label, snapshot) in parse_prometheus_text(&text)? {
                samples.push(EngineSample { endpoint, engine_label, timestamp: now, snapshot });
            }
        }
        if samples.is_empty() {
            return Ok(());
        }
        self.fetch_max_model_lens().await;
        let (metrics, load) = self.observe(samples);
        if let Some(tx) = &self.load_tx {
            tx.send(load).await.map_err(|_| anyhow::anyhow!("load sample receiver closed"))?;
        }
        if !metrics.is_empty() {
            if let Some(tx) = &self.metrics_tx {
                tx.send(metrics).await.map_err(|_| anyhow::anyhow!("metrics receiver closed"))?;
            }
        }
        Ok(())
    }

    /// Metrics and load samples of one poll's engine samples; they become the next baseline.
    fn observe(&mut self, samples: Vec<EngineSample>) -> (Metrics, Vec<EngineLoadSample>) {
        let metrics = if self.log_metrics { self.build_metrics(&samples) } else { Metrics::new() };
        let load: Vec<EngineLoadSample> = samples.iter().map(|s| self.build_load_sample(s)).collect();
        for sample in samples {
            let key = (self.endpoints[sample.endpoint].key.clone(), sample.engine_label);
            self.previous.insert(key, TimedSnapshot { timestamp: sample.timestamp, snapshot: sample.snapshot });
        }
        (metrics, load)
    }

    /// Cache each engine's max context length from `/v1/models` (static per engine lifetime;
    /// retried next poll on failure).
    async fn fetch_max_model_lens(&mut self) {
        let missing: Vec<&MetricsEndpoint> =
            self.endpoints.iter().filter(|e| !self.max_model_len_by_endpoint.contains_key(&e.key)).collect();
        let results = futures::future::join_all(missing.iter().map(|endpoint| async move {
            let result = async {
                let response = endpoint.client.get("/v1/models").timeout(FETCH_TIMEOUT).send().await?.error_for_status()?;
                anyhow::Ok(response.json::<Value>().await?)
            }
            .await;
            match result {
                Ok(body) => body
                    .get("data")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|card| card.get("max_model_len").and_then(Value::as_u64).filter(|&l| l > 0))
                    .max()
                    .map(|len| (endpoint.key.clone(), len)),
                Err(error) => {
                    tracing::debug!("Failed to fetch max_model_len from {}: {error}", endpoint.client.base_url());
                    None
                }
            }
        }))
        .await;
        self.max_model_len_by_endpoint.extend(results.into_iter().flatten());
    }

    fn previous_of(&self, sample: &EngineSample) -> Option<&TimedSnapshot> {
        self.previous.get(&(self.endpoints[sample.endpoint].key.clone(), sample.engine_label.clone()))
    }

    fn engine_id(&self, sample: &EngineSample) -> String {
        format!("{}.{}", self.endpoints[sample.endpoint].name, sample.engine_label)
    }

    fn build_load_sample(&self, sample: &EngineSample) -> EngineLoadSample {
        let cache = &sample.snapshot.cache_config;
        let digits = |key: &str| cache.get(key).filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit())).and_then(|v| v.parse::<u64>().ok());
        // `kv_cache_size_tokens` is authoritative; `num_gpu_blocks` ignores `num_gpu_blocks_override`.
        let kv_capacity_tokens = digits("kv_cache_size_tokens").or_else(|| Some(digits("num_gpu_blocks")? * digits("block_size")?));
        // The first poll has no baseline: a nonzero cumulative counter is not a fresh preemption.
        let mut preemptions_delta = 0i64;
        if let Some(previous) = self.previous_of(sample) {
            for name in ["num_preemptions", "num_preemptions_total"] {
                if let Some(value) = sample.snapshot.counters.get(name) {
                    let delta = value - previous.snapshot.counters.get(name).copied().unwrap_or(0.0);
                    preemptions_delta = preemptions_delta.max(delta.trunc() as i64);
                }
            }
        }
        let gauge = |name: &str| sample.snapshot.gauges.get(name).copied();
        let endpoint = &self.endpoints[sample.endpoint];
        EngineLoadSample {
            engine_id: self.engine_id(sample),
            role: endpoint.role,
            kv_capacity_tokens,
            max_model_len: self.max_model_len_by_endpoint.get(&endpoint.key).copied(),
            kv_usage: gauge("kv_cache_usage_perc").unwrap_or(0.0),
            running: gauge("num_requests_running").unwrap_or(0.0).trunc() as u64,
            waiting: gauge("num_requests_waiting").unwrap_or(0.0).trunc() as u64,
            waiting_capacity: gauge("num_requests_waiting_reason_capacity").map(|v| v.trunc() as u64),
            preemptions_delta: preemptions_delta as u64,
        }
    }

    fn build_metrics(&self, samples: &[EngineSample]) -> Metrics {
        let values_per_engine: Vec<IndexMap<String, f64>> =
            samples.iter().map(|s| engine_values(&s.snapshot, s.timestamp, self.previous_of(s))).collect();
        let deltas_per_engine: Vec<IndexMap<String, Vec<(f64, f64)>>> =
            samples.iter().map(|s| bucket_deltas(&s.snapshot, self.previous_of(s))).collect();

        let mut metrics = Metrics::new();
        for (sample, values) in samples.iter().zip(&values_per_engine) {
            let engine_id = self.engine_id(sample);
            for (name, value) in values {
                metrics.insert(format!("inference/{engine_id}/{name}"), (*value).into());
            }
        }

        let mut scopes: Vec<(&str, Vec<usize>)> = vec![("agg", (0..samples.len()).collect())];
        if self.has_pd_roles {
            for role in [InferenceRole::Decode, InferenceRole::Prefill] {
                let indices: Vec<usize> = (0..samples.len()).filter(|&i| self.endpoints[samples[i].endpoint].role == Some(role)).collect();
                if !indices.is_empty() {
                    scopes.push((role.as_str(), indices));
                }
            }
        }
        for (scope, indices) in scopes {
            let mut counter_deltas: IndexMap<String, f64> = IndexMap::new();
            for &i in &indices {
                let Some(previous) = self.previous_of(&samples[i]) else { continue };
                for (name, value) in &samples[i].snapshot.counters {
                    let delta = value - previous.snapshot.counters.get(name).copied().unwrap_or(0.0);
                    if delta >= 0.0 {
                        *counter_deltas.entry(name.clone()).or_insert(0.0) += delta;
                    }
                }
            }
            let values: Vec<&IndexMap<String, f64>> = indices.iter().map(|&i| &values_per_engine[i]).collect();
            let deltas: Vec<&IndexMap<String, Vec<(f64, f64)>>> = indices.iter().map(|&i| &deltas_per_engine[i]).collect();
            for (key, value) in build_scope_metrics(scope, &values, &deltas, &counter_deltas) {
                metrics.insert(key, value.into());
            }
        }
        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPOSITION: &str = r#"# HELP vllm:num_requests_running Running requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0"} RUNNING
# TYPE vllm:num_requests_waiting_by_reason gauge
vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 1.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0"} 0.5
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",engine="0",num_gpu_blocks="100"} 1.0
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0"} HITS
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0"} QUERIES
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0"} PREEMPTIONS
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0"} LE1
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf"} COUNT
vllm:e2e_request_latency_seconds_count{engine="0"} COUNT
vllm:e2e_request_latency_seconds_sum{engine="0"} SUM
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 1.0
"#;

    fn poll(collector: &mut InferenceMetricsCollector, timestamp: f64, values: [(&str, &str); 7]) -> (Metrics, Vec<EngineLoadSample>) {
        let text = values.iter().fold(EXPOSITION.to_string(), |text, (k, v)| text.replace(k, v));
        let samples = parse_prometheus_text(&text)
            .unwrap()
            .into_iter()
            .map(|(engine_label, snapshot)| EngineSample { endpoint: 0, engine_label, timestamp, snapshot })
            .collect();
        collector.observe(samples)
    }

    /// Two polls 5 s apart; expected values from the Python collector on the same input.
    #[test]
    fn rates_means_quantiles_and_load_match_python() {
        let client = AdminClient::new("http://h:8000/v1", Default::default()).unwrap();
        let mut collector = InferenceMetricsCollector::new(&[client], None, true, None, None).unwrap();
        poll(&mut collector, 0.0, [("RUNNING", "3.0"), ("HITS", "10.0"), ("QUERIES", "40.0"), ("PREEMPTIONS", "5.0"), ("LE1", "2.0"), ("COUNT", "4.0"), ("SUM", "6.0")]);
        let (metrics, load) =
            poll(&mut collector, 5.0, [("RUNNING", "4.0"), ("HITS", "30.0"), ("QUERIES", "80.0"), ("PREEMPTIONS", "7.0"), ("LE1", "6.0"), ("COUNT", "8.0"), ("SUM", "14.0")]);

        let keys: Vec<&str> = metrics.keys().take(14).map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "inference/server0.0/num_requests_running",
                "inference/server0.0/num_requests_waiting_reason_capacity",
                "inference/server0.0/kv_cache_usage_perc",
                "inference/server0.0/prefix_cache_hits_total",
                "inference/server0.0/prefix_cache_queries_total",
                "inference/server0.0/num_preemptions_total",
                "inference/server0.0/e2e_request_latency_seconds_sum",
                "inference/server0.0/e2e_request_latency_seconds_count",
                "inference/server0.0/prefix_cache_hits_total:rate",
                "inference/server0.0/prefix_cache_queries_total:rate",
                "inference/server0.0/num_preemptions_total:rate",
                "inference/server0.0/e2e_request_latency_seconds:rate",
                "inference/server0.0/e2e_request_latency_seconds:mean",
                "inference/server0.0/prefix_cache_hit_rate",
            ]
        );
        for (key, value) in [
            ("inference/server0.0/prefix_cache_hits_total:rate", 4.0),
            ("inference/server0.0/e2e_request_latency_seconds:rate", 0.8),
            ("inference/server0.0/e2e_request_latency_seconds:mean", 2.0),
            ("inference/agg/num_requests_running/median", 4.0),
            ("inference/agg/e2e_request_latency_seconds/p50", 0.5),
            ("inference/agg/e2e_request_latency_seconds/p99", 0.99),
            ("inference/agg/prefix_cache_hit_rate/pooled", 0.5),
        ] {
            assert_eq!(metrics[key].as_f64(), Some(value), "{key}");
        }
        assert_eq!(metrics.len(), 88);
        assert_eq!(
            load,
            [EngineLoadSample {
                engine_id: "server0.0".into(),
                role: None,
                kv_capacity_tokens: Some(1600),
                max_model_len: None,
                kv_usage: 0.5,
                running: 4,
                waiting: 0,
                waiting_capacity: Some(1),
                preemptions_delta: 2,
            }]
        );
    }

    #[test]
    fn python_sum_is_compensated() {
        assert_eq!(py_sum(&[0.1; 10]), 1.0);
        assert_eq!(py_mean(&[1e16, 1.0, -1e16]), 1.0 / 3.0);
    }
}
