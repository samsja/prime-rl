//! Episode-native train and evaluation metrics (`orchestrator/metrics.py`).

use std::collections::HashSet;
use std::sync::Arc;

use indexmap::IndexMap;
use prl_types::{DispatchFailure, GroupCancellation, Metrics, Scalar, Subset};
use prl_wire::{Episode, Trace};

use crate::algo::{is_trainable, scalar_advantage};
use crate::episodes::{episode_env_name, episode_group_id, EpisodeRef};

/// Metrics for requests that failed outside the verifier episode boundary.
pub fn dispatch_failure_metrics(failures: &[DispatchFailure], prefix: &str, total_attempts: usize) -> Metrics {
    let mut out = Metrics::new();
    if total_attempts == 0 {
        return out;
    }
    out.insert(format!("{prefix}/dispatch_failure/mean"), Scalar::Float(failures.len() as f64 / total_attempts as f64));
    let mut types: Vec<&str> = failures.iter().map(|f| f.error.type_.as_str()).collect();
    types.sort();
    types.dedup();
    for ty in types {
        let count = failures.iter().filter(|f| f.error.type_ == ty).count();
        out.insert(format!("{prefix}/dispatch_failure/{ty}"), Scalar::Float(count as f64));
    }
    out
}

fn comb(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 0..k {
        result = result * u128::from(n - i) / u128::from(i + 1);
    }
    result
}

/// Unbiased pass@k and pass^k for one example's binary rewards, k = powers of 2 in [1, n].
pub fn compute_pass_metrics(rewards: &[f64]) -> IndexMap<String, f64> {
    let n = rewards.len() as u64;
    let c = rewards.iter().filter(|&&r| r == 1.0).count() as u64;
    let mut out = IndexMap::new();
    let mut k = 1;
    while k <= n {
        let n_choose_k = comb(n, k) as f64;
        out.insert(format!("pass@{k}"), 1.0 - comb(n - c, k) as f64 / n_choose_k);
        out.insert(format!("pass^{k}"), comb(c, k) as f64 / n_choose_k);
        k *= 2;
    }
    out
}

/// A distribution with mean, extrema and linearly interpolated percentiles.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stat {
    pub values: Vec<f64>,
}

impl Stat {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.values.iter().sum::<f64>() / self.values.len() as f64
        }
    }

    pub fn max(&self) -> f64 {
        self.values.iter().copied().fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| if v > m { v } else { m }))).unwrap_or(0.0)
    }

    pub fn min(&self) -> f64 {
        self.values.iter().copied().fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| if v < m { v } else { m }))).unwrap_or(0.0)
    }

    pub fn percentile(&self, q: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mut values = self.values.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rank = q / 100.0 * (values.len() - 1) as f64;
        let low = rank as usize;
        let high = (low + 1).min(values.len() - 1);
        values[low] + (values[high] - values[low]) * (rank - low as f64)
    }

    pub fn sum(&self) -> f64 {
        self.values.iter().sum()
    }

    pub fn to_dict(&self, prefix: &str) -> Vec<(String, f64)> {
        if self.values.is_empty() {
            return Vec::new();
        }
        vec![
            (format!("{prefix}/mean"), self.mean()),
            (format!("{prefix}/max"), self.max()),
            (format!("{prefix}/min"), self.min()),
            (format!("{prefix}/p10"), self.percentile(10.0)),
            (format!("{prefix}/p90"), self.percentile(90.0)),
        ]
    }
}

/// Mergeable numeric observations of a reporting window. Rate families share a denominator.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricWindow {
    pub distributions: IndexMap<String, Vec<f64>>,
    pub counts: IndexMap<String, f64>,
    pub rates: IndexMap<String, (IndexMap<String, f64>, usize)>,
}

impl MetricWindow {
    pub fn distribution(&mut self, key: impl Into<String>, stat: &Stat) {
        self.distributions.entry(key.into()).or_default().extend_from_slice(&stat.values);
    }

    pub fn rate<'a>(&mut self, prefix: impl Into<String>, counts: impl IntoIterator<Item = (&'a str, f64)>, total: usize) {
        let entry = self.rates.entry(prefix.into()).or_insert_with(|| (IndexMap::new(), 0));
        for (name, count) in counts {
            *entry.0.entry(name.to_owned()).or_insert(0.0) += count;
        }
        entry.1 += total;
    }

    pub fn count(&mut self, key: impl Into<String>, value: f64) {
        *self.counts.entry(key.into()).or_insert(0.0) += value;
    }

    pub fn extend(&mut self, other: &MetricWindow) {
        for (key, values) in &other.distributions {
            self.distributions.entry(key.clone()).or_default().extend_from_slice(values);
        }
        for (key, &value) in &other.counts {
            self.count(key.clone(), value);
        }
        for (prefix, (counts, total)) in &other.rates {
            self.rate(prefix.clone(), counts.iter().map(|(k, &v)| (k.as_str(), v)), *total);
        }
    }

    pub fn is_empty_counts(&self) -> bool {
        self.counts.is_empty()
    }

    pub fn to_dict(&self) -> Metrics {
        let mut out = Metrics::new();
        for (key, &value) in &self.counts {
            out.insert(key.clone(), Scalar::Float(value));
        }
        for (key, values) in &self.distributions {
            for (k, v) in Stat::new(values.clone()).to_dict(key) {
                out.insert(k, Scalar::Float(v));
            }
        }
        for (prefix, (counts, total)) in &self.rates {
            for (name, &count) in counts {
                let key = if name.is_empty() { prefix.clone() } else { format!("{prefix}/{name}") };
                out.insert(key, Scalar::Float(if *total > 0 { count / *total as f64 } else { 0.0 }));
            }
        }
        out
    }
}

/// A trace joined with its episode and training selection state.
#[derive(Debug, Clone, Copy)]
pub struct TraceRecord<'a> {
    /// Index of the episode in the collection's selected episodes.
    pub episode_index: usize,
    pub episode: &'a EpisodeRef,
    pub trace: &'a Trace,
    pub sampled: bool,
    pub admitted: bool,
    pub cancelled: bool,
}

fn group_id(episode: &EpisodeRef) -> &str {
    episode_group_id(&episode.episode).unwrap_or_default()
}

fn bool_f(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// Trace-level statistics of one agent's records.
pub struct TraceMetrics<'r, 'a> {
    pub records: &'r [TraceRecord<'a>],
}

const TRACE_DISTRIBUTIONS: [&str; 6] =
    ["reward", "num_total_tokens", "num_input_tokens", "num_output_tokens", "num_turns", "num_branches"];
const TRACE_RATES: [&str; 3] = ["is_truncated", "is_timeout", "is_completed"];

fn trace_attr(trace: &Trace, name: &str) -> f64 {
    match name {
        "reward" => trace.reward(),
        "num_total_tokens" => trace.num_total_tokens() as f64,
        "num_input_tokens" => trace.num_input_tokens() as f64,
        "num_output_tokens" => trace.num_output_tokens() as f64,
        "num_turns" => trace.num_turns() as f64,
        "num_branches" => trace.num_branches() as f64,
        "is_truncated" => bool_f(trace.is_truncated()),
        "is_timeout" => bool_f(trace.is_timeout),
        "is_completed" => bool_f(trace.is_completed),
        other => unreachable!("unknown trace attribute {other}"),
    }
}

impl TraceMetrics<'_, '_> {
    pub fn stat(&self, name: &str) -> Stat {
        Stat::new(self.records.iter().map(|r| trace_attr(r.trace, name)).collect())
    }

    fn timing(&self) -> Vec<(String, Stat)> {
        let traces = || self.records.iter().map(|r| &r.trace.timing);
        let phase = |t: &prl_wire::Timing, p: &str| match p {
            "setup" => t.setup.duration(),
            "agent" => t.agent.duration(),
            "finalize" => t.finalize.duration(),
            "scoring" => t.scoring.duration(),
            _ => unreachable!(),
        };
        let phases = ["setup", "agent", "finalize", "scoring"];
        let mut out: Vec<(String, Stat)> =
            phases.iter().map(|p| (p.to_string(), Stat::new(traces().map(|t| phase(t, p)).collect()))).collect();
        out.push(("agent/model".into(), Stat::new(traces().map(|t| t.agent.model.duration).collect())));
        out.push(("agent/harness".into(), Stat::new(traces().map(|t| t.agent.harness.duration).collect())));
        out.push(("total".into(), Stat::new(traces().map(|t| phases.iter().map(|p| phase(t, p)).sum()).collect())));
        out
    }

    fn custom(&self, rewards: bool) -> Vec<(String, Stat)> {
        let mut names: Vec<&str> = Vec::new();
        for r in self.records {
            if rewards {
                names.extend(r.trace.rewards.keys().map(String::as_str));
            } else {
                names.extend(r.trace.metrics.keys().map(String::as_str));
            }
        }
        names.sort();
        names.dedup();
        names
            .into_iter()
            .map(|name| {
                let values = self
                    .records
                    .iter()
                    .filter_map(|r| {
                        if rewards {
                            r.trace.rewards.get(name).map(|v| v.as_ref().map_or(0.0, |rw| rw.value()))
                        } else {
                            r.trace.metrics.get(name).map(|v| v.unwrap_or(0.0))
                        }
                    })
                    .collect();
                (name.to_owned(), Stat::new(values))
            })
            .collect()
    }

    fn solve_rates(&self) -> [(&'static str, f64); 3] {
        let mut groups: IndexMap<&str, Vec<f64>> = IndexMap::new();
        for r in self.records {
            groups.entry(group_id(r.episode)).or_default().push(r.trace.reward());
        }
        let num_groups = groups.len() as f64;
        let solved_none = groups.values().filter(|g| g.iter().sum::<f64>() == 0.0).count() as f64;
        let solved_all = groups.values().filter(|g| g.iter().all(|&r| r == 1.0)).count() as f64;
        [
            ("solved_none", solved_none / num_groups),
            ("solved_all", solved_all / num_groups),
            ("solved_some", 1.0 - (solved_none + solved_all) / num_groups),
        ]
    }

    pub fn snapshot(&self, prefix: &str, subset: Subset) -> MetricWindow {
        let mut out = MetricWindow::default();
        let n = self.records.len();
        for name in TRACE_DISTRIBUTIONS {
            out.distribution(format!("{prefix}/{name}"), &self.stat(name));
        }
        for name in TRACE_RATES {
            out.rate(format!("{prefix}/{name}"), [("mean", self.stat(name).sum())], n);
        }
        for (group, stats) in [("timing", self.timing()), ("metrics", self.custom(false)), ("rewards", self.custom(true))] {
            for (key, stat) in stats {
                out.distribution(format!("{prefix}/{group}/{key}"), &stat);
            }
        }
        if subset == Subset::All {
            let has_error = self.records.iter().filter(|r| r.trace.has_error()).count() as f64;
            out.rate(format!("{prefix}/has_error"), [("mean", has_error)], n);
            let cancelled = self.records.iter().filter(|r| r.cancelled).count() as f64;
            out.rate(format!("{prefix}/cancelled"), [("mean", cancelled)], n);
            let mut error_types: Vec<&str> = self
                .records
                .iter()
                .filter(|r| r.trace.has_error())
                .filter_map(|r| r.trace.last_error().map(|e| e.type_.as_str()))
                .collect();
            error_types.sort();
            for (ty, count) in counted(&error_types) {
                out.count(format!("{prefix}/error/{ty}"), count as f64);
            }
            let dropped = self.records.iter().filter(|r| r.trace.is_dropped()).count() as f64;
            out.rate(format!("{prefix}/dropped"), [("mean", dropped)], n);
            let mut reasons: Vec<&str> = self.records.iter().filter_map(|r| r.trace.dropped.as_deref()).collect();
            reasons.sort();
            for (reason, count) in counted(&reasons) {
                out.count(format!("{prefix}/dropped/{reason}"), count as f64);
            }
            let num_groups = self.records.iter().map(|r| group_id(r.episode)).collect::<HashSet<_>>().len();
            let rates = self.solve_rates();
            out.rate(prefix, rates.iter().map(|&(k, v)| (k, v * num_groups as f64)), num_groups);
        }
        let truncated = self
            .records
            .iter()
            .filter(|r| r.trace.is_truncated() && r.trace.stop_condition.as_deref() != Some("prompt_too_long"))
            .count() as f64;
        out.rate(format!("{prefix}/stop_condition/generation_truncated"), [("", truncated)], n);
        let conditions: Vec<&str> = self.records.iter().filter_map(|r| r.trace.stop_condition.as_deref()).collect();
        let mut unique: IndexMap<&str, f64> = IndexMap::new();
        for c in &conditions {
            *unique.entry(c).or_insert(0.0) += 1.0;
        }
        out.rate(format!("{prefix}/stop_condition"), unique.iter().map(|(&k, &v)| (k, v)), conditions.len());
        out
    }

    pub fn to_dict(&self, prefix: &str, subset: Subset) -> Metrics {
        self.snapshot(prefix, subset).to_dict()
    }
}

/// Runs of equal adjacent items with their counts (input sorted).
fn counted<'a>(sorted: &[&'a str]) -> Vec<(&'a str, usize)> {
    let mut out: Vec<(&str, usize)> = Vec::new();
    for &item in sorted {
        match out.last_mut() {
            Some((last, count)) if *last == item => *count += 1,
            _ => out.push((item, 1)),
        }
    }
    out
}

/// How a collection narrows its records (`TrainEpisodes.effective` / `EvalEpisodes.effective`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Predicate {
    TrainEffective,
    EvalEffective,
}

impl Predicate {
    fn keep(self, r: &TraceRecord<'_>) -> bool {
        let clean = !r.trace.has_error() && !r.trace.is_dropped() && r.trace.agent.trainable;
        match self {
            Predicate::TrainEffective => r.admitted && r.sampled && !r.cancelled && clean,
            Predicate::EvalEffective => clean,
        }
    }
}

/// Episodes with training selection state; the shared core of `TrainEpisodes` and `EvalEpisodes`.
#[derive(Debug, Clone, Default)]
pub struct EpisodeCollection {
    pub episodes: Vec<EpisodeRef>,
    pub sampled_trace_ids: Arc<HashSet<String>>,
    pub admitted: Arc<HashSet<String>>,
    pub cancelled: Arc<HashSet<String>>,
    predicate: Option<Predicate>,
}

impl EpisodeCollection {
    fn all_records(&self) -> Vec<TraceRecord<'_>> {
        let mut out = Vec::new();
        for (episode_index, episode) in self.episodes.iter().enumerate() {
            for trace in episode.traces() {
                out.push(TraceRecord {
                    episode_index,
                    episode,
                    trace,
                    sampled: self.sampled_trace_ids.contains(&trace.id),
                    admitted: self.admitted.contains(episode.id()),
                    cancelled: self.cancelled.contains(episode.id()),
                });
            }
        }
        out
    }

    /// Records of the selected episodes (episode indices refer to `selected_episodes`).
    pub fn records(&self) -> Vec<TraceRecord<'_>> {
        let records = self.all_records();
        let Some(predicate) = self.predicate else { return records };
        let kept: Vec<TraceRecord<'_>> = records.into_iter().filter(|r| predicate.keep(r)).collect();
        // Renumber episode indices to the selected episodes.
        let mut renumber: IndexMap<usize, usize> = IndexMap::new();
        for r in &kept {
            let next = renumber.len();
            renumber.entry(r.episode_index).or_insert(next);
        }
        let mut order: Vec<usize> = renumber.keys().copied().collect();
        order.sort();
        let position: IndexMap<usize, usize> = order.iter().enumerate().map(|(i, &e)| (e, i)).collect();
        kept.into_iter().map(|r| TraceRecord { episode_index: position[&r.episode_index], ..r }).collect()
    }

    pub fn selected_episodes(&self) -> Vec<&EpisodeRef> {
        let Some(predicate) = self.predicate else { return self.episodes.iter().collect() };
        let selected: HashSet<usize> = self.all_records().iter().filter(|r| predicate.keep(r)).map(|r| r.episode_index).collect();
        self.episodes.iter().enumerate().filter(|(i, _)| selected.contains(i)).map(|(_, e)| e).collect()
    }

    pub fn num_traces(&self) -> usize {
        self.records().len()
    }

    pub fn num_total_tokens(&self) -> i64 {
        self.records().iter().map(|r| r.trace.num_total_tokens()).sum()
    }

    pub fn len(&self) -> usize {
        self.selected_episodes().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The selected episodes as verifier episodes; a narrowed collection keeps only its records'
    /// traces and stamps each one's scalar advantage into `info["advantage"]`.
    pub fn vf_episodes(&self) -> Vec<Episode> {
        if self.predicate.is_none() {
            return self.episodes.iter().map(EpisodeRef::to_episode).collect();
        }
        let records = self.records();
        let selected = self.selected_episodes();
        let mut by_episode: Vec<Vec<Trace>> = vec![Vec::new(); selected.len()];
        for r in &records {
            let mut trace = r.trace.clone();
            if let Some(advantage) = scalar_advantage(&trace) {
                trace.info.insert("advantage".to_string(), advantage.into());
            }
            by_episode[r.episode_index].push(trace);
        }
        selected
            .iter()
            .zip(by_episode)
            .map(|(episode, traces)| {
                let mut out = EpisodeRef::subset(episode.episode.clone(), Vec::new()).to_episode();
                out.traces = traces;
                out
            })
            .collect()
    }

    fn narrowed(&self, predicate: Predicate) -> Self {
        Self { predicate: Some(predicate), ..self.clone() }
    }

    fn with_episodes(&self, episodes: Vec<EpisodeRef>) -> Self {
        Self { episodes, ..self.clone() }
    }

    fn episode_metrics<'s>(&'s self, records: &'s [TraceRecord<'s>]) -> EpisodeMetrics<'s> {
        EpisodeMetrics { episodes: self.selected_episodes(), records, cancelled: &self.cancelled }
    }
}

/// Episode- and agent-level statistics.
pub struct EpisodeMetrics<'a> {
    pub episodes: Vec<&'a EpisodeRef>,
    pub records: &'a [TraceRecord<'a>],
    pub cancelled: &'a HashSet<String>,
}

impl<'a> EpisodeMetrics<'a> {
    pub fn by_agent(&self) -> Vec<(&'a str, Vec<TraceRecord<'a>>)> {
        let mut per_agent: IndexMap<&'a str, Vec<TraceRecord<'a>>> = IndexMap::new();
        for r in self.records {
            per_agent.entry(r.trace.agent.name.as_str()).or_default().push(*r);
        }
        per_agent.sort_keys();
        per_agent.into_iter().collect()
    }

    fn episode_stat(&self, attr: &str) -> Stat {
        let mut by_episode = vec![0f64; self.episodes.len()];
        for r in self.records {
            by_episode[r.episode_index] += trace_attr(r.trace, attr);
        }
        Stat::new(by_episode)
    }

    pub fn num_total_tokens(&self) -> Stat {
        self.episode_stat("num_total_tokens")
    }
    pub fn num_input_tokens(&self) -> Stat {
        self.episode_stat("num_input_tokens")
    }
    pub fn num_output_tokens(&self) -> Stat {
        self.episode_stat("num_output_tokens")
    }
    pub fn num_turns(&self) -> Stat {
        self.episode_stat("num_turns")
    }
    pub fn num_branches(&self) -> Stat {
        self.episode_stat("num_branches")
    }

    pub fn reward(&self) -> Stat {
        Stat::new(self.records.iter().map(|r| r.trace.reward()).collect())
    }

    pub fn is_truncated(&self) -> Stat {
        Stat::new(self.records.iter().map(|r| bool_f(r.trace.is_truncated())).collect())
    }

    pub fn is_timeout(&self) -> Stat {
        Stat::new(self.records.iter().map(|r| bool_f(r.trace.is_timeout)).collect())
    }

    pub fn has_error(&self) -> Stat {
        Stat::new(self.episodes.iter().map(|e| bool_f(!e.ok() || e.traces().any(Trace::has_error))).collect())
    }

    pub fn cancelled(&self) -> Stat {
        Stat::new(self.episodes.iter().map(|e| bool_f(self.cancelled.contains(e.id()))).collect())
    }

    pub fn snapshot(&self, prefix: &str, subset: Subset) -> MetricWindow {
        let mut out = MetricWindow::default();
        if self.episodes.is_empty() {
            return out;
        }
        let metric_prefix = format!("{prefix}/{}", subset.as_str());
        for name in ["num_total_tokens", "num_input_tokens", "num_output_tokens", "num_turns", "num_branches"] {
            out.distribution(format!("{metric_prefix}/{name}"), &self.episode_stat(name));
        }
        if subset == Subset::All {
            out.rate(format!("{metric_prefix}/has_error"), [("mean", self.has_error().sum())], self.episodes.len());
            out.rate(format!("{metric_prefix}/cancelled"), [("mean", self.cancelled().sum())], self.episodes.len());
        }
        for (agent, records) in self.by_agent() {
            out.extend(&TraceMetrics { records: &records }.snapshot(&format!("{metric_prefix}/{agent}"), subset));
        }
        out
    }

    /// `TrainMetrics.snapshot`: plus per-agent `is_trainable` / `is_admitted` rates.
    pub fn train_snapshot(&self, prefix: &str, subset: Subset) -> MetricWindow {
        let mut out = self.snapshot(prefix, subset);
        for (agent, records) in self.by_agent() {
            let metric_prefix = format!("{prefix}/{}/{agent}", subset.as_str());
            let trainable = records.iter().filter(|r| is_trainable(r.trace)).count() as f64;
            out.rate(format!("{metric_prefix}/is_trainable"), [("mean", trainable)], records.len());
            let admitted = records.iter().filter(|r| r.admitted).count() as f64;
            out.rate(format!("{metric_prefix}/is_admitted"), [("mean", admitted)], records.len());
        }
        out
    }
}

/// pass@k / pass^k averaged over examples, for binary rewards only.
pub fn pass_at_k(records: &[TraceRecord<'_>]) -> IndexMap<String, f64> {
    if !records.iter().all(|r| r.trace.reward() == 0.0 || r.trace.reward() == 1.0) {
        return IndexMap::new();
    }
    let mut by_example: IndexMap<&str, Vec<f64>> = IndexMap::new();
    for r in records {
        by_example.entry(group_id(r.episode)).or_default().push(r.trace.reward());
    }
    let per_example: Vec<IndexMap<String, f64>> = by_example.values().map(|g| compute_pass_metrics(g)).collect();
    let mut keys: Vec<&String> = per_example.iter().flat_map(|m| m.keys()).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .map(|key| {
            let values: Vec<f64> = per_example.iter().filter_map(|m| m.get(key).copied()).collect();
            (key.clone(), values.iter().sum::<f64>() / values.len() as f64)
        })
        .collect()
}

/// Train-side collection (`TrainEpisodes`).
#[derive(Debug, Clone, Default)]
pub struct TrainEpisodes(pub EpisodeCollection);

impl TrainEpisodes {
    /// `TrainEpisodes(episodes, sampled_trace_ids)`: every episode admitted.
    pub fn new(episodes: Vec<EpisodeRef>, sampled_trace_ids: HashSet<String>) -> Self {
        let admitted = episodes.iter().map(|e| e.id().to_owned()).collect();
        Self(EpisodeCollection {
            episodes,
            sampled_trace_ids: Arc::new(sampled_trace_ids),
            admitted: Arc::new(admitted),
            cancelled: Arc::default(),
            predicate: None,
        })
    }

    /// `TrainEpisodes(cancelled=...).extend(group, admitted=...)` of the train sink.
    pub fn group(episodes: Vec<EpisodeRef>, admitted: bool, cancelled: HashSet<String>) -> Self {
        let admitted = if admitted { episodes.iter().map(|e| e.id().to_owned()).collect() } else { HashSet::new() };
        Self(EpisodeCollection {
            episodes,
            sampled_trace_ids: Arc::default(),
            admitted: Arc::new(admitted),
            cancelled: Arc::new(cancelled),
            predicate: None,
        })
    }

    pub fn effective(&self) -> Self {
        Self(self.0.narrowed(Predicate::TrainEffective))
    }

    pub fn by_env(&self) -> IndexMap<String, TrainEpisodes> {
        let mut grouped: IndexMap<String, Vec<EpisodeRef>> = IndexMap::new();
        for episode in self.0.selected_episodes() {
            let env = episode_env_name(&episode.episode).unwrap_or_default().to_owned();
            grouped.entry(env).or_default().push(episode.clone());
        }
        grouped.into_iter().map(|(env, episodes)| (env, Self(self.0.with_episodes(episodes)))).collect()
    }

    /// Runs `f` over the collection's `TrainMetrics`.
    pub fn with_metrics<R>(&self, f: impl FnOnce(&EpisodeMetrics<'_>) -> R) -> R {
        let records = self.0.records();
        f(&self.0.episode_metrics(&records))
    }

    pub fn snapshot(&self, prefix: &str, subset: Subset) -> MetricWindow {
        self.with_metrics(|m| m.train_snapshot(prefix, subset))
    }

    pub fn to_wandb(&self, prefix: &str, subset: Subset) -> Metrics {
        self.snapshot(prefix, subset).to_dict()
    }
}

impl std::ops::Deref for TrainEpisodes {
    type Target = EpisodeCollection;
    fn deref(&self) -> &EpisodeCollection {
        &self.0
    }
}

/// Eval-side collection (`EvalEpisodes`): one env's epoch.
#[derive(Debug, Clone, Default)]
pub struct EvalEpisodes {
    pub inner: EpisodeCollection,
    pub group_size: usize,
}

impl EvalEpisodes {
    pub fn new(episodes: Vec<Arc<Episode>>, group_size: usize) -> Self {
        let episodes: Vec<EpisodeRef> = episodes.into_iter().map(EpisodeRef::whole).collect();
        let admitted = episodes.iter().map(|e| e.id().to_owned()).collect();
        Self {
            inner: EpisodeCollection {
                episodes,
                sampled_trace_ids: Arc::default(),
                admitted: Arc::new(admitted),
                cancelled: Arc::default(),
                predicate: None,
            },
            group_size,
        }
    }

    pub fn effective(&self) -> Self {
        Self { inner: self.inner.narrowed(Predicate::EvalEffective), group_size: self.group_size }
    }

    pub fn with_metrics<R>(&self, f: impl FnOnce(&EpisodeMetrics<'_>) -> R) -> R {
        let records = self.inner.records();
        f(&self.inner.episode_metrics(&records))
    }

    /// `EvalMetrics.to_wandb`: plus per-agent `avg@{group_size}` and, on `effective`, pass@k / pass^k.
    pub fn to_wandb(&self, prefix: &str, subset: Subset) -> Metrics {
        self.with_metrics(|m| {
            let mut out = m.snapshot(prefix, subset).to_dict();
            for (agent, records) in m.by_agent() {
                let metric_prefix = format!("{prefix}/{}/{agent}", subset.as_str());
                let rewards = Stat::new(records.iter().map(|r| r.trace.reward()).collect());
                out.insert(format!("{metric_prefix}/avg@{}", self.group_size), Scalar::Float(rewards.mean()));
                if subset == Subset::Effective {
                    for (key, value) in pass_at_k(&records) {
                        out.insert(format!("{metric_prefix}/{key}"), Scalar::Float(value));
                    }
                }
            }
            out
        })
    }
}

impl std::ops::Deref for EvalEpisodes {
    type Target = EpisodeCollection;
    fn deref(&self) -> &EpisodeCollection {
        &self.inner
    }
}

/// Completed-group accounting of one reporting window, independent of batch fill.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RolloutWindow {
    pub metrics: MetricWindow,
    pub tokens: i64,
    pub traces: usize,
    pub groups: usize,
    pub attempts: usize,
    pub discarded: usize,
    pub stale: usize,
    pub errored: usize,
    pub cancelled: usize,
}

impl RolloutWindow {
    pub fn observe(
        &mut self,
        episodes: &TrainEpisodes,
        failures: &[DispatchFailure],
        cancellation: Option<&GroupCancellation>,
        env_name: &str,
        queued_trace_ids: &HashSet<String>,
    ) {
        let cancelled = cancellation.map_or(0, |c| c.count);
        let selected = episodes.selected_episodes();
        let num_traces = episodes.num_traces();
        self.tokens += episodes.num_total_tokens();
        self.traces += num_traces;
        self.groups += 1;
        self.attempts += selected.len() + failures.len() + cancelled;
        let discarded: Vec<&EpisodeRef> =
            selected.iter().copied().filter(|e| !e.traces().any(|t| queued_trace_ids.contains(&t.id))).collect();
        self.discarded += discarded.len() + failures.len() + cancelled;
        let voided = discarded.iter().filter(|e| episodes.cancelled.contains(e.id())).count();
        if cancellation.is_some() {
            self.cancelled += voided + cancelled;
        } else {
            self.stale += voided + cancelled;
        }
        self.errored += failures.len()
            + discarded
                .iter()
                .filter(|e| !episodes.cancelled.contains(e.id()) && (!e.ok() || e.traces().any(Trace::has_error)))
                .count();
        let snapshot = |prefix: &str| episodes.snapshot(prefix, Subset::All);
        for prefix in ["train/agg".to_string(), format!("train/{env_name}")] {
            self.metrics.extend(&snapshot(&prefix));
            self.metrics.rate(
                format!("{prefix}/all/dispatch_failure"),
                [("mean", failures.len() as f64)],
                selected.len() + failures.len(),
            );
            for failure in failures {
                self.metrics.count(format!("{prefix}/all/dispatch_failure/{}", failure.error.type_), 1.0);
            }
        }
        self.metrics.rate("rollout/env_share", [(env_name, num_traces as f64)], num_traces);
        self.metrics.count("rollout/traces/queued", queued_trace_ids.len() as f64);
        self.metrics.count("rollout/traces/discarded", num_traces as f64 - queued_trace_ids.len() as f64);
    }
}
