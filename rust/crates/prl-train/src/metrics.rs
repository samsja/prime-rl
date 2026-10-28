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
    let types: Vec<&str> = failures.iter().map(|f| f.error.type_.as_str()).collect();
    for (ty, count) in counted(types) {
        out.insert(format!("{prefix}/dispatch_failure/{ty}"), Scalar::Float(count as f64));
    }
    out
}

/// `(item, count)` per distinct item, sorted.
fn counted(mut items: Vec<&str>) -> Vec<(&str, usize)> {
    items.sort();
    let mut out: Vec<(&str, usize)> = Vec::new();
    for item in items {
        match out.last_mut() {
            Some((last, count)) if *last == item => *count += 1,
            _ => out.push((item, 1)),
        }
    }
    out
}

fn comb(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    (0..k.min(n - k)).fold(1u128, |acc, i| acc * u128::from(n - i) / u128::from(i + 1))
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

/// A distribution with mean, extrema and linearly interpolated percentiles (0.0 when empty).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stat {
    pub values: Vec<f64>,
}

impl Stat {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    pub fn sum(&self) -> f64 {
        self.values.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum() / self.values.len() as f64
        }
    }

    pub fn max(&self) -> f64 {
        self.values.iter().copied().reduce(f64::max).unwrap_or(0.0)
    }

    pub fn min(&self) -> f64 {
        self.values.iter().copied().reduce(f64::min).unwrap_or(0.0)
    }

    pub fn percentile(&self, q: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mut values = self.values.clone();
        values.sort_by(f64::total_cmp);
        let rank = q / 100.0 * (values.len() - 1) as f64;
        let low = rank as usize;
        let high = (low + 1).min(values.len() - 1);
        values[low] + (values[high] - values[low]) * (rank - low as f64)
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
    pub fn distribution(&mut self, key: impl Into<String>, values: &[f64]) {
        self.distributions.entry(key.into()).or_default().extend_from_slice(values);
    }

    pub fn rate<'a>(&mut self, prefix: impl Into<String>, counts: impl IntoIterator<Item = (&'a str, f64)>, total: usize) {
        let entry = self.rates.entry(prefix.into()).or_default();
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
            self.distribution(key.clone(), values);
        }
        for (key, &value) in &other.counts {
            self.count(key.clone(), value);
        }
        for (prefix, (counts, total)) in &other.rates {
            self.rate(prefix.clone(), counts.iter().map(|(k, &v)| (k.as_str(), v)), *total);
        }
    }

    /// Counts, then distributions (`/mean,/max,/min,/p10,/p90`), then rates (`prefix/name`).
    pub fn to_dict(&self) -> Metrics {
        let mut out = Metrics::new();
        for (key, &value) in &self.counts {
            out.insert(key.clone(), value.into());
        }
        for (key, values) in &self.distributions {
            if values.is_empty() {
                continue;
            }
            let stat = Stat::new(values.clone());
            for (name, value) in [("mean", stat.mean()), ("max", stat.max()), ("min", stat.min()), ("p10", stat.percentile(10.0)), ("p90", stat.percentile(90.0))] {
                out.insert(format!("{key}/{name}"), value.into());
            }
        }
        for (prefix, (counts, total)) in &self.rates {
            for (name, &count) in counts {
                let key = if name.is_empty() { prefix.clone() } else { format!("{prefix}/{name}") };
                out.insert(key, (if *total > 0 { count / *total as f64 } else { 0.0 }).into());
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

fn f(b: bool) -> f64 {
    f64::from(u8::from(b))
}

fn values<'a>(records: &[TraceRecord<'a>], value: impl Fn(&'a Trace) -> f64) -> Vec<f64> {
    records.iter().map(|r| value(r.trace)).collect()
}

type TraceValue = fn(&Trace) -> f64;
type TraceFlag = fn(&Trace) -> bool;

/// `TraceMetrics.snapshot`: trace-level statistics of one agent's records.
fn trace_snapshot(records: &[TraceRecord<'_>], prefix: &str, subset: Subset) -> MetricWindow {
    let mut out = MetricWindow::default();
    let n = records.len();
    let distributions: [(&str, TraceValue); 6] = [
        ("reward", |t| t.reward()),
        ("num_total_tokens", |t| t.num_total_tokens() as f64),
        ("num_input_tokens", |t| t.num_input_tokens() as f64),
        ("num_output_tokens", |t| t.num_output_tokens() as f64),
        ("num_turns", |t| t.num_turns() as f64),
        ("num_branches", |t| t.num_branches() as f64),
    ];
    for (name, value) in distributions {
        out.distribution(format!("{prefix}/{name}"), &values(records, value));
    }
    let rates: [(&str, TraceFlag); 3] =
        [("is_truncated", |t| t.is_truncated()), ("is_timeout", |t| t.is_timeout), ("is_completed", |t| t.is_completed)];
    for (name, value) in rates {
        out.rate(format!("{prefix}/{name}"), [("mean", values(records, |t| f(value(t))).iter().sum())], n);
    }

    let phases: [(&str, TraceValue); 4] = [
        ("setup", |t| t.timing.setup.duration()),
        ("agent", |t| t.timing.agent.duration()),
        ("finalize", |t| t.timing.finalize.duration()),
        ("scoring", |t| t.timing.scoring.duration()),
    ];
    for (name, phase) in phases {
        out.distribution(format!("{prefix}/timing/{name}"), &values(records, phase));
    }
    out.distribution(format!("{prefix}/timing/agent/model"), &values(records, |t| t.timing.agent.model.duration));
    out.distribution(format!("{prefix}/timing/agent/harness"), &values(records, |t| t.timing.agent.harness.duration));
    out.distribution(format!("{prefix}/timing/total"), &values(records, |t| phases.iter().map(|(_, phase)| phase(t)).sum()));

    // Named metrics / rewards: a trace without the name is skipped, an unscored one reads 0.0.
    let mut names: Vec<&str> = records.iter().flat_map(|r| r.trace.metrics.keys().map(String::as_str)).collect();
    names.sort();
    names.dedup();
    for name in names {
        let values: Vec<f64> = records.iter().filter_map(|r| r.trace.metrics.get(name).map(|v| v.unwrap_or(0.0))).collect();
        out.distribution(format!("{prefix}/metrics/{name}"), &values);
    }
    let mut names: Vec<&str> = records.iter().flat_map(|r| r.trace.rewards.keys().map(String::as_str)).collect();
    names.sort();
    names.dedup();
    for name in names {
        let values: Vec<f64> =
            records.iter().filter_map(|r| r.trace.rewards.get(name).map(|v| v.as_ref().map_or(0.0, |rw| rw.value()))).collect();
        out.distribution(format!("{prefix}/rewards/{name}"), &values);
    }

    if subset == Subset::All {
        let count = |pred: &dyn Fn(&TraceRecord<'_>) -> bool| records.iter().filter(|r| pred(r)).count() as f64;
        out.rate(format!("{prefix}/has_error"), [("mean", count(&|r| r.trace.has_error()))], n);
        out.rate(format!("{prefix}/cancelled"), [("mean", count(&|r| r.cancelled))], n);
        let error_types = records.iter().filter(|r| r.trace.has_error()).filter_map(|r| r.trace.last_error()).map(|e| e.type_.as_str());
        for (ty, c) in counted(error_types.collect()) {
            out.count(format!("{prefix}/error/{ty}"), c as f64);
        }
        out.rate(format!("{prefix}/dropped"), [("mean", count(&|r| r.trace.is_dropped()))], n);
        for (reason, c) in counted(records.iter().filter_map(|r| r.trace.dropped.as_deref()).collect()) {
            out.count(format!("{prefix}/dropped/{reason}"), c as f64);
        }
        let mut groups: IndexMap<&str, Vec<f64>> = IndexMap::new();
        for r in records {
            groups.entry(group_id(r.episode)).or_default().push(r.trace.reward());
        }
        let num_groups = groups.len() as f64;
        let none = groups.values().filter(|g| g.iter().sum::<f64>() == 0.0).count() as f64;
        let all = groups.values().filter(|g| g.iter().all(|&r| r == 1.0)).count() as f64;
        let solve_rates = [
            ("solved_none", none / num_groups),
            ("solved_all", all / num_groups),
            ("solved_some", 1.0 - (none + all) / num_groups),
        ];
        out.rate(prefix, solve_rates.map(|(k, v)| (k, v * num_groups)), groups.len());
    }
    let truncated = records
        .iter()
        .filter(|r| r.trace.is_truncated() && r.trace.stop_condition.as_deref() != Some("prompt_too_long"))
        .count() as f64;
    out.rate(format!("{prefix}/stop_condition/generation_truncated"), [("", truncated)], n);
    let mut conditions: IndexMap<&str, f64> = IndexMap::new();
    for c in records.iter().filter_map(|r| r.trace.stop_condition.as_deref()) {
        *conditions.entry(c).or_insert(0.0) += 1.0;
    }
    let total = conditions.values().sum::<f64>() as usize;
    out.rate(format!("{prefix}/stop_condition"), conditions.iter().map(|(&k, &v)| (k, v)), total);
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
    fn new(episodes: Vec<EpisodeRef>, sampled_trace_ids: HashSet<String>, admitted: bool, cancelled: HashSet<String>) -> Self {
        let admitted = if admitted { episodes.iter().map(|e| e.id().to_owned()).collect() } else { HashSet::new() };
        Self {
            episodes,
            sampled_trace_ids: Arc::new(sampled_trace_ids),
            admitted: Arc::new(admitted),
            cancelled: Arc::new(cancelled),
            predicate: None,
        }
    }

    /// The selected episodes (all, or those with a record the predicate keeps) and their records.
    fn select(&self) -> (Vec<&EpisodeRef>, Vec<TraceRecord<'_>>) {
        let (mut episodes, mut records) = (Vec::new(), Vec::new());
        for episode in &self.episodes {
            let before = records.len();
            for trace in episode.traces() {
                let record = TraceRecord {
                    episode_index: episodes.len(),
                    episode,
                    trace,
                    sampled: self.sampled_trace_ids.contains(&trace.id),
                    admitted: self.admitted.contains(episode.id()),
                    cancelled: self.cancelled.contains(episode.id()),
                };
                if self.predicate.is_none_or(|p| p.keep(&record)) {
                    records.push(record);
                }
            }
            if self.predicate.is_none() || records.len() > before {
                episodes.push(episode);
            }
        }
        (episodes, records)
    }

    pub fn records(&self) -> Vec<TraceRecord<'_>> {
        self.select().1
    }

    pub fn selected_episodes(&self) -> Vec<&EpisodeRef> {
        self.select().0
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
        let (episodes, records) = self.select();
        let mut traces: Vec<Vec<Trace>> = vec![Vec::new(); episodes.len()];
        for r in &records {
            let mut trace = r.trace.clone();
            if let Some(advantage) = scalar_advantage(&trace) {
                trace.info.insert("advantage".to_string(), advantage.into());
            }
            traces[r.episode_index].push(trace);
        }
        episodes.iter().zip(traces).map(|(episode, traces)| episode.with_traces(traces)).collect()
    }

    fn narrowed(&self, predicate: Predicate) -> Self {
        Self { predicate: Some(predicate), ..self.clone() }
    }

    fn with_metrics<R>(&self, f: impl FnOnce(&EpisodeMetrics<'_>) -> R) -> R {
        let (episodes, records) = self.select();
        f(&EpisodeMetrics { episodes, records: &records, cancelled: &self.cancelled })
    }
}

/// Episode- and agent-level statistics.
pub struct EpisodeMetrics<'a> {
    episodes: Vec<&'a EpisodeRef>,
    records: &'a [TraceRecord<'a>],
    cancelled: &'a HashSet<String>,
}

impl<'a> EpisodeMetrics<'a> {
    fn by_agent(&self) -> Vec<(&'a str, Vec<TraceRecord<'a>>)> {
        let mut per_agent: IndexMap<&'a str, Vec<TraceRecord<'a>>> = IndexMap::new();
        for r in self.records {
            per_agent.entry(r.trace.agent.name.as_str()).or_default().push(*r);
        }
        per_agent.sort_keys();
        per_agent.into_iter().collect()
    }

    /// Per selected episode, the sum over its records.
    fn episode_stat(&self, value: impl Fn(&Trace) -> f64) -> Stat {
        let mut by_episode = vec![0f64; self.episodes.len()];
        for r in self.records {
            by_episode[r.episode_index] += value(r.trace);
        }
        Stat::new(by_episode)
    }

    pub fn num_turns(&self) -> Stat {
        self.episode_stat(|t| t.num_turns() as f64)
    }

    pub fn num_branches(&self) -> Stat {
        self.episode_stat(|t| t.num_branches() as f64)
    }

    pub fn reward(&self) -> Stat {
        Stat::new(values(self.records, |t| t.reward()))
    }

    pub fn is_truncated(&self) -> Stat {
        Stat::new(values(self.records, |t| f(t.is_truncated())))
    }

    pub fn is_timeout(&self) -> Stat {
        Stat::new(values(self.records, |t| f(t.is_timeout)))
    }

    pub fn has_error(&self) -> Stat {
        Stat::new(self.episodes.iter().map(|e| f(!e.episode.ok || e.traces().any(Trace::has_error))).collect())
    }

    fn snapshot(&self, prefix: &str, subset: Subset) -> MetricWindow {
        let mut out = MetricWindow::default();
        if self.episodes.is_empty() {
            return out;
        }
        let metric_prefix = format!("{prefix}/{}", subset.as_str());
        let episode_stats: [(&str, TraceValue); 5] = [
            ("num_total_tokens", |t| t.num_total_tokens() as f64),
            ("num_input_tokens", |t| t.num_input_tokens() as f64),
            ("num_output_tokens", |t| t.num_output_tokens() as f64),
            ("num_turns", |t| t.num_turns() as f64),
            ("num_branches", |t| t.num_branches() as f64),
        ];
        for (name, value) in episode_stats {
            out.distribution(format!("{metric_prefix}/{name}"), &self.episode_stat(value).values);
        }
        if subset == Subset::All {
            let n = self.episodes.len();
            out.rate(format!("{metric_prefix}/has_error"), [("mean", self.has_error().sum())], n);
            let cancelled = self.episodes.iter().filter(|e| self.cancelled.contains(e.id())).count() as f64;
            out.rate(format!("{metric_prefix}/cancelled"), [("mean", cancelled)], n);
        }
        for (agent, records) in self.by_agent() {
            out.extend(&trace_snapshot(&records, &format!("{metric_prefix}/{agent}"), subset));
        }
        out
    }
}

/// pass@k / pass^k averaged over examples, for binary rewards only.
fn pass_at_k(records: &[TraceRecord<'_>]) -> IndexMap<String, f64> {
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
pub struct TrainEpisodes(EpisodeCollection);

impl TrainEpisodes {
    /// A shipped cohort: every episode admitted.
    pub fn new(episodes: Vec<EpisodeRef>, sampled_trace_ids: HashSet<String>) -> Self {
        Self(EpisodeCollection::new(episodes, sampled_trace_ids, true, HashSet::new()))
    }

    /// A finalized group as the train sink records it.
    pub fn group(episodes: Vec<EpisodeRef>, admitted: bool, cancelled: HashSet<String>) -> Self {
        Self(EpisodeCollection::new(episodes, HashSet::new(), admitted, cancelled))
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
        grouped.into_iter().map(|(env, episodes)| (env, Self(EpisodeCollection { episodes, ..self.0.clone() }))).collect()
    }

    pub fn with_metrics<R>(&self, f: impl FnOnce(&EpisodeMetrics<'_>) -> R) -> R {
        self.0.with_metrics(f)
    }

    /// `TrainMetrics.snapshot`: plus per-agent `is_trainable` / `is_admitted` rates.
    pub fn snapshot(&self, prefix: &str, subset: Subset) -> MetricWindow {
        self.with_metrics(|m| {
            let mut out = m.snapshot(prefix, subset);
            for (agent, records) in m.by_agent() {
                let metric_prefix = format!("{prefix}/{}/{agent}", subset.as_str());
                let trainable = records.iter().filter(|r| is_trainable(r.trace)).count() as f64;
                out.rate(format!("{metric_prefix}/is_trainable"), [("mean", trainable)], records.len());
                let admitted = records.iter().filter(|r| r.admitted).count() as f64;
                out.rate(format!("{metric_prefix}/is_admitted"), [("mean", admitted)], records.len());
            }
            out
        })
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
    inner: EpisodeCollection,
    group_size: usize,
}

impl EvalEpisodes {
    pub fn new(episodes: Vec<Arc<Episode>>, group_size: usize) -> Self {
        let episodes = episodes.into_iter().map(EpisodeRef::whole).collect();
        Self { inner: EpisodeCollection::new(episodes, HashSet::new(), true, HashSet::new()), group_size }
    }

    pub fn effective(&self) -> Self {
        Self { inner: self.inner.narrowed(Predicate::EvalEffective), group_size: self.group_size }
    }

    pub fn with_metrics<R>(&self, f: impl FnOnce(&EpisodeMetrics<'_>) -> R) -> R {
        self.inner.with_metrics(f)
    }

    /// `EvalMetrics.to_wandb`: plus per-agent `avg@{group_size}` and, on `effective`, pass@k / pass^k.
    pub fn to_wandb(&self, prefix: &str, subset: Subset) -> Metrics {
        self.with_metrics(|m| {
            let mut out = m.snapshot(prefix, subset).to_dict();
            for (agent, records) in m.by_agent() {
                let metric_prefix = format!("{prefix}/{}/{agent}", subset.as_str());
                let rewards = Stat::new(values(&records, |t| t.reward()));
                out.insert(format!("{metric_prefix}/avg@{}", self.group_size), rewards.mean().into());
                if subset == Subset::Effective {
                    for (key, value) in pass_at_k(&records) {
                        out.insert(format!("{metric_prefix}/{key}"), value.into());
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
        let (selected, records) = episodes.select();
        let num_traces = records.len();
        self.tokens += records.iter().map(|r| r.trace.num_total_tokens()).sum::<i64>();
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
                .filter(|e| !episodes.cancelled.contains(e.id()) && (!e.episode.ok || e.traces().any(Trace::has_error)))
                .count();
        for prefix in ["train/agg".to_string(), format!("train/{env_name}")] {
            self.metrics.extend(&episodes.snapshot(&prefix, Subset::All));
            let attempts = selected.len() + failures.len();
            self.metrics.rate(format!("{prefix}/all/dispatch_failure"), [("mean", failures.len() as f64)], attempts);
            for failure in failures {
                self.metrics.count(format!("{prefix}/all/dispatch_failure/{}", failure.error.type_), 1.0);
            }
        }
        self.metrics.rate("rollout/env_share", [(env_name, num_traces as f64)], num_traces);
        self.metrics.count("rollout/traces/queued", queued_trace_ids.len() as f64);
        self.metrics.count("rollout/traces/discarded", num_traces as f64 - queued_trace_ids.len() as f64);
    }
}
