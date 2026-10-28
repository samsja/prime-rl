//! Training-side group finalization, scalar accounting and batch assembly (`train_sink.py`).
//!
//! `add` takes one completed episode, `fail` a request that produced no episode, and `cancel` a
//! dropped group's `GroupCancellation`. Before every readiness check the sink sweeps the queued
//! traces for tokens past `max_staleness` — this sweep is what guarantees no stale token trains.
//! The consumer drains `take_rollout_window` before shipping a batch.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use indexmap::IndexMap;
use prl_types::{CancelReason, DispatchFailure, GroupCancellation, Kind};
use prl_wire::payload::trace_payload_urls;
use prl_wire::Episode;

use crate::algo::{iter_trainable_traces, stamp_loss_routing, Algorithm};
use crate::episodes::{episode_env_name, episode_group_id, train_work, EpisodeRef};
use crate::metrics::{RolloutWindow, TrainEpisodes};
use crate::sample::{sample_payload_urls, TrainingSample};
use crate::staleness::{mask_stale_tokens, min_fresh_version, stale_token_runs, token_staleness};
use crate::trajectories::{iter_trainable_branches, trace_to_samples};

const KEEP_ARRAYS_HINT: &str = "Serve the env with prime-rl's `env-server` entrypoint, which keeps them for the trainer, \
and check its log for arrays it could not keep.";

/// The curriculum's verdict on a finalized group (`on_result`).
#[async_trait]
pub trait Admission: Send {
    async fn admit(&mut self, group: &[Episode]) -> Result<bool>;
}

/// Admits every group (no curriculum).
pub struct AdmitAll;

#[async_trait]
impl Admission for AdmitAll {
    async fn admit(&mut self, _group: &[Episode]) -> Result<bool> {
        Ok(true)
    }
}

/// Frees env-server arrays of samples and traces that will never ship.
#[derive(Clone)]
pub struct PayloadReleaser(Arc<dyn Fn(Vec<String>) + Send + Sync>);

impl PayloadReleaser {
    pub fn new(release: impl Fn(Vec<String>) + Send + Sync + 'static) -> Self {
        Self(Arc::new(release))
    }

    pub fn release(&self, urls: Vec<String>) {
        if !urls.is_empty() {
            (self.0)(urls);
        }
    }
}

impl Default for PayloadReleaser {
    /// Fire-and-forget HTTP DELETEs (`delete_payloads`).
    fn default() -> Self {
        Self::new(prl_wire::payload::delete_payloads)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinkConfig {
    pub batch_size: usize,
    pub max_staleness: i64,
    pub constant_trainer_batch_size: bool,
}

/// What the sink needs of one train env.
pub struct TrainEnvSpec {
    pub group_size: usize,
    pub temperature: f32,
    pub requires_sampling_masks: bool,
    pub requires_top_logprobs: bool,
    pub algorithm: Box<dyn Algorithm>,
}

/// Exactly the shipped cohort and its trainer payload.
pub struct TrainBatch {
    pub cohort: TrainEpisodes,
    pub samples: Vec<TrainingSample>,
}

/// Remove zero-advantage tokens from the RL component; false when the sample has no loss left.
pub fn prune_zero_advantages(sample: &mut TrainingSample) -> Result<bool> {
    let Some(advantages) = &sample.advantages else { return Ok(true) };
    let rl_weights: Vec<f32> = match &sample.rl_weights {
        Some(w) => w.clone(),
        None => sample.mask.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect(),
    };
    if !(sample.mask.len() == advantages.len() && advantages.len() == rl_weights.len()) {
        bail!(
            "mask, advantages and rl_weights must align: {}, {}, {}",
            sample.mask.len(),
            advantages.len(),
            rl_weights.len()
        );
    }
    let pruned: Vec<bool> =
        (0..rl_weights.len()).map(|i| sample.mask[i] && advantages[i] == 0.0 && rl_weights[i] != 0.0).collect();
    if !pruned.iter().any(|&p| p) {
        return Ok(true);
    }
    let mut rl_weights = rl_weights;
    for (w, p) in rl_weights.iter_mut().zip(pruned) {
        if p {
            *w = 0.0;
        }
    }
    sample.rl_weights = Some(rl_weights);
    Ok(has_loss(sample))
}

pub fn has_loss(sample: &TrainingSample) -> bool {
    let has_rl = match &sample.rl_weights {
        Some(w) => sample.mask.iter().zip(w).any(|(&m, &w)| m && w != 0.0),
        None => sample.mask.iter().any(|&m| m),
    };
    let any_nonzero = |w: &Option<Vec<f32>>| w.as_ref().is_some_and(|w| w.iter().any(|&x| x != 0.0));
    has_rl || any_nonzero(&sample.ce_weights) || any_nonzero(&sample.ref_kl_weights)
}

/// Score native episodes, admit groups, then compile trainer payloads.
pub struct TrainSink {
    cfg: SinkConfig,
    envs: IndexMap<String, TrainEnvSpec>,
    payloads: PayloadReleaser,
    pub rollout_window: RolloutWindow,
    pending_groups: HashMap<String, Vec<Episode>>,
    pending_group_failures: HashMap<String, Vec<DispatchFailure>>,
    /// A dropped group's terminal marker; its `count` fills in for the undelivered episodes.
    pending_group_cancellations: HashMap<String, GroupCancellation>,
    pending_batch: IndexMap<String, Vec<TrainingSample>>,
    episode_by_trace: HashMap<String, EpisodeRef>,
    /// Step of the last full staleness sweep: queued traces only age when the step advances.
    swept_step: i64,
    zero_output_units: usize,
    reported_zero_output_windows: usize,
}

impl TrainSink {
    pub fn new(cfg: SinkConfig, envs: IndexMap<String, TrainEnvSpec>, payloads: PayloadReleaser) -> Self {
        Self {
            cfg,
            envs,
            payloads,
            rollout_window: RolloutWindow::default(),
            pending_groups: HashMap::new(),
            pending_group_failures: HashMap::new(),
            pending_group_cancellations: HashMap::new(),
            pending_batch: IndexMap::new(),
            episode_by_trace: HashMap::new(),
            swept_step: 0,
            zero_output_units: 0,
            reported_zero_output_windows: 0,
        }
    }

    /// Connect every env's algorithm resources.
    pub async fn setup(&mut self) -> Result<()> {
        for env in self.envs.values_mut() {
            env.algorithm.setup().await?;
        }
        Ok(())
    }

    /// Delete the env-server arrays of samples that will never ship.
    pub fn release_samples(&self, samples: &[TrainingSample]) {
        self.payloads.release(sample_payload_urls(samples));
    }

    /// `rollout_window.metrics.count("rollout/queued/cancelled", n)`.
    pub fn count_queued_cancelled(&mut self, n: f64) {
        self.rollout_window.metrics.count("rollout/queued/cancelled", n);
    }

    fn env(&self, env_name: &str) -> Result<&TrainEnvSpec> {
        self.envs.get(env_name).ok_or_else(|| anyhow!("unknown train env {env_name:?}"))
    }

    pub fn batch_progress(&self) -> (usize, usize) {
        (self.pending_batch.len(), self.cfg.batch_size)
    }

    pub fn buffered_count(&self) -> usize {
        self.pending_groups.values().map(Vec::len).sum::<usize>() + self.pending_group_failures.values().map(Vec::len).sum::<usize>()
    }

    pub fn pending_batch_by_env(&self) -> IndexMap<String, usize> {
        let mut counts = IndexMap::new();
        for trace_id in self.pending_batch.keys() {
            let env = episode_env_name(&self.episode_by_trace[trace_id].episode).unwrap_or_default();
            *counts.entry(env.to_owned()).or_insert(0) += 1;
        }
        counts
    }

    /// Process one completed episode; returns a batch when ready.
    pub async fn add(&mut self, mut episode: Episode, step: u64, admission: &mut dyn Admission) -> Result<Option<TrainBatch>> {
        let env_name = episode_env_name(&episode)?.to_owned();
        let group_id = episode_group_id(&episode)?.to_owned();
        let env = self.envs.get_mut(&env_name).ok_or_else(|| anyhow!("unknown train env {env_name:?}"))?;
        env.algorithm.finalize_episode(&mut episode).await?;
        self.pending_groups.entry(group_id.clone()).or_default().push(episode);
        if !self.group_complete(&group_id, &env_name)? {
            return Ok(None);
        }
        self.process_group(&group_id, step, admission).await?;
        self.take_batch(step).await
    }

    /// A dropped group's terminal marker; a `shutdown` drop voids the group's arrived episodes.
    pub async fn cancel(&mut self, cancellation: GroupCancellation, step: u64, admission: &mut dyn Admission) -> Result<Option<TrainBatch>> {
        let (group_id, env_name) = (cancellation.group_id.clone(), cancellation.env_name.clone());
        self.pending_group_cancellations.insert(group_id.clone(), cancellation);
        if !self.group_complete(&group_id, &env_name)? {
            return Ok(None);
        }
        self.process_group(&group_id, step, admission).await?;
        self.take_batch(step).await
    }

    /// Count a request failure toward its group without presenting it as an episode.
    pub async fn fail(&mut self, failure: DispatchFailure, step: u64, admission: &mut dyn Admission) -> Result<Option<TrainBatch>> {
        if failure.kind != Kind::Train {
            bail!("TrainSink cannot process a {} dispatch failure", failure.kind);
        }
        let (group_id, env_name) = (failure.group_id.clone(), failure.env_name.clone());
        self.pending_group_failures.entry(group_id.clone()).or_default().push(failure);
        if !self.group_complete(&group_id, &env_name)? {
            return Ok(None);
        }
        self.process_group(&group_id, step, admission).await?;
        self.take_batch(step).await
    }

    fn group_complete(&self, group_id: &str, env_name: &str) -> Result<bool> {
        let cancelled = self.pending_group_cancellations.get(group_id).map_or(0, |c| c.count);
        let failed = self.pending_group_failures.get(group_id).map_or(0, Vec::len);
        let arrived = self.pending_groups.get(group_id).map_or(0, Vec::len);
        Ok(arrived + failed + cancelled >= self.env(env_name)?.group_size)
    }

    /// Sweep stale queued traces, then cut a batch if the survivors still meet the threshold.
    pub async fn take_batch(&mut self, step: u64) -> Result<Option<TrainBatch>> {
        self.drop_stale(step as i64, None)?;
        if self.pending_batch.len() >= self.cfg.batch_size {
            return Ok(Some(self.process_batch(step as i64).await?));
        }
        Ok(None)
    }

    /// Mask queued tokens past `max_staleness` out of the loss and void the samples left with
    /// nothing to train. `trace_ids` scopes the check to a freshly inserted group; otherwise the
    /// full sweep runs once per step.
    fn drop_stale(&mut self, step: i64, trace_ids: Option<Vec<String>>) -> Result<()> {
        let trace_ids = match trace_ids {
            Some(ids) => ids,
            None => {
                if self.swept_step == step {
                    return Ok(());
                }
                self.swept_step = step;
                self.pending_batch.keys().cloned().collect()
            }
        };
        let min_version = min_fresh_version(step, self.cfg.max_staleness);
        if min_version <= 0 {
            return Ok(());
        }
        let mut dropped = 0usize;
        for trace_id in trace_ids {
            let episode = self.episode_by_trace[&trace_id].clone();
            let Some(policy) = train_work(&episode.episode)?.1 else { continue };
            if policy.start >= min_version {
                continue;
            }
            let branches = episode.first_trace().branches();
            let samples = self.pending_batch.get_mut(&trace_id).expect("queued trace has samples");
            let mut kept = Vec::new();
            let mut voided = Vec::new();
            for mut sample in std::mem::take(samples) {
                let branch = &branches[sample.branch_index.unwrap_or_default() as usize];
                let masked = mask_stale_tokens(&mut sample, branch, policy.start, min_version);
                if masked == 0 || has_loss(&sample) {
                    kept.push(sample);
                } else {
                    voided.push(sample);
                }
            }
            self.release_samples(&voided);
            if !kept.is_empty() {
                *self.pending_batch.get_mut(&trace_id).unwrap() = kept;
                continue;
            }
            self.pending_batch.shift_remove(&trace_id);
            self.episode_by_trace.remove(&trace_id);
            dropped += 1;
        }
        if dropped > 0 {
            self.rollout_window.metrics.count("staleness/rollout/dropped", dropped as f64);
            tracing::warn!(
                "Dropped {dropped} queued traces with every token past max_staleness={}. Consider increasing it to avoid this.",
                self.cfg.max_staleness
            );
        }
        Ok(())
    }

    async fn process_group(&mut self, group_id: &str, step: u64, admission: &mut dyn Admission) -> Result<()> {
        let delivered: Vec<(String, Vec<String>)> = self
            .pending_groups
            .get(group_id)
            .into_iter()
            .flatten()
            .flat_map(|e| e.traces.iter())
            .map(|t| (t.id.clone(), trace_payload_urls(t)))
            .collect();
        let result = self.process_group_inner(group_id, step as i64, admission).await;
        // Every delivered trace's arrays are either queued for a batch or deleted here.
        let mut release = Vec::new();
        for (trace_id, urls) in delivered {
            let queued: HashSet<String> = self.pending_batch.get(&trace_id).map(sample_payload_urls).unwrap_or_default().into_iter().collect();
            release.extend(urls.into_iter().filter(|u| !queued.contains(u)));
        }
        self.payloads.release(release);
        result
    }

    async fn process_group_inner(&mut self, group_id: &str, step: i64, admission: &mut dyn Admission) -> Result<()> {
        let mut group = self.pending_groups.remove(group_id).unwrap_or_default();
        let failures = self.pending_group_failures.remove(group_id).unwrap_or_default();
        let cancellation = self.pending_group_cancellations.remove(group_id);
        if group.is_empty() && failures.is_empty() && cancellation.is_none() {
            return Ok(());
        }
        let env_name = match (group.first(), failures.first(), &cancellation) {
            (Some(e), _, _) => episode_env_name(e)?.to_owned(),
            (None, Some(f), _) => f.env_name.clone(),
            (None, None, Some(c)) => c.env_name.clone(),
            _ => unreachable!(),
        };
        let num_traces: usize = group.iter().map(|e| e.traces.len()).sum();
        let task_idx = group
            .iter()
            .flat_map(|e| e.traces.iter())
            .next()
            .map_or("None".to_string(), |t| t.task.idx().map_or("None".to_string(), |i| i.to_string()));
        let num_errored = group.iter().flat_map(|e| e.traces.iter()).filter(|t| t.has_error()).count()
            + group.iter().filter(|e| e.traces.is_empty() && !e.ok).count()
            + failures.len();
        let n_owed = group.len() + failures.len() + cancellation.as_ref().map_or(0, |c| c.count);

        // A shutdown drop voids the whole group and bypasses the curriculum.
        if let Some(c) = cancellation.as_ref().filter(|c| c.reason == CancelReason::Shutdown) {
            let group = arc_all(group);
            let cancelled = group.iter().map(|e| e.id().to_owned()).collect();
            self.record_group(&group, &failures, cancellation.as_ref(), &env_name, false, cancelled, HashSet::new());
            self.record_zero_output(&group, 0, n_owed);
            tracing::debug!(
                "Dropped group | env={env_name} task_idx={task_idx} | episodes={} traces={num_traces} (errored={num_errored}) | reason=cancelled ({})",
                group.len(),
                c.reason.as_str()
            );
            return Ok(());
        }

        let survivors = iter_trainable_traces(&group);
        if !survivors.is_empty() {
            self.envs.get_mut(&env_name).ok_or_else(|| anyhow!("unknown train env {env_name:?}"))?.algorithm.score_group(&mut group).await?;
        }
        let admitted = if group.is_empty() { false } else { admission.admit(&group).await? };
        let group = arc_all(group);
        if survivors.is_empty() || !admitted {
            self.record_group(&group, &failures, cancellation.as_ref(), &env_name, admitted, HashSet::new(), HashSet::new());
            self.record_zero_output(&group, survivors.len(), n_owed);
            let reason = if survivors.is_empty() { "no trainable survivors" } else { "rejected by curriculum" };
            tracing::debug!(
                "Dropped group | env={env_name} task_idx={task_idx} | episodes={} traces={num_traces} (errored={num_errored}) | reason={reason}",
                group.len()
            );
            return Ok(());
        }

        let env = self.env(&env_name)?;
        let (temperature, requires_masks, requires_top) = (env.temperature, env.requires_sampling_masks, env.requires_top_logprobs);
        let action_loss_type = env.algorithm.action_loss_type();
        let constant = self.cfg.constant_trainer_batch_size;
        let jobs: Vec<(Arc<Episode>, usize)> = survivors.iter().map(|&(e, t)| (group[e].episode.clone(), t)).collect();
        let env_for_samples = env_name.clone();
        let built: Vec<(String, Vec<TrainingSample>)> = tokio::task::spawn_blocking(move || -> Result<_> {
            let mut out = Vec::new();
            for (episode, t) in jobs {
                let trace = &episode.traces[t];
                let mut samples = trace_to_samples(trace, &env_for_samples)?;
                for sample in &mut samples {
                    sample.temperatures = vec![temperature; sample.token_ids.len()];
                    if requires_masks && !sample.kept("sampling_mask") {
                        bail!(
                            "env '{env_for_samples}' samples with truncation (top_p/top_k) but its rollouts carry no sampling masks. \
Set `enable_return_sampling_mask = true` on the inference server config (the rl entrypoint does this automatically) - \
it requires vLLM's native sampling-mask capture (>= 0.28). {KEEP_ARRAYS_HINT}"
                        );
                    }
                    if requires_top && !(sample.kept("top_logprobs") || sample.kept("sampling_mask_logprobs")) {
                        bail!("env '{env_for_samples}' requested sampler top logprobs but returned none. {KEEP_ARRAYS_HINT}");
                    }
                    stamp_loss_routing(sample, action_loss_type);
                }
                if constant {
                    let mut kept = Vec::new();
                    for mut sample in samples {
                        if prune_zero_advantages(&mut sample)? {
                            kept.push(sample);
                        }
                    }
                    samples = kept;
                }
                if !samples.is_empty() {
                    out.push((trace.id.clone(), samples));
                }
            }
            Ok(out)
        })
        .await??;

        if built.is_empty() {
            self.record_group(&group, &failures, cancellation.as_ref(), &env_name, true, HashSet::new(), HashSet::new());
            self.record_zero_output(&group, survivors.len(), n_owed);
            return Ok(());
        }

        let new_ids: Vec<String> = built.iter().map(|(id, _)| id.clone()).collect();
        let new_set: HashSet<&String> = new_ids.iter().collect();
        for (trace_id, samples) in built {
            self.pending_batch.insert(trace_id, samples);
        }
        for episode in &group {
            for (t, trace) in episode.episode.traces.iter().enumerate() {
                if new_set.contains(&trace.id) {
                    self.episode_by_trace.insert(trace.id.clone(), EpisodeRef::subset(episode.episode.clone(), vec![t]));
                }
            }
        }
        self.drop_stale(step, Some(new_ids.clone()))?;
        let queued: HashSet<String> = new_ids.iter().filter(|id| self.pending_batch.contains_key(*id)).cloned().collect();
        let stale: HashSet<String> = group
            .iter()
            .filter(|e| e.traces().any(|t| new_set.contains(&t.id)) && !e.traces().any(|t| queued.contains(&t.id)))
            .map(|e| e.id().to_owned())
            .collect();
        let none_queued = queued.is_empty();
        self.record_group(&group, &failures, cancellation.as_ref(), &env_name, true, stale, queued);
        // A fully-voided group shipped nothing: advance the zero-output tally instead of resetting it.
        if none_queued {
            self.record_zero_output(&group, 0, n_owed);
            return Ok(());
        }
        self.zero_output_units = 0;
        self.reported_zero_output_windows = 0;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_group(
        &mut self,
        group: &[EpisodeRef],
        failures: &[DispatchFailure],
        cancellation: Option<&GroupCancellation>,
        env_name: &str,
        admitted: bool,
        cancelled: HashSet<String>,
        queued_trace_ids: HashSet<String>,
    ) {
        let episodes = TrainEpisodes::group(group.to_vec(), admitted, cancelled);
        self.rollout_window.observe(&episodes, failures, cancellation, env_name, &queued_trace_ids);
    }

    /// Transfer accumulated step statistics before shipment (`take_rollout_window(force=True)`).
    pub fn take_rollout_window(&mut self) -> Option<RolloutWindow> {
        if self.rollout_window.groups == 0 && self.rollout_window.metrics.counts.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.rollout_window))
    }

    /// Release accepted traces that cannot ship after training stops.
    pub fn discard_queued(&mut self) {
        if !self.pending_batch.is_empty() {
            self.count_queued_cancelled(self.pending_batch.len() as f64);
        }
        let mut urls: Vec<String> = sample_payload_urls(self.pending_batch.values().flatten());
        urls.extend(self.pending_groups.values().flatten().flat_map(|e| e.traces.iter()).flat_map(trace_payload_urls));
        self.payloads.release(urls);
        self.pending_batch.clear();
        self.episode_by_trace.clear();
    }

    /// `n_owed` counts the group's full episode budget, so dropped groups advance the zero-output
    /// tally at the same rate as fully-delivered ones.
    fn record_zero_output(&mut self, group: &[EpisodeRef], survivors: usize, n_owed: usize) {
        let returned_traces: usize = group.iter().map(EpisodeRef::num_traces).sum();
        self.zero_output_units += if survivors > 0 {
            survivors
        } else if returned_traces > 0 {
            returned_traces
        } else {
            n_owed
        };
        let windows = self.zero_output_units / self.cfg.batch_size;
        if windows <= self.reported_zero_output_windows {
            return;
        }
        self.reported_zero_output_windows = windows;
        tracing::warn!(
            "No admitted train payload after {} finalized units ({windows} zero-output batch equivalents)",
            self.zero_output_units
        );
    }

    async fn process_batch(&mut self, step: i64) -> Result<TrainBatch> {
        let take = self.cfg.batch_size.min(self.pending_batch.len());
        let selected: Vec<(String, Vec<TrainingSample>)> = self.pending_batch.drain(..take).collect();
        let selected_ids: Vec<String> = selected.iter().map(|(id, _)| id.clone()).collect();
        let selected_urls: Vec<String> = selected.iter().flat_map(|(_, s)| sample_payload_urls(s)).collect();
        let jobs: Vec<(String, EpisodeRef, Vec<TrainingSample>)> = selected
            .into_iter()
            .map(|(id, samples)| {
                let episode = self.episode_by_trace[&id].clone();
                (id, episode, samples)
            })
            .collect();
        let (constant, max_staleness) = (self.cfg.constant_trainer_batch_size, self.cfg.max_staleness);
        let (shipped, kept, masked) = tokio::task::spawn_blocking(move || -> Result<_> {
            let mut shipped: Vec<(String, EpisodeRef, Vec<TrainingSample>)> = Vec::new();
            for (id, episode, samples) in jobs {
                let samples = if constant {
                    samples
                } else {
                    let mut kept = Vec::new();
                    for mut sample in samples {
                        if prune_zero_advantages(&mut sample)? {
                            kept.push(sample);
                        }
                    }
                    kept
                };
                if !samples.is_empty() {
                    shipped.push((id, episode, samples));
                }
            }
            let trained_version = step - 1;
            let min_version = min_fresh_version(step, max_staleness);
            let (mut kept, mut masked) = (0usize, 0usize);
            for (_, episode, samples) in &mut shipped {
                let policy = train_work(&episode.episode)?.1.cloned();
                let trace = episode.first_trace();
                let branches = trace.branches();
                if let Some(policy) = &policy {
                    for sample in samples.iter_mut() {
                        let branch = &branches[sample.branch_index.unwrap_or_default() as usize];
                        sample.staleness = Some(token_staleness(branch, policy.start, trained_version));
                    }
                }
                let stale = policy.as_ref().is_some_and(|p| p.start < min_version);
                for sample in samples.iter() {
                    kept += sample.mask.iter().filter(|&&m| m).count();
                }
                if let (true, Some(policy)) = (stale, &policy) {
                    // Shared fork prefixes train only in their first branch; count each stale token once.
                    let trainable: HashMap<usize, Vec<bool>> =
                        iter_trainable_branches(trace).into_iter().map(|(b, mask)| (b.index, mask)).collect();
                    for sample in samples.iter() {
                        let index = sample.branch_index.unwrap_or_default() as usize;
                        let mask = &trainable[&index];
                        for (start, end) in stale_token_runs(&branches[index], policy.start, min_version) {
                            masked += mask[start..end].iter().filter(|&&m| m).count();
                        }
                    }
                }
            }
            Ok((shipped, kept, masked))
        })
        .await??;

        let shipped_urls: HashSet<String> = shipped.iter().flat_map(|(_, _, s)| sample_payload_urls(s)).collect();
        self.payloads.release(selected_urls.into_iter().filter(|u| !shipped_urls.contains(u)).collect());
        self.rollout_window.metrics.rate("staleness/token/masked", [("", masked as f64)], kept + masked);

        let shipped_ids: HashSet<String> = shipped.iter().map(|(id, _, _)| id.clone()).collect();
        let mut cohort: IndexMap<String, (Arc<Episode>, Vec<usize>)> = IndexMap::new();
        for trace_id in &selected_ids {
            let episode = self.episode_by_trace.remove(trace_id).expect("selected trace is tracked");
            if shipped_ids.contains(trace_id) {
                cohort
                    .entry(episode.id().to_owned())
                    .or_insert_with(|| (episode.episode.clone(), Vec::new()))
                    .1
                    .extend_from_slice(episode.trace_indices());
            }
        }
        let pruned = selected_ids.len() - shipped_ids.len();
        if pruned > 0 {
            self.rollout_window.metrics.count("rollout/queued/pruned", pruned as f64);
        }
        let cohort_episodes = cohort.into_values().map(|(episode, traces)| EpisodeRef::subset(episode, traces)).collect();
        let samples = shipped.into_iter().flat_map(|(_, _, s)| s).collect();
        Ok(TrainBatch { cohort: TrainEpisodes::new(cohort_episodes, shipped_ids), samples })
    }
}

fn arc_all(group: Vec<Episode>) -> Vec<EpisodeRef> {
    group.into_iter().map(|e| EpisodeRef::whole(Arc::new(e))).collect()
}
