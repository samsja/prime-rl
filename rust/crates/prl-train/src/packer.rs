//! `BatchPacker` (`orchestrator/batch.py`): packs a step's samples into one micro batch list per
//! trainer data rank. CPU-bound: call `pack` from `spawn_blocking`.

use anyhow::{bail, ensure, Result};

use crate::partition::balanced_partition;
use crate::sample::{MMImageRef, MMRefs, TrainingSample};
use crate::transport::MicroBatch;

/// The trainer model's `FlopCounter` tabulated per document length (`per_len[n] = bin_cost([n])`);
/// the cost of packed documents is the sum of theirs.
pub struct FlopTable {
    per_len: Vec<i128>,
}

impl FlopTable {
    pub fn new(per_len: Vec<i128>) -> Self {
        Self { per_len }
    }

    /// Longest document length the table must cover for a packer config.
    pub fn max_len(seq_len: usize, pad_to_multiple_of: usize) -> usize {
        seq_len + pad_to_multiple_of
    }

    pub fn cost(&self, seq_lens: &[usize]) -> i128 {
        seq_lens.iter().map(|&n| self.per_len[n]).sum()
    }
}

pub struct BatchPacker {
    seq_len: usize,
    num_train_workers: usize,
    pad_to_multiple_of: usize,
    bin_cost: FlopTable,
}

impl BatchPacker {
    /// `pad_to_seq_len` pads every row to exactly `seq_len`.
    pub fn new(seq_len: usize, num_train_workers: usize, pad_to_multiple_of: usize, pad_to_seq_len: bool, bin_cost: FlopTable) -> Self {
        let pad_to_multiple_of = if pad_to_seq_len { seq_len } else { pad_to_multiple_of };
        Self { seq_len, num_train_workers, pad_to_multiple_of, bin_cost }
    }

    pub fn pack(&self, samples: Vec<TrainingSample>) -> Result<Vec<Vec<MicroBatch>>> {
        prepare_batch(samples, self.seq_len, self.num_train_workers, &self.bin_cost, self.pad_to_multiple_of)
    }
}

/// Truncation never splits an image: a cut inside one moves back to its first placeholder.
fn truncate_mm_refs(refs: &MMRefs, seq_len: usize) -> (usize, Option<MMRefs>) {
    let mut cut = seq_len;
    let mut kept = Vec::new();
    for image in &refs.images {
        let end = image.offset + image.length;
        if image.offset < seq_len && seq_len < end {
            cut = image.offset;
            break;
        }
        if end <= seq_len {
            kept.push(image.clone());
        }
    }
    (cut, (!kept.is_empty()).then_some(MMRefs { images: kept }))
}

fn check_multimodal(sample: &TrainingSample) -> Result<()> {
    let n = sample.token_ids.len();
    if let Some(mm) = &sample.mm_token_type_ids {
        ensure!(mm.len() == n, "mm_token_type_ids length must match token_ids length ({} != {n})", mm.len());
    }
    if let Some(refs) = &sample.mm_refs {
        ensure!(sample.mm_token_type_ids.is_some(), "multimodal samples require mm_token_type_ids");
        let mut previous_end = 0;
        for image in &refs.images {
            ensure!(
                image.offset >= previous_end && image.offset + image.length <= n,
                "multimodal image refs must be ordered, non-overlapping, and within token_ids"
            );
            previous_end = image.offset + image.length;
        }
    }
    Ok(())
}

/// One sample as a single-sequence micro batch, truncated to `seq_len`.
pub fn prepare_sample(sample: TrainingSample, seq_len: usize) -> Result<MicroBatch> {
    check_multimodal(&sample)?;
    ensure!(sample.env_name != "all", "env_name='all' is reserved for aggregate metric keys");
    let advantages = match sample.advantages {
        Some(a) => a,
        None => {
            let has_rl_members = match &sample.rl_weights {
                None => sample.mask.iter().any(|&m| m),
                Some(w) => sample.mask.iter().zip(w).any(|(&m, &w)| m && w != 0.0),
            };
            if has_rl_members {
                bail!(
                    "sample from env '{}' has rl member tokens but no advantages — the producer must stamp the advantage stream (the orchestrator broadcasts the rollout scalar)",
                    sample.env_name
                );
            }
            vec![0.0; sample.token_ids.len()]
        }
    };
    let mut mb = MicroBatch {
        position_ids: (0..sample.token_ids.len() as i32).collect(),
        input_ids: sample.token_ids,
        loss_mask: sample.mask,
        advantages,
        inference_logprobs: sample.logprobs,
        temperatures: sample.temperatures,
        env_names: vec![sample.env_name],
        ref_logprobs: sample.ref_logprobs,
        mm_refs: sample.mm_refs,
        mm_token_type_ids: sample.mm_token_type_ids,
        rl_weights: sample.rl_weights,
        ce_weights: sample.ce_weights,
        ref_kl_weights: sample.ref_kl_weights,
        trace_ids: Some(vec![sample.trace_id.unwrap_or_default()]),
        branch_indices: Some(vec![sample.branch_index.unwrap_or(-1)]),
        staleness: sample.staleness,
        payloads: sample.payload.map(|p| vec![Some(p)]),
        ..Default::default()
    };
    if mb.input_ids.len() > seq_len {
        let (cut, mm_refs) = match &mb.mm_refs {
            Some(refs) => truncate_mm_refs(refs, seq_len),
            None => (seq_len, None),
        };
        mb.mm_refs = mm_refs;
        mb.input_ids.truncate(cut);
        mb.loss_mask.truncate(cut);
        mb.advantages.truncate(cut);
        mb.inference_logprobs.truncate(cut);
        mb.position_ids.truncate(cut);
        mb.temperatures.truncate(cut);
        for stream in [&mut mb.ref_logprobs, &mut mb.rl_weights, &mut mb.ce_weights, &mut mb.ref_kl_weights].into_iter().flatten() {
            stream.truncate(cut);
        }
        if let Some(mm) = &mut mb.mm_token_type_ids {
            mm.truncate(cut);
        }
        if let Some(s) = &mut mb.staleness {
            s.truncate(cut);
        }
        if let Some(p) = mb.payloads.as_mut().and_then(|ps| ps[0].as_mut()) {
            p.num_tokens = cut;
        }
    }
    let n = mb.input_ids.len();
    let lens = [
        mb.loss_mask.len(),
        mb.advantages.len(),
        mb.inference_logprobs.len(),
        mb.temperatures.len(),
        mb.ref_logprobs.as_ref().map_or(n, Vec::len),
        mb.rl_weights.as_ref().map_or(n, Vec::len),
        mb.ce_weights.as_ref().map_or(n, Vec::len),
        mb.ref_kl_weights.as_ref().map_or(n, Vec::len),
        mb.staleness.as_ref().map_or(n, Vec::len),
    ];
    ensure!(
        lens.iter().all(|&len| len == n),
        "per-token streams must match input_ids ({n}): mask, advantages, logprobs, temperatures, ref_logprobs, rl, ce, ref_kl, staleness = {lens:?}"
    );
    mb.sequence_lengths = vec![n];
    mb.seq_lens = vec![n];
    Ok(mb)
}

struct Bin {
    samples: Vec<MicroBatch>,
    length: usize,
}

impl Bin {
    fn from_sample(sample: MicroBatch) -> Self {
        let length = sample.input_ids.len();
        Self { samples: vec![sample], length }
    }

    fn can_add(&self, sample: &MicroBatch, max_seq_len: usize) -> bool {
        self.length + sample.input_ids.len() <= max_seq_len && self.samples[0].has_routing() == sample.has_routing()
    }

    fn add(&mut self, sample: MicroBatch) {
        self.length += sample.input_ids.len();
        self.samples.push(sample);
    }

    fn workload(&self, bin_cost: &FlopTable) -> i128 {
        let lens: Vec<usize> = self.samples.iter().map(|s| s.input_ids.len()).collect();
        bin_cost.cost(&lens)
    }

    /// Longest-processing-time split: heaviest sample to the lighter side.
    fn split_by_workload(self, bin_cost: &FlopTable) -> (Bin, Bin) {
        let mut ranked: Vec<(i128, MicroBatch)> =
            self.samples.into_iter().map(|s| (bin_cost.cost(&[s.input_ids.len()]), s)).collect();
        ranked.sort_by_key(|(workload, _)| std::cmp::Reverse(*workload));
        let (mut left, mut right) = (Vec::new(), Vec::new());
        let (mut left_workload, mut right_workload) = (0i128, 0i128);
        for (workload, sample) in ranked {
            if left_workload <= right_workload {
                left.push(sample);
                left_workload += workload;
            } else {
                right.push(sample);
                right_workload += workload;
            }
        }
        let len = |v: &Vec<MicroBatch>| v.iter().map(|s| s.input_ids.len()).sum();
        (Bin { length: len(&left), samples: left }, Bin { length: len(&right), samples: right })
    }
}

/// A weight stream's backfill when a packed sample lacks it: rl 1.0 on the loss mask, ce/ref_kl absent (0.0).
const RL_FILL: f32 = 1.0;
const CE_FILL: f32 = 0.0;
const REF_KL_FILL: f32 = 0.0;

fn concat_optional<T: Clone>(
    samples: &[MicroBatch],
    get: impl Fn(&MicroBatch) -> &Option<Vec<T>>,
    fill: T,
) -> Option<Vec<T>> {
    if !samples.iter().any(|s| get(s).is_some()) {
        return None;
    }
    let mut out = Vec::new();
    for s in samples {
        match get(s) {
            Some(v) => out.extend_from_slice(v),
            None => out.extend(std::iter::repeat_n(fill.clone(), s.input_ids.len())),
        }
    }
    Some(out)
}

fn materialize_bin(bin: Bin) -> Result<MicroBatch> {
    let samples = bin.samples;
    let ref_logprobs = concat_optional(&samples, |s| &s.ref_logprobs, 0.0f32);
    let mm_token_type_ids = concat_optional(&samples, |s| &s.mm_token_type_ids, 0i32);
    let rl_weights = concat_optional(&samples, |s| &s.rl_weights, RL_FILL);
    let ce_weights = concat_optional(&samples, |s| &s.ce_weights, CE_FILL);
    let ref_kl_weights = concat_optional(&samples, |s| &s.ref_kl_weights, REF_KL_FILL);
    let staleness = concat_optional(&samples, |s| &s.staleness, -1i16);

    let mut out = MicroBatch {
        ref_logprobs,
        mm_token_type_ids,
        rl_weights,
        ce_weights,
        ref_kl_weights,
        staleness,
        trace_ids: Some(Vec::new()),
        branch_indices: Some(Vec::new()),
        ..Default::default()
    };
    let mut payloads = Vec::new();
    let mut offset = 0;
    for s in samples {
        let len = s.input_ids.len();
        let num_sequences = s.sequence_lengths.len();
        out.input_ids.extend_from_slice(&s.input_ids);
        out.loss_mask.extend_from_slice(&s.loss_mask);
        out.advantages.extend_from_slice(&s.advantages);
        out.inference_logprobs.extend_from_slice(&s.inference_logprobs);
        out.position_ids.extend_from_slice(&s.position_ids);
        out.temperatures.extend_from_slice(&s.temperatures);
        out.env_names.extend(s.env_names);
        if let Some(refs) = s.mm_refs {
            out.mm_refs.get_or_insert_with(|| MMRefs { images: Vec::new() }).images.extend(refs.images.into_iter().map(
                |image| MMImageRef { url: image.url, offset: offset + image.offset, length: image.length },
            ));
        }
        out.seq_lens.extend(s.seq_lens);
        out.trace_ids.as_mut().unwrap().extend(s.trace_ids.unwrap_or_else(|| vec![String::new(); num_sequences]));
        out.branch_indices.as_mut().unwrap().extend(s.branch_indices.unwrap_or_else(|| vec![-1; num_sequences]));
        payloads.extend(s.payloads.unwrap_or_else(|| vec![None; num_sequences]));
        out.sequence_lengths.push(len);
        offset += len;
    }
    ensure!(out.sequence_lengths.iter().sum::<usize>() == offset, "sequence_lengths do not sum to {offset}");
    ensure!(out.seq_lens.iter().sum::<usize>() == offset, "seq_lens do not sum to {offset}");
    out.payloads = payloads.iter().any(Option::is_some).then_some(payloads);
    Ok(out)
}

fn expand_bins_by_splitting(bins: &mut Vec<Bin>, target_count: usize, bin_cost: &FlopTable) {
    while bins.len() < target_count {
        let Some((_, idx)) = bins
            .iter()
            .enumerate()
            .filter(|(_, b)| b.samples.len() > 1)
            .map(|(idx, b)| (b.workload(bin_cost), idx))
            .max()
        else {
            break;
        };
        let placeholder = Bin { samples: Vec::new(), length: 0 };
        let bin = std::mem::replace(&mut bins[idx], placeholder);
        let (left, right) = bin.split_by_workload(bin_cost);
        bins[idx] = left;
        bins.push(right);
    }
}

/// First-fit decreasing packing (never truncates), then split bins until the count is a
/// multiple of the worker count.
fn packed_samples_into_micro_bs(
    mut samples: Vec<MicroBatch>,
    max_seq_len: usize,
    num_train_workers: usize,
    bin_cost: &FlopTable,
) -> Result<Vec<MicroBatch>> {
    samples.sort_by_key(|s| std::cmp::Reverse(s.input_ids.len()));
    let mut bins: Vec<Bin> = Vec::new();
    for sample in samples {
        match bins.iter_mut().find(|b| b.can_add(&sample, max_seq_len)) {
            Some(bin) => bin.add(sample),
            None => bins.push(Bin::from_sample(sample)),
        }
    }
    if num_train_workers > 1 {
        let target = (bins.len().div_ceil(num_train_workers) * num_train_workers).max(num_train_workers);
        expand_bins_by_splitting(&mut bins, target, bin_cost);
    }
    bins.into_iter().map(materialize_bin).collect()
}

/// Pad a micro batch with loss-masked tokens up to the next multiple of `pad_to_multiple_of`.
pub fn pad_micro_batch(mut mb: MicroBatch, pad_to_multiple_of: usize) -> Result<MicroBatch> {
    let n = mb.input_ids.len();
    let padding = (pad_to_multiple_of - n % pad_to_multiple_of) % pad_to_multiple_of;
    ensure!(
        mb.env_names.len() == mb.sequence_lengths.len(),
        "MicroBatch.env_names must name every sequence: env_names={}, sequences={}",
        mb.env_names.len(),
        mb.sequence_lengths.len()
    );
    if !(pad_to_multiple_of > 1 && padding > 0) {
        return Ok(mb);
    }
    // Text rows pad with their own tokens (identical pad ids would overload one MoE expert);
    // multimodal rows keep a constant id.
    if mb.mm_refs.is_none() {
        let pad: Vec<i32> = (0..padding).map(|i| mb.input_ids[i % n]).collect();
        mb.input_ids.extend(pad);
    } else {
        mb.input_ids.extend(std::iter::repeat_n(1, padding));
    }
    mb.advantages.extend(std::iter::repeat_n(0.0, padding));
    mb.loss_mask.extend(std::iter::repeat_n(false, padding));
    mb.position_ids.extend(0..padding as i32);
    *mb.sequence_lengths.last_mut().unwrap() += padding;
    *mb.seq_lens.last_mut().unwrap() += padding;
    mb.padding += padding;
    mb.inference_logprobs.extend(std::iter::repeat_n(0.0, padding));
    mb.temperatures.extend(std::iter::repeat_n(1.0, padding));
    for stream in [&mut mb.ref_logprobs, &mut mb.rl_weights, &mut mb.ce_weights, &mut mb.ref_kl_weights].into_iter().flatten() {
        stream.extend(std::iter::repeat_n(0.0, padding));
    }
    if let Some(mm) = &mut mb.mm_token_type_ids {
        mm.extend(std::iter::repeat_n(0, padding));
    }
    if let Some(s) = &mut mb.staleness {
        s.extend(std::iter::repeat_n(-1, padding));
    }
    Ok(mb)
}

/// A zero-loss copy of a micro batch, preserving its modality and routing.
fn make_dummy_batch(source: &MicroBatch) -> MicroBatch {
    let mut dummy = source.clone();
    let n = dummy.input_ids.len();
    dummy.advantages = vec![0.0; n];
    dummy.loss_mask = vec![false; n];
    dummy.rl_weights = None;
    dummy.ce_weights = None;
    dummy.ref_kl_weights = None;
    dummy.payloads = if source.has_routing() {
        dummy.payloads.map(|ps| {
            ps.into_iter()
                .map(|p| p.map(|mut p| {
                    p.fields = vec!["routed_experts".into()];
                    p
                }))
                .collect()
        })
    } else {
        None
    };
    dummy.trace_ids = None;
    dummy.branch_indices = None;
    dummy
}

fn pad_group_for_distribution(group: &mut Vec<MicroBatch>, num_train_workers: usize) {
    let num_padding = (num_train_workers - group.len() % num_train_workers) % num_train_workers;
    if num_padding > 0 && !group.is_empty() {
        let dummy = make_dummy_batch(&group[0]);
        group.extend(std::iter::repeat_n(dummy, num_padding));
    }
}

fn distribute_group(group: Vec<MicroBatch>, num_train_workers: usize, bin_cost: &FlopTable) -> Vec<Vec<MicroBatch>> {
    assert!(group.len().is_multiple_of(num_train_workers), "Number of micro batches is not divisible by number of data ranks");
    if group.is_empty() {
        return (0..num_train_workers).map(|_| Vec::new()).collect();
    }
    let weights: Vec<i128> = group.iter().map(|mb| bin_cost.cost(&mb.sequence_lengths)).collect();
    let partitions = balanced_partition(&weights, num_train_workers);
    let mut slots: Vec<Option<MicroBatch>> = group.into_iter().map(Some).collect();
    partitions
        .into_iter()
        .map(|p| p.into_iter().map(|i| slots[i].take().expect("partition index used twice")).collect())
        .collect()
}

/// Per data rank, its micro batches. At every step index all ranks see the same modality.
pub fn prepare_batch(
    rollouts: Vec<TrainingSample>,
    seq_len: usize,
    num_train_workers: usize,
    bin_cost: &FlopTable,
    pad_to_multiple_of: usize,
) -> Result<Vec<Vec<MicroBatch>>> {
    let samples = rollouts.into_iter().map(|s| prepare_sample(s, seq_len)).collect::<Result<Vec<_>>>()?;
    let micro_batches = packed_samples_into_micro_bs(samples, seq_len, num_train_workers, bin_cost)?;
    let micro_batches =
        micro_batches.into_iter().map(|mb| pad_micro_batch(mb, pad_to_multiple_of)).collect::<Result<Vec<_>>>()?;
    let (mut mm, mut text): (Vec<_>, Vec<_>) = micro_batches.into_iter().partition(MicroBatch::is_multimodal);
    pad_group_for_distribution(&mut mm, num_train_workers);
    pad_group_for_distribution(&mut text, num_train_workers);
    let mut per_worker: Vec<Vec<MicroBatch>> = (0..num_train_workers).map(|_| Vec::new()).collect();
    for group in [mm, text] {
        for (worker, batches) in distribute_group(group, num_train_workers, bin_cost).into_iter().enumerate() {
            per_worker[worker].extend(batches);
        }
    }
    Ok(per_worker)
}
