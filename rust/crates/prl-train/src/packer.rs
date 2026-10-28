//! `BatchPacker` (`orchestrator/batch.py`): packs a step's samples into one micro batch list per
//! trainer data rank. CPU-bound: call `pack` from `spawn_blocking`.

use std::sync::Arc;

use anyhow::{bail, ensure, Result};

use crate::partition::balanced_partition;
use crate::sample::{MMImageRef, MMRefs, TrainingSample};
use crate::transport::MicroBatch;

/// Training cost of packed documents; additive over sequences.
pub trait BinCost: Send + Sync {
    fn cost(&self, seq_lens: &[usize]) -> i128;
}

impl<F: Fn(&[usize]) -> i128 + Send + Sync> BinCost for F {
    fn cost(&self, seq_lens: &[usize]) -> i128 {
        self(seq_lens)
    }
}

/// `FlopCounter` of the trainer model, tabulated per document length (`table[n] = bin_cost([n])`).
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
}

impl BinCost for FlopTable {
    fn cost(&self, seq_lens: &[usize]) -> i128 {
        seq_lens
            .iter()
            .map(|&n| *self.per_len.get(n).unwrap_or_else(|| panic!("flop table covers {} tokens, got {n}", self.per_len.len() - 1)))
            .sum()
    }
}

pub struct BatchPacker {
    pub seq_len: usize,
    pub num_train_workers: usize,
    pub pad_to_multiple_of: usize,
    pub bin_cost: Arc<dyn BinCost>,
}

impl BatchPacker {
    /// `pad_to_seq_len` pads every row to exactly `seq_len`.
    pub fn new(seq_len: usize, num_train_workers: usize, pad_to_multiple_of: usize, pad_to_seq_len: bool, bin_cost: Arc<dyn BinCost>) -> Self {
        let pad_to_multiple_of = if pad_to_seq_len { seq_len } else { pad_to_multiple_of };
        Self { seq_len, num_train_workers, pad_to_multiple_of, bin_cost }
    }

    pub fn pack(&self, samples: Vec<TrainingSample>) -> Result<Vec<Vec<MicroBatch>>> {
        prepare_batch(samples, self.seq_len, self.num_train_workers, self.bin_cost.as_ref(), self.pad_to_multiple_of)
    }
}

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

pub fn multimodal_sample_error(sample: &TrainingSample) -> Option<String> {
    if let Some(mm) = &sample.mm_token_type_ids {
        if mm.len() != sample.token_ids.len() {
            return Some(format!(
                "mm_token_type_ids length must match token_ids length ({} != {})",
                mm.len(),
                sample.token_ids.len()
            ));
        }
    }
    if let Some(refs) = &sample.mm_refs {
        if sample.mm_token_type_ids.is_none() {
            return Some("multimodal samples require mm_token_type_ids".into());
        }
        let mut previous_end = 0;
        for image in &refs.images {
            if image.offset < previous_end || image.offset + image.length > sample.token_ids.len() {
                return Some("multimodal image refs must be ordered, non-overlapping, and within token_ids".into());
            }
            previous_end = image.offset + image.length;
        }
    }
    None
}

/// One sample as a single-sequence micro batch, truncated to `seq_len`.
pub fn prepare_sample(sample: TrainingSample, seq_len: usize) -> Result<MicroBatch> {
    if let Some(error) = multimodal_sample_error(&sample) {
        bail!(error);
    }
    let TrainingSample {
        mut token_ids,
        mask: mut loss_mask,
        logprobs: mut inference_logprobs,
        mut temperatures,
        env_name,
        mut ref_logprobs,
        mut mm_refs,
        mut mm_token_type_ids,
        mut rl_weights,
        mut ce_weights,
        mut ref_kl_weights,
        advantages,
        trace_id,
        branch_index,
        mut staleness,
        mut payload,
    } = sample;
    let mut advantages = match advantages {
        Some(a) => a,
        None => {
            let has_rl_members = match &rl_weights {
                None => loss_mask.iter().any(|&m| m),
                Some(w) => loss_mask.iter().zip(w).any(|(&m, &w)| m && w != 0.0),
            };
            if has_rl_members {
                bail!(
                    "sample from env '{env_name}' has rl member tokens but no advantages — the producer must stamp the advantage stream (the orchestrator broadcasts the rollout scalar)"
                );
            }
            vec![0.0; token_ids.len()]
        }
    };
    ensure!(env_name != "all", "env_name='all' is reserved for aggregate metric keys");
    let mut position_ids: Vec<i32> = (0..token_ids.len() as i32).collect();

    if token_ids.len() > seq_len {
        let mut cut = seq_len;
        if let Some(refs) = &mm_refs {
            let (c, kept) = truncate_mm_refs(refs, seq_len);
            cut = c;
            mm_refs = kept;
        }
        token_ids.truncate(cut);
        loss_mask.truncate(cut);
        inference_logprobs.truncate(cut);
        position_ids.truncate(cut);
        advantages.truncate(cut);
        temperatures.truncate(cut);
        for stream in [&mut ref_logprobs, &mut rl_weights, &mut ce_weights, &mut ref_kl_weights].into_iter().flatten() {
            stream.truncate(cut);
        }
        if let Some(mm) = &mut mm_token_type_ids {
            mm.truncate(cut);
        }
        if let Some(p) = &mut payload {
            p.num_tokens = cut;
        }
        if let Some(s) = &mut staleness {
            s.truncate(cut);
        }
    }

    let n = token_ids.len();
    ensure!(
        n == advantages.len() && n == loss_mask.len() && n == inference_logprobs.len() && n == temperatures.len(),
        "input_ids: {n}, advantages: {}, loss_mask: {}, position_ids: {}, inference_logprobs: {}, temperatures: {}",
        advantages.len(),
        loss_mask.len(),
        position_ids.len(),
        inference_logprobs.len(),
        temperatures.len()
    );
    for (name, len) in [
        ("ref_logprobs", ref_logprobs.as_ref().map(Vec::len)),
        ("rl_weights", rl_weights.as_ref().map(Vec::len)),
        ("ce_weights", ce_weights.as_ref().map(Vec::len)),
        ("ref_kl_weights", ref_kl_weights.as_ref().map(Vec::len)),
        ("staleness", staleness.as_ref().map(Vec::len)),
    ] {
        if let Some(len) = len {
            ensure!(len == n, "{name}: {len}");
        }
    }

    Ok(MicroBatch {
        input_ids: token_ids,
        advantages,
        loss_mask,
        position_ids,
        inference_logprobs,
        sequence_lengths: vec![n],
        ref_logprobs,
        temperatures,
        mm_token_type_ids,
        env_names: vec![env_name],
        mm_refs,
        rl_weights,
        ce_weights,
        ref_kl_weights,
        seq_lens: vec![n],
        trace_ids: Some(vec![trace_id.unwrap_or_default()]),
        branch_indices: Some(vec![branch_index.unwrap_or(-1)]),
        padding: 0,
        staleness,
        payloads: payload.map(|p| vec![Some(p)]),
    })
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

    fn workload(&self, bin_cost: &dyn BinCost) -> i128 {
        let lens: Vec<usize> = self.samples.iter().map(|s| s.input_ids.len()).collect();
        bin_cost.cost(&lens)
    }

    /// Longest-processing-time split: heaviest sample to the lighter side.
    fn split_by_workload(self, bin_cost: &dyn BinCost) -> (Bin, Bin) {
        let mut ranked: Vec<(i128, MicroBatch)> =
            self.samples.into_iter().map(|s| (bin_cost.cost(&[s.input_ids.len()]), s)).collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0));
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

fn expand_bins_by_splitting(bins: &mut Vec<Bin>, target_count: usize, bin_cost: &dyn BinCost) {
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
pub fn packed_samples_into_micro_bs(
    mut samples: Vec<MicroBatch>,
    max_seq_len: usize,
    num_train_workers: usize,
    bin_cost: &dyn BinCost,
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

fn assert_token_arrays_aligned(mb: &MicroBatch) -> Result<()> {
    let n = mb.input_ids.len();
    let lens = [
        ("loss_mask", Some(mb.loss_mask.len())),
        ("advantages", Some(mb.advantages.len())),
        ("inference_logprobs", Some(mb.inference_logprobs.len())),
        ("position_ids", Some(mb.position_ids.len())),
        ("temperatures", Some(mb.temperatures.len())),
        ("ref_logprobs", mb.ref_logprobs.as_ref().map(Vec::len)),
        ("rl_weights", mb.rl_weights.as_ref().map(Vec::len)),
        ("ce_weights", mb.ce_weights.as_ref().map(Vec::len)),
        ("ref_kl_weights", mb.ref_kl_weights.as_ref().map(Vec::len)),
        ("mm_token_type_ids", mb.mm_token_type_ids.as_ref().map(Vec::len)),
        ("staleness", mb.staleness.as_ref().map(Vec::len)),
    ];
    for (name, len) in lens {
        if let Some(len) = len {
            ensure!(len == n, "{name} misaligned after packing: {len} != {n} tokens");
        }
    }
    ensure!(mb.sequence_lengths.iter().sum::<usize>() == n, "sequence_lengths sum != {n} tokens");
    let num_sequences = mb.sequence_lengths.len();
    for (name, len) in [
        ("env_names", Some(mb.env_names.len())),
        ("trace_ids", mb.trace_ids.as_ref().map(Vec::len)),
        ("branch_indices", mb.branch_indices.as_ref().map(Vec::len)),
        ("payloads", mb.payloads.as_ref().map(Vec::len)),
    ] {
        if let Some(len) = len {
            ensure!(len == num_sequences, "{name} misaligned after packing: {len} != {num_sequences} sequences");
        }
    }
    ensure!(mb.seq_lens.iter().sum::<usize>() == n, "seq_lens sum != {n} tokens");
    Ok(())
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

fn distribute_group(group: Vec<MicroBatch>, num_train_workers: usize, bin_cost: &dyn BinCost) -> Vec<Vec<MicroBatch>> {
    assert!(group.len() % num_train_workers == 0, "Number of micro batches is not divisible by number of data ranks");
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
    bin_cost: &dyn BinCost,
    pad_to_multiple_of: usize,
) -> Result<Vec<Vec<MicroBatch>>> {
    let samples = rollouts.into_iter().map(|s| prepare_sample(s, seq_len)).collect::<Result<Vec<_>>>()?;
    let micro_batches = packed_samples_into_micro_bs(samples, seq_len, num_train_workers, bin_cost)?;
    let micro_batches =
        micro_batches.into_iter().map(|mb| pad_micro_batch(mb, pad_to_multiple_of)).collect::<Result<Vec<_>>>()?;
    let (mut mm, mut text): (Vec<_>, Vec<_>) = micro_batches.into_iter().partition(MicroBatch::is_multimodal);
    pad_group_for_distribution(&mut mm, num_train_workers);
    pad_group_for_distribution(&mut text, num_train_workers);
    for mb in mm.iter().chain(&text) {
        assert_token_arrays_aligned(mb)?;
    }
    let mut per_worker: Vec<Vec<MicroBatch>> = (0..num_train_workers).map(|_| Vec::new()).collect();
    for group in [mm, text] {
        for (worker, batches) in distribute_group(group, num_train_workers, bin_cost).into_iter().enumerate() {
            per_worker[worker].extend(batches);
        }
    }
    Ok(per_worker)
}
