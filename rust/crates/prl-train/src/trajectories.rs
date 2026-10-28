//! Native traces to training samples (`orchestrator/trajectories.py`).

use std::collections::HashSet;

use anyhow::{bail, Result};
use prl_wire::{Branch, Trace, Value};

use prl_wire::payload::PAYLOAD_INFO_KEY;

use crate::sample::{MMImageRef, MMRefs, PayloadSlice, TrainingSample};

fn image_urls(branch: &Branch<'_>) -> Vec<String> {
    let mut urls = Vec::new();
    for node in &branch.nodes {
        let Some(parts) = node.message.get("content").and_then(Value::as_array) else { continue };
        for part in parts {
            if part.get("type").and_then(Value::as_str) == Some("image_url") {
                if let Some(url) = part.get("image_url").and_then(|i| i.get("url")).and_then(Value::as_str) {
                    urls.push(url.to_owned());
                }
            }
        }
    }
    urls
}

fn image_runs(token_types: &[i32]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = None;
    for (index, &token_type) in token_types.iter().chain(std::iter::once(&0)).enumerate() {
        if token_type == 1 && start.is_none() {
            start = Some(index);
        } else if token_type != 1 {
            if let Some(s) = start.take() {
                runs.push((s, index - s));
            }
        }
    }
    runs
}

fn build_mm_refs(urls: Vec<String>, token_types: &[i32]) -> Result<Option<MMRefs>> {
    let runs = image_runs(token_types);
    if urls.len() != runs.len() {
        bail!(
            "Inline image count does not match expanded placeholder runs: images={}, runs={}",
            urls.len(),
            runs.len()
        );
    }
    if urls.is_empty() {
        return Ok(None);
    }
    Ok(Some(MMRefs {
        images: urls.into_iter().zip(runs).map(|(url, (offset, length))| MMImageRef { url, offset, length }).collect(),
    }))
}

/// Each branch that yields a training sample, with its trainable-token mask. A sampled node shared
/// by several branches trains only in the first branch containing it; branches left with no
/// trainable tokens are skipped.
pub fn iter_trainable_branches(trace: &Trace) -> Vec<(Branch<'_>, Vec<bool>)> {
    let mut trained_nodes: HashSet<usize> = HashSet::new();
    let mut out = Vec::new();
    for branch in trace.branches() {
        if !branch.trainable {
            continue;
        }
        let mut mask = Vec::with_capacity(branch.num_tokens());
        for (&node_id, node) in branch.node_ids.iter().zip(&branch.nodes) {
            let trainable_node = node.sampled && node.mask.iter().any(|&m| m);
            if trainable_node && trained_nodes.contains(&node_id) {
                mask.extend(std::iter::repeat_n(false, node.mask.len()));
            } else {
                if trainable_node {
                    trained_nodes.insert(node_id);
                }
                mask.extend_from_slice(&node.mask);
            }
        }
        if mask.iter().any(|&m| m) {
            out.push((branch, mask));
        }
    }
    out
}

/// Flatten one graph-native loss stream, training each shared node once.
fn loss_weights(
    branch: &Branch<'_>,
    name: &str,
    trained_nodes: &mut HashSet<usize>,
) -> Result<Option<Vec<f32>>> {
    let mut weights = Vec::with_capacity(branch.num_tokens());
    for (&node_id, node) in branch.node_ids.iter().zip(&branch.nodes) {
        let node_weights = node.loss_weights.as_ref().and_then(|w| w.get(name));
        match node_weights {
            Some(w) if !trained_nodes.contains(&node_id) => {
                if w.len() != node.token_ids.len() {
                    bail!(
                        "loss weight stream {name:?} must align with node token_ids: got {}, expected {}",
                        w.len(),
                        node.token_ids.len()
                    );
                }
                weights.extend_from_slice(w);
                if w.iter().any(|&x| x != 0.0) {
                    trained_nodes.insert(node_id);
                }
            }
            _ => weights.extend(std::iter::repeat_n(0.0, node.token_ids.len())),
        }
    }
    Ok(weights.iter().any(|&x| x != 0.0).then_some(weights))
}

/// The branch's arrays an env server kept (`trace.info[PAYLOAD_INFO_KEY]`), if any.
fn payload_slice(trace: &Trace, branch_index: usize, num_tokens: usize) -> Option<PayloadSlice> {
    let info = trace.info.get(PAYLOAD_INFO_KEY).filter(|v| v.truthy())?;
    let reference = info.get("branches")?.get(&branch_index.to_string())?;
    let get = |key: &str| reference.get(key).filter(|v| !v.is_nil());
    Some(PayloadSlice {
        url: info.get("url")?.as_str()?.to_owned(),
        id: get("id")?.as_str()?.to_owned(),
        num_tokens,
        fields: get("fields")?.as_array()?.iter().filter_map(|f| f.as_str().map(str::to_owned)).collect(),
        routed_shape: get("routed_shape").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_i64).collect()),
        routed_dtype: get("routed_dtype").and_then(Value::as_str).map(str::to_owned),
    })
}

/// One `TrainingSample` per trainable branch. `temperatures` is left empty for the sink to fill.
pub fn trace_to_samples(trace: &Trace, env_name: &str) -> Result<Vec<TrainingSample>> {
    let mut samples = Vec::new();
    let mut trained: [HashSet<usize>; 3] = Default::default();
    for (branch, mask) in iter_trainable_branches(trace) {
        let token_ids = branch.token_ids();
        let urls = image_urls(&branch);
        let (mm_token_type_ids, mm_refs) = if urls.is_empty() {
            (None, None)
        } else {
            let Some(types) = branch.mm_token_type_ids() else {
                bail!("Inline images have no expanded multimodal prompt tokens");
            };
            let refs = build_mm_refs(urls, &types)?;
            (Some(types), refs)
        };
        let [rl, ce, ref_kl] = &mut trained;
        let payload = payload_slice(trace, branch.index, token_ids.len());
        samples.push(TrainingSample {
            mask,
            logprobs: branch.logprobs(),
            temperatures: Vec::new(),
            env_name: env_name.to_owned(),
            ref_logprobs: branch.reference_logprobs(),
            mm_refs,
            mm_token_type_ids,
            rl_weights: loss_weights(&branch, "rl", rl)?,
            ce_weights: loss_weights(&branch, "ce", ce)?,
            ref_kl_weights: loss_weights(&branch, "ref_kl", ref_kl)?,
            advantages: branch.advantages(),
            trace_id: Some(trace.id.clone()),
            branch_index: Some(branch.index as i64),
            staleness: None,
            payload,
            token_ids,
        });
    }
    if samples.is_empty() {
        tracing::warn!(
            "No trainable samples (error={}, stop={}, num_turns={}).",
            if trace.has_error() { "True" } else { "False" },
            trace.stop_condition.as_deref().unwrap_or("None"),
            trace.num_turns()
        );
    }
    Ok(samples)
}

/// Nodes with sampled (mask-True) tokens on the trace's trainable branches, in graph order.
pub fn trainable_node_ids(trace: &Trace) -> Vec<usize> {
    let mut on_trainable: HashSet<usize> = HashSet::new();
    for branch in trace.branches() {
        if branch.trainable {
            on_trainable.extend(branch.node_ids.iter().copied());
        }
    }
    (0..trace.nodes.len()).filter(|i| on_trainable.contains(i) && trace.nodes[*i].mask.iter().any(|&m| m)).collect()
}

pub(crate) fn count_true(mask: &[bool]) -> usize {
    mask.iter().filter(|&&m| m).count()
}
