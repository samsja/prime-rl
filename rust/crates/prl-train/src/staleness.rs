//! Token and rollout staleness (`orchestrator/utils.py`, `train_sink.py`).

use prl_wire::{Branch, Episode, MessageNode};

use crate::episodes::train_policy;
use crate::sample::TrainingSample;

/// Oldest policy version whose tokens may still train in batch `step`.
pub fn min_fresh_version(step: i64, max_staleness: i64) -> i64 {
    (step - 1) - max_staleness
}

/// `(start, end, version)` runs over a node's sampled tokens, indexed into `node.token_ids`.
/// Tokens without a numeric engine label get `fallback`.
pub fn node_policy_versions(node: &MessageNode, fallback: i64) -> Vec<(usize, usize, i64)> {
    let sampled = node.mask.iter().filter(|&&m| m).count();
    let offset = node.mask.len() - sampled;
    match node.weight_versions.as_deref() {
        None | Some([]) => {
            if sampled > 0 {
                vec![(offset, offset + sampled, fallback)]
            } else {
                Vec::new()
            }
        }
        Some(spans) => spans
            .iter()
            .map(|span| {
                let numeric = !span.version.is_empty() && span.version.chars().all(|c| c.is_ascii_digit());
                let version = if numeric { span.version.parse().unwrap_or(fallback) } else { fallback };
                (offset + span.start as usize, offset + span.end as usize, version)
            })
            .collect(),
    }
}

/// Staleness of one train rollout in batch `training_step`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Staleness {
    pub rollout: i64,
    pub in_flight: i64,
    pub in_queue: i64,
    pub token_sum: i64,
    pub token_count: i64,
}

/// `None` for frozen-sourced episodes (no policy span), which never go stale.
pub fn rollout_staleness<'a>(episode: &Episode, traces: impl IntoIterator<Item = &'a prl_wire::Trace>, training_step: i64) -> anyhow::Result<Option<Staleness>> {
    let Some(policy) = train_policy(episode)? else {
        return Ok(None);
    };
    let trained_version = training_step - 1;
    let (mut rollout, mut token_sum, mut token_count) = (0i64, 0i64, 0i64);
    for trace in traces {
        for node in &trace.nodes {
            for (start, end, version) in node_policy_versions(node, policy.start) {
                let staleness = (trained_version - version).max(0);
                rollout = rollout.max(staleness);
                token_sum += staleness * (end - start) as i64;
                token_count += (end - start) as i64;
            }
        }
    }
    let in_queue = rollout.min((trained_version - policy.end).max(0));
    Ok(Some(Staleness { rollout, in_flight: rollout - in_queue, in_queue, token_sum, token_count }))
}

/// `[start, end)` runs of the branch's sampled tokens whose policy version is older than `min_version`.
pub fn stale_token_runs(branch: &Branch<'_>, fallback: i64, min_version: i64) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut offset = 0;
    for node in &branch.nodes {
        for (start, end, version) in node_policy_versions(node, fallback) {
            if version < min_version {
                runs.push((offset + start, offset + end));
            }
        }
        offset += node.token_ids.len();
    }
    runs
}

/// Per-token staleness of the branch when it trains on `trained_version`; -1 on unsampled tokens.
pub fn token_staleness(branch: &Branch<'_>, fallback: i64, trained_version: i64) -> Vec<i16> {
    let total = branch.nodes.iter().map(|n| n.token_ids.len()).sum();
    let mut staleness = vec![-1i16; total];
    let mut offset = 0;
    for node in &branch.nodes {
        if node.sampled {
            for (start, end, version) in node_policy_versions(node, fallback) {
                let value = (trained_version - version).max(0) as i16;
                staleness[offset + start..offset + end].fill(value);
            }
        }
        offset += node.token_ids.len();
    }
    staleness
}

/// Remove the sample's stale tokens from every loss component; returns how many trainable tokens
/// this newly masked.
pub fn mask_stale_tokens(sample: &mut TrainingSample, branch: &Branch<'_>, fallback: i64, min_version: i64) -> usize {
    let mut masked = 0;
    for (start, end) in stale_token_runs(branch, fallback, min_version) {
        masked += sample.mask[start..end].iter().filter(|&&m| m).count();
        sample.mask[start..end].fill(false);
        for weights in [&mut sample.rl_weights, &mut sample.ce_weights, &mut sample.ref_kl_weights].into_iter().flatten() {
            weights[start..end].fill(0.0);
        }
    }
    masked
}
