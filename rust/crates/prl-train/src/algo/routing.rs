//! Training annotations and transport loss routing (`algo/routing.py`).

use anyhow::{bail, Result};
use prl_config::ActionLossType;
use prl_wire::Trace;

use crate::sample::TrainingSample;
use crate::trajectories::{count_true, trainable_node_ids};

/// Credit for a trace's sampled tokens: one scalar broadcast, or one value per sampled token.
pub enum Advantages<'a> {
    Scalar(f32),
    PerToken(&'a [f32]),
}

/// Assign credit in compact sampled-token order across the trainable graph paths.
pub fn assign_advantages(trace: &mut Trace, values: Advantages<'_>) -> Result<()> {
    let nodes = trainable_node_ids(trace);
    for &n in &nodes {
        let node = &trace.nodes[n];
        if node.mask.len() != node.token_ids.len() {
            bail!(
                "node masks must align with token ids in trace {:?}: got {}, expected {}",
                trace.id,
                node.mask.len(),
                node.token_ids.len()
            );
        }
    }
    let counts: Vec<usize> = nodes.iter().map(|&n| count_true(&trace.nodes[n].mask)).collect();
    let total: usize = counts.iter().sum();
    let advantages: Vec<f32> = match values {
        Advantages::Scalar(v) => vec![v; total],
        Advantages::PerToken(v) => {
            if v.len() != total {
                bail!(
                    "advantages must align with trace {:?}'s sampled tokens: got {}, expected {total}",
                    trace.id,
                    v.len()
                );
            }
            v.to_vec()
        }
    };
    let mut offset = 0;
    for (&n, count) in nodes.iter().zip(counts) {
        trace.nodes[n].advantages = Some(advantages[offset..offset + count].to_vec());
        offset += count;
    }
    Ok(())
}

/// Project branch-aligned reference logprobs onto the branch's sampled nodes (first writer wins).
pub fn assign_reference_logprobs(trace: &mut Trace, node_ids: &[usize], values: &[f32]) -> Result<()> {
    let num_tokens: usize = node_ids.iter().map(|&n| trace.nodes[n].token_ids.len()).sum();
    if values.len() != num_tokens {
        bail!(
            "reference logprobs must align with branch tokens: got {}, expected {num_tokens}",
            values.len()
        );
    }
    let mut offset = 0;
    for &n in node_ids {
        let node = &mut trace.nodes[n];
        let end = offset + node.token_ids.len();
        if node.mask.iter().any(|&m| m) && node.reference_logprobs.is_none() {
            node.reference_logprobs =
                Some(values[offset..end].iter().zip(&node.mask).filter(|(_, &m)| m).map(|(&v, _)| v).collect());
        }
        offset = end;
    }
    Ok(())
}

/// Mean nonzero token advantage, 0.0 for an assigned-zero trace, `None` when unassigned.
pub fn scalar_advantage(trace: &Trace) -> Option<f64> {
    let mut any = false;
    let (mut sum, mut count) = (0f64, 0usize);
    for node in &trace.nodes {
        if let Some(advantages) = &node.advantages {
            for &a in advantages {
                any = true;
                if a != 0.0 {
                    sum += f64::from(a);
                    count += 1;
                }
            }
        }
    }
    if !any {
        return None;
    }
    Some(if count > 0 { sum / count as f64 } else { 0.0 })
}

/// Whether any sampled token carries nonzero RL credit.
pub fn is_trainable(trace: &Trace) -> bool {
    trace.nodes.iter().filter_map(|n| n.advantages.as_ref()).any(|a| a.iter().any(|&x| x != 0.0))
}

/// Route action tokens into the algorithm's declared loss component.
pub fn stamp_loss_routing(sample: &mut TrainingSample, action_loss_type: ActionLossType) {
    if action_loss_type == ActionLossType::Rl {
        return;
    }
    let seq_len = sample.token_ids.len();
    sample.rl_weights = Some(vec![0.0; seq_len]);
    let mut action_weights = match (&sample.ce_weights, action_loss_type) {
        (Some(ce), ActionLossType::Ce) => ce.clone(),
        _ => vec![0.0; seq_len],
    };
    for (w, &m) in action_weights.iter_mut().zip(&sample.mask) {
        if m {
            *w = 1.0;
        }
    }
    if action_loss_type == ActionLossType::Ce {
        sample.ce_weights = Some(action_weights);
    } else {
        sample.ref_kl_weights = Some(action_weights);
    }
}
