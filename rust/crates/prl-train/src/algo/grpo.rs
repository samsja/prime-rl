//! Scalar credit-assignment algorithms: grpo, echo, max_rl, rae, hierarchical_grpo, sft, debug.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{bail, Result};
use async_trait::async_trait;
use indexmap::IndexMap;
use prl_config::{ActionLossType, EchoAlgoConfig, LengthPenaltyConfig, LossAggregation};
use prl_types::Bridge;
use prl_wire::{min_new_input_tokens, Episode, Trace};

use super::routing::{assign_advantages, Advantages};
use super::{iter_trainable_traces, Algorithm};
use crate::trajectories::{count_true, iter_trainable_branches, trainable_node_ids};

/// Weighted output and new input tokens for branches with no shared prefixes, discounting
/// semantic subagent calls (`subagent_call` edges and what continues from them).
pub fn length_penalty_tokens(trace: &Trace, subagent_token_weight: f64) -> (f64, f64) {
    let mut subagents: HashSet<usize> = HashSet::new();
    let mut successors: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node_id, node) in trace.nodes.iter().enumerate() {
        for link in &node.semantic_parents {
            if link.type_ == "subagent_call" {
                subagents.insert(node_id);
            }
            if matches!(link.type_.as_str(), "subagent_call" | "continuation" | "compaction_attempt" | "compaction") {
                successors.entry(link.node).or_default().push(node_id);
            }
        }
    }
    let mut pending: Vec<usize> = subagents.iter().copied().collect();
    while let Some(node) = pending.pop() {
        for &child in successors.get(&node).into_iter().flatten() {
            if subagents.insert(child) {
                pending.push(child);
            }
        }
    }
    let (mut output_tokens, mut input_tokens) = (0f64, 0f64);
    for branch in trace.branches() {
        for (i, increment) in min_new_input_tokens(branch.calls.iter().copied()) {
            let call = branch.calls[i];
            let weight = if call.node.is_some_and(|n| subagents.contains(&n)) { subagent_token_weight } else { 1.0 };
            let completion = call.usage.as_ref().map_or(0, |u| u.completion_tokens);
            output_tokens += weight * completion as f64;
            input_tokens += weight * increment as f64;
        }
    }
    (output_tokens, input_tokens)
}

/// Mean of f32 values as torch computes it (f32 result); accumulated in f64.
fn mean_f32(values: &[f32]) -> f32 {
    (values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64) as f32
}

fn max_clamp1(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max).max(1.0)
}

/// Group Relative Policy Optimization: credit = reward minus the group mean (optionally
/// length-shaped); action tokens feed the `rl` loss.
pub struct GrpoAlgorithm {
    pub(super) action_loss_type: ActionLossType,
    pub(super) length_penalty: Option<LengthPenaltyConfig>,
    pub(super) length_weighted_baseline: bool,
    pub(super) loss_aggregation: LossAggregation,
}

impl GrpoAlgorithm {
    fn score(&self, episodes: &mut [Episode]) -> Result<()> {
        let ids = iter_trainable_traces(episodes);
        let traces: Vec<&Trace> = ids.iter().map(|&(e, t)| &episodes[e].traces[t]).collect();
        let rewards: Vec<f32> = traces.iter().map(|t| t.reward() as f32).collect();
        if rewards.is_empty() {
            return Ok(());
        }
        let shaped: Vec<f32> = match &self.length_penalty {
            None => rewards.clone(),
            Some(penalty) => {
                let (w_out, w_in, w_turns) = match penalty {
                    LengthPenaltyConfig::Linear(c) => (c.num_output_tokens_weight, c.num_input_tokens_weight, c.num_turns_weight),
                    LengthPenaltyConfig::LinearSubagent(c) => {
                        (c.num_output_tokens_weight, c.num_input_tokens_weight, c.num_turns_weight)
                    }
                };
                let (output, input): (Vec<f32>, Vec<f32>) = match penalty {
                    LengthPenaltyConfig::LinearSubagent(c) => traces
                        .iter()
                        .map(|t| {
                            let (o, i) = length_penalty_tokens(t, c.subagent_token_weight);
                            (o as f32, i as f32)
                        })
                        .unzip(),
                    LengthPenaltyConfig::Linear(_) => traces
                        .iter()
                        .map(|t| {
                            let output = t.num_output_tokens() as f32;
                            (output, t.num_total_tokens() as f32 - output)
                        })
                        .unzip(),
                };
                let turns: Vec<f32> = traces.iter().map(|t| t.num_turns() as f32).collect();
                let (max_out, max_in, max_turns) = (max_clamp1(&output), max_clamp1(&input), max_clamp1(&turns));
                let pass_rate = mean_f32(&rewards);
                (0..rewards.len())
                    .map(|i| {
                        let frac = w_out as f32 * (output[i] / max_out)
                            + w_in as f32 * (input[i] / max_in)
                            + w_turns as f32 * (turns[i] / max_turns);
                        rewards[i] - pass_rate * frac
                    })
                    .collect()
            }
        };
        let baseline = if self.length_weighted_baseline {
            let lengths: Vec<f32> = traces
                .iter()
                .map(|t| trainable_node_ids(t).iter().map(|&n| count_true(&t.nodes[n].mask)).sum::<usize>() as f32)
                .collect();
            let weighted: f64 = lengths.iter().zip(&shaped).map(|(&l, &r)| f64::from(l * r)).sum();
            let total: f64 = lengths.iter().map(|&l| f64::from(l)).sum();
            (weighted as f32) / (total as f32)
        } else {
            mean_f32(&shaped)
        };
        let advantages: Vec<f32> = shaped.iter().map(|&r| r - baseline).collect();
        for (&(e, t), &advantage) in ids.iter().zip(&advantages) {
            assign_advantages(&mut episodes[e].traces[t], Advantages::Scalar(advantage))?;
        }
        if self.loss_aggregation == LossAggregation::Prompt {
            let nodes: Vec<(usize, usize, usize)> = ids
                .iter()
                .flat_map(|&(e, t)| trainable_node_ids(&episodes[e].traces[t]).into_iter().map(move |n| (e, t, n)))
                .collect();
            if !nodes.is_empty() {
                let total: usize = nodes.iter().map(|&(e, t, n)| count_true(&episodes[e].traces[t].nodes[n].mask)).sum();
                let weight = (1.0 / total as f64) as f32;
                for (e, t, n) in nodes {
                    let node = &mut episodes[e].traces[t].nodes[n];
                    let rl: Vec<f32> = node.mask.iter().map(|&m| if m { weight } else { 0.0 }).collect();
                    node.loss_weights.get_or_insert_with(IndexMap::new).insert("rl".into(), rl);
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Algorithm for GrpoAlgorithm {
    fn action_loss_type(&self) -> ActionLossType {
        self.action_loss_type
    }

    async fn score_group(&mut self, episodes: &mut [Episode]) -> Result<()> {
        self.score(episodes)
    }
}

/// GRPO on action tokens plus weighted CE on env-provided tokens of later turns, selected by role.
pub struct EchoAlgorithm {
    grpo: GrpoAlgorithm,
    role_weights: IndexMap<&'static str, f32>,
    filter: Option<(String, serde_json::Value, Arc<dyn Bridge>)>,
}

impl EchoAlgorithm {
    pub fn new(action_loss_type: ActionLossType, config: &EchoAlgoConfig, bridge: Option<Arc<dyn Bridge>>) -> Result<Self> {
        let roles = &config.roles;
        let role_weights = [("system", &roles.system), ("user", &roles.user), ("assistant", &roles.assistant), ("tool", &roles.tool)]
            .into_iter()
            .filter_map(|(role, c)| c.as_ref().map(|c| (role, c.alpha as f32)))
            .collect();
        let filter = match &config.filter {
            None => None,
            Some(f) => {
                let Some(bridge) = bridge else { bail!("echo filter {:?} needs the Python bridge", f.import_path) };
                Some((f.import_path.clone(), f.kwargs.clone(), bridge))
            }
        };
        Ok(Self {
            grpo: GrpoAlgorithm {
                action_loss_type,
                length_penalty: config.length_penalty.clone(),
                length_weighted_baseline: config.length_weighted_baseline,
                loss_aggregation: config.loss_aggregation,
            },
            role_weights,
            filter,
        })
    }

    async fn filter_masks(&self, trace: &Trace, branch_lengths: &[usize]) -> Result<Option<Vec<Vec<bool>>>> {
        let Some((import_path, kwargs, bridge)) = &self.filter else { return Ok(None) };
        let record = serde_json::from_slice(&prl_wire::json::to_vec(&trace.to_record(None)))?;
        let masks = bridge.echo_filter(import_path, kwargs.clone(), record).await?;
        if masks.len() != branch_lengths.len() {
            bail!(
                "echo filter must return one keep-mask per trainable branch: got {}, expected {}",
                masks.len(),
                branch_lengths.len()
            );
        }
        for (branch_idx, (mask, &expected)) in masks.iter().zip(branch_lengths).enumerate() {
            if mask.len() != expected {
                bail!(
                    "echo filter mask for branch {branch_idx} must span the branch's tokens: got {}, expected {expected}",
                    mask.len()
                );
            }
        }
        Ok(Some(masks))
    }

    /// Write graph-native `ce` weights over the env-provided observation tokens of later turns.
    async fn weight_observations(&self, trace: &mut Trace) -> Result<()> {
        // (node ids, per-token weights) of each trainable branch, computed before any write.
        let branches: Vec<(Vec<usize>, usize)> =
            iter_trainable_branches(trace).iter().map(|(b, _)| (b.node_ids.clone(), b.num_tokens())).collect();
        let lengths: Vec<usize> = branches.iter().map(|(_, n)| *n).collect();
        let filter_masks = self.filter_masks(trace, &lengths).await?;
        let mut updates = Vec::new();
        for (branch_idx, (node_ids, num_tokens)) in branches.iter().enumerate() {
            let mut weights = vec![0f32; *num_tokens];
            let keep = filter_masks.as_ref().map(|m| &m[branch_idx]);
            let mut offset = 0;
            let mut seen_response = false;
            for &n in node_ids {
                let node = &trace.nodes[n];
                let span = node.token_ids.len();
                if seen_response && !node.sampled {
                    if let Some(&alpha) = self.role_weights.get(node.role()) {
                        for i in 0..span {
                            let content = node.is_content.len() != span || node.is_content[i];
                            let kept = keep.is_none_or(|k| k[offset + i]);
                            if content && kept {
                                weights[offset + i] = alpha;
                            }
                        }
                    }
                }
                if node.sampled {
                    seen_response = true;
                }
                offset += span;
            }
            updates.push((node_ids.clone(), weights));
        }
        for (node_ids, weights) in updates {
            let mut offset = 0;
            for n in node_ids {
                let node = &mut trace.nodes[n];
                let end = offset + node.token_ids.len();
                let node_weights = &weights[offset..end];
                if node_weights.iter().any(|&w| w != 0.0) {
                    let streams = node.loss_weights.get_or_insert_with(IndexMap::new);
                    let current = streams.get("ce").cloned().unwrap_or_else(|| vec![0.0; end - offset]);
                    let merged = current.iter().zip(node_weights).map(|(&a, &b)| a.max(b)).collect();
                    streams.insert("ce".into(), merged);
                }
                offset = end;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Algorithm for EchoAlgorithm {
    fn action_loss_type(&self) -> ActionLossType {
        self.grpo.action_loss_type
    }

    async fn score_episode(&mut self, episode: &mut Episode) -> Result<()> {
        for trace in &mut episode.traces {
            if !trace.has_error() && trace.agent.trainable {
                self.weight_observations(trace).await?;
            }
        }
        Ok(())
    }

    async fn score_group(&mut self, episodes: &mut [Episode]) -> Result<()> {
        self.grpo.score(episodes)
    }
}

/// MaxRL: `(reward - group mean) / group mean`; zeros when the mean is <= 0.
pub struct MaxRlAlgorithm {
    pub action_loss_type: ActionLossType,
}

#[async_trait]
impl Algorithm for MaxRlAlgorithm {
    fn action_loss_type(&self) -> ActionLossType {
        self.action_loss_type
    }

    async fn score_group(&mut self, episodes: &mut [Episode]) -> Result<()> {
        let ids = iter_trainable_traces(episodes);
        let rewards: Vec<f32> = ids.iter().map(|&(e, t)| episodes[e].traces[t].reward() as f32).collect();
        if rewards.is_empty() {
            return Ok(());
        }
        let mean = mean_f32(&rewards);
        for (&(e, t), &reward) in ids.iter().zip(&rewards) {
            let advantage = if mean <= 0.0 { 0.0 } else { (reward - mean) / mean };
            assign_advantages(&mut episodes[e].traces[t], Advantages::Scalar(advantage))?;
        }
        Ok(())
    }
}

/// RAE: reward minus a per-agent EMA baseline, scored before the baseline update.
pub struct RaeAlgorithm {
    pub(super) action_loss_type: ActionLossType,
    pub(super) decay: f64,
    pub(super) baselines: HashMap<String, f64>,
}

#[async_trait]
impl Algorithm for RaeAlgorithm {
    fn action_loss_type(&self) -> ActionLossType {
        self.action_loss_type
    }

    async fn score_group(&mut self, episodes: &mut [Episode]) -> Result<()> {
        for (e, t) in iter_trainable_traces(episodes) {
            let trace = &mut episodes[e].traces[t];
            let reward = trace.reward();
            let baseline = self.baselines.get(&trace.agent.name).copied().unwrap_or(0.0);
            assign_advantages(trace, Advantages::Scalar((reward - baseline) as f32))?;
            self.baselines.insert(trace.agent.name.clone(), self.decay * baseline + (1.0 - self.decay) * reward);
        }
        Ok(())
    }
}

/// GRPO for proposer-solver envs: episode-scoped peers for `episode_agents`, group-scoped otherwise.
pub struct HierarchicalGrpoAlgorithm {
    pub(super) action_loss_type: ActionLossType,
    pub(super) episode_agents: HashSet<String>,
}

#[async_trait]
impl Algorithm for HierarchicalGrpoAlgorithm {
    fn action_loss_type(&self) -> ActionLossType {
        self.action_loss_type
    }

    async fn score_group(&mut self, episodes: &mut [Episode]) -> Result<()> {
        let mut peers: IndexMap<(String, Option<String>), Vec<(usize, usize)>> = IndexMap::new();
        for (e, t) in iter_trainable_traces(episodes) {
            let name = episodes[e].traces[t].agent.name.clone();
            let scope = self.episode_agents.contains(&name).then(|| episodes[e].id.clone());
            peers.entry((name, scope)).or_default().push((e, t));
        }
        for members in peers.values() {
            let baseline = members.iter().map(|&(e, t)| episodes[e].traces[t].reward()).sum::<f64>() / members.len() as f64;
            for &(e, t) in members {
                let trace = &mut episodes[e].traces[t];
                let advantage = trace.reward() - baseline;
                assign_advantages(trace, Advantages::Scalar(advantage as f32))?;
            }
        }
        Ok(())
    }
}

/// Hard distillation: CE on the frozen teacher's sampled tokens; assigns nothing.
pub struct SftAlgorithm {
    pub action_loss_type: ActionLossType,
}

#[async_trait]
impl Algorithm for SftAlgorithm {
    fn action_loss_type(&self) -> ActionLossType {
        self.action_loss_type
    }
}

/// Infra debugging: a constant advantage on every sampled token of every clean trainable trace.
pub struct DebugAlgorithm {
    pub action_loss_type: ActionLossType,
    pub advantage: f32,
}

#[async_trait]
impl Algorithm for DebugAlgorithm {
    fn action_loss_type(&self) -> ActionLossType {
        self.action_loss_type
    }

    async fn score_episode(&mut self, episode: &mut Episode) -> Result<()> {
        for (_, t) in iter_trainable_traces(std::slice::from_ref(episode)) {
            assign_advantages(&mut episode.traces[t], Advantages::Scalar(self.advantage))?;
        }
        Ok(())
    }
}
