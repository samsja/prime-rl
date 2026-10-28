//! Orchestrator-side algorithm runtime (`orchestrator/algo/*`). One `Algorithm` per train env,
//! built by [`build_algorithm`]; algorithms annotate native verifier graphs (advantages, reference
//! logprobs, loss-weight streams) and the train sink compiles samples after admission.

mod distill;
mod grpo;
pub mod routing;

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use prl_config::{ActionLossType, AlgoConfig, AlgoParams, FrozenModelConfig};
use prl_types::{Bridge, Scorer};
use prl_wire::Episode;

use distill::{OpdAlgorithm, OpsdAlgorithm};
use grpo::{DebugAlgorithm, EchoAlgorithm, GrpoAlgorithm, HierarchicalGrpoAlgorithm, MaxRlAlgorithm, RaeAlgorithm, SftAlgorithm};
pub use routing::{is_trainable, scalar_advantage, stamp_loss_routing};

/// `(episode index, trace index)` of every clean, kept, trainable trace with sampled tokens.
pub fn iter_trainable_traces(episodes: &[Episode]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (e, episode) in episodes.iter().enumerate() {
        for (t, trace) in episode.traces.iter().enumerate() {
            if !episode.ok || trace.has_error() || trace.is_dropped() || !trace.agent.trainable || trace.num_turns() == 0 {
                continue;
            }
            if trace.nodes.iter().any(|n| n.mask.iter().any(|&m| m)) {
                out.push((e, t));
            }
        }
    }
    out
}

/// Assigns training annotations to verifier episodes. Override `score_episode` for rollout-local
/// work and `score_group` for cohort-relative work.
#[async_trait]
pub trait Algorithm: Send + Sync {
    fn action_loss_type(&self) -> ActionLossType;

    /// Connect resources owned by the algorithm (frozen model pools, renderers).
    async fn setup(&mut self) -> Result<()> {
        Ok(())
    }

    async fn score_episode(&mut self, _episode: &mut Episode) -> Result<()> {
        Ok(())
    }

    async fn score_group(&mut self, _episodes: &mut [Episode]) -> Result<()> {
        Ok(())
    }

    /// Run rollout-local scoring when the episode has trainable traces.
    async fn finalize_episode(&mut self, episode: &mut Episode) -> Result<()> {
        if !iter_trainable_traces(std::slice::from_ref(episode)).is_empty() {
            self.score_episode(episode).await?;
        }
        Ok(())
    }
}

/// Connects a frozen model endpoint (`connect_frozen_client`).
pub type ConnectFrozen = Arc<dyn Fn(FrozenModelConfig) -> BoxFuture<'static, Result<Arc<dyn Scorer>>> + Send + Sync>;

/// What algorithms may need from the rest of the orchestrator.
#[derive(Clone)]
pub struct AlgorithmDeps {
    /// Prefill scoring against the live policy pool (opsd's teacher).
    pub policy: Arc<dyn Scorer>,
    /// The policy model name (opsd builds its hint renderer from its tokenizer).
    pub policy_model_name: String,
    pub connect_frozen: ConnectFrozen,
    /// Python bridge: opsd hint rendering, echo user filters.
    pub bridge: Option<Arc<dyn Bridge>>,
}

pub fn build_algorithm(config: &AlgoConfig, deps: &AlgorithmDeps) -> Result<Box<dyn Algorithm>> {
    let action_loss_type = config.action_loss_type;
    Ok(match &config.params {
        AlgoParams::Grpo(c) => Box::new(GrpoAlgorithm {
            action_loss_type,
            length_penalty: c.length_penalty.clone(),
            length_weighted_baseline: c.length_weighted_baseline,
            loss_aggregation: c.loss_aggregation,
        }),
        AlgoParams::Echo(c) => Box::new(EchoAlgorithm::new(action_loss_type, c, deps.bridge.clone())?),
        AlgoParams::MaxRl => Box::new(MaxRlAlgorithm { action_loss_type }),
        AlgoParams::Rae(c) => Box::new(RaeAlgorithm { action_loss_type, decay: c.decay, baselines: Default::default() }),
        AlgoParams::HierarchicalGrpo(c) => Box::new(HierarchicalGrpoAlgorithm { action_loss_type, episode_agents: c.episode_agents.iter().cloned().collect() }),
        AlgoParams::Opd(c) => Box::new(OpdAlgorithm::new(action_loss_type, c.teacher.clone(), deps.connect_frozen.clone())),
        AlgoParams::Opsd(c) => Box::new(OpsdAlgorithm::new(action_loss_type, c, deps)?),
        AlgoParams::Sft => Box::new(SftAlgorithm { action_loss_type }),
        AlgoParams::Debug(c) => Box::new(DebugAlgorithm { action_loss_type, advantage: c.advantage as f32 }),
    })
}
