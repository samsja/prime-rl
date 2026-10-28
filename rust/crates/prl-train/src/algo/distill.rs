//! Reference-KL algorithms: opd (frozen teacher) and opsd (demo-conditioned live policy).

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use futures::future::try_join_all;
use prl_config::{ActionLossType, FrozenModelConfig, OPSDAlgoConfig};
use prl_types::{Bridge, Scorer};
use prl_wire::{Episode, Value};

use super::routing::assign_reference_logprobs;
use super::{iter_trainable_traces, Algorithm, AlgorithmDeps, ConnectFrozen};
use crate::episodes::episode_env_name;
use crate::trajectories::iter_trainable_branches;

/// `(node ids, token ids)` of every trainable branch of the trace.
fn trainable_branches(trace: &prl_wire::Trace) -> Vec<(Vec<usize>, Vec<i32>)> {
    iter_trainable_branches(trace).into_iter().map(|(b, _)| (b.node_ids.clone(), b.token_ids())).collect()
}

/// On-policy distillation: each trainable branch is prefill-scored under the frozen teacher.
pub struct OpdAlgorithm {
    action_loss_type: ActionLossType,
    teacher: FrozenModelConfig,
    connect_frozen: ConnectFrozen,
    teacher_client: Option<Arc<dyn Scorer>>,
}

impl OpdAlgorithm {
    pub fn new(action_loss_type: ActionLossType, teacher: FrozenModelConfig, connect_frozen: ConnectFrozen) -> Self {
        Self { action_loss_type, teacher, connect_frozen, teacher_client: None }
    }

    /// Score with an already-connected teacher (tests, shared pools).
    pub fn with_teacher(action_loss_type: ActionLossType, teacher: FrozenModelConfig, client: Arc<dyn Scorer>) -> Self {
        let connect_frozen: ConnectFrozen = Arc::new(|_| Box::pin(async { Err(anyhow!("teacher already connected")) }));
        Self { action_loss_type, teacher, connect_frozen, teacher_client: Some(client) }
    }
}

#[async_trait]
impl Algorithm for OpdAlgorithm {
    fn action_loss_type(&self) -> ActionLossType {
        self.action_loss_type
    }

    async fn setup(&mut self) -> Result<()> {
        self.teacher_client = Some((self.connect_frozen)(self.teacher.clone()).await?);
        Ok(())
    }

    async fn score_episode(&mut self, episode: &mut Episode) -> Result<()> {
        let teacher = self.teacher_client.clone().ok_or_else(|| anyhow!("teacher not connected — Algorithm.setup() must run first"))?;
        let mut branches = Vec::new();
        for (_, t) in iter_trainable_traces(std::slice::from_ref(episode)) {
            for (node_ids, token_ids) in trainable_branches(&episode.traces[t]) {
                branches.push((t, node_ids, token_ids));
            }
        }
        let scores = try_join_all(branches.iter().map(|(_, _, tokens)| teacher.score(tokens.clone()))).await?;
        for ((t, node_ids, _), logprobs) in branches.iter().zip(scores) {
            assign_reference_logprobs(&mut episode.traces[*t], node_ids, &logprobs)?;
        }
        Ok(())
    }
}

/// On-policy self-distillation: the live policy scores each sample with the expert demonstration
/// prepended as a leading system message.
pub struct OpsdAlgorithm {
    action_loss_type: ActionLossType,
    demo_key: String,
    template: String,
    renderer_config: serde_json::Value,
    policy_model_name: String,
    teacher: Arc<dyn Scorer>,
    bridge: Arc<dyn Bridge>,
}

impl OpsdAlgorithm {
    pub fn new(action_loss_type: ActionLossType, config: &OPSDAlgoConfig, deps: &AlgorithmDeps) -> Result<Self> {
        let bridge = deps.bridge.clone().ok_or_else(|| anyhow!("opsd needs the Python bridge to render its hint block"))?;
        Ok(Self {
            action_loss_type,
            demo_key: config.demo_key.clone(),
            template: config.template.clone(),
            renderer_config: config.renderer.clone(),
            policy_model_name: deps.policy_model_name.clone(),
            teacher: deps.policy.clone(),
            bridge,
        })
    }

    fn demonstration(&self, episode: &Episode, trace: &prl_wire::Trace) -> Result<String> {
        let value = trace
            .info
            .get(&self.demo_key)
            .filter(|v| !v.is_nil())
            .or_else(|| trace.task.data.get(&self.demo_key).filter(|v| !v.is_nil()));
        match value {
            Some(v) => Ok(python_str(v)),
            None => bail!(
                "opsd requires '{}' in the trace info dict or on the task (env '{}', task {}).",
                self.demo_key,
                episode_env_name(episode)?,
                trace.task.idx().map_or("None".to_string(), |i| i.to_string())
            ),
        }
    }
}

/// `str(value)` for the JSON-like values a demonstration can be.
fn python_str(value: &Value) -> String {
    match value {
        Value::Str(s) => s.clone(),
        other => prl_wire::json::to_python_json(other, false),
    }
}

/// `template.format(demonstration=...)`: `{demonstration}` substituted, `{{`/`}}` unescaped.
pub fn format_template(template: &str, demonstration: &str) -> Result<String> {
    let mut out = String::with_capacity(template.len() + demonstration.len());
    let mut rest = template;
    while let Some(i) = rest.find(['{', '}']) {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        if let Some(after) = tail.strip_prefix("{{") {
            out.push('{');
            rest = after;
        } else if let Some(after) = tail.strip_prefix("}}") {
            out.push('}');
            rest = after;
        } else if let Some(after) = tail.strip_prefix("{demonstration}") {
            out.push_str(demonstration);
            rest = after;
        } else {
            bail!("opsd template may only reference {{demonstration}}: {template:?}");
        }
    }
    out.push_str(rest);
    Ok(out)
}

#[async_trait]
impl Algorithm for OpsdAlgorithm {
    fn action_loss_type(&self) -> ActionLossType {
        self.action_loss_type
    }

    async fn score_episode(&mut self, episode: &mut Episode) -> Result<()> {
        for (_, t) in iter_trainable_traces(std::slice::from_ref(episode)) {
            let hint = format_template(&self.template, &self.demonstration(episode, &episode.traces[t])?)?;
            let hint_block = self.bridge.render_ids(&self.policy_model_name, self.renderer_config.clone(), &hint).await?;
            let branches = trainable_branches(&episode.traces[t]);
            let scores = try_join_all(branches.iter().map(|(_, tokens)| {
                let mut full = hint_block.clone();
                full.extend_from_slice(tokens);
                self.teacher.score(full)
            }))
            .await?;
            for ((node_ids, _), full_logprobs) in branches.iter().zip(scores) {
                let logprobs = full_logprobs.get(hint_block.len()..).unwrap_or_default();
                assign_reference_logprobs(&mut episode.traces[t], node_ids, logprobs)?;
            }
        }
        Ok(())
    }
}
