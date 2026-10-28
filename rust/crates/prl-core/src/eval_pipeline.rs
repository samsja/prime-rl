//! Eval side of the step loop: eval results from the dispatcher (episodes logged to `all` on
//! arrival, then counted into per-env epochs), eval triggers after every applied policy, and
//! epoch finalization (`trigger_eval` / `finalize_eval_batch` of `orchestrator.py`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Result};
use async_trait::async_trait;
use indexmap::IndexMap;
use prl_io::logging::format_time;
use prl_monitors::annotations::{stamp_arrival, stamp_batch};
use prl_monitors::Monitors;
use prl_train::{dispatch_failure_metrics, EvalEpisodes};
use prl_types::{DispatchResult, Kind, Metrics, Subset};
use prl_wire::{Episode, RunInfo};
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::dispatcher::DispatcherHandle;
use crate::eval_sink::{EvalEpoch, EvalSink};
use crate::watcher::UpdateHook;

/// Per eval env: group size and number of examples.
#[derive(Debug, Clone)]
pub struct EvalEnvShape {
    pub group_size: usize,
    pub num_examples: usize,
}

pub struct EvalPipelineConfig {
    pub envs: IndexMap<String, EvalEnvShape>,
    pub resume_step: Option<u64>,
    pub retrigger_on_resume: bool,
    pub max_steps: Option<u64>,
}

/// `(env, step, arrived, expected)` per open epoch, for the pipeline view.
pub type EvalProgress = Vec<(String, u64, usize, usize)>;

enum Command {
    PolicyApplied { step: u64, ack: oneshot::Sender<Result<()>> },
}

#[derive(Clone)]
pub struct EvalPipelineHandle {
    tx: mpsc::UnboundedSender<Command>,
}

#[async_trait]
impl UpdateHook for EvalPipelineHandle {
    async fn on_update(&self, step: u64) -> Result<()> {
        let (ack, rx) = oneshot::channel();
        self.tx.send(Command::PolicyApplied { step, ack }).map_err(|_| anyhow::anyhow!("eval pipeline stopped"))?;
        rx.await.map_err(|_| anyhow::anyhow!("eval pipeline stopped"))?
    }
}

pub struct EvalPipeline {
    config: EvalPipelineConfig,
    sink: EvalSink<Arc<Episode>>,
    dispatcher: DispatcherHandle,
    monitors: Monitors,
    triggered_steps: HashSet<u64>,
    triggered_at: HashMap<(String, u64), Instant>,
    progress: watch::Sender<EvalProgress>,
}

/// Delete the env-server arrays of an episode (eval never trains on them).
pub fn release_episode(episode: &Episode) {
    prl_wire::payload::delete_payloads(episode.traces.iter().flat_map(prl_wire::payload::trace_payload_urls));
}

fn eval_step(episode: &Episode) -> Result<u64> {
    let Some(run) = &episode.run else { bail!("Dispatched episode is missing run identity") };
    let RunInfo::Train(run) = run else { bail!("Orchestrated episode is missing training-run provenance") };
    Ok(run.work.step() as u64)
}

impl EvalPipeline {
    pub fn new(
        config: EvalPipelineConfig,
        dispatcher: DispatcherHandle,
        monitors: Monitors,
    ) -> (Self, EvalPipelineHandle, watch::Receiver<EvalProgress>, EvalInbox) {
        let epoch_sizes = config.envs.iter().map(|(n, e)| (n.clone(), e.group_size * e.num_examples)).collect();
        let (progress, progress_rx) = watch::channel(Vec::new());
        let (tx, cmd_rx) = mpsc::unbounded_channel();
        let this = Self {
            config,
            sink: EvalSink::new(epoch_sizes),
            dispatcher,
            monitors,
            triggered_steps: HashSet::new(),
            triggered_at: HashMap::new(),
            progress,
        };
        (this, EvalPipelineHandle { tx }, progress_rx, EvalInbox { cmd_rx })
    }

    pub async fn run(
        mut self,
        inbox: EvalInbox,
        mut results: mpsc::Receiver<DispatchResult<Episode>>,
        cancel: CancellationToken,
    ) -> Result<()> {
        let mut cmd_rx = inbox.cmd_rx;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                Some(cmd) = cmd_rx.recv() => match cmd {
                    Command::PolicyApplied { step, ack } => {
                        let _ = ack.send(self.trigger_eval(step).await);
                    }
                },
                result = results.recv() => {
                    let Some(result) = result else { return Ok(()) };
                    self.handle(result).await?;
                    self.progress.send_replace(self.sink.batch_progress());
                }
            }
        }
    }

    async fn handle(&mut self, result: DispatchResult<Episode>) -> Result<()> {
        let epoch = match result {
            DispatchResult::Episode(mut episode) => {
                let step = eval_step(&episode)?;
                stamp_arrival(std::slice::from_mut(&mut episode), Kind::Eval, step);
                let env_name = episode.env.name.clone().unwrap_or_default();
                let episode = Arc::new(episode);
                self.monitors.log_episodes(vec![episode.clone()], step, Kind::Eval, Subset::All).await;
                release_episode(&episode);
                self.sink.add(&env_name, step, episode)
            }
            DispatchResult::Failure(failure) => self.sink.fail(failure)?,
            DispatchResult::Cancelled(cancellation) => self.sink.cancel(&cancellation)?,
        };
        if let Some(epoch) = epoch {
            self.finalize(epoch).await?;
        }
        Ok(())
    }

    /// Fire eligible eval epochs after inference applied v{step} and prefer eval work.
    async fn trigger_eval(&mut self, step: u64) -> Result<()> {
        if self.triggered_steps.contains(&step) {
            return Ok(());
        }
        if self.config.resume_step == Some(step) && !self.config.retrigger_on_resume {
            return Ok(());
        }
        let is_final = self.config.max_steps.is_some_and(|max| step >= max);
        let reason = format!("eval was triggered at step {step}");
        let outcome = self.dispatcher.trigger_eval(step, is_final, Vec::new(), Some(reason)).await?;
        if outcome.fired.is_empty() {
            return Ok(());
        }
        self.triggered_steps.insert(step);
        let now = Instant::now();
        let mut total = 0;
        for env_name in &outcome.fired {
            self.triggered_at.insert((env_name.clone(), step), now);
        }
        for env_name in &outcome.fired {
            let shape = &self.config.envs[env_name];
            let expected = shape.group_size * shape.num_examples;
            total += expected;
            self.monitors.log_eval_plan(env_name.clone(), step, expected as u64).await;
        }
        info!("Starting evals in {} ({total} total rollouts)", outcome.fired.join(", "));
        Ok(())
    }

    async fn finalize(&mut self, epoch: EvalEpoch<Arc<Episode>>) -> Result<()> {
        let EvalEpoch { env_name, step, episodes, failures, .. } = epoch;
        if episodes.is_empty() && failures.is_empty() {
            warn!("Eval @ step={step} env={env_name}: no attempts returned, skipping log");
            return Ok(());
        }
        let group_size = self.config.envs[&env_name].group_size;
        let mut versions: Vec<u64> = Vec::with_capacity(episodes.len() + failures.len());
        for episode in &episodes {
            let policy = match &episode.run {
                Some(RunInfo::Train(run)) => run.work.policy().map(|p| p.start as u64),
                _ => None,
            };
            let Some(start) = policy else { bail!("Eval {env_name} step {step} is missing policy provenance") };
            versions.push(start);
        }
        versions.extend(failures.iter().map(|f| f.policy_version));
        let policy_version = versions.into_iter().min().expect("non-empty epoch");
        let total_attempts = episodes.len() + failures.len();

        let all = EvalEpisodes::new(episodes, group_size);
        let effective = all.effective();
        if !effective.is_empty() {
            let records: Vec<Arc<Episode>> = effective.vf_episodes().into_iter().map(Arc::new).collect();
            self.monitors.log_episodes(records.clone(), step, Kind::Eval, Subset::Effective).await;
            self.monitors.log_annotations(stamp_batch(records.iter().map(|e| e.as_ref()), step)).await;
        }
        let mut metrics = Metrics::new();
        let prefix = format!("eval/{env_name}");
        metrics.extend(all.to_wandb(&prefix, Subset::All));
        metrics.extend(effective.to_wandb(&prefix, Subset::Effective));
        metrics.extend(dispatch_failure_metrics(&failures, &format!("eval/{env_name}/all"), total_attempts));
        metrics.insert(format!("eval/{env_name}/policy_version"), (policy_version as f64).into());
        metrics.insert("step".into(), (step as f64).into());
        self.monitors.log_metrics(metrics, Some(step)).await;

        let (reward, turns, branches, truncated) = effective
            .with_metrics(|m| (m.reward().mean(), m.num_turns().mean(), m.num_branches().mean(), m.is_truncated().mean()));
        let (errored, timed_out) = all.with_metrics(|m| (m.has_error().mean(), m.is_timeout().mean()));
        let elapsed =
            self.triggered_at.remove(&(env_name.clone(), step)).map_or(0.0, |t| t.elapsed().as_secs_f64());
        prl_io::success!(
            "Evaluated {env_name} | Policy v{policy_version} | {:>7} | Reward {:.4} | Turns {:.1} | Branches {:.1} | \
             Error {} | Truncation {} | Timeout {}",
            format_time(elapsed),
            reward,
            turns,
            branches,
            crate::fmt::pct(errored),
            crate::fmt::pct(truncated),
            crate::fmt::pct(timed_out),
        );
        Ok(())
    }
}

pub struct EvalInbox {
    cmd_rx: mpsc::UnboundedReceiver<Command>,
}
