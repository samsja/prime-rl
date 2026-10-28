//! Train side of the step loop (`main_loop` train branch, `handle_train_result`,
//! `finalize_train_batch`, `start_draining` of `orchestrator.py`).
//!
//! Packing and sending a batch (which blocks until every trainer rank requested it) runs in its
//! own task while results keep flowing into the sink. Results that arrive during a send belong
//! to the next batch window (`progress.step + 1`), as in Python, which consumed them only after
//! the send. The post-send work (step advance, checkpoint, metrics, drain) runs in order once
//! the send completes; batches completed meanwhile wait in a queue. Control submissions are
//! applied at any time.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use async_trait::async_trait;
use indexmap::IndexMap;
use prl_io::ckpt::{CheckpointManager, OrchestratorState};
use prl_io::control::Controller;
use prl_io::heartbeat::Heartbeat;
use prl_io::logging::format_time;
use prl_monitors::annotations::{stamp_arrival, stamp_batch};
use prl_monitors::Monitors;
use prl_train::algo::routing::is_trainable;
use prl_train::sink::{Admission, TrainBatch, TrainSink};
use prl_train::staleness::{rollout_staleness, Staleness};
use prl_train::{BatchPacker, BatchSender, TrainEpisodes};
use prl_types::{DispatchResult, Kind, Metrics, Progress, Subset};
use prl_wire::{Episode, RunInfo};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::curriculum::GroupOutcome;
use crate::dispatcher::DispatcherHandle;
use crate::fmt::pct;
use crate::train_source::TrainSourceHandle;

/// The train sink's progress for the pipeline view.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainView {
    pub batch: usize,
    pub target: usize,
    pub buffered: usize,
    pub batch_by_env: IndexMap<String, usize>,
}

/// What the curriculum reads off a finalized group.
pub fn group_outcome(group: &[Episode]) -> GroupOutcome {
    let mut outcome = GroupOutcome::default();
    for episode in group {
        outcome.task_keys.push(episode.task.key.clone());
        for trace in &episode.traces {
            if !episode.ok || trace.has_error() || !trace.agent.trainable {
                continue;
            }
            if trace.num_turns() > 0 {
                outcome.rewards.push(trace.reward());
            }
            for node in &trace.nodes {
                if let Some(advantages) = &node.advantages {
                    outcome.advantages.extend_from_slice(advantages);
                }
            }
            if !trace.is_dropped() && trace.num_turns() > 0 && trace.nodes.iter().any(|n| n.mask.iter().any(|m| *m)) {
                outcome.any_trainable = true;
            }
        }
    }
    outcome
}

/// `TrainSource.on_result` as the sink's admission callback.
pub struct SourceAdmission(pub TrainSourceHandle);

#[async_trait]
impl Admission for SourceAdmission {
    async fn admit(&mut self, group: &[Episode]) -> Result<bool> {
        let Some(first) = group.first() else { bail!("Cannot report an empty rollout group") };
        let env_name = first.env.name.clone().ok_or_else(|| anyhow::anyhow!("Orchestrated episode is missing its environment name"))?;
        self.0.on_result(&env_name, group_outcome(group)).await
    }
}

pub struct TrainPipelineConfig {
    pub max_steps: Option<u64>,
    pub ckpt_interval: Option<u64>,
    /// `ckpt` configured: checkpoints are saved (a resume without it only loads).
    pub ckpt_enabled: bool,
    pub weight_broadcast_timeout: Duration,
    pub train_env_names: Vec<String>,
}

enum Command {
    /// The final checkpoint state (`progress.step - 1`), when anything shipped.
    FinalState { reply: oneshot::Sender<Result<Option<OrchestratorState>>> },
    Stop { ack: oneshot::Sender<()> },
    /// Answered once everything handed to the pipeline before it was handled.
    Sync { ack: oneshot::Sender<()> },
}

#[derive(Clone)]
pub struct TrainPipelineHandle {
    tx: mpsc::UnboundedSender<Command>,
}

impl TrainPipelineHandle {
    pub async fn final_state(&mut self) -> Result<Option<OrchestratorState>> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Command::FinalState { reply }).map_err(|_| anyhow::anyhow!("train pipeline stopped"))?;
        rx.await.map_err(|_| anyhow::anyhow!("train pipeline stopped"))?
    }

    /// Wait until the pipeline finished handling what it is working on.
    pub async fn sync(&self) -> Result<()> {
        let (ack, rx) = oneshot::channel();
        self.tx.send(Command::Sync { ack }).map_err(|_| anyhow::anyhow!("train pipeline stopped"))?;
        rx.await.map_err(|_| anyhow::anyhow!("train pipeline stopped"))
    }

    /// Close the batch sender and exit.
    pub async fn stop(&self) -> Result<()> {
        let (ack, rx) = oneshot::channel();
        self.tx.send(Command::Stop { ack }).map_err(|_| anyhow::anyhow!("train pipeline stopped"))?;
        rx.await.map_err(|_| anyhow::anyhow!("train pipeline stopped"))
    }
}

pub struct TrainPipelineInbox {
    cmd_rx: mpsc::UnboundedReceiver<Command>,
}

pub struct TrainPipeline {
    pub config: TrainPipelineConfig,
    pub sink: TrainSink,
    pub admission: SourceAdmission,
    pub source: TrainSourceHandle,
    pub progress: Progress,
    pub step_tx: watch::Sender<u64>,
    /// The last policy version whose update hooks all ran.
    pub applied: watch::Receiver<u64>,
    pub dispatcher: DispatcherHandle,
    pub packer: Arc<BatchPacker>,
    /// None while a batch is being shipped (the ship task owns it).
    sender: Option<BatchSender>,
    pub ckpt: CheckpointManager,
    pub controller: Option<Controller>,
    pub monitors: Monitors,
    pub heartbeat: Option<Arc<Heartbeat>>,
    pub view: watch::Sender<TrainView>,
    shipped_state: Option<(u64, OrchestratorState)>,
    last_batch_at: Instant,
    draining: bool,
    shipping: Option<Shipping>,
    /// Batches completed while the previous one was still shipping.
    queued: VecDeque<TrainBatch>,
}

/// Pack and send of one batch, the result of the ship task.
struct Shipped {
    sender: BatchSender,
    pack_time: f64,
    send_batch_time: f64,
    wait_for_policy_time: f64,
}

/// A batch whose pack/send runs in its own task while results keep flowing into the sink.
struct Shipping {
    step: u64,
    step_time: f64,
    effective: TrainEpisodes,
    rollout_metrics: Metrics,
    task: JoinHandle<Result<Shipped>>,
}

impl TrainPipeline {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: TrainPipelineConfig,
        sink: TrainSink,
        source: TrainSourceHandle,
        progress: Progress,
        applied: watch::Receiver<u64>,
        dispatcher: DispatcherHandle,
        packer: Arc<BatchPacker>,
        sender: BatchSender,
        ckpt: CheckpointManager,
        controller: Option<Controller>,
        monitors: Monitors,
        heartbeat: Option<Arc<Heartbeat>>,
        step_tx: watch::Sender<u64>,
    ) -> (Self, TrainPipelineHandle, TrainPipelineInbox, watch::Receiver<TrainView>) {
        step_tx.send_replace(progress.step);
        let (view, view_rx) = watch::channel(TrainView::default());
        let (tx, cmd_rx) = mpsc::unbounded_channel();
        let this = Self {
            config,
            sink,
            admission: SourceAdmission(source.clone()),
            source,
            progress,
            step_tx,
            applied,
            dispatcher,
            packer,
            sender: Some(sender),
            ckpt,
            controller,
            monitors,
            heartbeat,
            view,
            shipped_state: None,
            last_batch_at: Instant::now(),
            draining: false,
            shipping: None,
            queued: VecDeque::new(),
        };
        this.publish_view();
        (this, TrainPipelineHandle { tx }, TrainPipelineInbox { cmd_rx }, view_rx)
    }

    fn publish_view(&self) {
        let (batch, target) = self.sink.batch_progress();
        self.view.send_replace(TrainView {
            batch,
            target,
            buffered: self.sink.buffered_count(),
            batch_by_env: self.sink.pending_batch_by_env(),
        });
    }

    fn beat(&self) {
        if let Some(heartbeat) = &self.heartbeat {
            heartbeat.beat();
        }
    }

    pub async fn run(
        mut self,
        inbox: TrainPipelineInbox,
        mut results: mpsc::Receiver<DispatchResult<Episode>>,
        cancel: CancellationToken,
    ) -> Result<()> {
        let mut cmd_rx = inbox.cmd_rx;
        // Anchor the step-time clock so the first step measures startup -> first batch.
        self.last_batch_at = Instant::now();
        let submitted = self.controller.as_ref().map(Controller::submitted);
        // The view walks the pending batch: refresh it on a tick, not per result.
        let mut view_tick = tokio::time::interval(Duration::from_secs(1));
        view_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut results_open = true;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                _ = view_tick.tick() => self.publish_view(),
                Some(cmd) = cmd_rx.recv() => match cmd {
                    Command::FinalState { reply } => {
                        let _ = reply.send(self.final_state().await);
                    }
                    Command::Stop { ack } => {
                        self.stop().await;
                        let _ = ack.send(());
                        return Ok(());
                    }
                    Command::Sync { ack } => {
                        let _ = ack.send(());
                    }
                },
                _ = async { submitted.as_ref().expect("controller").notified().await }, if submitted.is_some() => {
                    self.apply_control()?;
                }
                shipped = async { (&mut self.shipping.as_mut().expect("shipping").task).await }, if self.shipping.is_some() => {
                    let shipping = self.shipping.take().expect("shipping");
                    self.finish_ship(shipping, shipped??, &cancel).await?;
                    self.ship_next(None, &cancel).await?;
                }
                result = results.recv(), if results_open => match result {
                    Some(result) => self.handle(result, &cancel).await?,
                    // The dispatcher stopped; keep serving commands until the supervisor stops us.
                    None => results_open = false,
                },
            }
        }
    }

    async fn stop(&mut self) {
        if let Some(shipping) = self.shipping.take() {
            shipping.task.abort();
        }
        if let Some(sender) = self.sender.take() {
            sender.close().await;
        }
    }

    async fn handle(&mut self, result: DispatchResult<Episode>, cancel: &CancellationToken) -> Result<()> {
        // During a send the batch window collecting is the next one.
        let step = self.progress.step + u64::from(self.shipping.is_some());
        let batch = match result {
            DispatchResult::Episode(mut episode) => {
                let Some(run) = &episode.run else { bail!("Dispatched episode is missing run identity") };
                if !matches!(run, RunInfo::Train(_)) {
                    bail!("Orchestrated episode is missing training-run provenance");
                }
                // Every completed rollout lands in the `all` traces the moment it arrives, in the
                // batch window currently collecting.
                stamp_arrival(std::slice::from_mut(&mut episode), Kind::Train, step);
                self.monitors.log_episodes(vec![Arc::new(episode.clone())], step, Kind::Train, Subset::All).await;
                self.sink.add(episode, step, &mut self.admission).await?
            }
            DispatchResult::Failure(failure) => self.sink.fail(failure, step, &mut self.admission).await?,
            DispatchResult::Cancelled(cancellation) => self.sink.cancel(cancellation, step, &mut self.admission).await?,
        };
        self.handle_train_result(batch, cancel).await
    }

    async fn handle_train_result(&mut self, batch: Option<TrainBatch>, cancel: &CancellationToken) -> Result<()> {
        self.beat();
        self.ship_next(batch, cancel).await
    }

    /// Ship `batch` (or the next ready one) unless a send is in flight; batches that cannot
    /// ship after training stops are released.
    async fn ship_next(&mut self, mut batch: Option<TrainBatch>, cancel: &CancellationToken) -> Result<()> {
        loop {
            if self.draining {
                for batch in batch.take().into_iter().chain(self.queued.drain(..)) {
                    self.sink.count_queued_cancelled(batch.cohort.num_traces() as f64);
                    self.sink.release_samples(&batch.samples);
                }
                self.sink.discard_queued();
                return Ok(());
            }
            if self.shipping.is_some() {
                self.queued.extend(batch);
                return Ok(());
            }
            let next = match batch.take().or_else(|| self.queued.pop_front()) {
                Some(next) => next,
                None => match self.sink.take_batch(self.progress.step).await? {
                    Some(next) => next,
                    None => return Ok(()),
                },
            };
            self.start_ship(next, cancel).await?;
        }
    }

    /// Stop scheduling train work and let the pipeline empty; triggered evals still complete.
    async fn start_draining(&mut self, reason: &str, cancel: &CancellationToken) -> Result<()> {
        self.draining = true;
        self.sink.discard_queued();
        let cancellations = self.dispatcher.drain().await?;
        let n: usize = cancellations.iter().map(|c| c.count).sum();
        for cancellation in cancellations {
            let batch = self.sink.cancel(cancellation, self.progress.step, &mut self.admission).await?;
            Box::pin(self.handle_train_result(batch, cancel)).await?;
        }
        info!(
            "{reason} — draining pipeline (cancelled {n} outstanding train episode(s); any in-flight evals will complete)"
        );
        Ok(())
    }

    fn take_rollout_metrics(&mut self) -> Metrics {
        let Some(window) = self.sink.take_rollout_window() else { return Metrics::new() };
        let p = &mut self.progress;
        p.rollout_window += 1;
        p.total_tokens += window.tokens.max(0) as u64;
        p.total_samples += window.traces as u64;
        p.total_problems += window.groups as u64;
        let mut metrics = Metrics::new();
        metrics.insert("staleness/rollout/dropped".into(), 0.0.into());
        metrics.insert("rollout/queued/pruned".into(), 0.0.into());
        metrics.insert("rollout/queued/cancelled".into(), 0.0.into());
        metrics.extend(window.metrics.to_dict());
        let no_signal = window.discarded - window.stale - window.errored - window.cancelled;
        for (key, value) in [
            ("rollout/attempts", window.attempts),
            ("rollout/discarded", window.discarded),
            ("rollout/stale", window.stale),
            ("rollout/errored", window.errored),
            ("rollout/cancelled", window.cancelled),
            ("rollout/no_signal", no_signal),
            ("progress/tokens", window.tokens.max(0) as usize),
            ("progress/rollouts", window.traces),
            ("progress/tasks", window.groups),
            ("progress/total_tokens", p.total_tokens as usize),
            ("progress/total_rollouts", p.total_samples as usize),
            ("progress/total_tasks", p.total_problems as usize),
        ] {
            metrics.insert(key.into(), value.into());
        }
        if window.attempts > 0 && window.discarded as f64 / window.attempts as f64 > 0.5 {
            warn!(
                "Discarded {}/{} episodes ({}): stale={}, errored={}, cancelled={}, no_signal={no_signal}. \
                 Review max_staleness, episode errors, and reward signal.",
                window.discarded,
                window.attempts,
                pct(window.discarded as f64 / window.attempts as f64),
                window.stale,
                window.errored,
                window.cancelled,
            );
        }
        metrics
    }

    async fn snapshot(&mut self) -> Result<OrchestratorState> {
        Ok(OrchestratorState { progress: self.progress.clone(), train_source: self.source.state().await? })
    }

    async fn final_state(&mut self) -> Result<Option<OrchestratorState>> {
        if self.progress.step <= 1 {
            return Ok(None);
        }
        let mut state = self.snapshot().await?;
        state.progress.step -= 1;
        Ok(Some(state))
    }

    /// Save the state as of shipping the last step if a pause is due at it.
    fn apply_control(&mut self) -> Result<()> {
        let (Some(controller), Some((step, state))) = (&self.controller, &self.shipped_state) else { return Ok(()) };
        if controller.update(*step) {
            self.ckpt.save(state, *step)?;
            controller.mark_saved(*step);
            info!("Paused at step {step}, saved its state");
        }
        Ok(())
    }

    async fn wait_for_version(&mut self, version: u64, reason: &str, cancel: &CancellationToken) -> Result<()> {
        wait_for_version(&mut self.applied, version, reason, self.config.weight_broadcast_timeout, cancel).await
    }

    /// Pre-send half of `finalize_train_batch`: budget/empty checks, cohort logs, then pack and
    /// send in the ship task.
    async fn start_ship(&mut self, batch: TrainBatch, cancel: &CancellationToken) -> Result<()> {
        let step = self.progress.step;
        if let Some(max_steps) = self.config.max_steps.filter(|max| step > *max) {
            self.sink.count_queued_cancelled(batch.cohort.num_traces() as f64);
            self.sink.release_samples(&batch.samples);
            return Box::pin(self.start_draining(&format!("Step {step} exceeds max_steps={max_steps}"), cancel)).await;
        }
        if batch.samples.is_empty() {
            warn!("Step {step}: skipping empty train batch after zero-advantage pruning");
            return Ok(());
        }
        let now = Instant::now();
        let step_time = now.duration_since(self.last_batch_at).as_secs_f64();
        self.last_batch_at = now;

        let TrainBatch { cohort, samples } = batch;
        let effective = cohort.effective();
        drop(cohort);
        let n_effective = effective.num_traces();
        let n_trainable = effective.records().iter().filter(|r| is_trainable(r.trace)).count();
        if n_effective > 0 && n_trainable as f64 / n_effective as f64 <= 0.1 {
            warn!(
                "Only {n_trainable}/{n_effective} effective traces are trainable ({}) — consider reviewing task difficulty",
                pct(n_trainable as f64 / n_effective as f64)
            );
        }

        let rollout_metrics = self.take_rollout_metrics();

        let effective_episodes: Vec<Arc<Episode>> = effective.vf_episodes().into_iter().map(Arc::new).collect();
        self.monitors.log_episodes(effective_episodes.clone(), step, Kind::Train, Subset::Effective).await;
        self.monitors.log_annotations(stamp_batch(effective_episodes.iter().map(|e| e.as_ref()), step)).await;

        let packer = self.packer.clone();
        let mut sender = self.sender.take().expect("no send in flight");
        let task = tokio::spawn(async move {
            let pack_start = Instant::now();
            let grid = tokio::task::spawn_blocking(move || packer.pack(samples)).await??;
            let pack_time = pack_start.elapsed().as_secs_f64();
            let send_start = Instant::now();
            let send_batch_time = sender.send(grid).await?;
            // The send waits until every trainer rank requested the batch: the time the
            // orchestrator spent waiting on the trainer for the next policy.
            let wait_for_policy_time = (send_start.elapsed().as_secs_f64() - send_batch_time).max(0.0);
            Ok(Shipped { sender, pack_time, send_batch_time, wait_for_policy_time })
        });
        self.shipping = Some(Shipping { step, step_time, effective, rollout_metrics, task });
        Ok(())
    }

    /// Post-send half of `finalize_train_batch`: advance the step, checkpoint, log, drain.
    async fn finish_ship(&mut self, shipping: Shipping, shipped: Shipped, cancel: &CancellationToken) -> Result<()> {
        let Shipping { step, step_time, effective, rollout_metrics, .. } = shipping;
        let Shipped { sender, pack_time, send_batch_time, wait_for_policy_time } = shipped;
        self.sender = Some(sender);
        self.progress.step += 1;
        self.step_tx.send_replace(self.progress.step);
        if self.controller.is_some() {
            self.shipped_state = Some((step, self.snapshot().await?));
            self.apply_control()?;
        }
        let save_ckpt_time = self.maybe_save_ckpt(step).await?;

        let source_metrics = self.source.metrics().await?;
        let shipped = shipped_metrics(&effective, step, step_time, pack_time, save_ckpt_time, send_batch_time)?;
        let mut metrics = rollout_metrics;
        metrics.extend(shipped);
        metrics.extend(source_metrics);
        metrics.insert("time/wait_for_policy".into(), wait_for_policy_time.into());
        self.monitors.log_metrics(metrics, Some(step)).await;

        let active_step_time = (step_time - wait_for_policy_time).max(0.0);
        if step_time > 0.0 && wait_for_policy_time >= active_step_time {
            warn!(
                "Orchestrator waited {} for policy updates, at least as long as its {} active step time. \
                 Train-inference compute is imbalanced; add more trainer nodes.",
                format_time(wait_for_policy_time),
                format_time(active_step_time)
            );
        }
        self.beat();
        log_train_batch(&effective, step, step_time, self.config.train_env_names.len())?;

        if self.config.max_steps.is_some_and(|max| step >= max) {
            self.wait_for_version(step, "before shutdown", cancel).await?;
            // Drain right after shipping the final batch: a further batch can never train.
            self.start_draining("Shipped the final batch", cancel).await?;
        }
        Ok(())
    }

    /// Checkpoint the step just shipped on an interval boundary (the final step is saved at
    /// teardown). Returns the elapsed time.
    async fn maybe_save_ckpt(&mut self, step: u64) -> Result<f64> {
        let Some(interval) = self.config.ckpt_interval.filter(|i| *i > 0 && self.config.ckpt_enabled) else {
            return Ok(0.0);
        };
        if self.config.max_steps.is_some_and(|max| step >= max) || step % interval != 0 {
            return Ok(0.0);
        }
        info!("Saving checkpoint at step {step}");
        let t = Instant::now();
        let state = self.snapshot().await?;
        self.ckpt.save(&state, step)?;
        Ok(t.elapsed().as_secs_f64())
    }
}

/// Bounded wait until inference serves v{version} and its update hooks ran; warns and
/// proceeds on timeout.
pub async fn wait_for_version(
    applied: &mut watch::Receiver<u64>,
    version: u64,
    reason: &str,
    timeout: Duration,
    cancel: &CancellationToken,
) -> Result<()> {
    if *applied.borrow() >= version {
        return Ok(());
    }
    info!("Waiting for inference to apply policy v{version} {reason}");
    tokio::select! {
        _ = cancel.cancelled() => bail!("cancelled while waiting for policy v{version}"),
        result = tokio::time::timeout(timeout, applied.wait_for(|v| *v >= version)) => match result {
            Ok(changed) => { changed.map_err(|_| anyhow::anyhow!("policy watch closed"))?; }
            Err(_) => warn!(
                "Inference did not apply policy v{version} within {}s — proceeding anyway",
                timeout.as_secs_f64()
            ),
        },
    }
    Ok(())
}

fn shipped_metrics(
    effective: &TrainEpisodes,
    step: u64,
    step_time: f64,
    pack_time: f64,
    save_ckpt_time: f64,
    send_batch_time: f64,
) -> Result<Metrics> {
    let mut metrics = effective.to_wandb("train/agg", Subset::Effective);
    let by_env = effective.by_env();
    for (env_name, pool) in &by_env {
        metrics.extend(pool.to_wandb(&format!("train/{env_name}"), Subset::Effective));
    }
    let records = effective.records();
    let num_input: i64 = records.iter().map(|r| r.trace.num_input_tokens()).sum();
    let num_output: i64 = records.iter().map(|r| r.trace.num_output_tokens()).sum();
    metrics.insert("progress/input_tokens".into(), num_input.into());
    metrics.insert("progress/output_tokens".into(), num_output.into());
    metrics.insert("time/step".into(), step_time.into());
    metrics.insert("time/pack".into(), pack_time.into());
    metrics.insert("time/save_ckpt".into(), save_ckpt_time.into());
    metrics.insert("time/send_batch".into(), send_batch_time.into());
    metrics.insert("step".into(), step.into());
    let mut staleness = Vec::new();
    for episode in effective.selected_episodes() {
        if let Some(s) = rollout_staleness(&episode.episode, episode.traces(), step as i64)? {
            staleness.push(s);
        }
    }
    if !staleness.is_empty() {
        let n = staleness.len() as f64;
        let mean = |f: fn(&Staleness) -> i64| staleness.iter().map(f).sum::<i64>() as f64 / n;
        let max = |f: fn(&Staleness) -> i64| staleness.iter().map(f).max().unwrap_or(0) as f64;
        metrics.insert("staleness/rollout/mean".into(), mean(|s| s.rollout).into());
        metrics.insert("staleness/rollout/max".into(), max(|s| s.rollout).into());
        metrics.insert("staleness/rollout/in_flight/mean".into(), mean(|s| s.in_flight).into());
        metrics.insert("staleness/rollout/in_flight/max".into(), max(|s| s.in_flight).into());
        metrics.insert("staleness/rollout/in_queue/mean".into(), mean(|s| s.in_queue).into());
        metrics.insert("staleness/rollout/in_queue/max".into(), max(|s| s.in_queue).into());
        let tokens: i64 = staleness.iter().map(|s| s.token_count).sum();
        if tokens > 0 {
            let sum: i64 = staleness.iter().map(|s| s.token_sum).sum();
            metrics.insert("staleness/token/mean".into(), (sum as f64 / tokens as f64).into());
        }
    }
    let total = effective.num_traces() as f64;
    for (env_name, pool) in &by_env {
        metrics.insert(format!("batch/{env_name}"), (pool.num_traces() as f64 / total).into());
    }
    Ok(metrics)
}

fn max_staleness(pool: &TrainEpisodes, step: u64) -> Result<i64> {
    let mut max = 0;
    for episode in pool.selected_episodes() {
        if let Some(s) = rollout_staleness(&episode.episode, episode.traces(), step as i64)? {
            max = max.max(s.rollout);
        }
    }
    Ok(max)
}

/// Quality and env share of exactly the traces shipped this step.
fn log_train_batch(effective: &TrainEpisodes, step: u64, step_time: f64, num_train_envs: usize) -> Result<()> {
    let n_effective = effective.num_traces();
    let n_trainable = effective.records().iter().filter(|r| is_trainable(r.trace)).count();
    let (reward, turns, branches, truncated, timed_out) = effective.with_metrics(|m| {
        (m.reward().mean(), m.num_turns().mean(), m.num_branches().mean(), m.is_truncated().mean(), m.is_timeout().mean())
    });
    let trainable_rate = if n_effective > 0 { n_trainable as f64 / n_effective as f64 } else { 0.0 };
    let head = format!(
        "Step {step} | {:>7} | Reward {:.4} | Trainable {n_trainable}/{n_effective} ({}) | Turns {:.1} | Branches {:.1} | \
         Max Staleness {} | Truncation {} | Timeout {}",
        format_time(step_time),
        reward,
        pct(trainable_rate),
        turns,
        branches,
        max_staleness(effective, step)?,
        pct(truncated),
        pct(timed_out),
    );
    if num_train_envs <= 1 {
        prl_io::success!("{head}");
        return Ok(());
    }
    let by_env = effective.by_env();
    let mut names: Vec<&String> = by_env.keys().collect();
    names.sort();
    let width = names.iter().map(|n| n.chars().count()).max().unwrap_or(0);
    let mut lines = vec![head];
    for name in names {
        let pool = &by_env[name];
        let (reward, turns, branches, truncated, timed_out) = pool.with_metrics(|m| {
            (m.reward().mean(), m.num_turns().mean(), m.num_branches().mean(), m.is_truncated().mean(), m.is_timeout().mean())
        });
        let ratio = if n_effective > 0 { pool.num_traces() as f64 / n_effective as f64 } else { 0.0 };
        lines.push(format!(
            "╰─ {name:<width$} | Ratio {} | Reward {:.4} | Turns {:.1} | Branches {:.1} | Max Staleness {} | \
             Truncation {} | Timeout {}",
            pct(ratio),
            reward,
            turns,
            branches,
            max_staleness(pool, step)?,
            pct(truncated),
            pct(timed_out),
        ));
    }
    prl_io::success!("{}", lines.join("\n\t\t "));
    Ok(())
}
