//! Online evals (port of `eval/online.py`): evaluate the trainer's weight broadcasts as they appear.

use std::collections::{BTreeSet, HashSet};

/// What one broadcasts-dir scan does with a step, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalkAction {
    /// An eval-due step that broadcast cleaning removed before it was evaluated.
    SkipDeleted(u64),
    /// An unpublished step below a published one: an abandoned partial write.
    SkipAbandoned(u64),
    /// Receive the broadcast and run the due evals (`force` on the final step).
    Evaluate { step: u64, force: bool },
}

/// Eval-due steps up to the newest published broadcast that are missing from the broadcasts dir.
pub fn deleted_due_steps(steps: &[u64], newest_published: Option<u64>, intervals: &[u64]) -> BTreeSet<u64> {
    let Some(newest) = newest_published else { return BTreeSet::new() };
    let present: HashSet<u64> = steps.iter().copied().collect();
    intervals
        .iter()
        .flat_map(|&interval| (interval..=newest).step_by(interval as usize))
        .filter(|step| !present.contains(step))
        .collect()
}

/// One scan of the broadcasts dir: `steps` are the `step_*` dirs, `published` the ones with a
/// `.sender_ready` marker. Steps at or below `last_step` were already handled.
pub fn plan_walk(
    steps: &[u64],
    published: &HashSet<u64>,
    intervals: &[u64],
    last_step: u64,
    max_steps: Option<u64>,
) -> Vec<WalkAction> {
    let newest_published = steps.iter().copied().filter(|step| published.contains(step)).max();
    let present: HashSet<u64> = steps.iter().copied().collect();
    let mut walk: BTreeSet<u64> = steps.iter().copied().collect();
    walk.extend(deleted_due_steps(steps, newest_published, intervals));
    let mut actions = Vec::new();
    for step in walk {
        if step <= last_step {
            continue;
        }
        if !present.contains(&step) {
            actions.push(WalkAction::SkipDeleted(step));
            continue;
        }
        if !published.contains(&step) {
            if newest_published.is_none_or(|newest| newest < step) {
                break; // still being written — later steps can't be ready either
            }
            actions.push(WalkAction::SkipAbandoned(step));
            continue;
        }
        actions.push(WalkAction::Evaluate { step, force: max_steps.is_some_and(|max| step >= max) });
    }
    actions
}


// ── driver ──────────────────────────────────────────────────────────────────

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use prl_config::{FileSystemWeightBroadcastConfig, SFTOnlineEvalConfig, WeightBroadcastConfig};
use prl_io::pathing::all_ckpt_steps;
use prl_io::weights::{NixlBridge, WeightReceiver};
use prl_monitors::prime::eval::EvalSourceSpec;
use prl_monitors::{Monitors, WandbBridge};
use prl_types::Bridge;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::runner::{EvalRunner, Schedule, POLL_INTERVAL_S};

/// Budget for the trainer's startup broadcast: it is always coming, but only after the trainer
/// has finished loading the model.
pub const STARTUP_BROADCAST_TIMEOUT: Duration = Duration::from_secs(1200);

fn py_opt<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "None".to_owned(), |v| v.to_string())
}

/// Python `str()` of the weight broadcast config, as the init line prints it.
fn broadcast_repr(config: &WeightBroadcastConfig) -> String {
    match config {
        WeightBroadcastConfig::Filesystem(c) => format!("timeout={} type='filesystem'", c.timeout),
        WeightBroadcastConfig::Nccl(c) => format!(
            "timeout={} host='{}' port={} inference_world_size={} type='nccl'",
            c.timeout, c.host, c.port, c.inference_world_size
        ),
        other => format!("type='{}'", other.tag()),
    }
}

/// Spawned by the `sft` launcher next to the trainer: watches the broadcasts dir, moves the
/// inference server onto each offered broadcast and runs the due eval sources against it,
/// sequentially per broadcast so every epoch measures exactly one policy version.
pub async fn run_online_eval(
    config: SFTOnlineEvalConfig,
    bridge: Arc<dyn Bridge>,
    wandb: Option<Box<dyn WandbBridge>>,
    nixl: Option<Box<dyn NixlBridge>>,
) -> Result<()> {
    let model = config.model.clone().context("the sft launcher fills eval.model")?;
    let cancel = CancellationToken::new();
    let sources = config
        .served
        .source
        .iter()
        .map(|s| EvalSourceSpec { resolved_name: s.resolved_name.clone(), taskset_id: s.env_id.clone(), group_size: s.group_size })
        .collect();
    let (monitors, monitors_task) = crate::eval::setup_monitors(
        &config.monitors,
        &config.output_dir,
        "online-eval",
        "sft",
        &model,
        sources,
        wandb,
        cancel.child_token(),
    )
    .await?;
    let result = run(&config, &model, monitors.clone(), bridge, nixl).await;
    if result.is_ok() {
        // Finalize only on a clean exit — a crashed run must not mark the run completed.
        monitors.finalize(false).await;
    }
    drop(monitors);
    cancel.cancel();
    let _ = monitors_task.await;
    result
}

async fn run(
    config: &SFTOnlineEvalConfig,
    model: &str,
    monitors: Monitors,
    bridge: Arc<dyn Bridge>,
    nixl: Option<Box<dyn NixlBridge>>,
) -> Result<()> {
    let schedule = Schedule {
        intervals: config.intervals.clone(),
        skip_first_step: config.skip_first_step,
        is_resumed: config.resume_step.is_some(),
    };
    let mut runner =
        EvalRunner::setup(&config.served, model, &config.output_dir, config.log.interval, Some(schedule), monitors, bridge)
            .await?;
    let result = async {
        let broadcasts_dir = config.broadcasts_dir.clone().context("broadcasts_dir is resolved by the config")?;
        // A hand-written config may omit the transport; broadcasts are then plain filesystem checkpoints.
        let weight_broadcast = config
            .weight_broadcast
            .clone()
            .unwrap_or(WeightBroadcastConfig::Filesystem(FileSystemWeightBroadcastConfig { timeout: 3600 }));
        info!("Initializing weight broadcast ({})", broadcast_repr(&weight_broadcast));
        let admin = runner.admin_plane.take().expect("admin plane is set up");
        let mut receiver = WeightReceiver::new(broadcasts_dir.clone(), weight_broadcast, admin, model, nixl)?;
        receiver.initialize().await?;
        runner.start()?;
        let mut online = OnlineEval { config, runner: &mut runner, receiver, broadcasts_dir, last_step: config.resume_step.unwrap_or(0) };
        online.watch().await?;
        runner.drain().await;
        prl_io::success!("Online evals finished!");
        Ok(())
    }
    .await;
    runner.stop().await;
    result
}

struct OnlineEval<'a> {
    config: &'a SFTOnlineEvalConfig,
    runner: &'a mut EvalRunner,
    receiver: WeightReceiver,
    broadcasts_dir: PathBuf,
    /// The last weight-broadcast step already handled (evaluated or skipped).
    last_step: u64,
}

impl OnlineEval<'_> {
    fn is_final(&self, step: u64) -> bool {
        self.config.max_steps.is_some_and(|max| step >= max)
    }

    /// Evaluate each eligible weight broadcast as it appears.
    async fn watch(&mut self) -> Result<()> {
        let config = self.config;
        // Rendezvous with the trainer's startup broadcast (v0 fresh, the checkpoint step on
        // resume), for every transport.
        let startup = config.resume_step.unwrap_or(0);
        self.receiver.sync_startup(startup, STARTUP_BROADCAST_TIMEOUT).await?;
        self.runner.policy.send_modify(|p| p.version = startup);

        match config.resume_step {
            None => self.maybe_run_eval(0, false, false).await?,
            Some(step) if config.retrigger_on_resume => self.maybe_run_eval(step, false, self.is_final(step)).await?,
            Some(_) => {}
        }

        info!(
            "Watching {} for new weight broadcasts (max_steps={})",
            self.broadcasts_dir.display(),
            py_opt(config.max_steps)
        );
        let intervals: Vec<u64> = config.intervals.values().copied().collect();
        loop {
            let steps = all_ckpt_steps(&self.broadcasts_dir)?;
            let published = steps.iter().copied().filter(|s| self.receiver.is_published(*s)).collect();
            for action in plan_walk(&steps, &published, &intervals, self.last_step, config.max_steps) {
                match action {
                    WalkAction::SkipDeleted(step) => {
                        warn!(
                            "Weight broadcast for eval step {step} was deleted before it could be evaluated \
                             (broadcast cleaning outpaced the evals process) - skipping its evals"
                        );
                        self.last_step = self.last_step.max(step);
                    }
                    WalkAction::SkipAbandoned(step) => {
                        warn!(
                            "Weight broadcast step {step} is not marked published but newer broadcasts are - \
                             treating it as abandoned and skipping its evals"
                        );
                        self.last_step = self.last_step.max(step);
                    }
                    WalkAction::Evaluate { step, force } => self.maybe_run_eval(step, true, force).await?,
                }
            }
            if config.max_steps.is_some_and(|max| self.last_step >= max) {
                return Ok(());
            }
            self.runner.check_components()?;
            tokio::time::sleep(Duration::from_secs(POLL_INTERVAL_S)).await;
        }
    }

    /// Fire eligible envs for one checkpoint step and run the full epoch(s), reloading the
    /// inference weights first. A broadcast offered by the trainer is always received (the
    /// trainer blocks inside the handshake), eval or no eval.
    async fn maybe_run_eval(&mut self, step: u64, reload: bool, force: bool) -> Result<()> {
        let broadcast_dir = self.receiver.step_dir(step);
        if reload && !self.receiver.is_published(step) {
            warn!("No published weight broadcast for step {step} ({}) - skipping eval", broadcast_dir.display());
            self.last_step = self.last_step.max(step);
            return Ok(());
        }
        // Block scheduling before the trigger: nothing of this epoch may run on the old weights.
        if reload {
            self.runner.dispatcher.version_pending().await?;
        }
        let outcome = self.runner.trigger(step, force, Vec::new()).await?;
        self.last_step = self.last_step.max(step);
        if reload {
            info!("Updating inference weights to broadcast step {step} ({})", broadcast_dir.display());
            self.receiver.receive(step).await?;
            self.runner.policy.send_modify(|p| p.version = step);
            self.runner.dispatcher.version_applied().await?;
        } else {
            self.runner.policy.send_modify(|p| p.version = step);
        }
        if outcome.fired.is_empty() {
            return Ok(());
        }
        let receiver = &self.receiver;
        let broadcasts_dir = &self.broadcasts_dir;
        let mut next_published = move || -> Option<u64> {
            all_ckpt_steps(broadcasts_dir).ok()?.into_iter().find(|&c| c > step && receiver.is_published(c))
        };
        let superseding: Option<&mut (dyn FnMut() -> Option<u64> + Send)> =
            if self.config.cancel_on_new_checkpoint { Some(&mut next_published) } else { None };
        self.runner.run_epoch(outcome, step, superseding).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use WalkAction::*;

    fn set(steps: &[u64]) -> HashSet<u64> {
        steps.iter().copied().collect()
    }

    #[test]
    fn evaluates_published_steps_in_order_above_last_step() {
        let actions = plan_walk(&[1, 2, 3, 4], &set(&[1, 2, 3, 4]), &[2], 2, Some(10));
        assert_eq!(actions, [Evaluate { step: 3, force: false }, Evaluate { step: 4, force: false }]);
    }

    #[test]
    fn stops_at_a_step_still_being_written() {
        let actions = plan_walk(&[1, 2, 3], &set(&[1]), &[1], 0, None);
        assert_eq!(actions, [Evaluate { step: 1, force: false }]);
    }

    #[test]
    fn skips_abandoned_steps_below_a_published_one() {
        let actions = plan_walk(&[1, 2, 3], &set(&[1, 3]), &[5], 0, None);
        assert_eq!(actions, [Evaluate { step: 1, force: false }, SkipAbandoned(2), Evaluate { step: 3, force: false }]);
    }

    #[test]
    fn reports_due_steps_removed_by_broadcast_cleaning() {
        // trainer keeps only the newest broadcast: steps 2 and 4 (due every 2) vanished
        let actions = plan_walk(&[5], &set(&[5]), &[2], 0, None);
        assert_eq!(actions, [SkipDeleted(2), SkipDeleted(4), Evaluate { step: 5, force: false }]);
        assert_eq!(deleted_due_steps(&[5], None, &[2]), BTreeSet::new());
    }

    #[test]
    fn final_step_forces_every_env() {
        let actions = plan_walk(&[9, 10], &set(&[9, 10]), &[100], 8, Some(10));
        assert_eq!(actions, [Evaluate { step: 9, force: false }, Evaluate { step: 10, force: true }]);
    }
}
