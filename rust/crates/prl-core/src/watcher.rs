//! Discovers new policy versions, applies them to inference, and notifies the pipeline.
//! Port of `watcher.py` as an actor: the single writer of the `policy` and `applied` watches.

use std::time::{Duration, Instant};

use anyhow::Result;
use prl_io::logging::format_time;
use prl_io::weights::WeightReceiver;
use prl_types::{Metrics, Policy};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::dispatcher::DispatcherHandle;
use crate::eval_pipeline::EvalPipelineHandle;

const POLL_INTERVAL: Duration = Duration::from_millis(100);

pub struct WeightWatcher {
    pub receiver: WeightReceiver,
    pub policy: watch::Sender<Policy>,
    /// The last version whose update notifications (eval trigger, dispatcher resume) ran.
    pub applied: watch::Sender<u64>,
    pub gauges: watch::Sender<Metrics>,
    pub dispatcher: DispatcherHandle,
    pub eval: Option<EvalPipelineHandle>,
    pub ckpt_step: u64,
    pub last_update_weights_time: f64,
    pub last_wait_for_ckpt_time: f64,
    pub update_count: u64,
}

impl WeightWatcher {
    /// Apply the startup policy and run the update notifications.
    pub async fn sync_startup(&mut self, step: u64, timeout: Duration) -> Result<()> {
        self.receiver.sync_startup(step, timeout).await?;
        self.ckpt_step = step;
        self.policy.send_modify(|p| p.version = step);
        self.notify_update(step).await
    }

    /// Poll until cancelled. An update in flight finishes before the cancellation is observed
    /// (the trainer blocks inside its broadcast until the apply completes); only the wait for
    /// the trainer's offer is interruptible.
    pub async fn run(mut self, cancel: CancellationToken) -> Result<()> {
        loop {
            let next = self.receiver.next_version(self.ckpt_step)?;
            if next > self.ckpt_step {
                match self.apply_policy_update(next, &cancel).await {
                    Err(_) if cancel.is_cancelled() => return Ok(()),
                    result => result?,
                }
            }
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                _ = tokio::time::sleep(POLL_INTERVAL) => {}
            }
        }
    }

    async fn apply_policy_update(&mut self, next: u64, cancel: &CancellationToken) -> Result<()> {
        let t0 = Instant::now();
        self.receiver.wait_published(next, cancel).await?;
        self.last_wait_for_ckpt_time = t0.elapsed().as_secs_f64();
        self.ckpt_step = next;
        // Pause scheduling BEFORE the engines pause: aborts must settle while they step.
        self.dispatcher.version_pending().await?;
        debug!("Updating inference weights to policy v{next}");
        let t1 = Instant::now();
        self.receiver.receive(next).await?;
        self.last_update_weights_time = t1.elapsed().as_secs_f64();
        self.update_count += 1;
        self.policy.send_modify(|p| p.version = next);
        debug!("Updated inference weights to policy v{next} in {}", format_time(self.last_update_weights_time));
        self.notify_update(next).await
    }

    /// The eval trigger runs before the dispatcher resumes, so nothing is scheduled on the new
    /// policy before an eval trigger switched the dispatcher to PREFER_EVAL.
    async fn notify_update(&mut self, step: u64) -> Result<()> {
        self.publish_gauges();
        if let Some(eval) = &self.eval {
            eval.policy_applied(step).await?;
        }
        self.dispatcher.version_applied().await?;
        self.applied.send_replace(step);
        Ok(())
    }

    fn publish_gauges(&self) {
        let mut m = Metrics::new();
        m.insert("watcher/policy_version".into(), (self.policy.borrow().version as f64).into());
        m.insert("watcher/update_count".into(), (self.update_count as f64).into());
        m.insert("watcher/last_update_weights_time".into(), self.last_update_weights_time.into());
        m.insert("watcher/last_wait_for_ckpt_time".into(), self.last_wait_for_ckpt_time.into());
        for (phase, times) in self.receiver.update_times().phases() {
            if times.is_empty() {
                continue;
            }
            let min = times.iter().copied().fold(f64::INFINITY, f64::min);
            let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            m.insert(format!("watcher/{phase}/min"), min.into());
            m.insert(format!("watcher/{phase}/mean"), mean.into());
            m.insert(format!("watcher/{phase}/max"), max.into());
        }
        self.gauges.send_replace(m);
    }
}
