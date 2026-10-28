//! Discovers new policy versions, applies them to inference, and notifies observers.
//! Port of `watcher.py` as an actor: the single writer of the `policy` watch.

use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use indexmap::IndexMap;
use prl_types::{Metrics, Policy};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use prl_io::logging::format_time;

pub const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The weight transport as the watcher drives it (implemented over prl-io's receiver).
#[async_trait]
pub trait PolicyReceiver: Send + 'static {
    /// Newest version offered beyond `current`; `current` if none.
    async fn next_version(&mut self, current: u64) -> Result<u64>;
    /// Block until the trainer offers v{step}; returns early with an error when cancelled.
    async fn wait_published(&mut self, step: u64, cancel: &CancellationToken) -> Result<()>;
    /// Apply v{step} to inference; returns per-phase engine times (`pause`/`transfer`/`resume`).
    async fn receive(&mut self, step: u64) -> Result<IndexMap<String, Vec<f64>>>;
    /// Apply the trainer's startup broadcast.
    async fn sync_startup(&mut self, step: u64, timeout: Duration) -> Result<()>;
}

/// Notified around each policy update (`VersionObserver`). Errors are logged, not fatal.
#[async_trait]
pub trait VersionObserver: Send + Sync {
    fn name(&self) -> &str;
    /// Before inference is paused for the update.
    async fn on_version_pending(&self, step: u64) -> Result<()>;
    /// After the new weights are live and `policy` advanced.
    async fn on_new_version(&self, step: u64) -> Result<()>;
}

/// Runs after every applied policy (including the startup sync), in registration order.
/// Errors are fatal to the watcher (and so to the run), as in Python.
#[async_trait]
pub trait UpdateHook: Send + Sync {
    async fn on_update(&self, step: u64) -> Result<()>;
}

pub struct WeightWatcher<R: PolicyReceiver> {
    receiver: R,
    policy: watch::Sender<Policy>,
    observers: Vec<Box<dyn VersionObserver>>,
    hooks: Vec<Box<dyn UpdateHook>>,
    ckpt_step: u64,
    poll_interval: Duration,
    last_update_weights_time: f64,
    last_wait_for_ckpt_time: f64,
    update_count: u64,
    update_times: IndexMap<String, Vec<f64>>,
    gauges: watch::Sender<Metrics>,
    /// The last version whose update notifications (observers and hooks) all ran.
    applied: watch::Sender<u64>,
}

impl<R: PolicyReceiver> WeightWatcher<R> {
    pub fn new(
        receiver: R,
        policy: watch::Sender<Policy>,
        applied: watch::Sender<u64>,
        observers: Vec<Box<dyn VersionObserver>>,
        hooks: Vec<Box<dyn UpdateHook>>,
        ckpt_step: u64,
    ) -> (Self, watch::Receiver<Metrics>) {
        let (gauges, gauges_rx) = watch::channel(Metrics::new());
        let this = Self {
            receiver,
            policy,
            observers,
            hooks,
            ckpt_step,
            poll_interval: POLL_INTERVAL,
            last_update_weights_time: 0.0,
            last_wait_for_ckpt_time: 0.0,
            update_count: 0,
            update_times: IndexMap::new(),
            gauges,
            applied,
        };
        this.publish_gauges();
        (this, gauges_rx)
    }

    pub fn with_poll_interval(mut self, poll_interval: Duration) -> Self {
        self.poll_interval = poll_interval;
        self
    }

    /// Apply the startup policy and run the update notifications.
    pub async fn sync_startup(&mut self, step: u64, timeout: Duration) -> Result<()> {
        self.receiver.sync_startup(step, timeout).await?;
        self.ckpt_step = step;
        self.policy.send_modify(|p| p.version = step);
        self.publish_gauges();
        self.notify_update(step).await
    }

    /// Poll until cancelled. An update in flight finishes before the loop observes the
    /// cancellation (the trainer blocks inside its broadcast until the apply completes);
    /// only the wait for the trainer's offer is interruptible.
    pub async fn run(mut self, cancel: CancellationToken) -> Result<()> {
        loop {
            let next = self.receiver.next_version(self.ckpt_step).await?;
            if next > self.ckpt_step {
                if let Err(err) = self.apply_policy_update(next, &cancel).await {
                    if cancel.is_cancelled() {
                        return Ok(());
                    }
                    return Err(err);
                }
            }
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                _ = tokio::time::sleep(self.poll_interval) => {}
            }
        }
    }

    async fn apply_policy_update(&mut self, next: u64, cancel: &CancellationToken) -> Result<()> {
        let t0 = Instant::now();
        self.receiver.wait_published(next, cancel).await?;
        self.last_wait_for_ckpt_time = t0.elapsed().as_secs_f64();
        self.ckpt_step = next;

        // Observers cancel rollouts BEFORE the engines pause (aborts must settle while the
        // engines are stepping; see the Python watcher for the NIXL story).
        for observer in &self.observers {
            if let Err(err) = observer.on_version_pending(next).await {
                warn!("Observer {}.on_version_pending({next}) raised: {err:?}", observer.name());
            }
        }

        debug!("Updating inference weights to policy v{next}");
        let t1 = Instant::now();
        self.update_times = self.receiver.receive(next).await?;
        self.last_update_weights_time = t1.elapsed().as_secs_f64();
        self.update_count += 1;
        self.policy.send_modify(|p| p.version = next);
        debug!("Updated inference weights to policy v{next} in {}", format_time(self.last_update_weights_time));
        self.publish_gauges();
        self.notify_update(next).await
    }

    async fn notify_update(&self, step: u64) -> Result<()> {
        for observer in &self.observers {
            if let Err(err) = observer.on_new_version(step).await {
                warn!("Observer {}.on_new_version({step}) raised: {err:?}", observer.name());
            }
        }
        for hook in &self.hooks {
            hook.on_update(step).await?;
        }
        self.applied.send_replace(step);
        Ok(())
    }

    fn publish_gauges(&self) {
        let mut m = Metrics::new();
        m.insert("watcher/policy_version".into(), (self.policy.borrow().version as f64).into());
        m.insert("watcher/update_count".into(), (self.update_count as f64).into());
        m.insert("watcher/last_update_weights_time".into(), self.last_update_weights_time.into());
        m.insert("watcher/last_wait_for_ckpt_time".into(), self.last_wait_for_ckpt_time.into());
        for (phase, times) in &self.update_times {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct FakeReceiver {
        published: Arc<Mutex<u64>>,
        log: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl PolicyReceiver for FakeReceiver {
        async fn next_version(&mut self, current: u64) -> Result<u64> {
            Ok((*self.published.lock().unwrap()).max(current))
        }
        async fn wait_published(&mut self, _step: u64, _cancel: &CancellationToken) -> Result<()> {
            Ok(())
        }
        async fn receive(&mut self, step: u64) -> Result<IndexMap<String, Vec<f64>>> {
            self.log.lock().unwrap().push(format!("receive {step}"));
            Ok([("pause".to_string(), vec![1.0, 3.0])].into_iter().collect())
        }
        async fn sync_startup(&mut self, step: u64, _timeout: Duration) -> Result<()> {
            self.log.lock().unwrap().push(format!("sync {step}"));
            Ok(())
        }
    }

    struct Obs(Arc<Mutex<Vec<String>>>);

    #[async_trait]
    impl VersionObserver for Obs {
        fn name(&self) -> &str {
            "Obs"
        }
        async fn on_version_pending(&self, step: u64) -> Result<()> {
            self.0.lock().unwrap().push(format!("pending {step}"));
            anyhow::bail!("ignored")
        }
        async fn on_new_version(&self, step: u64) -> Result<()> {
            self.0.lock().unwrap().push(format!("new {step}"));
            Ok(())
        }
    }

    struct Hook(Arc<Mutex<Vec<String>>>);

    #[async_trait]
    impl UpdateHook for Hook {
        async fn on_update(&self, step: u64) -> Result<()> {
            self.0.lock().unwrap().push(format!("hook {step}"));
            anyhow::ensure!(step < 5, "hook failed");
            Ok(())
        }
    }

    #[tokio::test]
    async fn applies_newest_version_in_observer_then_hook_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let published = Arc::new(Mutex::new(0));
        let (policy_tx, mut policy_rx) = watch::channel(Policy::default());
        let (applied_tx, _applied) = watch::channel(0);
        let (mut w, gauges) = WeightWatcher::new(
            FakeReceiver { published: published.clone(), log: log.clone() },
            policy_tx,
            applied_tx,
            vec![Box::new(Obs(log.clone()))],
            vec![Box::new(Hook(log.clone()))],
            0,
        );
        w.sync_startup(0, Duration::from_secs(1)).await.unwrap();
        let cancel = CancellationToken::new();
        let task = tokio::spawn(w.with_poll_interval(Duration::from_millis(5)).run(cancel.clone()));
        *published.lock().unwrap() = 3;
        policy_rx.wait_for(|p| p.version == 3).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            *log.lock().unwrap(),
            vec!["sync 0", "new 0", "hook 0", "pending 3", "receive 3", "new 3", "hook 3"]
        );
        assert_eq!(gauges.borrow()["watcher/pause/mean"], 2.0.into());
        // a failing hook kills the watcher
        *published.lock().unwrap() = 7;
        assert!(task.await.unwrap().is_err());
    }
}
