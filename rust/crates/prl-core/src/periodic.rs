//! The pipeline-view logger (`periodic_logger.py`): every `interval` seconds (and once more on
//! stop) collect `(console body, payload)` in one call, log the body and send the payload to
//! the monitors as a time-keyed row.

use std::future::Future;
use std::time::Duration;

use prl_monitors::Monitors;
use prl_types::Metrics;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::info;

pub struct PeriodicLogger {
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl PeriodicLogger {
    pub fn spawn<F, Fut>(interval: Duration, monitors: Monitors, mut collect: F) -> Self
    where
        F: FnMut() -> Fut + Send + 'static,
        Fut: Future<Output = (String, Metrics)> + Send,
    {
        let (stop, mut stopped) = oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            loop {
                let stopping = tokio::select! {
                    _ = &mut stopped => true,
                    _ = tokio::time::sleep(interval) => false,
                };
                let (body, payload) = collect().await;
                info!("{body}");
                if !payload.is_empty() {
                    monitors.log_metrics(payload, None).await;
                }
                if stopping {
                    return;
                }
            }
        });
        Self { stop: Some(stop), task }
    }

    /// Emit a final tick and wait for it.
    pub async fn stop(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let _ = self.task.await;
    }
}
