//! Runtime scheduling lag: how late a 100 ms sleep wakes (`EventLoopLagMonitor`).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use prl_train::Stat;
use prl_types::Metrics;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

const MEASURE_INTERVAL: Duration = Duration::from_millis(100);
const MAX_MEASUREMENTS: usize = 1000;

/// `event_loop_lag/*` of the pipeline view (numpy-style percentiles); empty before any sample.
fn lag_metrics(samples: Vec<f64>) -> Metrics {
    let mut m = Metrics::new();
    if samples.is_empty() {
        return m;
    }
    let n = samples.len() as f64;
    let stat = Stat::new(samples);
    for (key, value) in [
        ("min", stat.min()),
        ("mean", stat.mean()),
        ("median", stat.percentile(50.0)),
        ("p90", stat.percentile(90.0)),
        ("p99", stat.percentile(99.0)),
        ("max", stat.max()),
        ("n", n),
    ] {
        m.insert(format!("event_loop_lag/{key}"), value.into());
    }
    m
}

#[derive(Clone)]
pub struct LagHandle {
    tx: mpsc::Sender<oneshot::Sender<Metrics>>,
}

impl LagHandle {
    pub async fn metrics(&self) -> Metrics {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(tx).await.is_err() {
            return Metrics::new();
        }
        rx.await.unwrap_or_default()
    }
}

/// Spawn the lag monitor; it stops on `cancel`.
pub fn spawn(cancel: CancellationToken) -> LagHandle {
    let (tx, mut rx) = mpsc::channel::<oneshot::Sender<Metrics>>(8);
    tokio::spawn(async move {
        let mut lags: VecDeque<f64> = VecDeque::with_capacity(MAX_MEASUREMENTS);
        loop {
            let deadline = Instant::now() + MEASURE_INTERVAL;
            let sleep = tokio::time::sleep(MEASURE_INTERVAL);
            tokio::pin!(sleep);
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    Some(reply) = rx.recv() => {
                        let _ = reply.send(lag_metrics(lags.iter().copied().collect()));
                        continue;
                    }
                    _ = &mut sleep => {}
                }
                break;
            }
            if lags.len() == MAX_MEASUREMENTS {
                lags.pop_front();
            }
            lags.push_back(Instant::now().saturating_duration_since(deadline).as_secs_f64());
        }
    });
    LagHandle { tx }
}
