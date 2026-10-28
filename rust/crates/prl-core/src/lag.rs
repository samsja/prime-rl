//! Runtime scheduling lag: how late a 100 ms sleep wakes (`EventLoopLagMonitor`).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use prl_types::Metrics;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub const MEASURE_INTERVAL: Duration = Duration::from_millis(100);
pub const MAX_MEASUREMENTS: usize = 1000;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LagStats {
    pub min: f64,
    pub mean: f64,
    pub median: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
    pub n: usize,
}

impl LagStats {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        Self {
            min: sorted[0],
            mean: samples.iter().sum::<f64>() / samples.len() as f64,
            median: percentile(&sorted, 50.0),
            p90: percentile(&sorted, 90.0),
            p99: percentile(&sorted, 99.0),
            max: sorted[sorted.len() - 1],
            n: samples.len(),
        }
    }

    /// `event_loop_lag/*` keys of the pipeline view (empty when nothing was measured).
    pub fn metrics(&self) -> Metrics {
        let mut m = Metrics::new();
        if self.n == 0 {
            return m;
        }
        for (key, value) in [
            ("min", self.min),
            ("mean", self.mean),
            ("median", self.median),
            ("p90", self.p90),
            ("p99", self.p99),
            ("max", self.max),
            ("n", self.n as f64),
        ] {
            m.insert(format!("event_loop_lag/{key}"), value.into());
        }
        m
    }
}

/// numpy's default (linear) percentile over sorted samples.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    let pos = q / 100.0 * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

#[derive(Clone)]
pub struct LagHandle {
    tx: mpsc::Sender<oneshot::Sender<LagStats>>,
}

impl LagHandle {
    pub async fn stats(&self) -> LagStats {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(tx).await.is_err() {
            return LagStats::default();
        }
        rx.await.unwrap_or_default()
    }
}

/// Spawn the lag monitor; it stops on `cancel`.
pub fn spawn(cancel: CancellationToken) -> (LagHandle, tokio::task::JoinHandle<()>) {
    let (tx, mut rx) = mpsc::channel::<oneshot::Sender<LagStats>>(8);
    let task = tokio::spawn(async move {
        let mut lags: VecDeque<f64> = VecDeque::with_capacity(MAX_MEASUREMENTS);
        let mut deadline = Instant::now() + MEASURE_INTERVAL;
        let mut sleep = Box::pin(tokio::time::sleep(MEASURE_INTERVAL));
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                Some(reply) = rx.recv() => {
                    let samples: Vec<f64> = lags.iter().copied().collect();
                    let _ = reply.send(LagStats::from_samples(&samples));
                }
                _ = &mut sleep => {
                    let lag = Instant::now().saturating_duration_since(deadline).as_secs_f64();
                    if lags.len() == MAX_MEASUREMENTS {
                        lags.pop_front();
                    }
                    lags.push_back(lag);
                    deadline = Instant::now() + MEASURE_INTERVAL;
                    sleep.as_mut().reset(tokio::time::Instant::now() + MEASURE_INTERVAL);
                }
            }
        }
    });
    (LagHandle { tx }, task)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_match_numpy() {
        // np.percentile([0.1, 0.4, 0.2, 0.3], [50, 90, 99]) = 0.25, 0.37, 0.397
        let s = LagStats::from_samples(&[0.1, 0.4, 0.2, 0.3]);
        assert!((s.median - 0.25).abs() < 1e-12);
        assert!((s.p90 - 0.37).abs() < 1e-12);
        assert!((s.p99 - 0.397).abs() < 1e-12);
        assert_eq!((s.min, s.max, s.n), (0.1, 0.4, 4));
        assert!(LagStats::from_samples(&[]).metrics().is_empty());
    }
}
