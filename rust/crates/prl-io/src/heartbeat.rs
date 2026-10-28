//! Better Stack heartbeat (`utils/heartbeat.py`).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Default)]
struct BeatState {
    pending: bool,
    last_beat_at: Option<Instant>,
}

/// `beat()` is called on every progress event; a ping is only sent when the last one is at least
/// `min_interval` old and none is in flight. Surplus beats are dropped without a request.
#[derive(Debug, Clone)]
pub struct Heartbeat {
    url: String,
    min_interval: Duration,
    client: reqwest::Client,
    state: Arc<Mutex<BeatState>>,
}

impl Heartbeat {
    pub fn new(url: impl Into<String>, min_interval_s: f64) -> Self {
        Self {
            url: url.into(),
            min_interval: Duration::from_secs_f64(min_interval_s),
            client: reqwest::Client::new(),
            state: Arc::default(),
        }
    }

    /// Returns immediately; the ping (1 s timeout) runs on a background task. Must be called from
    /// within a tokio runtime.
    pub fn beat(&self) {
        {
            let mut state = self.state.lock().expect("heartbeat lock");
            let now = Instant::now();
            if state.pending || state.last_beat_at.is_some_and(|last| now - last < self.min_interval) {
                return;
            }
            state.pending = true;
            state.last_beat_at = Some(now);
        }
        let this = self.clone();
        tokio::spawn(async move {
            match this.client.get(&this.url).timeout(Duration::from_secs(1)).send().await {
                Ok(response) if response.status().as_u16() != 200 => {
                    tracing::warn!("BetterStack heartbeat failed with status code: {}", response.status().as_u16())
                }
                Ok(_) => {}
                Err(error) => tracing::warn!("BetterStack heartbeat error: {error}"),
            }
            this.state.lock().expect("heartbeat lock").pending = false;
        });
    }
}
