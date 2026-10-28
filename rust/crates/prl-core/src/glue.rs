//! Implementations of core's actor traits over the sibling crates' concrete types.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use indexmap::IndexMap;
use prl_io::clients::InferenceClient;
use prl_io::weights::WeightReceiver;
use prl_monitors::{LiveEvent, Monitors};
use tokio_util::sync::CancellationToken;

use crate::dispatcher::{LiveSink, SessionFinisher};
use crate::watcher::PolicyReceiver;

#[async_trait]
impl PolicyReceiver for WeightReceiver {
    async fn next_version(&mut self, current: u64) -> Result<u64> {
        WeightReceiver::next_version(self, current)
    }

    async fn wait_published(&mut self, step: u64, cancel: &CancellationToken) -> Result<()> {
        WeightReceiver::wait_published(self, step, cancel).await
    }

    async fn receive(&mut self, step: u64) -> Result<IndexMap<String, Vec<f64>>> {
        let times = WeightReceiver::receive(self, step).await?;
        Ok(times.phases().iter().map(|(phase, t)| (phase.to_string(), t.to_vec())).collect())
    }

    async fn sync_startup(&mut self, step: u64, timeout: Duration) -> Result<()> {
        WeightReceiver::sync_startup(self, step, timeout).await.map(|_| ())
    }
}

/// Session release for runs sampled from an inference endpoint.
pub struct ClientSessions(pub Arc<InferenceClient>);

#[async_trait]
impl SessionFinisher for ClientSessions {
    async fn finish_sessions(&self, session_ids: Vec<String>) {
        if session_ids.is_empty() {
            return;
        }
        self.0.finish_sessions(&session_ids).await;
    }
}

#[async_trait]
impl LiveSink for Monitors {
    async fn log_live(&self, events: Vec<LiveEvent>) {
        Monitors::log_live(self, events).await;
    }
}
