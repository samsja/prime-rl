//! Consumer side of the trainer's weight broadcast (`transports/weights`): moves the inference
//! engines onto trainer-offered policy versions through the admin plane.
//!
//! Broadcast-dir handshake per version (`<output_dir>/broadcasts/step_N`): the trainer raises
//! `.sender_ready`, the receiver replies `.receiver_ready`, the trainer raises `.started`, moves
//! the weights and raises `.finished`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use prl_config::WeightBroadcastConfig;
use tokio_util::sync::CancellationToken;

use crate::clients::{init_nixl_broadcast, load_lora_adapter, AdminPlane, Transport, UpdateTimes};
use crate::pathing::{all_ckpt_steps, step_path, wait_for_path};

const SENDER_READY_MARKER: &str = ".sender_ready";
const RECEIVER_READY_MARKER: &str = ".receiver_ready";
const FINISHED_MARKER: &str = ".finished";

/// The NIXL agent (UCX) and ModelExpress (gRPC) calls of the NIXL receiver, which have no
/// Rust bindings; implemented over the Python bridge.
#[async_trait::async_trait]
pub trait NixlBridge: Send + Sync {
    /// Publish the orchestrator's NIXL agent metadata to ModelExpress (role `orchestrator`).
    async fn publish(&mut self) -> Result<()>;
    /// Resolve and connect the trainer peer on first use, then wait (up to `timeout_s`) for its
    /// `policy:{step:016x}:ready` notification.
    async fn wait_ready(&mut self, step: u64, timeout_s: u64) -> Result<()>;
    /// Send `policy:{step:016x}:complete` to the trainer peer.
    async fn send_complete(&mut self, step: u64) -> Result<()>;
}

pub struct WeightReceiver {
    broadcast_dir: PathBuf,
    config: WeightBroadcastConfig,
    admin: AdminPlane,
    model_name: String,
    nixl: Option<Box<dyn NixlBridge>>,
    update_times: UpdateTimes,
}

impl WeightReceiver {
    /// `nixl` is required for (and only used by) the NIXL transport.
    pub fn new(
        broadcast_dir: PathBuf,
        config: WeightBroadcastConfig,
        admin: AdminPlane,
        model_name: impl Into<String>,
        nixl: Option<Box<dyn NixlBridge>>,
    ) -> Result<Self> {
        if matches!(config, WeightBroadcastConfig::Nixl(_)) && nixl.is_none() {
            bail!("the NIXL weight receiver needs a NIXL bridge");
        }
        Ok(Self { broadcast_dir, config, admin, model_name: model_name.into(), nixl, update_times: UpdateTimes::default() })
    }

    /// Per-engine times of the last full weight update (empty before the first one; LoRA
    /// hot-swaps leave it unchanged).
    pub fn update_times(&self) -> &UpdateTimes {
        &self.update_times
    }

    /// One-time transport bootstrap (rendezvous groups, sessions).
    pub async fn initialize(&mut self) -> Result<()> {
        match &self.config {
            WeightBroadcastConfig::Filesystem(_) => Ok(()),
            WeightBroadcastConfig::Nccl(c) => self.admin.initialize_nccl(&c.host, c.port, c.timeout, c.inference_world_size).await,
            WeightBroadcastConfig::Nixl(c) => {
                init_nixl_broadcast(self.admin.clients(), &c.host, c.port, c.timeout, c.inference_world_size, &c.session_id).await?;
                self.nixl.as_mut().expect("checked in new").publish().await
            }
        }
    }

    pub fn step_dir(&self, step: u64) -> PathBuf {
        step_path(&self.broadcast_dir, step)
    }

    /// Whether the trainer has offered v{step}.
    pub fn is_published(&self, step: u64) -> bool {
        self.step_dir(step).join(SENDER_READY_MARKER).exists()
    }

    /// Newest version offered beyond `current`; `current` if none.
    pub fn next_version(&self, current: u64) -> Result<u64> {
        let steps = all_ckpt_steps(&self.broadcast_dir)?;
        Ok(steps.into_iter().filter(|&s| self.is_published(s)).max().unwrap_or(current))
    }

    /// Block until the trainer offers v{step} (polls every 0.1 s). Errors when `cancel` fires.
    pub async fn wait_published(&self, step: u64, cancel: &CancellationToken) -> Result<()> {
        let sender_ready = self.step_dir(step).join(SENDER_READY_MARKER);
        while !sender_ready.exists() {
            if cancel.is_cancelled() {
                bail!("cancelled while waiting for policy v{step} to be published");
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Ok(())
    }

    /// Acknowledge the offered version, unblocking the waiting trainer.
    fn ack(step_dir: &Path) -> Result<()> {
        std::fs::OpenOptions::new().create(true).append(true).open(step_dir.join(RECEIVER_READY_MARKER))?;
        Ok(())
    }

    /// Acknowledge v{step} and move the engines onto it. Returns the update times of the last
    /// full update.
    pub async fn receive(&mut self, step: u64) -> Result<UpdateTimes> {
        let step_dir = self.step_dir(step);
        match &self.config {
            WeightBroadcastConfig::Filesystem(_) => {
                Self::ack(&step_dir)?;
                wait_for_path(&step_dir.join(FINISHED_MARKER)).await;
                if step_dir.join("adapter_config.json").exists() {
                    // An adapter is hot-swapped under live traffic: no engine pause.
                    load_lora_adapter(self.admin.clients(), &self.model_name, &step_dir, step).await?;
                } else {
                    self.update_times = self.admin.update_weights(Some(&step_dir), Transport::Filesystem, step, || Ok(())).await?;
                }
            }
            WeightBroadcastConfig::Nccl(_) => {
                // Pause, acknowledge, then enter the receive RPC: the trainer only joins the
                // collective once the engines wait in it.
                let ack_dir = step_dir.clone();
                self.update_times = self.admin.update_weights(Some(&step_dir), Transport::Nccl, step, move || Self::ack(&ack_dir)).await?;
            }
            WeightBroadcastConfig::Nixl(c) => {
                Self::ack(&step_dir)?;
                let nixl = self.nixl.as_mut().ok_or_else(|| anyhow!("missing NIXL bridge"))?;
                nixl.wait_ready(step, c.timeout).await?;
                self.update_times = self.admin.update_weights(None, Transport::Nixl, step, || Ok(())).await?;
                self.nixl.as_mut().expect("checked above").send_complete(step).await?;
            }
        }
        Ok(self.update_times.clone())
    }

    /// Rendezvous with the trainer's startup broadcast of v{step}.
    pub async fn sync_startup(&mut self, step: u64, timeout: Duration) -> Result<UpdateTimes> {
        let never = CancellationToken::new();
        tokio::time::timeout(timeout, self.wait_published(step, &never))
            .await
            .map_err(|_| anyhow!("timed out after {}s waiting for the trainer's startup broadcast of policy v{step}", timeout.as_secs()))??;
        self.receive(step).await
    }
}
