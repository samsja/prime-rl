//! Trainer-facing batch transport (`transports/batch/*`, sender side).

mod filesystem;
mod types;
mod zmq;

use std::path::PathBuf;

use anyhow::{ensure, Result};
use prl_config::TransportConfig;

pub use filesystem::FileSystemBatchSender;
pub use types::{encode_micro_batches, MicroBatch};
pub use zmq::ZmqBatchSender;

/// Sends a step's micro batch grid once every trainer rank requested it.
pub enum BatchSender {
    FileSystem(FileSystemBatchSender),
    Zmq(ZmqBatchSender),
}

impl BatchSender {
    /// `setup_batch_sender`.
    pub async fn new(output_dir: PathBuf, data_world_size: usize, current_step: u64, transport: &TransportConfig) -> Result<Self> {
        Ok(match transport {
            TransportConfig::Filesystem => {
                BatchSender::FileSystem(FileSystemBatchSender::new(output_dir, data_world_size, current_step))
            }
            TransportConfig::Zmq(zmq) => {
                BatchSender::Zmq(ZmqBatchSender::new(data_world_size, current_step, &zmq.host, zmq.port).await?)
            }
        })
    }

    /// Returns the seconds spent handing the step to the transport, excluding the wait for requests.
    pub async fn send(&mut self, grid: Vec<Vec<MicroBatch>>) -> Result<f64> {
        match self {
            BatchSender::FileSystem(s) => s.send(grid).await,
            BatchSender::Zmq(s) => s.send(grid).await,
        }
    }

    pub fn current_step(&self) -> u64 {
        match self {
            BatchSender::FileSystem(s) => s.current_step,
            BatchSender::Zmq(s) => s.current_step,
        }
    }

    pub async fn close(self) {
        if let BatchSender::Zmq(s) = self {
            s.close().await;
        }
    }
}

fn check_grid(grid: &[Vec<MicroBatch>], data_world_size: usize) -> Result<()> {
    ensure!(grid.len() == data_world_size, "Number of micro batch lists must match data world size");
    for list in grid {
        ensure!(list.len() == grid[0].len(), "All micro batch lists must have the same length");
    }
    Ok(())
}

/// Encode every rank's list off the reactor.
async fn encode_grid(grid: Vec<Vec<MicroBatch>>) -> Result<Vec<Vec<u8>>> {
    Ok(tokio::task::spawn_blocking(move || grid.iter().map(|list| encode_micro_batches(list)).collect()).await?)
}
