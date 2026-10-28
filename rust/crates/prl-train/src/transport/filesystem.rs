use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::{check_grid, encode_grid, MicroBatch};

const REQUEST_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Writes `{output_dir}/batches/step_{n}/rank_{r}.bin.tmp`, waits for every trainer rank's
/// `rank_*.request` marker (each holds the trainer world size), then renames to `.bin`.
pub struct FileSystemBatchSender {
    batch_dir: PathBuf,
    data_world_size: usize,
    pub current_step: u64,
}

impl FileSystemBatchSender {
    pub fn new(output_dir: PathBuf, data_world_size: usize, current_step: u64) -> Self {
        Self { batch_dir: output_dir.join("batches"), data_world_size, current_step }
    }

    pub async fn send(&mut self, grid: Vec<Vec<MicroBatch>>) -> Result<f64> {
        check_grid(&grid, self.data_world_size)?;
        let step_path = self.batch_dir.join(format!("step_{}", self.current_step));
        tokio::fs::create_dir_all(&step_path).await?;
        let buffers = encode_grid(grid).await?;
        let dir = step_path.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            for (rank, buffer) in buffers.iter().enumerate() {
                std::fs::write(dir.join(format!("rank_{rank}.bin.tmp")), buffer)?;
            }
            Ok(())
        })
        .await??;
        wait_for_requests(&step_path).await?;
        let start = Instant::now();
        for rank in 0..self.data_world_size {
            tokio::fs::rename(step_path.join(format!("rank_{rank}.bin.tmp")), step_path.join(format!("rank_{rank}.bin")))
                .await?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        self.current_step += 1;
        Ok(elapsed)
    }
}

async fn wait_for_requests(step_path: &Path) -> Result<()> {
    loop {
        let mut requests = Vec::new();
        let mut entries = tokio::fs::read_dir(step_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("rank_") && name.ends_with(".request") {
                requests.push(entry.path());
            }
        }
        if let Some(first) = requests.first() {
            let text = tokio::fs::read_to_string(first).await?;
            let world_size: usize = text.trim().parse().with_context(|| format!("bad request marker {first:?}"))?;
            if requests.len() == world_size {
                return Ok(());
            }
        }
        tokio::time::sleep(REQUEST_POLL_INTERVAL).await;
    }
}
