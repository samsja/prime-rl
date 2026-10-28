//! The file monitor (`monitors/file/monitor.py`): metrics JSONL, the episode trace stream and
//! its index, per-producer annotation streams, live traces and the eval plan, all under
//! `{output_dir}/monitors/file/`.

pub mod chunks;
pub mod index;
pub mod live;
pub mod update;

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use prl_types::{Kind, Subset};
use prl_wire::Value;
use serde::Deserialize;

use crate::pyjson::{dumps_row, sanitize, Row, Scalar};
use crate::{EpisodeRef, Monitor};
use chunks::ChunkedJsonl;
use live::{LiveEvent, LiveWriter};

/// `FileMonitorConfig` (the `monitors.file` block of the config JSON).
#[derive(Debug, Clone, Deserialize)]
pub struct FileMonitorConfig {
    #[serde(default = "default_path")]
    pub path: PathBuf,
    #[serde(default = "default_chunk_bytes")]
    pub chunk_bytes: u64,
    #[serde(default = "default_true")]
    pub compress: bool,
    #[serde(default = "default_float_decimals")]
    pub float_decimals: Option<u32>,
}

fn default_path() -> PathBuf {
    "metrics.jsonl".into()
}
fn default_chunk_bytes() -> u64 {
    5 * 1024 * 1024 * 1024
}
fn default_true() -> bool {
    true
}
fn default_float_decimals() -> Option<u32> {
    Some(4)
}

impl From<&prl_config::FileMonitorConfig> for FileMonitorConfig {
    fn from(c: &prl_config::FileMonitorConfig) -> Self {
        FileMonitorConfig {
            path: c.path.clone(),
            chunk_bytes: c.chunk_bytes,
            compress: c.compress,
            float_decimals: c.float_decimals,
        }
    }
}

impl Default for FileMonitorConfig {
    fn default() -> Self {
        FileMonitorConfig {
            path: default_path(),
            chunk_bytes: default_chunk_bytes(),
            compress: true,
            float_decimals: default_float_decimals(),
        }
    }
}

pub fn file_monitor_dir(output_dir: &Path) -> PathBuf {
    output_dir.join("monitors").join("file")
}
pub fn trace_dir(output_dir: &Path) -> PathBuf {
    file_monitor_dir(output_dir).join("traces")
}
pub fn trace_stream(output_dir: &Path) -> PathBuf {
    trace_dir(output_dir).join("stream")
}
pub fn annotations_dir(output_dir: &Path) -> PathBuf {
    trace_dir(output_dir).join("annotations")
}
pub fn eval_plan_path(output_dir: &Path) -> PathBuf {
    file_monitor_dir(output_dir).join("plan.json")
}
/// A stream's index, beside it.
pub fn index_path(stream: &Path) -> PathBuf {
    let name = stream
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    stream.with_file_name(format!(
        "{}.index.jsonl",
        name.strip_suffix(".jsonl").unwrap_or(&name)
    ))
}

pub fn unix_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_secs_f64()
}

/// Lines in a file as Python's line iteration counts them.
fn count_lines(path: &Path) -> Result<u64> {
    let data = fs::read(path)?;
    let newlines = data.iter().filter(|b| **b == b'\n').count() as u64;
    Ok(newlines + u64::from(data.last().is_some_and(|b| *b != b'\n')))
}

fn open_append(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))
}

pub struct FileMonitor {
    config: FileMonitorConfig,
    output_dir: PathBuf,
    producer: Option<String>,
    logged: u64,
    path: PathBuf,
    file: File,
    live: LiveWriter,
    streams: HashMap<PathBuf, (ChunkedJsonl, File)>,
}

impl FileMonitor {
    pub fn init(
        config: FileMonitorConfig,
        output_dir: &Path,
        producer: Option<String>,
    ) -> Result<Self> {
        let index = index_path(&trace_stream(output_dir));
        // a relaunch appends to the stream it finds, so the numbering carries on
        let logged = if index.is_file() {
            count_lines(&index)?
        } else {
            0
        };
        let path = file_monitor_dir(output_dir).join(&config.path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = open_append(&path)?;
        tracing::info!(
            "Logging metrics and episodes to the local filesystem ({})",
            output_dir.display()
        );
        Ok(FileMonitor {
            config,
            output_dir: output_dir.to_path_buf(),
            producer,
            logged,
            path,
            file,
            live: LiveWriter::new(output_dir),
            streams: HashMap::new(),
        })
    }

    /// `step=None` logs a time-keyed row.
    pub fn log_metrics(&mut self, metrics: &Row, step: Option<u64>) -> Result<()> {
        let (sanitized, dropped) = sanitize(metrics);
        if !dropped.is_empty() {
            tracing::warn!(
                "Dropping {} non-finite value(s) from {}: {}",
                dropped.len(),
                self.config.path.display(),
                dropped
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let mut row = Row::new();
        row.insert("step".into(), step.map_or(Scalar::Null, Scalar::from));
        row.insert("time".into(), Scalar::Float(unix_time()));
        row.extend(sanitized);
        if let Some(producer) = &self.producer {
            row.insert("producer".into(), producer.as_str().into());
        }
        let mut line = dumps_row(&row);
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        Ok(())
    }

    /// Merge the epoch's expected count into `plan.json` (atomic replace).
    pub fn log_eval_plan(&mut self, env_name: &str, step: u64, expected: u64) -> Result<()> {
        let path = eval_plan_path(&self.output_dir);
        let mut plan: serde_json::Map<String, serde_json::Value> = if path.is_file() {
            serde_json::from_slice(&fs::read(&path)?)?
        } else {
            Default::default()
        };
        let env = plan
            .entry(env_name.to_string())
            .or_insert_with(|| serde_json::Value::Object(Default::default()));
        env.as_object_mut()
            .with_context(|| format!("{} entry for {env_name} is not an object", path.display()))?
            .insert(step.to_string(), expected.into());
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_vec(&plan)?)?;
        fs::rename(&tmp, &path)?;
        Ok(())
    }

    pub fn log_live(&mut self, events: &[LiveEvent]) -> Result<()> {
        self.live.write(events)
    }

    /// A stream and its index, opened on first use and kept open.
    fn stream(&mut self, directory: PathBuf) -> Result<&mut (ChunkedJsonl, File)> {
        if !self.streams.contains_key(&directory) {
            let stream =
                ChunkedJsonl::open(&directory, self.config.chunk_bytes, self.config.compress)?;
            let index = open_append(&index_path(&directory))?;
            self.streams.insert(directory.clone(), (stream, index));
        }
        Ok(self.streams.get_mut(&directory).expect("inserted above"))
    }

    /// Append each episode of the `all` subset to the trace stream as it completes, with its
    /// index row. The `effective` cohort writes no second copy.
    pub fn log_episodes(
        &mut self,
        episodes: &[EpisodeRef],
        _step: u64,
        _kind: Kind,
        subset: Subset,
    ) -> Result<()> {
        if subset == Subset::Effective {
            return Ok(());
        }
        let float_decimals = self.config.float_decimals;
        let mut logged = self.logged;
        let (stream, index) = self.stream(trace_stream(&self.output_dir.clone()))?;
        for episode in episodes {
            let record = episode.to_record(float_decimals);
            let mut line = prl_wire::json::to_vec(&record);
            line.push(b'\n');
            let (chunk, offset) = stream.append(&line)?;
            logged += 1;
            let mut row = prl_wire::json::to_vec(&Value::Map(index::index_row(
                logged, &record, chunk, offset,
            )));
            row.push(b'\n');
            index.write_all(&row)?;
        }
        stream.flush()?;
        index.flush()?;
        self.logged = logged;
        Ok(())
    }

    /// Append trace updates to this producer's annotation stream.
    pub fn log_annotations(&mut self, updates: &[Value]) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let directory =
            annotations_dir(&self.output_dir).join(self.producer.as_deref().unwrap_or("unknown"));
        let (stream, index) = self.stream(directory)?;
        for update in updates {
            let mut line = prl_wire::json::to_vec(update);
            line.push(b'\n');
            let (chunk, offset) = stream.append(&line)?;
            let mut row = prl_wire::json::to_vec(&update::update_index_row(update, chunk, offset));
            row.push(b'\n');
            index.write_all(&row)?;
        }
        stream.flush()?;
        index.flush()?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        for (stream, index) in self.streams.values_mut() {
            stream.flush()?;
            index.flush()?;
        }
        self.file.flush()?;
        Ok(())
    }

    pub fn finalize(&mut self) -> Result<()> {
        for (_, (stream, mut index)) in self.streams.drain() {
            stream.close()?;
            index.flush()?;
        }
        self.file.flush()?;
        tracing::info!("Finalized metrics at {}", self.path.display());
        Ok(())
    }
}

/// The file monitor on the monitors actor: its blocking writes run on the blocking pool, one
/// at a time (the state moves into the task and back), so appends never interleave.
pub struct FileMonitorActor {
    inner: Option<FileMonitor>,
}

impl FileMonitorActor {
    pub async fn init(
        config: FileMonitorConfig,
        output_dir: PathBuf,
        producer: Option<String>,
    ) -> Result<Self> {
        let monitor =
            tokio::task::spawn_blocking(move || FileMonitor::init(config, &output_dir, producer))
                .await??;
        Ok(FileMonitorActor {
            inner: Some(monitor),
        })
    }

    async fn blocking<R: Send + 'static>(
        &mut self,
        f: impl FnOnce(&mut FileMonitor) -> Result<R> + Send + 'static,
    ) -> Result<R> {
        let mut monitor = self
            .inner
            .take()
            .context("file monitor lost after a panic")?;
        let (monitor, result) = tokio::task::spawn_blocking(move || {
            let result = f(&mut monitor);
            (monitor, result)
        })
        .await?;
        self.inner = Some(monitor);
        result
    }
}

#[async_trait]
impl Monitor for FileMonitorActor {
    fn name(&self) -> &'static str {
        "FileMonitor"
    }

    async fn log_metrics(&mut self, metrics: &Row, step: Option<u64>) -> Result<()> {
        // a single line write: cheap enough for the actor thread
        self.inner
            .as_mut()
            .context("file monitor lost after a panic")?
            .log_metrics(metrics, step)
    }

    async fn log_episodes(
        &mut self,
        episodes: &[EpisodeRef],
        step: u64,
        kind: Kind,
        subset: Subset,
    ) -> Result<()> {
        if subset == Subset::Effective {
            return Ok(());
        }
        let episodes = episodes.to_vec();
        self.blocking(move |m| m.log_episodes(&episodes, step, kind, subset))
            .await
    }

    async fn log_annotations(&mut self, updates: &Arc<Vec<Value>>) -> Result<()> {
        let updates = Arc::clone(updates);
        self.blocking(move |m| m.log_annotations(&updates)).await
    }

    async fn log_live(&mut self, events: &Arc<Vec<LiveEvent>>) -> Result<()> {
        let events = Arc::clone(events);
        self.blocking(move |m| m.log_live(&events)).await
    }

    async fn log_eval_plan(&mut self, env_name: &str, step: u64, expected: u64) -> Result<()> {
        let env_name = env_name.to_string();
        self.blocking(move |m| m.log_eval_plan(&env_name, step, expected))
            .await
    }

    async fn finalize(&mut self, _paused: bool) -> Result<()> {
        self.blocking(|m| m.finalize()).await
    }

    /// Like an exiting Python process: flush what is open, seal nothing.
    async fn crash(&mut self) -> Result<()> {
        self.blocking(|m| m.flush()).await
    }
}
