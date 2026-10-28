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
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use async_trait::async_trait;
pub use prl_config::FileMonitorConfig;
use prl_types::{Kind, Scalar, Subset};
use prl_wire::Value;

use crate::{sanitize, EpisodeRef, Monitor, Row};
use chunks::ChunkedJsonl;
use live::{LiveEvent, LiveWriter};

pub fn file_monitor_dir(output_dir: &Path) -> PathBuf {
    output_dir.join("monitors/file")
}

pub fn trace_stream(output_dir: &Path) -> PathBuf {
    file_monitor_dir(output_dir).join("traces/stream")
}

/// A stream's index, beside it: `stream` -> `stream.index.jsonl`.
fn index_path(stream: &Path) -> PathBuf {
    let mut name = stream.as_os_str().to_owned();
    name.push(".index.jsonl");
    name.into()
}

pub fn unix_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_secs_f64()
}

pub(crate) fn open_append(path: &Path) -> Result<File> {
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
    /// Episodes in the stream so far: index rows number them from 1.
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
        // a relaunch appends to the stream it finds, so the numbering carries on
        let index = index_path(&trace_stream(output_dir));
        let logged = match fs::read(&index) {
            Ok(data) => data
                .split(|b| *b == b'\n')
                .filter(|line| !line.is_empty())
                .count() as u64,
            Err(_) => 0,
        };
        let path = file_monitor_dir(output_dir).join(&config.path);
        fs::create_dir_all(path.parent().expect("metrics file has a parent"))?;
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
            live: LiveWriter::new(&file_monitor_dir(output_dir).join("traces/live")),
            streams: HashMap::new(),
        })
    }

    /// One row of `metrics.jsonl`, written with Python's `json.dumps`. `step=None` logs a
    /// time-keyed row.
    pub fn log_metrics(&mut self, metrics: &Row, step: Option<u64>) -> Result<()> {
        let (metrics, dropped) = sanitize(metrics);
        if !dropped.is_empty() {
            tracing::warn!(
                "Dropping {} non-finite value(s) from {}: {}",
                dropped.len(),
                self.config.path.display(),
                dropped[..dropped.len().min(5)].join(", ")
            );
        }
        let mut row = Row::new();
        row.insert("step".into(), step.map_or(Scalar::Null, Scalar::from));
        row.insert("time".into(), Scalar::Float(unix_time()));
        row.extend(metrics);
        if let Some(producer) = &self.producer {
            row.insert("producer".into(), producer.as_str().into());
        }
        let value = prl_wire::value::to_value(&row)?;
        writeln!(
            self.file,
            "{}",
            prl_wire::json::to_python_json(&value, false)
        )?;
        Ok(())
    }

    /// Merge the epoch's expected count into `plan.json` (atomic replace).
    pub fn log_eval_plan(&mut self, env_name: &str, step: u64, expected: u64) -> Result<()> {
        let path = file_monitor_dir(&self.output_dir).join("plan.json");
        let mut plan: serde_json::Map<String, serde_json::Value> = if path.is_file() {
            serde_json::from_slice(&fs::read(&path)?)?
        } else {
            Default::default()
        };
        plan.entry(env_name)
            .or_insert_with(|| serde_json::json!({}))
            .as_object_mut()
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

    /// Append records to a stream, opened on first use and kept open, and their index rows
    /// to the index beside it. The stream is flushed before the index is written: a row a
    /// reader sees points at a record it can read.
    fn append(
        &mut self,
        directory: PathBuf,
        records: &[Value],
        mut row: impl FnMut(&Value, u64, u64) -> Value,
    ) -> Result<()> {
        if !self.streams.contains_key(&directory) {
            let stream =
                ChunkedJsonl::open(&directory, self.config.chunk_bytes, self.config.compress)?;
            let index = open_append(&index_path(&directory))?;
            self.streams.insert(directory.clone(), (stream, index));
        }
        let (stream, index) = self.streams.get_mut(&directory).expect("inserted above");
        let mut rows = Vec::new();
        for record in records {
            let mut line = prl_wire::json::to_vec(record);
            line.push(b'\n');
            let (chunk, offset) = stream.append(&line)?;
            prl_wire::json::write(&mut rows, &row(record, chunk, offset));
            rows.push(b'\n');
        }
        stream.flush()?;
        index.write_all(&rows)?;
        Ok(())
    }

    /// Append each episode of the `all` subset to the trace stream as it completes, with its
    /// index row. The `effective` cohort writes no second copy.
    pub fn log_episodes(&mut self, episodes: &[EpisodeRef], subset: Subset) -> Result<()> {
        if subset == Subset::Effective {
            return Ok(());
        }
        let records: Vec<Value> = episodes
            .iter()
            .map(|e| e.to_record(self.config.float_decimals))
            .collect();
        let mut line = self.logged;
        self.append(
            trace_stream(&self.output_dir),
            &records,
            |record, chunk, offset| {
                line += 1;
                Value::Map(index::index_row(line, record, chunk, offset))
            },
        )?;
        self.logged = line;
        Ok(())
    }

    /// Append trace updates to this producer's annotation stream.
    pub fn log_annotations(&mut self, updates: &[Value]) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let producer = self.producer.as_deref().unwrap_or("unknown");
        let directory = file_monitor_dir(&self.output_dir)
            .join("traces/annotations")
            .join(producer);
        self.append(directory, updates, update::update_index_row)
    }

    /// Like an exiting Python process: flush what is open, seal nothing.
    pub fn flush(&mut self) -> Result<()> {
        for (stream, _) in self.streams.values_mut() {
            stream.flush()?;
        }
        Ok(())
    }

    pub fn finalize(&mut self) -> Result<()> {
        for (_, (stream, _)) in self.streams.drain() {
            stream.close()?;
        }
        tracing::info!("Finalized metrics at {}", self.path.display());
        Ok(())
    }
}

/// The file monitor on the monitors actor. Its writes run on the blocking pool one at a time
/// (the monitor moves into the task and back), so appends never interleave.
pub struct FileMonitorActor(Option<FileMonitor>);

impl FileMonitorActor {
    pub async fn init(
        config: FileMonitorConfig,
        output_dir: PathBuf,
        producer: Option<String>,
    ) -> Result<Self> {
        let monitor =
            tokio::task::spawn_blocking(move || FileMonitor::init(config, &output_dir, producer))
                .await??;
        Ok(FileMonitorActor(Some(monitor)))
    }

    async fn blocking(
        &mut self,
        f: impl FnOnce(&mut FileMonitor) -> Result<()> + Send + 'static,
    ) -> Result<()> {
        let mut monitor = self.0.take().context("file monitor lost after a panic")?;
        let (monitor, result) = tokio::task::spawn_blocking(move || {
            let result = f(&mut monitor);
            (monitor, result)
        })
        .await?;
        self.0 = Some(monitor);
        result
    }
}

#[async_trait]
impl Monitor for FileMonitorActor {
    fn name(&self) -> &'static str {
        "FileMonitor"
    }

    async fn log_metrics(&mut self, metrics: &Row, step: Option<u64>) -> Result<()> {
        let metrics = metrics.clone();
        self.blocking(move |m| m.log_metrics(&metrics, step)).await
    }

    async fn log_episodes(
        &mut self,
        episodes: &[EpisodeRef],
        _step: u64,
        _kind: Kind,
        subset: Subset,
    ) -> Result<()> {
        let episodes = episodes.to_vec();
        self.blocking(move |m| m.log_episodes(&episodes, subset))
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
        self.blocking(FileMonitor::finalize).await
    }

    async fn crash(&mut self) -> Result<()> {
        self.blocking(FileMonitor::flush).await
    }
}
