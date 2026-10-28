//! Orchestrator checkpoints (`orchestrator/ckpt.py`): `<output_dir>/checkpoints/step_N/orchestrator/progress.pt`,
//! written in `torch.save`'s zip format so Python and Rust runs resume each other.

mod pickle;
mod pyrandom;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use indexmap::IndexMap;
use prl_types::Progress;

use crate::logging::format_time;
use crate::pathing::{ckpt_dir, step_path};
pub use pickle::PyValue;
pub use pyrandom::{PyRandom, PyRandomState};

const PROGRESS_MODULE: &str = "prime_rl.orchestrator.types";
const PROGRESS_CLASS: &str = "Progress";

/// Everything the orchestrator checkpoints (`build_state`).
#[derive(Debug, Clone, PartialEq)]
pub struct OrchestratorState {
    pub progress: Progress,
    pub train_source: TrainSourceState,
}

/// `TrainSource.state_dict()`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainSourceState {
    pub rng: PyRandomState,
    pub envs: IndexMap<String, CurriculumState>,
}

/// `Curriculum.state_dict()`.
#[derive(Debug, Clone, PartialEq)]
pub struct CurriculumState {
    pub sampler: SamplerState,
    /// Gate name -> gate state (the built-in gates keep none: `{}`).
    pub gates: IndexMap<String, PyValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SamplerState {
    /// `StandardSampler`: `{"cursor": int}`.
    Standard { cursor: u64 },
    /// `DifficultyPoolSampler`: `{"rng": getstate(), "task_rewards": {key: float}}`.
    DifficultyPool { rng: Box<PyRandomState>, task_rewards: IndexMap<String, f64> },
}

impl TrainSourceState {
    /// `TrainSource.load_state_dict`'s env check against the configured envs.
    pub fn validate_envs(&self, configured: &[String]) -> Result<()> {
        let mut saved: Vec<&str> = self.envs.keys().map(String::as_str).collect();
        let mut expected: Vec<&str> = configured.iter().map(String::as_str).collect();
        saved.sort_unstable();
        expected.sort_unstable();
        ensure!(
            saved == expected,
            "Train-source checkpoint envs {saved:?} do not match configured envs {expected:?}"
        );
        Ok(())
    }
}

fn expect_fields(value: &PyValue, expected: &[&str], what: &str) -> Result<()> {
    let mut keys = value.str_keys()?;
    keys.sort_unstable();
    let mut sorted = expected.to_vec();
    sorted.sort_unstable();
    ensure!(keys == sorted, "{what} checkpoint fields must be {sorted:?}");
    Ok(())
}

impl OrchestratorState {
    fn to_py(&self) -> PyValue {
        let p = &self.progress;
        let progress = PyValue::Object {
            module: PROGRESS_MODULE.into(),
            name: PROGRESS_CLASS.into(),
            state: Box::new(PyValue::dict([
                ("step", PyValue::Int(p.step as i64)),
                ("total_tokens", PyValue::Int(p.total_tokens as i64)),
                ("total_samples", PyValue::Int(p.total_samples as i64)),
                ("total_problems", PyValue::Int(p.total_problems as i64)),
                ("rollout_window", PyValue::Int(p.rollout_window as i64)),
            ])),
        };
        let envs = self.train_source.envs.iter().map(|(name, curriculum)| {
            let sampler = match &curriculum.sampler {
                SamplerState::Standard { cursor } => PyValue::dict([("cursor", PyValue::Int(*cursor as i64))]),
                SamplerState::DifficultyPool { rng, task_rewards } => PyValue::dict([
                    ("rng", rng.to_py()),
                    (
                        "task_rewards",
                        PyValue::dict(task_rewards.iter().map(|(k, v)| (k.clone(), PyValue::Float(*v)))),
                    ),
                ]),
            };
            let gates = PyValue::dict(curriculum.gates.iter().map(|(k, v)| (k.clone(), v.clone())));
            (name.clone(), PyValue::dict([("sampler", sampler), ("gates", gates)]))
        });
        PyValue::dict([
            ("progress", progress),
            (
                "train_source",
                PyValue::dict([("rng", self.train_source.rng.to_py()), ("envs", PyValue::dict(envs))]),
            ),
        ])
    }

    fn from_py(value: &PyValue) -> Result<Self> {
        let PyValue::Object { state: progress, .. } = value.get("progress")? else {
            bail!("checkpoint progress must be a Progress object");
        };
        let mut p = Progress::default();
        for (key, field) in progress.as_dict()? {
            let v = field.as_int()?;
            let v = u64::try_from(v).with_context(|| format!("negative progress field {key:?}"))?;
            match key.as_str()? {
                "step" => p.step = v,
                "total_tokens" => p.total_tokens = v,
                "total_samples" => p.total_samples = v,
                "total_problems" => p.total_problems = v,
                "rollout_window" => p.rollout_window = v,
                _ => {}
            }
        }

        let source = value.get("train_source")?;
        expect_fields(source, &["rng", "envs"], "Train-source")?;
        let rng = PyRandomState::from_py(source.get("rng")?)?;
        let mut envs = IndexMap::new();
        for (name, curriculum) in source.get("envs")?.as_dict()? {
            expect_fields(curriculum, &["sampler", "gates"], "Curriculum")?;
            let sampler = curriculum.get("sampler")?;
            let mut keys = sampler.str_keys()?;
            keys.sort_unstable();
            let sampler = match keys.as_slice() {
                ["cursor"] => SamplerState::Standard { cursor: u64::try_from(sampler.get("cursor")?.as_int()?)? },
                ["rng", "task_rewards"] => SamplerState::DifficultyPool {
                    rng: Box::new(PyRandomState::from_py(sampler.get("rng")?)?),
                    task_rewards: sampler
                        .get("task_rewards")?
                        .as_dict()?
                        .iter()
                        .map(|(k, v)| Ok((k.as_str()?.to_owned(), v.as_float()?)))
                        .collect::<Result<_>>()?,
                },
                other => bail!("unsupported task sampler checkpoint state with fields {other:?}"),
            };
            let gates = curriculum
                .get("gates")?
                .as_dict()?
                .iter()
                .map(|(k, v)| Ok((k.as_str()?.to_owned(), v.clone())))
                .collect::<Result<_>>()?;
            envs.insert(name.as_str()?.to_owned(), CurriculumState { sampler, gates });
        }
        Ok(Self { progress: p, train_source: TrainSourceState { rng, envs } })
    }
}

/// Saves and loads orchestrator checkpoints.
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    ckpt_dir: PathBuf,
}

impl CheckpointManager {
    pub fn new(output_dir: &Path) -> Self {
        Self { ckpt_dir: ckpt_dir(output_dir) }
    }

    fn ckpt_path(&self, step: u64) -> PathBuf {
        step_path(&self.ckpt_dir, step).join("orchestrator")
    }

    /// Write `state` as the checkpoint of `step` (temp file + atomic rename).
    pub fn save(&self, state: &OrchestratorState, step: u64) -> Result<()> {
        let ckpt_path = self.ckpt_path(step);
        std::fs::create_dir_all(&ckpt_path)?;
        let start = Instant::now();
        let tmp = ckpt_path.join(format!("progress.pt.{}.tmp", uuid::Uuid::new_v4().simple()));
        let written = write_torch_zip(&tmp, &pickle::dumps(&state.to_py()))
            .and_then(|()| std::fs::rename(&tmp, ckpt_path.join("progress.pt")).map_err(Into::into));
        if let Err(error) = written {
            let _ = std::fs::remove_file(&tmp);
            return Err(error);
        }
        tracing::debug!(
            "Orchestrator checkpoint saved to {} in {}",
            ckpt_path.display(),
            format_time(start.elapsed().as_secs_f64())
        );
        Ok(())
    }

    /// Load the checkpoint of `step`; `path` overrides where it is read from (an external run's
    /// `step_<N>/orchestrator`).
    pub fn load(&self, step: u64, path: Option<&Path>) -> Result<OrchestratorState> {
        let ckpt_path = path.map(Path::to_path_buf).unwrap_or_else(|| self.ckpt_path(step));
        let state_file = ckpt_path.join("progress.pt");
        if !state_file.exists() {
            bail!("Orchestrator checkpoint not found at {}", state_file.display());
        }
        tracing::debug!("Loading checkpoint from {}", state_file.display());
        let start = Instant::now();
        let data = read_torch_zip(&state_file)?;
        let state = OrchestratorState::from_py(&pickle::loads(&data)?)
            .with_context(|| format!("invalid orchestrator checkpoint {}", state_file.display()))?;
        for name in state.train_source.envs.keys() {
            tracing::info!("Resumed curriculum state for env {name}");
        }
        tracing::debug!("Orchestrator checkpoint loaded in {}", format_time(start.elapsed().as_secs_f64()));
        Ok(state)
    }
}

/// Lay `pickle` out as `torch.save(obj, f)` does (no tensors): stored, 64-byte aligned entries
/// under `archive/`.
fn write_torch_zip(path: &Path, pickle: &[u8]) -> Result<()> {
    let mut zip = zip::ZipWriter::new(std::fs::File::create(path)?);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .with_alignment(64);
    let serialization_id = format!("{:020}{:020}", rand::random::<u64>() % 10u64.pow(19), rand::random::<u64>() % 10u64.pow(19));
    let records: [(&str, &[u8]); 6] = [
        ("data.pkl", pickle),
        (".format_version", b"1"),
        (".storage_alignment", b"64"),
        ("byteorder", b"little"),
        ("version", b"3\n"),
        (".data/serialization_id", serialization_id.as_bytes()),
    ];
    for (name, data) in records {
        zip.start_file(format!("archive/{name}"), options)?;
        zip.write_all(data)?;
    }
    zip.finish()?;
    Ok(())
}

/// The `data.pkl` record of a `torch.save` zip (any archive prefix).
fn read_torch_zip(path: &Path) -> Result<Vec<u8>> {
    let mut zip = zip::ZipArchive::new(std::fs::File::open(path)?)
        .with_context(|| format!("{} is not a torch zip checkpoint", path.display()))?;
    let name = zip
        .file_names()
        .find(|name| name.ends_with("/data.pkl"))
        .map(str::to_owned)
        .with_context(|| format!("{} has no data.pkl record", path.display()))?;
    let mut data = Vec::new();
    zip.by_name(&name)?.read_to_end(&mut data)?;
    Ok(data)
}
