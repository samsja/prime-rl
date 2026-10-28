//! Monitors (`prime_rl.monitors`): the orchestrator's metrics and episode sinks — local file
//! traces, the Prime platform and W&B — behind one ordered actor.
//!
//! Every producer logs through a cloned [`Monitors`] handle; the actor consumes one message at a
//! time and fans it out to each registered monitor in registration order (prime, wandb, file).
//! A monitor that fails to log is warned about and skipped: monitoring never takes the run down.
//! A monitor whose init fails does: a configured monitor must work.

pub mod annotations;
pub mod file;
pub mod prime;
pub mod pyjson;
pub mod records;
pub mod wandb;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use prl_types::{Kind, Subset};
use prl_wire::Value;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub use file::live::{stage, LiveEvent, STAGES};
pub use file::FileMonitorConfig;
pub use prime::{PrimeTrainMonitor, PrimeTrainSetup};
pub use pyjson::{Row, Scalar};
pub use wandb::{WandbBridge, WandbMonitor};

/// Default precision of per-token float streams in uploaded records (`RECORD_FLOAT_DECIMALS`).
pub const RECORD_FLOAT_DECIMALS: u32 = 4;
const INBOX: usize = 1024;

/// Episodes are shared with the sinks: the arrival record is serialized later, off the
/// producer's path, while the same episode flows on.
pub type EpisodeRef = Arc<prl_wire::Episode>;

/// One monitor (`prime_rl.monitors.base.Monitor`). Monitors that only carry scalars ignore the
/// rest through the default no-ops.
#[async_trait]
pub trait Monitor: Send {
    /// The class name the fan-out warnings use.
    fn name(&self) -> &'static str;
    async fn log_metrics(&mut self, metrics: &Row, step: Option<u64>) -> Result<()>;
    async fn log_episodes(
        &mut self,
        episodes: &[EpisodeRef],
        step: u64,
        kind: Kind,
        subset: Subset,
    ) -> Result<()>;
    async fn log_annotations(&mut self, _updates: &Arc<Vec<Value>>) -> Result<()> {
        Ok(())
    }
    async fn log_live(&mut self, _events: &Arc<Vec<LiveEvent>>) -> Result<()> {
        Ok(())
    }
    async fn log_eval_plan(&mut self, _env_name: &str, _step: u64, _expected: u64) -> Result<()> {
        Ok(())
    }
    async fn log_eval_epoch(
        &mut self,
        _env_name: &str,
        _step: u64,
        _episodes: &[EpisodeRef],
    ) -> Result<()> {
        Ok(())
    }
    /// A paused run stopped to resume later and did not complete.
    async fn finalize(&mut self, paused: bool) -> Result<()>;
    /// The process is going away without a clean finalize (Python: atexit hooks, open files
    /// closed by the interpreter). Defaults to doing nothing.
    async fn crash(&mut self) -> Result<()> {
        Ok(())
    }
}

enum MonitorMsg {
    Metrics {
        row: Row,
        step: Option<u64>,
    },
    Episodes {
        episodes: Vec<EpisodeRef>,
        step: u64,
        kind: Kind,
        subset: Subset,
    },
    Annotations(Arc<Vec<Value>>),
    Live(Arc<Vec<LiveEvent>>),
    EvalPlan {
        env_name: String,
        step: u64,
        expected: u64,
    },
    EvalEpoch {
        env_name: String,
        step: u64,
        episodes: Vec<EpisodeRef>,
    },
    Finalize {
        paused: bool,
        done: oneshot::Sender<()>,
    },
}

/// The producers' handle to the monitors actor. Messages are processed in send order; each
/// method waits only for the send (the inbox is bounded, so a stalled monitor backs pressure
/// up to producers like the awaited Python fan-out did), `finalize` for completion.
#[derive(Clone)]
pub struct Monitors {
    tx: mpsc::Sender<MonitorMsg>,
}

impl Monitors {
    async fn send(&self, msg: MonitorMsg) {
        if self.tx.send(msg).await.is_err() {
            tracing::debug!("Monitors already stopped; dropping a log call");
        }
    }

    /// A row of scalar metrics; `step=None` logs a time-keyed row.
    pub async fn log_metrics(&self, row: Row, step: Option<u64>) {
        self.send(MonitorMsg::Metrics { row, step }).await
    }

    pub async fn log_episodes(
        &self,
        episodes: Vec<EpisodeRef>,
        step: u64,
        kind: Kind,
        subset: Subset,
    ) {
        self.send(MonitorMsg::Episodes {
            episodes,
            step,
            kind,
            subset,
        })
        .await
    }

    /// Trace updates (`make_update` records) for this producer's annotation stream.
    pub async fn log_annotations(&self, updates: Vec<Value>) {
        self.send(MonitorMsg::Annotations(Arc::new(updates))).await
    }

    pub async fn log_live(&self, events: Vec<LiveEvent>) {
        self.send(MonitorMsg::Live(Arc::new(events))).await
    }

    pub async fn log_eval_plan(&self, env_name: String, step: u64, expected: u64) {
        self.send(MonitorMsg::EvalPlan {
            env_name,
            step,
            expected,
        })
        .await
    }

    pub async fn log_eval_epoch(&self, env_name: String, step: u64, episodes: Vec<EpisodeRef>) {
        self.send(MonitorMsg::EvalEpoch {
            env_name,
            step,
            episodes,
        })
        .await
    }

    /// Finalize every monitor and stop the actor. Only on a clean exit: a crashed run is
    /// reported crashed when the actor stops without it.
    pub async fn finalize(&self, paused: bool) {
        let (done, wait) = oneshot::channel();
        self.send(MonitorMsg::Finalize { paused, done }).await;
        let _ = wait.await;
    }
}

/// Run one call on every monitor, warning about (and skipping) a failing one.
macro_rules! fan_out {
    ($monitors:expr, $verb:literal, |$m:ident| $call:expr) => {
        for $m in $monitors.iter_mut() {
            if let Err(e) = $call.await {
                tracing::warn!("Failed to {} {}: {e:#}", $verb, $m.name());
            }
        }
    };
}

async fn handle(
    monitors: &mut [Box<dyn Monitor>],
    msg: MonitorMsg,
) -> Option<(bool, oneshot::Sender<()>)> {
    match msg {
        MonitorMsg::Metrics { row, step } => {
            fan_out!(monitors, "log to", |m| m.log_metrics(&row, step))
        }
        MonitorMsg::Episodes {
            episodes,
            step,
            kind,
            subset,
        } => {
            fan_out!(monitors, "log to", |m| m
                .log_episodes(&episodes, step, kind, subset))
        }
        MonitorMsg::Annotations(updates) => {
            fan_out!(monitors, "log to", |m| m.log_annotations(&updates))
        }
        MonitorMsg::Live(events) => fan_out!(monitors, "log to", |m| m.log_live(&events)),
        MonitorMsg::EvalPlan {
            env_name,
            step,
            expected,
        } => {
            fan_out!(monitors, "log to", |m| m
                .log_eval_plan(&env_name, step, expected))
        }
        MonitorMsg::EvalEpoch {
            env_name,
            step,
            episodes,
        } => {
            fan_out!(monitors, "log to", |m| m
                .log_eval_epoch(&env_name, step, &episodes))
        }
        MonitorMsg::Finalize { paused, done } => return Some((paused, done)),
    }
    None
}

async fn run(
    mut monitors: Vec<Box<dyn Monitor>>,
    mut rx: mpsc::Receiver<MonitorMsg>,
    cancel: CancellationToken,
) -> Result<()> {
    loop {
        let msg = tokio::select! {
            msg = rx.recv() => msg,
            () = cancel.cancelled() => None,
        };
        let Some(msg) = msg else { break };
        if let Some((paused, done)) = handle(&mut monitors, msg).await {
            fan_out!(monitors, "finalize", |m| m.finalize(paused));
            let _ = done.send(());
            return Ok(());
        }
    }
    // Stopped without a finalize: what producers already handed over still goes out, then
    // each monitor reports the unclean exit.
    while let Ok(msg) = rx.try_recv() {
        if let Some((_, done)) = handle(&mut monitors, msg).await {
            let _ = done.send(());
        }
    }
    fan_out!(monitors, "finalize", |m| m.crash());
    Ok(())
}

/// The Prime monitor a component runs.
pub enum PrimeSetup {
    /// `PrimeTrainMonitorConfig`: `name` from the config, `run` the registration (None for a
    /// config-less init).
    Train {
        name: Option<String>,
        run: Option<PrimeTrainSetup>,
    },
    /// `PrimeEvalMonitorConfig` (eval processes).
    Eval(prime::eval::PrimeEvalSetup),
}

/// `monitors.setup(...)`: one monitor per configured block.
pub struct MonitorsSetup {
    pub output_dir: PathBuf,
    /// Stamped on metrics rows and naming the annotation stream (`"orch"`).
    pub producer: Option<String>,
    pub prime: Option<PrimeSetup>,
    pub wandb: Option<Box<dyn WandbBridge>>,
    pub file: Option<FileMonitorConfig>,
    pub train_env_names: Vec<String>,
    pub eval_env_names: Vec<String>,
    /// `"rl"`, `"sft"` or `"eval"`: which curated W&B overview to provision.
    pub overview_flavor: String,
}

impl MonitorsSetup {
    /// The orchestrator's `monitors.setup(producer="orch", ...)`.
    pub fn from_orchestrator(
        config: &prl_config::OrchestratorConfig,
        wandb: Option<Box<dyn WandbBridge>>,
    ) -> Self {
        let sources = &config.train.source;
        let prime = config
            .monitors
            .prime
            .as_ref()
            .map(|prime| PrimeSetup::Train {
                name: prime.name.clone(),
                run: Some(PrimeTrainSetup {
                    name: prime.name.clone(),
                    model: config.model.name.clone(),
                    environments: sources.iter().map(|s| s.env_id.clone()).collect(),
                    training: prime::backend::TrainingSpec {
                        max_steps: config.max_steps.unwrap_or(0),
                        batch_size: Some(config.batch_size),
                        rollouts_per_example: sources.iter().map(|s| s.group_size).max(),
                        seq_len: Some(config.seq_len),
                        wandb_project: config.monitors.wandb.as_ref().map(|w| w.project.clone()),
                    },
                    run_config: config
                        .prime_run_config
                        .as_object()
                        .cloned()
                        .unwrap_or_default(),
                }),
            });
        MonitorsSetup {
            output_dir: config.output_dir.clone(),
            producer: Some("orch".into()),
            prime,
            wandb,
            file: config.monitors.file.as_ref().map(FileMonitorConfig::from),
            train_env_names: sources.iter().map(|s| s.resolved_name.clone()).collect(),
            eval_env_names: config.eval.as_ref().map_or_else(Vec::new, |e| {
                e.source.iter().map(|s| s.resolved_name.clone()).collect()
            }),
            overview_flavor: "rl".into(),
        }
    }
}

/// Construct and initialize every configured monitor (only on rank 0), then start the actor.
pub async fn spawn(
    setup: MonitorsSetup,
    cancel: CancellationToken,
) -> Result<(Monitors, JoinHandle<Result<()>>)> {
    let rank: u64 = std::env::var("RANK")
        .or_else(|_| std::env::var("DP_RANK"))
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(0);
    let mut monitors: Vec<Box<dyn Monitor>> = Vec::new();
    if rank == 0 {
        let names: Vec<&str> = [
            setup.prime.as_ref().map(|_| "prime"),
            setup.wandb.as_ref().map(|_| "wandb"),
            setup.file.as_ref().map(|_| "file"),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !names.is_empty() {
            tracing::info!("Initializing monitors ({})", names.join(", "));
        }
        let t0 = std::time::Instant::now();
        match setup.prime {
            Some(PrimeSetup::Train { name, run }) => {
                tracing::info!("Initializing prime monitor (name={})", py_opt(&name));
                let output_dir = Some(setup.output_dir.clone());
                monitors.push(Box::new(
                    PrimeTrainMonitor::init(run, name, output_dir).await?,
                ));
            }
            Some(PrimeSetup::Eval(eval)) => {
                tracing::info!("Initializing prime monitor (name={})", py_opt(&eval.name));
                let output_dir = Some(setup.output_dir.clone());
                monitors.push(Box::new(
                    prime::eval::PrimeEvalMonitor::init(eval, output_dir).await?,
                ));
            }
            None => {}
        }
        if let Some(bridge) = setup.wandb {
            tracing::info!("Initializing wandb monitor");
            let monitor = WandbMonitor::init(
                bridge,
                setup.train_env_names.clone(),
                setup.eval_env_names.clone(),
                setup.overview_flavor.clone(),
            )
            .await?;
            monitors.push(Box::new(monitor));
        }
        if let Some(config) = setup.file {
            tracing::info!(
                "Initializing file monitor (path=PosixPath('{}') chunk_bytes={} compress={} float_decimals={})",
                config.path.display(),
                config.chunk_bytes,
                if config.compress { "True" } else { "False" },
                config.float_decimals.map_or("None".into(), |d| d.to_string())
            );
            monitors.push(Box::new(
                file::FileMonitorActor::init(
                    config,
                    setup.output_dir.clone(),
                    setup.producer.clone(),
                )
                .await?,
            ));
        }
        if !names.is_empty() {
            tracing::debug!("Initialized monitors in {:.2}s", t0.elapsed().as_secs_f64());
        }
    }
    Ok(spawn_monitors(monitors, cancel))
}

/// Start the actor over already-initialized monitors.
pub fn spawn_monitors(
    monitors: Vec<Box<dyn Monitor>>,
    cancel: CancellationToken,
) -> (Monitors, JoinHandle<Result<()>>) {
    let (tx, rx) = mpsc::channel(INBOX);
    (Monitors { tx }, tokio::spawn(run(monitors, rx, cancel)))
}

fn py_opt(value: &Option<String>) -> String {
    value.as_ref().map_or("None".into(), |v| format!("'{v}'"))
}
