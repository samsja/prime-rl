//! The orchestrator supervisor (`orchestrator.py`): setup, actor wiring, the step loop's
//! lifecycle (drain -> final broadcast -> final checkpoint -> finalize) and bounded teardown.
//! Any actor that exits early fails the run, like `_on_component_done`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use indexmap::IndexMap;
use prl_config::{AdmissionGateConfig, ModelReference, OrchestratorConfig, Split, TaskSamplerConfig, TrainSourceConfig};
use prl_io::ckpt::CheckpointManager;
use prl_io::clients::{connect_frozen_client, AdminPlane, ClientType, InferenceClient};
use prl_io::control::Controller;
use prl_io::heartbeat::Heartbeat;
use prl_io::inference_metrics::InferenceMetricsCollector;
use prl_io::logging::format_time;
use prl_io::weights::{NixlBridge, WeightReceiver};
use prl_monitors::{Monitors, MonitorsSetup, WandbBridge};
use prl_train::algo::{build_algorithm, AlgorithmDeps};
use prl_train::{BatchPacker, BatchSender, FlopTable, PayloadReleaser, SinkConfig, TrainEnvSpec, TrainSink};
use prl_types::{Bridge, DispatchResult, Kind, Metrics, Policy, Progress, RunIdentity, Scorer};
use prl_wire::Episode;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::concurrency::{ConcurrencyController, ConcurrencyHandle, ConcurrencyIo, ConcurrencyParams};
use crate::curriculum::{AdvRangeGate, CurriculumConfig, DifficultyPool, SamplerConfig, Tasks};
use crate::dispatcher::{DispatchEnv, Dispatcher, DispatcherConfig, DispatcherHandle, DispatcherIo, EvalSide, TrainSide};
use crate::envs::{config_dir, replica_address_files, start_env, StartedEnv};
use crate::eval_pipeline::{EvalEnvShape, EvalPipeline, EvalPipelineConfig, EvalProgress};
use crate::eval_source::{EvalEnvTasks, EvalSource};
use crate::fmt::pct;
use crate::lag::{self, LagHandle};
use crate::periodic::PeriodicLogger;
use crate::train_pipeline::{wait_for_version, TrainPipeline, TrainPipelineConfig, TrainPipelineHandle, TrainPipelineInbox, TrainView};
use crate::train_source::{TrainEnvTasks, TrainSource};
use crate::watcher::WeightWatcher;

/// Wall-clock budget for teardown; the process force-exits past it.
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(300);

type Components = JoinSet<(&'static str, Result<()>)>;
type Results = mpsc::Receiver<DispatchResult<Episode>>;

fn spawn_component(
    tasks: &mut Components,
    name: &'static str,
    fut: impl std::future::Future<Output = Result<()>> + Send + 'static,
) {
    tasks.spawn(async move { (name, fut.await) });
}

fn curriculum_config(source: &TrainSourceConfig) -> CurriculumConfig {
    let Some(config) = &source.curriculum else { return CurriculumConfig::default() };
    let sampler = match &config.sampler {
        TaskSamplerConfig::Standard => SamplerConfig::Standard,
        TaskSamplerConfig::DifficultyPool(c) => SamplerConfig::DifficultyPool {
            pools: c
                .pools
                .iter()
                .map(|(name, p)| (name.clone(), DifficultyPool { threshold: p.threshold, weight: p.weight }))
                .collect(),
            seed: c.seed.unsigned_abs(),
        },
    };
    let gates = config
        .gates
        .iter()
        .map(|(name, AdmissionGateConfig::AdvantageRange(g))| {
            (name.clone(), AdvRangeGate { reject_min: g.reject_min, reject_max: g.reject_max })
        })
        .collect();
    CurriculumConfig { sampler, gates }
}

/// The vf `SamplingConfig` of train requests (`TrainEnv.__init__`).
fn train_sampling(source: &TrainSourceConfig, live: bool) -> Result<prl_wire::Map> {
    let s = &source.sampling;
    let sampling = prl_wire::sampling::TrainSampling {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k.map(|k| k as i64),
        max_completion_tokens: s.max_completion_tokens.map(|k| k as i64),
        logprobs: s.logprobs.map(|k| k as i64),
        extra_body: json_map(&s.extra_body)?,
    };
    Ok(prl_wire::sampling::train_sampling(&sampling, live))
}

fn json_map(value: &serde_json::Value) -> Result<prl_wire::Map> {
    serde_json::from_value::<prl_wire::Value>(value.clone())?
        .into_map()
        .ok_or_else(|| anyhow!("expected a mapping, got {value}"))
}

/// The micro-batch packer with the trainer model's FLOP cost table (from the bridge).
async fn build_packer(config: &OrchestratorConfig, bridge: &Arc<dyn Bridge>) -> Result<BatchPacker> {
    let (seq_len, multiple) = (config.seq_len as usize, config.pad_to_multiple_of as usize);
    let max_len = if config.pad_to_seq_len { seq_len } else { FlopTable::max_len(seq_len, multiple) };
    let table = FlopTable::new(bridge.flop_cost_table(max_len).await?);
    Ok(BatchPacker::new(seq_len, config.num_train_workers as usize, multiple, config.pad_to_seq_len, table))
}

/// Connect an env per source (concurrently), logging the phase like Python.
async fn start_envs(config: &OrchestratorConfig, kind: Kind, names: &[String], bridge: &Arc<dyn Bridge>) -> Result<Vec<StartedEnv>> {
    info!("Loading {} environments ({})", kind.as_str(), names.join(", "));
    let t0 = Instant::now();
    let split = if kind == Kind::Train { Split::Train } else { Split::Eval };
    let dir = config_dir(&config.output_dir);
    let envs = futures::future::try_join_all(names.iter().map(|name| {
        let address = config.env_addresses.get(&(split, name.clone())).cloned().flatten();
        let files = replica_address_files(&dir, kind, name, config.env_server_replicas);
        start_env(kind, name, address, files, bridge.clone())
    }))
    .await?;
    let label = if kind == Kind::Train { "Train" } else { "Eval" };
    prl_io::success!("{label} environments ready in {}", format_time(t0.elapsed().as_secs_f64()));
    Ok(envs)
}

/// Read-only handles the pipeline-view logger polls.
#[derive(Clone)]
struct PipelineView {
    dispatcher: DispatcherHandle,
    concurrency: ConcurrencyHandle,
    lag: LagHandle,
    watcher: watch::Receiver<Metrics>,
    train: watch::Receiver<TrainView>,
    eval: Option<watch::Receiver<EvalProgress>>,
    gc: Option<Metrics>,
    train_envs: Vec<String>,
    eval_envs: Vec<String>,
}

impl PipelineView {
    /// `collect_pipeline_view`: `(console body, payload)`.
    async fn collect(&self) -> (String, Metrics) {
        let Ok(disp) = self.dispatcher.view().await else { return ("Dispatcher stopped".into(), Metrics::new()) };
        let train = self.train.borrow().clone();
        let eval = self.eval.as_ref().map(|e| e.borrow().clone()).unwrap_or_default();
        let multi_train = self.train_envs.len() > 1;
        let multi_eval = self.eval_envs.len() > 1;

        let train_pct = if train.target > 0 { train.batch as f64 / train.target as f64 } else { 0.0 };
        let mut body = format!("Train batch {}/{} ({})", train.batch, train.target, pct(train_pct));
        if multi_train {
            let pairs: Vec<String> = self
                .train_envs
                .iter()
                .map(|e| format!("{e}={}", train.batch_by_env.get(e).copied().unwrap_or(0)))
                .collect();
            body += &format!(" ({})", pairs.join(", "));
        }
        if train.buffered > 0 {
            body += &format!(" (+{} buffered)", train.buffered);
        }
        for (env, _step, arrived, expected) in &eval {
            let p = if *expected > 0 { *arrived as f64 / *expected as f64 } else { 0.0 };
            body += &format!(" | {env} {arrived}/{expected} ({})", pct(p));
        }
        let mut inflight = format!(
            "{} inflight episodes (train={}, eval={}",
            disp.inflight_train + disp.inflight_eval,
            disp.inflight_train,
            disp.inflight_eval
        );
        if multi_train || multi_eval {
            let count = |kind: Kind, e: &String| disp.inflight_by_env.get(&(kind, e.clone())).copied().unwrap_or(0);
            let mut pairs: Vec<String> =
                self.train_envs.iter().map(|e| format!("{e}={}", count(Kind::Train, e))).collect();
            pairs.extend(self.eval_envs.iter().map(|e| format!("{e}={}", count(Kind::Eval, e))));
            inflight += &format!(" | {}", pairs.join(", "));
        }
        inflight += ")";
        body += &format!("; {inflight}");

        let mut payload = disp.gauges;
        payload.extend(disp.drained);
        payload.extend(self.watcher.borrow().clone());
        payload.extend(self.concurrency.gauges().await.unwrap_or_default());
        if let Some(gc) = &self.gc {
            payload.extend(gc.clone());
        }
        payload.extend(self.lag.metrics().await);
        (body, payload)
    }
}

/// Run the orchestrator until it exits. `shutdown` (SIGINT) cancels it like Python's main task.
pub async fn run(
    config: OrchestratorConfig,
    bridge: Arc<dyn Bridge>,
    wandb: Option<Box<dyn WandbBridge>>,
    nixl: Option<Box<dyn NixlBridge>>,
    shutdown: CancellationToken,
) -> Result<()> {
    let cancel = CancellationToken::new();
    let mut tasks = Components::new();
    let setup = tokio::select! {
        r = Orchestrator::setup(config, bridge, wandb, nixl, &mut tasks, cancel.clone()) => r,
        _ = shutdown.cancelled() => Err(anyhow!("interrupted during setup")),
    };
    match setup {
        Ok(orchestrator) => orchestrator.start(tasks, shutdown).await,
        Err(err) => {
            cancel.cancel();
            while tasks.join_next().await.is_some() {}
            Err(err)
        }
    }
}

struct Orchestrator {
    config: OrchestratorConfig,
    cancel: CancellationToken,
    metrics_cancel: CancellationToken,
    dispatcher: DispatcherHandle,
    train: TrainPipelineHandle,
    train_pipeline: Option<(TrainPipeline, TrainPipelineInbox, Results)>,
    watcher: Option<WeightWatcher>,
    applied: watch::Receiver<u64>,
    idle: watch::Receiver<bool>,
    monitors: Monitors,
    view: PipelineView,
    controller: Option<Controller>,
    ckpt: CheckpointManager,
    envs: Vec<StartedEnv>,
}

impl Orchestrator {
    /// `Orchestrator.setup()`: same order and log lines as Python.
    async fn setup(
        config: OrchestratorConfig,
        bridge: Arc<dyn Bridge>,
        wandb: Option<Box<dyn WandbBridge>>,
        nixl: Option<Box<dyn NixlBridge>>,
        tasks: &mut Components,
        cancel: CancellationToken,
    ) -> Result<Self> {
        info!("Initializing policy inference pool ({})", config.reprs.model);
        let policy_client = Arc::new(InferenceClient::new(
            &config.model.client,
            &config.model.name,
            ClientType::Renderer,
            ClientType::ChatCompletions,
            Some(&config.renderer),
        )?);
        let roles = config.inference_metrics_roles.clone();
        let mut admin = AdminPlane::setup(&config.model.client, &config.model.name, roles)?;

        let (monitors, monitors_task) =
            prl_monitors::spawn(MonitorsSetup::from_orchestrator(&config, wandb), cancel.child_token()).await?;
        tasks.spawn(async move { ("monitors", monitors_task.await.map_err(anyhow::Error::from).and_then(|r| r)) });

        // The launcher-set $PRL_RUN_ID is the run identity; standalone runs mint one.
        let env_var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        let run = RunIdentity {
            id: env_var("PRL_RUN_ID").unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()),
            name: env_var("PRL_RUN_NAME"),
        };
        let heartbeat = config.heartbeat.as_ref().map(|h| Arc::new(Heartbeat::new(h.url.clone(), h.min_interval)));
        let ckpt = CheckpointManager::new(&config.output_dir);

        let deps = AlgorithmDeps {
            policy: policy_client.clone() as Arc<dyn Scorer>,
            policy_model_name: policy_client.model_name.clone(),
            connect_frozen: Arc::new(|model| {
                Box::pin(async move { Ok(Arc::new(connect_frozen_client(&model, None).await?) as Arc<dyn Scorer>) })
            }),
            bridge: Some(bridge.clone()),
        };
        let mut algorithms = Vec::new();
        for source in &config.train.source {
            info!("Initializing {} algorithm for {}", source.algo.params.tag(), source.resolved_name);
            algorithms.push(build_algorithm(&source.algo, &deps)?);
        }

        let resume_step = match &config.resume {
            None => None,
            Some(resume) if resume.dir.is_some() => resume.dir_step,
            Some(resume) => match resume.step {
                Some(step) => Some(step),
                None => prl_io::pathing::resolve_latest_ckpt_step(&prl_io::pathing::ckpt_dir(&config.output_dir))?,
            },
        };
        let mut progress = Progress::default();
        match resume_step {
            Some(step) => {
                progress.step = step + 1;
                info!("Resuming from step {step}");
            }
            None => info!("Starting from scratch"),
        }

        let packer = Arc::new(build_packer(&config, &bridge).await?);
        info!("Initializing micro batch sender ({})", config.reprs.rollout_transport);
        let workers = config.num_train_workers as usize;
        let sender = BatchSender::new(config.output_dir.clone(), workers, progress.step, &config.rollout_transport).await?;

        let train_names: Vec<String> = config.train.source.iter().map(|s| s.resolved_name.clone()).collect();
        let mut train_envs = start_envs(&config, Kind::Train, &train_names, &bridge).await?;
        let eval_config = config.eval.clone();
        let eval_sources = eval_config.as_ref().map(|e| e.source.clone()).unwrap_or_default();
        let eval_names: Vec<String> = eval_sources.iter().map(|s| s.resolved_name.clone()).collect();
        let eval_envs =
            if eval_names.is_empty() { Vec::new() } else { start_envs(&config, Kind::Eval, &eval_names, &bridge).await? };
        let eval_examples: Vec<_> = eval_envs.iter().map(StartedEnv::examples).collect::<Result<_>>()?;

        let mut source_envs = Vec::new();
        for (source, env) in config.train.source.iter().zip(train_envs.iter_mut()) {
            source_envs.push(TrainEnvTasks {
                name: source.resolved_name.clone(),
                ratio: source.ratio,
                curriculum: curriculum_config(source),
                tasks: std::mem::replace(&mut env.tasks, Tasks::Finite(Vec::new())),
            });
        }
        let mut train_source = TrainSource::new(source_envs)?;
        if let Some(step) = resume_step {
            let path = config.resume.as_ref().and_then(|r| r.dir.as_ref()).map(|dir| dir.join("orchestrator"));
            let state = ckpt.load(step, path.as_deref())?;
            progress = state.progress.clone();
            train_source.load(&state.train_source).await?;
            progress.step = step + 1;
        }

        info!("Waiting for policy inference pool to be ready");
        let t0 = Instant::now();
        admin.wait_for_ready(&config.model.name).await?;
        prl_io::success!("Policy inference pool ready after {}", format_time(t0.elapsed().as_secs_f64()));
        // Frozen generation sources and the algorithms' reference pools.
        let frozen = futures::future::try_join_all(config.train.source.iter().map(|s| async {
            match &s.algo.sampling.source {
                ModelReference::Frozen(model) => {
                    Ok::<_, anyhow::Error>(Some(Arc::new(connect_frozen_client(model, Some(&config.renderer)).await?)))
                }
                ModelReference::Policy => Ok(None),
            }
        }))
        .await?;
        futures::future::try_join_all(algorithms.iter_mut().map(|a| a.setup())).await?;

        let admin_clients = admin.clients().to_vec();
        info!("Initializing weight broadcast ({})", config.reprs.weight_broadcast);
        let t0 = Instant::now();
        let broadcast_dir = prl_io::pathing::broadcast_dir(&config.output_dir);
        let wb = config.weight_broadcast.clone();
        let mut receiver = WeightReceiver::new(broadcast_dir, wb, admin, config.model.name.clone(), nixl)?;
        receiver.initialize().await?;
        debug!("Initialized weight broadcast in {}", format_time(t0.elapsed().as_secs_f64()));

        // Shared values.
        let (policy_tx, policy_rx) = watch::channel(Policy { version: 0, model_name: policy_client.model_name.clone() });
        // Written once a policy update's eval trigger and dispatcher resume ran: version waits
        // use it so they never overtake the eval trigger of that version.
        let (applied_tx, applied_rx) = watch::channel(0u64);
        let (step_tx, step_rx) = watch::channel(progress.step);

        let concurrency = ConcurrencyController::new(
            ConcurrencyParams {
                min_inflight: config.concurrency.min_inflight as usize,
                max_inflight: config.concurrency.max_inflight.map(|m| m as usize),
                initial_inflight: config.concurrency.initial_inflight.map(|m| m as usize),
            },
            config.seq_len,
        );
        let (limit_tx, limit_rx) = watch::channel(concurrency.max_inflight);
        let (inflight_tx, inflight_rx) = watch::channel(0usize);
        let (overload_tx, overload_rx) = mpsc::unbounded_channel();
        let (load_tx, load_rx) = mpsc::channel(16);
        let io = ConcurrencyIo { samples: load_rx, inflight: inflight_rx, limit: limit_tx, overload: Some(overload_tx) };
        let (concurrency, concurrency_task) = concurrency.spawn(io, cancel.child_token());
        tasks.spawn(async move { ("concurrency", concurrency_task.await.map_err(anyhow::Error::from).and_then(|r| r)) });

        // Results channels, bounded like Python's out_q.
        let bound = config.concurrency.max_inflight.map_or(1 << 20, |m| (m as usize).max(8));
        let (train_results_tx, train_results_rx) = mpsc::channel(bound);
        let (eval_results_tx, eval_results_rx) = mpsc::channel(bound);

        let (source, source_task) = train_source.spawn();
        tasks.spawn(async move { ("train_source", source_task.await.map_err(anyhow::Error::from)) });

        let mut train_dispatch = IndexMap::new();
        let mut env_specs = IndexMap::new();
        for (((source_config, env), frozen), algorithm) in
            config.train.source.iter().zip(&train_envs).zip(frozen).zip(algorithms)
        {
            let live = frozen.is_none();
            let clients = frozen.unwrap_or_else(|| policy_client.clone());
            let name = source_config.resolved_name.clone();
            let sampling = &source_config.sampling;
            env_specs.insert(
                name.clone(),
                TrainEnvSpec {
                    group_size: source_config.group_size as usize,
                    temperature: sampling.temperature as f32,
                    requires_sampling_masks: sampling.truncates_distribution && live,
                    requires_top_logprobs: sampling.logprobs.is_some() && live,
                    algorithm,
                },
            );
            train_dispatch.insert(
                name.clone(),
                DispatchEnv {
                    name,
                    group_size: source_config.group_size as usize,
                    runner: env.router.clone(),
                    sampling: train_sampling(source_config, live)?,
                    client: serde_json::to_value(&clients.train_client)?,
                    frozen_model: (!live).then(|| clients.model_name.clone()),
                    sessions: clients,
                },
            );
        }

        let eval_side = match &eval_config {
            Some(eval) if !eval_envs.is_empty() => {
                let mut envs = IndexMap::new();
                let mut tasks_by_env = Vec::new();
                for ((s, env), examples) in eval_sources.iter().zip(&eval_envs).zip(&eval_examples) {
                    let interval = *eval
                        .intervals
                        .get(&s.resolved_name)
                        .ok_or_else(|| anyhow!("No eval interval configured for env {:?}", s.resolved_name))?;
                    tasks_by_env.push(EvalEnvTasks {
                        name: s.resolved_name.clone(),
                        tasks: examples.clone(),
                        group_size: s.group_size as usize,
                        interval,
                    });
                    envs.insert(
                        s.resolved_name.clone(),
                        DispatchEnv {
                            name: s.resolved_name.clone(),
                            group_size: s.group_size as usize,
                            runner: env.router.clone(),
                            sampling: json_map(&s.sampling)?,
                            client: serde_json::to_value(&policy_client.eval_client)?,
                            frozen_model: None,
                            sessions: policy_client.clone(),
                        },
                    );
                }
                let source = EvalSource::new(tasks_by_env, eval.skip_first_step, resume_step.is_some());
                Some(EvalSide { envs, source, results: eval_results_tx })
            }
            _ => None,
        };

        let (idle_tx, idle_rx) = watch::channel(false);
        let (dispatcher, dispatcher_handle, dispatcher_inbox) = Dispatcher::new(
            DispatcherConfig {
                dispatch_per_minute: config.dispatch_per_minute.map(|d| d as f64),
                run,
                policy_weights_change: true,
            },
            Some(TrainSide {
                envs: train_dispatch,
                source: source.clone(),
                progress_step: step_rx,
                results: train_results_tx,
            }),
            eval_side,
            DispatcherIo {
                policy: policy_rx,
                limit: limit_rx,
                inflight: inflight_tx,
                overload: overload_rx,
                concurrency: Some(concurrency.clone()),
                live: monitors.clone(),
                idle: idle_tx,
            },
        );
        spawn_component(tasks, "dispatcher", dispatcher.run(dispatcher_inbox, cancel.child_token()));

        let controller = config.control.as_ref().map(|_| Controller::new(true));
        let sink = TrainSink::new(
            SinkConfig {
                batch_size: config.batch_size as usize,
                max_staleness: config.max_staleness as i64,
                constant_trainer_batch_size: config.constant_trainer_batch_size,
            },
            env_specs,
            PayloadReleaser::new(prl_wire::payload::delete_payloads),
        );
        let (train_pipeline, train, train_inbox, train_view) = TrainPipeline::new(
            TrainPipelineConfig {
                max_steps: config.max_steps,
                ckpt_interval: config.ckpt.as_ref().and_then(|c| c.interval),
                ckpt_enabled: config.ckpt.is_some(),
                weight_broadcast_timeout: Duration::from_secs(config.weight_broadcast.timeout()),
                train_env_names: train_names.clone(),
            },
            sink,
            source,
            progress,
            applied_rx.clone(),
            dispatcher_handle.clone(),
            packer,
            sender,
            ckpt.clone(),
            controller.clone(),
            monitors.clone(),
            heartbeat,
            step_tx,
        );

        // The collector always polls (it feeds the concurrency controller); its metrics fan out
        // to the monitors when collection is on.
        let metrics_cancel = cancel.child_token();
        let (metrics_tx, mut metrics_rx) = mpsc::channel::<Metrics>(16);
        let mut collector = InferenceMetricsCollector::new(
            &admin_clients,
            config.inference_metrics_roles.as_deref(),
            config.collect_inference_metrics,
            Some(load_tx),
            Some(metrics_tx),
        )?;
        // One awaited scrape so the controller derives its initial limit before the loop starts.
        collector.probe(1, Duration::ZERO).await;
        tokio::spawn(collector.run(metrics_cancel.clone()));
        let metrics_monitors = monitors.clone();
        tokio::spawn(async move {
            while let Some(row) = metrics_rx.recv().await {
                metrics_monitors.log_metrics(row, None).await;
            }
        });

        let mut eval_progress = None;
        let mut eval_handle = None;
        if let Some(eval) = eval_config.as_ref().filter(|_| !eval_envs.is_empty()) {
            let shapes = eval_sources
                .iter()
                .zip(&eval_examples)
                .map(|(s, ex)| (s.resolved_name.clone(), EvalEnvShape { group_size: s.group_size as usize, num_examples: ex.len() }))
                .collect();
            let (pipeline, handle, progress, inbox) = EvalPipeline::new(
                EvalPipelineConfig {
                    envs: shapes,
                    resume_step,
                    retrigger_on_resume: eval.retrigger_on_resume,
                    max_steps: config.max_steps,
                },
                dispatcher_handle.clone(),
                monitors.clone(),
            );
            spawn_component(tasks, "eval_pipeline", pipeline.run(inbox, eval_results_rx, cancel.child_token()));
            eval_handle = Some(handle);
            eval_progress = Some(progress);
        }

        let sync_version = resume_step.unwrap_or(0);
        let (gauges_tx, gauges_rx) = watch::channel(Metrics::new());
        let mut watcher = WeightWatcher {
            receiver,
            policy: policy_tx,
            applied: applied_tx,
            gauges: gauges_tx,
            dispatcher: dispatcher_handle.clone(),
            eval: eval_handle,
            ckpt_step: sync_version,
            last_update_weights_time: 0.0,
            last_wait_for_ckpt_time: 0.0,
            update_count: 0,
        };

        let lag = lag::spawn(cancel.child_token());
        let view = PipelineView {
            dispatcher: dispatcher_handle.clone(),
            concurrency,
            lag,
            watcher: gauges_rx,
            train: train_view,
            eval: eval_progress,
            gc: config.gc.as_ref().map(|gc| prl_io::gc::GarbageCollector::new(gc.interval).gauges()),
            train_envs: train_names,
            eval_envs: eval_names,
        };

        info!("Syncing inference to the trainer's startup broadcast (v{sync_version})");
        let t0 = Instant::now();
        watcher.sync_startup(sync_version, Duration::from_secs(config.weight_broadcast.timeout())).await?;
        debug!("Synced inference to policy v{sync_version} in {}", format_time(t0.elapsed().as_secs_f64()));

        Ok(Self {
            config,
            cancel,
            metrics_cancel,
            dispatcher: dispatcher_handle,
            train,
            train_pipeline: Some((train_pipeline, train_inbox, train_results_rx)),
            watcher: Some(watcher),
            applied: applied_rx,
            idle: idle_rx,
            monitors,
            view,
            controller,
            ckpt,
            envs: train_envs.into_iter().chain(eval_envs).collect(),
        })
    }

    async fn start(mut self, mut tasks: Components, shutdown: CancellationToken) -> Result<()> {
        let max_steps = self.config.max_steps.map_or_else(|| "infinite".to_string(), |m| m.to_string());
        info!("Starting orchestrator loop (max_steps={max_steps})");
        let start = Instant::now();

        let periodic = {
            let view = self.view.clone();
            PeriodicLogger::spawn(Duration::from_secs_f64(self.config.log.interval), self.monitors.clone(), move || {
                let view = view.clone();
                async move { view.collect().await }
            })
        };
        let watcher = self.watcher.take().expect("watcher");
        spawn_component(&mut tasks, "watcher", watcher.run(self.cancel.child_token()));
        let (pipeline, inbox, results) = self.train_pipeline.take().expect("train pipeline");
        spawn_component(&mut tasks, "train_pipeline", pipeline.run(inbox, results, self.cancel.child_token()));
        if let (Some(control), Some(controller)) = (&self.config.control, &self.controller) {
            let (controller, host, port, cancel) =
                (controller.clone(), control.host.clone(), control.port, self.cancel.child_token());
            spawn_component(&mut tasks, "control", async move {
                prl_io::control::serve(controller, &host, port, cancel).await
            });
        }
        self.dispatcher.start()?;

        let loop_result = self.step_loop(&mut tasks, &shutdown).await;
        let loop_done = loop_result.is_ok();
        let clean_exit = match loop_result {
            Ok(()) => {
                let max = self.config.max_steps;
                let timeout = Duration::from_secs(self.config.weight_broadcast.timeout());
                match max {
                    None => true,
                    Some(max) => {
                        let mut applied = self.applied.clone();
                        let wait = wait_for_version(&mut applied, max, "before shutdown", timeout, &self.cancel);
                        tokio::select! {
                            r = wait => r.is_ok(),
                            failure = next_failure(&mut tasks) => { log_failure(&failure); false }
                            _ = shutdown.cancelled() => false,
                        }
                    }
                }
            }
            Err(err) => {
                error!("{err:#}");
                false
            }
        };

        let elapsed = format_time(start.elapsed().as_secs_f64());
        // The final finished step is `progress.step - 1`; checkpointed once the step loop
        // finished (a crash mid-loop leaves no consistent state for that step).
        if loop_done && self.config.ckpt.is_some() {
            if let Some(state) = self.train.final_state().await? {
                let step = state.progress.step;
                info!("Saving final checkpoint at step {step}");
                self.ckpt.save(&state, step)?;
            }
        }
        if clean_exit {
            prl_io::success!("Orchestrator step loop done in {elapsed}");
            self.metrics_cancel.cancel();
            periodic.stop().await;
            self.monitors.finalize(false).await;
        } else {
            warn!("Orchestrator interrupted after {elapsed} — forcing cleanup (not a clean exit)");
            periodic.stop().await;
        }
        self.stop(tasks, clean_exit).await;
        if clean_exit {
            prl_io::success!("Orchestrator finished");
            Ok(())
        } else {
            warn!("Orchestrator cleanup complete (forced)");
            Err(anyhow!("orchestrator did not exit cleanly"))
        }
    }

    /// Until the pipeline drained (after the final step) — or a component failed / a signal
    /// arrived.
    async fn step_loop(&mut self, tasks: &mut Components, shutdown: &CancellationToken) -> Result<()> {
        let mut idle = self.idle.clone();
        tokio::select! {
            r = idle.wait_for(|idle| *idle) => {
                r.map_err(|_| anyhow!("dispatcher stopped"))?;
            }
            failure = next_failure(tasks) => {
                log_failure(&failure);
                return Err(failure.1.unwrap_err_or_exit(failure.0));
            }
            _ = shutdown.cancelled() => bail!("interrupted"),
        }
        // The train pipeline finishes its drain (and its log line) first.
        self.train.sync().await?;
        // Nothing is in flight and no eval work remains: stop the dispatcher so the sinks see
        // their result channels close, and let the eval pipeline consume what it was handed.
        self.dispatcher.stop().await?;
        let mut pending: Vec<&str> = vec!["dispatcher"];
        if self.view.eval.is_some() {
            pending.push("eval_pipeline");
        }
        while !pending.is_empty() {
            let exited = tokio::select! {
                exited = next_failure(tasks) => exited,
                _ = shutdown.cancelled() => bail!("interrupted"),
            };
            match exited {
                (name, Ok(())) if pending.contains(&name) => pending.retain(|n| *n != name),
                failure => {
                    log_failure(&failure);
                    return Err(failure.1.unwrap_err_or_exit(failure.0));
                }
            }
        }
        info!("Pipeline drained, exiting main loop");
        Ok(())
    }

    /// Bounded teardown of every component (`stop`).
    async fn stop(self, mut tasks: Components, clean_exit: bool) {
        info!("Stopping orchestrator components");
        let t0 = Instant::now();
        let teardown = async {
            debug!("Closing micro batch sender");
            let _ = self.train.stop().await;
            debug!("Stopping dispatcher");
            let _ = self.dispatcher.stop().await;
            debug!("Stopping weight watcher");
            debug!("Stopping inference metrics collector");
            self.metrics_cancel.cancel();
            self.cancel.cancel();
            while let Some(joined) = tasks.join_next().await {
                if let Ok((name, Err(err))) = joined {
                    debug!("Component {name} stopped with: {err:#}");
                }
            }
            debug!("Stopping train env clients");
            for env in &self.envs {
                env.close().await;
            }
        };
        if tokio::time::timeout(SHUTDOWN_TIMEOUT, teardown).await.is_err() {
            warn!(
                "Orchestrator shutdown did not complete within {}s; forcing process exit. Training artifacts are already persisted.",
                SHUTDOWN_TIMEOUT.as_secs()
            );
            std::process::exit(if clean_exit { 0 } else { 1 });
        }
        debug!("Stopped orchestrator components in {}", format_time(t0.elapsed().as_secs_f64()));
    }
}

trait ExitResult {
    fn unwrap_err_or_exit(self, name: &str) -> anyhow::Error;
}

impl ExitResult for Result<()> {
    fn unwrap_err_or_exit(self, name: &str) -> anyhow::Error {
        match self {
            Err(err) => err,
            Ok(()) => anyhow!("{name} exited unexpectedly"),
        }
    }
}

/// The next component to exit (any exit before teardown is a failure).
async fn next_failure(tasks: &mut Components) -> (&'static str, Result<()>) {
    match tasks.join_next().await {
        Some(Ok(done)) => done,
        Some(Err(err)) => ("component", Err(anyhow!("component task panicked: {err}"))),
        None => std::future::pending().await,
    }
}

fn log_failure((name, result): &(&'static str, Result<()>)) {
    match result {
        Err(err) => error!("Orchestrator component {name} failed: {err:#}"),
        Ok(()) => error!("Orchestrator component {name} exited unexpectedly"),
    }
}
