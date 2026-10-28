//! The orchestrator supervisor (`orchestrator.py`): setup, actor wiring, the step loop's
//! lifecycle (drain -> final broadcast -> final checkpoint -> finalize) and bounded teardown.
//!
//! Actors: dispatcher, train source, train pipeline (sink + ship), eval pipeline (sink +
//! triggers), weight watcher, concurrency controller, inference metrics collector, monitors,
//! control server, lag monitor and the pipeline-view logger. They share `policy` (watcher),
//! `progress_step` (train pipeline), `limit` (concurrency) and `inflight` (dispatcher) watch
//! channels. Any actor that exits early fails the run, like `_on_component_done`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use indexmap::IndexMap;
use prl_config::{AdmissionGateConfig, OrchestratorConfig, TaskSamplerConfig, TrainSourceConfig};
use prl_io::ckpt::CheckpointManager;
use prl_io::clients::{connect_frozen_client, AdminPlane, ClientType, InferenceClient};
use prl_io::control::Controller;
use prl_io::heartbeat::Heartbeat;
use prl_io::inference_metrics::InferenceMetricsCollector;
use prl_io::logging::format_time;
use prl_io::weights::{NixlBridge, WeightReceiver};
use prl_monitors::{Monitors, MonitorsSetup, WandbBridge};
use prl_types::{Bridge, Kind, Metrics, Policy, Progress, RunIdentity};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::concurrency::{ConcurrencyController, ConcurrencyHandle, ConcurrencyIo, ConcurrencyParams};
use crate::curriculum::{AdvRangeGate, CurriculumConfig, DifficultyPool, SamplerConfig};
use crate::dispatcher::{
    DispatchEnv, Dispatcher, DispatcherConfig, DispatcherHandle, DispatcherIo, EvalSide, TrainSide,
};
use crate::envs::{config_dir, replica_address_files, start_env, StartedEnv};
use crate::eval_pipeline::{EvalEnvShape, EvalPipeline, EvalPipelineConfig, EvalProgress};
use crate::eval_source::{EvalEnvTasks, EvalSource};
use crate::fmt::pct;
use crate::glue::ClientSessions;
use crate::lag::{self, LagHandle};
use crate::periodic::PeriodicLogger;
use crate::train_pipeline::{wait_for_version, TrainPipeline, TrainPipelineConfig, TrainView};
use crate::train_source::{TrainEnvTasks, TrainSource};
use crate::watcher::{UpdateHook, VersionObserver, WeightWatcher};

/// Wall-clock budget for teardown; the process force-exits past it.
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(300);

type Components = JoinSet<(&'static str, Result<()>)>;

fn spawn_component(
    tasks: &mut Components,
    name: &'static str,
    fut: impl std::future::Future<Output = Result<()>> + Send + 'static,
) {
    tasks.spawn(async move { (name, fut.await) });
}

/// The dispatcher pauses scheduling across a weight update.
#[async_trait]
impl VersionObserver for DispatcherHandle {
    fn name(&self) -> &str {
        "Dispatcher"
    }

    async fn on_version_pending(&self, _step: u64) -> Result<()> {
        self.version_pending().await
    }

    async fn on_new_version(&self, _step: u64) -> Result<()> {
        Ok(())
    }
}

/// Resumes the dispatcher after the update hooks ran, so an eval trigger switches the mode
/// before anything is scheduled on the new policy.
struct ResumeDispatch(DispatcherHandle);

#[async_trait]
impl UpdateHook for ResumeDispatch {
    async fn on_update(&self, _step: u64) -> Result<()> {
        self.0.version_applied().await
    }
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
        .map(|(name, gate)| match gate {
            AdmissionGateConfig::AdvantageRange(g) => {
                (name.clone(), AdvRangeGate { reject_min: g.reject_min, reject_max: g.reject_max })
            }
        })
        .collect();
    CurriculumConfig { sampler, gates }
}

fn train_sampling(source: &TrainSourceConfig, live: bool) -> Result<prl_wire::Map> {
    let s = &source.sampling;
    let extra_body = serde_json::from_value::<prl_wire::Value>(s.extra_body.clone())?
        .into_map()
        .unwrap_or_default();
    let sampling = prl_wire::sampling::TrainSampling {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k.map(|k| k as i64),
        max_completion_tokens: s.max_completion_tokens.map(|k| k as i64),
        logprobs: s.logprobs.map(|k| k as i64),
        extra_body,
    };
    Ok(prl_wire::sampling::train_sampling(&sampling, live))
}

fn json_map(value: &serde_json::Value) -> Result<prl_wire::Map> {
    serde_json::from_value::<prl_wire::Value>(value.clone())?
        .into_map()
        .ok_or_else(|| anyhow!("expected a mapping, got {value}"))
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
        payload.extend(self.lag.stats().await.metrics());
        (body, payload)
    }
}

/// Run the orchestrator until it exits.
pub async fn run(
    config: OrchestratorConfig,
    bridge: Arc<dyn Bridge>,
    wandb: Option<Box<dyn WandbBridge>>,
    nixl: Option<Box<dyn NixlBridge>>,
    shutdown: CancellationToken,
) -> Result<()> {
    let cancel = CancellationToken::new();
    let mut tasks = Components::new();
    let result = tokio::select! {
        r = Orchestrator::setup(config, bridge, wandb, nixl, &mut tasks, cancel.clone()) => r,
        _ = shutdown.cancelled() => Err(anyhow!("interrupted during setup")),
    };
    let orchestrator = match result {
        Ok(orchestrator) => orchestrator,
        Err(err) => {
            cancel.cancel();
            while tasks.join_next().await.is_some() {}
            return Err(err);
        }
    };
    orchestrator.start(tasks, shutdown).await
}

struct Orchestrator {
    config: OrchestratorConfig,
    cancel: CancellationToken,
    metrics_cancel: CancellationToken,
    dispatcher: DispatcherHandle,
    train: crate::train_pipeline::TrainPipelineHandle,
    train_pipeline: Option<(TrainPipeline, crate::train_pipeline::TrainPipelineInbox, mpsc::Receiver<prl_types::DispatchResult<prl_wire::Episode>>)>,
    watcher: Option<WeightWatcher<WeightReceiver>>,
    policy: watch::Receiver<Policy>,
    idle: watch::Receiver<bool>,
    monitors: Monitors,
    view: PipelineView,
    controller: Option<Controller>,
    ckpt: CheckpointManager,
    envs: Vec<StartedEnv>,
}

impl Orchestrator {
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
        let mut admin =
            AdminPlane::setup(&config.model.client, &config.model.name, config.inference_metrics_roles.clone())?;

        let (monitors, monitors_task) =
            prl_monitors::spawn(MonitorsSetup::from_orchestrator(&config, wandb), cancel.child_token()).await?;
        tasks.spawn(async move { ("monitors", monitors_task.await.map_err(anyhow::Error::from).and_then(|r| r)) });

        // The launcher-set $PRL_RUN_ID is the run identity; standalone runs mint one.
        let run = RunIdentity {
            id: std::env::var("PRL_RUN_ID").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()),
            name: std::env::var("PRL_RUN_NAME").ok().filter(|s| !s.is_empty()),
        };
        let heartbeat = config.heartbeat.as_ref().map(|h| Arc::new(Heartbeat::new(h.url.clone(), h.min_interval)));
        let config_dir = config_dir(&config.output_dir);
        let ckpt = CheckpointManager::new(&config.output_dir);

        let mut algorithms = Vec::new();
        for source in &config.train.source {
            info!("Initializing {} algorithm for {}", source.algo.params.tag(), source.resolved_name);
            algorithms.push(crate::train_glue::build_algorithm(&config, source, &policy_client, &bridge)?);
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

        let packer = Arc::new(crate::train_glue::build_packer(&config, &bridge).await?);
        info!("Initializing micro batch sender ({})", config.reprs.rollout_transport);
        let sender = prl_train::BatchSender::new(
            config.output_dir.clone(),
            config.num_train_workers as usize,
            progress.step,
            &config.rollout_transport,
        )
        .await?;

        let train_names: Vec<String> = config.train.source.iter().map(|s| s.resolved_name.clone()).collect();
        info!("Loading train environments ({})", train_names.join(", "));
        let t0 = Instant::now();
        let train_envs = futures::future::try_join_all(config.train.source.iter().map(|s| {
            let address = config.env_addresses.get(&(prl_config::Split::Train, s.resolved_name.clone())).cloned().flatten();
            let files = replica_address_files(&config_dir, Kind::Train, &s.resolved_name, config.env_server_replicas);
            start_env(Kind::Train, &s.resolved_name, address, files, bridge.clone())
        }))
        .await?;
        prl_io::success!("Train environments ready in {}", format_time(t0.elapsed().as_secs_f64()));

        let eval_sources = config.eval.as_ref().map(|e| e.source.clone()).unwrap_or_default();
        let mut eval_envs = Vec::new();
        if !eval_sources.is_empty() {
            let names: Vec<String> = eval_sources.iter().map(|s| s.resolved_name.clone()).collect();
            info!("Loading eval environments ({})", names.join(", "));
            let t0 = Instant::now();
            eval_envs = futures::future::try_join_all(eval_sources.iter().map(|s| {
                let address =
                    config.env_addresses.get(&(prl_config::Split::Eval, s.resolved_name.clone())).cloned().flatten();
                let files = replica_address_files(&config_dir, Kind::Eval, &s.resolved_name, config.env_server_replicas);
                start_env(Kind::Eval, &s.resolved_name, address, files, bridge.clone())
            }))
            .await?;
            prl_io::success!("Eval environments ready in {}", format_time(t0.elapsed().as_secs_f64()));
        }
        let eval_examples: Vec<Vec<prl_types::TaskRecord>> =
            eval_envs.iter().map(StartedEnv::examples).collect::<Result<_>>()?;

        // Train source over the env tasksets (each consumed once).
        let mut train_envs = train_envs;
        let mut source_envs = Vec::new();
        for (source, env) in config.train.source.iter().zip(train_envs.iter_mut()) {
            let tasks = std::mem::replace(&mut env.tasks, crate::curriculum::Tasks::Finite(Vec::new()));
            source_envs.push(TrainEnvTasks {
                name: source.resolved_name.clone(),
                ratio: source.ratio,
                curriculum: curriculum_config(source),
                tasks,
            });
        }
        let mut train_source = TrainSource::new(source_envs)?;
        if let Some(step) = resume_step {
            let path: Option<PathBuf> =
                config.resume.as_ref().and_then(|r| r.dir.as_ref()).map(|dir| dir.join("orchestrator"));
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
                prl_config::ModelReference::Frozen(model) => {
                    Ok::<_, anyhow::Error>(Some(Arc::new(connect_frozen_client(model, Some(&config.renderer)).await?)))
                }
                prl_config::ModelReference::Policy => Ok(None),
            }
        }))
        .await?;
        futures::future::try_join_all(algorithms.iter_mut().map(|a| a.setup())).await?;

        let admin_clients = admin.clients().to_vec();
        info!("Initializing weight broadcast ({})", config.reprs.weight_broadcast);
        let t0 = Instant::now();
        let mut receiver = WeightReceiver::new(
            prl_io::pathing::broadcast_dir(&config.output_dir),
            config.weight_broadcast.clone(),
            admin,
            config.model.name.clone(),
            nixl,
        )?;
        receiver.initialize().await?;
        debug!("Initialized weight broadcast in {}", format_time(t0.elapsed().as_secs_f64()));
        let sync_version = resume_step.unwrap_or(0);

        // Shared values.
        let (policy_tx, policy_rx) = watch::channel(Policy { version: 0, model_name: policy_client.model_name.clone() });
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
        let (concurrency_handle, concurrency_task) = concurrency.spawn(
            ConcurrencyIo { samples: load_rx, inflight: inflight_rx, limit: limit_tx, overload: Some(overload_tx) },
            cancel.child_token(),
        );
        tasks.spawn(async move { ("concurrency", concurrency_task.await.map_err(anyhow::Error::from).and_then(|r| r)) });

        // Results channels, bounded like Python's out_q.
        let bound = config.concurrency.max_inflight.map_or(1 << 20, |m| (m as usize).max(8));
        let (train_results_tx, train_results_rx) = mpsc::channel(bound);
        let (eval_results_tx, eval_results_rx) = mpsc::channel(bound);

        let (source_handle, source_task) = train_source.spawn();
        tasks.spawn(async move { ("train_source", source_task.await.map_err(anyhow::Error::from)) });

        let mut train_dispatch = IndexMap::new();
        for (i, source) in config.train.source.iter().enumerate() {
            let live = frozen[i].is_none();
            let clients = frozen[i].clone().unwrap_or_else(|| policy_client.clone());
            train_dispatch.insert(
                source.resolved_name.clone(),
                DispatchEnv {
                    name: source.resolved_name.clone(),
                    group_size: source.group_size as usize,
                    runner: train_envs[i].router.clone(),
                    sampling: train_sampling(source, live)?,
                    client: serde_json::to_value(&clients.train_client)?,
                    frozen_model: (!live).then(|| clients.model_name.clone()),
                    sessions: Arc::new(ClientSessions(clients.clone())),
                },
            );
        }
        let progress_step_placeholder = watch::channel(progress.step);
        let ckpt_enabled = config.ckpt.is_some();
        let ckpt_interval = config.ckpt.as_ref().and_then(|c| c.interval);
        let controller = config.control.as_ref().map(|_| Controller::new(true));
        let train_specs = crate::train_glue::train_env_specs(&config, algorithms)?;
        let sink = crate::train_glue::build_sink(&config, train_specs)?;
        let train_config = TrainPipelineConfig {
            max_steps: config.max_steps,
            ckpt_interval,
            ckpt_enabled,
            weight_broadcast_timeout: Duration::from_secs(config.weight_broadcast.timeout()),
            train_env_names: train_names.clone(),
        };

        // The dispatcher reads `progress_step`, written by the train pipeline.
        let (step_tx, step_rx) = progress_step_placeholder;
        {
            let eval_side = if eval_envs.is_empty() {
                None
            } else {
                let eval_config = config.eval.as_ref().expect("eval envs imply eval config");
                let tasks = eval_sources
                    .iter()
                    .zip(eval_examples.iter())
                    .map(|(s, examples)| {
                        let interval = *eval_config
                            .intervals
                            .get(&s.resolved_name)
                            .ok_or_else(|| anyhow!("No eval interval configured for env {:?}", s.resolved_name))?;
                        Ok(EvalEnvTasks {
                            name: s.resolved_name.clone(),
                            tasks: examples.clone(),
                            group_size: s.group_size as usize,
                            interval,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Some((tasks, eval_config.skip_first_step))
            };
            let eval_side = match eval_side {
                None => None,
                Some((tasks, skip_first)) => {
                    let mut envs = IndexMap::new();
                    for (s, env) in eval_sources.iter().zip(eval_envs.iter()) {
                        envs.insert(
                            s.resolved_name.clone(),
                            DispatchEnv {
                                name: s.resolved_name.clone(),
                                group_size: s.group_size as usize,
                                runner: env.router.clone(),
                                sampling: json_map(&s.sampling)?,
                                client: serde_json::to_value(&policy_client.eval_client)?,
                                frozen_model: None,
                                sessions: Arc::new(ClientSessions(policy_client.clone())),
                            },
                        );
                    }
                    Some(EvalSide {
                        envs,
                        source: EvalSource::new(tasks, skip_first, resume_step.is_some()),
                        results: eval_results_tx,
                    })
                }
            };
            let (idle_tx, idle_rx) = watch::channel(false);
            let built = Dispatcher::new(
                DispatcherConfig {
                    dispatch_per_minute: config.dispatch_per_minute.map(|d| d as f64),
                    run: run.clone(),
                    policy_weights_change: true,
                },
                Some(TrainSide {
                    envs: train_dispatch,
                    source: source_handle.clone(),
                    progress_step: step_rx,
                    results: train_results_tx,
                }),
                eval_side,
                DispatcherIo {
                    policy: policy_rx.clone(),
                    limit: limit_rx,
                    inflight: inflight_tx,
                    overload: overload_rx,
                    concurrency: Some(concurrency_handle.clone()),
                    live: Arc::new(monitors.clone()),
                    release: Arc::new(crate::eval_pipeline::release_episode),
                    idle: idle_tx,
                },
            );
            let (dispatcher, dispatcher_handle, dispatcher_inbox) = built;
            let (train_pipeline, train_handle, train_inbox, view_rx) = TrainPipeline::new(
                train_config,
                sink,
                source_handle.clone(),
                progress.clone(),
                policy_rx.clone(),
                dispatcher_handle.clone(),
                packer,
                sender,
                ckpt.clone(),
                controller.clone(),
                monitors.clone(),
                heartbeat.clone(),
                step_tx,
            );
            spawn_component(tasks, "dispatcher", dispatcher.run(dispatcher_inbox, cancel.child_token()));
            Self::finish_setup(
                config,
                cancel,
                tasks,
                monitors,
                policy_client,
                admin_clients,
                load_tx,
                concurrency_handle,
                dispatcher_handle,
                train_handle,
                (train_pipeline, train_inbox, train_results_rx),
                view_rx,
                eval_results_rx,
                eval_sources.iter().zip(eval_examples.iter()).map(|(s, ex)| (s.resolved_name.clone(), EvalEnvShape { group_size: s.group_size as usize, num_examples: ex.len() })).collect(),
                resume_step,
                receiver,
                policy_tx,
                policy_rx,
                idle_rx,
                sync_version,
                controller,
                ckpt,
                train_envs.into_iter().chain(eval_envs).collect(),
                train_names,
            )
            .await
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn finish_setup(
        config: OrchestratorConfig,
        cancel: CancellationToken,
        tasks: &mut Components,
        monitors: Monitors,
        _policy_client: Arc<InferenceClient>,
        admin_clients: Vec<prl_io::http::AdminClient>,
        load_tx: mpsc::Sender<Vec<prl_io::inference_metrics::EngineLoadSample>>,
        concurrency: ConcurrencyHandle,
        dispatcher: DispatcherHandle,
        train: crate::train_pipeline::TrainPipelineHandle,
        train_pipeline: (TrainPipeline, crate::train_pipeline::TrainPipelineInbox, mpsc::Receiver<prl_types::DispatchResult<prl_wire::Episode>>),
        train_view: watch::Receiver<TrainView>,
        eval_results: mpsc::Receiver<prl_types::DispatchResult<prl_wire::Episode>>,
        eval_shapes: IndexMap<String, EvalEnvShape>,
        resume_step: Option<u64>,
        receiver: WeightReceiver,
        policy_tx: watch::Sender<Policy>,
        policy_rx: watch::Receiver<Policy>,
        idle_rx: watch::Receiver<bool>,
        sync_version: u64,
        controller: Option<Controller>,
        ckpt: CheckpointManager,
        envs: Vec<StartedEnv>,
        train_names: Vec<String>,
    ) -> Result<Self> {
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
        {
            let monitors = monitors.clone();
            let metrics_cancel = metrics_cancel.clone();
            tokio::spawn(collector.run(metrics_cancel));
            tokio::spawn(async move {
                while let Some(row) = metrics_rx.recv().await {
                    monitors.log_metrics(row, None).await;
                }
            });
        }

        let mut hooks: Vec<Box<dyn UpdateHook>> = Vec::new();
        let mut eval_progress = None;
        let eval_envs: Vec<String> = eval_shapes.keys().cloned().collect();
        if !eval_shapes.is_empty() {
            let eval_config = config.eval.as_ref().expect("eval envs imply eval config");
            let (pipeline, handle, progress, inbox) = EvalPipeline::new(
                EvalPipelineConfig {
                    envs: eval_shapes,
                    resume_step,
                    retrigger_on_resume: eval_config.retrigger_on_resume,
                    max_steps: config.max_steps,
                },
                dispatcher.clone(),
                monitors.clone(),
            );
            spawn_component(tasks, "eval_pipeline", pipeline.run(inbox, eval_results, cancel.child_token()));
            hooks.push(Box::new(handle));
            eval_progress = Some(progress);
        }
        hooks.push(Box::new(ResumeDispatch(dispatcher.clone())));
        let (mut watcher, watcher_gauges) =
            WeightWatcher::new(receiver, policy_tx, vec![Box::new(dispatcher.clone())], hooks, sync_version);

        let (lag, lag_task) = lag::spawn(cancel.child_token());
        tokio::spawn(async move {
            let _ = lag_task.await;
        });
        let view = PipelineView {
            dispatcher: dispatcher.clone(),
            concurrency,
            lag,
            watcher: watcher_gauges,
            train: train_view,
            eval: eval_progress,
            gc: config.gc.as_ref().map(|gc| prl_io::gc::GarbageCollector::new(gc.interval).gauges()),
            train_envs: train_names,
            eval_envs,
        };

        info!("Syncing inference to the trainer's startup broadcast (v{sync_version})");
        let t0 = Instant::now();
        watcher.sync_startup(sync_version, Duration::from_secs(config.weight_broadcast.timeout())).await?;
        debug!("Synced inference to policy v{sync_version} in {}", format_time(t0.elapsed().as_secs_f64()));

        Ok(Self {
            config,
            cancel,
            metrics_cancel,
            dispatcher,
            train,
            train_pipeline: Some(train_pipeline),
            watcher: Some(watcher),
            policy: policy_rx,
            idle: idle_rx,
            monitors,
            view,
            controller,
            ckpt,
            envs,
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
                        let mut policy = self.policy.clone();
                        let wait = wait_for_version(&mut policy, max, "before shutdown", timeout, &self.cancel);
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
