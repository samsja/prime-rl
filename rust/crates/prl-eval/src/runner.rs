//! The eval engine shared by `uv run eval` and online evals (port of `eval/runner.py`).

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use prl_config::{ControlConfig, ServedEvalConfig, Split};
use prl_core::concurrency::{ConcurrencyController, ConcurrencyHandle, ConcurrencyIo, ConcurrencyParams};
use prl_core::dispatcher::{
    release_episode, DispatchEnv, Dispatcher, DispatcherConfig, DispatcherHandle, DispatcherIo, EvalSide, Mode,
    TriggerOutcome,
};
use prl_core::envs::{replica_address_files, start_env, ReplicaRouter};
use prl_core::eval_sink::{EvalEpoch, EvalSink};
use prl_core::eval_source::{EvalEnvTasks, EvalSource};
use prl_core::fmt::pct;
use prl_core::periodic::PeriodicLogger;
use prl_io::clients::{AdminPlane, ClientType, InferenceClient};
use prl_io::heartbeat::Heartbeat;
use prl_io::inference_metrics::InferenceMetricsCollector;
use prl_io::logging::format_time;
use prl_monitors::annotations::{stamp_arrival, stamp_batch};
use prl_monitors::Monitors;
use prl_train::episodes::episode_env_name;
use prl_train::metrics::{dispatch_failure_metrics, EvalEpisodes};
use prl_types::{Bridge, DispatchResult, Kind, Metrics, Policy, RunIdentity, Subset};
use prl_wire::{Episode, RunInfo};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

/// How often `run_epoch` re-checks for a superseding checkpoint while it waits for episodes.
pub const POLL_INTERVAL: Duration = Duration::from_secs(2);

/// The source fields the runner reads (`EvalSourceConfig` / `OnlineEvalSourceConfig`).
pub trait SourceView {
    fn name(&self) -> &str;
    fn sampling(&self) -> &serde_json::Value;
    fn group_size(&self) -> usize;
}

macro_rules! source_view {
    ($ty:ty) => {
        impl SourceView for $ty {
            fn name(&self) -> &str {
                &self.resolved_name
            }
            fn sampling(&self) -> &serde_json::Value {
                &self.sampling
            }
            fn group_size(&self) -> usize {
                self.group_size as usize
            }
        }
    };
}
source_view!(prl_config::EvalSourceConfig);
source_view!(prl_config::OnlineEvalSourceConfig);

/// Interval scheduling of online evals; a standalone eval fires every env on its one trigger.
pub struct Schedule {
    pub intervals: IndexMap<String, u64>,
    pub skip_first_step: bool,
    pub is_resumed: bool,
}

/// `(env, step, arrived, expected)` per epoch in progress.
type Progress = Vec<(String, u64, usize, usize)>;

/// A superseding-checkpoint probe: the newest published step after the epoch's, if any.
pub type Superseding<'a> = Option<&'a mut (dyn FnMut() -> Option<u64> + Send)>;

fn eval_step(episode: &Episode) -> Result<u64> {
    let Some(RunInfo::Train(run)) = &episode.run else { bail!("Dispatched episode is missing run identity") };
    Ok(run.work.step() as u64)
}

/// An eval-only dispatcher under the adaptive concurrency controller, fed by the inference
/// metrics collector. Eval episodes are version-pinned measurements and are never cancelled on
/// load — a controller cut only blocks admission until the pool drains.
pub struct EvalRunner {
    pub dispatcher: DispatcherHandle,
    /// Single writer of the served policy (the watcher of an online eval).
    pub policy: watch::Sender<Policy>,
    /// Handed to the weight receiver by online evals.
    pub admin_plane: Option<AdminPlane>,
    monitors: Monitors,
    concurrency: ConcurrencyHandle,
    group_sizes: IndexMap<String, usize>,
    epoch_sizes: IndexMap<String, usize>,
    results: mpsc::Receiver<DispatchResult<Episode>>,
    sink: EvalSink<Arc<Episode>>,
    progress: watch::Sender<Progress>,
    triggered_at: HashMap<(String, u64), Instant>,
    heart: Option<Heartbeat>,
    envs: Vec<Arc<ReplicaRouter>>,
    /// Any of these failing fails the eval.
    actors: JoinSet<Result<()>>,
    actors_cancel: CancellationToken,
    periodic: Option<PeriodicLogger>,
    control: Option<ControlConfig>,
    _overload: mpsc::UnboundedSender<usize>,
}

impl EvalRunner {
    pub async fn setup<S: SourceView>(
        served: &ServedEvalConfig<S>,
        model: &str,
        run_dir: &Path,
        log_interval: f64,
        schedule: Option<Schedule>,
        monitors: Monitors,
        bridge: Arc<dyn Bridge>,
    ) -> Result<Self> {
        let run = RunIdentity {
            id: std::env::var("PRL_RUN_ID")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()),
            name: std::env::var("PRL_RUN_NAME").ok(),
        };

        info!("Initializing inference pool (base_url={}, model={model})", served.client.base_url);
        let clients = Arc::new(InferenceClient::new(
            &served.client,
            model,
            ClientType::ChatCompletions,
            ClientType::ChatCompletions,
            None,
        )?);
        let mut admin_plane = AdminPlane::standard(&served.client, None)?;

        info!("Loading eval environment(s)");
        let config_dir = prl_core::envs::config_dir(run_dir);
        let started = futures::future::try_join_all(served.source.iter().map(|source| {
            let name = source.name().to_owned();
            let address = served.env_addresses.get(&(Split::Eval, name.clone())).cloned().flatten();
            let files = replica_address_files(&config_dir, Kind::Eval, &name, None);
            let bridge = bridge.clone();
            async move { start_env(Kind::Eval, &name, address, files, bridge).await }
        }))
        .await?;
        let names: Vec<&str> = started.iter().map(|env| env.name.as_str()).collect();
        info!("Eval environment(s) ready ({})", names.join(", "));

        info!("Waiting for inference pool to be ready");
        admin_plane.wait_for_ready(model).await?;
        info!("Inference pool ready");

        let client = serde_json::to_value(&clients.eval_client)?;
        let (mut env_tasks, mut dispatch_envs, mut envs) = (Vec::new(), IndexMap::new(), Vec::new());
        let (mut group_sizes, mut epoch_sizes) = (IndexMap::new(), IndexMap::new());
        for (source, env) in served.source.iter().zip(started) {
            let (name, group_size, tasks) = (env.name.clone(), source.group_size(), env.examples()?);
            group_sizes.insert(name.clone(), group_size);
            epoch_sizes.insert(name.clone(), tasks.len() * group_size);
            let interval = schedule.as_ref().map_or(1, |s| s.intervals[&name]);
            env_tasks.push(EvalEnvTasks { name: name.clone(), tasks, group_size, interval });
            envs.push(env.router.clone());
            let dispatch_env = DispatchEnv {
                name: name.clone(),
                group_size,
                runner: env.router,
                sampling: serde_json::from_value(source.sampling().clone()).context("eval sampling must be a mapping")?,
                client: client.clone(),
                frozen_model: None,
                sessions: clients.clone(),
            };
            dispatch_envs.insert(name, dispatch_env);
        }
        let (skip_first_step, is_resumed) = schedule.as_ref().map_or((false, false), |s| (s.skip_first_step, s.is_resumed));
        let source = EvalSource::new(env_tasks, skip_first_step, is_resumed);

        let actors_cancel = CancellationToken::new();
        let mut actors = JoinSet::new();
        let (policy, policy_rx) = watch::channel(Policy { version: 0, model_name: model.to_owned() });

        let band = &served.concurrency;
        let params = ConcurrencyParams {
            min_inflight: band.min_inflight as usize,
            max_inflight: band.max_inflight.map(|n| n as usize),
            initial_inflight: band.initial_inflight.map(|n| n as usize),
        };
        // Pessimistic per-episode token cost, only used when the engine reports no max context length.
        let max_tokens = served.source.iter().filter_map(|s| s.sampling()["max_tokens"].as_u64()).max();
        let controller = ConcurrencyController::new(params, max_tokens.filter(|&n| n > 0).unwrap_or(8192));
        let (samples_tx, samples) = mpsc::channel(16);
        let (inflight_tx, inflight) = watch::channel(0);
        let (limit, limit_rx) = watch::channel(controller.max_inflight);
        let (concurrency, concurrency_task) =
            controller.spawn(ConcurrencyIo { samples, inflight, limit, overload: None }, actors_cancel.clone());
        actors.spawn(async move { concurrency_task.await? });

        let (results_tx, results) = mpsc::channel(band.max_inflight.map_or(1024, |n| (n as usize).max(8)));
        let (overload_tx, overload) = mpsc::unbounded_channel();
        let (dispatcher, dispatcher_handle, inbox) = Dispatcher::new(
            DispatcherConfig {
                dispatch_per_minute: served.dispatch_per_minute.map(|n| n as f64),
                run,
                policy_weights_change: schedule.is_some(),
            },
            None,
            Some(EvalSide { envs: dispatch_envs, source, results: results_tx }),
            DispatcherIo {
                policy: policy_rx,
                limit: limit_rx,
                inflight: inflight_tx,
                overload,
                concurrency: Some(concurrency.clone()),
                live: monitors.clone(),
                idle: watch::channel(false).0,
            },
        );
        let cancel = actors_cancel.clone();
        actors.spawn(async move { dispatcher.run(inbox, cancel).await.context("dispatcher failed") });

        // The collector always polls: it feeds the controller; its rows go to the monitors.
        let (metrics_tx, mut metrics_rx) = mpsc::channel::<Metrics>(16);
        let mut collector =
            InferenceMetricsCollector::new(admin_plane.clients(), None, true, Some(samples_tx), Some(metrics_tx))?;
        let forward = monitors.clone();
        actors.spawn(async move {
            while let Some(row) = metrics_rx.recv().await {
                forward.log_metrics(row, None).await;
            }
            Ok(())
        });
        // External API endpoints expose no vLLM /metrics: without a load signal the cap would
        // silently sit at min_inflight, so only a pinned band may run against them.
        if !collector.probe(3, Duration::from_secs(2)).await {
            if Some(band.min_inflight) != band.max_inflight {
                let urls: Vec<String> = admin_plane.clients().iter().map(|c| c.base_url()).collect();
                bail!(
                    "No engine metrics at {} - adaptive concurrency has no load signal. The endpoint does not expose \
                     vLLM /metrics (e.g. an external inference API); pin the concurrency with `-c N` \
                     (concurrency.min_inflight = max_inflight = N).",
                    urls.join(", ")
                );
            }
            info!("No engine metrics - running with concurrency pinned at {}", band.min_inflight);
        }
        let cancel = actors_cancel.clone();
        actors.spawn(async move {
            collector.run(cancel).await;
            Ok(())
        });

        let progress = watch::channel(Vec::new()).0;
        let periodic = {
            let (dispatcher, concurrency, progress) = (dispatcher_handle.clone(), concurrency.clone(), progress.subscribe());
            PeriodicLogger::spawn(Duration::from_secs_f64(log_interval), monitors.clone(), move || {
                let (dispatcher, concurrency, progress) = (dispatcher.clone(), concurrency.clone(), progress.borrow().clone());
                async move { pipeline_view(&dispatcher, &concurrency, progress).await }
            })
        };
        Ok(Self {
            dispatcher: dispatcher_handle,
            policy,
            admin_plane: Some(admin_plane),
            monitors,
            concurrency,
            group_sizes,
            sink: EvalSink::new(epoch_sizes.clone()),
            epoch_sizes,
            results,
            progress,
            triggered_at: HashMap::new(),
            heart: served.heartbeat.as_ref().map(|h| Heartbeat::new(h.url.clone(), h.min_interval)),
            envs,
            actors,
            actors_cancel,
            periodic: Some(periodic),
            control: served.control.clone(),
            _overload: overload_tx,
        })
    }

    /// Start dispatching and serve the control API (when configured).
    pub fn start(&mut self) -> Result<()> {
        self.dispatcher.start()?;
        if let Some(control) = self.control.clone() {
            let (concurrency, cancel) = (self.concurrency.clone(), self.actors_cancel.clone());
            self.actors.spawn(async move { crate::control::serve(concurrency, &control.host, control.port, cancel).await });
        }
        Ok(())
    }

    /// Fire the eligible envs at `step` (`EvalSource.trigger`), reusing `completed` episodes.
    pub async fn trigger(&self, step: u64, force: bool, completed: Vec<Episode>) -> Result<TriggerOutcome> {
        self.dispatcher.trigger_eval(step, force, completed, None).await
    }

    /// Run the epoch `trigger` queued for `step` and finalize each env's batch as it completes.
    /// Restored episodes rejoin it first; once `superseding` reports a newer checkpoint, the
    /// epoch's unfinished episodes are cancelled so the caller can move on to it.
    pub async fn run_epoch(&mut self, outcome: TriggerOutcome, step: u64, mut superseding: Superseding<'_>) -> Result<()> {
        let TriggerOutcome { fired, restored, owed } = outcome;
        let now = Instant::now();
        for env in &fired {
            self.monitors.log_eval_plan(env.clone(), step, self.epoch_sizes[env] as u64).await;
            self.triggered_at.insert((env.clone(), step), now);
        }
        let restored_part = if restored.is_empty() { String::new() } else { format!(", {} restored", restored.len()) };
        info!("Starting evals in {} at step {step} ({owed} total rollouts{restored_part})", fired.join(", "));
        // Checked before the mode switch too: the dispatcher schedules concurrently, so an
        // already superseded epoch must not start episodes it would cancel right away.
        let mut cancelled = None;
        self.cancel_if_superseded(step, &mut superseding, &mut cancelled).await?;
        self.dispatcher.switch_mode(Mode::PreferEval, format!("eval was triggered at step {step}"))?;

        let mut pending: BTreeSet<String> = fired.into_iter().filter(|env| self.epoch_sizes[env] > 0).collect();
        for episode in restored {
            self.land(episode, &mut pending).await?;
        }
        let mut poll = tokio::time::interval(POLL_INTERVAL);
        while !pending.is_empty() {
            self.cancel_if_superseded(step, &mut superseding, &mut cancelled).await?;
            tokio::select! {
                item = self.results.recv() => {
                    let epoch = match item.context("dispatcher stopped before the eval epoch finished")? {
                        DispatchResult::Cancelled(c) => self.sink.cancel(&c)?,
                        DispatchResult::Failure(f) => self.sink.fail(f)?,
                        DispatchResult::Episode(mut episode) => {
                            let at = eval_step(&episode)?;
                            stamp_arrival(std::slice::from_mut(&mut episode), Kind::Eval, at);
                            if let Some(heart) = &self.heart {
                                heart.beat();
                            }
                            self.land(episode, &mut pending).await?;
                            continue;
                        }
                    };
                    self.progress.send_replace(self.sink.batch_progress());
                    if let Some(epoch) = epoch {
                        pending.remove(&epoch.env_name);
                        self.finalize(epoch).await?;
                    }
                }
                _ = poll.tick(), if superseding.is_some() => {}
                Some(joined) = self.actors.join_next() => {
                    joined??;
                    bail!("an eval component stopped before the eval epoch finished");
                }
            }
        }
        if let Some((newer, n)) = cancelled {
            warn!("Cancelled {n} unfinished eval episodes for step {step}; advancing to checkpoint {newer}");
        }
        Ok(())
    }

    async fn cancel_if_superseded(
        &self,
        step: u64,
        superseding: &mut Superseding<'_>,
        cancelled: &mut Option<(u64, usize)>,
    ) -> Result<()> {
        if cancelled.is_none() {
            if let Some(newer) = superseding.as_mut().and_then(|f| f()) {
                warn!("Checkpoint {newer} is ready - cancelling unfinished eval episodes for step {step}");
                *cancelled = Some((newer, self.dispatcher.cancel_eval_step(step).await?));
            }
        }
        Ok(())
    }

    /// Fail when a component stopped (a dispatcher error, the control server's bind, ...).
    pub fn check_components(&mut self) -> Result<()> {
        if let Some(joined) = self.actors.try_join_next() {
            joined??;
            bail!("an eval component stopped unexpectedly");
        }
        Ok(())
    }

    /// One episode of the epoch, arrived or restored: through the monitors into its env's batch.
    async fn land(&mut self, episode: Episode, pending: &mut BTreeSet<String>) -> Result<()> {
        let step = eval_step(&episode)?;
        let env = episode_env_name(&episode)?.to_owned();
        let episode = Arc::new(episode);
        self.monitors.log_episodes(vec![episode.clone()], step, Kind::Eval, Subset::All).await;
        // Env servers keep every episode's training arrays; eval never trains on them.
        release_episode(&episode);
        let epoch = self.sink.add(&env, step, episode);
        self.progress.send_replace(self.sink.batch_progress());
        if let Some(epoch) = epoch {
            pending.remove(&epoch.env_name);
            self.finalize(epoch).await?;
        }
        Ok(())
    }

    /// Log one completed epoch: effective episodes, the epoch upload and the `eval/{env}/...` metrics.
    async fn finalize(&mut self, epoch: EvalEpoch<Arc<Episode>>) -> Result<()> {
        let EvalEpoch { env_name, step, episodes, failures, cancelled } = epoch;
        if episodes.is_empty() && failures.is_empty() && cancelled == 0 {
            warn!("Eval @ step={step} env={env_name}: no attempts returned, skipping log");
            return Ok(());
        }
        let completed = episodes.len();
        let all = EvalEpisodes::new(episodes, self.group_sizes[&env_name]);
        let effective = all.effective();
        if !effective.is_empty() {
            let records: Vec<Arc<Episode>> = effective.vf_episodes().into_iter().map(Arc::new).collect();
            self.monitors.log_episodes(records.clone(), step, Kind::Eval, Subset::Effective).await;
            self.monitors.log_annotations(stamp_batch(records.iter().map(|e| e.as_ref()), step)).await;
        }
        let records = all.vf_episodes().into_iter().map(Arc::new).collect();
        self.monitors.log_eval_epoch(env_name.clone(), step, records).await;

        let prefix = format!("eval/{env_name}");
        let total = completed + failures.len() + cancelled;
        let mut metrics = all.to_wandb(&prefix, Subset::All);
        metrics.extend(effective.to_wandb(&prefix, Subset::Effective));
        metrics.extend(dispatch_failure_metrics(&failures, &format!("{prefix}/all"), total));
        if cancelled > 0 {
            metrics.insert(format!("{prefix}/all/cancelled/count"), (cancelled as f64).into());
            metrics.insert(format!("{prefix}/all/cancelled/mean"), (cancelled as f64 / total as f64).into());
        }
        metrics.insert(format!("{prefix}/policy_version"), (step as f64).into());
        metrics.insert("step".into(), (step as f64).into());
        self.monitors.log_metrics(metrics, Some(step)).await;

        let (reward, turns, branches, truncated) = effective
            .with_metrics(|m| (m.reward().mean(), m.num_turns().mean(), m.num_branches().mean(), m.is_truncated().mean()));
        let (error, timeout) = all.with_metrics(|m| (m.has_error().mean(), m.is_timeout().mean()));
        let elapsed = self.triggered_at.remove(&(env_name.clone(), step)).map_or(0.0, |t| t.elapsed().as_secs_f64());
        let elapsed = format_time(elapsed);
        if cancelled > 0 {
            warn!(
                "Partially evaluated {env_name} (Step {step}) | {elapsed:>7} | Reward {reward:.4} | Error {} | \
                 Completed {completed}/{total} | Cancelled {cancelled}/{total}",
                pct(error)
            );
        } else {
            prl_io::success!(
                "Evaluated {env_name} (Step {step}) | {elapsed:>7} | Reward {reward:.4} | Turns {turns:.1} | \
                 Branches {branches:.1} | Error {} | Truncation {} | Timeout {}",
                pct(error),
                pct(truncated),
                pct(timeout)
            );
        }
        Ok(())
    }

    /// Stop the periodic logger so nothing logs after the monitors finalize.
    pub async fn drain(&mut self) {
        if let Some(periodic) = self.periodic.take() {
            periodic.stop().await;
        }
    }

    /// Teardown; tolerates a failed run.
    pub async fn stop(mut self) {
        self.drain().await;
        let _ = self.dispatcher.stop().await;
        self.actors_cancel.cancel();
        while self.actors.join_next().await.is_some() {}
        for router in &self.envs {
            for client in router.clients() {
                client.close().await;
            }
        }
    }
}

/// The periodic `Eval` line: per-env epoch progress plus the in-flight pool against its cap.
async fn pipeline_view(dispatcher: &DispatcherHandle, concurrency: &ConcurrencyHandle, mut progress: Progress) -> (String, Metrics) {
    let (Ok(view), Ok(gauges)) = (dispatcher.view().await, concurrency.gauges().await) else {
        return ("Eval components stopped".into(), Metrics::new());
    };
    progress.sort();
    let parts: Vec<String> = progress
        .iter()
        .map(|(env, _, arrived, expected)| match expected {
            0 => env.clone(),
            _ => format!("{env} {arrived}/{expected} ({})", pct(*arrived as f64 / *expected as f64)),
        })
        .collect();
    let progress = if parts.is_empty() { "Idle".to_owned() } else { parts.join(" | ") };
    let signal = match gauges.get("concurrency/signal").and_then(|s| s.as_f64()) {
        Some(s) if s >= 2.0 => "hard",
        Some(s) if s >= 1.0 => "soft",
        _ => "clear",
    };
    let stages = if view.stage_counts.is_empty() { String::new() } else { format!(" - {}", view.stage_counts) };
    let body = format!(
        "{progress}; {} inflight episodes (cap {}, signal {signal}){stages}",
        view.inflight_eval, view.max_inflight
    );
    let mut payload = view.gauges;
    payload.extend(view.drained);
    payload.extend(gauges);
    (body, payload)
}
