//! The eval engine shared by `uv run eval` and online evals (port of `eval/runner.py`).

use prl_io::logging::format_time;

/// How often `run_epoch` re-checks for a superseding checkpoint while it waits for episodes.
pub const POLL_INTERVAL_S: u64 = 2;

/// Python `f"{x:.1%}"`.
pub fn pct(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

/// Means of one finished epoch that its log line reports: reward/turns/branches/truncation over
/// the effective episodes, error/timeout over all of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochSummary {
    pub reward: f64,
    pub turns: f64,
    pub branches: f64,
    pub truncated: f64,
    pub error: f64,
    pub timeout: f64,
}

/// SUCCESS line of a complete epoch.
pub fn evaluated_line(env: &str, step: u64, elapsed: f64, s: &EpochSummary) -> String {
    format!(
        "Evaluated {env} (Step {step}) | {:>7} | Reward {:.4} | Turns {:.1} | Branches {:.1} | Error {} | Truncation {} | Timeout {}",
        format_time(elapsed),
        s.reward,
        s.turns,
        s.branches,
        pct(s.error),
        pct(s.truncated),
        pct(s.timeout),
    )
}

/// WARNING line of an epoch a newer checkpoint cut short.
pub fn partially_evaluated_line(
    env: &str,
    step: u64,
    elapsed: f64,
    s: &EpochSummary,
    completed: usize,
    cancelled: usize,
    total: usize,
) -> String {
    format!(
        "Partially evaluated {env} (Step {step}) | {:>7} | Reward {:.4} | Error {} | Completed {completed}/{total} | Cancelled {cancelled}/{total}",
        format_time(elapsed),
        s.reward,
        pct(s.error),
    )
}

/// Body of the periodic `Eval` line: per-epoch progress, then the in-flight pool against its cap.
pub fn pipeline_body(
    mut progress: Vec<(String, u64, usize, usize)>,
    inflight_eval: usize,
    max_inflight: usize,
    signal: &str,
    stage_counts: &str,
) -> String {
    progress.sort();
    let parts: Vec<String> = progress
        .iter()
        .map(|(env, _step, arrived, expected)| {
            if *expected > 0 {
                format!("{env} {arrived}/{expected} ({})", pct(*arrived as f64 / *expected as f64))
            } else {
                env.clone()
            }
        })
        .collect();
    let progress = if parts.is_empty() { "Idle".to_owned() } else { parts.join(" | ") };
    let stages = if stage_counts.is_empty() { String::new() } else { format!(" - {stage_counts}") };
    format!("{progress}; {inflight_eval} inflight episodes (cap {max_inflight}, signal {signal}){stages}")
}

/// `Starting evals in ...` line of a triggered epoch.
pub fn starting_line(fired: &[String], step: u64, owed: usize, restored: usize) -> String {
    let restored = if restored > 0 { format!(", {restored} restored") } else { String::new() };
    format!("Starting evals in {} at step {step} ({owed} total rollouts{restored})", fired.join(", "))
}


// ── engine ──────────────────────────────────────────────────────────────────

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use prl_config::{ConcurrencyConfig, OnlineEvalSourceConfig, ServedEvalConfig, Split};
use prl_core::concurrency::{ConcurrencyController, ConcurrencyHandle, ConcurrencyIo, ConcurrencyParams};
use prl_core::dispatcher::{
    DispatchEnv, Dispatcher, DispatcherConfig, DispatcherHandle, DispatcherIo, EvalSide, Mode, TriggerOutcome,
};
use prl_core::envs::{replica_address_files, start_env, ReplicaRouter, StartedEnv};
use prl_core::eval_sink::{EvalEpoch, EvalSink};
use prl_core::eval_source::{EvalEnvTasks, EvalSource};
use prl_core::glue::ClientSessions;
use prl_core::periodic::PeriodicLogger;
use prl_io::clients::{AdminPlane, ClientType, InferenceClient};
use prl_io::heartbeat::Heartbeat;
use prl_io::inference_metrics::InferenceMetricsCollector;
use prl_monitors::annotations::{stamp_arrival, stamp_batch};
use prl_monitors::Monitors;
use prl_train::metrics::{dispatch_failure_metrics, EvalEpisodes};
use prl_types::{Bridge, DispatchResult, Kind, Metrics, Policy, RunIdentity, Subset};
use prl_wire::{EnvClient, Episode, RunInfo};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

/// One eval source as the runner needs it (`EvalSourceConfig` / `OnlineEvalSourceConfig`).
pub trait SourceView {
    fn resolved_name(&self) -> &str;
    fn sampling(&self) -> &serde_json::Value;
    fn group_size(&self) -> u64;
}

impl SourceView for prl_config::EvalSourceConfig {
    fn resolved_name(&self) -> &str {
        &self.resolved_name
    }
    fn sampling(&self) -> &serde_json::Value {
        &self.sampling
    }
    fn group_size(&self) -> u64 {
        self.group_size
    }
}

impl SourceView for OnlineEvalSourceConfig {
    fn resolved_name(&self) -> &str {
        &self.resolved_name
    }
    fn sampling(&self) -> &serde_json::Value {
        &self.sampling
    }
    fn group_size(&self) -> u64 {
        self.group_size
    }
}

/// Interval scheduling of online evals; a standalone eval fires every env on its one trigger.
pub struct Schedule {
    pub intervals: IndexMap<String, u64>,
    pub skip_first_step: bool,
    pub is_resumed: bool,
}

/// `(env, step, arrived, expected)` per epoch in progress.
type Progress = Vec<(String, u64, usize, usize)>;

/// Pessimistic per-episode token cost for the controller's starting cap (only used when the
/// engine does not report its max context length).
pub fn fallback_cost<S: SourceView>(sources: &[S]) -> u64 {
    let max = sources.iter().map(|s| s.sampling().get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(0)).max();
    match max {
        Some(n) if n > 0 => n,
        _ => 8192,
    }
}

fn eval_step(episode: &Episode) -> Result<u64> {
    let Some(RunInfo::Train(run)) = &episode.run else { bail!("Dispatched episode is missing run identity") };
    Ok(run.work.step() as u64)
}

fn env_name(episode: &Episode) -> String {
    episode.env.name.clone().unwrap_or_else(|| episode.env.id.clone())
}

fn severity_name(severity: Option<f64>) -> &'static str {
    match severity {
        Some(s) if s >= 2.0 => "hard",
        Some(s) if s >= 1.0 => "soft",
        _ => "clear",
    }
}

/// Delete the env-server arrays of an episode (eval never trains on them).
fn release_episode(episode: &Episode) {
    prl_wire::payload::delete_payloads(episode.traces.iter().flat_map(prl_wire::payload::trace_payload_urls));
}

/// The eval engine: an eval-only dispatcher under the adaptive concurrency controller, fed by
/// the inference metrics collector. Eval episodes are version-pinned measurements and are never
/// cancelled on load — a controller cut only blocks admission until the pool drains.
pub struct EvalRunner {
    pub monitors: Monitors,
    pub dispatcher: DispatcherHandle,
    pub concurrency: ConcurrencyHandle,
    /// Single writer of the served policy (the watcher of an online eval).
    pub policy: watch::Sender<Policy>,
    /// Admin plane of the inference pool (handed to the weight receiver by online evals).
    pub admin_plane: Option<AdminPlane>,
    pub group_sizes: IndexMap<String, usize>,
    results: mpsc::Receiver<DispatchResult<Episode>>,
    sink: EvalSink<Arc<Episode>>,
    epoch_sizes: IndexMap<String, usize>,
    progress: watch::Sender<Progress>,
    triggered_at: HashMap<(String, u64), Instant>,
    heart: Option<Heartbeat>,
    envs: Vec<Arc<ReplicaRouter<EnvClient>>>,
    /// Actors the runner owns; any of them failing fails the eval.
    actors: JoinSet<Result<()>>,
    actors_cancel: CancellationToken,
    periodic: Option<PeriodicLogger>,
    control: Option<prl_config::ControlConfig>,
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
        let heart = served.heartbeat.as_ref().map(|h| Heartbeat::new(h.url.clone(), h.min_interval));
        let run = RunIdentity {
            id: std::env::var("PRL_RUN_ID").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()),
            name: std::env::var("PRL_RUN_NAME").ok(),
        };

        info!("Initializing inference pool (base_url={}, model={model})", served.client.base_url);
        let clients =
            Arc::new(InferenceClient::new(&served.client, model, ClientType::ChatCompletions, ClientType::ChatCompletions, None)?);
        let mut admin_plane = AdminPlane::standard(&served.client, None)?;

        info!("Loading eval environment(s)");
        let config_dir = prl_core::envs::config_dir(run_dir);
        let starts = served.source.iter().map(|source| {
            let name = source.resolved_name().to_owned();
            let address = served.env_addresses.get(&(Split::Eval, name.clone())).cloned().flatten();
            let files = replica_address_files(&config_dir, Kind::Eval, &name, None);
            let bridge = bridge.clone();
            async move { start_env(Kind::Eval, &name, address, files, bridge).await }
        });
        let started: Vec<StartedEnv> = futures::future::try_join_all(starts).await?;
        let envs = started
            .into_iter()
            .map(|env| Ok((env.name.clone(), env.router.clone(), env.examples()?)))
            .collect::<Result<Vec<_>>>()?;
        let names: Vec<String> = envs.iter().map(|(name, _, _)| name.clone()).collect();
        info!("Eval environment(s) ready ({})", names.join(", "));

        info!("Waiting for inference pool to be ready");
        admin_plane.wait_for_ready(model).await?;
        info!("Inference pool ready");

        let mut env_tasks = Vec::new();
        let mut dispatch_envs = IndexMap::new();
        let mut group_sizes = IndexMap::new();
        let mut epoch_sizes = IndexMap::new();
        let client_json = serde_json::to_value(&clients.eval_client)?;
        let sessions = Arc::new(ClientSessions(clients.clone()));
        let mut routers = Vec::new();
        for (source, (name, router, tasks)) in served.source.iter().zip(envs) {
            let group_size = source.group_size() as usize;
            group_sizes.insert(name.clone(), group_size);
            epoch_sizes.insert(name.clone(), tasks.len() * group_size);
            let interval = schedule.as_ref().map_or(1, |s| s.intervals[&name]);
            env_tasks.push(EvalEnvTasks { name: name.clone(), tasks, group_size, interval });
            let sampling = source.sampling().as_object().cloned().context("eval sampling must be a mapping")?;
            routers.push(router.clone());
            dispatch_envs.insert(
                name.clone(),
                DispatchEnv {
                    name: name.clone(),
                    group_size,
                    runner: router,
                    sampling: serde_json::from_value(serde_json::Value::Object(sampling))?,
                    client: client_json.clone(),
                    frozen_model: None,
                    sessions: sessions.clone(),
                },
            );
        }
        let (skip_first_step, is_resumed) = schedule.as_ref().map_or((false, false), |s| (s.skip_first_step, s.is_resumed));
        let source = EvalSource::new(env_tasks, skip_first_step, is_resumed);

        let actors_cancel = CancellationToken::new();
        let mut actors = JoinSet::new();
        let (policy, policy_rx) = watch::channel(Policy { version: 0, model_name: model.to_owned() });

        let ConcurrencyConfig { initial_inflight, min_inflight, max_inflight } = served.concurrency.clone();
        let params = ConcurrencyParams {
            min_inflight: min_inflight as usize,
            max_inflight: max_inflight.map(|n| n as usize),
            initial_inflight: initial_inflight.map(|n| n as usize),
        };
        let controller = ConcurrencyController::new(params, fallback_cost(&served.source));
        let (samples_tx, samples) = mpsc::channel(16);
        let (inflight_tx, inflight) = watch::channel(0usize);
        let (limit, limit_rx) = watch::channel(controller.max_inflight);
        let (concurrency, concurrency_task) =
            controller.spawn(ConcurrencyIo { samples, inflight, limit, overload: None }, actors_cancel.clone());
        actors.spawn(async move { concurrency_task.await? });

        let (results_tx, results) = mpsc::channel(max_inflight.map_or(1024, |n| (n as usize).max(8)));
        let (overload_tx, overload) = mpsc::unbounded_channel();
        let (idle, _) = watch::channel(false);
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
                live: Arc::new(monitors.clone()),
                release: Arc::new(release_episode),
                idle,
            },
        );
        let cancel = actors_cancel.clone();
        actors.spawn(async move { dispatcher.run(inbox, cancel).await.context("dispatcher failed") });

        // The collector always polls: it feeds the concurrency controller; its metrics rows
        // fan out to every monitor.
        let (metrics_tx, mut metrics_rx) = mpsc::channel::<Metrics>(16);
        let mut collector =
            InferenceMetricsCollector::new(admin_plane.clients(), None, true, Some(samples_tx), Some(metrics_tx))?;
        let forward = monitors.clone();
        actors.spawn(async move {
            while let Some(row) = metrics_rx.recv().await {
                forward.log_metrics(row.into(), None).await;
            }
            Ok(())
        });
        // Fail fast when adaptivity has no signal: external API endpoints expose no vLLM
        // /metrics, so the cap would silently sit at min_inflight forever.
        if !collector.probe(3, Duration::from_secs(2)).await {
            if Some(min_inflight) != max_inflight {
                let urls: Vec<String> = admin_plane.clients().iter().map(|c| c.base_url()).collect();
                bail!(
                    "No engine metrics at {} - adaptive concurrency has no load signal. The endpoint does not expose \
                     vLLM /metrics (e.g. an external inference API); pin the concurrency with `-c N` \
                     (concurrency.min_inflight = max_inflight = N).",
                    urls.join(", ")
                );
            }
            info!("No engine metrics - running with concurrency pinned at {min_inflight}");
        }
        let cancel = actors_cancel.clone();
        actors.spawn(async move {
            collector.run(cancel).await;
            Ok(())
        });

        let (progress, _) = watch::channel(Vec::new());
        Ok(Self {
            monitors,
            dispatcher: dispatcher_handle,
            concurrency,
            policy,
            admin_plane: Some(admin_plane),
            group_sizes,
            results,
            sink: EvalSink::new(epoch_sizes.clone()),
            epoch_sizes,
            progress,
            triggered_at: HashMap::new(),
            heart,
            envs: routers,
            actors,
            actors_cancel,
            periodic: None,
            control: served.control.clone(),
            _overload: overload_tx,
        })
        .map(|mut this| {
            this.start_periodic(log_interval);
            this
        })
    }

    fn start_periodic(&mut self, interval: f64) {
        let dispatcher = self.dispatcher.clone();
        let concurrency = self.concurrency.clone();
        let progress = self.progress.subscribe();
        self.periodic = Some(PeriodicLogger::spawn(Duration::from_secs_f64(interval), self.monitors.clone(), move || {
            let dispatcher = dispatcher.clone();
            let concurrency = concurrency.clone();
            let progress = progress.borrow().clone();
            async move { collect_pipeline_view(&dispatcher, &concurrency, progress).await }
        }));
    }

    /// Start dispatching and serve the control API (when configured).
    pub fn start(&mut self) -> Result<()> {
        self.dispatcher.start()?;
        if let Some(control) = self.control.clone() {
            let concurrency = self.concurrency.clone();
            let cancel = self.actors_cancel.clone();
            self.actors.spawn(async move {
                crate::control::serve(concurrency, &control.host, control.port, cancel).await
            });
        }
        Ok(())
    }

    /// Fire the eligible envs at `step` (`EvalSource.trigger`), reusing `completed` episodes.
    pub async fn trigger(&self, step: u64, force: bool, completed: Vec<Episode>) -> Result<TriggerOutcome> {
        self.dispatcher.trigger_eval(step, force, completed, None).await
    }

    /// Run the epoch `trigger` queued for `step` in the fired envs and finalize each env's
    /// batch as it completes. `restored` episodes landed before a resume and rejoin it first;
    /// when `superseding_step` returns a newer checkpoint, the unfinished episodes of this epoch
    /// are cancelled so the caller can move on to it.
    pub async fn run_epoch(
        &mut self,
        outcome: TriggerOutcome,
        step: u64,
        mut superseding_step: Option<&mut (dyn FnMut() -> Option<u64> + Send)>,
    ) -> Result<()> {
        let TriggerOutcome { fired, restored, owed } = outcome;
        for env in &fired {
            self.monitors.log_eval_plan(env.clone(), step, self.epoch_sizes[env] as u64).await;
        }
        let now = Instant::now();
        for env in &fired {
            self.triggered_at.insert((env.clone(), step), now);
        }
        info!("{}", starting_line(&fired, step, owed, restored.len()));
        let mut cancelled: Option<(u64, usize)> = None;
        // Check before the mode switch: the dispatcher schedules concurrently, so an epoch
        // that is already superseded must not start episodes it would cancel right away.
        self.cancel_if_superseded(step, &mut superseding_step, &mut cancelled).await?;
        self.dispatcher.switch_mode(Mode::PreferEval, format!("eval was triggered at step {step}"))?;

        let mut pending: BTreeSet<String> = fired.iter().filter(|env| self.epoch_sizes[*env] > 0).cloned().collect();
        for episode in restored {
            self.land(episode, &mut pending).await?;
        }
        let mut poll = tokio::time::interval(Duration::from_secs(POLL_INTERVAL_S));
        poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        while !pending.is_empty() {
            self.cancel_if_superseded(step, &mut superseding_step, &mut cancelled).await?;
            tokio::select! {
                item = self.results.recv() => {
                    let Some(item) = item else { bail!("dispatcher stopped before the eval epoch finished") };
                    let epoch = match item {
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
                    if let Some(epoch) = epoch {
                        pending.remove(&epoch.env_name);
                        self.finalize(epoch).await?;
                    }
                    self.publish_progress();
                }
                _ = poll.tick(), if superseding_step.is_some() => {}
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

    /// Cancel the epoch's unfinished episodes once a newer checkpoint is ready (once per epoch).
    async fn cancel_if_superseded(
        &self,
        step: u64,
        superseding_step: &mut Option<&mut (dyn FnMut() -> Option<u64> + Send)>,
        cancelled: &mut Option<(u64, usize)>,
    ) -> Result<()> {
        if cancelled.is_some() {
            return Ok(());
        }
        if let Some(newer) = superseding_step.as_mut().and_then(|f| f()) {
            warn!("Checkpoint {newer} is ready - cancelling unfinished eval episodes for step {step}");
            *cancelled = Some((newer, self.dispatcher.cancel_eval_step(step).await?));
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

    fn publish_progress(&self) {
        self.progress.send_replace(self.sink.batch_progress());
    }

    /// One episode of the epoch, arrived or restored: through the monitors and into its env's
    /// batch, which is finalized once the epoch is complete.
    async fn land(&mut self, episode: Episode, pending: &mut BTreeSet<String>) -> Result<()> {
        let step = eval_step(&episode)?;
        let env = env_name(&episode);
        let episode = Arc::new(episode);
        self.monitors.log_episodes(vec![episode.clone()], step, Kind::Eval, Subset::All).await;
        // Env servers keep every episode's training arrays; eval never trains on them.
        release_episode(&episode);
        let epoch = self.sink.add(&env, step, episode);
        self.publish_progress();
        if let Some(epoch) = epoch {
            pending.remove(&epoch.env_name);
            self.finalize(epoch).await?;
        }
        Ok(())
    }

    /// Persist + log one completed eval epoch through the monitors: effective episodes plus
    /// the `eval/{env}/...` metric dict.
    async fn finalize(&mut self, epoch: EvalEpoch<Arc<Episode>>) -> Result<()> {
        let EvalEpoch { env_name, step, episodes, failures, cancelled } = epoch;
        if episodes.is_empty() && failures.is_empty() && cancelled == 0 {
            warn!("Eval @ step={step} env={env_name}: no attempts returned, skipping log");
            return Ok(());
        }
        let all = EvalEpisodes::new(episodes.clone(), self.group_sizes[&env_name]);
        let effective = all.effective();
        if !effective.is_empty() {
            let records: Vec<Arc<Episode>> = effective.vf_episodes().into_iter().map(Arc::new).collect();
            self.monitors.log_episodes(records.clone(), step, Kind::Eval, Subset::Effective).await;
            self.monitors.log_annotations(stamp_batch(records.iter().map(|e| e.as_ref()), step)).await;
        }
        self.monitors.log_eval_epoch(env_name.clone(), step, all.vf_episodes().into_iter().map(Arc::new).collect()).await;

        let prefix = format!("eval/{env_name}");
        let total_attempts = episodes.len() + failures.len() + cancelled;
        let mut metrics = Metrics::new();
        metrics.extend(all.to_wandb(&prefix, Subset::All));
        metrics.extend(effective.to_wandb(&prefix, Subset::Effective));
        metrics.extend(dispatch_failure_metrics(&failures, &format!("{prefix}/all"), total_attempts));
        if cancelled > 0 {
            metrics.insert(format!("{prefix}/all/cancelled/count"), (cancelled as f64).into());
            metrics.insert(format!("{prefix}/all/cancelled/mean"), (cancelled as f64 / total_attempts as f64).into());
        }
        metrics.insert(format!("{prefix}/policy_version"), (step as f64).into());
        metrics.insert("step".into(), (step as f64).into());
        self.monitors.log_metrics(metrics.into(), Some(step)).await;

        let (reward, turns, branches, truncated) = effective
            .with_metrics(|m| (m.reward().mean(), m.num_turns().mean(), m.num_branches().mean(), m.is_truncated().mean()));
        let (error, timeout) = all.with_metrics(|m| (m.has_error().mean(), m.is_timeout().mean()));
        let summary = EpochSummary { reward, turns, branches, truncated, error, timeout };
        let elapsed = self.triggered_at.remove(&(env_name.clone(), step)).map_or(0.0, |t| t.elapsed().as_secs_f64());
        if cancelled > 0 {
            warn!(
                "{}",
                partially_evaluated_line(&env_name, step, elapsed, &summary, episodes.len(), cancelled, total_attempts)
            );
        } else {
            prl_io::success!("{}", evaluated_line(&env_name, step, elapsed, &summary));
        }
        Ok(())
    }

    /// Stop the background loggers so nothing logs after the monitors finalize.
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
        while let Some(joined) = self.actors.join_next().await {
            if let Ok(Err(e)) = joined {
                tracing::debug!("eval component ended with {e:#}");
            }
        }
        for router in std::mem::take(&mut self.envs) {
            for client in router.clients() {
                client.close().await;
            }
        }
    }
}

/// The pipeline view for the periodic logger: per-env epoch progress plus the in-flight pool
/// against the controller's current cap.
async fn collect_pipeline_view(
    dispatcher: &DispatcherHandle,
    concurrency: &ConcurrencyHandle,
    progress: Progress,
) -> (String, Metrics) {
    let (Ok(view), Ok(gauges)) = (dispatcher.view().await, concurrency.gauges().await) else {
        return ("Eval components stopped".into(), Metrics::new());
    };
    let signal = severity_name(gauges.get("concurrency/signal").and_then(|s| s.as_f64()));
    let body = pipeline_body(progress, view.inflight_eval, view.max_inflight, signal, &view.stage_counts);
    let mut payload = view.gauges;
    payload.extend(view.drained);
    payload.extend(gauges);
    (body, payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_match_python_format() {
        let s = EpochSummary { reward: 0.123456, turns: 1.0, branches: 1.25, truncated: 0.0625, error: 0.00049, timeout: 1.0 };
        assert_eq!(
            evaluated_line("gsm8k", 0, 75.0, &s),
            "Evaluated gsm8k (Step 0) |  1m 15s | Reward 0.1235 | Turns 1.0 | Branches 1.2 | Error 0.0% | Truncation 6.2% | Timeout 100.0%"
        );
        assert_eq!(
            partially_evaluated_line("gsm8k", 5, 0.5, &s, 3, 5, 8),
            "Partially evaluated gsm8k (Step 5) |   500ms | Reward 0.1235 | Error 0.0% | Completed 3/8 | Cancelled 5/8"
        );
        assert_eq!(starting_line(&["a".into(), "b".into()], 0, 12, 0), "Starting evals in a, b at step 0 (12 total rollouts)");
        assert_eq!(starting_line(&["a".into()], 3, 2, 4), "Starting evals in a at step 3 (2 total rollouts, 4 restored)");
    }

    #[test]
    fn pipeline_body_matches_python() {
        assert_eq!(pipeline_body(vec![], 0, 128, "clear", ""), "Idle; 0 inflight episodes (cap 128, signal clear)");
        let progress = vec![("b".into(), 0, 1, 3), ("a".into(), 0, 2, 4)];
        assert_eq!(
            pipeline_body(progress, 5, 64, "soft", "model=3, env=2"),
            "a 2/4 (50.0%) | b 1/3 (33.3%); 5 inflight episodes (cap 64, signal soft) - model=3, env=2"
        );
    }
}
