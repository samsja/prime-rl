//! Schedules environment runs under a shared permit counter and emits every dispatched
//! attempt exactly once (episode, `DispatchFailure`, or under a `GroupCancellation`).
//! Port of `dispatcher.py` as an actor.
//!
//! - Capacity (`max_inflight`) is shared across train + eval; one permit is one episode. The
//!   cap follows the concurrency controller's `limit` watch; net admissions are burst-capped
//!   per 5 s window (natural completions refund one admission).
//! - Optional rate limiting (`dispatch_per_minute`, aiolimiter leaky-bucket semantics),
//!   non-blocking: a limited admission arms a wake-up instead of stalling the actor.
//! - `Mode::PreferTrain` / `PreferEval` picks the kind to schedule; PREFER_EVAL flips back once
//!   the eval queue and every opened eval group are fully dispatched.
//! - Train scheduling respects the dispatch gate `(progress_step - 1) - policy.version <=
//!   TARGET_LAG`, read live from the `policy` and `progress_step` watches.
//! - Results go to per-kind outboxes drained into the sink channels inside the select loop,
//!   so commands (drain, overload) are served while a sink is busy; results are handed over
//!   with `try_send` before every scheduling pass, and scheduling pauses only while a sink
//!   channel is full (the bounded-queue backpressure of Python's `out_q`).

use std::collections::{BTreeSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use indexmap::IndexMap;
use prl_io::clients::InferenceClient;
use prl_monitors::{LiveEvent, Monitors};
use prl_types::{
    CancelReason, DispatchFailure, DispatchResult, ErrorInfo, GroupCancellation, Kind, Metrics, Policy, RunIdentity,
    TaskRecord,
};
use prl_wire::{
    EnvClientError, Episode, Error as VfError, GroupInfo, OnDelta, PolicySpan, RunInfo, TrainRunInfo, Value, WorkInfo,
};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::{AbortHandle, JoinSet};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::concurrency::ConcurrencyHandle;
use crate::envs::ReplicaRouter;
use crate::eval_source::{EvalSource, TaskRequest};
use crate::live::{self, DispatchInfo, LiveTrace};
use crate::train_source::TrainSourceHandle;

pub const LIVE_INTERVAL: Duration = Duration::from_millis(500);
/// Buffered live events before the oldest deltas are dropped.
pub const LIVE_EVENT_CAP: usize = 20_000;
pub const ADMISSION_WINDOW: Duration = Duration::from_secs(5);
/// Maximum shipped batches inference has not applied the policy for yet.
pub const TARGET_LAG: i64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    PreferTrain,
    PreferEval,
}

/// One `run` request against an env server.
#[derive(Debug, Clone)]
pub struct RunRequest {
    pub task_data: serde_json::Value,
    /// vf `ClientConfig`, as dumped.
    pub client: serde_json::Value,
    pub model: String,
    /// vf `SamplingConfig`, as dumped (with `cache_salt` when salted).
    pub sampling: prl_wire::Map,
}

/// Delete the env-server arrays of an episode no sink will train on.
pub fn release_episode(episode: &Episode) {
    prl_wire::payload::delete_payloads(episode.traces.iter().flat_map(prl_wire::payload::trace_payload_urls));
}

/// An env as the dispatcher schedules it.
pub struct DispatchEnv {
    pub name: String,
    pub group_size: usize,
    pub runner: Arc<ReplicaRouter>,
    /// vf `SamplingConfig` dump (`prl_wire::sampling`).
    pub sampling: prl_wire::Map,
    pub client: serde_json::Value,
    /// Fixed model of a frozen generation source; None samples the live policy.
    pub frozen_model: Option<String>,
    /// Releases the inference sessions of finished episodes.
    pub sessions: Arc<InferenceClient>,
}

impl DispatchEnv {
    fn live_sourced(&self) -> bool {
        self.frozen_model.is_none()
    }
}

pub struct TrainSide {
    pub envs: IndexMap<String, DispatchEnv>,
    pub source: TrainSourceHandle,
    pub progress_step: watch::Receiver<u64>,
    pub results: mpsc::Sender<DispatchResult<Episode>>,
}

pub struct EvalSide {
    pub envs: IndexMap<String, DispatchEnv>,
    pub source: EvalSource,
    pub results: mpsc::Sender<DispatchResult<Episode>>,
}

pub struct DispatcherConfig {
    pub dispatch_per_minute: Option<f64>,
    pub run: RunIdentity,
    /// Live-sourced endpoints receive weight updates, so their requests carry a cache salt.
    pub policy_weights_change: bool,
}

pub struct DispatcherIo {
    pub policy: watch::Receiver<Policy>,
    pub limit: watch::Receiver<usize>,
    pub inflight: watch::Sender<usize>,
    pub overload: mpsc::UnboundedReceiver<usize>,
    pub concurrency: Option<ConcurrencyHandle>,
    pub live: Monitors,
    /// Set once draining with nothing in flight, no eval work, and every result handed off.
    pub idle: watch::Sender<bool>,
}

#[derive(Debug, Clone)]
pub struct TriggerOutcome {
    pub fired: Vec<String>,
    pub restored: Vec<Episode>,
    /// Rollouts queued for the triggered step.
    pub owed: usize,
}

#[derive(Debug, Clone)]
pub struct DispatcherView {
    pub gauges: Metrics,
    pub drained: Metrics,
    pub inflight_by_env: IndexMap<(Kind, String), usize>,
    pub inflight_train: usize,
    pub inflight_eval: usize,
    pub stage_counts: String,
    pub max_inflight: usize,
}

enum Command {
    Start,
    SwitchMode { mode: Mode, reason: String },
    VersionPending { ack: oneshot::Sender<()> },
    VersionApplied { ack: oneshot::Sender<()> },
    TriggerEval { step: u64, force: bool, completed: Vec<Episode>, reason: Option<String>, reply: oneshot::Sender<TriggerOutcome> },
    CancelEvalStep { step: u64, reply: oneshot::Sender<usize> },
    Drain { reply: oneshot::Sender<Vec<GroupCancellation>> },
    View { reply: oneshot::Sender<DispatcherView> },
    Stop { ack: oneshot::Sender<()> },
}

#[derive(Clone)]
pub struct DispatcherHandle {
    tx: mpsc::UnboundedSender<Command>,
}

fn stopped() -> anyhow::Error {
    anyhow::anyhow!("dispatcher stopped")
}

impl DispatcherHandle {
    async fn call<T>(&self, make: impl FnOnce(oneshot::Sender<T>) -> Command) -> Result<T> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(make(tx)).map_err(|_| stopped())?;
        rx.await.map_err(|_| stopped())
    }

    /// Begin scheduling (the dispatcher idles until started).
    pub fn start(&self) -> Result<()> {
        self.tx.send(Command::Start).map_err(|_| stopped())
    }

    pub fn switch_mode(&self, mode: Mode, reason: impl Into<String>) -> Result<()> {
        self.tx.send(Command::SwitchMode { mode, reason: reason.into() }).map_err(|_| stopped())
    }

    /// Block scheduling across a weight update (`on_version_pending`).
    pub async fn version_pending(&self) -> Result<()> {
        self.call(|ack| Command::VersionPending { ack }).await
    }

    /// Resume scheduling after the new policy is live (`on_new_version`).
    pub async fn version_applied(&self) -> Result<()> {
        self.call(|ack| Command::VersionApplied { ack }).await
    }

    /// Fire eligible eval epochs; with a `reason`, switches to PREFER_EVAL if anything fires.
    pub async fn trigger_eval(
        &self,
        step: u64,
        force: bool,
        completed: Vec<Episode>,
        reason: Option<String>,
    ) -> Result<TriggerOutcome> {
        self.call(|reply| Command::TriggerEval { step, force, completed, reason, reply }).await
    }

    /// Cancel queued and active eval groups of a superseded step; scheduling stays paused
    /// until `version_applied`.
    pub async fn cancel_eval_step(&self, step: u64) -> Result<usize> {
        self.call(|reply| Command::CancelEvalStep { step, reply }).await
    }

    /// Stop train scheduling and cancel in-flight train work; the cancellations come back
    /// directly instead of through the result channel.
    pub async fn drain(&self) -> Result<Vec<GroupCancellation>> {
        self.call(|reply| Command::Drain { reply }).await
    }

    pub async fn view(&self) -> Result<DispatcherView> {
        self.call(|reply| Command::View { reply }).await
    }

    /// Cancel everything in flight (no markers), flush the live view and exit.
    pub async fn stop(&self) -> Result<()> {
        self.call(|ack| Command::Stop { ack }).await
    }
}

/// aiolimiter's `AsyncLimiter(max_rate, 60)`: a leaky bucket of `max_rate` tokens.
#[derive(Debug)]
struct RateLimiter {
    max_rate: f64,
    rate_per_sec: f64,
    level: f64,
    last: Instant,
}

impl RateLimiter {
    fn new(per_minute: f64) -> Self {
        Self { max_rate: per_minute, rate_per_sec: per_minute / 60.0, level: 0.0, last: Instant::now() }
    }

    fn try_acquire(&mut self, now: Instant) -> std::result::Result<(), Instant> {
        let elapsed = now.saturating_duration_since(self.last).as_secs_f64();
        self.level = (self.level - elapsed * self.rate_per_sec).max(0.0);
        self.last = now;
        if self.level + 1.0 <= self.max_rate {
            self.level += 1.0;
            return Ok(());
        }
        Err(now + Duration::from_secs_f64((self.level + 1.0 - self.max_rate) / self.rate_per_sec))
    }
}

/// Per-tick drain counters for the pipeline view (cleared on read).
#[derive(Debug, Default)]
struct DrainCounters {
    cancelled: IndexMap<(Kind, String), usize>,
    errored: IndexMap<(Kind, String), usize>,
}

impl DrainCounters {
    fn record_cancellation(&mut self, kind: Kind, env: &str, n: usize) {
        *self.cancelled.entry((kind, env.to_owned())).or_default() += n;
    }

    fn record_error(&mut self, kind: Kind, env: &str) {
        *self.errored.entry((kind, env.to_owned())).or_default() += 1;
    }

    /// The full pre-registered key set every tick (zeros when idle), then clear.
    fn drained(&mut self, train_envs: &[String], eval_envs: &[String]) -> Metrics {
        let get = |map: &IndexMap<(Kind, String), usize>, kind: Kind, env: &str| {
            map.get(&(kind, env.to_owned())).copied().unwrap_or(0)
        };
        let mut out = Metrics::new();
        for (kind, envs) in [(Kind::Train, train_envs), (Kind::Eval, eval_envs)] {
            let cancelled: usize = envs.iter().map(|e| get(&self.cancelled, kind, e)).sum();
            let errored: usize = envs.iter().map(|e| get(&self.errored, kind, e)).sum();
            out.insert(format!("dispatcher/cancelled/{kind}"), (cancelled as f64).into());
            out.insert(format!("dispatcher/errored/{kind}"), (errored as f64).into());
        }
        let mut all: Vec<&String> = train_envs.iter().collect();
        all.extend(eval_envs.iter().filter(|e| !train_envs.contains(e)));
        for env in all {
            let c = get(&self.cancelled, Kind::Train, env) + get(&self.cancelled, Kind::Eval, env);
            let e = get(&self.errored, Kind::Train, env) + get(&self.errored, Kind::Eval, env);
            out.insert(format!("dispatcher/cancelled/{env}"), (c as f64).into());
            out.insert(format!("dispatcher/errored/{env}"), (e as f64).into());
        }
        self.cancelled.clear();
        self.errored.clear();
        out
    }
}

#[derive(Debug)]
struct GroupState {
    kind: Kind,
    env_name: String,
    task: Arc<TaskRecord>,
    step: u64,
    episodes_to_schedule: usize,
    target_episodes: usize,
    emitted: usize,
    policy_version_at_start: u64,
}

struct Inflight {
    kind: Kind,
    env_name: String,
    group_id: String,
    task: Arc<TaskRecord>,
    policy_version: u64,
    step: u64,
    started_at: Instant,
    info: DispatchInfo,
    live: IndexMap<String, LiveTrace>,
    session_ids: BTreeSet<String>,
    abort: AbortHandle,
    /// Latched when the episode is claimed, so its run stops relaying deltas.
    claimed: Arc<Mutex<bool>>,
    sessions: Arc<InferenceClient>,
}

type Completion = (u64, std::result::Result<Episode, EnvClientError>);

enum Scheduled {
    Yes,
    No,
    WaitUntil(Instant),
}

pub struct Dispatcher {
    cfg: DispatcherConfig,
    train: Option<TrainSide>,
    eval: Option<EvalSide>,
    policy: watch::Receiver<Policy>,
    inflight_tx: watch::Sender<usize>,
    concurrency: Option<ConcurrencyHandle>,
    live_sink: Monitors,
    idle_tx: watch::Sender<bool>,

    max_inflight: usize,
    rate_limiter: Option<RateLimiter>,
    admission_window_start: Instant,
    admissions_in_window: usize,
    min_burst: usize,

    inflight: IndexMap<u64, Inflight>,
    next_slot: u64,
    groups: IndexMap<String, GroupState>,
    cache_salt_prefix: String,
    train_out: VecDeque<DispatchResult<Episode>>,
    eval_out: VecDeque<DispatchResult<Episode>>,

    mode: Mode,
    started: bool,
    train_disabled: bool,
    draining: bool,
    policy_update_pending: bool,
    gate_open: bool,
    counters: DrainCounters,
    live_events: Vec<LiveEvent>,
    /// Deltas among `live_events`.
    live_deltas: usize,
    live_dropped: bool,
    wake_at: Option<Instant>,
    delta_tx: mpsc::UnboundedSender<(u64, Value)>,
}

fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn py_opt_repr(s: Option<&str>) -> String {
    s.map_or_else(|| "None".to_string(), py_repr)
}

impl Dispatcher {
    pub fn new(
        cfg: DispatcherConfig,
        train: Option<TrainSide>,
        eval: Option<EvalSide>,
        io: DispatcherIo,
    ) -> (Self, DispatcherHandle, DispatcherInbox) {
        let (tx, cmd_rx) = mpsc::unbounded_channel();
        let (delta_tx, delta_rx) = mpsc::unbounded_channel();
        let min_burst = train.as_ref().and_then(|t| t.envs.values().map(|e| e.group_size).max()).unwrap_or(8);
        let max_inflight = *io.limit.borrow();
        let this = Self {
            rate_limiter: cfg.dispatch_per_minute.filter(|r| *r > 0.0).map(RateLimiter::new),
            cfg,
            train,
            eval,
            policy: io.policy,
            inflight_tx: io.inflight,
            concurrency: io.concurrency,
            live_sink: io.live,
            idle_tx: io.idle,
            max_inflight,
            admission_window_start: Instant::now(),
            admissions_in_window: 0,
            min_burst,
            inflight: IndexMap::new(),
            next_slot: 0,
            groups: IndexMap::new(),
            cache_salt_prefix: uuid::Uuid::new_v4().simple().to_string()[..8].to_string(),
            train_out: VecDeque::new(),
            eval_out: VecDeque::new(),
            mode: Mode::PreferTrain,
            started: false,
            train_disabled: false,
            draining: false,
            policy_update_pending: false,
            gate_open: true,
            counters: DrainCounters::default(),
            live_events: Vec::new(),
            live_deltas: 0,
            live_dropped: false,
            wake_at: None,
            delta_tx,
        };
        let inbox = DispatcherInbox { cmd_rx, delta_rx, limit: io.limit, overload: io.overload };
        (this, DispatcherHandle { tx }, inbox)
    }

    // ── counts ─────────────────────────────────────────────────────────────

    fn inflight_count(&self, kind: Kind) -> usize {
        self.inflight.values().filter(|m| m.kind == kind).count()
    }

    fn eval_has_work(&self) -> bool {
        self.eval.as_ref().is_some_and(|e| !e.source.is_empty())
            || self.groups.values().any(|g| g.kind == Kind::Eval && g.episodes_to_schedule > 0)
    }

    fn is_idle(&self) -> bool {
        self.inflight.is_empty() && !self.eval_has_work() && self.train_out.is_empty() && self.eval_out.is_empty()
    }

    fn train_env_names(&self) -> Vec<String> {
        self.train.as_ref().map(|t| t.envs.keys().cloned().collect()).unwrap_or_default()
    }

    fn eval_env_names(&self) -> Vec<String> {
        self.eval.as_ref().map(|e| e.envs.keys().cloned().collect()).unwrap_or_default()
    }

    fn env(&self, kind: Kind, name: &str) -> &DispatchEnv {
        match kind {
            Kind::Train => &self.train.as_ref().expect("train side").envs[name],
            Kind::Eval => &self.eval.as_ref().expect("eval side").envs[name],
        }
    }

    // ── gate ───────────────────────────────────────────────────────────────

    fn lead(&self) -> Option<(i64, u64, u64)> {
        let train = self.train.as_ref()?;
        let step = *train.progress_step.borrow();
        let version = self.policy.borrow().version;
        Some(((step as i64 - 1) - version as i64, step, version))
    }

    fn gate_allows(&self) -> bool {
        self.lead().is_none_or(|(lead, _, _)| lead <= TARGET_LAG)
    }

    /// Log gate transitions on every step / policy change (`update_dispatch_gate`).
    fn update_gate(&mut self) {
        // Python evaluates the gate first when the startup policy is applied; transitions
        // before the dispatcher starts are not logged.
        if !self.started {
            return;
        }
        let Some((lead, step, version)) = self.lead() else { return };
        let open = lead <= TARGET_LAG;
        if open == self.gate_open {
            return;
        }
        if open {
            info!("Resuming dispatcher (policy v{version})");
        } else {
            info!(
                "Pausing dispatcher until inference applies policy v{} (currently v{version})",
                step as i64 - 1 - TARGET_LAG
            );
        }
        self.gate_open = open;
    }

    // ── admission ──────────────────────────────────────────────────────────

    fn admission_budget(&mut self, now: Instant) -> i64 {
        if now.saturating_duration_since(self.admission_window_start) >= ADMISSION_WINDOW {
            self.admission_window_start = now;
            self.admissions_in_window = 0;
        }
        let burst_cap = self.min_burst.max(self.max_inflight / 10);
        burst_cap as i64 - self.admissions_in_window as i64
    }

    fn arm_wake(&mut self, at: Instant) {
        self.wake_at = Some(self.wake_at.map_or(at, |w| w.min(at)));
    }

    fn switch_mode(&mut self, mode: Mode, reason: &str) {
        if mode == self.mode {
            return;
        }
        let prefer = if mode == Mode::PreferEval { "eval" } else { "train" };
        info!("Switching dispatcher mode to prefer {prefer} episodes because {reason}");
        self.mode = mode;
    }

    /// Schedule up to the cap, honoring the mode. Eval ignores the dispatch gate.
    async fn fill(&mut self, tasks: &mut JoinSet<Completion>) -> Result<()> {
        loop {
            if !self.started || self.policy_update_pending {
                return Ok(());
            }
            if !self.train_out.is_empty() || !self.eval_out.is_empty() {
                return Ok(());
            }
            let now = Instant::now();
            if self.max_inflight <= self.inflight.len() {
                return Ok(());
            }
            if self.admission_budget(now) <= 0 {
                self.arm_wake(self.admission_window_start + ADMISSION_WINDOW);
                return Ok(());
            }
            let kind = match self.mode {
                Mode::PreferEval => {
                    if !self.eval_has_work() {
                        self.switch_mode(Mode::PreferTrain, "the eval queue drained");
                        continue;
                    }
                    Kind::Eval
                }
                Mode::PreferTrain => {
                    if !self.gate_allows() {
                        return Ok(());
                    }
                    Kind::Train
                }
            };
            match self.try_schedule(kind, tasks).await? {
                Scheduled::Yes => {}
                Scheduled::No => return Ok(()),
                Scheduled::WaitUntil(at) => {
                    self.arm_wake(at);
                    return Ok(());
                }
            }
        }
    }

    /// Continue an open group of `kind` (prefix-cache locality) or open a fresh one.
    async fn try_schedule(&mut self, kind: Kind, tasks: &mut JoinSet<Completion>) -> Result<Scheduled> {
        match kind {
            Kind::Train if self.train.is_none() || self.train_disabled => return Ok(Scheduled::No),
            Kind::Eval if self.eval.is_none() => return Ok(Scheduled::No),
            _ => {}
        }
        if let Some(gid) =
            self.groups.iter().find(|(_, g)| g.kind == kind && g.episodes_to_schedule > 0).map(|(k, _)| k.clone())
        {
            return self.schedule_group_episode(&gid, tasks);
        }
        let request = match kind {
            Kind::Train => {
                let train = self.train.as_ref().expect("train side");
                let step = *train.progress_step.borrow();
                Some(train.source.next_task(step).await?)
            }
            Kind::Eval => self.eval.as_mut().expect("eval side").source.next_task(),
        };
        let Some(request) = request else { return Ok(Scheduled::No) };
        let TaskRequest { env_name, task, step, rollouts, group_id } = request;
        let rollouts = rollouts.unwrap_or_else(|| self.env(kind, &env_name).group_size);
        let gid = group_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let group = GroupState {
            kind,
            env_name,
            task: Arc::new(task),
            step,
            episodes_to_schedule: rollouts,
            target_episodes: rollouts,
            emitted: 0,
            policy_version_at_start: self.policy.borrow().version,
        };
        self.groups.insert(gid.clone(), group);
        self.schedule_group_episode(&gid, tasks)
    }

    fn schedule_group_episode(&mut self, gid: &str, tasks: &mut JoinSet<Completion>) -> Result<Scheduled> {
        let now = Instant::now();
        if let Some(limiter) = &mut self.rate_limiter {
            if let Err(at) = limiter.try_acquire(now) {
                return Ok(Scheduled::WaitUntil(at));
            }
        }
        let policy = self.policy.borrow().clone();
        let group = &self.groups[gid];
        let (kind, env_name, task, step, version) =
            (group.kind, group.env_name.clone(), group.task.clone(), group.step, group.policy_version_at_start);
        let env = self.env(kind, &env_name);
        let model = env.frozen_model.clone().unwrap_or_else(|| policy.model_name.clone());
        // Only endpoints receiving policy weight updates need a cache salt.
        let salt = (env.live_sourced() && self.cfg.policy_weights_change)
            .then(|| format!("{}-{version}", self.cache_salt_prefix));
        let sampling = prl_wire::sampling::with_cache_salt(&env.sampling, salt.as_deref());
        let request = RunRequest { task_data: task.data.clone(), client: env.client.clone(), model, sampling };
        let runner = env.runner.clone();
        let sessions = env.sessions.clone();

        self.groups[gid].episodes_to_schedule -= 1;
        self.admissions_in_window += 1;

        let slot = self.next_slot;
        self.next_slot += 1;
        let claimed = Arc::new(Mutex::new(false));
        let on_delta: OnDelta = {
            let claimed = claimed.clone();
            let tx = self.delta_tx.clone();
            Box::new(move |delta: &Value| {
                let claimed = claimed.lock().expect("claim latch");
                if !*claimed {
                    let _ = tx.send((slot, delta.clone()));
                }
            })
        };
        let abort = tasks.spawn(async move { (slot, runner.run(request, on_delta).await) });
        let info = DispatchInfo {
            id: uuid::Uuid::new_v4().simple().to_string(),
            kind,
            env: env_name.clone(),
            group: gid.to_owned(),
            task: live::task_label(&task.data),
            policy_version: version,
            step,
            started: prl_monitors::file::unix_time(),
        };
        self.live_events.push(live::pending_event(&info));
        self.inflight.insert(
            slot,
            Inflight {
                kind,
                env_name,
                group_id: gid.to_owned(),
                task,
                policy_version: version,
                step,
                started_at: now,
                info,
                live: IndexMap::new(),
                session_ids: BTreeSet::new(),
                abort,
                claimed,
                sessions,
            },
        );
        self.inflight_tx.send_replace(self.inflight.len());
        Ok(Scheduled::Yes)
    }

    // ── live ───────────────────────────────────────────────────────────────

    fn on_delta(&mut self, slot: u64, delta: Value) {
        let Some(meta) = self.inflight.get_mut(&slot) else { return };
        if let Some(trace) = delta.get("trace").and_then(Value::as_str) {
            meta.session_ids.insert(trace.to_owned());
        }
        let first = meta.live.is_empty();
        live::apply(&mut meta.live, &delta);
        let streamed = first && !meta.live.is_empty();
        let dispatch = meta.info.to_value();
        let dispatched = live::dispatched_event(&meta.info);
        self.queue_live(LiveEvent::Delta { delta, dispatch });
        // After the delta, and only once a trace streams.
        if streamed {
            self.live_events.push(dispatched);
        }
    }

    /// Buffer a delta for the next flush. Past the cap the oldest deltas go (never the
    /// bookkeeping events that create or remove files). Trimming waits until the deltas alone
    /// exceed the cap, so it stays amortized O(1) per delta even when bookkeeping events of
    /// many in-flight episodes fill the buffer.
    fn queue_live(&mut self, event: LiveEvent) {
        self.live_events.push(event);
        self.live_deltas += 1;
        if self.live_events.len() > LIVE_EVENT_CAP && self.live_deltas > LIVE_EVENT_CAP {
            let (deltas, mut kept): (Vec<_>, Vec<_>) =
                std::mem::take(&mut self.live_events).into_iter().partition(|e| matches!(e, LiveEvent::Delta { .. }));
            let dropped = deltas.len().saturating_sub(LIVE_EVENT_CAP / 2);
            kept.extend(deltas.into_iter().skip(dropped));
            self.live_events = kept;
            self.live_deltas = LIVE_EVENT_CAP / 2;
            if !self.live_dropped {
                self.live_dropped = true;
                warn!("Live trace buffer over {LIVE_EVENT_CAP} events - dropping the oldest deltas");
            }
        }
    }

    async fn flush_live(&mut self) {
        if self.live_events.is_empty() {
            return;
        }
        let events = std::mem::take(&mut self.live_events);
        self.live_deltas = 0;
        self.live_sink.log_live(events).await;
    }

    /// The episode left the in-flight set: its live traces are over (and its placeholder if
    /// no trace ever streamed), and its inference sessions are released.
    fn retire(&mut self, meta: &Inflight, episode: Option<&Episode>) {
        if meta.live.is_empty() {
            self.live_events.push(live::dispatched_event(&meta.info));
        }
        self.live_events.extend(meta.live.keys().map(|id| LiveEvent::Done(id.clone())));
        let mut ids = meta.session_ids.clone();
        if let Some(episode) = episode {
            ids.extend(episode.traces.iter().map(|t| t.id.clone()));
        }
        let sessions = meta.sessions.clone();
        let ids: Vec<String> = ids.into_iter().collect();
        if !ids.is_empty() {
            tokio::spawn(async move { sessions.finish_sessions(&ids).await });
        }
    }

    /// Latch and abort a claimed episode; a run that already returned is released by
    /// `handle_completed` when its result surfaces.
    fn claim(&mut self, meta: &Inflight) {
        *meta.claimed.lock().expect("claim latch") = true;
        meta.abort.abort();
    }

    // ── completion ─────────────────────────────────────────────────────────

    fn push_result(&mut self, kind: Kind, result: DispatchResult<Episode>) {
        match kind {
            Kind::Train => self.train_out.push_back(result),
            Kind::Eval => self.eval_out.push_back(result),
        }
    }

    fn complete_group_member(&mut self, meta: &Inflight) -> u64 {
        let Some(group) = self.groups.get_mut(&meta.group_id) else { return meta.policy_version };
        let version = group.policy_version_at_start;
        group.emitted += 1;
        if group.emitted >= group.target_episodes {
            self.groups.shift_remove(&meta.group_id);
        }
        version
    }

    fn handle_completed(&mut self, slot: u64, result: std::result::Result<Episode, EnvClientError>) -> Result<()> {
        let Some(meta) = self.inflight.shift_remove(&slot) else {
            // Claimed (dropped / cancelled) after it returned: nobody will see its arrays.
            if let Ok(episode) = &result {
                release_episode(episode);
            }
            return Ok(());
        };
        self.retire(&meta, result.as_ref().ok());
        // A natural completion refunds one admission (cancellations do not).
        self.admissions_in_window = self.admissions_in_window.saturating_sub(1);
        self.inflight_tx.send_replace(self.inflight.len());
        let (gid, env) = (meta.group_id.as_str(), meta.env_name.as_str());
        let mut episode = match result {
            Err(error) => {
                let (type_name, message) = (error.type_name().to_string(), error.to_string());
                warn!("Environment request failed in group {gid} ({env}): {type_name}({})", py_repr(&message));
                self.counters.record_error(meta.kind, env);
                let policy_version = self.complete_group_member(&meta);
                let failure = DispatchFailure {
                    kind: meta.kind,
                    env_name: meta.env_name.clone(),
                    group_id: meta.group_id.clone(),
                    step: meta.step,
                    policy_version,
                    task_type: "Task".into(),
                    task_key: meta.task.key.clone(),
                    task_hash: meta.task.hash.clone(),
                    error: ErrorInfo {
                        traceback: Some(format!("{type_name}: {message}\n")),
                        type_: type_name,
                        message,
                        status_code: None,
                    },
                };
                self.push_result(meta.kind, DispatchResult::Failure(failure));
                return Ok(());
            }
            Ok(episode) => episode,
        };

        for trace in &mut episode.traces {
            if trace.is_timeout && !trace.has_error() && meta.kind == Kind::Train {
                // Training keeps a timed-out rollout out of the batch; eval keeps it as a score.
                let condition = trace.stop_condition.clone();
                trace.errors.push(VfError::new("Timeout", format!("Trace stopped by {}", condition.as_deref().unwrap_or("None"))));
                trace.ok = false;
                episode.ok = false;
            }
            if trace.is_dropped() {
                info!("Trace dropped in group {gid} ({env}) — {}", trace.dropped.as_deref().unwrap_or("None"));
            }
            if trace.has_error() {
                self.counters.record_error(meta.kind, env);
                if let Some(error) = trace.last_error() {
                    warn!("Trace failed in group {gid} ({env}) — {}: {}", error.type_, error.message);
                }
            }
        }
        if !episode.ok {
            if !episode.traces.iter().any(|t| t.has_error()) {
                self.counters.record_error(meta.kind, env);
            }
            let detail =
                episode.last_error().map(|e| format!(" — {}: {}", e.type_, e.message)).unwrap_or_default();
            warn!("Episode failed in group {gid} ({env}){detail}");
        }
        if let Some(concurrency) = &self.concurrency {
            concurrency.record_episode(episode.num_total_tokens().max(0) as u64, self.inflight.len());
        }
        self.emit_episode(&meta, episode)
    }

    fn emit_episode(&mut self, meta: &Inflight, mut episode: Episode) -> Result<()> {
        let expected = (Some(meta.task.key.as_str()), Some(meta.task.hash.as_str()));
        let actual = (episode.task.key.as_deref(), episode.task.hash.as_deref());
        if actual != expected {
            bail!(
                "Episode task provenance ({}, {}) does not match dispatched task ({}, {})",
                py_opt_repr(actual.0),
                py_opt_repr(actual.1),
                py_repr(&meta.task.key),
                py_repr(&meta.task.hash)
            );
        }
        let policy_version = self.complete_group_member(meta);
        episode.env.name = Some(meta.env_name.clone());
        episode.group = Some(GroupInfo { id: meta.group_id.clone() });
        let live_policy = self.env(meta.kind, &meta.env_name).live_sourced() || meta.kind == Kind::Eval;
        let policy = if live_policy {
            let end = self.policy.borrow().version;
            Some(PolicySpan::new(policy_version as i64, end as i64).map_err(anyhow::Error::msg)?)
        } else {
            None
        };
        let step = meta.step as i64;
        let work = match meta.kind {
            Kind::Eval => WorkInfo::Eval { step, policy },
            Kind::Train => WorkInfo::Train { step, policy },
        };
        episode.record_run(RunInfo::Train(TrainRunInfo {
            id: self.cfg.run.id.clone(),
            name: self.cfg.run.name.clone(),
            work,
        }));
        self.push_result(meta.kind, DispatchResult::Episode(episode));
        Ok(())
    }

    // ── cancellation ───────────────────────────────────────────────────────

    /// Claim every in-flight member of `gid` and emit one `GroupCancellation` covering the
    /// in-flight and never-dispatched episodes. Returns the owed count.
    fn drop_group(&mut self, gid: &str, reason: CancelReason) -> usize {
        let group = self.groups.shift_remove(gid);
        let slots: Vec<u64> = self.inflight.iter().filter(|(_, m)| m.group_id == gid).map(|(s, _)| *s).collect();
        let mut claimed = Vec::with_capacity(slots.len());
        for slot in slots {
            let meta = self.inflight.shift_remove(&slot).expect("claimed slot");
            self.retire(&meta, None);
            self.claim(&meta);
            claimed.push(meta);
        }
        self.inflight_tx.send_replace(self.inflight.len());
        let inflight_cancelled = claimed.len();
        let unscheduled = group.as_ref().map_or(0, |g| g.episodes_to_schedule);
        let cancelled = inflight_cancelled + unscheduled;
        if cancelled > 0 {
            let last = claimed.last();
            let kind = group.as_ref().map_or_else(|| last.expect("claimed").kind, |g| g.kind);
            let env_name = group.as_ref().map_or_else(|| last.expect("claimed").env_name.clone(), |g| g.env_name.clone());
            let step = group.as_ref().map_or_else(|| last.expect("claimed").step, |g| g.step);
            self.counters.record_cancellation(kind, &env_name, cancelled);
            debug!(
                "Dropped {kind} group | group={} env={env_name} reason={} | cancelled={cancelled} (inflight={inflight_cancelled} unscheduled={unscheduled})",
                &gid[..gid.len().min(8)],
                reason.as_str()
            );
            self.push_result(
                kind,
                DispatchResult::Cancelled(GroupCancellation {
                    kind,
                    env_name,
                    group_id: gid.to_owned(),
                    step,
                    count: cancelled,
                    reason,
                }),
            );
        }
        cancelled
    }

    /// Overload cut: drop roughly `n` in-flight train episodes, youngest groups first (a
    /// group's age is its oldest member's start).
    fn cancel_inflight(&mut self, n: usize) {
        let mut group_age: IndexMap<String, Instant> = IndexMap::new();
        for meta in self.inflight.values().filter(|m| m.kind == Kind::Train) {
            let age = group_age.entry(meta.group_id.clone()).or_insert(meta.started_at);
            *age = (*age).min(meta.started_at);
        }
        let mut order: Vec<(String, Instant)> = group_age.into_iter().collect();
        order.sort_by_key(|(_, age)| std::cmp::Reverse(*age));
        let mut shed = 0;
        for (gid, _) in order {
            if shed >= n {
                break;
            }
            let live = self.inflight.values().filter(|m| m.group_id == gid).count();
            self.drop_group(&gid, CancelReason::Overload);
            shed += live;
        }
        if shed > 0 {
            warn!("Cancelled {shed} youngest in-flight episodes after overload cut");
        }
    }

    /// Stop train scheduling; cancel train groups and in-flight train episodes, returning
    /// the markers directly.
    fn drain(&mut self) -> Vec<GroupCancellation> {
        self.train_disabled = true;
        self.draining = true;
        let mut by_group: IndexMap<String, GroupCancellation> = IndexMap::new();
        let train_groups: Vec<String> =
            self.groups.iter().filter(|(_, g)| g.kind == Kind::Train).map(|(k, _)| k.clone()).collect();
        for gid in train_groups {
            let group = self.groups.shift_remove(&gid).expect("train group");
            self.counters.record_cancellation(Kind::Train, &group.env_name, group.episodes_to_schedule);
            by_group.insert(
                gid.clone(),
                GroupCancellation {
                    kind: Kind::Train,
                    env_name: group.env_name,
                    group_id: gid,
                    step: group.step,
                    count: group.episodes_to_schedule,
                    reason: CancelReason::Shutdown,
                },
            );
        }
        let slots: Vec<u64> =
            self.inflight.iter().filter(|(_, m)| m.kind == Kind::Train).map(|(s, _)| *s).collect();
        for slot in slots {
            let meta = self.inflight.shift_remove(&slot).expect("train slot");
            self.retire(&meta, None);
            self.claim(&meta);
            self.counters.record_cancellation(Kind::Train, &meta.env_name, 1);
            by_group
                .entry(meta.group_id.clone())
                .or_insert_with(|| GroupCancellation {
                    kind: Kind::Train,
                    env_name: meta.env_name.clone(),
                    group_id: meta.group_id.clone(),
                    step: meta.step,
                    count: 0,
                    reason: CancelReason::Shutdown,
                })
                .count += 1;
        }
        self.inflight_tx.send_replace(self.inflight.len());
        by_group.into_values().filter(|c| c.count > 0).collect()
    }

    fn cancel_eval_step(&mut self, step: u64) -> usize {
        let Some(eval) = self.eval.as_mut() else { return 0 };
        self.policy_update_pending = true;
        let queued = eval.source.cancel_step(step);
        let gids: Vec<String> =
            self.groups.iter().filter(|(_, g)| g.kind == Kind::Eval && g.step == step).map(|(k, _)| k.clone()).collect();
        let mut cancelled: usize = gids.iter().map(|gid| self.drop_group(gid, CancelReason::Superseded)).sum();
        for request in queued {
            let count = request.rollouts.unwrap_or_else(|| self.env(Kind::Eval, &request.env_name).group_size);
            cancelled += count;
            self.counters.record_cancellation(Kind::Eval, &request.env_name, count);
            self.push_result(
                Kind::Eval,
                DispatchResult::Cancelled(GroupCancellation {
                    kind: Kind::Eval,
                    env_name: request.env_name,
                    group_id: uuid::Uuid::new_v4().to_string(),
                    step: request.step,
                    count,
                    reason: CancelReason::Superseded,
                }),
            );
        }
        cancelled
    }

    /// Shutdown: cancel everything in flight without markers.
    fn cancel_all(&mut self) {
        let slots: Vec<u64> = self.inflight.keys().copied().collect();
        for slot in slots {
            let meta = self.inflight.shift_remove(&slot).expect("slot");
            self.counters.record_cancellation(meta.kind, &meta.env_name, 1);
            self.retire(&meta, None);
            self.claim(&meta);
        }
        self.groups.clear();
        self.inflight_tx.send_replace(self.inflight.len());
    }

    fn view(&mut self) -> DispatcherView {
        let mut inflight_by_env: IndexMap<(Kind, String), usize> = IndexMap::new();
        for meta in self.inflight.values() {
            *inflight_by_env.entry((meta.kind, meta.env_name.clone())).or_default() += 1;
        }
        let (inflight_train, inflight_eval) = (self.inflight_count(Kind::Train), self.inflight_count(Kind::Eval));
        let mut gauges = Metrics::new();
        gauges.insert("dispatcher/inflight/train".into(), (inflight_train as f64).into());
        gauges.insert("dispatcher/inflight/eval".into(), (inflight_eval as f64).into());
        let queued = self.eval.as_ref().map_or(0, |e| e.source.len());
        gauges.insert("dispatcher/queued/eval".into(), (queued as f64).into());
        gauges.insert("dispatcher/mode".into(), f64::from(u8::from(self.mode == Mode::PreferEval)).into());
        let (train_envs, eval_envs) = (self.train_env_names(), self.eval_env_names());
        DispatcherView {
            gauges,
            drained: self.counters.drained(&train_envs, &eval_envs),
            inflight_by_env,
            inflight_train,
            inflight_eval,
            stage_counts: live::stage_counts(self.inflight.values().map(|m| &m.live)),
            max_inflight: self.max_inflight,
        }
    }

    // ── actor loop ─────────────────────────────────────────────────────────

    /// Returns `true` when the actor should exit.
    async fn handle_command(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::Start => {
                self.started = true;
                self.gate_open = self.gate_allows();
            }
            Command::SwitchMode { mode, reason } => self.switch_mode(mode, &reason),
            Command::VersionPending { ack } => {
                self.policy_update_pending = true;
                let _ = ack.send(());
            }
            Command::VersionApplied { ack } => {
                self.policy_update_pending = false;
                let _ = ack.send(());
            }
            Command::TriggerEval { step, force, completed, reason, reply } => {
                let outcome = match self.eval.as_mut() {
                    None => TriggerOutcome { fired: vec![], restored: vec![], owed: 0 },
                    Some(eval) => {
                        let (fired, restored) = eval.source.trigger(step, force, completed);
                        let owed = eval
                            .source
                            .queued()
                            .filter(|r| r.step == step)
                            .map(|r| r.rollouts.unwrap_or_else(|| eval.envs[&r.env_name].group_size))
                            .sum();
                        TriggerOutcome { fired, restored, owed }
                    }
                };
                if let Some(reason) = reason.filter(|_| !outcome.fired.is_empty()) {
                    self.switch_mode(Mode::PreferEval, &reason);
                }
                let _ = reply.send(outcome);
            }
            Command::CancelEvalStep { step, reply } => {
                let n = self.cancel_eval_step(step);
                let _ = reply.send(n);
            }
            Command::Drain { reply } => {
                let cancellations = self.drain();
                let _ = reply.send(cancellations);
            }
            Command::View { reply } => {
                let view = self.view();
                let _ = reply.send(view);
            }
            Command::Stop { ack } => {
                self.cancel_all();
                self.flush_live().await;
                let _ = ack.send(());
                return true;
            }
        }
        false
    }

    fn drain_deltas(&mut self, delta_rx: &mut mpsc::UnboundedReceiver<(u64, Value)>) {
        while let Ok((slot, delta)) = delta_rx.try_recv() {
            self.on_delta(slot, delta);
        }
    }

    fn flush_outbox(
        &mut self,
        train_tx: Option<&mpsc::Sender<DispatchResult<Episode>>>,
        eval_tx: Option<&mpsc::Sender<DispatchResult<Episode>>>,
    ) -> Result<()> {
        for (outbox, tx) in [(&mut self.train_out, train_tx), (&mut self.eval_out, eval_tx)] {
            let Some(tx) = tx else { continue };
            while let Some(result) = outbox.pop_front() {
                match tx.try_send(result) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(result)) => {
                        outbox.push_front(result);
                        break;
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => bail!("sink channel closed"),
                }
            }
        }
        Ok(())
    }

    /// Run until stopped or cancelled.
    pub async fn run(mut self, inbox: DispatcherInbox, cancel: CancellationToken) -> Result<()> {
        let DispatcherInbox { mut cmd_rx, mut delta_rx, mut limit, mut overload } = inbox;
        let mut tasks: JoinSet<Completion> = JoinSet::new();
        let mut live_tick = tokio::time::interval(LIVE_INTERVAL);
        live_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let train_tx = self.train.as_ref().map(|t| t.results.clone());
        let eval_tx = self.eval.as_ref().map(|e| e.results.clone());
        let mut step_rx = self.train.as_ref().map(|t| t.progress_step.clone());
        let mut policy_rx = self.policy.clone();
        let far = tokio::time::Instant::now() + Duration::from_secs(86_400 * 365);
        loop {
            // Hand results over without blocking first: the outbox only stays non-empty (and
            // stops admissions) when a sink channel is genuinely full.
            self.flush_outbox(train_tx.as_ref(), eval_tx.as_ref())?;
            self.fill(&mut tasks).await?;
            if self.draining && self.is_idle() {
                self.idle_tx.send_replace(true);
            }
            let wake = self.wake_at.map_or(far, tokio::time::Instant::from_std);
            let has_train_out = !self.train_out.is_empty();
            let has_eval_out = !self.eval_out.is_empty();
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                cmd = cmd_rx.recv() => {
                    let Some(cmd) = cmd else { return Ok(()) };
                    self.drain_deltas(&mut delta_rx);
                    if self.handle_command(cmd).await {
                        return Ok(());
                    }
                }
                Some(joined) = tasks.join_next(), if !tasks.is_empty() => {
                    match joined {
                        Ok((slot, result)) => {
                            self.drain_deltas(&mut delta_rx);
                            self.handle_completed(slot, result)?;
                        }
                        Err(err) if err.is_cancelled() => {}
                        Err(err) => return Err(anyhow::anyhow!("episode task panicked: {err}")),
                    }
                }
                Some((slot, delta)) = delta_rx.recv() => self.on_delta(slot, delta),
                Some(n) = overload.recv() => {
                    self.drain_deltas(&mut delta_rx);
                    self.cancel_inflight(n);
                }
                Ok(()) = limit.changed() => self.max_inflight = *limit.borrow_and_update(),
                Ok(()) = policy_rx.changed() => {
                    policy_rx.borrow_and_update();
                    self.update_gate();
                }
                Ok(()) = async { step_rx.as_mut().expect("train side").changed().await }, if step_rx.is_some() => {
                    self.update_gate();
                }
                permit = async { train_tx.as_ref().expect("train sink").reserve().await }, if has_train_out => {
                    let permit = permit.map_err(|_| anyhow::anyhow!("train sink closed"))?;
                    permit.send(self.train_out.pop_front().expect("train outbox"));
                }
                permit = async { eval_tx.as_ref().expect("eval sink").reserve().await }, if has_eval_out => {
                    let permit = permit.map_err(|_| anyhow::anyhow!("eval sink closed"))?;
                    permit.send(self.eval_out.pop_front().expect("eval outbox"));
                }
                _ = live_tick.tick() => self.flush_live().await,
                _ = tokio::time::sleep_until(wake), if self.wake_at.is_some() => self.wake_at = None,
            }
        }
    }
}

/// The dispatcher's input channels, consumed by `Dispatcher::run`.
pub struct DispatcherInbox {
    cmd_rx: mpsc::UnboundedReceiver<Command>,
    delta_rx: mpsc::UnboundedReceiver<(u64, Value)>,
    limit: watch::Receiver<usize>,
    overload: mpsc::UnboundedReceiver<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_matches_aiolimiter() {
        let now = Instant::now();
        let mut rl = RateLimiter::new(2.0);
        rl.last = now;
        assert!(rl.try_acquire(now).is_ok());
        assert!(rl.try_acquire(now).is_ok());
        assert_eq!(rl.try_acquire(now).unwrap_err() - now, Duration::from_secs(30));
        assert!(rl.try_acquire(now + Duration::from_secs(30)).is_ok());
    }

    #[test]
    fn py_repr_quotes_like_python() {
        assert_eq!(py_repr("a"), "'a'");
        assert_eq!(py_repr("it's"), "\"it's\"");
    }
}
