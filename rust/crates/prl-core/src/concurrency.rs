//! Adaptive cap on in-flight units of inference work (AIMD). Port of `concurrency.py`.
//!
//! A pure state machine: the metrics collector pushes `observe(samples)` every poll, the
//! dispatcher reports `record_episode(tokens, inflight)` per completed unit, and every
//! decision comes back as [`Action`]s (new limit, overload excess to cancel).

use std::time::{Duration, Instant};

use anyhow::Result;
use prl_types::Metrics;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::info;

/// Cap growth per pipeline turnover at zero GPU-KV utilization.
pub const TURNOVER_GROWTH_MAX: f64 = 1.2;
/// The cap counts as binding when inflight reaches this fraction of it.
pub const BINDING_FRACTION: f64 = 0.9;
/// Above this usage, soft-trim: lower the cap and let completions drain the pool.
pub const KV_USAGE_SOFT_CAP: f64 = 0.8;
/// Above this usage, hard-trim: also cancel the pool excess.
pub const KV_USAGE_HARD_CAP: f64 = 0.9;
/// A trim resizes to inflight * target / usage.
pub const KV_USAGE_TARGET: f64 = 0.7;
/// Polls between kv-headroom trims.
pub const KV_TRIM_COOLDOWN_POLLS: u32 = 6;
/// HARD once capacity-queued requests exceed this fraction of running requests for the persistence window.
pub const QUEUE_RATIO: f64 = 0.5;
/// Consecutive polls of queue overload before the HARD cut.
pub const QUEUE_PERSISTENCE_POLLS: u32 = 6;
/// A queue cut targets this fraction of the in-flight pool.
pub const QUEUE_CUT_FRACTION: f64 = 0.9;
/// A preemption cut targets this fraction of the in-flight pool.
pub const PREEMPTION_CUT_FRACTION: f64 = 0.8;
/// Cut fraction when overload survives a full drain.
pub const ESCALATED_CUT_FRACTION: f64 = 0.5;
/// Polls after a drain during which a repeat overload cuts at the escalated fraction.
pub const ESCALATION_GRACE_POLLS: i64 = 6;
/// The growth gate expires this long after the poll that set it.
pub const GROWTH_GATE_TTL: Duration = Duration::from_secs(15);

pub use prl_io::inference_metrics::EngineLoadSample;
use prl_config::InferenceRole;

fn is_prefill(sample: &EngineLoadSample) -> bool {
    sample.role == Some(InferenceRole::Prefill)
}

/// Engine pressure, in increasing severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Signal {
    Clear = 0,
    Soft = 1,
    Hard = 2,
}

impl Signal {
    pub fn as_str(self) -> &'static str {
        match self {
            Signal::Clear => "clear",
            Signal::Soft => "soft",
            Signal::Hard => "hard",
        }
    }
}

/// Outbound effect of a controller decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Move the dispatcher's in-flight cap.
    SetLimit(usize),
    /// Cancel roughly this many in-flight train episodes, youngest groups first.
    Overload(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConcurrencyParams {
    pub min_inflight: usize,
    pub max_inflight: Option<usize>,
    pub initial_inflight: Option<usize>,
}

#[derive(Debug)]
pub struct ConcurrencyController {
    params: ConcurrencyParams,
    fallback_cost: u64,
    cap: f64,
    pub max_inflight: usize,
    /// Python keeps the clamped float until the first applied limit (`128.0 -> 49` in logs).
    limit_applied: bool,
    bootstrapped: bool,
    /// A manual cap that overrides the band and the adaptive loop until cleared.
    pub target: Option<usize>,
    engine_max_len: Option<u64>,
    capacity_by_engine: indexmap::IndexMap<String, u64>,
    turnover: f64,
    signal: Signal,
    growth_multiplier: f64,
    can_grow: bool,
    can_grow_until: Option<Instant>,
    prev_waiting: indexmap::IndexMap<String, u64>,
    queue_overload_polls: u32,
    trim_cooldown: u32,
    escalation_grace: i64,
    draining: bool,
    escalated: bool,
}

impl ConcurrencyController {
    pub fn new(params: ConcurrencyParams, fallback_cost: u64) -> Self {
        let mut this = Self {
            params,
            fallback_cost,
            cap: params.initial_inflight.unwrap_or(params.min_inflight) as f64,
            max_inflight: 0,
            limit_applied: false,
            bootstrapped: params.initial_inflight.is_some(),
            target: None,
            engine_max_len: None,
            capacity_by_engine: Default::default(),
            turnover: 0.0,
            signal: Signal::Clear,
            growth_multiplier: 1.0,
            can_grow: false,
            can_grow_until: None,
            prev_waiting: Default::default(),
            queue_overload_polls: 0,
            trim_cooldown: 0,
            escalation_grace: 0,
            draining: false,
            escalated: false,
        };
        this.max_inflight = this.clamp(this.cap) as usize;
        this
    }

    /// One completed unit. `inflight` is the dispatcher's count after the unit's permit was
    /// released (the completing unit is added back for the binding check).
    pub fn record_episode(&mut self, tokens: u64, inflight: usize) -> Vec<Action> {
        let mut actions = Vec::new();
        let inflight = inflight + 1;
        let fraction = 1.0 / inflight as f64;
        self.turnover += fraction;
        if tokens == 0 {
            // Error markers consumed no engine capacity; growing on them floods engines later.
            return actions;
        }
        let gate_open = self.can_grow_until.is_some_and(|until| Instant::now() < until);
        if self.target.is_none()
            && self.can_grow
            && gate_open
            && inflight as f64 >= BINDING_FRACTION * self.max_inflight as f64
        {
            self.cap = self.clamp(self.cap * self.growth_multiplier.powf(fraction));
            self.apply_limit(self.cap as usize, None, &mut actions);
        }
        actions
    }

    /// Per-poll engine load push: classify the signal, gate growth, apply trims and cuts.
    pub fn observe(&mut self, samples: &[EngineLoadSample], inflight: usize) -> Vec<Action> {
        let mut actions = Vec::new();
        if samples.is_empty() {
            return actions;
        }
        for sample in samples {
            if let Some(capacity) = sample.kv_capacity_tokens.filter(|c| *c > 0) {
                if !is_prefill(sample) {
                    self.capacity_by_engine.insert(sample.engine_id.clone(), capacity);
                }
            }
            if let Some(len) = sample.max_model_len.filter(|l| *l > 0) {
                self.engine_max_len = Some(self.engine_max_len.unwrap_or(0).max(len));
            }
        }
        // Overload signals read decode engines only.
        let samples: Vec<&EngineLoadSample> =
            samples.iter().filter(|s| !is_prefill(s)).collect();
        if samples.is_empty() {
            return actions;
        }

        let mut worst = Signal::Clear;
        let mut max_usage: f64 = 0.0;
        let mut total_running: u64 = 0;
        let mut total_queued: u64 = 0;
        let mut preempted = false;
        for sample in &samples {
            if sample.preemptions_delta > 0 {
                preempted = true;
                worst = Signal::Hard;
            }
            if sample.waiting > 0 && self.prev_waiting.get(&sample.engine_id).copied().unwrap_or(0) > 0 {
                worst = worst.max(Signal::Soft);
            }
            max_usage = max_usage.max(sample.kv_usage);
            total_running += sample.running;
            total_queued += sample.waiting_capacity.unwrap_or(sample.waiting);
        }
        self.prev_waiting = samples.iter().map(|s| (s.engine_id.clone(), s.waiting)).collect();

        if total_running > 0 && total_queued as f64 > QUEUE_RATIO * total_running as f64 {
            self.queue_overload_polls += 1;
        } else {
            self.queue_overload_polls = 0;
        }
        let queue_overload = self.queue_overload_polls >= QUEUE_PERSISTENCE_POLLS;
        if queue_overload {
            worst = worst.max(Signal::Hard);
        } else if max_usage > KV_USAGE_SOFT_CAP {
            worst = worst.max(Signal::Soft);
        }
        self.signal = worst;

        // Release the drain only once the engines have settled too.
        if self.draining && inflight <= self.max_inflight && !preempted && !queue_overload {
            self.draining = false;
            self.escalation_grace = ESCALATION_GRACE_POLLS;
        }
        if !self.draining && self.escalated {
            self.escalation_grace -= 1;
            if self.escalation_grace <= 0 {
                self.escalated = false;
            }
        }
        self.trim_cooldown = self.trim_cooldown.saturating_sub(1);
        self.can_grow = worst == Signal::Clear && total_queued == 0 && !self.draining;
        self.growth_multiplier =
            1.0 + (TURNOVER_GROWTH_MAX - 1.0) * (1.0 - max_usage / KV_USAGE_SOFT_CAP).max(0.0);
        self.can_grow_until = Some(Instant::now() + GROWTH_GATE_TTL);
        if self.target.is_some() {
            return actions;
        }

        if !self.bootstrapped {
            if let Some(capacity) = self.capacity() {
                self.bootstrapped = true;
                let cost = self.engine_max_len.unwrap_or(self.fallback_cost) as f64;
                self.cap = self.clamp(capacity as f64 / cost);
                info!(
                    "Derived initial max inflight {} - {} KV cache tokens / {} tokens per episode",
                    self.cap as usize,
                    format_num(capacity as f64, 1, false),
                    format_num(cost, 1, true)
                );
                self.apply_limit(self.cap as usize, None, &mut actions);
            }
        }

        if self.draining {
            return actions;
        }

        if preempted || queue_overload {
            let fraction = self.escalated.then_some(ESCALATED_CUT_FRACTION);
            let (target, reason) = if queue_overload {
                self.queue_overload_polls = 0;
                (self.clamp(inflight as f64 * fraction.unwrap_or(QUEUE_CUT_FRACTION)), "queue overload".to_string())
            } else {
                (self.clamp(inflight as f64 * fraction.unwrap_or(PREEMPTION_CUT_FRACTION)), "preemptions".to_string())
            };
            self.resize_down(target as usize, inflight, &reason, true, &mut actions);
            self.draining = true;
            self.escalated = true;
            return actions;
        }

        if max_usage > KV_USAGE_SOFT_CAP && inflight > 0 && self.trim_cooldown == 0 {
            let hard = max_usage > KV_USAGE_HARD_CAP;
            let target = self.clamp(inflight as f64 * KV_USAGE_TARGET / max_usage) as usize;
            let reason =
                format!("kv headroom (usage {:.2}, {} trim)", max_usage, if hard { "hard" } else { "soft" });
            self.resize_down(target, inflight, &reason, hard, &mut actions);
            self.trim_cooldown = KV_TRIM_COOLDOWN_POLLS;
        }
        actions
    }

    /// Pin the cap at `target` (outside the band too), or hand it back to the adaptive loop.
    pub fn set_target(&mut self, target: Option<usize>) -> Vec<Action> {
        let mut actions = Vec::new();
        self.target = target;
        self.cap = match target {
            Some(t) => t as f64,
            None => self.clamp(self.cap),
        };
        let reason = if target.is_some() { "manual target" } else { "adaptive" };
        self.apply_limit(self.cap as usize, Some(reason), &mut actions);
        actions
    }

    fn resize_down(&mut self, target: usize, inflight: usize, reason: &str, cancel: bool, actions: &mut Vec<Action>) {
        let target = target.min(self.max_inflight);
        self.cap = target as f64;
        self.apply_limit(target, Some(reason), actions);
        if cancel && inflight > target {
            actions.push(Action::Overload(inflight - target));
        }
    }

    fn apply_limit(&mut self, n_max: usize, reason: Option<&str>, actions: &mut Vec<Action>) {
        if n_max == self.max_inflight {
            return;
        }
        if let Some(reason) = reason {
            let verb = if n_max > self.max_inflight { "Increased" } else { "Decreased" };
            let previous =
                if self.limit_applied { self.max_inflight.to_string() } else { format!("{}.0", self.max_inflight) };
            info!(
                "{verb} concurrency {previous} -> {n_max} ({reason}) - turnover={:.1} signal={}",
                self.turnover,
                self.signal.as_str()
            );
        }
        self.max_inflight = n_max;
        self.limit_applied = true;
        actions.push(Action::SetLimit(n_max));
    }

    fn clamp(&self, n_max: f64) -> f64 {
        let ceiling = self.params.max_inflight.map_or(f64::INFINITY, |m| m as f64);
        n_max.max(self.params.min_inflight as f64).min(ceiling)
    }

    /// Total KV tokens across decode engines; None until the first poll.
    pub fn capacity(&self) -> Option<u64> {
        let total: u64 = self.capacity_by_engine.values().sum();
        (total > 0).then_some(total)
    }

    pub fn gauges(&self) -> Metrics {
        let mut m = Metrics::new();
        m.insert("concurrency/max_inflight".into(), (self.max_inflight as f64).into());
        m.insert("concurrency/turnover".into(), self.turnover.into());
        m.insert("concurrency/capacity".into(), (self.capacity().unwrap_or(0) as f64).into());
        m.insert("concurrency/signal".into(), (self.signal as u8 as f64).into());
        m.insert("concurrency/growth_multiplier".into(), self.growth_multiplier.into());
        m
    }
}

/// `prime_rl.utils.utils.format_num`; `is_float` mirrors Python's `isinstance(num, float)`.
pub fn format_num(num: f64, precision: usize, is_float: bool) -> String {
    let sign = if num < 0.0 { "-" } else { "" };
    let n = num.abs();
    if n < 1e3 {
        return if is_float { format!("{sign}{n:.precision$}") } else { format!("{sign}{}", n as i64) };
    }
    let (value, suffix) = if n < 1e6 {
        (n / 1e3, "K")
    } else if n < 1e9 {
        (n / 1e6, "M")
    } else {
        (n / 1e9, "B")
    };
    format!("{sign}{value:.precision$}{suffix}")
}

// ── actor ──────────────────────────────────────────────────────────────────

/// Point-in-time view for the eval control API (`GET/PUT /v1/concurrency`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ConcurrencyState {
    pub target: Option<usize>,
    pub cap: usize,
    pub inflight: usize,
}

enum Command {
    Episode { tokens: u64, inflight: usize },
    SetTarget { target: Option<usize>, reply: oneshot::Sender<ConcurrencyState> },
    Get { reply: oneshot::Sender<ConcurrencyState> },
    Gauges { reply: oneshot::Sender<Metrics> },
}

/// Handle onto the concurrency actor. Cloned by the dispatcher (completions), the pipeline
/// view (gauges) and the eval control API.
#[derive(Clone)]
pub struct ConcurrencyHandle {
    tx: mpsc::UnboundedSender<Command>,
}

impl ConcurrencyHandle {
    /// One completed unit; `inflight` is the dispatcher's count after releasing its permit.
    pub fn record_episode(&self, tokens: u64, inflight: usize) {
        let _ = self.tx.send(Command::Episode { tokens, inflight });
    }

    pub async fn set_target(&self, target: Option<usize>) -> Result<ConcurrencyState> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Command::SetTarget { target, reply }).map_err(|_| anyhow::anyhow!("concurrency actor stopped"))?;
        Ok(rx.await?)
    }

    pub async fn get(&self) -> Result<ConcurrencyState> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Command::Get { reply }).map_err(|_| anyhow::anyhow!("concurrency actor stopped"))?;
        Ok(rx.await?)
    }

    pub async fn gauges(&self) -> Result<Metrics> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Command::Gauges { reply }).map_err(|_| anyhow::anyhow!("concurrency actor stopped"))?;
        Ok(rx.await?)
    }
}

/// Wiring of the concurrency actor.
pub struct ConcurrencyIo {
    /// Engine load polls from the inference metrics collector.
    pub samples: mpsc::Receiver<Vec<EngineLoadSample>>,
    /// The dispatcher's in-flight count.
    pub inflight: watch::Receiver<usize>,
    /// The dispatcher's in-flight cap (single writer: this actor).
    pub limit: watch::Sender<usize>,
    /// Overload excess to cancel; None never cancels (eval: a cut only blocks admission).
    pub overload: Option<mpsc::UnboundedSender<usize>>,
}

impl ConcurrencyController {
    /// Run the controller as an actor until `cancel` fires or every input closes.
    pub fn spawn(mut self, io: ConcurrencyIo, cancel: CancellationToken) -> (ConcurrencyHandle, JoinHandle<Result<()>>) {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let ConcurrencyIo { mut samples, inflight, limit, overload } = io;
        limit.send_replace(self.max_inflight);
        let task = tokio::spawn(async move {
            let apply = |actions: Vec<Action>| {
                for action in actions {
                    match action {
                        Action::SetLimit(n) => {
                            limit.send_replace(n);
                        }
                        Action::Overload(n) => {
                            if let Some(overload) = &overload {
                                let _ = overload.send(n);
                            }
                        }
                    }
                }
            };
            let mut samples_open = true;
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => return Ok(()),
                    batch = samples.recv(), if samples_open => match batch {
                        Some(batch) => {
                            let n = *inflight.borrow();
                            apply(self.observe(&batch, n));
                        }
                        None => samples_open = false,
                    },
                    cmd = rx.recv() => {
                        let Some(cmd) = cmd else { return Ok(()) };
                        match cmd {
                            Command::Episode { tokens, inflight } => apply(self.record_episode(tokens, inflight)),
                            Command::SetTarget { target, reply } => {
                                apply(self.set_target(target));
                                let _ = reply.send(self.state(*inflight.borrow()));
                            }
                            Command::Get { reply } => {
                                let _ = reply.send(self.state(*inflight.borrow()));
                            }
                            Command::Gauges { reply } => {
                                let _ = reply.send(self.gauges());
                            }
                        }
                    }
                }
            }
        });
        (ConcurrencyHandle { tx }, task)
    }

    fn state(&self, inflight: usize) -> ConcurrencyState {
        ConcurrencyState { target: self.target, cap: self.max_inflight, inflight }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(kv_usage: f64, running: u64, waiting: u64, preemptions_delta: u64) -> EngineLoadSample {
        EngineLoadSample {
            engine_id: "engine-0".into(),
            role: None,
            kv_capacity_tokens: Some(1_000_000),
            max_model_len: Some(8192),
            kv_usage,
            running,
            waiting,
            waiting_capacity: Some(waiting),
            preemptions_delta,
        }
    }

    fn controller() -> ConcurrencyController {
        ConcurrencyController::new(
            ConcurrencyParams { min_inflight: 1, max_inflight: Some(2000), initial_inflight: Some(2000) },
            8192,
        )
    }

    #[test]
    fn preemption_cut_and_manual_target() {
        let mut c = controller();
        assert_eq!(c.observe(&[sample(0.95, 1800, 0, 4)], 1800), vec![Action::SetLimit(1440), Action::Overload(360)]);
        let mut c = controller();
        assert_eq!(c.set_target(Some(1500)), vec![Action::SetLimit(1500)]);
        c.observe(&[sample(0.95, 1800, 0, 4)], 1800);
        assert_eq!(c.max_inflight, 1500);
    }

    #[test]
    fn queue_overload_needs_persistence_then_escalates() {
        let mut c = controller();
        for _ in 0..5 {
            assert!(c.observe(&[sample(0.5, 100, 80, 0)], 1000).iter().all(|a| !matches!(a, Action::Overload(_))));
        }
        assert_eq!(c.observe(&[sample(0.5, 100, 80, 0)], 1000), vec![Action::SetLimit(900), Action::Overload(100)]);
        // Overload while draining is stale; after the drain a repeat cuts at the escalated fraction.
        assert!(c.observe(&[sample(0.95, 1800, 0, 4)], 1000).is_empty());
        c.observe(&[sample(0.5, 1800, 0, 0)], 800);
        assert_eq!(c.observe(&[sample(0.95, 1800, 0, 4)], 800), vec![Action::SetLimit(400), Action::Overload(400)]);
    }

    #[test]
    fn kv_headroom_trims() {
        let mut c = controller();
        assert_eq!(c.observe(&[sample(0.85, 1800, 0, 0)], 1000), vec![Action::SetLimit(823)]);
        let mut c = controller();
        assert_eq!(c.observe(&[sample(0.95, 1800, 0, 0)], 1000), vec![Action::SetLimit(736), Action::Overload(264)]);
    }

    #[test]
    fn format_num_matches_python() {
        assert_eq!(format_num(1_000_000.0, 1, false), "1.0M");
        assert_eq!(format_num(8192.0, 1, true), "8.2K");
        assert_eq!(format_num(512.0, 1, false), "512");
    }
}
