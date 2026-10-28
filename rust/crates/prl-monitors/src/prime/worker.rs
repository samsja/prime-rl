//! Background uploader (`prime_runs.worker`): one task draining a bounded queue of record
//! batches into a list of sinks. A producer that outruns it blocks briefly, then drops
//! (counted). Record-specific failures drop only their batch, transient failures get a few
//! consecutive strikes before the sink is retired (for good, or until `retire_cooldown`), and
//! any other failure retires the sink at once.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;

use super::client::ApiError;

pub const DEFAULT_QUEUE_SIZE: usize = 256;
pub const DEFAULT_PUT_TIMEOUT: Duration = Duration::from_secs(5);
/// Consecutive transient failures before a sink is retired.
pub const TRANSIENT_FAILURE_LIMIT: u32 = 3;

/// A sink stored only part of a batch: `cause` drives retry/retirement, `failed_records` the
/// loss accounting.
#[derive(Debug)]
pub struct SinkError {
    pub cause: ApiError,
    pub failed_records: usize,
}

#[async_trait]
pub trait Sink<R>: Send {
    fn name(&self) -> &'static str;
    fn enabled(&self) -> bool;
    fn set_enabled(&mut self, enabled: bool);
    async fn write(&mut self, records: &[R]) -> Result<(), SinkError>;
    /// Episodes the sink confirmed stored (the traces sink), for the eval summary.
    fn episodes_written(&self) -> Option<usize> {
        None
    }
}

/// Send a batch one unit at a time (`send_each`): a record rejection or transient failure costs
/// its own unit, a sink-wide failure stops the batch and costs the rest too.
pub fn send_each_result(
    results: impl IntoIterator<Item = Result<(), ApiError>>,
    total: usize,
) -> Result<(), SinkError> {
    let mut failed = 0;
    let mut reported: Option<ApiError> = None;
    for (position, result) in results.into_iter().enumerate() {
        let Err(e) = result else { continue };
        failed += 1;
        if !(e.is_record_rejection() || e.is_transient()) {
            failed += total - position - 1;
            reported = Some(e);
            break;
        }
        if reported.is_none() || e.is_transient() {
            reported = Some(e);
        }
    }
    match reported {
        Some(cause) => Err(SinkError {
            cause,
            failed_records: failed,
        }),
        None => Ok(()),
    }
}

enum Item<R> {
    Records(Vec<R>),
    Flush(oneshot::Sender<()>),
    Stop,
}

/// What a closed worker leaves behind.
#[derive(Debug, Default)]
pub struct WorkerReport {
    pub failed_records: HashMap<&'static str, usize>,
    pub errors: Vec<String>,
    pub episodes_written: Vec<usize>,
}

struct State<R> {
    sinks: Vec<Box<dyn Sink<R>>>,
    retire_cooldown: Option<Duration>,
    report: WorkerReport,
    transient: HashMap<&'static str, u32>,
    retired: HashSet<&'static str>,
    cooldowns: HashMap<&'static str, Instant>,
}

impl<R: Send + Sync + 'static> State<R> {
    async fn dispatch(&mut self, records: &[R]) {
        for i in 0..self.sinks.len() {
            self.maybe_revive(i);
            let name = self.sinks[i].name();
            if !self.sinks[i].enabled() {
                if self.retired.contains(name) {
                    *self.report.failed_records.entry(name).or_default() += records.len();
                }
                continue;
            }
            match self.sinks[i].write(records).await {
                Ok(()) => {
                    self.transient.remove(name);
                }
                Err(e) => self.fail_sink(i, e.cause, e.failed_records),
            }
        }
    }

    fn maybe_revive(&mut self, i: usize) {
        let name = self.sinks[i].name();
        match self.cooldowns.get(name) {
            Some(until) if Instant::now() >= *until => {}
            _ => return,
        }
        self.cooldowns.remove(name);
        self.retired.remove(name);
        self.transient.remove(name);
        self.sinks[i].set_enabled(true);
        tracing::info!("Sink {name} re-enabled after its cooldown");
    }

    fn fail_sink(&mut self, i: usize, e: ApiError, dropped: usize) {
        let name = self.sinks[i].name();
        if dropped > 0 {
            *self.report.failed_records.entry(name).or_default() += dropped;
        }
        let what = format!("{}: {e}", e.kind());
        self.report
            .errors
            .push(format!("writing to the {name} sink failed: {what}"));
        if dropped > 0 && e.is_record_rejection() {
            tracing::warn!("Sink {name} rejected a batch of {dropped} record(s) ({what}); still enabled for later records.");
            return;
        }
        if e.is_transient() {
            let strikes = self.transient.entry(name).or_default();
            *strikes += 1;
            let strikes = *strikes;
            if strikes < TRANSIENT_FAILURE_LIMIT {
                tracing::warn!(
                    "Sink {name} dropped a batch of {dropped} record(s) ({what}); strike {strikes} of {TRANSIENT_FAILURE_LIMIT}, still enabled."
                );
                return;
            }
            self.sinks[i].set_enabled(false);
            self.retired.insert(name);
            match self.retire_cooldown {
                None => tracing::warn!(
                    "Sink {name} disabled after {strikes} consecutive transient failures: {what}"
                ),
                Some(cooldown) => {
                    self.cooldowns.insert(name, Instant::now() + cooldown);
                    tracing::warn!(
                        "Sink {name} paused for {:.0}s after {strikes} consecutive transient failures: {what}",
                        cooldown.as_secs_f64()
                    );
                }
            }
        } else {
            self.sinks[i].set_enabled(false);
            self.retired.insert(name);
            tracing::warn!("Sink {name} disabled after an error: {what}");
        }
    }
}

async fn run<R: Send + Sync + 'static>(
    mut rx: mpsc::Receiver<Item<R>>,
    mut state: State<R>,
) -> WorkerReport {
    let mut pending: Option<Item<R>> = None;
    loop {
        let item = match pending.take() {
            Some(item) => item,
            None => match rx.recv().await {
                Some(item) => item,
                None => break,
            },
        };
        match item {
            Item::Stop => break,
            Item::Flush(done) => {
                let _ = done.send(());
            }
            Item::Records(mut batch) => {
                // Merge everything already queued behind it, stopping at a control item, so the
                // request rate tracks upload latency rather than rollout throughput.
                loop {
                    match rx.try_recv() {
                        Ok(Item::Records(more)) => batch.extend(more),
                        Ok(control) => {
                            pending = Some(control);
                            break;
                        }
                        Err(_) => break,
                    }
                }
                state.dispatch(&batch).await;
            }
        }
    }
    state.report.episodes_written = state
        .sinks
        .iter()
        .filter_map(|s| s.episodes_written())
        .collect();
    state.report
}

/// The handle to one uploader task.
pub struct UploadWorker<R> {
    tx: Option<mpsc::Sender<Item<R>>>,
    task: Option<JoinHandle<WorkerReport>>,
    put_timeout: Duration,
    pub dropped: usize,
}

impl<R: Send + Sync + 'static> UploadWorker<R> {
    pub fn spawn(sinks: Vec<Box<dyn Sink<R>>>, retire_cooldown: Option<Duration>) -> Self {
        Self::with_queue(
            sinks,
            retire_cooldown,
            DEFAULT_QUEUE_SIZE,
            DEFAULT_PUT_TIMEOUT,
        )
    }

    pub fn with_queue(
        sinks: Vec<Box<dyn Sink<R>>>,
        retire_cooldown: Option<Duration>,
        queue: usize,
        put_timeout: Duration,
    ) -> Self {
        if sinks.is_empty() {
            return UploadWorker {
                tx: None,
                task: None,
                put_timeout,
                dropped: 0,
            };
        }
        let (tx, rx) = mpsc::channel(queue);
        let state = State {
            sinks,
            retire_cooldown,
            report: WorkerReport::default(),
            transient: HashMap::new(),
            retired: HashSet::new(),
            cooldowns: HashMap::new(),
        };
        UploadWorker {
            tx: Some(tx),
            task: Some(tokio::spawn(run(rx, state))),
            put_timeout,
            dropped: 0,
        }
    }

    /// Queue a batch; `false` means it was dropped (queue full for the whole put timeout).
    pub async fn submit(&mut self, records: Vec<R>) -> bool {
        let Some(tx) = &self.tx else { return true };
        let count = records.len();
        match tx
            .send_timeout(Item::Records(records), self.put_timeout)
            .await
        {
            Ok(()) => true,
            Err(_) => {
                self.dropped += count;
                tracing::warn!(
                    "Upload queue full after {:.1}s; dropped {count} item(s) ({} total). The producer is outrunning the uploader.",
                    self.put_timeout.as_secs_f64(),
                    self.dropped
                );
                false
            }
        }
    }

    /// Wait until everything queued so far has been written.
    pub async fn flush(&self, deadline: Instant) -> bool {
        let Some(tx) = &self.tx else { return true };
        let (done_tx, done_rx) = oneshot::channel();
        if tokio::time::timeout_at(deadline, tx.send(Item::Flush(done_tx)))
            .await
            .is_err()
        {
            tracing::warn!("Could not enqueue a flush barrier before the drain timeout");
            return false;
        }
        matches!(tokio::time::timeout_at(deadline, done_rx).await, Ok(Ok(())))
    }

    /// Drain, stop the task and return its report (None when it outlived the deadline).
    pub async fn close(&mut self, deadline: Instant) -> Option<WorkerReport> {
        let tx = self.tx.take()?;
        let mut task = self.task.take()?;
        if tokio::time::timeout_at(deadline, tx.send(Item::Stop))
            .await
            .is_err()
        {
            tracing::warn!("Upload queue remained saturated through the close timeout");
        }
        drop(tx);
        match tokio::time::timeout_at(deadline, &mut task).await {
            Ok(Ok(report)) => Some(report),
            Ok(Err(e)) => {
                tracing::warn!("Uploader task failed: {e}");
                None
            }
            Err(_) => {
                tracing::warn!(
                    "Uploader still running after the close timeout; leaving it. Records still in flight may not finish before the process exits."
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct FakeSink {
        enabled: bool,
        writes: Arc<Mutex<Vec<Vec<u32>>>>,
        errors: Arc<Mutex<Vec<ApiError>>>,
    }

    #[async_trait]
    impl Sink<u32> for FakeSink {
        fn name(&self) -> &'static str {
            "fake"
        }
        fn enabled(&self) -> bool {
            self.enabled
        }
        fn set_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }
        async fn write(&mut self, records: &[u32]) -> Result<(), SinkError> {
            self.writes.lock().unwrap().push(records.to_vec());
            match self.errors.lock().unwrap().pop() {
                Some(cause) => Err(SinkError {
                    cause,
                    failed_records: records.len(),
                }),
                None => Ok(()),
            }
        }
    }

    fn transient() -> ApiError {
        ApiError::Retryable {
            status: 503,
            message: "busy".into(),
            code: None,
            retry_after: None,
        }
    }

    type Writes = Arc<Mutex<Vec<Vec<u32>>>>;

    fn worker(errors: Vec<ApiError>, cooldown: Option<Duration>) -> (UploadWorker<u32>, Writes) {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let sink = FakeSink {
            enabled: true,
            writes: writes.clone(),
            errors: Arc::new(Mutex::new(errors)),
        };
        (UploadWorker::spawn(vec![Box::new(sink)], cooldown), writes)
    }

    fn soon() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    #[tokio::test]
    async fn transient_strikes_retire_the_sink() {
        let (mut w, writes) = worker(vec![transient(), transient(), transient()], None);
        for i in 0..5 {
            w.submit(vec![i]).await;
            w.flush(soon()).await;
        }
        let report = w.close(soon()).await.unwrap();
        // three strikes, then retired: the last two batches never reach it and count as lost
        assert_eq!(writes.lock().unwrap().len(), 3);
        assert_eq!(report.failed_records["fake"], 5);
        assert_eq!(report.errors.len(), 3);
    }

    #[tokio::test]
    async fn record_rejections_keep_the_sink() {
        let rejection = ApiError::ValidationRejected {
            message: "bad".into(),
            code: None,
        };
        let (mut w, writes) = worker(vec![rejection], None);
        w.submit(vec![1]).await;
        w.flush(soon()).await;
        w.submit(vec![2]).await;
        let report = w.close(soon()).await.unwrap();
        assert_eq!(*writes.lock().unwrap(), [vec![1], vec![2]]);
        assert_eq!(report.failed_records["fake"], 1);
    }

    #[tokio::test(start_paused = true)]
    async fn retired_sink_revives_after_cooldown() {
        let (mut w, writes) = worker(
            vec![transient(), transient(), transient()],
            Some(Duration::from_secs(300)),
        );
        for i in 0..4 {
            w.submit(vec![i]).await;
            w.flush(soon()).await;
        }
        assert_eq!(writes.lock().unwrap().len(), 3);
        tokio::time::advance(Duration::from_secs(301)).await;
        w.submit(vec![9]).await;
        w.close(soon()).await.unwrap();
        assert_eq!(writes.lock().unwrap().last().unwrap(), &vec![9]);
    }

    #[tokio::test]
    async fn queued_batches_coalesce_and_a_full_queue_drops() {
        let (tx, rx) = mpsc::channel(4);
        let writes = Arc::new(Mutex::new(Vec::new()));
        let sink = FakeSink {
            enabled: true,
            writes: writes.clone(),
            errors: Arc::default(),
        };
        for i in 0..3 {
            tx.send(Item::Records(vec![i])).await.unwrap();
        }
        tx.send(Item::Stop).await.unwrap();
        let state = State {
            sinks: vec![Box::new(sink) as Box<dyn Sink<u32>>],
            retire_cooldown: None,
            report: WorkerReport::default(),
            transient: HashMap::new(),
            retired: HashSet::new(),
            cooldowns: HashMap::new(),
        };
        run(rx, state).await;
        assert_eq!(*writes.lock().unwrap(), [vec![0, 1, 2]]);

        // a worker whose queue stays full drops after the put timeout
        let (tx, _rx) = mpsc::channel(1);
        let mut w: UploadWorker<u32> = UploadWorker {
            tx: Some(tx),
            task: None,
            put_timeout: Duration::from_millis(10),
            dropped: 0,
        };
        assert!(w.submit(vec![1]).await);
        assert!(!w.submit(vec![2, 3]).await);
        assert_eq!(w.dropped, 2);
    }
}
