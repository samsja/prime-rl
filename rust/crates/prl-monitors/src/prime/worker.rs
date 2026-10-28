//! Background uploader (`prime_runs.worker`): one task draining a bounded queue of record
//! batches into a sink. A producer that outruns it blocks briefly, then drops (counted).
//! Record-specific failures drop only their batch, transient failures get a few consecutive
//! strikes before the sink is retired (for good, or until `retire_cooldown` passes), and any
//! other failure retires it at once.

use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;

use super::client::ApiError;

const QUEUE_SIZE: usize = 256;
const PUT_TIMEOUT: Duration = Duration::from_secs(5);
/// Consecutive transient failures before a sink is retired.
const TRANSIENT_FAILURE_LIMIT: u32 = 3;

/// A sink stored only part of a batch: `cause` drives retirement, `failed_records` the loss
/// accounting.
#[derive(Debug)]
pub struct SinkError {
    pub cause: ApiError,
    pub failed_records: usize,
}

#[async_trait]
pub trait Sink<R>: Send {
    fn name(&self) -> &'static str;
    /// False once the sink switched itself off (it then skips every batch quietly).
    fn enabled(&self) -> bool;
    fn set_enabled(&mut self, enabled: bool);
    async fn write(&mut self, records: &[R]) -> Result<(), SinkError>;
    /// Episodes the sink confirmed stored (the traces sink), for the eval summary.
    fn episodes_written(&self) -> Option<usize> {
        None
    }
}

enum Item<R> {
    Records(Vec<R>),
    Flush(oneshot::Sender<()>),
}

/// What a closed worker leaves behind.
#[derive(Debug, Default)]
pub struct WorkerReport {
    /// Records the sink could not store, by sink name.
    pub failed_records: Option<(&'static str, usize)>,
    pub errors: Vec<String>,
    pub episodes_written: Option<usize>,
}

struct State<R> {
    sink: Box<dyn Sink<R>>,
    retire_cooldown: Option<Duration>,
    failed: usize,
    errors: Vec<String>,
    strikes: u32,
    /// Retired by the worker (records skipping it count as lost), until `revive_at`.
    retired: bool,
    revive_at: Option<Instant>,
}

impl<R: Send + Sync + 'static> State<R> {
    async fn dispatch(&mut self, records: &[R]) {
        let name = self.sink.name();
        if self.revive_at.is_some_and(|at| Instant::now() >= at) {
            (self.revive_at, self.retired, self.strikes) = (None, false, 0);
            self.sink.set_enabled(true);
            tracing::info!("Sink {name} re-enabled after its cooldown");
        }
        if !self.sink.enabled() {
            if self.retired {
                self.failed += records.len();
            }
            return;
        }
        let Err(SinkError {
            cause,
            failed_records,
        }) = self.sink.write(records).await
        else {
            self.strikes = 0;
            return;
        };
        self.failed += failed_records;
        let what = cause.describe();
        self.errors
            .push(format!("writing to the {name} sink failed: {what}"));
        if failed_records > 0 && cause.is_record_rejection() {
            tracing::warn!("Sink {name} rejected a batch of {failed_records} record(s) ({what}); still enabled for later records.");
            return;
        }
        if cause.is_transient() {
            self.strikes += 1;
            let strikes = self.strikes;
            if strikes < TRANSIENT_FAILURE_LIMIT {
                tracing::warn!("Sink {name} dropped a batch of {failed_records} record(s) ({what}); strike {strikes} of {TRANSIENT_FAILURE_LIMIT}, still enabled.");
                return;
            }
            match self.retire_cooldown {
                None => tracing::warn!(
                    "Sink {name} disabled after {strikes} consecutive transient failures: {what}"
                ),
                Some(cooldown) => {
                    self.revive_at = Some(Instant::now() + cooldown);
                    tracing::warn!("Sink {name} paused for {}s after {strikes} consecutive transient failures: {what}", cooldown.as_secs());
                }
            }
        } else {
            tracing::warn!("Sink {name} disabled after an error: {what}");
        }
        self.sink.set_enabled(false);
        self.retired = true;
    }
}

async fn run<R: Send + Sync + 'static>(
    mut rx: mpsc::Receiver<Item<R>>,
    mut state: State<R>,
) -> WorkerReport {
    let mut pending = None;
    loop {
        let Some(item) = (match pending.take() {
            Some(item) => Some(item),
            None => rx.recv().await,
        }) else {
            break;
        };
        match item {
            Item::Flush(done) => {
                let _ = done.send(());
            }
            Item::Records(mut batch) => {
                // Merge everything already queued behind it, stopping at a flush, so the
                // request rate tracks upload latency rather than rollout throughput.
                while let Ok(item) = rx.try_recv() {
                    match item {
                        Item::Records(more) => batch.extend(more),
                        flush => {
                            pending = Some(flush);
                            break;
                        }
                    }
                }
                state.dispatch(&batch).await;
            }
        }
    }
    WorkerReport {
        failed_records: (state.failed > 0).then(|| (state.sink.name(), state.failed)),
        errors: state.errors,
        episodes_written: state.sink.episodes_written(),
    }
}

/// The handle to one uploader task (none for a disabled run).
pub struct UploadWorker<R> {
    worker: Option<(mpsc::Sender<Item<R>>, JoinHandle<WorkerReport>)>,
    pub dropped: usize,
}

impl<R: Send + Sync + 'static> UploadWorker<R> {
    pub fn disabled() -> Self {
        UploadWorker {
            worker: None,
            dropped: 0,
        }
    }

    pub fn spawn(sink: Box<dyn Sink<R>>, retire_cooldown: Option<Duration>) -> Self {
        let (tx, rx) = mpsc::channel(QUEUE_SIZE);
        let state = State {
            sink,
            retire_cooldown,
            failed: 0,
            errors: Vec::new(),
            strikes: 0,
            retired: false,
            revive_at: None,
        };
        UploadWorker {
            worker: Some((tx, tokio::spawn(run(rx, state)))),
            dropped: 0,
        }
    }

    /// Queue a batch; dropped (and counted) when the queue stays full for the put timeout.
    pub async fn submit(&mut self, records: Vec<R>) {
        let Some((tx, _)) = &self.worker else { return };
        let count = records.len();
        if tx
            .send_timeout(Item::Records(records), PUT_TIMEOUT)
            .await
            .is_err()
        {
            self.dropped += count;
            tracing::warn!(
                "Upload queue full after {}s; dropped {count} item(s) ({} total). The producer is outrunning the uploader.",
                PUT_TIMEOUT.as_secs(),
                self.dropped
            );
        }
    }

    /// Wait until everything queued so far has been written; false past the deadline.
    pub async fn flush(&self, deadline: Instant) -> bool {
        let Some((tx, _)) = &self.worker else {
            return true;
        };
        let (done, wait) = oneshot::channel();
        tokio::time::timeout_at(deadline, async {
            tx.send(Item::Flush(done)).await.is_ok() && wait.await.is_ok()
        })
        .await
        .unwrap_or(false)
    }

    /// Drain, stop the task and return its report (None when it outlived the deadline).
    pub async fn close(&mut self, deadline: Instant) -> Option<WorkerReport> {
        let (tx, mut task) = self.worker.take()?;
        drop(tx);
        match tokio::time::timeout_at(deadline, &mut task).await {
            Ok(report) => Some(report.expect("uploader task panicked")),
            Err(_) => {
                tracing::warn!("Uploader still running after the close timeout; records still in flight may not finish before the process exits.");
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
        errors: Vec<ApiError>,
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
            self.errors.pop().map_or(Ok(()), |cause| {
                Err(SinkError {
                    cause,
                    failed_records: records.len(),
                })
            })
        }
    }

    #[tokio::test]
    async fn transient_strikes_retire_the_sink_until_its_cooldown() {
        let busy = ApiError::Http {
            status: 503,
            code: None,
            message: "busy".into(),
            retry_after: None,
        };
        let writes = Arc::new(Mutex::new(Vec::new()));
        let sink = FakeSink {
            enabled: true,
            writes: writes.clone(),
            errors: vec![busy.clone(), busy.clone(), busy],
        };
        let mut worker = UploadWorker::spawn(Box::new(sink), Some(Duration::from_millis(200)));
        let soon = || Instant::now() + Duration::from_secs(5);
        for i in 0..5 {
            worker.submit(vec![i]).await;
            worker.flush(soon()).await;
        }
        // three strikes, then retired: the next two batches never reach it and count as lost
        assert_eq!(writes.lock().unwrap().len(), 3);
        tokio::time::sleep(Duration::from_millis(250)).await;
        worker.submit(vec![9]).await;
        let report = worker.close(soon()).await.unwrap();
        assert_eq!(writes.lock().unwrap().last().unwrap(), &vec![9]);
        assert_eq!(report.failed_records, Some(("fake", 5)));
        assert_eq!(report.errors.len(), 3);
    }
}
