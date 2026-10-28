//! `EnvClient`: the serve protocol's client side (`verifiers/v1/serve/client.py`).
//!
//! One actor task per client owns a libzmq DEALER socket and drives it from tokio through
//! the socket's notification fd (the same integration pyzmq's asyncio sockets use). The
//! actor never decodes: it routes each `[request_id, kind, data]` frame to the request's
//! own ordered channel, so a run's deltas and then its reply are applied by the run's
//! task, in arrival order. Dropping a run before its reply sends `cancel` to the server.

use std::collections::HashMap;
use std::os::fd::{AsRawFd, RawFd};
use std::time::Duration;

use serde::Serialize;
use tokio::io::unix::AsyncFd;
use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use crate::assembly::{AssemblyError, EpisodeAssembly};
use crate::model::{Episode, ValidationError};
use crate::protocol::{Response, KIND_DELTA, METHOD_CANCEL, METHOD_HEALTH, METHOD_RUN};
use crate::value::{self, Map, Value};

const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);

/// Receives each decoded delta of a run, after it was applied to the run's assembly.
pub type OnDelta = Box<dyn FnMut(&Value) + Send>;

#[derive(Debug, thiserror::Error)]
pub enum EnvClientError {
    /// A reply with `success=False` (`RuntimeError(error or "env server request failed")`).
    #[error("{0}")]
    Server(String),
    #[error(transparent)]
    Assembly(#[from] AssemblyError),
    #[error(transparent)]
    Validation(#[from] ValidationError),
    #[error("invalid env-server reply: {0}")]
    Reply(String),
    #[error("env client for {0} is closed")]
    Closed(String),
    #[error("{0}")]
    Timeout(String),
    #[error("zmq: {0}")]
    Zmq(#[from] zmq::Error),
}

impl EnvClientError {
    /// The Python exception class the same failure raises in `verifiers.v1.serve.EnvClient`.
    pub fn type_name(&self) -> &'static str {
        match self {
            EnvClientError::Server(_) | EnvClientError::Assembly(_) | EnvClientError::Closed(_) => "RuntimeError",
            EnvClientError::Validation(_) | EnvClientError::Reply(_) => "ValidationError",
            EnvClientError::Timeout(_) => "TimeoutError",
            EnvClientError::Zmq(_) => "ZMQError",
        }
    }
}

enum Frame {
    Delta(Vec<u8>),
    Reply(Vec<u8>),
}

enum Command {
    Request { request_id: String, method: &'static str, payload: Vec<u8>, frames: mpsc::UnboundedSender<Frame>, deltas: bool },
    /// Stop routing frames to an abandoned request.
    Forget { request_id: String },
    /// A fire-and-forget request whose reply nobody reads (`cancel`).
    Send { method: &'static str, payload: Vec<u8> },
    Close { done: oneshot::Sender<()> },
}

/// A cheap, cloneable handle onto one env server's socket actor.
#[derive(Clone)]
pub struct EnvClient {
    address: String,
    inbox: mpsc::UnboundedSender<Command>,
}

use crate::model::new_id as new_request_id;

fn pack(value: &Value) -> Vec<u8> {
    value::to_msgpack(value)
}

impl EnvClient {
    /// Create the socket and connect it (libzmq connects and reconnects in the background,
    /// queueing requests meanwhile). Must be called inside a tokio runtime.
    pub fn connect(address: &str) -> Result<EnvClient, EnvClientError> {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::DEALER)?;
        socket.set_linger(0)?;
        socket.set_sndhwm(0)?;
        socket.set_rcvhwm(0)?;
        socket.connect(address)?;
        // SAFETY: the fd belongs to `socket`, which the actor keeps open for as long as
        // it holds the `AsyncFd` (and drops the `AsyncFd` first).
        let fd = unsafe { AsyncFd::register_with_interest(ZmqFd(socket.get_fd()?), tokio::io::Interest::READABLE) }
            .map_err(|e| EnvClientError::Zmq(zmq::Error::from_raw(e.into_parts().1.raw_os_error().unwrap_or(0))))?;
        let (inbox, commands) = mpsc::unbounded_channel();
        tokio::spawn(Actor { fd, socket, _ctx: ctx, commands, pending: HashMap::new() }.run());
        Ok(EnvClient { address: address.to_owned(), inbox })
    }

    async fn request(
        &self,
        method: &'static str,
        payload: Value,
        request_id: String,
        mut on_delta: Option<&mut (dyn FnMut(&Value) + Send)>,
    ) -> Result<Response, EnvClientError> {
        let (frames, mut rx) = mpsc::unbounded_channel();
        let mut guard = PendingGuard { inbox: &self.inbox, request_id: request_id.clone(), cancel: method == METHOD_RUN };
        self.inbox
            .send(Command::Request { request_id, method, payload: pack(&payload), frames, deltas: on_delta.is_some() })
            .map_err(|_| EnvClientError::Closed(self.address.clone()))?;
        let data = loop {
            match rx.recv().await {
                Some(Frame::Delta(data)) => match value::from_msgpack(&data) {
                    Ok(delta) => {
                        if let Some(on_delta) = on_delta.as_mut() {
                            on_delta(&delta);
                        }
                    }
                    Err(e) => warn!("dropping a malformed env-server frame: {e}"),
                },
                Some(Frame::Reply(data)) => break data,
                None => return Err(EnvClientError::Closed(self.address.clone())),
            }
        };
        guard.cancel = false;
        drop(guard);
        let reply = value::from_msgpack(&data).map_err(|e| EnvClientError::Reply(e.to_string()))?;
        let response = Response::from_value(&reply).map_err(EnvClientError::Reply)?;
        if !response.success {
            return Err(EnvClientError::Server(response.error.unwrap_or_else(|| "env server request failed".into())));
        }
        Ok(response)
    }

    /// Whether the server answers `health` within `timeout`.
    pub async fn health(&self, timeout: Duration) -> Result<bool, EnvClientError> {
        match tokio::time::timeout(timeout, self.request(METHOD_HEALTH, Value::map(), new_request_id(), None)).await {
            Ok(response) => response.map(|r| r.success),
            Err(_) => Ok(false),
        }
    }

    /// Poll `health` until the server answers or `timeout` elapses (interval 1s).
    pub async fn wait_for_server_startup(&self, timeout: Duration) -> Result<(), EnvClientError> {
        let interval = Duration::from_secs(1);
        let deadline = tokio::time::Instant::now() + timeout;
        while tokio::time::Instant::now() < deadline {
            if self.health(interval.min(HEALTH_TIMEOUT)).await? {
                return Ok(());
            }
            tokio::time::sleep(interval).await;
        }
        Err(EnvClientError::Timeout(format!(
            "env server at {} did not become healthy in {}s",
            self.address,
            python_seconds(timeout)
        )))
    }

    /// Run one rollout and return its episode, assembled from the streamed deltas.
    /// `on_delta` sees each delta after it was applied. Dropping the future before it
    /// resolves sends `cancel` for the run.
    pub async fn run<C: Serialize + ?Sized, S: Serialize + ?Sized>(
        &self,
        task_data: &Map,
        client: &C,
        model: &str,
        sampling: &S,
        mut on_delta: Option<OnDelta>,
    ) -> Result<Episode, EnvClientError> {
        let encode = |e: rmp_serde::encode::Error| EnvClientError::Reply(e.to_string());
        let payload: Value = [
            ("task_data", Value::Map(task_data.clone())),
            ("client", value::to_value(client).map_err(encode)?),
            ("model", Value::from(model)),
            ("sampling", value::to_value(sampling).map_err(encode)?),
        ]
        .into_iter()
        .collect();
        let request_id = new_request_id();
        let mut assembly = EpisodeAssembly::new();
        let mut apply = |delta: &Value| match assembly.apply(delta) {
            Ok(()) => {
                if let Some(on_delta) = on_delta.as_mut() {
                    on_delta(delta);
                }
            }
            Err(e) => warn!("dropping a malformed env-server frame: {e}"),
        };
        let response = self.request(METHOD_RUN, payload, request_id.clone(), Some(&mut apply)).await?;
        let Some(head) = response.head else {
            return Err(EnvClientError::Reply("run reply without an episode head".into()));
        };
        let episode = assembly
            .finish(&head, &response.traces)
            .map_err(EnvClientError::from)
            .and_then(|record| Episode::from_wire(&record).map_err(EnvClientError::from));
        if episode.is_err() {
            // The server counts the episode delivered: tell it this caller never got it.
            self.send_cancel(&request_id);
        }
        episode
    }

    fn send_cancel(&self, run_request_id: &str) {
        let payload: Value = [("request_id", Value::from(run_request_id))].into_iter().collect();
        // A closed actor has nothing left to cancel on.
        let _ = self.inbox.send(Command::Send { method: METHOD_CANCEL, payload: pack(&payload) });
    }

    /// Send every queued command (cancels included), then close the socket.
    pub async fn close(&self) {
        let (done, wait) = oneshot::channel();
        if self.inbox.send(Command::Close { done }).is_ok() {
            let _ = wait.await;
        }
    }
}

/// Seconds as Python formats a float (`600.0`) or an int.
fn python_seconds(d: Duration) -> String {
    let mut out = Vec::new();
    crate::json::write_f64(&mut out, d.as_secs_f64());
    String::from_utf8(out).unwrap()
}

/// Forgets a request whose caller went away; a dropped run also cancels server-side.
struct PendingGuard<'a> {
    inbox: &'a mpsc::UnboundedSender<Command>,
    request_id: String,
    cancel: bool,
}

impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        let _ = self.inbox.send(Command::Forget { request_id: self.request_id.clone() });
        if self.cancel {
            let payload: Value = [("request_id", Value::from(self.request_id.as_str()))].into_iter().collect();
            let _ = self.inbox.send(Command::Send { method: METHOD_CANCEL, payload: pack(&payload) });
        }
    }
}

struct ZmqFd(RawFd);

impl AsRawFd for ZmqFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

struct Pending {
    frames: mpsc::UnboundedSender<Frame>,
    deltas: bool,
}

struct Actor {
    // drop order: the fd registration, then the socket, then the context
    fd: AsyncFd<ZmqFd>,
    socket: zmq::Socket,
    _ctx: zmq::Context,
    commands: mpsc::UnboundedReceiver<Command>,
    pending: HashMap<String, Pending>,
}

impl Actor {
    async fn run(mut self) {
        loop {
            // The notification fd is edge-triggered: drain everything readable after every
            // wakeup and every send, before waiting again.
            if let Err(e) = self.drain() {
                warn!("env client receive failed: {e}");
            }
            tokio::select! {
                biased;
                command = self.commands.recv() => match command {
                    Some(Command::Close { done }) => {
                        let _ = done.send(());
                        break;
                    }
                    Some(command) => self.handle(command),
                    None => break,
                },
                ready = self.fd.readable() => match ready {
                    Ok(mut guard) => guard.clear_ready(),
                    Err(e) => {
                        warn!("env client socket fd failed: {e}");
                        break;
                    }
                },
            }
        }
    }

    fn handle(&mut self, command: Command) {
        let sent = match command {
            Command::Request { request_id, method, payload, frames, deltas } => {
                let frames_out = [request_id.as_bytes(), method.as_bytes(), payload.as_slice()];
                self.pending.insert(request_id.clone(), Pending { frames, deltas });
                self.socket.send_multipart(frames_out, zmq::DONTWAIT)
            }
            Command::Forget { request_id } => {
                self.pending.remove(&request_id);
                Ok(())
            }
            Command::Send { method, payload } => {
                let request_id = new_request_id();
                self.socket.send_multipart([request_id.as_bytes(), method.as_bytes(), payload.as_slice()], zmq::DONTWAIT)
            }
            Command::Close { .. } => unreachable!("handled by run"),
        };
        // SNDHWM=0 never refuses a send; a failure means the socket itself is broken.
        if let Err(e) = sent {
            warn!("env client send failed: {e}");
        }
    }

    fn drain(&mut self) -> Result<(), zmq::Error> {
        loop {
            let frames = match self.socket.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => frames,
                Err(zmq::Error::EAGAIN) => return Ok(()),
                Err(e) => return Err(e),
            };
            self.route(frames);
        }
    }

    fn route(&mut self, frames: Vec<Vec<u8>>) {
        let Ok([request_id, kind, data]) = <[Vec<u8>; 3]>::try_from(frames) else {
            warn!(
                "dropping a malformed env-server frame: expected [request_id, kind, data] - is the env server speaking the same serve protocol?"
            );
            return;
        };
        let Ok(request_id) = String::from_utf8(request_id) else {
            warn!("dropping a malformed env-server frame: request id is not UTF-8");
            return;
        };
        if kind == KIND_DELTA {
            if let Some(pending) = self.pending.get(&request_id).filter(|p| p.deltas) {
                let _ = pending.frames.send(Frame::Delta(data));
            }
        } else if let Some(pending) = self.pending.remove(&request_id) {
            let _ = pending.frames.send(Frame::Reply(data));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    /// A bare ROUTER speaking the serve protocol on a thread; tests script its frames.
    struct FakeServer {
        router: zmq::Socket,
        _ctx: zmq::Context,
        address: String,
    }

    impl FakeServer {
        fn new() -> FakeServer {
            let ctx = zmq::Context::new();
            let router = ctx.socket(zmq::ROUTER).unwrap();
            router.set_linger(0).unwrap();
            router.bind("tcp://127.0.0.1:*").unwrap();
            let address = router.get_last_endpoint().unwrap().unwrap();
            router.set_rcvtimeo(10_000).unwrap();
            FakeServer { router, _ctx: ctx, address }
        }

        /// `[identity, request_id, method, payload]`
        fn recv(&self) -> Vec<Vec<u8>> {
            self.router.recv_multipart(0).expect("a request within 10s")
        }

        fn shared(self) -> Arc<Mutex<FakeServer>> {
            Arc::new(Mutex::new(self))
        }

        fn send(&self, frames: &[&[u8]]) {
            self.router.send_multipart(frames, 0).unwrap();
        }
    }

    fn pack_seq(seq: i64) -> Vec<u8> {
        pack(&[("seq", Value::from(seq))].into_iter().collect())
    }

    fn health_reply() -> Vec<u8> {
        pack(&[("success", Value::Bool(true)), ("error", Value::Nil)].into_iter().collect())
    }

    async fn on_thread<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        tokio::task::spawn_blocking(f).await.unwrap()
    }

    /// Runs streaming at once each see their own deltas in order, and a run's reply lands
    /// only after its last delta. A malformed frame and an undecodable delta among them are
    /// skipped without losing their neighbours.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn interleaved_runs_see_their_deltas_in_order_before_the_reply() {
        let (n_runs, n_deltas) = (20usize, 50i64);
        let server = FakeServer::new();
        let client = EnvClient::connect(&server.address).unwrap();
        let mut runs = Vec::new();
        for _ in 0..n_runs {
            let client = client.clone();
            runs.push(tokio::spawn(async move {
                let seen = Arc::new(Mutex::new(Vec::new()));
                let sink = seen.clone();
                let mut on_delta = move |d: &Value| sink.lock().unwrap().push(d.get("seq").unwrap().as_i64().unwrap());
                client.request(METHOD_HEALTH, Value::map(), new_request_id(), Some(&mut on_delta)).await.unwrap();
                let seen = seen.lock().unwrap().clone();
                seen
            }));
        }
        on_thread(move || {
            let requests: Vec<Vec<Vec<u8>>> = (0..n_runs).map(|_| server.recv()).collect();
            for seq in 0..n_deltas {
                for r in &requests {
                    server.send(&[&r[0], &r[1], b"delta", &pack_seq(seq)]);
                }
                if seq == n_deltas / 2 {
                    server.send(&[&requests[0][0], b"not-the-protocol"]);
                    server.send(&[&requests[0][0], &requests[0][1], b"delta", b"\xc1"]);
                }
            }
            for r in &requests {
                server.send(&[&r[0], &r[1], b"reply", &health_reply()]);
            }
            server
        })
        .await;
        for run in runs {
            assert_eq!(run.await.unwrap(), (0..n_deltas).collect::<Vec<_>>());
        }
    }

    /// Deltas arriving after a run was dropped are not seen; the dropped run's server-side
    /// rollout gets a `cancel`; a run streaming beside it still sees all of its own.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_cancelled_run_stops_seeing_its_stream() {
        let server = FakeServer::new();
        let client = EnvClient::connect(&server.address).unwrap();
        let server = server.shared();
        let seen_a = Arc::new(Mutex::new(Vec::new()));
        let seen_b = Arc::new(Mutex::new(Vec::new()));
        let spawn_run = |seen: Arc<Mutex<Vec<i64>>>| {
            let client = client.clone();
            tokio::spawn(async move {
                let mut on_delta = move |d: &Value| seen.lock().unwrap().push(d.get("seq").unwrap().as_i64().unwrap());
                client.request(METHOD_RUN, Value::map(), new_request_id(), Some(&mut on_delta)).await
            })
        };
        let a = spawn_run(seen_a.clone());
        let s = server.clone();
        let request_a = on_thread(move || s.lock().unwrap().recv()).await;
        let b = spawn_run(seen_b.clone());
        let s = server.clone();
        let request_b = on_thread(move || s.lock().unwrap().recv()).await;
        for r in [&request_a, &request_b] {
            server.lock().unwrap().send(&[&r[0], &r[1], b"delta", &pack_seq(0)]);
        }
        while seen_a.lock().unwrap().is_empty() || seen_b.lock().unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        a.abort();
        assert!(a.await.unwrap_err().is_cancelled());
        let s = server.clone();
        let cancel = on_thread(move || s.lock().unwrap().recv()).await;
        assert_eq!(cancel[2], b"cancel");
        let cancel_payload = value::from_msgpack(&cancel[3]).unwrap();
        assert_eq!(cancel_payload.get("request_id").unwrap().as_str().unwrap().as_bytes(), request_a[1].as_slice());
        for seq in 1..10 {
            for r in [&request_a, &request_b] {
                server.lock().unwrap().send(&[&r[0], &r[1], b"delta", &pack_seq(seq)]);
            }
        }
        for r in [&request_a, &request_b] {
            server.lock().unwrap().send(&[&r[0], &r[1], b"reply", &health_reply()]);
        }
        b.await.unwrap().unwrap();
        assert_eq!(*seen_a.lock().unwrap(), vec![0]);
        assert_eq!(*seen_b.lock().unwrap(), (0..10).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn a_failed_reply_raises_its_error_and_health_times_out() {
        let server = FakeServer::new();
        let client = EnvClient::connect(&server.address).unwrap();
        let server = server.shared();
        let s = server.clone();
        let reply = tokio::task::spawn_blocking(move || {
            let s = s.lock().unwrap();
            let r = s.recv();
            let failed: Value = [("success", Value::Bool(false)), ("error", Value::from("Cancelled: rollout aborted"))].into_iter().collect();
            s.send(&[&r[0], &r[1], b"reply", &pack(&failed)]);
        });
        let err = client.request(METHOD_HEALTH, Value::map(), new_request_id(), None).await.unwrap_err();
        reply.await.unwrap();
        assert_eq!((err.type_name(), err.to_string().as_str()), ("RuntimeError", "Cancelled: rollout aborted"));
        // nobody answers now
        assert!(!client.health(Duration::from_millis(100)).await.unwrap());
        let err = client.wait_for_server_startup(Duration::from_millis(300)).await.unwrap_err();
        assert_eq!(err.type_name(), "TimeoutError");
        assert!(err.to_string().ends_with("did not become healthy in 0.3s"), "{err}");
    }
}
