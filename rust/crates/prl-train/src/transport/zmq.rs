use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

use super::{check_grid, encode_grid, MicroBatch};

/// How often a blocked request wait checks whether the sender was closed.
const POLL_INTERVAL_MS: i64 = 100;

/// A trainer rank asking for its micro batches of one step (msgpack map).
#[derive(Debug, Clone, Deserialize)]
struct BatchRequest {
    world_size: usize,
    data_rank: usize,
    step: u64,
}

/// One libzmq ROUTER socket answering every trainer rank's DEALER, replying on the connection it
/// asked on (`ROUTER_MANDATORY`: replying to a rank that went away errors). The socket is blocking,
/// so each send runs on a blocking thread.
pub struct ZmqBatchSender {
    socket: Option<zmq::Socket>,
    _context: zmq::Context,
    closed: Arc<AtomicBool>,
    data_world_size: usize,
    pub current_step: u64,
}

impl ZmqBatchSender {
    pub async fn new(data_world_size: usize, current_step: u64, host: &str, port: u16) -> Result<Self> {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::ROUTER)?;
        socket.set_router_mandatory(true)?;
        socket.set_linger(0)?;
        let endpoint = format!("tcp://{}:{port}", bind_host(host, port).await?);
        socket.bind(&endpoint).with_context(|| format!("binding {endpoint} (host {host})"))?;
        tracing::debug!("Initialized ZMQ micro batch sender (endpoint={endpoint})");
        Ok(Self {
            socket: Some(socket),
            _context: context,
            closed: Arc::new(AtomicBool::new(false)),
            data_world_size,
            current_step,
        })
    }

    pub async fn send(&mut self, grid: Vec<Vec<MicroBatch>>) -> Result<f64> {
        check_grid(&grid, self.data_world_size)?;
        let buffers = encode_grid(grid).await?;
        let socket = self.socket.take().ok_or_else(|| anyhow!("ZMQ batch sender is closed"))?;
        let (step, closed) = (self.current_step, self.closed.clone());
        let (socket, result) = tokio::task::spawn_blocking(move || {
            let result = send_blocking(&socket, &buffers, step, &closed);
            (socket, result)
        })
        .await?;
        self.socket = Some(socket);
        let (elapsed, num_ranks) = result?;
        tracing::debug!("Sent micro batch grid for step {} to {num_ranks} ranks", self.current_step);
        self.current_step += 1;
        Ok(elapsed)
    }

    pub async fn close(mut self) {
        self.closed.store(true, Ordering::Relaxed);
        drop(self.socket.take());
        tracing::debug!("Closed ZMQ micro batch sender");
    }
}

impl Drop for ZmqBatchSender {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Relaxed);
    }
}

fn send_blocking(socket: &zmq::Socket, buffers: &[Vec<u8>], step: u64, closed: &AtomicBool) -> Result<(f64, usize)> {
    let requests = wait_for_requests(socket, step, closed)?;
    let start = Instant::now();
    for (identity, request) in &requests {
        let buffer = buffers
            .get(request.data_rank)
            .ok_or_else(|| anyhow!("trainer rank requested data rank {} of {}", request.data_rank, buffers.len()))?;
        socket.send(identity.as_slice(), zmq::SNDMORE)?;
        socket.send(buffer.as_slice(), 0)?;
    }
    Ok((start.elapsed().as_secs_f64(), requests.len()))
}

fn receive_request(socket: &zmq::Socket, step: u64, closed: &AtomicBool) -> Result<(Vec<u8>, BatchRequest)> {
    while socket.poll(zmq::POLLIN, POLL_INTERVAL_MS)? == 0 {
        if closed.load(Ordering::Relaxed) {
            bail!("ZMQ batch sender closed while waiting for trainer requests");
        }
    }
    let mut frames = socket.recv_multipart(0)?;
    if frames.len() != 2 {
        bail!("expected [identity, request] frames, got {}", frames.len());
    }
    let payload = frames.pop().unwrap();
    let identity = frames.pop().unwrap();
    let request: BatchRequest = rmp_serde::from_slice(&payload).context("decoding BatchRequest")?;
    if request.step != step {
        bail!(
            "A trainer rank requested step {}, but the orchestrator is on step {step}. Resume both from the same checkpoint step.",
            request.step
        );
    }
    Ok((identity, request))
}

/// One request per trainer rank; each carries the trainer world size.
fn wait_for_requests(socket: &zmq::Socket, step: u64, closed: &AtomicBool) -> Result<Vec<(Vec<u8>, BatchRequest)>> {
    let (identity, request) = receive_request(socket, step, closed)?;
    let world_size = request.world_size;
    let mut by_identity: HashMap<Vec<u8>, usize> = HashMap::from([(identity.clone(), 0)]);
    let mut requests = vec![(identity, request)];
    while requests.len() < world_size {
        let (identity, request) = receive_request(socket, step, closed)?;
        match by_identity.get(&identity) {
            Some(&index) => requests[index] = (identity, request),
            None => {
                by_identity.insert(identity.clone(), requests.len());
                requests.push((identity, request));
            }
        }
    }
    Ok(requests)
}

/// libzmq treats a bind host as an interface or address; resolve names to an address
/// first (IPv4 preferred, as libzmq without IPv6 enabled), keep `*` and literals.
async fn bind_host(host: &str, port: u16) -> Result<String> {
    if host == "*" || host.parse::<std::net::IpAddr>().is_ok() {
        return Ok(host.to_owned());
    }
    let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host, port)).await?.collect();
    let addr = addrs.iter().find(|a| a.is_ipv4()).or(addrs.first()).ok_or_else(|| anyhow!("cannot resolve {host}"))?;
    Ok(addr.ip().to_string())
}
