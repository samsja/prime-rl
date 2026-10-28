use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use serde::Deserialize;
use zeromq::{RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use super::{check_grid, encode_grid, MicroBatch};

/// A trainer rank asking for its micro batches of one step (msgpack map).
#[derive(Debug, Clone, Deserialize)]
struct BatchRequest {
    world_size: usize,
    data_rank: usize,
    step: u64,
}

/// One ROUTER socket answering every trainer rank's DEALER, replying on the connection it asked on.
pub struct ZmqBatchSender {
    socket: RouterSocket,
    data_world_size: usize,
    pub current_step: u64,
}

impl ZmqBatchSender {
    pub async fn new(data_world_size: usize, current_step: u64, host: &str, port: u16) -> Result<Self> {
        let host = bind_host(host, port).await?;
        let endpoint = format!("tcp://{host}:{port}");
        let mut socket = RouterSocket::new();
        socket.bind(&endpoint).await.with_context(|| format!("binding {endpoint}"))?;
        tracing::debug!("Initialized ZMQ micro batch sender (endpoint={endpoint})");
        Ok(Self { socket, data_world_size, current_step })
    }

    pub async fn send(&mut self, grid: Vec<Vec<MicroBatch>>) -> Result<f64> {
        check_grid(&grid, self.data_world_size)?;
        let buffers: Vec<Bytes> = encode_grid(grid).await?.into_iter().map(Bytes::from).collect();
        let requests = self.wait_for_requests().await?;
        let start = Instant::now();
        let num_ranks = requests.len();
        for (identity, request) in requests {
            let buffer = buffers
                .get(request.data_rank)
                .ok_or_else(|| anyhow!("trainer rank requested data rank {} of {}", request.data_rank, buffers.len()))?
                .clone();
            let message = ZmqMessage::try_from(vec![identity, buffer]).map_err(|e| anyhow!("{e}"))?;
            self.socket.send(message).await?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        tracing::debug!("Sent micro batch grid for step {} to {num_ranks} ranks", self.current_step);
        self.current_step += 1;
        Ok(elapsed)
    }

    async fn receive_request(&mut self) -> Result<(Bytes, BatchRequest)> {
        let message = self.socket.recv().await?;
        let mut frames = message.into_vec();
        if frames.len() != 2 {
            bail!("expected [identity, request] frames, got {}", frames.len());
        }
        let payload = frames.pop().unwrap();
        let identity = frames.pop().unwrap();
        let request: BatchRequest = rmp_serde::from_slice(&payload).context("decoding BatchRequest")?;
        if request.step != self.current_step {
            bail!(
                "A trainer rank requested step {}, but the orchestrator is on step {}. Resume both from the same checkpoint step.",
                request.step,
                self.current_step
            );
        }
        Ok((identity, request))
    }

    /// One request per trainer rank; each carries the trainer world size.
    async fn wait_for_requests(&mut self) -> Result<Vec<(Bytes, BatchRequest)>> {
        let (identity, request) = self.receive_request().await?;
        let world_size = request.world_size;
        let mut by_identity: HashMap<Bytes, usize> = HashMap::from([(identity.clone(), 0)]);
        let mut requests = vec![(identity, request)];
        while requests.len() < world_size {
            let (identity, request) = self.receive_request().await?;
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

    pub async fn close(self) {
        let _ = self.socket.close().await;
        tracing::debug!("Closed ZMQ micro batch sender");
    }
}

/// libzmq binds a name to its first IPv4 address and `*` to every interface.
async fn bind_host(host: &str, port: u16) -> Result<String> {
    if host == "*" {
        return Ok("0.0.0.0".into());
    }
    if host.parse::<std::net::IpAddr>().is_ok() {
        return Ok(host.into());
    }
    let addrs: Vec<_> = tokio::net::lookup_host((host, port)).await?.collect();
    let addr = addrs
        .iter()
        .find(|a| a.is_ipv4())
        .or(addrs.first())
        .ok_or_else(|| anyhow!("cannot resolve {host}"))?;
    Ok(addr.ip().to_string())
}
