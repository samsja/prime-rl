//! A fake env server for orchestrator benchmarks: speaks the verifiers v1 serve protocol
//! (ROUTER socket, `[request_id, kind, data]` frames) and answers every `run` by replaying a
//! recorded episode stream (`record_template.py`) after `--latency-ms`: fresh episode and
//! trace ids, the request's task (data, key = hash = `task_key(data)`), and a random reward
//! so groups have variance. The orchestrator receives exactly what a real server sends.

use std::os::fd::{AsRawFd, RawFd};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use prl_wire::value::{from_msgpack, to_msgpack};
use prl_wire::{task_hash, Map, Value};
use rand::Rng;
use tokio::io::unix::AsyncFd;
use tokio::sync::mpsc;

#[derive(Parser)]
struct Args {
    /// Recorded episode stream: msgpack {"deltas": [bytes], "head": map, "traces": [summary]}.
    #[arg(long)]
    template: PathBuf,
    /// Publish the bound address here (like `env-server`'s address_file).
    #[arg(long)]
    address_file: PathBuf,
    #[arg(long, default_value = "tcp://127.0.0.1:0")]
    bind: String,
    /// Time each run takes before its stream is sent.
    #[arg(long, default_value_t = 0)]
    latency_ms: u64,
}

struct Template {
    deltas: Vec<Value>,
    head: Map,
    traces: Vec<Value>,
}

fn new_id() -> String {
    prl_wire::new_id()
}

fn episode_frames(template: &Template, request: &Value) -> (Vec<Vec<u8>>, Vec<u8>) {
    let task_data = request.get("task_data").and_then(Value::as_map).cloned().unwrap_or_default();
    let hash = task_hash(&task_data);
    let task_type = template.head.get("task").and_then(|t| t.get("type")).cloned().unwrap_or_default();
    let task: Value = [
        ("type", task_type),
        ("data", Value::Map(task_data)),
        ("key", Value::from(hash.as_str())),
        ("hash", Value::from(hash.as_str())),
    ]
    .into_iter()
    .collect();
    let mut ids: Map = Map::new();
    let mut rename = |old: &str| -> String {
        ids.entry(old.to_owned()).or_insert_with(|| Value::from(new_id())).as_str().unwrap().to_owned()
    };
    let reward: f64 = rand::thread_rng().gen();
    let deltas = template
        .deltas
        .iter()
        .map(|delta| {
            let mut delta = delta.clone();
            let id = rename(delta.get("trace").and_then(Value::as_str).unwrap_or_default());
            delta.insert("trace", id.as_str());
            if let Some(open) = delta.get_mut("open").and_then(Value::as_map_mut) {
                open.insert("id".into(), id.as_str().into());
                open.insert("task".into(), task.clone());
            }
            if let Some(rewards) = delta.get_mut("set").and_then(|s| s.get_mut("rewards")).and_then(Value::as_map_mut) {
                for reward_value in rewards.values_mut() {
                    if let Some(r) = reward_value.as_map_mut() {
                        r.insert("score".into(), reward.into());
                    }
                }
            }
            to_msgpack(&delta)
        })
        .collect();
    let mut head = template.head.clone();
    head.insert("id".into(), new_id().into());
    head.insert("task".into(), task);
    let traces: Vec<Value> = template
        .traces
        .iter()
        .map(|summary| {
            let mut summary = summary.clone();
            let id = rename(summary.get("id").and_then(Value::as_str).unwrap_or_default());
            summary.insert("id", id.as_str());
            summary
        })
        .collect();
    let reply: Value = [
        ("success", Value::Bool(true)),
        ("error", Value::Nil),
        ("head", Value::Map(head)),
        ("traces", Value::Array(traces)),
    ]
    .into_iter()
    .collect();
    (deltas, to_msgpack(&reply))
}

struct ZmqFd(RawFd);

impl AsRawFd for ZmqFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

/// Frames to send: `[identity, request_id, kind, data]`.
type Outgoing = Vec<Vec<u8>>;

fn main() {
    let args = Args::parse();
    let raw = from_msgpack(&std::fs::read(&args.template).unwrap()).unwrap();
    let template = Arc::new(Template {
        deltas: raw.get("deltas").unwrap().as_array().unwrap().iter().map(|d| from_msgpack(d.as_bytes().unwrap()).unwrap()).collect(),
        head: raw.get("head").unwrap().as_map().unwrap().clone(),
        traces: raw.get("traces").unwrap().as_array().unwrap().clone(),
    });
    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::ROUTER).unwrap();
    socket.set_linger(0).unwrap();
    socket.set_sndhwm(0).unwrap();
    socket.set_rcvhwm(0).unwrap();
    socket.bind(&args.bind).unwrap();
    let address = socket.get_last_endpoint().unwrap().unwrap();
    let tmp = args.address_file.with_extension("address.tmp");
    std::fs::write(&tmp, &address).unwrap();
    std::fs::rename(&tmp, &args.address_file).unwrap();
    eprintln!("fake env server on {address}");
    let latency = Duration::from_millis(args.latency_ms);
    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    runtime.block_on(async move {
        // SAFETY: the fd belongs to `socket`, which outlives the AsyncFd in this scope.
        let fd = unsafe { AsyncFd::register_with_interest(ZmqFd(socket.get_fd().unwrap()), tokio::io::Interest::READABLE) }.unwrap();
        let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Outgoing>();
        let ok: Vec<u8> = to_msgpack(&[("success", Value::Bool(true)), ("error", Value::Nil)].into_iter().collect());
        // Runs received per 100 ms, as `<unix time> <count>` lines on stderr: the admission timeline.
        let received = Arc::new(std::sync::atomic::AtomicU64::new(0));
        {
            let received = received.clone();
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(Duration::from_millis(100));
                loop {
                    tick.tick().await;
                    let n = received.swap(0, std::sync::atomic::Ordering::Relaxed);
                    if n > 0 {
                        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
                        eprintln!("{now:.1} {n}");
                    }
                }
            });
        }
        loop {
            loop {
                let frames = match socket.recv_multipart(zmq::DONTWAIT) {
                    Ok(frames) => frames,
                    Err(zmq::Error::EAGAIN) => break,
                    Err(e) => panic!("recv: {e}"),
                };
                let Ok([identity, request_id, method, payload]) = <[Vec<u8>; 4]>::try_from(frames) else { continue };
                if method == b"run" {
                    received.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let (template, out_tx) = (template.clone(), out_tx.clone());
                    tokio::spawn(async move {
                        if !latency.is_zero() {
                            tokio::time::sleep(latency).await;
                        }
                        let request = from_msgpack(&payload).unwrap();
                        let (deltas, reply) = episode_frames(&template, &request);
                        for delta in deltas {
                            let _ = out_tx.send(vec![identity.clone(), request_id.clone(), b"delta".to_vec(), delta]);
                        }
                        let _ = out_tx.send(vec![identity, request_id, b"reply".to_vec(), reply]);
                    });
                } else {
                    socket.send_multipart([identity, request_id, b"reply".to_vec(), ok.clone()], zmq::DONTWAIT).unwrap();
                }
            }
            tokio::select! {
                frames = out_rx.recv() => {
                    let frames = frames.unwrap();
                    socket.send_multipart(frames, zmq::DONTWAIT).unwrap();
                    while let Ok(frames) = out_rx.try_recv() {
                        socket.send_multipart(frames, zmq::DONTWAIT).unwrap();
                    }
                }
                ready = fd.readable() => ready.unwrap().clear_ready(),
            }
        }
    });
}
