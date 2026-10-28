//! Drive env servers directly (no orchestrator) to measure what they and the inference
//! behind them sustain: keeps `--concurrency` runs in flight, spread least-loaded over the
//! replicas, for `--seconds`, and prints rollouts/s as JSON.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use prl_wire::{EnvClient, Map, Value};

#[derive(Parser)]
struct Args {
    /// Files holding the replicas' addresses (as env servers publish them).
    #[arg(long, num_args = 1.., required = true)]
    address_files: Vec<std::path::PathBuf>,
    /// JSON list of task data dicts to cycle through.
    #[arg(long)]
    tasks: std::path::PathBuf,
    /// JSON client config (vf ClientConfig dump).
    #[arg(long)]
    client: std::path::PathBuf,
    /// JSON sampling config (vf SamplingConfig dump).
    #[arg(long)]
    sampling: std::path::PathBuf,
    #[arg(long)]
    model: String,
    #[arg(long, default_value_t = 256)]
    concurrency: usize,
    #[arg(long, default_value_t = 30)]
    seconds: u64,
    /// Seconds of ramp-up excluded from the measurement.
    #[arg(long, default_value_t = 5)]
    warmup: u64,
}

fn read_json(path: &std::path::Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let tasks: Vec<Map> = read_json(&args.tasks).into_array().unwrap().into_iter().map(|t| t.into_map().unwrap()).collect();
    let tasks = Arc::new(tasks);
    let client_config = Arc::new(read_json(&args.client));
    let sampling = Arc::new(read_json(&args.sampling));
    let mut clients = Vec::new();
    for file in &args.address_files {
        let address = loop {
            match std::fs::read_to_string(file) {
                Ok(a) if !a.trim().is_empty() => break a.trim().to_owned(),
                _ => tokio::time::sleep(Duration::from_millis(200)).await,
            }
        };
        let client = EnvClient::connect(&address).unwrap();
        client.wait_for_server_startup(Duration::from_secs(600)).await.unwrap();
        clients.push(client);
    }
    let in_flight: Arc<Vec<AtomicU64>> = Arc::new(clients.iter().map(|_| AtomicU64::new(0)).collect());
    let done = Arc::new(AtomicU64::new(0));
    let failed = Arc::new(AtomicU64::new(0));
    let deadline = Instant::now() + Duration::from_secs(args.warmup + args.seconds);
    for worker in 0..args.concurrency {
        let (clients, in_flight, tasks, client_config, sampling, model, done, failed) = (
            clients.clone(),
            in_flight.clone(),
            tasks.clone(),
            client_config.clone(),
            sampling.clone(),
            args.model.clone(),
            done.clone(),
            failed.clone(),
        );
        tokio::spawn(async move {
            let mut i = worker;
            while Instant::now() < deadline {
                let replica = (0..clients.len()).min_by_key(|&r| in_flight[r].load(Ordering::Relaxed)).unwrap();
                in_flight[replica].fetch_add(1, Ordering::Relaxed);
                let result = clients[replica].run(&tasks[i % tasks.len()], &*client_config, &model, &*sampling, None).await;
                in_flight[replica].fetch_sub(1, Ordering::Relaxed);
                match result {
                    Ok(episode) if episode.ok => done.fetch_add(1, Ordering::Relaxed),
                    _ => failed.fetch_add(1, Ordering::Relaxed),
                };
                i += 1;
            }
        });
    }
    tokio::time::sleep(Duration::from_secs(args.warmup)).await;
    let (done0, t0) = (done.load(Ordering::Relaxed), Instant::now());
    tokio::time::sleep(Duration::from_secs(args.seconds)).await;
    let (done1, t1) = (done.load(Ordering::Relaxed), Instant::now());
    let rate = (done1 - done0) as f64 / (t1 - t0).as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"concurrency": args.concurrency, "replicas": clients.len(), "rollouts_per_s": rate, "ok": done1, "failed": failed.load(Ordering::Relaxed)})
    );
    std::process::exit(0);
}
