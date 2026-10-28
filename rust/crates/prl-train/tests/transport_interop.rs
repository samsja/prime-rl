//! The Python trainer's batch receivers decode what the Rust senders send, losslessly.
//! Needs the repo's Python env: `cargo test -p prl-train --test transport_interop -- --ignored`.

use std::path::PathBuf;
use std::process::Stdio;

use prl_train::packer::prepare_batch;
use prl_train::{encode_micro_batches, BatchSender, MicroBatch, PayloadSlice, TrainingSample, TransportConfig};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, BufReader};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..").canonicalize().unwrap()
}

fn sample(len: usize, token: i32, with_payload: bool) -> TrainingSample {
    TrainingSample {
        token_ids: (0..len as i32).map(|i| token + i).collect(),
        mask: (0..len).map(|i| i >= len / 2).collect(),
        logprobs: (0..len).map(|i| -(i as f32) / 8.0).collect(),
        temperatures: vec![1.0; len],
        env_name: format!("env-{}", token % 3),
        advantages: Some((0..len).map(|i| if i >= len / 2 { 0.5 } else { 0.0 }).collect()),
        ce_weights: (token % 2 == 0).then(|| vec![0.25; len]),
        staleness: Some(vec![1; len]),
        trace_id: Some(format!("trace-{token}")),
        branch_index: Some(0),
        payload: with_payload.then(|| PayloadSlice {
            url: "http://127.0.0.1:1".into(),
            id: format!("t{token}-0"),
            num_tokens: len,
            fields: vec!["sampling_mask".into()],
            routed_shape: None,
            routed_dtype: None,
        }),
        ..Default::default()
    }
}

fn grid(step: i32) -> Vec<Vec<MicroBatch>> {
    let samples = (0..7).map(|i| sample(5 + i as usize * 3, 100 * step + i, i % 3 == 0)).collect();
    let cost = |lens: &[usize]| lens.iter().map(|&n| n as i128).sum::<i128>();
    prepare_batch(samples, 32, 2, &cost, 4).unwrap()
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

async fn run(kind: &str) {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let steps = 2;
    let mut child = tokio::process::Command::new("uv")
        .args(["run", "--no-sync", "python", "rust/crates/prl-train/tests/fixtures/transport_receivers.py", kind])
        .arg(dir.path())
        .arg(port.to_string())
        .arg(steps.to_string())
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("uv must be on PATH");
    let transport = match kind {
        "zmq" => TransportConfig::Zmq { host: "127.0.0.1".into(), port },
        _ => TransportConfig::FileSystem,
    };
    let mut sender = BatchSender::new(dir.path().to_path_buf(), 2, 0, &transport).await.unwrap();
    let mut expected = Vec::new();
    for step in 0..steps {
        let grid = grid(step as i32);
        expected.push(grid.iter().map(|list| format!("{:x}", Sha256::digest(encode_micro_batches(list)))).collect::<Vec<_>>());
        sender.send(grid).await.unwrap();
    }
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    let mut seen = 0;
    while let Some(line) = lines.next_line().await.unwrap() {
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let (step, rank) = (v["step"].as_u64().unwrap() as usize, v["rank"].as_u64().unwrap() as usize);
        let data_rank = [0, 0, 1, 1][rank];
        assert_eq!(v["sha256"].as_str().unwrap(), expected[step][data_rank], "{kind} step {step} rank {rank}");
        seen += 1;
    }
    assert!(child.wait().await.unwrap().success());
    assert_eq!(seen, 4 * steps);
    sender.close().await;
}

#[tokio::test]
#[ignore = "needs the repo's Python env (uv run)"]
async fn python_filesystem_receivers_decode_rust_batches() {
    run("filesystem").await;
}

#[tokio::test]
#[ignore = "needs the repo's Python env (uv run)"]
async fn python_zmq_receivers_decode_rust_batches() {
    run("zmq").await;
}
