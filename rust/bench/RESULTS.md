# Orchestrator throughput: Rust vs Python (CPU only)

How far each orchestrator scales when nothing else is the bottleneck. One orchestrator per
run, against a fake world that does no real work:

- **inference**: `tools/mock-inference`, a fake vLLM. It serves `/inference/v1/generate`
  (token-in/out, packed logprobs), `/v1/models`, `/metrics` and the admin routes
  (pause/resume/update_weights). Every generate returns one of six canned completions.
- **env servers**: `--env fake` (the default) runs `tools/fake-env-server`, which speaks
  the verifiers v1 serve protocol. It replays one real reverse-text episode stream
  recorded from a real `env-server` (`record_template.py`), with fresh episode/trace ids,
  the dispatched task (key and hash recomputed with `task_key`) and a random reward.
  Every run is answered after `--env-latency-ms`, which stands in for generation time.
  The orchestrator receives exactly what a real server streams.
  `--env real` uses real `env-server` pools for reverse-text with an in-process one-call
  harness (`prl_bench_harness.py`); the built-in `null` harness starts a Python program
  per rollout and caps a worker at about 8 rollouts/s.
- **trainer**: `trainer_stub.py` speaks the ZMQ batch transport and the filesystem weight
  broadcast. It pulls each step's micro batches and immediately publishes the next policy
  version (sentinel files only).

Config: reverse-text, GRPO, group size 16, `seq_len` 2048, `max_completion_tokens` 128,
fixed concurrency (`min_inflight = max_inflight = initial_inflight`), batch size =
max_inflight unless noted, zmq rollout transport, filesystem weight broadcast, file monitor
only (Prime/W&B off). Measured over steps 5-12 (warmup: 4 steps; real-env rows: steps 4-10).
"Avg in flight" is the orchestrator's `dispatcher/inflight/train` gauge. CPU and RSS are
the orchestrator process from `/proc`. "Loop lag" is `event_loop_lag/*`: the asyncio
event-loop lag for Python, and the scheduling delay of the Rust runtime's equivalent probe.

Versions: Python = a checkout of `8acab25e4` (the branch point). Rust = this branch
at `7b25613dd` (16k at 250 ms with and without file monitor), `0b9b9704f` (4k at 250 ms),
`d2991ab7e` (16k at 50 ms, 65k, real env), `f38172da5` (4k at 50 ms, before the memory
change), and before `f38172da5` (256, 1024; not affected by the fixes below).

Machine: 30 CPUs, shared with other agents' runs (load average 1-23 during the
measurements). Each row is a single run, so expect ±10-20 % noise.

## Results

| orchestrator | max_inflight | batch | env | rollouts/s | steps/s | avg in flight | CPU % | RSS max (GB) | loop lag p99 (ms) | loop lag max (ms) | time/pack (s) | time/step (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| python | 256 | 256 | fake 250 ms | 78 | 0.303 | 30 | 25 | 1.5 | 7.9 | 14.9 | 0.010 | 3.30 |
| rust | 256 | 256 | fake 250 ms | 85 | 0.334 | 40 | 12 | 1.5 | 1.9 | 2.3 | 0.001 | 3.12 |
| python | 1024 | 1024 | fake 250 ms | 368 | 0.359 | 127 | 71 | 1.6 | 58.0 | 121.8 | 0.040 | 3.10 |
| rust | 1024 | 1024 | fake 250 ms | 333 | 0.325 | 180 | 48 | 1.8 | 1.9 | 2.1 | 0.003 | 3.15 |
| python | 4096 | 4096 | fake 250 ms | 484 | 0.118 | 643 | 107 | 2.1 | 106.9 | 165.2 | 0.250 | 8.36 |
| rust | 4096 | 4096 | fake 250 ms | 1253 | 0.306 | 507 | 183 | 2.4 | 4.5 | 8.5 | 0.012 | 3.26 |
| python | 16384 | 16384 | fake 250 ms | 436 | 0.027 | 2762 | 107 | 4.2 | 230.8 | 392.4 | 0.804 | 39.56 |
| rust | 16384 | 16384 | fake 250 ms | 1936 | 0.118 | 574 | 318 | 5.9 | 6.6 | 19.7 | 0.046 | 8.50 |
| python | 4096 | 4096 | fake 50 ms | 518 | 0.127 | 496 | 107 | 2.1 | 83.1 | 117.3 | 0.135 | 8.40 |
| rust | 4096 | 4096 | fake 50 ms | 2455 | 0.599 | 406 | 303 | 3.4 | 5.3 | 8.6 | 0.016 | 1.69 |
| python | 16384 | 16384 | fake 50 ms | 588 | 0.036 | 2871 | 107 | 4.4 | 368.5 | 836.6 | 0.682 | 28.05 |
| rust | 16384 | 16384 | fake 50 ms | 1917 | 0.117 | 348 | 307 | 4.6 | 8.4 | 33.7 | 0.063 | 8.45 |
| rust | 65536 | 16384 | fake 250 ms | 2142 | 0.131 | 4123 | 356 | 6.4 | 13.8 | 50.2 | 0.044 | 7.90 |
| python | 16384 | 16384 | fake 250 ms, no file monitor | 575 | 0.035 | 0 | 102 | 3.8 | - | - | nan | nan |
| rust | 16384 | 16384 | fake 250 ms, no file monitor | 4262 | 0.260 | 0 | 431 | 4.2 | - | - | nan | nan |
| python | 2048 | 1024 | real | 374 | 0.365 | 93 | 110 | 1.7 | 82.9 | 236.8 | 0.043 | 2.37 |
| rust | 2048 | 1024 | real | 540 | 0.527 | 204 | 108 | 2.0 | 3.9 | 8.3 | 0.007 | 1.93 |

## Findings

- **Python saturates at about 450-590 rollouts/s.** From 4k in flight on, the Python
  orchestrator pins one core (107 % CPU, the asyncio loop plus helper threads), and adding
  in-flight work stops helping.
  - Event-loop lag grows with load: p99 0.1 s at 4k, 0.23-0.37 s at 16k, max 0.84 s.
  - Packing a 16k batch takes 0.7-0.8 s (`time/pack`).
  - Time to the first step at 16k is 166-211 s (Rust: 27-31 s).
  - It keeps 2.7k episodes in flight but cannot process them faster.
- **Rust reaches 1.9-2.5k rollouts/s with the file monitor on**: 4.4x Python at 16k in
  flight, 2.6x at 4k, and 4.7x with 50 ms episodes at 4k. Without the file monitor it
  reaches 4.3k rollouts/s at 16k, 7.4x Python (575/s, which is still CPU-bound at 102 %).
  - It uses 1.8-3.6 cores, and no single thread is saturated (the busiest is about 30 % at 16k).
  - Runtime lag p99 stays at 2-14 ms, and packing a 16k batch takes 0.05 s.
- **Memory at 16k is close**: Rust 4.2-5.9 GB, Python 3.8-4.4 GB. It was 8-9.5 GB before
  `d2991ab7e` (see below).
- **With real env servers** (8 replicas x 2 workers, 2048 in flight), Rust delivers 540
  rollouts/s against Python's 374 (+45 %).
  - Python is CPU-bound there: 110 %, with env servers at 500 %.
  - With Rust, the env servers become the limit: they reach 794 %, near their measured
    capacity (below).
- **Below about 1k in flight both are latency-bound** and equal: throughput is roughly the
  number in flight divided by episode latency.

## Bottlenecks found and fixed during the benchmark

| where | symptom | fix |
|---|---|---|
| `TrainPipeline::publish_view` -> `TrainSink::pending_batch_by_env` (prl-core/prl-train) | O(batch) per episode, O(batch²) per step; the train-pipeline thread at 100 %, 680 rollouts/s at 16k | view refreshed on a 1 s tick (`f38172da5`) -> 1.9k/s |
| `Dispatcher::queue_live` (prl-core) | once there were more than 20k bookkeeping events (≥ 20k really in flight) every delta re-partitioned the live buffer; 50 rollouts/s at 65k | trim only when deltas exceed the cap (core) -> 2.1k/s at 65k |
| `prl_wire::Episode` | 35 KB per episode, half of it the agent config and task data, deep-copied by every `Episode` clone (sink per-trace copies, monitor snapshots); 8-9.5 GB RSS at 16k | `AgentInfo.config` / `TraceTask.data` behind `Arc`, shared across an env's episodes and a group's rollouts (`d2991ab7e`) -> 4.5 GB |

The live-buffer case exists in Python too (same algorithm). Python just never gets enough
episodes in flight to hit it.

## What limits Rust now: the file monitor

At 16k in flight, turning the file monitor off (`--no-file-monitor`) raises Rust from 1.9k to
4.3k rollouts/s (3.7 s/step instead of 8.5 s). Python goes from 436 to 575.

How the monitor caps throughput:
- The train pipeline awaits `monitors.log_episodes` for every episode, and each one is
  written serially: record, JSON, zstd chunk and index row.
- At about 2k episodes/s that writer is the ceiling. The train pipeline falls behind, the
  results channel fills, and the dispatcher stops admitting.
- `fake-env-server` logs its admissions every 100 ms, and that timeline shows dispatch
  stalling for 0.5-2 s several times per step.

The dispatcher fixes `0b9b9704f` (ship batches in a task) and `7b25613dd` (results handed
over before scheduling) remove real stalls but don't change this ceiling. Making the
record write parallel or batched is the next step (monitors).
`prl_wire::Episode::to_json` costs 46 us per reverse-text episode, down from 60 us now that
shared subtrees are dumped once, so it is a small part of that writer.

## Env server capacity (no orchestrator: `env-load` drives the replicas directly)

| env | concurrency | replicas x workers | rollouts/s | env CPU |
|---|---|---|---|---|
| real, `null` harness | 512 | 2 x 2 | 33 | - |
| real, bench harness | 512 | 2 x 2 | 496 | 2 x 234 % |
| real, bench harness | 4096 | 4 x 4 | 887 | 4 x 310 % |
| real, bench harness | 4096 | 12 x 1 | 726 | 12 x 92 % |
| fake, 1 s latency | 4096 | 2 | 4006 (= concurrency / latency) | 2 x 40 % |

The mock inference used under 15 % CPU in every run.

## Reproduce

```
cd rust/bench/tools && CARGO_TARGET_DIR=../../target/wire-bench cargo build --release && cd -
uv run --no-sync python rust/bench/record_template.py rust/bench/template.msgpack   # once (committed)
uv run --no-sync python rust/bench/run_matrix.py --inflight 256 1024 4096 16384 --env-latency-ms 250 --env-replicas 6 \
    --python-repo <checkout of 8acab25e4 with its .venv synced>
uv run --no-sync python rust/bench/run_bench.py rust 2048 1024 --env real --env-replicas 8 --env-workers 2
uv run --no-sync python rust/bench/run_bench.py rust 16384 16384 --env-latency-ms 250 --env-replicas 6 --no-file-monitor
uv run --no-sync python rust/bench/run_bench.py env-load 4096 0 --env real --env-replicas 4 --env-workers 4
uv run --no-sync python rust/bench/summarize.py /tmp/prl-orch-bench/results.jsonl
```
Raw results for this table: `results.jsonl`. Each run's logs and metrics are kept under
`--work-dir` (default `/tmp/prl-orch-bench/<name>`); traces and broadcasts are deleted.
