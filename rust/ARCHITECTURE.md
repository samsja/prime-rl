# Rust orchestrator — architecture

The prime-rl orchestrator, `uv run eval` and the SFT online eval run in Rust. Behavior matches the former Python
orchestrator at commit 8acab25e4 (the parity reference). The pydantic `OrchestratorConfig` / `EvalConfig`
(packages/prime-rl-configs) stay the only config API: Python validates the config, Rust reads the validated
pydantic object directly through pyo3.

## Compatibility contracts

- Config: same TOML/CLI surface, same pydantic models.
- Env servers: unchanged verifiers v1 env server (`env-server`), ZMQ DEALER + msgpack serve protocol
  (`deps/verifiers/verifiers/v1/serve`). Rust speaks the same protocol.
- Trainer: unchanged. Micro batches sent through `transports/batch` (filesystem / zmq) are byte-identical to what
  the Python orchestrator produced.
- Inference: same vLLM HTTP endpoints (OpenAI API, admin endpoints, weight updates, /metrics).
- Outputs: same metric keys, same monitor outputs (wandb, prime platform, file traces), same checkpoint format
  (`progress.pt` stays `torch.load`-able), same console log lines.
- Launcher: `uv run rl ...` is unchanged; the `orchestrator @ config.json` entrypoint runs the Rust orchestrator.
  End-to-end parity is checked against a checkout of 8acab25e4 with `rust/tools/e2e_compare.py`.

## Layout

```
rust/
  Cargo.toml                 workspace, members = ["crates/*"]
  docs/specs/<area>.md       behavior inventory + Rust design, one per area
  crates/
    prl-config     pyo3 extraction of OrchestratorConfig / EvalConfig / SFTOnlineEvalConfig
    prl-wire       verifiers v1 wire types, msgpack + orjson-compatible codecs, ZMQ EnvClient, payload refs
    prl-types      shared runtime types: Policy, Progress, Kind, DispatchFailure, GroupCancellation, Bridge, ...
    prl-train      train sink, algorithms, trajectories, episode metrics, batch packer, batch transport
    prl-io         inference clients/admin plane, inference metrics, weight receivers, control API,
                   heartbeat, checkpoints, logging
    prl-monitors   monitors actor: file (traces + live), prime, wandb (via Python)
    prl-core       dispatcher, concurrency, train/eval sources, curriculum, watcher, eval sink,
                   periodic logger, supervisor (`prl_core::run`)
    prl-eval       `uv run eval` and online eval on the core actors
    prl-py         pyo3 extension module `prime_rl_orch_rs`
  tools/, bench/   e2e parity compare, throughput benchmark
packages/prime-rl-orch-rs/   maturin python package (uv workspace member) building prl-py
```

Dependency direction: `prl-config, prl-wire -> prl-types -> prl-train, prl-io, prl-monitors -> prl-core ->
prl-eval -> prl-py`.

## Python boundary

- `orchestrator @ cfg.toml` parses `OrchestratorConfig` in Python, then `src/prime_rl/rust_bridge.py` calls
  `prime_rl_orch_rs.run(config, bridge)`, which releases the GIL and runs a tokio runtime until the orchestrator
  exits. `uv run eval` and the SFT online eval do the same through `run_eval` / `run_online_eval`.
- Config: `prl-config` declares only the fields Rust reads, as `#[derive(FromPyObject)]` structs extracted once
  from the live pydantic object at startup. Derived values (`resolved_name`, `env_addresses`, ...) are read off the
  Python object; sub-configs Rust only forwards are `model_dump`ed. There is no schema copy and no validation in
  Rust. A test extracts every config under `configs/` and `examples/`, so a renamed pydantic field fails CI.
- The bridge (`PyBridge` in `rust_bridge.py`, see `docs/specs/boundary.md`) covers what has no Rust equivalent:
  taskset loading, the W&B monitor, the NIXL agent, FLOP cost tables, and the opsd/echo renderer hooks. Calls run
  on `spawn_blocking` with the GIL.
- Everything else is native Rust: HTTP via `reqwest`, ZMQ via the `zmq` crate (vendored libzmq), HTTP server via
  `axum`.

## Runtime design: event-driven actors

The Python orchestrator is a central `main_loop` that pulls from `dispatcher.out_q` and calls into components,
plus callbacks (`watcher.on_update`, `concurrency.bind`) and shared mutable objects (`Policy`, `Progress`,
`dispatch_allowed` Event). The Rust design pushes this to native async message passing:

- Each component is an actor: one tokio task that owns its state exclusively and reacts to messages
  (`tokio::select!` over its inbox(es), timers and cancellation). No `Arc<Mutex<_>>` over component state.
- Commands go point-to-point over `mpsc` (component inbox enums). Request/response uses a `oneshot` in the
  command (e.g. the periodic logger asking for gauges).
- Shared slowly-changing values are `tokio::sync::watch` channels with a single writer:
  `policy: watch<Policy>` (written by the watcher), `progress_step: watch<u64>` (written by the shipper),
  `dispatch_gate: watch<bool>` (derived by a tiny gate actor from policy + step, replacing
  `update_dispatch_gate` being called from two places), `concurrency_limit: watch<usize>`.
- Broadcast facts as events (`PolicyAdvanced{version}`, `BatchShipped{step}`, `EvalTriggered{step, envs}`,
  `Draining`) to the components that subscribe. Components never reference the supervisor.
- Data flow:
  `TrainSource/EvalSource -> Dispatcher -> EnvClient(s)` ; completed runs flow
  `Dispatcher -> TrainSink | EvalSink` (episodes, failures, group cancellations) ;
  `TrainSink -> Shipper` (TrainBatch: pack -> send -> ckpt -> metrics -> BatchShipped) ;
  `EvalSink -> EvalReporter` ; `WeightWatcher -> PolicyAdvanced` ;
  `InferenceMetricsCollector -> ConcurrencyController -> concurrency_limit / overload -> Dispatcher` ;
  everything that logs -> `Monitors` actor (single ordered consumer that fans out to file / prime / wandb).
- CPU-heavy work (scoring big groups, packing, metric aggregation over a batch) runs on `spawn_blocking`/rayon,
  never on the reactor threads.
- Lifecycle: a supervisor owns a `CancellationToken` and a `JoinSet`. Any actor that returns an error cancels the
  whole tree (same semantics as `_on_component_done`). Draining is an event, not a flag poked from outside.
  Shutdown has the same global timeout semantics as Python (`SHUTDOWN_TIMEOUT_S`).

## Working rules

- Behavior reference: the Python orchestrator at commit 8acab25e4. Byte-level formats (trainer batches, trace
  files, records, checkpoints) are tested against fixtures generated from that checkout.
- Errors propagate (`anyhow`/`thiserror`), no silent catch-alls. Mirror intentional fault tolerance only.
- Use a per-agent target dir to avoid lock contention: `CARGO_TARGET_DIR=rust/target/<name>`.
