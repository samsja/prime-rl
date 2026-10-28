# Rust orchestrator (POC) — architecture

Goal: a 1-1 functional rewrite of `src/prime_rl/orchestrator` in Rust. The pydantic
`OrchestratorConfig` (packages/prime-rl-configs) stays the only config API. Python
validates the config; Rust consumes `config.model_dump(mode="json")`.

## Hard compatibility contracts (must not change)

- Config: same TOML/CLI surface, same pydantic models. Rust mirrors the JSON dump with serde.
- Env servers: unchanged verifiers v1 env server (`env-server`), ZMQ DEALER + msgpack serve protocol
  (`deps/verifiers/verifiers/v1/serve`). Rust speaks the same protocol.
- Trainer: unchanged. Micro batches sent through `transports/batch` (filesystem / zmq) must be byte-
  compatible with what the Python trainer receives today.
- Inference: same vLLM HTTP endpoints (OpenAI API, admin endpoints, weight updates, /metrics).
- Outputs: same metric keys, same monitor outputs (wandb, prime platform, file traces), same checkpoint
  format (a Python run can resume a Rust checkpoint and vice versa), same console log lines where practical.
- Launcher: `uv run rl ...` keeps working. The `orchestrator @ config.json` entrypoint always runs the Rust
  orchestrator (no backend switch). The Python orchestrator stays in-tree only as the behavior reference until
  Rust reaches parity, then it is deleted. The parity baseline for e2e runs is an untouched main checkout.

## Layout

```
rust/
  Cargo.toml                 workspace, members = ["crates/*"]
  ARCHITECTURE.md            this file
  docs/specs/<area>.md       behavior inventory of the Python code + Rust design, one per area
  crates/
    prl-config     serde mirror of OrchestratorConfig JSON              (owner: config)
    prl-wire       verifiers v1 wire types, msgpack codec, ZMQ EnvClient (owner: wire)
    prl-types      shared runtime types: Policy, Progress, events,       (owner: core)
                   DispatchFailure, GroupCancellation, Kind, ...
    prl-train      train sink, algorithms, routing, trajectories,         (owner: train)
                   TrainEpisodes metrics, batch packer, batch transport
    prl-io         inference clients/admin plane, inference metrics,      (owner: io)
                   weight receivers, control server, heartbeat, ckpt, gc
    prl-monitors   monitors facade: file (traces + live), prime, wandb    (owner: monitors)
    prl-core       dispatcher, concurrency, train/eval sources,           (owner: core)
                   curriculum, watcher, eval sink, periodic logger,
                   orchestrator supervisor/wiring
    prl-eval       `uv run eval` on the Rust core (port of src/prime_rl/eval) (owner: eval)
    prl-py         pyo3 extension module `prime_rl_orch_rs`               (owner: scaffold, then core)
packages/prime-rl-orch-rs/   maturin python package (uv workspace member) building prl-py
```

Dependency direction: `prl-config, prl-wire -> prl-types -> prl-train, prl-io, prl-monitors -> prl-core -> prl-py`.

## Python boundary

- `orchestrator @ cfg.toml` (Python) parses `OrchestratorConfig` exactly as today, then calls `prime_rl_orch_rs.run(config_json: str, bridge: PyBridge)` which releases the GIL and runs a
  tokio runtime until the orchestrator exits.
- Things with no sane Rust equivalent stay in Python and are reached through the bridge, called from
  `spawn_blocking` with the GIL: taskset loading (`vf.load_taskset(...).select(...)` -> task data dicts, lazily for
  unbounded tasksets), the wandb monitor, and anything else a spec proves cannot be ported. Keep the bridge small
  and explicit: one Python class, a handful of methods, plain JSON-able / bytes arguments.
- Everything else is native Rust: HF tokenizer via the `tokenizers` crate if needed, HTTP via `reqwest`,
  ZMQ via the `zeromq` crate (or `zmq` bindings), msgpack via `rmp-serde`/`rmpv`, HTTP server via `axum`.

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

## Working rules for all agents

- Python code under `src/prime_rl/orchestrator` and its deps is the source of truth for behavior. Port its unit
  tests (`tests/unit/orchestrator/*`) as Rust tests where they check real logic.
- Own your crate. Coordinate a cross-crate interface directly with the owning sibling via `agent_message`.
- Errors propagate (`anyhow`/`thiserror`), no silent catch-alls. Mirror Python's intentional fault tolerance only.
- Use your own target dir to avoid lock contention: `CARGO_TARGET_DIR=rust/target/<your-agent-name>`.
- Keep the Python side untouched except where your task explicitly says so (entrypoint, bridge, config field).
