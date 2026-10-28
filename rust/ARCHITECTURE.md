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

## Layout

```
rust/
  Cargo.toml                 workspace, members = ["crates/*"]
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
- The bridge (`PyBridge` in `rust_bridge.py`) covers what has no Rust equivalent: taskset loading, the W&B
  monitor, the NIXL agent, FLOP cost tables, and the opsd/echo renderer hooks. Calls take plain arguments, return
  JSON `bytes`, and run on `spawn_blocking` with the GIL; a Python exception becomes an error with its traceback.
- Python keeps its signal handlers: the main thread polls them while Rust runs, and a handler that raises
  (Ctrl-C, a launcher's SIGTERM handler) cancels the run, then re-raises once Rust has stopped.
- Everything else is native Rust: HTTP via `reqwest`, ZMQ via the `zmq` crate (vendored libzmq), HTTP server via
  `axum`.

## Runtime design: event-driven actors

The Python orchestrator is a central `main_loop` over `dispatcher.out_q` plus callbacks and shared mutable
objects. The Rust orchestrator (`prl_core::run`) is a set of tokio actors: each owns its state, reacts in a
`tokio::select!` over its inbox(es), timers and cancellation, and is reached through a cloneable handle
(`mpsc` commands; request/response with a `oneshot`). No `Arc<Mutex<_>>` over component state.

Actors (`prl-core` unless noted):

- `Dispatcher`: groups, in-flight env runs (a `JoinSet`), permits, admission window, rate limiter, mode
  (PREFER_TRAIN / PREFER_EVAL) and the `EvalSource`. Draws train tasks from the `TrainSource` actor. The
  dispatch gate (`(progress_step - 1) - policy.version <= 1`) is evaluated inside the dispatcher from the
  `policy` and `progress_step` watches. Results go to per-kind outboxes drained into bounded channels
  (`DispatchResult<Episode>`: episode, `DispatchFailure`, `GroupCancellation`) to the two pipelines, so the
  dispatcher keeps serving commands while a sink is busy. Run tasks relay env-server deltas on a channel; the
  live view is flushed to `Monitors` every 0.5 s.
- `TrainSource`: train tasks across envs + curricula; `on_result` admission, metrics, checkpoint state.
- `TrainPipeline`: `prl_train::TrainSink` + the ship flow, checkpoint saves and control pauses. Single
  writer of `progress_step`. Pack + send run in a ship task while results keep flowing into the sink
  (stamped with the next step, the window Python consumed them in); the post-send work (step advance, ckpt,
  metrics, drain) runs in order when the send completes, and batches completed meanwhile queue behind it.
- `EvalPipeline`: eval results -> `EvalSink` epochs -> finalize; `trigger_eval` after each applied policy.
- `WeightWatcher`: polls the weight receiver (prl-io), single writer of `policy`. Per update: dispatcher
  `version_pending` -> receive -> `policy` -> hooks in order: `EvalPipeline` (eval trigger, may switch the
  dispatcher to PREFER_EVAL) -> dispatcher `version_applied`.
- `ConcurrencyController`: pure AIMD state machine wrapped in an actor; fed engine load samples by prl-io's
  `InferenceMetricsCollector` and per-episode completions by the dispatcher; single writer of `limit`; sends
  overload excess to the dispatcher.
- `Monitors` (prl-monitors): single ordered consumer of every log call (metrics, episodes, annotations,
  live events, eval plans), fanning out to file / prime / wandb. Episodes are logged to `all` on arrival
  before they reach a sink.
- Also: control server (prl-io), lag monitor, pipeline-view `PeriodicLogger` (polls the dispatcher and
  concurrency handles and reads the watcher / pipeline progress watches).

Watch channels (single writer): `policy: Policy` (watcher), `progress_step: u64` (train pipeline),
`limit: usize` (concurrency), `inflight: usize` (dispatcher), `idle: bool` (dispatcher), plus progress
watches the pipelines and watcher publish for the pipeline view.

Lifecycle: the supervisor owns a `CancellationToken` and a `JoinSet` of named components; any component
exiting early fails the run (`_on_component_done` semantics). Draining starts in the train pipeline after the
final batch (`dispatcher.drain()` returns the train cancellations directly). When the dispatcher reports
`idle`, the supervisor stops it, lets the eval pipeline consume its closed channel, waits for the final
broadcast, saves the final checkpoint, finalizes the monitors, and tears down within `SHUTDOWN_TIMEOUT`
(300 s, then process exit).

## Working rules

- Behavior reference: the Python orchestrator at commit 8acab25e4. Byte-level formats (trainer batches, trace
  files, records, checkpoints) are tested against fixtures generated from that checkout.
- Errors propagate (`anyhow`/`thiserror`), no silent catch-alls. Mirror intentional fault tolerance only.
- Use a per-agent target dir to avoid lock contention: `CARGO_TARGET_DIR=rust/target/<name>`.
