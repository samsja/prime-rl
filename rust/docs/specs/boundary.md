# Python <-> Rust boundary

Owner: scaffold (Python side: `src/prime_rl/orchestrator/rust.py`, entrypoint, packaging;
Rust side of the call: `rust/crates/prl-py`). Core/monitors request bridge methods; scaffold adds them.

## Packaging

- `rust/crates/prl-py` is a pyo3 (abi3, cp312) `cdylib` named `prime_rl_orch_rs`.
  The `extension-module` cargo feature is enabled only by maturin, so `cargo test` links libpython.
- `packages/prime-rl-orch-rs` is a maturin package (`manifest-path = ../../rust/crates/prl-py/Cargo.toml`,
  release profile). It is a uv workspace member and a dependency of `prime-rl`, so `uv sync` builds it.
  `[tool.uv] cache-keys` cover every `rust/**/Cargo.toml`, `rust/crates/**/*.rs` and `rust/Cargo.lock`,
  so `uv sync` / `uv run` rebuild the extension when Rust sources change.
- `rust/.cargo/config.toml` sets `PYO3_PYTHON` to the project venv (`.venv/bin/python`) for plain cargo
  commands; maturin sets its own. `prl-py`'s build.rs gives its tests an rpath to libpython and the interpreter's
  `PYTHONHOME` (uv's standalone Python does not find its stdlib when embedded), so `cargo test -p prl-py` just works.
- While crates are in flux, run Python with `uv run --no-sync ...`: a plain `uv run` rebuilds the extension
  (release) and fails if any workspace crate does not build.

## Entry

`orchestrator @ cfg.toml` (`prime_rl.entrypoints.orchestrator:main`, used by the `rl` launcher) always runs the
Rust orchestrator. There is no backend switch. The Python orchestrator (`prime_rl.orchestrator.orchestrator`) stays
in-tree only as the behavior reference and is deleted once Rust reaches parity.

1. `cli(OrchestratorConfig)` parses and validates the config exactly as before.
2. `prime_rl.orchestrator.rust.run_rust_orchestrator(config)`:
   - `write_resolved_config(config.output_dir, config, "orchestrator")` (same file as the Python orchestrator),
   - `setup_logger(config.log.level, json_logging=config.log.json_logging)` and
     `intercept_vf_logging("verifiers.v1", "WARN")` for Python-side logs (bridge, verifiers),
   - `prime_rl_orch_rs.run(config, PyBridge(config))`.
3. `prime_rl_orch_rs.run(config: OrchestratorConfig, bridge: PyBridge) -> None` extracts the config with the GIL
   held (prl-config's pyo3 `FromPyObject`, reading the live validated pydantic object; owner: config), then
   releases the GIL (`py.allow_threads`) and runs a multi-thread tokio runtime (`block_on`) on
   `prl_core::run(config: prl_config::OrchestratorConfig, bridge: Arc<dyn prl_types::Bridge>)` until the
   orchestrator exits. An `Err` is raised in Python as `RuntimeError(f"{err:#}")`; the process exits non-zero.
   (Until prl-core lands, `run` is a stub that logs and returns.)

## Threading and GIL rules

- The main thread is inside `block_on` with the GIL released for the whole run.
- Rust calls the bridge only through `prime_rl_orch_rs::bridge::PyBridge` (Rust), which runs each call on
  `tokio::task::spawn_blocking` and takes the GIL there (`Python::with_gil`). Never call Python on a reactor thread.
- Bridge calls take plain arguments (`str`, `int`, `bytes`) and return JSON-encoded `bytes`, decoded with serde.
  No Python objects cross the boundary or are held by Rust besides the bridge object itself.
- A Python exception becomes an `anyhow::Error` with context `PyBridge.<method> failed` and the formatted traceback.
- Concurrent bridge calls are allowed; they serialize on the GIL (Python code releasing the GIL, e.g. dataset IO,
  can overlap). The bridge pre-seeds datasets' tqdm lock, as `Envs.start` does, for concurrent taskset loads.
- Signals: while Python waits in `run`, Python signal handlers do not run. The Rust supervisor owns
  SIGINT/SIGTERM (`tokio::signal`) and turns them into cancellation.

## Bridge methods (`prime_rl.orchestrator.rust.PyBridge`)

`PyBridge(config)` takes an `OrchestratorConfig` (sources: `train.source`, `eval.source`) or an `EvalConfig` /
`SFTOnlineEvalConfig` (eval-only: `config.source`; `flop_cost_table` is orchestrator-only).
`split` is `"train"` or `"eval"`; `env_name` is the source's `resolved_name`.

| method | returns (JSON bytes) | Python behavior ported from |
|---|---|---|
| `load_tasks(split, env_name)` | `{"num_tasks": int \| null, "tasks": [TaskRecord]}` | `Env.start`: `vf.load_taskset(source.env.taskset).select(source.select)`; bounded -> materialize all; unbounded -> `num_tasks: null`, `tasks: []`, iterator kept for `next_tasks` |
| `next_tasks(split, env_name, n)` | `[TaskRecord]`, fewer than `n` = stream ended | `iter(taskset)` of an unbounded taskset (`TrainSource` / curriculum pull) |
| `flop_cost_table(max_len)` | `[str]`: decimal `bin_cost([n])` for `n in 0..=max_len` (may exceed 64 bits) | `BatchPacker.__init__`: `FlopCounter(get_model(ModelConfig(name=model.name, attn="flash_attention_3", trust_remote_code=tokenizer.trust_remote_code), device="meta"))` |
| `render_ids(model_name, renderer_config_json, system_content)` | `[int]` | opsd hint block: `create_renderer(load_tokenizer(model_name), RendererConfig).render_ids([{"role": "system", "content": ...}], add_generation_prompt=False)`; renderer cached per (model, config) |
| `echo_filter(import_path, kwargs_json, trace_json: bytes)` | `[[bool]]`, unchecked | echo: `partial(import_object(import_path), **kwargs)(vf.WireTrace.model_validate_json(trace))`; filter cached per (path, kwargs); shape validation stays with the Rust caller as in `EchoAlgorithm._filter_masks` |

`TaskRecord = {"key": str, "hash": str, "data": object}`:
- `data` = `task.data.model_dump(mode="json")`, shipped verbatim as the run request's `task_data` (dispatcher.py).
- `key` = `task.key`, `hash` = `task.hash` (a taskset may override `key`, so Rust must not recompute it);
  compared against the returned episode's `task.key` / `task.hash`.

Rust side: the trait `prl_types::Bridge` (owner: core) with `TaskRecord` / `LoadedTasks`; implemented by
`prime_rl_orch_rs::bridge::PyBridge` (generic `call_json(method, args)` + one trait method per bridge method).
Tests: `rust/crates/prl-py/tests/bridge.rs` (fake Python bridge object: round-trips, exception -> error with
traceback, i128 flop costs, bytes args).
Eval sources with an unbounded taskset are an error in Python (`EvalEnv.start`); the Rust caller keeps that check.

### W&B (`PyBridge.wandb`)

`PyBridge.wandb` is `prime_rl.monitors.wandb.bridge.WandbBridge(config.monitors.wandb, output_dir, config)` or `None`
when `monitors.wandb` is unset (`output_dir` = `run_dir` for `EvalConfig`, else `output_dir`, as each Python process's
`monitors.setup`). Methods (owner of the Python class: monitors): `init(train_env_names, eval_env_names,
overview_flavor)`, `log(row_json, step | None)` (row may hold `NaN`/`Infinity`), `finish(paused)`.
Rust: `prime_rl_orch_rs::wandb::PyWandb::from_bridge(py, &bridge) -> Option<Box<dyn prl_monitors::WandbBridge>>`,
built in `run` with the GIL held and handed to the orchestrator. prl-monitors calls it from a blocking thread; each
call takes the GIL. Test: `wandb_forwards_calls`.

### NIXL weight receiver (`PyBridge.nixl_receiver()`)

Only built when `weight_broadcast.type == "nixl"`. Returns
`prime_rl.transports.weights.nixl.bridge.NixlReceiverBridge(config.weight_broadcast)` (owner: io): the consumer's
NIXL agent (UCX, `set_ucx_env_defaults(0)`, agent name `orchestrator-<host>-r0`) and ModelExpress session (role
`orchestrator`, worker `orchestrator`). Methods, all blocking: `publish()` (agent metadata -> ModelExpress, after
`/init_broadcaster`), `wait_ready(step, timeout)` (resolve + connect the trainer peer on first use, wait for
`policy:{step:016x}:ready`), `send_complete(step)`. The Rust `prl_io::weights::WeightReceiver` drives the protocol
(markers, `/init_broadcaster` with `session_id`, `/update_weights` with a null `weight_dir`).
Rust: `prime_rl_orch_rs::nixl::PyNixl::from_bridge(py, &bridge) -> Box<dyn prl_io::weights::NixlBridge>`, passed to
`WeightReceiver::new`; calls run on the blocking pool with the GIL. Tests: `drives_python_bridge` (fake object),
`real_nixl_handshake` (`PRL_NIXL_LIVE=1`: real NIXL agents over UCX/TCP, ModelExpress faked),
`tests/unit/transports/test_nixl_receiver_bridge.py`.

Planned / on request: anything else a spec proves cannot be ported.
To add a method: Python method on `PyBridge` (plain args, JSON `bytes` result), trait method in `prl_types::Bridge`
(core), impl in `prl-py/src/bridge.rs`, a row here.
