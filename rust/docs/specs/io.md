# prl-io spec — inference clients, admin plane, inference metrics, weight receivers, control, heartbeat, checkpoint, gc, logging

Owner: io. Crate: `rust/crates/prl-io`. Python reference (source of truth):
`orchestrator/clients.py`, `inference/dynamo.py` (DynamoAdminPlane, reached via `setup_admin_plane`),
`orchestrator/inference_metrics.py`, `transports/weights/{base,filesystem,nccl,nixl/nixl.py}` (receiver side only),
`control/{controller,app,types,server}.py`, `utils/heartbeat.py`, `orchestrator/ckpt.py`, `orchestrator/gc.py`,
`utils/logger.py` (format + `format_time`), `utils/pathing.py` (ckpt/broadcast dir helpers, `wait_for_path`).

## 1. Python behavior inventory

### 1.1 HTTP clients (`clients.py`)

Admin clients (`setup_admin_clients(cfg)`): one per URL in `cfg.admin_base_url or [cfg.base_url]`.
- base URL = `url.rstrip("/").removesuffix("/v1")`; httpx renders it with a trailing `/` in messages
  (`f"{client.base_url}"` -> `http://h:8000/`).
- headers = `resolve_headers(cfg)` (static `headers`, then `headers_from_env` values that are set, case-insensitive
  merge keeping the first spelling; for `PRIME_API_KEY`-keyed `*.pinference.ai` hosts add `X-Prime-Team-ID` from
  `$PRIME_TEAM_ID` or `~/.prime/config.json:team_id` unless present) + `Authorization: Bearer <key>` when
  `resolve_api_key(cfg) != "EMPTY"` (key = `$<api_key_var>`, falling back to `~/.prime/config.json:api_key` for
  pinference hosts keyed by `PRIME_API_KEY`).
- pool: max 4 connections, 1 keepalive; **no default timeout** (`Timeout(None)`).

`_admin_post(client, path, timeout_s=300, **kw)`: POST with per-attempt timeout connect=10 read=timeout_s
write=60 pool=10, `raise_for_status`. tenacity: retry if HTTP status >= 500 or transport/timeout error; stop after
`2*timeout_s` seconds since start OR 10 attempts; wait_exponential(multiplier=1,min=1,max=10) -> 1,2,4,8,10,10..s;
reraise last error. Constants: `ADMIN_TIMEOUT_S=300`, `UPDATE_WEIGHTS_TIMEOUT_S=720`.

`_post_all(clients, path)`: concurrent `_admin_post`, returns each engine's wall time.

`_pause_engines(clients, step, roles)`: debug `Pausing inference engines to update weights to policy v{step}`;
prefill-role clients first (all concurrently), then the rest; `POST /pause?mode=keep&clear_cache=false`; debug
`All inference engines paused in {min:.2f}-{max:.2f}s`. Returns pause times (prefill first).
`_resume_engines`: `POST /resume` to all, debug `All inference engines resumed`.

`AdminPlane(client_config, roles)`:
- `clients` (engine admin clients, order = GPU rank order = metrics role order), `_router_clients` (only when
  `admin_base_url` is set: admin clients for `base_url`), `update_times: {pause, transfer, resume}: list[float]`.
- `wait_for_ready(model)`: concurrently `check_health(clients, timeout=wait_for_ready_timeout, quiet=bool(router))`
  and `check_health(router_clients, ...)`; then `maybe_check_has_model(clients, model, skip_model_check)`.
- `initialize_nccl(host, port, timeout, inference_world_size)`: info
  `Initializing NCCL broadcast: {n} servers, inference_world_size={w}, gpus_per_server={w//n}`; concurrently plain
  `POST /init_broadcaster` (no retry, no timeout) json `{host, port, rank_offset: i*gpus_per_server,
  inference_world_size, timeout}`; a 404 logs warning `The route /init_broadcaster does not exist. Skipping NCCL
  broadcast initialization.`; other HTTP errors are **swallowed** (only 404 is logged; transport errors propagate).
- `update_weights(weight_dir|None, transport, step, on_paused)`: pause (above); then `on_paused()`; then
  `_post_all(/update_weights, json={"weight_dir": posix|null, "weight_version": step}, timeout_s=720)`; resume in
  `finally`; `update_times` replaced only on success.
- `aclose()`.

`check_health(clients, interval=1, log_interval=30, timeout=3600, quiet=False)` per client: debug `Pinging /health
until the inference server is ready`; loop while `wait_time < timeout`: GET `/health` (no timeout); 404 -> warning
`The route /health does not exist. Skipping health check.` and return; 2xx -> debug `Inference pool is ready after
{wait_time} seconds`; any error/non-2xx -> when `wait_time % log_interval == 0 and wait_time > 0` log (info, or debug
when quiet) `Waiting for inference server at {base_url} to start up ({wait_time}s elapsed)` + debug `Inference server
at {base_url} not reachable: {e!r}`; sleep interval. Exhausted -> error log + `TimeoutError("Inference server is not
ready after {wait_time} (>{timeout}) seconds. Aborting...")`.

`maybe_check_has_model(clients, model, skip)`: GET `/v1/models` on all; body parsed as JSON only if content-type
starts with `application/json`; status != 200 or no `data` -> `RuntimeError("Listing the models of {base_url} failed
with status {code}: {text[:300]}")`; model must equal some `id` or `root` else `ValueError("Model {m} was not found in
the inference pool on {base_url}")`. Debug lines `Checking if model {m} is in the inference pool` / `Model {m} was
found in the inference pool`.

`load_lora_adapter(admin, lora_name, lora_path, step)`: per engine (concurrently) POST `/load_lora_adapter`
json `{lora_name, lora_path}` params `weight_version=step`, per-attempt timeout read=30; retry on 404/500 or
transport errors; stop 120 s or 10 attempts; same exponential wait. Debug `Sending request to load LoRA adapter
{name} from {path}`.

`init_nixl_broadcast(admin, host, port, timeout, world, session_id)`: `_admin_post(/init_broadcaster,
timeout_s=max(300, timeout), json={host, port, rank_offset: i*(world//n), inference_world_size, timeout,
session_id})` on all engines.

`setup_admin_plane(cfg, model, roles)`: `DynamoAdminPlane` when `cfg.dynamo` is set and enabled, else `AdminPlane`.
`check_inference_ready(cfg, model)`: transient AdminPlane `wait_for_ready` + close.
`connect_frozen_client(frozen_cfg, renderer_config)`: info `Initializing frozen model pool (model={name},
base_url={base_url})`; InferenceClient (train type `renderer` iff renderer config given); `check_inference_ready`.

`InferenceClient(cfg, model_name, train_client_type, eval_client_type, renderer_config)`:
- `train_client`/`eval_client` = `setup_client(...)`: vf `TrainClientConfig` (`type="train"`, + `renderer`,
  `renderer_model_name` = model_name iff train type is renderer, `multiplex=256`) for type `renderer`, else
  `EvalClientConfig` (`type="eval"`); both carry the BaseClientConfig fields (base_url, api_key_var, headers,
  headers_from_env, timeout). These are serialized onto the env-server wire (owner: wire).
- `score(token_ids)` = `prefill_logprobs`: POST `{base_url without /v1}/inference/v1/generate` via the OpenAI SDK
  (Authorization `Bearer <key or EMPTY>`, resolved headers, `max_retries=0`, client timeouts from cfg.timeout), body
  `{model, token_ids, sampling_params: {max_tokens:1, temperature:1.0, top_p:1.0, prompt_logprobs:1}}`; returns one
  float per `prompt_logprobs` entry: `None`/empty -> 0.0, else the first value's `logprob` (null -> 0.0).
- `finish_sessions(ids)`: only when `admin_base_url` is set (router admin client); concurrently
  `_admin_post(/finish_session?session_id=..., timeout_s=5)`; failures debug `Failed to release inference session
  {id}: {e!r}`.

Tests: `tests/unit/orchestrator/test_clients.py` (lora retry predicate, lora request shape, init_broadcaster shape,
prefill-before-decode pause order, check_health retries non-2xx, setup_client shapes),
`test_prefill_logprobs.py`.

### 1.2 Dynamo admin plane (`inference/dynamo.py`)

`DynamoAdminPlane(policy_cfg, model)`: discovery URL = `dynamo.discovery_url` or `base_url` with port+1 and empty
path (errors when no port or port 65535). Headers: static + env headers, `Authorization: Bearer $key` replacing any
case-variant when key set and != EMPTY. `wait_for_ready`: check frontend health + model, then poll
`GET {discovery}/v1/rl/workers` (strip trailing `/v1` of path) every 1 s until two consecutive identical
`(instance_id, admin_base_url)` fingerprints, health-check that worker (quiet), bind `clients=[worker]` (timeout
`min(30,max(1,t))`). Snapshot parsing (`protocol_version: 1`, `workers[]` filtered by `model`): pending on
`error`/missing admin URL/no match; error on `world_size != 1`, multiple matches, non-origin URL, host mismatch.
`ensure_topology_current` before every mutation (3 attempts; changed twice -> error). NCCL: one-shot
`init_broadcaster` via `POST /collective_rpc {"method", "timeout", "args", "kwargs": {}}` expecting
`{"results":[null]}`; NCCL update = pause, on_paused, `collective_rpc(update_weights_from_path, [dir])`, resume;
failure -> terminal state. Non-NCCL updates delegate to the base plane. Tests: `tests/unit/inference/test_dynamo.py`.

### 1.3 Inference metrics (`inference_metrics.py`)

Constants: `POLL_INTERVAL=5`, `FETCH_TIMEOUT=5`, prefix `vllm:`, `vllm:cache_config_info`, roles `prefill|decode`,
quantiles `p50/p90/p99`, aggregations `min,max,sum,mean,median`, ratio `prefix_cache_hit_rate` from
`prefix_cache_hits[_total] / prefix_cache_queries[_total]`.

Parsing = `prometheus_client.parser.text_string_to_metric_families` semantics (HELP/TYPE families, allowed sample
names per type, counter family `_total` munging, untyped singletons for out-of-family samples), then per engine
label (`engine`, default `"0"`): skip non-`vllm:` families, types other than counter/gauge/histogram, `_created`
samples and non-finite values; `cache_config_info` labels -> `cache_config`; `num_requests_waiting_by_reason` ->
`num_requests_waiting_reason_{reason}`; gauges/counters summed over extra labels keyed by sample name (prefix
stripped); histograms keyed by family name: `_sum`, `_count`, `_bucket{le}` summed.

Endpoints: `name = f"{role or 'server'}{index}"`, `key = base_url.rstrip('/')`; role count mismatch / unknown
role -> ValueError. `has_pd_roles` iff roles == {prefill, decode}. Init info log `Collecting inference metrics from
server0=http://h:8000, ...`.

Per poll: GET `/metrics` (5 s) on all endpoints (failures debug-logged, skipped); samples sorted by engine label;
none -> return; GET `/v1/models` once per endpoint until a max `max_model_len` is cached; metrics (if
`log_metrics`); load samples; store previous snapshots; `on_load(load_samples)`; `monitors.log(metrics, step=None)`
if non-empty. Poll loop: errors warn `Inference metrics poll failed: {e!r}`, sleep 5 s after each poll. `probe(3, 2s)`
-> bool (warn `Inference metrics probe failed: {e!r}`).

Metric keys (insertion order preserved): `inference/{engine_id}/{name}` for every gauge, counter, histogram
`{h}_sum`/`{h}_count`, and with a previous snapshot (dt>0): `{counter}:rate` (non-negative deltas), `{h}:rate`,
`{h}:mean` (count delta>0), per-engine `prefix_cache_hit_rate`; then per scope (`agg`, plus `decode`, `prefill` when
P/D): `inference/{scope}/{name}/{min|max|sum|mean|median}` (names sorted), `inference/{scope}/{h}/{p50|p90|p99}`
(pooled bucket deltas, Prometheus `histogram_quantile`), `inference/{scope}/prefix_cache_hit_rate/pooled`.
`engine_id = "{endpoint.name}.{engine_label}"`.

`EngineLoadSample` (what ConcurrencyController consumes): `engine_id, role, kv_capacity_tokens` (cache_config
`kv_cache_size_tokens` if digits, else `num_gpu_blocks*block_size`), `max_model_len`, `kv_usage`
(`kv_cache_usage_perc`), `running`, `waiting`, `waiting_capacity` (`num_requests_waiting_reason_capacity` or None),
`preemptions_delta` (max of `num_preemptions[_total]` deltas; 0 on first poll). No Python unit tests; Rust tests
use Python-generated fixtures.

### 1.4 Weight receiver (orchestrator side of `transports/weights`)

Broadcast dir `<output_dir>/broadcasts/step_N` with markers `.sender_ready`, `.receiver_ready`, `.started`,
`.finished`. Receiver: `is_published(step)` = `.sender_ready` exists; `next_version(current)` = max published step
among `step_*` dirs (default current); `wait_published(step, cancelled)` polls 0.1 s; `_ack` touches
`.receiver_ready`; `sync_startup(step, timeout)` = `wait_for(wait_published, timeout)` then `receive`.
- filesystem: ack; `wait_for_path(.finished)` (1 s poll, debug logs every 10 s); `adapter_config.json` present ->
  `load_lora_adapter(model_name, dir, step)`, else `update_weights(dir, "filesystem", step)`.
- nccl: `initialize` = `admin.initialize_nccl(...)`; `receive` = `update_weights(dir, "nccl", step,
  on_paused=ack)`.
- nixl: `initialize` = `init_nixl_broadcast(...)` + ModelExpress publish of the NIXL agent metadata; `receive` = ack,
  lazily resolve the trainer peer through ModelExpress (gRPC) and add it to the local NIXL agent, wait for the NIXL
  notification `policy:{step:016x}:ready`, `update_weights(None, "nixl", step)`, send `policy:{step:016x}:complete`.

### 1.5 Control (`control/`)

`Controller(requires_step=True for the orchestrator)`. State: actions by id, `pending`, `saved_steps`, `step`,
`stop_step`. `submit(type, step)`: validate step presence (`InvalidActionError`), then conflicts (`stop_step` set,
pending exists, step < current step -> `ControlConflictError`); action `{id: uuid4 hex, type: "pause", state:
"pending", step}`; then `on_submit()`. `update(step) -> bool`: record step; apply pending if its step is None or
== step: state accepted, step set, `stop_step=step`, info `Control action {id}: pause at step {step}`.
`get(id)`: unknown -> 404 message; accepted & step in saved_steps -> committed. `mark_saved(step)`.
HTTP (FastAPI, uvicorn on `control.host:port`): `GET /v1/actions/{id}` -> 200 Action | 404 `{"detail": msg}`;
`POST /v1/actions` body `{type: "pause", step?: int}` -> 202 Action | 409/422 `{"detail": msg}`; invalid body ->
422 FastAPI validation error. Orchestrator glue (core): after each ship `shipped_state=(step, snapshot)`,
`apply_control()`: `if controller.update(step): ckpt.save_state(state, step); mark_saved(step); info "Paused at step
{step}, saved its state"`; a submit also triggers `apply_control` on the loop. Tests: `tests/unit/control/test_controller.py`.

### 1.6 Heartbeat (`utils/heartbeat.py`)

`beat()` non-blocking: drop if a ping is pending or last ping < `min_interval` s ago; else GET `url` with 1 s
timeout in background; non-200 -> warning `BetterStack heartbeat failed with status code: {code}`; request error ->
warning `BetterStack heartbeat error: {e}`.

### 1.7 Checkpoint (`orchestrator/ckpt.py`)

Path `<output_dir>/checkpoints/step_N/orchestrator/progress.pt`. `torch.save({"progress": Progress,
"train_source": {"rng": random.getstate(), "envs": {env: {"sampler": {...}, "gates": {name: {}}}}}})` written to
`progress.pt.XXXX.tmp` in the same dir then `os.replace` (tmp removed on failure). Sampler state: standard
`{"cursor": int}`, difficulty pool `{"rng": getstate(), "task_rewards": {key: float}}`. `random.getstate()` =
`(3, (625 ints: 624 MT words + index), None|gauss_next)`. Format: zip (stored, 64-byte aligned) with
`archive/data.pkl` (pickle protocol 2; Progress as `GLOBAL prime_rl.orchestrator.types Progress`, `EMPTY_TUPLE`,
`NEWOBJ`, state dict, `BUILD`), `archive/.format_version` (`1`), `archive/.storage_alignment` (`64`),
`archive/byteorder` (`little`), `archive/version` (`3\n`), `archive/.data/serialization_id`. Load: missing file
-> `FileNotFoundError("Orchestrator checkpoint not found at {file}")`; progress fields copied by name; train source
`load_state_dict` validates field set `{rng, envs}` and env set; info `Resumed curriculum state for env {name}` per
env. Debug lines `Orchestrator checkpoint saved to {path} in {format_time}` / `Loading checkpoint from {file}` /
`Orchestrator checkpoint loaded in {t}`. `resolve_latest_ckpt_step`: warn `No checkpoints found in {dir}. Starting
from scratch.` / info `Found latest checkpoint in {dir}: {step}`.

### 1.8 GC (`orchestrator/gc.py`)

CPython-GC specific: freeze startup heap, disable automatic gen-2, scheduled full collection every
`gc.interval` steps, pause stats -> gauges `gc/collections`, `gc/pause_max`, `gc/pause_total`.

### 1.9 Logging (`utils/logger.py`)

Console line (loguru, always colorized): `\x1b[2mHH:MM:SS\x1b[0m <C>{level:>7}\x1b[0m <C>\x1b[22m{[tag] }{message}\x1b[0m\x1b[0m`
with level colors `<C>`: DEBUG `\x1b[34m\x1b[1m`, INFO `\x1b[1m`, SUCCESS `\x1b[32m\x1b[1m`, WARNING
`\x1b[33m\x1b[1m`, ERROR `\x1b[31m\x1b[1m`; at level DEBUG every line gets suffix `<C>\x1b[22m [file::line]\x1b[0m\x1b[0m`.
JSON mode: `{"timestamp": iso, "level", "message", "module", "function", "line"}` (+`exception`, `tag`, `extra`).
Optional plain file sink `HH:MM:SS {level:>7} {message}`. `format_time`: `<1s` -> `{ms:.0f}ms`, `<60` -> `{s:.1f}s`,
`<3600` -> `{m}m {s}s`, `<86400` -> `{h}h {m}m`, else `{d}d {h}h`.

### 1.10 Deleting `src/prime_rl/orchestrator`: who still imports io-area code

Python importers outside `orchestrator/` and `eval/` (both become Rust), by symbol:

| Importer | Symbols from `orchestrator/clients.py` | Used by |
|---|---|---|
| `transports/weights/__init__.py` | `AdminPlane` | `setup_weight_receiver` signature |
| `transports/weights/base.py` | `AdminPlane` | `WeightReceiver.__init__` type |
| `transports/weights/filesystem.py` | `load_lora_adapter` | `FileSystemWeightReceiver.receive` |
| `transports/weights/nixl/nixl.py` | `init_nixl_broadcast` | `NIXLWeightReceiver.initialize` |
| `inference/dynamo.py` | `ADMIN_TIMEOUT_S`, `UPDATE_WEIGHTS_TIMEOUT_S`, `AdminPlane`, `_admin_post`, `check_health`, `maybe_check_has_model`, `setup_admin_clients` | `DynamoAdminPlane` |
| `tests/unit/orchestrator/test_clients.py`, `test_prefill_logprobs.py` | clients symbols | ported to `prl-io` tests |
| `tests/unit/inference/test_dynamo.py` | `AdminPlane`, `setup_admin_plane` (+ `inference.dynamo`) | ported to `prl_io::dynamo` tests |

Nothing outside imports `inference_metrics.py`, `ckpt.py` or `gc.py`. Every importer above is consumer-side
code that the Rust port replaces; the trainer only needs `setup_weight_sender` and `prune_broadcasts_beyond`
(`trainer/{rl,sft}/train.py`), which reach `clients.py` solely through the receiver imports in
`transports/weights/__init__.py`.

Proposal: nothing moves; delete instead.
1. `transports/weights`: delete the consumer side — `WeightReceiver` (base.py), `FileSystemWeightReceiver`,
   `NCCLWeightReceiver`, `NIXLWeightReceiver`, `setup_weight_receiver`, and the `nixl/__init__.py` export of
   `NIXLWeightReceiver`. Keep the senders, `prune_broadcasts_beyond`, the marker constants, and
   `nixl/bridge.py` (`NixlReceiverBridge`, used by the Rust receiver; imports no orchestrator code). Then
   `transports/weights` imports nothing from `orchestrator/`.
2. `inference/dynamo.py`: its only user is `clients.setup_admin_plane`; delete it with
   `tests/unit/inference/test_dynamo.py` (behavior lives in `prl_io::dynamo`). `DynamoConfig` stays in the configs.
3. Then delete `clients.py`, `inference_metrics.py`, `ckpt.py`, `gc.py` and their unit tests.

Other `prime_rl.orchestrator` imports from outside, not io-owned but in the way of the deletion:
`orchestrator/rust.py` (the PyBridge + `run_rust_orchestrator`, imported by `entrypoints/orchestrator.py`) must move
out (e.g. `prime_rl/rust_bridge.py`); it and `entrypoints/env_server.py` import `intercept_vf_logging` /
`setup_env_server_logging` from `orchestrator/utils.py` (proposal: move both into `utils/logger.py`);
`tests/unit/transports/test_payload.py` (`orchestrator.batch`, `orchestrator.trajectories`),
`tests/unit/eval/test_control.py` (`orchestrator.concurrency`), `tests/unit/eval/test_resume.py`
(`orchestrator.eval_source`). `control/`, `utils/heartbeat.py`, `utils/logger.py` stay (the trainer uses them).

## 2. Rust design (`prl_io`) — as implemented

All network calls are `reqwest` (rustls); waits are tokio timers; everything is cancel-safe by drop.
Configs are `prl_config` types (pyo3-extracted pydantic objects).

```rust
// logging
pub fn setup_logger(level: &str, json_logging: bool, log_file: Option<&Path>, console_level: Option<&str>) -> Result<()>;
macro success!(...)                       // loguru SUCCESS = info event with `success = true`
pub fn format_time(seconds: f64) -> String;
// pathing
ckpt_dir, broadcast_dir, step_path, all_ckpt_steps(dir) -> io::Result<Vec<u64>>, resolve_latest_ckpt_step, wait_for_path
// http
resolve_api_key(var, base_url), resolve_headers(headers, headers_from_env, var, base_url)
pub struct AdminClient;                    // Clone; base_url() renders like httpx, key() = rstrip("/")
setup_admin_clients(&ClientConfig) -> Vec<AdminClient>; router_admin_client(&ClientConfig)
pub struct RetryPolicy { max_delay, max_attempts }   // tenacity stop/wait semantics
admin_post(client, path, &AdminRequest, timeout_s); post_all(clients, path, &AdminRequest, timeout_s) -> Vec<f64>
ADMIN_TIMEOUT_S = 300, UPDATE_WEIGHTS_TIMEOUT_S = 720
// clients
pub enum ClientType { Renderer, ChatCompletions }
pub struct VfClientConfig;                 // Serialize == pydantic model_dump(mode="json") (fixture-tested)
setup_client(&ClientConfig, ClientType, renderer: Option<&Value>, renderer_model_name: Option<&str>) -> VfClientConfig
pub struct InferenceClient { pub train_client, pub eval_client, pub model_name, .. }
  ::new(&ClientConfig, model, train: ClientType, eval: ClientType, renderer: Option<&Value>)
  .score(&[i32]) -> Vec<f64>; .finish_sessions(&[String]); impl prl_types::Scorer
connect_frozen_client(&FrozenModelConfig, renderer: Option<&Value>) -> InferenceClient
check_health(clients, interval, log_interval, timeout, quiet); maybe_check_has_model; check_inference_ready
pub enum AdminPlane { Standard(StandardAdminPlane), Dynamo(DynamoAdminPlane) }
  ::standard(&ClientConfig, roles: Option<Vec<InferenceRole>>); ::setup(&PolicyClientConfig, model, roles)
  .clients() -> &[AdminClient]; .wait_for_ready(model); .initialize_nccl(host, port, timeout, world)
  .update_weights(dir: Option<&Path>, Transport, step, on_paused: FnOnce() -> Result<()>) -> UpdateTimes
pub struct UpdateTimes { pause, transfer, resume: Vec<f64> }
pause_engines, resume_engines, update_engines, load_lora_adapter(clients, name, path, step), init_nixl_broadcast(...)
// dynamo
DynamoAdminPlane, parse_dynamo_worker, resolve_dynamo_discovery_url, DiscoveryError::{Pending, Retryable, Fatal}
// inference_metrics
parse_metric_families (prometheus_client parser semantics), parse_prometheus_text -> Vec<(engine_label, EngineSnapshot)>
engine_values, bucket_deltas, build_scope_metrics, histogram_quantile, py_sum (3.12 compensated), py_mean, py_median
pub struct EngineLoadSample { engine_id, role: Option<InferenceRole>, kv_capacity_tokens, max_model_len, kv_usage,
                              running, waiting, waiting_capacity, preemptions_delta }
pub struct InferenceMetricsCollector
  ::new(clients, roles: Option<&[InferenceRole]>, log_metrics, load_tx: Option<mpsc::Sender<Vec<EngineLoadSample>>>,
        metrics_tx: Option<mpsc::Sender<Metrics>>)
  .probe(attempts, interval) -> bool; .run(cancel)  (actor: poll every 5 s); .collect_and_log(); .observe(samples)
// weights
pub struct WeightReceiver ::new(broadcast_dir, WeightBroadcastConfig, AdminPlane, model, nixl: Option<Box<dyn NixlBridge>>)
  .initialize(); .step_dir(step); .is_published(step); .next_version(cur) -> Result<u64>;
  .wait_published(step, &CancellationToken); .receive(step) -> UpdateTimes (of the last full update);
  .sync_startup(step, Duration) -> UpdateTimes; .update_times(); .admin()
#[async_trait] pub trait NixlBridge { publish(); wait_ready(step, timeout_s); send_complete(step) }   // Python side
// control
pub struct Controller (Clone, shared state) ::new(requires_step); submit/get/update/mark_saved; submitted() -> Arc<Notify>
serve(controller, host, port, cancel); serve_on(controller, listener, cancel); router(controller)
// heartbeat
Heartbeat::new(url, min_interval_s).beat()
// ckpt
CheckpointManager::new(output_dir) .ckpt_path(step) .save(&OrchestratorState, step) .load(step, path)
OrchestratorState { progress, train_source: TrainSourceState { rng: PyRandomState, envs: IndexMap<String, CurriculumState> } }
CurriculumState { sampler: SamplerState::{Standard{cursor}, DifficultyPool{rng: Box<PyRandomState>, task_rewards}}, gates }
TrainSourceState::validate_envs(&[String]); PyValue + pickle::{dumps, loads}; torchzip::{write, read}
PyRandom::{new(seed), new_from_entropy(), from_state, state, random, choices_index(weights)}
// gc
GarbageCollector::new(interval).gauges()
```

Tests (`cargo test -p prl-io`; `PRL_IO_PYTHON=1` additionally has Python `torch.load` a Rust checkpoint):
unit tests per module (controller cases from `tests/unit/control/test_controller.py`, dynamo parsing/derivation
from `tests/unit/inference/test_dynamo.py`, loguru byte-exact lines, `format_time`, pickle opcodes, tenacity waits);
`tests/ckpt_compat.rs` (loads a Python-written checkpoint, round-trips, CPython RNG sequences/states/choices);
`tests/inference_metrics_compat.rs` (two polls of a P/D deployment: every metric key in order + value, load samples);
`tests/clients_http.rs` (mock vLLM: LoRA shape + 404 retry, 5xx-only retry, init_broadcaster shape, prefill-before-
decode pause order, health retries/timeout message, model check, update flow incl. resume after failure, prefill
logprobs, setup_client vs pydantic dumps, filesystem receiver handshake); `tests/control_http.rs` (routes, codes,
payloads). Fixtures: `uv run --no-sync python rust/crates/prl-io/tools/gen_fixtures.py`.

## 3. Gaps / deliberate differences

- GC: Rust has no tracing GC. `GarbageCollector` keeps the API (`start`, `run(step)`, `gauges()`) and emits the same
  gauge keys with zeros; no log lines (they would be false). Memory return to the OS is the train crate's
  `release_and_trim` concern.
- NIXL: the orchestrator side needs the NIXL C++ agent (UCX) and the ModelExpress gRPC client. The Rust receiver
  keeps the protocol (ack, init_broadcaster with session_id, ready/complete notifications) and delegates the agent +
  ModelExpress calls to `NixlBridge`, implemented by `prime_rl_orch_rs::nixl::PyNixl` over
  `transports/weights/nixl/bridge.py::NixlReceiverBridge` (see boundary.md). Verified: fake-agent tests, and real
  NIXL agents exchanging the ready/complete notifications over UCX/TCP (`PRL_NIXL_LIVE=1`). Not verified here: a real
  ModelExpress server and an actual RDMA weight pull by vLLM workers (needs InfiniBand).
- Per-attempt httpx timeouts (connect/read/write/pool) map to reqwest `connect_timeout` (10 s on admin clients,
  also for the otherwise unbounded `/health` polls) + per-request total timeout = read timeout.
- `statistics.mean` is computed as `fsum(values) / n` (Python divides the exact fraction): may differ in the last ulp.
- JSON log lines have an empty `function` (tracing has no function names).
- The pickle reader copies memoized containers (no shared references); orchestrator checkpoints never share them.
- `{e!r}` in log lines renders Rust error Display instead of Python reprs.
