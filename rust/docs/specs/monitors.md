# Monitors (prl-monitors) — behavior inventory + Rust design

Owner: `monitors`. Ports `src/prime_rl/monitors/*` as used by the orchestrator
(`producer="orch"`, `prime=PrimeTrainMonitorConfig`), plus the `prime_runs` / `prime_traces`
SDK paths the train monitor drives.

## 1. Python behavior inventory

### 1.1 Facade (`monitors/__init__.py`)
- `setup(wandb, prime, file, output_dir, producer, run_config, train_env_names, eval_env_names)`:
  only rank 0 (`$RANK`, else `$DP_RANK`, else 0) registers; order is **prime, wandb, file**.
  Logs `Initializing monitors (prime, wandb, file)`, then per monitor `Initializing {name} monitor ({config repr})`,
  then debug `Initialized monitors in {format_time}`. A monitor whose `init` raises **crashes the run**.
- Fan-out (`log`, `log_annotations`, `log_eval_plan`, `log_live`, `log_eval_epoch`, `finalize`):
  sequential over monitors in registration order; an exception is logged as
  `Failed to log to {ClassName}: {e}` (`Failed to finalize {ClassName}: {e}`) and swallowed.
- `log(dict, step)` -> `log_metrics`; `log(episodes, step, kind, subset)` -> `log_episodes`.
- Orchestrator call sites (all producer=`orch`):
  - every arriving episode: `log([episode], step, kind, "all")` (train: `progress.step`; eval: eval step)
  - shipped batch: `log(effective, step, "train", "effective")`, `log_annotations(annotations)`, `log(metrics, step)`
  - eval epoch: `log_eval_plan(env, step, expected)` when counted; on completion
    `log(effective, step, "eval", "effective")`, `log_annotations(stamp_batch(...))`, `log(metrics, step)`
  - wall-time rows: inference metrics and periodic logger: `log(metrics, step=None)`
  - dispatcher: `log_live(events)` every 0.5 s (and a final flush on stop)
  - `finalize()` (never `paused=True` from the orchestrator). `log_eval_epoch` is **not** called by the orchestrator.

### 1.2 File monitor (`monitors/file/monitor.py`, `file/traces/*`)
Paths (`output_dir/monitors/file/...`):
- metrics: `monitors/file/{config.path}` (default `metrics.jsonl`; absolute path wins), opened append, line buffered.
- trace stream: `monitors/file/traces/stream/NNNNN.jsonl[.zst]` + index `monitors/file/traces/stream.index.jsonl`.
- annotations: `monitors/file/traces/annotations/{producer or "unknown"}/NNNNN.jsonl[.zst]` + `.../annotations/{producer}.index.jsonl`.
- live: `monitors/file/traces/live/{trace_id}.jsonl`, pending placeholders `monitors/file/traces/live/pending/{dispatch_id}.json`.
- eval plan: `monitors/file/plan.json`.

`init`: `_logged` = number of lines in an existing stream index (relaunch continues numbering).
Log `Logging metrics and episodes to the local filesystem ({output_dir})`.

`log_metrics(metrics, step)`: `sanitize` drops non-finite floats recursively (dotted paths `a.b`, `a[0]`); warns
`Dropping {n} non-finite value(s) from {path}: {first 5 joined ", "}`. Row = `{"step": step, "time": time.time(), **sanitized, "producer": producer}`
written with **stdlib `json.dumps`** (separators `", "`/`": "`, `ensure_ascii=True`, Python float repr e.g. `1e-05`, `1e+16`, ints stay ints) + `\n`.

`log_eval_plan(env, step, expected)`: read `plan.json` (orjson) or `{}`, `plan.setdefault(env, {})[str(step)] = expected`,
write compact orjson to `plan.json.tmp`, atomic rename.

`log_episodes(episodes, step, kind, subset)`: `effective` is a no-op. For `all`: per episode
`record = episode.to_record(float_decimals=config.float_decimals)`; `line = orjson.dumps(record, default=str, OPT_APPEND_NEWLINE|OPT_SERIALIZE_NUMPY)`;
`(chunk, offset) = stream.append(line)`; `_logged += 1`; index line = `orjson.dumps(index_row(_logged, record, chunk, offset))`.
Flush stream then index after the batch.

`ChunkedJsonl(dir, max_bytes, compress)`: on open, numbers = chunk files `NNNNN.jsonl*`; live chunk = last if its plain file exists
else last+1; any other plain chunk is sealed (if compress). `append` rolls when `size>0 && size+len>max_bytes` (seal old chunk in a
background thread, zstd level 3, seekable format with 4 MiB max frame content, written to `.jsonl.zst.tmp` then renamed, then plain unlinked).
`close`: close; empty live chunk is unlinked, else sealed (if compress); join sealing threads.

`index_row` = `summarize_episode` minus `rewards/metrics/timing` plus `chunk`. Key order:
`cost, line, offset, id, kind, trace_ids, env, group, ok, num_errors, reward, advantage, input_tokens, output_tokens, turns, branches,
stop_condition, truncated, timeout, dispatch, arrival, duration, chunk` (see `traces/index.py` for each field; ported verbatim).

`log_annotations(updates)`: empty is a no-op. Per update: `orjson.dumps(update)` into the producer's annotation stream,
index row `{"trace_id", "chunk", "offset", "info": update.info or {}}`.

`log_live(events)`: first call per process `rm -rf live/` (stale previous attempt). `mkdir -p live/pending`. Per event:
- `{"done": tid}`: drop pending appends for that file, unlink `live/{tid}.jsonl`.
- `{"pending": id, "dispatch": d}`: write `orjson(d)` to `pending/{id}.json.tmp`, rename to `pending/{id}.json`.
- `{"dispatched": id}`: unlink `pending/{id}.json`.
- `{"delta": delta, "dispatch": d}`: path `live/{delta.trace}.jsonl`; `delta.discard` -> drop pending appends + unlink.
  Else line = copy of delta; nodes have `routed_experts/sampling_mask/top_logprobs` removed and `__nd__` arrays decoded to lists
  (recursively inside dict values, e.g. `loss_weights`); if `"open" in delta`, `line["dispatch"] = d`.
  `orjson.dumps(line, default=str, OPT_APPEND_NEWLINE|OPT_SERIALIZE_NUMPY|OPT_NON_STR_KEYS)`. Then one append per file of all its lines
  (each line gets an **extra** `\n`: orjson already appends one, Python joins `line + b"\n"` -> blank lines between deltas; readers skip blank lines; kept byte-identical).
- float32 arrays print float32-shortest (orjson numpy), not widened.

`finalize`: close all streams (seal live chunks) and index files; log `Finalized metrics at {path}`.

Helpers used elsewhere: `live.stage(trace)` and `STAGES` (dispatcher progress line, owned by core), `update.make_update`
(orchestrator annotations, owned by train). Both ported here as pure functions.

### 1.3 Prime train monitor (`monitors/prime.py` + `prime_runs` 0.1.4 + `prime_traces`)
`init(config, output_dir)`:
- init kwargs: `$RUN_ID` set -> attach `{"id": RUN_ID}`; else `{name, model=config.model.name, environments=[s.env_id for train sources],
  training=TrainingSpec(max_steps or 0, batch_size, rollouts_per_example=max(group_size), seq_len, wandb_project), config=config.model_dump(exclude_none=True, mode="json")}`.
- mode: `$PRIME_RUNS_MODE` if set, else requires an API key (`$PRIME_API_KEY` or `~/.prime/config.json` api_key) else RuntimeError
  `The prime monitor found no Prime API key. ... or turn the monitor off with --no-monitors.prime`.
- base_url: `$PRIME_API_BASE` stripped of `/` and `/rft`; else SDK config (`$PRIME_API_BASE_URL`|`$PRIME_BASE_URL`|config file|`https://api.primeintellect.ai`, minus `/api/v1`).
  API root: `/api/internal(/rft)` suffix -> internal (`{origin}/api/internal`, extra `x-api-key` header); else `{origin}/api/v1`.
- create (`kind=train`): requires model and team (`team_id` from `$PRIME_TEAM_ID` (empty = none) or config file); `POST /rft/external-runs`
  `{base_model, max_steps, environments:[{"id": env}], team_id, name?, batch_size?, rollouts_per_example?, seq_len?, run_config?, wandb_project?}`,
  not replayable; id = `response.run.id`; url = `{frontend}/dashboard/training/{id}` (frontend: `$PRIME_FRONTEND_URL` | config `frontend_url` | `https://app.primeintellect.ai`).
  ConfigurationError/APIError on create (not attach) -> RuntimeError `Failed to register the Prime run (...). Select a team with `prime switch` ...`.
- disabled mode: run id `disabled-{16 hex}`, no url, no sinks; logs `Prime run disabled`.
- logs `Logging metrics and episodes to Prime run {id} ({url})[ (attached via $RUN_ID)]`.
- links/records: `register_links(output_dir, "prime", 1 if url else 0)`; if url: locked merge of `monitors/prime/run.json` with
  `{"kind": "train", "id", "url"}` and `publish_link(output_dir, "prime", "run", url)` (`monitors/links.json`, flock on `.lock`, pid tmp + rename, indent 2).
`log_metrics`: sanitize + warn `Dropping {n} non-finite metric value(s): ...`; SDK row = finite-cleaned values + `step` (if not None) + `_timestamp`; queued.
`log_episodes`: only subset `all`, non-empty; queued.
`finalize(paused)`: `flush` if paused else `finish` (drain <= 60 s, close workers, RFT finalize `POST /rft/finalize {"run_id", "exit_code": 0}`
(idempotent; APIError -> `PUT /rft/external-runs/{id}/status {"status": "completed"}`); attached runs are finalized the same way on COMPLETED).
SDK atexit: a run never finished reports `crashed` -> `PUT .../status {"status": "failed", "error_message": "crashed: process exited without finishing the run"}`
(attached: left to the launcher).
Upload machinery: two workers (samples, metrics), bounded queue 256, put timeout 5 s then drop (counted, warning), coalescing of queued
batches, per-sink strike policy (record rejection: drop batch; transient: 3 strikes then pause 300 s (train); other: disable).
- metrics sink: `POST /rft/metrics {"run_id", "metrics": row}` per row, idempotent replay.
- traces sink: records = `stamp_run(episode.to_record(), run_id, "train")` (run.id/type on envelope and every trace), JSONL lines (orjson),
  batched at 30 MiB (64 MiB line cap -> TraceTooLarge), `POST {traces_url}/api/v1/traces` multipart: `metadata`
  (`{"schema_version": 1, "context": {"source": "prime-runs", "run_type": "train", "model": ...}}`), `traces` (gzip mtime=0,
  `Content-Encoding: gzip`), headers `Idempotency-Key: sha256:<hex of uncompressed>`, `X-Prime-Line-Format: episode`,
  `Authorization`, `X-Prime-Team-ID`. Retries 429/502/503/504/transport up to 5 attempts (Retry-After capped 60 s; else jittered exp backoff 1..30 s).
  403 `service_not_enabled` before any receipt -> sink retires, legacy fallback takes over.
- retry policy of the platform client: 5 attempts; connect errors & 429 always replayed; ambiguous (502/503/504, read timeouts) only when idempotent.

### 1.4 W&B monitor (`monitors/wandb/*`)
Python only (no Rust SDK). `init` handles `$WANDB_ARGS`, shared mode env vars, retries, overview view, links. `log_metrics`:
`step=None` rows define `{prefix}/*` against `_timestamp` on first sight and log `{**metrics, "_timestamp": now}`; stepped rows log
`{**metrics, "step": step, "_timestamp": now}`. Episodes ignored. `finalize`: `wandb.finish(exit_code=0)`.

### 1.5 Python tests covering this area
- `tests/unit/orchestrator/test_episode_records.py` (file monitor stream/live/annotations from a wire episode; prime traces encoding).
- `tests/unit/test_dashboard_artifacts.py::test_live_trace_removed_between_stat_and_open` (reader side, stays Python).
- `tests/unit/utils/test_links.py` (links record), `tests/unit/train/sft/test_prime_monitor.py` (SFT side, not ported).

### 1.6 Imports from `prime_rl.orchestrator` outside it (to handle when the Python orchestrator is deleted)
None in `prime_rl.monitors`. The reverse direction (orchestrator imports `monitors.file.traces.{live.STAGES, live.stage, update.make_update}`)
moves to Rust. Readers of the outputs (dashboard, trainer annotations, eval) stay Python and keep reading the same files.

## 2. Rust design

### 2.1 Crate layout (`rust/crates/prl-monitors`)
```
src/lib.rs          Monitors actor + handle + Monitor trait + setup
src/pyjson.rs       Python json.dumps-compatible writer (metrics.jsonl), sanitize
src/file/mod.rs     FileMonitor
src/file/chunks.rs  ChunkedJsonl (zstd seekable sealing)
src/file/index.rs   summarize_episode / index_row
src/file/update.rs  make_update / update_index_row
src/file/live.rs    live event writer, stage(), STAGES
src/records.rs      links.json / run.json locked records
src/prime/*         platform client, RFT backend, upload worker, metrics + traces sinks, PrimeTrainMonitor
src/wandb.rs        WandbMonitor over the `WandbBridge` trait
```

### 2.2 Public API
```rust
pub enum Scalar { Int(i64), Float(f64), Bool(bool), Str(String), Null }   // metrics.jsonl keeps ints as ints
pub type Row = IndexMap<String, Scalar>;                                     // + From<prl_types::Metrics>
pub enum LiveEvent { Delta { delta: prl_wire::Value, dispatch: prl_wire::Value }, Done(String),
                     Pending { id: String, dispatch: prl_wire::Value }, Dispatched(String) }
#[derive(Clone)] pub struct Monitors { tx: mpsc::Sender<MonitorMsg> }      // single channel => send order
impl Monitors {
    async fn log_metrics(&self, row: Row, step: Option<u64>);
    async fn log_episodes(&self, episodes: Vec<Arc<prl_wire::Episode>>, step: u64, kind: Kind, subset: Subset);
    async fn log_annotations(&self, updates: Vec<prl_wire::Value>);
    async fn log_live(&self, events: Vec<LiveEvent>);
    async fn log_eval_plan(&self, env_name: String, step: u64, expected: u64);
    async fn finalize(&self, paused: bool);                                  // awaits completion
}
pub struct MonitorsSetup { output_dir, producer: Option<String>, file: Option<FileMonitorConfig>,
                           prime: Option<PrimeTrainSetup>, wandb: Option<Box<dyn WandbBridge>>, ... }
pub async fn spawn(setup: MonitorsSetup, cancel: CancellationToken) -> Result<(Monitors, JoinHandle<Result<()>>)>;
pub trait WandbBridge: Send + 'static {                   // implemented in prl-py over PyBridge (GIL, spawn_blocking)
    fn init(&mut self, train_env_names: &[String], eval_env_names: &[String]) -> Result<()>;
    fn log(&mut self, row_json: &str, step: Option<u64>) -> Result<()>;
    fn finish(&mut self, paused: bool) -> Result<()>;
}
pub mod live { pub const STAGES: [&str; 8]; pub fn stage(trace: &Value) -> &'static str; }
pub mod update { pub fn make_update(trace_id, info, branches) -> Value; }
pub mod annotations { pub fn stamp_arrival(&mut [Episode], Kind, step); pub fn stamp_batch(&[..Episode], step) -> Vec<Value>; }
```
Records come from `prl_wire::Episode::to_record(float_decimals)` / `to_json` (orjson byte-compatible, owned by wire);
index rows, live lines and annotation lines are written with `prl_wire::json::to_vec` (orjson-compatible writer).
`live_value` (ndarray decoding of live node fields) is `prl_wire::ndarray::live_value`.
- `spawn` runs every monitor's `init` in setup order (prime, wandb, file) before returning; an init error is returned (crashes the run).
- The actor is the single ordered consumer: it processes messages one at a time and fans out sequentially in registration order,
  with the same warning on per-monitor failure. Blocking file I/O runs on `spawn_blocking` (state moved in and back, awaited before
  the next message, so writes never interleave). W&B calls run on `spawn_blocking` holding the GIL via the bridge.
- Prime uploads run on their own tokio tasks (samples / metrics workers) fed by bounded queues, like the SDK's uploader threads.
- If the inbox closes or cancellation fires without `Finalize`, the actor finalizes as a crash: file streams are closed, prime reports
  `crashed` (SDK atexit semantics), wandb `finish`.

### 2.3 Python bridge (wandb)
`src/prime_rl/orchestrator/rust_monitors.py: WandbBridge(config)` (owned by monitors; `PyBridge` exposes
`wandb_init(train_env_names_json, eval_env_names_json)`, `wandb_log(row_json, step|None)`, `wandb_finish(paused)` delegating to it).
It wraps the existing `WandbMonitor` (same init path, overview view, links, shared mode); `log` is `json.loads` + `WandbMonitor.log_metrics`.
Only scalar rows cross the boundary (W&B ignores episodes; no tables are logged today).

### 2.4 Needed from siblings
- wire (agreed): `Episode::to_record/to_json`, `prl_wire::Value`, `prl_wire::json::to_vec`, `prl_wire::ndarray::live_value`;
  episode accessors for annotations (trace.info, run.work.step, trace.timing.start, branches).
- core: owns spawning (`prl_monitors::spawn`), sends `MonitorMsg`s; converts `prl_types::Metrics` into `Row` via `From`.
- config: fields for `monitors.{file,prime,wandb}`, `model.name`, `max_steps`, `batch_size`, `seq_len`, per train source `env_id`/`group_size`/`resolved_name`.
- scaffold: `PyBridge` delegates three wandb methods to `rust_monitors.WandbBridge`; prl-py implements `WandbBridge` trait.

### 2.5 Known gaps (planned)
- Legacy samples fallback (parquet to `/rft/samples` when Prime Traces is not enabled for the account): not ported; the traces sink logs
  that traces are not enabled and retires (records are lost for such accounts, as with `traces_opt_out`). `PRIME_TRACES_OPT_OUT=1` errors at init.
- `PrimeEvalMonitor` and `log_eval_epoch`: not used by the orchestrator, not ported.
