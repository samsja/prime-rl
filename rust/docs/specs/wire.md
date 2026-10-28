# prl-wire — verifiers v1 serve protocol (client side) and the wire data model

Owner: `wire`. Crate: `rust/crates/prl-wire`.

## 1. Python behavior being ported

Sources: `deps/verifiers/verifiers/v1/serve/{client,delta,encoding,types,server}.py`,
`verifiers/v1/{episode,trace,graph,types,task}.py`, the orchestrator side of
`src/prime_rl/transports/payload.py`, and how `src/prime_rl/orchestrator/envs.py` /
`dispatcher.py` use `EnvClient`.

### 1.1 Transport (client.py, server.py)

- Client socket: ZMQ `DEALER`, `LINGER=0`, `SNDHWM=0`, `RCVHWM=0`, `connect(address)`
  (address like `tcp://127.0.0.1:5000`). The server is a `ROUTER` (`ROUTER_MANDATORY`).
- Request frames (client -> server): `[request_id, method, payload]`
  - `request_id`: `uuid4().hex` as UTF-8 bytes. `method`: `b"health" | b"run" | b"cancel"`.
  - `payload`: `msgpack.packb(request.model_dump(mode="json"), use_bin_type=True)`.
    - health: `{}`
    - run: `{"task_data": dict, "client": ClientConfig, "model": str, "sampling": SamplingConfig}`
    - cancel: `{"request_id": <run request id>}`; sent fire-and-forget with a fresh request id;
      its reply is ignored (no pending future, so the receive loop drops it).
- Server frames (server -> client): `[request_id, kind, data]`, `kind ∈ {b"delta", b"reply"}`.
  - A frame with != 3 parts is logged (`"dropping a malformed env-server frame"`) and skipped.
  - `delta`: routed to the run's delta handler if one is registered; deltas of unknown /
    cancelled requests are dropped silently. A delta that fails to decode, or whose handler
    raises, is logged with the same warning and skipped.
  - `reply`: resolves the pending future of that request id (unknown ids are dropped).
  - Order: per connection, ZMQ preserves frame order, so a run's reply is processed after its
    last delta.
- Reply payload: `msgpack.unpackb(data, raw=False, strict_map_key=False)` validated into
  `BaseResponse{success: bool = True, error: str | None}` (+ `HealthResponse`, `CancelResponse
  {cancelled}`, `RunResponse{head: dict | None, traces: list[TraceSummary{id, nodes, calls}]}`).
  `success == False` -> `RuntimeError(error or "env server request failed")`.
  Server-side errors are formatted `"{ExceptionType}: {message}"`; a server-side abort replies
  `"Cancelled: rollout aborted"`; unknown methods `"unknown method 'x'"`.
- Timeouts: only health uses one (`asyncio.wait_for`); `health(timeout=2.0)` returns False on
  timeout (the pending entry is removed). Runs are untimed.
- `wait_for_server_startup(timeout=120, interval=1)`: loop until deadline: `health(timeout=
  min(interval, 2))`; sleep `interval`; then `TimeoutError("env server at {address} did not
  become healthy in {timeout}s")`. prime-rl calls it with `ENV_SERVER_STARTUP_TIMEOUT = 600`.
- Cancellation: if the awaiting caller of a `run` is cancelled, the pending entry is removed
  and a `cancel` request for the run id is sent (fire-and-forget, errors suppressed). Same if
  the episode fails to validate after the reply landed (server then discards kept arrays).
  `close()` flushes queued cancels before closing the socket.
- `run(client, model, sampling, task_data, on_delta)`: assembles deltas into an
  `EpisodeAssembly`, calls `on_delta(delta)` after applying each, and on reply returns
  `WireEpisode.model_validate(assembly.finish(head, traces))`; `head is None` is an assert.
- Server semantics relevant to the client: one run per request; the server holds kept
  training arrays (payload hook) until the reply is delivered; a cancel within
  `LATE_CANCEL_SECONDS = 10` after delivery still discards them.

### 1.2 Delta stream and assembly (delta.py)

Delta: `{"trace": id, ...}` with optional keys
- `"discard": True` — drop the trace from the assembly (retried attempt).
- `"open"`: header `{version, id, verifiers, task}` (first delta of a trace).
- `"links"`: `{node_index(int key): [ParentLink...]}` appended to earlier nodes'
  `semantic_parents`.
- list fields `nodes, calls, errors, extra_usage, request_rewrites, response_rewrites`:
  items appended. Nodes stream without `routed_experts/sampling_mask/top_logprobs`, with
  per-token streams as `__nd__` dicts (`{"__nd__": True, "dtype", "shape", "data": bin}`).
- `"set"`: whole-value updates of scalar fields `agent, tools, mm_token_type_id_map, rewards,
  metrics, info, root_reply, is_completed, ok, stop_condition, is_timeout, dropped, timing,
  num_input_tokens, num_output_tokens, num_total_tokens`.
- `"pending"`: replaces the preview list; a delta with `nodes` clears it.
Assembly: a new trace dict = `{**open, nodes: [], calls: [], errors: [], extra_usage: [],
request_rewrites: [], response_rewrites: [], pending: []}`; node dicts are copied with their own
`semantic_parents` list. `finish(head, summaries)`: traces in summary order; missing trace ->
`RuntimeError("served episode is missing trace {id}")`; node/call count mismatch ->
`RuntimeError("served trace {id} assembled {n} nodes / {c} calls, server reports {N} / {C}")`;
`pending` stripped; returns `{**head, "traces": [...]}`.

### 1.3 msgpack encoding (encoding.py, types.py)

- Server packs with `use_bin_type=True` and `msgpack_encoder` (Path/UUID -> str, set -> list,
  Enum -> value, datetime -> isoformat, numpy scalar -> item, ndarray/tensor ->
  `{"__torch_tensor__": True, dtype, shape, data}`, pydantic -> model_dump, dataclass -> asdict).
- Node streams use `encode_ndarray` (`__nd__` marker, C-contiguous, `dtype` numpy name such as
  `int32`, `bool`, `float32`, `uint8`; little-endian raw bytes). `decode_ndarray` reverses it.
- Floats are always float64 on the wire; map keys may be ints (`links`).

### 1.4 Data model (episode.py, trace.py, graph.py, types.py)

`WireEpisode = Episode[WireTaskData, State, WireAgentConfig]` with `Trace[WireTaskData, ...]`.
Field order (= dump order; extras after fields, computed fields last):
- Episode: `id, env{id, name}, task{type, data, key, hash}, group{id}?, run?, ok, is_timeout,
  errors, traces` + computed `num_input_tokens, num_output_tokens, num_total_tokens`.
- RunInfo (discriminated by `type`): `TrainRunInfo{type="train", id, name, work}`,
  `EvalRunInfo{type="eval", id, name}`; WorkInfo: `{type="train"|"eval", step, policy:
  PolicySpan{start, end}?}`.
- Trace: `version, id, verifiers{version, commit}, task, agent{config, runtime, name,
  trainable}, tools, nodes, calls, mm_token_type_id_map (str keys when dumped), request_rewrites,
  response_rewrites, rewards{name: Reward{score, weight}|None}, metrics{name: float|None}, info,
  root_reply, extra_usage, is_completed, ok, stop_condition, is_timeout, errors, dropped,
  timing` + computed `num_input_tokens, num_output_tokens, num_total_tokens`.
- MessageNode: `parent, tools (omitted when empty), semantic_parents[ParentLink{node, type}],
  message, sampled, timestamp, token_ids:int32, mask:bool, is_content:bool, logprobs:f32,
  advantages?, reference_logprobs?, trainer_logprobs?, entropies?, loss_weights?{name: f32},
  routed_experts?, sampling_mask?, weight_versions?[{version, start, end}], top_logprobs?`.
- ModelCall: `node, model, sampling, endpoint, finish_reason, usage{prompt_tokens,
  completion_tokens, cached_input_tokens, reasoning_tokens, cost}, time{start, end}, error,
  policy, acp`. Timing: `start, boot, setup, agent{start, end, model{duration},
  harness{duration}}, finalize, scoring`. Error: `type, message, status_code, traceback`.
- Derived logic used by the orchestrator: `Trace.branches` (leaves = nodes that are no node's
  parent, path walked by `parent`; trainable unless the leaf has a `compaction_attempt`
  semantic link and no node has a `compaction` link pointing at it), `Branch.token_ids /
  sampled_mask / logprobs / advantages / reference_logprobs / loss_weights(name) /
  mm_token_type_ids / spread`, `reward` (sum score*weight of non-None), `has_error = not ok`,
  `is_dropped`, `num_turns` (sampled nodes), `num_branches`, `usage` (Usage.aggregate),
  `min_new_input_tokens`, `num_input/output/total_tokens` (trace: per-call increments counted
  once across branches; episode: sums), `is_truncated`, `last_error`, `last_reply`.
- Attributes the orchestrator reads/writes on vf objects (grep over orchestrator/monitors):
  episode `{id, env.name (write), group (write), run (record_run), ok (write), task.key/hash,
  last_error, num_total_tokens, traces, to_record, model_copy}`; trace `{id, agent.name/
  trainable, branches, dropped/is_dropped, errors (append), has_error, info (read/write),
  is_timeout, is_truncated, last_error, nodes, num_output/total_tokens, num_turns, ok (write),
  reward, rewards, metrics, stop_condition, task, timing.*.duration}`; node `{advantages,
  reference_logprobs, loss_weights (write), is_content, mask, message, sampled,
  semantic_parents, token_ids, weight_versions}`; call `{node, usage}`.

### 1.5 Records (`Episode.to_record` / `to_json`), what the file monitor writes

`to_record(float_decimals=4)`: `model_dump(mode="json", exclude=<wire tensors + record streams
of every node>, exclude_none=True)`, then each node record is extended (`dict.update`, so at the
end) with `RECORD_STREAMS = token_ids, mask, is_content, logprobs, advantages,
reference_logprobs, trainer_logprobs, entropies, loss_weights` as numpy arrays; `None` streams
are omitted (exclude_none); float streams in `ROUNDED_STREAMS` are `np.round(float64(x),
decimals)` (`x*10^d`, round-half-even, `/10^d`); `loss_weights` widen to float64 unrounded.
`to_json` = `orjson.dumps(record, OPT_SERIALIZE_NUMPY)`.
pydantic semantics that matter for byte identity (verified):
- dump order: declared fields, then extras (extra="allow" models: WireTaskData, Tool,
  SamplingConfig, WireHarnessConfig), then computed fields;
- `exclude_none` drops `None` model fields and `None` extras, recursively through nested models
  (also inside lists/dicts of models), but never `None` values of plain dicts (`info`,
  `metrics`, a `dict[str, Reward|None]` keeps `None` values);
- orjson float format: shortest round-trip digits, decimal notation for 1e-5 <= |x| < 1e16
  (`100.0`, `0.00001`), else scientific `1e-6`, `1.5e+16`; NaN/inf -> `null`; float32 numpy
  arrays print f32-shortest; string escapes only `"`, `\`, and control chars (`\n \r \t \b \f`,
  others `\u00XX`), no escaping of non-ASCII.

### 1.6 Payload references (payload.py, orchestrator side only)

`PAYLOAD_INFO_KEY = "prime_rl_payload"` (from `transports/batch/types.py`): `trace.info["prime_rl_payload"] =
{"url": str, "branches": {str(branch_index): {"id", "fields", "routed_shape"?,
"routed_dtype"?}}}`. Orchestrator uses `trace_payload_urls(trace)` (`f"{url}/{id}"` per branch)
and `delete_payloads(urls)` (fire-and-forget HTTP `DELETE` from 8 daemon threads, no proxy,
10s timeout, errors ignored). The rest of payload.py (PayloadHook/PayloadServer on the env
server, `resolve_payloads` in the trainer, `pack_token_arrays`) is not orchestrator code and
stays Python. `sample_payload_urls` is over prl-train's `TrainingSample` -> prl-train.

### 1.7 Python tests covering this

- `deps/verifiers/tests/unit/serve/test_client.py` — interleaved runs see their deltas in
  order before the reply; malformed frames skipped (2 warnings); cancelled run stops seeing its
  stream; multi-MB deltas intact; deep backlog doesn't starve the loop; close drains cancels.
- `deps/verifiers/tests/unit/serve/test_delta.py` — assembly of streamed traces, discard,
  pending preview, `finish` refusing a gap.
- `deps/verifiers/tests/unit/test_trace.py` — branches / token accounting.
- `tests/unit/orchestrator/test_episode_records.py` — received episodes write plain records.
- `tests/unit/transports/test_payload.py` — payload hook/assembly (Python-only parts).

### 1.8 Python code outside the orchestrator that depends on what is ported here

None of the ported verifiers code moves: env servers, trainer and the trace viewer keep using
Python verifiers (`monitors/file/traces/live.py` uses `EpisodeAssembly` to read live files).
`transports/payload.py` stays (trainer + env server import it). When `src/prime_rl/orchestrator`
is deleted, `monitors/*` type-only `vf` imports are unaffected by prl-wire.

## 2. Rust design

### 2.1 Modules

- `value`: `Value` — ordered, msgpack-faithful dynamic value: `Nil, Bool, Int(i64), UInt(u64),
  F32(f32), F64(f64), Str, Bin, Array(Vec), Map(IndexMap<String, Value>)` (`type Map`). Integer
  map keys are stringified on decode (only `links` uses them; JSON writes them as strings anyway).
  `impl Serialize/Deserialize`; msgpack encode/decode (`value::{from_msgpack, to_msgpack}`).
- `json`: `to_vec(&Value) -> Vec<u8>`: orjson-byte-compatible writer.
- `ndarray`: `NdArray{dtype, shape, data}` (`__nd__` codec), typed decode to `Vec<i32/bool/f32>`,
  `live_value(&Value)` (= file monitor `_live_value`).
- `schema`: `wire_episode.json` generated from pydantic by `schema/gen_schema.py` and embedded;
  drives the generic JSON-mode dump (`exclude_none`, extras, computed fields, union choice by
  discriminator or key set) for the opaque submodels (agent config, runtime, messages, tools,
  task data...). Regenerate when verifiers changes.
- `model`: typed `Episode`, `Trace`, `MessageNode`, `ModelCall`, `Usage`, `Error`, `Reward`,
  `Timing`, `TimeSpan`, `AgentInfo`, `TraceTask`, `EnvInfo`, `GroupInfo`, `RunInfo`, `WorkInfo`,
  `PolicySpan`, `ParentLink`, `WeightVersionSpan`, `Branch<'a>`; everything the orchestrator does
  not read stays `Value`. `Episode::from_wire(Value)`, `to_record(Option<u32>) -> Value`,
  `to_json(Option<u32>) -> Vec<u8>`, plus the derived logic of 1.4.
- `assembly`: `EpisodeAssembly{apply(&Value), finish(head, &[TraceSummary]) -> Result<Value>}`.
- `protocol`: frames, `TraceSummary`, response decoding, errors.
- `client`: `EnvClient` (tokio actor owning one libzmq DEALER socket via the `zmq` crate with
  vendored libzmq, driven through the socket's notification fd with `AsyncFd`, like pyzmq's
  asyncio integration). The pure-Rust `zeromq` crate was rejected: it cannot queue sends before
  the connection exists, does not reconnect, and panics on peer read errors.
- `sampling`: `train_sampling` / `sampling_config` / `with_cache_salt` — the vf `SamplingConfig`
  dicts of `TrainEnv.__init__` / `Env._sampling`, key order included.
- `json::to_python_json`: Python `json.dumps` (for `task_key` hashes; `TraceTask::content_hash`).
- `payload`: `PAYLOAD_INFO_KEY`, `trace_payload_urls(&Trace)`, `delete_payloads(urls)`.

### 2.2 EnvClient (actor)

```rust
let client = EnvClient::connect("tcp://127.0.0.1:5000").await?;      // Clone handle
client.wait_for_server_startup(Duration::from_secs(600)).await?;
let ok: bool = client.health(Duration::from_secs(2)).await;
let episode: Episode = client.run(&task_data /*Map*/, &client_cfg /*impl Serialize*/,
                                  "model", &sampling /*impl Serialize*/,
                                  Some(Box::new(|delta: &Value| ...))).await?;
```
- One actor task owns the socket: `select!` over the command inbox (send request / forget
  request) and `socket.recv()`. Pending requests: `HashMap<request_id, PendingReq{reply:
  oneshot::Sender<Bytes>, deltas: Option<mpsc::UnboundedSender<Bytes>>}>`.
- Deltas are forwarded raw to the run's own task, which decodes and applies them in order (the
  reactor never decodes), then the reply is sent after them on the same ordered path, so a run's
  reply is processed after its last delta.
- Cancel-on-drop: the run future holds a guard; dropping it before completion (or failing to
  validate the episode) removes the pending entry and sends `cancel{request_id}` with a fresh id.
- Errors: `RunError` (`type_name()` gives the Python exception class: `RuntimeError` for
  `success=false`, assembly gaps and validation failures; `TimeoutError` for startup).

### 2.3 Tests

- `tests/json.rs`: orjson float/string formatting (f64, numpy f32) and msgpack round trip.
- `tests/model.rs`: a rich episode streamed by Python's `DeltaStreamer` (`tests/fixtures/
  gen_episode.py`): assembly == Python's, `to_json(4|None)` and `Trace.to_record(2)` byte-equal,
  branches/token counts equal, same after the orchestrator's mutations, record reload round trip,
  pydantic strictness of record validation.
- `tests/misc.rs`: sampling dicts, `task_key` hashes, Python `json.dumps`.
- `src/client.rs` unit tests (port of `serve/test_client.py`): interleaved runs see their deltas
  in order before the reply, malformed frames skipped; a dropped run stops seeing its stream and
  sends `cancel`; multi-MB deltas intact; failed replies; health timeout; a run assembles the
  fixture stream. `src/payload.rs`: DELETEs reach a server.
- `tests/env_server.rs` (ignored) + `tests/e2e/run_env_server_e2e.py`: against a real
  `uv run env-server` (gsm8k, null harness) with a mock OpenAI endpoint (eval client) or vLLM
  (`--train URL MODEL`: token-in/out, sampling logprobs, payload hook blobs GET 200 then 404 after
  `delete_payloads`); Python re-validates every received episode and its record is byte-equal to
  the Rust one; a dropped run is cancelled server-side.

### 2.4 What I need from siblings

- io: `VfClientConfig` (serde, pydantic field names) — passed to `run` as `impl Serialize`.
- core/config: the vf `SamplingConfig` dict (from `TrainSourceConfig.sampling` with logprobs
  rule of `TrainEnv.__init__`, or eval sampling) and the task data dicts + `Task.key/hash`
  from the Python taskset bridge.
