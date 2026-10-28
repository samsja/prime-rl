# prl-config: the orchestrator config in Rust

Owner: config. Crate: `rust/crates/prl-config`.

## Approach

Pydantic is the only config schema and does all validation. The `orchestrator` entrypoint
parses `OrchestratorConfig` as today and passes the validated **pydantic object** to Rust
(`prime_rl_orch_rs.run(config, bridge)`). Rust extracts it once at startup:

```rust
let config = prl_config::OrchestratorConfig::extract(&config_obj)?; // GIL held, startup only
```

Extraction is pyo3 `FromPyObject` over attributes. After it, Rust holds plain values and
never touches the GIL to read config.

Rules:

- Declare only fields the orchestrator reads. No full mirror, no defaults, no validation.
- Derived values come from Python (attributes, properties, or method calls on the
  object). None is recomputed in Rust.
- Sub-configs that Rust only forwards are `model_dump(mode="json")`ed into
  `serde_json::Value`.
- `Literal[...]` fields become Rust enums (`str_enum!`), extracted from the `str`.
- Discriminated unions become Rust enums. Extraction reads the discriminator (`type`) and
  then extracts the matching variant struct.
- All fields are `pub` and every type derives `Debug, Clone, PartialEq`. Other crates
  build configs in plain Rust for their unit tests.

How drift is caught: `tests/extract.rs` loads every orchestrator-relevant TOML through
the real pydantic loader and extracts it (`tools/load_configs.py`). It also loads two
variant TOMLs (`tools/extra/*.toml`) that cover the union variants no checked-in config
uses. A renamed or removed field fails `getattr`. An unknown union tag fails the tag match.
Run it with `CARGO_TARGET_DIR=rust/target/<you> cargo test -p prl-config` from `rust/`.
It embeds the project's `.venv` Python, the same way `prl-py` tests do.

## Derived values read off Python

| Rust field | Python source |
|---|---|
| `TrainSourceConfig.resolved_name`, `OnlineEvalSourceConfig.resolved_name` | `EnvConfig.resolved_name` (property) |
| `*.env_id` | `EnvConfig.env_id` (property) |
| `TrainSamplingConfig.truncates_distribution` | `TrainSamplingConfig.truncates_distribution()` |
| `AlgoConfig.action_loss_type` | algorithm `ClassVar` `action_loss_type` |
| `ResumeConfig.dir_step` | `ResumeConfig.dir_step` (read only when `dir` is set) |
| `RLOnlineEvalConfig.intervals` | `ScheduledEvalConfig.intervals` (property) |
| `OrchestratorConfig.env_addresses` | `OrchestratorConfig.env_addresses`: `IndexMap<(Split, String), Option<String>>` (train first, then eval) |
| `OrchestratorConfig.any_policy_sourced` | property |
| `TokenizerConfig.name/trust_remote_code`, `monitors.prime.name`, eval `sampling.top_k = -1`, policy `extra_body.extra_args.prl_replay`, train `top_k` default | already set by the model validators |

The resume step logic (`dir_step` if `dir` else `step`) is orchestrator code. It is a
single line, so core keeps it.

## Forwarded as JSON (`serde_json::Value`, from `model_dump(mode="json")`)

| Field | Why opaque |
|---|---|
| `OrchestratorConfig.renderer` | `renderers.RendererConfig`: about 35 variants on `name`. It goes to the env server inside the policy's `TrainClientConfig` (renderer client). Read `renderer["name"]` if needed. |
| `OPSDAlgoConfig.renderer` | Same type. The OPSD hint renderer runs in Python or the renderer layer. |
| `OnlineEvalSourceConfig.sampling` | `verifiers.v1.SamplingConfig` (`extra="allow"`). Sent to the env server unchanged (`EvalEnv.sampling = config.sampling`). |
| `TrainSamplingConfig.extra_body` | `dict[str, Any]`. |
| `EchoFilterConfig.kwargs` | `dict[str, Any]` passed to a Python callable. |
| `OrchestratorConfig.dump` | The whole config, as the monitors' `run_config` (wandb config, platform). |

Not extracted:

- Source `env`, `select`, `serve`. The taskset is loaded in Python through
  `PyBridge.load_tasks`, which reads the Python config itself. `serve.address` reaches Rust
  through `env_addresses`.
- `env_vars`, group-level defaults (`train.sampling/select/group_size/algo/env`,
  `eval.sampling/...`). They are launcher or validation inputs, already merged into each
  source.
- `model.trust_remote_code`, `model.vlm`. The orchestrator does not read them.
- `write_resolved_config` stays in Python (`prime_rl.orchestrator.rust.run_rust_orchestrator`).

Orchestrator logic over config that is **not** in this crate, so its owner must port or
bridge it:

- The train request sampling. `TrainEnv.__init__` builds
  `vf.SamplingConfig(**sampling.model_dump(exclude_none=True, exclude={"logprobs"}), logprobs=...)`.
  The `normalize` validator merges `extra_body` into top-level keys and renames
  `max_completion_tokens` to `max_tokens`.
- `setup_client` (v1 `TrainClientConfig`/`EvalClientConfig` from a `ClientConfig` plus the
  renderer dump), and `resolve_api_key`/`resolve_headers`. These read env vars at runtime.

## Public API (all re-exported at the crate root)

- `OrchestratorConfig { model, train, tokenizer, renderer: Value, eval: Option<RLOnlineEvalConfig>, log, monitors, collect_inference_metrics, inference_metrics_roles: Option<Vec<InferenceRole>>, ckpt: Option<CheckpointConfig>, resume: Option<ResumeConfig>, weight_broadcast, rollout_transport, env_server_replicas: Option<u64>, output_dir: PathBuf, dispatch_per_minute: Option<u64>, batch_size, constant_trainer_batch_size, concurrency, gc: Option<GCConfig>, seq_len, num_train_workers, pad_to_multiple_of, pad_to_seq_len, max_steps: Option<u64>, max_staleness, heartbeat: Option<HeartbeatConfig>, control: Option<ControlConfig>, env_addresses, any_policy_sourced, dump: Value }`
  and `OrchestratorConfig::extract(&Bound<PyAny>) -> PyResult<Self>`.
- Shared types: `ModelConfig { name, client: PolicyClientConfig }`.
  `PolicyClientConfig { client: ClientConfig, dynamo: Option<DynamoConfig> }` derefs to
  `ClientConfig`.
  `ClientConfig { base_url, api_key_var, headers, headers_from_env: IndexMap<String,String>, timeout: ClientTimeoutConfig, wait_for_ready_timeout, skip_model_check, admin_base_url: Option<Vec<String>> }`.
  `TokenizerConfig`, `LogConfig`, `GCConfig`, `HeartbeatConfig`, `ControlConfig`,
  `CheckpointConfig`, `ResumeConfig { step, dir, dir_step }`.
  `WeightBroadcastConfig::{Filesystem, Nccl, Nixl}` (`.tag()`, `.timeout()`).
  `TransportConfig::{Filesystem, Zmq(ZMQTransportConfig)}`.
- Sources: `TrainConfig { source }`.
  `TrainSourceConfig { resolved_name, env_id, sampling: TrainSamplingConfig, ratio, group_size, algo: AlgoConfig, curriculum: Option<CurriculumConfig> }`.
  `RLOnlineEvalConfig { source: Vec<OnlineEvalSourceConfig>, skip_first_step, retrigger_on_resume, intervals }`.
  `OnlineEvalSourceConfig { resolved_name, env_id, sampling: Value, group_size, interval }`.
  `ConcurrencyConfig`, `Split`, `InferenceRole`.
- Curriculum: `CurriculumConfig { sampler: TaskSamplerConfig, gates: IndexMap<String, AdmissionGateConfig> }`.
  `TaskSamplerConfig::{Standard, DifficultyPool(DifficultyPoolSamplerConfig { pools, seed })}`.
  `AdmissionGateConfig::AdvantageRange(AdvRangeGateConfig)`.
- Algorithms: `AlgoConfig { sampling: AlgoSamplingConfig { source: ModelReference }, action_loss_type: ActionLossType, params: AlgoParams }`.
  `ModelReference::{Policy, Frozen(Box<FrozenModelConfig>)}`. `FrozenModelConfig { client, name }` derefs to `ClientConfig`.
  `AlgoParams::{Grpo(GRPOAlgoConfig), Echo(EchoAlgoConfig), MaxRl, Rae(RAEAlgoConfig), HierarchicalGrpo(..), Opd(OPDAlgoConfig), Opsd(OPSDAlgoConfig), Sft, Debug(DebugAlgoConfig)}` (`.tag()`).
  `LengthPenaltyConfig::{Linear, LinearSubagent}`, `LossAggregation`, `EchoRolesConfig`, `EchoFilterConfig`.
- Monitors: `TrainMonitorsConfig { wandb: Option<WandbMonitorConfig>, file: Option<FileMonitorConfig>, prime: Option<PrimeTrainMonitorConfig> }`.

Need another field? Ask config, or add it yourself: add the field to the struct and a
check in `tests/extract.rs`.

## Eval configs (`prl_config::eval`, for prl-eval)

- `EvalConfig::extract` (`uv run eval`): `{ served: ServedEvalConfig<EvalSourceConfig>, model: String, run: RunConfig{name, dir}, output_dir, run_dir (property), resume: Option<EvalResumeConfig{skip_checks}>, log, monitors: EvalMonitorsConfig, dump }`.
- `SFTOnlineEvalConfig::extract` (the sft online-eval process): `{ served: ServedEvalConfig<OnlineEvalSourceConfig>, intervals (property), skip_first_step, retrigger_on_resume, cancel_on_new_checkpoint, model: Option<String>, weight_broadcast: Option<WeightBroadcastConfig>, broadcasts_dir: Option<PathBuf>, max_steps, resume_step, output_dir, log, monitors, dump }`.
- `ServedEvalConfig<S> { source: Vec<S>, client: ClientConfig, concurrency, dispatch_per_minute, control, heartbeat, env_addresses (property) }`. It is the shared Python base `ServedEvalConfig`.
- `EvalSourceConfig { resolved_name, env_id, sampling: Value, group_size }`.
  `EvalMonitorsConfig { wandb, file, prime: Option<PrimeEvalMonitorConfig{name}> }`.
- Not extracted: `clean`, `dry_run`, `dashboard`, `select`, `group_size` (group level). The
  Python entrypoint handles the first three. The last two are already merged into the sources.
- The test loads every `uv run eval` TOML under configs/examples and each sft config's
  `[eval]` block, plus `tools/extra/eval.toml` and `tools/extra/sft_online_eval.toml`.

## Python tests covering this config

`tests/unit/test_configs.py` (pydantic validation) stays in Python. It is unchanged,
because the validation never moves.

## Imports of `prime_rl.orchestrator` from outside it (flag for deletion at parity)

`eval/runner.py`, `eval/control.py`, `transports/weights/{__init__,filesystem,base}.py`,
`transports/weights/nixl/nixl.py`, `inference/dynamo.py`, `entrypoints/orchestrator.py`,
`entrypoints/env_server.py`.
