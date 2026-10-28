# prl-eval — spec (owner: eval)

Ports `src/prime_rl/eval/{runner,eval,online,control,resume}.py` (the `uv run eval` engine and the SFT
online-eval process) onto the Rust core, plus the Rust handoff in `src/prime_rl/entrypoints/eval.py` and the
online-eval module the `sft` launcher spawns. The pydantic configs (`configs/eval.py`) stay the only config API.

## 1. Python behavior inventory

### 1.1 Entrypoint (`entrypoints/eval.py`) — stays Python (launcher), only the run call changes
- `USAGE` text; `-h/--help` or no args prints it and `cli(EvalConfig)` help.
- `expand_shorthands(argv)`: positional `<taskset-id>` and `--env.<path> <value>` fold into one JSON
  `--source '[{"env": {...}}]'` (keys `-` -> `_`, JSON list/object values parsed, rest left to pydantic);
  next to a TOML that defines `[[source]]`, `--env.*` passes through to the shared `env` block; positional +
  source TOML -> `SystemExit("The <taskset-id> shorthand describes a single source and cannot be combined ...")`;
  `-c N` -> `--concurrency.min_inflight N --concurrency.max_inflight N`; a flag without value -> `"{flag} needs a value"`
  (numbers like `-1` are values). Tests: `tests/unit/eval/test_cli.py` (stay Python).
- `prepare_run(config)`, attempt dirs (`PRL_ATTEMPT_CONFIG_DIR/LOG_DIR`), logger to `log_dir/eval.log`,
  `Starting eval`, launch artifacts, `config_dir/eval.json` (indent 2), one env server per source without
  `serve.address` (`env-server @ config_dir/envs/eval/{name}.json`, log `log_dir/envs/eval/{name}.log`), `--dry-run`
  exit, dashboard, `Configs:` / log message lines, SIGTERM -> cleanup, console quiet (`console_level="ERROR"`) while
  running, links banner thread until done, `Eval finished!` success at the end.
- **Change**: `asyncio.run(run_eval(config))` -> `prime_rl_orch_rs.run_eval(config, PyBridge(config), resolved_json)`
  (GIL released for the run; the banner thread keeps running). Rust errors -> `RuntimeError`; the Python
  `clean_exit` behavior (log `Fatal error in run_eval` with traceback, exit 1) is kept by the entrypoint wrapper.

### 1.2 `Eval.run` (`eval.py`)
1. `config.resume` set -> `landed = resume.take_landed(run_dir, config)` **before** monitors setup.
2. `Initializing monitors ({config.monitors})`; `monitors.setup(producer="eval", wandb, prime, file, output_dir=run_dir,
   run_config=config, eval_env_names=[resolved names], overview_flavor="eval")`.
3. `resume.stamp_config(run_dir, dump_resolved_config(config))` -> `run_dir/monitors/file/eval.json`
   (`orjson OPT_INDENT_2`).
4. `runner.setup()`; `fired, restored = eval_source.trigger(0, completed=landed)`; with resume:
   `Resuming from the trace stream: {len(restored)} episodes restored, {owed} rollouts owed`.
5. `runner.start()`, `runner.run_epoch(fired, 0, restored=restored)`, `runner.drain()`.
6. `run_eval`: clean exit -> `monitors.finalize()` (not on a crash); always `runner.stop()`.

### 1.3 `EvalRunner` (`runner.py`)
setup:
- heartbeat (`config.heartbeat`) beaten only by landed (arrived) episodes; no startup ping.
- run id `$PRL_RUN_ID` or uuid4 hex; run name `$PRL_RUN_NAME`.
- `Initializing inference pool (base_url={client.base_url}, model={model})`: `InferenceClient(client, model)`
  (eval client = openai chat completions), `AdminPlane(client)`.
- `Loading eval environment(s)` -> `EvalEnvs(config.source, config.env_addresses, get_config_dir(run_dir)).start()`
  -> `Eval environment(s) ready ({names})`.
- `Waiting for inference pool to be ready` -> `admin_plane.wait_for_ready(model)` -> `Inference pool ready`.
- `EvalSource(envs, intervals=config.intervals if online else None, skip_first_step, is_resumed)`, `EvalSink`,
  `Policy(version=0, model_name=model)`.
- `fallback_cost = max(source.sampling.max_tokens or 0) or 8192`; `ConcurrencyController(config.concurrency, fallback_cost)`.
- `Dispatcher(train=None, eval envs/source, policy_weights_change = online, progress=None,
  initial_max_inflight = concurrency.max_inflight, ceiling = config.concurrency.max_inflight, dispatch_per_minute,
  on_episode_complete = concurrency.record_episode)`; concurrency bound to the dispatcher **without `on_overload`**
  (eval episodes are never cancelled on load; a cut only blocks admission).
- `InferenceMetricsCollector(admin clients, on_load=concurrency.observe)`; `probe()` false and
  `min_inflight != max_inflight` -> `ValueError("No engine metrics at {urls} - adaptive concurrency has no load signal. ...
  pin the concurrency with `-c N` (concurrency.min_inflight = max_inflight = N).")`; false and pinned ->
  `No engine metrics - running with concurrency pinned at {n}`; then the collector starts (metrics rows go to monitors).
- `PeriodicLogger(name="Eval", collect=collect_pipeline_view, interval=log.interval)`.

start: dispatcher task; control server when `config.control` (`GET/PUT /v1/concurrency` on `control.host:port`);
periodic logger.

`run_epoch(fired, step, restored=(), superseding_step=None)`:
- per fired env `monitors.log_eval_plan(env, step, batch_size_for(env))` (= examples * group_size).
- record trigger time per (env, step); `Starting evals in {a, b} at step {step} ({owed} total rollouts[, {n} restored])`
  where owed = sum of rollouts queued for this step in fired envs.
- `dispatcher.switch_mode(PREFER_EVAL, "eval was triggered at step {step}")`.
- `pending` = fired envs with batch size > 0; restored episodes go through `land` first (no arrival stamp, no beat).
- loop while pending: if `superseding_step()` returns a newer step (checked every `POLL_INTERVAL_S = 2` s while
  waiting) -> warning `Checkpoint {n} is ready - cancelling unfinished eval episodes for step {step}` and
  `dispatcher.cancel_eval_step(step)` (once). Items: `GroupCancellation` -> `eval_sink.cancel`; `DispatchFailure` ->
  `eval_sink.fail`; episode -> `stamp_arrival([ep], "eval", work.step)`, heartbeat beat, `land`. A completed batch ->
  `finalize_eval_batch`, env leaves pending. After the loop with a cancellation: warning
  `Cancelled {n} unfinished eval episodes for step {step}; advancing to checkpoint {newer}`.
- `land(ep)`: `monitors.log([ep], step, "eval", "all")`, then `eval_sink.add` (releases payload URLs) -> maybe finalize.

`finalize_eval_batch(batch)`:
- no episodes, failures or cancellations -> warning `Eval @ step={s} env={e}: no attempts returned, skipping log`.
- effective non-empty -> `monitors.log(effective, step, "eval", "effective")` + `log_annotations(stamp_batch(effective, step))`.
- `monitors.log_eval_epoch(env, step, all episodes)`.
- metrics: `episodes.metrics.to_wandb("eval/{env}", "all")` | `effective.metrics.to_wandb("eval/{env}", "effective")` |
  `dispatch_failure_metrics(failures, "eval/{env}/all", total_attempts = eps + failures + cancelled)` |
  (cancelled > 0) `eval/{env}/all/cancelled/count`, `eval/{env}/all/cancelled/mean` | `eval/{env}/policy_version = step` |
  `step`; `monitors.log(metrics, step)`.
- cancelled > 0 -> WARNING `Partially evaluated {env} (Step {s}) | {elapsed:>7} | Reward {eff.reward.mean:.4f} |
  Error {full.has_error.mean:.1%} | Completed {n}/{total} | Cancelled {c}/{total}`; else SUCCESS
  `Evaluated {env} (Step {s}) | {elapsed:>7} | Reward {r:.4f} | Turns {t:.1f} | Branches {b:.1f} | Error {e:.1%} |
  Truncation {x:.1%} | Timeout {y:.1%}` (reward/turns/branches/truncation over effective, error/timeout over all;
  elapsed = format_time since trigger, 0 when unknown).

`collect_pipeline_view` (INFO every `log.interval` s + time-keyed row): body
`{env arrived/expected (p:.1%) | ...  or "Idle"}; {inflight_eval} inflight episodes (cap {max_inflight}, signal {signal})[ - {stage_counts}]`
(progress parts sorted by (env, step, ...)); payload = dispatcher gauges | drained(train_envs=∅, eval_envs) | concurrency gauges.

drain: stop periodic logger (final tick) + inference metrics. stop (always, tolerates partial setup): control server,
periodic logger, metrics collector, dispatcher (cancels inflight), clients.

### 1.4 Online eval (`online.py`, `python -m prime_rl.eval.online @ eval.json`, spawned by the `sft` launcher)
- `main`: proc title `OnlineEvals`, `cli(SFTOnlineEvalConfig)`, attempt dirs under `output_dir`, write
  `config_dir/eval.json`, logger.
- `run`: monitors setup (`producer="online-eval"`, `overview_flavor="sft"`, output_dir), runner setup
  (`skip_first_step`, `is_resumed = resume_step is not None`), `Initializing weight broadcast ({wb})` with
  `weight_broadcast or FileSystemWeightBroadcastConfig()`, receiver on `broadcasts_dir` (admin plane, model),
  `initialize()`, start, `watch()`, drain, `Online evals finished!`.
- `watch`: `receiver.sync_startup(resume_step or 0, timeout=1200)`; policy.version = startup step; fresh run ->
  `maybe_run_eval(0)`; resumed with `retrigger_on_resume` -> `maybe_run_eval(resume_step, force=resume_step >= max_steps)`.
  `Watching {dir} for new weight broadcasts (max_steps={n})`. Loop every 2 s: steps = `step_*` dirs; published =
  `.sender_ready`; newest published; walk `sorted(steps ∪ deleted_due_steps)` above `last_step`:
  vanished due step -> warning `Weight broadcast for eval step {s} was deleted before it could be evaluated
  (broadcast cleaning outpaced the evals process) - skipping its evals`; unpublished step: above newest published ->
  stop the walk (still writing), else warning `Weight broadcast step {s} is not marked published but newer broadcasts
  are - treating it as abandoned and skipping its evals`; published -> `maybe_run_eval(s, reload_weights=True,
  force = s >= max_steps)`. Exit when `last_step >= max_steps`.
- `deleted_due_steps` = multiples of each env interval up to newest published, minus present steps.
- `maybe_run_eval(step, reload, force)`: reload and not published -> warning `No published weight broadcast for step
  {s} ({dir}) - skipping eval`; trigger **before** reload; `last_step = max`; reload: `Updating inference weights to
  broadcast step {s} ({dir})`, `dispatcher.on_version_pending(s)`, `receiver.receive(s)`, policy.version = s,
  `dispatcher.on_new_version(s)`; nothing fired -> return; `run_epoch(fired, s, superseding_step = next published
  step > s if cancel_on_new_checkpoint)`.

### 1.5 Control API (`control.py`)
`GET /v1/concurrency` -> `{"target": int|null, "cap": int, "inflight": int}`; `PUT /v1/concurrency {"target": int>=1|null}`
-> `set_target` then the same body; invalid body (missing `target`, `target < 1`) -> 422.
Tests: `tests/unit/eval/test_control.py` (pinned target above/below band, back to adaptive, invalid bodies, target
holds under overload).

### 1.6 Resume (`resume.py`)
- `stamp_config(run_dir, cfg)`: mkdir `run_dir/monitors/file`, write `eval.json` (orjson indent 2).
- `read_records(stream)`: chunks in number order (plain `NNNNN.jsonl` or sealed `NNNNN.jsonl.zst`), line by line; a
  torn line (JSON error) ends the stream.
- `archives(run_dir)`: `monitors/file.attempt_*` sorted by N.
- `take_landed(run_dir, config)`: directories = archives (+ current if it exists or there are no archives);
  expected = `model_dump(mode="json", exclude={"resume"})`; `skip_checks` -> warning `Skipping resume compatibility
  checks; saved episodes may use different experiment settings.`; per directory (unless skip_checks): missing
  `eval.json` -> `ValueError("--resume: no saved config at {path}. Set --resume.skip-checks to reuse episodes without a
  saved config.")`; top-level keys whose values differ (or missing on one side; `resume` dropped from saved) ->
  `ValueError("--resume: config differs from {path} in [{sorted keys}]. Use the saved config, start a fresh run, or
  set --resume.skip-checks if the changes are intentional.")`. Records -> `WireEpisode.model_validate` (invalid ->
  error, nothing rotated); keep `ok` ones whose record has a `traces` key. Finally the current dir is renamed to
  `file.attempt_{len(archives)+1}`.
- `EvalSource.trigger(0, completed=landed)` (`plan_rollouts`: per task occurrence up to group_size saved episodes by
  content hash, dedup by episode id; each occurrence gets a fresh group id shared by its restored episodes and its
  queued request; only owed rollouts are queued).
Tests: `tests/unit/eval/test_resume.py` (plan, trigger owed rollouts, take_landed across attempts, validation,
config checks, missing saved config, source order/shared defaults).

## 2. Rust design (`rust/crates/prl-eval`)

```rust
pub async fn run_eval(config: prl_config::EvalConfig, resolved: serde_json::Value /* model_dump(mode="json") */,
                      bridge: Arc<dyn Bridge>) -> anyhow::Result<()>;
pub async fn run_online_eval(config: prl_config::SFTOnlineEvalConfig, bridge: Arc<dyn Bridge>) -> anyhow::Result<()>;
pub mod resume;   // stamp_config, read_records, archives, take_landed (pure fs + serde)
pub mod control;  // axum Router over ConcurrencyHandle: GET/PUT /v1/concurrency
pub mod runner;   // EvalRunner: setup/start/run_epoch/finalize/pipeline view/stop
pub mod online;   // OnlineEval watch loop (pure step-walk planner + driver)
```
- The eval driver is one task (the Python `run_epoch` loop): it owns `EvalSink<Arc<Episode>>`, trigger times and the
  heartbeat, and `select!`s over the dispatcher's eval results channel and a 2 s superseding-check interval.
  Everything else is a core/io actor: Dispatcher (eval-only, owns EvalSource), Concurrency actor
  (`cancel_on_overload = false`), InferenceMetricsCollector -> Concurrency, PeriodicLogger("Eval"), Monitors,
  control server (axum, cancelled on stop).
- Online eval: the driver is the single writer of `policy: watch<Policy>` (it is the watcher), calling
  `dispatcher.version_pending/version_applied` around `WeightReceiver::receive`. The watch-loop step walk is a pure
  function (`plan_walk(steps, published, intervals, last_step) -> Vec<WalkAction>`) so its edge cases are unit-tested.
- Lifecycle: the driver plus the actors run under a `CancellationToken` + `JoinSet`; an actor error cancels everything
  and is returned. `monitors.finalize()` only on a clean exit; `stop` order as Python.
- Resume: records are parsed with prl-wire (`Episode` from a record); config compatibility compares top-level keys of
  `resolved` (minus `resume`) against each saved `eval.json` (numbers compared numerically like Python `==`).
  `eval.json` is written from the Python-made resolved dict with an orjson-compatible indent-2 writer.

### Needs from siblings
- core: eval-only Dispatcher (`train: None`, `policy_weights_change`, eval results `mpsc::Sender<DispatchResult<..>>`),
  handle `switch_mode`, `version_pending/applied`, `cancel_eval_step -> usize`, `trigger_eval{step, force, completed}
  -> TriggerOutcome{fired, restored, owed}`, `pipeline_view() -> DispatcherView`, `stop`; Concurrency actor with
  `cancel_on_overload=false` + handle `get/set_target/gauges`; `EvalSink<E>`; `envs` builder; `PeriodicLogger` with a
  collector closure. (agreed)
- train: `EvalEpisodes` (+ `effective`, `metrics.to_wandb(prefix, subset)`, reward/num_turns/num_branches/
  is_truncated/has_error/is_timeout means), `dispatch_failure_metrics`, `stamp_arrival`/`stamp_batch`, payload release.
- io: `InferenceClient`, `AdminPlane` (`wait_for_ready`, `clients`), `InferenceMetricsCollector` (`probe`, `run`),
  `WeightReceiver` (`initialize`, `sync_startup`, `is_published`, `step_dir`, `receive`), `Heartbeat`, logging
  (`success!`, `format_time`), pathing (`all_ckpt_steps`, config dir).
- monitors: setup with `producer` `eval`/`online-eval`, `overview_flavor` `eval`/`sft`, eval env names;
  `log_eval_epoch(env, step, episodes)`; `PrimeEvalMonitor` (on by default for `uv run eval`).
- wire: `Episode` from a trace-stream record (missing `traces` -> [], `null`/`[{}]` -> error).
- config: `EvalConfig`, `SFTOnlineEvalConfig` extraction.
- scaffold: `prime_rl_orch_rs.run_eval / run_online_eval` pyfunctions; `PyBridge` serving split `eval` from
  top-level `config.source`.

### Python side after the switch
- `entrypoints/eval.py` calls Rust; `python -m prime_rl.eval.online` is replaced by an online-eval entry under
  `prime_rl.entrypoints` that parses `SFTOnlineEvalConfig` and calls Rust (sft launcher + sbatch template updated).
- `tests/unit/eval/test_cli.py` stays (Python shorthands). `test_control.py` and `test_resume.py` move to Rust tests;
  `src/prime_rl/eval/` is deleted with the Python orchestrator.

## 3. Rust tests
- control: port of `test_control.py` (axum `oneshot` requests over a real Concurrency actor).
- resume: port of `test_resume.py` take_landed cases with fixtures written by Python (`ChunkedJsonl` plain + sealed zstd
  chunks, saved configs from `EvalConfig.model_dump`), torn last line; trigger/plan cases are core's EvalSource tests.
- online: step-walk planner (published / abandoned / vanished-due / still-writing / max_steps exit).
- runner: finalize metrics + log lines on a fake episode set; epoch with cancellations (Partially evaluated line).
- e2e: `uv run eval` on a small env against local vLLM, compared with the main checkout's outputs.
