# prl-core / prl-types — spec (owner: core)

Ports: `orchestrator.py` (supervisor, lifecycle, step loop, ship flow, drain, eval trigger, control, shutdown),
`dispatcher.py`, `concurrency.py`, `train_source.py`, `eval_source.py`, `curriculum/*`, `watcher.py`, `eval_sink.py`,
`periodic_logger.py`, `types.py`, `live.py`, `envs.py`, the orchestrator-only helpers of `utils.py`, and
`utils/async_utils.py::EventLoopLagMonitor`. Staleness/metrics helpers of `utils.py` are ported by train (see train.md).

## 1. Python behavior inventory

### 1.1 Supervisor (`Orchestrator`)
Setup order (each line is a log line; `success` = loguru SUCCESS):
1. (Python side, before Rust) write resolved config, setup logger, `Starting orchestrator`.
2. `Initializing tokenizer ({tokenizer})` — the Rust port needs no tokenizer (train confirmed); skipped.
3. `Initializing policy inference pool ({model})`: `InferenceClient(model.client, name, train=renderer, eval=openai_chat_completions)`, `AdminPlane(roles=inference_metrics_roles)`.
4. `monitors.setup(producer="orch", wandb, prime, file, output_dir, run_config, train_env_names, eval_env_names)`.
5. run id = `$PRL_RUN_ID` or uuid4 hex; run name = `$PRL_RUN_NAME` (also base sandbox labels — env servers read the env var themselves; in-process sandbox labels are irrelevant to Rust).
6. Heartbeat if configured. TrainEnvs/EvalEnvs built (log `Initializing {algo.type} algorithm for {env}` per train env).
7. Resume step: `resume.dir` -> `resume.dir_step`; else `resume.step` or latest ckpt step in ckpt dir. `progress.step = resume_step+1`, log `Resuming from step {n}` / `Starting from scratch`.
8. `Initializing micro batch sender ({rollout_transport})`; sender at `progress.step`.
9. `Loading train environments (a, b)` -> envs connect (wait for address files `config_dir/.../env address file`, poll 0.5 s, 600 s timeout, then `wait_for_server_startup(600)`), load taskset via bridge; per env `Env {name} ready in {t} (num_tasks={n|infinite})`; success `Train environments ready in {t}`. Same for eval (`Eval environments ready`); eval env with unbounded taskset -> error `Eval env {name} has an infinite taskset — set select.limit to bound it`.
10. TrainSource built; on resume, ckpt load (path override `resume.dir/orchestrator`), `Resumed curriculum state for env {name}` per env.
11. `Waiting for policy inference pool to be ready` -> `admin_plane.wait_for_ready` -> success `Policy inference pool ready after {t}`; `env.setup()` for all train envs (frozen sampling source + algorithm pools).
12. `Initializing weight broadcast ({weight_broadcast})`; `receiver.initialize()`.
13. EvalSource(intervals, skip_first_step, is_resumed). ConcurrencyController, Dispatcher, InferenceMetricsCollector (start + one `probe()`), TrainSink, EvalSink, WeightWatcher (observer: dispatcher; hooks: trigger_eval, on_policy_update), lag monitor, PeriodicLogger("Pipeline", log.interval).
14. `Syncing inference to the trainer's startup broadcast (v{sync})` -> `watcher.sync_startup(sync, weight_broadcast.timeout)` (sync = resume_step or 0); this runs the update hooks, so the startup eval trigger fires here.

Start: `Starting orchestrator loop (max_steps={n|infinite})`; spawn lag monitor, gc, periodic logger, dispatcher, watcher, control (`Controller(requires_step=True)` + ControlServer + serve_control task). A failing component (dispatcher/watcher/control) logs `Orchestrator component {name} failed` (with exception) and cancels the step loop; the error propagates. `last_batch_at` anchored at loop start.

Teardown (`finally`):
- If the step loop finished (`loop_done`) and `ckpt` configured and `progress.step > 1`: `progress.step -= 1`, `Saving final checkpoint at step {n}`, save ckpt.
- clean exit (loop done + final broadcast waited): success `Orchestrator step loop done in {t}`, stop inference metrics + periodic logger, `monitors.finalize()`. Else warning `Orchestrator interrupted after {t} — forcing cleanup (not a clean exit)`.
- `stop()`: `Stopping orchestrator components`, teardown under `SHUTDOWN_TIMEOUT_S = 300`: sender.close, await release, dispatcher.stop (cancels all inflight, records cancellations, flushes live), watcher.stop (waits for an in-flight apply), periodic logger, lag task, component tasks, inference metrics, clients/admin aclose, env clients aclose. Timeout -> warning `Orchestrator shutdown did not complete within 300s; forcing process exit. Training artifacts are already persisted.` and `os._exit(0|1)`.
- success `Orchestrator finished` / warning `Orchestrator cleanup complete (forced)`; malloc_trim.

Step loop (`main_loop`): until stopped: raise if a component stopped; if `draining && dispatcher.is_idle` -> `Pipeline drained, exiting main loop`, stop. Else take one item from `dispatcher.out_q` (0.5 s timeout):
- `GroupCancellation` (train only) -> `train_sink.cancel` -> `handle_train_result`.
- `DispatchFailure` eval -> `eval_sink.fail` -> maybe `finalize_eval_batch`; train -> `train_sink.fail` -> `handle_train_result`.
- episode: must have `run` (`Dispatched episode is missing run identity`) of type TrainRunInfo; `step = work.step` (eval) or `progress.step` (train); `stamp_arrival`; **`monitors.log([ep], step, kind, "all")` before the sink**; eval -> `eval_sink.add` -> maybe finalize; train -> `train_sink.add` -> `handle_train_result`.

`handle_train_result(batch)`: heartbeat beat; while batch and not draining/stopped: `finalize_train_batch(batch)`, then `batch = train_sink.take_batch()`. If draining/stopped afterwards: leftover batch -> count `rollout/queued/cancelled += num_traces`, release its samples; `train_sink.discard_queued()`.

`finalize_train_batch(batch)` (step = progress.step):
1. `max_steps` and `step > max_steps` -> count queued/cancelled, release samples, `start_draining("Step {step} exceeds max_steps={n}")`, return.
2. empty samples -> warning `Step {step}: skipping empty train batch after zero-advantage pruning`, return (no step advance).
3. `step_time = now - last_batch_at`; `last_batch_at = now`.
4. trainable share: warning `Only {k}/{n} effective traces are trainable ({p:.1%}) — consider reviewing task difficulty` when `k/n <= 0.1`.
5. `rollout_metrics = take_rollout_metrics()` (below; mutates progress totals before the ckpt).
6. `monitors.log(effective.vf_episodes, step, "train", "effective")`, then `monitors.log_annotations(stamp_batch(effective, step))`.
7. pack (thread) -> `pack_time`; `send_batch_time = sender.send(grid)`; `wait_for_policy = elapsed - send_batch_time`.
8. `progress.step += 1`; `update_dispatch_gate()`; if controller: `shipped_state = (step, snapshot(progress, train_source))`; `apply_control()`.
9. `save_ckpt_time = maybe_save_ckpt(step)`: only if `ckpt.interval`, not the final step (`step >= max_steps` skipped — saved in teardown), and `step % interval == 0`; log `Saving checkpoint at step {step}`.
10. metrics = rollout_metrics | shipped_metrics | train_source.metrics(), plus `time/wait_for_policy`; `monitors.log(metrics, step)`. shipped_metrics: `effective.metrics.to_wandb("train/agg","effective")`, per env `train/{env}`, `progress/input_tokens`, `progress/output_tokens`, `time/step`, `time/pack`, `time/save_ckpt`, `time/send_batch`, `step`; staleness of effective episodes with a policy span: `staleness/rollout/{mean,max}`, `staleness/rollout/in_flight/{mean,max}`, `staleness/rollout/in_queue/{mean,max}`, `staleness/token/mean` (if tokens); `batch/{env}` = env share.
11. warning `Orchestrator waited {t} for policy updates, at least as long as its {a} active step time. Train-inference compute is imbalanced; add more trainer nodes.` when `step_time > 0 and wait >= max(step_time - wait, 0)`.
12. heartbeat beat; `log_train_batch` success line: `Step {step} | {time:>7} | Reward {r:.4f} | Trainable {k}/{n} ({p:.1%}) | Turns {t:.1f} | Branches {b:.1f} | Max Staleness {s} | Truncation {x:.1%} | Timeout {y:.1%}`; with >1 train env, extra lines joined by `"\n\t\t "`: `╰─ {env:<w} | Ratio {p:.1%} | Reward … | Turns … | Branches … | Max Staleness … | Truncation … | Timeout …` (envs sorted).
13. final step (`step >= max_steps`): `wait_for_version(step, "before shutdown")` then `start_draining("Shipped the final batch")`.
14. await previous release; `gc.run(step)`; schedule release of this batch's buffers.

`take_rollout_metrics()`: window = `train_sink.take_rollout_window(force=True)`; none -> `{}`. Else progress `rollout_window += 1`, `total_tokens += tokens`, `total_samples += traces`, `total_problems += groups`; metrics = `{staleness/rollout/dropped: 0, rollout/queued/pruned: 0, rollout/queued/cancelled: 0} | window.metrics.to_dict() | {rollout/attempts, rollout/discarded, rollout/stale, rollout/errored, rollout/cancelled, rollout/no_signal = discarded-stale-errored-cancelled, progress/tokens, progress/rollouts, progress/tasks, progress/total_tokens, progress/total_rollouts, progress/total_tasks}`. Warning when `discarded/attempts > 0.5`: `Discarded {d}/{a} episodes ({p:.1%}): stale={s}, errored={e}, cancelled={c}, no_signal={n}. Review max_staleness, episode errors, and reward signal.`

`start_draining(reason)`: `draining = True`; `train_sink.discard_queued()`; `dispatcher.disable_train_scheduling()`; `cancellations = dispatcher.cancel_inflight_train_episodes()` (returned directly, not via the queue) -> each through `train_sink.cancel` + `handle_train_result`; log `{reason} — draining pipeline (cancelled {n} outstanding train episode(s); any in-flight evals will complete)`.

`wait_for_version(v, reason)`: return if `policy.version >= v`; log `Waiting for inference to apply policy v{v} {reason}`; wait (failing if a component died) up to `weight_broadcast.timeout`; timeout -> warning `Inference did not apply policy v{v} within {t}s — proceeding anyway`. `wait_for_final_broadcast` = `wait_for_version(max_steps, "before shutdown")` when max_steps set.

`update_dispatch_gate()` (called after ship and after every applied policy): `lead = (progress.step - 1) - policy.version`; `lead > TARGET_LAG(=1)` -> close gate, logging once on transition `Pausing dispatcher until inference applies policy v{progress.step-1-1} (currently v{policy.version})`; else open, logging on transition `Resuming dispatcher (policy v{policy.version})`.

`trigger_eval(step)` (watcher hook, after every applied policy incl. startup sync): no eval source or step already triggered -> return; `resume_step == step and not eval.retrigger_on_resume` -> return; `is_final = max_steps and step >= max_steps`; `fired = eval_source.trigger(step, force=is_final)`; none -> return. Record step; `dispatcher.switch_mode(PREFER_EVAL, "eval was triggered at step {step}")`; `eval_triggered_at[(env, step)] = now`; census `group_size * len(examples)` per env -> `monitors.log_eval_plan(env, step, expected)`; log `Starting evals in {a, b} ({total} total rollouts)`.

`finalize_eval_batch(batch)`: no episodes and no failures -> warning `Eval @ step={s} env={e}: no attempts returned, skipping log`. Effective (non-errored) non-empty -> `monitors.log(effective.vf_episodes, step, "eval", "effective")` + `log_annotations(stamp_batch(...))`. Every episode needs `eval_work(ep).policy` (`Eval {env} step {s} is missing policy provenance`); `policy_version = min(span.start ∪ failure.policy_version)`. Metrics: `to_wandb("eval/{env}", "all")`, `to_wandb("eval/{env}", "effective")`, `dispatch_failure_metrics(failures, "eval/{env}/all", total_attempts=len(eps)+len(failures))`, `eval/{env}/policy_version`, `step`; `monitors.log(metrics, step)`. Success: `Evaluated {env} | Policy v{v} | {elapsed:>7} | Reward {r:.4f} | Turns {t:.1f} | Branches {b:.1f} | Error {e:.1%} | Truncation {x:.1%} | Timeout {y:.1%}` (reward/turns/branches/truncation over effective; error/timeout over all).

Control: `apply_control()`: with `shipped_state=(step, state)`, `controller.update(step)` true -> `ckpt.save_state(state, step)`, `controller.mark_saved(step)`, log `Paused at step {step}, saved its state`. Also applied whenever a submission arrives (`serve_control`), including while a send blocks.

Pipeline view (`collect_pipeline_view`, every `log.interval` s, info line + time-keyed metrics row `step=None`):
body `Train batch {b}/{target} ({p:.1%})[ (env=N, …) if >1 train env][ (+{buffered} buffered)]` + per eval epoch ` | {env} {arrived}/{expected} ({p:.1%})` + `; {total} inflight episodes (train={t}, eval={e}[ | env=N, … if >1 train or >1 eval env])`.
payload: dispatcher gauges (`dispatcher/inflight/{train,eval}`, `dispatcher/queued/eval`, `dispatcher/mode`), drained counters (`dispatcher/{cancelled,errored}/{train,eval}`, `dispatcher/{cancelled,errored}/{env}` for every env, cleared per tick), watcher gauges (`watcher/policy_version`, `watcher/update_count`, `watcher/last_update_weights_time`, `watcher/last_wait_for_ckpt_time`, `watcher/{pause,transfer,resume}/{min,mean,max}`), concurrency gauges (`concurrency/{max_inflight,turnover,capacity,signal,growth_multiplier}`), gc gauges, `event_loop_lag/{min,mean,median,p90,p99,max,n}` when n > 0.

### 1.2 Dispatcher
State: inflight episodes (kind, env, group id, task, policy_version, step, client config, started_at monotonic, dispatch_id uuid hex, live traces), groups (`GroupState`: kind, env, task, step, episodes_to_schedule, target_episodes, emitted, policy_version_at_start, group_id), `max_inflight` (dynamic), `current_inflight` permits, rate limiter (`dispatch_per_minute` tokens / 60 s, aiolimiter leaky bucket), admission window (5 s, burst cap `max(min_burst, max_inflight // 10)`, `min_burst = max(train group sizes, default 8)`; natural completions refund one admission), mode (PREFER_TRAIN start), `train_scheduling_disabled`, `dispatch_allowed` gate, `policy_update_pending`, cache salt prefix (uuid hex[:8]), drain-on-read metrics.
- `fill_inflight`: loop: stop if `policy_update_pending`, no permits, or admission budget <= 0. PREFER_EVAL: no eval work (queue empty and no eval group with episodes_to_schedule > 0) -> switch to PREFER_TRAIN (`Switching dispatcher mode to prefer train episodes because the eval queue drained`), continue; else schedule eval (eval ignores the gate). PREFER_TRAIN: gate closed -> stop; schedule train. Scheduling nothing -> stop. Note: in PREFER_TRAIN eval groups are never scheduled; after the flip, leftover eval work cannot exist.
- `try_schedule(kind)`: train disabled -> false; continue the first existing group of that kind (dict order) with episodes left; else open a fresh group from the source (train: `train_source.next_task(step=progress.step)`; eval: `eval_source.next_task()`), group id = request.group_id or uuid4, rollouts = request.rollouts or env group size, `policy_version_at_start = policy.version`.
- `schedule_group_episode`: eval -> policy clients `eval_client`, model `policy.model_name`; train -> env clients `train_client` (`uses_live_policy` -> `policy.model_name`, else frozen client's model name). `cache_salt = "{prefix}-{policy_version_at_start}"` when live-sourced and weights change, else None. `episodes_to_schedule -= 1`, acquire (rate limiter, then permit), `admissions_in_window += 1`; run `env.run(client, model, cache_salt, task_data, on_delta)` on the least-loaded replica; queue live `pending` event. On completion/cancel: `clients.finish_sessions(sorted(session ids from deltas + episode traces))` (shielded); a returned-but-cancelled episode gets its payloads released.
- `on_delta`: add trace id to session ids; `live.apply`; queue `{"delta": d, "dispatch": dispatch_info}`; on the first live trace append `{"dispatched": id}`. Live buffer cap 20 000: drop oldest deltas down to cap/2 (warn once `Live trace buffer over 20000 events - dropping the oldest deltas`). Flushed every 0.5 s via `monitors.log_live`; final flush at stop.
- completion (`handle_completed_request`): already claimed -> ignore. Retire (live: `dispatched` if no trace streamed, `done` per live trace), release permit with admission refund. Cancelled -> nothing. Exception -> warning `Environment request failed in group {gid} ({env}): {exc!r}`, record error, advance group accounting, emit `DispatchFailure{policy_version = group's start version, task_type = "Task", task_key, task_hash, error{type, message, traceback}}`. Episode: per trace: train + timeout + no error -> append `Error(type="Timeout", message="Trace stopped by {stop_condition}")`, trace.ok = episode.ok = False; dropped -> info `Trace dropped in group {gid} ({env}) — {dropped}`; has_error -> record error, warning `Trace failed in group {gid} ({env}) — {type}: {message}`. Episode not ok: record error if no trace errored; warning `Episode failed in group {gid} ({env})[ — {type}: {message}]`. `concurrency.record_episode(num_total_tokens)`. Emit: task key/hash must equal the dispatched task (`Episode task provenance ... does not match dispatched task ...`, an error that kills the dispatcher); group accounting (`emitted += 1`, remove group at target); `env.name = env`, `group = {id}`, run = `TrainRunInfo{id, name, work: Train|EvalWorkInfo{step, policy: PolicySpan{start: group start version, end: current policy.version} if live-sourced else None}}`.
- `drop_group(gid, reason)`: pop group; claim its inflight tasks (release permits w/o refund, retire); `count = claimed + episodes_to_schedule`; if > 0: record cancellations, debug `Dropped {kind} group | group={gid[:8]} env={env} reason={reason} | cancelled={n} (inflight={i} unscheduled={u})`, emit `GroupCancellation`; cancel claimed tasks (release payloads of already-returned ones).
- `cancel_inflight(n)` (overload): groups of train inflight ordered youngest first by the *oldest* member's start; drop whole groups (`overload`) until shed >= n (shed counts only live members); warning `Cancelled {shed} youngest in-flight episodes after overload cut`.
- `cancel_inflight_train_episodes()`: disable train; after any scheduling in progress: every train group -> cancellation with its unscheduled count; every train inflight -> claimed, +1 to its group's cancellation (created with count 0 if group gone); record cancellations; return those with count > 0 (`shutdown`). Not emitted on out_q.
- `cancel_inflight_episodes()` (stop): record cancellation per inflight, release, retire, clear groups; no markers.
- `on_version_pending`: `policy_update_pending = True`, wait for any in-progress scheduling. `on_new_version`: pending = False.
- `is_idle`: no inflight, no eval work, out_q empty.
- out_q bounded `max(8, concurrency.max_inflight)` when a ceiling is configured, else unbounded.
- `cancel_eval_step` is used only by `eval/runner.py` (online eval), not by the orchestrator.

### 1.3 ConcurrencyController
Pure AIMD state machine — ported verbatim (constants, thresholds, log lines `Derived initial max inflight {n} - {cap} KV cache tokens / {cost} tokens per episode`, `{Increased|Decreased} concurrency {a} -> {b} ({reason}) - turnover={t:.1f} signal={s}`), `record_episode(tokens)` (growth per completion), `observe(samples)` (signal, growth gate with 15 s TTL, bootstrap, queue/preemption cuts with escalation, kv trims), `set_target`, `gauges()`. Inputs: dispatcher inflight count, `on_overload(excess)` -> dispatcher cancel_inflight. Python tests: `tests/unit/eval/test_control.py` (uses it).

### 1.4 TrainSource / curriculum
- `TrainSource`: `random.Random(42)`; env weights = `ratio`; `next_task(step)` = `rng.choices(env_names, weights)` then `next(curriculum.sampler)`; `on_result(group)` -> curriculum admission, counts admitted/rejected per env; `metrics()`: per env `curriculum/{env}/admission_rate` (if any, popped), `curriculum/{env}/sampler/...`, `curriculum/{env}/gate/{name}/...`; state `{"rng": getstate(), "envs": {env: curriculum state}}`; load validates field sets (`Train-source checkpoint fields must be ['envs', 'rng']`, env set mismatch message).
- `Curriculum(config, tasks)`: sampler Standard | DifficultyPool; gates AdvRange by name. `on_result`: group non-empty, single non-null task key, `sampler.observe`, all gates admit **and** any trainable trace. State `{"sampler": …, "gates": {name: {}}}` with field validation.
- `StandardSampler`: finite: unique keys (`Task keys must be unique within a taskset: [...]`), non-empty (`A finite curriculum needs at least one task`), cycle in order; infinite: stream. `cursor` counts draws; load: finite -> `cursor % len` skips; infinite -> skip `cursor` items.
- `DifficultyPoolSampler`: finite only (`DifficultyPoolSampler requires a finite taskset` / `... at least one task`), unique keys, `Random(seed)`, pools ordered by threshold, task pool = first pool with `score <= threshold` else last; weight 1.0 unseen; `rng.choices(tasks, weights)`; all-zero -> `DifficultyPoolSampler has no tasks in a pool with positive weight`. observe: mean reward of trainable ok non-error traces with turns > 0. metrics `pool/unseen`, `pool/{name}` occupancy. State `{"rng", "task_rewards"}`.
- `AdvRangeGate`: concat advantages of nodes of ok/no-error/trainable traces; empty -> admit; reject iff all within `[reject_min, reject_max]`.
- Python tests: `tests/unit/orchestrator/test_curriculum.py`.

### 1.5 EvalSource / EvalSink
- EvalSource: tasks per env (eval examples), group sizes, intervals (online evals: config intervals; else 1). `trigger(step, force)`: first trigger (fresh run) fires all envs unless `skip_first_step` (then consumes the first trigger and fires nothing); else `step % interval == 0` or force; only envs with tasks. Round-robin queue across fired envs (A1, B1, A2, …), each example one request with fresh group id, rollouts = group size (`plan_rollouts` with no saved episodes = every task once with group_size rollouts). `next_task` pops front; `cancel_step`; len/bool.
- EvalSink: per (env, step) episodes / failures / cancellations; epoch completes when their total >= `len(examples) * group_size`; returns `EvalBatch(env, step, EvalEpisodes(episodes, group_size), failures, cancelled)`; episodes' payloads are released on arrival; `batch_progress()` per open epoch.
- Python tests: `tests/unit/eval/test_resume.py` (EvalSource with saved episodes — eval-runner only), `test_train_sink_payloads.py` (eval sink payload release).

### 1.6 WeightWatcher
Poll every 0.1 s: `next = receiver.next_version(ckpt_step)` (newest published, versions may skip); if newer: under update lock, `wait_published(next, cancelled=stopped)`, `ckpt_step = next`, observers `on_version_pending(next)` (exceptions logged as warning `Observer {name}.on_version_pending({n}) raised: {e!r}`), debug `Updating inference weights to policy v{n}`, `receiver.receive(next)`, `update_count += 1`, `policy.version = next`, debug `Updated inference weights to policy v{n} in {t}`, then observers `on_new_version` (warning on raise) and hooks in order (trigger_eval, on_policy_update — exceptions propagate and kill the watcher -> run fails). `sync_startup(step, timeout)` sets version and runs the same notifications. `stop()` waits for an in-flight apply before cancelling.

### 1.7 PeriodicLogger, lag monitor, live, envs
- PeriodicLogger: every `interval` s (and once more when stopped) collect -> info body, `monitors.log(payload, step=None)` if non-empty.
- Lag monitor: sleep 0.1 s repeatedly, record overshoot (deque 1000); stats min/mean/median/p90/p99 (numpy linear percentile)/max/n. In Rust it measures tokio reactor scheduling lag of a timer task (same semantics: how late a 100 ms sleep wakes).
- live: `dispatch_info = {id, kind, env, group, task: str(data.name) if truthy else "idx={data.idx}", policy_version, step, started: wall time of dispatch}`; `pending_event`, `dispatched_event`, `apply(delta)` (discard pops; turns += len(calls); errors -> stage "error"; `set.timing` -> `stage({"timing": …})` from the file monitor's stage table), `stage_counts` (unused by the orchestrator body).
- envs: Env per source with one EnvClient per replica (address from `env_addresses[(split, name)]`, else replica address files `env_address_file(config_dir, split, name[, replica])`), least-loaded replica routing (ties -> lowest index), sampling config: train = `config.sampling` minus None + `logprobs = (sampling.logprobs or True) if live else None`; `requires_sampling_masks = truncates_distribution() and live`, `requires_top_logprobs = sampling.logprobs is not None and live`; eval = `config.sampling`. Python test: `tests/unit/orchestrator/test_env_replicas.py`.

### 1.8 Imports of these modules from outside the orchestrator (to handle when Python orchestrator is deleted)
- `src/prime_rl/eval/runner.py` (standalone/online eval) uses Dispatcher (+`cancel_eval_step`, `policy_weights_change`, `progress=None`), DispatcherMode, ConcurrencyController, EvalEnvs, EvalSink, EvalSource (with saved episodes / `plan_rollouts`), InferenceMetricsCollector, PeriodicLogger, live, types (DispatchFailure, EvalBatch, GroupCancellation, Policy), utils (`eval_work`, `intercept_vf_logging`, `set_default_executor`, …).
- `src/prime_rl/eval/control.py` and `tests/unit/eval/test_control.py`: ConcurrencyController, EngineLoadSample.
- `tests/unit/eval/test_resume.py`: EvalSource.
- `src/prime_rl/entrypoints/env_server.py`: `orchestrator.utils.setup_env_server_logging` (must move, not port).
- `src/prime_rl/orchestrator/rust.py` (new PyBridge) imports `orchestrator.utils.intercept_vf_logging`.
- `transports/weights/*`, `inference/dynamo.py`: `orchestrator.clients` (io's area).

## 2. Rust design (prl-core) — as implemented

Every component is an actor (one tokio task owning its state; `mpsc` inbox; request/response via
`oneshot`). Shared values are `watch` channels with one writer. The supervisor owns a
`CancellationToken` and a `JoinSet` of named components; any early exit fails the run.

| Module | Python | Notes |
|---|---|---|
| `orchestrator` | `Orchestrator` | `run(config, bridge, wandb, nixl, shutdown)`: setup order and log lines of 1.1, `start` (spawn, drain barrier, final broadcast, final ckpt, finalize), bounded teardown (`SHUTDOWN_TIMEOUT` = 300 s, `process::exit(0/1)` on expiry). `shutdown` (cancelled by prl-py on SIGINT) = the main task being cancelled. |
| `dispatcher` | `Dispatcher` | Actor. Owns groups, in-flight episodes (a `JoinSet` of env runs), permits, admission window, rate limiter, mode, `EvalSource`. Train tasks come from the `TrainSource` actor. Results go to per-kind outboxes drained into bounded sink channels inside the select loop. |
| `train_source`, `curriculum` | `TrainSource`, `curriculum/*` | `TrainSource` is pure + an actor wrapper (`TrainSourceHandle`: `next_task`, `on_result`, `metrics`, `state`). CPython-exact `random.Random` via `prl_io::ckpt::PyRandom`; state types are prl-io's `TrainSourceState` (torch.save-compatible). |
| `train_pipeline` | `main_loop` train branch, `handle_train_result`, `finalize_train_batch`, `start_draining`, `apply_control`, `maybe_save_ckpt` | Actor wrapping `prl_train::TrainSink`; writer of `progress_step`; ship flow inline (serial with ingestion, as in Python); control submissions applied while a send blocks. |
| `eval_pipeline` | `main_loop` eval branch, `trigger_eval`, `finalize_eval_batch` | Actor; the watcher's first update hook. |
| `eval_source`, `eval_sink` | `EvalSource`, `EvalSink` | Pure; generic over the episode type (shared with prl-eval). |
| `watcher` | `WeightWatcher` | Actor; writer of `policy`; `PolicyReceiver` trait implemented over `prl_io::weights::WeightReceiver`. Gauges published on a `watch`. |
| `concurrency` | `ConcurrencyController` | Pure state machine returning `Action`s + actor (`ConcurrencyHandle`: `record_episode`, `get`, `set_target`, `gauges`); writer of `limit`. |
| `envs` | `Env`, `Envs` | `start_env` (address files, health, taskset via bridge), `ReplicaRouter` (least-loaded routing) implementing the dispatcher's `EnvRunner`. |
| `live`, `periodic`, `lag` | `live.py`, `PeriodicLogger`, `EventLoopLagMonitor` | |

Ordering and semantics notes:
- **'all' before sink**: the pipelines stamp arrival and send `log_episodes(All)` before handing the
  episode to their sink; later effective logs follow on the same monitors channel.
- **Gate**: no separate actor. The dispatcher evaluates `(progress_step - 1) - policy.version <= 1`
  from the two watches whenever it schedules train work (always current), and logs the
  pause/resume transitions when either watch changes.
- **Policy update order**: watcher -> `version_pending` (dispatcher stops scheduling) -> receive ->
  `policy` watch -> hooks in order: eval pipeline (`trigger_eval`, may switch to PREFER_EVAL) ->
  dispatcher `version_applied`. So nothing is scheduled on a new policy before an eval trigger
  switched the mode (Python got the same result by running both synchronously).
- **Eval runs in its own actor**: eval episodes are consumed while a train send blocks (Python
  stalled them behind the send). Same records and metrics; only timing differs.
- **Drain**: the train pipeline calls `dispatcher.drain()` (disable train, cancel train groups and
  in-flight train episodes, return the markers directly). The dispatcher sets `idle` once draining
  with nothing in flight, no eval work and empty outboxes; the supervisor then stops the
  dispatcher, waits until the eval pipeline consumed its closed channel (`Pipeline drained, exiting
  main loop`), waits for the final broadcast, saves the final checkpoint, finalizes.
- **Rate limiter**: aiolimiter leaky-bucket math, but non-blocking: a limited admission arms a
  wake-up timer instead of blocking the actor.
- **Deltas**: run tasks relay deltas to the dispatcher on a channel; a per-episode latch stops
  relaying once the episode is claimed (dropped/cancelled), and pending deltas are drained before
  any completion or claim, so no live trace is left without its `done` event.
- **Not ported**: `gc.run`/`malloc_trim` (no Python GC; gc gauges are kept), `set_default_executor`,
  `intercept_vf_logging` (stays in Python), the tokenizer load (unused by the Rust sink).

## 3. Rust tests (port of Python tests)
- `test_curriculum.py` -> curriculum/train_source tests (resume finite/infinite, unique keys, composition + state + metrics, difficulty pools + gate + resume, adv range gate), with Python-generated RNG fixtures for exact sampling sequences.
- `test_env_replicas.py` -> least-loaded routing.
- `eval/test_control.py` concurrency cases -> controller unit tests.
- New focused tests for real logic without Python counterparts: dispatcher scheduling/group accounting/drop_group/overload ordering/drain cancellations (with a fake env runner), eval source round-robin + skip_first_step, eval sink completion, gate transitions, pipeline view body formatting.
