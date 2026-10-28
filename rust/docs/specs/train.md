# prl-train — spec (owner: train)

Ports: `orchestrator/train_sink.py`, `batch.py` (BatchPacker), `trajectories.py`, `algo/*`, `annotations.py`,
`metrics.py`, the staleness helpers of `orchestrator/utils.py`, the orchestrator half of `transports/payload.py`
(`trace_payload_urls`, `sample_payload_urls`, `delete_payloads`) and `transports/batch/*` (types + senders).

## 1. Python behavior inventory

### 1.1 TrainSink (`train_sink.py`)
State: `rollout_window: RolloutWindow`, `pending_groups[group_id] -> [Episode]`, `pending_group_failures[group_id]`,
`pending_group_cancellations[group_id]`, `pending_batch: ordered dict trace_id -> [TrainingSample]` (insertion order
matters: batches cut the first `batch_size` traces), `episode_by_trace[trace_id] -> Episode copy with traces=[that trace]`,
`_swept_step`, `zero_output_units`, `reported_zero_output_windows`.

- `add(episode)`: `algorithm.finalize_episode(episode)` (= `score_episode` iff the episode has a trainable trace), append to
  its group; when `episodes + failures + cancellation.count >= group_size(env)` -> `process_group`, then `take_batch()`.
- `fail(failure)`: `kind != train` -> error. Counted toward the group, same completion rule.
- `cancel(c)`: stores the cancellation; its `count` fills in for undelivered episodes. Same completion rule.
- `take_batch()`: `_drop_stale()` full sweep (once per `progress.step`), then `process_batch()` iff `len(pending_batch) >= batch_size`.
- `_drop_stale(trace_ids?)`: `min_version = (step-1) - max_staleness`; `<= 0` -> no-op. For each queued trace whose
  episode policy span `start < min_version`: `mask_stale_tokens` on each sample (zero mask + rl/ce/ref_kl weights on
  stale runs, return newly masked count); samples with `masked && !has_loss` are released (payload URLs deleted) and
  dropped; a trace with no samples left is removed. If any traces dropped: `count("staleness/rollout/dropped", n)` and
  WARNING `Dropped {n} queued traces with every token past max_staleness={k}. Consider increasing it to avoid this.`
  Frozen-sourced episodes (`policy is None`) never go stale.
- `process_group(gid)`: always (finally) releases payload URLs of every delivered trace that is not queued.
  `_process_group`:
  - pops group/failures/cancellation; nothing -> return. `env_name` from first episode | failure | cancellation.
  - `num_errored = traces with has_error + trace-less !ok episodes + failures`; `n_owed = episodes + failures + cancel.count`.
  - shutdown cancellation: record (admitted=False, cancelled=all episodes), zero-output tally, DEBUG
    `Dropped group | env={env} task_idx={idx} | episodes={n} traces={t} (errored={e}) | reason=cancelled (shutdown)`, return.
  - `survivors = iter_trainable_traces(group)`; if any -> `algorithm.score_group(group)`. `admitted = on_result(group)`
    (curriculum callback; True if none; False for empty group). No survivors or rejected -> record, tally, DEBUG
    `... | reason=no trainable survivors` / `rejected by curriculum`, return.
  - per survivor: `trace_to_samples` (off-thread), stamp `temperatures = full(len, env.sampling.temperature)`; errors:
    `requires_sampling_masks && "sampling_mask" not in payload.fields` -> RuntimeError (message in source, incl.
    `_KEEP_ARRAYS_HINT`); `requires_top_logprobs && neither top_logprobs nor sampling_mask_logprobs kept` -> RuntimeError;
    `stamp_loss_routing(sample, algorithm.action_loss_type)`. If `constant_trainer_batch_size`: prune samples with
    `_prune_zero_advantages == False`. Keep traces with samples.
  - none kept -> record + tally, return. Else insert into `pending_batch` (dict update order = survivor order), store
    `episode_by_trace`, `_drop_stale(new ids)`, `queued = new ∩ pending`; `stale` = episodes whose sampled traces all got
    dropped; record (queued_trace_ids=queued, cancelled=stale). No queued -> tally, return; else reset zero-output counters.
- `_prune_zero_advantages(sample)`: no advantages -> keep. Else `pruned = mask & adv==0 & rl_w!=0` (rl_w default mask as
  f32); none pruned -> keep unchanged; else set those rl_weights to 0 (materializes rl_weights) and return `has_loss`.
- `has_loss`: `any(mask & rl_w!=0) || any(ce!=0) || any(ref_kl!=0)`.
- zero-output tally: `units += len(survivors) or returned_traces or n_owed`; warn once per `units // batch_size` window:
  WARNING `No admitted train payload after {units} finalized units ({w} zero-output batch equivalents)`.
- `process_batch()`: take first `batch_size` traces; if not constant mode, prune zero advantages now (off-thread), drop
  empty traces; release URLs of selected-but-not-shipped samples; stamp `staleness` (int16 per token, -1 on unsampled
  nodes, `max(0, (step-1) - version)`) for traces with a policy span; `rate("staleness/token/masked", {"": masked},
  kept + masked)` where masked counts stale tokens (by `iter_trainable_branches` mask) of traces past min_version
  (already masked earlier) and kept = shipped mask sum; `count("rollout/queued/pruned", selected - shipped)`;
  cohort = `TrainEpisodes(episodes regrouped by episode id (order of first shipped trace), sampled_trace_ids=shipped)`.
- `take_rollout_window(force)`: only with force; None when no groups and no counts.
- `discard_queued()`: `count("rollout/queued/cancelled", len(pending_batch))` if non-empty; release all queued samples
  and all pending-group traces' URLs; clear pending_batch/episode_by_trace (pending groups stay).
- `batch_progress() -> (len(pending_batch), batch_size)`, `buffered_count()`, `pending_batch_by_env()`.
- `stale_token_runs`, `token_staleness`, `mask_stale_tokens`: node offsets across `branch.nodes`, runs from
  `node_policy_versions(node, fallback)`.

### 1.2 utils (staleness)
- `min_fresh_version(step, k) = (step-1) - k`.
- `node_policy_versions(node, fallback)`: `sampled = count(mask)`, `offset = len(mask) - sampled`; no weight_versions ->
  one run `(offset, offset+sampled, fallback)` if sampled; else per span `(offset+start, offset+end, int(version) if
  version.isdigit() else fallback)`.
- `rollout_staleness(episode, step)` -> `Staleness{rollout, in_flight, in_queue, token_sum, token_count}` or None;
  `in_queue = min(rollout, max(0, (step-1) - policy.end))`. `max_rollout_staleness`. `compute_pass_metrics`.

### 1.3 trajectories
- `iter_trainable_branches(trace)`: trainable branches; a sampled node with any mask True is trainable only in the first
  branch containing it (later: zeros); branches with no True mask skipped.
- `trace_to_samples(trace, env_name)`: per trainable branch: `token_ids` (i32), mask (above), `logprobs` (spread),
  `ref_logprobs` (spread or None), mm refs (image_url parts of node messages paired with runs of mm_token_type_ids==1;
  count mismatch -> error; urls without mm ids -> error), `rl/ce/ref_kl_weights` (`_loss_weights`: node stream or zeros,
  a node trains each stream once per trace; all-zero -> None; misaligned -> error), `advantages` (spread or None),
  `trace_id`, `branch_index`, `payload = PayloadSlice(url, id, num_tokens=len, fields, routed_shape?, routed_dtype?)` from
  `trace.info["prime_rl_payload"]["branches"][str(index)]`. No samples -> WARNING
  `No trainable samples (error={has_error}, stop={stop_condition}, num_turns={num_turns}).`

### 1.4 algorithms (`algo/*`)
- `iter_trainable_traces(episodes)`: skip `!episode.ok`, `trace.has_error`, dropped, `!agent.trainable`, `num_turns==0`;
  require any node mask True.
- routing: `trainable_nodes` (nodes on trainable branches with any mask, graph order); `assign_advantages(trace, scalar|vec)`
  (float32, compact per sampled token, alignment errors); `assign_reference_logprobs(branch, values)` (len == branch
  tokens; per node with mask and no existing ref: values[mask]); `scalar_advantage` (mean of nonzero, f64; 0.0 if all zero;
  None if none); `is_trainable` (any nonzero advantage); `stamp_loss_routing` (rl: no-op; ce/ref_kl: rl_weights zeros,
  action weights = mask->1.0 over existing ce stream for ce).
- grpo: rewards f32; optional length penalty (linear: output/total-output/turns; linear-subagent: per-call weighted tokens via
  semantic edges `subagent_call` closure over `subagent_call|continuation|compaction_attempt|compaction`, `min_new_input_tokens`);
  `penalty = mean(rewards) * Σ w_i * x_i / max(x_i.max(), 1)`; baseline mean or length-weighted; `loss_aggregation=prompt`
  writes `rl` loss weights `1/T_q` on masked tokens of all trainable nodes.
- echo: grpo + ce loss weights on non-sampled nodes after the first sampled node whose role is in the role table
  (is_content-narrowed when aligned; optional user filter — Python callable, see gaps); merged with max into node `ce` stream.
- max_rl: `(r - mean)/mean`, zeros if mean <= 0. rae: per-agent EMA baseline, score before update.
- hierarchical_grpo: peers keyed by (agent name, episode id if agent in episode_agents else None); reward − peer mean.
- opd: teacher `score(branch.token_ids)` per trainable branch (concurrently), `assign_reference_logprobs`.
- opsd: hint = template.format(demonstration=trace.info[demo_key] or task.data[demo_key]) (missing -> ValueError);
  `hint_block = renderer.render_ids([system hint], add_generation_prompt=False)`; score `hint_block + branch tokens`
  against the live policy; slice off the hint prefix.
- sft: no-op. debug: constant advantage on clean trainable traces (score_episode).

### 1.5 annotations
- `stamp_arrival(episodes, kind, step)`: `trace.info.kind`, `trace.info.dispatch = {step: work.step, time: trace.timing.start}`
  (when run.work exists), `trace.info.arrival = {step, time: now}`.
- `stamp_batch(episodes, step)` -> `[make_update(trace.id, info={"effective": True, "ship": {step, time}, ["advantage"]},
  branches={index: {"advantages": [...]}} for trainable branches with advantages)]`;
  `make_update = {"version": 1, "trace_id", "info", "branches": [{"index": i, **streams}]}`.

### 1.6 metrics
`Stat` (mean/max/min/p10/p90 with linear interpolation, empty -> {}), `MetricWindow` (distributions/counts/rates; to_dict
order: counts, distributions (`/mean,/max,/min,/p10,/p90`), rates (`prefix/name` or `prefix` for name "")),
`TraceMetrics.snapshot(prefix, subset)` (distributions reward/num_total_tokens/num_input_tokens/num_output_tokens/num_turns/
num_branches, rates is_truncated/is_timeout/is_completed `/mean`, timing/{setup,agent,finalize,scoring,agent/model,agent/harness,total},
metrics/*, rewards/*; all-only: has_error, cancelled, error/{type} counts, dropped (+/{reason}), solve rates
`prefix/solved_none|solved_all|solved_some`; always `stop_condition/generation_truncated` and `stop_condition/{cond}`),
`EpisodeMetrics` (`{prefix}/{subset}/num_*` per-episode sums, all-only has_error/cancelled, per-agent TraceMetrics),
`TrainMetrics` (+ `is_trainable`, `is_admitted` rates per agent), `EvalMetrics.to_wandb` (+ `avg@{group_size}`, pass@k/pass^k on
effective), `EpisodeCollection/TrainEpisodes/EvalEpisodes` (`effective` predicates, `by_env`, `vf_episodes` with
`info.advantage = scalar_advantage`), `RolloutWindow.observe` (tokens/traces/groups/attempts/discarded/stale/errored/cancelled +
`train/agg` and `train/{env}` snapshots, `.../all/dispatch_failure` rate and per-type counts, `rollout/env_share/{env}`,
`rollout/traces/queued`, `rollout/traces/discarded`), `dispatch_failure_metrics`.

### 1.7 BatchPacker (`batch.py`)
- `prepare_sample(sample, seq_len)`: mm validation errors; missing advantages with live rl members -> ValueError, else zeros;
  truncation to `seq_len` (cut back to an image start if the cut splits an image; payload `num_tokens = cut`); emits a
  one-sequence MicroBatch with `trace_ids=[id or ""]`, `branch_indices=[idx or -1]`, `payloads=[p] or None`.
- `packed_samples_into_micro_bs`: stable sort by -len; first-fit (bins hold <= seq_len tokens, routing-or-not uniform per bin);
  `num_train_workers>1`: split the heaviest multi-sample bin (bin_cost, max by (workload, idx)) via LPT until
  `ceil(bins/W)*W` (>= W) bins; `_materialize_bin` with STREAM_FILL backfill (rl 1.0, ce/ref_kl 0.0, ref 0.0, mm 0, staleness -1).
- `pad_micro_batch`: pad to multiple (text rows repeat their own tokens (`np.resize`), mm rows pad id 1; position ids restart at 0;
  temperatures 1.0; streams 0.0; staleness -1; last sequence length += pad; `padding += pad`).
- modality split (mm / text), each group padded to a multiple of W with deep copies of a zero-loss dummy of `group[0]`
  (no rl/ce/ref_kl, payloads only `routed_experts` field if routing else None, trace ids/branch indices None);
  `_assert_token_arrays_aligned`; `_distribute_group` = Karmarkar-Karp (heapq ordering) + best-swap refinement on
  `bin_cost(sequence_lengths)`; mm batches first, then text, per rank.
- `bin_cost` = `FlopCounter(meta model)`: additive over sequences: `n * matmul_flops_per_token + Σ attention_flops(n)`.
- `pad_to_multiple_of = seq_len if pad_to_seq_len else pad_to_multiple_of`.

### 1.8 Batch transport (`transports/batch/*`)
- Wire: msgspec msgpack, `array_like=True, omit_defaults=True` (trailing default fields omitted). `MicroBatch` field order:
  input_ids, loss_mask, advantages, inference_logprobs, position_ids, sequence_lengths, temperatures, env_names, seq_lens,
  ref_logprobs, routed_experts, mm_refs, mm_token_type_ids, rl_weights, ce_weights, ref_kl_weights, sampling_mask,
  trace_ids, branch_indices, top_logprobs, padding, staleness, payloads. Streams = msgpack bin of raw little-endian bytes
  (`MICRO_BATCH_DTYPES`: i32 ids/positions/mm, bool mask (1 byte), f32 floats, i16 staleness). `PayloadSlice` (array,
  no omit_defaults): [url, id, num_tokens, fields, routed_shape|nil, routed_dtype|nil]. `MMRefs`=[[ [url,offset,length], ...]].
  Per rank: msgpack array of micro batches.
- Filesystem sender: `{output_dir}/batches/step_{n}/rank_{r}.bin.tmp` written first; wait (poll 0.1 s) until
  `rank_*.request` count == int(first marker content); rename all `.tmp` -> `.bin`; `current_step += 1`; returns rename time.
- ZMQ sender: ROUTER bound at `tcp://{host}:{port}` (ROUTER_MANDATORY); encode per rank first; collect `BatchRequest`
  (msgpack map `{world_size, data_rank, step}`) until `world_size` distinct identities; wrong step -> RuntimeError
  `A trainer rank requested step {s}, but the orchestrator is on step {n}. Resume both from the same checkpoint step.`;
  reply `[identity, buffer[data_rank]]`; DEBUG `Sent micro batch grid for step {n} to {k} ranks`.
- Grid asserts: `len(grid) == data_world_size`, equal list lengths.

### Python tests covering this area
`tests/unit/orchestrator/{test_train_sink,test_train_sink_payloads,test_batch,test_algorithms,test_advantage,test_metrics,
test_rollout_staleness}.py`, `tests/unit/transports/{test_batch_encode,test_batch_transport}.py`.

## 2. Rust API (crate `prl-train`, as implemented)

- `sample`: `TrainingSample`, `PayloadSlice`, `MMRefs`/`MMImageRef`, `sample_payload_urls`.
- `trajectories`: `iter_trainable_branches(&Trace) -> Vec<(Branch, mask)>`, `trace_to_samples(&Trace, env) -> Result<Vec<TrainingSample>>`, `trainable_node_ids`.
- `staleness`: `min_fresh_version`, `node_policy_versions`, `rollout_staleness(&Episode, traces, step) -> Result<Option<Staleness>>`,
  `stale_token_runs`, `token_staleness`, `mask_stale_tokens`.
- `algo`: `trait Algorithm { action_loss_type, setup, score_episode(&mut Episode), score_group(&mut [Episode]), finalize_episode }`
  (async), `build_algorithm(&AlgoConfig, &AlgorithmDeps)`, `AlgorithmDeps { policy: Arc<dyn prl_types::Scorer>, policy_model_name,
  connect_frozen: ConnectFrozen, bridge: Option<Arc<dyn prl_types::Bridge>> }`, `iter_trainable_traces(&[Episode]) -> Vec<(e, t)>`,
  routing (`assign_advantages`, `assign_reference_logprobs`, `scalar_advantage`, `is_trainable`, `stamp_loss_routing`).
- `metrics`: `Stat`, `MetricWindow`, `TraceRecord`, `EpisodeMetrics` (snapshot / train_snapshot, Stat accessors), `TrainEpisodes`,
  `EvalEpisodes::new(Vec<Arc<Episode>>, group_size)`, `RolloutWindow`, `dispatch_failure_metrics`, `compute_pass_metrics`, `pass_at_k`.
  Collections hold `EpisodeRef { Arc<Episode>, trace indices }` (Python's shallow `model_copy(update={"traces": ...})`).
- `sink`: `TrainSink::new(SinkConfig, IndexMap<String, TrainEnvSpec>, PayloadReleaser)`, `setup`, `add/fail/cancel(.., step, &mut dyn Admission)`,
  `take_batch(step)`, `take_rollout_window()` (= `force=True`), `count_queued_cancelled`, `release_samples`, `discard_queued`,
  `batch_progress`, `buffered_count`, `pending_batch_by_env`; `TrainBatch { cohort: TrainEpisodes, samples }`; `Admission` (curriculum
  verdict, async), `AdmitAll`; `PayloadReleaser::default()` = `prl_wire::payload::delete_payloads`. The sink is a plain struct (core's
  actor owns it); the step is passed explicitly; sample building, pruning and staleness stamping run on `spawn_blocking`.
- `packer`: `BatchPacker::new(seq_len, num_train_workers, pad_to_multiple_of, pad_to_seq_len, Arc<dyn BinCost>)`, `pack` (sync; call
  from `spawn_blocking`), `FlopTable::new(Vec<i128>)` from `Bridge::flop_cost_table(seq_len + pad_to_multiple_of)`; `prepare_batch`,
  `prepare_sample`, `pad_micro_batch`; `partition::balanced_partition` (KK with Python-`heapq`-identical ordering + swap refinement, i128).
- `transport`: `MicroBatch`, `encode_micro_batches` (hand-written msgpack writer, byte-identical to msgspec), `BatchSender::{new(output_dir,
  data_world_size, current_step, &prl_config::TransportConfig), send(grid) -> seconds, close}`. ZMQ uses libzmq (`zmq` crate, ROUTER with
  `ROUTER_MANDATORY`) on a blocking thread; the pure-Rust `zeromq` 0.4 ROUTER loses wakeups with several peers (its fair queue returns
  `Pending` while other streams are ready), so it is not used.

## 3. Tests
- `tests/packer_parity.rs`: 60 random sample sets packed by Python `prepare_batch` and encoded by `BatchSender.encode`
  (`fixtures/gen_packer.py`): Rust bytes identical (sum, flop-shaped and >int64 bin costs; 1-8 workers; padding; mm; payloads).
- `tests/sink_parity.rs`: 30 random scenarios (`fixtures/gen_sink.py`): real `vf.Episode`s (forks, compaction attempts, subagent edges,
  weight versions, errors, drops, judges, payload refs), failures, overload/shutdown cancellations, curriculum verdicts, all non-calling
  algorithms, constant/deferred pruning, max_staleness 0-4, step advances — replayed through the Rust sink: every batch's samples, cohort,
  effective train metrics (`train/agg` + per env), rollout staleness, advantages, rollout windows (metrics + counters), released payload
  URLs, batch progress match. Also `dispatch_failure_metrics`.
- `tests/algo_parity.rs`: opd / opsd / echo-with-user-filter against deterministic fake teacher, renderer and filter
  (`fixtures/gen_algos.py`); ported unit tests for `stamp_loss_routing`, zero-advantage pruning, opsd template.
- `tests/transport_interop.rs` (`--ignored`, needs the repo Python env): the Python trainer's `FileSystemBatchReceiver` /
  `ZMQBatchReceiver` (4 ranks, 2 data ranks, 2 steps) decode Rust-sent batches losslessly.

## 4. Gaps / deviations
- Echo user filters and the opsd hint renderer run in Python through `prl_types::Bridge` (`echo_filter`, `render_ids`).
- GRPO / MaxRL use f32 arithmetic like torch, but sums accumulate in f64 (torch: f32 cascade sums), so advantages can differ by an ULP.
- Python iterates a `set` for the cohort episode order and for `stop_condition/*` keys; Rust uses first-seen order (Python's order is
  hash-randomized, so no consumer can rely on it).
- `compute_pass_metrics` divides `u128` binomials as f64 (Python divides exact ints): identical below 2^53.
- `annotations.py` (`stamp_arrival`, `stamp_batch`) moved to prl-monitors; `delete_payloads`/`trace_payload_urls` live in prl-wire.

## 5. Imports of these modules from outside the orchestrator (to handle when Python orchestrator is deleted)
- `src/prime_rl/eval/runner.py`: `orchestrator.annotations.{stamp_arrival, stamp_batch}`, `orchestrator.metrics.dispatch_failure_metrics`,
  `orchestrator.utils.{eval_work, intercept_vf_logging, set_default_executor}` (plus non-train modules: live, clients, concurrency,
  dispatcher, envs, eval_sink, eval_source, inference_metrics, periodic_logger, types). Eval runner metrics also use
  `EvalEpisodes` via eval_sink.
- `src/prime_rl/entrypoints/env_server.py`: `orchestrator.utils.setup_env_server_logging` (env-server logging; must move, not port).
- `tests/unit/transports/test_payload.py`: `orchestrator.batch.prepare_batch`, `orchestrator.trajectories.trace_to_samples`.
- `transports/batch/*` (receivers + types) and `transports/payload.py` (PayloadHook, resolve_payloads) stay in Python: the
  trainer and env servers use them. Only the orchestrator-side senders/packer move to Rust.
