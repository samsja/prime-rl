# Imports the removed Python orchestrator: run it from a checkout that still has it (e.g. the baseline worktree at 8acab25e4).
"""Generate prl-train sink fixtures: random native episodes replayed through the Python TrainSink.

Run: uv run --no-sync python rust/crates/prl-train/tests/fixtures/gen_sink.py
Each scenario records its config, the event sequence (episode records as JSON, failures,
cancellations, curriculum verdicts) and, per event, what the sink returned: batches (samples,
cohort metrics), rollout windows and the payload URLs it released.
"""

import asyncio
import gzip
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.semantic import ParentLink
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import AssistantMessage, ToolMessage, Usage, UserMessage, WeightVersionSpan

import prime_rl.orchestrator.train_sink as train_sink_module
from prime_rl.configs.algorithm import (
    DebugAlgoConfig,
    EchoAlgoConfig,
    GRPOAlgoConfig,
    HierarchicalGRPOAlgoConfig,
    MaxRLAlgoConfig,
    RAEAlgoConfig,
    SFTAlgoConfig,
)
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.algo import build_algorithm
from prime_rl.orchestrator.metrics import dispatch_failure_metrics
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, Progress
from prime_rl.orchestrator.utils import rollout_staleness
from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY
from prime_rl.utils.logger import setup_logger

setup_logger("error")

RELEASED: list[str] = []
train_sink_module.delete_payloads = lambda urls: RELEASED.extend(urls)

ALGOS = {
    "grpo": lambda: GRPOAlgoConfig(),
    "grpo_linear": lambda: GRPOAlgoConfig(
        length_penalty={"type": "linear", "num_output_tokens_weight": 0.3}, length_weighted_baseline=True
    ),
    "grpo_subagent_prompt": lambda: GRPOAlgoConfig(
        length_penalty={"type": "linear-subagent", "subagent_token_weight": 0.5}, loss_aggregation="prompt"
    ),
    "max_rl": lambda: MaxRLAlgoConfig(),
    "rae": lambda: RAEAlgoConfig(decay=0.9),
    "hierarchical_grpo": lambda: HierarchicalGRPOAlgoConfig(episode_agents=["solver"]),
    "debug": lambda: DebugAlgoConfig(advantage=0.5),
    "sft": lambda: SFTAlgoConfig(sampling={"source": {"name": "teacher", "base_url": "http://teacher:1/v1"}}),
    "echo": lambda: EchoAlgoConfig(roles={"tool": {"alpha": 0.25}, "user": {"alpha": 0.5}}),
}


def q(x):
    return float(np.round(x * 64) / 64)


class Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.next_token = 0

    def tokens(self, n):
        ids = list(range(self.next_token, self.next_token + n))
        self.next_token = (self.next_token + n) % 5000
        return ids

    def trace(self, *, agent_name="agent", frozen=False, trainable=True):
        rng = self.rng
        nodes: list[MessageNode] = []
        calls: list[ModelCall] = []
        n = rng.randint(1, 3)
        nodes.append(
            MessageNode(
                message=UserMessage(content="q"),
                token_ids=self.tokens(n),
                mask=[False] * n,
                is_content=[True] * n if rng.random() < 0.5 else [],
            )
        )
        parent = 0
        total = n
        turns = rng.randint(0, 3) if rng.random() < 0.1 else rng.randint(1, 3)
        assistant_ids = []
        for turn in range(turns):
            scaffold = rng.randint(0, 2)
            sampled = rng.randint(1, 5)
            mask = [False] * scaffold + [True] * sampled
            wv = None
            if not frozen and rng.random() < 0.4:
                cut = rng.randint(0, sampled)
                wv = [
                    WeightVersionSpan(version=rng.choice(["default", "1", "2", "3", "5"]), start=0, end=cut),
                    WeightVersionSpan(version=rng.choice(["2", "4", "6"]), start=cut, end=sampled),
                ]
            node = MessageNode(
                parent=parent,
                message=AssistantMessage(content="a"),
                sampled=True,
                token_ids=self.tokens(scaffold + sampled),
                mask=mask,
                logprobs=[q(-rng.random()) for _ in range(sampled)],
                weight_versions=wv,
            )
            nodes.append(node)
            parent = len(nodes) - 1
            assistant_ids.append(parent)
            calls.append(
                ModelCall(
                    node=parent,
                    usage=Usage(prompt_tokens=total, completion_tokens=sampled),
                    finish_reason=rng.choice(["stop", "length", "tool_calls"]),
                )
            )
            total += scaffold + sampled
            if turn < turns - 1 or rng.random() < 0.3:
                k = rng.randint(1, 4)
                role = rng.choice(["tool", "user"])
                message = ToolMessage(content="r", tool_call_id="c") if role == "tool" else UserMessage(content="u")
                nodes.append(
                    MessageNode(
                        parent=parent,
                        message=message,
                        token_ids=self.tokens(k),
                        mask=[False] * k,
                        is_content=([False] + [True] * (k - 1)) if rng.random() < 0.5 else [],
                    )
                )
                parent = len(nodes) - 1
                total += k
        # A fork: a second response from an earlier node (a second branch sharing the prefix).
        if assistant_ids and rng.random() < 0.3:
            fork_parent = rng.choice([0, *assistant_ids[:-1]]) if len(assistant_ids) > 1 else 0
            sampled = rng.randint(1, 4)
            nodes.append(
                MessageNode(
                    parent=fork_parent,
                    message=AssistantMessage(content="f"),
                    sampled=True,
                    token_ids=self.tokens(sampled),
                    mask=[True] * sampled,
                    logprobs=[q(-rng.random()) for _ in range(sampled)],
                    semantic_parents=[
                        ParentLink(
                            node=fork_parent, type=rng.choice(["compaction_attempt", "subagent_call", "continuation"])
                        )
                    ],
                )
            )
            calls.append(ModelCall(node=len(nodes) - 1, usage=Usage(prompt_tokens=3, completion_tokens=sampled)))
        ok = rng.random() > 0.1
        trace = vf.Trace(
            task=vf.TraceTask(type="Task", data=vf.TaskData(idx=rng.randint(0, 9), prompt=None)),
            agent=vf.AgentInfo(config=vf.AgentConfig(), name=agent_name, trainable=trainable),
            nodes=nodes,
            calls=calls,
            ok=ok,
            is_completed=True,
            is_timeout=rng.random() < 0.1,
            stop_condition=rng.choice([None, "agent_completed", "max_turns", "prompt_too_long"]),
            rewards={
                "correct": vf.Reward(score=rng.choice([0.0, 1.0, 1.0, 0.5])),
                **({"style": vf.Reward(score=0.25, weight=2.0)} if rng.random() < 0.3 else {}),
            },
            metrics={"m": q(rng.random())} if rng.random() < 0.5 else {"m": None},
            errors=[] if ok else [vf.Error(type=rng.choice(["ProviderError", "TimeoutError"]), message="x")],
            dropped="filtered" if rng.random() < 0.05 else None,
        )
        trace.timing.setup.start, trace.timing.setup.end = 10.0, 10.0 + q(rng.random())
        trace.timing.agent.start, trace.timing.agent.end = 11.0, 11.0 + q(rng.random() * 4)
        trace.timing.agent.model.duration = q(rng.random())
        if rng.random() < 0.4:
            branches = {
                str(b.index): {"id": f"{trace.id}-{b.index}", "fields": ["sampling_mask"]}
                for b in trace.branches
                if b.trainable
            }
            if branches:
                trace.info[PAYLOAD_INFO_KEY] = {"url": "http://env:9", "branches": branches}
        return trace

    def episode(self, env_name, group_id, step, version, hierarchical=False, frozen=False):
        rng = self.rng
        if hierarchical:
            traces = [self.trace(agent_name="proposer")] + [
                self.trace(agent_name="solver") for _ in range(rng.randint(1, 3))
            ]
        else:
            traces = [self.trace(frozen=frozen) for _ in range(rng.choice([1, 1, 1, 2]))]
            if rng.random() < 0.1:
                traces.append(self.trace(agent_name="judge", trainable=False))
        policy = None if frozen else vf.PolicySpan(start=version, end=version + rng.randint(0, 2))
        return vf.Episode(
            env=vf.EnvInfo(id=env_name, name=env_name),
            task=traces[0].task,
            group=vf.GroupInfo(id=group_id),
            traces=[] if rng.random() < 0.03 else traces,
            ok=rng.random() > 0.05,
            run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=step, policy=policy)),
        )


def sample_json(s):
    def arr(a):
        return None if a is None else np.asarray(a).tolist()

    return {
        "token_ids": arr(s.token_ids),
        "mask": arr(s.mask),
        "logprobs": arr(s.logprobs),
        "temperatures": arr(s.temperatures),
        "env_name": s.env_name,
        "ref_logprobs": arr(s.ref_logprobs),
        "rl_weights": arr(s.rl_weights),
        "ce_weights": arr(s.ce_weights),
        "ref_kl_weights": arr(s.ref_kl_weights),
        "advantages": arr(s.advantages),
        "trace_id": s.trace_id,
        "branch_index": s.branch_index,
        "staleness": arr(s.staleness),
        "payload": None if s.payload is None else [s.payload.url, s.payload.id, s.payload.num_tokens, s.payload.fields],
    }


def window_json(window):
    if window is None:
        return None
    return {
        "metrics": window.metrics.to_dict(),
        "counters": [
            window.tokens,
            window.traces,
            window.groups,
            window.attempts,
            window.discarded,
            window.stale,
            window.errored,
            window.cancelled,
        ],
    }


def batch_json(batch, step):
    effective = batch.cohort.effective
    metrics = effective.metrics.to_wandb(prefix="train/agg", subset="effective")
    for env_name, pool in effective.by_env().items():
        metrics |= pool.metrics.to_wandb(prefix=f"train/{env_name}", subset="effective")
    staleness = [s for episode in effective if (s := rollout_staleness(episode, step)) is not None]
    advantages = {
        trace.id: trace.info.get("advantage") for episode in effective.vf_episodes for trace in episode.traces
    }
    eff = effective.metrics
    return {
        "samples": [sample_json(s) for s in batch.samples],
        "cohort_episodes": sorted(e.id for e in batch.cohort.episodes),
        "cohort_traces": sorted(t.id for e in batch.cohort.episodes for t in e.traces),
        "effective_metrics": metrics,
        "staleness": sorted([s.rollout, s.in_flight, s.in_queue, s.token_sum, s.token_count] for s in staleness),
        "advantages": advantages,
        "means": [
            eff.reward.mean(),
            eff.num_turns.mean(),
            eff.num_branches.mean(),
            eff.is_truncated.mean(),
            eff.is_timeout.mean(),
            eff.has_error.mean(),
        ],
    }


async def run_scenario(seed: int):
    rng = random.Random(seed)
    b = Builder(rng)
    algo_names = rng.sample(sorted(ALGOS), k=rng.choice([1, 2]))
    batch_size = rng.choice([2, 3, 5, 8])
    constant = rng.random() < 0.5
    max_staleness = rng.choice([0, 1, 2, 4])
    config = OrchestratorConfig(
        batch_size=batch_size, constant_trainer_batch_size=constant, max_staleness=max_staleness
    )
    envs = {}
    env_specs = {}
    for i, algo_name in enumerate(algo_names):
        env_name = f"env{i}"
        group_size = rng.choice([1, 2, 3])
        temperature = rng.choice([0.5, 1.0])
        algo_config = ALGOS[algo_name]()
        envs[env_name] = SimpleNamespace(
            config=SimpleNamespace(group_size=group_size, sampling=SimpleNamespace(temperature=temperature)),
            requires_sampling_masks=False,
            requires_top_logprobs=False,
            algorithm=build_algorithm(algo_config, None),
        )
        env_specs[env_name] = {"algo": algo_name, "group_size": group_size, "temperature": temperature}
    verdicts: dict[str, bool] = {}

    def on_result(group):
        gid = group[0].group.id
        verdicts[gid] = rng.random() > 0.15
        return verdicts[gid]

    progress = Progress()
    sink = train_sink_module.TrainSink(
        config,
        tokenizer=None,
        train_envs=SimpleNamespace(get=lambda name: envs[name]),
        progress=progress,
        batch_size=batch_size,
        on_result=on_result,
    )
    # Build the event stream: groups interleaved, with failures and cancellations.
    pending = []
    for g in range(rng.randint(4, 14)):
        env_name = rng.choice(sorted(envs))
        spec = env_specs[env_name]
        gid = f"s{seed}g{g}"
        hierarchical = spec["algo"] == "hierarchical_grpo"
        frozen = spec["algo"] == "sft"
        items = []
        for _ in range(spec["group_size"]):
            roll = rng.random()
            if roll < 0.08:
                items.append(("fail", env_name, gid))
            else:
                items.append(("episode", env_name, gid, hierarchical, frozen))
        if rng.random() < 0.15:
            cut = rng.randint(0, len(items) - 1)
            items = items[:cut] + [("cancel", env_name, gid, len(items) - cut, rng.choice(["overload", "shutdown"]))]
        pending.append(items)
    events = []
    out_events = []
    while any(pending):
        group_items = rng.choice([p for p in pending if p])
        item = group_items.pop(0)
        step = progress.step
        RELEASED.clear()
        if item[0] == "episode":
            _, env_name, gid, hierarchical, frozen = item
            version = max(0, step - 1 - rng.randint(0, 5))
            episode = b.episode(env_name, gid, step, version, hierarchical=hierarchical, frozen=frozen)
            events.append({"type": "episode", "record": json.loads(episode.to_json(float_decimals=None))})
            batch = await sink.add(episode)
        elif item[0] == "fail":
            _, env_name, gid = item
            failure = DispatchFailure(
                "train",
                env_name,
                gid,
                step,
                0,
                "Task",
                "k",
                "h",
                vf.Error(type=rng.choice(["TransportError", "ProviderError"]), message="boom"),
            )
            events.append(
                {"type": "fail", "env_name": env_name, "group_id": gid, "step": step, "error_type": failure.error.type}
            )
            batch = await sink.fail(failure)
        else:
            _, env_name, gid, count, reason = item
            events.append(
                {
                    "type": "cancel",
                    "env_name": env_name,
                    "group_id": gid,
                    "step": step,
                    "count": count,
                    "reason": reason,
                }
            )
            batch = await sink.cancel(GroupCancellation("train", env_name, gid, step, count, reason))
        batches = []
        while batch is not None:
            window = sink.take_rollout_window(force=True)
            batches.append(
                {"step": progress.step, "batch": batch_json(batch, progress.step), "window": window_json(window)}
            )
            progress.step += 1
            batch = await sink.take_batch()
        if rng.random() < 0.2:
            progress.step += 1  # a step shipped elsewhere: queued traces age
            events[-1]["advance_after"] = True
        out_events.append(
            {
                "batches": batches,
                "released": sorted(RELEASED),
                "progress": list(sink.batch_progress()),
                "buffered": sink.buffered_count(),
            }
        )
    RELEASED.clear()
    sink.discard_queued()
    final = {"released": sorted(RELEASED), "window": window_json(sink.take_rollout_window(force=True))}
    return {
        "seed": seed,
        "batch_size": batch_size,
        "constant": constant,
        "max_staleness": max_staleness,
        "envs": env_specs,
        "verdicts": verdicts,
        "events": events,
        "results": out_events,
        "final": final,
    }


async def main():
    scenarios = [await run_scenario(seed) for seed in range(30)]
    failures = [
        DispatchFailure("eval", "e", "g", 1, 0, "T", "k", "h", vf.Error(type=t, message="m")) for t in ["B", "A", "B"]
    ]
    extra = {"dispatch_failure_metrics": dispatch_failure_metrics(failures, prefix="eval/e/all", total_attempts=7)}
    out = Path(__file__).parent / "sink_cases.json.gz"
    out.write_bytes(gzip.compress(json.dumps({"scenarios": scenarios, **extra}).encode(), 9))
    print(f"wrote {len(scenarios)} scenarios to {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
