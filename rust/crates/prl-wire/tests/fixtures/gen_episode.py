"""A rich episode streamed the way an env server streams it, and what Python makes of it.

    uv run --no-sync python rust/crates/prl-wire/tests/fixtures/gen_episode.py

Writes `episode_stream.msgpack`: {"deltas": [delta payload bytes...], "reply": reply payload
bytes} exactly as the server frames them, `episode_assembled.msgpack` (the assembled
record), and records/derived values of the validated `WireEpisode`, before and after the
orchestrator-side mutations the Rust tests replay.
"""

import asyncio
import json
from pathlib import Path

import msgpack
import numpy as np

import verifiers.v1 as vf
from verifiers.v1.episode import EnvInfo, Episode, WireEpisode
from verifiers.v1.graph import MessageNode
from verifiers.v1.semantic import ParentLink
from verifiers.v1.serve.delta import DeltaStreamer, EpisodeAssembly, TraceSummary, dump, pack, unpack
from verifiers.v1.serve.encoding import msgpack_encoder
from verifiers.v1.serve.types import RunResponse
from verifiers.v1.trace import InterceptRecord, ModelCall, TimeSpan, TraceTask
from verifiers.v1.types import (
    AssistantMessage,
    ImageUrlContentPart,
    ImageUrlSource,
    SystemMessage,
    TextContentPart,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
    WeightVersionSpan,
)

OUT = Path(__file__).parent


class MyTask(vf.TaskData):
    answer: str
    meta: dict
    weight: float = 1.0


def make_trace(name: str, trainable: bool) -> vf.Trace:
    task = MyTask(
        idx=7,
        prompt=[UserMessage(content="what is 2+2?")],
        answer="4",
        meta={"source": "unit", "missing": None, "tags": ["a", None]},
    )
    config = vf.AgentConfig(max_turns=5, sampling=vf.SamplingConfig(temperature=0.7, top_k=20))
    return vf.Trace(
        task=TraceTask(type="MyTask", data=task, key="k-7", hash="h-7"),
        agent=vf.AgentInfo(config=config, name=name, trainable=trainable),
    )


async def build() -> tuple[Episode, list[bytes]]:
    frames: list[bytes] = []
    traces: list[vf.Trace] = []

    async def send(delta: dict) -> None:
        frames.append(pack(delta))

    streamer = DeltaStreamer(lambda: traces, send)
    tools = [Tool(name="calc", description="adds", parameters={"type": "object", "properties": {}}, extra_flag=True)]

    a = make_trace("solver", True)
    discarded = make_trace("solver", True)
    traces.extend([a, discarded])
    await streamer.flush()

    # turn 1: system + user(image) -> assistant tool call
    a.nodes.append(
        MessageNode(
            message=SystemMessage(content="be brief"),
            tools=tools,
            token_ids=[1, 2],
            mask=[False, False],
            is_content=[False, True],
            timestamp=1.5,
        )
    )
    a.nodes.append(
        MessageNode(
            parent=0,
            message=UserMessage(
                content=[TextContentPart(text="look"), ImageUrlContentPart(image_url=ImageUrlSource(url="data:x"))]
            ),
            token_ids=[3, 151655, 151655, 4],
            mask=[False] * 4,
            timestamp=1.5,
        )
    )
    a.nodes.append(
        MessageNode(
            parent=1,
            message=AssistantMessage(
                content=None,
                reasoning_content="think",
                tool_calls=[ToolCall(id="c1", name="calc", arguments='{"a": 1}')],
            ),
            sampled=True,
            token_ids=[5, 6, 7, 8],
            mask=[False, True, True, True],
            is_content=[False, True, True, False],
            logprobs=[-0.123456789, -1e-8, -12.5],
            weight_versions=[WeightVersionSpan(version="3", start=0, end=3)],
            routed_experts=np.ones((4, 2, 2), dtype=np.uint8),
            timestamp=2.25,
        )
    )
    a.calls.append(
        ModelCall(
            node=2,
            model="m",
            sampling=vf.SamplingConfig(temperature=0.7, max_tokens=64, top_k=20),
            endpoint="/chat/completions",
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=4, completion_tokens=3, cached_input_tokens=2, cost=0.001),
            time=TimeSpan(start=1.5, end=2.25),
        )
    )
    a.mm_token_type_id_map = {151655: 1}
    a.tools = tools
    a.timing.setup = TimeSpan(start=1.0, end=1.4)
    a.preview(object(), [ToolMessage(content="2", tool_call_id="c1", name="calc")])
    await streamer.flush()

    # the retried attempt goes away; turn 2 commits a tool result and the final answer
    traces.remove(discarded)
    a._pending.clear()
    a.nodes.append(
        MessageNode(
            parent=2,
            message=ToolMessage(content="2", tool_call_id="c1", name="calc"),
            token_ids=[9],
            mask=[False],
            timestamp=3.0,
        )
    )
    a.nodes.append(
        MessageNode(
            parent=3,
            message=AssistantMessage(content="  4  "),
            sampled=True,
            token_ids=[10, 11, 12],
            mask=[False, True, True],
            logprobs=[-0.5, -0.25],
            timestamp=3.5,
        )
    )
    a.calls.append(
        ModelCall(
            node=4,
            finish_reason="length",
            usage=Usage(prompt_tokens=8, completion_tokens=2),
            time=TimeSpan(start=3.0, end=3.5),
        )
    )
    # a compaction attempt branching off the user turn, resumed from later
    a.nodes.append(
        MessageNode(
            parent=1,
            message=AssistantMessage(content="summary"),
            sampled=True,
            token_ids=[13, 14],
            mask=[False, True],
            logprobs=[-0.75],
            semantic_parents=[ParentLink(node=2, type="compaction_attempt")],
            timestamp=4.0,
        )
    )
    a.calls.append(
        ModelCall(
            node=5,
            usage=Usage(prompt_tokens=6, completion_tokens=1),
            error=vf.Error(type="ProviderError", message="boom", status_code=500),
        )
    )
    await streamer.flush()
    a.nodes[4].semantic_parents.append(ParentLink(node=5, type="compaction"))
    a.extra_usage.append(Usage(prompt_tokens=10, completion_tokens=1, reasoning_tokens=1))
    a.request_rewrites.append(InterceptRecord(handler="strip"))
    a.rewards = {"correct": vf.Reward(score=0.5, weight=2.0), "skipped": None}
    a.metrics = {"m": 1.5, "n": None, "tiny": 1e-7}
    a.info = {"a": None, "nested": {"x": [1, None]}, "f": 1e-7, "big": 12345678901234567890}
    a.root_reply = None
    a.timing.agent.start, a.timing.agent.end = 1.4, 3.6
    a.split_agent_time()
    a.timing.scoring = TimeSpan(start=3.6, end=3.7)
    a.stop("agent_completed")
    a.ok = True
    await streamer.flush()

    b = make_trace("judge", False)
    traces.append(b)
    b.nodes.append(MessageNode(message=UserMessage(content="judge this"), token_ids=[], mask=[]))
    b.record_error(RuntimeError("judge failed"))
    b.errors[-1].traceback = "Traceback: x"
    b.root_reply = " verdict "
    b.dropped = "no reward"
    await streamer.flush()

    episode = Episode(
        env=EnvInfo(id="my-env"),
        task=a.task,
        ok=False,
        errors=[vf.Error(type="RuntimeError", message="judge failed")],
        traces=[a, b],
    )
    return episode, frames


episode, frames = asyncio.run(build())
response = RunResponse(
    head=dump(episode, exclude={"traces"}),
    traces=[TraceSummary(id=t.id, nodes=len(t.nodes), calls=len(t.calls)) for t in episode.traces],
)
reply = msgpack.packb(response.model_dump(mode="python"), default=msgpack_encoder, use_bin_type=True)
(OUT / "episode_stream.msgpack").write_bytes(msgpack.packb({"deltas": frames, "reply": reply}, use_bin_type=True))

assembly = EpisodeAssembly()
for frame in frames:
    assembly.apply(unpack(frame))
response = RunResponse.model_validate(unpack(reply))
assembled = assembly.finish(response.head, response.traces)
(OUT / "episode_assembled.msgpack").write_bytes(pack(assembled))
received = WireEpisode.model_validate(assembled)


def derived(ep: Episode) -> dict:
    out = {
        "num_input_tokens": ep.num_input_tokens,
        "num_output_tokens": ep.num_output_tokens,
        "num_total_tokens": ep.num_total_tokens,
        "traces": [],
    }
    for t in ep.traces:
        out["traces"].append(
            {
                "reward": float(t.reward),
                "has_error": t.has_error,
                "is_truncated": t.is_truncated,
                "num_turns": t.num_turns,
                "num_branches": t.num_branches,
                "num_input_tokens": t.num_input_tokens,
                "num_output_tokens": t.num_output_tokens,
                "num_total_tokens": t.num_total_tokens,
                "last_reply": t.last_reply,
                "branches": [
                    {
                        "index": br.index,
                        "trainable": br.trainable,
                        "token_ids": br.token_ids.tolist(),
                        "sampled_mask": br.sampled_mask.tolist(),
                        "logprobs": br.logprobs.tolist(),
                        "advantages": None if br.advantages is None else br.advantages.tolist(),
                        "loss_weights_rl": None if br.loss_weights("rl") is None else br.loss_weights("rl").tolist(),
                        "mm_token_type_ids": None if br.mm_token_type_ids is None else br.mm_token_type_ids.tolist(),
                        "num_input_tokens": br.num_input_tokens,
                        "num_output_tokens": br.num_output_tokens,
                        "num_total_tokens": br.num_total_tokens,
                    }
                    for br in t.branches
                ],
            }
        )
    return out


(OUT / "episode_record_4.json").write_bytes(received.to_json())
(OUT / "episode_record_none.json").write_bytes(received.to_json(float_decimals=None))
(OUT / "episode_trace0_record_2.json").write_bytes(received.traces[0].to_json(float_decimals=2))
(OUT / "episode_derived.json").write_text(json.dumps(derived(received)))

# the orchestrator's mutations (replayed by tests/model.rs::mutate)
received.env.name = "my-env-name"
received.group = vf.GroupInfo(id="g1")
received.record_run(vf.TrainRunInfo(id="run1", work=vf.TrainWorkInfo(step=3, policy=vf.PolicySpan(start=1, end=2))))
a, b = received.traces
a.info["kind"] = "train"
a.info["arrival"] = {"step": 3, "time": 1.25}
a.nodes[4].advantages = np.full(2, 0.7, dtype=np.float32)
a.nodes[2].advantages = np.full(3, -1 / 3, dtype=np.float32)
a.nodes[4].loss_weights = {"rl": np.full(3, 1 / 3, dtype=np.float32)}
a.nodes[4].reference_logprobs = np.array([-0.3, -0.123456], dtype=np.float32)
b.errors.append(vf.Error(type="Timeout", message="Trace stopped by error"))
b.ok = False
received.ok = False
(OUT / "episode_mutated_record_4.json").write_bytes(received.to_json())
(OUT / "episode_mutated_derived.json").write_text(json.dumps(derived(received)))
