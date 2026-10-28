# Imports the removed Python orchestrator: run it from a checkout that still has it (e.g. the baseline worktree at 8acab25e4).
"""Generate file-monitor fixtures: run the Python FileMonitor over wire-received episodes and
keep both its inputs (records, updates, live events) and its outputs.

    uv run --no-sync python rust/crates/prl-monitors/tests/fixtures/gen_file_monitor.py
"""

import asyncio
import json
import math
import shutil
import sys
from pathlib import Path

import msgpack
import numpy as np
import orjson
import verifiers.v1 as vf
from verifiers.v1.serve.delta import DeltaStreamer, EpisodeAssembly, TraceSummary, dump, pack, unpack
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import Usage

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from prime_rl.configs.monitors import FileMonitorConfig  # noqa: E402
from prime_rl.monitors.file.monitor import FileMonitor  # noqa: E402
from prime_rl.orchestrator.algo.routing import assign_advantages  # noqa: E402
from prime_rl.orchestrator.annotations import stamp_arrival, stamp_batch  # noqa: E402
from tests.unit.orchestrator.test_train_sink import episode, trace  # noqa: E402

OUT = Path(__file__).parent / "file_monitor"


async def _noop(_):
    return None


def over_the_wire(source):
    traces = source.traces
    assembly = EpisodeAssembly()
    deltas = []
    for _, delta, _ in DeltaStreamer(lambda: traces, _noop).diff():
        delta = unpack(pack(delta))
        assembly.apply(delta)
        deltas.append(delta)
    summaries = [TraceSummary(id=t.id, nodes=len(t.nodes), calls=len(t.calls)) for t in traces]
    head = dump(source, exclude={"traces"})
    return vf.WireEpisode.model_validate(assembly.finish(head, summaries)), deltas


def rich_trace(i):
    t = trace(advantage=0.25 * i, ok=i % 3 != 2)
    t.id = f"trace{i:02d}"
    t.rewards = {"correct": vf.Reward(score=float(i % 2)), "style": vf.Reward(score=0.5, weight=0.2)}
    t.metrics = {"len": 3.0 + i, "count": float(i)}
    t.calls = [
        ModelCall(
            node=2,
            usage=Usage(prompt_tokens=2, completion_tokens=2, cost=0.001 * (i + 1)),
            finish_reason="length" if i == 1 else "stop",
        )
    ]
    t.stop_condition = "max_turns" if i == 4 else None
    t.is_timeout = i == 5
    t.timing.start = 1000.0 + i
    t.timing.boot.start, t.timing.boot.end = 1000.0 + i, 1001.5 + i
    t.timing.scoring.start, t.timing.scoring.end = 1003.0 + i, 1003.25 + i
    t.info = {"advantage": 0.25 * i} if i % 2 else {}
    return t


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    out_dir = OUT / "run"
    received, live_events = [], []
    for i in range(6):
        traces = [rich_trace(i)] if i != 3 else [rich_trace(i), rich_trace(10 + i)]
        ep = episode(traces, group=f"g{i // 2}")
        ep.id = f"episode{i:02d}"
        got, deltas = over_the_wire(ep)
        received.append(got)
        dispatch = {
            "id": f"d{i}",
            "kind": "train",
            "env": "test",
            "group": f"g{i // 2}",
            "task": "idx=0",
            "policy_version": 0,
            "step": 1,
            "started": 1000.0 + i,
        }
        live_events.append({"pending": f"d{i}", "dispatch": dispatch})
        live_events.append({"dispatched": f"d{i}"})
        live_events.extend({"delta": delta, "dispatch": dispatch} for delta in deltas)
        if i % 2 == 0:
            live_events.append({"done": got.traces[0].id})
    live_events.append({"delta": {"trace": "gone", "discard": True}, "dispatch": {}})
    for got in received:
        for t in got.traces:
            assign_advantages(t, 0.7)
    stamp_arrival(received, "train", 2)

    monitor = FileMonitor(FileMonitorConfig(chunk_bytes=8000))
    asyncio.run(monitor.init(out_dir, producer="orch"))
    records = [orjson.dumps(ep.to_record(float_decimals=4), option=orjson.OPT_SERIALIZE_NUMPY) for ep in received]
    asyncio.run(monitor.log_episodes(received[:4], step=2, kind="train", subset="all"))
    asyncio.run(monitor.log_episodes(received[:2], step=2, kind="train", subset="effective"))
    asyncio.run(monitor.log_episodes(received[4:], step=3, kind="eval", subset="all"))
    updates = stamp_batch(received[:3], step=2)
    asyncio.run(monitor.log_annotations(updates))
    asyncio.run(monitor.log_live(live_events))
    asyncio.run(monitor.log_eval_plan("gsm8k", 2, 64))
    asyncio.run(monitor.log_eval_plan("gsm8k", 4, 64))
    asyncio.run(monitor.log_eval_plan("aime", 4, 30))
    asyncio.run(
        monitor.log_metrics(
            {
                "loss/mean": 0.5,
                "progress/total_tokens": 1234,
                "tiny": 1e-05,
                "big": 1e16,
                "nan": math.nan,
                "inf/x": math.inf,
                "flag": True,
                "name": "é😀",
            },
            step=2,
        )
    )
    asyncio.run(monitor.log_metrics({"inference/agg/x": 3.0}, step=None))
    asyncio.run(monitor.finalize())

    (OUT / "records.jsonl").write_bytes(b"".join(r + b"\n" for r in records))
    (OUT / "updates.jsonl").write_bytes(
        b"".join(orjson.dumps(u, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n" for u in updates)
    )
    (OUT / "live_events.msgpack").write_bytes(msgpack.packb(live_events, use_bin_type=True))


if __name__ == "__main__":
    main()
