"""Random episode-record dicts and their Python index rows (`traces/index.py`), to check the
Rust summary arithmetic and key handling.

    uv run --no-sync python rust/crates/prl-monitors/tests/fixtures/gen_index_rows.py
"""

import random
from pathlib import Path

import orjson

from prime_rl.monitors.file.traces.index import index_row

rng = random.Random(7)


def num():
    return rng.choice([rng.random() * 10, rng.randint(0, 5), 0.1, 0.2, 1e-7, 0, 0.0, -1.5])


def maybe(value, p=0.8):
    return value() if rng.random() < p else None


def trace():
    t = {}
    if rng.random() < 0.9:
        t["id"] = rng.choice(["t" + str(rng.randint(0, 99)), ""])
    if rng.random() < 0.7:
        t["rewards"] = {f"r{i}": rng.choice([{"score": num(), "weight": maybe(num)}, {"score": num()}, {"score": None}, None]) for i in range(rng.randint(0, 3))}
    if rng.random() < 0.6:
        t["metrics"] = {f"m{i}": rng.choice([num(), None, "x"]) for i in range(rng.randint(0, 3))}
    if rng.random() < 0.7:
        t["calls"] = [{"usage": rng.choice([{"cost": num()}, {}, None, {"cost": None}]), "finish_reason": rng.choice(["stop", "length", None]), **({"error": {"t": 1}} if rng.random() < 0.2 else {})} for _ in range(rng.randint(0, 3))]
    if rng.random() < 0.8:
        n = rng.randint(0, 4)
        t["nodes"] = [{**({"parent": rng.randint(0, i - 1)} if i and rng.random() < 0.8 else {}), "sampled": rng.random() < 0.5} for i in range(n)]
    if rng.random() < 0.7:
        timing = {"start": 100.0 + num()}
        for span in ["boot", "setup", "agent", "scoring"]:
            if rng.random() < 0.6:
                timing[span] = {"start": num(), "end": num() + 5, **({"model": {"duration": num()}} if rng.random() < 0.3 else {})}
        if rng.random() < 0.2:
            timing["duration"] = num()
        t["timing"] = timing
    info = {}
    if rng.random() < 0.5:
        info["advantage"] = rng.choice([num(), None])
    if rng.random() < 0.5:
        info["kind"] = rng.choice(["train", "eval", "other"])
    if rng.random() < 0.5:
        info["dispatch"] = {"step": 1, "time": rng.choice([1000.0 + num(), 1000, None])}
        info["arrival"] = {"step": 2, "time": rng.choice([1010.0 + num(), 1010, 999.0])}
    if info or rng.random() < 0.5:
        t["info"] = info
    if rng.random() < 0.3:
        t["stop_condition"] = rng.choice(["max_turns", "done", None])
    if rng.random() < 0.2:
        t["is_timeout"] = rng.random() < 0.5
    return t


def record(i):
    rec = {"id": f"e{i}"}
    if rng.random() < 0.9:
        rec["traces"] = [trace() for _ in range(rng.randint(0, 3))]
    for key in ["num_input_tokens", "num_output_tokens"]:
        if rng.random() < 0.8:
            rec[key] = rng.randint(0, 1000)
    if rng.random() < 0.8:
        rec["env"] = rng.choice([{"id": "a", "name": "n"}, {"id": "a"}, {"name": ""}, {}])
    if rng.random() < 0.7:
        rec["group"] = {"id": "g"}
    if rng.random() < 0.8:
        rec["ok"] = rng.random() < 0.7
    if rng.random() < 0.5:
        rec["errors"] = [{}] * rng.randint(0, 2)
    if rng.random() < 0.5:
        rec["run"] = rng.choice([{"type": "train", "work": {"type": "train"}}, {"type": "eval"}, {"work": {}}])
    if rng.random() < 0.2:
        rec["is_timeout"] = True
    return rec


HERE = Path(__file__).parent
records = [record(i) for i in range(400)]
(HERE / "index_records.jsonl").write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in records))
rows = [index_row(i + 1, r, i // 10, 1000 * i) for i, r in enumerate(records)]
(HERE / "index_rows.jsonl").write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))
