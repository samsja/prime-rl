"""Generate prl-eval resume fixtures from the Python implementation.

Run from the repo root: uv run --no-sync python rust/crates/prl-eval/tests/fixtures/gen_resume.py
"""

import copy
import json
import shutil
import tempfile
import uuid
from pathlib import Path

import orjson

from prime_rl.configs.eval import EvalConfig
from prime_rl.eval import resume
from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl
from prime_rl.utils.pathing import get_file_monitor_dir

OUT = Path(__file__).parent


def record(env: str, key: str, *, ok: bool = True) -> dict:
    return {
        "id": key,
        "env": {"id": env, "name": env},
        "task": {"type": "Task", "data": {"idx": 0}, "key": key, "hash": key},
        "group": {"id": f"group-{key}"},
        "ok": ok,
        "traces": [],
    }


def config_cases() -> list[dict]:
    """Saved vs current configs and the error Python raises (None = compatible)."""
    original = {"source": [{"env": {"id": "single_agent"}}], "run": {"name": "resume-test"}}
    updates_list = [
        {},
        {"model": "different-model"},
        {"sampling": {"temperature": 0.3}},
        {"select": {"limit": 2}},
        {"group_size": 2},
        {"client": {"base_url": "https://different.invalid/v1"}},
        {"env": {"timeout": {"episode": 60}}},
        {"log": {"level": "debug"}, "dashboard": False},
        {"concurrency": {"min_inflight": 8, "max_inflight": 8}, "dispatch_per_minute": 10},
        {"output_dir": "./another-output-location"},
    ]
    cases = []
    saved = EvalConfig.model_validate_json(orjson.dumps(original))
    for updates in updates_list:
        if "client" in updates:
            updates = updates | {"client": saved.client.model_dump() | updates["client"]}
        current = EvalConfig.model_validate(original | updates | {"resume": {}})
        cases.append({"saved": saved.model_dump(mode="json"), "current": current.model_dump(mode="json")})
    two = EvalConfig(
        source=[{"env": {"id": "single_agent"}}, {"env": {"id": "single_agent"}, "name": "other", "group_size": 2}],
        sampling={"temperature": 1.0},
    )
    reordered = two.model_dump(mode="json")
    reordered["source"].reverse()
    cases.append({"saved": two.model_dump(mode="json"), "current": EvalConfig.model_validate(reordered | {"resume": {}}).model_dump(mode="json")})
    for case in cases:
        tmp = Path(tempfile.mkdtemp())
        resume.stamp_config(tmp, case["saved"])
        current = EvalConfig.model_validate(copy.deepcopy(case["current"]))
        try:
            resume.take_landed(tmp, current)
            case["error"] = None
        except ValueError as e:
            case["error"] = str(e).replace(str(tmp), "<RUN>")
        shutil.rmtree(tmp)
    return cases


def streams() -> None:
    """A run dir with one archived attempt (sealed zstd chunks) and a current attempt with a torn tail."""
    run = OUT / "resume_run"
    shutil.rmtree(run, ignore_errors=True)
    config = EvalConfig(source=[{"env": {"id": "single_agent"}}], run={"name": "resume-test"}, resume={})
    dump = config.model_dump(mode="json")
    (OUT / "resume_config.json").write_text(json.dumps(dump))

    resume.stamp_config(run, dump)
    stream = ChunkedJsonl(get_trace_stream(run), max_bytes=600, compress=True)
    for i in range(12):
        stream.append(orjson.dumps(record("math", f"a{i}", ok=i % 3 != 0), option=orjson.OPT_APPEND_NEWLINE))
    no_traces = record("math", "a-no-traces")
    del no_traces["traces"]
    stream.append(orjson.dumps(no_traces, option=orjson.OPT_APPEND_NEWLINE))
    stream.close()
    directory = get_file_monitor_dir(run)
    directory.rename(directory.with_name("file.attempt_1"))

    resume.stamp_config(run, dump)
    stream = ChunkedJsonl(get_trace_stream(run), max_bytes=1 << 20, compress=False)
    for key in ["b0", "b1"]:
        stream.append(orjson.dumps(record("math", key), option=orjson.OPT_APPEND_NEWLINE))
    stream.append(b'{"id": "torn", "ok": tr')
    stream.close()

    work = Path(tempfile.mkdtemp()) / "run"
    shutil.copytree(run, work)
    landed = resume.take_landed(work, config)
    expected = {
        "landed": [episode.id for episode in landed],
        "archives": [path.name for path in resume.archives(work)],
        "stamp": (get_file_monitor_dir(run) / resume.CONFIG_NAME).read_text(),
    }
    (OUT / "resume_expected.json").write_text(json.dumps(expected, indent=2))


if __name__ == "__main__":
    (OUT / "resume_config_cases.json").write_text(json.dumps(config_cases(), indent=1))
    streams()
