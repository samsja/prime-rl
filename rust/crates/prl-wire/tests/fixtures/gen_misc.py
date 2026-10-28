"""Fixtures for sampling dicts, task hashes and Python-json formatting.

uv run --no-sync python rust/crates/prl-wire/tests/fixtures/gen_misc.py
"""

import json
from pathlib import Path

import msgpack
import numpy as np

import verifiers.v1 as vf
from prime_rl.configs.orchestrator import TrainSamplingConfig
from verifiers.v1.task import task_key

OUT = Path(__file__).parent

cases = [
    {},
    {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "max_completion_tokens": 128, "logprobs": 5},
    {"extra_body": {"min_tokens": 3, "max_completion_tokens": 9, "chat_template_kwargs": {"enable_thinking": False}}},
    {"max_completion_tokens": 64, "extra_body": {"max_tokens": 10, "repetition_penalty": 1.1, "temperature": 0.2}},
]
sampling = []
for case in cases:
    config = TrainSamplingConfig(**case)
    for live in (True, False):
        vf_sampling = vf.SamplingConfig(
            **config.model_dump(exclude_none=True, exclude={"logprobs"}),
            logprobs=(config.logprobs or True) if live else None,
        )
        salted = vf_sampling.model_copy(update={"cache_salt": "abc-3"})
        sampling.append(
            {
                "input": config.model_dump(mode="json"),
                "live": live,
                "expected": vf_sampling.model_dump(mode="json"),
                "salted": salted.model_dump(mode="json"),
            }
        )
(OUT / "sampling.msgpack").write_bytes(msgpack.packb(sampling, use_bin_type=True))


class MyTask(vf.TaskData):
    answer: str
    meta: dict
    weight: float = 1.0


tasks = [
    vf.TaskData(idx=0, prompt="hi"),
    MyTask(
        idx=7,
        prompt=[vf.UserMessage(content="what is 2+2? é ✓ 🎉")],
        answer="4",
        meta={"x": None, "f": 1e-5, "big": 1e16},
    ),
]
hashes = [
    {"data": t.model_dump(mode="python"), "hash": task_key(t.model_dump(mode="json", exclude_none=True))} for t in tasks
]
(OUT / "task_hashes.msgpack").write_bytes(msgpack.packb(hashes, use_bin_type=True))

rng = np.random.default_rng(1)
floats = np.concatenate(
    [rng.standard_normal(500) * 10.0 ** rng.integers(-25, 25, 500), 10.0 ** np.arange(-8, 20), [0.0, -0.0, 1.0, 0.5]]
).tolist()
value = {"floats": floats, "s": "a\nb é ✓ 🎉 \x7f \x01", "b": [True, None], "z": 1, "a": {"k": 2}}
(OUT / "python_json_input.msgpack").write_bytes(msgpack.packb(value, use_bin_type=True))
(OUT / "python_json_expected.json").write_text(json.dumps(value, sort_keys=True))
