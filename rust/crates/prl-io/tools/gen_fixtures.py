"""Generate prl-io test fixtures from the Python reference implementation.

Run from the repo root: uv run --no-sync python rust/crates/prl-io/tools/gen_fixtures.py
"""

import json
import random
from pathlib import Path

from prime_rl.orchestrator.ckpt import CheckpointManager
from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.types import Progress

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def state_json(state: dict) -> dict:
    """JSON view of a checkpoint state that the Rust test compares against."""

    def rng(s):
        version, words, gauss = s
        return {"version": version, "key": list(words[:624]), "pos": words[624], "gauss_next": gauss}

    def sampler(s):
        if "cursor" in s:
            return {"cursor": s["cursor"]}
        return {"rng": rng(s["rng"]), "task_rewards": s["task_rewards"]}

    p = state["progress"]
    return {
        "progress": {k: getattr(p, k) for k in ("step", "total_tokens", "total_samples", "total_problems", "rollout_window")},
        "train_source": {
            "rng": rng(state["train_source"]["rng"]),
            "envs": {
                name: {"sampler": sampler(c["sampler"]), "gates": c["gates"]}
                for name, c in state["train_source"]["envs"].items()
            },
        },
    }


class FakeTrainSource:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state


def checkpoints() -> None:
    source_rng = random.Random(42)
    for _ in range(7):
        source_rng.random()
    pool_rng = random.Random(1234)
    pool_rng.gauss(0, 1)  # leaves gauss_next set
    state = {
        "rng": source_rng.getstate(),
        "envs": {
            "math": {"sampler": {"cursor": 12345}, "gates": {}},
            "code": {
                "sampler": {"rng": pool_rng.getstate(), "task_rewards": {"t-1": 0.5, "t-2": 1.0, "t-3": 0.0}},
                "gates": {"adv": {}},
            },
        },
    }
    progress = Progress(step=17, total_tokens=2**40 + 3, total_samples=4096, total_problems=512, rollout_window=3)
    out = FIXTURES / "ckpt_run"
    manager = CheckpointManager(out, CheckpointConfig())
    manager.save(progress, FakeTrainSource(state), step=16)
    expected = state_json({"progress": progress, "train_source": state})
    (FIXTURES / "ckpt_expected.json").write_text(json.dumps(expected, indent=1))


def pyrandom() -> None:
    cases = []
    for seed in (42, 0, 1, 2**32 - 1, 2**32, 2**40 + 7):
        r = random.Random(seed)
        draws = [r.random() for _ in range(5)]
        weights = [1.0, 0.0, 2.5, 0.5]
        picks = [random.Random(seed).choices(range(4), weights=weights, k=1)[0]]
        r2 = random.Random(seed)
        picks = [r2.choices(range(4), weights=weights, k=1)[0] for _ in range(50)]
        state_after = r2.getstate()
        cases.append(
            {
                "seed": seed,
                "draws": draws,
                "weights": weights,
                "picks": picks,
                "state_after": {"key": list(state_after[1][:624]), "pos": state_after[1][624]},
            }
        )
    (FIXTURES / "pyrandom.json").write_text(json.dumps(cases))


if __name__ == "__main__":
    FIXTURES.mkdir(parents=True, exist_ok=True)
    checkpoints()
    pyrandom()
