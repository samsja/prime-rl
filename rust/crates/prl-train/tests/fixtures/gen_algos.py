# Imports the removed Python orchestrator: run it from a checkout that still has it (e.g. the baseline worktree at 8acab25e4).
"""Generate prl-train algorithm fixtures for the algorithms that call out (opd, opsd, echo with a
user filter): episodes before/after scoring with deterministic fake teachers, renderer and filter.

Run: uv run --no-sync python rust/crates/prl-train/tests/fixtures/gen_algos.py
"""

import asyncio
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gen_sink import Builder  # noqa: E402

from prime_rl.configs.algorithm import EchoAlgoConfig, OPDAlgoConfig, OPSDAlgoConfig  # noqa: E402
from prime_rl.orchestrator.algo.echo import EchoAlgorithm  # noqa: E402
from prime_rl.orchestrator.algo.opd import OPDAlgorithm  # noqa: E402
from prime_rl.orchestrator.algo.opsd import OPSDAlgorithm  # noqa: E402


class FakeScorer:
    """Prefill logprobs: -(token % 97) / 64 per position (exact in float32)."""

    model_name = "policy"

    async def score(self, token_ids):
        return [-(t % 97) / 64 for t in token_ids]


class FakeRenderer:
    def render_ids(self, messages, add_generation_prompt=False):
        content = messages[0]["content"]
        return [9000 + len(content) % 7, 9001, 9002]


def record(episode):
    return json.loads(episode.to_json(float_decimals=None))


async def main():
    rng = random.Random(7)
    b = Builder(rng)
    cases = []
    teacher = {"name": "teacher", "base_url": "http://teacher:1/v1"}
    for i in range(12):
        episodes = [b.episode("env", f"g{i}", 1, 0, hierarchical=rng.random() < 0.3) for _ in range(rng.randint(1, 3))]
        for episode in episodes:
            episode.ok = True
            for trace in episode.traces:
                if rng.random() < 0.8:
                    trace.info["demonstration"] = f"demo {{x}} {i}-{trace.id[:4]}"
                else:
                    trace.info.pop("demonstration", None)
                    trace.ok = False  # opsd requires a demonstration on every trainable trace
                    trace.errors = []
        before = [record(e) for e in episodes]

        opd = OPDAlgorithm(OPDAlgoConfig(teacher=teacher), None)
        opd.teacher_clients = FakeScorer()
        opd_eps = [e.model_copy(deep=True) for e in episodes]
        for e in opd_eps:
            await opd.finalize_episode(e)

        opsd = OPSDAlgorithm(OPSDAlgoConfig(template="Demo:\n{demonstration}\n{{end}}"), FakeScorer())
        opsd.renderer = FakeRenderer()
        opsd_eps = [e.model_copy(deep=True) for e in episodes]
        for e in opsd_eps:
            await opsd.finalize_episode(e)

        masks_by_trace = {}

        def echo_filter(trace, keep_every=2):
            from prime_rl.orchestrator.trajectories import iter_trainable_branches

            masks = [
                [(j % keep_every) == 0 for j in range(len(branch.token_ids))]
                for branch, _ in iter_trainable_branches(trace)
            ]
            masks_by_trace[trace.id] = masks
            return masks

        echo = EchoAlgorithm(EchoAlgoConfig(roles={"tool": {"alpha": 0.25}, "user": {"alpha": 0.5}}), None)
        echo.filter_fn = echo_filter
        echo_eps = [e.model_copy(deep=True) for e in episodes]
        for e in echo_eps:
            await echo.finalize_episode(e)
        await echo.score_group(echo_eps)

        cases.append(
            {
                "before": before,
                "opd": [record(e) for e in opd_eps],
                "opsd": [record(e) for e in opsd_eps],
                "echo": [record(e) for e in echo_eps],
                "echo_masks": masks_by_trace,
            }
        )
    out = Path(__file__).parent / "algo_cases.json.gz"
    out.write_bytes(gzip.compress(json.dumps(cases).encode(), 9))
    print(f"wrote {len(cases)} cases to {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
