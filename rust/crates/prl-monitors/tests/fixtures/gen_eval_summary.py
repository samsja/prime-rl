"""`prime_runs.metrics.from_episodes` over the file-monitor fixture records.

uv run --no-sync python rust/crates/prl-monitors/tests/fixtures/gen_eval_summary.py
"""

import json
from pathlib import Path

import prime_runs as pr
import verifiers.v1 as vf

HERE = Path(__file__).parent
episodes = [
    vf.WireEpisode.model_validate_json(line) for line in (HERE / "file_monitor/records.jsonl").read_bytes().splitlines()
]
(HERE / "eval_summary.json").write_text(json.dumps(pr.metrics.from_episodes(episodes)))
