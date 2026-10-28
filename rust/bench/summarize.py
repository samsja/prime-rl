"""Render results.jsonl as the markdown table used in RESULTS.md.

    uv run --no-sync python rust/bench/summarize.py /tmp/prl-orch-bench/results.jsonl
"""

import argparse
import json
from pathlib import Path


def envcol(r: dict) -> str:
    return f"fake {r['env_latency_ms']} ms" if r.get("env") == "fake" else "real"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--names", nargs="*", help="only these run names, in this order")
    args = parser.parse_args()
    rows = {}
    for line in args.results.read_text().splitlines():
        result = json.loads(line)
        if result["impl"] != "env-load":
            rows[result["name"]] = result
    names = args.names or list(rows)
    print(
        "| orchestrator | max_inflight | batch | env | rollouts/s | steps/s | avg in flight | CPU % | RSS max (GB) "
        "| loop lag p99 (ms) | loop lag max (ms) | time/pack (s) | time/step (s) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        r = rows[name]
        m = r.get("metrics", {})
        lag99, lagmax = m.get("event_loop_lag/p99"), m.get("event_loop_lag/max")
        print(
            f"| {r['impl']} | {r['max_inflight']} | {r['batch_size']} | {envcol(r)} "
            f"| {r['samples_per_s']:.0f} | {r['steps_per_s']:.3f} | {m.get('dispatcher/inflight/train', 0):.0f} "
            f"| {r['orchestrator'].get('cpu_pct', 0):.0f} | {r['orchestrator'].get('rss_mb_max', 0) / 1024:.1f} "
            f"| {1000 * lag99:.1f} | {1000 * lagmax:.1f} | {m.get('time/pack', float('nan')):.3f} | {m.get('time/step', float('nan')):.2f} |"
            if lag99 is not None and lagmax is not None
            else f"| {r['impl']} | {r['max_inflight']} | {r['batch_size']} | {envcol(r)} | {r['samples_per_s']:.0f} "
            f"| {r['steps_per_s']:.3f} | {m.get('dispatcher/inflight/train', 0):.0f} | {r['orchestrator'].get('cpu_pct', 0):.0f} "
            f"| {r['orchestrator'].get('rss_mb_max', 0) / 1024:.1f} | - | - | {m.get('time/pack', float('nan')):.3f} | {m.get('time/step', float('nan')):.2f} |"
        )


if __name__ == "__main__":
    main()
