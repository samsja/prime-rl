"""Run run_bench.py over a grid: each orchestrator at each max_inflight (batch size =
max_inflight unless --batch-size is given). Results accumulate in <work-dir>/results.jsonl.

    uv run --no-sync python rust/bench/run_matrix.py --inflight 256 1024 4096 16384
"""

import argparse
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impls", nargs="+", default=["python", "rust"])
    parser.add_argument("--inflight", nargs="+", type=int, default=[256, 1024, 4096, 16384])
    parser.add_argument("--batch-size", type=int, default=None)
    args, rest = parser.parse_known_args()
    for inflight in args.inflight:
        for impl in args.impls:
            batch = args.batch_size or inflight
            cmd = [sys.executable, BENCH / "run_bench.py", impl, str(inflight), str(batch), *rest]
            print(" ".join(map(str, cmd)), flush=True)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"run failed with {result.returncode}", flush=True)


if __name__ == "__main__":
    main()
