"""A no-op trainer for orchestrator benchmarks.

Speaks the trainer side of the ZMQ batch transport and the filesystem weight broadcast:
broadcasts the startup policy v0, then for each step requests the step's micro batches,
and as soon as they arrive publishes policy v{step} (sentinel files only, no weights).
Appends one JSON line per received step to LOG.

    uv run --no-sync python rust/bench/trainer_stub.py OUTPUT_DIR PORT MAX_STEPS LOG
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import msgspec
import zmq

from prime_rl.transports.batch.types import MicroBatch


class BatchRequest(msgspec.Struct):
    world_size: int
    data_rank: int
    step: int


def broadcast(output_dir: Path, step: int) -> None:
    broadcasts = output_dir / "broadcasts"
    step_dir = broadcasts / f"step_{step}"
    shutil.rmtree(step_dir, ignore_errors=True)
    step_dir.mkdir(parents=True)
    (step_dir / ".sender_ready").touch()
    while not (step_dir / ".receiver_ready").exists():
        time.sleep(0.01)
    (step_dir / ".started").touch()
    (step_dir / ".finished").touch()
    for old in broadcasts.glob("step_*"):
        if int(old.name.removeprefix("step_")) < step - 1:
            shutil.rmtree(old, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("port", type=int)
    parser.add_argument("max_steps", type=int)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    socket = zmq.Context.instance().socket(zmq.DEALER)
    socket.connect(f"tcp://127.0.0.1:{args.port}")
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(list[MicroBatch])
    broadcast(args.output_dir, 0)
    with args.log.open("a") as log:
        for step in range(1, args.max_steps + 1):
            socket.send(encoder.encode(BatchRequest(world_size=1, data_rank=0, step=step)))
            payload = socket.recv()
            received = time.time()
            micro_batches = decoder.decode(payload)
            row = {
                "step": step,
                "time": received,
                "bytes": len(payload),
                "micro_batches": len(micro_batches),
                "samples": sum(len(mb.env_names) for mb in micro_batches),
                "tokens": sum(sum(mb.sequence_lengths) for mb in micro_batches),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            broadcast(args.output_dir, step)


if __name__ == "__main__":
    main()
