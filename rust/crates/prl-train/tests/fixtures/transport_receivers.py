"""Python trainer-side receivers for the prl-train transport interop test.

Usage: python transport_receivers.py {filesystem|zmq} OUTPUT_DIR PORT STEPS
Ranks 0..3 map to data ranks [0, 0, 1, 1]; prints one JSON line per (step, rank) with the sha256
of the decoded micro batches re-encoded by msgspec (equal to the sent bytes iff decoding is lossless).
"""

import hashlib
import json
import sys
import threading
from pathlib import Path

import msgspec

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.filesystem import FileSystemBatchReceiver
from prime_rl.transports.batch.zmq import ZMQBatchReceiver

RANK_TO_DATA_RANK = [0, 0, 1, 1]


def main():
    kind, output_dir, port, steps = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    world = len(RANK_TO_DATA_RANK)
    if kind == "filesystem":
        receivers = [FileSystemBatchReceiver(output_dir, d, r, world) for r, d in enumerate(RANK_TO_DATA_RANK)]
    else:
        transport = ZMQTransportConfig(host="127.0.0.1", port=port)
        receivers = [ZMQBatchReceiver(output_dir, d, r, world, 0, transport) for r, d in enumerate(RANK_TO_DATA_RANK)]
    lock = threading.Lock()

    def run(rank, receiver):
        for step in range(steps):
            receiver.wait()
            batches = receiver.receive()
            digest = hashlib.sha256(msgspec.msgpack.encode(batches)).hexdigest()
            with lock:
                print(json.dumps({"step": step, "rank": rank, "n": len(batches), "sha256": digest}), flush=True)

    threads = [threading.Thread(target=run, args=(r, rec)) for r, rec in enumerate(receivers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for rec in receivers:
        rec.close()


if __name__ == "__main__":
    main()
