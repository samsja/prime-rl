"""Generate prl-train packer/encoder fixtures: random TrainingSamples, the Python packer's
per-rank encoded bytes. Run: uv run --no-sync python rust/crates/prl-train/tests/fixtures/gen_packer.py"""

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.transports.batch.base import BatchSender
from prime_rl.transports.batch.types import MMImageRef, MMRefs, PayloadSlice, TrainingSample


class Sender(BatchSender):
    async def send(self, grid):
        raise NotImplementedError


def flop_cost(seq_lens):
    return sum(64 * n + n * n for n in seq_lens)


def big_cost(seq_lens):
    # exceeds int64 when summed over a batch
    return sum(n * 4 * 10**16 + n * n * 10**12 for n in seq_lens)


COSTS = {"sum": sum, "flop": flop_cost, "big": big_cost}


def q(x):
    """Quantize to multiples of 1/256 so JSON floats are short and exact in f32."""
    return np.round(np.asarray(x) * 256) / 256


def random_sample(rng, seq_len, case):
    length = int(rng.integers(1, seq_len + seq_len // 2 + 2))
    prompt = int(rng.integers(0, length))
    mask = np.array([False] * prompt + [True] * (length - prompt))
    if rng.random() < 0.2:
        mask = rng.random(length) < 0.5
    tokens = rng.integers(0, 1000, size=length).astype(np.int32)
    s = TrainingSample(
        token_ids=tokens,
        mask=mask,
        logprobs=q(np.where(mask, -rng.random(length), 0.0)).astype(np.float32),
        temperatures=np.full(length, float(rng.choice([0.7, 1.0])), np.float32),
        env_name=f"env-{int(rng.integers(0, 3))}",
        advantages=q(np.where(mask, rng.normal(), 0.0)).astype(np.float32),
    )
    if rng.random() < 0.2:
        s.ref_logprobs = q(-rng.random(length)).astype(np.float32)
    if rng.random() < 0.2:
        s.rl_weights = q(rng.random(length)).astype(np.float32)
    if rng.random() < 0.2:
        s.ce_weights = (rng.random(length) < 0.3).astype(np.float32) * 0.125
    if rng.random() < 0.1:
        s.ref_kl_weights = mask.astype(np.float32)
    if rng.random() < 0.5:
        s.staleness = np.where(mask, rng.integers(0, 5, length), -1).astype(np.int16)
    if rng.random() < 0.6:
        s.trace_id = f"trace-{case}-{int(rng.integers(0, 10**6))}"
        s.branch_index = int(rng.integers(0, 3))
    if rng.random() < 0.3:
        routed = case % 3 == 0
        s.payload = PayloadSlice(
            url="http://127.0.0.1:4242",
            id=f"p{int(rng.integers(0, 10**6))}",
            num_tokens=length,
            fields=["routed_experts", "sampling_mask"] if routed else ["sampling_mask"],
            routed_shape=[4, 2] if routed else None,
            routed_dtype="uint8" if routed else None,
        )
    if rng.random() < 0.1 and length >= 4:
        mm = np.zeros(length, np.int32)
        images = []
        start = 1
        while start + 2 <= length and len(images) < 2:
            mm[start : start + 2] = 1
            images.append(MMImageRef(url=f"data:image/png;base64,{case}-{start}", offset=start, length=2))
            start += 3
        s.mm_token_type_ids = mm
        s.mm_refs = MMRefs(images=images)
        s.advantages = np.where(mm == 1, 0.0, s.advantages).astype(np.float32)
    if rng.random() < 0.05:
        s.advantages = None
        s.mask = np.zeros(length, bool)
    return s


def to_json(s):
    def arr(a):
        return None if a is None else a.tolist()

    return {
        "token_ids": s.token_ids.tolist(),
        "mask": s.mask.tolist(),
        "logprobs": s.logprobs.tolist(),
        "temperatures": s.temperatures.tolist(),
        "env_name": s.env_name,
        "ref_logprobs": arr(s.ref_logprobs),
        "mm_refs": None if s.mm_refs is None else [[i.url, i.offset, i.length] for i in s.mm_refs.images],
        "mm_token_type_ids": arr(s.mm_token_type_ids),
        "rl_weights": arr(s.rl_weights),
        "ce_weights": arr(s.ce_weights),
        "ref_kl_weights": arr(s.ref_kl_weights),
        "advantages": arr(s.advantages),
        "trace_id": s.trace_id,
        "branch_index": s.branch_index,
        "staleness": arr(s.staleness),
        "payload": None
        if s.payload is None
        else {
            "url": s.payload.url,
            "id": s.payload.id,
            "num_tokens": s.payload.num_tokens,
            "fields": s.payload.fields,
            "routed_shape": s.payload.routed_shape,
            "routed_dtype": s.payload.routed_dtype,
        },
    }


def main():
    rng = np.random.default_rng(1234)
    sender = Sender(Path("/tmp"), data_world_size=1)
    cases = []
    for case in range(60):
        seq_len = int(rng.choice([8, 16, 32, 48]))
        workers = int(rng.choice([1, 2, 3, 4, 8]))
        pad = int(rng.choice([1, 1, 4, 6]))
        cost = ["sum", "flop", "big"][case % 3]
        samples = [random_sample(rng, seq_len, case) for _ in range(int(rng.integers(1, 20)))]
        inputs = [to_json(s) for s in samples]
        grid = prepare_batch(samples, seq_len=seq_len, num_train_workers=workers, bin_cost=COSTS[cost], pad_to_multiple_of=pad)
        cases.append(
            {
                "seq_len": seq_len,
                "num_train_workers": workers,
                "pad_to_multiple_of": pad,
                "cost": cost,
                "samples": inputs,
                # Full bytes for the first cases (readable diffs), digests for the rest (fixture size).
                "ranks": [base64.b64encode(sender.encode(mbs)).decode() for mbs in grid] if case < 12 else None,
                "rank_sha256": [hashlib.sha256(sender.encode(mbs)).hexdigest() for mbs in grid],
            }
        )
    out = Path(__file__).parent / "packer_cases.json"
    out.write_text(json.dumps(cases))
    print(f"wrote {len(cases)} cases to {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
