"""Training arrays kept by env servers: the trainer must assemble each branch's arrays,
cut, packed and padded like its tokens, and the store must serve and free them."""

import base64
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.episode import EnvInfo, Episode
from verifiers.v1.types import SamplingMask, TopLogprobs, TurnTokens

from prime_rl.trainer.rl.data import DataLoader
from prime_rl.transports import payload
from prime_rl.transports.batch import types as wire
from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY, MicroBatch, PayloadSlice, encode_array
from prime_rl.transports.payload import (
    PayloadHook,
    PayloadServer,
    delete_payloads,
    payload_host,
    resolve_payloads,
    wait_for_deletes,
)


def _episode(
    rng: np.random.Generator, prompt_len: int, completion_len: int, *, routed: bool = True, masked: bool = False
) -> vf.Episode:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    rows = rng.integers(0, 128, size=(prompt_len + completion_len - 1, 3, 2)).astype(np.uint8)
    counts = rng.integers(1, 5, size=completion_len).astype(np.int32)
    head = TopLogprobs(
        ids=rng.integers(0, 1000, size=int(counts.sum())).astype(np.int32),
        logprobs=-rng.random(int(counts.sum())).astype(np.float32),
        counts=counts,
    )
    graph.prepare_turn(trace, [vf.UserMessage(content="u")]).commit(
        vf.Response(
            id="a",
            created=0,
            model="t",
            message=vf.AssistantMessage(content="a"),
            finish_reason="stop",
            tokens=TurnTokens.model_construct(
                prompt_ids=list(range(prompt_len)),
                completion_ids=list(range(completion_len)),
                completion_logprobs=[-0.5] * completion_len,
                message_spans=[(0, prompt_len)],
                routed_experts={
                    "data": base64.b64encode(rows.tobytes()).decode(),
                    "shape": list(rows.shape),
                    "start": 0,
                }
                if routed
                else None,
                top_logprobs=head,
                # A mask the head covers exactly carries the head's logprobs.
                sampling_mask=SamplingMask(ids=head.ids, counts=head.counts) if masked else None,
            ),
        )
    )
    trace.stop("agent_completed")
    trace.ok = True
    return Episode(env=EnvInfo(id="e"), task=trace.task, ok=True, traces=[trace])


def _micro_batch(lengths: list[int], payloads: list[PayloadSlice | None], padding: int = 0) -> MicroBatch:
    """A packed micro batch as the trainer receives it: ``lengths`` per sequence, the last one
    extended by ``padding``, each sequence referencing its kept arrays."""
    lengths = [*lengths[:-1], lengths[-1] + padding]
    n = sum(lengths)
    micro_batch = MicroBatch(
        input_ids=np.zeros(n, np.int32),
        loss_mask=np.ones(n, np.bool_),
        advantages=np.zeros(n, np.float32),
        inference_logprobs=np.zeros(n, np.float32),
        position_ids=np.zeros(n, np.int32),
        sequence_lengths=lengths,
        temperatures=np.ones(n, np.float32),
        env_names=["env"] * len(lengths),
        seq_lens=lengths,
        padding=padding,
        payloads=payloads,
    )
    return msgspec.msgpack.decode(msgspec.msgpack.encode(micro_batch, enc_hook=encode_array), type=MicroBatch)


def test_heads_and_masks_survive_truncation_mixed_packing_and_padding():
    """Explicit values: a top-k head, a mask with sampler logprobs and a plain mask reach
    the trainer cut to their slices, packed together and padded."""
    server = PayloadServer("127.0.0.1")

    def ref(key: str, fields: list[str], num_tokens: int = 4, **arrays) -> PayloadSlice:
        server.put(key, payload._encoder.encode(wire.BranchArrays(**arrays)))
        return PayloadSlice(url=server.url, id=key, num_tokens=num_tokens, fields=fields)

    head = wire.TopLogprobs(
        ids=np.array([3, 7, 4], np.int32).tobytes(),
        logprobs=np.array([-0.3, -1.8, -0.2], np.float32).tobytes(),
        counts=np.array([0, 0, 2, 1], np.int32).tobytes(),
    )
    replayed = wire.SamplingMask(
        ids=np.array([5, 6, 8], np.int32).tobytes(),
        counts=np.array([0, 0, 1, 2], np.int32).tobytes(),
        logprobs=np.log(np.array([1.0, 0.75, 0.25], np.float32)).tobytes(),
    )
    plain = wire.SamplingMask(ids=np.array([9], np.int32).tobytes(), counts=np.array([0, 0, 0, 1], np.int32).tobytes())
    payloads = [
        # Truncated to 3 of its 4 tokens: the head's last row is cut.
        ref("head", ["top_logprobs"], num_tokens=3, top_logprobs=head),
        ref("replayed", ["sampling_mask", "sampling_mask_logprobs"], sampling_mask=replayed),
        ref("plain", ["sampling_mask"], sampling_mask=plain),
        # No arrays at all: it shares the micro batch and gets empty rows.
        None,
    ]
    packed = _micro_batch([3, 4, 4, 4], payloads, padding=1)
    with ThreadPoolExecutor(2) as pool:
        assert resolve_payloads([packed], pool)[0] == []
    assert packed.payloads is None
    assert len(packed.top_logprobs.ids) == 2 * 4
    tensor = DataLoader._micro_batch_to_tensor(None, packed)
    ids, values = tensor["top_logprobs_ids"][0], tensor["top_logprobs_values"][0]
    assert ids.shape == values.shape
    assert ids.shape[0] == sum(packed.sequence_lengths)
    rows = {
        tuple(row[row >= 0].tolist()): value[row >= 0].tolist() for row, value in zip(ids, values) if (row >= 0).any()
    }
    expected = {(3, 7): [-0.3, -1.8], (5,): [0.0], (6, 8): np.log([0.75, 0.25]), (9,): [np.nan]}
    assert rows.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_allclose(rows[key], value, rtol=1e-6)


def test_unfetchable_micro_batches_are_reported():
    server = PayloadServer("127.0.0.1")
    missing = PayloadSlice(
        url=server.url,
        id="missing",
        num_tokens=16,
        fields=["routed_experts"],
        routed_shape=[3, 2],
        routed_dtype="uint8",
    )
    micro_batch = _micro_batch([16], [missing])
    with ThreadPoolExecutor(1) as pool:
        assert resolve_payloads([micro_batch], pool)[0] == [0]
    # Still replayable: the model's routing shape, every top-k slot a different expert.
    routed = micro_batch.routed_experts
    assert routed.shape == [16, 3, 2]
    rows = np.frombuffer(routed.data, dtype=routed.dtype).reshape(routed.shape)
    assert (rows[..., 0] != rows[..., 1]).all()


@pytest.mark.parametrize("support", ["complete", "mismatch"])
def test_sampler_evidence_moves_masked_top_logprobs_onto_the_mask(support):
    import verifiers.v1 as vf

    from prime_rl.transports.payload import _encode_sampler_evidence

    mask = vf.SamplingMask(ids=np.array([5, 6, 8], np.int32), counts=np.array([0, 0, 1, 2], np.int32))
    head = vf.TopLogprobs(
        ids=np.array([1, 2, 5, 8, 6 if support == "complete" else 7], np.int32),
        logprobs=np.array([-0.1, -2.5, 0.0, -0.3, -1.4], np.float32),
        counts=np.array([0, 2, 1, 2], np.int32),
    )
    if support == "mismatch":
        with pytest.raises(ValueError):
            _encode_sampler_evidence(mask, head, 4)
        return
    sampling_mask, top_logprobs = _encode_sampler_evidence(mask, head, 4)
    np.testing.assert_array_equal(np.frombuffer(sampling_mask.ids, np.int32), [5, 8, 6])
    np.testing.assert_array_equal(np.frombuffer(sampling_mask.counts, np.int32), [0, 0, 1, 2])
    np.testing.assert_array_equal(np.frombuffer(sampling_mask.logprobs, np.float32), np.float32([0.0, -0.3, -1.4]))
    np.testing.assert_array_equal(np.frombuffer(top_logprobs.ids, np.int32), [1, 2])
    np.testing.assert_array_equal(np.frombuffer(top_logprobs.logprobs, np.float32), np.float32([-0.1, -2.5]))
    np.testing.assert_array_equal(np.frombuffer(top_logprobs.counts, np.int32), [0, 2, 0, 0])


def _two_trace_episode(rng: np.random.Generator) -> Episode:
    first, second = _episode(rng, 6, 8), _episode(rng, 6, 8)
    return Episode(env=first.env, task=first.task, ok=True, traces=[first.traces[0], second.traces[0]])


def _failing_second_encode(monkeypatch, error: Exception) -> None:
    encode = payload.encode_branch_arrays
    calls = []

    def flaky(branch, num_tokens):
        calls.append(branch)
        if len(calls) == 2:
            raise error
        return encode(branch, num_tokens)

    monkeypatch.setattr(payload, "encode_branch_arrays", flaky)


def test_keep_frees_what_it_kept_when_it_fails(monkeypatch):
    hook = PayloadHook()
    episode = _two_trace_episode(np.random.default_rng(3))
    _failing_second_encode(monkeypatch, MemoryError())
    with pytest.raises(MemoryError):
        hook.keep(episode, vf.SamplingConfig())
    assert not hook.server.blobs


def test_keep_keeps_routing_when_the_sampler_evidence_does_not_encode(monkeypatch):
    """The episode still succeeds (eval never needs the evidence) and keeps its routing
    rows; a train run that needs the mask or head stops at the train sink."""
    hook = PayloadHook()
    episode = _two_trace_episode(np.random.default_rng(4))
    _failing_second_encode(monkeypatch, ValueError("Top logprobs are not aligned to branch tokens"))
    hook.keep(episode, vf.SamplingConfig())
    first, second = episode.traces
    assert first.info[PAYLOAD_INFO_KEY]["branches"]["0"]["fields"] == ["routed_experts", "top_logprobs"]
    assert second.info[PAYLOAD_INFO_KEY]["branches"]["0"]["fields"] == ["routed_experts"]
    assert all(node.routed_experts is None for node in second.nodes)


def test_a_hook_that_keeps_nothing_starts_no_server():
    """A pool broker installs the hook too; it must not open a payload server."""
    hook = PayloadHook()
    hook.discard(_episode(np.random.default_rng(5), 4, 4))
    assert hook._server is None


def test_fetches_bypass_http_proxies(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    server = PayloadServer("127.0.0.1")
    server.put("a", payload._encoder.encode(wire.BranchArrays()))
    assert payload._fetch(f"{server.url}/a") == wire.BranchArrays()


def test_short_routing_rows_fail_the_alignment_check():
    rows = wire.RoutedExperts(data=bytes(3 * 2 * 2), shape=[4, 2, 2], dtype="uint8")
    mb = SimpleNamespace(routed_experts=rows, sampling_mask=None, top_logprobs=None)
    with pytest.raises(AssertionError, match="bytes"):
        payload._assert_aligned(mb, num_tokens=4)


def test_payload_server_binds_the_env_server_host():
    assert payload_host(None) == ("127.0.0.1", "127.0.0.1")
    assert payload_host("tcp://10.1.2.3:0") == ("10.1.2.3", "10.1.2.3")
    bind, url = payload_host("tcp://*:5000")
    assert bind == "0.0.0.0" and not url.startswith("127.")


def test_store_serves_and_deletes():
    server = PayloadServer("127.0.0.1")
    server.put("a", b"x" * 40)
    server.put("b", b"y" * 40)
    with urllib.request.urlopen(f"{server.url}/b") as response:
        assert response.read() == b"y" * 40
    delete_payloads([f"{server.url}/b", f"{server.url}/missing"])
    wait_for_deletes()
    assert server.get("b") is None and server.get("a") is not None


def test_discard_frees_what_keep_kept():
    hook = PayloadHook()
    episode = _episode(np.random.default_rng(2), 6, 8)
    hook.keep(episode, vf.SamplingConfig())
    assert hook.server.blobs
    hook.discard(episode)
    assert not hook.server.blobs
