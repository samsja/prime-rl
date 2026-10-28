# Imports the removed Python orchestrator: run it from a checkout that still has it (e.g. the baseline worktree at 8acab25e4).
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
        "progress": {
            k: getattr(p, k) for k in ("step", "total_tokens", "total_samples", "total_problems", "rollout_window")
        },
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


def run_collector(base_urls, roles, max_model_len, polls) -> dict:
    """Feed `(timestamp, [text per endpoint])` polls through the Python collector; the last
    endpoint's max_model_len is preset."""
    import httpx

    from prime_rl.orchestrator.inference_metrics import (
        EngineSample,
        InferenceMetricsCollector,
        TimedSnapshot,
        parse_prometheus_text,
    )

    collector = InferenceMetricsCollector([httpx.AsyncClient(base_url=url) for url in base_urls], roles=roles)
    collector.max_model_len_by_endpoint[collector.endpoints[-1].key] = max_model_len
    out = []
    for timestamp, texts in polls:
        samples = [
            EngineSample(endpoint=endpoint, engine_label=label, timestamp=timestamp, snapshot=snapshot)
            for endpoint, text in zip(collector.endpoints, texts)
            for label, snapshot in sorted(parse_prometheus_text(text).items())
        ]
        metrics = collector.build_metrics(samples)
        load = [vars(collector.build_load_sample(s)) for s in samples]
        for s in samples:
            collector.previous[s.key] = TimedSnapshot(timestamp=s.timestamp, snapshot=s.snapshot)
        out.append({"timestamp": timestamp, "texts": texts, "metrics": list(metrics.items()), "load": load})
    return {"base_urls": base_urls, "roles": roles, "max_model_len": max_model_len, "polls": out}


def real_vllm_metrics() -> None:
    """Two scrapes of a real Qwen3-0.6B prime-rl inference server (vllm_metrics_{1,2}.txt)."""
    texts = [(FIXTURES / f"vllm_metrics_{i}.txt").read_text() for i in (1, 2)]
    fixture = run_collector(["http://localhost:8731"], None, 4096, [(10.0, [texts[0]]), (15.0, [texts[1]])])
    (FIXTURES / "inference_metrics_vllm.json").write_text(json.dumps(fixture, indent=1))


def inference_metrics() -> None:
    """Two polls of two engines' expositions (one prefill server with 2 engines, one decode
    server) and the metrics/load samples the Python collector derives from them."""
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

    def exposition(engines: list[str], scale: float) -> str:
        registry = CollectorRegistry()
        running = Gauge("vllm:num_requests_running", "Running", ["engine", "model_name"], registry=registry)
        waiting = Gauge("vllm:num_requests_waiting", "Waiting", ["engine", "model_name"], registry=registry)
        by_reason = Gauge(
            "vllm:num_requests_waiting_by_reason", "Waiting by reason", ["engine", "reason"], registry=registry
        )
        kv = Gauge("vllm:kv_cache_usage_perc", "KV \\ usage\nhelp", ["engine"], registry=registry)
        info = Gauge(
            "vllm:cache_config_info",
            "cache",
            ["engine", "num_gpu_blocks", "block_size", "kv_cache_size_tokens"],
            registry=registry,
        )
        hits = Counter("vllm:prefix_cache_hits", "hits", ["engine"], registry=registry)
        queries = Counter("vllm:prefix_cache_queries", "queries", ["engine"], registry=registry)
        preempt = Counter("vllm:num_preemptions", "preemptions", ["engine"], registry=registry)
        finished = Counter("vllm:request_success", "done", ["engine", "finished_reason"], registry=registry)
        ttft = Histogram(
            "vllm:time_to_first_token_seconds", "ttft", ["engine"], buckets=[0.01, 0.1, 1.0, 10.0], registry=registry
        )
        other = Gauge("process_cpu_seconds", "not vllm", registry=registry)
        other.set(3)
        for i, engine in enumerate(engines):
            running.labels(engine, "m").set(int(4 * scale) + i)
            waiting.labels(engine, "m").set(2 * scale + i)
            by_reason.labels(engine, "capacity").set(1 * scale)
            by_reason.labels(engine, "deferred").set(1)
            kv.labels(engine).set(0.25 * scale + 0.1 * i)
            info.labels(engine, "1000", "16", "16000" if i == 0 else "").set(1)
            hits.labels(engine).inc(100 * scale + i)
            queries.labels(engine).inc(300 * scale + 7 * i)
            preempt.labels(engine).inc(int(3 * scale) + i)
            finished.labels(engine, "stop").inc(10 * scale)
            finished.labels(engine, "length").inc(2 * scale)
            for v in [0.005, 0.05, 0.5 * scale, 5.0, 0.3 * i]:
                for _ in range(int(scale)):
                    ttft.labels(engine).observe(v)
        return generate_latest(registry).decode() + 'vllm:untyped_thing{engine="0"} 5\nvllm:weird{engine="0"} NaN\n'

    polls = [(ts, [exposition(e, scale) for e in [["0", "1"], ["0"]]]) for ts, scale in [(100.0, 1.0), (105.5, 3.0)]]
    fixture = run_collector(["http://prefill-a:8000", "http://decode-a:8000"], ["prefill", "decode"], 32768, polls)
    (FIXTURES / "inference_metrics.json").write_text(json.dumps(fixture, indent=1))


def client_configs() -> None:
    """`setup_client(...).model_dump(mode="json")` for the shapes the orchestrator builds."""
    from renderers import Qwen3VLRendererConfig

    from prime_rl.configs.shared import ClientConfig
    from prime_rl.orchestrator.clients import setup_client

    config = ClientConfig(
        base_url="http://worker-a:8000/v1",
        api_key_var="VLLM_API_KEY",
        headers={"X-Test": "test"},
        headers_from_env={"X-Token": "PROVIDER_TOKEN"},
        timeout={"read": 12},
    )
    renderer = Qwen3VLRendererConfig()
    cases = {
        "renderer": setup_client(
            config, client_type="renderer", renderer_config=renderer, renderer_model_name="Qwen/Qwen3-VL-4B-Instruct"
        ),
        "renderer_auto": setup_client(config, client_type="renderer"),
        "chat": setup_client(config),
    }
    fixture = {
        "renderer_config": renderer.model_dump(mode="json"),
        "cases": {name: client.model_dump(mode="json") for name, client in cases.items()},
    }
    (FIXTURES / "client_configs.json").write_text(json.dumps(fixture, indent=1))


if __name__ == "__main__":
    FIXTURES.mkdir(parents=True, exist_ok=True)
    checkpoints()
    pyrandom()
    inference_metrics()
    real_vllm_metrics()
    client_configs()
