"""Record one real episode stream for `fake-env-server`.

Starts `mock-inference` and one real `env-server` (reverse-text, bench harness), runs one
episode through the Python `EnvClient`, and writes the deltas it streamed plus the reply's
head and trace summaries to OUT (msgpack).

    uv run --no-sync python rust/bench/record_template.py rust/bench/template.msgpack
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import msgpack

from run_bench import BENCH, ENV_NAME, MODEL, ROOT, TOOLS, completions, orchestrator_toml, spawn, stop, write_env_server_configs


async def record(address: str, task: dict, client, sampling) -> dict:
    from verifiers.v1.serve import EnvClient
    from verifiers.v1.serve.delta import TraceSummary, dump, pack

    deltas: list[bytes] = []
    env_client = EnvClient(address)
    await env_client.wait_for_server_startup(timeout=300)
    episode = await env_client.run(client=client, model=MODEL, sampling=sampling, task_data=task, on_delta=lambda d: deltas.append(pack(d)))
    await env_client.close()
    assert episode.ok, episode.errors
    summaries = [TraceSummary(id=t.id, nodes=len(t.nodes), calls=len(t.calls)).model_dump() for t in episode.traces]
    return {"deltas": deltas, "head": dump(episode, exclude={"traces"}), "traces": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--mock-port", type=int, default=18321)
    args = parser.parse_args()

    import tomllib

    import verifiers.v1 as vf

    from prime_rl.configs.orchestrator import OrchestratorConfig

    work = Path(tempfile.mkdtemp(prefix="prl-bench-record-"))
    bench_args = argparse.Namespace(steps=1, batch_size=16, env_replicas=1, env_workers=1, max_inflight=16, group_size=16,
                                    mock_port=args.mock_port, batch_port=0)
    config = OrchestratorConfig.model_validate(tomllib.loads(orchestrator_toml(bench_args, work / "run")))
    config_dir = work / "run" / "configs" / "resolved"
    (env_config,) = write_env_server_configs(bench_args, config, work / "run", config_dir)
    (work / "completions.json").write_text(json.dumps(completions()))
    env = {**os.environ, "PRL_BENCH_KEY": "EMPTY", "PYTHONPATH": os.pathsep.join(filter(None, [str(BENCH), os.environ.get("PYTHONPATH")]))}
    procs = [
        spawn([TOOLS / "mock-inference", "--port", args.mock_port, "--model", MODEL, "--completions", work / "completions.json"], work / "mock.log", work),
        spawn([ROOT / ".venv/bin/env-server", "@", env_config], work / "env.log", ROOT, env),
    ]
    try:
        address_file = config_dir / "envs" / "train" / f"{ENV_NAME}.0.address"
        while not address_file.exists():
            time.sleep(0.5)
        task = next(iter(vf.load_taskset(config.train.source[0].env.taskset))).data.model_dump(mode="json")
        client = vf.TrainClientConfig(base_url=f"http://127.0.0.1:{args.mock_port}/v1", api_key_var="PRL_BENCH_KEY",
                                      renderer=config.renderer, renderer_model_name=MODEL)
        sampling = vf.SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=128, logprobs=True)
        template = asyncio.run(record(address_file.read_text().strip(), task, client, sampling))
        args.out.write_bytes(msgpack.packb(template, use_bin_type=True))
        print(f"recorded {len(template['deltas'])} deltas to {args.out}")
    finally:
        for proc in procs:
            stop(proc)


if __name__ == "__main__":
    main()
