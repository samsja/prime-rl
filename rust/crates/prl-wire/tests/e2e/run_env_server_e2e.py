"""End-to-end check of prl-wire against a real env server.

Starts `uv run env-server` for gsm8k (null harness) and either a mock OpenAI endpoint
(default, eval client) or uses a running vLLM (`--train URL MODEL`: token-in/out train
client, sampling logprobs, payload hook arrays), runs the ignored Rust test `env_server`
against it, then validates every episode the Rust client received with Python's
`WireEpisode` and checks the records match byte for byte.

    uv run --no-sync python rust/crates/prl-wire/tests/e2e/run_env_server_e2e.py
    uv run --no-sync python rust/crates/prl-wire/tests/e2e/run_env_server_e2e.py --train http://127.0.0.1:8000/v1 Qwen/Qwen3-0.6B
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import msgpack

import verifiers.v1 as vf
from prime_rl.configs.env_server import EnvServerConfig
from verifiers.v1.episode import WireEpisode

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).parent


def main() -> None:
    train = sys.argv[1:2] == ["--train"]
    work = Path(tempfile.mkdtemp(prefix="prl-wire-e2e-"))
    port = 18900 + os.getpid() % 90
    (work / "env.toml").write_text(
        f'address_file = "{work}/addr"\noutput_dir = "{work}/out"\n\n[env]\ntaskset.id = "gsm8k"\n'
        'agent.harness.id = "null"\nagent.runtime.type = "subprocess"\n'
    )
    procs = [subprocess.Popen(["uv", "run", "--no-sync", "env-server", "@", str(work / "env.toml")], cwd=ROOT)]
    if not train:
        procs.append(subprocess.Popen([sys.executable, HERE / "mock_openai.py", str(port)]))
    try:
        config = EnvServerConfig.model_validate(tomllib.loads((work / "env.toml").read_text()))
        tasks = []
        for task in vf.load_taskset(config.env.taskset):
            tasks.append({"data": task.data.model_dump(mode="json"), "key": task.key, "hash": task.hash})
            if len(tasks) == 4:
                break
        (work / "tasks.json").write_text(json.dumps(tasks))
        if train:
            base_url, model = sys.argv[2], sys.argv[3]
            client = vf.TrainClientConfig(base_url=base_url, api_key_var="PRL_WIRE_E2E_KEY", renderer_model_name=model)
            sampling = vf.SamplingConfig(temperature=1.0, max_tokens=64, logprobs=5)
        else:
            model = "m"
            client = vf.EvalClientConfig(base_url=f"http://127.0.0.1:{port}/v1", api_key_var="PRL_WIRE_E2E_KEY")
            sampling = vf.SamplingConfig(temperature=1.0, max_tokens=64)
        (work / "model").write_text(model)
        (work / "client.json").write_text(json.dumps(client.model_dump(mode="json")))
        (work / "sampling.json").write_text(json.dumps(sampling.model_dump(mode="json")))
        deadline = time.monotonic() + 300
        while not (work / "addr").exists():
            assert time.monotonic() < deadline, "env server never published its address"
            time.sleep(0.5)
        env = {**os.environ, "PRL_WIRE_E2E_DIR": str(work), "CARGO_TARGET_DIR": str(ROOT / "rust/target/wire")}
        subprocess.run(
            [os.path.expanduser("~/.cargo/bin/cargo"), "test", "-p", "prl-wire", "--test", "env_server", "--", "--ignored", "--nocapture"],
            cwd=ROOT / "rust",
            env=env,
            check=True,
        )
        for i in range(len(tasks)):
            wire = msgpack.unpackb((work / f"episode_{i}.msgpack").read_bytes(), raw=False, strict_map_key=False)
            episode = WireEpisode.model_validate(wire)
            expected = episode.to_json()
            actual = (work / f"episode_{i}.json").read_bytes()
            assert actual == expected, f"episode {i} record differs:\n{actual[:2000]}\n{expected[:2000]}"
        print(f"ok: {len(tasks)} episodes from a real env server, records identical ({work})")
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
