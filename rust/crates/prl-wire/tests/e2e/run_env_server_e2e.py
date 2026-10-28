"""End-to-end check of prl-wire against a real env server.

Starts a mock OpenAI endpoint and `uv run env-server` for gsm8k (null harness), runs the
ignored Rust test `env_server` against it, then validates every episode the Rust client
received with Python's `WireEpisode` and checks the records match byte for byte.

    uv run --no-sync python rust/crates/prl-wire/tests/e2e/run_env_server_e2e.py
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
    work = Path(tempfile.mkdtemp(prefix="prl-wire-e2e-"))
    port = 18900 + os.getpid() % 90
    (work / "env.toml").write_text(
        f'address_file = "{work}/addr"\noutput_dir = "{work}/out"\n\n[env]\ntaskset.id = "gsm8k"\n'
        'agent.harness.id = "null"\nagent.runtime.type = "subprocess"\n'
    )
    procs = [
        subprocess.Popen([sys.executable, HERE / "mock_openai.py", str(port)]),
        subprocess.Popen(["uv", "run", "--no-sync", "env-server", "@", str(work / "env.toml")], cwd=ROOT),
    ]
    try:
        config = EnvServerConfig.model_validate(tomllib.loads((work / "env.toml").read_text()))
        tasks = []
        for task in vf.load_taskset(config.env.taskset):
            tasks.append({"data": task.data.model_dump(mode="json"), "key": task.key, "hash": task.hash})
            if len(tasks) == 4:
                break
        (work / "tasks.json").write_text(json.dumps(tasks))
        client = vf.EvalClientConfig(base_url=f"http://127.0.0.1:{port}/v1", api_key_var="PRL_WIRE_E2E_KEY")
        (work / "client.json").write_text(json.dumps(client.model_dump(mode="json")))
        sampling = vf.SamplingConfig(temperature=1.0, max_tokens=64)
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
