"""Orchestrator throughput benchmark: Python vs Rust, CPU only.

One run = one orchestrator (the Python one from --python-repo or the Rust one from
--rust-repo) against a fake world that is never the bottleneck:

- inference: `mock-inference` (rust/bench/tools), a fast fake vLLM token-in/out endpoint
  that answers every generate with one of a few canned completions;
- env servers: `--env-replicas` real `env-server` pools for reverse-text (null harness),
  each with `--env-workers` static workers;
- trainer: `trainer_stub.py`, which pulls each step's micro batches over ZMQ and at once
  publishes the next policy version (filesystem broadcast, no weights).

It samples the orchestrator's CPU and RSS from /proc, reads the trainer stub's step log and
the orchestrator's metrics.jsonl, and appends one JSON line of results to
<work-dir>/results.jsonl.

    uv run --no-sync python rust/bench/run_bench.py rust 1024 1024
    uv run --no-sync python rust/bench/run_bench.py python 1024 1024 --env-replicas 6
    uv run --no-sync python rust/bench/run_bench.py env-load 4096 0   # env servers + mock only
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
TOOLS = ROOT / "rust/target/wire-bench/release"
MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
ENV_NAME = "reverse-text"
CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")


def completions() -> list[list[int]]:
    """A few canned replies (as token ids ending in <|im_end|>), so rewards vary."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    stop = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # tagged, so reverse-text's LCS reward parses them; different overlaps give different rewards
    texts = [f"<reversed_text>{t}</reversed_text>" for t in ["eht", "a e i o u", "dlrow olleh", "xof nworb kciuq eht", "s", "ereh txet"]]
    return [tokenizer.encode(text, add_special_tokens=False) + [stop] for text in texts]


def orchestrator_toml(args, run_dir: Path) -> str:
    return f"""output_dir = "{run_dir}"
max_steps = {args.steps}
seq_len = 2048
batch_size = {max(args.batch_size, args.group_size)}
env_server_replicas = {args.env_replicas}
collect_inference_metrics = false

[model]
name = "{MODEL}"

[model.client]
base_url = "http://127.0.0.1:{args.mock_port}/v1"
api_key_var = "PRL_BENCH_KEY"

[renderer]
name = "prime-qwen3"

[log]
level = "info"
interval = 5.0

[monitors.file]

[concurrency]
min_inflight = {args.max_inflight}
max_inflight = {args.max_inflight}
initial_inflight = {args.max_inflight}

[rollout_transport]
type = "zmq"
host = "127.0.0.1"
port = {args.batch_port}

[weight_broadcast]
type = "filesystem"

[train]
group_size = {args.group_size}

[train.algo]
type = "grpo"

[train.sampling]
max_completion_tokens = 128

[[train.source]]
name = "{ENV_NAME}"
env.taskset.id = "reverse-text"
env.agent.harness.id = "prl-bench-harness"
env.agent.runtime.type = "subprocess"
"""


def write_env_server_configs(args, config, run_dir: Path, config_dir: Path) -> list[Path]:
    import verifiers.v1 as vf

    from prime_rl.configs.env_server import EnvServerConfig
    from prime_rl.utils.pathing import env_address_file

    source = config.train.source[0]
    paths = []
    for replica in range(args.env_replicas):
        server = EnvServerConfig(
            env=source.env,
            serve=vf.ServeConfig(pool=vf.StaticPoolConfig(num_workers=args.env_workers)),
            address_file=env_address_file(config_dir, "train", ENV_NAME, replica),
            output_dir=run_dir,
        )
        server.log.level = "warning"
        path = config_dir / "envs" / "train" / f"{ENV_NAME}.{replica}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(server.model_dump(mode="json")))
        paths.append(path)
    return paths


def descendants(pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                ppid = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError):
                continue
            children.setdefault(ppid, []).append(int(entry.name))
    out, stack = [], [pid]
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(children.get(current, []))
    return out


def cpu_rss(pids: list[int]) -> tuple[float, float]:
    """Summed CPU seconds and RSS bytes of `pids`."""
    cpu = rss = 0.0
    for pid in pids:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            cpu += (int(fields[11]) + int(fields[12])) / CLK_TCK
            rss += int(fields[21]) * PAGE
        except (OSError, IndexError):
            pass
    return cpu, rss


class Sampler(threading.Thread):
    """Samples CPU seconds and RSS of process groups once a second."""

    def __init__(self, groups: dict[str, int]):
        super().__init__(daemon=True)
        self.groups = groups
        self.samples: list[dict] = []
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.is_set():
            row = {"time": time.time()}
            for name, pid in self.groups.items():
                row[name] = cpu_rss(descendants(pid))
            self.samples.append(row)
            self.stopped.wait(1.0)

    def window(self, name: str, start: float, end: float) -> dict:
        rows = [r for r in self.samples if start <= r["time"] <= end]
        if len(rows) < 2:
            return {}
        cpu = (rows[-1][name][0] - rows[0][name][0]) / (rows[-1]["time"] - rows[0]["time"])
        rss = [r[name][1] for r in rows]
        return {"cpu_pct": 100 * cpu, "rss_mb_mean": sum(rss) / len(rss) / 2**20, "rss_mb_max": max(rss) / 2**20}


def metrics_summary(path: Path, start: float, end: float) -> dict:
    """Means of the orchestrator's `time/*` and `event_loop_lag/*` metrics over the window."""
    sums: dict[str, list[float]] = {}
    if not path.is_file():
        return {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not start <= row.get("time", 0) <= end:
            continue
        for key, value in row.items():
            if isinstance(value, (int, float)) and (key.startswith("time/") or key.startswith("event_loop_lag/") or key.startswith("perf/")):
                sums.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in sorted(sums.items())}


def wait_for(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError(what)
        time.sleep(0.5)


def stop(proc: subprocess.Popen, sig=signal.SIGTERM, timeout: float = 30) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def spawn(cmd: list, log: Path, cwd: Path, env: dict | None = None) -> subprocess.Popen:
    return subprocess.Popen(
        [str(c) for c in cmd], cwd=cwd, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("impl", choices=["python", "rust", "env-load"])
    parser.add_argument("max_inflight", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--env", choices=["fake", "real"], default="fake",
                        help="fake: fake-env-server replaying a recorded episode; real: env-server pools")
    parser.add_argument("--template", type=Path, default=BENCH / "template.msgpack", help="recorded episode for --env fake")
    parser.add_argument("--env-latency-ms", type=int, default=1000, help="per-episode latency of --env fake")
    parser.add_argument("--env-replicas", type=int, default=4)
    parser.add_argument("--env-workers", type=int, default=4)
    parser.add_argument("--mock-threads", type=int, default=4)
    parser.add_argument("--mock-latency-ms", type=int, default=0)
    parser.add_argument("--mock-port", type=int, default=18311)
    parser.add_argument("--batch-port", type=int, default=18312)
    parser.add_argument("--load-seconds", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--python-repo", type=Path, default=Path("/home/ubuntu/prime-rl-orch-baseline"))
    parser.add_argument("--rust-repo", type=Path, default=ROOT)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/prl-orch-bench"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    import tomllib

    import verifiers.v1 as vf

    from prime_rl.configs.orchestrator import OrchestratorConfig

    name = f"{args.impl}-{args.env}-n{args.max_inflight}-b{args.batch_size}-r{args.env_replicas}x{args.env_workers}{args.tag}"
    work = args.work_dir / name
    shutil.rmtree(work, ignore_errors=True)
    run_dir, config_dir = work / "run", work / "run" / "configs" / "resolved"
    config_dir.mkdir(parents=True)
    raw = tomllib.loads(orchestrator_toml(args, run_dir))
    # file monitor only: never create platform runs from a benchmark (Prime is on by default)
    raw["monitors"]["prime"] = None
    (work / "orchestrator.json").write_text(json.dumps(raw))
    config = OrchestratorConfig.model_validate(raw)
    assert config.monitors.prime is None and config.monitors.wandb is None
    (work / "completions.json").write_text(json.dumps(completions()))
    env_configs = write_env_server_configs(args, config, run_dir, config_dir)
    env = {**os.environ, "PRL_ATTEMPT_CONFIG_DIR": str(config_dir), "PRL_RUN_ID": name, "PRL_BENCH_KEY": "EMPTY",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(BENCH), os.environ.get("PYTHONPATH")]))}

    procs: dict[str, subprocess.Popen] = {}
    try:
        procs["mock"] = spawn(
            [TOOLS / "mock-inference", "--port", args.mock_port, "--model", MODEL, "--completions", work / "completions.json",
             "--threads", args.mock_threads, "--latency-ms", args.mock_latency_ms],
            work / "mock.log", work,
        )
        address_files = [config_dir / "envs" / "train" / f"{ENV_NAME}.{r}.address" for r in range(args.env_replicas)]
        for replica, path in enumerate(env_configs):
            if args.env == "fake":
                cmd = [TOOLS / "fake-env-server", "--template", args.template, "--address-file", address_files[replica],
                       "--latency-ms", args.env_latency_ms]
            else:
                cmd = [args.rust_repo / ".venv/bin/env-server", "@", path]
            procs[f"env{replica}"] = spawn(cmd, work / f"env{replica}.log", args.rust_repo, env)
        wait_for(lambda: all(f.exists() for f in address_files), 600, "env servers never published their addresses")

        if args.impl == "env-load":
            source = config.train.source[0]
            tasks = []
            for task in vf.load_taskset(source.env.taskset):
                tasks.append(task.data.model_dump(mode="json"))
                if len(tasks) == 256:
                    break
            (work / "tasks.json").write_text(json.dumps(tasks))
            client = vf.TrainClientConfig(base_url=f"http://127.0.0.1:{args.mock_port}/v1", api_key_var="PRL_BENCH_KEY",
                                          renderer=config.renderer, renderer_model_name=MODEL)
            (work / "client.json").write_text(json.dumps(client.model_dump(mode="json")))
            sampling = vf.SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=128, logprobs=True)
            (work / "sampling.json").write_text(json.dumps(sampling.model_dump(mode="json")))
            sampler = Sampler({name: proc.pid for name, proc in procs.items()})
            sampler.start()
            t0 = time.time()
            out = subprocess.run(
                [TOOLS / "env-load", "--address-files", *address_files, "--tasks", work / "tasks.json", "--client", work / "client.json",
                 "--sampling", work / "sampling.json", "--model", MODEL, "--concurrency", str(args.max_inflight),
                 "--seconds", str(args.load_seconds), "--warmup", "10"],
                env=env, capture_output=True, text=True, check=True,
            )
            result = {"impl": "env-load", **json.loads(out.stdout.strip().splitlines()[-1]), "env_workers": args.env_workers,
                      "mock": sampler.window("mock", t0 + 10, time.time()),
                      "env_servers": {k: sampler.window(k, t0 + 10, time.time()) for k in procs if k.startswith("env")}}
        else:
            repo = args.python_repo if args.impl == "python" else args.rust_repo
            procs["trainer"] = spawn(
                [args.rust_repo / ".venv/bin/python", BENCH / "trainer_stub.py", run_dir, args.batch_port, args.steps, work / "steps.jsonl"],
                work / "trainer.log", args.rust_repo, env,
            )
            procs["orchestrator"] = spawn([repo / ".venv/bin/orchestrator", "@", work / "orchestrator.json"], work / "orchestrator.log", repo, env)
            sampler = Sampler({name: proc.pid for name, proc in procs.items()})
            sampler.start()
            steps_log = work / "steps.jsonl"

            def steps() -> list[dict]:
                return [json.loads(line) for line in steps_log.read_text().splitlines()] if steps_log.exists() else []

            deadline = time.monotonic() + args.timeout
            while len(steps()) < args.steps:
                if procs["orchestrator"].poll() is not None:
                    raise RuntimeError(f"orchestrator exited with {procs['orchestrator'].returncode}, see {work / 'orchestrator.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"only {len(steps())} steps in {args.timeout}s")
                time.sleep(1)
            rows = steps()
            first, last = rows[args.warmup_steps - 1], rows[-1]
            window = last["time"] - first["time"]
            samples = sum(r["samples"] for r in rows[args.warmup_steps:])
            mock_generated = None
            result = {
                "impl": args.impl,
                "max_inflight": args.max_inflight,
                "batch_size": args.batch_size,
                "group_size": args.group_size,
                "env": args.env,
                "env_latency_ms": args.env_latency_ms if args.env == "fake" else None,
                "env_replicas": args.env_replicas,
                "env_workers": args.env_workers,
                "steady_steps": len(rows) - args.warmup_steps,
                "steps_per_s": (len(rows) - args.warmup_steps) / window,
                "samples_per_s": samples / window,
                "startup_s": rows[0]["time"] - sampler.samples[0]["time"],
                "orchestrator": sampler.window("orchestrator", first["time"], last["time"]),
                "mock": sampler.window("mock", first["time"], last["time"]),
                "env_servers_cpu_pct": sum(sampler.window(k, first["time"], last["time"]).get("cpu_pct", 0) for k in procs if k.startswith("env")),
                "metrics": metrics_summary(run_dir / "metrics.jsonl", first["time"], last["time"]),
            }
            del mock_generated
        sampler.stopped.set()
        result["name"] = name
        print(json.dumps(result))
        with (args.work_dir / "results.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
    finally:
        if "orchestrator" in procs:
            stop(procs["orchestrator"], signal.SIGINT, timeout=60)
        for proc in procs.values():
            stop(proc)
        # benchmark runs keep logs and metrics only
        shutil.rmtree(run_dir / "broadcasts", ignore_errors=True)


if __name__ == "__main__":
    main()
