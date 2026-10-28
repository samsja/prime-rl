"""Python side of the Rust orchestrator, run by the ``orchestrator`` entrypoint.

``run_rust_orchestrator`` hands the validated config to ``prime_rl_orch_rs.run``, which
releases the GIL and runs the orchestrator on a tokio runtime until it exits. The Rust side
calls back into ``PyBridge`` (from blocking threads, holding the GIL) for the work that stays
in Python. See ``rust/docs/specs/boundary.md``.
"""

from __future__ import annotations

from collections.abc import Iterator

import orjson
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import EnvConfig, OrchestratorConfig
from prime_rl.orchestrator.utils import intercept_vf_logging
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import write_resolved_config


def task_record(task: vf.Task) -> dict:
    """A task as the Rust side sees it: ``data`` is what the dispatcher ships as
    ``task_data``; ``key``/``hash`` are the identity checked against the returned episode."""
    return {"key": task.key, "hash": task.hash, "data": task.data.model_dump(mode="json")}


class PyBridge:
    """Callbacks from the Rust orchestrator into Python. Arguments are plain values; results
    are JSON-encoded ``bytes``."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._streams: dict[tuple[str, str], Iterator[vf.Task]] = {}
        # datasets' parallel arrow read races on tqdm's lock when tasksets load concurrently
        # (see ``Envs.start``); pre-seed it.
        from datasets.utils import tqdm as hf_tqdm

        hf_tqdm.set_lock(hf_tqdm.get_lock())

    def _source(self, split: str, env_name: str) -> EnvConfig:
        if split == "train":
            sources = self.config.train.source
        elif split == "eval":
            sources = self.config.eval.source if self.config.eval is not None else []
        else:
            raise ValueError(f"Unknown split {split!r}, expected 'train' or 'eval'")
        for source in sources:
            if source.resolved_name == env_name:
                return source
        raise KeyError(f"No {split} env named {env_name!r}")

    def load_tasks(self, split: str, env_name: str) -> bytes:
        """Load the env's taskset client-side, as ``Env.start`` does.

        Returns ``{"num_tasks": int | null, "tasks": [task_record, ...]}``. A bounded
        selection is materialized whole. An unbounded one returns ``num_tasks: null`` and no
        tasks; pull them with ``next_tasks``."""
        source = self._source(split, env_name)
        taskset = vf.load_taskset(source.env.taskset).select(source.select)
        if taskset.bounded:
            tasks = [task_record(task) for task in taskset]
            return orjson.dumps({"num_tasks": len(tasks), "tasks": tasks})
        self._streams[(split, env_name)] = iter(taskset)
        return orjson.dumps({"num_tasks": None, "tasks": []})

    def next_tasks(self, split: str, env_name: str, n: int) -> bytes:
        """Up to ``n`` more tasks of an unbounded taskset opened by ``load_tasks``, as a JSON
        list of task records. Fewer than ``n`` means the stream ended."""
        stream = self._streams[(split, env_name)]
        tasks = []
        for task in stream:
            tasks.append(task_record(task))
            if len(tasks) == n:
                break
        return orjson.dumps(tasks)


def run_rust_orchestrator(config: OrchestratorConfig) -> None:
    import prime_rl_orch_rs

    write_resolved_config(config.output_dir, config, "orchestrator")
    setup_logger(config.log.level, json_logging=config.log.json_logging)
    intercept_vf_logging(logger="verifiers.v1", level="WARN")
    get_logger().info("Starting orchestrator")
    prime_rl_orch_rs.run(config.model_dump_json(), PyBridge(config))
