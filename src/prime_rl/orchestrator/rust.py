"""Python side of the Rust orchestrator, run by the ``orchestrator`` entrypoint.

``run_rust_orchestrator`` hands the validated config to ``prime_rl_orch_rs.run``, which
releases the GIL and runs the orchestrator on a tokio runtime until it exits. The Rust side
calls back into ``PyBridge`` (from blocking threads, holding the GIL) for the work that stays
in Python. See ``rust/docs/specs/boundary.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from typing import TYPE_CHECKING

import orjson
import verifiers.v1 as vf
from pydantic import TypeAdapter
from renderers import RendererConfig

from prime_rl.configs.eval import EvalConfig, SFTOnlineEvalConfig
from prime_rl.configs.orchestrator import EnvConfig, OrchestratorConfig
from prime_rl.monitors.wandb.bridge import WandbBridge
from prime_rl.orchestrator.utils import intercept_vf_logging
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import write_resolved_config
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from renderers.base import Renderer


def task_record(task: vf.Task) -> dict:
    """A task as the Rust side sees it: ``data`` is what the dispatcher ships as
    ``task_data``; ``key``/``hash`` are the identity checked against the returned episode."""
    return {"key": task.key, "hash": task.hash, "data": task.data.model_dump(mode="json")}


class PyBridge:
    """Callbacks from the Rust orchestrator into Python. Arguments are plain values; results
    are JSON-encoded ``bytes``."""

    def __init__(self, config: OrchestratorConfig | EvalConfig | SFTOnlineEvalConfig):
        self.config = config
        if isinstance(config, OrchestratorConfig):
            eval_sources = config.eval.source if config.eval is not None else []
            self._sources: dict[str, list[EnvConfig]] = {"train": config.train.source, "eval": eval_sources}
        else:
            self._sources = {"train": [], "eval": config.source}
        # The W&B monitor, driven by the Rust monitors actor (``prl_monitors::WandbBridge``).
        # Same output dir as the Python ``monitors.setup`` call of each process.
        output_dir = config.run_dir if isinstance(config, EvalConfig) else config.output_dir
        wandb = config.monitors.wandb
        self.wandb = WandbBridge(wandb, output_dir, config) if wandb is not None else None
        self._streams: dict[tuple[str, str], Iterator[vf.Task]] = {}
        self._renderers: dict[tuple[str, str], Renderer] = {}
        self._echo_filters: dict[tuple[str, str], Callable[[vf.Trace], list]] = {}
        # datasets' parallel arrow read races on tqdm's lock when tasksets load concurrently
        # (see ``Envs.start``); pre-seed it.
        from datasets.utils import tqdm as hf_tqdm

        hf_tqdm.set_lock(hf_tqdm.get_lock())

    def _source(self, split: str, env_name: str) -> EnvConfig:
        for source in self._sources[split]:
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

    def flop_cost_table(self, max_len: int) -> bytes:
        """``[bin_cost([n]) for n in 0..=max_len]`` of the FlopCounter ``BatchPacker`` builds
        for the trainer model. Entries are decimal strings: they can exceed 64 bits.
        Orchestrator only."""
        assert isinstance(self.config, OrchestratorConfig)
        import torch

        from prime_rl.configs.trainer import ModelConfig
        from prime_rl.trainer.model import get_model
        from prime_rl.trainer.perf import FlopCounter

        model_config = ModelConfig(
            name=self.config.model.name,
            attn="flash_attention_3",
            trust_remote_code=self.config.tokenizer.trust_remote_code,
        )
        bin_cost = FlopCounter(get_model(model_config, device=torch.device("meta")))
        return orjson.dumps([str(bin_cost([n])) for n in range(max_len + 1)])

    def render_ids(self, model_name: str, renderer_config: str, system_content: str) -> bytes:
        """opsd's hint block: token ids of one system message (no generation prompt), rendered
        with ``create_renderer(load_tokenizer(model_name), renderer_config)``.
        ``renderer_config`` is the JSON of a ``RendererConfig``."""
        cache_key = (model_name, renderer_config)
        renderer = self._renderers.get(cache_key)
        if renderer is None:
            from renderers.base import create_renderer, load_tokenizer

            config = TypeAdapter(RendererConfig).validate_json(renderer_config)
            renderer = self._renderers[cache_key] = create_renderer(load_tokenizer(model_name), config)
        ids = renderer.render_ids([{"role": "system", "content": system_content}], add_generation_prompt=False)
        return orjson.dumps(list(ids))

    def echo_filter(self, import_path: str, kwargs: str, trace: bytes) -> bytes:
        """Masks from echo's user filter ``import_object(import_path)(trace, **kwargs)``.
        ``kwargs`` is a JSON object; ``trace`` is a JSON-encoded ``vf.WireTrace``. Returns the
        filter's masks unchecked (the caller validates their shape, as ``EchoAlgorithm`` does)."""
        cache_key = (import_path, kwargs)
        filter_fn = self._echo_filters.get(cache_key)
        if filter_fn is None:
            filter_fn = self._echo_filters[cache_key] = partial(import_object(import_path), **orjson.loads(kwargs))
        masks = filter_fn(vf.WireTrace.model_validate_json(trace))
        return orjson.dumps(masks, option=orjson.OPT_SERIALIZE_NUMPY)


def run_rust_orchestrator(config: OrchestratorConfig) -> None:
    import prime_rl_orch_rs

    write_resolved_config(config.output_dir, config, "orchestrator")
    setup_logger(config.log.level, json_logging=config.log.json_logging)
    intercept_vf_logging(logger="verifiers.v1", level="WARN")
    get_logger().info("Starting orchestrator")
    prime_rl_orch_rs.run(config, PyBridge(config))
