"""Python side of the Rust orchestrator (``prime_rl_orch_rs``), used by the ``orchestrator``,
``eval`` and ``online_eval`` entrypoints.

``prime_rl_orch_rs.run*`` release the GIL and run on a tokio runtime until they exit. The Rust
side calls back into ``PyBridge`` (from blocking threads, holding the GIL) for the work that
stays in Python. Arguments are plain values; results are JSON-encoded ``bytes``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import cache, partial
from typing import TYPE_CHECKING

import orjson
import verifiers.v1 as vf
from pydantic import TypeAdapter
from renderers import RendererConfig

from prime_rl.configs.eval import EvalConfig, SFTOnlineEvalConfig
from prime_rl.configs.orchestrator import EnvConfig, OrchestratorConfig
from prime_rl.monitors.wandb.bridge import WandbBridge
from prime_rl.utils.logger import get_logger, intercept_vf_logging, setup_logger
from prime_rl.utils.pathing import write_resolved_config
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from renderers.base import Renderer


def task_record(task: vf.Task) -> dict:
    """``data`` is shipped as the run request's ``task_data``; ``key``/``hash`` identify the task
    on the returned episode (a taskset may override ``key``)."""
    return {"key": task.key, "hash": task.hash, "data": task.data.model_dump(mode="json")}


@cache
def _renderer(model_name: str, renderer_config: str) -> Renderer:
    from renderers.base import create_renderer, load_tokenizer

    return create_renderer(load_tokenizer(model_name), TypeAdapter(RendererConfig).validate_json(renderer_config))


@cache
def _echo_filter(import_path: str, kwargs: str) -> Callable[[vf.Trace], list]:
    return partial(import_object(import_path), **orjson.loads(kwargs))


class PyBridge:
    def __init__(self, config: OrchestratorConfig | EvalConfig | SFTOnlineEvalConfig):
        self.config = config
        if isinstance(config, OrchestratorConfig):
            eval_sources = config.eval.source if config.eval is not None else []
            self._sources: dict[str, list[EnvConfig]] = {"train": config.train.source, "eval": eval_sources}
        else:
            self._sources = {"train": [], "eval": config.source}
        # Driven by the Rust monitors actor; same output dir as each process's Python monitors.
        output_dir = config.run_dir if isinstance(config, EvalConfig) else config.output_dir
        wandb = config.monitors.wandb
        self.wandb = WandbBridge(wandb, output_dir, config) if wandb is not None else None
        self._streams: dict[tuple[str, str], Iterator[vf.Task]] = {}
        # datasets' parallel arrow read races on tqdm's lock when tasksets load concurrently.
        from datasets.utils import tqdm as hf_tqdm

        hf_tqdm.set_lock(hf_tqdm.get_lock())

    def load_tasks(self, split: str, env_name: str) -> bytes:
        """``{"num_tasks": int | null, "tasks": [task_record]}``. A bounded selection comes back
        whole; an unbounded one has ``num_tasks: null`` and is pulled with ``next_tasks``."""
        (source,) = (s for s in self._sources[split] if s.resolved_name == env_name)
        taskset = vf.load_taskset(source.env.taskset).select(source.select)
        if taskset.bounded:
            tasks = [task_record(task) for task in taskset]
            return orjson.dumps({"num_tasks": len(tasks), "tasks": tasks})
        self._streams[(split, env_name)] = iter(taskset)
        return orjson.dumps({"num_tasks": None, "tasks": []})

    def next_tasks(self, split: str, env_name: str, n: int) -> bytes:
        """Up to ``n`` more task records of an unbounded taskset; fewer means it ended."""
        stream = self._streams[(split, env_name)]
        return orjson.dumps([task_record(task) for _, task in zip(range(n), stream)])

    def flop_cost_table(self, max_len: int) -> bytes:
        """``[bin_cost([n]) for n in 0..=max_len]`` of the trainer model's FlopCounter, as decimal
        strings (they can exceed 64 bits)."""
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
        """opsd's hint block: token ids of one system message, without generation prompt."""
        renderer = _renderer(model_name, renderer_config)
        ids = renderer.render_ids([{"role": "system", "content": system_content}], add_generation_prompt=False)
        return orjson.dumps(list(ids))

    def echo_filter(self, import_path: str, kwargs: str, trace: str) -> bytes:
        """Echo's user filter masks for one ``vf.WireTrace`` (JSON), unchecked."""
        masks = _echo_filter(import_path, kwargs)(vf.WireTrace.model_validate_json(trace))
        return orjson.dumps(masks, option=orjson.OPT_SERIALIZE_NUMPY)

    def nixl_receiver(self):
        """The consumer's NIXL agent + ModelExpress session, for ``nixl`` weight broadcasts."""
        from prime_rl.transports.weights.nixl.bridge import NixlReceiverBridge

        return NixlReceiverBridge(self.config.weight_broadcast)


def run_rust_orchestrator(config: OrchestratorConfig) -> None:
    import prime_rl_orch_rs

    write_resolved_config(config.output_dir, config, "orchestrator")
    setup_logger(config.log.level, json_logging=config.log.json_logging)
    intercept_vf_logging(logger="verifiers.v1", level="WARN")
    get_logger().info("Starting orchestrator")
    prime_rl_orch_rs.run(config, PyBridge(config))
