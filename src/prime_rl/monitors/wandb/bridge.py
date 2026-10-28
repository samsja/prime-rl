"""The W&B monitor behind the Rust monitors actor (``prl_monitors::WandbBridge``).

There is no Rust W&B SDK, so the Rust side calls this from a blocking thread holding the GIL.
Only scalar rows cross the boundary, as JSON (``NaN``/``Infinity`` allowed); episodes never
reach W&B.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from prime_rl.configs.monitors import WandbMonitorConfig
from prime_rl.monitors.wandb.monitor import WandbMonitor
from prime_rl.utils.config import BaseConfig


class WandbBridge:
    def __init__(self, config: WandbMonitorConfig, output_dir: Path, run_config: BaseConfig | None = None):
        self.monitor = WandbMonitor(config)
        self.output_dir = output_dir
        self.run_config = run_config

    def init(self, train_env_names: list[str], eval_env_names: list[str], overview_flavor: str) -> None:
        asyncio.run(
            self.monitor.init(
                output_dir=self.output_dir,
                config=self.run_config,
                train_env_names=train_env_names,
                eval_env_names=eval_env_names,
                overview_flavor=overview_flavor,
            )
        )

    def log(self, row_json: str, step: int | None) -> None:
        asyncio.run(self.monitor.log_metrics(json.loads(row_json), step=step))

    def finish(self, paused: bool) -> None:
        asyncio.run(self.monitor.finalize(paused=paused))
