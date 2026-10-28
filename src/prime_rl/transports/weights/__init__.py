from pathlib import Path

import torch

from prime_rl.configs.shared import WeightBroadcastConfig
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transports.weights.base import WeightSender, prune_broadcasts_beyond
from prime_rl.transports.weights.filesystem import FileSystemWeightSender
from prime_rl.transports.weights.nccl import NCCLWeightSender
from prime_rl.transports.weights.nixl import NIXLWeightSender

__all__ = [
    "WeightSender",
    "prune_broadcasts_beyond",
    "setup_weight_sender",
]


def setup_weight_sender(
    output_dir: Path,
    config: WeightBroadcastConfig,
    parallel_dims: ParallelDims,
    lora_config: LoRAConfig | None = None,
) -> WeightSender:
    if config.type == "nccl":
        return NCCLWeightSender(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        return FileSystemWeightSender(output_dir, config, lora_config)
    elif config.type == "nixl":
        return NIXLWeightSender(output_dir, config, parallel_dims)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
