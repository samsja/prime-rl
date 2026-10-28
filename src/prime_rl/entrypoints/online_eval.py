"""Online evals of an ``sft`` run on the Rust core: ``python -m prime_rl.entrypoints.online_eval @ eval.json``.

Spawned by the ``sft`` launcher next to the trainer. The process watches the run's broadcasts
directory, moves the inference server onto each offered weight broadcast and runs the due eval
sources against it (``prl_eval::run_online_eval``)."""

import json
import os

from prime_rl.configs.eval import SFTOnlineEvalConfig
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import prepare_attempt_dirs
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("OnlineEvals")
    config = cli(SFTOnlineEvalConfig)
    config_dir, log_dir = prepare_attempt_dirs(config.output_dir)
    os.environ["PRL_ATTEMPT_CONFIG_DIR"] = str(config_dir)
    os.environ["PRL_ATTEMPT_LOG_DIR"] = str(log_dir)
    (config_dir / "eval.json").write_text(json.dumps(dump_resolved_config(config), indent=2))
    setup_logger(config.log.level, json_logging=config.log.json_logging)

    import prime_rl_orch_rs

    from prime_rl.orchestrator.rust import PyBridge
    from prime_rl.orchestrator.utils import intercept_vf_logging

    intercept_vf_logging(logger="verifiers.v1", level="WARN")
    prime_rl_orch_rs.run_online_eval(config, PyBridge(config))


if __name__ == "__main__":
    main()
