"""Launcher for the orchestrator.

Parses ``OrchestratorConfig`` (the config API stays in Python) and runs the Rust
orchestrator (``prime_rl_orch_rs``) through ``prime_rl.rust_bridge``. Heavy
imports are deferred until after ``cli()`` so ``orchestrator --help`` stays fast.
"""

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def main():
    set_proc_title("Orchestrator")
    config = cli(OrchestratorConfig)
    from prime_rl.rust_bridge import run_rust_orchestrator

    run_rust_orchestrator(config)


if __name__ == "__main__":
    main()
