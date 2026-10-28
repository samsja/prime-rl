"""Load every orchestrator-relevant TOML through the real pydantic-config loader.

Used by the prl-config extraction test (`tests/extract.rs`), which extracts each loaded
`OrchestratorConfig` into the Rust structs, so a renamed or removed pydantic field fails
there. RL configs contribute `.orchestrator`; other configs must parse as an
`OrchestratorConfig` themselves (sft/eval/inference configs don't and are skipped).
Overlays load on top of their base. `extra/*.toml` cover variants no checked-in config uses.

Print what loads: uv run --no-sync python rust/crates/prl-config/tools/load_configs.py
"""

import contextlib
import io
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXTRA_DIR = Path(__file__).resolve().parent / "extra"

OVERLAYS = {
    # Overlays whose base is not a sibling `start.toml` (named in each overlay's header).
    "examples/advanced/glm-4.5-air/ipo-ablation/icepop.toml": "examples/advanced/glm-4.5-air/ipo-ablation/base.toml",
    "examples/advanced/glm-4.5-air/ipo-ablation/ipo.toml": "examples/advanced/glm-4.5-air/ipo-ablation/base.toml",
    "examples/advanced/glm-4.5-air/swe-2-node.toml": "examples/advanced/glm-4.5-air/swe-budget.toml",
    "examples/advanced/glm-4.5-air/swe-4-node.toml": "examples/advanced/glm-4.5-air/swe-budget.toml",
}


def _setup() -> None:
    # Taskset/env packages resolve their config classes at validation time. They are uv
    # workspace members, not necessarily installed, so put their source roots on the path.
    envs = [
        *ROOT.glob("deps/verifiers/environments/*"),
        *ROOT.glob("deps/prime-envs/environments/*/*"),
        *ROOT.glob("deps/prime-envs/packages/*"),
    ]
    for env_dir in sorted(envs):
        if env_dir.is_dir():
            sys.path.append(str(env_dir / "src") if (env_dir / "src").is_dir() else str(env_dir))
    for var in ("PRL_OUTPUT_DIR", "PRIME_LOG_LEVEL", "PRIME_VF_LOG_LEVEL", "PRIME_INFERENCE_URL"):
        os.environ.pop(var, None)
    os.chdir(ROOT)


def _load(cls, args: list[str]):
    """The config, or the error text when it does not validate."""
    from prime_rl.utils.config import cli

    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            return cli(cls, args=[*args, "--output-dir", "outputs"])
    except (SystemExit, Exception) as error:
        return f"{type(error).__name__}: {' | '.join(str(error).splitlines()[:4])}"


def _candidates(path: Path) -> list[list[str]]:
    rel = str(path.relative_to(ROOT))
    if rel in OVERLAYS:
        return [["@", OVERLAYS[rel], "@", rel]]
    args = [["@", rel]]
    base = path.with_name("start.toml")
    if path.name != "start.toml" and base.exists():
        args.append(["@", str(base.relative_to(ROOT)), "@", rel])
    return args


def load_all() -> tuple[list[tuple[str, object]], list[tuple[str, str]]]:
    """``(loaded, skipped)``: ``(name, OrchestratorConfig)`` and ``(name, error)``."""
    _setup()
    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.configs.rl import RLConfig

    loaded, skipped = [], []
    for path in sorted([*ROOT.glob("configs/**/*.toml"), *ROOT.glob("examples/**/*.toml")]):
        name = str(path.relative_to(ROOT))
        data = tomllib.loads(path.read_text())
        cls = RLConfig if "orchestrator" in data or "trainer" in data else OrchestratorConfig
        config = None
        for args in _candidates(path):
            config = _load(cls, args)
            if not isinstance(config, str):
                break
        if isinstance(config, str):
            skipped.append((name, config))
            continue
        loaded.append((name, config.orchestrator if isinstance(config, RLConfig) else config))
    for path in sorted(EXTRA_DIR.glob("*.toml")):
        config = _load(OrchestratorConfig, ["@", str(path)])
        if isinstance(config, str):
            raise ValueError(f"{path}: {config}")
        loaded.append((str(path.relative_to(ROOT)), config))
    return loaded, skipped


if __name__ == "__main__":
    loaded, skipped = load_all()
    print(f"loaded {len(loaded)}:")
    for name, _ in loaded:
        print(f"  {name}")
    print(f"skipped {len(skipped)}:")
    for name, error in skipped:
        print(f"  {name}: {error}")
