"""Load every TOML under configs/ and examples/ through the real pydantic-config loader.

Used by the prl-config extraction test (`tests/extract.rs`), which extracts each loaded
config into the Rust structs, so a renamed or removed pydantic field fails there. Each
config lands in one kind:

- ``orchestrator``: `OrchestratorConfig` (an RL config's `.orchestrator`, or standalone)
- ``eval``: `EvalConfig` (`uv run eval`)
- ``sft_online_eval``: `SFTOnlineEvalConfig` (an sft config's `.eval`)

Configs of none of these (inference, sft without eval) are skipped. Overlays load on top of
their base. `extra/<kind>*.toml` cover fields and variants no checked-in config uses.

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


def _load_any(path: Path) -> tuple[str, object] | str:
    """``(kind, config)`` of the first config class that validates the TOML, else the errors."""
    from prime_rl.configs.eval import EvalConfig
    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.configs.rl import RLConfig
    from prime_rl.configs.sft import SFTConfig

    data = tomllib.loads(path.read_text())
    if "orchestrator" in data or "trainer" in data:
        kinds = [("orchestrator", RLConfig, lambda c: c.orchestrator)]
    else:
        kinds = [
            ("orchestrator", OrchestratorConfig, lambda c: c),
            ("eval", EvalConfig, lambda c: c),
            ("sft_online_eval", SFTConfig, lambda c: c.eval),
        ]
    errors = []
    for args in _candidates(path):
        for kind, cls, pick in kinds:
            config = _load(cls, args)
            if isinstance(config, str):
                errors.append(f"{cls.__name__}: {config}")
            elif (picked := pick(config)) is not None:
                return kind, picked
            else:
                errors.append(f"{cls.__name__}: no eval block")
    return " || ".join(errors)


def load_all() -> tuple[dict[str, list[tuple[str, object]]], list[tuple[str, str]]]:
    """``(loaded, skipped)``: ``{kind: [(name, config)]}`` and ``[(name, errors)]``."""
    _setup()
    from prime_rl.configs.eval import EvalConfig, SFTOnlineEvalConfig
    from prime_rl.configs.orchestrator import OrchestratorConfig

    loaded: dict[str, list[tuple[str, object]]] = {"orchestrator": [], "eval": [], "sft_online_eval": []}
    skipped = []
    for path in sorted([*ROOT.glob("configs/**/*.toml"), *ROOT.glob("examples/**/*.toml")]):
        name = str(path.relative_to(ROOT))
        result = _load_any(path)
        if isinstance(result, str):
            skipped.append((name, result))
            continue
        kind, config = result
        loaded[kind].append((name, config))
    extra_classes = {"orchestrator": OrchestratorConfig, "eval": EvalConfig, "sft_online_eval": SFTOnlineEvalConfig}
    for kind, cls in extra_classes.items():
        for path in sorted(EXTRA_DIR.glob(f"{kind}*.toml")):
            config = _load(cls, ["@", str(path)])
            if isinstance(config, str):
                raise ValueError(f"{path}: {config}")
            loaded[kind].append((str(path.relative_to(ROOT)), config))
    return loaded, skipped


if __name__ == "__main__":
    loaded, skipped = load_all()
    for kind, configs in loaded.items():
        print(f"{kind}: {len(configs)}")
        for name, _ in configs:
            print(f"  {name}")
    print(f"skipped {len(skipped)}:")
    for name, error in skipped:
        print(f"  {name}: {error}")
