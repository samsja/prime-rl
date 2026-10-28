"""Compare two prime-rl run directories (e.g. a Python-orchestrator baseline and a Rust-orchestrator run).

Usage: uv run --no-sync python rust/tools/e2e_compare.py RUN_A RUN_B [--steps N] [--threshold 0.25]

RUN_A / RUN_B are run directories (holding monitors/, logs/, checkpoints/) or an output dir with a
single run directory inside. Sections:
  layout     relative file layout (step numbers, chunk numbers and ids normalized)
  metrics    metrics.jsonl: key sets per producer and per step, key-metric trends, large deviations
  traces     trace stream / annotations / index: record counts per kind, record schemas
  logs       orchestrator.log line shapes (numbers, ids, paths normalized)
  ckpt       checkpoint layout and orchestrator progress.pt contents

Trainer-owned payloads (broadcasts/, weights/, checkpoints/step_*/trainer) are ignored: they are
not written by the orchestrator, and parity runs prune them to save disk.
"""

import argparse
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

KEY_METRICS = [
    r"^train/agg/(all|effective)/agent/reward/mean$",
    r"^train/agg/effective/agent/num_(output|total|input)_tokens/mean$",
    r"^train/agg/effective/agent/num_turns/mean$",
    r"^train/agg/all/agent/is_truncated/mean$",
    r"^train/agg/all/has_error/mean$",
    r"^staleness/",
    r"^eval/.*/(all|effective)/agent/reward/mean$",
    r"^eval/.*/avg@",
    r"^progress/",
    r"^rollout/(attempts|discarded|stale|errored|cancelled|no_signal)$",
    r"^time/",
    r"^perf/",
    r"^throughput",
    r"tokens_per_s",
    r"^loss/",
    r"^optim/",
    r"^mismatch",
]

SECTION = "=" * 100


def resolve_run(path: Path) -> Path:
    if (path / "monitors").is_dir() or (path / "logs").is_dir():
        return path
    runs = [p for p in path.iterdir() if p.is_dir() and ((p / "monitors").is_dir() or (p / "logs").is_dir())]
    if len(runs) != 1:
        raise SystemExit(f"{path}: expected a run directory or exactly one run inside, found {[p.name for p in runs]}")
    return runs[0]


def normalize_path(rel: str) -> str:
    rel = re.sub(r"step_\d+", "step_N", rel)
    rel = re.sub(r"attempt_\d+", "attempt_N", rel)
    rel = re.sub(r"\b\d{5}\.jsonl", "NNNNN.jsonl", rel)
    rel = re.sub(r"[0-9a-f]{32}", "<id>", rel)
    rel = re.sub(r"[0-9a-f]{8,}(-[0-9a-f]+)*", "<hex>", rel)
    rel = re.sub(r"rank_\d+", "rank_N", rel)
    rel = re.sub(r"<id>_\w+", "<id>", rel)  # torchrun run dirs
    rel = re.sub(r"\.\d+\.tmp$", ".tmp", rel)
    return rel


def section(title: str) -> None:
    print(f"\n{SECTION}\n{title}\n{SECTION}")


def show_set_diff(name: str, a: set, b: set, limit: int = 40) -> int:
    only_a, only_b = sorted(a - b), sorted(b - a)
    print(f"{name}: {len(a)} vs {len(b)} ({len(a & b)} shared)")
    for label, items in (("only in A", only_a), ("only in B", only_b)):
        if items:
            print(f"  {label} ({len(items)}):")
            for item in items[:limit]:
                print(f"    {item}")
            if len(items) > limit:
                print(f"    ... {len(items) - limit} more")
    return len(only_a) + len(only_b)


# ---------------------------------------------------------------------------- layout


def layout(run: Path) -> dict[str, int]:
    counts: Counter = Counter()
    for path in run.rglob("*"):
        rel = path.relative_to(run).as_posix()
        # Trainer-owned payloads (weights, trainer checkpoints) are unchanged by the orchestrator and
        # are pruned after runs to save disk.
        if (
            rel in ("weights", "broadcasts")
            or rel.startswith(("weights/", "broadcasts/"))
            or re.match(r"checkpoints/step_\d+/trainer", rel)
        ):
            continue
        counts[normalize_path(rel) + ("/" if path.is_dir() else "")] += 1
    return counts


def compare_layout(a: Path, b: Path) -> int:
    section("LAYOUT (normalized relative paths)")
    la, lb = layout(a), layout(b)
    return show_set_diff("paths", set(la), set(lb))


# ---------------------------------------------------------------------------- metrics


def load_metrics(run: Path) -> list[dict]:
    path = run / "monitors" / "file" / "metrics.jsonl"
    if not path.is_file():
        print(f"  (no {path})")
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_producer(rows: list[dict]) -> dict[str, dict[int | None, dict]]:
    """producer -> step -> merged row (time-keyed rows under step None are merged by key set only)."""
    out: dict[str, dict] = defaultdict(dict)
    for row in rows:
        producer = row.get("producer") or "-"
        step = row.get("step")
        merged = out[producer].setdefault(step, {})
        merged.update({k: v for k, v in row.items() if k not in ("time", "producer")})
    return out


def numeric(value) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def compare_metrics(a: Path, b: Path, steps: int | None, threshold: float) -> int:
    section("METRICS (monitors/file/metrics.jsonl)")
    pa, pb = by_producer(load_metrics(a)), by_producer(load_metrics(b))
    issues = show_set_diff("producers", set(pa), set(pb))
    for producer in sorted(set(pa) & set(pb)):
        print(f"\n--- producer {producer}")
        sa = {s: r for s, r in pa[producer].items() if s is not None and (steps is None or s <= steps)}
        sb = {s: r for s, r in pb[producer].items() if s is not None and (steps is None or s <= steps)}
        print(
            f"steps: A {sorted(sa)[:3]}..{max(sa, default=None)} ({len(sa)})  B {sorted(sb)[:3]}..{max(sb, default=None)} ({len(sb)})"
        )
        keys_a = set().union(*(r.keys() for r in sa.values())) if sa else set()
        keys_b = set().union(*(r.keys() for r in sb.values())) if sb else set()
        issues += show_set_diff("step-keyed metric keys (union over steps)", keys_a, keys_b)
        ta = set().union(*(r.keys() for s, r in pa[producer].items() if s is None)) if None in pa[producer] else set()
        tb = set().union(*(r.keys() for s, r in pb[producer].items() if s is None)) if None in pb[producer] else set()
        if ta or tb:
            issues += show_set_diff("time-keyed metric keys", ta, tb)

        def value_types(rows: dict) -> dict[str, set]:
            types: dict[str, set] = defaultdict(set)
            for row in rows.values():
                for key, value in row.items():
                    types[key].add(type(value).__name__)
            return types

        types_a, types_b = value_types(pa[producer]), value_types(pb[producer])
        type_diffs = [
            (k, sorted(types_a[k]), sorted(types_b[k]))
            for k in sorted(set(types_a) & set(types_b))
            if types_a[k] != types_b[k]
        ]
        if type_diffs:
            print(f"JSON value types differ ({len(type_diffs)}; e.g. int vs float):")
            for key, ta, tb in type_diffs[:30]:
                print(f"  {key}: A {ta} B {tb}")
            issues += len(type_diffs)
        per_step = []
        for step in sorted(set(sa) & set(sb)):
            ka, kb = set(sa[step]), set(sb[step])
            if ka != kb:
                per_step.append((step, sorted(ka - kb), sorted(kb - ka)))
        if per_step:
            print(f"steps whose key sets differ ({len(per_step)}):")
            for step, only_a, only_b in per_step[:10]:
                print(
                    f"  step {step}: only A {only_a[:6]}{'...' if len(only_a) > 6 else ''} | only B {only_b[:6]}{'...' if len(only_b) > 6 else ''}"
                )
        shared = sorted(keys_a & keys_b)
        common_steps = sorted(set(sa) & set(sb))
        key_metrics = [k for k in shared if any(re.search(p, k) for p in KEY_METRICS)]
        if key_metrics and common_steps:
            print("\nkey metrics (mean over common steps | first | last), A vs B:")
            for key in key_metrics:
                va = [numeric(sa[s].get(key)) for s in common_steps]
                vb = [numeric(sb[s].get(key)) for s in common_steps]
                va = [v for v in va if v is not None]
                vb = [v for v in vb if v is not None]
                if not va or not vb:
                    continue
                ma, mb = sum(va) / len(va), sum(vb) / len(vb)
                rel = abs(ma - mb) / max(abs(ma), abs(mb), 1e-12)
                flag = "  <-- differs" if rel > threshold else ""
                print(
                    f"  {key:70s} {ma:12.4g} {va[0]:10.4g} {va[-1]:10.4g} | {mb:12.4g} {vb[0]:10.4g} {vb[-1]:10.4g}{flag}"
                )
        deviations = []
        for key in shared:
            va = [numeric(sa[s].get(key)) for s in common_steps]
            vb = [numeric(sb[s].get(key)) for s in common_steps]
            va = [v for v in va if v is not None]
            vb = [v for v in vb if v is not None]
            if not va or not vb or key == "step":
                continue
            ma, mb = sum(va) / len(va), sum(vb) / len(vb)
            scale = max(abs(ma), abs(mb))
            if scale < 1e-9:
                continue
            rel = abs(ma - mb) / scale
            if rel > threshold:
                deviations.append((rel, key, ma, mb))
        deviations.sort(reverse=True)
        if deviations:
            print(
                f"\nmetrics whose step-mean differs by > {threshold:.0%} ({len(deviations)}; time/* are expected to):"
            )
            for rel, key, ma, mb in deviations[:40]:
                print(f"  {key:70s} A {ma:12.4g}  B {mb:12.4g}  ({rel:.0%})")
    return issues


# ---------------------------------------------------------------------------- traces


def read_chunks(directory: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.jsonl*")):
        if path.name.endswith(".tmp"):
            continue
        if path.suffix == ".zst":
            import pyzstd

            data = pyzstd.decompress(path.read_bytes())
        else:
            data = path.read_bytes()
        records.extend(json.loads(line) for line in io.BytesIO(data).read().splitlines() if line.strip())
    return records


def schema(value, prefix: str = "", out: dict | None = None) -> dict[str, set]:
    """Key path -> set of value type names; list elements share the path `[]`."""
    out = out if out is not None else defaultdict(set)
    if isinstance(value, dict):
        out[prefix or "."].add("dict")
        for key, item in value.items():
            # Free-form maps keyed by data (branch indices, metric names) collapse to `*`.
            key = "*" if re.fullmatch(r"\d+|[0-9a-f]{32}(-\d+)?", str(key)) else key
            schema(item, f"{prefix}.{key}", out)
    elif isinstance(value, list):
        out[prefix or "."].add("list")
        for item in value[:50]:
            schema(item, f"{prefix}[]", out)
    else:
        name = type(value).__name__
        out[prefix or "."].add("number" if name in ("int", "float") else name)
    return out


def merged_schema(records: list[dict]) -> dict[str, set]:
    out: dict[str, set] = defaultdict(set)
    for record in records:
        schema(record, "", out)
    return out


def compare_schemas(name: str, ra: list[dict], rb: list[dict]) -> int:
    sa, sb = merged_schema(ra), merged_schema(rb)
    issues = show_set_diff(f"{name} key paths", set(sa), set(sb))
    type_diffs = [
        (k, sorted(sa[k]), sorted(sb[k]))
        for k in sorted(set(sa) & set(sb))
        if sa[k] - {"NoneType"} != sb[k] - {"NoneType"}
    ]
    if type_diffs:
        print(f"  {name} value types differ ({len(type_diffs)}):")
        for key, ta, tb in type_diffs[:30]:
            print(f"    {key}: A {ta} B {tb}")
    return issues + len(type_diffs)


def episode_kind(record: dict) -> str:
    for trace in record.get("traces") or []:
        if (kind := (trace.get("info") or {}).get("kind")) in ("train", "eval"):
            return kind
    run = record.get("run") or {}
    return (run.get("work") or {}).get("type") or "?"


def read_index(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def compare_traces(a: Path, b: Path) -> int:
    section("TRACES (monitors/file/traces)")
    issues = 0
    base_a, base_b = a / "monitors/file/traces", b / "monitors/file/traces"
    ra, rb = read_chunks(base_a / "stream"), read_chunks(base_b / "stream")
    print(f"\n--- stream: {len(ra)} vs {len(rb)} episode records")
    ca, cb = Counter(episode_kind(r) for r in ra), Counter(episode_kind(r) for r in rb)
    print(f"  episodes per kind: A {dict(ca)}  B {dict(cb)}")
    print(f"  traces: A {sum(len(r.get('traces') or []) for r in ra)}  B {sum(len(r.get('traces') or []) for r in rb)}")
    if ra or rb:
        issues += compare_schemas("stream", ra, rb)
    ia, ib = read_index(base_a / "stream.index.jsonl"), read_index(base_b / "stream.index.jsonl")
    print(f"\n--- stream.index.jsonl: {len(ia)} vs {len(ib)} rows")
    if ia or ib:
        issues += compare_schemas("stream index", ia, ib)
    producers = sorted({p.name for base in (base_a, base_b) for p in (base / "annotations").glob("*") if p.is_dir()})
    for producer in producers:
        ra, rb = read_chunks(base_a / "annotations" / producer), read_chunks(base_b / "annotations" / producer)
        print(f"\n--- annotations/{producer}: {len(ra)} vs {len(rb)} records")
        if ra or rb:
            issues += compare_schemas(f"annotations/{producer}", ra, rb)
        ia = read_index(base_a / "annotations" / f"{producer}.index.jsonl")
        ib = read_index(base_b / "annotations" / f"{producer}.index.jsonl")
        print(f"--- annotations/{producer}.index.jsonl: {len(ia)} vs {len(ib)} rows")
        if ia or ib:
            issues += compare_schemas(f"annotations/{producer} index", ia, ib)
    for name in ("plan.json",):
        pa, pb = a / "monitors/file" / name, b / "monitors/file" / name
        if pa.is_file() or pb.is_file():
            va = json.loads(pa.read_text()) if pa.is_file() else None
            vb = json.loads(pb.read_text()) if pb.is_file() else None
            same = va == vb
            print(f"\n--- {name}: {'identical' if same else f'A {va}  B {vb}'}")
            issues += 0 if same else 1
    return issues


# ---------------------------------------------------------------------------- logs

LOG_PREFIX = re.compile(r"^\S*\s*\d{2}:\d{2}:\d{2}(\.\d+)?\s*")


def line_shape(line: str) -> str:
    line = LOG_PREFIX.sub("", line.rstrip())
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    line = re.sub(r"(/[\w.\-]+)+", "<path>", line)
    line = re.sub(r"[0-9a-f]{8,}", "<hex>", line)
    line = re.sub(r"\d+(\.\d+)?(e[+-]?\d+)?", "<n>", line)
    line = re.sub(r"<n>(ms|s|m|h)\b( <n>(ms|s|m|h)\b)*", "<duration>", line)  # format_time
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def compare_logs(a: Path, b: Path, limit: int) -> int:
    section("LOGS (orchestrator.log line shapes)")
    issues = 0
    for name in ("orchestrator.log",):
        files_a = sorted((a / "logs").glob(f"attempt_*/{name}"))
        files_b = sorted((b / "logs").glob(f"attempt_*/{name}"))
        if not files_a or not files_b:
            print(f"  missing {name}: A {bool(files_a)} B {bool(files_b)}")
            continue
        sa = Counter(line_shape(l) for l in files_a[-1].read_text(errors="replace").splitlines() if l.strip())
        sb = Counter(line_shape(l) for l in files_b[-1].read_text(errors="replace").splitlines() if l.strip())
        print(f"{name}: {sum(sa.values())} vs {sum(sb.values())} lines, {len(sa)} vs {len(sb)} shapes")
        for label, mine, other in (("only in A", sa, sb), ("only in B", sb, sa)):
            only = [(count, shape) for shape, count in mine.items() if shape not in other]
            only.sort(reverse=True)
            if only:
                print(f"  {label} ({len(only)} shapes):")
                for count, shape in only[:limit]:
                    print(f"    {count:5d}x {shape[:180]}")
            issues += len(only)
        for level in ("WARNING", "ERROR"):
            ca = sum(c for s, c in sa.items() if level in s)
            cb = sum(c for s, c in sb.items() if level in s)
            print(f"  {level} lines: A {ca}  B {cb}")
    return issues


# ---------------------------------------------------------------------------- ckpt


def ckpt_summary(run: Path) -> dict:
    out = {}
    for path in sorted((run / "checkpoints").glob("step_*/orchestrator/progress.pt")):
        step = path.parent.parent.name
        try:
            import torch

            state = torch.load(path, weights_only=False)
            progress = state["progress"]
            fields = vars(progress) if hasattr(progress, "__dict__") else progress
            out[step] = {
                "keys": sorted(state),
                "progress": dict(fields),
                "train_source_keys": sorted(state.get("train_source", {})),
                "envs": sorted(state.get("train_source", {}).get("envs", {})),
            }
        except Exception as error:
            out[step] = {"error": repr(error)}
    return out


def compare_ckpt(a: Path, b: Path) -> int:
    section("CHECKPOINTS (checkpoints/step_*/)")
    issues = 0

    def ckpt_files(run: Path) -> set[str]:
        root = run / "checkpoints"
        if not root.is_dir():
            return set()
        rels = (p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        return {rel for rel in rels if not re.match(r"step_\d+/trainer", rel)}

    files_a, files_b = ckpt_files(a), ckpt_files(b)
    issues += show_set_diff("checkpoint files", files_a, files_b)
    ca, cb = ckpt_summary(a), ckpt_summary(b)
    for step in sorted(set(ca) | set(cb), key=lambda s: int(s.split("_")[1])):
        va, vb = ca.get(step), cb.get(step)
        if va is None or vb is None:
            print(f"  {step}: only in {'A' if vb is None else 'B'}")
            issues += 1
            continue
        for label, v in (("A", va), ("B", vb)):
            if "error" in v:
                print(
                    f"  {step}: cannot load {label}'s progress.pt ({v['error']}); run this tool from an env that "
                    "has prime_rl.orchestrator (the baseline worktree)"
                )
                issues += 1
        structure_same = {k: v for k, v in va.items() if k != "progress"} == {
            k: v for k, v in vb.items() if k != "progress"
        }
        print(
            f"  {step}: structure {'same' if structure_same else 'DIFFERS'}; progress A {va.get('progress')}  B {vb.get('progress')}"
        )
        if not structure_same:
            print(f"    A {va}\n    B {vb}")
            issues += 1
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--steps", type=int, default=None, help="only compare steps <= this")
    parser.add_argument("--threshold", type=float, default=0.25, help="relative step-mean difference to flag")
    parser.add_argument("--limit", type=int, default=25, help="max log shapes listed per side")
    args = parser.parse_args()
    a, b = resolve_run(args.run_a), resolve_run(args.run_b)
    print(f"A: {a}\nB: {b}")
    issues = {
        "layout": compare_layout(a, b),
        "metrics": compare_metrics(a, b, args.steps, args.threshold),
        "traces": compare_traces(a, b),
        "logs": compare_logs(a, b, args.limit),
        "ckpt": compare_ckpt(a, b),
    }
    section("SUMMARY (structural differences per section; value deviations are listed above)")
    for name, count in issues.items():
        print(f"  {name:8s} {count}")


if __name__ == "__main__":
    main()
