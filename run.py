"""Benchmark orchestrator.

Runs each system over each task for k seeds, judges every output with the
oracle, and aggregates the metrics the paper reports. Fully runnable with just
the SUT (deterministic, no model); the LLM baselines B1–B3 need a provider,
selected with --model / the LLM_PROVIDER env (see systems/llm.py).

    # SUT only (no model needed):
    python run.py --systems SUT --seeds 1

    # SUT + baselines on a configured provider, 5 seeds:
    python run.py --systems SUT,B1,B2,B3 --seeds 5 --model claude-opus-4-8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from oracle import Intent, judge                    # noqa: E402
from systems import sut                              # noqa: E402
from systems.base import SystemOutput               # noqa: E402


def _intent_obj(intent: dict) -> Intent:
    return Intent(kind=intent["kind"], name=intent["name"],
                  field_path=tuple(intent["field_path"]),
                  new_value=intent["new_value"], namespace=intent.get("namespace"))


def build_systems(names: list[str], model: str | None, max_tokens: int = 16000):
    """Map requested system names to callables `(original, intent) -> SystemOutput`."""
    systems: dict[str, object] = {}
    need_llm = any(n in ("B1", "B2", "B3") for n in names)
    llm = None
    if need_llm:
        from systems.llm import KubeAstraLLM
        llm = KubeAstraLLM(model=model, max_tokens=max_tokens)
        # Construct the provider once, single-threaded, so concurrent workers
        # never race on the lazy import / sys.path insert on first use.
        llm._get_provider()
    for n in names:
        if n == "SUT":
            systems["SUT-span-edit"] = sut.run
        elif n == "B1":
            from systems.baselines import B1FullFile
            systems[B1FullFile.name] = B1FullFile(llm).run
        elif n == "B2":
            from systems.baselines import B2UnifiedDiff
            systems[B2UnifiedDiff.name] = B2UnifiedDiff(llm).run
        elif n == "B3":
            from systems.baselines import B3DiffRetry
            systems[B3DiffRetry.name] = B3DiffRetry(llm).run
        else:
            raise SystemExit(f"unknown system: {n}")
    return systems


def build_row(task: dict, sys_name: str, seed: int, out: SystemOutput,
              original: str, intent_obj: Intent) -> dict:
    """Judge one system's output and shape the result row. Shared by the
    synchronous runner and the batched runner so both emit identical schemas."""
    row = {"task_id": task["task_id"], "field_type": task["intent"]["field_type"],
           "system": sys_name, "seed": seed,
           "refused": out.refused, "reason": out.reason}
    if out.refused:
        row.update(correct=False, collateral=None, fabricated=None,
                   comments_preserved=None, format_preserved=None,
                   diff_added=None, diff_removed=None)
    else:
        v = judge(original, out.output_text, intent_obj)
        row.update(correct=v.correct,
                   collateral=len(v.collateral_changes),
                   fabricated=len(v.fabricated_paths),
                   comments_preserved=v.comments_preserved,
                   format_preserved=v.format_preserved,
                   diff_added=v.diff_added, diff_removed=v.diff_removed,
                   reason=v.reason)
    return row


def _run_cell(job: tuple) -> dict:
    """Run one (task, system, seed) cell and judge it. Pure w.r.t. shared state
    — reads only its captured args — so it is safe to run on a worker thread."""
    task, sys_name, fn, seed, original, intent, intent_obj = job
    try:
        out: SystemOutput = fn(original, intent)
    except Exception as exc:                       # a provider error, etc.
        out = SystemOutput.refuse(f"error: {exc.__class__.__name__}: {exc}")
    return build_row(task, sys_name, seed, out, original, intent_obj)


def run_matrix(tasks: list[dict], systems: dict, seeds: int, corpus_dir: Path,
               concurrency: int = 1) -> list[dict]:
    """Every (task, system, seed) is an independent cell. With concurrency > 1
    they run on a thread pool — the calls are I/O-bound on the LLM API, so this
    cuts wall-clock ~Nx (bounded by the provider's rate limits, which the LLM
    adapter's backoff absorbs). Manifests are read once, up front, so the pool
    shares only read-only data; `executor.map` preserves submission order, so
    `rows.jsonl` is byte-identical whatever the concurrency."""
    cache: dict[str, str] = {}
    jobs: list[tuple] = []
    for task in tasks:
        mp = task["manifest_path"]
        original = cache.setdefault(mp, (corpus_dir / mp).read_text())
        intent = task["intent"]
        intent_obj = _intent_obj(intent)
        for sys_name, fn in systems.items():
            for seed in range(seeds):
                jobs.append((task, sys_name, fn, seed, original, intent, intent_obj))

    if concurrency <= 1:
        return [_run_cell(j) for j in jobs]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(_run_cell, jobs))


def aggregate(rows: list[dict]) -> dict:
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)

    summary = {}
    for sys_name, rs in by_sys.items():
        n = len(rs)
        applied = [r for r in rs if not r["refused"]]
        correct = [r for r in rs if r["correct"]]
        diffs = [r["diff_added"] + r["diff_removed"] for r in applied]
        summary[sys_name] = {
            "runs": n,
            "correct_rate": round(len(correct) / n, 4) if n else 0.0,
            "refused_rate": round(sum(r["refused"] for r in rs) / n, 4) if n else 0.0,
            "collateral_rate": round(
                sum(1 for r in applied if r["collateral"]) / n, 4) if n else 0.0,
            "fabrication_rate": round(
                sum(1 for r in applied if r["fabricated"]) / n, 4) if n else 0.0,
            "comments_preserved_rate": round(
                sum(1 for r in applied if r["comments_preserved"]) / n, 4) if n else 0.0,
            "format_preserved_rate": round(
                sum(1 for r in applied if r["format_preserved"]) / n, 4) if n else 0.0,
            "diff_size_mean": round(mean(diffs), 2) if diffs else None,
            "diff_size_std": round(pstdev(diffs), 2) if len(diffs) > 1 else 0.0,
        }
    return summary


def _print_summary(summary: dict) -> None:
    cols = ["correct_rate", "refused_rate", "collateral_rate", "fabrication_rate",
            "format_preserved_rate", "diff_size_mean"]
    width = max(len(s) for s in summary) if summary else 10
    print(f"\n{'system':<{width}}  " + "  ".join(f"{c:>18}" for c in cols))
    for s, m in summary.items():
        print(f"{s:<{width}}  " + "  ".join(f"{str(m[c]):>18}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="SUT",
                    help="comma-separated: SUT,B1,B2,B3")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--model", default=None, help="provider model id for baselines")
    ap.add_argument("--tasks", default=str(HERE / "tasks.jsonl"))
    ap.add_argument("--corpus", default=str(HERE / "corpus"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel worker threads for LLM calls (1 = sequential)")
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="output cap; must exceed the largest file so B1 can "
                         "emit it whole (a low cap silently truncates B1)")
    ap.add_argument("--batch", action="store_true",
                    help="use the Anthropic Message Batches API (50%% cheaper, "
                         "async); Anthropic provider only")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    names = [s.strip() for s in args.systems.split(",") if s.strip()]

    batch_usage = None
    if args.batch:
        from batch import run_batched, anthropic_executor
        executor, model = anthropic_executor(args.model)
        rows = run_batched(tasks, names, args.seeds, Path(args.corpus),
                           executor, model, args.max_tokens)
        batch_usage = executor.usage
    else:
        systems = build_systems(names, args.model, args.max_tokens)
        rows = run_matrix(tasks, systems, args.seeds, Path(args.corpus),
                          concurrency=args.concurrency)
    summary = aggregate(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if batch_usage is not None:
        (out_dir / "usage.json").write_text(json.dumps(batch_usage, indent=2))

    _print_summary(summary)
    if batch_usage is not None:
        print(f"\nbatch token usage: {batch_usage}")
    print(f"\nwrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
