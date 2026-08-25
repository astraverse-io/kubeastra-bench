"""Strict-vs-fuzzy applier study (peer-review blocker #1).

Reviewers rightly note that the main benchmark applies model diffs STRICTLY,
which is a lower bound. This study re-applies the *same* model-authored B2 diffs
under every available applier (strict, YAML-safe offset-tolerant, whitespace-
insensitive, and GNU patch with fuzz) and scores each with the oracle — so we can
separate "the diff didn't line up" from "the diff was genuinely wrong," and show
that whitespace-fuzzy application trades rejections for silent misapplication.

Design: the model's raw diffs are captured ONCE to `raw-diffs.jsonl`; scoring is
free and reproducible, so adding an applier or re-scoring never spends API calls.

    # capture (spends) + score:
    python applier_study.py --model claude-sonnet-5 --seeds 3 \
        --tasks tasks-real.jsonl --corpus corpus-real --out results/appliers
    # re-score only (free — reuses results/appliers/raw-diffs.jsonl):
    python applier_study.py --tasks tasks-real.jsonl --corpus corpus-real \
        --out results/appliers
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from appliers import available_appliers               # noqa: E402
from oracle import Intent, judge                       # noqa: E402
from systems import prompts                            # noqa: E402
from systems.baselines import strip_fences            # noqa: E402


def _intent_obj(intent: dict) -> Intent:
    return Intent(kind=intent["kind"], name=intent["name"],
                  field_path=tuple(intent["field_path"]),
                  new_value=intent["new_value"], namespace=intent.get("namespace"))


def _load_corpus(corpus_dir: Path):
    cache: dict[str, str] = {}

    def original(mp: str) -> str:
        if mp not in cache:
            cache[mp] = (corpus_dir / mp).read_text()
        return cache[mp]
    return original


# ── phase 1: capture the model's raw B2 diffs (spends; cached) ─────────────────

def capture_diffs(tasks, seeds, corpus_dir, model, max_tokens, out_path: Path) -> list[dict]:
    if out_path.exists():
        print(f"reusing captured diffs: {out_path}")
        return [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]

    from systems.llm import KubeAstraLLM
    llm = KubeAstraLLM(model=model, max_tokens=max_tokens)
    llm._get_provider()                                # fail fast on misconfig
    original = _load_corpus(corpus_dir)
    records: list[dict] = []
    for task in tasks:
        o, intent = original(task["manifest_path"]), task["intent"]
        for seed in range(seeds):
            system, user = prompts.build_unified_diff(o, intent)
            try:
                raw = strip_fences(llm.complete(system, user))
                err = None
            except Exception as exc:                   # provider error
                raw, err = "", f"{exc.__class__.__name__}: {exc}"
            records.append({"task_id": task["task_id"], "seed": seed,
                            "manifest_path": task["manifest_path"],
                            "intent": intent, "raw_diff": raw, "error": err})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"captured {len(records)} raw diffs -> {out_path}")
    return records


# ── phase 2: apply under every applier + score (free) ─────────────────────────

def score(records: list[dict], corpus_dir: Path) -> dict:
    original = _load_corpus(corpus_dir)
    appliers = available_appliers()
    tally = {name: defaultdict(int) for name in appliers}     # name -> counters
    for rec in records:
        if rec.get("error"):                            # never produced a diff
            for name in appliers:
                tally[name]["provider_error"] += 1
            continue
        o = original(rec["manifest_path"])
        io = _intent_obj(rec["intent"])
        for name, fn in appliers.items():
            tally[name]["n"] += 1
            try:
                patched = fn(o, rec["raw_diff"])
            except Exception:
                patched = None
            if patched is None:
                tally[name]["rejected"] += 1
                continue
            tally[name]["applied"] += 1
            v = judge(o, patched, io)
            if v.correct:
                tally[name]["correct"] += 1
            else:
                tally[name]["misapplied"] += 1          # applied but wrong (unsafe!)

    summary = {}
    for name, c in tally.items():
        n = c["n"] or 1
        summary[name] = {
            "n": c["n"],
            "apply_rate": round(c["applied"] / n, 4),
            "correct_rate": round(c["correct"] / n, 4),
            "misapplied_rate": round(c["misapplied"] / n, 4),   # applied but wrong
            "rejected_rate": round(c["rejected"] / n, 4),
        }
    return summary


def _print(summary: dict) -> None:
    cols = ["n", "apply_rate", "correct_rate", "misapplied_rate", "rejected_rate"]
    w = max((len(s) for s in summary), default=10)
    print(f"\n{'applier':<{w}}  " + "  ".join(f"{c:>15}" for c in cols))
    for name, m in summary.items():
        print(f"{name:<{w}}  " + "  ".join(f"{str(m[c]):>15}" for c in cols))
    print("\napply_rate = fraction that applied; correct_rate = applied AND right;")
    print("misapplied_rate = applied but WRONG (the cost of unsafe leniency).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="required for capture (not for re-score)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16000)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    out_dir = Path(args.out)
    raw_path = out_dir / "raw-diffs.jsonl"
    if not raw_path.exists() and not args.model:
        raise SystemExit("no captured diffs and no --model to capture them")

    records = capture_diffs(tasks, args.seeds, Path(args.corpus),
                            args.model, args.max_tokens, raw_path)
    summary = score(records, Path(args.corpus))
    (out_dir / "applier-summary.json").write_text(json.dumps(summary, indent=2))
    _print(summary)
    print(f"\nwrote {out_dir/'applier-summary.json'}")


if __name__ == "__main__":
    main()
