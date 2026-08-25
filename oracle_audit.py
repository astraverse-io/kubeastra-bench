"""Independent manual audit of the correctness oracle (paper Sec 7, oracle validity).

Every rate in the paper rests on `oracle.judge()`. This harness surfaces a
random, stratified sample of REAL cases with enough context for a human to
adjudicate the ground-truth verdict independently, then compare to the oracle:

  - SUT edits (deterministic span-edit output),
  - B2 diffs applied with GNU `patch --fuzz` that the oracle called CORRECT,
  - B2 diffs applied with GNU `patch --fuzz` that the oracle called MISAPPLIED
    (applied-but-wrong) -- the cases the 14-20% headline depends on.

For each case it prints the intent, the original->output unified diff, and the
oracle's verdict. Deterministic (fixed seed) so the audit is reproducible.

    python oracle_audit.py --tasks tasks-real.jsonl --corpus corpus-real \
        --diffs results/appliers/raw-diffs.jsonl --per-stratum 10
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from oracle import Intent, judge                       # noqa: E402
from appliers import apply_patch_fuzz                  # noqa: E402
from systems import sut                                # noqa: E402


def _intent_obj(intent: dict) -> Intent:
    return Intent(kind=intent["kind"], name=intent["name"],
                  field_path=tuple(intent["field_path"]),
                  new_value=intent["new_value"], namespace=intent.get("namespace"))


def _describe(intent: dict) -> str:
    fp = ".".join(map(str, intent["field_path"]))
    return (f"{intent['kind']}/{intent['name']}  set  {fp}  ->  "
            f"{intent['new_value']!r}")


def _trim_diff(original: str, output: str) -> str:
    lines = list(difflib.unified_diff(
        original.splitlines(), output.splitlines(), n=1, lineterm=""))
    body = [l for l in lines if not l.startswith(("---", "+++"))]
    return "\n".join("      " + l for l in body) if body else "      (no change)"


def _verdict_line(v) -> str:
    return (f"correct={v.correct}  target_found={v.target_found}  "
            f"collateral={[list(k) for k in v.collateral_changes][:3]}  "
            f"diff=+{v.diff_added}/-{v.diff_removed}\n      reason: {v.reason}")


def build_cases(tasks, corpus_dir: Path, diffs_path: Path):
    corpus: dict[str, str] = {}

    def orig(mp: str) -> str:
        if mp not in corpus:
            corpus[mp] = (corpus_dir / mp).read_text()
        return corpus[mp]

    sut_cases, b2_correct, b2_misapplied = [], [], []

    # SUT stratum (deterministic) — oracle should call these correct.
    if sut.AVAILABLE:
        for t in tasks:
            o, intent = orig(t["manifest_path"]), t["intent"]
            out = sut.run(o, intent)
            if out.refused:
                continue
            v = judge(o, out.output_text, _intent_obj(intent))
            sut_cases.append(("SUT", t["task_id"], intent, o, out.output_text, v))

    # B2 stratum — apply the tolerant tool, split by the oracle's call.
    for rec in (json.loads(l) for l in diffs_path.read_text().splitlines() if l.strip()):
        if rec.get("error"):
            continue
        o, intent = orig(rec["manifest_path"]), rec["intent"]
        patched = None
        try:
            patched = apply_patch_fuzz(o, rec["raw_diff"])
        except Exception:
            patched = None
        if patched is None:
            continue                                   # rejected, not an oracle case
        v = judge(o, patched, _intent_obj(intent))
        tag = ("B2-patchfuzz", rec["task_id"], intent, o, patched, v)
        (b2_correct if v.correct else b2_misapplied).append(tag)

    return {"SUT (oracle=correct)": sut_cases,
            "B2 patch-fuzz (oracle=CORRECT)": b2_correct,
            "B2 patch-fuzz (oracle=MISAPPLIED)": b2_misapplied}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--diffs", required=True)
    ap.add_argument("--per-stratum", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    strata = build_cases(tasks, Path(args.corpus), Path(args.diffs))
    rng = random.Random(args.seed)

    print(f"# Oracle audit — stratified sample (seed={args.seed}, "
          f"{args.per_stratum}/stratum)\n")
    for name, pool in strata.items():
        print(f"## {name}   (pool: {len(pool)})")
        sample = rng.sample(pool, min(args.per_stratum, len(pool)))
        for i, (system, tid, intent, o, out, v) in enumerate(sample, 1):
            print(f"\n[{name.split()[0]}-{i}] {system}  task={tid}")
            print(f"  INTENT: {_describe(intent)}")
            print(f"  DIFF (original -> output):\n{_trim_diff(o, out)}")
            print(f"  ORACLE: {_verdict_line(v)}")
        print()


if __name__ == "__main__":
    main()
