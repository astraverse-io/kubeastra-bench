"""Baseline tests using FakeLLM — no real model, no API key."""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from oracle import Intent, judge                       # noqa: E402
from systems.llm import FakeLLM                        # noqa: E402
from systems.baselines import (                        # noqa: E402
    B1FullFile, B2UnifiedDiff, B3DiffRetry, apply_unified_diff, strip_fences,
)

# A complete manifest so the oracle can locate Deployment/x. `replicas` is on
# line 6 (1-based), which the unified diffs below target.
ORIG = ("apiVersion: apps/v1\n"
        "kind: Deployment\n"
        "metadata:\n"
        "  name: x\n"
        "spec:\n"
        "  replicas: 2            # runbook\n"
        "  minReadySeconds: 10\n")

INTENT = {"kind": "Deployment", "name": "x", "field_path": ["spec", "replicas"],
          "field_type": "replicas", "new_value": 3}

GOOD_DIFF = ("@@ -6,1 +6,1 @@\n"
             "-  replicas: 2            # runbook\n"
             "+  replicas: 3            # runbook\n")


def _intent():
    return Intent("Deployment", "x", ("spec", "replicas"), 3)


def _with_replicas(value: int, keep_comment: bool = True) -> str:
    line = f"  replicas: {value}            # runbook" if keep_comment else f"  replicas: {value}"
    return ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\n"
            "spec:\n" + line + "\n  minReadySeconds: 10\n")


# ── helpers ───────────────────────────────────────────────────────────────────

def test_strip_fences():
    assert strip_fences("```yaml\nspec: 1\n```") == "spec: 1"
    assert strip_fences("no fences") == "no fences"


def test_apply_unified_diff_clean_and_mismatch():
    out = apply_unified_diff(ORIG, GOOD_DIFF)
    assert out is not None and "replicas: 3" in out and "minReadySeconds: 10" in out
    bad = ("@@ -6,1 +6,1 @@\n"
           "-  replicas: 9            # wrong context\n"
           "+  replicas: 3\n")
    assert apply_unified_diff(ORIG, bad) is None       # context mismatch → fail, not misapply


# ── B1: full file ─────────────────────────────────────────────────────────────

def test_b1_correct_edit():
    out = B1FullFile(FakeLLM([_with_replicas(3)])).run(ORIG, INTENT)
    assert not out.refused
    assert judge(ORIG, out.output_text, _intent()).correct


def test_b1_reflow_is_caught_as_not_faithful():
    out = B1FullFile(FakeLLM([_with_replicas(3, keep_comment=False)])).run(ORIG, INTENT)
    v = judge(ORIG, out.output_text, _intent())
    assert v.correct                 # semantically right
    assert not v.comments_preserved  # but dropped the comment


def test_b1_fenced_output_is_unwrapped():
    out = B1FullFile(FakeLLM(["```yaml\n" + _with_replicas(3) + "```"])).run(ORIG, INTENT)
    assert judge(ORIG, out.output_text, _intent()).correct


# ── B2: unified diff ──────────────────────────────────────────────────────────

def test_b2_correct_diff():
    out = B2UnifiedDiff(FakeLLM([GOOD_DIFF])).run(ORIG, INTENT)
    assert not out.refused
    assert judge(ORIG, out.output_text, _intent()).correct


def test_b2_unapplyable_diff_refuses():
    out = B2UnifiedDiff(FakeLLM(["@@ -6,1 +6,1 @@\n-  nope\n+  x\n"])).run(ORIG, INTENT)
    assert out.refused and "did not apply" in out.reason


# ── B3: diff + retry ──────────────────────────────────────────────────────────

def test_b3_recovers_on_retry():
    bad = "@@ -6,1 +6,1 @@\n-  nope\n+  x\n"                 # won't apply
    llm = FakeLLM([bad, GOOD_DIFF])                         # 1st fails, retry succeeds
    out = B3DiffRetry(llm, retries=2).run(ORIG, INTENT)
    assert not out.refused
    assert judge(ORIG, out.output_text, _intent()).correct
    assert len(llm.calls) == 2                              # retried exactly once


def test_b3_gives_up_after_retries():
    llm = FakeLLM(["@@ -6,1 +6,1 @@\n-  nope\n+  x\n"])      # always bad
    out = B3DiffRetry(llm, retries=2).run(ORIG, INTENT)
    assert out.refused and "gave up after 3 attempts" in out.reason
    assert len(llm.calls) == 3
