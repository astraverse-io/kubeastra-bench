"""Tests for the strict-vs-fuzzy diff appliers.

They pin the study's core claims: (a) an offset-only-wrong diff is rejected by
the strict applier but recovered by the YAML-safe offset-tolerant one; (b) a
mis-indented diff is *correctly* rejected by the offset-tolerant applier while
the whitespace-insensitive one 'applies' it and corrupts the file; (c) a
genuinely wrong diff is rejected by everything; (d) ambiguity is refused.
"""
from __future__ import annotations
import sys
from pathlib import Path

import shutil

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from appliers import (  # noqa: E402
    apply_strict, apply_offset_tolerant, apply_ws_insensitive,
    apply_patch_fuzz, available_appliers,
)

M = ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\nspec:\n"
     "  replicas: 2            # runbook\n  minReadySeconds: 10\n")

# correct content, correct line number
GOOD = ("@@ -6,1 +6,1 @@\n"
        "-  replicas: 2            # runbook\n"
        "+  replicas: 3            # runbook\n")
# correct content, WRONG line number (@@ says line 1)
WRONG_LINE = ("@@ -1,1 +1,1 @@\n"
              "-  replicas: 2            # runbook\n"
              "+  replicas: 3            # runbook\n")
# right idea, WRONG indentation (model dropped the two-space indent)
WRONG_INDENT = ("@@ -6,1 +6,1 @@\n"
                "-replicas: 2            # runbook\n"
                "+replicas: 3            # runbook\n")
# genuinely wrong context — matches nothing
WRONG_CONTENT = ("@@ -6,1 +6,1 @@\n"
                 "-  replicas: 9            # nope\n"
                 "+  replicas: 3\n")


def test_good_diff_applies_everywhere():
    for fn in (apply_strict, apply_offset_tolerant, apply_ws_insensitive):
        out = fn(M, GOOD)
        assert out is not None and "  replicas: 3            # runbook" in out


def test_offset_recovers_what_strict_rejects():
    # the whole point: content is right, only the line number is wrong
    assert apply_strict(M, WRONG_LINE) is None                 # strict: hard fail
    out = apply_offset_tolerant(M, WRONG_LINE)
    assert out is not None and "  replicas: 3            # runbook" in out
    assert "  minReadySeconds: 10" in out                       # nothing else moved


def test_ws_insensitive_applies_but_corrupts_yaml():
    # offset-tolerant is YAML-safe: it refuses a mis-indented hunk
    assert apply_offset_tolerant(M, WRONG_INDENT) is None
    # ws-insensitive 'applies' it — and writes replicas at column 0 (broken)
    out = apply_ws_insensitive(M, WRONG_INDENT)
    assert out is not None
    assert "\nreplicas: 3" in out              # unindented → structurally wrong
    assert "  replicas: 3" not in out          # the correct (indented) form is absent


def test_wrong_content_rejected_by_all():
    for fn in (apply_strict, apply_offset_tolerant, apply_ws_insensitive):
        assert fn(M, WRONG_CONTENT) is None


def test_ambiguous_block_is_refused():
    dup = "a:\n  x: 1\nb:\n  x: 1\n"
    diff = "@@ -2,1 +2,1 @@\n-  x: 1\n+  x: 2\n"     # '  x: 1' appears twice
    assert apply_offset_tolerant(dup, diff) is None            # won't guess which


def test_registry_has_python_appliers():
    avail = available_appliers()
    assert "strict" in avail and "offset_tolerant" in avail and "ws_insensitive" in avail


@pytest.mark.skipif(not shutil.which("patch"), reason="GNU patch not installed")
def test_patch_fuzz_recovers_offset_when_available():
    out = apply_patch_fuzz(M, WRONG_LINE)
    assert out is not None and "replicas: 3" in out
