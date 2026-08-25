"""Tests for the G5 refusal/ambiguity stratum — regression-lock the fail-closed
behavior, including the one documented leak (YAML aliases)."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from systems import sut                                # noqa: E402
import refusal                                         # noqa: E402

pytestmark = pytest.mark.skipif(not sut.AVAILABLE, reason="span-edit pipeline unavailable")


def _rows():
    return {r["id"]: r for r in (refusal.evaluate_case(c) for c in refusal.CASES)}


def test_controls_edit_correctly():
    rows = _rows()
    for r in rows.values():
        if r["expected"] == "edit":
            assert r["refused"] is False and r["correct_edit"] is True, r


def test_refusal_precision_is_perfect():
    # no resolvable/control case is ever refused
    rows = _rows()
    for r in rows.values():
        if r["expected"] == "edit":
            assert r["refused"] is False


def test_fail_closed_holds_except_the_known_alias_leak():
    rows = _rows()
    for r in rows.values():
        if r["expected"] == "refuse" and r["id"] != "alias-target":
            assert r["refused"] is True, f"{r['id']} should refuse but did not"


def test_alias_target_is_the_documented_leak():
    # Fail-closed *should* fire here but currently doesn't: an alias resolves to
    # its anchor node, so the locator edits the wrong line. If this ever starts
    # refusing (a fix), this test flips — update §7 of the paper accordingly.
    r = _rows()["alias-target"]
    assert r["refused"] is False and r["as_expected"] is False


def test_study_summary_numbers():
    s = refusal.run_study()
    assert s["refusal_recall"] == 0.889          # 8/9 should-refuse cases refuse
    assert s["refusal_precision"] == 1.0          # every refusal warranted
    assert s["control_coverage"] == 1.0           # every resolvable case edited right
    assert s["leaks"] == ["alias-target"]         # the single documented gap
