"""Integration: the oracle blesses the REAL span-edit pipeline's output.

Runs KubeAstra's actual locator + editor (the system under test) on a manifest
and asserts the oracle judges the result correct, minimal (1-line diff), and
fully faithful. This validates the oracle against the real SUT, and doubles as
a sanity check that the SUT satisfies its own guarantees on a concrete case.

Skips cleanly if the pipeline isn't importable (so the oracle's own tests still
run standalone)."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _find_backend():
    """Locate the KubeAstra `ui/backend` (which holds the span-edit pipeline).

    Search this file's ancestors and their siblings
    for a directory containing `ui/backend/gitops`. Falls back to None → the
    integration test skips (the oracle's own unit tests don't need this)."""
    start = Path(__file__).resolve()
    for anc in start.parents:
        if (anc / "ui" / "backend" / "gitops").is_dir():
            return anc / "ui" / "backend"
        parent = anc.parent
        if parent.exists():
            for sib in parent.iterdir():
                if (sib / "ui" / "backend" / "gitops").is_dir():
                    return sib / "ui" / "backend"
    return None


BACKEND = _find_backend()
for p in (str(HERE), str(BACKEND) if BACKEND else None):
    if p and p not in sys.path:
        sys.path.insert(0, p)

from oracle import Intent, judge  # noqa: E402

try:
    from gitops.locate import find_span            # noqa: E402
    from gitops.edit import apply_span             # noqa: E402
    _PIPELINE = True
except Exception:
    _PIPELINE = False

pytestmark = pytest.mark.skipif(not _PIPELINE, reason="span-edit pipeline not importable")

MANIFEST = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 2            # bumped during the Nov incident
  template:
    spec:
      containers:
        - name: api
          image: ghcr.io/acme/api:v1.0.0
          resources:
            limits:
              memory: 128Mi
"""


def _sut_edit(text, field_path, new_value):
    """Exactly the system-under-test path: locate the scalar span, replace it."""
    span = find_span(text, 0, field_path)
    assert span is not None
    return apply_span(text, span, new_value)


def test_oracle_blesses_real_pipeline_replicas():
    out = _sut_edit(MANIFEST, ("spec", "replicas"), 5)
    v = judge(MANIFEST, out, Intent("Deployment", "api-gateway",
                                    ("spec", "replicas"), 5))
    assert v.correct and v.comments_preserved and v.format_preserved, v.reason
    assert v.diff_added == 1 and v.diff_removed == 1
    assert v.collateral_changes == [] and v.fabricated_paths == []


def test_oracle_blesses_real_pipeline_nested_memory_limit():
    fp = ("spec", "template", "spec", "containers", "api",
          "resources", "limits", "memory")
    out = _sut_edit(MANIFEST, fp, "512Mi")
    v = judge(MANIFEST, out, Intent("Deployment", "api-gateway", fp, "512Mi"))
    assert v.correct and v.format_preserved, v.reason
    assert v.diff_added == 1 and v.diff_removed == 1
