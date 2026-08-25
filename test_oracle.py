"""Tests for the benchmark oracle. A buggy judge invalidates the paper, so the
judge is tested like production code: known-good and known-bad edits, and the
load-bearing property that a *semantically correct but reflowed* edit passes
correctness while failing the fidelity metrics."""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from oracle import Intent, judge  # noqa: E402

BASE = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 3            # keep in sync with the HPA min
  template:
    spec:
      containers:
        - name: sidecar
          image: envoy:1.29
        - name: api
          image: ghcr.io/acme/api:v1.4.2
          resources:
            limits:
              memory: 128Mi
"""


def _intent(**kw):
    base = dict(kind="Deployment", name="api-gateway",
                field_path=("spec", "replicas"), new_value=5)
    base.update(kw)
    return Intent(**base)


# ── the ideal span-edit: minimal, faithful ────────────────────────────────────

def test_minimal_faithful_edit_is_correct():
    out = BASE.replace("replicas: 3", "replicas: 5")
    v = judge(BASE, out, _intent())
    assert v.correct and v.parses and v.target_found
    assert v.collateral_changes == [] and v.fabricated_paths == []
    assert v.comments_preserved and v.format_preserved
    assert v.diff_added == 1 and v.diff_removed == 1


def test_named_container_resolves_and_sidecar_untouched():
    out = BASE.replace("ghcr.io/acme/api:v1.4.2", "ghcr.io/acme/api:v1.5.0")
    v = judge(BASE, out, _intent(
        field_path=("spec", "template", "spec", "containers", "api", "image"),
        new_value="ghcr.io/acme/api:v1.5.0"))
    assert v.correct, v.reason
    # and the sidecar image change must be flagged as collateral, proving
    # name-addressing distinguishes the two containers
    bad = BASE.replace("envoy:1.29", "envoy:1.30")
    vb = judge(BASE, bad, _intent(
        field_path=("spec", "template", "spec", "containers", "api", "image"),
        new_value="ghcr.io/acme/api:v1.5.0"))
    assert not vb.correct


# ── the load-bearing test: semantic correctness vs. fidelity are separate ──────

def test_reflowed_but_semantically_correct_edit_passes_correctness_fails_fidelity():
    # Same semantic change (replicas 3->5) but the model dropped the comment and
    # re-quoted an unrelated scalar. A baseline that does this is still CORRECT
    # (right change, nothing else semantically altered) but NOT faithful.
    out = ("apiVersion: apps/v1\n"
           "kind: Deployment\n"
           "metadata:\n"
           "  name: api-gateway\n"
           "spec:\n"
           "  replicas: 5\n"                       # comment dropped
           "  template:\n"
           "    spec:\n"
           "      containers:\n"
           "        - name: sidecar\n"
           '          image: "envoy:1.29"\n'       # requoted, same value
           "        - name: api\n"
           "          image: ghcr.io/acme/api:v1.4.2\n"
           "          resources:\n"
           "            limits:\n"
           "              memory: 128Mi\n")
    v = judge(BASE, out, _intent())
    assert v.correct, "semantic change is right → must count as correct"
    assert v.collateral_changes == [], "requoting is not a semantic change"
    assert not v.comments_preserved, "the HPA comment was dropped"
    assert not v.format_preserved, "requoting an unrelated scalar is a reflow"


# ── known-bad edits ───────────────────────────────────────────────────────────

def test_collateral_change_is_caught():
    out = (BASE.replace("replicas: 3", "replicas: 5")
               .replace("memory: 128Mi", "memory: 256Mi"))
    v = judge(BASE, out, _intent())
    assert not v.correct
    assert any("memory" in p for p in v.collateral_changes)


def test_fabricated_field_is_caught():
    out = BASE.replace(
        "spec:\n  replicas: 3            # keep in sync with the HPA min",
        "spec:\n  replicas: 5\n  paused: true")   # invented field
    v = judge(BASE, out, _intent())
    assert not v.correct
    assert any("paused" in p for p in v.fabricated_paths)


def test_wrong_value_is_caught():
    out = BASE.replace("replicas: 3", "replicas: 4")
    v = judge(BASE, out, _intent(new_value=5))
    assert not v.correct and "expected 5" in v.reason


def test_type_mismatch_is_caught():
    # model wrote a quoted string "5"; intent wanted the int 5. Semantically
    # different in YAML (str vs int) — a real defect for a numeric field.
    out = BASE.replace("replicas: 3", 'replicas: "5"')
    v = judge(BASE, out, _intent(new_value=5))
    assert not v.correct


def test_missing_target_is_caught():
    out = BASE.replace(
        "  replicas: 3            # keep in sync with the HPA min\n", "")
    v = judge(BASE, out, _intent())
    assert not v.correct and not v.target_found


def test_nonparseable_output_is_caught():
    out = BASE + "\n:\n  - broken: [unterminated\n"
    v = judge(BASE, out, _intent())
    assert not v.parses and not v.correct


# ── multi-document files ──────────────────────────────────────────────────────

MULTI = """apiVersion: v1
kind: ConfigMap
metadata:
  name: api-config
data:
  LEVEL: info
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 2
"""


def test_multidoc_edit_in_second_doc_is_correct():
    out = MULTI.replace("replicas: 2", "replicas: 4")
    v = judge(MULTI, out, _intent(new_value=4))
    assert v.correct, v.reason


def test_multidoc_collateral_in_sibling_doc_is_caught():
    out = (MULTI.replace("replicas: 2", "replicas: 4")
                .replace("LEVEL: info", "LEVEL: debug"))
    v = judge(MULTI, out, _intent(new_value=4))
    assert not v.correct
    assert any("LEVEL" in p for p in v.collateral_changes)
