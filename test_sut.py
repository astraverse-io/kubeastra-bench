"""SUT adapter tests, incl. the end-to-end pipeline builder → SUT → oracle over
the whole seed corpus. Asserts C1's guarantees hold UNIVERSALLY (no collateral,
no fabrication, minimal diff, correctness) — the property the paper claims."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from oracle import Intent, judge          # noqa: E402
from corpus_builder import build          # noqa: E402
from systems import sut                    # noqa: E402

pytestmark = pytest.mark.skipif(not sut.AVAILABLE, reason="span-edit pipeline unavailable")

CORPUS = HERE / "corpus"


def _intent_of(task: dict) -> Intent:
    it = task["intent"]
    return Intent(kind=it["kind"], name=it["name"],
                  field_path=tuple(it["field_path"]), new_value=it["new_value"],
                  namespace=it.get("namespace"))


def _load_tasks(tmp_path):
    out = tmp_path / "tasks.jsonl"
    build(CORPUS, out)
    return [json.loads(l) for l in out.read_text().splitlines()]


def test_sut_edits_replicas_and_oracle_blesses(tmp_path):
    tasks = _load_tasks(tmp_path)
    t = next(x for x in tasks if x["intent"]["field_type"] == "replicas")
    original = (CORPUS / t["manifest_path"]).read_text()
    out = sut.run(original, t["intent"])
    assert not out.refused, out.reason
    v = judge(original, out.output_text, _intent_of(t))
    assert v.correct and v.format_preserved, v.reason
    assert v.diff_added == 1 and v.diff_removed == 1


def test_sut_refuses_when_resource_absent():
    original = (CORPUS / "deployment-api.yaml").read_text()
    intent = {"kind": "Deployment", "name": "does-not-exist",
              "field_path": ["spec", "replicas"], "new_value": 5, "namespace": None}
    out = sut.run(original, intent)
    assert out.refused and "no Deployment/does-not-exist" in out.reason


def test_whole_corpus_span_edit_satisfies_guarantees(tmp_path):
    """Every generated task, run through the REAL span-edit pipeline, must:
    apply, be semantically correct, add a minimal (<=1 line) diff, and introduce
    zero collateral changes and zero fabricated content. This is the paper's
    core claim, checked over the whole corpus."""
    tasks = _load_tasks(tmp_path)
    assert len(tasks) >= 15, "seed corpus should generate a meaningful task set"

    failures = []
    for t in tasks:
        original = (CORPUS / t["manifest_path"]).read_text()
        out = sut.run(original, t["intent"])
        if out.refused:
            failures.append((t["task_id"], t["intent"]["field_type"], "REFUSED: " + out.reason))
            continue
        v = judge(original, out.output_text, _intent_of(t))
        if not (v.correct and not v.collateral_changes and not v.fabricated_paths
                and v.diff_added <= 1 and v.diff_removed <= 1):
            failures.append((t["task_id"], t["intent"]["field_type"], v.reason))

    assert not failures, "span-edit violated a guarantee on:\n" + "\n".join(map(str, failures))


def test_span_edit_is_deterministic(tmp_path):
    tasks = _load_tasks(tmp_path)
    t = tasks[0]
    original = (CORPUS / t["manifest_path"]).read_text()
    outs = {sut.run(original, t["intent"]).output_text for _ in range(5)}
    assert len(outs) == 1                      # identical every run
