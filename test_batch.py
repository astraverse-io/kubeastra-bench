"""Batched-runner tests with a fake executor — no Anthropic API, no key.

They verify the orchestration the batch path adds on top of the (already tested)
baseline post-processing: single-shot B1/B2 in one batch, B3 batched round by
round, and the give-up path — plus that identical rows come out.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from systems import sut                               # noqa: E402
from batch import run_batched, Result                 # noqa: E402

pytestmark = pytest.mark.skipif(not sut.AVAILABLE, reason="span-edit pipeline unavailable")

MANIFEST = ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\nspec:\n"
            "  replicas: 2            # runbook\n  minReadySeconds: 10\n")
GOOD_B1 = MANIFEST.replace("replicas: 2", "replicas: 3")
GOOD_DIFF = ("@@ -6,1 +6,1 @@\n"
             "-  replicas: 2            # runbook\n"
             "+  replicas: 3            # runbook\n")
BAD_DIFF = "@@ -6,1 +6,1 @@\n-  nope\n+  x\n"           # won't apply

TASK = {"task_id": "abc123def456", "manifest_path": "m.yaml",
        "intent": {"kind": "Deployment", "name": "x",
                   "field_path": ["spec", "replicas"], "field_type": "replicas",
                   "new_value": 3, "namespace": None, "old_value": 2}}


class FakeExecutor:
    """Scripts a Result per request from a (cid) -> text|Result responder, and
    counts how many batches (rounds) were submitted."""

    def __init__(self, responder):
        self._responder = responder
        self.rounds = 0

    def submit(self, requests):
        self.rounds += 1
        out = {}
        for r in requests:
            v = self._responder(r["custom_id"])
            out[r["custom_id"]] = v if isinstance(v, Result) else Result(ok=True, text=v)
        return out


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "m.yaml").write_text(MANIFEST)
    return tmp_path


def _by_sys(rows):
    return {r["system"]: r for r in rows}


def test_sut_needs_no_executor(corpus):
    ex = FakeExecutor(lambda cid: pytest.fail("executor must not be called for SUT"))
    rows = run_batched([TASK], ["SUT"], seeds=1, corpus_dir=corpus,
                       executor=ex, model="m", max_tokens=16000)
    assert ex.rounds == 0
    assert _by_sys(rows)["SUT-span-edit"]["correct"] is True


def test_b1_b2_single_shot_one_batch(corpus):
    def responder(cid):
        return GOOD_B1 if cid.startswith("b1_") else GOOD_DIFF
    ex = FakeExecutor(responder)
    rows = run_batched([TASK], ["B1", "B2"], seeds=1, corpus_dir=corpus,
                       executor=ex, model="m", max_tokens=16000)
    assert ex.rounds == 1                        # B1 + B2 share a single batch
    bysys = _by_sys(rows)
    assert bysys["B1-full-file"]["correct"] is True
    assert bysys["B2-unified-diff"]["correct"] is True


def test_b3_recovers_on_second_round(corpus):
    # round 0 diff won't apply; round 1 diff is correct
    def responder(cid):
        return BAD_DIFF if cid.startswith("b3r0_") else GOOD_DIFF
    ex = FakeExecutor(responder)
    rows = run_batched([TASK], ["B3"], seeds=1, corpus_dir=corpus,
                       executor=ex, model="m", max_tokens=16000, retries=2)
    assert ex.rounds == 2                         # one retry round only
    assert _by_sys(rows)["B3-diff-retry"]["correct"] is True


def test_b3_gives_up_after_all_rounds(corpus):
    ex = FakeExecutor(lambda cid: BAD_DIFF)       # never applies
    rows = run_batched([TASK], ["B3"], seeds=1, corpus_dir=corpus,
                       executor=ex, model="m", max_tokens=16000, retries=2)
    assert ex.rounds == 3                          # retries + 1 attempts
    row = _by_sys(rows)["B3-diff-retry"]
    assert row["refused"] is True and "gave up after 3 attempts" in row["reason"]


def test_batch_error_result_becomes_refusal(corpus):
    ex = FakeExecutor(lambda cid: Result(ok=False, error="batch errored"))
    rows = run_batched([TASK], ["B1"], seeds=1, corpus_dir=corpus,
                       executor=ex, model="m", max_tokens=16000)
    row = _by_sys(rows)["B1-full-file"]
    assert row["refused"] is True and "batch errored" in row["reason"]


def test_seeds_multiply_rows(corpus):
    ex = FakeExecutor(lambda cid: GOOD_B1)
    rows = run_batched([TASK], ["B1"], seeds=4, corpus_dir=corpus,
                       executor=ex, model="m", max_tokens=16000)
    assert len(rows) == 4                          # one system x one task x 4 seeds
    assert {r["seed"] for r in rows} == {0, 1, 2, 3}
