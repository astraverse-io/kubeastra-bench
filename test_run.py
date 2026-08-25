"""End-to-end orchestrator test with FakeLLM baselines over the seed corpus.

Proves the whole matrix runs and aggregates, and — the point of the paper —
that the deterministic SUT beats LLM-authored baselines on the guarantee
metrics, using a FakeLLM scripted to make the usual baseline mistakes."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from corpus_builder import build                    # noqa: E402
from systems import sut                              # noqa: E402
from systems.llm import FakeLLM                      # noqa: E402
from systems.baselines import B1FullFile            # noqa: E402
import run as runner                                 # noqa: E402

pytestmark = pytest.mark.skipif(not sut.AVAILABLE, reason="span-edit pipeline unavailable")

CORPUS = HERE / "corpus"


def _tasks(tmp_path):
    out = tmp_path / "tasks.jsonl"
    build(CORPUS, out)
    import json
    return [json.loads(l) for l in out.read_text().splitlines()]


def test_matrix_runs_and_sut_is_perfect(tmp_path):
    tasks = _tasks(tmp_path)
    systems = {"SUT-span-edit": sut.run}
    rows = runner.run_matrix(tasks, systems, seeds=1, corpus_dir=CORPUS)
    summary = runner.aggregate(rows)
    s = summary["SUT-span-edit"]
    assert s["correct_rate"] == 1.0
    assert s["collateral_rate"] == 0.0 and s["fabrication_rate"] == 0.0
    assert s["diff_size_mean"] is not None and s["diff_size_mean"] <= 2.0


def test_sut_beats_a_reflowing_b1_baseline(tmp_path):
    """A B1 that always drops comments (a real LLM habit) stays semantically
    correct but loses fidelity — so SUT should dominate on format preservation
    while both are 'correct'. Demonstrates the metric separation end-to-end."""
    tasks = [t for t in _tasks(tmp_path)
             if t["intent"]["field_type"] == "replicas"]     # simple, comment-bearing

    def reflow_responder(system, user):
        # echo the file with the replicas line changed AND comments stripped.
        # The manifest now lives in `system` (for prompt caching); it is the last
        # "FILE:\n" block (the earlier one is the few-shot example).
        import re
        file_text = system.rsplit("FILE:\n", 1)[1]
        lines = []
        for ln in file_text.splitlines():
            ln = re.sub(r"\s+#.*$", "", ln)               # drop comments (reflow)
            m = re.match(r"^(\s*replicas:\s*)\d+", ln)
            if m:
                ln = m.group(1) + "5"
            lines.append(ln)
        return "\n".join(lines) + "\n"

    systems = {"SUT-span-edit": sut.run,
               "B1-full-file": B1FullFile(FakeLLM(reflow_responder)).run}
    rows = runner.run_matrix(tasks, systems, seeds=1, corpus_dir=CORPUS)
    summary = runner.aggregate(rows)

    # both make the right value change...
    assert summary["SUT-span-edit"]["correct_rate"] == 1.0
    # ...but only SUT preserves comments/format
    assert summary["SUT-span-edit"]["format_preserved_rate"] == 1.0
    assert summary["B1-full-file"]["comments_preserved_rate"] < 1.0


def _replicas_reflow(system, user):
    """Deterministic B1 responder: echo the file with replicas->5, comments
    stripped. Pure in (system,user), so output can't depend on call order.
    The manifest is the last "FILE:\n" block in `system` (prompt-caching layout)."""
    import re
    file_text = system.rsplit("FILE:\n", 1)[1]
    lines = []
    for ln in file_text.splitlines():
        ln = re.sub(r"\s+#.*$", "", ln)
        m = re.match(r"^(\s*replicas:\s*)\d+", ln)
        if m:
            ln = m.group(1) + "5"
        lines.append(ln)
    return "\n".join(lines) + "\n"


def test_parallel_matches_sequential(tmp_path):
    """Concurrency is an optimization, not a semantic change: the same matrix
    run sequentially and on a thread pool must yield byte-identical rows in the
    same order. Uses a stateless callable FakeLLM so any order-dependence would
    show up as a diff."""
    tasks = [t for t in _tasks(tmp_path)
             if t["intent"]["field_type"] == "replicas"]

    def systems():
        return {"SUT-span-edit": sut.run,
                "B1-full-file": B1FullFile(FakeLLM(_replicas_reflow)).run}

    seq = runner.run_matrix(tasks, systems(), seeds=3, corpus_dir=CORPUS, concurrency=1)
    par = runner.run_matrix(tasks, systems(), seeds=3, corpus_dir=CORPUS, concurrency=4)
    assert seq == par
    assert len(seq) == len(tasks) * 2 * 3        # 2 systems x 3 seeds
