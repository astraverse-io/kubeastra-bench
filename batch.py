"""Batched execution over the Anthropic Message Batches API (50% cheaper).

The synchronous runner (`run.run_matrix`) makes one live call per cell. This
module instead collects requests and submits them as batches:

  * B1 and B2 are single-shot — one request per (task, seed), all in one batch.
  * B3's retry loop is batched **round by round**: round r submits only the
    cells that still failed after round r-1, with the failure text appended.
  * SUT needs no model and is computed inline.

Prompt caching still applies — every request carries the manifest in a
`cache_control`'d system block, so the batch stacks the caching discount on top
of the 50% batch discount.

The batches API is reached only through `Executor.submit(requests) -> {cid:
Result}`, so the orchestration is unit-tested with a fake executor and no key.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from oracle import Intent  # noqa: F401  (re-exported intent type lives in run)
from systems import prompts, sut
from systems.base import SystemOutput
from systems.baselines import strip_fences, apply_unified_diff, _parses
from run import build_row, _intent_obj

# custom_id must match ^[a-zA-Z0-9_-]{1,64}$ — task_id is 12 hex chars, so
# "<code>_<task_id>_<seed>" stays well under 64 and uses only legal characters.
_CODE = {"B1": "b1", "B2": "b2"}
_NAME = {"b1": "B1-full-file", "b2": "B2-unified-diff", "b3": "B3-diff-retry"}


@dataclass
class Result:
    ok: bool
    text: str = ""
    error: str = ""


def _params(system: str, user: str, model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
    }


class AnthropicBatchExecutor:
    """Submit a batch, wait for it to end, return {custom_id: Result}."""

    def __init__(self, client, poll_seconds: float = 15.0):
        self._client = client
        self._poll = poll_seconds
        # cumulative token usage across every batch this executor submits, so the
        # run can report exact cost instead of an estimate.
        self.usage = {"requests": 0, "input": 0, "output": 0,
                      "cache_read": 0, "cache_write": 0}

    def _tally(self, u) -> None:
        self.usage["requests"] += 1
        self.usage["input"] += getattr(u, "input_tokens", 0) or 0
        self.usage["output"] += getattr(u, "output_tokens", 0) or 0
        self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0

    def submit(self, requests: list[dict]) -> dict[str, Result]:
        if not requests:
            return {}
        batch = self._client.messages.batches.create(requests=requests)
        while True:
            b = self._client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            time.sleep(self._poll)
        out: dict[str, Result] = {}
        for r in self._client.messages.batches.results(batch.id):
            if r.result.type == "succeeded":
                self._tally(r.result.message.usage)
                text = "".join(getattr(x, "text", "") for x in r.result.message.content
                               if getattr(x, "type", None) == "text")
                out[r.custom_id] = Result(ok=bool(text.strip()), text=text,
                                          error="" if text.strip() else "empty response")
            else:
                out[r.custom_id] = Result(ok=False, error=f"batch {r.result.type}")
        return out


def anthropic_executor(model: str | None, poll_seconds: float = 15.0):
    """Build an AnthropicBatchExecutor + resolved model id from KubeAstra settings."""
    from systems.base import find_mcp
    mcp = find_mcp()
    if mcp and str(mcp) not in sys.path:
        sys.path.insert(0, str(mcp))
    from config.settings import get_settings
    settings = get_settings()
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return AnthropicBatchExecutor(client, poll_seconds), (model or settings.anthropic_model)


# ── output post-processing (mirrors systems/baselines.py) ─────────────────────

def _b1_output(original: str, res: Result) -> SystemOutput:
    if res is None or not res.ok:
        return SystemOutput.refuse(f"error: {res.error if res else 'no result'}")
    out = strip_fences(res.text)
    if original.endswith("\n") and not out.endswith("\n"):
        out += "\n"
    return SystemOutput.edit(out)


def _b2_output(original: str, res: Result) -> SystemOutput:
    if res is None or not res.ok:
        return SystemOutput.refuse(f"error: {res.error if res else 'no result'}")
    patched = apply_unified_diff(original, strip_fences(res.text))
    if patched is None:
        return SystemOutput.refuse("diff did not apply")
    return SystemOutput.edit(patched)


def _b3_step(original: str, res: Result) -> tuple[SystemOutput | None, str | None]:
    """One B3 attempt → (final output, None) on success, or (None, error) to
    retry. Mirrors the validate-and-retry logic of systems.baselines.B3."""
    if res is None or not res.ok:
        return None, f"the previous attempt failed: {res.error if res else 'no result'}"
    patched = apply_unified_diff(original, strip_fences(res.text))
    if patched is None:
        return None, "the diff did not apply cleanly"
    if not _parses(patched):
        return None, "the patched file did not parse as YAML"
    return SystemOutput.edit(patched), None


# ── orchestration ─────────────────────────────────────────────────────────────

def run_batched(tasks: list[dict], names: list[str], seeds: int, corpus_dir: Path,
                executor, model: str, max_tokens: int, retries: int = 2) -> list[dict]:
    corpus_dir = Path(corpus_dir)
    cache: dict[str, str] = {}

    def original(mp: str) -> str:
        if mp not in cache:
            cache[mp] = (corpus_dir / mp).read_text()
        return cache[mp]

    rows: list[dict] = []

    # SUT — deterministic, no API.
    if "SUT" in names:
        for task in tasks:
            o, intent = original(task["manifest_path"]), task["intent"]
            io = _intent_obj(intent)
            for seed in range(seeds):
                rows.append(build_row(task, "SUT-span-edit", seed,
                                      sut.run(o, intent), o, io))

    # B1 / B2 — single-shot, one batch.
    reqs: list[dict] = []
    meta: dict[str, tuple[dict, str, int]] = {}
    for task in tasks:
        o, intent = original(task["manifest_path"]), task["intent"]
        for seed in range(seeds):
            for code in ("B1", "B2"):
                if code not in names:
                    continue
                builder = prompts.build_full_file if code == "B1" else prompts.build_unified_diff
                s, u = builder(o, intent)
                cid = f"{_CODE[code]}_{task['task_id']}_{seed}"
                reqs.append({"custom_id": cid, "params": _params(s, u, model, max_tokens)})
                meta[cid] = (task, _NAME[_CODE[code]], seed)
    if reqs:
        results = executor.submit(reqs)
        for cid, (task, sys_name, seed) in meta.items():
            o, io = original(task["manifest_path"]), _intent_obj(task["intent"])
            res = results.get(cid)
            out = (_b1_output(o, res) if sys_name == "B1-full-file"
                   else _b2_output(o, res))
            rows.append(build_row(task, sys_name, seed, out, o, io))

    # B3 — batched round by round.
    if "B3" in names:
        rows.extend(_run_b3(tasks, seeds, original, executor, model, max_tokens, retries))
    return rows


def _run_b3(tasks, seeds, original, executor, model, max_tokens, retries) -> list[dict]:
    # pending cells carry the error text to feed the next attempt (None first).
    pending = [(t, s, None) for t in tasks for s in range(seeds)]
    final: dict[tuple[str, int], SystemOutput] = {}

    for round_i in range(retries + 1):
        if not pending:
            break
        reqs, meta = [], {}
        for task, seed, err in pending:
            o = original(task["manifest_path"])
            s, u = prompts.build_unified_diff(o, task["intent"], err)
            cid = f"b3r{round_i}_{task['task_id']}_{seed}"
            reqs.append({"custom_id": cid, "params": _params(s, u, model, max_tokens)})
            meta[cid] = (task, seed)
        results = executor.submit(reqs)
        nxt = []
        for cid, (task, seed) in meta.items():
            o = original(task["manifest_path"])
            out, err = _b3_step(o, results.get(cid))
            if out is not None:
                final[(task["task_id"], seed)] = out
            else:
                nxt.append((task, seed, err))
        pending = nxt

    # anything still failing after the last round is a refusal
    for task, seed, err in pending:
        final[(task["task_id"], seed)] = SystemOutput.refuse(
            f"gave up after {retries + 1} attempts: {err}")

    rows = []
    for task in tasks:
        io = _intent_obj(task["intent"])
        o = original(task["manifest_path"])
        for seed in range(seeds):
            rows.append(build_row(task, "B3-diff-retry", seed,
                                  final[(task["task_id"], seed)], o, io))
    return rows
