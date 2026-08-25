"""LLM-authored-edit baselines B1–B3.

  B1  the model writes the full edited file.
  B2  the model writes a unified diff, applied strictly (any context mismatch =
      apply failure, a real and measured B2 weakness).
  B3  B2 plus validate-and-retry: if the diff won't apply or the result won't
      parse, re-prompt with the error, up to R retries.

Each is a System: `.name` + `.run(original, intent) -> SystemOutput`. They call
an injected LLM, so they run against FakeLLM in tests and any real provider at
run time.
"""
from __future__ import annotations

import re

import yaml

from .base import SystemOutput
from .llm import LLM
from . import prompts


# ── output cleaning + diff application ────────────────────────────────────────

def strip_fences(text: str) -> str:
    """Remove a surrounding ```lang ... ``` block if the model added one."""
    t = text.strip()
    if not t.startswith("```"):
        return text
    lines = t.splitlines()
    lines = lines[1:]                                   # drop opening fence
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]                              # drop closing fence
    return "\n".join(lines)


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_unified_diff(original: str, diff: str) -> str | None:
    """Apply a unified diff strictly. Returns the patched text, or None if any
    hunk's context/removed lines don't match (a legitimate B2 failure mode)."""
    orig = original.splitlines()
    out: list[str] = []
    oi = 0
    body: list[str] = []
    hunks: list[tuple[int, list[str]]] = []
    old_start = None
    for line in diff.splitlines():
        m = _HUNK.match(line)
        if m:
            if old_start is not None:
                hunks.append((old_start, body))
            old_start, body = int(m.group(1)), []
        elif old_start is not None:
            if line[:3] in ("---", "+++"):
                continue
            if line and line[0] in " +-":
                body.append(line)
            elif line == "":
                body.append(" ")                        # blank context line
    if old_start is not None:
        hunks.append((old_start, body))
    if not hunks:
        return None

    for start, lines in hunks:
        target = start - 1
        if target < oi or target > len(orig):
            return None
        out.extend(orig[oi:target])
        oi = target
        for l in lines:
            tag, content = l[0], l[1:]
            if tag == " ":
                if oi >= len(orig) or orig[oi] != content:
                    return None
                out.append(content); oi += 1
            elif tag == "-":
                if oi >= len(orig) or orig[oi] != content:
                    return None
                oi += 1
            elif tag == "+":
                out.append(content)
            else:
                return None
    out.extend(orig[oi:])
    result = "\n".join(out)
    if original.endswith("\n"):
        result += "\n"
    return result


def _parses(text: str) -> bool:
    try:
        list(yaml.safe_load_all(text))
        return True
    except yaml.YAMLError:
        return False


# ── the systems ───────────────────────────────────────────────────────────────

class B1FullFile:
    name = "B1-full-file"

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, original_text: str, intent: dict) -> SystemOutput:
        system, user = prompts.build_full_file(original_text, intent)
        out = strip_fences(self.llm.complete(system, user))
        if original_text.endswith("\n") and not out.endswith("\n"):
            out += "\n"
        return SystemOutput.edit(out)


class B2UnifiedDiff:
    name = "B2-unified-diff"

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, original_text: str, intent: dict) -> SystemOutput:
        system, user = prompts.build_unified_diff(original_text, intent)
        diff = strip_fences(self.llm.complete(system, user))
        patched = apply_unified_diff(original_text, diff)
        if patched is None:
            return SystemOutput.refuse("diff did not apply")
        return SystemOutput.edit(patched)


class B3DiffRetry:
    name = "B3-diff-retry"

    def __init__(self, llm: LLM, retries: int = 2):
        self.llm = llm
        self.retries = retries

    def run(self, original_text: str, intent: dict) -> SystemOutput:
        error = None
        for _ in range(self.retries + 1):
            system, user = prompts.build_unified_diff(original_text, intent, error)
            diff = strip_fences(self.llm.complete(system, user))
            patched = apply_unified_diff(original_text, diff)
            if patched is None:
                error = "the diff did not apply cleanly"
                continue
            if not _parses(patched):
                error = "the patched file did not parse as YAML"
                continue
            return SystemOutput.edit(patched)
        return SystemOutput.refuse(f"gave up after {self.retries + 1} attempts: {error}")
