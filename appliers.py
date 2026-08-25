"""Diff appliers for the strict-vs-fuzzy study (peer-review blocker #1).

The main benchmark applies model-authored diffs STRICTLY: any context mismatch
is a hard failure. A reviewer will (rightly) object that real GitOps/PR
workflows apply patches with tolerance — GNU `patch` has a fuzz factor and can
ignore whitespace — so the strict number is a *lower bound*. This module
re-applies the same diffs under several appliers so the paper can report the
spread. (We also evaluated `git apply`; it is *stricter* than patch on the
headerless model diffs and is excluded — see the note by the registry.)

The load-bearing nuance: **for whitespace-significant YAML, the two tolerances
are not equal.**
  * *Offset* tolerance (the `@@` line number is wrong but the content is right)
    is SAFE — it recovers a diff the model located incorrectly.
  * *Whitespace* tolerance is UNSAFE — indentation is semantic in YAML, so
    ignoring it can apply a mis-indented hunk and silently change structure.
So each applier is scored on BOTH apply-success and *correctness* (via the
oracle, in applier_study.py): a whitespace-fuzzy applier may apply more diffs
while landing more of them wrong — itself evidence for the paper's thesis.

Interface: `apply(original: str, diff: str) -> str | None`  (None = did not apply).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# strict applier is the one the main benchmark already uses
from systems.baselines import apply_unified_diff as apply_strict  # noqa: F401

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


# ── pure-Python appliers (always available, deterministic, reproducible) ───────

def _parse_hunks(diff: str) -> list[tuple[list[str], list[str]]]:
    """(old_block, new_block) per hunk, tags stripped, `@@` line numbers ignored.
    old_block = context+removed lines (what must be present); new_block =
    context+added (what replaces it)."""
    hunks: list[tuple[list[str], list[str]]] = []
    old: list[str] | None = None
    new: list[str] = []
    for line in diff.splitlines():
        if _HUNK.match(line):
            if old is not None:
                hunks.append((old, new))
            old, new = [], []
        elif old is not None:
            if line[:3] in ("---", "+++"):
                continue
            if line == "":
                old.append(""); new.append(""); continue          # blank context
            tag, content = line[0], line[1:]
            if tag == " ":
                old.append(content); new.append(content)
            elif tag == "-":
                old.append(content)
            elif tag == "+":
                new.append(content)
            # "\ No newline at end of file" and stray lines are ignored
    if old is not None:
        hunks.append((old, new))
    return hunks


def _locate(hay: list[str], needle: list[str], start: int, norm) -> int:
    """First index ≥ start where `needle` matches `hay` under `norm`, requiring a
    UNIQUE match (ambiguous ⇒ -1, so we never guess which occurrence to edit)."""
    if not needle:
        return -1
    hits = [i for i in range(start, len(hay) - len(needle) + 1)
            if all(norm(hay[i + j]) == norm(needle[j]) for j in range(len(needle)))]
    return hits[0] if len(hits) == 1 else -1


def _apply_by_content(original: str, diff: str, norm) -> str | None:
    lines = original.splitlines()
    out: list[str] = []
    cur = 0
    for old, new in _parse_hunks(diff):
        if not old:                        # pure insertion, no anchor ⇒ unsafe
            return None
        idx = _locate(lines, old, cur, norm)
        if idx < 0:
            return None
        out.extend(lines[cur:idx])
        out.extend(new)
        cur = idx + len(old)
    out.extend(lines[cur:])
    res = "\n".join(out)
    if original.endswith("\n"):
        res += "\n"
    return res


def apply_offset_tolerant(original: str, diff: str) -> str | None:
    """Locate each hunk by EXACT content anywhere in the file (ignoring the
    `@@` numbers). YAML-safe: tolerates a wrong line number, not a wrong indent."""
    return _apply_by_content(original, diff, norm=lambda s: s)


def apply_ws_insensitive(original: str, diff: str) -> str | None:
    """Offset-tolerant AND whitespace-insensitive (trailing+leading). UNSAFE for
    YAML — included to quantify how often ignoring indentation lets a wrong hunk
    apply. Correctness (not just apply-success) is what exposes the cost."""
    return _apply_by_content(original, diff, norm=lambda s: s.strip())


# ── real-tool appliers (credibility; skipped when the tool is absent) ──────────

def _write(dirpath: Path, name: str, text: str) -> Path:
    p = dirpath / name
    p.write_text(text if text.endswith("\n") else text + "\n")
    return p


def apply_patch_fuzz(original: str, diff: str) -> str | None:
    """GNU `patch --fuzz=3` — offset + context-fuzz, whitespace-SENSITIVE."""
    return _run_patch(original, diff, ["--fuzz=3"])


def apply_patch_fuzz_ws(original: str, diff: str) -> str | None:
    """GNU `patch --fuzz=3 -l` — also ignore whitespace (UNSAFE for YAML)."""
    return _run_patch(original, diff, ["--fuzz=3", "-l"])


def _run_patch(original: str, diff: str, extra: list[str]) -> str | None:
    if not shutil.which("patch"):
        return None
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        f = _write(dp, "f", original)
        _write(dp, "p", diff)
        with (dp / "p").open() as stdin:
            # cwd=d so any .rej/.orig reject files patch writes land in the
            # temp dir (auto-cleaned), not the caller's working directory.
            r = subprocess.run(["patch", *extra, "-o", str(dp / "o"), str(f)],
                               stdin=stdin, capture_output=True, text=True, cwd=d)
        if r.returncode != 0 or not (dp / "o").exists():
            return None
        return (dp / "o").read_text()


# NB: `git apply` was evaluated and deliberately excluded. It is *stricter* than
# GNU patch, not more lenient: it has no fuzz factor, and on the headerless,
# no-context single-line hunks models emit (`@@ -6,1 +6,1 @@` with just one `-`
# and one `+`) it rejects even well-formed diffs (`patch does not apply`). So the
# canonical *lenient* real tool for this study is GNU patch; git apply would only
# make the baselines look worse, and unfairly. We note this in the paper rather
# than report a number from a tool that is the wrong fit for headerless diffs.


# ── registry ──────────────────────────────────────────────────────────────────

ALL_APPLIERS = {
    "strict": apply_strict,                    # the benchmark's default (lower bound)
    "offset_tolerant": apply_offset_tolerant,  # YAML-safe leniency (offset only)
    "ws_insensitive": apply_ws_insensitive,    # UNSAFE leniency (offset + whitespace)
    "patch_fuzz": apply_patch_fuzz,            # GNU patch --fuzz=3 (ws-sensitive)
    "patch_fuzz_ws": apply_patch_fuzz_ws,      # GNU patch --fuzz=3 -l (ws-insensitive)
}

# which of the tool-backed ones are usable in this environment
_TOOLS = {"patch_fuzz": "patch", "patch_fuzz_ws": "patch"}


def available_appliers() -> dict:
    """The appliers usable here (pure-Python always; shell ones iff the tool exists)."""
    return {name: fn for name, fn in ALL_APPLIERS.items()
            if name not in _TOOLS or shutil.which(_TOOLS[name])}
