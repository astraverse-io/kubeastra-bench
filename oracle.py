"""Correctness oracle for the C1 (span-edit) benchmark.

The oracle is the *judge*: given the original manifest, a system's edited output,
and the intended field change, it decides whether the edit is correct and
measures how minimal/faithful it is. Every quantitative result in the paper
rests on this module, so its semantic core is deliberately simple and is
unit-tested against known-good/known-bad pairs (see test_oracle.py).

Two tiers of check, with different reliability:

  * SEMANTIC (rigorous). Parse the original and the output into document trees
    and compare leaf-by-leaf. A correct edit changes EXACTLY the target scalar
    to the intended value and leaves every other value byte-identical. This is
    what `correct`, `collateral_changes`, `fabricated_paths`, and
    `removed_paths` are built on. It is objective: correctness is judged on the
    parsed structure, so a baseline that makes the right change in a different
    style still counts as correct — fairness by construction.

  * FIDELITY (heuristic). `comments_preserved` and `format_preserved` work on
    raw text (PyYAML discards comments, so structure alone can't measure them).
    These are best-effort and marked as such; the unit tests pin their
    behavior. They are secondary metrics — they never affect `correct`.

Lists of mappings that carry a unique `name` child (containers, env, volumes)
are addressed BY NAME, matching the span-edit pipeline's convention, so a
reordered manifest still aligns. Other lists are addressed by index.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import yaml


# ── Inputs ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Intent:
    """The change the task asked for. `field_path` uses named-list segments
    (a container/env addressed by its name), not positional indices. Provide
    `new_value` in its intended parsed type (int for replicas, str for an image
    tag) — a str/int mismatch is a real semantic difference and is reported."""
    kind: str
    name: str
    field_path: tuple
    new_value: Any
    namespace: Optional[str] = None


@dataclass
class Verdict:
    parses: bool
    correct: bool
    target_found: bool
    collateral_changes: list          # semantic paths changed that shouldn't be
    fabricated_paths: list            # leaf paths in output absent from original
    removed_paths: list               # leaf paths in original absent from output
    comments_preserved: bool          # heuristic
    comments_lost: list               # heuristic
    format_preserved: bool            # heuristic
    diff_added: int
    diff_removed: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── YAML tree flattening (named-list aware) ───────────────────────────────────

def _all_named_mappings(seq: list) -> bool:
    return bool(seq) and all(
        isinstance(x, dict) and isinstance(x.get("name"), (str, int)) for x in seq
    )


def _flatten_node(node: Any, prefix: tuple, out: dict) -> None:
    if isinstance(node, dict):
        if not node:
            out[prefix] = {}          # record empty mapping as a leaf
            return
        for k, v in node.items():
            _flatten_node(v, prefix + (k,), out)
    elif isinstance(node, list):
        if not node:
            out[prefix] = []
            return
        if _all_named_mappings(node):
            names = [x["name"] for x in node]
            if len(set(names)) == len(names):     # unique names → address by name
                for x in node:
                    _flatten_node(x, prefix + (x["name"],), out)
                return
        for i, x in enumerate(node):              # fallback → address by index
            _flatten_node(x, prefix + (i,), out)
    else:
        out[prefix] = node


def _flatten_docs(docs: list) -> dict:
    out: dict = {}
    for i, doc in enumerate(docs):
        _flatten_node(doc, (i,), out)
    return out


def _find_doc_index(docs: list, kind: str, name: str) -> Optional[int]:
    for i, d in enumerate(docs):
        if isinstance(d, dict) and d.get("kind") == kind:
            meta = d.get("metadata") or {}
            if isinstance(meta, dict) and meta.get("name") == name:
                return i
    return None


# ── Fidelity (heuristic, text-level) ──────────────────────────────────────────

def _strip_quoted(line: str) -> str:
    line = re.sub(r'"[^"]*"', "", line)
    line = re.sub(r"'[^']*'", "", line)
    return line


def _comments(text: str) -> list:
    """Best-effort comment extraction: the text after a `#` that is not inside a
    quoted scalar. Naive on escapes; adequate for set-containment comparison
    because original and output are processed identically."""
    found = []
    for line in text.splitlines():
        stripped = _strip_quoted(line)
        idx = stripped.find("#")
        if idx != -1:
            comment = stripped[idx:].strip()
            if comment != "#":
                found.append(comment)
    return found


def _only_scalar_changed(original: str, output: str, old: Any, new: Any) -> bool:
    """Heuristic format check: every changed line is the same line with only the
    old scalar's text swapped for the new one — no reflow, requote, or reorder.
    Whole-line insert/delete of unrelated lines fails the check."""
    a = original.splitlines()
    b = output.splitlines()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace":
            return False                          # pure insert/delete of a line
        rem, add = a[i1:i2], b[j1:j2]
        if len(rem) != len(add):
            return False
        for r, s in zip(rem, add):
            if r.replace(str(old), str(new), 1) != s:
                return False
    return True


def _diff_counts(original: str, output: str) -> tuple:
    added = removed = 0
    for line in difflib.unified_diff(original.splitlines(), output.splitlines(),
                                     lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


# ── The judgment ──────────────────────────────────────────────────────────────

def judge(original: str, output: str, intent: Intent) -> Verdict:
    added, removed = _diff_counts(original, output)

    # 1. Output must parse.
    try:
        docs_o = list(yaml.safe_load_all(output))
    except yaml.YAMLError as exc:
        return Verdict(
            parses=False, correct=False, target_found=False,
            collateral_changes=[], fabricated_paths=[], removed_paths=[],
            comments_preserved=False, comments_lost=[], format_preserved=False,
            diff_added=added, diff_removed=removed,
            reason=f"output does not parse: {exc.__class__.__name__}",
        )

    docs_m = list(yaml.safe_load_all(original))   # corpus is pre-validated

    # 2. Locate the target resource. It must sit at the same document index in
    #    both files — a faithful edit does not reorder documents.
    idx = _find_doc_index(docs_m, intent.kind, intent.name)
    if idx is None:
        return Verdict(
            parses=True, correct=False, target_found=False,
            collateral_changes=[], fabricated_paths=[], removed_paths=[],
            comments_preserved=True, comments_lost=[], format_preserved=False,
            diff_added=added, diff_removed=removed,
            reason=f"target {intent.kind}/{intent.name} not found in ORIGINAL "
                   f"(malformed task)",
        )
    target_key = (idx,) + tuple(intent.field_path)

    flat_m = _flatten_docs(docs_m)
    flat_o = _flatten_docs(docs_o)

    keys_m, keys_o = set(flat_m), set(flat_o)
    fabricated = sorted(keys_o - keys_m, key=repr)
    removed_paths = sorted(keys_m - keys_o, key=repr)
    changed = sorted(
        (k for k in keys_m & keys_o if flat_m[k] != flat_o[k]), key=repr
    )
    collateral = [k for k in changed if k != target_key] + fabricated + removed_paths
    collateral = [k for k in collateral if k != target_key]

    target_found = target_key in flat_o
    target_ok = target_found and flat_o.get(target_key) == intent.new_value
    correct = bool(target_ok and not collateral)

    # 3. Fidelity (heuristic; never affects `correct`).
    comments_m, comments_o = _comments(original), _comments(output)
    from collections import Counter
    lost = list((Counter(comments_m) - Counter(comments_o)).elements())
    comments_preserved = not lost
    old_val = flat_m.get(target_key)
    format_preserved = comments_preserved and _only_scalar_changed(
        original, output, old_val, intent.new_value
    )

    if correct:
        reason = "correct: exactly the target scalar changed"
    elif not target_found:
        reason = f"target field {'.'.join(map(str, intent.field_path))} missing in output"
    elif not target_ok:
        reason = (f"target value is {flat_o.get(target_key)!r}, "
                  f"expected {intent.new_value!r}")
    else:
        reason = f"collateral change(s): {[list(k) for k in collateral][:5]}"

    return Verdict(
        parses=True, correct=correct, target_found=target_found,
        collateral_changes=[list(k) for k in collateral],
        fabricated_paths=[list(k) for k in fabricated],
        removed_paths=[list(k) for k in removed_paths],
        comments_preserved=comments_preserved, comments_lost=lost,
        format_preserved=format_preserved,
        diff_added=added, diff_removed=removed, reason=reason,
    )
