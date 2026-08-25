"""System under test: KubeAstra's deterministic span-edit pipeline (C1).

Runs the REAL pipeline per task — index the manifest by (kind, name), resolve
the target scalar's character span, replace it — so the benchmark measures the
shipped system, not a reimplementation. Refuses (rather than guesses) on zero or
ambiguous matches or an absent field, exactly as the product does; a refusal is
recorded as data (coverage / refusal-correctness), not a crash.
"""
from __future__ import annotations

import sys
from typing import Any

from .base import SystemOutput, find_backend

_BACKEND = find_backend()
if _BACKEND and str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

try:
    from gitops.index import RepoFile, build_index      # noqa: E402
    from gitops.locate import FieldChange, find_span    # noqa: E402
    from gitops.edit import apply_span                  # noqa: E402
    AVAILABLE = True
except Exception:                                       # pragma: no cover
    AVAILABLE = False


NAME = "SUT-span-edit"


def run(original_text: str, intent: dict) -> SystemOutput:
    """Apply the pipeline to a single-manifest task.

    `intent` is the task's intent dict (kind, name, field_path, new_value, ...).
    Uses the real index+locate+edit, so it exercises match resolution and
    refusal, not just the span replacement.
    """
    if not AVAILABLE:
        return SystemOutput.refuse("span-edit pipeline unavailable")

    kind, name = intent["kind"], intent["name"]
    field_path = tuple(intent["field_path"])
    new_value: Any = intent["new_value"]

    index = build_index([RepoFile("manifest.yaml", original_text)])
    matches = index.get((kind, name), [])
    if not matches:
        return SystemOutput.refuse(f"no {kind}/{name} in manifest")
    if len(matches) > 1:
        return SystemOutput.refuse(f"ambiguous: {len(matches)} matches for {kind}/{name}")

    span = find_span(original_text, matches[0].doc_index, field_path)
    if span is None:
        return SystemOutput.refuse(
            f"field {'.'.join(map(str, field_path))} absent on {kind}/{name}")

    # FieldChange is constructed for parity with the product call path even
    # though the span already fully determines the edit.
    _ = FieldChange(kind=kind, name=name, namespace=intent.get("namespace"),
                    field_path=field_path, new_value=new_value,
                    reason=intent.get("field_type", ""))
    return SystemOutput.edit(apply_span(original_text, span, new_value))
