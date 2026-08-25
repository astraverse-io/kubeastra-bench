"""Shared types + backend locator for benchmark systems.

A *system* takes the original manifest text and a task intent and returns a
SystemOutput. The span-edit SUT is deterministic; the LLM baselines (B1-B3) are
not. `find_backend()` locates KubeAstra's `ui/backend` so the SUT can import the
real span-edit pipeline, searching this file's ancestors and their siblings, so
a KubeAstra checkout placed beside this repo is found automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SystemOutput:
    """Outcome of running one system on one task.

    Exactly one of `output_text` (an edit to apply/judge) or a refusal is
    meaningful: if `refused`, `output_text` is None and `reason` says why.
    `added_files` is set only for the Kustomize patch-fallback path.
    """
    output_text: Optional[str]
    refused: bool
    reason: str
    added_files: Optional[dict] = None    # path -> content, fallback only

    @classmethod
    def edit(cls, text: str) -> "SystemOutput":
        return cls(output_text=text, refused=False, reason="edited")

    @classmethod
    def refuse(cls, reason: str) -> "SystemOutput":
        return cls(output_text=None, refused=True, reason=reason)


def find_backend() -> Optional[Path]:
    """Return KubeAstra's `ui/backend` dir, or None if not found."""
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


def find_mcp() -> Optional[Path]:
    """Return KubeAstra's `mcp` dir (holds config.settings + services.llm),
    which is a sibling of `ui/`. Needed only by the real LLM adapter."""
    backend = find_backend()
    if backend is None:
        return None
    mcp = backend.parent.parent / "mcp"     # <repo>/ui/backend -> <repo>/mcp
    return mcp if (mcp / "services" / "llm").is_dir() else None
