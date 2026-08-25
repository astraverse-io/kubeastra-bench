"""Prompts for the LLM baselines B1 (full-file) and B2/B3 (unified diff).

Deliberately engineered — minimal-change instruction, comment/format
preservation, and a worked few-shot example — so the baselines are a fair,
strong comparison rather than a strawman. These exact strings go in the paper
appendix. The intent is expressed in natural language (not a raw path), which is
the fair way to ask a model to apply a change.
"""
from __future__ import annotations


def describe_change(intent: dict) -> str:
    """Human-readable statement of the change, from the structured intent."""
    fp = intent["field_path"]
    kind, name, nv, ft = intent["kind"], intent["name"], intent["new_value"], intent["field_type"]
    if ft == "replicas":
        return f"In {kind}/{name}, set spec.replicas to {nv}."
    # container-scoped changes: field_path = (..., 'containers', <cname>, ...)
    cname = fp[fp.index("containers") + 1] if "containers" in fp else "?"
    if ft == "image":
        return f"In {kind}/{name}, set the image of container '{cname}' to {nv}."
    if ft == "env":
        env_name = fp[fp.index("env") + 1]
        return (f"In {kind}/{name}, set the value of env var '{env_name}' in "
                f"container '{cname}' to {nv}.")
    if ft.startswith(("limit.", "request.")):
        section, key = ft.split(".")
        return (f"In {kind}/{name}, set resources.{section}s.{key} of container "
                f"'{cname}' to {nv}.")
    return f"In {kind}/{name}, set {'.'.join(map(str, fp))} to {nv}."


_RULES = (
    "Apply EXACTLY the requested change and nothing else. Change only the single "
    "field named. Preserve every comment, all formatting, quoting style, and key "
    "order. Do not reformat, re-quote, reorder, or touch any other field."
)

_FEWSHOT_FILE = """spec:
  replicas: 2            # set by the on-call runbook
  minReadySeconds: 10"""

_FEWSHOT_FILE_EDITED = """spec:
  replicas: 3            # set by the on-call runbook
  minReadySeconds: 10"""

_FEWSHOT_DIFF = """@@ -1,2 +1,2 @@
-  replicas: 2            # set by the on-call runbook
+  replicas: 3            # set by the on-call runbook"""


# The manifest goes at the END of `system`, not in `user`. Prompt caching is a
# prefix match, so the large, reused content must sit in a stable prefix: the
# same manifest backs ~80 of the 83 tasks and all 5 seeds, so an identical
# `system` lets those calls read the manifest from cache (~90% input savings)
# instead of re-billing ~5.7K tokens every time. The per-task instruction (and
# B3's retry note) stays in the small, volatile `user` message.

def build_full_file(original: str, intent: dict) -> tuple[str, str]:
    system = (
        "You edit Kubernetes YAML manifests. " + _RULES + " Output ONLY the "
        "complete edited file — no explanation, no markdown code fences.\n\n"
        "Example — change spec.replicas to 3:\nFILE:\n" + _FEWSHOT_FILE +
        "\nOUTPUT:\n" + _FEWSHOT_FILE_EDITED +
        "\n\nApply the change described in the next message to this manifest, "
        "and output the complete edited file.\nFILE:\n" + original
    )
    user = describe_change(intent)
    return system, user


def build_unified_diff(original: str, intent: dict, error: str | None = None) -> tuple[str, str]:
    system = (
        "You edit Kubernetes YAML manifests by emitting a unified diff. " + _RULES +
        " Output ONLY a unified diff with @@ hunk headers that applies cleanly to "
        "the file — no explanation, no markdown code fences.\n\n"
        "Example — change spec.replicas to 3:\n" + _FEWSHOT_DIFF +
        "\n\nApply the change described in the next message to this manifest.\n"
        "FILE:\n" + original
    )
    user = describe_change(intent)
    if error:
        user += (f"\n\nYour previous diff was rejected: {error}\n"
                 "Return a corrected unified diff that applies cleanly.")
    return system, user
