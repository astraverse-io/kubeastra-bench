"""Corpus builder / task generator for the C1 (span-edit) benchmark.

Walks a directory of Kubernetes manifests and emits one field-change *task* per
eligible scalar (replicas, container image tag, resource requests/limits, env
value). Each task carries the intent (kind, name, field_path, old/new value) and
provenance, written as JSON lines to `tasks.jsonl`.

New values are chosen deterministically (replicas +2, image patch +1, quantity
doubled, env value suffixed) so the corpus is reproducible: same manifests in →
same tasks out. Field paths use named-list segments (a container by its `name`),
matching the span-edit pipeline and the oracle.

Scope (v1): resolvable single-file edits. Refusal/ambiguity cases (S4) and the
Kustomize patch-fallback are authored separately, not generated here.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import yaml

# kind → (pod-template path prefix, whether `spec.replicas` applies)
WORKLOADS: dict[str, dict] = {
    "Deployment":  {"template": ("spec", "template", "spec"), "replicas": True},
    "StatefulSet": {"template": ("spec", "template", "spec"), "replicas": True},
    "ReplicaSet":  {"template": ("spec", "template", "spec"), "replicas": True},
    "DaemonSet":   {"template": ("spec", "template", "spec"), "replicas": False},
    "Job":         {"template": ("spec", "template", "spec"), "replicas": False},
    "CronJob":     {"template": ("spec", "jobTemplate", "spec", "template", "spec"),
                    "replicas": False},
    "Pod":         {"template": ("spec",), "replicas": False},
}

_MAX_ENV_PER_CONTAINER = 2   # cap so env-heavy manifests don't dominate the corpus


# ── deterministic value mutations ─────────────────────────────────────────────

def bump_replicas(v: Any) -> Optional[int]:
    return v + 2 if isinstance(v, int) and not isinstance(v, bool) else None


def bump_image(v: Any) -> Optional[str]:
    """Bump the tag of `name:tag`. Tag is the part after the last ':' that
    follows the last '/', so registry ports (`reg:5000/img`) aren't mistaken
    for tags. Semver patch +1; otherwise append `-next`."""
    if not isinstance(v, str):
        return None
    slash = v.rfind("/")
    colon = v.find(":", slash + 1)
    if colon == -1:
        return None                       # no explicit tag
    name, tag = v[:colon], v[colon + 1:]
    m = re.match(r"^(v?)(\d+)\.(\d+)\.(\d+)(.*)$", tag)
    new_tag = f"{m.group(1)}{m.group(2)}.{m.group(3)}.{int(m.group(4)) + 1}{m.group(5)}" \
        if m else f"{tag}-next"
    return f"{name}:{new_tag}"


def _double_numeric(num: str) -> Any:
    """Double a bare numeric string, returning a value whose str() round-trips
    (int for whole numbers, float otherwise) — so it matches what the span-edit
    writes and re-parses to."""
    if "." in num:
        d = float(num) * 2
        return int(d) if d.is_integer() else d
    return int(num) * 2


def bump_quantity(v: Any) -> Optional[Any]:
    """Double a Kubernetes quantity. Return type is chosen to round-trip through
    the pipeline's `str(new_value)` write:

      * unit-bearing (128Mi, 100m, 1Gi) → a STRING ("256Mi") — re-parses to str.
      * bare numeric (int/float/"1"/"0.5") → a NUMBER (int/float) — a numeric
        replacement is written unquoted and re-parses to that number.

    A returning-a-string "2" here would never match the int the file parses to,
    so this typing is load-bearing for the corpus's correctness.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v * 2
    if isinstance(v, float):
        d = v * 2
        return int(d) if d.is_integer() else d
    if not isinstance(v, str):
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)([a-zA-Z]*)$", v.strip())
    if not m:
        return None
    num, unit = m.group(1), m.group(2)
    if unit:
        if "." in num:
            d = float(num) * 2
            new_num = str(int(d)) if d.is_integer() else str(d)
        else:
            new_num = str(int(num) * 2)
        return f"{new_num}{unit}"
    # A *string* input means the original scalar was quoted (safe_load returns
    # str only for quoted numerics). The pipeline now preserves quote style, so
    # keep the type a string too — `cpu: "1"` → `cpu: "2"` round-trips.
    return str(_double_numeric(num))


def bump_env(v: Any) -> Optional[str]:
    return v + "_v2" if isinstance(v, str) else None


# ── navigation ────────────────────────────────────────────────────────────────

def _dig(node: Any, path: tuple) -> Any:
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _task_id(rel_path: str, doc_index: int, field_path: tuple) -> str:
    h = hashlib.sha1(f"{rel_path}|{doc_index}|{'/'.join(map(str, field_path))}".encode())
    return h.hexdigest()[:12]


def _make_task(*, kind, name, namespace, field_path, old, new, field_type,
               rel_path, doc_index, prov) -> dict:
    return {
        "task_id": _task_id(rel_path, doc_index, field_path),
        "stratum": prov.get("stratum", "unknown"),
        "difficulty": prov.get("difficulty", []),
        "source": {"file": rel_path, "provenance": prov.get("source", "unknown"),
                   "license": prov.get("license", "unknown")},
        "manifest_path": rel_path,
        "intent": {
            "kind": kind, "name": name, "namespace": namespace,
            "field_path": list(field_path), "field_type": field_type,
            "old_value": old, "new_value": new,
        },
    }


# ── enumeration ───────────────────────────────────────────────────────────────

def enumerate_tasks(text: str, rel_path: str, prov: Optional[dict] = None) -> list[dict]:
    prov = prov or {}
    tasks: list[dict] = []
    try:
        docs = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        return tasks                      # templated / non-YAML: skip

    for doc_index, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        kind = doc.get("kind")
        meta = doc.get("metadata") or {}
        name = meta.get("name") if isinstance(meta, dict) else None
        if kind not in WORKLOADS or not name:
            continue
        wl = WORKLOADS[kind]
        namespace = meta.get("namespace")

        def add(field_path, old, new, ftype):
            if new is not None and new != old:
                tasks.append(_make_task(
                    kind=kind, name=name, namespace=namespace,
                    field_path=field_path, old=old, new=new, field_type=ftype,
                    rel_path=rel_path, doc_index=doc_index, prov=prov))

        if wl["replicas"]:
            r = _dig(doc, ("spec", "replicas"))
            add(("spec", "replicas"), r, bump_replicas(r), "replicas")

        containers = _dig(doc, wl["template"] + ("containers",))
        if not isinstance(containers, list):
            continue
        for c in containers:
            if not isinstance(c, dict):
                continue
            cname = c.get("name")
            if not cname:
                continue
            base = wl["template"] + ("containers", cname)

            add(base + ("image",), c.get("image"), bump_image(c.get("image")), "image")

            res = c.get("resources") or {}
            for section in ("limits", "requests"):
                sect = res.get(section) or {}
                for rk in ("memory", "cpu"):
                    val = sect.get(rk)
                    add(base + ("resources", section, rk), val,
                        bump_quantity(val), f"{section[:-1]}.{rk}")

            env_added = 0
            for e in (c.get("env") or []):
                if env_added >= _MAX_ENV_PER_CONTAINER:
                    break
                if not isinstance(e, dict):
                    continue
                en, ev = e.get("name"), e.get("value")
                if en is None or not isinstance(ev, str):
                    continue
                add(base + ("env", en, "value"), ev, bump_env(ev), "env")
                env_added += 1
    return tasks


def build(corpus_dir: str | Path, out_path: str | Path) -> int:
    corpus_dir = Path(corpus_dir)
    prov_all = {}
    prov_file = corpus_dir / "provenance.json"
    if prov_file.exists():
        prov_all = json.loads(prov_file.read_text())

    all_tasks: list[dict] = []
    for mf in sorted(corpus_dir.rglob("*.y*ml")):
        rel = str(mf.relative_to(corpus_dir))
        all_tasks.extend(enumerate_tasks(mf.read_text(), rel, prov_all.get(rel, {})))

    out_path = Path(out_path)
    with out_path.open("w") as f:
        for t in all_tasks:
            f.write(json.dumps(t) + "\n")
    return len(all_tasks)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    n = build(here / "corpus", here / "tasks.jsonl")
    print(f"wrote {n} tasks to {here / 'tasks.jsonl'}")
