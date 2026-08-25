"""G5 refusal/ambiguity stratum — empirically test the fail-closed contract.

The main corpus never makes location hard, so the fail-closed design goal (G5:
refuse on ambiguous/absent/unresolvable targets rather than guess) was asserted
but unmeasured. This module is a small *labeled* adversarial set: each case has a
manifest, an intent, and the expected outcome (refuse vs. edit). Running the real
SUT over it measures refusal precision/recall and locator coverage — and, just as
importantly, surfaces where fail-closed *leaks*.

Honest finding baked in: the pipeline refuses correctly on six adversarial
categories, but has a real gap on YAML **aliases** — resolving an alias to its
anchor node and editing the wrong location instead of refusing. We label those
cases `should_refuse` and let the study report the leak rather than hide it.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from oracle import Intent, judge                       # noqa: E402
from systems import sut                                # noqa: E402

# ── manifests ─────────────────────────────────────────────────────────────────

DEP = ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: api\nspec:\n"
       "  replicas: 3\n  template:\n    spec:\n      containers:\n"
       "        - name: web\n          image: nginx:1.25\n")
DUP = DEP + "---\n" + DEP                               # two Deployment/api
TMPL = ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: api\nspec:\n"
        "  replicas: {{ .Values.replicas }}\n")         # Go-templated, unparseable
ANCHOR = ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: api\nspec:\n"
          "  replicas: &r 3\n  minReplicas: *r\n")       # alias references the anchor
MAPTGT = ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: api\nspec:\n"
          "  replicas: 3\n  template:\n    spec: {}\n")


def _intent(kind, name, path, val):
    return {"kind": kind, "name": name, "field_path": path, "field_type": "x",
            "new_value": val, "namespace": None}


# ── the labeled stratum: (id, category, manifest, intent, expected) ───────────
# expected ∈ {"refuse", "edit"}
CASES = [
    # controls — must EDIT (guards refusal *precision*: not refusing everything)
    ("ctl-replicas", "control", DEP, _intent("Deployment", "api", ["spec", "replicas"], 5), "edit"),
    ("ctl-image", "control", DEP, _intent("Deployment", "api",
        ["spec", "template", "spec", "containers", "web", "image"], "nginx:1.26"), "edit"),
    ("ctl-crd", "control",
        "apiVersion: argoproj.io/v1alpha1\nkind: Rollout\nmetadata:\n  name: api\nspec:\n  replicas: 3\n",
        _intent("Rollout", "api", ["spec", "replicas"], 5), "edit"),

    # absent resource — (kind,name) not in the manifest
    ("abs-res-name", "absent_resource", DEP, _intent("Deployment", "ghost", ["spec", "replicas"], 5), "refuse"),
    ("abs-res-kind", "absent_resource", DEP, _intent("StatefulSet", "api", ["spec", "replicas"], 5), "refuse"),

    # absent field — resource present, field not
    ("abs-field", "absent_field", DEP, _intent("Deployment", "api", ["spec", "paused"], True), "refuse"),
    ("abs-field-deep", "absent_field", DEP, _intent("Deployment", "api",
        ["spec", "strategy", "rollingUpdate", "maxSurge"], 2), "refuse"),

    # named-list miss — container name not present
    ("cont-miss", "container_miss", DEP, _intent("Deployment", "api",
        ["spec", "template", "spec", "containers", "nope", "image"], "x:1"), "refuse"),

    # ambiguous — duplicate (kind,name)
    ("ambiguous", "ambiguous", DUP, _intent("Deployment", "api", ["spec", "replicas"], 5), "refuse"),

    # templated / unparseable — index skips it ⇒ resource not found
    ("templated", "templated", TMPL, _intent("Deployment", "api", ["spec", "replicas"], 5), "refuse"),

    # non-scalar target — path lands on a map, not a scalar
    ("non-scalar", "non_scalar", MAPTGT, _intent("Deployment", "api", ["spec", "template"], "oops"), "refuse"),

    # KNOWN GAP — alias target: fail-closed *should* fire but currently leaks
    ("alias-target", "alias_target", ANCHOR, _intent("Deployment", "api", ["spec", "minReplicas"], 9), "refuse"),
]


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_case(case) -> dict:
    _id, category, manifest, intent, expected = case
    out = sut.run(manifest, intent)
    refused = out.refused
    correct_edit = False
    parses = True
    if not refused:
        try:
            import yaml
            list(yaml.safe_load_all(out.output_text))
        except Exception:
            parses = False
        if expected == "edit":
            io = Intent(kind=intent["kind"], name=intent["name"],
                        field_path=tuple(intent["field_path"]),
                        new_value=intent["new_value"], namespace=intent.get("namespace"))
            correct_edit = judge(manifest, out.output_text, io).correct
    ok = refused if expected == "refuse" else correct_edit
    return {"id": _id, "category": category, "expected": expected,
            "refused": refused, "correct_edit": correct_edit,
            "parses": parses, "as_expected": ok}


def run_study() -> dict:
    rows = [evaluate_case(c) for c in CASES]
    should_refuse = [r for r in rows if r["expected"] == "refuse"]
    controls = [r for r in rows if r["expected"] == "edit"]
    refused = [r for r in rows if r["refused"]]
    return {
        "rows": rows,
        "refusal_recall": round(sum(r["refused"] for r in should_refuse) / len(should_refuse), 3),
        "refusal_precision": round(
            sum(1 for r in refused if r["expected"] == "refuse") / len(refused), 3) if refused else 1.0,
        "control_coverage": round(sum(r["correct_edit"] for r in controls) / len(controls), 3),
        "leaks": [r["id"] for r in should_refuse if not r["refused"]],
    }


if __name__ == "__main__":
    s = run_study()
    print(f"refusal recall     : {s['refusal_recall']}  (should-refuse cases that refused)")
    print(f"refusal precision  : {s['refusal_precision']}  (refusals that were warranted)")
    print(f"control coverage   : {s['control_coverage']}  (resolvable cases edited correctly)")
    print(f"fail-closed LEAKS  : {s['leaks']}  (should have refused, didn't)")
    print()
    for r in s["rows"]:
        mark = "ok " if r["as_expected"] else "LEAK" if r["expected"] == "refuse" else "MISS"
        act = "refused" if r["refused"] else ("edited-correct" if r["correct_edit"]
                                              else "edited-WRONG" if not r["parses"] else "edited")
        print(f"  [{mark}] {r['category']:16} {r['id']:14} expected={r['expected']:7} -> {act}")
