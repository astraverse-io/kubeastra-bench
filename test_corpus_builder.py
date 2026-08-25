"""Unit tests for the corpus builder / task generator."""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from corpus_builder import (  # noqa: E402
    bump_replicas, bump_image, bump_quantity, bump_env, enumerate_tasks,
)

DEPLOY = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: sidecar
          image: envoy:1.29
        - name: api
          image: ghcr.io/acme/api:v1.4.2
          env:
            - name: LOG_LEVEL
              value: info
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              memory: 256Mi
"""


# ── mutation helpers ──────────────────────────────────────────────────────────

def test_bump_replicas():
    assert bump_replicas(3) == 5
    assert bump_replicas(True) is None          # bool is not a replica count
    assert bump_replicas("3") is None


def test_bump_image_semver_and_nonsemver_and_notag():
    assert bump_image("ghcr.io/acme/api:v1.4.2") == "ghcr.io/acme/api:v1.4.3"
    assert bump_image("envoy:1.29") == "envoy:1.29-next"      # not 3-part semver
    assert bump_image("nginx") is None                        # no tag
    assert bump_image("myreg:5000/img") is None               # ':' is a port, not a tag


def test_bump_quantity_roundtrip_types():
    assert bump_quantity("128Mi") == "256Mi"     # unit → string
    assert bump_quantity("100m") == "200m"
    assert bump_quantity("1Gi") == "2Gi"
    assert bump_quantity("1") == "2"             # quoted numeric → string (preserve type)
    assert bump_quantity(1) == 2                 # int → int
    assert bump_quantity(0.5) == 1               # whole float → int


def test_bump_env():
    assert bump_env("info") == "info_v2"
    assert bump_env(5) is None


# ── enumeration ───────────────────────────────────────────────────────────────

def test_enumerate_finds_expected_field_types():
    tasks = enumerate_tasks(DEPLOY, "d.yaml", {"stratum": "S3", "difficulty": []})
    by_type = {}
    for t in tasks:
        by_type.setdefault(t["intent"]["field_type"], []).append(t)
    assert "replicas" in by_type and by_type["replicas"][0]["intent"]["new_value"] == 5
    assert len(by_type["image"]) == 2            # both containers
    assert "limit.memory" in by_type and "request.cpu" in by_type
    assert "env" in by_type


def test_container_image_task_uses_named_path():
    tasks = enumerate_tasks(DEPLOY, "d.yaml")
    api_img = next(t for t in tasks if t["intent"]["field_type"] == "image"
                   and t["intent"]["field_path"][-2] == "api")
    assert api_img["intent"]["field_path"] == \
        ["spec", "template", "spec", "containers", "api", "image"]
    assert api_img["intent"]["old_value"] == "ghcr.io/acme/api:v1.4.2"
    assert api_img["intent"]["new_value"] == "ghcr.io/acme/api:v1.4.3"


def test_task_ids_are_deterministic():
    a = enumerate_tasks(DEPLOY, "d.yaml")
    b = enumerate_tasks(DEPLOY, "d.yaml")
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert len({t["task_id"] for t in a}) == len(a)   # unique
