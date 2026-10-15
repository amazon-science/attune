# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""log_pod.py accepts a log pod that mounts the shared volume read-only and refuses any writable mount of it.
Run: python3 -m pytest harness/tests"""
import json
import subprocess
import sys

import pytest

import log_pod


def pod(*mounts, volumes=None):
    volumes = volumes or [{"name": "shared", "persistentVolumeClaim": {"claimName": "fsx"}}]
    return {"spec": {"volumes": volumes, "containers": [{"name": "main", "volumeMounts": list(mounts)}]}}


RO = {"name": "shared", "mountPath": "/shared", "readOnly": True}
PVC_RO = [{"name": "shared", "persistentVolumeClaim": {"claimName": "fsx", "readOnly": True}}]


@pytest.mark.parametrize("p", [pod(RO), pod(dict(RO, mountPath="/shared/")), pod(dict(RO, readOnly=False),
                                                                                volumes=PVC_RO),
                               pod(dict(RO, mountPath="/"), {"name": "tmp", "mountPath": "/tmp"},
                                   volumes=[{"name": "shared", "hostPath": {"path": "/x"}},
                                            {"name": "tmp", "emptyDir": {}}])])
def test_read_only(p):
    assert log_pod.problems(p, "/shared") == []


@pytest.mark.parametrize("p,why", [
    (pod(dict(RO, readOnly=False)), "writable"), (pod({"name": "shared", "mountPath": "/shared"}), "writable"),
    (pod({"name": "shared", "mountPath": "/data"}), "mounts nothing at /shared"),
    (pod(RO, {"name": "shared", "mountPath": "/shared/experiments"}), "writable"),
    (pod(RO, {"name": "again", "mountPath": "/data"},
         volumes=[{"name": "shared", "persistentVolumeClaim": {"claimName": "fsx"}},
                  {"name": "again", "persistentVolumeClaim": {"claimName": "fsx"}}]), "again"),
    (pod({"name": "shared", "mountPath": "/", "readOnly": False}), "writable")])
def test_writable(p, why):
    found = log_pod.problems(p, "/shared")
    assert found and why in "; ".join(found)


def test_cli_shows_the_mount_to_use():
    p = subprocess.run([sys.executable, log_pod.__file__, "/shared", "the log pod lp"], capture_output=True,
                       text=True, input=json.dumps(pod({"name": "fsx", "mountPath": "/shared"})), check=False)
    assert p.returncode == 1
    assert "the log pod lp must mount" in p.stderr and "readOnly: true" in p.stderr and "name: fsx" in p.stderr
    ok = subprocess.run([sys.executable, log_pod.__file__, "/shared"], capture_output=True, text=True,
                        input=json.dumps(pod(RO)), check=False)
    assert ok.returncode == 0 and ok.stderr == ""
