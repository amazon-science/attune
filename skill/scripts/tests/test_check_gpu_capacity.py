# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""check_gpu_capacity.py against a fake kubectl on PATH (no cluster).

data/capacity_*.json are recorded from a live cluster read by a namespace-scoped identity (names replaced): 4 GPU nodes,
2 of them running a pod of the namespace, and the namespace's LocalQueues. The ClusterQueue, which that identity could
not read, is written here in the Kueue v1beta1 shape.
Run: python3 -m pytest skill/scripts/tests"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "check_gpu_capacity.py"
DATA = HERE / "data"
FAKE = '''#!/usr/bin/env python3
import json, os, sys
call = " ".join(sys.argv[1:])
state = json.load(open(os.environ["FAKE_STATE"]))
for needle, err in state.get("fail", {}).items():
    if needle in call:
        sys.exit(print(err, file=sys.stderr) or 1)
for needle, out in state["answers"].items():
    if needle in call:
        sys.exit(print(json.dumps(out)) or 0)
sys.exit(print("fake kubectl: unexpected call " + call, file=sys.stderr) or 3)
'''
FORBIDDEN = ('Error from server (Forbidden): %s is forbidden: User "arn:aws:sts::1:assumed-role/Reader/x" cannot list '
             'resource "%s" at the cluster scope')
CQ = {"metadata": {"name": "team-main"},
      "spec": {"resourceGroups": [{"coveredResources": ["cpu", "memory", "nvidia.com/gpu"], "flavors": [
          {"name": "gpu-p5en", "resources": [{"name": "nvidia.com/gpu", "nominalQuota": "96"}]},
          {"name": "gpu-p5e", "resources": [{"name": "nvidia.com/gpu", "nominalQuota": "64"}]}]}]},
      "status": {"pendingWorkloads": 0, "admittedWorkloads": 4, "flavorsReservation": [
          {"name": "gpu-p5en", "resources": [{"name": "nvidia.com/gpu", "total": "80"}]},
          {"name": "gpu-p5e", "resources": [{"name": "nvidia.com/gpu", "total": "64"}]}]}}


def load(name):
    return json.loads((DATA / name).read_text())


def answers(cq=CQ):
    pods = load("capacity_pods_namespace.json")
    return {"get nodes": load("capacity_nodes.json"), "get pods -A": pods, "-n team get pods": pods,
            "-n team get localqueues": load("capacity_localqueues.json"), "get clusterqueue team-main": cq,
            "get clusterqueue team-urgent": dict(CQ, metadata={"name": "team-urgent"})}


def run(tmp_path, fail=None, args=(), ans=None):
    (tmp_path / "kubectl").write_text(FAKE)
    (tmp_path / "kubectl").chmod(0o755)
    (tmp_path / "state.json").write_text(json.dumps({"answers": ans or answers(), "fail": fail or {}}))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AUTOTUNE_", "ATTUNE_", "KUBECTL_"))}
    env.update(PATH="%s:%s" % (tmp_path, os.environ["PATH"]), FAKE_STATE=str(tmp_path / "state.json"),
               AUTOTUNE_KUBECTL_CONTEXT="ctx", KUBECTL_NS="team", AUTOTUNE_INSTANCE_TYPES="p5en.48xlarge")
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=60)


def baseline(tmp_path, queue="team-main", workers=0):
    import yaml
    gpu = {"resources": {"limits": {"nvidia.com/gpu": 8}}}
    rj = {"kind": "RayJob", "metadata": {"name": "b", "labels": {"kueue.x-k8s.io/queue-name": queue}},
          "spec": {"rayClusterSpec": {"headGroupSpec": {"template": {"spec": {"containers": [dict(gpu, name="h")]}}},
                                      "workerGroupSpecs": [{"replicas": workers, "template": {"spec": {
                                          "containers": [dict(gpu, name="w")]}}}] if workers else []}}}
    p = tmp_path / "base.yaml"
    p.write_text(yaml.safe_dump(rj))
    return str(p)


def lines(p):
    assert p.returncode == 0, p.stderr
    return p.stdout.splitlines()


def test_quota_and_cluster_wide_pods(tmp_path):
    out = lines(run(tmp_path, args=["--baseline", baseline(tmp_path)]))
    assert "[p5en.48xlarge]  Kueue team-main/gpu-p5en: quota 96 GPUs, reserved 80, free 16 (2 nodes)" in out
    assert any(ln.startswith("[p5en.48xlarge]  free=2  occupied=2") for ln in out)
    assert out[-2] == "source: Kueue ClusterQueue quota and nodes (pods in all namespaces)"
    assert out[-1] == "fits: yes (need 1 node, 2 free)"


def test_namespace_scoped_identity(tmp_path):
    """The recorded case: no cluster-wide pods, no ClusterQueue; nodes, namespace pods and LocalQueues readable."""
    fail = {"get pods -A": FORBIDDEN % ("pods", "pods"), "get clusterqueue": FORBIDDEN % ("clusterqueues", "clusterqueues")}
    out = lines(run(tmp_path, fail=fail, args=["--baseline", baseline(tmp_path)]))
    assert sum(ln.startswith("FREE:") for ln in out) == 2
    assert any("Kueue LocalQueue team-main -> ClusterQueue team-main (quota not readable): this queue reserves 280 GPUs "
               "on gpu-p5en (p5en.48xlarge); 0 workloads pending, 91 admitted" in ln for ln in out)
    assert out[-2] == ("source: nodes and pods in namespace team (pods of other namespaces are not visible: free is an "
                       "upper bound) + LocalQueue team-main")
    assert out[-1] == "fits: yes (need 1 node, 2 free)"
    out = lines(run(tmp_path, fail=fail, args=["--baseline", baseline(tmp_path, workers=3)]))
    assert out[-1] == "fits: no (need 4 nodes, 2 free): queue it"


def test_quota_spent(tmp_path):
    cq = json.loads(json.dumps(CQ))
    cq["status"]["flavorsReservation"][0]["resources"][0]["total"] = "96"
    cq["status"]["pendingWorkloads"] = 3
    ans = answers(cq)
    out = lines(run(tmp_path, args=["--nodes", "1", "--baseline", baseline(tmp_path)], ans=ans))
    assert out[-1] == "fits: no (need 1 node, 0 free, 3 workloads pending): queue it"


def test_nothing_readable_is_one_line(tmp_path):
    fail = {"get nodes": FORBIDDEN % ("nodes", "nodes"), "get pods": FORBIDDEN % ("pods", "pods"),
            "get localqueues": FORBIDDEN % ("localqueues", "localqueues"),
            "get clusterqueue": FORBIDDEN % ("clusterqueues", "clusterqueues")}
    p = run(tmp_path, fail=fail)
    assert p.returncode == 0 and p.stderr == ""
    assert p.stdout.splitlines() == ["capacity unknown: Kueue admits runs (this identity may read neither the nodes nor "
                                     "the Kueue queues)"]


def test_other_kubectl_failures_still_fail(tmp_path):
    p = run(tmp_path, fail={"get nodes": "Unable to connect to the server: timeout"})
    assert p.returncode != 0 and "Unable to connect" in p.stderr


def test_cluster_without_kueue(tmp_path):
    no_kueue = 'error: the server doesn\'t have a resource type "localqueues"'
    out = lines(run(tmp_path, fail={"get localqueues": no_kueue}))
    assert out[-2] == "source: nodes and pods in all namespaces"
    assert out[-1] == "fits: yes (need 1 node, 2 free)"
