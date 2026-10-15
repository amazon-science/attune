#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report GPU nodes as free, occupied, broken, cordoned, not ready or tainted, per instance type.

    check_gpu_capacity.py [INSTANCE_TYPE ...] [--baseline RUN.yaml]

Instance types come from the arguments, else from $AUTOTUNE_INSTANCE_TYPES (comma-separated). Every kubectl call
passes --context $AUTOTUNE_KUBECTL_CONTEXT; nodes and pods are listed cluster-wide. FREE = Ready, schedulable, no
blocking taint and no GPU requested by a pod that has not finished. A NoSchedule or NoExecute taint blocks unless it
is nvidia.com/gpu (tolerated by GPU pods), a node.kubernetes.io/* condition taint (reported through Ready and
cordoned instead), or tolerated by the pods of --baseline (a RayJob YAML).
"""
import argparse
import json
import os
import subprocess
import sys

TOLERATED_KEYS = ("nvidia.com/gpu",)
TOLERATED_PREFIXES = ("node.kubernetes.io/",)


def kubectl_json(ctx, args):
    cmd = ["kubectl", "--context", ctx] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        sys.exit(f"{' '.join(cmd)} timed out after 120 s")
    if r.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{r.stderr}")
    return json.loads(r.stdout)


def baseline_tolerations(path):
    """Tolerations of every pod template of the RayJob(s) in a YAML file."""
    import yaml
    with open(path) as f:
        docs = [d for d in yaml.safe_load_all(f) if isinstance(d, dict) and d.get("kind") == "RayJob"]
    out = []
    for job in docs:
        rcs = (job.get("spec") or {}).get("rayClusterSpec") or {}
        for g in [rcs.get("headGroupSpec") or {}] + list(rcs.get("workerGroupSpecs") or []):
            out += ((g.get("template") or {}).get("spec") or {}).get("tolerations") or []
    return out


def tolerates(tol, taint):
    """Kubernetes toleration matching: effect (empty = any), key (empty with Exists = any), Exists or Equal value."""
    if tol.get("effect") and tol["effect"] != taint.get("effect"):
        return False
    if tol.get("operator", "Equal") == "Exists":
        return not tol.get("key") or tol["key"] == taint.get("key")
    return tol.get("key") == taint.get("key") and tol.get("value", "") == taint.get("value", "")


def blocking_taints(node, tolerations):
    out = []
    for t in (node.get("spec") or {}).get("taints") or []:
        key = t.get("key", "")
        if t.get("effect") not in ("NoSchedule", "NoExecute") or key in TOLERATED_KEYS:
            continue
        if key.startswith(TOLERATED_PREFIXES) or any(tolerates(x, t) for x in tolerations):
            continue
        out.append("%s=%s:%s" % (key, t.get("value", ""), t.get("effect")))
    return out


def gpus_in_use(pods):
    """node name -> GPUs requested by pods that are scheduled and not finished (Pending-but-bound pods hold GPUs)."""
    def req(c):
        return int(((c.get("resources") or {}).get("requests") or {}).get("nvidia.com/gpu", 0))
    used = {}
    for p in pods:
        spec = p.get("spec") or {}
        if (p.get("status") or {}).get("phase") in ("Succeeded", "Failed") or not spec.get("nodeName"):
            continue
        gpu = sum(req(c) for c in spec.get("containers") or [])
        gpu += sum(req(c) for c in spec.get("initContainers") or [] if c.get("restartPolicy") == "Always")
        if gpu:
            used[spec["nodeName"]] = used.get(spec["nodeName"], 0) + gpu
    return used


def node_state(n, used, tolerations):
    """(state, detail): the first of broken, not_ready, cordoned, tainted, occupied, free that applies."""
    alloc = int(((n.get("status") or {}).get("allocatable") or {}).get("nvidia.com/gpu", 0))
    ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in (n.get("status") or {}).get("conditions") or [])
    taints = blocking_taints(n, tolerations)
    if alloc == 0:
        return "broken", "allocatable_gpu=0"
    if not ready:
        return "not_ready", "Ready!=True"
    if (n.get("spec") or {}).get("unschedulable"):
        return "cordoned", f"allocatable_gpu={alloc} used_gpu={used}"
    if taints:
        return "tainted", "blocking taint " + ", ".join(taints)
    return ("occupied" if used else "free"), f"allocatable_gpu={alloc} used_gpu={used}"


def main(argv=None):
    try:
        import campaign
    except ImportError:  # a copy in the campaign directory: the environment decides
        campaign = None
    if campaign:
        campaign.require_settings()  # the campaign's config.env decides its settings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instance_types", nargs="*", help="node.kubernetes.io/instance-type values")
    ap.add_argument("--baseline", help="RayJob YAML whose pod tolerations also count as tolerated")
    a = ap.parse_args(argv)
    ctx = os.environ.get("AUTOTUNE_KUBECTL_CONTEXT", "").strip()
    if not ctx:
        sys.exit("set AUTOTUNE_KUBECTL_CONTEXT to the target cluster context")
    types = a.instance_types or [t.strip() for t in os.environ.get("AUTOTUNE_INSTANCE_TYPES", "").split(",") if t.strip()]
    if not types:
        sys.exit("give the GPU node instance types as arguments or in AUTOTUNE_INSTANCE_TYPES")
    tolerations = baseline_tolerations(a.baseline) if a.baseline else []
    nodes = kubectl_json(ctx, ["get", "nodes", "-l", "node.kubernetes.io/instance-type in (%s)" % ",".join(types),
                               "-o", "json"])["items"]
    used = gpus_in_use(kubectl_json(ctx, ["get", "pods", "-A", "-o", "json"])["items"])
    totals = {t: {} for t in types}
    for n in nodes:
        name = n["metadata"]["name"]
        itype = (n["metadata"].get("labels") or {}).get("node.kubernetes.io/instance-type", "unknown")
        state, detail = node_state(n, used.get(name, 0), tolerations)
        bucket = totals.setdefault(itype, {})
        bucket[state] = bucket.get(state, 0) + 1
        if state != "occupied":
            print(f"{state.upper() + ':':11s}{name.split('.')[0]}  ({itype})  {detail}")
    print("\n--- Summary (FREE = Ready, schedulable, no blocking taint, 0 GPUs requested) ---")
    for itype, b in totals.items():
        counts = "  ".join(f"{k}={b.get(k, 0)}" for k in ("free", "occupied", "broken", "cordoned", "not_ready", "tainted"))
        print(f"[{itype}]  {counts}  total={sum(b.values())}")
    print(f"Total nodes: {len(nodes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
