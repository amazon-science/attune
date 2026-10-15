#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report the GPU capacity a run can get, per instance type, and whether the run fits.

    check_gpu_capacity.py [INSTANCE_TYPE ...] [--baseline RUN.yaml] [--nodes N]

Instance types come from the arguments, else from $AUTOTUNE_INSTANCE_TYPES (comma-separated). Every kubectl call
passes --context $AUTOTUNE_KUBECTL_CONTEXT, and namespaced reads -n $KUBECTL_NS. It uses the best source the identity
may read, and says which:
  1. Kueue quota: the ClusterQueue behind the run's LocalQueue (the --baseline RayJob's kueue.x-k8s.io/queue-name, else
     every LocalQueue in the namespace). Free GPUs = nominal quota - reserved, per flavor of the instance type; pending
     workloads queue ahead of a new run.
  2. Nodes: each node free, occupied, broken, cordoned, not ready or tainted, from its allocatable GPUs and the GPUs
     requested by unfinished pods, listed cluster-wide or, when that is refused, in $KUBECTL_NS only (an upper bound:
     pods of other namespaces are not visible), with the LocalQueue's reserved GPUs and pending workloads.
  3. Neither readable: one line, "capacity unknown: Kueue admits runs", and exit 0.
FREE = Ready, schedulable, no blocking taint and no GPU requested by a pod that has not finished. A NoSchedule or
NoExecute taint blocks unless it is nvidia.com/gpu (tolerated by GPU pods), a node.kubernetes.io/* condition taint
(reported through Ready and cordoned instead), or tolerated by the pods of --baseline (a RayJob YAML). The last line
is `fits: yes|no|unknown (...)` for the nodes the run needs: --nodes, else the GPU pods of the --baseline RayJob, else 1.
"""
import argparse
import json
import os
import subprocess
import sys

TOLERATED_KEYS = ("nvidia.com/gpu",)
TOLERATED_PREFIXES = ("node.kubernetes.io/",)


class Refused(Exception):
    """kubectl was refused by RBAC (Forbidden)."""


def kubectl_json(ctx, args):
    cmd = ["kubectl", "--context", ctx] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        sys.exit(f"{' '.join(cmd)} timed out after 120 s")
    if r.returncode != 0:
        if "forbidden" in r.stderr.lower():
            raise Refused(" ".join(a for a in args if a not in ("-o", "json")))
        if "doesn't have a resource type" in r.stderr or "(NotFound)" in r.stderr:
            raise Absent(r.stderr.strip())
        sys.exit(f"{' '.join(cmd)} failed:\n{r.stderr}")
    return json.loads(r.stdout)


class Absent(Exception):
    """The resource type or object does not exist (e.g. a cluster without Kueue)."""


def readable(ctx, args, optional=False):
    """kubectl_json, or None when RBAC refuses the read (or, `optional`, when the resource does not exist)."""
    try:
        return kubectl_json(ctx, args)
    except Refused:
        return None
    except Absent:
        if optional:
            return None
        raise SystemExit(f"kubectl {' '.join(args)}: not found")


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


def baseline_info(path):
    """(queue name or None, GPU pods) of the RayJob(s) in a YAML file."""
    import yaml
    with open(path) as f:
        docs = [d for d in yaml.safe_load_all(f) if isinstance(d, dict) and d.get("kind") == "RayJob"]
    queue, pods = None, 0
    for job in docs:
        queue = queue or ((job.get("metadata") or {}).get("labels") or {}).get("kueue.x-k8s.io/queue-name")
        rcs = (job.get("spec") or {}).get("rayClusterSpec") or {}
        for g in [rcs.get("headGroupSpec") or {}] + list(rcs.get("workerGroupSpecs") or []):
            cs = ((g.get("template") or {}).get("spec") or {}).get("containers") or []
            if any(int(((c.get("resources") or {}).get("limits") or (c.get("resources") or {}).get("requests") or {})
                       .get("nvidia.com/gpu", 0)) for c in cs):
                pods += int(g.get("replicas", 1))
    return queue, pods


def _gpu(resources):
    """nvidia.com/gpu of a Kueue [{name, total|nominalQuota}] list."""
    for r in resources or []:
        if r.get("name") == "nvidia.com/gpu":
            return int(float(str(r.get("total", r.get("nominalQuota", 0)))))
    return 0


def queue_flavors(lq, types):
    """{flavor: instance type} of the LocalQueue's flavors whose node labels pick one of `types`."""
    out = {}
    for f in (lq.get("status") or {}).get("flavors") or []:
        t = (f.get("nodeLabels") or {}).get("node.kubernetes.io/instance-type")
        if t in types:
            out[f["name"]] = t
    return out


def kueue_lines(lq, cq, types, per_node):
    """(lines, {instance type: free nodes by quota} or None) for one LocalQueue and its ClusterQueue (None: unreadable)."""
    st = lq.get("status") or {}
    flavors = queue_flavors(lq, types)
    reserved_lq = {f["name"]: _gpu(f.get("resources")) for f in st.get("flavorsReservation") or []}
    name, cq_name = lq["metadata"]["name"], (lq.get("spec") or {}).get("clusterQueue")
    lines, free = [], None
    if cq is not None:
        quota = {}
        for rg in (cq.get("spec") or {}).get("resourceGroups") or []:
            for f in rg.get("flavors") or []:
                quota[f["name"]] = _gpu(f.get("resources"))
        cst = cq.get("status") or {}
        reserved = {f["name"]: _gpu(f.get("resources")) for f in cst.get("flavorsReservation") or []}
        free = {}
        for fl, t in sorted(flavors.items()):
            left = max(0, quota.get(fl, 0) - reserved.get(fl, 0))
            free[t] = free.get(t, 0) + left // per_node.get(t, 8)
            lines.append(f"[{t}]  Kueue {cq_name}/{fl}: quota {quota.get(fl, 0)} GPUs, reserved {reserved.get(fl, 0)}, "
                         f"free {left} ({left // per_node.get(t, 8)} nodes)")
        lines.append(f"Kueue ClusterQueue {cq_name}: {cst.get('pendingWorkloads', 0)} workloads pending, "
                     f"{cst.get('admittedWorkloads', 0)} admitted")
    else:
        held = ", ".join(f"{reserved_lq.get(fl, 0)} GPUs on {fl} ({t})" for fl, t in sorted(flavors.items())) or "none"
        lines.append(f"Kueue LocalQueue {name} -> ClusterQueue {cq_name} (quota not readable): this queue reserves {held}; "
                     f"{st.get('pendingWorkloads', 0)} workloads pending, {st.get('admittedWorkloads', 0)} admitted")
    pending = int((cq.get("status") or {}).get("pendingWorkloads", 0)) if cq is not None else int(st.get("pendingWorkloads", 0))
    return lines, free, pending


def main(argv=None):
    try:
        import campaign
    except ImportError:  # a copy in the campaign directory: the environment decides
        campaign = None
    if campaign:
        campaign.require_settings()  # the campaign's config.env decides its settings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instance_types", nargs="*", help="node.kubernetes.io/instance-type values")
    ap.add_argument("--baseline", help="RayJob YAML whose pod tolerations also count as tolerated, and whose queue and "
                                       "GPU pods are the run's")
    ap.add_argument("--nodes", type=int, help="nodes the run needs (default: the --baseline RayJob's GPU pods, else 1)")
    a = ap.parse_args(argv)
    ctx = os.environ.get("AUTOTUNE_KUBECTL_CONTEXT", "").strip()
    if not ctx:
        sys.exit("set AUTOTUNE_KUBECTL_CONTEXT to the target cluster context")
    ns = os.environ.get("KUBECTL_NS", "").strip()
    types = a.instance_types or [t.strip() for t in os.environ.get("AUTOTUNE_INSTANCE_TYPES", "").split(",") if t.strip()]
    if not types:
        sys.exit("give the GPU node instance types as arguments or in AUTOTUNE_INSTANCE_TYPES")
    tolerations = baseline_tolerations(a.baseline) if a.baseline else []
    queue, base_pods = baseline_info(a.baseline) if a.baseline else (None, 0)
    need = a.nodes or base_pods or 1

    nodes = readable(ctx, ["get", "nodes", "-l", "node.kubernetes.io/instance-type in (%s)" % ",".join(types),
                           "-o", "json"])
    pods, pod_scope = readable(ctx, ["get", "pods", "-A", "-o", "json"]), "all namespaces"
    if pods is None and ns:
        pods, pod_scope = readable(ctx, ["-n", ns, "get", "pods", "--field-selector",
                                         "status.phase!=Succeeded,status.phase!=Failed", "-o", "json"]), "namespace " + ns
    lqs = (readable(ctx, ["-n", ns, "get", "localqueues", "-o", "json"], optional=True) or {}).get("items") if ns else None
    if lqs and queue:
        lqs = [q for q in lqs if q["metadata"]["name"] == queue] or lqs
    cqs = {}
    for q in lqs or []:
        name = (q.get("spec") or {}).get("clusterQueue")
        if name and name not in cqs:
            cqs[name] = readable(ctx, ["get", "clusterqueue", name, "-o", "json"], optional=True)
    if nodes is None and not lqs:
        print("capacity unknown: Kueue admits runs (this identity may read neither the nodes nor the Kueue queues)")
        return 0

    per_node, free_nodes = {}, {}
    if nodes is not None:
        used = gpus_in_use(pods["items"]) if pods is not None else {}
        totals = {t: {} for t in types}
        for n in nodes["items"]:
            name = n["metadata"]["name"]
            itype = (n["metadata"].get("labels") or {}).get("node.kubernetes.io/instance-type", "unknown")
            alloc = int(((n.get("status") or {}).get("allocatable") or {}).get("nvidia.com/gpu", 0))
            if alloc:
                per_node[itype] = max(per_node.get(itype, 0), alloc)
            state, detail = node_state(n, used.get(name, 0), tolerations)
            bucket = totals.setdefault(itype, {})
            bucket[state] = bucket.get(state, 0) + 1
            if state != "occupied" and pods is not None:
                print(f"{state.upper() + ':':11s}{name.split('.')[0]}  ({itype})  {detail}")
        if pods is not None:
            print("\n--- Summary (FREE = Ready, schedulable, no blocking taint, 0 GPUs requested) ---")
            for itype, b in totals.items():
                counts = "  ".join(f"{k}={b.get(k, 0)}" for k in ("free", "occupied", "broken", "cordoned",
                                                                   "not_ready", "tainted"))
                print(f"[{itype}]  {counts}  total={sum(b.values())}")
                free_nodes[itype] = b.get("free", 0)
        else:
            for itype, b in totals.items():
                print(f"[{itype}]  nodes={sum(b.values())}  allocatable_gpu/node={per_node.get(itype, 0)}  "
                      f"(GPU use per node not readable)")
        print(f"Total nodes: {len(nodes['items'])}")
    quota_free, pending, klines = None, 0, []
    for q in lqs or []:
        lines, free, pend = kueue_lines(q, cqs.get((q.get("spec") or {}).get("clusterQueue")), types, per_node)
        klines += lines
        pending = max(pending, pend)
        if free is not None:
            quota_free = {t: quota_free.get(t, 0) + v for t, v in free.items()} if quota_free else free
    for ln in klines:
        print(ln)

    if quota_free is not None:
        source = "Kueue ClusterQueue quota"
        free = sum(quota_free.values())
        if free_nodes:
            free = min(free, sum(free_nodes.values()))
            source += " and nodes (pods in %s)" % pod_scope
    elif free_nodes:
        free = sum(free_nodes.values())
        source = "nodes and pods in %s" % pod_scope
        if pod_scope != "all namespaces":
            source += " (pods of other namespaces are not visible: free is an upper bound)"
        if lqs:
            source += " + LocalQueue " + ", ".join(q["metadata"]["name"] for q in lqs)
    else:
        source = "LocalQueue only (quota and node use not readable)"
        free = None
    print(f"source: {source}")
    if free is None:
        verdict = f"unknown (need {need} node{'s' * (need != 1)}; Kueue admits runs)"
    elif free >= need and not pending:
        verdict = f"yes (need {need} node{'s' * (need != 1)}, {free} free)"
    elif free >= need:
        verdict = f"yes, behind {pending} pending workloads (need {need}, {free} free)"
    else:
        verdict = f"no (need {need} node{'s' * (need != 1)}, {free} free{f', {pending} workloads pending' if pending else ''}): queue it"
    print(f"fits: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
