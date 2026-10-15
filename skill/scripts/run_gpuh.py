#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-hours a run has consumed: num_gpus x (admission -> job end). Queue time before admission is free.

    run_gpuh.py <run_id> [--context CTX] [--namespace NS]            # measure, print JSON
    run_gpuh.py <run_id> --launch --gpus N [--log FILE]               # gpuh.jsonl line at submit
    run_gpuh.py <run_id> --record [--status S] [--log FILE]           # measure, append the measured line
    run_gpuh.py --total [--log FILE]                                  # campaign total so far

On a cluster with Kueue it reads the run's Workload (admission time, GPU count from its pod sets) and its
RayJob (end time); a run with no Workload (rejected before admission) costs 0 GPU-h. Without the Kueue
Workload API it charges RayJob startTime -> endTime, with the GPU count summed from the RayJob's pod
templates. A still-running job is charged up to now (`final: false`). Call it in finalize BEFORE teardown:
the Workload and RayJob are deleted with the run.
Prints one JSON object: {run_id, num_gpus, admitted, end, hours, gpuh, final, source}.
The campaign GPU-hour log (FILE, default $AUTOTUNE_GPUH_LOG, else ./gpuh.jsonl) is append-only (the launcher's
integrity check holds it to that): lines {"run", "gpus", "submitted", "admitted", "ended", "gpuh", "cum_gpuh",
"status"}, where --launch appends a run's first line with nulls and every --record appends a measured line. A run's
last line is its current state; cum_gpuh is the sum of every run's last measured gpuh, this one included.
Measuring needs the cluster: --context (default $AUTOTUNE_KUBECTL_CONTEXT) and --namespace (default $KUBECTL_NS);
every kubectl call passes both.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys


def _kubectl(args, ctx, ns):
    p = subprocess.run(["kubectl", "--context", ctx, "-n", ns] + args, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError("kubectl %s: %s" % (" ".join(args), p.stderr.strip()[:300]))
    return json.loads(p.stdout)


def _ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc) if s else None


def _gpus(workload):
    n = 0
    for ps in workload.get("spec", {}).get("podSets", []):
        per = 0
        for c in ps.get("template", {}).get("spec", {}).get("containers", []):
            r = c.get("resources", {})
            per += int((r.get("limits") or {}).get("nvidia.com/gpu") or (r.get("requests") or {}).get("nvidia.com/gpu") or 0)
        n += per * int(ps.get("count", 1))
    return n


def _rayjob_gpus(rj):
    rcs = (rj.get("spec") or {}).get("rayClusterSpec") or {}
    n = 0
    for g in [rcs.get("headGroupSpec") or {}] + list(rcs.get("workerGroupSpecs") or []):
        reps = int(g.get("replicas", 1))
        for c in ((g.get("template") or {}).get("spec") or {}).get("containers", []):
            r = c.get("resources", {})
            n += reps * int((r.get("limits") or {}).get("nvidia.com/gpu") or (r.get("requests") or {}).get("nvidia.com/gpu") or 0)
    return n


def measure_rayjob(run_id, ctx, ns):
    """No Kueue Workload API on the cluster: charge RayJob startTime -> endTime (or now)."""
    try:
        rj = _kubectl(["get", "rayjob", run_id, "-o", "json"], ctx, ns)
    except RuntimeError:
        return {"run_id": run_id, "num_gpus": None, "admitted": None, "end": None, "hours": 0.0, "gpuh": 0.0,
                "final": True, "source": "no RayJob (never created, or already torn down: check the ledger)"}
    st = rj.get("status") or {}
    t0, end = _ts(st.get("startTime")), _ts(st.get("endTime"))
    final = end is not None
    src = "rayjob startTime -> endTime" if final else "rayjob startTime -> now (running)"
    if end is None:
        end = dt.datetime.now(dt.timezone.utc)
    n = _rayjob_gpus(rj)
    hours = max((end - t0).total_seconds(), 0.0) / 3600.0 if t0 else 0.0
    return {"run_id": run_id, "num_gpus": n, "admitted": t0.isoformat() if t0 else None, "end": end.isoformat(),
            "hours": round(hours, 4), "gpuh": round(n * hours, 3), "final": final, "source": src}


def measure(run_id, ctx, ns):
    try:
        wls = _kubectl(["get", "workloads", "-o", "json"], ctx, ns)["items"]
    except RuntimeError as e:
        if "doesn't have a resource type" not in str(e):
            raise
        return measure_rayjob(run_id, ctx, ns)
    mine = [w for w in wls if any(o.get("name") == run_id for o in w["metadata"].get("ownerReferences", []))]
    if not mine:
        return {"run_id": run_id, "num_gpus": None, "admitted": None, "end": None, "hours": 0.0, "gpuh": 0.0,
                "final": True, "source": "no Kueue Workload (never admitted, or already torn down: check the ledger)"}
    w = mine[0]
    cond = {c["type"]: c for c in w.get("status", {}).get("conditions", [])}
    adm = cond.get("Admitted")
    if not adm or adm.get("status") != "True":
        if "Finished" not in cond:
            return {"run_id": run_id, "num_gpus": _gpus(w), "admitted": None, "end": None, "hours": 0.0,
                    "gpuh": 0.0, "final": False, "source": "queued, not admitted"}
    t0 = _ts((adm or {}).get("lastTransitionTime"))
    end, src = None, "workload"
    try:
        rj = _kubectl(["get", "rayjob", run_id, "-o", "json"], ctx, ns)
        end = _ts((rj.get("status") or {}).get("endTime"))
        src = "workload admission -> rayjob endTime" if end else src
    except RuntimeError:
        pass
    if end is None and cond.get("Finished", {}).get("status") == "True":
        end, src = _ts(cond["Finished"]["lastTransitionTime"]), "workload admission -> workload Finished"
    final = end is not None
    if end is None:
        end, src = dt.datetime.now(dt.timezone.utc), "workload admission -> now (running)"
    n = _gpus(w)
    hours = max((end - t0).total_seconds(), 0.0) / 3600.0 if t0 else 0.0
    return {"run_id": run_id, "num_gpus": n, "admitted": t0.isoformat() if t0 else None, "end": end.isoformat(),
            "hours": round(hours, 4), "gpuh": round(n * hours, 3), "final": final, "source": src}


def _now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _latest(rows):
    """run -> its last line."""
    out = {}
    for r in rows:
        out[r.get("run")] = r
    return out


def _append_log(path, fn):
    """fn(rows) -> (line to append or None, result), under an exclusive lock; the log only grows."""
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows = read_log(path)
        line, out = fn(rows)
        if line is not None:
            with open(path, "a") as f:
                f.write(json.dumps(line) + "\n")
        return out


def _total(rows):
    """The campaign's charge so far: the sum of every run's last measured gpuh (a later unmeasured line leaves it)."""
    measured = {r.get("run"): r["gpuh"] for r in rows if isinstance(r.get("gpuh"), (int, float))}
    return round(sum(measured.values()), 3)


def log_launch(path, run_id, gpus):
    def fn(rows):
        if any(r.get("run") == run_id for r in rows):
            return None, {"run": run_id, "note": "already logged"}
        row = {"run": run_id, "gpus": gpus, "submitted": _now(), "admitted": None, "ended": None,
               "gpuh": None, "cum_gpuh": None, "status": "submitted"}
        return row, row
    return _append_log(path, fn)


def log_record(path, m, status):
    def fn(rows):
        prev = _latest(rows).get(m["run_id"]) or {}  # launched without a --launch line: submit time unknown
        row = {"run": m["run_id"], "gpus": m["num_gpus"] if m["num_gpus"] is not None else prev.get("gpus"),
               "submitted": prev.get("submitted"), "admitted": m["admitted"],
               "ended": m["end"] if m["final"] else None, "gpuh": m["gpuh"] if m["final"] else None,
               "cum_gpuh": None, "status": status if m["final"] else "running"}
        if m["final"]:
            row["cum_gpuh"] = _total(rows + [row])
        return row, row
    return _append_log(path, fn)


def main():
    import campaign

    campaign.require_settings()  # the campaign's config.env decides its settings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--log", default=os.environ.get("AUTOTUNE_GPUH_LOG") or "gpuh.jsonl")
    ap.add_argument("--launch", action="store_true", help="append the run's gpuh.jsonl line at submit")
    ap.add_argument("--gpus", type=int, help="GPU count for --launch")
    ap.add_argument("--record", action="store_true", help="measure and complete the run's gpuh.jsonl line")
    ap.add_argument("--status", default="done", help="status for --record (e.g. done, failed, oom, rejected)")
    ap.add_argument("--total", action="store_true", help="print the campaign GPU-h total from the log")
    ap.add_argument("--context", default=os.environ.get("AUTOTUNE_KUBECTL_CONTEXT"),
                    help="kubectl context (default $AUTOTUNE_KUBECTL_CONTEXT)")
    ap.add_argument("--namespace", default=os.environ.get("KUBECTL_NS"), help="namespace (default $KUBECTL_NS)")
    a = ap.parse_args()
    if a.total:
        rows = read_log(a.log)
        latest = _latest(rows)
        print(json.dumps({"cum_gpuh": _total(rows), "runs": len(latest),
                          "open": [r for r, row in latest.items() if row.get("gpuh") is None]}))
        return
    if not a.run_id:
        ap.error("run_id required")
    if a.launch:
        if a.gpus is None:
            ap.error("--launch needs --gpus N")
        print(json.dumps(log_launch(a.log, a.run_id, a.gpus)))
        return
    if not a.context or not a.namespace:
        ap.error("measuring needs --context and --namespace (or $AUTOTUNE_KUBECTL_CONTEXT and $KUBECTL_NS)")
    m = measure(a.run_id, a.context, a.namespace)
    print(json.dumps(log_record(a.log, m, a.status) if a.record else m))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": "%s: %s" % (type(e).__name__, e)}))
        sys.exit(1)
