#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The baseline calibration a campaign ranks with: the band b and the baseline denominator.

    calibration.py            # JSON {band, denominator, sigma_alloc, runs, num_steps, window, source};
                              # exit 1 if $AUTOTUNE_CALIBRATION is unusable
    calibration.py --require  # exit 1 unless $AUTOTUNE_CALIBRATION names a usable file
    calibration.py build --out <file> --num-gpus N --replicate <run_id> <train.log> <node,node,...> [--replicate ...]
    calibration.py stamp <file> --num-steps N --window START END

The file (JSON) holds `denominator` = mean tokens/s/GPU of >= 5 baseline replicates on distinct allocations,
`sigma_alloc` (their sample standard deviation), `band` = max(0.03, 3 sigma_alloc / denominator), `runs`: the
replicate count, or the list of replicates, each with the `nodes` it ran on (the allocations must then be distinct),
and the stamps `num_steps` and `window` [start, end] of the measurement the replicates used, which must equal the
campaign's workload (workload.py: 20 steps, window [5, 19] by default). Without $AUTOTUNE_CALIBRATION the band is 0.03
and there is no denominator.

`build` scores each replicate's log with read_metrics.sh over the campaign's window and writes a stamped file; it
writes nothing unless the result is usable. `stamp` adds the stamps to a file written before they existed: give the
step count and window its replicates were actually measured with (a file measured over another window stays unusable;
rebuild it from replicates measured under this workload). Stamps a file already carries are never changed.
"""
import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import workload

ENV = "AUTOTUNE_CALIBRATION"
HERE = Path(__file__).resolve().parent
RULE = "denominator = mean of >= 5 baseline replicates on distinct allocations; band = max(0.03, 3 sigma_alloc / denominator)"


def problems(d, wl=None):
    """Why a calibration file is unusable (empty = usable); `wl` is the campaign's workload.resolve()."""
    runs = d.get("runs")
    n = runs if isinstance(runs, int) else len(runs or [])
    p = []
    if n < 5:
        p.append("%d replicates (need >= 5)" % n)
    if isinstance(runs, list):
        allocs = [tuple(sorted(r.get("nodes") or [])) for r in runs]
        if any(not a for a in allocs) or len(set(allocs)) != len(allocs):
            p.append("replicates need recorded, distinct allocations")
    if not isinstance(d.get("denominator"), (int, float)) or d["denominator"] <= 0:
        p.append("no denominator")
    if not isinstance(d.get("band"), (int, float)) or d["band"] < 0.03:
        p.append("band missing or below 0.03")
    if wl is not None:
        want = (wl["num_steps"], [wl["window_start"], wl["window_end"]])
        if d.get("num_steps") is None or d.get("window") is None:
            p.append("no num_steps/window stamps (written before they existed): rebuild it with `calibration.py build` "
                     "from replicates measured with num_steps %d, window %s, or record how its replicates were "
                     "measured with `calibration.py stamp <file> --num-steps N --window START END`" % want)
        elif (d.get("num_steps"), d.get("window")) != want:
            p.append("measured with num_steps %s, window %s; the campaign measures num_steps %d, window %s: rebuild it "
                     "with `calibration.py build` from replicates measured under this workload"
                     % (d.get("num_steps"), d.get("window"), want[0], want[1]))
    return p


def _workload():
    try:
        return workload.resolve()
    except ValueError as e:
        sys.exit("calibration: %s" % e)


def build(a):
    wl = _workload()
    window = [wl["window_start"], wl["window_end"]]
    runs, errs = [], []
    for run_id, log, nodes in a.replicate:
        p = subprocess.run(["bash", str(HERE / "read_metrics.sh"), log, str(window[0]), str(window[1]), str(a.num_gpus)],
                           capture_output=True, text=True)
        try:
            m = json.loads(p.stdout)
        except ValueError:
            m = {}
        if p.returncode != 0 or not m.get("valid"):
            errs.append("%s: no valid window %d-%d (%s)" % (run_id, window[0], window[1],
                                                             m.get("invalid_reason") or p.stderr.strip()[:200]))
            continue
        runs.append({"run_id": run_id, "log": log, "nodes": [x for x in nodes.split(",") if x],
                     "tok_s_gpu": m["aggregate_tok_per_s_per_gpu"]})
    if errs:
        sys.exit("calibration build: " + "; ".join(errs))
    toks = [r["tok_s_gpu"] for r in runs]
    mean = statistics.mean(toks)
    sd = statistics.stdev(toks) if len(toks) > 1 else 0.0
    d = {"denominator": mean, "sigma_alloc": sd, "band": max(0.03, 3 * sd / mean), "runs": runs,
         "num_gpus": a.num_gpus, "num_steps": wl["num_steps"], "window": window, "rule": RULE,
         "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    bad = problems(d, wl)
    if bad:
        sys.exit("calibration build: not usable, nothing written: %s" % "; ".join(bad))
    Path(a.out).write_text(json.dumps(d, indent=1) + "\n")
    print(json.dumps({"out": a.out, "band": d["band"], "denominator": mean, "sigma_alloc": sd, "runs": len(runs),
                      "num_steps": d["num_steps"], "window": window}))
    return 0


def stamp(a):
    try:
        d = json.loads(Path(a.file).read_text())
    except (OSError, ValueError) as e:
        sys.exit("calibration stamp: cannot read %s: %s" % (a.file, e))
    given = {"num_steps": a.num_steps, "window": list(a.window)}
    for k, v in given.items():
        if d.get(k) is not None and d[k] != v:
            sys.exit("calibration stamp: %s already records %s %s; stamps record how the replicates were measured "
                     "and are not changed" % (a.file, k, d[k]))
    d.update(given)
    Path(a.file).write_text(json.dumps(d, indent=1) + "\n")
    bad = problems(d, _workload())
    print("stamped %s: num_steps %d, window %s%s" % (a.file, a.num_steps, given["window"],
                                                     "; still unusable: " + "; ".join(bad) if bad else ""))
    return 0


def show(a):
    path = os.environ.get(ENV, "").strip()
    if not path:
        if a.require:
            sys.exit("calibration: $%s is not set (--require)" % ENV)
        print(json.dumps({"band": 0.03, "denominator": None, "source": "default (no calibration file)"}))
        return 0
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        sys.exit("calibration: cannot read %s: %s" % (path, e))
    bad = problems(d, _workload())
    if bad:
        sys.exit("calibration: %s is not usable: %s" % (path, "; ".join(bad)))
    runs = d.get("runs")
    print(json.dumps({"band": d["band"], "denominator": d["denominator"], "sigma_alloc": d.get("sigma_alloc"),
                      "runs": runs if isinstance(runs, int) else len(runs), "num_steps": d["num_steps"],
                      "window": d["window"], "source": path}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require", action="store_true", help="fail unless $%s names a usable file" % ENV)
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build", help="score >= 5 baseline replicates and write a stamped calibration file")
    b.add_argument("--out", required=True, help="calibration file to write")
    b.add_argument("--num-gpus", type=int, required=True, help="the baseline's GPU count")
    b.add_argument("--replicate", nargs=3, action="append", required=True, metavar=("RUN_ID", "LOG", "NODES"),
                   help="one replicate: its run id, train.log (read through $AUTOTUNE_LOG_POD when set) and the "
                        "comma-separated nodes it ran on")
    s = sub.add_parser("stamp", help="record the step count and window an unstamped file's replicates were measured with")
    s.add_argument("file")
    s.add_argument("--num-steps", type=int, required=True)
    s.add_argument("--window", type=int, nargs=2, required=True, metavar=("START", "END"))
    a = ap.parse_args(argv)
    return {"build": build, "stamp": stamp}.get(a.cmd, show)(a)


if __name__ == "__main__":
    sys.exit(main())
