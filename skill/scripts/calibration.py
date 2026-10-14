#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print the baseline calibration a campaign ranks with: the band b and the baseline denominator.

    calibration.py            # JSON {band, denominator, sigma_alloc, runs}; exit 1 if $AUTOTUNE_CALIBRATION is unusable
    calibration.py --require  # exit 1 unless $AUTOTUNE_CALIBRATION names a usable file

The file (JSON) holds `denominator` = mean tokens/s/GPU of >= 5 baseline replicates on distinct allocations,
`sigma_alloc`, `band` = max(0.03, 3 sigma_alloc / denominator), and `runs`: the replicate count, or the list of
replicates, each with the `nodes` it ran on (the allocations must then be distinct). Without $AUTOTUNE_CALIBRATION the
band is 0.03 and there is no denominator.
"""
import argparse
import json
import os
import sys

ENV = "AUTOTUNE_CALIBRATION"


def problems(d):
    """Why a calibration file is unusable (empty = usable)."""
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
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require", action="store_true", help="fail unless $%s names a usable file" % ENV)
    a = ap.parse_args(argv)
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
    bad = problems(d)
    if bad:
        sys.exit("calibration: %s is not usable: %s" % (path, "; ".join(bad)))
    runs = d.get("runs")
    print(json.dumps({"band": d["band"], "denominator": d["denominator"], "sigma_alloc": d.get("sigma_alloc"),
                      "runs": runs if isinstance(runs, int) else len(runs), "source": path}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
