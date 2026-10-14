#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark workload: GBS, row length, dataset and step counts, read from the environment.

    AUTOTUNE_GBS                global batch size of every ranked run (default 128)
    AUTOTUNE_SEQ_LEN            tokens per row; every row has exactly this many (default 8192)
    AUTOTUNE_DATASET            dataset path stamped as --prompt-data (default: leave the baseline's)
    AUTOTUNE_NUM_STEPS          --num-rollout of an UNPROFILED ranked run (default 14: steps 0-13)
    AUTOTUNE_NUM_STEPS_PROFILED --num-rollout of a PROFILED ranked run (default 17: steps 0-16)
    AUTOTUNE_WINDOW_START       first measured step (default 4; steps 0..START-1 are discarded)

Both modes measure the SAME window, steps WINDOW_START..N-1 of the unprofiled layout (4-13 by default,
10 samples), so a profiled and an unprofiled run are comparable. A profiled run adds the warmup step N
and the capture steps N+1..M-1 after that window (14, then 15-16 by default; --profile-step-end is
exclusive, so 17). Completion: the last step is N-1 unprofiled (13) and M-1 profiled (16).

    workload.py            # print the resolved workload as JSON
    workload.py --shell    # print shell assignments (WINDOW_START=4 ...)
"""
import argparse
import json
import os
import sys

DEFAULT_GBS = 128
DEFAULT_SEQ_LEN = 8192
DEFAULT_NUM_STEPS = 14            # unprofiled: steps 0-13, measured 4-13
DEFAULT_NUM_STEPS_PROFILED = 17   # profiled: + warmup 14 and capture 15-16
DEFAULT_WINDOW_START = 4
MIN_WINDOW = 3                    # a measured window of at least 3 steps


def _int_env(name, default, minimum=1):
    raw = os.environ.get(name, "")
    if raw.strip() == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer")
    if v < minimum:
        raise ValueError(f"{name}={v} is below the minimum {minimum}")
    return v


def resolve():
    start = _int_env("AUTOTUNE_WINDOW_START", DEFAULT_WINDOW_START, 1)
    n = _int_env("AUTOTUNE_NUM_STEPS", DEFAULT_NUM_STEPS, start + MIN_WINDOW)
    m = _int_env("AUTOTUNE_NUM_STEPS_PROFILED", max(DEFAULT_NUM_STEPS_PROFILED, n + 3), n + 3)
    return {
        "gbs": _int_env("AUTOTUNE_GBS", DEFAULT_GBS),
        "seq_len": _int_env("AUTOTUNE_SEQ_LEN", DEFAULT_SEQ_LEN),
        "dataset": os.environ.get("AUTOTUNE_DATASET", "").strip() or None,
        "num_steps": n,                     # --num-rollout of an unprofiled run
        "num_steps_profiled": m,            # --num-rollout of a profiled run
        "window_start": start,
        "window_end": n - 1,                # the same measured window in both modes
        "profile_warmup_step": n,
        "profile_start": n + 1,
        "profile_end": m,                   # end-exclusive
        "target_step": n - 1,               # completion marker, unprofiled
        "target_step_profiled": m - 1,      # completion marker, profiled
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shell", action="store_true", help="print NAME=value shell assignments instead of JSON")
    a = ap.parse_args(argv)
    w = resolve()
    if a.shell:
        for k, v in w.items():
            print(f"{k.upper()}={'' if v is None else v}")
    else:
        print(json.dumps(w))


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
