#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark workload: GBS, row length, dataset and step counts, read from the environment.

    AUTOTUNE_GBS                global batch size of every ranked run (default 128)
    AUTOTUNE_SEQ_LEN            tokens per row; every row has exactly this many (default 8192)
    AUTOTUNE_DATASET            dataset path stamped as --prompt-data (default: leave the baseline's)
    AUTOTUNE_NUM_STEPS          --num-rollout of an UNPROFILED ranked run (default 20: steps 0-19; at least 20,
                                the steps of the convergence check)
    AUTOTUNE_PROFILE_CAPTURE    recorded profiler steps K of a PROFILED ranked run (default 1, at most 3)
    AUTOTUNE_WINDOW_START       first measured step (default 5; steps 0..START-1 are warm-up)

Both modes measure the SAME window, steps WINDOW_START..N-1 of the unprofiled layout (5-19 by default, 15 samples),
so a profiled and an unprofiled run are comparable. slime steps its torch profiler at the end of each step, inside
the step's timer, so with --profile-step-start S the profiler's setup lands in step S-2 and its warm-up in step S-1.
A profiled run therefore starts recording at S = N+2: step N (20) carries the setup, step N+1 (21) the warm-up, and
steps N+2..N+1+K (22, up to 24) are recorded; --profile-step-end N+2+K is exclusive and the run has N+2+K steps
(23, 24 or 25). Completion: the last step is N-1 unprofiled (19) and N+1+K profiled (22 at K=1). The convergence check
reads steps 0-19 of unprofiled runs only: a profiled run changes --num-rollout, which slime's LR and weight-decay
schedules can depend on.

    workload.py            # print the resolved workload as JSON
    workload.py --shell    # print shell assignments (WINDOW_START=5 ...)
"""
import argparse
import json
import os
import shlex
import sys

DEFAULT_GBS = 128
DEFAULT_SEQ_LEN = 8192
DEFAULT_NUM_STEPS = 20            # unprofiled: steps 0-19, measured 5-19
MIN_NUM_STEPS = 20                # every run holds the 20 steps of the convergence check
DEFAULT_PROFILE_CAPTURE = 1       # profiled: + setup 20, warm-up 21, capture 22
MAX_PROFILE_CAPTURE = 3
DEFAULT_WINDOW_START = 5
MIN_WINDOW = 3                    # a measured window of at least 3 steps


def _int_env(name, default, minimum=1, why=""):
    """$name as an int, else `default`; either must be at least `minimum`."""
    raw = os.environ.get(name, "")
    if raw.strip() == "":
        if default < minimum:
            raise ValueError(f"{name} is unset and its default {default} is below the minimum {minimum}{why}")
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer")
    if v < minimum:
        raise ValueError(f"{name}={v} is below the minimum {minimum}{why}")
    return v


def resolve():
    """The unprofiled workload. Profiled step counts come from profiled(); a bad $AUTOTUNE_PROFILE_CAPTURE does not
    affect an unprofiled render or poll_run.sh."""
    start = _int_env("AUTOTUNE_WINDOW_START", DEFAULT_WINDOW_START, 1)
    # N >= 20 (the convergence check's steps) and the window START..N-1 holds >= MIN_WINDOW steps
    n = _int_env("AUTOTUNE_NUM_STEPS", DEFAULT_NUM_STEPS, max(MIN_NUM_STEPS, start + MIN_WINDOW),
                 f" (at least {MIN_NUM_STEPS}, and a window from AUTOTUNE_WINDOW_START={start} to step N-1 needs "
                 f"N >= {start + MIN_WINDOW})")
    if os.environ.get("AUTOTUNE_NUM_STEPS_PROFILED", "").strip():
        print("workload: AUTOTUNE_NUM_STEPS_PROFILED is no longer read; a profiled run has AUTOTUNE_NUM_STEPS + 2 + "
              "AUTOTUNE_PROFILE_CAPTURE steps", file=sys.stderr)
    return {
        "gbs": _int_env("AUTOTUNE_GBS", DEFAULT_GBS),
        "seq_len": _int_env("AUTOTUNE_SEQ_LEN", DEFAULT_SEQ_LEN),
        "dataset": os.environ.get("AUTOTUNE_DATASET", "").strip() or None,
        "num_steps": n,                     # --num-rollout of an unprofiled run
        "window_start": start,
        "window_end": n - 1,                # the same measured window in both modes
        "target_step": n - 1,               # completion marker, unprofiled
    }


def profiled(wl, capture=None):
    """Step counts of a profiled run of workload `wl` with K recorded steps: `capture`, else
    $AUTOTUNE_PROFILE_CAPTURE, else 1. ValueError outside 1..MAX_PROFILE_CAPTURE."""
    k = capture if capture is not None else _int_env("AUTOTUNE_PROFILE_CAPTURE", DEFAULT_PROFILE_CAPTURE)
    if not 1 <= k <= MAX_PROFILE_CAPTURE:
        raise ValueError(f"profile capture {k} is outside 1..{MAX_PROFILE_CAPTURE}")
    n = wl["num_steps"]
    return {
        "profile_capture": k,
        "profile_setup_step": n,            # slime's profiler setup lands in step START-2
        "profile_warmup_step": n + 1,
        "profile_start": n + 2,
        "profile_end": n + 2 + k,           # end-exclusive
        "num_steps_profiled": n + 2 + k,    # --num-rollout of a profiled run
        "target_step_profiled": n + 1 + k,  # completion marker, profiled
    }


def main(argv=None):
    import campaign

    campaign.require_settings()  # the campaign's config.env decides its settings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shell", action="store_true", help="print NAME=value shell assignments instead of JSON")
    a = ap.parse_args(argv)
    w = resolve()
    try:
        w.update(profiled(w))
    except ValueError as e:
        w["profile_error"] = str(e)
    if a.shell:
        for k, v in w.items():
            print(f"{k.upper()}={shlex.quote('' if v is None else str(v))}")
    else:
        print(json.dumps(w))


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
