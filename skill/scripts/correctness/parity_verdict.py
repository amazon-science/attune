#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness check of a candidate recipe against its baseline (references/measurement.md §4, PROTOCOL.md here).

    parity_verdict.py <pair_dir | root_of_pair_dirs> [--json out.json]

A pair directory holds base.log and win.log, the train.log of the baseline's and the candidate's unprofiled runs (their
own benchmark runs, paired by pair_from_runs.py; or 20-step arms made by make_parity_arm.py, which load the parity
probe), and optionally gpus.json {"base": N, "win": M} and images.json {"base": ref, "win": ref}, the image each ran.
  L1    layer1.py on the two resolved-argument dumps, with the frozen-arguments file of $AUTOTUNE_FROZEN_ARGS (its
        directory is the setup's layer-1 directory) and, when images.json names one image pinned by digest for both
        arms, that image's rules. An unreadable frozen-arguments file fails L1. Unknown changes pass L1.
  STEP0 both logs hold the training loss of all 20 steps (0-19), each finite, and the candidate's step-0 loss is within
        STEP0_TOL (5%, relative) of the baseline's.
  Reported for review, never deciding: the loss-curve check, NVIDIA Megatron-Bridge's convergence check vendored
     verbatim in mbridge_convergence.py (scripts/performance/utils/evaluate.py:145-329 at c3c974d4f) and run with the
     thresholds its pipeline passes (argument_parser.py:1092-1125 defaults, via setup_experiment.py:1107-1116) on the
     candidate's per-step training loss (steps 0-19) against the baseline's: the first 20% of steps skipped,
     point-wise relative tolerance 10% / 5% / 2% for loss > 2.0 / 0.5-2.0 / < 0.5 with at most 10% of points outside,
     final-loss tolerance 5%, correlation >= 0.95 and a 3-sigma outlier check; and the candidate's mean relative loss
     deviation from the baseline over steps 10-19. A verdict's `review` list holds every unknown change of L1 (name,
     from, to, reason) and a failed or unscored loss-curve check.
  Reported for orientation: whether the probe's per-step rows, tokens and loss-mask tokens are identical (when the
     probe ran), the step-0 grad-norm relative difference, the 20-step mean loss gap, and tokens / logged step time
     per GPU over steps 5-19 (probe tokens when the probe ran, else the logged tokens; with gpus.json).
PASS = L1 and STEP0. Exit 0 when every pair passes, 1 when any pair fails or cannot be scored, 2 when no pair is found,
3 when the campaign fails its integrity check (harness/integrity.py, which also hashes the dataset again through the
log pod): then nothing is scored and no recipe ships, 4 when the log pod cannot be reached to hash the dataset: nothing
is scored, retry.
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import campaign  # noqa: E402
import frozen_args  # noqa: E402
import layer1  # noqa: E402

MB_PATH = os.path.join(HERE, "mbridge_convergence.py")
MB_SHA256 = "17fa9876476775a061fcb303b4ce96f2b96519285b07a9088818755dafc4cb37"  # loss-curve check: the vendored file
MB_CONFIG = {"skip_first_percent_loss": 0.20, "final_loss_tolerance": 0.05}  # the pipeline values that differ from the function's
STEPS = 20
STEP0_TOL = 0.05   # the candidate's step-0 loss may differ from the baseline's by at most 5% (relative)
REVIEW_START = 10  # the loss deviation reported for review: steps 10-19
WINDOW_START = 5   # orientation throughput only: the default measured window 5-19
COUNT_KEYS = ("rows", "tokens", "loss_tokens")  # the microbatch count is placement, not compared
_NUM = r"((?i:[+-]?(?:nan|inf(?:inity)?))|[0-9.eE+-]+)"


def _metric(name, text):
    """The float logged for `name` in a metrics dict line, None when absent or unparseable."""
    m = re.search(re.escape(name) + r".: " + _NUM, text)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def read_steps(log):
    """(loss, counts, perf) per step: {step: (loss, grad_norm)}, {rollout: probe counts}, {step: (tokens, step_time)}
    with the logged tokens = actor_train_tok_per_s x actor_train_time (None without actor_train_time). A loss that
    cannot be parsed counts as missing; an unparseable probe line is recorded as {"error": ...}."""
    loss, counts, perf = {}, {}, {}
    with open(log, errors="ignore") as f:
        for line in f:
            m = re.search(r"PARITY_COUNTS rollout (\d+) step \d+: (\{.*\})", line)
            if m:
                try:
                    counts[int(m.group(1))] = json.loads(m.group(2))
                except ValueError as e:
                    counts[int(m.group(1))] = {"error": "unparseable PARITY_COUNTS line: %s" % e}
            m = re.search(r" step (\d+): \{(.*)", line)
            if m and "train/loss" in m.group(2):
                lo = _metric("train/loss", m.group(2))
                if lo is not None:
                    loss[int(m.group(1))] = (lo, _metric("train/grad_norm", m.group(2)))
            m = re.search(r"perf (\d+): \{(.*)", line)
            if m:
                tk = _metric("perf/actor_train_tok_per_s", m.group(2))
                tt = _metric("perf/actor_train_time", m.group(2))
                stt = _metric("perf/step_time", m.group(2))
                if tk is not None and stt is not None:
                    perf[int(m.group(1))] = (tk * tt if tt is not None else None, stt)
    return loss, counts, perf


def _span(steps):
    """[3, 4, 5, 9] -> '3-5, 9'."""
    out = []
    for t in steps:
        if out and out[-1][1] == t - 1:
            out[-1][1] = t
        else:
            out.append([t, t])
    return ", ".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in out)


def bad_steps(loss):
    """Steps 0-19 whose training loss is missing or not finite."""
    return [t for t in range(STEPS) if t not in loss or not math.isfinite(loss[t][0])]


def step0(base_loss, win_loss):
    """(passed, shift, problems): shift = candidate step-0 loss / baseline's - 1; passes when |shift| <= STEP0_TOL and
    both logs hold all 20 losses, finite."""
    problems = ["%s: training loss missing or not finite at steps %s" % (side, _span(bad))
                for side, bad in (("baseline", bad_steps(base_loss)), ("candidate", bad_steps(win_loss))) if bad]
    ok0 = all(0 in d and math.isfinite(d[0][0]) for d in (base_loss, win_loss))
    shift = win_loss[0][0] / base_loss[0][0] - 1 if ok0 and base_loss[0][0] else None
    if ok0 and shift is None:
        problems.append("baseline step-0 loss is 0")
    return not problems and abs(shift) <= STEP0_TOL, shift, problems


def deviation(base_loss, win_loss):
    """The candidate's mean relative loss deviation from the baseline over steps 10-19 (None unless all are finite)."""
    ts, bad = range(REVIEW_START, STEPS), set(bad_steps(base_loss)) | set(bad_steps(win_loss))
    if bad & set(ts) or any(not base_loss[t][0] for t in ts):
        return None
    return st.mean(win_loss[t][0] / base_loss[t][0] - 1 for t in ts)


def convergence(base_loss, win_loss, common):
    """(result, error): Megatron-Bridge validate_convergence on the 20 common steps."""
    if len(common) < STEPS:
        return None, "loss for %d of the %d steps in both logs" % (len(common), STEPS)
    nonfinite = [t for t in common if not (math.isfinite(base_loss[t][0]) and math.isfinite(win_loss[t][0]))]
    if nonfinite:
        return None, "loss not finite at steps %s" % _span(nonfinite)
    with open(MB_PATH, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != MB_SHA256:
            return None, "mbridge_convergence.py is not the vendored copy (sha256 %s)" % MB_SHA256[:12]
    import logging

    import mbridge_convergence as mb
    import numpy as np
    lg = logging.getLogger("mbridge_convergence")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    r = mb.validate_convergence(np.array([win_loss[t][0] for t in common]), np.array([base_loss[t][0] for t in common]),
                                [str(t) for t in common], lg, config=dict(MB_CONFIG))
    metrics = {k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "item") else v) for k, v in r["metrics"].items()}
    return {"passed": bool(r["passed"]), "failed_metrics": list(r["failed_metrics"]), "metrics": metrics}, None


def _rel(a, b):
    return abs(a - b) / b if b else None


def _tput(perf, counts, gpus):
    """Tokens / logged step time per GPU over the default measured window, steps 5-19: probe tokens when every
    window step has probe counts, else the logged tokens."""
    ts = [t for t in range(WINDOW_START, STEPS) if t in perf]
    if len(ts) < STEPS - WINDOW_START or not gpus:
        return None
    if all(t in counts for t in ts):
        tokens = [counts[t]["tokens"] for t in ts]
    elif all(perf[t][0] is not None for t in ts):
        tokens = [perf[t][0] for t in ts]
    else:
        return None
    return sum(tokens) / sum(perf[t][1] for t in ts) / gpus


def pair_image(pair_dir):
    """The image both arms ran, from images.json, when both name the same digest-pinned reference; else None."""
    path = os.path.join(pair_dir, "images.json")
    try:
        with open(path) as f:
            refs = json.load(f)
        base, win = refs.get("base"), refs.get("win")
    except (OSError, ValueError, AttributeError):
        return None
    pinned = isinstance(base, str) and re.search(r"@sha256:[0-9a-f]{64}$", base)
    return base if pinned and base == win else None


def _layer1(base_args, win_args, image=None):
    try:
        spec = frozen_args.load()
    except frozen_args.SpecError as e:
        return {"L1": False, "errors": [str(e)], "changes": {}, "frozen": {}}, None
    return layer1.verdict(base_args, win_args, spec, image), spec


def verdict(pair_dir):
    base_log, win_log = os.path.join(pair_dir, "base.log"), os.path.join(pair_dir, "win.log")
    image = pair_image(pair_dir)
    l1, spec = _layer1(frozen_args.resolved_args(base_log), frozen_args.resolved_args(win_log), image)
    (lb, cb, pb), (lw, cw, pw) = read_steps(base_log), read_steps(win_log)
    common = sorted(set(lb) & set(lw) & set(range(STEPS)))
    cnt_steps = sorted(set(cb) & set(cw))
    s0_ok, shift, s0_problems = step0(lb, lw)
    review = [dict(kind="unknown", **u) for u in l1.get("unknown", [])]

    def guarded(what, fn):  # review-only: an error is recorded in `review` and never changes the verdict
        try:
            return fn()
        except Exception as e:
            review.append({"kind": "review_error", "what": what, "error": "%s: %s" % (type(e).__name__, e)})
            return None
    try:
        conv, curve_error = convergence(lb, lw, common)
    except Exception as e:
        conv, curve_error = None, "the check raised %s: %s" % (type(e).__name__, e)
    if curve_error or not conv["passed"]:
        review.append({"kind": "curve_check", "error": curve_error,
                       "failed_metrics": conv["failed_metrics"] if conv else None})

    def read_gpus():
        path = os.path.join(pair_dir, "gpus.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return dict(json.load(f))
    gpus = guarded("gpus.json", read_gpus) or {}
    ch = l1["changes"]
    finite = not set(common) & (set(bad_steps(lb)) | set(bad_steps(lw)))
    orientation = {
        "deviation_10_19": guarded("steps 10-19 deviation", lambda: deviation(lb, lw)),
        "counts_identical": guarded("probe counts", lambda: bool(cnt_steps) and len(cnt_steps) >= len(common) and all(
            cb[s].get(k) == cw[s].get(k) for s in cnt_steps for k in COUNT_KEYS)),
        "step0_gradnorm_rel": guarded("step-0 grad norm", lambda: _rel(lw[0][1], lb[0][1])
                                      if 0 in common and lb[0][1] and lw[0][1] is not None else None),
        "gap20_rel": guarded("20-step loss gap", lambda: abs(st.mean(lw[t][0] - lb[t][0] for t in common))
                             / st.mean(lb[t][0] for t in common) if common and finite else None),
        "base_tok_s_gpu": guarded("baseline throughput", lambda: _tput(pb, cb, gpus.get("base"))),
        "win_tok_s_gpu": guarded("candidate throughput", lambda: _tput(pw, cw, gpus.get("win"))),
    }
    return {
        "pair": os.path.basename(os.path.abspath(pair_dir)), "steps": len(common), "count_steps": len(cnt_steps),
        "L1": l1["L1"], "L1_errors": l1["errors"], "L1_objective": ch.get("objective", {}),
        "L1_defect": ch.get("defect", {}), "L1_defect_why": ch.get("defect_why", {}),
        "L1_ct_failed": ch.get("ct_failed", {}), "L1_ct_failed_why": ch.get("ct_failed_why", {}),
        "L1_derived_violation": ch.get("derived_violation", {}),
        "L1_frozen": l1["frozen"], "frozen_spec": spec and spec["path"], "allowlisted": ch.get("allowlisted", {}),
        "image": image, "setup_dir": l1.get("setup_dir"), "image_dir": l1.get("image_dir"),
        "derived": ch.get("derived", {}),
        "STEP0": s0_ok, "step0_shift": shift, "step0_tol": STEP0_TOL, "step0_problems": s0_problems,
        "curve_check": conv, "curve_error": curve_error, "review": review,
        "PASS": l1["L1"] and s0_ok, **orientation,
    }


def find_pairs(root):
    def is_pair(d):
        return os.path.isfile(os.path.join(d, "base.log")) and os.path.isfile(os.path.join(d, "win.log"))
    if is_pair(root):
        return [root]
    subdirs = sorted(os.path.join(root, d) for d in os.listdir(root)) if os.path.isdir(root) else []
    return [d for d in subdirs if os.path.isdir(d) and is_pair(d)]


def _pct(x, sign=""):
    return "-" if x is None else ("%" + sign + ".3f%%") % (100 * x)


def report(r):
    """A few readable lines for one pair: the decision, then what failed, then the review lines."""
    if "error" in r:
        return "%s: ERROR %s" % (r["pair"], r["error"])
    out = ["%s: %s  (L1 %s, step-0 shift %s, limit %g%%; %d loss steps, %d count steps)" % (
        r["pair"], "PASS" if r["PASS"] else "FAIL", "pass" if r["L1"] else "FAIL", _pct(r["step0_shift"], "+"),
        100 * r["step0_tol"], r["steps"], r["count_steps"])]

    def moves(changes, why=None):
        lines = []
        for n, (a, b) in changes.items():
            w = (why or {}).get(n)
            if isinstance(w, dict):
                via = [k for k in w.get("changed", []) if k != n]
                w = "%s; evidence: %s%s" % (str(w["reason"]).rstrip("."), w["evidence"],
                                            "; reached by changing %s" % ", ".join(via) if via else "")
            lines.append("%s %s -> %s%s" % (n, a, b, " (%s)" % w if w else ""))
        return lines
    l1 = [("errors", r["L1_errors"]), ("objective changes", moves(r["L1_objective"])),
          ("known defects", moves(r["L1_defect"], r["L1_defect_why"])),
          ("failed component tests", moves(r["L1_ct_failed"], r["L1_ct_failed_why"])),
          ("derived violations", moves(r["L1_derived_violation"])),
          ("frozen violations", ["%s (%s)" % (v["arg"], k) for k in ("base", "win") for v in r["L1_frozen"].get(k, [])])]
    out += ["  L1 %s: %s" % (name, "; ".join(map(str, items))) for name, items in l1 if items]
    out += ["  step-0 check: %s" % p for p in r["step0_problems"]]
    out += ["  review: unknown %s %s -> %s (%s)" % (u["name"], u["from"], u["to"], u["reason"])
            for u in r["review"] if u["kind"] == "unknown"]
    out += ["  review: %s not computed (%s)" % (u["what"], u["error"])
            for u in r["review"] if u["kind"] == "review_error"]
    c = r["curve_check"]
    curve = ("not scored (%s)" % r["curve_error"] if r["curve_error"] else "pass" if c["passed"]
             else "FAIL (%s)" % ", ".join(c["failed_metrics"]))
    out.append("  review: loss curve %s; steps 10-19 mean deviation %s" % (curve, _pct(r["deviation_10_19"], "+")))
    probe = "off" if not r["count_steps"] else "identical" if r["counts_identical"] else "DIFFER or missing"
    out.append("  probe counts %s; step-0 grad norm %s; 20-step loss gap %s" % (
        probe, _pct(r["step0_gradnorm_rel"]), _pct(r["gap20_rel"])))
    if r["base_tok_s_gpu"] and r["win_tok_s_gpu"]:
        out.append("  tok/s/GPU (steps 5-19, orientation only): base %.1f, candidate %.1f"
                   % (r["base_tok_s_gpu"], r["win_tok_s_gpu"]))
    return "\n".join(out)


def main(argv=None):
    campaign.require_settings()  # the campaign's config.env decides its settings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a pair directory, or a directory whose subdirectories are pair directories")
    ap.add_argument("--json", metavar="FILE", help="also write every pair's result as JSON")
    a = ap.parse_args(argv)
    problems = campaign.integrity_check("parity_verdict.py (before a recipe ships)", teardown=True, dataset=True)
    lasting = [p for p in problems if not p.get("transient")]
    if lasting:
        print("INTEGRITY VIOLATION: no recipe ships; the campaign has failed: %s; %s"
              % (campaign.describe(lasting), campaign.teardown_note()), file=sys.stderr)
        return 3
    if problems:
        print("cannot verify the campaign's dataset now, so nothing is scored; retry (exit 4): %s"
              % campaign.describe(problems), file=sys.stderr)
        return 4
    pairs = find_pairs(a.root)
    if not pairs:
        print("no pair directory (base.log + win.log) at or under %s" % a.root, file=sys.stderr)
        return 2
    results = []
    for p in pairs:
        try:
            results.append(verdict(p))
        except Exception as e:  # an unscorable pair is reported and fails; the others are still scored
            results.append({"pair": os.path.basename(os.path.abspath(p)), "error": "%s: %s" % (type(e).__name__, e)})
    for r in results:
        try:
            print(report(r))
        except Exception as e:  # the report is for reading; the verdict stands without it
            print("%s: %s  (report failed: %s: %s)" % (r.get("pair"), "PASS" if r.get("PASS") else "FAIL",
                                                      type(e).__name__, e))
    if a.json:
        try:
            with open(a.json, "w") as f:
                json.dump(results, f, indent=1, default=str)
        except Exception as e:  # the exit code is the verdict's; a lost copy of it does not change it
            print("cannot write %s: %s: %s" % (a.json, type(e).__name__, e), file=sys.stderr)
    return 0 if all(r.get("PASS") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
