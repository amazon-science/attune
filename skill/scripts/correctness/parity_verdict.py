#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness check of a candidate recipe against its baseline (references/measurement.md §4, PROTOCOL.md here).

    parity_verdict.py <pair_dir | root_of_pair_dirs> [--json out.json]

A pair directory holds base.log and win.log, the train.log of the baseline's and the candidate's unprofiled runs (their
own benchmark runs, paired by pair_from_runs.py; or 20-step arms made by make_parity_arm.py, which load the parity
probe), and optionally gpus.json {"base": N, "win": M} and images.json {"base": ref, "win": ref}, the image each ran.
  L1 layer1.py on the two resolved-argument dumps, with the frozen-arguments file of $AUTOTUNE_FROZEN_ARGS (its
     directory is the setup's layer-1 directory) and, when images.json names one image pinned by digest for both
     arms, that image's rules. An unreadable frozen-arguments file fails L1.
  L2 NVIDIA Megatron-Bridge's convergence check, vendored verbatim in mbridge_convergence.py
     (scripts/performance/utils/evaluate.py:145-329 at c3c974d4f) and run with the thresholds its pipeline passes
     (argument_parser.py:1092-1125 defaults, via setup_experiment.py:1107-1116) on the candidate's per-step training
     loss (steps 0-19, all 20 present) against the baseline's: the first 20% of steps skipped, point-wise relative
     tolerance 10% / 5% / 2% for loss > 2.0 / 0.5-2.0 / < 0.5 with at most 10% of points outside, final-loss
     tolerance 5%, correlation >= 0.95 and a 3-sigma outlier check. The check is tuned to avoid false rejects: it
     catches gross breakage and can miss subtle faults.
  Reported, not gating: whether the probe's per-step rows, tokens and loss-mask tokens are identical (when the probe
     ran), the step-0 loss and grad-norm relative differences, the 20-step mean loss gap, and tokens / logged step
     time per GPU over steps 5-19 (probe tokens when the probe ran, else the logged tokens; with gpus.json) for
     orientation.
PASS = L1 and L2. Exit 0 when every pair passes, 1 when any pair fails or cannot be scored, 2 when no pair is found,
3 when the campaign fails its integrity check (harness/integrity.py, which also hashes the dataset again through the
log pod): then nothing is scored and no recipe ships, 4 when the log pod cannot be reached to hash the dataset: nothing
is scored, retry.
"""
import argparse
import hashlib
import json
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
MB_SHA256 = "0897172c6a3d8654177d080e725209baa45bbc5239538e484269057b99e0d018"  # layer 2: the vendored file
MB_CONFIG = {"skip_first_percent_loss": 0.20, "final_loss_tolerance": 0.05}  # the pipeline values that differ from the function's
STEPS = 20
WINDOW_START = 5   # orientation throughput only: the default measured window 5-19
COUNT_KEYS = ("rows", "tokens", "loss_tokens")  # the microbatch count is placement, not compared
_NUM = r"([0-9.eE+-]+)"


def read_steps(log):
    """(loss, counts, perf) per step: {step: (loss, grad_norm)}, {rollout: probe counts}, {step: (tokens, step_time)}
    with the logged tokens = actor_train_tok_per_s x actor_train_time (None without actor_train_time)."""
    loss, counts, perf = {}, {}, {}
    with open(log, errors="ignore") as f:
        for line in f:
            m = re.search(r"PARITY_COUNTS rollout (\d+) step \d+: (\{.*\})", line)
            if m:
                counts[int(m.group(1))] = json.loads(m.group(2))
            m = re.search(r" step (\d+): \{(.*)", line)
            if m and "train/loss" in m.group(2):
                lo = re.search(r"train/loss.: " + _NUM, m.group(2))
                gn = re.search(r"train/grad_norm.: " + _NUM, m.group(2))
                loss[int(m.group(1))] = (float(lo.group(1)), float(gn.group(1)) if gn else None)
            m = re.search(r"perf (\d+): \{(.*)", line)
            if m:
                tk = re.search(r"perf/actor_train_tok_per_s.: " + _NUM, m.group(2))
                tt = re.search(r"perf/actor_train_time.: " + _NUM, m.group(2))
                stt = re.search(r"perf/step_time.: " + _NUM, m.group(2))
                if tk and stt:
                    tokens = float(tk.group(1)) * float(tt.group(1)) if tt else None
                    perf[int(m.group(1))] = (tokens, float(stt.group(1)))
    return loss, counts, perf


def convergence(base_loss, win_loss, common):
    """(result, error): Megatron-Bridge validate_convergence on the 20 common steps."""
    if len(common) < STEPS:
        return None, "loss for %d of the %d steps in both logs" % (len(common), STEPS)
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
    conv, l2_error = convergence(lb, lw, common)
    gpus = {}
    if os.path.exists(os.path.join(pair_dir, "gpus.json")):
        with open(os.path.join(pair_dir, "gpus.json")) as f:
            gpus = json.load(f)
    ch = l1["changes"]
    return {
        "pair": os.path.basename(os.path.abspath(pair_dir)), "steps": len(common), "count_steps": len(cnt_steps),
        "L1": l1["L1"], "L1_errors": l1["errors"], "L1_numeric": ch.get("numeric", {}),
        "L1_allowlist_violation": ch.get("allowlist_violation", {}), "L1_derived_violation": ch.get("derived_violation", {}),
        "L1_frozen": l1["frozen"], "frozen_spec": spec and spec["path"], "allowlisted": ch.get("allowlisted", {}),
        "image": image, "setup_dir": l1.get("setup_dir"), "image_dir": l1.get("image_dir"),
        "derived": ch.get("derived", {}),
        "L2": bool(conv and conv["passed"]), "L2_error": l2_error, "convergence": conv,
        "counts_identical": bool(cnt_steps) and len(cnt_steps) >= len(common) and all(
            cb[s].get(k) == cw[s].get(k) for s in cnt_steps for k in COUNT_KEYS),
        "step0_loss_rel": _rel(lw[0][0], lb[0][0]) if 0 in common else None,
        "step0_gradnorm_rel": _rel(lw[0][1], lb[0][1]) if 0 in common and lb[0][1] and lw[0][1] is not None else None,
        "gap20_rel": (abs(st.mean(lw[t][0] - lb[t][0] for t in common)) / st.mean(lb[t][0] for t in common)
                      if common else None),
        "PASS": l1["L1"] and bool(conv and conv["passed"]),
        "base_tok_s_gpu": _tput(pb, cb, gpus.get("base")), "win_tok_s_gpu": _tput(pw, cw, gpus.get("win")),
    }


def find_pairs(root):
    def is_pair(d):
        return os.path.isfile(os.path.join(d, "base.log")) and os.path.isfile(os.path.join(d, "win.log"))
    if is_pair(root):
        return [root]
    subdirs = sorted(os.path.join(root, d) for d in os.listdir(root)) if os.path.isdir(root) else []
    return [d for d in subdirs if os.path.isdir(d) and is_pair(d)]


def _pct(x):
    return "-" if x is None else "%.3f%%" % (100 * x)


def report(r):
    """A few readable lines for one pair."""
    if "error" in r:
        return "%s: ERROR %s" % (r["pair"], r["error"])
    out = ["%s: %s  (L1 %s, L2 %s; %d loss steps, %d count steps)" % (
        r["pair"], "PASS" if r["PASS"] else "FAIL", "pass" if r["L1"] else "FAIL", "pass" if r["L2"] else "FAIL",
        r["steps"], r["count_steps"])]
    def moves(changes):
        return ["%s %s -> %s" % (n, a, b) for n, (a, b) in changes.items()]
    l1 = [("errors", r["L1_errors"]), ("numeric changes", moves(r["L1_numeric"])),
          ("allowlist violations", moves(r["L1_allowlist_violation"])),
          ("derived violations", moves(r["L1_derived_violation"])),
          ("frozen violations", ["%s (%s)" % (v["arg"], k) for k in ("base", "win") for v in r["L1_frozen"].get(k, [])])]
    out += ["  L1 %s: %s" % (name, "; ".join(map(str, items))) for name, items in l1 if items]
    if r["L2_error"]:
        out.append("  L2 not scored: %s" % r["L2_error"])
    elif not r["L2"]:
        out.append("  L2 failed metrics: %s" % ", ".join(r["convergence"]["failed_metrics"]))
    probe = "off" if not r["count_steps"] else "identical" if r["counts_identical"] else "DIFFER or missing"
    out.append("  probe counts %s; step-0 loss %s, grad norm %s; 20-step loss gap %s" % (
        probe, _pct(r["step0_loss_rel"]),
        _pct(r["step0_gradnorm_rel"]), _pct(r["gap20_rel"])))
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
        print(report(r))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1, default=str)
    return 0 if all(r.get("PASS") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
