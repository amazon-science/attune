#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make a correctness-check pair from two finished benchmark runs, so the check needs no new launch.

    pair_from_runs.py --base-log <baseline train.log> --base-yaml <its run.yaml> \\
                      --win-log <candidate train.log> --win-yaml <its run.yaml> --pair <dir>
                      [--allow-image-change] [--image REF]

The baseline's unprofiled benchmark run is the reference side and the candidate's own unprofiled benchmark run is the
candidate side: both are runs of at least 20 steps on the same rows, order, seed and GBS (the workload contract of
make_run_yaml.py --benchmark), so their loss curves over steps 0-19 compare directly. The pair is refused unless both
logs hold Megatron's resolved-argument dump with num_rollout and every training loss of steps 0-19, neither run is
profiled, both ran the same --num-rollout, and both YAMLs name one and the same training image. The training image is
that of the Ray container (the first container) of every RayJob head and worker group; sidecars and the submitter pod
do not count. When a run's groups name several, --image says which one trained. --allow-image-change accepts a
candidate that changes the image, or an ambiguous one recorded as unknown; layer 1 then applies no image rules.
A profiled run changes the run length, which slime's LR and weight-decay schedules can depend on, so it is never a
side: a profiled candidate gets one unprofiled confirm run, and that run is paired. When the campaign's runs are
shorter than 20 steps, both sides are confirm runs (make_run_yaml.py --benchmark --confirm), which have 20.

Everything is checked before anything is written. The pair directory gets base.log, win.log, gpus.json (each run's
GPU count) and images.json (each run's image), replacing earlier copies. Then run parity_verdict.py <dir>.
Exit 0 on a written pair, 1 when the runs cannot form one.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import frozen_args  # noqa: E402
import parity_verdict  # noqa: E402

PROFILER_ARGS = ("use_pytorch_profiler",)


def problems(log, side):
    """(reasons the log cannot be one side of the check, its num_rollout)."""
    if not Path(log).is_file():
        return [f"{side}: no log at {log}"], None
    out = []
    args = frozen_args.resolved_args(log)
    if not args:
        out.append(f"{side}: no resolved-argument dump in {log}")
    elif not args.get("num_rollout", "").isdigit():
        out.append(f"{side}: the argument dump has no numeric num_rollout ({args.get('num_rollout')!r})")
    if any(args.get(a) == "True" for a in PROFILER_ARGS):
        out.append(f"{side}: the run is profiled; run it once unprofiled and pass that run")
    loss, _, _ = parity_verdict.read_steps(log)
    missing = [s for s in range(parity_verdict.STEPS) if s not in loss]
    n = args.get("num_rollout", "")
    if n.isdigit() and int(n) < parity_verdict.STEPS:
        out.append(f"{side}: the run has {n} steps, fewer than the {parity_verdict.STEPS} the check scores; pair its "
                   f"{parity_verdict.STEPS}-step confirm run (make_run_yaml.py --benchmark --confirm)")
    elif missing:
        out.append(f"{side}: missing the training loss of steps {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return out, args.get("num_rollout")


def training_images(docs):
    """The images of the Ray containers of every RayJob head and worker group: KubeRay runs Ray in each group's first
    container, so sidecars and the submitter pod are ignored."""
    found = set()
    for job in (d for d in docs if isinstance(d, dict) and d.get("kind") == "RayJob"):
        rcs = (job.get("spec") or {}).get("rayClusterSpec") or {}
        for g in [rcs.get("headGroupSpec") or {}] + list(rcs.get("workerGroupSpecs") or []):
            cs = ((g.get("template") or {}).get("spec") or {}).get("containers") or []
            if cs and isinstance(cs[0], dict) and isinstance(cs[0].get("image"), str):
                found.add(cs[0]["image"])
    return sorted(found)


def run_meta(yaml_path):
    """(GPU count, training images) of a run YAML."""
    import yaml
    from make_run_yaml import count_gpus
    text = Path(yaml_path).read_text()
    return count_gpus(text), training_images([d for d in yaml.safe_load_all(text) if d is not None])


def check(a):
    """(errors, gpus, images) of the two runs."""
    errs, rollouts, gpus, images = [], {}, {}, {}
    for side, log, y in (("base", a.base_log, a.base_yaml), ("win", a.win_log, a.win_yaml)):
        e, rollouts[side] = problems(log, side)
        errs += e
        if not Path(y).is_file():
            errs.append(f"{side}: no run YAML at {y}")
            continue
        try:
            gpus[side], found = run_meta(y)
        except Exception as ex:   # an unreadable manifest is a refusal, not a traceback
            errs.append(f"{side}: cannot read {y}: {ex}")
            continue
        if not gpus[side]:
            errs.append(f"{side}: no GPUs in {y}")
        if len(found) == 1:
            images[side] = found[0]
        elif not found:
            errs.append(f"{side}: {y} has no RayJob head or worker container image")
        elif a.image in found:
            images[side] = a.image
        elif a.allow_image_change:
            images[side] = None   # unknown: layer 1 then applies no image rules
            print(f"note: {side}: {y} runs several training images {found}; recorded as unknown", file=sys.stderr)
        else:
            errs.append(f"{side}: {y} runs several training images {found}; name the training image with --image, "
                        "or pass --allow-image-change")
    if not errs and rollouts["base"] != rollouts["win"]:
        errs.append(f"--num-rollout differs ({rollouts['base']} vs {rollouts['win']}): both arms need the same run length")
    if not errs and images["base"] != images["win"] and not a.allow_image_change:
        errs.append(f"the runs used different images ({images['base']} vs {images['win']}); pass "
                    "--allow-image-change when the candidate's recipe changes the image")
    return errs, gpus, images


def main(argv=None):
    import campaign

    campaign.require_settings()  # the campaign's config.env decides its settings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-log", required=True, help="train.log of the baseline's unprofiled benchmark run")
    ap.add_argument("--base-yaml", required=True, help="the baseline run's RayJob YAML (GPU count and image)")
    ap.add_argument("--win-log", required=True, help="train.log of the candidate's unprofiled benchmark run")
    ap.add_argument("--win-yaml", required=True, help="the candidate run's RayJob YAML (GPU count and image)")
    ap.add_argument("--pair", required=True, help="pair directory to write")
    ap.add_argument("--allow-image-change", action="store_true",
                    help="accept runs on different images (the candidate's recipe changes the image), or a run whose "
                         "training image is ambiguous (recorded as unknown)")
    ap.add_argument("--image", metavar="REF",
                    help="the training image, for a run whose head and worker groups name several (must be one of "
                         "them)")
    a = ap.parse_args(argv)
    errs, gpus, images = check(a)
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    pair = Path(a.pair)
    pair.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.base_log, pair / "base.log")
    shutil.copyfile(a.win_log, pair / "win.log")
    (pair / "gpus.json").write_text(json.dumps(gpus) + "\n")
    (pair / "images.json").write_text(json.dumps(images) + "\n")
    print(f"Wrote {pair} (base.log, win.log, gpus.json, images.json); next: parity_verdict.py {pair}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
