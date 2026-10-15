#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make one arm of the correctness check from a ranked run's RayJob YAML.

    make_parity_arm.py --base <run.yaml> --run-id <arm_run_id> --out <arm.yaml> [--steps 20] [--pair <dir> --arm base|win]

Run it twice: once on the baseline's YAML and once on the candidate's. Each arm is the source YAML with only:
  - every K8s name / shared-volume path renamed to <arm_run_id>, which must pass the same run-id and prefix checks as
    make_run_yaml.py, and the owner label on the RayJob and its ConfigMaps,
  - `--num-rollout` set to --steps (20: steps 0-19), wherever the flag appears (inline or in a ConfigMap script),
  - the parity probe enabled on every pod through the hook chain (hook_chain.py: one sitecustomize loader,
    ConfigMap keys + read-only mount, PYTHONPATH prefix in the container env, runtimeEnvYAML and
    `export PYTHONPATH=`, and parity_probe added to $ATTUNE_HOOKS).
Data, seed, GBS, image and every training flag stay as in the source, so the two arms differ only by the recipe.
With --pair and --arm, the arm's image (the one image of its containers, or null when they differ) is recorded under
that key in <pair>/images.json, which parity_verdict.py reads for the image's layer-1 rules.
Then run parity_verdict.py on a directory holding base.log, win.log and (optionally) gpus.json and images.json.
"""
import json
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import hook_chain  # noqa: E402
from make_run_yaml import apply_name_suffix, check_run_id, infer_base_run_id, stamp_owner  # noqa: E402

ARM_HOOKS = ["parity_probe"]


def _walk_strings(node, fn):
    """Apply fn to every string scalar in a parsed YAML tree (in place); return the number of edits."""
    n = 0
    if isinstance(node, dict):
        for k in list(node):
            v = node[k]
            if isinstance(v, str):
                node[k], c = fn(v)
                n += c
            else:
                n += _walk_strings(v, fn)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str):
                node[i], c = fn(v)
                n += c
            else:
                n += _walk_strings(v, fn)
    return n


def arm_image(docs):
    """The one image of every pod's containers (init containers aside), or None when they name more than one."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for c in node.get("containers") or []:
                if isinstance(c, dict) and isinstance(c.get("image"), str):
                    found.add(c["image"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(docs)
    return found.pop() if len(found) == 1 else None


def record_image(pair_dir, arm, image):
    """Set images.json[arm] = image in pair_dir, keeping the other arm's entry."""
    path = Path(pair_dir) / "images.json"
    refs = json.loads(path.read_text()) if path.exists() else {}
    refs[arm] = image
    Path(pair_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(refs, indent=1) + "\n")


def make_arm(text, run_id, steps):
    import yaml
    check_run_id(run_id)
    base = infer_base_run_id(text)
    text = apply_name_suffix(text, base, run_id)
    docs = [d for d in yaml.safe_load_all(text) if d is not None]
    rollout_re = re.compile(r"(?<!\S)--num-rollout(\s+)\d+")
    n_roll = _walk_strings(docs, lambda s: rollout_re.subn(lambda m: "--num-rollout%s%d" % (m.group(1), steps), s))
    if n_roll == 0:
        raise RuntimeError("--num-rollout not found in the YAML or its ConfigMaps; add it to the training command")
    info = hook_chain.install(docs, run_id, enable=ARM_HOOKS)
    summary = {"num_rollout_edits": n_roll, "export_pythonpath_edits": info["export_edits"],
               "containers": info["containers"], "configmap": info["configmap"], "chain_sha256": info["chain_sha256"],
               "image": arm_image(docs)}
    return hook_chain.dump_yaml(stamp_owner(docs)), base, summary


def main():
    import campaign

    campaign.require_settings()  # the campaign's config.env decides its settings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="RayJob YAML of the ranked run (baseline or candidate)")
    ap.add_argument("--run-id", required=True, help="run id of this arm (every K8s name and shared-volume path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20, help="training steps (--num-rollout); the check uses 20")
    ap.add_argument("--pair", help="pair directory whose images.json records this arm's image")
    ap.add_argument("--arm", choices=("base", "win"), help="this arm's key in images.json (with --pair)")
    a = ap.parse_args()
    if bool(a.pair) != bool(a.arm):
        ap.error("--pair and --arm go together")
    out, base, info = make_arm(Path(a.base).read_text(), a.run_id, a.steps)
    Path(a.out).write_text(out)
    print("Wrote %s (from %s)" % (a.out, base))
    print("  --num-rollout %d (%d edit(s)); probe on %d container(s) via hook chain %s in ConfigMap %s; "
          "%d `export PYTHONPATH=` edit(s)" % (a.steps, info["num_rollout_edits"], info["containers"],
                                               info["chain_sha256"][:12], info["configmap"],
                                               info["export_pythonpath_edits"]))
    if a.pair:
        record_image(a.pair, a.arm, info["image"])
        print("  image %s recorded as %s in %s/images.json" % (info["image"], a.arm, a.pair))
    print("  before launch: delete a stale <run_id>/checkpoints/ dir; track with target_steps %d" % (a.steps - 1))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        sys.exit(1)
