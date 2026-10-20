#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy a campaign's proposed lessons to the curation inbox; launch_campaign.sh runs it when a session ends.

    lessons.py export <control_dir>                            # exit 0 exported or nothing to export, 1 copy failed
    lessons.py check-inbox <inbox> <campaign_dir> [<tree> ...]  # exit 2 with the reason when the inbox is refused

A campaign never edits the lever catalog: it writes the lessons it proposes to <project_folder>/LESSONS.md (SKILL.md
L4c). The launcher runs `export` only when the session-end integrity check passes; run by hand, it also exports
nothing from a campaign with a violation record. With $ATTUNE_LESSONS_INBOX set, it copies each project folder's file
to $ATTUNE_LESSONS_INBOX/<campaign id>/<folder>/LESSONS.md and writes manifest.json beside the folders: campaign id,
model, GPU, image and framework (from the campaign's config.env), catalog version (from the integrity record: the
sealed levers.md hash, the source tree's commit and whether it had uncommitted changes), campaign directory, time,
and each copy's folder, source and sha256. A later session replaces them. Unset, the lessons stay in the campaign
folder and it says so. The inbox must be absolute and outside the campaign directory and every ATTUNE tree's skill/
and harness/, which the launcher copies into each campaign. Curation is manual (harness/CURATION.md).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import integrity  # noqa: E402

NAME = "LESSONS.md"
INBOX = "ATTUNE_LESSONS_INBOX"
MANIFEST = "manifest.json"


def _write(path, data):
    tmp = os.path.join(os.path.dirname(path), ".%s.tmp" % os.path.basename(path))
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _recorded_source(campaign):
    try:
        return integrity._load(os.path.join(campaign, integrity.CONTROL_NAME, integrity.RECORD)).get("source")
    except (OSError, ValueError):
        return None


def inbox_refusal(inbox, campaign, trees=()):
    """Why `inbox` cannot collect the lessons of the campaign in `campaign` ("" when it can); `trees` and the tree
    the campaign's record names are ATTUNE trees."""
    if not os.path.isabs(inbox):
        return "%s is not an absolute path" % inbox
    real, campaign = os.path.realpath(inbox), os.path.realpath(campaign)
    if integrity._under(real, campaign):
        return "%s is the campaign directory %s or inside it" % (inbox, campaign)
    for tree in [t for t in list(trees) + [_recorded_source(campaign)] if t]:
        for d in integrity.SOURCE_DIRS:
            if integrity._under(real, os.path.realpath(os.path.join(tree, d))):
                return "%s is inside the ATTUNE tree's %s/ (%s), which the launcher copies into every campaign" % (
                    inbox, d, tree)
    return ""


def export(control, env=None):
    """One line on what happened to the campaign's lessons; raises when the copy fails."""
    env = os.environ if env is None else env
    p = integrity.Paths(control)
    cfg = integrity.read_config(p.config)
    campaign = os.path.realpath(cfg.get("CAMPAIGN_DIR") or cfg["AUTOTUNE_PROJECT_ROOT"])
    files = sorted(glob.glob(os.path.join(glob.escape(campaign), "*", NAME)))
    if not files:
        return "lessons: none written (no <project_folder>/%s)" % NAME
    where = ", ".join(files)
    if os.path.exists(p.violation):
        return "lessons: not exported, the campaign failed its integrity check; they stay in %s" % where
    inbox = env.get(INBOX)
    if not inbox:
        return "lessons: %s is unset, so they stay in the campaign folder: %s" % (INBOX, where)
    why = inbox_refusal(inbox, campaign)
    if why:
        raise ValueError("%s refused: %s" % (INBOX, why))
    cid = cfg.get("ATTUNE_CAMPAIGN_ID")
    if not cid:
        raise ValueError("%s has no ATTUNE_CAMPAIGN_ID" % p.config)
    rec = integrity._load(p.record)
    out = os.path.join(inbox, cid)
    copies = []
    for f in files:
        folder = os.path.basename(os.path.dirname(f))
        with open(f, "rb") as fh:
            data = fh.read()
        os.makedirs(os.path.join(out, folder), exist_ok=True)
        _write(os.path.join(out, folder, NAME), data)
        copies.append({"folder": folder, "source": f, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"campaign_id": cid, "model": cfg.get("ATTUNE_MODEL_NAME") or None,
                "gpu": cfg.get("ATTUNE_GPU_NAME") or None, "image": cfg.get("TRAINING_IMAGE") or None,
                "framework": cfg.get("ATTUNE_FRAMEWORK") or None,
                "catalog_version": {"levers_sha256": rec["files"].get(p.levers), "commit": rec.get("source_commit"),
                                    "dirty": rec.get("source_dirty")},
                "campaign_dir": campaign, "lessons": copies,
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        with open(os.path.join(out, MANIFEST), encoding="utf-8") as fh:
            before = {c.get("folder") for c in json.load(fh).get("lessons", [])}
    except (OSError, ValueError, AttributeError):
        before = set()
    _write(os.path.join(out, MANIFEST), (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
    for folder in before - {c["folder"] for c in copies}:  # a folder whose LESSONS.md is gone since the last export
        if isinstance(folder, str) and folder not in ("", ".", "..") and folder == os.path.basename(folder):
            try:
                os.remove(os.path.join(out, folder, NAME))
                os.rmdir(os.path.join(out, folder))
            except OSError:
                pass
    return "lessons: exported %s to %s/" % (where, out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "export":
        try:
            print(export(argv[1]))
        except Exception as e:
            print("lessons: not exported: %s: %s" % (type(e).__name__, e), file=sys.stderr)
            return 1
        return 0
    if len(argv) >= 3 and argv[0] == "check-inbox":
        why = inbox_refusal(argv[1], argv[2], argv[3:])
        if why:
            print("lessons: %s refused: %s" % (INBOX, why), file=sys.stderr)
            return 2
        return 0
    print("usage: lessons.py export <control_dir> | lessons.py check-inbox <inbox> <campaign_dir> [<tree> ...]",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
