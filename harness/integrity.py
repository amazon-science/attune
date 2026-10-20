#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tamper detection for the files an unattended campaign must keep unchanged.

    integrity.py seal <control_dir> [--dataset PATH] [--source DIR]
                                                       # first launch: record the protected files and the dataset
    integrity.py refresh <control_dir>                 # later launch: re-record the settings and prompts it rewrote
    integrity.py check <control_dir> [--where TEXT] [--relaunch] [--teardown] [--dataset]
                                                       # exit 0 intact, 3 failed (violation record), 4 cannot verify now
    integrity.py source-check <control_dir> [--where TEXT]
                                                       # session end: warn of changes to the source tree; exit 0

The control directory, <campaign_dir>/.attune, belongs to launch_campaign.sh: it holds the campaign's copies of the
skill and the harness, the campaign's config.env and this module's record, integrity.json. The record holds the
SHA-256 of every protected file: both copies (the lever catalog references/levers.md and the scoring, measurement and
correctness tools and rules among them), config.env, the goal and resume prompts, the rendered .claude/settings.json,
and the rules directory, frozen-arguments file (with its allowlist, component-test and defect files) and calibration
file that config.env names outside the copies. For the append-only files (every <campaign_dir>/*/AUTOTUNE_LEDGER.jsonl,
$AUTOTUNE_LEDGER and the GPU-hour log) it holds the length and hash of the content seen so far. It also holds the
dataset's rows_sha256 (and file sha256), read through the log pod at launch, and the tree the copies came from
($ATTUNE_HOME) with its git commit and whether its skill/ or harness/ had uncommitted changes, which with the
catalog's hash name the catalog version a campaign ran under.

A check passes when every protected file is unchanged, no file was added to the copies outside
skill/past_tuning_projects/ (new summaries), every append-only file still begins with the content last seen and,
with --dataset (a later launch, and before a recipe ships), the dataset still hashes the same; it then records the
longer content. A mismatch, or any error while checking, writes integrity_violation.json and the campaign has failed
for good: the Stop hook tears down the tracked runs and ends the session, ledger_append.py appends nothing (and tears
them down), parity_verdict.py ships nothing and launch_campaign.sh exits 3. `write_rule` gives the PreToolUse guard
the same classification for the file-edit tools, and also refuses edits of the source tree's skill/ and harness/.
A Bash command can still write there, so each launch also records the hashes of those two directories and
`source-check` names every change when the session ends, in its output and in the record's source_checks. That is a
warning, never a failure: the copies are unaffected, and an operator may change a shared tree during a campaign.
Runs on Python 3.7+ (the hooks import it).
"""

from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager

RECORD = "integrity.json"
VIOLATION = "integrity_violation.json"
CONFIG = "config.env"
LEDGER_NAME = "AUTOTUNE_LEDGER.jsonl"
CONTROL_NAME = ".attune"
LAUNCHER_FILES = (".claude/settings.json", "goal_prompt.rendered.txt", "goal_prompt.txt", "resume_prompt.txt")
OPERATOR_FILES = ("goal_prompt.txt", "resume_prompt.txt")  # the operator may edit these between sessions
SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache"})
SOURCE_DIRS = ("skill", "harness")  # what the launcher copies from the source tree
DEFAULT_CONTROL = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
VERSION = 1


# ---- paths -----------------------------------------------------------------------------------------------------

def read_config(path):
    """The KEY=value lines of the launcher's config.env (values shell-quoted)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, eq, value = line.partition("=")
            if not eq or not key.strip():
                raise ValueError("%s line %d is not KEY=value" % (path, n))
            words = shlex.split(value)
            out[key.strip()] = words[0] if words else ""
    return out


class Paths:
    """The files of one campaign, from its control directory."""

    def __init__(self, control, campaign=None):
        j = os.path.join
        self.control = os.path.realpath(control)
        self.skill, self.harness = j(self.control, "skill"), j(self.control, "harness")
        self.levers = j(self.skill, "references", "levers.md")
        self.summaries = j(self.skill, "past_tuning_projects")
        self.config, self.record, self.violation = (j(self.control, n) for n in (CONFIG, RECORD, VIOLATION))
        self.campaign = os.path.realpath(campaign) if campaign else None

    def launcher_files(self):
        return [os.path.join(self.campaign, f) for f in LAUNCHER_FILES] if self.campaign else []


def _under(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


def _files(root):
    """Every file under root, caches skipped."""
    out = []
    for d, subdirs, names in os.walk(root):
        subdirs[:] = sorted(s for s in subdirs if s not in SKIP_DIRS)
        out += [os.path.join(d, n) for n in sorted(names) if not n.endswith(".pyc")]
    return out


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _bytes(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _seen(path):
    data = _bytes(path) or b""
    return {"length": len(data), "sha256": _sha(data)}


def _external(cfg):
    """Rules, frozen-arguments and calibration files config.env names (anywhere; inside the copies is fine too)."""
    out = []
    rules = cfg.get("AUTOTUNE_RULES_DIR")
    if rules and os.path.isdir(rules):
        out += _files(os.path.realpath(rules))
    frozen = cfg.get("AUTOTUNE_FROZEN_ARGS")
    if frozen:
        setup = os.path.dirname(os.path.realpath(frozen))
        out += [os.path.realpath(frozen)] + [os.path.join(setup, n)
                                             for n in ("allowlist.yaml", "ct_status.yaml", "defects.yaml")]
    if cfg.get("AUTOTUNE_CALIBRATION"):
        out.append(os.path.realpath(cfg["AUTOTUNE_CALIBRATION"]))
    return [f for f in out if os.path.isfile(f)]


def _append_paths(p, cfg):
    gpuh = cfg.get("AUTOTUNE_GPUH_LOG") or os.path.join(p.campaign, "gpuh.jsonl")
    out = [os.path.realpath(gpuh)] + _ledgers(p.campaign)
    if cfg.get("AUTOTUNE_LEDGER"):
        out.append(os.path.realpath(cfg["AUTOTUNE_LEDGER"]))
    return out


def _ledgers(campaign):
    return [os.path.realpath(f) for f in glob.glob(os.path.join(glob.escape(campaign), "*", LEDGER_NAME))]


# ---- the record --------------------------------------------------------------------------------------------------

@contextmanager
def _locked(control):
    fd = os.open(control, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _write_json(control, path, obj):
    """Write path (in the read-only control directory) atomically, read-only."""
    mode = os.stat(control).st_mode
    os.chmod(control, mode | 0o200)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.write("\n")
        os.chmod(tmp, 0o444)
        os.replace(tmp, path)
    finally:
        os.chmod(control, mode)


def _load(path):
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    if not isinstance(rec, dict) or rec.get("version") != VERSION:
        raise ValueError("%s is not an integrity record of version %d" % (path, VERSION))
    return rec


def _digests(paths):
    return {f: _sha(_bytes(f)) for f in paths if os.path.isfile(f)}


def _git(tree, *args):
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", tree] + list(args), capture_output=True, text=True,
                           timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return None if r.returncode else r.stdout


def _version(tree):
    """(commit, dirty) of `tree` when it is the top of a git checkout, else (None, None); dirty is whether its skill/
    or harness/ has uncommitted changes, untracked files included."""
    words = (_git(tree, "rev-parse", "--show-toplevel", "HEAD") or "").split()
    if len(words) != 2 or os.path.realpath(words[0]) != os.path.realpath(tree):
        return None, None
    status = _git(tree, "status", "--porcelain", "--", *SOURCE_DIRS)
    return words[1], None if status is None else bool(status.strip())


def _snapshot(source):
    """{path relative to `source`: sha256, or None when unreadable} of every file in its skill/ and harness/."""
    out = {}
    for d in SOURCE_DIRS:
        for f in _files(os.path.join(source, d)):
            try:
                data = _bytes(f)
            except OSError:
                data = None
            out[os.path.relpath(f, source)] = None if data is None else _sha(data)
    return out


def seal(control, dataset=None, source=None):
    """Record the protected and append-only files of the campaign whose config.env is in `control`, with `dataset`
    the dataset's hashes read through the log pod, and with `source` the tree the copies came from."""
    p = Paths(control)
    cfg = read_config(p.config)
    campaign = cfg.get("CAMPAIGN_DIR") or cfg.get("AUTOTUNE_PROJECT_ROOT")
    if not campaign:
        raise ValueError("%s sets neither CAMPAIGN_DIR nor AUTOTUNE_PROJECT_ROOT" % p.config)
    p.campaign = os.path.realpath(campaign)
    appends = _append_paths(p, cfg)
    files = [f for f in _files(p.skill) + _files(p.harness) if f not in appends]
    files += [p.config] + p.launcher_files() + _external(cfg)
    rec = {"version": VERSION, "sealed_at": _now(), "control": p.control, "campaign": p.campaign,
           "campaign_id": cfg.get("ATTUNE_CAMPAIGN_ID", ""), "trees": [p.skill, p.harness], "open": [p.summaries],
           "files": _digests(files), "append_only": {f: _seen(f) for f in appends}}
    if source:
        commit, dirty = _version(source)
        rec.update(source=os.path.realpath(source), source_commit=commit, source_dirty=dirty,
                   source_files=_snapshot(source))
    if dataset:
        digest = dataset_digest(cfg, dataset)
        if digest.get("missing"):
            raise RuntimeError("the log pod %s finds no dataset at %s" % (cfg["AUTOTUNE_LOG_POD"], dataset))
        rec["dataset"] = dict(digest, path=dataset)
    with _locked(p.control):
        _write_json(p.control, p.record, rec)
    return rec


def refresh(control):
    """Re-record the files the launcher rewrites on a later launch (settings and prompts) and the source tree's
    hashes that source_check compares against; the rest stays as sealed."""
    p = Paths(control)
    with _locked(p.control):
        rec = _load(p.record)
        p.campaign = rec["campaign"]
        for f in p.launcher_files():
            rec["files"].pop(f, None)
        rec["files"].update(_digests(p.launcher_files()))
        if rec.get("source"):
            rec["source_files"] = _snapshot(rec["source"])
        _write_json(p.control, p.record, rec)
    return rec


def source_check(control, where="source-check"):
    """The changes to the source tree's skill/ and harness/ since this launch ([] when none), appended to the record's
    source_checks; None when the record holds no hashes of a source tree."""
    p = Paths(control)
    rec = _load(p.record)
    source, before = rec.get("source"), rec.get("source_files")
    if not source or not isinstance(before, dict):
        return None
    now = _snapshot(source)
    changes = [{"path": f, "problem": "deleted" if f not in now else "modified"}
               for f in sorted(before) if f not in now or now[f] != before[f]]
    changes += [{"path": f, "problem": "added"} for f in sorted(set(now) - set(before))]
    with _locked(p.control):
        rec = _load(p.record)
        rec.setdefault("source_checks", []).append({"at": _now(), "where": where, "changes": changes})
        _write_json(p.control, p.record, rec)
    return changes


# ---- the dataset, read through the log pod -------------------------------------------------------------------------

# Runs in the log pod (stdin of `python3 -`): the file's sha256, and rows_sha256 (as skill/scripts/data/verify_rows.py,
# from each row's tokens and loss mask) when the pod has pyarrow and the file holds the row metadata.
DATASET_HASH = r"""
import hashlib, json, os, sys
from array import array
path, out = sys.argv[1], {}
if not os.path.isfile(path):
    print(json.dumps({"missing": True}))
    sys.exit(0)
digest = hashlib.sha256()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(8 << 20), b""):
        digest.update(chunk)
out["parquet_sha256"] = digest.hexdigest()
try:
    import pyarrow.parquet as pq
    rows, n = hashlib.sha256(), 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["metadata"], batch_size=256):
        for md in batch.column(0).to_pylist():
            row = hashlib.sha256(array("i", md["tokens"]).tobytes() + bytes(md["loss_mask"]))
            rows.update((row.hexdigest() + "\n").encode())
            n += 1
    out.update(rows_sha256=rows.hexdigest(), rows=n)
except Exception:
    pass
print(json.dumps(out))
"""


def dataset_digest(cfg, path):
    """{"parquet_sha256", and "rows_sha256" and "rows" when the log pod can read the rows} of the dataset at `path`,
    or {"missing": True} when the log pod answers and the file is not there; RuntimeError when the pod cannot be
    reached."""
    missing = [n for n in ("AUTOTUNE_KUBECTL_CONTEXT", "KUBECTL_NS", "AUTOTUNE_LOG_POD") if not cfg.get(n)]
    if missing:
        raise RuntimeError("the campaign's settings lack %s" % ", ".join(missing))
    pod = ["kubectl", "--context", cfg["AUTOTUNE_KUBECTL_CONTEXT"], "-n", cfg["KUBECTL_NS"], "exec",
           cfg["AUTOTUNE_LOG_POD"], "-i", "--"]
    p = subprocess.run(pod + ["python3", "-", path], input=DATASET_HASH, capture_output=True, text=True,
                       timeout=1800)
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None
    except (ValueError, IndexError):
        out = None
    if isinstance(out, dict) and (out.get("parquet_sha256") or out.get("missing")):
        return out
    q = subprocess.run(pod + ["sh", "-c", '[ -f "$1" ] || { echo missing; exit 0; }; sha256sum "$1"', "sh", path],
                       capture_output=True, text=True, timeout=1800)  # a pod without python3
    words = q.stdout.split()
    if q.returncode == 0 and words == ["missing"]:
        return {"missing": True}
    if q.returncode == 0 and words and len(words[0]) == 64:
        return {"parquet_sha256": words[0]}
    raise RuntimeError("cannot reach the log pod %s to read %s: %s" % (cfg["AUTOTUNE_LOG_POD"], path,
                                                                       (p.stderr or q.stderr).strip()[-300:]))


def _dataset_change(then, now):
    """Why the dataset no longer matches its record ("" when it does); rows_sha256 decides when both have it."""
    if now.get("missing"):
        return "the dataset is gone: the log pod finds no file at this path"
    key = "rows_sha256" if then.get("rows_sha256") and now.get("rows_sha256") else "parquet_sha256"
    if then.get(key) != now.get(key):
        return "the dataset changed: %s %s at launch, %s now" % (key, then.get(key), now.get(key))
    return ""


# ---- checking ------------------------------------------------------------------------------------------------------

def _compare(rec):
    """(problems, the append-only record after this check)."""
    problems = []
    files = rec["files"]
    for path, digest in sorted(files.items()):
        data = _bytes(path)
        if data is None:
            problems.append({"path": path, "problem": "deleted"})
        elif _sha(data) != digest:
            problems.append({"path": path, "problem": "modified"})
    known = set(files) | set(rec["append_only"])
    for tree in rec["trees"]:
        for path in _files(tree):
            if path not in known and not any(_under(path, d) for d in rec.get("open", [])):
                problems.append({"path": path, "problem": "added to a protected copy"})
    seen = {}
    for path, last in sorted(rec["append_only"].items()):
        data = _bytes(path)
        if data is None:
            if last["length"]:
                problems.append({"path": path, "problem": "deleted (append-only)"})
            seen[path] = last
        elif len(data) < last["length"]:
            problems.append({"path": path, "problem": "truncated (append-only): %d bytes, %d seen before"
                             % (len(data), last["length"])})
        elif _sha(data[:last["length"]]) != last["sha256"]:
            problems.append({"path": path, "problem": "rewritten (append-only): its first %d bytes changed"
                             % last["length"]})
        else:
            seen[path] = {"length": len(data), "sha256": _sha(data)}
    for path in _ledgers(rec["campaign"]):
        seen.setdefault(path, _seen(path))
    return problems, seen


def check(control, where="check", relaunch=False, teardown=False, dataset=False):
    """The integrity problems of the campaign in `control` ([] when intact). A passing check records the content the
    append-only files have now; a failing one writes the violation record, and every later check fails with it. Any
    error while checking is a failure too.
    relaunch: the check before a later launch, which allows the operator's edits to the goal and resume prompts.
    teardown: on a failure, tear down the campaign's tracked runs (recorded in the violation record).
    dataset: hash the dataset again through the log pod; a changed or missing dataset fails the campaign, and a pod
    that cannot be reached gives a problem marked "transient", which refuses shipping without failing it."""
    p = Paths(control)
    try:
        problems = _check(p, where, relaunch, dataset)
    except Exception as e:
        problems = [{"path": p.record, "problem": "cannot verify: %s: %s" % (type(e).__name__, e)}]
        _note_violation(p, where, problems)
    if teardown and any(not x.get("transient") for x in problems):
        _teardown(p)
    return problems


def _check(p, where, relaunch, dataset):
    if not os.path.isdir(p.control):
        return [{"path": p.control, "problem": "the control directory is missing"}]
    if os.path.exists(p.violation):
        return stored(p.control)
    found, transient = [], []
    rec = _load(p.record)
    if dataset and rec.get("dataset"):   # read before taking the lock: it can take minutes
        try:
            why = _dataset_change(rec["dataset"], dataset_digest(read_config(p.config), rec["dataset"]["path"]))
        except Exception as e:
            transient.append({"path": rec["dataset"]["path"], "transient": True,
                              "problem": "cannot reach the log pod to hash the dataset now: %s" % e})
        else:
            found += [{"path": rec["dataset"]["path"], "problem": why}] if why else []
    with _locked(p.control):
        if os.path.exists(p.violation):
            return stored(p.control)
        rec = _load(p.record)
        problems, seen = _compare(rec)
        if relaunch:
            edits = {os.path.join(rec["campaign"], f) for f in OPERATOR_FILES}
            problems = [x for x in problems if x["path"] not in edits]
        problems += found
        if problems:
            _note_violation(p, where, problems)
            return problems
        if seen != rec["append_only"]:
            rec["append_only"] = seen
            _write_json(p.control, p.record, rec)
    return transient


def _violation(p):
    try:
        with open(p.violation, encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else None
    except (OSError, ValueError):
        return None


def _note_violation(p, where, problems):
    """Write the violation record unless one exists (best effort: a failure to write is reported, not raised)."""
    try:
        if not os.path.exists(p.violation):
            _write_json(p.control, p.violation, {
                "campaign_failed": True, "detected_at": _now(), "where": where, "violations": problems,
                "message": "Protected campaign files changed; the campaign has failed and no recipe ships."})
    except Exception as e:
        problems.append({"path": p.violation, "problem": "could not be written: %s: %s" % (type(e).__name__, e)})


def teardown_runs(p):
    """{run_id: result} of teardown_run.sh (the skill copy's) for every run in the campaign's tracked_jobs.jsonl."""
    cfg = read_config(p.config)
    campaign = cfg.get("CAMPAIGN_DIR") or cfg.get("AUTOTUNE_PROJECT_ROOT")
    runs = []
    for path in sorted(glob.glob(os.path.join(glob.escape(campaign), "*", "tracked_jobs.jsonl"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rid = json.loads(line).get("run_id")
                except (ValueError, AttributeError):
                    continue
                if isinstance(rid, str) and rid and rid not in runs:
                    runs.append(rid)
    env = dict(os.environ)
    env.update(cfg)  # teardown_run.sh holds the environment to the campaign's settings
    out = {}
    for rid in runs:
        try:
            r = subprocess.run(["bash", os.path.join(p.skill, "scripts", "teardown_run.sh"), rid], env=env,
                               cwd=campaign, capture_output=True, text=True, timeout=300)
            out[rid] = "torn down" if r.returncode == 0 else "FAILED (exit %d): %s" % (
                r.returncode, (r.stderr or r.stdout).strip()[-300:])
        except Exception as e:
            out[rid] = "FAILED: %s: %s" % (type(e).__name__, e)
    return out


def _teardown(p):
    """Tear down the tracked runs once per violation and record the result in the violation record."""
    done = (_violation(p) or {}).get("teardown")
    if isinstance(done, dict) and done.get("runs") is not None and all(
            v == "torn down" for v in done["runs"].values()):
        return done
    try:
        result = {"at": _now(), "runs": teardown_runs(p)}
    except Exception as e:
        result = {"at": _now(), "runs": None, "error": "%s: %s" % (type(e).__name__, e)}
    try:
        with _locked(p.control):
            v = _violation(p) or {"campaign_failed": True, "detected_at": _now(), "violations": []}
            v["teardown"] = result
            _write_json(p.control, p.violation, v)
    except Exception:
        pass
    return result


def teardown_note(control):
    """One line on the live-run teardown of a failed campaign."""
    t = (_violation(Paths(control)) or {}).get("teardown")
    if not isinstance(t, dict):
        return "live runs: not torn down"
    if t.get("runs") is None:
        return "live runs: teardown could not run (%s)" % t.get("error")
    if not t["runs"]:
        return "live runs: none tracked"
    good = [r for r, v in t["runs"].items() if v == "torn down"]
    bad = ["%s (%s)" % (r, v) for r, v in t["runs"].items() if v != "torn down"]
    return "; ".join((["live runs torn down: %s" % ", ".join(good)] if good else []) +
                     (["teardown FAILED for %s" % "; ".join(bad)] if bad else []))


def stored(control):
    """The problems of an earlier failed check, [] when there was none."""
    try:
        with open(os.path.join(control, VIOLATION), encoding="utf-8") as f:
            v = json.load(f)
        return list(v["violations"]) or [{"path": VIOLATION, "problem": "violation record without details"}]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [{"path": os.path.join(control, VIOLATION), "problem": "unreadable violation record: %s" % e}]


def summary(problems, limit=5):
    head = "; ".join("%s: %s" % (x.get("path"), x.get("problem")) for x in problems[:limit])
    more = len(problems) - limit
    return head + ("; and %d more" % more if more > 0 else "")


def hook_check(env, where):
    """The violation summary for a hook ("" when intact), after tearing down the tracked runs of a failed campaign.
    With the launcher's ATTUNE_CONTROL_DIR set, a campaign without its record has failed; without it (a checkout,
    the tests) a missing record means no campaign. Never raises: an error is a failure."""
    try:
        control = env.get("ATTUNE_CONTROL_DIR") or DEFAULT_CONTROL
        if not env.get("ATTUNE_CONTROL_DIR") and not os.path.isfile(os.path.join(control, RECORD)):
            return ""
        problems = check(control, where, teardown=True)
        if not problems:
            return ""
        return "%s; %s (record: %s)" % (summary(problems), teardown_note(control),
                                        os.path.join(os.path.realpath(control), VIOLATION))
    except Exception as e:
        return "the integrity check could not run: %s: %s" % (type(e).__name__, e)


# ---- the file-edit tools ------------------------------------------------------------------------------------------

def write_rule(path, control, campaign=None, append=()):
    """("protected" | "append" | None, why) for a file-edit tool writing the resolved absolute `path`. `append`
    names more append-only files (the GPU-hour log, $AUTOTUNE_LEDGER)."""
    p = Paths(control, campaign)
    rec = None
    if os.path.isfile(p.record):
        rec = _load(p.record)
        p.campaign = p.campaign or rec.get("campaign")
    recorded = rec["files"] if rec else {}
    appends = set(rec["append_only"]) if rec else set()
    appends.update(os.path.realpath(a) for a in append if a)
    if path in appends or os.path.basename(path) == LEDGER_NAME:
        return "append", "%s is append-only" % path
    lessons = "write proposed lessons to <project_folder>/LESSONS.md (SKILL.md L4c)"
    if path == p.levers:
        return "protected", "%s is the campaign's read-only lever catalog; %s" % (path, lessons)
    source = rec.get("source") if rec else None
    if source and any(_under(path, os.path.join(source, d)) for d in ("skill", "harness")):
        return "protected", ("%s is in the ATTUNE tree the campaign was copied from; its skill is the read-only copy "
                             "%s, and %s" % (path, p.skill, lessons))
    if _under(path, p.summaries) and path != p.summaries:
        if path in recorded:
            name = os.path.basename(path)
            tag = name[:-len("_summary.md")] if name.endswith("_summary.md") else os.path.splitext(name)[0]
            unique = "%s_%s_summary.md" % (tag, rec.get("campaign_id") or "<campaign id>")
            return "protected", ("%s was there at launch and stays unchanged; write this campaign's summary as "
                                 "past_tuning_projects/%s" % (path, unique))
        return None, ""
    roots = [p.control] if rec or os.path.basename(p.control) == CONTROL_NAME else [p.skill, p.harness]
    if any(_under(path, r) for r in roots) or path in (p.config, p.record, p.violation) or path in recorded:
        return "protected", "%s is a protected campaign file (the launcher records its hash)" % path
    if p.campaign and (_under(path, os.path.join(p.campaign, ".claude")) or path in p.launcher_files()):
        return "protected", "%s is launcher configuration" % path
    return None, ""


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("seal", "refresh", "check", "source-check"))
    ap.add_argument("control_dir")
    ap.add_argument("--where", default="integrity.py check", help="where the check runs (for the violation record)")
    ap.add_argument("--relaunch", action="store_true",
                    help="the check before a later launch: the operator's edits to goal_prompt.txt and "
                         "resume_prompt.txt are allowed (the launcher re-records them)")
    ap.add_argument("--teardown", action="store_true", help="on a failure, tear down the campaign's tracked runs")
    ap.add_argument("--dataset", nargs="?", const=True, default=None, metavar="PATH",
                    help="seal: record the hashes of the dataset at PATH (read through the log pod); check: hash "
                         "the recorded dataset again")
    ap.add_argument("--source", metavar="DIR", help="seal: the tree the copies came from ($ATTUNE_HOME)")
    a = ap.parse_args(argv)
    if a.command == "seal":
        try:
            rec = seal(a.control_dir, dataset=a.dataset if isinstance(a.dataset, str) else None, source=a.source)
        except Exception as e:
            print("integrity: cannot seal the campaign: %s: %s" % (type(e).__name__, e), file=sys.stderr)
            return 2
        ds = rec.get("dataset")
        print("integrity: sealed %d protected and %d append-only files in %s%s"
              % (len(rec["files"]), len(rec["append_only"]), os.path.join(rec["control"], RECORD),
                 "; dataset %s: %s" % (ds["path"], ds.get("rows_sha256") or ds["parquet_sha256"]) if ds else ""))
        return 0
    if a.command == "refresh":
        refresh(a.control_dir)
        print("integrity: re-recorded the launcher's settings and prompts")
        return 0
    if a.command == "source-check":
        try:
            changes = source_check(a.control_dir, a.where)
            source = _load(Paths(a.control_dir).record).get("source")
        except Exception as e:
            print("integrity: cannot check the source tree: %s: %s" % (type(e).__name__, e), file=sys.stderr)
            return 1
        if changes is None:
            print("integrity: the record names no source tree to check")
        elif not changes:
            print("integrity: the source tree's skill/ and harness/ are unchanged since launch (%s)" % source)
        else:
            print("WARNING: the source tree %s changed during the session: %s. The campaign's copies are unaffected; "
                  "unless an operator made these changes, the session wrote into the tree (recorded in %s)"
                  % (source, summary(changes, limit=10), os.path.join(os.path.realpath(a.control_dir), RECORD)),
                  file=sys.stderr)
        return 0
    problems = check(a.control_dir, a.where, relaunch=a.relaunch, teardown=a.teardown, dataset=bool(a.dataset))
    lasting = [x for x in problems if not x.get("transient")]
    if lasting:
        print("INTEGRITY VIOLATION: %s (record: %s)" % (summary(lasting), os.path.join(
            os.path.realpath(a.control_dir), VIOLATION)), file=sys.stderr)
        if a.teardown:
            print(teardown_note(a.control_dir), file=sys.stderr)
        return 3
    if problems:
        print("integrity: cannot verify now: %s" % summary(problems), file=sys.stderr)
        return 4
    print("integrity: intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
