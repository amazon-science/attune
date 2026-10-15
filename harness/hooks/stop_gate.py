#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code Stop hook: keeps an unattended campaign going until it converges (convergence.py).

Inactive unless AUTOTUNE_AUTONOMOUS=1. It first runs the campaign's integrity check (integrity.py); on a violation it
ends the session with a failure ({"continue": false}) and launch_campaign.sh exits nonzero. Otherwise a stop is
allowed when the campaign has converged, or after $AUTOTUNE_MAX_STALL (default 3) consecutive blocked stops without
progress: no new ranked result in ranked-count mode, no new spend in GPU-hour mode. Otherwise the stop is blocked
and the agent is told to continue. Any error blocks the stop too, unless the ledger already holds a baseline and a
STOP row.
The stall count lives in <project_root>/.stop_gate_state.json. Wire it in settings.json under hooks.Stop.
Runs on Python 3.7+.
"""
from __future__ import annotations

import sys

STATE_NAME = ".stop_gate_state.json"


def _stalled(progress: int, max_stall: int, path: str) -> bool:
    """Count one more blocked stop; True once `progress` has not moved for max_stall of them."""
    import json

    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        last, stall = int(state["progress"]), int(state["stall"])
    except (OSError, ValueError, KeyError, TypeError):
        last, stall = -1, 0
    stall = stall + 1 if progress == last else 0
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"progress": progress, "stall": stall}, f)
    return stall >= max_stall


def _state_path(env) -> str:
    import os

    root = env.get("AUTOTUNE_PROJECT_ROOT") or os.path.dirname(env.get("AUTOTUNE_LEDGER") or "")
    if not root:
        raise ValueError("AUTOTUNE_PROJECT_ROOT is not set")
    return os.path.join(root, STATE_NAME)


def respond(raw: str) -> str:
    """The hook's stdout: empty to allow the stop, a block decision otherwise."""
    env, convergence = {}, None
    try:
        import os

        env = os.environ
        if env.get("AUTOTUNE_AUTONOMOUS") != "1":
            return ""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            import integrity

            failed = integrity.hook_check(env, "Stop hook")
        except Exception as e:  # an integrity check that cannot run fails the campaign; it never blocks forever
            failed = _without_integrity(env, e)
        if failed:
            return _end("ATTUNE integrity violation, the campaign has failed and no recipe ships: %s" % failed)
        import convergence

        limits = convergence.Limits.from_env(env)
        ev = convergence.status(env, limits)
        if ev.converged or _stalled(ev.progress, limits.max_stall, _state_path(env)):
            return ""
        return _block(f"Campaign not converged: {ev.reason}. {convergence.continuation(limits)}")
    except Exception as e:
        try:
            if convergence is not None and convergence.stop_declared(env):
                return ""
        except Exception:
            pass
        return _block(f"Stop gate error, the campaign continues: {type(e).__name__}: {e}")


def _without_integrity(env, error) -> str:
    """integrity.py cannot load: tear down the tracked runs with the skill copy's teardown_run.sh and write a minimal
    violation record, both best effort."""
    import glob
    import json
    import os
    import subprocess
    import time

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    control = env.get("ATTUNE_CONTROL_DIR") or here
    root = env.get("AUTOTUNE_PROJECT_ROOT") or env.get("CAMPAIGN_DIR") or ""
    runs = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(root), "*", "tracked_jobs.jsonl"))) if root else []:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                rids = [json.loads(line).get("run_id") for line in f if line.strip()]
        except (OSError, ValueError, AttributeError):
            continue
        for rid in (r for r in rids if isinstance(r, str) and r):
            try:
                p = subprocess.run(["bash", os.path.join(control, "skill", "scripts", "teardown_run.sh"), rid],
                                   capture_output=True, text=True, timeout=300, cwd=root)
                runs[rid] = "torn down" if p.returncode == 0 else "FAILED (exit %d)" % p.returncode
            except Exception as e:
                runs[rid] = "FAILED: %s" % e
    record = {"campaign_failed": True, "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "where": "Stop hook", "violations": [{"path": "integrity.py", "problem": "cannot run: %s: %s" % (
                  type(error).__name__, error)}], "teardown": {"runs": runs}}
    try:
        mode = os.stat(control).st_mode
        os.chmod(control, mode | 0o200)
        try:
            with open(os.path.join(control, "integrity_violation.json"), "x", encoding="utf-8") as f:
                json.dump(record, f, indent=1)
        finally:
            os.chmod(control, mode)
    except OSError:
        pass
    return "the integrity check cannot run (%s: %s); live runs: %s" % (
        type(error).__name__, error, ", ".join("%s %s" % kv for kv in runs.items()) or "none tracked")


def _block(reason: str) -> str:
    import json

    return json.dumps({"decision": "block", "reason": reason})


def _end(reason: str) -> str:
    import json

    return json.dumps({"continue": False, "stopReason": reason, "systemMessage": reason})


if __name__ == "__main__":
    try:
        sys.stdout.write(respond(sys.stdin.read()))
    except Exception as e:  # exit code 2 blocks the stop even when no JSON could be written
        sys.stderr.write(f"Stop gate error, the campaign continues: {e}\n")
        sys.exit(2)
