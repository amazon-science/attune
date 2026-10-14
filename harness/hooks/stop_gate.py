#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code Stop hook: keeps an unattended campaign going until it converges (convergence.py).

Inactive unless AUTOTUNE_AUTONOMOUS=1. A stop is allowed when the campaign has converged, or after
$AUTOTUNE_MAX_STALL (default 3) consecutive blocked stops without progress: no new ranked result in
ranked-count mode, no new spend in GPU-hour mode. Otherwise the stop is blocked and the agent is told to
continue. Any error blocks the stop too, unless the ledger already holds a baseline and a STOP row.
The stall count lives in <project_root>/.stop_gate_state.json. Wire it in settings.json under hooks.Stop.
"""
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


def _block(reason: str) -> str:
    import json

    return json.dumps({"decision": "block", "reason": reason})


if __name__ == "__main__":
    try:
        sys.stdout.write(respond(sys.stdin.read()))
    except Exception as e:  # exit code 2 blocks the stop even when no JSON could be written
        sys.stderr.write(f"Stop gate error, the campaign continues: {e}\n")
        sys.exit(2)
