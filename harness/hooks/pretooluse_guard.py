#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code PreToolUse hook for unattended campaigns; inactive unless AUTOTUNE_AUTONOMOUS=1.

- Bash: the two rules of command_guard.py, kubectl scope and curl/wget only with web access on; everything else runs.
- Write, Edit, MultiEdit, NotebookEdit: refuses the protected campaign files, the lever catalog among them, and the
  skill/ and harness/ of the tree the campaign was copied from (integrity.write_rule). The append-only files (ledgers,
  the GPU-hour log) take only changes that keep their current text as the beginning: a Write that starts with it, an
  Edit or MultiEdit whose result starts with it.
- AskUserQuestion: denies with guidance, so the agent proceeds on a recorded default.
The guard prevents accidents to ATTUNE's own records and cluster scope; it is not a security boundary. Any error
denies the call; the settings run it as `python3 <this file> || exit 2`, so a failure to start blocks the call too.
Runs on Python 3.7+. Wire it in settings.json under hooks.PreToolUse with the matcher
"Bash|AskUserQuestion|Write|Edit|MultiEdit|NotebookEdit".
"""

from __future__ import annotations

import sys

EDIT_TOOLS = {"Write": "file_path", "Edit": "file_path", "MultiEdit": "file_path", "NotebookEdit": "notebook_path"}
ASK_REPLY = ("Operate autonomously: assume the recommended or first option, proceed, and record the assumption "
             "in the ledger. Do not ask again.")


def _keeps_prefix(tool, path, args):
    """True when the call leaves the file's current text as the beginning of its new text."""
    try:
        with open(path, encoding="utf-8") as f:
            old = f.read()
    except FileNotFoundError:
        old = ""
    if tool == "Write":
        content = args.get("content")
        return isinstance(content, str) and content.startswith(old)
    edits = [args] if tool == "Edit" else args.get("edits")
    if not isinstance(edits, list):
        return False
    text = old
    for e in edits:
        before, after = (e.get("old_string"), e.get("new_string")) if isinstance(e, dict) else (None, None)
        if not isinstance(before, str) or not isinstance(after, str) or (before and before not in text):
            return False
        if not before:
            if text:
                return False
            text = after
        else:
            text = text.replace(before, after) if e.get("replace_all") else text.replace(before, after, 1)
    return text.startswith(old)


def _decide(payload, env, command_guard, integrity):
    """The deny reason for one tool call, or None to let it through."""
    import os

    tool = payload.get("tool_name", "")
    args = payload.get("tool_input") or {}
    cwd = payload.get("cwd") or os.getcwd()
    if tool == "AskUserQuestion":
        return ASK_REPLY
    if tool == "Bash":
        why = command_guard.check_command(args.get("command", ""), env, cwd)
        return "attune guard: %s" % why if why else None
    if tool in EDIT_TOOLS:
        path = args.get(EDIT_TOOLS[tool])
        if not isinstance(path, str) or not path:
            return "attune guard: %s without a file path" % tool
        full = os.path.realpath(os.path.join(cwd, os.path.expanduser(path)))
        control = env.get("ATTUNE_CONTROL_DIR") or integrity.DEFAULT_CONTROL
        campaign = env.get("AUTOTUNE_PROJECT_ROOT") or env.get("CAMPAIGN_DIR")
        gpuh = env.get("AUTOTUNE_GPUH_LOG") or (os.path.join(campaign, "gpuh.jsonl") if campaign else "")
        kind, why = integrity.write_rule(full, control, campaign, append=(gpuh, env.get("AUTOTUNE_LEDGER", "")))
        if kind == "protected":
            return "attune guard: %s" % why
        if kind == "append" and (tool == "NotebookEdit" or not _keeps_prefix(tool, full, args)):
            return ("attune guard: %s; keep its current text and add after it (Write with the old text first, or "
                    "`>>` in Bash)" % why)
    return None


def respond(raw):
    """The hook's stdout for one call: empty to allow, a deny decision otherwise."""
    try:
        import json
        import os

        env = os.environ
        if env.get("AUTOTUNE_AUTONOMOUS") != "1":
            return ""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import command_guard
        import integrity

        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("hook input is not a JSON object")
        reason = _decide(payload, env, command_guard, integrity)
    except Exception as e:
        reason = "attune guard error, call refused: %s: %s" % (type(e).__name__, e)
    return _deny(reason) if reason else ""


def _deny(reason):
    import json

    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }})


if __name__ == "__main__":
    try:
        sys.stdout.write(respond(sys.stdin.read()))
    except BaseException as e:  # exit code 2 blocks the call even when no JSON could be written
        sys.stderr.write("attune guard error, call refused: %s\n" % e)
        sys.exit(2)
