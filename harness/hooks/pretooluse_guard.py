#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code PreToolUse hook for unattended campaigns; inactive unless AUTOTUNE_AUTONOMOUS=1.

- Bash: denies any command that command_guard.classify_command does not allow.
- Write, Edit, MultiEdit, NotebookEdit: denies writes to protected paths (command_guard.write_block_reason). Edits
  count as appends; a Write is an overwrite unless it keeps the file's current text as its beginning.
- AskUserQuestion: denies with guidance, so the agent proceeds on a recorded default.
Any error denies the call. Wire it in settings.json under hooks.PreToolUse with the matcher
"Bash|AskUserQuestion|Write|Edit|MultiEdit|NotebookEdit".
"""
import sys

EDIT_TOOLS = {"Write": "file_path", "Edit": "file_path", "MultiEdit": "file_path", "NotebookEdit": "notebook_path"}
ASK_REPLY = ("Operate autonomously: assume the recommended or first option, proceed, and record the assumption "
             "in the ledger. Do not ask again.")


def _only_adds(path: str, content) -> bool:
    """True when a Write creates `path` or keeps its current text as the new content's beginning."""
    try:
        with open(path, encoding="utf-8") as f:
            old = f.read()
    except FileNotFoundError:
        return True
    except (OSError, ValueError):
        return False
    return isinstance(content, str) and content.startswith(old)


def _decide(payload: dict, guard) -> str | None:
    """The deny reason for one tool call, or None to let it through."""
    import os

    tool = payload.get("tool_name", "")
    args = payload.get("tool_input") or {}
    cwd = payload.get("cwd")
    if tool == "AskUserQuestion":
        return ASK_REPLY
    if tool == "Bash":
        d = guard.classify_command(args.get("command", ""), cwd=cwd)
        return None if d.allow else f"safety guard: {d.reason}"
    if tool in EDIT_TOOLS:
        path = args.get(EDIT_TOOLS[tool])
        if not isinstance(path, str) or not path:
            return f"safety guard: {tool} without a file path"
        full = os.path.join(cwd or os.getcwd(), os.path.expanduser(path))
        append = tool != "Write" or _only_adds(full, args.get("content"))
        why = guard.write_block_reason(path, cwd, append=append)
        return f"safety guard: {why}" if why else None
    return None


def respond(raw: str) -> str:
    """The hook's stdout for one call: empty to allow, a deny decision otherwise."""
    try:
        import json
        import os

        if os.environ.get("AUTOTUNE_AUTONOMOUS") != "1":
            return ""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import command_guard

        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("hook input is not a JSON object")
        reason = _decide(payload, command_guard)
    except Exception as e:
        reason = f"safety guard error, call refused: {type(e).__name__}: {e}"
    return _deny(reason) if reason else ""


def _deny(reason: str) -> str:
    import json

    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }})


if __name__ == "__main__":
    try:
        sys.stdout.write(respond(sys.stdin.read()))
    except Exception as e:  # exit code 2 blocks the call even when no JSON could be written
        sys.stderr.write(f"safety guard error, call refused: {e}\n")
        sys.exit(2)
