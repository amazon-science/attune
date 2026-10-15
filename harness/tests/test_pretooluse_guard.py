# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PreToolUse hook fails closed, applies the kubectl and web rules to Bash, refuses edits of protected campaign
files, keeps append-only files append-only, answers questions with defaults, and starts under Python 3.7.
Run: python3 -m pytest harness/tests"""
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import command_guard
import pretooluse_guard as hook

HARNESS = Path(hook.__file__).resolve().parents[1]
HOOK_MODULES = [HARNESS / "hooks" / "pretooluse_guard.py", HARNESS / "hooks" / "stop_gate.py",
                HARNESS / "command_guard.py", HARNESS / "integrity.py", HARNESS / "convergence.py"]
PY37 = shutil.which("python3.7")


def _call(tool, cwd, **tool_input):
    return hook.respond(json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(cwd)}))


def _decision(out):
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "allow"


@pytest.fixture
def active(sealed, monkeypatch):
    monkeypatch.setenv("AUTOTUNE_AUTONOMOUS", "1")
    return sealed


def test_inactive_without_autonomous_mode(campaign):
    assert hook.respond("not json") == ""


def test_non_json_stdin_denies(active):
    p = subprocess.run([sys.executable, hook.__file__], input="not json", capture_output=True, text=True,
                       env=dict(os.environ), check=False)
    assert p.returncode == 0
    assert _decision(p.stdout) == "deny"
    assert "JSONDecodeError" in json.loads(p.stdout)["hookSpecificOutput"]["permissionDecisionReason"]


def test_a_crash_in_the_guard_denies(active, monkeypatch):
    def boom(*_, **__):
        raise RuntimeError("guard bug")
    monkeypatch.setattr(command_guard, "check_command", boom)
    out = _call("Bash", active.root, command="ls")
    assert _decision(out) == "deny" and "guard bug" in out


def test_bash_follows_the_command_rules(active):
    assert _decision(_call("Bash", active.root, command="rm -rf proj/scratch && python3 -c 'print(1)'")) == "allow"
    assert _decision(_call("Bash", active.root, command="kubectl --context c -n ns delete rayjob alice-tune-r1")) \
        == "allow"
    out = _call("Bash", active.root, command="kubectl --context c -n ns delete rayjob bob-r1")
    assert _decision(out) == "deny" and "alice-tune-" in out


def test_protected_files_are_refused(active):
    for path in (active.skill / "SKILL.md", active.skill / "scripts" / "read_metrics.sh",
                 active.skill / "scripts" / "new_tool.py", active.harness / "integrity.py",
                 active.control / "config.env", active.control / "integrity.json",
                 active.skill / "past_tuning_projects" / "README.md", active.root / ".claude" / "settings.json",
                 active.root / "goal_prompt.rendered.txt"):
        assert _decision(_call("Write", active.root, file_path=str(path), content="x")) == "deny", path
        assert _decision(_call("Edit", active.root, file_path=str(path), old_string="a", new_string="b")) == "deny"
    assert _decision(_call("NotebookEdit", active.root, notebook_path=".claude/x.ipynb")) == "deny"
    linked = ".claude/skills/autotune-training-throughput"
    (active.root / linked).symlink_to(active.skill)
    assert _decision(_call("Write", active.root, file_path=f"{linked}/SKILL.md", content="x")) == "deny"
    out = _call("Write", active.root, file_path=f"{linked}/past_tuning_projects/proj_summary.md", content="# p\n")
    assert _decision(out) == "deny" and f"proj_{active.campaign_id}_summary.md" in out   # an earlier campaign's
    unique = f"{linked}/past_tuning_projects/proj_{active.campaign_id}_summary.md"
    assert _decision(_call("Write", active.root, file_path=unique, content="# p\n")) == "allow"


def test_the_agents_own_files_are_free(active):
    for path in ("proj/RESULTS.md", "proj/notes.md", f"{active.root}/proj/aid.py", "/tmp/scratch.txt"):
        assert _decision(_call("Write", active.root, file_path=path, content="x")) == "allow", path
    assert _decision(_call("MultiEdit", active.root, file_path="proj/notes.md", edits=[])) == "allow"


@pytest.mark.parametrize("which", ["ledger", "gpuh", "levers"])
def test_append_only_files_take_only_appends(active, which):
    path = {"ledger": active.ledger, "gpuh": active.root / "gpuh.jsonl",
            "levers": active.skill / "references" / "levers.md"}[which]
    if which != "levers":
        path.write_text('{"run_id": "a"}\n')
    old = path.read_text()

    def decide(tool, **args):
        return _decision(_call(tool, active.root, file_path=str(path), **args))
    assert decide("Write", content=old + "new line\n") == "allow"
    assert decide("Write", content="new line\n" + old) == "deny"
    assert decide("Write", content="") == "deny"
    last = old.splitlines(keepends=True)[-1]
    assert decide("Edit", old_string=last, new_string=last + "new line\n") == "allow"
    assert decide("Edit", old_string=last, new_string="changed\n") == "deny"
    assert decide("Edit", old_string=old[:3], new_string=old[:3] + "x") == "deny"   # an insertion inside
    assert decide("Edit", old_string="not in the file", new_string="x") == "deny"
    assert decide("MultiEdit", edits=[{"old_string": last, "new_string": last + "a\n"},
                                      {"old_string": "a\n", "new_string": "a\nb\n"}]) == "allow"
    assert decide("MultiEdit", edits=[{"old_string": last, "new_string": last + "a\n"},
                                      {"old_string": last + "a\n", "new_string": "b\n"}]) == "deny"
    assert decide("NotebookEdit") == "deny"


def test_a_new_ledger_may_be_written(active):
    out = _call("Write", active.root, file_path=str(active.ledger), content='{"run_id": "a"}\n')
    assert _decision(out) == "allow"


def test_an_edit_without_a_path_denies(active):
    assert _decision(_call("Write", active.root, content="x")) == "deny"


def test_questions_become_recorded_defaults(active):
    out = _call("AskUserQuestion", active.root, questions=[])
    assert _decision(out) == "deny" and "record the assumption" in out


def test_other_tools_pass(active):
    assert _call("Read", active.root, file_path="/etc/hosts") == ""


@pytest.mark.parametrize("path", HOOK_MODULES, ids=lambda p: p.name)
def test_the_hook_modules_parse_as_python_37(path):
    ast.parse(path.read_text(), filename=str(path), feature_version=(3, 7))


@pytest.mark.skipif(not PY37, reason="no python3.7 on PATH")
def test_the_hook_runs_under_python_37(active):
    def run(command):
        payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(active.root)})
        p = subprocess.run([PY37, hook.__file__], input=payload, capture_output=True, text=True,
                           env=dict(os.environ), check=False)
        assert p.returncode == 0, p.stderr
        return _decision(p.stdout)
    assert run("kubectl --context c -n ns get pods") == "allow"
    assert run("kubectl get pods") == "deny"
    stop = subprocess.run([PY37, str(HARNESS / "hooks" / "stop_gate.py")], input="{}", capture_output=True,
                          text=True, env=dict(os.environ), check=False)
    assert stop.returncode == 0 and "Campaign not converged" in stop.stdout, stop.stderr


def test_a_hook_that_cannot_start_blocks(tmp_path):
    template = json.loads((HARNESS / "settings.template.json").read_text())
    commands = [h["command"] for event in template["hooks"].values() for m in event for h in m["hooks"]]
    assert commands and all(c.endswith("|| exit 2") for c in commands)
    missing = commands[0].replace("${ATTUNE_CONTROL_DIR}", str(tmp_path / "gone"))
    assert subprocess.run(["sh", "-c", missing], capture_output=True, check=False).returncode == 2
