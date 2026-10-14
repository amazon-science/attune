# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PreToolUse hook fails closed, screens shell commands and file edits, and answers questions with defaults.
Run: python3 -m pytest harness/tests"""
import json
import os
import subprocess
import sys

import pytest

import command_guard
import pretooluse_guard as hook


def _call(tool: str, cwd, **tool_input) -> str:
    return hook.respond(json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(cwd)}))


def _decision(out: str) -> str:
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "allow"


@pytest.fixture
def active(campaign, monkeypatch):
    monkeypatch.setenv("AUTOTUNE_AUTONOMOUS", "1")
    return campaign


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
    monkeypatch.setattr(command_guard, "classify_command", boom)
    out = _call("Bash", active.root, command="ls")
    assert _decision(out) == "deny" and "guard bug" in out


def test_bash_is_screened(active):
    assert _decision(_call("Bash", active.root, command="ls -la")) == "allow"
    out = _call("Bash", active.root, command="kubectl --context c -n ns delete rayjob x")
    assert _decision(out) == "deny" and "teardown_run.sh" in out


def test_write_to_a_protected_path_denies(active):
    assert _decision(_call("Write", active.root, file_path=f"{active.shared}/x", content="")) == "deny"
    assert _decision(_call("Write", active.root, file_path=f"{active.home}/.aws/config", content="")) == "deny"
    skill_file = f"{command_guard.SKILL_DIR}/references/measurement.md"
    assert _decision(_call("Edit", active.root, file_path=skill_file, old_string="a", new_string="b")) == "deny"
    assert _decision(_call("NotebookEdit", active.root, notebook_path=f"{active.root}/.claude/x.ipynb")) == "deny"


def test_the_skill_takes_lever_rules_and_summaries(active):
    levers = command_guard.LEVERS
    current = open(levers, encoding="utf-8").read()
    assert _decision(_call("Edit", active.root, file_path=levers, old_string="a", new_string="ab")) == "allow"
    assert _decision(_call("MultiEdit", active.root, file_path=levers, edits=[])) == "allow"
    assert _decision(_call("Write", active.root, file_path=levers, content=current + "- a new rule\n")) == "allow"
    assert _decision(_call("Write", active.root, file_path=levers, content="- only this rule\n")) == "deny"
    linked = ".claude/skills/autotune-training-throughput/past_tuning_projects/proj_summary.md"
    assert _decision(_call("Write", active.root, file_path=linked, content="# proj\n")) == "allow"
    past = command_guard.SUMMARIES_DIR
    for path in (f"{past}/README.md", f"{past}/sub/proj_summary.md", f"{past}/proj_notes.md",
                 f"{command_guard.SKILL_DIR}/SKILL.md", f"{command_guard.HARNESS_DIR}/convergence.py"):
        assert _decision(_call("Write", active.root, file_path=path, content="x")) == "deny", path


def test_write_in_the_campaign_directory_allows(active):
    assert _decision(_call("Write", active.root, file_path=f"{active.root}/proj/RESULTS.md", content="")) == "allow"
    assert _decision(_call("MultiEdit", active.root, file_path="proj/notes.md", edits=[])) == "allow"


def test_the_ledger_is_edited_not_overwritten(active):
    ledger = active.root / "proj" / "AUTOTUNE_LEDGER.jsonl"
    assert _decision(_call("Write", active.root, file_path=str(ledger), content="{}\n")) == "allow"
    ledger.write_text("{}\n")
    assert _decision(_call("Write", active.root, file_path=str(ledger), content="")) == "deny"
    assert _decision(_call("Edit", active.root, file_path=str(ledger), old_string="a", new_string="b")) == "allow"


def test_an_edit_without_a_path_denies(active):
    assert _decision(_call("Write", active.root, content="x")) == "deny"


def test_questions_become_recorded_defaults(active):
    out = _call("AskUserQuestion", active.root, questions=[])
    assert _decision(out) == "deny" and "record the assumption" in out


def test_other_tools_pass(active):
    assert _call("Read", active.root, file_path="/etc/hosts") == ""
