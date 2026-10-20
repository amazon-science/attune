# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""integrity.py passes an untouched campaign and appends, and fails rewrites, truncation, deletion, any change to the
lever catalog, additions to the copies and changed settings, for good, with a violation record; new summaries and
the project folder's LESSONS.md are allowed, and the tree the copies came from is refused to the file-edit tools,
recorded with its commit and dirty flag, and checked at session end, where a change is a warning, not a failure.
Run: python3 -m pytest harness/tests"""
import json
import os
import subprocess
import sys

import pytest
from conftest import seal_campaign

import integrity


def chmod_w(path):
    """What a deliberate write to a read-only copy needs first."""
    os.chmod(path, os.stat(path).st_mode | 0o200)


def edit(path, text, mode="w"):
    chmod_w(path.parent)
    if path.exists():
        chmod_w(path)
    with open(path, mode) as f:
        f.write(text)


def problems(sealed, where="test"):
    return integrity.check(str(sealed.control), where)


def test_an_untouched_campaign_passes(sealed):
    assert problems(sealed) == []
    assert problems(sealed) == []
    rec = json.loads((sealed.control / integrity.RECORD).read_text())
    assert str(sealed.skill / "SKILL.md") in rec["files"]
    assert str(sealed.control / "config.env") in rec["files"]
    assert str(sealed.root / ".claude" / "settings.json") in rec["files"]
    assert str(sealed.skill / "references" / "levers.md") in rec["files"]
    assert str(sealed.root / "gpuh.jsonl") in rec["append_only"] and len(rec["append_only"]) == 1
    assert not os.access(sealed.control / integrity.RECORD, os.W_OK)


def test_appends_pass_and_are_recorded(sealed):
    sealed.ledger.write_text('{"run_id": "a"}\n')        # a ledger created after launch
    (sealed.root / "gpuh.jsonl").write_text('{"run": "a"}\n')
    (sealed.root / "proj" / "LESSONS.md").write_text("## a lesson\n")
    assert problems(sealed) == []
    rec = json.loads((sealed.control / integrity.RECORD).read_text())
    assert rec["append_only"][str(sealed.ledger)]["length"] == len('{"run_id": "a"}\n')
    with open(sealed.ledger, "a") as f:
        f.write('{"run_id": "b"}\n')
    assert problems(sealed) == []


@pytest.mark.parametrize("change", ["rewrite", "truncate", "delete"])
def test_append_only_files_keep_what_was_seen(sealed, change):
    sealed.ledger.write_text('{"run_id": "a", "verdict": "lose"}\n')
    assert problems(sealed) == []   # the row is now seen
    if change == "rewrite":
        sealed.ledger.write_text('{"run_id": "a", "verdict": "win!"}\n{"run_id": "b"}\n')
    elif change == "truncate":
        sealed.ledger.write_text("")
    else:
        sealed.ledger.unlink()
    found = problems(sealed)
    assert len(found) == 1 and found[0]["path"] == str(sealed.ledger)
    assert {"rewrite": "rewritten", "truncate": "truncated", "delete": "deleted"}[change] in found[0]["problem"]


@pytest.mark.parametrize("mode,text", [("a", "- an appended rule\n"), ("w", "# levers\n- the rule, reworded\n")])
def test_the_lever_catalog_is_read_only(sealed, mode, text):
    levers = sealed.skill / "references" / "levers.md"
    assert not os.access(levers, os.W_OK)
    edit(levers, text, mode)
    assert problems(sealed) == [{"path": str(levers), "problem": "modified"}]


@pytest.mark.parametrize("rel", ["skill/SKILL.md", "skill/scripts/read_metrics.sh",
                                 "skill/scripts/correctness/rules/numerics_list.yaml", "harness/integrity.py",
                                 "config.env"])
def test_a_protected_file_that_changes_fails(sealed, rel):
    edit(sealed.control / rel, "changed\n")
    found = problems(sealed)
    assert found == [{"path": str(sealed.control / rel), "problem": "modified"}]


def test_a_deleted_protected_file_fails(sealed):
    path = sealed.skill / "references" / "measurement.md"
    chmod_w(path.parent)
    path.unlink()
    assert problems(sealed) == [{"path": str(path), "problem": "deleted"}]


def test_settings_and_prompts_are_protected(sealed):
    edit(sealed.root / ".claude" / "settings.json", '{"hooks": {"Stop": []}}\n')
    edit(sealed.root / "goal_prompt.rendered.txt", "/goal something else\n")
    paths = {p["path"] for p in problems(sealed)}
    assert paths == {str(sealed.root / ".claude" / "settings.json"), str(sealed.root / "goal_prompt.rendered.txt")}


def test_new_summaries_are_allowed_but_recorded_ones_and_other_additions_are_not(sealed):
    past = sealed.skill / "past_tuning_projects"
    mine = past / f"proj_{sealed.campaign_id}_summary.md"
    mine.write_text("# proj\n")
    (past / "sub").mkdir()
    (past / "sub" / "notes.md").write_text("x\n")
    assert problems(sealed) == []
    mine.write_text("# proj, revised\n")   # the campaign's own summary may change
    assert problems(sealed) == []
    edit(past / "proj_summary.md", "# an earlier campaign, overwritten from Bash\n")
    assert problems(sealed) == [{"path": str(past / "proj_summary.md"), "problem": "modified"}]


def test_an_existing_summary_name_gets_the_unique_one(sealed):
    kind, why = integrity.write_rule(str(sealed.skill / "past_tuning_projects" / "proj_summary.md"),
                                     str(sealed.control))
    assert kind == "protected" and f"past_tuning_projects/proj_{sealed.campaign_id}_summary.md" in why


def test_a_file_added_to_a_copy_fails(sealed):
    edit(sealed.skill / "scripts" / "sitecustomize.py", "import os\n")
    assert problems(sealed) == [{"path": str(sealed.skill / "scripts" / "sitecustomize.py"),
                                 "problem": "added to a protected copy"}]


def test_caches_are_ignored(sealed):
    cache = sealed.harness / "__pycache__"
    chmod_w(sealed.harness)
    cache.mkdir()
    (cache / "integrity.cpython-312.pyc").write_bytes(b"\0")
    assert problems(sealed) == []


def test_a_violation_is_recorded_and_final(sealed):
    path = sealed.skill / "SKILL.md"
    edit(path, "changed\n")
    assert problems(sealed, where="Stop hook")
    record = json.loads((sealed.control / integrity.VIOLATION).read_text())
    assert record["campaign_failed"] and record["where"] == "Stop hook"
    assert record["violations"] == [{"path": str(path), "problem": "modified"}]
    edit(path, "# skill\n")   # restoring the file does not undo the violation
    assert problems(sealed) == record["violations"]
    assert not os.access(sealed.control, os.W_OK), "the control directory stays read-only"


def test_files_named_outside_the_copies(campaign):
    rules, calib = campaign.tmp / "rules", campaign.tmp / "calibration.json"
    (rules / "layer1" / "setup").mkdir(parents=True)
    (rules / "numerics_list.yaml").write_text("numeric: []\n")
    frozen = rules / "layer1" / "setup" / "frozen_args.yaml"
    frozen.write_text("args: {}\n")
    (rules / "layer1" / "setup" / "allowlist.yaml").write_text("{}\n")
    calib.write_text('{"band": 0.03}\n')
    settings = {"CAMPAIGN_DIR": str(campaign.root), "AUTOTUNE_RULES_DIR": str(rules),
                "AUTOTUNE_FROZEN_ARGS": str(frozen), "AUTOTUNE_CALIBRATION": str(calib)}
    control = seal_campaign(campaign.root, settings)
    assert integrity.check(str(control)) == []
    calib.write_text('{"band": 0.5}\n')
    (rules / "layer1" / "setup" / "allowlist.yaml").write_text("{x: [1]}\n")
    assert {p["path"] for p in integrity.check(str(control))} == {
        str(calib), str(rules / "layer1" / "setup" / "allowlist.yaml")}


def test_setup_rule_files_beside_an_outside_frozen_args_file(campaign):
    setup = campaign.tmp / "setup"
    setup.mkdir()
    for name, text in (("frozen_args.yaml", "args: {}\n"), ("allowlist.yaml", "{}\n"), ("ct_status.yaml", "{}\n"),
                       ("defects.yaml", "defects: []\n")):
        (setup / name).write_text(text)
    control = seal_campaign(campaign.root, {"CAMPAIGN_DIR": str(campaign.root),
                                            "AUTOTUNE_FROZEN_ARGS": str(setup / "frozen_args.yaml")})
    assert integrity.check(str(control)) == []
    (setup / "defects.yaml").write_text("defects: []\n# emptied\n")
    assert [p["path"] for p in integrity.check(str(control))] == [str(setup / "defects.yaml")]


def test_rule_files_inside_the_skill_copy_are_protected(campaign):
    skill = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "skill")
    control = seal_campaign(campaign.root, {"CAMPAIGN_DIR": str(campaign.root)}, skill=skill)
    assert integrity.check(str(control)) == []
    path = control / "skill" / "scripts" / "correctness" / "rules" / "layer1" / "example" / "ct_status.yaml"
    os.chmod(path, 0o644)
    path.write_text("tests: {}\n")
    assert [p["path"] for p in integrity.check(str(control))] == [str(path)]


def test_a_later_launch_allows_the_operators_prompt_edits(sealed):
    (sealed.root / "resume_prompt.txt").write_text("The cluster was down for an hour.\n")
    assert integrity.check(str(sealed.control), relaunch=True) == []
    integrity.refresh(str(sealed.control))
    assert problems(sealed) == []
    (sealed.root / "resume_prompt.txt").write_text("changed during the session\n")
    assert problems(sealed)[0]["path"] == str(sealed.root / "resume_prompt.txt")


def test_hook_check_tears_down_the_tracked_runs(sealed):
    env = {"ATTUNE_CONTROL_DIR": str(sealed.control)}
    (sealed.root / "proj" / "tracked_jobs.jsonl").write_text(
        '{"run_id": "alice-tune-r1"}\n{"run_id": "alice-tune-stuck"}\nnot json\n{"run_id": "alice-tune-r1"}\n')
    assert integrity.hook_check(env, "Stop hook") == "" and sealed.torn() == []
    edit(sealed.skill / "SKILL.md", "changed\n")
    out = integrity.hook_check(env, "Stop hook")
    assert "SKILL.md: modified" in out and integrity.VIOLATION in out
    assert "live runs torn down: alice-tune-r1" in out and "teardown FAILED for alice-tune-stuck" in out
    assert sealed.torn() == ["alice-tune-r1", "alice-tune-stuck"]
    record = json.loads((sealed.control / integrity.VIOLATION).read_text())
    assert record["teardown"]["runs"]["alice-tune-r1"] == "torn down"
    integrity.hook_check(env, "Stop hook")   # a failed teardown is tried again, a done one is not repeated
    assert sealed.torn() == ["alice-tune-r1", "alice-tune-stuck", "alice-tune-r1", "alice-tune-stuck"]


def test_any_error_while_checking_fails_the_campaign(sealed, monkeypatch):
    def boom(*_):
        raise OSError("disk full")
    monkeypatch.setattr(integrity, "_write_json", boom)
    sealed.ledger.write_text("{}\n")   # a new ledger: the passing check must write the record, and cannot
    found = problems(sealed)
    assert "cannot verify: OSError: disk full" in found[0]["problem"]
    assert "could not be written" in found[-1]["problem"]


def test_an_unreadable_control_directory_fails_the_campaign(sealed):
    os.chmod(sealed.control, 0)
    try:
        out = integrity.hook_check({"ATTUNE_CONTROL_DIR": str(sealed.control)}, "Stop hook")
    finally:
        os.chmod(sealed.control, 0o555)
    assert "cannot verify" in out and "PermissionError" in out


def test_a_missing_record_fails_only_under_the_launcher(campaign):
    control = campaign.root / ".attune"
    control.mkdir()
    assert "cannot verify" in integrity.hook_check({"ATTUNE_CONTROL_DIR": str(control)}, "Stop hook")
    assert integrity.hook_check({}, "Stop hook") == ""   # a checkout: no campaign
    assert "missing" in integrity.hook_check({"ATTUNE_CONTROL_DIR": str(campaign.tmp / "gone")}, "x")


def test_cli_exit_codes(sealed):
    def run(*args):
        return subprocess.run([sys.executable, integrity.__file__, *args], capture_output=True, text=True,
                              check=False)
    p = run("check", str(sealed.control))
    assert p.returncode == 0 and "intact" in p.stdout
    edit(sealed.skill / "SKILL.md", "changed\n")
    p = run("check", str(sealed.control), "--where", "session end")
    assert p.returncode == 3 and "INTEGRITY VIOLATION" in p.stderr and "SKILL.md: modified" in p.stderr


@pytest.mark.parametrize("path,kind", [
    ("skill/SKILL.md", "protected"), ("harness/hooks/stop_gate.py", "protected"), ("config.env", "protected"),
    ("integrity.json", "protected"), ("skill/references/levers.md", "protected"),
    ("skill/past_tuning_projects/README.md", "protected"), ("skill/past_tuning_projects/p_summary.md", None),
    ("skill/scripts/new.py", "protected")])
def test_write_rule_in_the_control_directory(sealed, path, kind):
    assert integrity.write_rule(str(sealed.control / path), str(sealed.control))[0] == kind


@pytest.mark.parametrize("rel,kind", [
    (".claude/settings.json", "protected"), (".claude/x.ipynb", "protected"), ("goal_prompt.rendered.txt", "protected"),
    ("resume_prompt.txt", "protected"), ("gpuh.jsonl", "append"), ("proj/AUTOTUNE_LEDGER.jsonl", "append"),
    ("proj/RESULTS.md", None), ("proj/LESSONS.md", None), ("proj/aid.py", None)])
def test_write_rule_in_the_campaign_directory(sealed, rel, kind):
    assert integrity.write_rule(str(sealed.root / rel), str(sealed.control))[0] == kind


def test_the_lever_catalog_points_to_lessons(sealed):
    kind, why = integrity.write_rule(str(sealed.skill / "references" / "levers.md"), str(sealed.control))
    assert kind == "protected" and "LESSONS.md" in why


def test_the_source_tree_is_recorded_and_refused(campaign):
    tree = campaign.tmp / "tree"
    for d in ("skill/references", "harness", "attune-projects"):
        (tree / d).mkdir(parents=True)
    root = tree / "attune-projects" / "c1"   # a campaign directory inside the tree stays the agent's own
    (root / "proj").mkdir(parents=True)
    control = seal_campaign(root, {"CAMPAIGN_DIR": str(root)}, source=str(tree))
    rec = json.loads((control / integrity.RECORD).read_text())
    assert rec["source"] == str(tree) and rec["source_commit"] is None   # not a git checkout
    for rel in ("skill/references/levers.md", "skill/past_tuning_projects/p_summary.md", "harness/integrity.py"):
        kind, why = integrity.write_rule(str(tree / rel), str(control))
        assert kind == "protected" and "copied from" in why and "LESSONS.md" in why, rel
    for path in (tree / "README.md", root / "proj" / "LESSONS.md"):
        assert integrity.write_rule(str(path), str(control))[0] is None, path


def test_the_source_commit_and_dirty_flag_are_recorded(campaign):
    tree = campaign.tmp / "tree"
    (tree / "skill").mkdir(parents=True)
    (tree / "skill" / "SKILL.md").write_text("# skill\n")
    (tree / "README.md").write_text("# tree\n")
    git = ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    try:
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "."], check=True)
        subprocess.run(git + ["commit", "-q", "-m", "base"], check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no usable git")
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    def sealed_with(name):
        root = campaign.tmp / name
        (root / "proj").mkdir(parents=True)
        rec = json.loads((seal_campaign(root, {"CAMPAIGN_DIR": str(root)}, source=str(tree)) / integrity.RECORD)
                         .read_text())
        return rec["source_commit"], rec["source_dirty"]
    assert sealed_with("clean") == (head, False)
    (tree / "README.md").write_text("# tree, edited\n")   # outside skill/ and harness/
    assert sealed_with("readme") == (head, False)
    (tree / "harness").mkdir()
    (tree / "harness" / "new.py").write_text("")          # untracked counts
    assert sealed_with("untracked") == (head, True)


def _tree_campaign(campaign):
    tree = campaign.tmp / "tree"
    for rel in ("skill/references/levers.md", "skill/SKILL.md", "harness/integrity.py"):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text("# %s\n" % rel)
    control = seal_campaign(campaign.root, {"CAMPAIGN_DIR": str(campaign.root)}, source=str(tree))
    return tree, control


def test_a_bash_write_into_the_source_tree_is_reported_not_failed(campaign):
    tree, control = _tree_campaign(campaign)
    rec = json.loads((control / integrity.RECORD).read_text())
    assert set(rec["source_files"]) == {"skill/references/levers.md", "skill/SKILL.md", "harness/integrity.py"}
    assert integrity.source_check(str(control), "first") == []
    with open(tree / "skill" / "references" / "levers.md", "a") as f:   # cat >> levers.md <<'EOF'
        f.write("- a rule\n")
    (tree / "skill" / "SKILL.md").unlink()
    (tree / "harness" / "new.py").write_text("")
    (tree / "harness" / "__pycache__").mkdir()
    (tree / "harness" / "__pycache__" / "integrity.cpython-310.pyc").write_bytes(b"")   # caches are skipped
    changes = [{"path": "skill/SKILL.md", "problem": "deleted"},
               {"path": "skill/references/levers.md", "problem": "modified"},
               {"path": "harness/new.py", "problem": "added"}]
    assert integrity.source_check(str(control), "second") == changes
    assert integrity.check(str(control)) == [] and not (control / integrity.VIOLATION).exists()
    checks = json.loads((control / integrity.RECORD).read_text())["source_checks"]
    assert [(c["where"], c["changes"]) for c in checks] == [("first", []), ("second", changes)]


def test_a_later_launch_takes_the_trees_hashes_again(campaign):
    tree, control = _tree_campaign(campaign)
    (tree / "skill" / "SKILL.md").write_text("# an operator's change between sessions\n")
    integrity.refresh(str(control))
    assert integrity.source_check(str(control)) == []


def test_source_check_cli(campaign, sealed, capsys):
    assert integrity.main(["source-check", str(sealed.control)]) == 0
    assert "names no source tree" in capsys.readouterr().out
    root = campaign.tmp / "c2"
    (root / "proj").mkdir(parents=True)
    tree = campaign.tmp / "tree"
    (tree / "skill" / "references").mkdir(parents=True)
    (tree / "skill" / "references" / "levers.md").write_text("# levers\n")
    control = seal_campaign(root, {"CAMPAIGN_DIR": str(root)}, source=str(tree))
    assert integrity.main(["source-check", str(control)]) == 0
    assert "unchanged since launch" in capsys.readouterr().out
    (tree / "skill" / "references" / "levers.md").write_text("# levers\n- a rule\n")
    assert integrity.main(["source-check", str(control), "--where", "session end"]) == 0
    err = capsys.readouterr().err
    assert err.startswith("WARNING: the source tree %s changed" % tree), err
    assert "skill/references/levers.md: modified" in err
