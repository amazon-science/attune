# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""In a campaign (a sealed copy of the skill beside the launcher's record) the scripts take their settings from the
campaign's config.env and refuse an environment that disagrees; ledger_append.py appends once, and nothing after an
integrity violation; run_gpuh.py's log only grows.
Run: python3 -m pytest skill/scripts/tests"""
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parents[1]
SETTINGS = {"AUTOTUNE_KUBECTL_CONTEXT": "ctx", "KUBECTL_NS": "ns", "AUTOTUNE_USER": "alice",
            "ATTUNE_RUN_PREFIX": "alice-tune-", "AUTOTUNE_LOG_POD": "logpod", "AUTOTUNE_BUDGET": "20",
            "AUTOTUNE_GPUH_BUDGET": ""}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


@pytest.fixture
def camp(tmp_path):
    """run(script, *args, env=..., stdin=...) runs a script of the campaign's skill copy in the campaign directory."""
    root = tmp_path.resolve() / "campaign"
    control = root / ".attune"
    (root / "proj").mkdir(parents=True)
    shutil.copytree(REPO / "skill", control / "skill", ignore=IGNORE)
    shutil.copytree(REPO / "harness", control / "harness", ignore=IGNORE)
    settings = dict(SETTINGS, CAMPAIGN_DIR=str(root), AUTOTUNE_PROJECT_ROOT=str(root),
                    AUTOTUNE_GPUH_LOG=str(root / "gpuh.jsonl"))
    (control / "config.env").write_text("".join("%s=%s\n" % (k, shlex.quote(v)) for k, v in settings.items()))
    subprocess.run([sys.executable, str(control / "harness" / "integrity.py"), "seal", str(control)], check=True,
                   capture_output=True)
    base = {k: v for k, v in os.environ.items() if not k.startswith(("AUTOTUNE_", "ATTUNE_", "KUBECTL_"))}
    base.pop("CAMPAIGN_DIR", None)

    def run(script, *args, env=None, stdin=None):
        path = control / "skill" / "scripts" / script
        cmd = ["bash", str(path)] if script.endswith(".sh") else [sys.executable, str(path)]
        full = {**base, **settings, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
        return subprocess.run(cmd + list(args), capture_output=True, text=True, env=full, cwd=root, input=stdin,
                              timeout=60)
    run.root, run.control, run.settings = root, control, settings
    return run


@pytest.mark.parametrize("script,args", [("teardown_run.sh", ["alice-tune-r1"]), ("poll_run.sh", ["alice-tune-r1"]),
                                         ("read_metrics.sh", ["/x/train.log", "5", "19", "8"]),
                                         ("run_gpuh.py", ["--total"]), ("workload.py", []),
                                         ("make_run_yaml.py", ["--help"]), ("check_gpu_capacity.py", [])])
def test_a_changed_setting_is_refused(camp, script, args):
    for env in ({"KUBECTL_NS": "default"}, {"ATTUNE_RUN_PREFIX": "alice-"}, {"AUTOTUNE_GPUH_BUDGET": "1000"}):
        p = camp(script, *args, env=env)
        name = next(iter(env))
        assert p.returncode == 2 and name in p.stderr and "config.env" in p.stderr, (script, env, p.stderr)


def test_the_campaigns_settings_pass_and_fill_what_the_environment_lacks(camp):
    p = camp("workload.py")
    assert p.returncode == 0 and json.loads(p.stdout)["num_steps"] == 20, p.stderr
    assert camp("campaign.py", "check-settings").returncode == 0
    p = camp("campaign.py", "check-settings", "--shell", env={"KUBECTL_NS": "", "AUTOTUNE_LOG_POD": ""})
    assert p.returncode == 0 and p.stdout.splitlines() == ["export AUTOTUNE_LOG_POD=logpod", "export KUBECTL_NS=ns"]
    # an empty or unset value takes the campaign's, so the example config's empty AUTOTUNE_GPUH_LOG= is no override
    p = camp("run_gpuh.py", "--total", env={"AUTOTUNE_GPUH_LOG": ""})
    assert p.returncode == 0 and json.loads(p.stdout)["runs"] == 0, p.stderr
    p = camp("poll_run.sh", "alice-tune-r1", env={"KUBECTL_NS": "", "AUTOTUNE_LOG_POD": "",
                                                   "PATH": "/nonexistent:" + os.environ["PATH"]})
    assert "set KUBECTL_NS" not in p.stderr and "AUTOTUNE_LOG_POD" not in p.stderr, p.stderr


def test_outside_a_campaign_the_environment_decides():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AUTOTUNE_", "ATTUNE_", "KUBECTL_"))}
    p = subprocess.run([sys.executable, str(SCRIPTS / "campaign.py"), "check-settings"], capture_output=True,
                       text=True, env=dict(env, KUBECTL_NS="anything"), check=False)
    assert p.returncode == 0


def test_ledger_append(camp):
    ledger = "proj/AUTOTUNE_LEDGER.jsonl"
    row = {"run_id": "alice-tune-r1", "kind": "ranked", "status": "completed", "verdict": "tie"}
    out = json.loads(camp("ledger_append.py", ledger, json.dumps(row)).stdout)
    assert out["appended"] is True
    out = json.loads(camp("ledger_append.py", ledger, "-", stdin=json.dumps(dict(row, verdict="win"))).stdout)
    assert out["appended"] is False and "already" in out["note"]
    assert camp("ledger_append.py", ledger, "[1]").returncode == 2
    assert camp("ledger_append.py", ledger, "{not json").returncode == 2
    lines = (camp.root / ledger).read_text().splitlines()
    assert [json.loads(x)["verdict"] for x in lines] == ["tie"]


def test_ledger_append_refuses_and_tears_down_after_a_violation(camp, tmp_path):
    ledger = camp.root / "proj" / "AUTOTUNE_LEDGER.jsonl"
    row = {"run_id": "alice-tune-r1", "kind": "ranked", "status": "completed", "verdict": "lose"}
    assert camp("ledger_append.py", str(ledger), json.dumps(row)).returncode == 0
    (camp.root / "proj" / "tracked_jobs.jsonl").write_text('{"run_id": "alice-tune-r2"}\n')
    ledger.write_text(json.dumps(dict(row, verdict="win", note="rescored")) + "\n")   # the seen row is rewritten
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "kubectl").write_text("#!/bin/sh\necho 'Error from server (NotFound): x' >&2; exit 1\n")
    (bin_dir / "kubectl").chmod(0o755)
    p = camp("ledger_append.py", str(ledger), json.dumps(dict(row, run_id="alice-tune-r2")),
             env={"PATH": "%s:%s" % (bin_dir, os.environ["PATH"])})
    out = json.loads(p.stdout)
    assert p.returncode == 3 and out["appended"] is False and "rewritten" in out["violations"]
    assert out["teardown"] == "live runs torn down: alice-tune-r2"
    assert len(ledger.read_text().splitlines()) == 1
    assert (camp.control / "integrity_violation.json").exists()


def test_ledger_append_exits_3_on_any_error(tmp_path):
    p = subprocess.run([sys.executable, str(SCRIPTS / "ledger_append.py"), str(tmp_path), '{"run_id": "x"}'],
                       capture_output=True, text=True, check=False)
    assert p.returncode == 3 and json.loads(p.stdout)["appended"] is False and "IsADirectoryError" in p.stdout


def test_run_gpuh_log_only_grows(camp):
    log = camp.root / "gpuh.jsonl"
    sys.path.insert(0, str(SCRIPTS))
    import run_gpuh
    run_gpuh.log_launch(str(log), "alice-tune-r1", 8)
    assert run_gpuh.log_launch(str(log), "alice-tune-r1", 8)["note"] == "already logged"
    first = log.read_text()
    done = {"run_id": "alice-tune-r1", "num_gpus": 8, "admitted": "t0", "end": "t1", "gpuh": 4.0, "final": True}
    row = run_gpuh.log_record(str(log), done, "done")
    assert log.read_text().startswith(first) and row["cum_gpuh"] == 4.0 and row["submitted"]
    run_gpuh.log_launch(str(log), "alice-tune-r2", 8)
    row = run_gpuh.log_record(str(log), dict(done, run_id="alice-tune-r2", gpuh=2.5), "oom")
    assert row["cum_gpuh"] == 6.5
    row = run_gpuh.log_record(str(log), dict(done, gpuh=5.0), "done")   # a re-measure replaces the run's charge
    assert row["cum_gpuh"] == 7.5
    assert len(log.read_text().splitlines()) == 5
    p = camp("run_gpuh.py", "--total")
    assert json.loads(p.stdout) == {"cum_gpuh": 7.5, "runs": 2, "open": []}, p.stderr
    assert camp("campaign.py", "check-settings").returncode == 0
    p = subprocess.run([sys.executable, str(camp.control / "harness" / "integrity.py"), "check", str(camp.control)],
                       capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr


def test_the_aids_run_from_a_copy_in_the_campaign_directory(camp):
    """SKILL.md lets the agent copy check_gpu_capacity.py and analyze_profile.py (with profile_analyzer/)."""
    aids = camp.root / "proj" / "aids"
    aids.mkdir()
    scripts = camp.control / "skill" / "scripts"
    shutil.copy(scripts / "check_gpu_capacity.py", aids)
    shutil.copy(scripts / "analyze_profile.py", aids)
    shutil.copytree(scripts / "profile_analyzer", aids / "profile_analyzer", ignore=IGNORE)
    for aid in ("check_gpu_capacity.py", "analyze_profile.py"):
        p = subprocess.run([sys.executable, str(aids / aid), "--help"], capture_output=True, text=True, check=False,
                           cwd=camp.root)
        assert p.returncode == 0 and "usage" in p.stdout, (aid, p.stderr)
