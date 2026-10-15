# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: a campaign directory beside a shared volume, with the harness environment set, and a sealed
campaign (a control directory with copies of a skill and harness, config.env and the integrity record).
Run: python3 -m pytest harness/tests"""
import os
import shlex
import shutil
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HARNESS = Path(__file__).resolve().parent.parent
REPO = HARNESS.parent
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS / "hooks"))

import integrity  # noqa: E402

CLEARED = ("CAMPAIGN_DIR", "AUTOTUNE_LEDGER", "AUTOTUNE_ALLOW_WEB", "AUTOTUNE_SHARED_MOUNT", "AUTOTUNE_BUDGET",
           "AUTOTUNE_GPUH_BUDGET", "AUTOTUNE_GPUH_LOG", "AUTOTUNE_MAX_STALL", "CLAUDE_CONFIG_DIR",
           "AUTOTUNE_AUTONOMOUS", "ATTUNE_CONTROL_DIR", "ATTUNE_RUN_PREFIX", "AUTOTUNE_RULES_DIR", "AUTOTUNE_FROZEN_ARGS",
           "AUTOTUNE_CALIBRATION")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    tmp = tmp_path.resolve()
    root, shared, home = tmp / "campaign", tmp / "shared", tmp / "home"
    for d in (root / "proj", root / ".claude" / "skills", shared / "exp", home):
        d.mkdir(parents=True)
    for name in CLEARED:
        monkeypatch.delenv(name, raising=False)
    env = {"HOME": str(home), "AUTOTUNE_KUBECTL_CONTEXT": "c", "KUBECTL_NS": "ns", "AUTOTUNE_LOG_POD": "logpod",
           "ATTUNE_RUN_PREFIX": "alice-tune-", "AUTOTUNE_PROJECT_ROOT": str(root)}
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    yield SimpleNamespace(root=root, shared=shared, home=home, env=env, tmp=tmp)
    unlock(tmp)


def unlock(path):
    """Give every directory under `path` its write bit back, so the tmp_path cleanup can remove it."""
    for d, subdirs, _ in os.walk(path):
        os.chmod(d, os.stat(d).st_mode | stat.S_IWUSR)


TINY_SKILL = {"SKILL.md": "# skill\n", "references/levers.md": "# levers\n- a rule\n",
              "references/measurement.md": "# measurement\n", "scripts/read_metrics.sh": "echo metrics\n",
              "scripts/teardown_run.sh": 'echo "$1" >> "$TEARDOWN_LOG"; [ "$1" != alice-tune-stuck ]\n',
              "scripts/correctness/rules/numerics_list.yaml": "numeric: []\n",
              "past_tuning_projects/README.md": "# past\n",
              "past_tuning_projects/proj_summary.md": "# an earlier campaign\n"}
CAMPAIGN_ID = "alice-tune-20261015-120000-3fa2"
TINY_HARNESS = {"integrity.py": "# copy\n", "hooks/stop_gate.py": "# copy\n", "settings.template.json": "{}\n"}


def _tree(root, files):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def seal_campaign(root, settings, skill=None, harness=None):
    """A campaign as launch_campaign.sh leaves it: <root>/.attune holding read-only copies of the skill and harness
    (TINY_* trees, or the given source directories), config.env with `settings`, the rendered settings and goal
    prompt, and the integrity record. Returns the control directory."""
    control = root / ".attune"
    control.mkdir()
    for name, src, tiny in (("skill", skill, TINY_SKILL), ("harness", harness, TINY_HARNESS)):
        if src:
            shutil.copytree(src, control / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        else:
            _tree(control / name, tiny)
    (control / "config.env").write_text("".join("%s=%s\n" % (k, shlex.quote(v)) for k, v in settings.items()))
    (root / ".claude").mkdir(exist_ok=True)
    (root / ".claude" / "settings.json").write_text('{"hooks": {}}\n')
    (root / "goal_prompt.rendered.txt").write_text("/goal tune\n")
    walked = list(os.walk(control / "skill")) + list(os.walk(control / "harness"))
    for path in [os.path.join(sub, n) for sub, _, names in walked for n in names] + [sub for sub, _, _ in walked]:
        os.chmod(path, os.stat(path).st_mode & ~0o222)  # chmod -R a-w, as the launcher does
    os.chmod(control / "skill" / "references" / "levers.md", 0o644)
    os.chmod(control / "skill" / "past_tuning_projects", 0o755)
    integrity.seal(str(control))
    os.chmod(control, 0o555)
    return control


@pytest.fixture
def sealed(campaign, monkeypatch):
    """A sealed campaign with tiny copies; the hooks' environment points at its control directory."""
    settings = {"CAMPAIGN_DIR": str(campaign.root), "AUTOTUNE_PROJECT_ROOT": str(campaign.root),
                "AUTOTUNE_GPUH_LOG": str(campaign.root / "gpuh.jsonl"), "ATTUNE_CAMPAIGN_ID": CAMPAIGN_ID}
    control = seal_campaign(campaign.root, settings)
    monkeypatch.setenv("ATTUNE_CONTROL_DIR", str(control))
    monkeypatch.setenv("TEARDOWN_LOG", str(campaign.tmp / "teardown.log"))
    campaign.torn = lambda: (campaign.tmp / "teardown.log").read_text().split() if (
        campaign.tmp / "teardown.log").exists() else []
    monkeypatch.setenv("CAMPAIGN_DIR", str(campaign.root))
    campaign.control = control
    campaign.campaign_id = CAMPAIGN_ID
    campaign.skill = control / "skill"
    campaign.harness = control / "harness"
    campaign.ledger = campaign.root / "proj" / integrity.LEDGER_NAME
    return campaign
