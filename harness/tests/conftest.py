# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixture: a campaign directory beside a protected shared volume, with the harness environment set.
Run: python3 -m pytest harness/tests"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HARNESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS / "hooks"))

CLEARED = ("CAMPAIGN_DIR", "AUTOTUNE_LEDGER", "AUTOTUNE_ALLOW_WEB", "AUTOTUNE_SHARED_MOUNT", "AUTOTUNE_BUDGET",
           "AUTOTUNE_GPUH_BUDGET", "AUTOTUNE_GPUH_LOG", "AUTOTUNE_MAX_STALL", "CLAUDE_CONFIG_DIR", "AUTOTUNE_AUTONOMOUS")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    tmp = tmp_path.resolve()
    root, shared, home = tmp / "campaign", tmp / "shared", tmp / "home"
    for d in (root / "proj", root / ".claude" / "skills", shared / "exp", home):
        d.mkdir(parents=True)
    (root / ".claude" / "skills" / "autotune-training-throughput").symlink_to(HARNESS.parent / "skill")
    for name in CLEARED:
        monkeypatch.delenv(name, raising=False)
    env = {"HOME": str(home), "AUTOTUNE_KUBECTL_CONTEXT": "c", "KUBECTL_NS": "ns", "AUTOTUNE_LOG_POD": "logpod",
           "AUTOTUNE_PROTECTED_PATHS": str(shared), "AUTOTUNE_PROJECT_ROOT": str(root)}
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return SimpleNamespace(root=root, shared=shared, home=home, env=env)
