# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every command-line tool prints its usage for --help and exits 0, without a cluster or optional dependencies.
Run: python3 -m pytest skill/scripts/tests"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PYTHON_CLIS = ["skill/scripts/analyze_profile.py", "skill/scripts/calibration.py", "skill/scripts/campaign.py",
               "skill/scripts/check_gpu_capacity.py", "skill/scripts/env_rules.py", "skill/scripts/hook_chain.py",
               "skill/scripts/ledger_append.py", "skill/scripts/make_run_yaml.py",
               "skill/scripts/run_gpuh.py", "skill/scripts/workload.py", "skill/scripts/correctness/frozen_args.py",
               "skill/scripts/correctness/layer1.py", "skill/scripts/correctness/make_parity_arm.py",
               "skill/scripts/correctness/parity_verdict.py", "skill/scripts/correctness/pair_from_runs.py",
               "skill/scripts/data/analyze_token_distribution.py", "skill/scripts/data/generate_fixed_sft.py",
               "skill/scripts/data/patch_glm5_mask.py", "skill/scripts/data/verify_rows.py"]
SHELL_CLIS = ["skill/scripts/poll_run.sh", "skill/scripts/read_metrics.sh", "skill/scripts/teardown_run.sh"]


@pytest.mark.parametrize("cli", PYTHON_CLIS + SHELL_CLIS)
def test_help(cli):
    cmd = ["bash", cli] if cli.endswith(".sh") else [sys.executable, cli]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run(cmd + ["--help"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=60)
    assert p.returncode == 0 and "Traceback" not in p.stderr, p.stderr
    assert "sage" in p.stdout, p.stdout   # "usage:" (argparse) or "Usage:" (the shell headers)
