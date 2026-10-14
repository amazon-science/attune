# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The profile_rank hook keeps one traced rank per pipeline stage on slime's and on miles' profiler module."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import hook_chain  # noqa: E402

PARALLEL_STATE = """import os
def model_parallel_is_initialized(): return True
def get_tensor_model_parallel_rank(): return int(os.environ["FAKE_TP_RANK"])
def get_context_parallel_rank(): return 0
def get_data_parallel_rank(): return 0
def get_pipeline_model_parallel_rank(): return 0
def get_pipeline_model_parallel_world_size(): return 1
"""
PROFILE_UTILS = """class TrainProfiler:
    def __init__(self, args):
        self._torch_profiler_overall = "profiler" if args.use_pytorch_profiler else None
def _profile_simple_loop(iterator, args, name):
    yield from iterator
"""
DRIVER = """import types, importlib
m = importlib.import_module("{pkg}.utils.profile_utils")
p = m.TrainProfiler(types.SimpleNamespace(use_pytorch_profiler=True))
print("TRACES", p._torch_profiler_overall is not None)
"""


@pytest.mark.parametrize("pkg", ["slime", "miles"])
@pytest.mark.parametrize("tp_rank, traces", [(0, True), (1, False)])
def test_one_rank_per_stage(tmp_path, pkg, tp_rank, traces):
    fake = tmp_path / "fake"
    for rel, text in {"megatron/__init__.py": "", "megatron/core/__init__.py": "",
                      "megatron/core/parallel_state.py": PARALLEL_STATE, f"{pkg}/__init__.py": "",
                      f"{pkg}/utils/__init__.py": "", f"{pkg}/utils/profile_utils.py": PROFILE_UTILS}.items():
        (fake / rel).parent.mkdir(parents=True, exist_ok=True)
        (fake / rel).write_text(text)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for name, text in hook_chain.render()[0].items():
        (hooks / name).write_text(text)
    (tmp_path / "driver.py").write_text(DRIVER.format(pkg=pkg))
    env = dict(os.environ, PYTHONPATH="%s:%s" % (hooks, fake), ATTUNE_HOOKS="profile_rank", FAKE_TP_RANK=str(tp_rank))
    env.pop("SLIME_PROFILE_ALL_RANKS", None)
    p = subprocess.run([sys.executable, str(tmp_path / "driver.py")], capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "PROFILE_RANK_HOOK_ERROR" not in p.stdout + p.stderr
    assert f"TRACES {traces}" in p.stdout
