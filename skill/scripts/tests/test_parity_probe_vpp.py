# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity probe counts once per microbatch for any virtual pipeline size, so the verdict compares arms of any VPP.

The probe is loaded through the real hook-chain loader into a process whose slime and Megatron modules are small
stand-ins: train_one_step runs an interleaved schedule in which every virtual chunk calls get_batch for every
microbatch, as Megatron's interleaved 1F1B does. Run: python3 -m pytest skill/scripts/tests
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))
import fixtures  # noqa: E402
import hook_chain  # noqa: E402

torch = pytest.importorskip("torch")

FAKE = {
    "megatron/__init__.py": "",
    "megatron/core/__init__.py": "",
    "megatron/core/parallel_state.py": """import os
VP = int(os.environ.get("FAKE_VP", "1")); PP_RANK = int(os.environ.get("FAKE_PP_RANK", "0"))
def is_pipeline_first_stage(ignore_virtual=True, vp_stage=None):
    return PP_RANK == 0
def get_tensor_model_parallel_rank(): return 0
def get_context_parallel_rank(): return 0
def get_data_parallel_group(): return None
def get_data_parallel_rank(): return 0
def get_virtual_pipeline_model_parallel_world_size(): return VP if VP > 1 else None
""",
    "slime/__init__.py": "",
    "slime/backends/__init__.py": "",
    "slime/backends/megatron_utils/__init__.py": "",
    "slime/backends/megatron_utils/data.py": """import torch
def get_batch(data_iterator, keys=None, pad_multiplier=128):
    rows = next(data_iterator)
    return {"unconcat_tokens": [torch.ones(L, dtype=torch.long) for L, _ in rows],
            "loss_masks": [torch.cat([torch.zeros(L - n), torch.ones(n)]) for L, n in rows]}
""",
    "slime/backends/megatron_utils/model.py": """from slime.backends.megatron_utils.data import get_batch
def train_one_step(args, rollout_id, step_id, data_iterator, model, *rest):
    chunks = data_iterator if isinstance(data_iterator, list) else CHUNKS
    for mb in range(NUM_MB):                 # interleaved schedule: each chunk sees every microbatch
        for it in chunks:
            get_batch(it)
    return {}, 0.0
NUM_MB = 0
CHUNKS = []
""",
}
DRIVER = """import os, sys, json
import slime.backends.megatron_utils.model as m
VP, M = int(os.environ["FAKE_VP"]), 6
mbs = [[(16, 10 + i), (16, 3)] for i in range(M)]      # microbatch i: two rows of 16 tokens
def it(): return iter(list(mbs))
m.NUM_MB = M
chunks = [it() for _ in range(VP)]
m.train_one_step(None, 0, 0, chunks, None)              # normal path: data_iterator is the per-chunk list
m.CHUNKS = [it() for _ in range(VP)]
m.train_one_step(None, 1, 0, None, None)                # iterator not identifiable: divide by VP
"""


def _run(tmp_path, vp, pp_rank=0):
    data, _ = hook_chain.render()
    hooks = tmp_path / "hooks"
    fake = tmp_path / "fake"
    if not hooks.exists():
        hooks.mkdir()
        for k, v in data.items():
            (hooks / k).write_text(v)
        for rel, text in FAKE.items():
            (fake / rel).parent.mkdir(parents=True, exist_ok=True)
            (fake / rel).write_text(text)
        (tmp_path / "driver.py").write_text(DRIVER)
    env = dict(os.environ, PYTHONPATH="%s:%s" % (hooks, fake), ATTUNE_HOOKS="parity_probe", FAKE_VP=str(vp), FAKE_PP_RANK=str(pp_rank))
    p = subprocess.run([sys.executable, str(tmp_path / "driver.py")], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stderr
    assert "PARITY_PROBE_ERROR" not in p.stdout + p.stderr, p.stdout + p.stderr
    return [json.loads(line.split(": ", 1)[1]) for line in p.stdout.splitlines() if line.startswith("PARITY_COUNTS")]


@pytest.mark.parametrize("vp", [1, 2, 4, 8])
def test_counts_once_per_microbatch(tmp_path, vp):
    lines = _run(tmp_path, vp)
    want = {"rows": 12, "tokens": 12 * 16, "loss_tokens": sum(10 + i + 3 for i in range(6)), "microbatches": 6}
    assert len(lines) == 2
    assert lines[0] == dict(want, basis="chunk0")
    assert lines[1] == dict(want, basis="vp_divided")


def test_only_first_physical_stage_counts(tmp_path):
    assert _run(tmp_path, 4, pp_rank=1) == []


def _pair(tmp, base_counts, win_counts, base_args, win_args):
    pair = tmp / "pair"
    pair.mkdir(exist_ok=True)
    for name, counts, args in (("base", base_counts, base_args), ("win", win_counts, win_args)):
        text = fixtures.train_log(args=args, counts=False)
        extra = "".join("PARITY_COUNTS rollout %d step %d: %s\n" % (t, t, json.dumps(counts)) for t in range(20))
        (pair / (name + ".log")).write_text(text + extra)
    (pair / "gpus.json").write_text(json.dumps({"base": 96, "win": 96}))
    env = {k: v for k, v in os.environ.items() if k != "AUTOTUNE_FROZEN_ARGS"}
    p = subprocess.run([sys.executable, str(SCRIPTS / "correctness" / "parity_verdict.py"), str(pair), "--json",
                        str(pair / "v.json")], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads((pair / "v.json").read_text())[0]


C = {"rows": 48, "tokens": 48 * 3072, "loss_tokens": 48 * 3000, "microbatches": 24}


def test_verdict_vp6_vs_vp3_per_microbatch(tmp_path):
    r = _pair(tmp_path, dict(C, basis="chunk0"), dict(C, basis="chunk0"),
              {"virtual_pipeline_model_parallel_size": "6"}, {"virtual_pipeline_model_parallel_size": "3"})
    assert r["counts_identical"] and r["L2"], r


def test_miles_layout_counts(tmp_path):
    """miles keeps get_batch in training_utils.data and imports it into megatron_utils.model by name."""
    for rel, text in FAKE.items():
        if rel.startswith("slime/"):
            rel = rel.replace("slime/backends/megatron_utils/data.py", "slime/backends/training_utils/data.py")
            text = text.replace("from slime.backends.megatron_utils.data import get_batch",
                                "from ..training_utils.data import get_batch")
            (tmp_path / "fake" / rel.replace("slime/", "miles/", 1)).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / "fake" / rel.replace("slime/", "miles/", 1)).write_text(text)
    (tmp_path / "fake" / "miles/backends/training_utils/__init__.py").write_text("")
    for rel, text in FAKE.items():
        if not rel.startswith("slime/"):
            (tmp_path / "fake" / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / "fake" / rel).write_text(text)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for k, v in hook_chain.render()[0].items():
        (hooks / k).write_text(v)
    (tmp_path / "driver.py").write_text(DRIVER.replace("slime.backends", "miles.backends"))
    env = dict(os.environ, PYTHONPATH="%s:%s" % (hooks, tmp_path / "fake"), ATTUNE_HOOKS="parity_probe", FAKE_VP="2",
               FAKE_PP_RANK="0")
    p = subprocess.run([sys.executable, str(tmp_path / "driver.py")], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stderr
    lines = [json.loads(line.split(": ", 1)[1]) for line in p.stdout.splitlines() if line.startswith("PARITY_COUNTS")]
    want = {"rows": 12, "tokens": 12 * 16, "loss_tokens": sum(10 + i + 3 for i in range(6)), "microbatches": 6}
    assert lines == [dict(want, basis="chunk0"), dict(want, basis="vp_divided")]
