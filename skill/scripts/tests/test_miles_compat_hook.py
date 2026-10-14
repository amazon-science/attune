# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""miles_compat hook: slime's profiler switches and a log clock that the released read_metrics.sh parses."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import hook_chain  # noqa: E402

FAKE_TORCH = {
    "torch/__init__.py": "from . import profiler, distributed\n",
    "torch/distributed.py": "def get_rank(): return 0\n",
    "torch/profiler.py": """def schedule(**k): return k
def tensorboard_trace_handler(*a, **k): return None
def profile(**k): return k
""",
}
MILES_V010_PROFILER = """def _create_torch_profiler(args, name):
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            # TODO the train_actor and train_log_probs ones may need to have different args to control step
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.tensorboard_dir,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )
"""
FAKE_MILES = {
    "miles/__init__.py": "",
    "miles/utils/__init__.py": "",
    "miles/utils/profile_utils.py": "import torch\n\n\n" + MILES_V010_PROFILER,
    "miles/utils/logging_utils.py": """import logging
def configure_logger_raw(name: str = "") -> None:
    logging.basicConfig(level=logging.INFO, format=f"[%(asctime)s.%(msecs)03d {name}] %(filename)s:%(lineno)d - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", force=True)
def configure_logger(args, *, source=None) -> None:
    configure_logger_raw("actor_cell0_rank0")
""",
    "train_metric_utils.py": """import logging
def log(): logging.getLogger(__name__).info("perf 0: {'perf/actor_train_tok_per_s': 1.0}")
""",
}
DRIVER = """import json, sys, types
from miles.utils import profile_utils, logging_utils
args = types.SimpleNamespace(profile_step_start=15, profile_step_end=17, tensorboard_dir="/tmp")
k = profile_utils._create_torch_profiler(args, "train_overall")
print("PROFILER " + json.dumps({x: k[x] for x in ("record_shapes", "with_stack", "profile_memory", "with_flops")}))
logging_utils.configure_logger(None)
import train_metric_utils; train_metric_utils.log()
"""


def _run(tmp_path, fake_miles=None, **env):
    for rel, text in {**FAKE_TORCH, **FAKE_MILES, **(fake_miles or {})}.items():
        (tmp_path / "fake" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "fake" / rel).write_text(text)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for name, text in hook_chain.render()[0].items():
        (hooks / name).write_text(text)
    (tmp_path / "driver.py").write_text(DRIVER)
    e = {k: v for k, v in os.environ.items() if not k.startswith("SLIME_PROFILE_")}
    e.update(PYTHONPATH="%s:%s" % (hooks, tmp_path / "fake"), ATTUNE_HOOKS="", **env)
    p = subprocess.run([sys.executable, str(tmp_path / "driver.py")], capture_output=True, text=True, env=e, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "MILES_COMPAT_HOOK_ERROR" not in p.stdout + p.stderr
    return p


def test_profiler_records_only_what_slime_switches_ask_for(tmp_path):
    p = _run(tmp_path)
    off = json.loads(p.stdout.split("PROFILER ", 1)[1].splitlines()[0])
    assert off == {"record_shapes": False, "with_stack": False, "profile_memory": False, "with_flops": False}


@pytest.mark.parametrize("var, key", [("SLIME_PROFILE_WITH_STACK", "with_stack"), ("SLIME_PROFILE_MEMORY", "profile_memory")])
def test_profiler_switch_turns_one_on(tmp_path, var, key):
    p = _run(tmp_path, **{var: "1"})
    got = json.loads(p.stdout.split("PROFILER ", 1)[1].splitlines()[0])
    assert got[key] is True and sum(got.values()) == 1


def test_perf_line_clock_matches_the_released_read_metrics_pattern(tmp_path):
    p = _run(tmp_path)
    line = [x for x in p.stderr.splitlines() if "perf 0:" in x][0]
    src = (HERE.parent / "read_metrics.sh").read_text()
    pat = re.compile(re.search(r'pat_ts = re\.compile\(r"(.*)"\)', src).group(1))
    assert pat.search(line), line
    assert "[actor_cell0_rank0] train_metric_utils.py:" in line


def test_profiler_factory_other_than_v010_is_left_alone(tmp_path):
    changed = MILES_V010_PROFILER.replace("with_flops=True", "with_flops=False")
    p = _run(tmp_path, fake_miles={"miles/utils/profile_utils.py": "import torch\n\n\n" + changed})
    got = json.loads(p.stdout.split("PROFILER ", 1)[1].splitlines()[0])
    assert got == {"record_shapes": True, "with_stack": True, "profile_memory": True, "with_flops": False}
    assert "MILES_COMPAT_HOOK_SKIP miles.utils.profile_utils" in p.stdout
