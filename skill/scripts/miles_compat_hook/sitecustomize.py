# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# attune-hook-chain-member: loaded only by the hook chain (pod_hooks/sitecustomize.py)
"""miles compatibility hook: slime's profiler switches and log-timestamp format in miles processes.

make_run_yaml.py loads it through the hook chain (hook_chain.py; hook name miles_compat, default on). It does nothing
unless a miles module is imported. Two patches:
  - miles.utils.profile_utils._create_torch_profiler records shapes, stacks, memory and FLOPs only when
    SLIME_PROFILE_RECORD_SHAPES, SLIME_PROFILE_WITH_STACK, SLIME_PROFILE_MEMORY and SLIME_PROFILE_WITH_FLOPS are set
    (the slime defaults; miles turns all four on, which slows profiled steps and inflates traces). It replaces the
    function only when its source is miles v0.1.0's; otherwise it prints MILES_COMPAT_HOOK_SKIP and keeps miles' own.
  - miles.utils.logging_utils writes "[<time>] [<worker>] file:N - msg" instead of "[<time> <worker>] ...", so
    read_metrics.sh finds the log clock on perf lines as it does for slime.
Never raises into the host program.
"""
import hashlib
import importlib.abc
import importlib.util
import inspect
import os
import sys

_FLAGS = {"record_shapes": "SLIME_PROFILE_RECORD_SHAPES", "with_stack": "SLIME_PROFILE_WITH_STACK",
          "profile_memory": "SLIME_PROFILE_MEMORY", "with_flops": "SLIME_PROFILE_WITH_FLOPS"}
# sha256 of inspect.getsource(miles.utils.profile_utils._create_torch_profiler) at miles v0.1.0 (78527b91)
PROFILER_SHA256 = "55f252b49bde651f2e0250ff2faefc45f21522b7a6b132cda316b1e9aa72582c"


def _flag(name):
    return os.environ.get(name, "0").lower() not in ("0", "", "false", "no")


def _patch_profile_utils(m):
    try:
        sha = hashlib.sha256(inspect.getsource(m._create_torch_profiler).encode()).hexdigest()
    except Exception:
        sha = None
    if sha != PROFILER_SHA256:
        print("MILES_COMPAT_HOOK_SKIP miles.utils.profile_utils: _create_torch_profiler is not miles v0.1.0's; "
              "SLIME_PROFILE_* switches not applied", flush=True)
        return

    def _create_torch_profiler(args, name):
        import torch
        return torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=max(args.profile_step_start - 1, 0),
                warmup=1 if args.profile_step_start > 0 else 0,
                active=args.profile_step_end - args.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                args.tensorboard_dir, worker_name=f"{name}_rank_{torch.distributed.get_rank()}", use_gzip=True),
            **{k: _flag(v) for k, v in _FLAGS.items()},
        )
    m._create_torch_profiler = _create_torch_profiler


def _patch_logging_utils(m):
    orig = m.configure_logger_raw

    def configure_logger_raw(name: str = "") -> None:
        orig(name)
        try:
            import logging
            for h in logging.getLogger().handlers:
                fmt = getattr(h.formatter, "_fmt", None) or ""
                if fmt.startswith("[%(asctime)s.%(msecs)03d ") and "] " in fmt:
                    head, rest = fmt.split("] ", 1)
                    worker = head[len("[%(asctime)s.%(msecs)03d "):]
                    new = "[%(asctime)s.%(msecs)03d]" + (" [%s] " % worker if worker else " ") + rest
                    h.setFormatter(logging.Formatter(new, datefmt=h.formatter.datefmt))
        except Exception as e:
            print("MILES_COMPAT_HOOK_ERROR log format %s: %s" % (type(e).__name__, e), flush=True)
    m.configure_logger_raw = configure_logger_raw


_TARGETS = {"miles.utils.profile_utils": _patch_profile_utils, "miles.utils.logging_utils": _patch_logging_utils}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module
        patch = _TARGETS[name]

        def exec_module(module):
            orig_exec(module)
            try:
                patch(module)
            except Exception as e:
                print("MILES_COMPAT_HOOK_ERROR %s %s: %s" % (name, type(e).__name__, e), flush=True)

        spec.loader.exec_module = exec_module
        return spec


try:
    sys.meta_path.insert(0, _Finder())
except Exception:
    pass
