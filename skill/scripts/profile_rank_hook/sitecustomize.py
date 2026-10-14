# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# attune-hook-chain-member: loaded only by the hook chain (pod_hooks/sitecustomize.py)
"""Profiler rank selection for slime's or miles' torch profiler: one traced rank per pipeline stage.

make_run_yaml.py loads it through the hook chain (hook_chain.py; hook name profile_rank). On import of
slime.utils.profile_utils or miles.utils.profile_utils (same TrainProfiler API) it limits torch-profiler creation to
ranks whose tensor-, context- and
data-parallel ranks are all 0, i.e. the first rank of each pipeline stage. Trace options, steps and output dir
are unchanged. SLIME_PROFILE_ALL_RANKS=1 in the pod env restores stock behavior (every rank traces).
Never raises into the host program.
"""
import importlib.abc
import importlib.util
import os
import sys

_TARGETS = ("slime.utils.profile_utils", "miles.utils.profile_utils")


def _selected():
    """(selected, description). Falls back to stock behavior (selected) when topology is unknown."""
    if os.environ.get("SLIME_PROFILE_ALL_RANKS") == "1":
        return True, "all ranks (SLIME_PROFILE_ALL_RANKS=1)"
    try:
        from megatron.core import parallel_state as mpu
        if not mpu.model_parallel_is_initialized():
            return True, "parallel state not initialized; stock behavior"
        sel = (mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0
               and mpu.get_data_parallel_rank() == 0)
        return sel, "pp stage %d/%d" % (mpu.get_pipeline_model_parallel_rank(),
                                        mpu.get_pipeline_model_parallel_world_size())
    except Exception as e:
        return True, "selection failed (%s); stock behavior" % type(e).__name__


def _rank():
    try:
        import torch
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    except Exception:
        return -1


def _patch(m):
    TP = m.TrainProfiler
    orig_init = TP.__init__
    orig_loop = m._profile_simple_loop

    def __init__(self, args, *a, **k):
        orig_init(self, args, *a, **k)
        try:
            if getattr(args, "use_pytorch_profiler", False):
                sel, why = _selected()
                if not sel:
                    self._torch_profiler_overall = None
                else:
                    print("PROFILE_RANK_HOOK: rank %d traces (%s)" % (_rank(), why), flush=True)
        except Exception as e:
            print("PROFILE_RANK_HOOK_ERROR init %s: %s" % (type(e).__name__, e), flush=True)

    def _profile_simple_loop(iterator, args, name):
        try:
            sel = _selected()[0]
        except Exception:
            sel = True
        if not sel:
            yield from iterator
            return
        yield from orig_loop(iterator, args, name)

    TP.__init__ = __init__
    m._profile_simple_loop = _profile_simple_loop


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        sys.meta_path.remove(self)  # one-shot: find the real spec, then wrap its loader
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            return None
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                _patch(module)
            except Exception as e:
                print("PROFILE_RANK_HOOK_ERROR patch %s: %s" % (type(e).__name__, e), flush=True)

        spec.loader.exec_module = exec_module
        return spec


def _main():
    try:
        if not any(t in sys.modules for t in _TARGETS):
            sys.meta_path.insert(0, _Finder())
    except Exception:
        pass


_main()
