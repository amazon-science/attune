# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# attune-hook-chain-member: loaded only by the hook chain (pod_hooks/sitecustomize.py)
"""Parity probe: log exact per-step token / loss-mask / row counts for the correctness gate.

make_parity_arm.py loads it through the hook chain (hook_chain.py; hook name parity_probe) in every Python process of a
check arm; PARITY_PROBE=0 disables it. Wraps slime's or miles' get_batch (to count) and train_one_step (to reset,
reduce across data-parallel replicas, and print one line per step):
  PARITY_COUNTS rollout <r> step <s>: {"rows": R, "tokens": T, "loss_tokens": L, "microbatches": M, "basis": B}
Counts are taken once per microbatch on one rank per data-parallel replica (physical first pipeline stage, TP rank 0,
CP rank 0) from pre-CP-slicing data, so they are independent of TP, PP, VPP, CP and DP. With interleaved pipelining
every virtual chunk on that rank calls get_batch for every microbatch; only calls on chunk 0's data iterator (the
first element of train_one_step's data_iterator list) are counted (basis "chunk0"). If the iterator cannot be
identified, every call is counted and the totals are divided by the virtual pipeline size (basis "vp_divided").
Never raises into training: any error disables the probe and prints PARITY_PROBE_ERROR once.
"""
import importlib.abc
import importlib.util
import json
import os
import sys

_STATE = {"active": False, "rows": 0, "tokens": 0, "loss_tokens": 0, "mb": 0, "err": False, "chunk0": None}

def _err(e):
    if not _STATE["err"]:
        _STATE["err"] = True
        print(f"PARITY_PROBE_ERROR {type(e).__name__}: {e}", flush=True)

def _counting_rank():
    from megatron.core import parallel_state as mpu
    return (mpu.is_pipeline_first_stage(ignore_virtual=True) and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0)


def _vp_size():
    try:
        from megatron.core import parallel_state as mpu
        return int(mpu.get_virtual_pipeline_model_parallel_world_size() or 1)
    except Exception:
        return 1

def _wrap_get_batch(orig):
    def get_batch(*a, **k):
        batch = orig(*a, **k)
        if _STATE["active"] and not _STATE["err"]:
            try:
                di = a[0] if a else k.get("data_iterator")
                if _STATE["chunk0"] is not None and id(di) != _STATE["chunk0"]:
                    return batch  # a later virtual chunk re-reading the same microbatch
                if _counting_rank():
                    toks = batch.get("unconcat_tokens")
                    masks = batch.get("loss_masks")
                    if toks is not None:
                        _STATE["rows"] += len(toks)
                        _STATE["tokens"] += int(sum(int(t.numel()) for t in toks))
                    if masks is not None:
                        _STATE["loss_tokens"] += int(sum(int(m.sum().item()) for m in masks))
                    _STATE["mb"] += 1
            except Exception as e:
                _err(e)
        return batch
    get_batch._parity_wrapped = True
    return get_batch

def _wrap_train_one_step(orig):
    def train_one_step(*a, **k):
        it = k.get("data_iterator", a[3] if len(a) > 3 else None)
        first = it[0] if isinstance(it, (list, tuple)) and it else it
        _STATE.update(active=True, rows=0, tokens=0, loss_tokens=0, mb=0, chunk0=id(first) if first is not None else None)
        try:
            return orig(*a, **k)
        finally:
            _STATE["active"] = False
            if not _STATE["err"]:
                try:
                    import torch
                    from megatron.core import parallel_state as mpu
                    if _counting_rank():
                        vals = [_STATE["rows"], _STATE["tokens"], _STATE["loss_tokens"], _STATE["mb"]]
                        basis = "chunk0"
                        if _STATE["chunk0"] is None:
                            vp, basis = _vp_size(), "vp_divided"
                            if any(v % vp for v in vals):
                                raise ValueError("counts %s are not multiples of the virtual pipeline size %d" % (vals, vp))
                            vals = [v // vp for v in vals]
                        dist = torch.distributed.is_available() and torch.distributed.is_initialized()
                        t = torch.tensor(vals, dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu")
                        if dist:
                            torch.distributed.all_reduce(t, group=mpu.get_data_parallel_group())
                        if mpu.get_data_parallel_rank() == 0:
                            rid = k.get("rollout_id", a[1] if len(a) > 1 else -1)
                            sid = k.get("step_id", a[2] if len(a) > 2 else -1)
                            rows, tokens, loss_tokens, mb = (int(x) for x in t.tolist())
                            print(f"PARITY_COUNTS rollout {rid} step {sid}: " + json.dumps(
                                {"rows": rows, "tokens": tokens, "loss_tokens": loss_tokens, "microbatches": mb,
                                 "basis": basis}), flush=True)
                except Exception as e:
                    _err(e)
    train_one_step._parity_wrapped = True
    return train_one_step

_TARGETS = {
    "slime.backends.megatron_utils.data": ("get_batch", _wrap_get_batch),
    "slime.backends.megatron_utils.model": ("train_one_step", _wrap_train_one_step),
    "miles.backends.training_utils.data": ("get_batch", _wrap_get_batch),  # model.py imports it by name, after this
    "miles.backends.megatron_utils.model": ("train_one_step", _wrap_train_one_step),
}

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module
        attr, wrap = _TARGETS[name]
        def exec_module(module):
            orig_exec(module)
            try:
                f = getattr(module, attr, None)
                if f is not None and not getattr(f, "_parity_wrapped", False):
                    setattr(module, attr, wrap(f))
            except Exception as e:
                _err(e)
        spec.loader.exec_module = exec_module
        return spec

if os.environ.get("PARITY_PROBE", "1") == "1":
    sys.meta_path.insert(0, _Finder())
