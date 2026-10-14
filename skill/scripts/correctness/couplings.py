# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conditional frozen rules ("couplings") of a frozen-arguments file, which the skill spec format (eq/in/products)
cannot express. One evaluator for manifests (pre-launch) and resolved-argument dumps (layer 1).

The `require` coupling, e.g. `--tp-comm-overlap` runs only with
  - `--tp-comm-bootstrap-backend gloo` (slime's ReloadableProcessGroup breaks the nccl bootstrap),
  - `--sequence-parallel` (Megatron arguments.py:914-915),
  - `--seq-length` = the workload's row length and exact buffer fill: Megatron sizes the userbuffers as
    seq_length x micro_batch_size / CP (initialize.py:264-274; slime passes both to the schedule, model.py:478,648) and
    every microbatch must fill them exactly, so with dynamic batching
    max_tokens_per_gpu x CP == seq_length x micro_batch_size (fixed-length rows; slime splits bins only when their
    count is not a multiple of DP x microbatch group, dp_schedule.py:167-176),
  - the recipe's published userbuffers config (`--tp-comm-overlap-cfg` basename, or none = TE defaults). Its content is
    per-GEMM method/SM settings, not shapes, so any layout that satisfies the rules above may use it.
frozen_args.py calls violations() with a getter over manifest flags (pre-launch) or a resolved-argument dump (layer 1).
"""
import os
import re as _re

TRUE = ("true", "1", "yes")


def _truthy(v):
    return v is True or (isinstance(v, str) and v.strip().strip("'\"").lower() in TRUE)


def _num(v, default=None):
    if v is None or v is True:
        return default
    try:
        return float(str(v).strip("'\""))
    except ValueError:
        return default


def _vpp_middle_stages(c, get):
    """Virtual stages need a middle pipeline stage when both uneven end stages are set
    (Megatron get_transformer_layer_offset divides the middle layers by pp minus the set end stages)."""
    vpp = _num(get("num_virtual_stages_per_pipeline_rank"))
    if vpp is None and _num(get("num_layers_per_virtual_pipeline_stage")) is not None:
        vpp = 2.0
    if vpp is None or vpp <= 1:
        return []
    pp = _num(get("pipeline_model_parallel_size"), 1.0)
    ends = sum(_num(get(n)) is not None for n in ("decoder_first_pipeline_num_layers", "decoder_last_pipeline_num_layers"))
    if pp - ends <= 0:
        return [f"{c['id']}: virtual pipeline stages need a middle pipeline stage; PP {pp:g} with {ends} uneven end "
                f"stage(s) has none (Megatron transformer_layer.py:137 divides by it)"]
    return []


def _vpp_off_at_cp(c, get):
    """At CP >= cp_min the interleaved schedule can hang at step 0 (NCCL unbatched P2P) while the same layout with VPP
    off trains normally. Virtual stages must be off at that CP."""
    cp = _num(get("context_parallel_size"), 1.0)
    vpp = _num(get("num_virtual_stages_per_pipeline_rank"))
    if vpp is None and _num(get("num_layers_per_virtual_pipeline_stage")) is not None:
        vpp = 2.0
    if cp >= float(c["cp_min"]) and vpp is not None and vpp > 1:
        return [f"{c['id']}: virtual pipeline stages must be off at CP {cp:g} (the interleaved schedule hangs at "
                f"CP >= {c['cp_min']} on this setup); got {vpp:g}"]
    return []


def _edp_dp_overlap(c, get):
    """With expert data parallelism EDP = GPUs / (EP x ETP x PP) above 1, --overlap-grad-reduce /
    --overlap-param-gather can hang at step 1-2 on a stack; such layouts must then run without them."""
    tp = _num(get("tensor_model_parallel_size"), 1.0)
    ep = _num(get("expert_model_parallel_size"), 1.0)
    etp = _num(get("expert_tensor_parallel_size"), tp)        # Megatron defaults ETP to TP (arguments.py:410-411)
    pp = _num(get("pipeline_model_parallel_size"), 1.0)
    edp = float(c["num_gpus"]) / (ep * etp * pp)
    on = [n for n in ("overlap_grad_reduce", "overlap_param_gather") if _truthy(get(n))]
    if edp > 1 and on:
        return [f"{c['id']}: expert data parallel size {edp:g} > 1 (EP {ep:g} x ETP {etp:g} x PP {pp:g} of "
                f"{c['num_gpus']} GPUs) needs --{' and --'.join(n.replace('_', '-') for n in on)} off (hangs on this stack)"]
    return []


def _flex_shared_overlap(c, get):
    """Megatron transformer_config.py:1117-1123 raises for --moe-shared-expert-overlap with any dispatcher but alltoall."""
    disp = str(get("moe_token_dispatcher_type") or "").strip("'\"")
    if disp and disp != "alltoall" and _truthy(get("moe_shared_expert_overlap")):
        return [f"{c['id']}: --moe-shared-expert-overlap works with the alltoall dispatcher only, got {disp}"]
    return []


def _min_value(c, get):
    """A parallel degree below which the model trains wrongly on the stack; unset means Megatron's 1."""
    v = _num(get(c["arg"]), 1.0)
    if v < float(c["min"]):
        return [f"{c['id']}: --{c['arg'].replace('_', '-')} {v:g} is below {c['min']} ({c['why']})"]
    return []


def _max_value(c, get):
    """A degree above the largest one the setup's checks cover; unset means Megatron's 1."""
    v = _num(get(c["arg"]), 1.0)
    if v > float(c["max"]):
        return [f"{c['id']}: --{c['arg'].replace('_', '-')} {v:g} is above {c['max']} ({c['why']})"]
    return []


def _max_when_eq(c, get):
    """Cap one degree when another equals a value (e.g. EP 1 needs ETP <= 2); unset ETP is TP."""
    w = _num(get(c["when_arg"]), 1.0)
    v = _num(get(c["arg"]), _num(get(c["default_from"]), 1.0) if c.get("default_from") else 1.0)
    if w == float(c["when_eq"]) and v > float(c["max"]):
        return [f"{c['id']}: --{c['arg'].replace('_', '-')} {v:g} exceeds {c['max']} when "
                f"--{c['when_arg'].replace('_', '-')} is {c['when_eq']} ({c['why']})"]
    return []


def _layout_stages(layout):
    """Megatron PipelineParallelLayerLayout.get_num_stages_from_str (pipeline_parallel_layer_layout.py:277-301): unroll
    '(...)*n' and 'x*n', drop cosmetic commas, count '|' + 1."""
    s = str(layout).strip("'\"").replace(",", "")
    for pat in (r"\(([^)]+)\)\*(\d+)", r"(.)\*(\d+)"):
        while _re.search(pat, s):
            s = _re.sub(pat, lambda m: m.group(1) * int(m.group(2)), s)
    return s.count("|") + 1


def _vpp_of(get):
    """Virtual stages per pipeline rank as Megatron resolves them (arguments.py:516-531): from the layout (stages / PP,
    1 -> None), --num-virtual-stages-per-pipeline-rank, --num-layers-per-virtual-pipeline-stage, or the resolved value."""
    pp = _num(get("pipeline_model_parallel_size"), 1.0)
    lay = get("pipeline_model_parallel_layout")
    if lay not in (None, True, ""):
        v = _layout_stages(lay) / pp
        return v if v > 1 else None
    for n in ("virtual_pipeline_model_parallel_size", "num_virtual_stages_per_pipeline_rank"):
        v = _num(get(n))
        if v is not None:
            return v
    nl, per = _num(get("num_layers")), _num(get("num_layers_per_virtual_pipeline_stage"))
    if nl and per:
        return nl / pp / per
    return None


def _a2a_overlap_requires(c, get):
    """Megatron startup asserts for the EP all-to-all overlap (core/transformer/transformer_config.py, 1dcf0daf):
    PP > 1 needs virtual stages (:1886-1890); no full recompute, no recompute method or layer count (:1900-1908); shared-
    expert overlap off (:1915-1917); EP > 1 (:1892-1894); alltoall or flex dispatcher (:1895-1898); and
    --delay-wgrad-compute needs the overlap (:1937-1941). Each would crash a launch after its allocation."""
    out, cid = [], c["id"]
    on = _truthy(get("overlap_moe_expert_parallel_comm"))
    if _truthy(get("delay_wgrad_compute")) and not on:
        out.append(f"{cid}: --delay-wgrad-compute needs --overlap-moe-expert-parallel-comm (transformer_config.py:1937-1941)")
    if not on:
        return out
    pp = _num(get("pipeline_model_parallel_size"), 1.0)
    vpp = _vpp_of(get)
    if pp > 1 and (vpp is None or vpp <= 1):
        out.append(f"{cid}: the EP a2a overlap at PP {pp:g} needs virtual pipeline stages (got none; "
                   "transformer_config.py:1886-1890)")
    rg = str(get("recompute_granularity") or "").strip("'\"")
    if rg == "full" or get("recompute_method") not in (None, "") or _num(get("recompute_num_layers")) is not None:
        out.append(f"{cid}: the EP a2a overlap forbids full recompute and --recompute-method/--recompute-num-layers "
                   "(transformer_config.py:1900-1908)")
    if _truthy(get("moe_shared_expert_overlap")):
        out.append(f"{cid}: the EP a2a overlap needs --moe-shared-expert-overlap off (transformer_config.py:1915-1917)")
    if _num(get("expert_model_parallel_size"), 1.0) <= 1:
        out.append(f"{cid}: the EP a2a overlap needs expert parallelism > 1 (transformer_config.py:1892-1894)")
    disp = str(get("moe_token_dispatcher_type") or "alltoall").strip("'\"")
    if disp not in ("alltoall", "flex"):
        out.append(f"{cid}: the EP a2a overlap needs the alltoall or flex dispatcher, got {disp} (transformer_config.py:1895-1898)")
    return out


def _require(c, get):
    """`require`: when the `when` flag is on, the listed settings must hold (the TP comm overlap rules above)."""
    if not _truthy(get(c["when"])):
        return []
    out, req, cid = [], c["require"], c["id"]
    if "tp_comm_bootstrap_backend" in req:
        got = str(get("tp_comm_bootstrap_backend") or "nccl").strip("'\"")
        if got != req["tp_comm_bootstrap_backend"]:
            out.append(f"{cid}: --tp-comm-bootstrap-backend must be {req['tp_comm_bootstrap_backend']}, got {got}")
    if req.get("sequence_parallel") and not _truthy(get("sequence_parallel")):
        out.append(f"{cid}: --sequence-parallel is required")
    seq = _num(get("seq_length"))
    if "seq_length" in req and seq != req["seq_length"]:
        out.append(f"{cid}: --seq-length must be {req['seq_length']} (the row length), got {get('seq_length')}")
    cp = _num(get("context_parallel_size"), 1.0)
    tp = _num(get("tensor_model_parallel_size"), 1.0)
    if req.get("exact_buffer_fill") and _truthy(get("use_dynamic_batch_size")):
        mt, mbs = _num(get("max_tokens_per_gpu")), _num(get("micro_batch_size"), 1.0)
        if mt is None or seq is None or mt * cp != seq * mbs:
            out.append(f"{cid}: userbuffers need max_tokens_per_gpu x CP == seq_length x micro_batch_size "
                       f"(exact fill), got {mt} x {cp} vs {seq} x {mbs}")
    if "tp_comm_overlap_cfg" in req:
        cfg = get("tp_comm_overlap_cfg")
        base = os.path.basename(str(cfg).strip("'\"")) if cfg else None
        if base not in req["tp_comm_overlap_cfg"]:
            out.append(f"{cid}: --tp-comm-overlap-cfg must be one of {req['tp_comm_overlap_cfg']} "
                       f"(None = TE defaults), got {base}")
    if req.get("tp_gt_1") and tp <= 1:
        out.append(f"{cid}: TP comm overlap needs TP > 1 (Megatron turns sequence parallel off at TP 1)")
    return out


KINDS = {
    "vpp_middle_stages": _vpp_middle_stages,
    "vpp_off_at_cp": _vpp_off_at_cp,
    "edp_dp_overlap": _edp_dp_overlap,
    "flex_shared_overlap": _flex_shared_overlap,
    "min_value": _min_value,
    "max_value": _max_value,
    "max_when_eq": _max_when_eq,
    "a2a_overlap_requires": _a2a_overlap_requires,
    "require": _require,
}


def violations(spec_raw, get):
    """List of strings. spec_raw: the loaded frozen-arguments file; get: dest name -> value or None. A coupling without
    a `kind` is a `require` rule; an unknown kind raises (the caller fails closed)."""
    out = []
    for c in spec_raw.get("couplings") or []:
        kind = c.get("kind", "require")
        if kind not in KINDS:
            raise ValueError(f"coupling {c.get('id')}: unknown kind {kind!r}")
        out += KINDS[kind](c, get)
    return out
