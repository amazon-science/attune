# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Name-based classification tables: GPU kernels, memcpy directions, host frames, parallel dims.

Everything here matches generic names (CUDA/NCCL/PyTorch/Megatron conventions), never a model name.
"""
import re

# ---- GPU activity buckets, in precedence order (earlier wins when streams overlap) ----
COMPUTE_CLASSES = ("gemm", "grouped_gemm", "attention", "elementwise_norm", "optimizer", "other_compute")
COMM_DIMS = ("PP", "EP", "TP", "CP", "ETP", "DP", "other_comm")
MEMCPY_DIRS = ("D2H", "H2D", "P2P", "D2D", "memset", "memcpy_other")

HOST_CLASSES = (
    "gc",
    "allocator_cache_release",
    "device_sync_event_wait",
    "blocking_allocation",
    "host_device_copy_wait",
    "comm_host_call",
    "data_loading",
    "checkpoint_io",
    "optimizer_cpu_step",
    "kernel_launch_bound",
    "python_other",
)
HOST_CLASS_LABEL = {
    "gc": "garbage collection",
    "allocator_cache_release": "allocator cache release (empty_cache)",
    "device_sync_event_wait": "device sync / event wait",
    "blocking_allocation": "blocking allocation / malloc / free",
    "host_device_copy_wait": "host<->device copy wait",
    "comm_host_call": "collective / p2p host call",
    "data_loading": "data loading",
    "checkpoint_io": "checkpoint / file I/O",
    "optimizer_cpu_step": "optimizer CPU step",
    "kernel_launch_bound": "kernel-launch-bound (host in launch API)",
    "python_other": "python/host other",
}

GEMM_KEYS = ("gemm", "nvjet", "cublas", "cutlass", "matmul", "xmma", "s16816", "s1688", "wgmma", "gemv",
             "sm90_fp8", "deep_gemm", "_mm_", "splitk")
ATTN_KEYS = ("flash", "fmha", "attn", "attention", "sdpa", "delta_rule", "mha_fwd", "mha_bwd", "_mla_")
OPT_KEYS = ("adam", "multi_tensor", "lamb_", "sgd_", "optimizer", "l2norm")
ELEM_KEYS = ("elementwise", "vectorized", "unrolled", "norm", "softmax", "reduce", "silu", "gelu", "swiglu",
             "act_and_mul", "dropout", "triton_poi", "triton_red", "triton_per", "copy_kernel", "fill_kernel",
             "cast", "rope", "rotary", "bias", "sigmoid", "exp_", "mul_", "add_")
COMM_KERNEL_KEYS = ("nccl", "deep_ep", "cross_device_reduce", "userbuffers", "all_reduce_kernel", "allreduce_kernel")

LAUNCH_API_KEYS = ("launchkernel", "culaunch", "cudalaunch", "graphlaunch", "cudalaunchcooperative")
ALLOC_API_KEYS = ("cudamalloc", "cudafree", "cumemalloc", "cumemfree", "cumemcreate", "cumemmap", "cumemrelease",
                  "cudahostalloc", "cudafreehost", "cudahostregister", "cudamemgetinfo", "cumemgetinfo")


def is_comm_kernel(nl):
    return any(k in nl for k in COMM_KERNEL_KEYS)


def compute_class(nl):
    """Class of a non-comm kernel from its lowercased name (grouped/optimizer refined by launch context)."""
    if any(k in nl for k in ATTN_KEYS) or ("cudnn" in nl and ("fprop" in nl or "bprop" in nl)):
        return "attention"
    if any(k in nl for k in GEMM_KEYS):
        if "grouped" in nl or "group_gemm" in nl or "moe_gemm" in nl:
            return "grouped_gemm"
        return "gemm"
    if any(k in nl for k in OPT_KEYS):
        return "optimizer"
    if any(k in nl for k in ELEM_KEYS):
        return "elementwise_norm"
    return "other_compute"


def memcpy_dir(name, is_memset):
    if is_memset:
        return "memset"
    n = name.lower()
    if "htod" in n or "host -> device" in n or "pinned -> device" in n or "pageable -> device" in n:
        return "H2D"
    if "dtoh" in n or "-> pinned" in n or "-> pageable" in n:
        return "D2H"
    if "ptop" in n or "peer" in n:
        return "P2P"
    if "dtod" in n:
        return "D2D"
    return "memcpy_other"


# ---- v1 kernel categories (kept verbatim so profile_top_categories stays comparable) ----
# Kernel-name classification rules adapted from slime's tools/analyze_profile.py
# (https://github.com/THUDM/slime, Apache-2.0). See THIRD_PARTY_NOTICES.
def v1_category(name):
    nl = name.lower()
    if "nccl" in nl:
        return "NCCL Communication"
    if "deep_ep" in nl:
        if "dispatch" in nl:
            return "DeepEP Dispatch (MoE)"
        if "combine" in nl:
            return "DeepEP Combine (MoE)"
        if "clean" in nl:
            return "DeepEP Buffer Clean"
        return "DeepEP (other)"
    if ("flash" in nl and "attn" in nl) or "flash::" in nl or "flash_fwd" in nl or "flash_bwd" in nl:
        return "Flash Attention"
    if "sdpa" in nl or ("cudnn" in nl and ("fprop" in nl or "bprop" in nl) and "flash" in nl):
        return "Attention (cuDNN fused)"
    if "sparse_attn" in nl:
        return "Sparse Attention (MLA)"
    if "paged_mqa" in nl:
        return "Paged MQA/MLA Logits"
    if "attention" in nl or "fmha" in nl:
        return "Attention (other)"
    if "deep_gemm" in nl or "sm90_fp8_gemm" in nl:
        return "DeepGEMM (FP8)"
    if "nvjet" in nl:
        return "NvJet GEMM"
    if "gemm" in nl or "cutlass" in nl or "matmul" in nl or "cublas" in nl:
        return "GEMM/MatMul (other)"
    if "topk" in nl:
        return "TopK / MoE Routing"
    if "quant" in nl:
        return "Quantization (FP8)"
    if "rmsnorm" in nl or "layernorm" in nl or "norm" in nl:
        return "Normalization"
    if any(k in nl for k in ["triton", "fused"]):
        return "Fused/Triton Kernels"
    if "elementwise" in nl or "vectorized_elementwise" in nl:
        return "Elementwise Ops"
    if "reduce" in nl or "softmax" in nl:
        return "Reduce/Softmax"
    if "embedding" in nl:
        return "Embedding"
    if "memcpy" in nl or "memset" in nl:
        return "Memory Ops"
    if "index" in nl or "scatter" in nl or "gather" in nl:
        return "Index/Gather/Scatter"
    if "cat" in nl and "batched" in nl:
        return "Concat/Cat"
    return "Other Compute"


def v1_copy_category(name, is_memset):
    nl = name.lower()
    if "dtoh" in nl or "d2h" in nl or "device -> host" in nl or "devicetohost" in nl:
        return "Memcpy D2H (offload)"
    if "htod" in nl or "h2d" in nl or "host -> device" in nl or "hosttodevice" in nl:
        return "Memcpy H2D (offload)"
    if is_memset:
        return "Memset"
    return "Memcpy (other)"


# ---- host frames ----
_FRAME_RE = re.compile(r"^(.*?\.py)\((\d+)\): (.*)$")


def split_frame(name):
    """'path/file.py(12): func' -> ('path/file.py', 'func'); other names -> ('', name)."""
    m = _FRAME_RE.match(name)
    if m:
        return m.group(1), m.group(3)
    return "", name


_HOST_RULES = (
    ("gc", ("<built-in function collect>", "gc_collect", "collect_garbage", "gc.collect")),
    ("allocator_cache_release", ("emptycache", "empty_cache", "release_cached", "releasepool")),
    ("device_sync_event_wait", ("devicesynchronize", "streamsynchronize", "eventsynchronize", "ctxsynchronize",
                                "_cuda_synchronize", "aten::synchronize", "cudastreamquery_wait")),
    ("blocking_allocation", ("_empty_strided_cuda", "aten::empty", "aten::resize_", "aten::zeros", "aten::full",
                             "cudamalloc", "cudafree", "cumemalloc", "cumemfree", "cumemcreate", "cumemmap",
                             "cudahostalloc", "cudafreehost", "cudahostregister")),
    ("host_device_copy_wait", ("cudamemcpy", "cumemcpy", "aten::copy_", "aten::_to_copy", "aten::item",
                               "aten::_local_scalar_dense", "aten::nonzero", "=aten::to", "<built-in method item",
                               "<built-in method tolist", "<built-in method cpu", "<method 'cpu'")),
    ("comm_host_call", ("nccl:", "gloo:", "c10d::", "record_param_comms", "batch_isend_irecv", "processgroup",
                        "<built-in method wait of", "barrier", "all_reduce", "all_gather", "reduce_scatter",
                        "broadcast", "isend", "irecv", "_communicate", "send_forward", "recv_forward",
                        "send_backward", "recv_backward")),
    ("data_loading", ("get_batch", "_next_data", "collate", "pin_memory", "dataloader", "data_iterator")),
    ("checkpoint_io", ("save_checkpoint", "load_checkpoint", "fsync", "safetensors", "torch.save", "torch.load",
                       "dist_checkpointing")),
    ("optimizer_cpu_step", ("optimizer.step#", "clip_grad", "get_grad_norm", "_step_impl", "optimizer_step",
                            "step_with_ready_grads")),
    ("kernel_launch_bound", LAUNCH_API_KEYS),
)
_HOST_PRIORITY = {c: i for i, c in enumerate(HOST_CLASSES)}


def host_class(name):
    """Class matched by a single frame name (or None). Patterns match the function part only."""
    path, func = split_frame(name)
    fl = func.lower()
    for cls, keys in _HOST_RULES:
        for k in keys:
            if (fl == k[1:]) if k[0] == "=" else (k in fl):
                return cls
    if path and "optim" in path.lower() and fl in ("step", "_step", "wrapper", "_use_grad", "maybe_fallback"):
        return "optimizer_cpu_step"
    if path and ("serialization.py" in path or "checkpointing" in path) and ("save" in fl or "load" in fl):
        return "checkpoint_io"
    return None


def host_priority(cls):
    return _HOST_PRIORITY.get(cls, len(HOST_CLASSES))


def is_launch_api(name):
    nl = name.lower()
    return any(k in nl for k in LAUNCH_API_KEYS)


def is_alloc_api(name):
    nl = name.lower()
    return any(k in nl for k in ALLOC_API_KEYS)


# ---- phases ----
def phase_kind(name, cat):
    """Phase marker implied by a host frame: 'forward', 'backward', 'recompute_frame', 'optimizer', or None."""
    path, func = split_frame(name)
    fl = func.lower()
    if name.startswith("Optimizer.step#"):
        return "optimizer"
    if path:
        pl = path.lower()
        if fl == "forward_step" and "schedules" in pl:
            return "forward"
        if fl == "backward_step" and "schedules" in pl:
            return "backward"
        if fl == "backward" and ("random.py" in pl or "checkpoint" in pl):
            return "recompute_frame"
        if "optim" in pl and fl == "step" and "torch/optim" not in pl:
            return "optimizer"
        return None
    if name == "CheckpointFunctionBackward":
        return "recompute_frame"
    return None


PHASES = ("forward", "backward", "recompute", "optimizer", "other")

# ---- parallel dimensions from process-group descriptions ----
_PG_EXACT = {
    "PIPELINE_MODEL_PARALLEL_GROUP": "PP",
    "EXPERT_MODEL_PARALLEL_GROUP": "EP",
    "EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP": "EP",
    "EXPERT_TENSOR_PARALLEL_GROUP": "ETP",
    "EXPERT_DATA_PARALLEL_GROUP": "DP",
    "TENSOR_MODEL_PARALLEL_GROUP": "TP",
    "TENSOR_AND_CONTEXT_PARALLEL_GROUP": "TP",
    "CONTEXT_PARALLEL_GROUP": "CP",
    "DATA_PARALLEL_GROUP": "DP",
    "DATA_PARALLEL_GROUP_WITH_CP": "DP",
    "TENSOR_AND_DATA_PARALLEL_GROUP": "DP",
    "TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP": "DP",
    "INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP": "DP",
    "INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP": "DP",
    "INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP": "DP",
    "INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP": "DP",
}


def dim_of_pg_desc(desc):
    if not desc:
        return None
    d = desc.upper()
    if d in _PG_EXACT:
        return _PG_EXACT[d]
    if "PIPELINE" in d and "EXPERT" not in d:
        return "PP"
    if d.startswith("EXPERT_TENSOR_AND_MODEL") or d.startswith("EXPERT_MODEL"):
        return "EP"
    if d.startswith("EXPERT_TENSOR"):
        return "ETP" if "MODEL" not in d else "other_comm"
    if "EXPERT_DATA" in d:
        return "DP"
    if "CONTEXT" in d and "TENSOR" not in d and "DATA" not in d:
        return "CP"
    if "DATA_PARALLEL" in d or "DISTRIBUTED_OPTIMIZER" in d:
        return "DP"
    if d.startswith("TENSOR"):
        return "TP"
    return "other_comm"


P2P_FRAME_KEYS = ("p2p_communication", "send_forward", "recv_forward", "send_backward", "recv_backward",
                  "batch_isend_irecv", "_communicate", "isend", "irecv")


def coll_type(collective_name, kernel_name):
    c = (collective_name or "").lower()
    k = kernel_name.lower()
    if "all_to_all" in c or "alltoall" in c:
        return "alltoall"
    if "allreduce" in c or "all_reduce" in c:
        return "allreduce"
    if "reduce_scatter" in c or "reducescatter" in c:
        return "reduce_scatter"
    if "allgather" in c or "all_gather" in c:
        return "allgather"
    if "broadcast" in c:
        return "broadcast"
    if c in ("send", "recv") or "sendrecv" in c:
        return "sendrecv"
    if "reduce" in c:
        return "reduce"
    if "deep_ep" in k:
        return "dispatch" if "dispatch" in k else ("combine" if "combine" in k else "deepep_other")
    if "allreduce" in k:
        return "allreduce"
    if "reducescatter" in k:
        return "reduce_scatter"
    if "allgather" in k:
        return "allgather"
    if "sendrecv" in k or "_send" in k or "_recv" in k:
        return "sendrecv"
    if "broadcast" in k:
        return "broadcast"
    return "other"


# NCCL bus-bandwidth factors: busbw = algbw * f(n)
def busbw_factor(ctype, n):
    n = max(int(n or 1), 1)
    if ctype == "allreduce":
        return 2.0 * (n - 1) / n
    if ctype in ("allgather", "reduce_scatter", "alltoall"):
        return (n - 1) / n
    return 1.0


DTYPE_BYTES = {"float": 4, "float32": 4, "float16": 2, "half": 2, "bfloat16": 2, "double": 8, "float64": 8,
               "int": 4, "int32": 4, "int64": 8, "long": 8, "uint8": 1, "int8": 1, "char": 1, "byte": 1, "bool": 1,
               "float8_e4m3fn": 1, "float8_e5m2": 1, "unsigned int": 4, "uint32": 4, "short": 2, "int16": 2}


def dtype_bytes(dtype):
    if not dtype:
        return None
    return DTYPE_BYTES.get(str(dtype).lower().replace("torch.", ""))


# ---- reference numbers (override with --link-bw / --peak-tflops) ----
# Dense BF16 tensor-core peak per GPU, TFLOP/s.
PEAK_BF16_TFLOPS = (("GB200", 2500.0), ("B200", 2250.0), ("B300", 2250.0), ("H200", 989.4), ("H100", 989.4),
                    ("H800", 989.4), ("H20", 148.0), ("A100", 312.0), ("A800", 312.0), ("L40S", 362.0))
# Nominal per-GPU unidirectional NVLink bandwidth, GB/s.
NVLINK_GBPS = (("GB200", 900.0), ("B200", 900.0), ("B300", 900.0), ("H200", 450.0), ("H100", 450.0),
               ("H800", 200.0), ("H20", 450.0), ("A100", 300.0), ("A800", 200.0))
# Nominal per-GPU network bandwidth for inter-node groups, GB/s (EFA 3.2 Tb/s per 8-GPU instance).
NET_GBPS_DEFAULT = 50.0


def lookup(table, gpu_name, default=None):
    g = (gpu_name or "").upper()
    for key, val in table:
        if key in g:
            return val
    return default
