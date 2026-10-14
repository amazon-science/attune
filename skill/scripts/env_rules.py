# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which container env vars a run may set or change relative to its baseline (the canonical-manifest rule).

A run may differ from the baseline manifest only in training CLI flag tokens and in the STRUCTURAL env vars
below: placement, scheduling, allocator and logging switches that cannot change the math. Every other env var keeps
the baseline's value (and presence). FROZEN names are classified explicitly because they select numerics or
kernels (Transformer Engine backends, TF32, determinism) and are also written into the setup's frozen-arguments
file; anything unclassified is held fixed the same way.

    python3 env_rules.py NAME ...      # prints each name's class
"""
import argparse

STRUCTURAL = frozenset({
    # scheduling and allocation
    "CUDA_DEVICE_MAX_CONNECTIONS", "PYTORCH_CUDA_ALLOC_CONF",
    # NCCL algorithm and protocol selectors
    "NCCL_ALGO", "NCCL_PROTO",
    # debug and dump switches
    "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE", "TORCH_NCCL_TRACE_BUFFER_SIZE", "TORCH_NCCL_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE", "TORCH_SHOW_CPP_STACKTRACES", "TORCH_DISTRIBUTED_DEBUG", "PYTHONFAULTHANDLER",
    "CUDA_LAUNCH_BLOCKING",
    # profiler switches (which ranks trace, what a trace records)
    "SLIME_PROFILE_ALL_RANKS", "SLIME_PROFILE_MEMORY", "SLIME_PROFILE_WITH_STACK", "SLIME_PROFILE_WITH_FLOPS",
    "SLIME_PROFILE_RECORD_SHAPES",
})
FROZEN_PREFIXES = ("NVTE_",)     # Transformer Engine backend and fusion selectors (NVTE_FLASH_ATTN, NVTE_FUSED_ATTN, ...)
FROZEN = frozenset({
    "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",       # TF32 switches
    "CUBLAS_WORKSPACE_CONFIG", "TORCH_DETERMINISTIC", "PYTHONHASHSEED",  # determinism switches
})


def classify(name):
    """"structural" (may differ), "frozen" (numeric or kernel selector) or "unclassified" (held fixed too)."""
    if name in STRUCTURAL:
        return "structural"
    if name in FROZEN or name.startswith(FROZEN_PREFIXES):
        return "frozen"
    return "unclassified"


def may_differ(name):
    return classify(name) == "structural"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", metavar="NAME", help="environment variable names")
    for n in ap.parse_args().names:
        print(n, classify(n))
