# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile analyzer: conserved step-time decomposition of PyTorch profiler traces, per rank.

Modules: parse (trace -> arrays + cache), intervals (interval algebra), taxonomy (name classification),
decompose (per-rank buckets, phases, idle attribution), comm (parallel dims, bus bandwidth), memory,
pipeline (cross-rank views), report (text), cli (entry point).
"""
