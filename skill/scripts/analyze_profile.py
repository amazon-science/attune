#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile trace analyzer.

    analyze_profile.py --profile-dir DIR [--rank R] [--top-n N] [--json | --json-full] [--quick]
    analyze_profile.py {ranks|rank R|gaps R|stack R|comm|kernels|phases R|memory R|diff A B} --profile-dir DIR

Each rank's profiled step is decomposed into disjoint buckets that sum to its wall time (compute, exposed
communication by parallel dimension, exposed memcpy, host-side idle by class, unattributed idle, fine-grained
idle), plus host-side memory management (allocator cleanup) per rank and per pipeline stage. It measures
and attributes; it does not diagnose or recommend. The key facts come first (text: the summary block;
JSON: the `summary` object). Implementation: profile_analyzer/. Python 3.10+, numpy; orjson optional.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profile_analyzer.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
