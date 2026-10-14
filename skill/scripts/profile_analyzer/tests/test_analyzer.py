# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic tests: overlap, nested frames, multi-stream, missing fields, stack-less traces, conservation.

Run with `python3 -m pytest scripts/profile_analyzer/tests` or `python3 scripts/profile_analyzer/tests/test_analyzer.py`.
"""
import io
import json
import os
import random
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # scripts/

import numpy as np  # noqa: E402

from profile_analyzer import cli, decompose, intervals as iv, parse, taxonomy as tx  # noqa: E402
from profile_analyzer.tests.synth import AUTOGRAD, MAIN, Trace  # noqa: E402


def analyze(trace, opts=None):
    meta, arr = parse.extract(trace.data(), "synthetic")
    r = decompose.Rank(meta, arr, opts or {})
    wins = r.windows()
    return r, [r.analyze_window(w) for w in wins]


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_conservation(res):
    c = res["conservation"]
    assert c["ok"], c
    b = res["buckets_us"]
    tot = b["compute"] + b["comm_exposed"] + b["memcpy_exposed"] + sum(b["host"].values()) + \
        b["unattributed_idle"] + b["fine_idle"]
    assert close(tot, res["window"]["wall_us"]), (tot, res["window"]["wall_us"])


# ---------------- interval algebra ----------------
def _brute(n, ivs):
    grid = np.zeros(n, dtype=bool)
    for s, e in ivs:
        grid[int(s):int(e)] = True
    return grid


def _to_grid(n, st):
    return _brute(n, list(zip(st[0], st[1])))


def test_interval_algebra_matches_brute_force():
    rng = random.Random(0)
    for _ in range(200):
        n = 200
        a = [(x, x + rng.randint(1, 30)) for x in (rng.randint(0, 170) for _ in range(rng.randint(0, 12)))]
        b = [(x, x + rng.randint(1, 30)) for x in (rng.randint(0, 170) for _ in range(rng.randint(0, 12)))]
        ua = iv.union([x[0] for x in a], [x[1] for x in a])
        ub = iv.union([x[0] for x in b], [x[1] for x in b])
        ga, gb = _brute(n, a), _brute(n, b)
        assert (_to_grid(n, ua) == ga).all()
        assert (_to_grid(n, iv.intersect(ua, ub)) == (ga & gb)).all()
        assert (_to_grid(n, iv.subtract(ua, ub)) == (ga & ~gb)).all()
        assert (_to_grid(n, iv.complement(ua, 0, n)) == ~ga).all()
        assert close(iv.total(ua), ga.sum())


# ---------------- overlap / precedence ----------------
def test_overlap_precedence_compute_comm_memcpy():
    t = Trace()
    t.step(0, 10000)
    t.kernel(0, 3000, "nvjet_gemm", stream=7)                 # compute
    t.nccl(2000, 5000, stream=20)                               # comm: 1000 overlapped, 2000 exposed
    t.memcpy(4000, 6000, stream=30)                             # memcpy: 1000 under comm, 1000 exposed
    t.frame(6000, 10000, "train.py(1): train")                  # host frame over the trailing idle
    r, (res,) = analyze(t)
    b = res["buckets_us"]
    assert close(b["compute"], 3000)
    assert close(b["comm_exposed"], 2000)
    assert close(b["memcpy_exposed"], 1000)
    assert close(res["comm"]["TP"]["overlapped_us"], 1000)
    assert close(res["memcpy"]["D2H"]["exposed_us"], 1000)
    assert close(b["host"]["python_other"], 4000)
    assert close(res["gpu_busy_us"], 6000)
    check_conservation(res)


def test_multi_stream_union_vs_sum():
    t = Trace()
    t.step(0, 5000)
    t.kernel(0, 2000, "nvjet_gemm_a", stream=7)
    t.kernel(1000, 3000, "nvjet_gemm_b", stream=8)             # overlaps the first on another stream
    t.kernel(1500, 2500, "flash_fwd_kernel", stream=9)          # attention under both
    r, (res,) = analyze(t)
    g = res["compute"]["gemm"]
    assert close(g["sum_us"], 4000) and close(g["union_us"], 3000) and close(g["exclusive_us"], 3000)
    a = res["compute"]["attention"]
    assert close(a["sum_us"], 1000) and close(a["exclusive_us"], 0)     # gemm has precedence
    assert close(res["buckets_us"]["compute"], 3000)
    check_conservation(res)


# ---------------- nested frames / classes ----------------
def test_nested_frames_innermost_and_priority():
    t = Trace()
    t.step(0, 10000)
    t.kernel(0, 1000, "nvjet_gemm")
    t.kernel(5000, 10000, "nvjet_gemm")
    t.frame(0, 9000, "train.py(1): train")
    t.frame(1500, 2500, "<built-in function collect>")
    t.frame(3000, 4000, "torch/cuda/memory.py(212): empty_cache")
    t.frame(3100, 3900, "<built-in function _cuda_emptyCache>")
    t.frame(3200, 3300, "cudaFree", cat="cuda_runtime")       # inner alloc API under empty_cache
    r, (res,) = analyze(t)
    h = res["buckets_us"]["host"]
    assert close(h["gc"], 1000), h
    assert close(h["allocator_cache_release"], 1000), h      # cache release outranks the nested cudaFree
    assert close(h["python_other"], 2000), h
    top = res["host_classes"]["gc"]["top_frames"][0]["frame"]
    assert top == "<built-in function collect>"
    check_conservation(res)


def test_cross_thread_shortest_frame_wins():
    t = Trace()
    t.step(0, 6000)
    t.kernel(0, 1000, "nvjet_gemm", launch_tid=MAIN)
    t.kernel(500, 900, "nvjet_gemm", launch_tid=AUTOGRAD)
    t.kernel(5000, 6000, "nvjet_gemm", launch_tid=MAIN)
    t.frame(0, 6000, "schedules.py(1): forward_backward", tid=MAIN)
    t.frame(1000, 5000, "<built-in method run_backward of X>", tid=MAIN)
    t.frame(2000, 4000, "<built-in function _empty_strided_cuda>", tid=AUTOGRAD)
    r, (res,) = analyze(t)
    h = res["buckets_us"]["host"]
    assert close(h["blocking_allocation"], 2000), h
    assert close(h["python_other"], 2000), h
    check_conservation(res)


def test_fine_idle_and_unattributed_and_context_frames():
    t = Trace()
    t.step(0, 12000)
    t.frame(0, 12000, "ray/_private/worker.py(1): main_loop")   # spans the window: context, not attribution
    for k in range(10):                                           # 9 gaps of 500 us -> fine idle
        t.kernel(k * 1000, k * 1000 + 500, "nvjet_gemm")
    r, (res,) = analyze(t)
    b = res["buckets_us"]
    assert close(b["fine_idle"], 9 * 500), b
    assert close(b["unattributed_idle"], 2500), b                 # trailing 2.5 ms gap, only a context frame
    assert close(sum(b["host"].values()), 0), b
    check_conservation(res)


def test_unattributed_coarse_gap():
    t = Trace()
    t.step(0, 10000)
    t.kernel(0, 1000, "nvjet_gemm")
    t.kernel(9000, 10000, "nvjet_gemm")
    r, (res,) = analyze(t)
    assert close(res["buckets_us"]["unattributed_idle"], 8000)
    check_conservation(res)


# ---------------- comm attribution ----------------
def test_comm_dims_and_busbw():
    pgs = [{"pg_name": "5", "pg_desc": "PIPELINE_MODEL_PARALLEL_GROUP", "pg_size": 2, "ranks": [0, 8],
            "backend_config": "cuda:nccl"},
           {"pg_name": "6", "pg_desc": "EXPERT_MODEL_PARALLEL_GROUP", "pg_size": 8, "ranks": list(range(8)),
            "backend_config": "cuda:nccl"}]
    t = Trace(rank=0, world=16, pgs=pgs)
    t.step(0, 20000)
    t.nccl(0, 1000, coll="allreduce", desc="TENSOR_MODEL_PARALLEL_GROUP", n=2, nelems=1 << 26)   # 128 MiB bf16
    t.nccl(1000, 2000, coll="all_to_allv", desc="EXPERT_MODEL_PARALLEL_GROUP", n=8, nelems=1 << 26,
           kname="ncclDevKernel_SendRecv")
    t.nccl(2000, 3000, coll="_allgather_base", desc="DATA_PARALLEL_GROUP", n=16, nelems=1 << 20,
           kname="ncclDevKernel_AllGather_RING_LL", ranks=list(range(16)))
    # p2p without PG args, launched under a p2p frame -> PP
    t.frame(2990, 4100, "megatron/core/pipeline_parallel/p2p_communication.py(464): send_forward")
    t.nccl(3000, 4000, coll=None, kname="ncclDevKernel_SendRecv")
    r, (res,) = analyze(t)
    c = res["comm"]
    assert c["TP"]["count"] == 1 and c["EP"]["count"] == 1 and c["DP"]["count"] == 1 and c["PP"]["count"] == 1
    bw = res["comm_bw"]
    ar = bw["TP"]["allreduce"]
    exp = (1 << 26) * 2 / 1e-3 / 1e9 * 2 * (2 - 1) / 2
    assert close(ar["busbw_gbps"], exp), (ar["busbw_gbps"], exp)
    assert bw["TP"]["allreduce"]["placement"] == "intra-node"
    assert bw["DP"]["allgather"]["placement"] == "inter-node"
    assert "launch-stack" in bw["PP"]["sendrecv"]["inferred"]
    check_conservation(res)


def test_pg_desc_mapping():
    m = tx.dim_of_pg_desc
    assert m("TENSOR_MODEL_PARALLEL_GROUP") == "TP"
    assert m("EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP") == "EP"
    assert m("EXPERT_TENSOR_PARALLEL_GROUP") == "ETP"
    assert m("EXPERT_DATA_PARALLEL_GROUP") == "DP"
    assert m("DATA_PARALLEL_GROUP_WITH_CP") == "DP"
    assert m("INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP") == "DP"
    assert m("PIPELINE_MODEL_PARALLEL_GROUP") == "PP"
    assert m("CONTEXT_PARALLEL_GROUP") == "CP"
    assert m("default_pg") == "other_comm"
    assert tx.busbw_factor("allreduce", 8) == 2 * 7 / 8.0
    assert tx.busbw_factor("allgather", 4) == 0.75
    assert tx.busbw_factor("sendrecv", 8) == 1.0


# ---------------- missing fields / stack-less ----------------
def test_missing_fields_do_not_crash():
    t = Trace(device_props=False, dist=False, flags={})
    t.ev.append({"ph": "X", "cat": "kernel", "name": "weird_kernel", "pid": 0, "tid": 7, "ts": 10.0})  # no dur/args
    t.ev.append({"ph": "X", "cat": "kernel", "name": "ncclDevKernel_SendRecv", "pid": 0, "tid": 9, "ts": 20.0,
                 "dur": 3000.0})
    t.ev.append({"ph": "X", "cat": "gpu_memcpy", "pid": 0, "tid": 9, "ts": 20.0, "dur": 5.0})         # no name/args
    t.ev.append({"ph": "X", "cat": "cpu_op", "ts": 0.0, "dur": 50.0})                                 # no name/tid
    t.ev.append({"ph": "i", "cat": "cpu_instant_event", "name": "[memory]", "ts": 1.0})              # no args
    t.ev.append("not-a-dict")
    r, res = analyze(t)
    assert len(res) == 1
    assert res[0]["window"]["source"].startswith("GPU active span")
    check_conservation(res[0])
    v1 = decompose.v1_compat(r)
    assert "profile_top_categories" in v1


def test_stackless_trace_uses_cpu_ops_and_annotations():
    t = Trace(flags={"with_stack": 0})
    t.step(0, 20000)
    t.kernel(0, 1000, "nvjet_gemm")
    t.frame(1000, 9000, "autograd::engine::evaluate_function: CheckpointFunctionBackward", cat="cpu_op", tid=AUTOGRAD)
    t.frame(1010, 8990, "CheckpointFunctionBackward", cat="cpu_op", tid=AUTOGRAD)
    t.kernel(1100, 3000, "nvjet_gemm", launch_tid=AUTOGRAD)                     # re-forward
    t.frame(4000, 8000, "autograd::engine::evaluate_function: MmBackward0", cat="cpu_op", tid=AUTOGRAD)
    t.kernel(4100, 8000, "nvjet_gemm", launch_tid=AUTOGRAD)
    t.frame(10000, 18000, "Optimizer.step#AdamW.step", cat="user_annotation")
    t.frame(12000, 16000, "aten::copy_", cat="cpu_op")
    t.memcpy(12000, 16000, "HtoD (Pinned -> Device)")
    r, (res,) = analyze(t)
    ph = res["phases"]
    assert "recompute" in ph and close(ph["recompute"]["wall_us"], 4000 - 1010), ph.get("recompute")
    assert "optimizer" in ph and close(ph["optimizer"]["wall_us"], 8000)
    assert close(res["memcpy"]["H2D"]["exposed_us"], 4000)
    assert close(res["buckets_us"]["host"].get("optimizer_cpu_step", 0), 4000)
    check_conservation(res)


def test_multi_step_windows():
    t = Trace()
    t.step(0, 5000, n=24)
    t.step(5000, 12000, n=25)
    t.kernel(0, 4000, "nvjet_gemm")
    t.kernel(5000, 11000, "nvjet_gemm")
    r, res = analyze(t)
    assert [x["window"]["step"] for x in res] == ["ProfilerStep#24", "ProfilerStep#25"]
    assert close(res[1]["window"]["wall_us"], 7000)
    for x in res:
        check_conservation(x)


def test_random_traces_conserve():
    rng = random.Random(7)
    for seed in range(40):
        t = Trace()
        T = 200000.0
        t.step(0, T)
        for _ in range(rng.randint(0, 60)):
            s = rng.uniform(-5000, T)
            d = rng.expovariate(1 / 3000.0)
            kind = rng.random()
            if kind < 0.5:
                t.kernel(s, s + d, rng.choice(["nvjet_gemm", "flash_fwd", "elementwise_kernel", "foo"]),
                         stream=rng.choice([7, 8, 9]), launch_tid=rng.choice([MAIN, AUTOGRAD]))
            elif kind < 0.8:
                t.nccl(s, s + d, desc=rng.choice(["TENSOR_MODEL_PARALLEL_GROUP", "EXPERT_MODEL_PARALLEL_GROUP"]),
                       stream=rng.choice([20, 21]))
            else:
                t.memcpy(s, s + d, rng.choice(["DtoH (Device -> Pinned)", "HtoD (Pageable -> Device)"]))
        for _ in range(rng.randint(0, 40)):
            s = rng.uniform(0, T)
            d = rng.expovariate(1 / 8000.0)
            t.frame(s, s + d, rng.choice(["<built-in function collect>", "a.py(1): f", "aten::empty",
                                          "cudaStreamSynchronize", "b.py(2): get_batch"]),
                    cat=rng.choice(["python_function", "cpu_op"]), tid=rng.choice([MAIN, AUTOGRAD]))
        r, res = analyze(t)
        for x in res:
            check_conservation(x)


# ---------------- memory ----------------
def test_memory_peaks():
    t = Trace()
    t.step(0, 10000)
    t.kernel(0, 10000, "nvjet_gemm")
    t.mem(100, 10 * 2 ** 30, 12 * 2 ** 30)
    t.mem(5000, 30 * 2 ** 30, 40 * 2 ** 30)
    t.mem(9000, 20 * 2 ** 30, 40 * 2 ** 30)
    r, (res,) = analyze(t)
    m = res["memory"]
    assert close(m["peak_allocated_gib"], 30) and close(m["peak_reserved_gib"], 40)
    assert close(m["allocator_unreserved_gib"], 40)


# ---------------- end-to-end CLI with cache ----------------
def _mk_run(d, busy_scale=1.0, gc_ms=0.0):
    pgs = [{"pg_name": "5", "pg_desc": "PIPELINE_MODEL_PARALLEL_GROUP", "pg_size": 2, "ranks": [0, 1],
            "backend_config": "cuda:nccl"}]
    for rank in (0, 1):
        t = Trace(rank=rank, world=2, pgs=pgs)
        t.step(0, 100000, n=24)
        t.frame(0, 100000, "schedules.py(1): forward_backward_pipelining", tid=MAIN)
        for k in range(8):
            s = k * 12000.0
            t.frame(s, s + 6000, "megatron/core/pipeline_parallel/schedules.py(314): forward_step")
            t.kernel(s, s + 5000 * busy_scale, "nvjet_gemm")
            t.frame(s + 6000, s + 6100, "megatron/core/pipeline_parallel/p2p_communication.py(1): send_forward")
            t.nccl(s + 6050, s + 11000, coll=None, kname="ncclDevKernel_SendRecv")
        if rank == 1 and gc_ms:
            t.frame(96000, 96000 + gc_ms, "<built-in function collect>")
        t.write(os.path.join(d, "rank_%d.pt.trace.json.gz" % rank))


def test_cli_end_to_end_cache_and_diff():
    tmp = tempfile.mkdtemp()
    try:
        a = os.path.join(tmp, "a")
        b = os.path.join(tmp, "b")
        os.makedirs(a)
        os.makedirs(b)
        _mk_run(a, gc_ms=3500)
        _mk_run(b, busy_scale=0.5)
        for argv in (["--profile-dir", a], ["--profile-dir", a, "--top-n", "5"],
                     ["ranks", "--profile-dir", a], ["rank", "1", "--profile-dir", a],
                     ["gaps", "1", "--profile-dir", a, "--top", "5"], ["phases", "0", "--profile-dir", a],
                     ["memory", "0", "--profile-dir", a], ["comm", "--profile-dir", a, "--dim", "PP"],
                     ["kernels", "--profile-dir", a, "--class", "gemm"],
                     ["stack", "0", "--profile-dir", a, "--t0", "5", "--t1", "7"], ["diff", a, b]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(argv)
            assert rc == 0, argv
            assert buf.getvalue().strip(), argv
        assert any(f.endswith(".pa2.npz") for f in os.listdir(a))
        assert any(f.endswith(".pa2.analysis.json") for f in os.listdir(a))
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["--profile-dir", a, "--json"])
        d = json.loads(buf.getvalue())
        for k in ("profile_top_categories", "profile_top_kernels", "gpu_util_pct", "total_kernel_ms", "ranks",
                  "coverage", "pipeline"):
            assert k in d, k
        assert d["schema"] == "profile_analyzer/v2.2"
        assert d["pipeline"]["pp_size"] == 2 and len(d["ranks"]) == 2
        # cache round trip gives identical arrays
        f = os.path.join(a, "rank_0.pt.trace.json.gz")
        m1, a1, hit1 = parse.load_rank(f)
        assert hit1
        m2, a2 = parse.extract(parse.read_json(f), f)
        for k in a2:
            assert np.array_equal(a1[k], a2[k], equal_nan=True) if a2[k].dtype.kind == "f" else np.array_equal(a1[k], a2[k]), k
        # the corrupt-trace path does not crash the report
        with open(os.path.join(a, "rank_9.pt.trace.json.gz"), "wb") as fh:
            fh.write(b"\x1f\x8b garbage")
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert cli.main(["--profile-dir", a]) == 0
        assert "ERROR" in buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------- output order, summary, labels, p2p, optimizer, streaming, speed ----------------
def _cli(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    assert rc == 0, argv
    return buf.getvalue()


def _mk_ranks(d, ranks, stall_rank=None, gc_ms=0.0, compute_rank=None):
    pgs = [{"pg_name": "5", "pg_desc": "PIPELINE_MODEL_PARALLEL_GROUP", "pg_size": len(ranks), "ranks": list(ranks),
            "backend_config": "cuda:nccl"}]
    for rank in ranks:
        t = Trace(rank=rank, world=max(ranks) + 1, pgs=pgs)
        t.step(0, 100000, n=24)
        t.frame(0, 100000, "schedules.py(1): forward_backward_pipelining", tid=MAIN)
        big = rank == compute_rank
        for k in range(8):
            s = k * 12000.0
            t.frame(s, s + 6000, "megatron/core/pipeline_parallel/schedules.py(314): forward_step")
            t.kernel(s, s + (11000 if big else 5000), "nvjet_gemm")
            if not big:
                t.frame(s + 6000, s + 6100, "megatron/core/pipeline_parallel/p2p_communication.py(1): send_forward")
                t.nccl(s + 6050, s + 11000, coll=None, kname="ncclDevKernel_SendRecv")
        if rank == stall_rank and gc_ms:
            t.frame(96000 - gc_ms, 96000, "<built-in function collect>")
        t.write(os.path.join(d, "trace_rank_%d.pt.trace.json.gz" % rank))


def _mk_cleanup_ranks(d, ranks, pressured=None, n_calls=10, call_ms=800.0):
    """PP ranks; the `pressured` rank calls a clear_memory helper (sync + gc.collect + empty_cache) n_calls times."""
    pgs = [{"pg_name": "5", "pg_desc": "PIPELINE_MODEL_PARALLEL_GROUP", "pg_size": len(ranks), "ranks": list(ranks),
            "backend_config": "cuda:nccl"}]
    for rank in ranks:
        t = Trace(rank=rank, world=max(ranks) + 1, pgs=pgs)
        wall = 20000.0 + n_calls * (call_ms * 1000 + 2000)
        t.step(0, wall, n=24)
        for k in range(4):
            t.kernel(k * 4000, k * 4000 + 3000, "nvjet_gemm")
        if rank in ((pressured,) if isinstance(pressured, int) else (pressured or ())):
            s = 20000.0
            for i in range(n_calls):
                e = s + call_ms * 1000
                t.frame(s - 50, e + 50, "slime/utils/reloadable_process_group.py(313): _wrap_low_level_call")
                t.frame(s, e, "slime/utils/memory_utils.py(11): clear_memory")
                t.frame(s + 10, s + 0.4 * (e - s), "<built-in function collect>")
                t.frame(s + 0.4 * (e - s), e - 10, "torch/cuda/memory.py(192): empty_cache")
                s = e + 2000
        t.write(os.path.join(d, "trace_rank_%d.pt.trace.json.gz" % rank))


def test_stage_memory_pressure_stall_flagged():
    tmp = tempfile.mkdtemp()
    try:
        _mk_cleanup_ranks(tmp, (0, 8, 16), pressured=16)
        txt = _cli(["--profile-dir", tmp])
        head = txt.splitlines()[:40]
        flag = [line for line in head if "STAGE MEMORY-PRESSURE STALL" in line]
        assert len(flag) == 1 and "pp2 (rank 16)" in flag[0] and "10 calls" in flag[0], head
        assert any("clear_memory" in line and "_wrap_low_level_call" in line for line in head), head
        d = json.loads(_cli(["--profile-dir", tmp, "--json"]))
        st = d["summary"]["stage_memory_pressure_stalls"]
        assert [x["pp"] for x in st] == [2] and st[0]["calls"] == 10, st
        assert close(st[0]["host_s"], 8.0, 1e-3), st
        mm = {r["rank"]: r["memory_mgmt"] for r in d["summary"]["per_rank"]}
        assert mm[0]["calls"] == 0 and mm[16]["by_kind"]["clear_memory"]["calls"] == 10, mm
        assert mm[16]["by_kind"]["gc_collect"]["calls"] == 10 and mm[16]["by_kind"]["empty_cache"]["calls"] == 10
        # all stages equally pressured: no single-stage flag
        tmp2 = tempfile.mkdtemp()
        try:
            _mk_cleanup_ranks(tmp2, (0, 8), pressured=(0, 8))
            d2 = json.loads(_cli(["--profile-dir", tmp2, "--json"]))
            assert d2["summary"]["stage_memory_pressure_stalls"] == [], d2["summary"]["stage_memory_pressure_stalls"]
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
        # one stage only: coverage names the missing stages
        tmp3 = tempfile.mkdtemp()
        try:
            _mk_cleanup_ranks(tmp3, (0, 8, 16), pressured=16)
            for r in (8, 16):
                os.remove(os.path.join(tmp3, "trace_rank_%d.pt.trace.json.gz" % r))
            t3 = _cli(["--profile-dir", tmp3])
            assert any("MISSING pp1,pp2" in line for line in t3.splitlines()[:6]), t3.splitlines()[:6]
            assert "only 1 of 3 pipeline stages profiled" in t3
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ranks_sorted_numerically_and_summary_first():
    tmp = tempfile.mkdtemp()
    try:
        _mk_ranks(tmp, (0, 8, 16, 24), stall_rank=16, gc_ms=30000)
        txt = _cli(["--profile-dir", tmp])
        rows = [line.split()[0] for line in txt.splitlines() if line[:2].strip().isdigit() and "pp" in line[:12]]
        assert rows[:4] == ["0", "8", "16", "24"], rows
        head = txt.splitlines()[:40]
        assert any("rank 16" in line and "host stall" in line and "gc" in line for line in head), head
        assert txt.index("== Summary") < txt.index("== Coverage")
        js = _cli(["--profile-dir", tmp, "--json"])
        d = json.loads(js)
        assert list(d)[0] == "summary" and d["schema"] == "profile_analyzer/v2.2"
        assert [r["rank"] for r in d["summary"]["per_rank"]] == [0, 8, 16, 24]
        assert [r["rank"] for r in d["ranks"]] == [0, 8, 16, 24]
        assert any("rank 16" in line and "host stall" in line for line in js.splitlines()[:40])
        assert "gaps_top" in d["ranks"][0]["primary"] and len(d["ranks"][0]["primary"]["gaps_top"]) <= 5
        full = json.loads(_cli(["--profile-dir", tmp, "--json-full"]))
        assert list(full)[0] == "summary" and "v1" in full["ranks"][0]
        q = _cli(["--profile-dir", tmp, "--quick"])
        assert "warm" in q.splitlines()[1] and "== Coverage" not in q and "--quick: summary only" in q
        qj = json.loads(_cli(["--profile-dir", tmp, "--json", "--quick"]))
        assert list(qj)[0] == "summary" and "ranks" not in qj
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_largest_bucket_names_compute_or_host():
    tmp = tempfile.mkdtemp()
    try:
        for rank in (0, 1):
            t = Trace(rank=rank, world=2)
            t.step(0, 100000)
            t.kernel(0, 20000 if rank == 0 else 90000, "nvjet_gemm")
            t.nccl(90000, 99000)
            if rank == 0:
                t.frame(20000, 90000, "<built-in function collect>")    # 70 ms of idle GPU under gc
            t.write(os.path.join(tmp, "rank_%d.pt.trace.json.gz" % rank))
        d = json.loads(_cli(["--profile-dir", tmp, "--json", "--no-cache"]))
        pr = {r["rank"]: r for r in d["summary"]["per_rank"]}
        assert pr[1]["largest_bucket"]["name"] == "compute", pr[1]
        assert pr[0]["largest_bucket"]["name"] == "host stall", pr[0]
        facts = d["summary"]["facts"]
        assert any(line.strip().startswith("rank 1") and "largest bucket compute" in line for line in facts), facts
        assert any(line.strip().startswith("rank 0") and "largest bucket host stall" in line and "[gc" in line for line in facts)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kernel_share_vs_step_share():
    t = Trace()
    t.step(0, 10000)
    t.kernel(0, 4000, "nvjet_gemm", stream=7)
    t.nccl(2000, 8000, stream=20)                  # 6000 kernel time, 2000 overlapped, 4000 exposed
    r, (res,) = analyze(t)
    ks = res["kernel_share"]["rows"]
    assert close(ks["TP"]["pct_of_gpu_activity_sum"], 60.0) and close(ks["TP"]["pct_of_step"], 40.0), ks["TP"]
    assert close(ks["compute"]["pct_of_gpu_activity_sum"], 40.0) and close(ks["compute"]["pct_of_step"], 40.0)
    check_conservation(res)


def test_unreserved_label_always_with_caveat():
    tmp = tempfile.mkdtemp()
    try:
        t = Trace(rank=0, world=1)
        t.step(0, 10000)
        t.kernel(0, 9000, "nvjet_gemm")
        t.mem(100, 10 * 2 ** 30, 12 * 2 ** 30)
        t.write(os.path.join(tmp, "rank_0.pt.trace.json.gz"))
        outs = [_cli(["--profile-dir", tmp]), _cli(["rank", "0", "--profile-dir", tmp]),
                _cli(["memory", "0", "--profile-dir", tmp]), _cli(["--profile-dir", tmp, "--quick"])]
        for o in outs:
            assert "headroom" not in o.lower()
            for line in o.splitlines():
                if "unreserved" in line.lower():
                    assert "NOT device free" in line, line
        assert any("unreserved" in o for o in outs)
        for argv in (["--profile-dir", tmp, "--json"], ["--profile-dir", tmp, "--json-full"],
                     ["memory", "0", "--profile-dir", tmp, "--json"], ["rank", "0", "--profile-dir", tmp, "--json"]):
            js = _cli(argv)
            assert "headroom" not in js.lower()
            n = [0]

            def walk(o):
                if isinstance(o, dict):
                    if "allocator_unreserved_gib" in o and o["allocator_unreserved_gib"] is not None:
                        assert "NOT device free" in o.get("allocator_unreserved_note", ""), o
                        n[0] += 1
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
                elif isinstance(o, str) and "unreserved" in o.lower():
                    assert "NOT device free" in o, o
            walk(json.loads(js))
            assert n[0] >= 1, argv
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_p2p_bytes_peer_and_wait_split():
    pgs = [{"pg_name": "5", "pg_desc": "PIPELINE_MODEL_PARALLEL_GROUP", "pg_size": 3, "ranks": [0, 8, 16],
            "backend_config": "cuda:nccl"}]
    t = Trace(rank=8, world=24, pgs=pgs)
    t.step(0, 50000)
    t.frame(1000, 12000, "megatron/core/pipeline_parallel/p2p_communication.py(1): send_forward_recv_backward")
    t.rpc(1100, 1150, "send", 25 * 10 ** 6, peer=16)            # 50 MB bf16 to the next stage
    t.rpc(1200, 1250, "recv", 25 * 10 ** 6, peer=16)            # 50 MB from the next stage
    t.nccl(1300, 11300, coll=None, kname="ncclDevKernel_SendRecv")   # 10 ms kernel
    t.frame(20000, 22000, "megatron/core/pipeline_parallel/p2p_communication.py(1): recv_forward")
    t.nccl(20100, 21100, coll=None, kname="ncclDevKernel_SendRecv")  # 1 ms kernel, no ops -> bytes unknown
    r, (res,) = analyze(t)
    x = res["p2p"]
    assert x["kernels"] == 2 and close(x["kernel_us"], 11000)
    # inter-node peer (rank 16 on another 8-GPU node): 50 MB / 50 GB/s = 1 ms transfer, 9 ms wait
    assert close(x["transfer_est_us"], 1000.0, 1e-3), x
    assert close(x["wait_est_us"], 9000.0, 1e-3), x
    assert close(x["bytes_unknown_kernel_us"], 1000.0), x
    top = x["by_peer"][0]
    assert top["peer"] == 16 and top["role"] == "recv" and close(top["kernel_us"], 10000)
    assert "waiting for the peer" in x["note"]
    check_conservation(res)
    run = {"ranks": [{"coords": {"rank": 8, "pp": 1, "pp_size": 3, "pp_ranks": [0, 8, 16]}, "primary": res}]}
    from profile_analyzer import pipeline as pl
    st = pl.stage_table(run["ranks"])[0]
    assert st["waits_on"][0]["stage"] == 2 and "pp2 (recv)" in pl.waits_text(st)


def test_optimizer_context_facts():
    t = Trace(rank=0, world=16)
    t.step(0, 40000)
    t.kernel(0, 5000, "nvjet_gemm")
    t.rpc(5000, 5100, "reduce_scatter_tensor_coalesced", 8 * 10 ** 9, dtype="Float", desc="DATA_PARALLEL_GROUP_WITH_CP",
          pg="6", gs=8)
    t.rpc(5200, 5300, "reduce_scatter_tensor_coalesced", 2 * 10 ** 9, dtype="Float", desc="EXPERT_DATA_PARALLEL_GROUP",
          pg="7", gs=1)
    t.frame(10000, 38000, "Optimizer.step#HybridDeviceOptimizer.step", cat="user_annotation")
    t.frame(10100, 30000, "megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py(150): step")
    t.frame(11000, 29000, "aten::_fused_adamw_", cat="cpu_op")
    t.memcpy(30000, 32000, "HtoD (Pageable -> Device)", nbytes=4 * 10 ** 9)
    r, (res,) = analyze(t)
    o = res["optimizer"]
    assert o["cpu_update_frames_present"] and any("HybridDeviceOptimizer" in x["frame"] for x in o["offload_indicators"])
    assert o["update_ops_host"][0]["op"] == "aten::_fused_adamw_" and close(o["update_ops_host"][0]["host_us"], 18000)
    assert close(o["copies_in_phase"]["H2D"]["bytes"], 4e9) and close(o["copies_in_phase"]["H2D"]["us"], 2000)
    sh = o["shard_estimate"]
    assert close(sh["params_on_rank"], 10e9) and close(sh["params_in_shard"], 1e9 + 2e9)
    assert close(sh["state_bytes_est"], 3e9 * 12)
    from profile_analyzer import summary as sm
    lines = sm.optimizer_lines([{"coords": {"rank": 0}, "primary": res}])
    assert "CPU-update frames present" in lines[0] and "Adam state" in lines[1] and "estimate" in lines[1]
    check_conservation(res)


def test_streaming_parser_matches_whole_load():
    t = Trace(rank=3, world=8)
    t.step(0, 20000)
    t.kernel(0, 3000, "nvjet_gemm_é中")                      # non-ASCII kernel name
    t.frame(3000, 9000, "a.py(1): f → g")
    t.memcpy(9000, 9500)
    t.mem(100, 2 ** 30, 2 * 2 ** 30)
    t.rpc(9600, 9700, "send", 1000, peer=4)
    t.nccl(9800, 12000, coll=None, kname="ncclDevKernel_SendRecv")
    t.top["traceName"] = "after-events"                                 # a top-level key after traceEvents
    tmp = tempfile.mkdtemp()
    try:
        for indent in (None, 1):
            f = os.path.join(tmp, "rank_3.%s.pt.trace.json" % indent)
            with open(f, "w") as fh:
                json.dump(t.top, fh, indent=indent, ensure_ascii=bool(indent))
            old = parse.STREAM_CHUNK
            parse.STREAM_CHUNK = 7                                       # force many refills mid-token
            try:
                top = {}
                m1, a1 = parse.extract_events(parse.iter_trace(f, top), top, f)
            finally:
                parse.STREAM_CHUNK = old
            m2, a2 = parse.extract(parse.read_json(f), f)
            assert top.get("traceName") == "after-events"
            assert m1["kernel_names"] == m2["kernel_names"] and m1["host_names"] == m2["host_names"]
            assert m1["dist"] == m2["dist"] and m1["device"] == m2["device"]
            for k in a2:
                eq = np.array_equal(a1[k], a2[k], equal_nan=True) if a2[k].dtype.kind == "f" else np.array_equal(a1[k], a2[k])
                assert eq, k
        bad = os.path.join(tmp, "rank_9.pt.trace.json")
        with open(bad, "w") as fh:
            fh.write('{"traceEvents": [{"ph": "X", "cat": "kernel"')
        try:
            list(parse.iter_trace(bad, {}))
            assert False, "truncated json must raise"
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_workers_degrade_when_ram_is_short():
    tmp = tempfile.mkdtemp()
    try:
        paths = []
        for i in range(4):
            f = os.path.join(tmp, "rank_%d.pt.trace.json.gz" % i)
            with open(f, "wb") as fh:
                fh.write(os.urandom(1 << 20))
            paths.append(f)
        old = cli._mem_available
        try:
            cli._mem_available = lambda: 64 << 30
            n, note = cli._workers(paths, {})
            assert n == min(4, os.cpu_count() or 1) and note is None
            cli._mem_available = lambda: int(1.5 * cli._need_bytes(paths[0]) / 0.8)
            n, note = cli._workers(paths, {})
            assert n == 1 and note and "RAM" in note
            cli._mem_available = lambda: 1 << 20
            n, note = cli._workers(paths, {})
            assert n == 1 and "may not fit" in note
        finally:
            cli._mem_available = old
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_all():
    names = sorted(k for k, v in globals().items() if k.startswith("test_") and callable(v))
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print("PASS", n)
        except Exception as e:  # pragma: no cover
            failed += 1
            import traceback
            traceback.print_exc()
            print("FAIL", n, e)
    print("%d passed, %d failed" % (len(names) - failed, failed))
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
