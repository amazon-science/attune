# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side memory management: time the host spends in allocator cleanup, per rank and per pipeline stage.

Counts wall time in cleanup calls whether or not the GPU is idle meanwhile: framework cleanup helpers
(clear_memory and similar), torch.cuda.empty_cache, gc.collect, the host allocator's empty-cache, and
cudaMalloc/cudaFree-family runtime calls. Nested calls count once (outermost frame). A stage whose cleanup
time or call count is far above the other stages' is flagged as a stage memory-pressure stall.
"""
import numpy as np

from . import intervals as iv
from . import memory as memmod
from . import taxonomy as tx

# Function-name keys (lowercased, substring) of cleanup frames, most specific kind first.
KINDS = (
    ("clear_memory", ("clear_memory", "clear_cuda_cache", "empty_cuda_cache", "release_memory")),
    ("empty_cache", ("empty_cache", "emptycache", "_host_emptycache", "release_cached", "releasepool")),
    ("gc_collect", ("<built-in function collect>", "gc.collect", "gc_collect", "collect_garbage")),
    ("malloc_trim", ("malloc_trim",)),
)
# cudaMalloc/cudaFree family (not cudaMemGetInfo, which is a query)
ALLOC_KEYS = ("malloc", "free", "memcreate", "memmap", "memrelease", "hostalloc", "hostregister")
KIND_NAMES = tuple(k for k, _ in KINDS) + ("alloc_api",)
# Stage flag: cleanup >= FLAG_MIN_PCT of the step and >= FLAG_MIN_S, and >= FLAG_RATIO x the other stages'
# median (time floored at FLAG_FLOOR_PCT of the step; calls floored at FLAG_MIN_CALLS).
FLAG_MIN_PCT = 2.0
FLAG_MIN_S = 0.5
FLAG_RATIO = 3.0
FLAG_FLOOR_PCT = 0.5
FLAG_MIN_CALLS = 5
RULE = ("cleanup >= %.0f%% of step and >= %.1f s, and >= %.0fx the other stages' median in time or calls" %
        (FLAG_MIN_PCT, FLAG_MIN_S, FLAG_RATIO))
LABEL = "memory management (host-side allocator cleanup)"
WHAT = "clear_memory-style helpers, empty_cache, gc.collect, cudaMalloc/cudaFree"


def kind_of(name):
    path, func = tx.split_frame(name)
    fl = func.lower()
    for kind, keys in KINDS:
        if any(k in fl for k in keys):
            return kind
    return None


def _caller(r, i):
    """Innermost non-cleanup frame on the same thread that encloses frame i (its name, or None)."""
    a = r.a
    idx = r.thread_frames(int(a["h_tid"][i]))
    s, e = r.h_start[i], r.h_end[i]
    k = np.searchsorted(r.h_start[idx], s, side="right")
    cand = idx[:k]
    cand = cand[(r.h_end[cand] >= e) & (cand != i)]
    best, bd = None, np.inf
    for c in cand[-64:].tolist():
        nm = r.hnames[int(a["h_name"][c])]
        if kind_of(nm) is None and a["h_dur"][c] < bd and not nm.startswith("ProfilerStep#"):
            best, bd = nm, a["h_dur"][c]
    return best


def window(r, t0, t1, idle):
    """Cleanup facts for one rank's step window. `idle` = the window's GPU-idle interval set."""
    a = r.a
    wall = t1 - t0
    hn = r.hnames
    kinds = [kind_of(n) for n in hn]
    anym = np.array([k is not None for k in kinds], dtype=bool) if hn else np.zeros(0, bool)
    top = memmod._outermost(r, anym, t0, t1)
    by = {}
    for kn in KIND_NAMES[:-1]:
        km = np.array([k == kn for k in kinds], dtype=bool) if hn else np.zeros(0, bool)
        f = memmod._outermost(r, km, t0, t1)
        if f.size:
            by[kn] = {"calls": int(f.size), "s": float(iv.total(iv.clip(iv.union(r.h_start[f], r.h_end[f]), t0, t1))) / 1e6}
    fu = iv.clip(iv.union(r.h_start[top], r.h_end[top]), t0, t1) if top.size else iv.EMPTY
    am = (a["a_ts"] < t1) & (a["a_ts"] + a["a_dur"] > t0)
    if am.any():
        ok = np.array([any(k in n.lower() for k in ALLOC_KEYS) for n in hn], dtype=bool)
        am &= ok[a["a_name"]]
    au = iv.clip(iv.union(a["a_ts"][am], a["a_ts"][am] + a["a_dur"][am]), t0, t1) if am.any() else iv.EMPTY
    api_in = int(iv.contains_points(fu, a["a_ts"][am]).sum()) if am.any() and fu[0].size else 0
    if am.any():
        by["alloc_api"] = {"calls": int(am.sum()), "s": iv.total(au) / 1e6, "inside_cleanup_calls": api_in}
    u = iv.union(np.concatenate([fu[0], au[0]]), np.concatenate([fu[1], au[1]])) if (fu[0].size or au[0].size) \
        else iv.EMPTY
    host_us = iv.total(u)
    idle_us = iv.overlap_total(u, idle)
    calls = int(top.size) + max(int(am.sum()) - api_in, 0)
    callers = {}
    for i in top[np.argsort(-a["h_dur"][top])][:200].tolist():
        c = _caller(r, i) or "(none)"
        x = callers.setdefault(c, [0, 0.0])
        x[0] += 1
        x[1] += float(a["h_dur"][i])
    durs = np.sort(a["h_dur"][top])[::-1] / 1e6 if top.size else np.zeros(0)
    return {"host_s": host_us / 1e6, "pct_of_step": 100.0 * host_us / wall if wall else 0.0,
            "gpu_idle_s": idle_us / 1e6, "gpu_idle_pct_of_step": 100.0 * idle_us / wall if wall else 0.0,
            "calls": calls, "cleanup_frames": int(top.size), "by_kind": by,
            "longest_calls_s": [float(x) for x in durs[:5]],
            "callers": [{"frame": k, "calls": v[0], "s": v[1] / 1e6}
                        for k, v in sorted(callers.items(), key=lambda kv: -kv[1][1])[:3]],
            "alloc_api_recorded": bool(am.any())}


def _median(xs):
    return float(np.median(xs)) if len(xs) else 0.0


def stage_view(ranks):
    """Per pipeline stage (worst profiled rank in each), plus stages flagged as memory-pressure stalls."""
    rs = [r for r in ranks if not r.get("error") and r["primary"].get("memory_mgmt")]
    if not rs:
        return {"stages": [], "flagged": [], "rule": RULE, "note": "no analyzed ranks"}
    pps = max((r["coords"].get("pp_size") or 1) for r in rs)
    by = {}
    for r in rs:
        k = r["coords"].get("pp") if pps > 1 else None
        by.setdefault(k, []).append(r)
    stages = []
    for k in sorted(by, key=lambda x: (x is None, x)):
        w = max(by[k], key=lambda r: r["primary"]["memory_mgmt"]["host_s"])
        m = w["primary"]["memory_mgmt"]
        stages.append({"pp": k, "rank": w["coords"].get("rank"), "ranks": [r["coords"].get("rank") for r in by[k]],
                       "wall_s": w["primary"]["window"]["wall_us"] / 1e6, "host_s": m["host_s"],
                       "pct_of_step": m["pct_of_step"], "gpu_idle_s": m["gpu_idle_s"], "calls": m["calls"],
                       "by_kind": m["by_kind"], "callers": m["callers"],
                       "peak_allocated_gib": w["primary"]["memory"].get("peak_allocated_gib")})
    out = {"pp_size": pps, "stages": stages, "flagged": [], "rule": RULE}
    if pps <= 1 or len(stages) < 2:
        out["note"] = ("no stage comparison: %s" % ("PP=1" if pps <= 1 else
                                                     "only %d of %d pipeline stages profiled" % (len(stages), pps)))
        return out
    for st in stages:
        others = [x for x in stages if x is not st]
        floor_s = FLAG_FLOOR_PCT / 100.0 * st["wall_s"]
        mt = max(_median([x["host_s"] for x in others]), floor_s)
        mc = max(_median([x["calls"] for x in others]), FLAG_MIN_CALLS)
        big = st["pct_of_step"] >= FLAG_MIN_PCT and st["host_s"] >= FLAG_MIN_S
        if big and (st["host_s"] >= FLAG_RATIO * mt or st["calls"] >= FLAG_RATIO * mc):
            st["flag"] = "stage memory-pressure stall"
            out["flagged"].append({"pp": st["pp"], "rank": st["rank"], "host_s": st["host_s"],
                                   "pct_of_step": st["pct_of_step"], "gpu_idle_s": st["gpu_idle_s"],
                                   "calls": st["calls"], "others_host_s": [x["host_s"] for x in others],
                                   "others_calls": [x["calls"] for x in others],
                                   "ratio_time": st["host_s"] / mt if mt else None,
                                   "peak_allocated_gib": st["peak_allocated_gib"],
                                   "others_peak_allocated_gib": [x["peak_allocated_gib"] for x in others]})
    return out
