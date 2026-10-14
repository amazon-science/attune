# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text rendering: overview and drill-down subcommands. Measurements only; no verdicts."""

from . import memory as memmod
from . import pipeline as pl
from . import summary as sm
from . import taxonomy as tx

SHORT = {
    "gc": "gc", "allocator_cache_release": "cache-release", "device_sync_event_wait": "sync-wait",
    "blocking_allocation": "blocking-alloc", "host_device_copy_wait": "h<->d-copy", "comm_host_call": "comm-host",
    "data_loading": "data-load", "checkpoint_io": "ckpt-io", "optimizer_cpu_step": "optim-cpu",
    "kernel_launch_bound": "launch-bound", "python_other": "py-other",
}


def s_(us):
    return "%.2f" % (us / 1e6)


def pct(x, w):
    return 100.0 * x / w if w else 0.0


def trunc(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n - 3] + "..."


def ok_ranks(run):
    return [r for r in run["ranks"] if not r.get("error")]


# ---------------- coverage ----------------
def coverage(run):
    rs = ok_ranks(run)
    cov = {"ranks_profiled": [r["coords"].get("rank") for r in rs], "errors": [
        {"path": r["path"], "error": r["error"]} for r in run["ranks"] if r.get("error")]}
    if not rs:
        return cov
    ws = max((r["coords"].get("world_size") or 0) for r in rs)
    cov["world_size"] = ws
    pps = max((r["coords"].get("pp_size") or 1) for r in rs)
    cov["pp_size"] = pps
    cov["pp_stages_covered"] = sorted(set(r["coords"].get("pp") for r in rs if r["coords"].get("pp") is not None))
    cov["steps"] = sorted(set(r["primary"]["window"]["step"] for r in rs))
    cov["steps_per_rank"] = max(len(r["windows"]) for r in rs)
    cov["window_source"] = sorted(set(r["primary"]["window"]["source"] for r in rs))
    ev = {}
    ev["python_stacks"] = all(r["event_counts"].get("python_function", 0) > 0 for r in rs)
    ev["shapes"] = all(bool(r["flags"].get("record_shapes")) for r in rs)
    ev["memory_events"] = all(r["primary"]["memory"].get("has_memory_events") for r in rs)
    ev["flops_ops"] = sum(r.get("flops_ops", 0) for r in rs)
    ev["cuda_runtime"] = all(r["event_counts"].get("cuda_runtime", 0) > 0 for r in rs)
    ev["malloc_free_api"] = any(r["primary"]["memory"].get("malloc_free_recorded") for r in rs)
    nk = sum(r["comm_meta"]["n"] for r in rs)
    ninf = sum(r["comm_meta"]["no_pg"] for r in rs)
    ev["nccl_kernels"] = nk
    ev["nccl_pg_metadata_pct"] = (100.0 * (nk - ninf) / nk) if nk else None
    ev["nccl_inferred"] = {}
    for r in rs:
        for k, v in r["comm_meta"]["inferred"].items():
            ev["nccl_inferred"][k] = ev["nccl_inferred"].get(k, 0) + v
    cov["events"] = ev
    blind = []
    if ws and len(rs) < ws:
        blind.append("ranks not profiled: %d of %d; per-rank numbers cover only the listed ranks -> "
                     "SLIME_PROFILE_ALL_RANKS=1 (profile all ranks)" % (ws - len(rs), ws))
    if pps > 1 and len(cov["pp_stages_covered"]) < pps:
        miss = [p for p in range(pps) if p not in cov["pp_stages_covered"]]
        blind.append("pipeline stages without a trace: %s -> the critical stage may be unprofiled" %
                     ",".join("pp%d" % p for p in miss))
    if not ev["python_stacks"]:
        blind.append("no python stacks: idle time resolves only to cpu_op/annotation names (gc, empty_cache, data "
                     "loading callers invisible) -> with_stack=True (SLIME_PROFILE_WITH_STACK=1)")
    if not ev["memory_events"]:
        blind.append("no [memory] events: peak memory unknown -> profile_memory=True (SLIME_PROFILE_MEMORY=1)")
    blind.append("device free memory is not in the trace; allocator unreserved (device total - peak reserved) is "
                 "NOT device free memory and overstates it (NCCL buffers, workspaces, fragmentation)")
    if not ev["flops_ops"]:
        blind.append("no FLOPs recorded: achieved TFLOP/s unknown -> with_flops=True (SLIME_PROFILE_WITH_FLOPS=1); "
                     "TE / custom GEMMs report no FLOPs even then")
    if not ev["malloc_free_api"]:
        blind.append("cudaMalloc/cudaFree not in the trace: allocator churn visible only as blocking allocations "
                     "and empty_cache calls")
    if ninf:
        inf = ", ".join("%s %d" % (k, v) for k, v in sorted(ev["nccl_inferred"].items()) if k != "deepep-kernel")
        blind.append("%d NCCL kernels carry no process-group metadata; dimension inferred (%s)" % (ninf, inf))
    blind.append("bus bandwidth = bytes / kernel duration, and kernel duration includes waiting for peers -> "
                 "a lower bound on link speed, not a link measurement; PP p2p transfer time is only estimated "
                 "(bytes / nominal link bandwidth)")
    outside = [(r["coords"].get("rank"), r.get("outside_window_us", 0.0), r["primary"]["window"]["wall_us"])
               for r in rs if r.get("outside_window_us", 0.0) > 0.05 * r["primary"]["window"]["wall_us"]]
    if outside:
        blind.append("GPU activity outside the analyzed step window(s) is not decomposed: %s -> inspect the raw trace "
                     "or profile whole steps only" % ", ".join("rank %s %.1f s" % (k, o / 1e6) for k, o, _ in outside[:6]))
    if cov["steps_per_rank"] <= 1:
        blind.append("one profiled step per rank: step-to-step variance unknown (profile >= 2 steps)")
    blind.append("host threads that launch no kernels (e.g. a gc/logging thread) are not used to attribute idle time")
    cov["blind_spots"] = blind
    return cov


# ---------------- overview ----------------
def overview(run, top_contrasts=5, quick=False):
    """Summary (key facts, first ~40 lines) then detail sections."""
    L, _ = sm.build(run)
    rs = ok_ranks(run)
    if quick or not rs:
        if quick:
            L.append("(--quick: summary only; run without --quick for coverage, kernel shares, pipeline, comm, memcpy "
                     "and host detail)")
        return "\n".join(L)
    L.append("")
    L.append("======== detail ========")
    L.append("traces (%d, rank order):" % len(run["ranks"]))
    shown = run["ranks"] if len(run["ranks"]) <= 8 else run["ranks"][:6]
    for r in shown:
        L.append("  rank %-4s %s%s" % (r.get("rank", "?"), r["path"], ("  ERROR: " + r["error"]) if r.get("error") else ""))
    if len(run["ranks"]) > len(shown):
        L.append("  ... %d more (see `ranks`)" % (len(run["ranks"]) - len(shown)))
    cov = run["coverage"]
    L.append("")
    L.append("== Coverage ==")
    L.append("ranks profiled: %d of world %s: %s" % (len(rs), cov.get("world_size"), trunc(cov["ranks_profiled"], 90)))
    if cov.get("pp_size", 1) > 1:
        L.append("pipeline stages covered: %s of PP=%d" % (",".join("pp%d" % p for p in cov["pp_stages_covered"]),
                                                            cov["pp_size"]))
    L.append("step window: %s; %d step(s) per trace; %s" % (",".join(cov["steps"]), cov["steps_per_rank"],
                                                            "; ".join(cov["window_source"])))
    ev = cov["events"]

    def yn(b):
        return "yes" if b else "NO"
    L.append("events: python stacks %s | shapes %s | [memory] %s | FLOPs %s | CUDA runtime %s | cudaMalloc/Free %s | "
             "NCCL PG metadata %s" % (yn(ev["python_stacks"]), yn(ev["shapes"]), yn(ev["memory_events"]),
                                     yn(ev["flops_ops"]), yn(ev["cuda_runtime"]), yn(ev["malloc_free_api"]),
                                     ("%.1f%%" % ev["nccl_pg_metadata_pct"]) if ev["nccl_pg_metadata_pct"] is not None else "n/a"))
    L.append("cannot answer from this trace (-> what would):")
    for b in cov["blind_spots"]:
        L.append("  - " + b)
    L.append("host = idle >= 1 ms spanned by a host frame on a kernel-launching thread; unattr = idle >= 1 ms with no "
             "such frame (or only step-level frames); fine = idle intervals < 1 ms")
    multi = [r for r in rs if len(r.get("per_step") or []) > 1]
    if multi:
        r0 = multi[0]
        L.append("per-step walls (rank %s): %s; tables show the last complete step (--step N selects another)" % (
            r0["coords"].get("rank"), ", ".join("%s %.2f s" % (x["step"].split("#")[-1], x["wall_us"] / 1e6)
                                                for x in r0["per_step"])))
    ks = [r["primary"]["kernels"] for r in rs if r["primary"]["kernels"].get("count")]
    if ks:
        lb = [pct(r["primary"]["idle"]["launch_api_overlap_idle_us"], r["primary"]["window"]["wall_us"]) for r in rs]
        L.append("kernels: median %.1f-%.1f us, %.0f-%.0f%% of launches < 10 us; GPU idle while host in launch APIs "
                 "%.1f-%.1f%% of wall" % (min(k["median_us"] for k in ks), max(k["median_us"] for k in ks),
                                          min(k["small_lt10us_count_pct"] for k in ks),
                                          max(k["small_lt10us_count_pct"] for k in ks), min(lb), max(lb)))

    L.append("")
    L += share_table(rs)

    pv = run["pipeline"]
    if pv.get("pp_size", 1) > 1 and pv.get("stages"):
        L.append("")
        L.append("== Pipeline (PP=%d, %s microbatches) ==" % (pv["pp_size"], pv.get("microbatches") or "?"))
        L.append("%-6s %-10s %8s %8s %11s %8s %9s %11s %9s  %s" % (
            "stage", "ranks", "wall s", "work s", "p2p-wait s", "host s", "peak GiB", "p2p-kern s", "xfer-est",
            "waits on (receives from)"))
        for st in pv["stages"]:
            L.append("%-6s %-10s %8.2f %8.2f %11.2f %8.2f %9s %11.2f %9.2f  %s" % (
                "pp%d" % st["pp"], trunc(",".join(str(x) for x in st["ranks"]), 10), st["wall_us"] / 1e6,
                st["work_us"] / 1e6, st["p2p_wait_us"] / 1e6, st["host_stall_us"] / 1e6,
                ("%.1f" % st["peak_alloc_gib"]) if st["peak_alloc_gib"] is not None else "n/a",
                st.get("p2p_kernel_us", 0.0) / 1e6, st.get("p2p_transfer_est_us", 0.0) / 1e6, pl.waits_text(st, 3)))
        L.append("work = GPU busy minus PP p2p kernels not overlapped by compute; p2p wait = those kernels. A p2p kernel "
                 "runs from launch until the peer's matching send/recv completes, so its duration includes waiting for "
                 "the peer; xfer-est = bytes / nominal link bandwidth (estimate)")

    L.append("")
    L.append("== Communication by dimension (mean per profiled rank; exposed = not overlapped by compute) ==")
    L.append("%-6s %-34s %7s %8s %8s %16s %9s %-11s %7s" % ("dim", "collectives (count/rank)", "GB", "kern s",
                                                          "union s", "exposed s (min-max)", "busbw GB/s",
                                                          "placement", "vs nom"))
    for row in comm_by_dim(run):
        L.append("%-6s %-34s %7.1f %8.2f %8.2f %16s %9s %-11s %7s" % (
            "other" if row["dim"] == "other_comm" else row["dim"], trunc(row["colls"], 34), row["gb"], row["sum_us"] / 1e6, row["union_us"] / 1e6,
            "%.2f (%.2f-%.2f)" % (row["exposed_us"] / 1e6, row["exposed_min_us"] / 1e6, row["exposed_max_us"] / 1e6),
            ("%.0f" % row["busbw"]) if row["busbw"] and row["bw_cov"] >= 0.5 else "n/a", row["placement"],
            ("%.2f" % row["vs_nominal"]) if row["vs_nominal"] and row["bw_cov"] >= 0.5 else "n/a"))
    L.append("busbw n/a when bytes are known for < 50% of the dimension's kernel time (e.g. coalesced p2p)")
    L.append("busbw uses NCCL factors (all-reduce 2(n-1)/n, all-gather/reduce-scatter/all-to-all (n-1)/n, p2p 1) and "
             "'vs nom' is against nominal NVLink/network per GPU; kernel time includes waiting for peers")

    mc = [r for r in rs if pct(r["primary"]["buckets_us"]["memcpy_exposed"], r["primary"]["window"]["wall_us"]) >= 1.0]
    if mc:
        L.append("")
        L.append("== Exposed memcpy by direction (ranks with >= 1% of wall; s exposed / s overlapping compute / GB) ==")
        for r in mc[:8]:
            p = r["primary"]
            parts = ["%s %.2f/%.2f/%.1f" % (d, v["exposed_us"] / 1e6, v["overlap_compute_us"] / 1e6, v["bytes"] / 1e9)
                     for d, v in p["memcpy"].items() if v["count"] and (v["union_us"] >= 0.001 * p["window"]["wall_us"])]
            L.append("  rank %-5s %s" % (r["coords"].get("rank"), " | ".join(parts)))
        if len(mc) > 8:
            L.append("  ... %d more ranks" % (len(mc) - 8))

    heavy = [r for r in rs if pct(r["primary"]["buckets_us"]["host_total"], r["primary"]["window"]["wall_us"]) >= 5.0]
    L.append("")
    L.append("== Host-side idle by class (ranks with host >= 5% of wall) ==")
    if not heavy:
        mx = max(rs, key=lambda r: pct(r["primary"]["buckets_us"]["host_total"], r["primary"]["window"]["wall_us"]))
        L.append("  none; largest is rank %s at %.1f%% of wall" % (mx["coords"].get("rank"), pct(
            mx["primary"]["buckets_us"]["host_total"], mx["primary"]["window"]["wall_us"])))
    for r in heavy[:8]:
        p = r["primary"]
        w = p["window"]["wall_us"]
        cl = sorted(p["host_classes"].items(), key=lambda kv: -kv[1]["us"])
        L.append("  rank %s: host %.1f s = %.1f%% of wall" % (r["coords"].get("rank"), p["buckets_us"]["host_total"] / 1e6,
                                                           pct(p["buckets_us"]["host_total"], w)))
        for c, v in cl[:4]:
            if v["us"] < 0.005 * w:
                continue
            fr = v["top_frames"][0]["frame"] if v["top_frames"] else ""
            L.append("     %-15s %7.2f s  %5d calls  top: %s" % (SHORT.get(c, c), v["us"] / 1e6, v["n_calls"], trunc(fr, 70)))
        mem = p["memory"]
        extra = []
        if mem.get("empty_cache_calls"):
            extra.append("empty_cache calls %d (%.0f ms)" % (mem["empty_cache_calls"], mem["empty_cache_ms"]))
        if mem.get("blocking_alloc_ge5ms_n"):
            extra.append("allocations >= 5 ms: %d (%.0f ms)" % (mem["blocking_alloc_ge5ms_n"], mem["blocking_alloc_ge5ms_ms"]))
        if extra:
            L.append("     " + "; ".join(extra))
    if len(heavy) > 8:
        L.append("  ... %d more ranks (see `rank R`)" % (len(heavy) - 8))
    L.append("")
    worst = max(rs, key=lambda r: r["primary"]["buckets_us"]["host_total"])
    L.append("drill down: rank %s | gaps %s --top 20 | phases %s | memory %s | comm --dim EP | kernels --class gemm | "
             "stack %s --t0 MS --t1 MS | diff DIR_A DIR_B | --json [--json-full] | --quick" % tuple([worst["coords"].get("rank")] * 5))
    return "\n".join(L)


def _share_cols(rs):
    cols = ["compute"]
    for d in tx.COMM_DIMS:
        if any(d in (r["primary"].get("kernel_share") or {}).get("rows", {}) for r in rs):
            cols.append(d)
    cols.append("memcpy")
    return cols


def share_table(rs, title=True):
    """Each compute/comm/memcpy bucket as a share of summed GPU activity time and of step wall, side by side."""
    cols = _share_cols(rs)
    L = []
    if title:
        L.append("== Kernel-time share vs step-wall share per rank (kern%/step%) ==")
    L.append("%-6s %-4s %s" % ("rank", "pp", " ".join("%13s" % ("other" if c == "other_comm" else c) for c in cols)))
    for r in rs[:24]:
        rows = (r["primary"].get("kernel_share") or {}).get("rows", {})
        cells = []
        for c in cols:
            v = rows.get(c)
            cells.append("%13s" % (("%.1f/%.1f" % (v["pct_of_gpu_activity_sum"], v["pct_of_step"])) if v else "-"))
        L.append("%-6s %-4s %s" % (r["coords"].get("rank"), _pp(r), " ".join(cells)))
    L.append("kern% = share of summed GPU kernel+memcpy durations (overlapping streams counted twice; what v1's "
             "'Communication / Total Kernel' measured); step% = exposed share of step wall (the conserved buckets)")
    return L


def _pp(r):
    c = r["coords"]
    return ("pp%d" % c["pp"]) if c.get("pp") is not None and (c.get("pp_size") or 1) > 1 else "-"


def _cnt(x):
    return ("%d" % round(x)) if x >= 1 else ("%.1f" % x)


def comm_by_dim(run):
    rs = ok_ranks(run)
    n = float(len(rs)) or 1.0
    rows = []
    for dim in tx.COMM_DIMS:
        tot = {"count": 0, "sum_us": 0.0, "union_us": 0.0, "exposed_us": 0.0, "bytes": 0.0}
        exps, colls, place, nom = [], {}, set(), []
        for r in rs:
            c = r["primary"]["comm"][dim]
            tot["sum_us"] += c["sum_us"]
            tot["union_us"] += c["union_us"]
            tot["exposed_us"] += c["exposed_us"]
            tot["count"] += c["count"]
            exps.append(c["exposed_us"])
            for ct, v in (r["primary"]["comm_bw"].get(dim) or {}).items():
                colls[ct] = colls.get(ct, 0) + v["count"]
                tot["bytes"] += v["bytes"]
                place.add(v["placement"])
                if v["nominal_gbps"]:
                    nom.append(v["nominal_gbps"])
        if tot["count"] == 0:
            continue
        # aggregate bus bandwidth: sum(bus_bytes)/sum(time with bytes) over all ranks and collective types
        bus_b, bus_t = 0.0, 0.0
        for r in rs:
            for ct, v in (r["primary"]["comm_bw"].get(dim) or {}).items():
                if v.get("bus_bytes") and v.get("bw_time_us"):
                    bus_b += v["bus_bytes"]
                    bus_t += v["bw_time_us"]
        busbw = bus_b / (bus_t * 1e-6) / 1e9 if bus_t > 0 else None
        pls = sorted(p for p in place if p != "unknown") or ["unknown"]
        if len(pls) > 1:
            pls = ["mixed"]
        nomv = max(nom) if nom and len(pls) == 1 else (min(nom) if nom else None)
        rows.append({"dim": dim, "colls": ", ".join("%s %s" % (k, _cnt(v / n)) for k, v in sorted(colls.items(), key=lambda kv: -kv[1])),
                     "gb": tot["bytes"] / n / 1e9, "sum_us": tot["sum_us"] / n, "union_us": tot["union_us"] / n,
                     "exposed_us": tot["exposed_us"] / n, "exposed_min_us": min(exps), "exposed_max_us": max(exps),
                     "busbw": busbw, "bw_cov": (bus_t / tot["sum_us"]) if tot["sum_us"] else 0.0,
                     "placement": "/".join(pls), "vs_nominal": (busbw / nomv) if (busbw and nomv) else None})
    return rows


# ---------------- ranks ----------------
def ranks_table(run):
    L = ["%-6s %-5s %-24s %8s %6s %s" % ("rank", "pp", "coords", "wall s", "busy%", "trace")]
    for r in run["ranks"]:
        if r.get("error"):
            L.append("%-6s ERROR %s  %s" % (r.get("rank", "?"), r["error"], r["path"]))
            continue
        c = r["coords"]
        cs = " ".join("%s%s/%s" % (k, c[k], c.get(k + "_size")) for k in ("tp", "cp", "dp", "ep") if c.get(k) is not None)
        p = r["primary"]
        L.append("%-6s %-5s %-24s %8.2f %6.1f %s" % (c.get("rank"), _pp(r), cs, p["window"]["wall_us"] / 1e6,
                                                    pct(p["gpu_busy_us"], p["window"]["wall_us"]), r["path"]))
    return "\n".join(L)


# ---------------- rank detail ----------------
def rank_detail(r, top_n=10):
    if r.get("error"):
        return "rank %s: ERROR %s (%s)" % (r.get("rank"), r["error"], r["path"])
    p = r["primary"]
    w = p["window"]["wall_us"]
    b = p["buckets_us"]
    L = ["RANK %s  %s  coords %s" % (r["coords"].get("rank"), r["path"], _coords_str(r["coords"]))]
    L.append("device %s | step %s | window %.2f s (%s) | driver threads %s" % (
        r["device"].get("name"), p["window"]["step"], w / 1e6, p["window"]["source"], r["driver_threads"]))
    L.append("")
    L.append("Decomposition of step wall (sums to 100%):")
    L.append("  %-44s %9s %6s" % ("bucket", "s", "%"))
    L.append("  %-44s %9s %6.1f" % ("compute (any compute kernel running)", s_(b["compute"]), pct(b["compute"], w)))
    for c, v in p["compute"].items():
        if v["exclusive_us"] > 0:
            L.append("    %-42s %9s %6.1f   work(sum) %s s, overlap %s s, n=%d" % (
                c, s_(v["exclusive_us"]), pct(v["exclusive_us"], w), s_(v["sum_us"]),
                s_(max(v["sum_us"] - v["union_us"], 0)), v["count"]))
    L.append("  %-44s %9s %6.1f" % ("communication exposed (no compute running)", s_(b["comm_exposed"]), pct(b["comm_exposed"], w)))
    for c, v in p["comm"].items():
        if v["count"]:
            L.append("    %-42s %9s %6.1f   own union %s s: exposed %s, overlapped %s; n=%d" % (
                c, s_(v["exclusive_us"]), pct(v["exclusive_us"], w), s_(v["union_us"]), s_(v["exposed_us"]),
                s_(v["overlapped_us"]), v["count"]))
    L.append("  %-44s %9s %6.1f" % ("memcpy/memset exposed (no compute/comm)", s_(b["memcpy_exposed"]), pct(b["memcpy_exposed"], w)))
    for c, v in p["memcpy"].items():
        if v["count"]:
            L.append("    %-42s %9s %6.1f   union %s s, overlaps compute %s s, %.2f GB, n=%d" % (
                c, s_(v["exclusive_us"]), pct(v["exclusive_us"], w), s_(v["union_us"]), s_(v["overlap_compute_us"]),
                v["bytes"] / 1e9, v["count"]))
    L.append("  %-44s %9s %6.1f" % ("host-side idle (>= 1 ms, host frame active)", s_(b["host_total"]), pct(b["host_total"], w)))
    for c in tx.HOST_CLASSES:
        v = p["host_classes"].get(c)
        if not v:
            continue
        L.append("    %-42s %9s %6.1f   %d calls; top: %s" % (
            tx.HOST_CLASS_LABEL[c], s_(v["us"]), pct(v["us"], w), v["n_calls"],
            "; ".join("%s %.0f ms x%d" % (trunc(f["frame"], 60), f["us"] / 1e3, f["n"]) for f in v["top_frames"][:2])))
    L.append("  %-44s %9s %6.1f" % ("unattributed idle (>= 1 ms, no host frame)", s_(b["unattributed_idle"]), pct(b["unattributed_idle"], w)))
    L.append("  %-44s %9s %6.1f   %d intervals; host in launch APIs during them %s s" % (
        "fine-grained idle (intervals < 1 ms)", s_(b["fine_idle"]), pct(b["fine_idle"], w), p["idle"]["fine_n"],
        s_(p["idle"]["launch_api_overlap_fine_us"])))
    c = p["conservation"]
    L.append("  conservation: sum %s s vs wall %s s, residual %.3f ms (%.4f%%) %s" % (
        s_(c["sum_us"]), s_(c["wall_us"]), c["residual_us"] / 1e3, c["residual_pct"], "OK" if c["ok"] else "VIOLATED"))
    L.append("  GPU busy %.1f%% | launch-bound signal: GPU idle while host in launch APIs %s s" % (
        pct(p["gpu_busy_us"], w), s_(p["idle"]["launch_api_overlap_idle_us"])))
    lg = sm.largest(r)
    L.append("  largest bucket: %s %.1f%% of step%s (next: %s %.1f%%)" % (
        lg["name"], lg["pct"], (" [%s]" % lg["note"]) if lg["note"] else "", lg["next_name"], lg["next_pct"]))
    L.append("")
    L += share_table([r])
    x = p.get("p2p") or {}
    if x.get("kernels"):
        L.append("")
        L.append("PP p2p: %d kernels, %.2f s kernel time (exposed %.2f s); longest %.2f s, median %.1f ms" % (
            x["kernels"], x["kernel_us"] / 1e6, p["comm"]["PP"]["exposed_us"] / 1e6, x.get("max_kernel_us", 0) / 1e6,
            x.get("median_kernel_us", 0) / 1e3))
        L.append("  %s" % x["note"])
        L.append("  bytes known for %.2f s of kernel time (%.2f GB): est. transfer %.2f s, wait %.2f s; bytes unknown "
                 "for %.2f s (transfer/wait split unknown there)" % (
                     x.get("bytes_known_kernel_us", 0) / 1e6, x.get("bytes", 0) / 1e9, x.get("transfer_est_us", 0) / 1e6,
                     x.get("wait_est_us", 0) / 1e6, x.get("bytes_unknown_kernel_us", 0) / 1e6))
        prk = r["coords"].get("pp_ranks") or []
        for bp in (x.get("by_peer") or [])[:4]:
            who = ("rank %s (pp%d)" % (bp["peer"], prk.index(bp["peer"]))) if bp["peer"] in prk else "rank %s" % bp["peer"]
            L.append("  %-9s %-18s %4d kernels %8.2f s kernel, est. wait %8.2f s" % (
                bp["role"], who, bp["n"], bp["kernel_us"] / 1e6, bp["wait_est_us"] / 1e6))
    o = p.get("optimizer") or {}
    if o:
        L.append("")
        L += sm.optimizer_lines([r]) if o.get("cpu_update_frames_present") else [
            "optimizer: no CPU-offload frames; phase %.2f s, GPU optimizer kernels %.2f s, GPU-idle host optimizer "
            "frames %.2f s" % (o.get("phase_wall_us", 0) / 1e6, o.get("gpu_optimizer_kernel_us", 0) / 1e6,
                               o.get("host_optimizer_us", 0) / 1e6)]
    L.append("")
    L.append(phases_text(r, header=False))
    k = p["kernels"]
    if k.get("count"):
        L.append("")
        L.append("Kernels: %d, median %.1f us, %.1f%% under 10 us (%.1f%% of kernel time)" % (
            k["count"], k["median_us"], k["small_lt10us_count_pct"], k["small_lt10us_time_pct"]))
        for t in k["top"][:top_n]:
            L.append("  %8.1f ms  n=%-6d %-16s %s" % (t["us"] / 1e3, t["count"], t["class"], trunc(t["kernel"], 80)))
        if k.get("tflops"):
            for cn, v in k["tflops"].items():
                L.append("  achieved %s: %.0f TFLOP/s (%s of dense BF16 peak) over %d/%d kernels with FLOPs" % (
                    cn, v["tflops"], ("%.0f%%" % (v["vs_dense_bf16_peak"] * 100)) if v["vs_dense_bf16_peak"] else "?",
                    v["kernels_with_flops"], v["of_class_kernels"]))
        else:
            L.append("  achieved TFLOP/s: unknown (no FLOPs recorded for GEMM/attention kernels)")
    L.append("")
    L.append(memory_text(r, None, header=False))
    return "\n".join(L)


def _coords_str(c):
    return " ".join("%s%s/%s" % (k, c[k], c.get(k + "_size")) for k in ("pp", "tp", "cp", "dp", "ep", "etp")
                    if c.get(k) is not None)


def phases_text(r, header=True):
    p = r["primary"]
    L = []
    if header:
        L.append("RANK %s phases (%s)" % (r["coords"].get("rank"), p["window"]["step"]))
    L.append("Phases (host-timeline windows; method: %s):" % ("; ".join(p["phase_method"]) or "none found"))
    L.append("  %-10s %8s %8s %9s %9s %8s %8s %7s  top host class" % ("phase", "wall s", "compute", "comm-exp",
                                                                   "memcpy-x", "host", "unattr", "fine"))
    for ph in tx.PHASES:
        v = p["phases"].get(ph)
        if not v:
            continue
        host = sum(v["host"].values())
        top = max(v["host"].items(), key=lambda kv: kv[1]) if v["host"] else None
        L.append("  %-10s %8.2f %8.2f %9.2f %9.2f %8.2f %8.2f %7.2f  %s" % (
            ph, v["wall_us"] / 1e6, v["compute"] / 1e6, v["comm_exposed"] / 1e6, v["memcpy_exposed"] / 1e6,
            host / 1e6, v["unattributed_idle"] / 1e6, v["fine_idle"] / 1e6,
            ("%s %.2f s" % (SHORT.get(top[0], top[0]), top[1] / 1e6)) if top and top[1] > 0 else "-"))
    if p.get("microbatches"):
        L.append("  microbatches (forward_step frames): %d" % p["microbatches"])
    return "\n".join(L)


def memory_text(r, tl=None, header=True):
    m = r["primary"]["memory"]
    L = []
    if header:
        L.append("RANK %s memory (%s)" % (r["coords"].get("rank"), r["primary"]["window"]["step"]))
    if m.get("has_memory_events") and m.get("peak_allocated_gib") is not None:
        L.append("Memory: peak allocated %.2f GiB at +%.0f ms (%s); peak reserved %.2f GiB at +%.0f ms (%s); device %.1f GiB"
                 % (m["peak_allocated_gib"], m["peak_allocated_t_ms"], m["peak_allocated_phase"], m["peak_reserved_gib"],
                    m["peak_reserved_t_ms"], m["peak_reserved_phase"], m["device_total_gib"] or 0))
        if m.get("allocator_unreserved_gib") is not None:
            L.append("  %s: %.2f GiB" % (memmod.UNRESERVED_LABEL, m["allocator_unreserved_gib"]))
    else:
        L.append("Memory: no [memory] events in this trace (profile_memory=True / SLIME_PROFILE_MEMORY=1)")
    L.append("  churn: empty_cache calls %d (%.0f ms); allocations >= 5 ms %d (%.0f ms); allocator API: %s" % (
        m["empty_cache_calls"], m["empty_cache_ms"], m["blocking_alloc_ge5ms_n"], m["blocking_alloc_ge5ms_ms"],
        ", ".join("%s n=%d" % (k, v["count"]) for k, v in list(m["allocator_api"].items())[:4]) or "none recorded"))
    mm = r["primary"].get("memory_mgmt")
    if mm:
        L.append("  memory management (host-side cleanup, whether or not the GPU idles): %.2f s = %.1f%% of step, GPU idle "
                 "%.2f s, %d calls; %s; longest %s s; called from: %s" % (
                     mm["host_s"], mm["pct_of_step"], mm["gpu_idle_s"], mm["calls"],
                     ", ".join("%s %.2f s x%d" % (k, v["s"], v["calls"]) for k, v in mm["by_kind"].items()) or "none",
                     ", ".join("%.2f" % x for x in mm["longest_calls_s"][:3]) or "-",
                     ", ".join("%s x%d" % (trunc(c["frame"], 70), c["calls"]) for c in mm["callers"]) or "-"))
    if tl:
        L.append("  timeline (max per bin, GiB):  t(ms)   alloc  reserved")
        for x in tl:
            if x["max_alloc_gib"] is None:
                continue
            L.append("                              %7.0f  %6.2f  %6.2f%s" % (x["t_ms"], x["max_alloc_gib"],
                                                                         x["max_reserved_gib"], "" if x["samples"] else "  (carried)"))
    return "\n".join(L)


def gaps_text(r, top=20):
    p = r["primary"]
    L = ["RANK %s largest GPU-idle intervals (>= 1 ms) in %s; t = ms from step start" % (
        r["coords"].get("rank"), p["window"]["step"])]
    L.append("  %9s %9s %-15s %6s  deciding frame | innermost frame | outer->inner chain" % ("t ms", "dur ms", "class", "attr%"))
    for g in p["gaps_top"][:top]:
        chain = " > ".join(trunc(x, 40) for x in reversed(g.get("chain", [])[:5]))
        L.append("  %9.1f %9.2f %-15s %6.0f  %s | %s | %s" % (
            g["t_ms"], g["dur_ms"], SHORT.get(g.get("class"), g.get("class", "?")), g.get("attributed_pct", 0),
            trunc(g.get("decided_by", "-"), 50), trunc(g.get("innermost", "-"), 50), chain))
    L.append("  coarse idle: %d intervals, %.2f s; fine idle: %d intervals, %.2f s" % (
        p["idle"]["coarse_n"], p["idle"]["coarse_us"] / 1e6, p["idle"]["fine_n"], p["idle"]["fine_us"] / 1e6))
    return "\n".join(L)


def comm_text(run, dim=None):
    L = ["%-6s %-5s %-6s %-15s %7s %9s %9s %10s %-11s %6s  %s" % (
        "rank", "pp", "dim", "collective", "count", "GB", "kern s", "busbw GB/s", "placement", "vs nom", "groups")]
    for r in ok_ranks(run):
        p = r["primary"]
        for d in tx.COMM_DIMS:
            if dim and d.upper() != dim.upper():
                continue
            for ct, v in sorted((p["comm_bw"].get(d) or {}).items(), key=lambda kv: -kv[1]["time_us"]):
                L.append("%-6s %-5s %-6s %-15s %7d %9.2f %9.2f %10s %-11s %6s  n=%s %s%s" % (
                    r["coords"].get("rank"), _pp(r), d, ct, v["count"], v["bytes"] / 1e9, v["time_us"] / 1e6,
                    ("%.0f" % v["busbw_gbps"]) if v["busbw_gbps"] else "n/a", v["placement"],
                    ("%.2f" % v["vs_nominal"]) if v["vs_nominal"] else "n/a", v["group_sizes"],
                    ",".join(v["pg_descs"])[:60], (" inferred:" + ",".join(v["inferred"])) if v["inferred"] else ""))
    L.append("")
    L.append("exposure per rank and dim (s): union / exposed / overlapped")
    for r in ok_ranks(run):
        p = r["primary"]
        parts = []
        for d in tx.COMM_DIMS:
            if dim and d.upper() != dim.upper():
                continue
            c = p["comm"][d]
            if c["count"]:
                parts.append("%s %.2f/%.2f/%.2f" % (d, c["union_us"] / 1e6, c["exposed_us"] / 1e6, c["overlapped_us"] / 1e6))
        L.append("  rank %-5s %s" % (r["coords"].get("rank"), " | ".join(parts)))
    return "\n".join(L)


def kernels_text(run, cls=None, rank=None, top_n=20):
    L = []
    for r in ok_ranks(run):
        if rank is not None and r["coords"].get("rank") != rank:
            continue
        k = r["primary"]["kernels"]
        if not k.get("count"):
            continue
        L.append("rank %s: %d kernels, median %.1f us, %.1f%% under 10 us" % (
            r["coords"].get("rank"), k["count"], k["median_us"], k["small_lt10us_count_pct"]))
        L.append("  %-16s %7s %9s %9s %9s %8s" % ("class", "count", "sum s", "union s", "median us", "<10us %"))
        for c, v in sorted(k["by_class"].items(), key=lambda kv: -kv[1]["sum_us"]):
            if cls and c != cls:
                continue
            un = r["primary"]["compute"].get(c, r["primary"]["comm"].get(c, r["primary"]["memcpy"].get(c, {})))
            L.append("  %-16s %7d %9.2f %9.2f %9.1f %8.1f" % (c, v["count"], v["sum_us"] / 1e6,
                                                              un.get("union_us", 0) / 1e6, v["median_us"],
                                                              v["small_lt10us_count_pct"]))
        n = 0
        for t in k["top"]:
            if cls and t["class"] != cls:
                continue
            L.append("    %8.1f ms n=%-6d %-16s %s" % (t["us"] / 1e3, t["count"], t["class"], trunc(t["kernel"], 90)))
            n += 1
            if n >= top_n:
                break
    return "\n".join(L) if L else "no kernels matched"


# ---------------- diff ----------------
def _bucket_rows(p):
    b = p["buckets_us"]
    rows = [("compute", b["compute"])]
    for d in tx.COMM_DIMS:
        rows.append(("comm-exposed " + d, p["comm"][d]["exclusive_us"]))
    for d in tx.MEMCPY_DIRS:
        rows.append(("memcpy-exposed " + d, p["memcpy"][d]["exclusive_us"]))
    for c in tx.HOST_CLASSES:
        rows.append(("host " + SHORT[c], b["host"].get(c, 0.0)))
    rows.append(("unattributed idle", b["unattributed_idle"]))
    rows.append(("fine idle", b["fine_idle"]))
    return rows


def diff_text(runA, runB):
    A = {pl.role_key(r["coords"]): r for r in ok_ranks(runA)}
    B = {pl.role_key(r["coords"]): r for r in ok_ranks(runB)}
    L = ["DIFF  A=%s\n      B=%s" % (runA["dir"], runB["dir"])]
    keys = [k for k in A if k in B]
    how = "role (pp, tp, cp, ep, dp)"
    if not keys:
        A = {r["coords"].get("rank"): r for r in ok_ranks(runA)}
        B = {r["coords"].get("rank"): r for r in ok_ranks(runB)}
        keys = [k for k in A if k in B]
        how = "global rank (roles differ between runs)"
    L.append("matched %d rank(s) by %s; unmatched A=%d B=%d" % (len(keys), how, len(A) - len(keys), len(B) - len(keys)))
    keys.sort(key=lambda k: (A[k]["coords"].get("pp") is None, A[k]["coords"].get("pp") or 0, A[k]["coords"].get("rank") or 0))
    for k in keys:
        a, b = A[k], B[k]
        pa, pb = a["primary"], b["primary"]
        wa, wb = pa["window"]["wall_us"], pb["window"]["wall_us"]
        L.append("")
        L.append("%s: A rank %s -> B rank %s | wall %.2f -> %.2f s (%+.2f s) | busy %.1f%% -> %.1f%%" % (
            _pp(a) if _pp(a) != "-" else "role", a["coords"].get("rank"), b["coords"].get("rank"), wa / 1e6, wb / 1e6,
            (wb - wa) / 1e6, pct(pa["gpu_busy_us"], wa), pct(pb["gpu_busy_us"], wb)))
        ra, rb = _bucket_rows(pa), _bucket_rows(pb)
        for (name, va), (_, vb) in zip(ra, rb):
            if max(va, vb) < 0.01 * max(wa, wb):
                continue
            L.append("   %-24s %8.2f -> %8.2f s  (%+8.2f)   %5.1f%% -> %5.1f%% of wall" % (
                name, va / 1e6, vb / 1e6, (vb - va) / 1e6, pct(va, wa), pct(vb, wb)))
    pa_, pb_ = runA.get("pipeline", {}), runB.get("pipeline", {})
    if pa_.get("stages") and pb_.get("stages"):
        L.append("")
        L.append("pipeline: critical stage A pp%s -> B pp%s" % (pa_.get("critical_pp"), pb_.get("critical_pp")))
    return "\n".join(L)


def stack_text(r, rank_obj, t0_ms, t1_ms, max_lines=80):
    """Host frames on kernel-launching threads around [t0, t1] (ms from step start).

    Frames enclosing the whole window are collapsed into one context line; the rest are depth-indented."""
    p = r["primary"]
    base = p["window"]["t0"]
    t0, t1 = base + t0_ms * 1e3, base + t1_ms * 1e3
    a = rank_obj.a

    def nm(i):
        return rank_obj.hnames[int(a["h_name"][i])]
    L = ["RANK %s host frames overlapping +%.1f..+%.1f ms (%s); columns: start ms, dur ms, frame" % (
        r["coords"].get("rank"), t0_ms, t1_ms, p["window"]["step"])]
    for tid in rank_obj.driver_tids:
        idx = rank_obj.thread_frames(tid)
        s = rank_obj.h_start[idx]
        e = rank_obj.h_end[idx]
        sel = idx[(s < t1) & (e > t0)]
        enc = [i for i in sel.tolist() if rank_obj.h_start[i] <= t0 and rank_obj.h_end[i] >= t1]
        inner = [i for i in sel.tolist() if not (rank_obj.h_start[i] <= t0 and rank_obj.h_end[i] >= t1)]
        L.append("thread %s (%s): %d enclosing frames, %d inside/partial" % (
            tid, r.get("thread_names", {}).get(str(tid), "?"), len(enc), len(inner)))
        if enc:
            L.append("  enclosing (outer->inner, last 8): " + " > ".join(trunc(nm(i), 60) for i in enc[-8:]))
        stack = []
        n = 0
        for i in inner:
            while stack and rank_obj.h_end[stack[-1]] <= rank_obj.h_start[i]:
                stack.pop()
            depth = len(stack)
            stack.append(i)
            if n < max_lines:
                L.append("  %+10.2f %9.2f %s%s" % ((rank_obj.h_start[i] - base) / 1e3, a["h_dur"][i] / 1e3,
                                                   "  " * min(depth, 12), trunc(nm(i), 110)))
            n += 1
        if n > max_lines:
            L.append("  ... %d more frames (narrow the window)" % (n - max_lines))
    g = rank_obj.g_sel
    m = (rank_obj.g_start[g] < t1) & (rank_obj.g_end[g] > t0)
    L.append("GPU activities overlapping the window: %d" % int(m.sum()))
    return "\n".join(L)
