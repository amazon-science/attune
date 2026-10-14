# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The summary: the key facts of a run in the first ~40 lines of text, and the `summary` object of --json.

Measurements only. The "largest bucket" line names the biggest conserved bucket; it is not a diagnosis.
"""
from . import memory as memmod
from . import memmgmt
from . import taxonomy as tx

SHORT = {
    "gc": "gc", "allocator_cache_release": "cache-release", "device_sync_event_wait": "sync-wait",
    "blocking_allocation": "blocking-alloc", "host_device_copy_wait": "h<->d-copy", "comm_host_call": "comm-host",
    "data_loading": "data-load", "checkpoint_io": "ckpt-io", "optimizer_cpu_step": "optim-cpu",
    "kernel_launch_bound": "launch-bound", "python_other": "py-other",
}
HOST_LIST_PCT = 5.0
GIB = 1024.0 ** 3


def pct(x, w):
    return 100.0 * x / w if w else 0.0


def ok_ranks(run):
    return [r for r in run["ranks"] if not r.get("error")]


def pp_of(r):
    c = r["coords"]
    return ("pp%d" % c["pp"]) if c.get("pp") is not None and (c.get("pp_size") or 1) > 1 else None


def rlabel(r):
    p = pp_of(r)
    return "rank %s%s" % (r["coords"].get("rank"), (" (%s)" % p) if p else "")


def dim_label(d):
    if d == "PP":
        return "exposed PP p2p"
    if d == "other_comm":
        return "exposed other comm"
    return "exposed %s comm" % d


def step_buckets(p):
    """Conserved buckets at 'which resource the step is on' granularity: compute, exposed comm per parallel
    dimension, exposed memcpy, host stall, unattributed idle, fine-grained idle. They sum to the wall."""
    b = p["buckets_us"]
    rows = [("compute", b["compute"])]
    for d in tx.COMM_DIMS:
        ex = p["comm"][d]["exclusive_us"]
        if ex > 0:
            rows.append((dim_label(d), ex))
    rows += [("exposed memcpy", b["memcpy_exposed"]), ("host stall", b["host_total"]),
             ("unattributed idle", b["unattributed_idle"]), ("fine-grained idle (<1 ms gaps)", b["fine_idle"])]
    return rows


def _top_host(p):
    cl = sorted(p["host_classes"].items(), key=lambda kv: -kv[1]["us"])
    return cl[0] if cl else None


def _top_memcpy(p):
    m = sorted(((d, v["exclusive_us"]) for d, v in p["memcpy"].items() if v["exclusive_us"] > 0), key=lambda x: -x[1])
    return m[0] if m else None


def largest(r):
    p = r["primary"]
    w = p["window"]["wall_us"]
    rows = sorted(step_buckets(p), key=lambda x: -x[1])
    first, second = rows[0], rows[1] if len(rows) > 1 else (None, 0.0)
    note = ""
    if first[0] == "host stall" and _top_host(p):
        c, v = _top_host(p)
        note = "%s %.1f%%" % (SHORT.get(c, c), pct(v["us"], w))
    elif first[0] == "exposed memcpy" and _top_memcpy(p):
        d, v = _top_memcpy(p)
        note = "%s %.1f%%" % (d, pct(v, w))
    return {"name": first[0], "pct": pct(first[1], w), "note": note,
            "next_name": second[0], "next_pct": pct(second[1], w)}


def _rng(vals, fmt="%.1f"):
    lo, hi = min(vals), max(vals)
    return (fmt % lo) if abs(hi - lo) < 0.05 else ("%s-%s" % (fmt % lo, fmt % hi))


def _ranks_label(group):
    rk = ",".join(str(r["coords"].get("rank")) for r in group)
    pps = [pp_of(r) for r in group if pp_of(r)]
    if len(group) == 1:
        return rlabel(group[0])
    return "ranks %s%s" % (rk, (" (%s)" % (pps[0] if len(set(pps)) == 1 else "%s-%s" % (pps[0], pps[-1]))) if pps else "")


def largest_lines(rs):
    groups = []
    for r in rs:
        lg = largest(r)
        key = (lg["name"], lg["next_name"])
        if groups and groups[-1][0] == key:
            groups[-1][1].append((r, lg))
        else:
            groups.append((key, [(r, lg)]))
    out = []
    for (name, nxt), items in groups:
        rr = [x[0] for x in items]
        lgs = [x[1] for x in items]
        note = lgs[0]["note"] if len(set(x["note"].split(" ")[0] for x in lgs)) == 1 and lgs[0]["note"] else ""
        if note and len(lgs) > 1:
            note = "%s %s%%" % (note.split(" ")[0], _rng([float(x["note"].split(" ")[1].rstrip("%")) for x in lgs]))
        out.append("  %s: largest bucket %s %s%% of step%s (next: %s %s%%)" % (
            _ranks_label(rr), name, _rng([x["pct"] for x in lgs]), (" [%s]" % note) if note else "", nxt,
            _rng([x["next_pct"] for x in lgs])))
    return out


def _table_rows(rs):
    return rs if len(rs) <= 24 else _outlier_subset(rs)


def _outlier_subset(rs):
    """For large rank counts: per stage the min/max busy ranks, plus the global max host stall."""
    by = {}
    for r in rs:
        by.setdefault(r["coords"].get("pp"), []).append(r)
    keep = []
    for k in sorted(by, key=lambda x: (x is None, x)):
        g = sorted(by[k], key=lambda r: r["primary"]["gpu_busy_us"] / r["primary"]["window"]["wall_us"])
        for r in (g[0], g[-1]):
            if r not in keep:
                keep.append(r)
    worst = max(rs, key=lambda r: r["primary"]["buckets_us"]["host_total"])
    if worst not in keep:
        keep.append(worst)
    keep.sort(key=lambda r: r["coords"].get("rank") if r["coords"].get("rank") is not None else -1)
    return keep[:24]


def host_line(r):
    p = r["primary"]
    w = p["window"]["wall_us"]
    b = p["buckets_us"]
    cl = sorted(p["host_classes"].items(), key=lambda kv: -kv[1]["us"])
    parts = ["%s %.2f s (%d calls)" % (SHORT.get(c, c), v["us"] / 1e6, v["n_calls"]) for c, v in cl[:4]
             if v["us"] >= 0.005 * w]
    top = ""
    if cl and cl[0][1]["top_frames"]:
        top = "; top frame: %s" % _trunc(cl[0][1]["top_frames"][0]["frame"], 70)
    return "  %s: host stall %.1f s = %.1f%% of step: %s%s" % (rlabel(r), b["host_total"] / 1e6,
                                                              pct(b["host_total"], w), " | ".join(parts), top)


def _trunc(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n - 3] + "..."


def coverage_brief(run):
    cov = run["coverage"]
    rs = ok_ranks(run)
    ev = cov.get("events", {})

    def yn(b):
        return "yes" if b else "NO"
    s = "coverage: %d of %s ranks %s" % (len(rs), cov.get("world_size") or "?", _trunc(cov.get("ranks_profiled"), 60))
    if cov.get("pp_size", 1) > 1:
        st = cov.get("pp_stages_covered") or []
        s += "; PP stages %s of %d" % (",".join("pp%d" % x for x in st) if len(st) <= 4 else
                                       "pp%d-pp%d (%d)" % (st[0], st[-1], len(st)), cov["pp_size"])
        miss = [p for p in range(cov["pp_size"]) if p not in st]
        if miss:
            s += " (MISSING %s: stage-local stalls there are invisible; analyze one trace per stage)" % (
                ",".join("pp%d" % p for p in miss) if len(miss) <= 6 else "%d stages" % len(miss))
    s += "; step %s (%d per trace); stacks %s, [memory] %s, FLOPs %s; %d blind spots under Coverage" % (
        ",".join(cov.get("steps") or []), cov.get("steps_per_rank", 0), yn(ev.get("python_stacks")),
        yn(ev.get("memory_events")), yn(ev.get("flops_ops")), len(cov.get("blind_spots") or []))
    if cov.get("errors"):
        s += "; %d trace(s) FAILED to parse" % len(cov["errors"])
    return s


def _short_frame(name):
    path, func = tx.split_frame(name)
    return _trunc(("%s: %s" % (path.rsplit("/", 1)[-1], func)) if path else name, 70)


def memmgmt_lines(run):
    mv = run.get("memory_mgmt") or {}
    sts = mv.get("stages") or []
    if not sts:
        return []
    L = ["%s: %s; host wall time per step, GPU idle during it, calls:" % (memmgmt.LABEL, memmgmt.WHAT)]
    parts = []
    for st in sts:
        lab = ("pp%s r%s" % (st["pp"], st["rank"])) if st["pp"] is not None else "r%s" % st["rank"]
        parts.append("%s %.2f s (%.1f%%; idle %.2f s; %d calls)" % (lab, st["host_s"], st["pct_of_step"],
                                                                   st["gpu_idle_s"], st["calls"]))
    for i in range(0, len(parts), 3):
        L.append("  " + " | ".join(parts[i:i + 3]))
    for f in mv.get("flagged") or []:
        st = next(x for x in sts if x["pp"] == f["pp"])
        kinds = ", ".join("%s %.2f s x%d" % (k, v["s"], v["calls"]) for k, v in
                          sorted(st["by_kind"].items(), key=lambda kv: -kv[1]["s"]) if v["calls"])
        cal = ", ".join("%s x%d" % (_short_frame(c["frame"]), c["calls"]) for c in st["callers"][:2])
        pk = [x for x in f["others_peak_allocated_gib"] if x is not None]
        mem = (" ; peak allocated %.1f GiB vs %s GiB on other stages" % (f["peak_allocated_gib"], _rng(pk))) \
            if f.get("peak_allocated_gib") is not None and pk else ""
        L.append("  STAGE MEMORY-PRESSURE STALL: pp%s (rank %s) %.2f s/step = %.1f%% of step in cleanup, GPU idle %.2f s; "
                 "%d calls vs %s on other stages (%s s)%s" % (
                     f["pp"], f["rank"], f["host_s"], f["pct_of_step"], f["gpu_idle_s"], f["calls"],
                     _rng(f["others_calls"], "%d"), _rng(f["others_host_s"], "%.2f"), mem))
        L.append("    by kind: %s; called from: %s" % (kinds or "n/a", cal or "n/a"))
    if mv.get("flagged"):
        L.append("  (flag rule: %s)" % mv["rule"])
    elif mv.get("note"):
        L.append("  (%s)" % mv["note"])
    return L


def timing_line(run):
    t = run.get("timing") or {}
    n = t.get("n_traces", 0)
    if t.get("parsed", 0) == 0 and t.get("analyzed", 0) == 0:
        how = "warm: analysis cache hit for %d/%d traces" % (t.get("analysis_cache_hits", 0), n)
    else:
        how = "cold: %d of %d traces parsed%s, %d re-analyzed from the parse cache, %d from the analysis cache; " \
              "%d worker(s)" % (t.get("parsed", 0), n, (" with %s" % t["reader"]) if t.get("reader") else "",
                                t.get("analyzed", 0), t.get("analysis_cache_hits", 0), t.get("workers", 1))
        if t.get("peak_worker_rss_gib"):
            how += ", peak worker RSS %.1f GiB" % t["peak_worker_rss_gib"]
    s = "analysis took %.2f s (%s)" % (t.get("seconds", 0.0), how)
    if t.get("no_cache"):
        s += "; caches disabled"
    return s


def memory_lines(rs):
    mem = [(r, r["primary"]["memory"]) for r in rs]
    have = [(r, m) for r, m in mem if m.get("peak_allocated_gib") is not None]
    L = []
    if not have:
        dev = [m.get("device_total_gib") for _, m in mem if m.get("device_total_gib")]
        L.append("memory: no [memory] events (peak memory unknown)%s" % (
            ("; device %.1f GiB" % max(dev)) if dev else ""))
    else:
        dev = max(m.get("device_total_gib") or 0 for _, m in have)
        L.append("memory: peak allocated/reserved GiB %s; device %.1f GiB" % (
            " | ".join("r%s %.1f/%.1f" % (r["coords"].get("rank"), m["peak_allocated_gib"], m["peak_reserved_gib"])
                       for r, m in have[:12]), dev))
        un = [(r, m) for r, m in have if m.get("allocator_unreserved_gib") is not None]
        if un:
            s = "  %s: %s" % (memmod.UNRESERVED_LABEL, " | ".join(
                "r%s %.1f" % (r["coords"].get("rank"), m["allocator_unreserved_gib"]) for r, m in un[:12]))
            L.append(s)
    return L


def _gb(b):
    return b / 1e9


def optimizer_lines(rs):
    ops = [(r, r["primary"].get("optimizer") or {}) for r in rs]
    off = [(r, o) for r, o in ops if o.get("cpu_update_frames_present")]
    L = []
    if not off:
        ph = [o.get("phase_wall_us", 0.0) / 1e6 for _, o in ops]
        gk = [o.get("gpu_optimizer_kernel_us", 0.0) / 1e6 for _, o in ops]
        ho = [o.get("host_optimizer_us", 0.0) / 1e6 for _, o in ops]
        s = ("optimizer: no CPU-offload frames on any rank; optimizer phase %s s, GPU optimizer kernels %s s, "
             "GPU-idle host optimizer frames %s s" % (_rng(ph, "%.2f"), _rng(gk, "%.2f"), _rng(ho, "%.2f")))
        sh = [o.get("shard_estimate") for _, o in ops if o.get("shard_estimate")]
        if sh:
            s += "; optimizer shard %s B params -> Adam state %s GiB (12 B/param, estimate)" % (
                _rng([x["params_in_shard"] / 1e9 for x in sh], "%.2f"), _rng([x["state_bytes_est"] / GIB for x in sh]))
        L.append(s)
        return L
    for r, o in off[:4]:
        ind = ", ".join("%s x%d" % (_trunc(x["frame"], 60), x["n"]) for x in o["offload_indicators"][:2])
        upd = ", ".join("%s %.2f s (%d calls)" % (x["op"], x["host_us"] / 1e6, x["n"]) for x in o["update_ops_host"][:1])
        L.append("optimizer %s: CPU-update frames present (%s); optimizer phase %.2f s; GPU-idle host optimizer "
                 "frames %.2f s; host time in update ops %s; GPU optimizer kernels %.2f s" % (
                     rlabel(r), ind, o["phase_wall_us"] / 1e6, o["host_optimizer_us"] / 1e6, upd or "none recorded",
                     o["gpu_optimizer_kernel_us"] / 1e6))
        cp = o.get("copies_in_phase") or {}
        cps = ", ".join("%s %.1f GB / %.2f s" % (d, _gb(v["bytes"]), v["us"] / 1e6) for d, v in sorted(cp.items()))
        sh = o.get("shard_estimate")
        m = r["primary"]["memory"]
        s = "  copies in optimizer phase: %s" % (cps or "none")
        if sh:
            s += "; params on rank %.2f B, in its optimizer shard %.2f B (DP reduce-scatter sizes) -> Adam state " \
                 "~%.1f GiB at 12 B/param (estimate)" % (sh["params_on_rank"] / 1e9, sh["params_in_shard"] / 1e9,
                                                          sh["state_bytes_est"] / GIB)
        else:
            s += "; optimizer shard size unknown (no DP reduce-scatter sizes in the trace)"
        s += "; device %s, peak allocated %s" % (
            ("%.1f GiB" % m["device_total_gib"]) if m.get("device_total_gib") else "?",
            ("%.1f GiB" % m["peak_allocated_gib"]) if m.get("peak_allocated_gib") is not None else "n/a (no [memory] events)")
        L.append(s)
    if len(off) > 4:
        L.append("  ... %d more ranks with CPU-update frames (see `rank R`)" % (len(off) - 4))
    return L


def pipeline_lines(run):
    pv = run.get("pipeline") or {}
    if pv.get("pp_size", 1) <= 1 or not pv.get("stages"):
        return []
    L = ["pipeline (PP=%d, %s microbatches):" % (pv["pp_size"], pv.get("microbatches") or "?")]
    for f in pv.get("facts", []):
        L.append("  " + f)
    return L


def build(run):
    """(lines, summary dict). Lines are the head of the text overview; summary['facts'] repeats them."""
    rs = ok_ranks(run)
    L = ["PROFILE ANALYZER | %s" % run["dir"], timing_line(run),
         "Measurements only, no diagnosis. Detail and blind spots follow this summary; verify surprising numbers "
         "against the raw trace (paths listed below)."]
    if not rs:
        L.append("no trace could be analyzed: %s" % "; ".join("%s: %s" % (r["path"], r.get("error")) for r in run["ranks"]))
        return L, {"facts": L, "per_rank": []}
    L.append(coverage_brief(run))
    L.append("== Summary: step decomposition per rank (% of step wall; compute..fine sum to 100) ==")
    L.append("%-6s %-4s %7s %6s | %7s %8s %8s %7s %7s %6s" % (
        "rank", "pp", "wall s", "busy%", "compute", "comm-exp", "memcpy-x", "host", "unattr", "fine"))
    rows = _table_rows(rs)
    for r in rows:
        p = r["primary"]
        b = p["buckets_us"]
        w = p["window"]["wall_us"]
        L.append("%-6s %-4s %7.2f %6.1f | %7.1f %8.1f %8.1f %7.1f %7.1f %6.1f" % (
            r["coords"].get("rank"), pp_of(r) or "-", w / 1e6, pct(p["gpu_busy_us"], w), pct(b["compute"], w),
            pct(b["comm_exposed"], w), pct(b["memcpy_exposed"], w), pct(b["host_total"], w),
            pct(b["unattributed_idle"], w), pct(b["fine_idle"], w)))
    if len(rows) < len(rs):
        L.append("(%d of %d ranks shown: per-stage extremes; `ranks` lists all)" % (len(rows), len(rs)))
    cons = [(r["coords"].get("rank"), c) for r in rs
            for c in [r["primary"]["conservation"]] + [x["conservation"] for x in (r.get("per_step") or [])]]
    bad = sorted(set(k for k, c in cons if not c["ok"]))
    L.append("conservation: max |residual| %.4f%% of wall over %d ranks x %d step(s) (tolerance 0.1%%) %s" % (
        max(abs(c["residual_pct"]) for _, c in cons), len(rs), max(1, max(len(r.get("per_step") or []) for r in rs)),
        "OK" if not bad else "VIOLATED on ranks %s" % bad))
    L.append("largest step-time bucket per rank (compute | exposed comm per dim | exposed memcpy | host stall | "
             "unattributed | fine idle):")
    L += largest_lines(rows)
    heavy = [r for r in rs if pct(r["primary"]["buckets_us"]["host_total"], r["primary"]["window"]["wall_us"]) >= HOST_LIST_PCT]
    if heavy:
        L.append("host stall (GPU idle >= 1 ms while a host frame runs), ranks with >= %.0f%% of step:" % HOST_LIST_PCT)
        L += [host_line(r) for r in heavy[:6]]
        if len(heavy) > 6:
            L.append("  ... %d more ranks (see Host-side idle below)" % (len(heavy) - 6))
    else:
        mx = max(rs, key=lambda r: pct(r["primary"]["buckets_us"]["host_total"], r["primary"]["window"]["wall_us"]))
        L.append("host stall: no rank >= %.0f%% of step (largest %s %.1f%%)" % (
            HOST_LIST_PCT, rlabel(mx), pct(mx["primary"]["buckets_us"]["host_total"], mx["primary"]["window"]["wall_us"])))
    L += memmgmt_lines(run)
    cs = run.get("contrasts") or []
    if cs:
        L.append("top contrasts across ranks:")
        L += ["  %d. %s" % (i + 1, c["text"]) for i, c in enumerate(cs[:5])]
    L += pipeline_lines(run)
    L += memory_lines(rs)
    L += optimizer_lines(rs)
    return L, {"facts": L, "analysis": run.get("timing"), "coverage_brief": coverage_brief(run),
               "per_rank": [rank_summary(r) for r in rs], "contrasts": [c["text"] for c in cs[:5]],
               "pipeline_facts": (run.get("pipeline") or {}).get("facts", []),
               "memory_mgmt_stages": (run.get("memory_mgmt") or {}).get("stages", []),
               "stage_memory_pressure_stalls": (run.get("memory_mgmt") or {}).get("flagged", []),
               "blind_spots": run["coverage"].get("blind_spots", [])}


def rank_summary(r):
    p = r["primary"]
    w = p["window"]["wall_us"]
    b = p["buckets_us"]
    m = p["memory"]
    lg = largest(r)
    hs = sorted(p["host_classes"].items(), key=lambda kv: -kv[1]["us"])
    out = {"rank": r["coords"].get("rank"), "pp": r["coords"].get("pp"), "wall_s": w / 1e6,
           "busy_pct": pct(p["gpu_busy_us"], w),
           "step_pct": {k: pct(v, w) for k, v in (("compute", b["compute"]), ("comm_exposed", b["comm_exposed"]),
                                                  ("memcpy_exposed", b["memcpy_exposed"]), ("host", b["host_total"]),
                                                  ("unattributed_idle", b["unattributed_idle"]),
                                                  ("fine_idle", b["fine_idle"]))},
           "exposed_comm_pct_by_dim": {d: pct(p["comm"][d]["exclusive_us"], w) for d in tx.COMM_DIMS
                                       if p["comm"][d]["exclusive_us"] > 0},
           "largest_bucket": {"name": lg["name"], "pct_of_step": lg["pct"], "detail": lg["note"]},
           "next_bucket": {"name": lg["next_name"], "pct_of_step": lg["next_pct"]},
           "host_stall": {"s": b["host_total"] / 1e6, "pct_of_step": pct(b["host_total"], w),
                          "classes": [{"class": c, "s": v["us"] / 1e6, "calls": v["n_calls"],
                                       "top_frame": v["top_frames"][0]["frame"] if v["top_frames"] else None}
                                      for c, v in hs[:5]]},
           "kernel_vs_step_share_pct": {k: {"of_gpu_activity_sum": v["pct_of_gpu_activity_sum"],
                                            "of_step": v["pct_of_step"]}
                                        for k, v in (p.get("kernel_share") or {}).get("rows", {}).items()},
           "memory": {"peak_allocated_gib": m.get("peak_allocated_gib"), "peak_reserved_gib": m.get("peak_reserved_gib"),
                      "device_total_gib": m.get("device_total_gib"),
                      "allocator_unreserved_gib": m.get("allocator_unreserved_gib"),
                      "allocator_unreserved_note": memmod.UNRESERVED_NOTE}}
    mm = p.get("memory_mgmt")
    if mm:
        out["memory_mgmt"] = {k: mm[k] for k in ("host_s", "pct_of_step", "gpu_idle_s", "calls", "by_kind", "callers")}
    x = p.get("p2p") or {}
    if x.get("kernels"):
        out["pp_p2p"] = {"kernel_s": x["kernel_us"] / 1e6, "exposed_s": p["comm"]["PP"]["exposed_us"] / 1e6,
                         "transfer_est_s": x.get("transfer_est_us", 0.0) / 1e6,
                         "wait_est_s": x.get("wait_est_us", 0.0) / 1e6,
                         "bytes_unknown_kernel_s": x.get("bytes_unknown_kernel_us", 0.0) / 1e6,
                         "max_kernel_s": x.get("max_kernel_us", 0.0) / 1e6, "note": x.get("note")}
    o = p.get("optimizer") or {}
    if o:
        sh = o.get("shard_estimate") or {}
        out["optimizer"] = {"cpu_update_frames_present": o.get("cpu_update_frames_present"),
                            "offload_indicators": o.get("offload_indicators"),
                            "phase_wall_s": o.get("phase_wall_us", 0.0) / 1e6,
                            "host_optimizer_idle_s": o.get("host_optimizer_us", 0.0) / 1e6,
                            "gpu_optimizer_kernel_s": o.get("gpu_optimizer_kernel_us", 0.0) / 1e6,
                            "update_ops_host": o.get("update_ops_host"),
                            "copies_in_phase": o.get("copies_in_phase"),
                            "params_on_rank": sh.get("params_on_rank"), "params_in_shard": sh.get("params_in_shard"),
                            "adam_state_gib_est": (sh["state_bytes_est"] / GIB) if sh else None,
                            "state_assumption": sh.get("state_assumption")}
    return out
