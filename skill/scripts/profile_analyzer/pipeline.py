# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-rank views: parallel coordinates, pipeline stages, contrasts."""
import numpy as np


COORD_GROUPS = (
    ("pp", ("PIPELINE_MODEL_PARALLEL_GROUP",)),
    ("tp", ("TENSOR_MODEL_PARALLEL_GROUP",)),
    ("cp", ("CONTEXT_PARALLEL_GROUP",)),
    ("dp", ("DATA_PARALLEL_GROUP", "DATA_PARALLEL_GROUP_WITH_CP")),
    ("ep", ("EXPERT_MODEL_PARALLEL_GROUP",)),
    ("etp", ("EXPERT_TENSOR_PARALLEL_GROUP",)),
)


def coords(meta):
    """(pp, tp, cp, dp, ep, etp) index and size of this rank, from distributedInfo process groups."""
    rank = (meta.get("dist") or {}).get("rank")
    out = {"rank": rank, "world_size": (meta.get("dist") or {}).get("world_size")}
    pgs = meta.get("pg_config") or []
    for key, descs in COORD_GROUPS:
        for d in descs:
            hit = [pg for pg in pgs if (pg.get("desc") or "").upper() == d and "nccl" in (pg.get("backend") or "nccl")]
            if hit:
                rk = [int(x) for x in hit[0].get("ranks") or []]
                out[key] = rk.index(rank) if rank in rk else None
                out[key + "_size"] = len(rk)
                if key == "pp":
                    out["pp_ranks"] = rk
                break
    return out


def role_key(c):
    return (c.get("pp"), c.get("tp"), c.get("cp"), c.get("ep"), c.get("dp"))


def pp_wait_us(p):
    return p["comm"]["PP"]["exposed_us"]


def stage_table(ranks):
    """Per pipeline stage: work, p2p wait, host stall, peak memory (mean over profiled ranks of the stage)."""
    by = {}
    for r in ranks:
        if r.get("error"):
            continue
        pp = r["coords"].get("pp")
        if pp is None:
            continue
        by.setdefault(pp, []).append(r)
    rows = []
    for pp in sorted(by):
        rs = by[pp]
        prim = [x["primary"] for x in rs]
        wall = np.mean([p["window"]["wall_us"] for p in prim])
        wait = np.mean([pp_wait_us(p) for p in prim])
        busy = np.mean([p["gpu_busy_us"] for p in prim])
        host = np.mean([p["buckets_us"]["host_total"] for p in prim])
        peak = [p["memory"].get("peak_allocated_gib") for p in prim if p["memory"].get("peak_allocated_gib") is not None]
        row = {"pp": pp, "ranks": [x["coords"]["rank"] for x in rs], "wall_us": float(wall),
               "work_us": float(busy - wait), "p2p_wait_us": float(wait), "host_stall_us": float(host),
               "non_wait_us": float(wall - wait), "peak_alloc_gib": max(peak) if peak else None}
        pp2 = [p.get("p2p") or {} for p in prim]
        for k in ("kernel_us", "transfer_est_us", "wait_est_us", "bytes_unknown_kernel_us"):
            row["p2p_" + k] = float(np.mean([x.get(k, 0.0) for x in pp2]))
        row["p2p_max_kernel_us"] = max([x.get("max_kernel_us", 0.0) for x in pp2] or [0.0])
        waits = {}
        for x, r in zip(pp2, rs):
            prk = r["coords"].get("pp_ranks") or []
            for b in x.get("by_peer") or []:
                st = prk.index(b["peer"]) if b["peer"] in prk else None
                k = (st, b["peer"] if st is None else None, b["role"])
                w = waits.setdefault(k, {"stage": st, "peer_rank": b["peer"], "role": b["role"], "kernel_us": 0.0,
                                         "wait_est_us": 0.0})
                w["kernel_us"] += b["kernel_us"] / len(rs)
                w["wait_est_us"] += b["wait_est_us"] / len(rs)
        row["waits_on"] = sorted(waits.values(), key=lambda w: -w["kernel_us"])
        rows.append(row)
    return rows


def _rng(vals, fmt="%.1f"):
    lo, hi = min(vals), max(vals)
    return (fmt % lo) if fmt % lo == fmt % hi else "%s-%s" % (fmt % lo, fmt % hi)


def waits_text(row, n=2):
    """'pp1 (recv) 40.3 s, pp0 (recv) 2.1 s' for the stages a stage's PP p2p kernels wait on."""
    parts = []
    for w in row.get("waits_on", [])[:n]:
        who = ("pp%d" % w["stage"]) if w["stage"] is not None else (
            "rank %s" % w["peer_rank"] if w["peer_rank"] is not None and w["peer_rank"] >= 0 else "unknown peer")
        parts.append("%s (%s) %.1f s" % (who, w["role"], w["kernel_us"] / 1e6))
    return ", ".join(parts) if parts else "no PP p2p kernels matched to a peer"


def pipeline_view(ranks):
    rows = stage_table(ranks)
    pp_sizes = [r["coords"].get("pp_size") for r in ranks if not r.get("error") and r["coords"].get("pp_size")]
    pp_size = max(pp_sizes) if pp_sizes else 1
    if pp_size <= 1 or not rows:
        return {"pp_size": pp_size, "stages": rows, "facts": []}
    crit = max(rows, key=lambda x: x["non_wait_us"])
    micro = [r["primary"].get("microbatches") for r in ranks if not r.get("error") and r["primary"].get("microbatches")]
    m = int(np.median(micro)) if micro else None
    facts = []
    others = [x for x in rows if x is not crit]

    def s(us):
        return "%.1f s" % (us / 1e6)
    facts.append("critical stage pp%d (rank %s): non-waiting time %s of %s wall; work %s, p2p wait %s, host stall %s"
                 % (crit["pp"], ",".join(str(x) for x in crit["ranks"]), s(crit["non_wait_us"]), s(crit["wall_us"]),
                    s(crit["work_us"]), s(crit["p2p_wait_us"]), s(crit["host_stall_us"])))
    if others:
        wk = [x["work_us"] for x in others]
        wt = [x["p2p_wait_us"] for x in others]
        pct = [x["p2p_wait_us"] / x["wall_us"] * 100 for x in others]
        facts.append("stages %s: work %.1f-%.1f s, p2p wait (waiting on neighbours) %.1f-%.1f s = %.0f-%.0f%% of wall"
                     % (",".join("pp%d" % x["pp"] for x in others), min(wk) / 1e6, max(wk) / 1e6,
                        min(wt) / 1e6, max(wt) / 1e6, min(pct), max(pct)))
    kt = [x["p2p_kernel_us"] for x in rows]
    if max(kt) > 0:
        te = [x["p2p_transfer_est_us"] for x in rows]
        unk = [x["p2p_bytes_unknown_kernel_us"] / x["p2p_kernel_us"] * 100 for x in rows if x["p2p_kernel_us"] > 0]
        facts.append("PP p2p kernel time includes waiting for the peer: per stage %s s of p2p kernels, of which est. "
                     "transfer (bytes / nominal link bw) %s s; longest single p2p kernel %.2f s; bytes unknown for "
                     "%s%% of the kernel time" % (_rng([x / 1e6 for x in kt]), _rng([x / 1e6 for x in te], "%.2f"),
                                                  max(x["p2p_max_kernel_us"] for x in rows) / 1e6,
                                                  _rng(unk or [0.0], "%.0f")))
        facts.append("waits on (PP p2p kernel time by the stage it receives from): " + " | ".join(
            "pp%d <- %s" % (x["pp"], waits_text(x, 1)) for x in rows))
    covered = sorted(x["pp"] for x in rows)
    missing = [p for p in range(pp_size) if p not in covered]
    bubble = None
    if m:
        bubble = (pp_size - 1.0) / (m + pp_size - 1.0)
        facts.append("ideal 1F1B bubble for PP=%d, m=%d microbatches: (p-1)/(m+p-1) = %.1f%% of step"
                     % (pp_size, m, bubble * 100))
    else:
        facts.append("microbatch count unknown (no forward_step frames) -> no ideal-bubble estimate")
    if missing:
        facts.append("stages not profiled: %s (critical stage may be among them)" % ",".join("pp%d" % p for p in missing))
    return {"pp_size": pp_size, "microbatches": m, "stages": rows, "critical_pp": crit["pp"],
            "ideal_1f1b_bubble_frac": bubble, "missing_stages": missing, "facts": facts}


METRICS = (
    ("busy_pct", "GPU busy", "%"),
    ("host_stall_pct", "host stall", "%"),
    ("pp_wait_pct", "PP p2p wait", "%"),
    ("comm_exposed_pct", "exposed comm", "%"),
    ("memcpy_exposed_pct", "exposed memcpy", "%"),
    ("compute_pct", "compute", "%"),
    ("unattributed_pct", "unattributed idle", "%"),
    ("fine_idle_pct", "fine-grained idle", "%"),
    ("wall_s", "step wall", "s"),
    ("peak_alloc_gib", "peak allocated", "GiB"),
)


def rank_metrics(r):
    p = r["primary"]
    w = p["window"]["wall_us"] or 1.0
    b = p["buckets_us"]
    return {
        "busy_pct": p["gpu_busy_us"] / w * 100, "host_stall_pct": b["host_total"] / w * 100,
        "pp_wait_pct": pp_wait_us(p) / w * 100, "comm_exposed_pct": b["comm_exposed"] / w * 100,
        "memcpy_exposed_pct": b["memcpy_exposed"] / w * 100, "compute_pct": b["compute"] / w * 100,
        "unattributed_pct": b["unattributed_idle"] / w * 100, "fine_idle_pct": b["fine_idle"] / w * 100,
        "wall_s": w / 1e6, "peak_alloc_gib": p["memory"].get("peak_allocated_gib"),
    }


def label(r):
    c = r["coords"]
    s = "rank %s" % c.get("rank")
    if c.get("pp_size", 1) and c.get("pp_size", 1) > 1 and c.get("pp") is not None:
        s += " (pp%d)" % c["pp"]
    return s


def contrasts(ranks, top=5):
    """Largest deviations of one rank from its peers, phrased as checkable facts."""
    ok = [r for r in ranks if not r.get("error")]
    if len(ok) < 2:
        return []
    mets = [rank_metrics(r) for r in ok]
    cands = []
    for key, name, unit in METRICS:
        vals = [m[key] for m in mets]
        if any(v is None for v in vals):
            continue
        for i, r in enumerate(ok):
            peers = [v for j, v in enumerate(vals) if j != i]
            med = float(np.median(peers))
            dev = vals[i] - med
            if unit == "%":
                score = abs(dev) / 100.0
            elif unit == "s":
                score = abs(dev) / max(med, 1e-9) * 0.5
            else:
                tot = ok[i]["primary"]["memory"].get("device_total_gib") or 80.0
                score = abs(dev) / tot
            if score < 0.02:
                continue
            fmt = (lambda v: "%.1f%%" % v) if unit == "%" else ((lambda v: "%.1f s" % v) if unit == "s"
                                                               else (lambda v: "%.1f GiB" % v))
            txt = "%s %s %s vs peers %s-%s" % (label(r), name, fmt(vals[i]), fmt(min(peers)), fmt(max(peers)))
            cands.append((score, key, i, txt))
    cands.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for score, key, i, txt in cands:
        # one line per (metric, direction) so the list is not five views of the same outlier metric
        if (key, i) in seen:
            continue
        seen.add((key, i))
        out.append({"score": score, "metric": key, "rank": ok[i]["coords"].get("rank"), "text": txt})
        if len(out) >= top:
            break
    return out
