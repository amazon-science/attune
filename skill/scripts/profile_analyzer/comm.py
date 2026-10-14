# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collective attribution: parallel dimension, collective type, bytes, bus bandwidth, intra/inter-node."""
import ast

import numpy as np

from . import taxonomy as tx


def _parse_ranks(s):
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
    except Exception:
        pass
    return None


def pg_ranks_map(meta):
    out = {}
    for pg in meta.get("pg_config") or []:
        out[str(pg.get("name"))] = (pg.get("desc"), list(pg.get("ranks") or []))
    return out


def gpus_per_node(meta, override=None):
    if override:
        return int(override)
    n = (meta.get("device") or {}).get("n_visible") or 0
    return n if n > 1 else 8


def comm_records(meta, arr, frames_at, opts=None):
    """Per comm-kernel attribution. `frames_at(tid, ts)` returns names of host frames enclosing a host point.

    Returns dict of numpy arrays indexed like arr['c_idx'].
    """
    opts = opts or {}
    n = arr["c_idx"].size
    knames = meta["kernel_names"]
    colls, descs, dtypes, rstrs, pgnames = (meta["coll_names"], meta["pg_descs"], meta["dtypes"],
                                            meta["rank_strs"], meta["pg_names"])
    pgmap = pg_ranks_map(meta)
    gpn = gpus_per_node(meta, opts.get("gpus_per_node"))
    pp_size = 1
    for pg in meta.get("pg_config") or []:
        if tx.dim_of_pg_desc(pg.get("desc")) == "PP" and (pg.get("desc") or "").upper().startswith("PIPELINE"):
            pp_size = max(pp_size, int(pg.get("size") or 1))
    dims, ctypes, desc_out, inferred = [], [], [], []
    nbytes = np.full(n, np.nan)
    gsize = np.zeros(n, dtype=np.int64)
    intra = np.full(n, -1, dtype=np.int8)
    rank_cache = {}
    for j in range(n):
        gi = int(arr["c_idx"][j])
        kname = knames[int(arr["g_name"][gi])]
        kl = kname.lower()
        ci = int(arr["c_coll"][j])
        di = int(arr["c_desc"][j])
        cname = colls[ci] if ci >= 0 else None
        desc = descs[di] if di >= 0 else None
        pgn = pgnames[int(arr["c_pg"][j])] if int(arr["c_pg"][j]) >= 0 else None
        ct = tx.coll_type(cname, kname)
        dim = tx.dim_of_pg_desc(desc)
        inf = ""
        if "deep_ep" in kl:
            dim, inf = "EP", "deepep-kernel"
        if dim is None and pgn is not None and pgn in pgmap:
            desc = pgmap[pgn][0]
            dim = tx.dim_of_pg_desc(desc)
        if dim is None:
            # No PG metadata on the kernel (e.g. batch_isend_irecv coalesced p2p): look at the launch site.
            lt = arr["g_launch_tid"][gi]
            lts = arr["g_launch_ts"][gi]
            names = frames_at(int(lt), float(lts)) if lt >= 0 and np.isfinite(lts) else []
            if any(any(k in nm for k in tx.P2P_FRAME_KEYS) for nm in names):
                dim, inf = "PP", "launch-stack"
            elif ct == "sendrecv" and pp_size > 1:
                dim, inf = "PP", "sendrecv-no-pg"
            else:
                dim, inf = "other_comm", "no-pg-metadata"
            if ct == "other" and dim == "PP":
                ct = "sendrecv"
        dims.append(dim)
        ctypes.append(ct)
        desc_out.append(desc or "")
        inferred.append(inf)
        gs = int(arr["c_gs"][j])
        rk = None
        ri = int(arr["c_rk"][j])
        if ri >= 0:
            rk = rank_cache.get(ri)
            if rk is None:
                rk = _parse_ranks(rstrs[ri]) or []
                rank_cache[ri] = rk
        if not rk and pgn is not None and pgn in pgmap:
            rk = pgmap[pgn][1]
        if gs <= 0 and rk:
            gs = len(rk)
        gsize[j] = gs
        if rk:
            intra[j] = 1 if len(set(int(r) // gpn for r in rk)) == 1 else 0
        dti = int(arr["c_dt"][j])
        b = tx.dtype_bytes(dtypes[dti]) if dti >= 0 else None
        if b:
            ine, oute = arr["c_in"][j], arr["c_out"][j]
            if ct == "allgather":
                v = oute
            elif ct in ("reduce_scatter", "alltoall", "allreduce", "broadcast", "reduce"):
                v = ine
            else:
                v = np.nanmax([ine, oute]) if np.isfinite(ine) or np.isfinite(oute) else np.nan
            nbytes[j] = v * b
    return {
        "dim": np.array(dims, dtype=object), "ctype": np.array(ctypes, dtype=object),
        "desc": np.array(desc_out, dtype=object), "inferred": np.array(inferred, dtype=object),
        "bytes": nbytes, "gsize": gsize, "intra": intra, "pp_size": pp_size, "gpn": gpn,
    }


def nominal_bw(gpu_name, is_intra, opts=None):
    opts = opts or {}
    if is_intra:
        v = opts.get("nvlink_gbps") or tx.lookup(tx.NVLINK_GBPS, gpu_name)
        return v, "NVLink"
    return opts.get("net_gbps") or tx.NET_GBPS_DEFAULT, "network"


def bandwidth_table(recs, gidx, dur_in_window, gpu_name, opts=None):
    """Aggregate per (dim, ctype): count, bytes, time, achieved bus bandwidth, intra/inter."""
    out = {}
    if gidx.size == 0:
        return out
    for j in range(gidx.size):
        d = dur_in_window[j]
        if d <= 0:
            continue
        key = (recs["dim"][j], recs["ctype"][j])
        o = out.get(key)
        if o is None:
            o = {"count": 0, "time_us": 0.0, "bytes": 0.0, "bw_time_us": 0.0, "bus_bytes": 0.0,
                 "gsizes": set(), "descs": set(), "intra": set(), "inferred": set(), "bytes_unknown": 0}
            out[key] = o
        o["count"] += 1
        o["time_us"] += d
        b = recs["bytes"][j]
        if np.isfinite(b) and b > 0:
            o["bytes"] += b
            o["bw_time_us"] += d
            o["bus_bytes"] += b * tx.busbw_factor(recs["ctype"][j], recs["gsize"][j])
        else:
            o["bytes_unknown"] += 1
        o["gsizes"].add(int(recs["gsize"][j]))
        if recs["desc"][j]:
            o["descs"].add(recs["desc"][j])
        o["intra"].add(int(recs["intra"][j]))
        if recs["inferred"][j]:
            o["inferred"].add(recs["inferred"][j])
    res = {}
    for (dim, ct), o in out.items():
        intra = o["intra"]
        is_intra = None if (-1 in intra and len(intra) == 1) else (intra == {1})
        bus = o["bus_bytes"] / (o["bw_time_us"] * 1e-6) / 1e9 if o["bw_time_us"] > 0 else None
        nom, link = nominal_bw(gpu_name, bool(is_intra), opts) if is_intra is not None else (None, None)
        res.setdefault(dim, {})[ct] = {
            "count": o["count"], "time_us": o["time_us"], "bytes": o["bytes"],
            "bus_bytes": o["bus_bytes"], "bw_time_us": o["bw_time_us"],
            "bytes_unknown_kernels": o["bytes_unknown"],
            "busbw_gbps": bus, "group_sizes": sorted(o["gsizes"]), "pg_descs": sorted(o["descs"]),
            "placement": ("intra-node" if is_intra else "inter-node") if is_intra is not None else "unknown",
            "nominal_gbps": nom, "nominal_link": link,
            "vs_nominal": (bus / nom) if (bus and nom) else None,
            "inferred": sorted(o["inferred"]),
        }
    return res


# ---------------- PP p2p: bytes and peers from record_param_comms ----------------
P2P_MATCH_WINDOW_US = 50000.0   # an isend/irecv op is coalesced into the next p2p kernel launched within 50 ms


def p2p_match(meta, arr, recs):
    """Per comm record: bytes sent/received and the main send/recv peer, for PP kernels.

    Coalesced p2p kernels (batch_isend_irecv) carry no sizes; the host-side record_param_comms send/recv ops
    do (with Dst/Src Rank). Each op is assigned to the first PP kernel launched from the same thread at or
    after the op's start.
    """
    n = arr["c_idx"].size
    out = {"send_bytes": np.zeros(n), "recv_bytes": np.zeros(n), "send_peer": np.full(n, -1, dtype=np.int64),
           "recv_peer": np.full(n, -1, dtype=np.int64), "n_ops": np.zeros(n, dtype=np.int64)}
    pts = arr.get("p_ts")
    if n == 0 or pts is None or pts.size == 0:
        return out
    colls = meta["coll_names"]
    dtypes = meta["dtypes"]
    kind = np.array([0 if c < 0 else (1 if colls[c] == "send" else (2 if colls[c] == "recv" else 0))
                     for c in arr["p_coll"].tolist()], dtype=np.int8)
    ops = np.flatnonzero(kind > 0)
    pp = np.flatnonzero(recs["dim"] == "PP")
    if ops.size == 0 or pp.size == 0:
        return out
    gi = arr["c_idx"][pp]
    lt = arr["g_launch_tid"][gi]
    lts = arr["g_launch_ts"][gi]
    peers = {}
    for tid in np.unique(arr["p_tid"][ops]):
        o = ops[arr["p_tid"][ops] == tid]
        o = o[np.argsort(arr["p_ts"][o], kind="mergesort")]
        k = pp[(lt == tid) & np.isfinite(lts)]
        if k.size == 0:
            continue
        kl = arr["g_launch_ts"][arr["c_idx"][k]]
        order = np.argsort(kl, kind="mergesort")
        k, kl = k[order], kl[order]
        pos = np.searchsorted(kl, arr["p_ts"][o], side="left")
        ok = pos < k.size
        o, pos = o[ok], pos[ok]
        ok = (kl[pos] - arr["p_ts"][o]) <= P2P_MATCH_WINDOW_US
        for i, j in zip(o[ok].tolist(), k[pos[ok]].tolist()):
            dti = int(arr["p_dt"][i])
            b = tx.dtype_bytes(dtypes[dti]) if dti >= 0 else None
            ne = np.nanmax([arr["p_in"][i], arr["p_out"][i]]) if (np.isfinite(arr["p_in"][i]) or
                                                                   np.isfinite(arr["p_out"][i])) else np.nan
            nb = float(ne * b) if (b and np.isfinite(ne)) else 0.0
            out["n_ops"][j] += 1
            side = "send" if kind[i] == 1 else "recv"
            out[side + "_bytes"][j] += nb
            pr = int(arr["p_peer"][i])
            if pr >= 0:
                d = peers.setdefault((j, side), {})
                d[pr] = d.get(pr, 0.0) + nb + 1.0
    for (j, side), d in peers.items():
        out[side + "_peer"][j] = max(d.items(), key=lambda kv: kv[1])[0]
    return out


def p2p_window(recs, p2p, dur, my_rank, gpu_name, opts=None):
    """PP p2p kernel time in a window, split into an estimated transfer part (bytes / nominal link bandwidth)
    and the rest (waiting for the peer), and attributed to the peer each kernel receives from."""
    pp = np.flatnonzero((recs["dim"] == "PP") & (dur > 0))
    out = {"kernels": int(pp.size), "kernel_us": float(dur[pp].sum()) if pp.size else 0.0,
           "note": "PP p2p kernel duration includes time spent waiting for the peer to post the matching "
                   "send/recv; transfer = bytes / nominal link bandwidth (estimate), wait = kernel time - transfer"}
    if pp.size == 0:
        return out
    gpn = recs.get("gpn") or 8
    known_us = unknown_us = transfer = wait = nbytes = 0.0
    by_peer = {}
    for j in pp.tolist():
        dj = float(dur[j])
        if p2p["n_ops"][j] == 0:
            unknown_us += dj
            continue
        est = 0.0
        for side in ("send", "recv"):
            b = float(p2p[side + "_bytes"][j])
            if b <= 0:
                continue
            peer = int(p2p[side + "_peer"][j])
            intra = (my_rank is not None and peer >= 0 and int(peer) // gpn == int(my_rank) // gpn)
            bw, _ = nominal_bw(gpu_name, intra, opts)
            est = max(est, b / (bw * 1e3) if bw else 0.0)
            nbytes += b
        est = min(est, dj)
        known_us += dj
        transfer += est
        wait += dj - est
        rp, sp = int(p2p["recv_peer"][j]), int(p2p["send_peer"][j])
        key = (rp, "recv") if rp >= 0 else ((sp, "send-only") if sp >= 0 else (-1, "unknown"))
        x = by_peer.setdefault(key, {"peer": key[0], "role": key[1], "n": 0, "kernel_us": 0.0, "wait_est_us": 0.0})
        x["n"] += 1
        x["kernel_us"] += dj
        x["wait_est_us"] += dj - est
    out.update({"bytes_known_kernel_us": known_us, "bytes_unknown_kernel_us": unknown_us, "bytes": nbytes,
                "transfer_est_us": transfer, "wait_est_us": wait,
                "max_kernel_us": float(dur[pp].max()), "median_kernel_us": float(np.median(dur[pp])),
                "by_peer": sorted(by_peer.values(), key=lambda x: -x["kernel_us"])})
    return out


def optimizer_shard_estimate(meta, arr, t0, t1):
    """Parameter elements per rank from the distributed optimizer's DP gradient reduce-scatter sizes.

    Sum of reduce-scatter input elements over DP groups = parameters held by this rank's model chunk;
    divided by each group's size = elements in this rank's optimizer shard."""
    pts = arr.get("p_ts")
    if pts is None or pts.size == 0:
        return None
    colls, descs = meta["coll_names"], meta["pg_descs"]
    pgsize = {}
    for pg in meta.get("pg_config") or []:
        pgsize[str(pg.get("name"))] = int(pg.get("size") or len(pg.get("ranks") or []) or 0)
    pgn = meta["pg_names"]
    groups = {}
    m = np.flatnonzero((pts >= t0) & (pts < t1))
    for i in m.tolist():
        c = int(arr["p_coll"][i])
        if c < 0 or "reduce_scatter" not in colls[c].lower():
            continue
        di = int(arr["p_desc"][i])
        desc = descs[di] if di >= 0 else None
        if tx.dim_of_pg_desc(desc) != "DP":
            continue
        ne = arr["p_in"][i]
        if not np.isfinite(ne) or ne <= 0:
            continue
        gs = int(arr["p_gs"][i])
        if gs <= 0:
            pi = int(arr["p_pg"][i])
            gs = pgsize.get(pgn[pi], 0) if pi >= 0 else 0
        g = groups.setdefault(desc, {"calls": 0, "in_elems": 0.0, "group_size": max(gs, 1)})
        g["calls"] += 1
        g["in_elems"] += float(ne)
    if not groups:
        return None
    for g in groups.values():
        g["shard_elems"] = g["in_elems"] / g["group_size"]
    return {"by_group": groups, "params_on_rank": sum(g["in_elems"] for g in groups.values()),
            "params_in_shard": sum(g["shard_elems"] for g in groups.values()),
            "method": "sum of DP gradient reduce-scatter input elements (record_param_comms) in the step"}
