# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device memory from `[memory]` events and allocator churn."""
import numpy as np

from . import intervals as iv
from . import taxonomy as tx

GIB = 1024.0 ** 3
UNRESERVED_NOTE = ("device total - peak reserved by the caching allocator; NOT device free memory "
                   "(NCCL buffers, library workspaces and fragmentation sit outside the allocator)")
UNRESERVED_LABEL = "allocator unreserved GiB (device total - peak reserved; NOT device free memory)"


def _phase_at(t, psets, phase_order):
    for p in phase_order:
        s = psets.get(p)
        if s is not None and s[0].size and iv.contains_points(s, [t])[0]:
            return p
    return "other"


def _outermost(r, name_mask, t0, t1):
    """Indices of frames whose name matches, dropping ones nested in an earlier match on the same thread."""
    a = r.a
    if not name_mask.any():
        return np.zeros(0, dtype=np.int64)
    sel = np.flatnonzero(name_mask[a["h_name"]] & (r.h_start < t1) & (r.h_end > t0))
    if sel.size == 0:
        return sel
    keep = []
    for tid in np.unique(a["h_tid"][sel]):
        ss = sel[a["h_tid"][sel] == tid]
        ss = ss[np.lexsort((-a["h_dur"][ss], r.h_start[ss]))]
        run_end = -np.inf
        for i in ss.tolist():
            if r.h_start[i] >= run_end:
                keep.append(i)
                run_end = r.h_end[i]
            else:
                run_end = max(run_end, r.h_end[i])
    return np.array(sorted(keep), dtype=np.int64)


def summary(r, t0, t1, psets, phase_order):
    a = r.a
    out = {}
    total = (r.meta.get("device") or {}).get("total_mem") or 0
    out["device_total_gib"] = total / GIB if total else None
    ts = a["m_ts"]
    if ts.size:
        m = a["m_dev"] == r.device
        if not m.any():
            m = np.ones(ts.size, dtype=bool)
        ts, al, rs = ts[m], a["m_alloc"][m], a["m_resv"][m]
        w = (ts >= t0) & (ts <= t1)
        out["has_memory_events"] = True
        if w.any():
            ia = int(np.argmax(np.where(w, al, -1)))
            ir = int(np.argmax(np.where(w, rs, -1)))
            out["peak_allocated_gib"] = float(al[ia] / GIB)
            out["peak_allocated_t_ms"] = float((ts[ia] - t0) / 1e3)
            out["peak_allocated_phase"] = _phase_at(ts[ia], psets, phase_order)
            out["peak_reserved_gib"] = float(rs[ir] / GIB)
            out["peak_reserved_t_ms"] = float((ts[ir] - t0) / 1e3)
            out["peak_reserved_phase"] = _phase_at(ts[ir], psets, phase_order)
            if total:
                out["allocator_unreserved_gib"] = float((total - rs[ir]) / GIB)
                out["allocator_unreserved_note"] = UNRESERVED_NOTE
        out["trace_peak_allocated_gib"] = float(al.max() / GIB)
        out["trace_peak_reserved_gib"] = float(rs.max() / GIB)
    else:
        out["has_memory_events"] = False
    # allocator churn
    hn = r.hnames
    ec = np.array([tx.host_class(n) == "allocator_cache_release" for n in hn], dtype=bool) if hn else np.zeros(0, bool)
    ecf = _outermost(r, ec, t0, t1)
    out["empty_cache_calls"] = int(ecf.size)
    out["empty_cache_ms"] = float(a["h_dur"][ecf].sum() / 1e3) if ecf.size else 0.0
    al_m = np.array([tx.host_class(n) == "blocking_allocation" and not tx.is_alloc_api(n) for n in hn], dtype=bool) \
        if hn else np.zeros(0, bool)
    alf = _outermost(r, al_m, t0, t1)
    slow = alf[a["h_dur"][alf] >= 5000.0] if alf.size else alf
    out["blocking_alloc_ge5ms_n"] = int(slow.size)
    out["blocking_alloc_ge5ms_ms"] = float(a["h_dur"][slow].sum() / 1e3) if slow.size else 0.0
    api = {}
    am = (a["a_ts"] >= t0) & (a["a_ts"] <= t1)
    for nid, d in zip(a["a_name"][am], a["a_dur"][am]):
        nm = hn[int(nid)]
        x = api.setdefault(nm, [0, 0.0])
        x[0] += 1
        x[1] += float(d)
    out["allocator_api"] = {k: {"count": v[0], "ms": v[1] / 1e3} for k, v in sorted(api.items(), key=lambda kv: -kv[1][0])}
    out["malloc_free_recorded"] = any(("malloc" in k.lower() or "free" in k.lower()) for k in api)
    return out


def timeline(r, t0, t1, bins=40):
    a = r.a
    ts = a["m_ts"]
    if ts.size == 0:
        return []
    m = (a["m_dev"] == r.device) & (ts >= t0) & (ts <= t1)
    ts, al, rs = ts[m], a["m_alloc"][m], a["m_resv"][m]
    if ts.size == 0:
        return []
    edges = np.linspace(t0, t1, bins + 1)
    b = np.clip(np.searchsorted(edges, ts, side="right") - 1, 0, bins - 1)
    out = []
    last_a, last_r = None, None
    for i in range(bins):
        mm = b == i
        if mm.any():
            last_a, last_r = float(al[mm].max() / GIB), float(rs[mm].max() / GIB)
            out.append({"t_ms": (edges[i] - t0) / 1e3, "max_alloc_gib": last_a, "max_reserved_gib": last_r,
                        "samples": int(mm.sum())})
        else:
            out.append({"t_ms": (edges[i] - t0) / 1e3, "max_alloc_gib": last_a, "max_reserved_gib": last_r,
                        "samples": 0})
    return out
