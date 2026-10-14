# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vectorized interval-set algebra on sorted, disjoint (starts, ends) float64 arrays.

An interval set is a pair of equal-length numpy arrays with starts[i] < ends[i] <= starts[i+1].
Times are microseconds (trace units).
"""
import numpy as np

EMPTY = (np.zeros(0), np.zeros(0))


def as_arrays(starts, ends):
    s = np.asarray(starts, dtype=np.float64)
    e = np.asarray(ends, dtype=np.float64)
    return s, e


def union(starts, ends):
    """Union of arbitrary (possibly overlapping, unsorted) intervals -> disjoint sorted set."""
    s, e = as_arrays(starts, ends)
    keep = e > s
    s, e = s[keep], e[keep]
    if s.size == 0:
        return EMPTY
    order = np.argsort(s, kind="mergesort")
    s, e = s[order], e[order]
    run_end = np.maximum.accumulate(e)
    # a new run starts where the start is beyond every earlier end
    new = np.empty(s.size, dtype=bool)
    new[0] = True
    new[1:] = s[1:] > run_end[:-1]
    idx = np.flatnonzero(new)
    out_s = s[idx]
    last = np.append(idx[1:] - 1, s.size - 1)
    out_e = run_end[last]
    return out_s, out_e


def total(iv):
    s, e = iv
    return float(np.sum(e - s)) if s.size else 0.0


def clip(iv, t0, t1):
    s, e = iv
    if s.size == 0:
        return EMPTY
    s2 = np.maximum(s, t0)
    e2 = np.minimum(e, t1)
    keep = e2 > s2
    return s2[keep], e2[keep]


def complement(iv, t0, t1):
    """Gaps of a disjoint set inside [t0, t1]."""
    s, e = clip(iv, t0, t1)
    if s.size == 0:
        return (np.array([t0], dtype=np.float64), np.array([t1], dtype=np.float64)) if t1 > t0 else EMPTY
    gs = np.concatenate(([t0], e))
    ge = np.concatenate((s, [t1]))
    keep = ge > gs
    return gs[keep], ge[keep]


def intersect(a, b):
    """Intersection of two disjoint sorted sets."""
    as_, ae = a
    bs, be = b
    if as_.size == 0 or bs.size == 0:
        return EMPTY
    # candidate pairs: for each a-interval, the b-intervals that overlap it
    lo = np.searchsorted(be, as_, side="right")
    hi = np.searchsorted(bs, ae, side="left")
    counts = np.maximum(hi - lo, 0)
    if counts.sum() == 0:
        return EMPTY
    ai = np.repeat(np.arange(as_.size), counts)
    offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    bi = np.repeat(lo, counts) + offs
    s = np.maximum(as_[ai], bs[bi])
    e = np.minimum(ae[ai], be[bi])
    keep = e > s
    return s[keep], e[keep]


def subtract(a, b):
    """a minus b, both disjoint sorted sets."""
    if a[0].size == 0:
        return EMPTY
    if b[0].size == 0:
        return a
    t0 = min(a[0][0], b[0][0])
    t1 = max(a[1][-1], b[1][-1])
    return intersect(a, complement(b, t0, t1))


def overlap_total(a, b):
    return total(intersect(a, b))


def contains_points(iv, pts):
    """Boolean mask: which points lie inside the set (half-open [s, e))."""
    s, e = iv
    pts = np.asarray(pts, dtype=np.float64)
    if s.size == 0:
        return np.zeros(pts.shape, dtype=bool)
    idx = np.searchsorted(s, pts, side="right") - 1
    ok = idx >= 0
    res = np.zeros(pts.shape, dtype=bool)
    res[ok] = pts[ok] < e[idx[ok]]
    return res


def segment_active(iv, seg_lo, seg_hi):
    """For elementary segments [seg_lo, seg_hi) that never straddle a boundary of `iv`,
    return a boolean mask of segments inside the set."""
    s, e = iv
    if s.size == 0:
        return np.zeros(seg_lo.shape, dtype=bool)
    idx = np.searchsorted(s, seg_lo, side="right") - 1
    ok = idx >= 0
    res = np.zeros(seg_lo.shape, dtype=bool)
    res[ok] = e[idx[ok]] >= seg_hi[ok]
    return res


def elementary(boundaries, t0, t1):
    """Sorted unique boundaries clipped to [t0, t1] -> (lo, hi) elementary segments."""
    b = np.asarray(boundaries, dtype=np.float64)
    b = b[(b > t0) & (b < t1)]
    b = np.unique(np.concatenate(([t0], b, [t1])))
    return b[:-1], b[1:]


def runs(mask, lo, hi):
    """Merge consecutive True segments (adjacent lo/hi) into intervals."""
    if not mask.any():
        return EMPTY
    s = lo[mask]
    e = hi[mask]
    # consecutive segments are adjacent when e[i] == s[i+1]
    brk = np.empty(s.size, dtype=bool)
    brk[0] = True
    brk[1:] = s[1:] > e[:-1]
    idx = np.flatnonzero(brk)
    last = np.append(idx[1:] - 1, s.size - 1)
    return s[idx], e[last]
