# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse a PyTorch profiler (kineto) trace into compact numpy arrays, with an on-disk cache.

Stdlib json is streamed event by event (constant memory); orjson, if importable, loads the whole trace.
A single pass keeps only what the analysis needs: GPU activities (kernels, memcpy, memset) with NCCL
metadata, host frames >= 10 us, kernel-launch API calls, allocator API calls, record_param_comms ops,
and device memory samples.
"""
import codecs
import gzip
import hashlib
import json
import os
import re
from json import scanner as _json_scanner

import numpy as np

try:  # optional accelerator
    import orjson as _orjson
except Exception:  # pragma: no cover
    _orjson = None

from . import taxonomy as tx

PARSER_VERSION = 4
CACHE_TAG = ".pa2"       # cache files: <trace>.pa2.npz and <trace>.pa2.analysis.json (both validated on load)
STREAM_CHUNK = 16 << 20
HOST_MIN_DUR_US = 10.0
HOST_CATS = {"python_function": 0, "cpu_op": 1, "user_annotation": 2, "cuda_runtime": 3, "cuda_driver": 4}
HOST_CAT_NAMES = ("python_function", "cpu_op", "user_annotation", "cuda_runtime", "cuda_driver")
GPU_KINDS = {"kernel": 0, "gpu_memcpy": 1, "gpu_memset": 2}
TRACE_GLOBS = (".pt.trace.json.gz", ".trace.json.gz", ".pt.trace.json", ".trace.json")


class Table(object):
    """String interning table."""

    def __init__(self, items=None):
        self.items = list(items or [])
        self.index = {s: i for i, s in enumerate(self.items)}

    def id(self, s):
        i = self.index.get(s)
        if i is None:
            i = len(self.items)
            self.index[s] = i
            self.items.append(s)
        return i


def is_trace_file(name):
    return any(name.endswith(g) for g in TRACE_GLOBS)


def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def read_json(path):
    with _open(path) as f:
        raw = f.read()
    if _orjson is not None:
        return _orjson.loads(raw)
    return json.loads(raw)


_WS = re.compile(r"[ \t\n\r]*")


def iter_trace(path, top):
    """Yield traceEvents one at a time; other top-level keys are stored in `top`. Memory stays ~ one chunk."""
    scan = _json_scanner.make_scanner(json.JSONDecoder())
    inc = codecs.getincrementaldecoder("utf-8")(errors="replace")
    st = {"buf": "", "pos": 0, "eof": False}
    with _open(path) as f:
        def fill():
            chunk = f.read(STREAM_CHUNK)
            b = st["buf"][st["pos"]:]
            if not chunk:
                st["eof"] = True
                b += inc.decode(b"", final=True)
            else:
                b += inc.decode(chunk)
            st["buf"], st["pos"] = b, 0

        def ensure(n=1 << 20):
            while not st["eof"] and len(st["buf"]) - st["pos"] < n:
                fill()

        def skip():
            ensure()
            st["pos"] = _WS.match(st["buf"], st["pos"]).end()
            ensure()

        def char():
            skip()
            if st["pos"] >= len(st["buf"]):
                raise ValueError("truncated trace json")
            return st["buf"][st["pos"]]

        def value():
            while True:
                try:
                    obj, end = scan(st["buf"], st["pos"])
                    st["pos"] = end
                    return obj
                except (StopIteration, ValueError):
                    if st["eof"]:
                        raise ValueError("invalid trace json near offset %d" % st["pos"])
                    fill()

        if char() != "{":
            raise ValueError("not a PyTorch profiler trace: top level is not an object")
        st["pos"] += 1
        while True:
            c = char()
            if c == "}":
                break
            if c == ",":
                st["pos"] += 1
                continue
            key = value()
            if char() != ":":
                raise ValueError("invalid trace json near offset %d" % st["pos"])
            st["pos"] += 1
            if key == "traceEvents" and char() == "[":
                top["traceEvents"] = True
                st["pos"] += 1
                while True:
                    ensure()
                    buf = st["buf"]
                    pos = _WS.match(buf, st["pos"]).end()
                    c = buf[pos] if pos < len(buf) else ""
                    if c == ",":
                        st["pos"] = pos + 1
                        continue
                    if c == "]":
                        st["pos"] = pos + 1
                        break
                    if not c and st["eof"]:
                        raise ValueError("truncated trace json")
                    st["pos"] = pos
                    try:
                        obj, end = scan(buf, pos)
                        st["pos"] = end
                    except (StopIteration, ValueError):
                        obj = value()
                    yield obj
            else:
                char()
                top[key] = value()


def _num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def extract(data, path=""):
    """Turn a loaded trace dict into (meta, arrays)."""
    events = data.get("traceEvents") if isinstance(data, dict) else None
    if not isinstance(events, list):
        raise ValueError("not a PyTorch profiler trace: no traceEvents list")
    return extract_events(events, data, path)


def _top_meta(meta, data):
    dev_props = data.get("deviceProperties") or []
    d0 = dev_props[0] if dev_props else {}
    meta["device"] = {
        "name": d0.get("name"), "total_mem": d0.get("totalGlobalMem"), "num_sms": d0.get("numSms"),
        "n_visible": len(dev_props),
    }
    meta["device_props"] = [{"id": d.get("id"), "name": d.get("name"), "total_mem": d.get("totalGlobalMem")}
                            for d in dev_props]
    di = data.get("distributedInfo") or {}
    meta["dist"] = {k: di.get(k) for k in ("backend", "rank", "world_size", "pg_count", "nccl_version")}
    pgs = []
    for pg in di.get("pg_config") or []:
        try:
            pgs.append({"name": str(pg.get("pg_name")), "desc": pg.get("pg_desc"), "size": pg.get("pg_size"),
                        "ranks": pg.get("ranks") or [], "backend": pg.get("backend_config")})
        except Exception:
            continue
    meta["pg_config"] = pgs
    meta["flags"] = {k: data.get(k) for k in ("with_stack", "profile_memory", "record_shapes", "with_flops")}
    meta["base_time_ns"] = data.get("baseTimeNanoseconds")
    meta["cuda_runtime_version"] = data.get("cuda_runtime_version")


def extract_events(events, data, path=""):
    """Single pass over an event iterable; `data` holds the top-level keys (filled by the time events end)."""
    meta = {"parser_version": PARSER_VERSION, "path": path}

    kn = Table()   # gpu activity names
    hn = Table()   # host frame names
    colls, descs, pgnames, dtypes, rankstrs = Table(), Table(), Table(), Table(), Table()
    counts = {}
    steps, threads = [], {}

    g_ts, g_dur, g_stream, g_kind, g_name, g_dev, g_bytes, g_corr, g_ext = [], [], [], [], [], [], [], [], []
    c_idx, c_coll, c_desc, c_pg, c_in, c_out, c_dt, c_gs, c_rk = [], [], [], [], [], [], [], [], []
    h_ts, h_dur, h_tid, h_name, h_cat = [], [], [], [], []
    l_ts, l_dur, l_tid = [], [], []
    a_ts, a_dur, a_tid, a_name = [], [], [], []
    r_corr, r_ts, r_tid = [], [], []
    m_ts, m_alloc, m_resv, m_dev = [], [], [], []
    # record_param_comms host ops (message sizes and peers; coalesced p2p kernels carry neither)
    p_ts, p_dur, p_tid, p_coll, p_desc, p_pg, p_in, p_out, p_dt, p_gs, p_peer = [], [], [], [], [], [], [], [], [], [], []
    flops_by_ext = {}
    launch_names = {}
    hmin = HOST_MIN_DUR_US
    host_cats = HOST_CATS

    for e in events:
        if not isinstance(e, dict):
            continue
        cat = e.get("cat")
        counts[cat] = counts.get(cat, 0) + 1
        if cat == "python_function":
            d = e.get("dur", 0)
            if d is not None and d >= hmin:
                h_ts.append(e.get("ts", 0.0))
                h_dur.append(d)
                h_tid.append(e.get("tid", -1))
                h_name.append(hn.id(e.get("name", "?")))
                h_cat.append(0)
            continue
        if cat == "ac2g" or cat == "fwdbwd":
            continue
        ph = e.get("ph")
        if cat == "cuda_runtime" or cat == "cuda_driver":
            if ph != "X":
                continue
            name = e.get("name", "?")
            d = e.get("dur", 0) or 0
            ts = e.get("ts", 0.0)
            tid = e.get("tid", -1)
            args = e.get("args") or {}
            corr = args.get("correlation")
            if corr is not None:
                r_corr.append(corr)
                r_ts.append(ts)
                r_tid.append(tid)
            ln = launch_names.get(name)
            if ln is None:
                ln = (tx.is_launch_api(name), tx.is_alloc_api(name))
                launch_names[name] = ln
            if ln[0]:
                l_ts.append(ts)
                l_dur.append(d)
                l_tid.append(tid)
            if ln[1]:
                a_ts.append(ts)
                a_dur.append(d)
                a_tid.append(tid)
                a_name.append(hn.id(name))
            if d >= hmin:
                h_ts.append(ts)
                h_dur.append(d)
                h_tid.append(tid)
                h_name.append(hn.id(name))
                h_cat.append(host_cats[cat])
            continue
        if cat == "cpu_op" or cat == "user_annotation":
            if ph != "X":
                continue
            name = e.get("name", "?")
            d = e.get("dur", 0) or 0
            if cat == "cpu_op":
                args = e.get("args")
                if args and "flops" in args:
                    ext = args.get("External id")
                    if ext is not None:
                        flops_by_ext[ext] = _num(args.get("flops"))
                if name == "record_param_comms" and args:
                    cn = args.get("Collective name")
                    p_ts.append(e.get("ts", 0.0))
                    p_dur.append(d)
                    p_tid.append(e.get("tid", -1))
                    p_coll.append(colls.id(cn) if cn is not None else -1)
                    ds = args.get("Process Group Description")
                    p_desc.append(descs.id(ds) if ds is not None else -1)
                    pn = args.get("Process Group Name")
                    p_pg.append(pgnames.id(str(pn)) if pn is not None else -1)
                    p_in.append(_num(args.get("In msg nelems"), np.nan))
                    p_out.append(_num(args.get("Out msg nelems"), np.nan))
                    dt = args.get("dtype")
                    p_dt.append(dtypes.id(dt) if dt is not None else -1)
                    p_gs.append(int(_num(args.get("Group size"), 0)))
                    pr = args.get("Dst Rank", args.get("Src Rank"))
                    p_peer.append(int(_num(pr, -1)))
            else:
                if name.startswith("ProfilerStep#"):
                    steps.append({"name": name, "ts": e.get("ts", 0.0), "dur": d, "tid": e.get("tid")})
            if d >= hmin or cat == "user_annotation":
                h_ts.append(e.get("ts", 0.0))
                h_dur.append(d)
                h_tid.append(e.get("tid", -1))
                h_name.append(hn.id(name))
                h_cat.append(host_cats[cat])
            continue
        kind = GPU_KINDS.get(cat)
        if kind is not None:
            if ph != "X":
                continue
            args = e.get("args") or {}
            name = e.get("name", "?")
            idx = len(g_ts)
            g_ts.append(e.get("ts", 0.0))
            g_dur.append(max(e.get("dur", 0) or 0, 0))
            g_stream.append(args.get("stream", e.get("tid", -1)))
            g_kind.append(kind)
            g_name.append(kn.id(name))
            g_dev.append(args.get("device", 0))
            g_corr.append(args.get("correlation", -1))
            g_ext.append(args.get("External id", -1))
            if kind == 0:
                g_bytes.append(0.0)
                nl = name.lower()
                if "nccl" in nl or "deep_ep" in nl:
                    c_idx.append(idx)
                    cn = args.get("Collective name")
                    c_coll.append(colls.id(cn) if cn is not None else -1)
                    ds = args.get("Process Group Description")
                    c_desc.append(descs.id(ds) if ds is not None else -1)
                    pn = args.get("Process Group Name")
                    c_pg.append(pgnames.id(str(pn)) if pn is not None else -1)
                    c_in.append(_num(args.get("In msg nelems"), np.nan))
                    c_out.append(_num(args.get("Out msg nelems"), np.nan))
                    dt = args.get("dtype")
                    c_dt.append(dtypes.id(dt) if dt is not None else -1)
                    c_gs.append(int(_num(args.get("Group size"), 0)))
                    rk = args.get("Process Group Ranks")
                    c_rk.append(rankstrs.id(str(rk)) if rk is not None else -1)
            else:
                g_bytes.append(_num(args.get("bytes"), 0.0))
            continue
        if cat == "cpu_instant_event":
            if e.get("name") == "[memory]":
                args = e.get("args") or {}
                if args.get("Device Type") == 1 and (args.get("Device Id", -1) or 0) >= 0:
                    m_ts.append(e.get("ts", 0.0))
                    m_alloc.append(_num(args.get("Total Allocated")))
                    m_resv.append(_num(args.get("Total Reserved")))
                    m_dev.append(args.get("Device Id", 0))
            continue
        if ph == "M":
            if e.get("name") == "thread_name":
                a = e.get("args") or {}
                threads[str(e.get("pid")) + ":" + str(e.get("tid"))] = a.get("name")
            continue

    _top_meta(meta, data if isinstance(data, dict) else {})
    meta["counts"] = {str(k): v for k, v in counts.items()}
    meta["steps"] = sorted(steps, key=lambda s: s["ts"])
    meta["threads"] = threads
    meta["kernel_names"] = kn.items
    meta["host_names"] = hn.items
    meta["coll_names"] = colls.items
    meta["pg_descs"] = descs.items
    meta["pg_names"] = pgnames.items
    meta["dtypes"] = dtypes.items
    meta["rank_strs"] = rankstrs.items
    meta["flops_ops"] = len(flops_by_ext)

    f8, i8 = np.float64, np.int64
    arr = {
        "g_ts": np.array(g_ts, f8), "g_dur": np.array(g_dur, f8), "g_stream": np.array(g_stream, i8),
        "g_kind": np.array(g_kind, np.int8), "g_name": np.array(g_name, np.int32),
        "g_dev": np.array(g_dev, np.int16), "g_bytes": np.array(g_bytes, f8),
        "c_idx": np.array(c_idx, i8), "c_coll": np.array(c_coll, np.int32), "c_desc": np.array(c_desc, np.int32),
        "c_pg": np.array(c_pg, np.int32), "c_in": np.array(c_in, f8), "c_out": np.array(c_out, f8),
        "c_dt": np.array(c_dt, np.int32), "c_gs": np.array(c_gs, np.int32), "c_rk": np.array(c_rk, np.int32),
        "h_ts": np.array(h_ts, f8), "h_dur": np.array(h_dur, f8), "h_tid": np.array(h_tid, i8),
        "h_name": np.array(h_name, np.int32), "h_cat": np.array(h_cat, np.int8),
        "l_ts": np.array(l_ts, f8), "l_dur": np.array(l_dur, f8), "l_tid": np.array(l_tid, i8),
        "a_ts": np.array(a_ts, f8), "a_dur": np.array(a_dur, f8), "a_tid": np.array(a_tid, i8),
        "a_name": np.array(a_name, np.int32),
        "m_ts": np.array(m_ts, f8), "m_alloc": np.array(m_alloc, f8), "m_resv": np.array(m_resv, f8),
        "m_dev": np.array(m_dev, np.int16),
        "p_ts": np.array(p_ts, f8), "p_dur": np.array(p_dur, f8), "p_tid": np.array(p_tid, i8),
        "p_coll": np.array(p_coll, np.int32), "p_desc": np.array(p_desc, np.int32), "p_pg": np.array(p_pg, np.int32),
        "p_in": np.array(p_in, f8), "p_out": np.array(p_out, f8), "p_dt": np.array(p_dt, np.int32),
        "p_gs": np.array(p_gs, np.int32), "p_peer": np.array(p_peer, i8),
    }
    # Launch site (host ts/tid) of each GPU activity, via the CUPTI correlation id.
    g_corr = np.array(g_corr, i8)
    n = g_corr.size
    launch_ts = np.full(n, np.nan)
    launch_tid = np.full(n, -1, dtype=i8)
    if r_corr and n:
        rc = np.array(r_corr, i8)
        order = np.argsort(rc, kind="mergesort")
        rc = rc[order]
        rts = np.array(r_ts, f8)[order]
        rtid = np.array(r_tid, i8)[order]
        pos = np.searchsorted(rc, g_corr)
        pos = np.clip(pos, 0, rc.size - 1)
        hit = rc[pos] == g_corr
        launch_ts[hit] = rts[pos[hit]]
        launch_tid[hit] = rtid[pos[hit]]
    arr["g_launch_ts"] = launch_ts
    arr["g_launch_tid"] = launch_tid
    g_flops = np.full(n, np.nan)
    if flops_by_ext and n:
        ext = np.array(g_ext, i8)
        for i in np.flatnonzero(arr["g_kind"] == 0):
            v = flops_by_ext.get(int(ext[i]))
            if v is not None:
                g_flops[i] = v
    arr["g_flops"] = g_flops
    return meta, arr


# ---------------- cache ----------------
def _cache_candidates(path, cache_dir=None):
    base = os.path.basename(path)
    out = []
    if cache_dir:
        h = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]
        out.append(os.path.join(cache_dir, h + "." + base))
    else:
        out.append(path)
        h = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]
        out.append(os.path.join(os.path.expanduser("~/.cache/profile_analyzer"), h + "." + base))
    return out


def cache_paths(path, suffix, cache_dir=None):
    return [c + suffix for c in _cache_candidates(path, cache_dir)]


def _stat_key(path):
    st = os.stat(path)
    return [int(st.st_size), int(st.st_mtime)]


def save_npz(path, meta, arr, cache_dir=None):
    meta = dict(meta)
    meta["source_stat"] = _stat_key(path)
    for target in cache_paths(path, CACHE_TAG + ".npz", cache_dir):
        try:
            d = os.path.dirname(target) or "."
            if not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            tmp = target + ".tmp%d.npz" % os.getpid()
            np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arr)
            os.replace(tmp, target)
            return target
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
            continue
    return None


def load_npz(path, cache_dir=None):
    for target in cache_paths(path, CACHE_TAG + ".npz", cache_dir):
        if not os.path.exists(target):
            continue
        try:
            z = np.load(target, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            if meta.get("parser_version") != PARSER_VERSION or meta.get("source_stat") != _stat_key(path):
                continue
            arr = {k: z[k] for k in z.files if k != "meta"}
            return meta, arr
        except Exception:
            continue
    return None


def load_rank(path, use_cache=True, cache_dir=None, stream=False):
    """(meta, arrays, from_cache). Streams with stdlib json unless orjson is importable (and not `stream`)."""
    if use_cache:
        hit = load_npz(path, cache_dir)
        if hit is not None:
            hit[0]["path"] = path
            return hit[0], hit[1], True
    if _orjson is not None and not stream:
        data = read_json(path)
        meta, arr = extract(data, path)
        del data
        meta["reader"] = "orjson (whole file)"
    else:
        top = {}
        meta, arr = extract_events(iter_trace(path, top), top, path)
        if not top.get("traceEvents"):
            raise ValueError("not a PyTorch profiler trace: no traceEvents list")
        meta["reader"] = "json (streamed)"
    meta["path"] = path
    meta["source_stat"] = _stat_key(path)
    if use_cache:
        meta["cache_file"] = save_npz(path, meta, arr, cache_dir)
    return meta, arr, False


_RANK_RE = re.compile(r"rank[_-]?(\d+)", re.IGNORECASE)


def rank_from_filename(path):
    m = _RANK_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1))
    m = re.search(r"TP-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None
