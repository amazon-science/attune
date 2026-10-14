# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command line: overview (default), drill-down subcommands, --json, diff. Parses ranks in parallel, caches."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import time

import numpy as np

from . import decompose as dc
from . import memory as memmod
from . import memmgmt
from . import parse
from . import pipeline as pl
from . import report
from . import summary as sm

SUBCOMMANDS = ("overview", "ranks", "rank", "gaps", "stack", "comm", "kernels", "phases", "memory", "diff")
SCHEMA = "profile_analyzer/v2.2"
ANALYSIS_SUFFIX = parse.CACHE_TAG + ".analysis.json"
# peak RSS per worker ~ this many times the compressed trace size (measured: stdlib json streamed 0.4 GiB for a
# 150 MB trace; orjson loads the whole file, ~90x)
RSS_PER_GZ_BYTE_STREAM = 5
RSS_PER_GZ_BYTE_WHOLE = 90


def log(msg):
    try:
        sys.stderr.write("[analyze_profile] %s\n" % msg)
        sys.stderr.flush()
    except Exception:
        pass


# ---------------- per-rank work (runs in worker processes) ----------------
def _opts_key(opts):
    keep = {k: opts.get(k) for k in ("gpus_per_node", "nvlink_gbps", "net_gbps", "peak_tflops", "step")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


def _analysis_key(path, opts):
    return {"parser": parse.PARSER_VERSION, "analysis": dc.ANALYSIS_VERSION, "opts": _opts_key(opts),
            "source_stat": parse._stat_key(path)}


def load_cached_analysis(path, opts):
    if opts.get("no_cache"):
        return None
    key = _analysis_key(path, opts)
    for c in parse.cache_paths(path, ANALYSIS_SUFFIX, opts.get("cache_dir")):
        try:
            with open(c) as f:
                d = json.load(f)
            if d.get("_key") == key:
                d["cache"]["analysis_from_cache"] = True
                return d
        except Exception:
            continue
    return None


def save_analysis(path, res, opts):
    if opts.get("no_cache"):
        return
    res["_key"] = _analysis_key(path, opts)
    for c in parse.cache_paths(path, ANALYSIS_SUFFIX, opts.get("cache_dir")):
        try:
            d = os.path.dirname(c) or "."
            os.makedirs(d, exist_ok=True)
            tmp = c + ".tmp%d" % os.getpid()
            with open(tmp, "w") as f:
                json.dump(res, f, default=_jsonable)
            os.replace(tmp, c)
            return
        except Exception:
            continue


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def build_rank_result(r, opts):
    meta = r.meta
    wins = r.windows()
    res = {"path": meta.get("path"), "rank": r.rank, "coords": pl.coords(meta), "device": meta.get("device"),
           "flags": meta.get("flags"), "event_counts": meta.get("counts"), "flops_ops": meta.get("flops_ops", 0),
           "driver_threads": r.driver_tids, "other_device_events": r.other_device_events,
           "thread_names": {k.split(":")[1]: v for k, v in (meta.get("threads") or {}).items()},
           "base_time_ns": meta.get("base_time_ns"), "windows": [{k: w[k] for k in ("step", "t0", "t1", "source")}
                                                                 for w in wins]}
    if res["coords"].get("rank") is None:
        res["coords"]["rank"] = parse.rank_from_filename(meta.get("path") or "")
        res["rank"] = res["coords"]["rank"]
    inf = {}
    for x in r.comm["inferred"]:
        if x:
            inf[x] = inf.get(x, 0) + 1
    res["comm_meta"] = {"n": int(r.a["c_idx"].size), "no_pg": int(sum(1 for x in r.comm["inferred"]
                                                                       if x and x != "deepep-kernel")),
                        "inferred": inf}
    res["reader"] = meta.get("reader")
    if not wins:
        raise ValueError("no GPU activity and no ProfilerStep annotation")
    want = opts.get("step")
    prim = None
    if want is not None:
        for w in wins:
            if w["step"].endswith("#%s" % want):
                prim = w
    if prim is None:
        prim = wins[-1]
    res["primary"] = r.analyze_window(prim)
    res["per_step"] = []
    if len(wins) > 1:
        for w in wins:
            x = prim if w is prim else w
            rr = res["primary"] if w is prim else r.analyze_window(w)
            res["per_step"].append({"step": x["step"], "wall_us": rr["window"]["wall_us"],
                                    "buckets_us": rr["buckets_us"], "conservation": rr["conservation"],
                                    "gpu_busy_us": rr["gpu_busy_us"]})
    v1 = dc.v1_compat(r)
    p = res["primary"]
    v1.update({
        "rank_name": "rank%s" % res["rank"],
        "gpu_name": (meta.get("device") or {}).get("name"),
        "world_size": (meta.get("dist") or {}).get("world_size"),
        "host_stall_ms": round(p["buckets_us"]["host_total"] / 1e3, 2),
        "host_stall_pct_of_span": round(p["buckets_us"]["host_total"] / p["window"]["wall_us"] * 100, 2),
        "host_top_frames": [{"frame": v["top_frames"][0]["frame"][:90], "class": c, "ms": round(v["us"] / 1e3, 1),
                             "count": v["n_calls"]} for c, v in sorted(p["host_classes"].items(),
                                                                       key=lambda kv: -kv[1]["us"])[:6]
                            if v["top_frames"]],
        "host_time_by_class_ms": {c: round(v["us"] / 1e3, 1) for c, v in p["host_classes"].items()},
        "blocking_alloc_count": p["memory"]["blocking_alloc_ge5ms_n"],
        "blocking_alloc_ms": round(p["memory"]["blocking_alloc_ge5ms_ms"], 1),
    })
    res["v1"] = v1
    span_us = v1.get("gpu_active_span_ms", 0.0) * 1e3
    res["outside_window_us"] = max(span_us - sum(w["t1"] - w["t0"] for w in wins), 0.0)
    return res


def analyze_path(args):
    path, opts = args
    t = time.time()
    try:
        hit = load_cached_analysis(path, opts)
        if hit is not None:
            hit["cache"]["seconds"] = time.time() - t
            hit["path"] = path
            return hit
        meta, arr, from_cache = parse.load_rank(path, use_cache=not opts.get("no_cache"), cache_dir=opts.get("cache_dir"),
                                                stream=bool(opts.get("stream")))
        t_parse = time.time() - t
        r = dc.Rank(meta, arr, opts)
        res = build_rank_result(r, opts)
        res["path"] = path
        res["cache"] = {"parse_from_cache": from_cache, "analysis_from_cache": False, "parse_s": t_parse,
                        "analyze_s": time.time() - t - t_parse, "seconds": time.time() - t,
                        "cache_file": meta.get("cache_file")}
        save_analysis(path, res, opts)
        return res
    except MemoryError:
        return {"path": path, "rank": parse.rank_from_filename(path), "error": "MemoryError while parsing"}
    except Exception as e:  # never crash the whole report on one bad trace
        import traceback
        return {"path": path, "rank": parse.rank_from_filename(path), "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()[-2000:]}


def load_rank_obj(path, opts):
    meta, arr, _ = parse.load_rank(path, use_cache=not opts.get("no_cache"), cache_dir=opts.get("cache_dir"))
    return dc.Rank(meta, arr, opts)


# ---------------- run (a directory of traces) ----------------
def find_traces(d):
    if os.path.isfile(d):
        return [d]
    fs = sorted(os.path.join(d, f) for f in os.listdir(d) if parse.is_trace_file(f))
    if not fs:
        for root, _dirs, files in os.walk(d):
            for f in files:
                if parse.is_trace_file(f):
                    fs.append(os.path.join(root, f))
        fs.sort()
    return fs


def _mem_available():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _need_bytes(path, stream=True):
    try:
        gz = os.path.getsize(path)
    except OSError:
        return 0
    if not path.endswith(".gz"):
        gz = gz / 8.0
    k = RSS_PER_GZ_BYTE_WHOLE if (parse._orjson is not None and not stream) else RSS_PER_GZ_BYTE_STREAM
    return int(gz * k + (512 << 20))


def _workers(paths, opts):
    """Parallel parse workers: bounded by cores, by the trace count, and by available RAM."""
    if opts.get("jobs"):
        return max(1, int(opts["jobs"])), None
    n = min(len(paths), os.cpu_count() or 1)
    avail = _mem_available()
    note = None
    if avail and parse._orjson is not None and not opts.get("stream"):
        # orjson loads whole files (fast, ~90x the gz size); stream instead when that would cost parallelism
        if avail * 0.8 < n * max(_need_bytes(p, stream=False) for p in paths):
            opts["stream"] = True
    if avail:
        need = max(_need_bytes(p, stream=parse._orjson is None or bool(opts.get("stream"))) for p in paths)
        fit = int(avail * 0.8 // max(need, 1))
        if fit < n:
            note = "RAM: %.1f GiB available, ~%.1f GiB per trace -> %d worker(s) instead of %d" % (
                avail / 2 ** 30, need / 2 ** 30, max(fit, 1), n)
            if fit < 1:
                note += " (even one may not fit; expect swapping or an OOM kill)"
        n = max(1, min(n, fit))
    return n, note


def load_run(d, opts, only_rank=None):
    paths = find_traces(d)
    if not paths:
        raise SystemExit("no trace files (*.pt.trace.json[.gz]) found in %s" % d)
    if only_rank is not None:
        sel = [p for p in paths if parse.rank_from_filename(p) == only_rank]
        paths = sel or paths
    todo = [(p, opts) for p in paths]
    t = time.time()
    cached = [load_cached_analysis(p, opts) for p in paths]
    for p, c in zip(paths, cached):
        if c is not None:
            c["path"] = p
    need = [todo[i] for i, c in enumerate(cached) if c is None]
    results = {}
    for i, c in enumerate(cached):
        if c is not None:
            results[paths[i]] = c
    nw = 0
    rss_self0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if need:
        nw, note = _workers([p for p, _ in need], opts)
        nw = min(nw, len(need))
        log("analyzing %d trace(s) without an analysis cache (%d cached) with %d worker(s); a first parse takes "
            "~40-60 s per 150 MB trace" % (len(need), len(paths) - len(need), nw))
        if note:
            log(note)
        done = [0]

        def _progress(res):
            done[0] += 1
            c = res.get("cache") or {}
            log("%d/%d rank %s %s in %.1f s%s" % (
                done[0], len(need), res.get("rank"), "ERROR" if res.get("error") else (
                    "re-analyzed from parse cache" if c.get("parse_from_cache") else "parsed"),
                c.get("seconds", time.time() - t), (": " + res["error"]) if res.get("error") else ""))
        if nw > 1:
            ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
            with ctx.Pool(nw) as pool:
                for res in pool.imap_unordered(analyze_path, need):
                    results[res["path"]] = res
                    _progress(res)
        else:
            for x in need:
                res = analyze_path(x)
                results[res["path"]] = res
                _progress(res)
    ranks = [results[p] for p in paths]
    # dedupe identical global ranks (e.g. a copy of the same trace)
    seen, uniq, dups = set(), [], []
    for r in ranks:
        k = r.get("rank")
        if not r.get("error") and k is not None and k in seen:
            dups.append(r["path"])
            continue
        seen.add(k)
        uniq.append(r)
    uniq.sort(key=lambda r: (r.get("rank") is None, r.get("rank") if r.get("rank") is not None else 0))
    if only_rank is not None:
        uniq = [r for r in uniq if r.get("rank") == only_rank] or uniq
    run = {"dir": os.path.abspath(d), "ranks": uniq, "duplicates_skipped": dups, "load_s": time.time() - t}
    parsed = [r for r in ranks if (r.get("cache") or {}).get("analysis_from_cache") is False
              and not (r.get("cache") or {}).get("parse_from_cache")]
    rss = 0
    if nw > 1:
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    elif need:
        rss = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rss_self0)
    readers = sorted(set(r.get("reader") for r in parsed if r.get("reader")))
    run["timing"] = {"n_traces": len(paths), "analysis_cache_hits": len(paths) - len(need),
                     "parsed": len(parsed), "analyzed": len(need) - len(parsed), "workers": max(nw, 1),
                     "reader": ", ".join(readers) if readers else None,
                     "peak_worker_rss_gib": (rss * 1024.0 / 2 ** 30) if rss else None,
                     "no_cache": bool(opts.get("no_cache"))}
    run["pipeline"] = pl.pipeline_view(uniq)
    run["contrasts"] = pl.contrasts(uniq)
    run["memory_mgmt"] = memmgmt.stage_view(uniq)
    run["coverage"] = report.coverage(run)
    return run


def run_json(run, full=False, quick=False):
    """`summary` first (the key facts), then v1 ledger keys, then detail. Compact per-rank detail unless full."""
    rs = report.ok_ranks(run)
    prim = rs[0] if rs else None
    _, summ = sm.build(run)
    out = {"summary": summ, "schema": SCHEMA, "analysis_version": dc.ANALYSIS_VERSION, "profile_dir": run["dir"],
           "detail": "full" if full else ("summary only (--quick)" if quick else
                                          "compact (--json-full adds all gaps, kernels, per-rank v1 keys)")}
    if quick:
        return out
    if prim is not None:
        out.update(prim["v1"])       # v1 top-level keys (ledger compatibility) for the lowest profiled rank
    out["coverage"] = run["coverage"]
    out["pipeline"] = run["pipeline"]
    out["contrasts"] = run["contrasts"]
    out["memory_mgmt"] = run.get("memory_mgmt")
    out["comm_by_dim"] = report.comm_by_dim(run)
    out["ranks"] = [({k: v for k, v in r.items() if k != "_key"} if full else compact_rank(r)) for r in run["ranks"]]
    out["duplicates_skipped"] = run["duplicates_skipped"]
    return out


def compact_rank(r):
    if r.get("error"):
        return {k: r.get(k) for k in ("rank", "path", "error")}
    out = {k: r[k] for k in ("rank", "path", "coords", "device", "flags", "windows", "comm_meta", "cache")
           if k in r}
    p = r["primary"]
    pc = {k: p[k] for k in ("window", "buckets_us", "conservation", "gpu_busy_us", "gpu_idle_us", "compute", "comm",
                            "memcpy", "idle", "phases", "phase_method", "microbatches", "memory", "kernel_share",
                            "p2p", "optimizer", "comm_bw", "memory_mgmt") if k in p}
    pc["host_classes"] = {c: {"us": v["us"], "n_calls": v["n_calls"], "top_frames": v["top_frames"][:2]}
                          for c, v in p["host_classes"].items()}
    pc["gaps_top"] = p["gaps_top"][:5]
    k = dict(p["kernels"])
    k["top"] = (k.get("top") or [])[:10]
    pc["kernels"] = k
    out["primary"] = pc
    out["per_step"] = r.get("per_step")
    return out


def dumps(o):
    return json.dumps(o, default=_jsonable)


def dump_json(out):
    """Valid JSON laid out for `head`: one top-level key per line, summary lists one item per line."""
    parts = []
    for k, v in out.items():
        if k == "summary" and isinstance(v, dict):
            sub = []
            for sk, sv in v.items():
                if isinstance(sv, list) and sv:
                    sub.append("  %s: [\n%s\n  ]" % (json.dumps(sk), ",\n".join("   " + dumps(x) for x in sv)))
                else:
                    sub.append("  %s: %s" % (json.dumps(sk), dumps(sv)))
            parts.append(' "summary": {\n%s\n }' % ",\n".join(sub))
        elif k == "ranks" and isinstance(v, list) and v:
            parts.append(' "ranks": [\n%s\n ]' % ",\n".join("  " + dumps(x) for x in v))
        else:
            parts.append(" %s: %s" % (json.dumps(k), dumps(v)))
    return "{\n" + ",\n".join(parts) + "\n}"


# ---------------- argument parsing ----------------
def _common(p):
    p.add_argument("--profile-dir", help="directory of *.pt.trace.json.gz (one per rank), or a single trace")
    p.add_argument("--json", action="store_true", help="emit JSON: `summary` first, v1 ledger keys, compact per-rank "
                                                       "detail")
    p.add_argument("--json-full", action="store_true", help="like --json with full per-rank detail")
    p.add_argument("--quick", action="store_true", help="summary only (the key facts); instant when the analysis "
                                                        "cache exists")
    p.add_argument("--no-cache", action="store_true", help="ignore and do not write caches")
    p.add_argument("--cache-dir", default=None, help="write caches here instead of next to the traces")
    p.add_argument("--jobs", type=int, default=None, help="parallel parse workers (default: by cores and RAM)")
    p.add_argument("--gpus-per-node", type=int, default=None)
    p.add_argument("--link-bw", default=None, help="override nominal GB/s: nvlink=450,net=50")
    p.add_argument("--peak-tflops", type=float, default=None, help="override dense BF16 peak per GPU")
    p.add_argument("--step", default=None, help="ProfilerStep number to analyze (default: last complete)")
    p.add_argument("--top-n", type=int, default=20)


def make_opts(a):
    o = {"no_cache": a.no_cache, "cache_dir": a.cache_dir, "jobs": a.jobs,
         "gpus_per_node": a.gpus_per_node, "peak_tflops": a.peak_tflops, "step": a.step, "top_n": a.top_n}
    if a.link_bw:
        for kv in a.link_bw.split(","):
            k, _, v = kv.partition("=")
            if k.strip() in ("nvlink", "net"):
                o[k.strip() + "_gbps"] = float(v)
    return o


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sub = "overview"
    if argv and argv[0] in SUBCOMMANDS:
        sub = argv.pop(0)
    p = argparse.ArgumentParser(prog="analyze_profile.py [%s]" % "|".join(SUBCOMMANDS),
                                description="Profile analyzer: conserved per-rank step-time decomposition "
                                            "(measurements, not verdicts).")
    _common(p)
    if sub == "overview":
        p.add_argument("--rank", type=int, default=None, help="analyze only this global rank")
    elif sub in ("rank", "gaps", "phases", "memory", "stack"):
        p.add_argument("R", type=int, help="global rank")
        if sub == "gaps":
            p.add_argument("--top", type=int, default=20)
        if sub == "memory":
            p.add_argument("--bins", type=int, default=40)
        if sub == "stack":
            p.add_argument("--t0", type=float, required=True, help="ms from step start")
            p.add_argument("--t1", type=float, required=True, help="ms from step start")
    elif sub == "comm":
        p.add_argument("--dim", default=None)
    elif sub == "kernels":
        p.add_argument("--class", dest="cls", default=None, choices=list(dc.CATS))
        p.add_argument("--rank", type=int, default=None)
    elif sub == "diff":
        p.add_argument("RUN_A")
        p.add_argument("RUN_B")
    a = p.parse_args(argv)
    if a.json_full:
        a.json = True
    opts = make_opts(a)
    if sub == "diff":
        ra = load_run(a.RUN_A, opts)
        rb = load_run(a.RUN_B, opts)
        if a.json:
            print(json.dumps({"schema": SCHEMA, "A": run_json(ra, a.json_full), "B": run_json(rb, a.json_full)},
                             default=_jsonable))
        else:
            print(report.diff_text(ra, rb))
        return 0
    if not a.profile_dir:
        p.error("--profile-dir is required")
    only = getattr(a, "rank", None) if sub == "overview" else None
    if sub in ("rank", "gaps", "phases", "memory", "stack"):
        only = a.R
    t_start = time.time()
    run = load_run(a.profile_dir, opts, only_rank=only)
    if sub == "overview":
        run["timing"]["seconds"] = time.time() - t_start
        if a.json:
            print(dump_json(run_json(run, full=a.json_full, quick=a.quick)))
        else:
            print(report.overview(run, quick=a.quick))
        return 0
    if sub == "ranks":
        print(json.dumps([{k: r.get(k) for k in ("rank", "path", "coords", "error")} for r in run["ranks"]],
                         default=_jsonable) if a.json else report.ranks_table(run))
        return 0
    if sub == "comm":
        print(json.dumps(report.comm_by_dim(run), default=_jsonable) if a.json else report.comm_text(run, a.dim))
        return 0
    if sub == "kernels":
        print(report.kernels_text(run, a.cls, a.rank, a.top_n))
        return 0
    r = _pick(run, a.R)
    if sub == "rank":
        print(json.dumps(r, default=_jsonable) if a.json else report.rank_detail(r, a.top_n))
    elif sub == "phases":
        print(json.dumps(r["primary"]["phases"], default=_jsonable) if a.json else report.phases_text(r))
    elif sub == "gaps":
        if a.top > len(r["primary"]["gaps_top"]) and not r.get("error"):
            # recompute with a longer list from the parse cache
            dc.TOP_GAPS_KEPT = a.top
            ro = load_rank_obj(r["path"], opts)
            r = dict(r)
            r["primary"] = ro.analyze_window(_window_of(r))
        print(json.dumps(r["primary"]["gaps_top"][:a.top], default=_jsonable) if a.json else report.gaps_text(r, a.top))
    elif sub == "memory":
        ro = load_rank_obj(r["path"], opts)
        w = r["primary"]["window"]
        tl = memmod.timeline(ro, w["t0"], w["t1"], a.bins)
        print(json.dumps({"summary": r["primary"]["memory"], "timeline": tl},
                         default=_jsonable) if a.json else report.memory_text(r, tl))
    elif sub == "stack":
        ro = load_rank_obj(r["path"], opts)
        print(report.stack_text(r, ro, a.t0, a.t1))
    return 0


def _window_of(r):
    w = r["primary"]["window"]
    return {"step": w["step"], "t0": w["t0"], "t1": w["t1"], "source": w["source"]}


def _pick(run, R):
    for r in run["ranks"]:
        if r.get("rank") == R:
            return r
    raise SystemExit("rank %s not found; profiled ranks: %s" % (R, [r.get("rank") for r in run["ranks"]]))
