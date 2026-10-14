# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-rank step-time decomposition with a conservation invariant.

wall = compute + comm_exposed + memcpy_exposed + sum(host idle classes) + unattributed_idle + fine_idle
"""
import numpy as np

from . import comm as commmod
from . import intervals as iv
from . import memory as memmod
from . import memmgmt
from . import taxonomy as tx

ANALYSIS_VERSION = 10
MIN_GAP_US = 1000.0          # idle intervals shorter than this are "fine-grained idle"
SMALL_KERNEL_US = 10.0
SLOW_ALLOC_US = 5000.0
TOP_GAPS_KEPT = 100
CONSERVATION_TOL = 1e-3      # 0.1 %
CONTEXT_SLACK_US = 1000.0    # a frame within min(1 ms, 0.1% of wall) of both window edges is step-level context

CATS = tx.COMPUTE_CLASSES + tx.COMM_DIMS + tx.MEMCPY_DIRS
CAT_IDX = {c: i for i, c in enumerate(CATS)}
N_COMPUTE = len(tx.COMPUTE_CLASSES)
N_COMM = len(tx.COMM_DIMS)
COMPUTE_IDS = list(range(N_COMPUTE))
COMM_IDS = list(range(N_COMPUTE, N_COMPUTE + N_COMM))
MEMCPY_IDS = list(range(N_COMPUTE + N_COMM, len(CATS)))

# host-frame names that indicate an optimizer whose update runs on the CPU (matched lowercased, anywhere)
OFFLOAD_KEYS = ("cpu_offloading", "hybriddeviceoptimizer", "hybrid_optimizer", "cpu_adam", "cpuadam",
                "offload_optimizer", "optimizer_offload")
OPT_UPDATE_KEYS = ("adam", "sgd", "lamb", "adagrad", "rmsprop")
ADAM_STATE_BYTES_PER_PARAM = 12   # fp32 master weight + exp_avg + exp_avg_sq


class Rank(object):
    """Arrays of one rank prepared for window analysis."""

    def __init__(self, meta, arr, opts=None):
        self.meta = meta
        self.a = arr
        self.opts = opts or {}
        a = arr
        self.rank = meta.get("dist", {}).get("rank")
        self.hnames = meta["host_names"]
        self.knames = meta["kernel_names"]
        # main device: the one that runs most kernels
        dev = a["g_dev"]
        if dev.size:
            vals, cnt = np.unique(dev, return_counts=True)
            self.device = int(vals[np.argmax(cnt)])
            self.other_device_events = int(dev.size - cnt.max())
        else:
            self.device, self.other_device_events = 0, 0
        on = dev == self.device
        self.g_sel = np.flatnonzero(on)
        self.g_start = a["g_ts"]
        self.g_end = a["g_ts"] + a["g_dur"]
        # host frame arrays
        self.h_start = a["h_ts"]
        self.h_end = a["h_ts"] + a["h_dur"]
        self.h_class_by_name = np.array([tx.host_priority(tx.host_class(n)) for n in self.hnames], dtype=np.int16) \
            if self.hnames else np.zeros(0, dtype=np.int16)
        self.h_phase_by_name = [tx.phase_kind(n, None) for n in self.hnames]
        self.step_name_mask = np.array([n.startswith("ProfilerStep#") for n in self.hnames], dtype=bool) \
            if self.hnames else np.zeros(0, dtype=bool)
        self.driver_tids = self._driver_threads()
        self._thread_index = {}
        self._classify_gpu()
        self.p2p = commmod.p2p_match(meta, arr, self.comm)
        low = [n.lower() for n in self.hnames]
        self.offload_name_mask = np.array([any(k in x for k in OFFLOAD_KEYS) for x in low], dtype=bool) \
            if low else np.zeros(0, bool)
        self.opt_update_mask = np.array([x.startswith("aten::") and any(k in x for k in OPT_UPDATE_KEYS)
                                         for x in low], dtype=bool) if low else np.zeros(0, bool)

    # ---------- threads / frames ----------
    def _driver_threads(self):
        lt = self.a["l_tid"]
        if lt.size == 0:
            ht = self.a["h_tid"]
            if ht.size == 0:
                return []
            vals, cnt = np.unique(ht, return_counts=True)
            return [int(vals[np.argmax(cnt)])]
        vals, cnt = np.unique(lt, return_counts=True)
        keep = cnt >= max(1, int(0.001 * lt.size))
        return [int(v) for v in vals[keep]]

    def thread_frames(self, tid):
        """Frame indices on a thread sorted by (start, -dur)."""
        idx = self._thread_index.get(tid)
        if idx is None:
            sel = np.flatnonzero(self.a["h_tid"] == tid)
            order = np.lexsort((-self.a["h_dur"][sel], self.h_start[sel]))
            idx = sel[order]
            self._thread_index[tid] = idx
        return idx

    def frames_at(self, tid, t):
        idx = self.thread_frames(tid)
        if idx.size == 0:
            return []
        k = np.searchsorted(self.h_start[idx], t, side="right")
        cand = idx[:k]
        m = self.h_end[cand] >= t
        return [self.hnames[int(self.a["h_name"][i])] for i in cand[m]]

    def frame_union(self, pred_name_mask, tids=None):
        """Per-thread unions of frames whose name satisfies a boolean mask over name ids."""
        out = {}
        if not pred_name_mask.any():
            return out
        sel = np.flatnonzero(pred_name_mask[self.a["h_name"]])
        if sel.size == 0:
            return out
        tid = self.a["h_tid"][sel]
        for t in np.unique(tid):
            ss = sel[tid == t]
            out[int(t)] = iv.union(self.h_start[ss], self.h_end[ss])
        return out

    def launched_in(self, unions, gidx):
        """Mask over gidx: GPU activity launched from inside a frame of the per-thread unions."""
        res = np.zeros(gidx.size, dtype=bool)
        if not unions or gidx.size == 0:
            return res
        lt = self.a["g_launch_tid"][gidx]
        lts = self.a["g_launch_ts"][gidx]
        for t, u in unions.items():
            m = lt == t
            if m.any():
                res[m] = iv.contains_points(u, lts[m])
        return res

    # ---------- GPU categories ----------
    def _classify_gpu(self):
        a = self.a
        n = a["g_ts"].size
        cat = np.full(n, -1, dtype=np.int16)
        kind = a["g_kind"]
        names = a["g_name"]
        # per-name compute class
        kc = []
        comm_name = []
        for nm in self.knames:
            nl = nm.lower()
            comm_name.append(tx.is_comm_kernel(nl))
            kc.append(CAT_IDX[tx.compute_class(nl)])
        kc = np.array(kc, dtype=np.int16) if kc else np.zeros(0, dtype=np.int16)
        comm_name = np.array(comm_name, dtype=bool) if comm_name else np.zeros(0, dtype=bool)
        is_k = kind == 0
        if n:
            cat[is_k] = kc[names[is_k]]
        # memcpy / memset
        for i in np.flatnonzero(~is_k):
            cat[i] = CAT_IDX[tx.memcpy_dir(self.knames[int(names[i])], kind[i] == 2)]
        # launch-context refinements
        hn = self.hnames
        grouped = np.array([("grouped" in x.lower()) for x in hn], dtype=bool) if hn else np.zeros(0, bool)
        optm = np.array([(self.h_phase_by_name[i] == "optimizer") or
                         (tx.host_class(hn[i]) == "optimizer_cpu_step") for i in range(len(hn))], dtype=bool) \
            if hn else np.zeros(0, bool)
        self.grouped_unions = self.frame_union(grouped)
        self.optimizer_unions = self.frame_union(optm)
        gemm_i = np.flatnonzero(cat == CAT_IDX["gemm"])
        if gemm_i.size:
            m = self.launched_in(self.grouped_unions, gemm_i)
            cat[gemm_i[m]] = CAT_IDX["grouped_gemm"]
        oth = np.flatnonzero(np.isin(cat, [CAT_IDX["elementwise_norm"], CAT_IDX["other_compute"]]))
        if oth.size:
            m = self.launched_in(self.optimizer_unions, oth)
            cat[oth[m]] = CAT_IDX["optimizer"]
        # communication kernels
        cidx = a["c_idx"]
        self.comm = commmod.comm_records(self.meta, a, self.frames_at, self.opts)
        if cidx.size:
            dims = np.array([CAT_IDX.get(d, CAT_IDX["other_comm"]) for d in self.comm["dim"]], dtype=np.int16)
            cat[cidx] = dims
        # any remaining comm-named kernels without a record (should not happen)
        if n:
            stray = is_k & comm_name[names] & ~np.isin(np.arange(n), cidx)
            cat[stray] = CAT_IDX["other_comm"]
        self.g_cat = cat
        self.comm_pos = {int(g): j for j, g in enumerate(cidx)}

    # ---------- windows ----------
    def windows(self):
        steps = [s for s in self.meta.get("steps") or [] if (s.get("dur") or 0) > 0]
        # keep the host-thread annotation, one per step name
        seen, out = set(), []
        for s in steps:
            if s["name"] in seen:
                continue
            seen.add(s["name"])
            out.append({"step": s["name"], "t0": float(s["ts"]), "t1": float(s["ts"] + s["dur"]),
                        "source": "ProfilerStep annotation"})
        if not out:
            sel = self.g_sel
            if sel.size:
                out.append({"step": "(no ProfilerStep)", "t0": float(self.g_start[sel].min()),
                            "t1": float(self.g_end[sel].max()), "source": "GPU active span (no step annotation)"})
        return out

    # ---------- phases ----------
    def phase_sets(self, t0, t1):
        """Interval sets for forward/backward/recompute/optimizer inside [t0, t1], and the method used."""
        a = self.a
        hn = a["h_name"]
        kinds = self.h_phase_by_name
        sel = np.flatnonzero((self.h_start < t1) & (self.h_end > t0))
        by = {"forward": [], "backward": [], "recompute_frame": [], "optimizer": []}
        for i in sel:
            k = kinds[int(hn[i])]
            if k is not None:
                by[k].append(i)
        method = []
        sets = {}
        for k in ("forward", "backward", "optimizer"):
            ii = np.array(by[k], dtype=np.int64)
            sets[k] = iv.clip(iv.union(self.h_start[ii], self.h_end[ii]), t0, t1) if ii.size else iv.EMPTY
        if by["forward"] or by["backward"]:
            method.append("python frames (schedules.forward_step/backward_step)")
        # backward fallback without stacks: autograd engine evaluate_function frames
        if not by["backward"]:
            evn = np.array([n.startswith("autograd::engine::evaluate_function") for n in self.hnames], dtype=bool) \
                if self.hnames else np.zeros(0, bool)
            ev_sel = sel[evn[hn[sel]]] if sel.size else sel
            if ev_sel.size:
                # bridge sub-ms gaps between consecutive autograd nodes
                u = iv.union(self.h_start[ev_sel] - 500.0, self.h_end[ev_sel] + 500.0)
                sets["backward"] = iv.clip((u[0] + 500.0, u[1] - 500.0), t0, t1)
                method.append("autograd evaluate_function spans (no backward_step frames)")
        # recompute: head of each checkpoint-backward frame, up to its first nested autograd node
        rc = np.array(by["recompute_frame"], dtype=np.int64)
        if rc.size:
            evn = np.array([n.startswith("autograd::engine::evaluate_function") for n in self.hnames], dtype=bool)
            ends = []
            for tid in np.unique(a["h_tid"][rc]):
                ti = self.thread_frames(int(tid))
                evs = ti[evn[hn[ti]]]
                es = np.sort(self.h_start[evs])
                r = rc[a["h_tid"][rc] == tid]
                k = np.searchsorted(es, self.h_start[r], side="right")
                nxt = np.where(k < es.size, es[np.minimum(k, max(es.size - 1, 0))] if es.size else np.inf, np.inf)
                ends.append((self.h_start[r], np.minimum(self.h_end[r], nxt)))
            s = np.concatenate([x[0] for x in ends])
            e = np.concatenate([x[1] for x in ends])
            sets["recompute"] = iv.clip(iv.union(s, e), t0, t1)
            method.append("checkpoint-backward head (re-forward before first nested autograd node)")
        else:
            sets["recompute"] = iv.EMPTY
        if by["optimizer"]:
            method.append("Optimizer.step annotation / optimizer step frames")
        n_micro = None
        if by["forward"]:
            n_micro = len(by["forward"])
        return sets, method, n_micro

    # ---------- the decomposition ----------
    def analyze_window(self, w):
        t0, t1 = w["t0"], w["t1"]
        wall = t1 - t0
        a = self.a
        sel = self.g_sel
        sel = sel[(self.g_start[sel] < t1) & (self.g_end[sel] > t0) & (self.g_cat[sel] >= 0)]
        gs = np.maximum(self.g_start[sel], t0)
        ge = np.minimum(self.g_end[sel], t1)
        gcat = self.g_cat[sel]
        unions = []
        for c in range(len(CATS)):
            m = gcat == c
            unions.append(iv.union(gs[m], ge[m]) if m.any() else iv.EMPTY)
        psets, pmethod, n_micro = self.phase_sets(t0, t1)
        phase_order = ("recompute", "optimizer", "backward", "forward")
        bnds = [np.concatenate([u[0], u[1]]) for u in unions]
        bnds += [np.concatenate([psets[p][0], psets[p][1]]) for p in phase_order]
        lo, hi = iv.elementary(np.concatenate(bnds) if bnds else np.zeros(0), t0, t1)
        seglen = hi - lo
        act = np.zeros((len(CATS), lo.size), dtype=bool)
        for c, u in enumerate(unions):
            if u[0].size:
                act[c] = iv.segment_active(u, lo, hi)
        busy = act.any(axis=0)
        owner = np.where(busy, np.argmax(act, axis=0), -1)
        comp_any = act[:N_COMPUTE].any(axis=0)
        comm_any = act[N_COMPUTE:N_COMPUTE + N_COMM].any(axis=0)
        # phase of each segment
        seg_phase = np.full(lo.size, len(tx.PHASES) - 1, dtype=np.int8)   # "other"
        for p in reversed(phase_order):   # lowest precedence first, overwritten by higher
            if psets[p][0].size:
                m = iv.segment_active(psets[p], lo, hi)
                seg_phase[m] = tx.PHASES.index(p)

        res = {"window": {"step": w["step"], "source": w["source"], "t0": t0, "t1": t1, "wall_us": wall}}
        excl = np.bincount(owner[busy], weights=seglen[busy], minlength=len(CATS)) if busy.any() \
            else np.zeros(len(CATS))
        sums = np.bincount(gcat, weights=ge - gs, minlength=len(CATS)) if gcat.size else np.zeros(len(CATS))
        cnts = np.bincount(gcat, minlength=len(CATS)) if gcat.size else np.zeros(len(CATS), dtype=int)
        compute = {}
        for c in COMPUTE_IDS:
            compute[CATS[c]] = {"exclusive_us": float(excl[c]), "union_us": iv.total(unions[c]),
                                "sum_us": float(sums[c]), "count": int(cnts[c])}
        comm = {}
        for c in COMM_IDS:
            own = act[c]
            exposed = float(seglen[own & ~comp_any].sum())
            comm[CATS[c]] = {"exclusive_us": float(excl[c]), "union_us": iv.total(unions[c]),
                             "exposed_us": exposed, "overlapped_us": iv.total(unions[c]) - exposed,
                             "sum_us": float(sums[c]), "count": int(cnts[c])}
        memcpy = {}
        for c in MEMCPY_IDS:
            own = act[c]
            m = gcat == c
            memcpy[CATS[c]] = {"exclusive_us": float(excl[c]), "union_us": iv.total(unions[c]),
                               "exposed_us": float(seglen[own & ~comp_any & ~comm_any].sum()),
                               "overlap_compute_us": float(seglen[own & comp_any].sum()),
                               "sum_us": float(sums[c]), "count": int(cnts[c]),
                               "bytes": float(a["g_bytes"][sel][m].sum()) if m.any() else 0.0}
        compute_us = float(excl[COMPUTE_IDS].sum())
        comm_exp_us = float(excl[COMM_IDS].sum())
        memcpy_exp_us = float(excl[MEMCPY_IDS].sum())

        # ---- idle ----
        gaps = iv.runs(~busy, lo, hi)
        glen = gaps[1] - gaps[0]
        fine_m = glen < MIN_GAP_US
        fine = (gaps[0][fine_m], gaps[1][fine_m])
        coarse = (gaps[0][~fine_m], gaps[1][~fine_m])
        fine_us = float(glen[fine_m].sum())
        att = self.attribute(coarse, psets, phase_order, t0, t1)
        host_us = {c: att["class_us"].get(c, 0.0) for c in tx.HOST_CLASSES}
        unattr_us = att["unattributed_us"]
        # launch-API signal (host inside launch calls while the GPU is idle)
        launch_u = self.launch_union(t0, t1)
        idle_set = (gaps[0], gaps[1])
        launch_idle = iv.overlap_total(idle_set, launch_u)
        launch_fine = iv.overlap_total(fine, launch_u)

        total = compute_us + comm_exp_us + memcpy_exp_us + sum(host_us.values()) + unattr_us + fine_us
        residual = total - wall
        # independent cross-check: busy from a plain union of all GPU intervals (not the segmentation)
        busy_ind = iv.total(iv.union(gs, ge))
        res_busy = (compute_us + comm_exp_us + memcpy_exp_us) - busy_ind
        res_idle = (sum(host_us.values()) + unattr_us + fine_us) - (wall - busy_ind)
        res["buckets_us"] = {"compute": compute_us, "comm_exposed": comm_exp_us, "memcpy_exposed": memcpy_exp_us,
                             "host": host_us, "host_total": float(sum(host_us.values())),
                             "unattributed_idle": unattr_us, "fine_idle": fine_us}
        worst = max(abs(residual), abs(res_busy), abs(res_idle))
        res["conservation"] = {"sum_us": total, "wall_us": wall, "residual_us": residual,
                               "busy_crosscheck_residual_us": res_busy, "idle_crosscheck_residual_us": res_idle,
                               "residual_pct": (worst / wall * 100.0) if wall > 0 else 0.0,
                               "ok": bool(worst <= CONSERVATION_TOL * max(wall, 1e-9))}
        res["gpu_busy_us"] = float(seglen[busy].sum())
        res["gpu_idle_us"] = float(glen.sum())
        res["compute"] = compute
        res["comm"] = comm
        res["memcpy"] = memcpy
        res["host_classes"] = att["classes"]
        res["idle"] = {"coarse_us": float((coarse[1] - coarse[0]).sum()), "coarse_n": int(coarse[0].size),
                       "fine_us": fine_us, "fine_n": int(fine[0].size),
                       "launch_api_overlap_idle_us": launch_idle, "launch_api_overlap_fine_us": launch_fine,
                       "min_gap_us": MIN_GAP_US}
        res["gaps_top"] = att["gaps_top"]

        # ---- phases ----
        phases = {}
        for pi, p in enumerate(tx.PHASES):
            m = seg_phase == pi
            pw = float(seglen[m].sum())
            if pw <= 0:
                continue
            ex = np.bincount(owner[m & busy], weights=seglen[m & busy], minlength=len(CATS)) if (m & busy).any() \
                else np.zeros(len(CATS))
            pset = iv.runs(m, lo, hi)
            pf = iv.overlap_total(fine, pset)
            ph = att["phase_class_us"].get(p, {})
            phases[p] = {"wall_us": pw, "compute": float(ex[COMPUTE_IDS].sum()),
                         "comm_exposed": float(ex[COMM_IDS].sum()),
                         "comm_exposed_by_dim": {CATS[c]: float(ex[c]) for c in COMM_IDS if ex[c] > 0},
                         "memcpy_exposed": float(ex[MEMCPY_IDS].sum()),
                         "host": {c: v for c, v in ph.items() if c != "_unattributed"},
                         "unattributed_idle": ph.get("_unattributed", 0.0), "fine_idle": pf}
        res["phases"] = phases
        res["phase_method"] = pmethod
        res["microbatches"] = n_micro

        # ---- kernels / comm detail / memory ----
        res["kernels"] = self.kernel_stats(sel, gs, ge, gcat, top_n=int(self.opts.get("top_n", 20)))
        res["comm_bw"], cdur = self.comm_detail(sel, gs, ge)
        res["p2p"] = commmod.p2p_window(self.comm, self.p2p, cdur, self.rank, self.meta["device"]["name"], self.opts) \
            if cdur is not None else {"kernels": 0, "kernel_us": 0.0}
        res["memory"] = memmod.summary(self, t0, t1, psets, phase_order)
        res["memory_mgmt"] = memmgmt.window(self, t0, t1, idle_set)
        # the same time two ways: share of summed GPU activity time vs exposed share of the step
        ksum = float(sums.sum())
        share = {"compute": (float(sums[COMPUTE_IDS].sum()), compute_us)}
        for c in COMM_IDS:
            if cnts[c]:
                share[CATS[c]] = (float(sums[c]), float(excl[c]))
        share["memcpy"] = (float(sums[MEMCPY_IDS].sum()), memcpy_exp_us)
        res["kernel_share"] = {"gpu_activity_sum_us": ksum, "rows": {
            k: {"sum_us": s, "pct_of_gpu_activity_sum": (100.0 * s / ksum) if ksum else 0.0,
                "step_us": e, "pct_of_step": 100.0 * e / wall if wall else 0.0} for k, (s, e) in share.items()},
            "note": "pct_of_gpu_activity_sum = summed kernel/memcpy durations (overlap counted twice); "
                    "pct_of_step = exposed share of step wall (the conserved buckets)"}
        res["optimizer"] = self.optimizer_context(t0, t1, psets, sel, gs, ge, gcat, compute, host_us)
        return res

    def optimizer_context(self, t0, t1, psets, sel, gs, ge, gcat, compute, host_us):
        """Facts bearing on the optimizer's cost: where the update runs, its time, copies, state size."""
        a = self.a
        out = {}
        ind = {}
        if self.offload_name_mask.any():
            m = np.flatnonzero(self.offload_name_mask[a["h_name"]] & (self.h_start < t1) & (self.h_end > t0))
            for i in m.tolist():
                nm = self.hnames[int(a["h_name"][i])]
                ind[nm] = ind.get(nm, 0) + 1
        out["offload_indicators"] = [{"frame": k, "n": v} for k, v in sorted(ind.items(), key=lambda kv: -kv[1])[:4]]
        out["cpu_update_frames_present"] = bool(ind)
        ops = psets.get("optimizer", iv.EMPTY)
        out["phase_wall_us"] = iv.total(ops)
        out["host_optimizer_us"] = float(host_us.get("optimizer_cpu_step", 0.0))
        out["gpu_optimizer_kernel_us"] = float(compute["optimizer"]["sum_us"])
        upd = memmod._outermost(self, self.opt_update_mask, t0, t1) if self.opt_update_mask.any() else np.zeros(0, int)
        if upd.size:
            names = {}
            for i in upd.tolist():
                nm = self.hnames[int(a["h_name"][i])]
                x = names.setdefault(nm, [0, 0.0])
                x[0] += 1
                x[1] += float(a["h_dur"][i])
            out["update_ops_host"] = [{"op": k, "n": v[0], "host_us": v[1]}
                                      for k, v in sorted(names.items(), key=lambda kv: -kv[1][1])[:3]]
        else:
            out["update_ops_host"] = []
        cp = {}
        if ops[0].size:
            mc = np.isin(gcat, MEMCPY_IDS)
            idx = np.flatnonzero(mc)
            if idx.size:
                inside = iv.contains_points(ops, gs[idx])
                for i in idx[inside].tolist():
                    d = CATS[int(gcat[i])]
                    x = cp.setdefault(d, {"n": 0, "us": 0.0, "bytes": 0.0})
                    x["n"] += 1
                    x["us"] += float(ge[i] - gs[i])
                    x["bytes"] += float(a["g_bytes"][sel[i]])
        out["copies_in_phase"] = cp
        est = commmod.optimizer_shard_estimate(self.meta, a, t0, t1)
        if est:
            est["state_bytes_est"] = est["params_in_shard"] * ADAM_STATE_BYTES_PER_PARAM
            est["state_assumption"] = ("Adam-style state: fp32 master weight + 2 fp32 moments = %d B/param; "
                                       "estimate" % ADAM_STATE_BYTES_PER_PARAM)
        out["shard_estimate"] = est
        return out

    def launch_union(self, t0, t1):
        a = self.a
        m = (a["l_ts"] < t1) & (a["l_ts"] + a["l_dur"] > t0) & np.isin(a["l_tid"], self.driver_tids)
        return iv.clip(iv.union(a["l_ts"][m], a["l_ts"][m] + a["l_dur"][m]), t0, t1)

    # ---------- idle attribution ----------
    def attribute(self, coarse, psets, phase_order, t0, t1):
        """Split each coarse idle interval among the innermost host frames active at each instant.

        Across driver threads, the innermost frame with the shortest duration wins; the class is the
        highest-priority class on that frame's ancestor chain.
        """
        a = self.a
        out = {"class_us": {}, "classes": {}, "unattributed_us": 0.0, "gaps_top": [], "phase_class_us": {}}
        if coarse[0].size == 0:
            return out
        gap_s, gap_e = coarse
        thread_segs = []
        frame_cls, frame_dec, frame_par = {}, {}, {}
        for tid in self.driver_tids:
            idx = self.thread_frames(tid)
            if idx.size == 0:
                continue
            s = self.h_start[idx]
            e = self.h_end[idx]
            k = np.searchsorted(gap_e, s, side="right")
            ok = k < gap_s.size
            ov = np.zeros(idx.size, dtype=bool)
            ov[ok] = gap_s[k[ok]] < e[ok]
            fr = idx[ov]
            # frames spanning the whole window (ProfilerStep, worker main loops) name nothing: not attribution
            slack = min(CONTEXT_SLACK_US, 0.001 * (t1 - t0))
            ctx = (self.h_start[fr] <= t0 + slack) & (self.h_end[fr] >= t1 - slack)
            ctx |= self.step_name_mask[a["h_name"][fr]]
            fr = fr[~ctx]
            if fr.size == 0:
                continue
            segs = self._sweep(fr, frame_cls, frame_dec, frame_par)
            if segs[0].size:
                thread_segs.append(segs)
        # elementary pieces within coarse gaps
        b = [gap_s, gap_e] + [np.concatenate([ts[0], ts[1]]) for ts in thread_segs]
        b += [np.concatenate([psets[p][0], psets[p][1]]) for p in phase_order]
        lo, hi = iv.elementary(np.concatenate(b), t0, t1)
        inside = iv.segment_active(coarse, lo, hi)
        lo, hi = lo[inside], hi[inside]
        best = np.full(lo.size, -1, dtype=np.int64)
        bestdur = np.full(lo.size, np.inf)
        for (ss, se, sf) in thread_segs:
            k = np.searchsorted(ss, lo, side="right") - 1
            ok = k >= 0
            kk = np.where(ok, k, 0)
            hit = ok & (se[kk] >= hi)
            f = np.where(hit, sf[kk], -1)
            d = np.where(f >= 0, a["h_dur"][np.maximum(f, 0)], np.inf)
            better = d < bestdur
            best[better] = f[better]
            bestdur[better] = d[better]
        plen = hi - lo
        # per-piece class and deciding frame
        cls = np.full(lo.size, -1, dtype=np.int16)
        dec = np.full(lo.size, -1, dtype=np.int64)
        has = best >= 0
        if has.any():
            uf = np.unique(best[has])
            cmap = {int(f): frame_cls[int(f)] for f in uf}
            dmap = {int(f): frame_dec[int(f)] for f in uf}
            cls[has] = [cmap[int(f)] for f in best[has]]
            dec[has] = [dmap[int(f)] for f in best[has]]
        out["unattributed_us"] = float(plen[~has].sum())
        classes = {}
        for ci, cname in enumerate(tx.HOST_CLASSES):
            m = cls == ci
            if not m.any():
                continue
            t = float(plen[m].sum())
            out["class_us"][cname] = t
            dd = dec[m]
            pl = plen[m]
            ud, inv = np.unique(dd, return_inverse=True)
            tt = np.bincount(inv, weights=pl)
            names = {}
            for j, f in enumerate(ud):
                nm = self.hnames[int(a["h_name"][f])]
                r = names.setdefault(nm, [0.0, 0])
                r[0] += tt[j]
                r[1] += 1
            top = sorted(names.items(), key=lambda kv: -kv[1][0])[:5]
            classes[cname] = {"us": t, "n_calls": int(ud.size),
                              "top_frames": [{"frame": nm, "us": v[0], "n": v[1]} for nm, v in top]}
        out["classes"] = classes
        # per-phase split of host time
        ph = np.full(lo.size, len(tx.PHASES) - 1, dtype=np.int8)
        for p in reversed(phase_order):
            if psets[p][0].size:
                ph[iv.segment_active(psets[p], lo, hi)] = tx.PHASES.index(p)
        for pi, p in enumerate(tx.PHASES):
            m = ph == pi
            if not m.any():
                continue
            d = {}
            for ci, cname in enumerate(tx.HOST_CLASSES):
                mm = m & (cls == ci)
                if mm.any():
                    d[cname] = float(plen[mm].sum())
            d["_unattributed"] = float(plen[m & ~has].sum())
            out["phase_class_us"][p] = d
        # largest gaps with their dominant frame
        order = np.argsort(-(gap_e - gap_s))[:TOP_GAPS_KEPT]
        gi_of_piece = np.searchsorted(gap_s, lo, side="right") - 1
        gaps_top = []
        for g in order:
            m = gi_of_piece == g
            rec = {"t_ms": (gap_s[g] - t0) / 1e3, "dur_ms": (gap_e[g] - gap_s[g]) / 1e3}
            if m.any():
                pl = plen[m]
                bb = best[m]
                cc = cls[m]
                j = int(np.argmax(pl))
                rec["attributed_pct"] = float(pl[bb >= 0].sum() / pl.sum() * 100)
                # class with most time in this gap
                ct = {}
                for x, y in zip(cc, pl):
                    ct[int(x)] = ct.get(int(x), 0.0) + y
                top_c = max(ct.items(), key=lambda kv: kv[1])[0]
                rec["class"] = tx.HOST_CLASSES[top_c] if top_c >= 0 else "unattributed"
                f = int(bb[j])
                if f >= 0:
                    rec["innermost"] = self.hnames[int(a["h_name"][f])]
                    rec["decided_by"] = self.hnames[int(a["h_name"][frame_dec[f]])]
                    chain = []
                    x = f
                    while x >= 0 and len(chain) < 8:
                        chain.append(self.hnames[int(a["h_name"][x])])
                        x = frame_par.get(x, -1)
                    rec["chain"] = chain
                    rec["tid"] = int(a["h_tid"][f])
            gaps_top.append(rec)
        out["gaps_top"] = gaps_top
        return out

    def _sweep(self, fr, frame_cls, frame_dec, frame_par):
        """Innermost-frame segmentation of one thread's (sorted, possibly nested) frames."""
        hs = self.h_start
        he = self.h_end
        hname = self.a["h_name"]
        pri = self.h_class_by_name
        npri = len(tx.HOST_CLASSES)
        other = tx.HOST_CLASSES.index("python_other")
        seg_s, seg_e, seg_f = [], [], []
        stack = []           # (end, idx)
        cur = -np.inf
        for i in fr.tolist():
            s = hs[i]
            e = he[i]
            while stack and stack[-1][0] <= s:
                end, top = stack.pop()
                if end > cur:
                    seg_s.append(max(cur, hs[top]))
                    seg_e.append(end)
                    seg_f.append(top)
                    cur = end
            if stack:
                pe, par = stack[-1]
                if s > cur:
                    seg_s.append(max(cur, hs[par]))
                    seg_e.append(s)
                    seg_f.append(par)
                if e > pe:
                    e = pe
                frame_par[i] = par
                pc = frame_cls[par]
                pd = frame_dec[par]
            else:
                frame_par[i] = -1
                pc, pd = npri, -1
            cur = max(cur, s)
            own = int(pri[hname[i]])
            if own < pc:
                frame_cls[i] = own
                frame_dec[i] = i
            else:
                frame_cls[i] = pc
                frame_dec[i] = pd
            stack.append((e, i))
        while stack:
            end, top = stack.pop()
            if end > cur:
                seg_s.append(max(cur, hs[top]))
                seg_e.append(end)
                seg_f.append(top)
                cur = end
        # frames with no classified ancestor are "python other", decided by the innermost frame
        for i in fr.tolist():
            if frame_cls[i] >= npri:
                frame_cls[i] = other
                frame_dec[i] = i
        return (np.array(seg_s, dtype=np.float64), np.array(seg_e, dtype=np.float64),
                np.array(seg_f, dtype=np.int64))

    # ---------- kernel stats ----------
    def kernel_stats(self, sel, gs, ge, gcat, top_n=20):
        a = self.a
        k = a["g_kind"][sel] == 0
        dur = (ge - gs)[k]
        names = a["g_name"][sel][k]
        cats = gcat[k]
        out = {"count": int(k.sum())}
        if dur.size == 0:
            return out
        out["median_us"] = float(np.median(dur))
        small = dur < SMALL_KERNEL_US
        out["small_lt10us_count_pct"] = float(small.mean() * 100)
        out["small_lt10us_time_pct"] = float(dur[small].sum() / max(dur.sum(), 1e-9) * 100)
        by = {}
        for c in np.unique(cats):
            m = cats == c
            d = dur[m]
            by[CATS[c]] = {"count": int(m.sum()), "sum_us": float(d.sum()), "median_us": float(np.median(d)),
                           "small_lt10us_count_pct": float((d < SMALL_KERNEL_US).mean() * 100)}
        out["by_class"] = by
        # top kernels by total time
        tot = np.bincount(names, weights=dur, minlength=len(self.knames))
        cnt = np.bincount(names, minlength=len(self.knames))
        order = np.argsort(-tot)[:max(top_n, 30)]
        cat_of = {}
        for n_, c_ in zip(names, cats):
            cat_of.setdefault(int(n_), int(c_))
        out["top"] = [{"kernel": self.knames[i], "class": CATS[cat_of.get(int(i), 0)], "us": float(tot[i]),
                       "count": int(cnt[i])} for i in order if tot[i] > 0]
        # achieved TFLOP/s where FLOPs were recorded
        fl = a["g_flops"][sel][k]
        tf = {}
        for cname in ("gemm", "grouped_gemm", "attention"):
            m = (cats == CAT_IDX[cname]) & np.isfinite(fl)
            if m.any() and dur[m].sum() > 0:
                v = fl[m].sum() / (dur[m].sum() * 1e-6) / 1e12
                peak = self.opts.get("peak_tflops") or tx.lookup(tx.PEAK_BF16_TFLOPS, self.meta["device"]["name"])
                tf[cname] = {"tflops": float(v), "kernels_with_flops": int(m.sum()),
                             "of_class_kernels": int((cats == CAT_IDX[cname]).sum()),
                             "vs_dense_bf16_peak": (v / peak) if peak else None}
        out["tflops"] = tf
        return out

    def comm_detail(self, sel, gs, ge):
        cidx = self.a["c_idx"]
        if cidx.size == 0:
            return {}, None
        pos = np.full(self.a["g_ts"].size, -1, dtype=np.int64)
        pos[cidx] = np.arange(cidx.size)
        d = np.zeros(cidx.size)
        p = pos[sel]
        m = p >= 0
        d[p[m]] = (ge - gs)[m]
        return commmod.bandwidth_table(self.comm, cidx, d, self.meta["device"]["name"], self.opts), d


def v1_compat(r):
    """Quantities v1 reported, computed the v1 way (whole trace, GPU active span)."""
    a = r.a
    out = {}
    kn = r.knames
    kind = a["g_kind"]
    dur = a["g_dur"]
    names = a["g_name"]
    n = kind.size
    if n == 0:
        return {"gpu_util_pct": 0.0, "gpu_idle_pct": 100.0, "total_kernel_ms": 0.0}
    cat_tot, cat_cnt = {}, {}
    per_name = np.bincount(names[kind == 0], weights=dur[kind == 0], minlength=len(kn))
    per_cnt = np.bincount(names[kind == 0], minlength=len(kn))
    for i in np.flatnonzero(per_cnt):
        c = tx.v1_category(kn[i])
        cat_tot[c] = cat_tot.get(c, 0.0) + per_name[i]
        cat_cnt[c] = cat_cnt.get(c, 0) + int(per_cnt[i])
    for i in np.flatnonzero(kind != 0):
        c = tx.v1_copy_category(kn[int(names[i])], kind[i] == 2)
        cat_tot[c] = cat_tot.get(c, 0.0) + dur[i]
        cat_cnt[c] = cat_cnt.get(c, 0) + 1
    total = sum(cat_tot.values()) or 1.0
    s, e = iv.union(a["g_ts"], a["g_ts"] + dur)
    busy = float((e - s).sum())
    span = float(e[-1] - s[0]) if s.size else 0.0
    gaps = np.sort(s[1:] - e[:-1])[::-1] if s.size > 1 else np.zeros(0)
    lower = [x.lower() for x in kn]
    is_nccl = np.array(["nccl" in x for x in lower], dtype=bool)
    is_dep = np.array(["deep_ep" in x for x in lower], dtype=bool)
    is_p2p = np.array([("sendrecv" in x or "_send" in x or "_recv" in x) and ("nccl" in x or "deep_ep" in x)
                       for x in lower], dtype=bool)
    k = kind == 0
    nccl_us = float(dur[k & is_nccl[names]].sum())
    dep_us = float(dur[k & is_dep[names]].sum())
    cm = k & (is_nccl[names] | is_dep[names])
    cu = iv.union(a["g_ts"][cm], a["g_ts"][cm] + dur[cm])
    pm = k & is_p2p[names]
    pu = iv.union(a["g_ts"][pm], a["g_ts"][pm] + dur[pm])
    top_names = np.argsort(-per_name)[:5]
    cats = sorted(cat_tot.items(), key=lambda kv: -kv[1])
    out.update({
        "gpu_util_pct": round(busy / span * 100, 2) if span else 0.0,
        "gpu_idle_pct": round(100 - busy / span * 100, 2) if span else 100.0,
        "gpu_active_span_ms": round(span / 1e3, 1), "gpu_busy_ms": round(busy / 1e3, 1),
        "total_kernel_ms": round(total / 1000, 2),
        "comm_pct_of_kernel": round((nccl_us + dep_us) / total * 100, 2),
        "nccl_ms": round(nccl_us / 1000, 2), "deep_ep_ms": round(dep_us / 1000, 2),
        "profile_top_categories": {c: round(v / total * 100, 2) for c, v in cats[:8]},
        "profile_all_categories_ms": {c: round(v / 1000, 1) for c, v in cats},
        "profile_all_categories_count": {c: cat_cnt.get(c, 0) for c, _ in cats},
        "profile_top_kernels": [{"kernel": kn[i][:90], "pct": round(per_name[i] / total * 100, 2),
                                 "count": int(per_cnt[i])} for i in top_names if per_cnt[i] > 0],
        "top_gaps_us": [round(float(g), 1) for g in gaps[:5]],
        "collective_wait_pct_of_span": round(iv.total(cu) / span * 100, 2) if span else 0.0,
        "p2p_wait_pct_of_span": round(iv.total(pu) / span * 100, 2) if span else 0.0,
    })
    if a["m_ts"].size:
        gib = 1024.0 ** 3
        tot_mem = r.meta["device"].get("total_mem") or 0
        pr = float(a["m_resv"].max())
        out["peak_reserved_gib"] = round(pr / gib, 2)
        out["device_total_gib"] = round(tot_mem / gib, 2) if tot_mem else None
        out["allocator_unreserved_gib"] = round((tot_mem - pr) / gib, 2) if tot_mem else None
        out["allocator_unreserved_note"] = memmod.UNRESERVED_NOTE
    else:
        out["peak_reserved_gib"] = None
        tot_mem = r.meta["device"].get("total_mem") or 0
        out["device_total_gib"] = round(tot_mem / 1024.0 ** 3, 2) if tot_mem else None
        out["allocator_unreserved_gib"] = None
    return out
