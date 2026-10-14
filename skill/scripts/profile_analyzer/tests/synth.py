# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build small kineto-shaped traces in memory for tests."""
import gzip
import json

PID = 100
MAIN = 100
AUTOGRAD = 101


class Trace(object):
    def __init__(self, rank=0, world=1, pgs=None, device_props=True, dist=True, flags=None):
        self.ev = []
        self.corr = 1000
        self.top = {"schemaVersion": 1, "traceEvents": self.ev}
        if device_props:
            self.top["deviceProperties"] = [{"id": i, "name": "NVIDIA H100 80GB HBM3", "totalGlobalMem": 80 * 2 ** 30,
                                             "numSms": 132} for i in range(8)]
        if dist:
            self.top["distributedInfo"] = {"backend": "nccl", "rank": rank, "world_size": world,
                                           "pg_config": pgs or []}
        self.top.update(flags or {"with_stack": 1, "profile_memory": 1, "record_shapes": 1, "with_flops": 0})
        self.top["baseTimeNanoseconds"] = 1700000000000000000

    def step(self, t0, t1, n=1):
        self.ev.append({"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#%d" % n, "pid": PID, "tid": MAIN,
                        "ts": t0, "dur": t1 - t0})

    def frame(self, t0, t1, name, cat="python_function", tid=MAIN):
        self.ev.append({"ph": "X", "cat": cat, "name": name, "pid": PID, "tid": tid, "ts": t0, "dur": t1 - t0})

    def kernel(self, t0, t1, name="sm90_xmma_gemm_bf16", stream=7, launch_tid=MAIN, args=None, cat="kernel"):
        self.corr += 1
        a = {"device": 0, "stream": stream, "correlation": self.corr}
        a.update(args or {})
        self.ev.append({"ph": "X", "cat": cat, "name": name, "pid": 0, "tid": stream, "ts": t0, "dur": t1 - t0,
                        "args": a})
        if launch_tid is not None:
            api = "cudaLaunchKernel" if cat == "kernel" else "cudaMemcpyAsync"
            self.ev.append({"ph": "X", "cat": "cuda_runtime", "name": api, "pid": PID, "tid": launch_tid,
                            "ts": max(t0 - 5.0, 0.0), "dur": 4.0, "args": {"correlation": self.corr}})

    def nccl(self, t0, t1, coll="allreduce", desc="TENSOR_MODEL_PARALLEL_GROUP", n=2, nelems=1 << 20,
             kname="ncclDevKernel_AllReduce_Sum_bf16_RING_LL", stream=20, ranks=None, launch_tid=MAIN):
        args = {}
        if coll is not None:
            args = {"Collective name": coll, "In msg nelems": nelems, "Out msg nelems": nelems, "Group size": n,
                    "dtype": "BFloat16", "Process Group Name": "7", "Process Group Description": desc,
                    "Process Group Ranks": str(ranks or list(range(n)))}
        self.kernel(t0, t1, kname, stream=stream, args=args, launch_tid=launch_tid)

    def memcpy(self, t0, t1, direction="DtoH (Device -> Pinned)", nbytes=1 << 20, stream=30):
        self.kernel(t0, t1, "Memcpy " + direction, stream=stream, args={"bytes": nbytes}, cat="gpu_memcpy")

    def rpc(self, t0, t1, coll, nelems, dtype="BFloat16", peer=None, desc=None, pg="0", gs=None, tid=MAIN):
        """A record_param_comms cpu_op (message size / peer metadata, as PyTorch records it)."""
        a = {"Collective name": coll, "In msg nelems": nelems, "Out msg nelems": nelems, "dtype": dtype,
             "Process Group Name": pg}
        if desc:
            a["Process Group Description"] = desc
        if gs:
            a["Group size"] = gs
        if peer is not None:
            a["Dst Rank" if coll == "send" else "Src Rank"] = peer
        self.ev.append({"ph": "X", "cat": "cpu_op", "name": "record_param_comms", "pid": PID, "tid": tid, "ts": t0,
                        "dur": t1 - t0, "args": a})

    def mem(self, t, alloc, resv):
        self.ev.append({"ph": "i", "cat": "cpu_instant_event", "s": "t", "name": "[memory]", "pid": PID, "tid": MAIN,
                        "ts": t, "args": {"Total Allocated": alloc, "Total Reserved": resv, "Device Id": 0,
                                          "Device Type": 1, "Bytes": 1}})

    def data(self):
        return self.top

    def write(self, path):
        with gzip.open(path, "wt") as f:
            json.dump(self.top, f)
        return path
