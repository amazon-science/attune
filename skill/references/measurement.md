# Measurement — logs, the metric, discipline, profiling (depth reference)

A tuning result is only as good as the measurement producing it. Most wasted loop-time comes
from acting on a number that was noise, or a number that measured the wrong thing. This file
covers **where the numbers live, how to reduce them, how to know one is real, and how to
profile the bottleneck.** SKILL.md owns the single-job protocol and verdict thresholds; this
is the how-to behind them.

---

## 0. Fixed-shape benchmark contract

- The workload comes from `scripts/workload.py`: GBS `$AUTOTUNE_GBS` (default 128), row length `$AUTOTUNE_SEQ_LEN` (default 8,192), dataset `$AUTOTUNE_DATASET`, N = `$AUTOTUNE_NUM_STEPS` steps unprofiled (default 20, at least 20) and N+2+K profiled, with K = `$AUTOTUNE_PROFILE_CAPTURE` recorded steps (default 1, so 23).
- Use frozen, model-tokenizer-specific rows of exactly the row length with no padding.
- Hold dataset bytes/order, seeds, objective, GBS, hardware type, image, and code fixed (and the GPU count, fixed at the baseline's count: SKILL.md Invariants).
- Run one N-step job per recipe. Do not restore, publish, or persist a Triton cache.
- Discard warm-up steps 0-4 and rank only the complete window 5..N-1 (5-19 at the default N=20), the same window for a profiled and an unprofiled run.
- When a run is profiled it runs N+2+K steps (default 23): the same ranked window, then the profiler's setup step N
  (20), its warm-up step N+1 (21) and the K recorded steps N+2..N+1+K (22; up to 22-24 with
  `make_run_yaml.py --profile-capture 3`); never profile the ranked window. Profiling is per launch, not part of the
  contract.
- Steps 0-19 of an unprofiled run are also its side of the final correctness check (`pair_from_runs.py`), so a
  shipped number and its correctness evidence come from the same run unless the ranked run was profiled. A profiled
  run changes `--num-rollout`, which the LR schedule can depend on, so it is never a side of the check; such a recipe
  gets one unprofiled confirm run for it (SKILL.md, Verdict).
- Do not apply this protocol to variable-shape data. Use a separately validated window there.

## 1. Where the logs live

- **Tee'd training log:** `$AUTOTUNE_EXPERIMENTS_DIR/<run_id>/logs/<timestamp>/train.log` on the shared volume
  (the scripts default `$AUTOTUNE_EXPERIMENTS_DIR` to `/shared/experiments`; the baseline YAML's training script
  writes the log, and `poll_run.sh` reports the newest one as `log_path`) — the most reliable source, grep-friendly, contains every
  `perf N:` line. `read_metrics.sh` consumes this.
- **Ray driver pod stdout:** `kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" logs <rayjob-submitter-pod>`
  — same output with Ray actor prefixes plus NCCL INFO noise.
- **Ray worker pod stdout:** the same `kubectl logs` on a worker pod — rank-0 output (what you want) is
  on the head/driver, not here.

The operator provides a long-lived log-access pod, `$AUTOTUNE_LOG_POD`, that mounts the shared volume read-only at
`$AUTOTUNE_SHARED_MOUNT` and is not part of any RayJob (any small Linux image with bash, coreutils and python3,
running `sleep infinity`). You read logs and traces through it with read-only `kubectl exec` (`ls`, `cat`,
`tail`, `grep`) without stopping the job. If it does not exist, ask the operator; do not create pods yourself.
The launcher starts a campaign only when every mount of the shared volume in the pod is read-only, as here:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: alice-log-access          # $AUTOTUNE_LOG_POD
spec:
  containers:
  - name: main
    image: <small Linux image with bash, coreutils and python3>
    command: ["sleep", "infinity"]
    volumeMounts:
    - name: shared
      mountPath: /shared          # $AUTOTUNE_SHARED_MOUNT
      readOnly: true
  volumes:
  - name: shared
    persistentVolumeClaim:
      claimName: <the shared volume's claim>
```

`read_metrics.sh`, `poll_run.sh`, and `teardown_run.sh` pass `$AUTOTUNE_KUBECTL_CONTEXT` and `$KUBECTL_NS` on every
call.

**One run_id per log before trusting metrics:** same-second launches sharing a `RUN_TS`-keyed
LOG_DIR interleave into one file and contaminate the aggregate. Always
`grep -oE "<run_id>" train.log | sort -u` and confirm exactly one before reading.

## 2. The `perf N:` line

Slime emits one line per rollout with all throughput metrics:

```
[ts] train_metric_utils.py:<line> - perf 17: {'perf/train_wait_time': <s>, 'perf/actor_train_time': <s>,
 'perf/log_probs_time': 0, 'perf/actor_train_tflops': <f>, 'perf/actor_train_tok_per_s': <f>,
 'perf/step_time': <s>, 'perf/wait_time_ratio': <f>, 'perf/actor_train_mfu_pct': <f>, ...}
```

`17` is the rollout/step ID.

| Field | Meaning |
|---|---|
| `perf/actor_train_tok_per_s` | **North-star numerator.** Training throughput in tokens/sec across all GPUs. |
| `perf/actor_train_tflops` | Aggregate training TFLOPS (MoE-aware formula). |
| `perf/actor_train_mfu_pct` | MFU %. Sanity: `actor_train_tflops / (N_GPUs × peak_TFLOPS) × 100`. |
| `perf/step_time` | Total wall-clock per step = `train_wait + train`. |
| `perf/actor_train_time` | Forward + backward + optimizer. |
| `perf/train_wait_time` | Time waiting on data/rollout. ~0 for SFT. **Read this before blaming rollout/topology for a big step_max** (see §5). |
| `perf/wait_time_ratio` | `train_wait / step_time`. Want ~0. |
| `perf/log_probs_time` | Log-probs forward (RL mode). 0 for SFT. |
| `perf/rollout_time` | CPU tokenization (SFT) or SGLang gen (RL). |

**Rollout perf line — different format, earlier in each step, don't confuse it:**
```
[ts] rollout.py:<line> - perf 0: {'rollout/response_len/mean': <f>, ..., 'perf/rollout_time': <s>, ...}
```

**Grep patterns:**
```bash
grep "train_metric_utils.*perf [0-9]" train.log                 # all train perf lines
grep -oE "'perf/actor_train_tok_per_s': [0-9.]+" train.log | awk -F: '{print $NF}'
```

## 3. The aggregate metric

```
aggregate_tok_per_s_per_gpu = total_tokens_in_window / total_step_time_in_window / num_gpus
```

- **Window: steps 5-19 of the single run** (15 samples at the default step count). Steps 0-4 absorb startup, fixed-shape
  compilation and allocator warm-up; a few layouts still run 10-15% slow at step 4. The window is predeclared and
  identical for every recipe.
- **Do NOT use `mean(per_step_tok_per_s)`** — averaging rates over-weights short steps.
  Aggregate-tokens-over-aggregate-time is the correct reduction.
- **Report the step_time distribution alongside the aggregate** (mean, standard deviation, CV,
  p50, min, max). A tight p50 with a catastrophic max is diagnostic — see §5. Within-run CV is
  a stability diagnostic, not independent replication.
- `read_metrics.sh <log> 5 19 <num_gpus>` implements this and emits JSON. The window defaults to
  `$AUTOTUNE_WINDOW_START` (5) through N-1 (19 at the default N=20). Prefer it over one-off greps. It exits
  non-zero / `valid:false` on a short window or bad data.

**MFU expectation:** large-MoE SFT at moderate GPU counts is comm-bound — expect MFU far below
dense-training "healthy" bands regardless of tuning; those bands are not a target here.

## 4. Discipline — is this number real?

**One run per candidate; no throughput reruns.** Each valid result is compared with the current leader by the
band of SKILL.md "Verdict" (win, tie or lose); a tie is documented, not rerun. The band is a predeclared noise
threshold: validate it once for the benchmark methodology on representative long traces and disjoint windows,
not per candidate.

**Single-variable ablations:** if two configs differ by >1 flag, the delta is unattributable. Only
exception: flags Megatron requires paired (e.g. `--overlap-grad-reduce` + `--overlap-param-gather`)
— call the coupling out in the hypothesis. Never stack "winner + new flag" on top of other
un-validated recent changes.

**Don't dynamically pick windows to make a config win.** The window (5-19) is pre-declared and
fixed across the sweep; consistency matters more than optimizing each comparison.

**Sanity checks before trusting any number:**
- Run reached its last step (19 unprofiled, 22 profiled, at the defaults; `poll_run.sh` prints the `target_step` it used)? A shorter `SUCCEEDED` RayJob is incomplete or
  swallowed an actor error.
- Every perf line present in 5-19? A short count = dropped emission → aggregate invalid
  (`read_metrics.sh` returns `valid:false`).
- No persistent compilation cache was restored, mounted, or published?
- No new shape or compilation tail appeared in the measured window? A late compile invalidates the run.
- `num_gpus` right? A 32-vs-16 miscount doubles throughput on paper.
- A fresh run_id? A run that resumes from a stale `<run_id>/checkpoints/` skips steps silently (§7).

**Correctness check.** Throughput says nothing about whether a recipe still trains the same model. Every shipped
recipe passes the two-layer correctness check (configuration, then 20-step convergence) run as the final cascade
(SKILL.md "When to stop"). `scripts/correctness/PROTOCOL.md` is the authoritative protocol: pair setup, the
`pair_from_runs.py` and `parity_verdict.py` commands (and `make_parity_arm.py` for diagnostic arms), the
thresholds, and how to validate the check on a new setup.

## 5. Compile-tail vs rollout-stall (canonical diagnosis)

**Symptom:** `step_time_max` is many times the p50 while p50 stays tight. Naive reading
"rollout stall" is usually **wrong**.

**Distinguish by reading `perf/train_wait_time` vs `perf/actor_train_time` — one grep:**

| train_wait_time | actor_train_time | Diagnosis |
|---|---|---|
| spiked | normal (≈ p50) | **Real rollout stall** — trainer waiting on inference. Needs warmup requests / persistent engine. |
| stays near zero | spiked | **JIT recompile tail** — a kernel rebuild triggered by a new microbatch shape (the diagnostic — `train_wait≈0` while `actor_train` spikes — is empirical/verified; the *mechanism* is uncited, likely Megatron/TE JIT — confirm against the framework/version before asserting it). Dynamic batching exposes new shapes early; higher PP depth × vpp enlarges the shape space. |

**Fixed-shape policy:** a fresh job may compile during startup, which is why steps 0-4 are excluded.
Exact fixed-length rows should not introduce a new shape later. Any compile-tail signature in the measured window
means the run violated that assumption; mark it invalid and diagnose the data path or runtime
rather than changing the window. Do not restore a persistent cache to hide the violation.
Do not add compile-cache wiring of your own.

## 6. Profiling

**Profiling is optional per launch, and the choice is yours.** Profile a run when the next decision needs a
diagnosis you do not have — the baseline is the natural first candidate, and a run whose bottleneck you cannot
name from its metrics is the next. A plain value sweep along a lever you have already diagnosed often needs no
trace. A profiled run is charged its full cost like any other, and `make_run_yaml.py --benchmark` produces an
unprofiled run unless you pass `--with-profiler`. Profile after the ranked window because the
profiler's setup, warm-up and captured steps can be materially slower and profiler overhead can vary by recipe.
Record one step by default; `--profile-capture 2|3` records up to three when the step-time CV is high or something
periodic is suspected.

On miles (v0.1.0) the default `miles_compat` hook applies the `SLIME_PROFILE_*` switches as on slime. On another
miles version it leaves miles' profiler factory in place and prints `MILES_COMPAT_HOOK_SKIP`; that factory records
shapes, stacks, memory and FLOPs, so expect larger traces than on slime.

**Standard flags** (`make_run_yaml.py --benchmark --with-profiler` adds these):
```
--use-pytorch-profiler --profile-target train_overall \
--profile-step-start 22 --profile-step-end 23 \
--tensorboard-dir $AUTOTUNE_EXPERIMENTS_DIR/<run_id>/profiles
```
- **`train_overall`** is stepped once per rollout by slime's TrainProfiler. Some slime builds never
  call the `train_actor` path, so that target writes no trace; check a new image's first run for a
  trace before trusting either target (levers.md "Profiler operability").
- **Step 20 sets up the profiler, step 21 warms it up, step 22 is captured (at the default step counts).** Slime's
  schedule is `wait = start - 1`, `warmup = 1`, `active = end - start`, and it steps the profiler at the end of each
  step inside the step's timer, so the setup lands in step START-2 and the warm-up in START-1; the end is exclusive,
  so `22-23` records exactly one step and `22-25` three. `make_run_yaml.py` refuses a range whose setup step falls
  inside steps 0..N-1 or whose end exceeds `--num-rollout`, so profiler work and trace flushing never bias the
  ranked window and a profile is never silently skipped; when the YAML's `--num-rollout` is not a literal integer
  (a shell variable, `--num-epoch`) it refuses the render unless you pass `--unchecked-num-rollout`.
- **One trace per pipeline stage** by default: `make_run_yaml.py --with-profiler` enables the `profile_rank`
  hook (`scripts/profile_rank_hook/`) in the hook chain (`scripts/hook_chain.py`) on every pod, which
  lets only the TP/CP/DP-rank-0 rank of each stage create the profiler. `--profile-ranks all` skips the hook,
  and `SLIME_PROFILE_ALL_RANKS=1` in the pod env (`make_run_yaml.py --env`) traces every rank. A YAML written by
  hand or by another generator needs the same hook, or it traces whatever the image's stock profiler traces.
  Stage-local stalls (one stage waiting on allocator cleanup, a straggler stage) are only visible
  with every stage traced. Metadata (`record_shapes`, `with_flops`, `with_stack`, `profile_memory`)
  follows the image's profiler settings; keep it off on host-RAM-tight configs (levers.md
  "Profiler operability").
- **The image must carry slime's PyTorch profiler controls** (`use_pytorch_profiler` and the profile step
  range in slime's `profile_utils.py`); the flags produce a trace only then. Verify it in the source of the
  image your baseline YAML pins, not from this doc (levers.md "Profiler operability").

**Mandatory post-run analysis** (the classification selects the next batch — not intuition, not
tier order; "no classification = no launch"):
```
run done → read_metrics.sh (ledger aggregate + MFU)
  → copy every trace off $AUTOTUNE_LOG_POD → analyze_profile.py --top-n 20 (ledger profile_top_categories)
  → classify: comm-bound-{sendrecv,allgather,rs} | compute-bound | memory-bound | bubble-bound
  → pick levers whose "applies when" matches (see levers.md §c/§d)
```
The analysis is cheap relative to the next run — skipping it is false economy.

**Running `analyze_profile.py`:**
```bash
# copy EVERY per-stage trace through the log-access pod (read-only) into the project folder, then analyze it
D=$AUTOTUNE_EXPERIMENTS_DIR/<run_id>/profiles; OUT=<project_folder>/traces/<run_id>; mkdir -p "$OUT"
for f in $(kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" exec "$AUTOTUNE_LOG_POD" -- ls "$D" \
           | grep '\.pt\.trace\.json\.gz$'); do
  kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" exec "$AUTOTUNE_LOG_POD" -- cat "$D/$f" \
    > "$OUT/$(basename "$f")"   # compare sizes with the source (ls -l)
done
python3 <this-skill-dir>/scripts/analyze_profile.py --profile-dir "$OUT"   # Python 3.10+
```
Run it with a 600000 ms Bash timeout or in the background: a cold parse of several ~0.5 GB traces
takes minutes (parallel workers, tens of GiB RSS); a rerun hits the parse cache in seconds. Check the
`coverage:` line before reading anything else. It must list every PP stage; `MISSING ppN` means the
classification below cannot see that stage. The `memory management` line gives each stage's
host-side allocator-cleanup time and calls, and `STAGE MEMORY-PRESSURE STALL` marks a stage far above
the others (levers.md §c).
`analyze_profile.py` reports the kernel-category breakdown (NCCL vs compute %), GPU util/bubble,
top kernels, and idle-gap causes — for both Megatron training and RL rollout traces. Add `--json`
to capture `profile_top_categories` for the ledger. It is descriptive only (no flag recommendations);
map the dominant category to a lever with §6 (kernel decoder) below + `levers.md` §c.

**NCCL kernel decoder** — the kernel *name* indicates which algorithm ran (RING/LL, NVLS, etc.);
confirm what each name means against the NCCL source for the version in your image before relying
on the semantics below (they are a working guide, not authoritative across NCCL versions):

| Kernel | What it likely is | How to shrink |
|---|---|---|
| `AllGather_RING_LL` | Distopt param reconstruct at forward start | `--overlap-param-gather` (hide behind compute); dominates without it |
| `ReduceScatter_Sum_bf16_RING_LL` | Distopt grad shard in backward | `--overlap-grad-reduce` (hide, structural). **NOT `--grad-reduce-in-bf16`** — it halves volume but is OBJECTIVE-CHANGING (nulls fp32 grad accum in `arguments.py` — the one framework-behavior claim here retained because it's the verified caveat behind a real regression); excluded from shipped recipes (levers.md; Invariants: objective-preserving) |
| `SendRecv` | PP stage boundary OR EP alltoall (token dispatch) | PP: reduce PP / add microbatches; EP: verify intranode (see levers.md §b) |
| `AllReduce_Sum_*` | TP grad sync (distopt off) | `--use-distributed-optimizer` → replaces with RS+AG pair |
| `*_NVLS_SIMPLE` / `*_NVLSTREE_SIMPLE` | Likely NVLink SHARP in-switch reduction | NVLS applicability (intranode-only vs multi-node AllReduce) and whether `NVLS_ENABLE` is a no-op when absent are NCCL-version-dependent — verify against NCCL source/docs rather than treating as fixed |

If one kernel is >50% of GPU time, the lever most likely to move the needle is the one targeting
that specific collective.

**When to extend profiling** (the `SLIME_PROFILE_*` switches go on with `make_run_yaml.py --env`): imbalance inside
a stage (TP/DP peers) → `SLIME_PROFILE_ALL_RANKS=1`; memory → `SLIME_PROFILE_MEMORY=1 --record-memory-history`;
kernel tuning → `SLIME_PROFILE_WITH_STACK=1`;
do not extend or shift the 5-19 ranking window after seeing results. A longer diagnostic is
unranked and recorded separately. **Skip profiling** only when the hypothesis is fully
tok/s-measurable and you already have a matching base-config profile.

**Bigger tools:** Perfetto (<https://ui.perfetto.dev>, practical for single traces up to ~100 MB);
HTA (Holistic Trace Analysis) for cross-rank/idle-time breakdown; Chrome tracing (same format).

## 7. Stale-checkpoint gotcha

If a run passes `--load` and `$AUTOTUNE_EXPERIMENTS_DIR/<run_id>/checkpoints/` exists from an earlier launch with the
same run_id, slime resumes at the latest step, **skipping the measured window and the profiler steps** and producing
no trace. Benchmark runs drop `--save/--load` (SKILL.md), and every launch gets a new run_id, so this cannot happen.
If a checkpoint directory must be removed, ask the operator; never delete files on the shared volume yourself.

## 8. Memory readings

**From Megatron stdout** (once per interval): `[Rank 0] (after 1 iterations) memory (MB) |
allocated: <A> | max allocated: <MA> | reserved: <R> | max reserved: <MR>`. Per-rank
max-allocated is the footprint; the tightest PP stage (usually last, holds the LM head) is the
bottleneck.

**From a profiled run:** the analyzer's `memory management` line gives each stage's allocator-cleanup
time and calls (§6); a stage far above the others is near its memory limit. Live readings inside the
training pods (`nvidia-smi`) need `kubectl exec` into them, which the agent does not run.

## 9. NCCL comm placement (verify EP/TP intranode)

With `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET`, grep for `NET/OFI Tuner init: comm with N ranks
and M nodes`. `M=1` = intranode (good for EP alltoall); `M>1` = cross-node (EFA). Map each "comm
with N ranks" line to its parallelism-group size to verify placement. Full method:
`print_topology.md`.

## 10. Python stack dumps for hangs (when no trace exists)

When a run hangs with no `perf N:` lines and the NCCL watchdog fires or TCPStore times out, no trace
was written. Two ways to see where the ranks are:
- **Flight Recorder:** relaunch with `TORCH_NCCL_DUMP_ON_TIMEOUT=1` and `TORCH_NCCL_TRACE_BUFFER_SIZE=20000`
  (`make_run_yaml.py --env`, both structural); the dump names the rank stuck on each collective. It fires only on
  the NCCL watchdog, not on c10d timeouts before NCCL init.
- **py-spy:** stack dumps need `kubectl exec` into the training pods, which the agent does not run. Ask the operator
  for `py-spy dump --pid <pid> --native` of every `ray::MegatronTrainRayActor.train` process on the head and worker
  pods, three snapshots 60 s apart, written to the shared volume (py-spy must be in the image).

Reading the snapshots: byte-identical stacks = true deadlock; shifting `libcuda.so`
PCs or flickering `clock_gettime` = slow, not hung. `"active"` thread = CPU running; `"idle"` =
blocked on mutex/syscall.
- c10d `TCPStore::doGet` waiting on ncclUniqueId broadcast — Flight Recorder can't see it (fires
  only on NCCL watchdog, not c10d pre-NCCL timeouts); py-spy is the only way.
- Ranks in Triton JIT `cuModuleLoadData` — looks hung, often just slow compilation; 3 snapshots tell.

**Pod IP ↔ node mapping is NOT guaranteed stable across runs** — with AWS VPC CNI a node's ENI
IP-pool can lead to IP reuse, but this is not a contract; do not assume "same pod IP = same
physical node" without verifying against current VPC-CNI behavior/docs. For node-specific hang
diagnosis, read the node directly from
`kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" get pods -o wide` (the `NODE` column) during the
run rather than inferring it from the IP.
