# Failure Triage

When a training run crashes, classify the failure *before* relaunching. Blind "try again with smaller batch" is almost always wrong.

## Triage procedure

1. **Pull the last 200 lines** of `train.log` from the shared experiments volume through `$AUTOTUNE_LOG_POD`
   (`kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" exec "$AUTOTUNE_LOG_POD" -- tail -200 <log path>`; `poll_run.sh` prints the path).
2. **Search for these patterns in order** — the first match wins.
3. **Classify** using the decision tree below.
4. **Write a ledger entry** with `status: "errored"` and the triage result.
5. **Propose a fix** — specific config diff, not a vague "retry."

If none of the patterns match, follow "What to do if triage doesn't match any pattern" at the end of this file.

## Pattern → classification → fix

### OOM (CUDA out of memory)

**Pattern:** `torch.cuda.OutOfMemoryError` or `CUDA out of memory`.

**Sub-classify by phase:**
- **Model load / weight placement OOM** (before any training step, during checkpoint load — look for "mlp.py: CUDA OutOfMemoryError encountered during tensors merging" repeating many times, or traceback during `dist_checkpointing`). The model itself doesn't fit. **Hard topology problem.**
  - Fix: raise PP to shard layers across more ranks; raise EP to shard experts further; or as a last resort `--optimizer-cpu-offload` (verify in the Megatron/slime source what cpu-offload actually relocates before assuming it will let the model fit — grep the offload flag; don't rely on a remembered scope).
  - Do not just "retry with smaller batch" — batch size doesn't change model-load memory.
- **Step 1 OOM** (first training step, not model load) → persistent memory + first peak transient (activations + AllGather buffers).
  - Fix: (1) drop `max-tokens-per-gpu`, (2) if already small, raise PP, (3) add `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`.
- **Step 2–5 OOM** → peak transient memory too big or variable-length spike.
  - Fix: (1) drop `max-tokens-per-gpu`, (2) swap `selective` → `full uniform` recompute.
- **Step 10+ OOM** → memory fragmentation or variable-seq-length pathology.
  - Fix: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` via env. If still fails, drop `max-tokens-per-gpu`.

**Always check which PP stage OOMed.** The last PP stage usually has LM head + vocab projection → tightest. If it's step 1 OOM on PP last-stage only, consider uneven PP (`--decoder-last-pipeline-num-layers`), and check that `--recompute-num-layers` divides every stage's layer count under uniform recompute (levers.md).

### NCCL communicator #104 timeout

**Pattern:** `DistBackendError: ... wait timeout ... keys: /default_pg/0//104//cuda//0`.

**Cause:** once observed to correlate with interleaved 1F1B at vpp_size > 2 per PP rank, or uneven PP via `--decoder-first-pipeline-num-layers`, on one older image and EFA generation. A `NCCL_PROTO=Simple` pin produced the same PP>1 comm-setup signature in that setup (below), the root cause was never confirmed, and later images ran uneven PP and vpp>2 cleanly. The mechanism (P2P pair count vs NCCL communicator pool, startup race) is unverified — verify against the P2P/comm-group setup in the Megatron source before treating it as fact.

**Fix:** first strip any `NCCL_PROTO` pin. If it still fails, revert to even PP with vpp_size ≤ 2 per rank for one run to test the correlation. Do not treat vpp>2 or uneven PP as categorically broken; both run on current images (levers.md, vpp and uneven-PP entries).

If vpp and PP are already even and you still hit #104, treat it as a novel comm-group pattern: record the exact config and try one more time with `TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_TRACE_BUFFER_SIZE=20000` (`make_run_yaml.py --env`) to capture a Flight Recorder trace. If that also fails, move on to a different topology in the queue; do not block the loop on diagnosing one crash.

### NCCL watchdog timeout on EXPERT_MODEL_PARALLEL_GROUP alltoall

**Pattern:** `DistBackendError ... PG GUID N(EXPERT_MODEL_PARALLEL_GROUP) ... ALLTOALL_BASE ... ran for 600018 milliseconds before timing out`. Training never reaches step 1 but gets past init (different from the store->get setup timeout).

**Cause:** EP group spans nodes and alltoall over EFA is hanging. Common when TP×CP×EP spans node boundaries. On 4×8-GPU nodes with TP=2 PP=1 EP=8, the EP=8 group contains ranks from all 4 nodes (not intranode as intuition might suggest).

**Fix:**
1. Check placement via `grep 'NCCL INFO.*comm with' train.log` — want EP groups to say `M=1 nodes`. If they span, either change topology or accept internode alltoall.
2. Options to get intranode EP: (a) PP=4 with EP ≤ 8 (each PP stage = 1 node, EP lives within stage); (b) reduce EP to the number of GPUs per node that contain a single DP replica of expert shards.
3. If internode EP is unavoidable, lower EP (e.g. EP=4) and accept experts being replicated, or capture a Flight Recorder trace (`TORCH_NCCL_DUMP_ON_TIMEOUT=1`) to tell a genuine hang from a very slow cross-node alltoall.

### EFA NIC count mismatch: `NET/OFI Unexpected number of remote rails`

**Pattern:** Training crashes at NCCL init (pre-step-1), before the first `perf N:` line. Log shows:
```
NET/OFI Unexpected number of remote rails for dev <N>. Expected <M> but got <K>
... NCCL WARN [Service thread] Error encountered progressing operation=Connect, res=3, closing connection
ncclInternalError: Internal check failed.
Last error: Socket recv failed while polling for opId=<hex>
```
…usually during `all_gather_object` inside Megatron optimizer-init. `poll_run.sh` reports `latest_step=-1`.

**Root cause:** the YAML's `vpc.amazonaws.com/efa: N` resource request **does not match the actual NIC count on the node**. NCCL's OFI-rdma plugin enumerates all NICs on the host but only gets handles for the requested count — peers on other nodes enumerate a different count → rail-discovery handshake fails. Not a soft fallback; process-group init dies.

**Correct values per instance type** (per the AWS EC2 instance specs — re-check for new instance types):

| Instance | Aggregate bandwidth | NIC count | `vpc.amazonaws.com/efa:` request |
|---|---|---|---|
| **p5e.48xlarge**  | 3.2 Tbps (EFAv2) | **32** × 100 Gbps | **`32`** |
| **p5en.48xlarge** | 3.2 Tbps (EFAv3) | **16** × 200 Gbps | **`16`** |

**Fix:** edit the YAML's `resources.limits.vpc.amazonaws.com/efa` and `resources.requests.vpc.amazonaws.com/efa` on BOTH head and worker containers to match the instance. Relaunch with a fresh run_id.

If some nodes advertise a different `allocatable` EFA count than the instance spec, report those nodes to the operator (or route around them) rather than changing the YAML.

**Verification before blaming anything else:**
```bash
# Node advertisement:
kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" get node <node-name> -o json | jq '.status.allocatable' | grep efa
```
The NIC count inside a running pod (`ls /sys/class/infiniband/ | wc -l` in a worker pod) needs `kubectl exec` into a
training pod; ask the operator for it. If the pod's NIC count doesn't match the YAML request, that's the bug — not a
slime / NCCL / Megatron issue.

**Why this used to look like a PP>1 bug:** Megatron's optimizer init calls `all_gather_object` across the full DP or EP group, which is the first inter-node collective after process-group setup. Bootstrap failure surfaces here. Historically attributed to `NCCL_PROTO=Simple` on EFAv2 (real, now fixed by stripping the pin — see "PP>1 hangs / init timeout traced to a pinned `NCCL_PROTO`" below) OR to NIC-count mismatch (this entry). Resolve the NIC mismatch first; revisit `NCCL_PROTO` if the hang persists.

### PP>1 hangs / init timeout traced to a pinned `NCCL_PROTO`

**Pattern:** PP>1 configs time out at init (comm #92/#104 `store->get`) or hang intra-step with GPUs at 0%. **Root cause seen in practice:** a base YAML inherited `NCCL_PROTO=Simple` from old AWS HPC recipes, which interacts badly with EFA for PP SendRecv. **Fix:** strip `NCCL_PROTO` (let NCCL auto-select) — see ledger.md anti-pattern #13. Don't re-add it.

### Intermittent NCCL init hang on AWS EFA (cross-node GDRDMA connect stall)

**Pattern:** a multi-node launch hangs during bring-up — model params may load, then SILENT: no checkpoint load, no step, and NO NCCL watchdog fire (it's before the first collective, so the process-group timeout isn't armed). The same YAML runs fine on other launches, including one launched at the same time on a disjoint node set ⇒ NOT a manifest bug.

**Diagnosis:** with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,BOOTSTRAP`, intranode P2P/IPC channels connect and ring/NVLS topology computes, then it stalls establishing a cross-node channel over EFA (a `NET/Libfabric/<n>/GDRDMA` `Channel …[receive]` line that never completes). A peer EFA endpoint's GDRDMA handshake wedges — a fabric/NIC-level transient, config-independent.

**Fix:** (1) relaunch on fresh nodes (usually clears; a scheduler like Kueue reassigns); (2) detect early — no checkpoint load and no step well past normal bring-up time ⇒ tear it down (`teardown_run.sh <run_id>`), don't wait for a watchdog that never fires; (3) report a recurring pattern to cluster ops (a persistently bad EFA NIC in the pool inflates the rate). Do NOT chase it in the YAML if the EFA env matches known-good runs (`FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`, `NCCL_SOCKET_IFNAME`, no `NCCL_PROTO` pin). Budget it as an external-infra retry (free; no ranked slot).

### Ray raylet crashes with `St12length_error: vector::reserve`

**Pattern:** log contains `(raylet) ... Unhandled exception: St12length_error` with `vector::reserve` on the stack. One or more actors become unavailable (`ActorUnavailableError: RPC error: Socket closed`). Training never reaches step 1.

**Observed pattern:** sporadic failure on cluster startup with many actors on the Ray version pinned in the training image. Non-deterministic; not caused by training config. No upstream issue is cited for it; treat "known Ray bug" as descriptive, not as a cited reference.

**Fix:** relaunch with a new `run_id`. If it recurs 3+ times in a row on the same config, it's not flaky — escalate to a human and consider a Ray version bump.


### NCCL comm setup timeout at init (any comm ID, including #92/#104)

**Pattern:** `DistBackendError: ... wait timeout after 600000ms, keys: /default_pg/0//<N>//cuda//0` during training start, before any `perf N:` line appears. The 600000ms = 10 min = default `--distributed-timeout-minutes`.

**Do NOT raise the timeout.** The default 10 minutes is already generous for comm group setup — healthy runs finish setup well within it. A timeout firing means a collective is stuck, not slow. Raising `--distributed-timeout-minutes` does not fix the hang; it only delays it and lets it burn cluster time.

**The timeout firing is the diagnostic signal, not the problem.** What it means:
- **One rank is in a different collective from the others** (typical cause) — e.g., one rank took a different code path, skipped a collective, or reached it with different group composition.
- **Network is genuinely failing** (rare) — check EFA health, try a different node set.
- **A new topology has a bug** — a config that worked before now hangs because something else was changed.

**Real fix: find out what changed.** Diff the config against the last known-good run — *every* flag, not just the "interesting" ones. Pay special attention to:
- `--num-layers-per-virtual-pipeline-stage` (vpp) — removing vpp is a big structural change; don't do it unless you have a specific reason
- PP/TP/EP topology — especially if any rank group ends up cross-node that was intranode before
- Recompute flags
- Overlap flags (`--overlap-param-gather` etc.)
- Environment changes on a new hardware type (EFA version, NVLink availability)

If the config truly matches a known-good baseline and it still hangs, enable Flight Recorder (`TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_TRACE_BUFFER_SIZE=20000`, set with `make_run_yaml.py --env`) and relaunch; the dump will show which rank is stuck on which collective.

### Assertion: config.no_sync_func must be None

**Pattern:** `AssertionError: config.no_sync_func must be None` — either on the second rollout, or at the first backward as `When overlap_grad_reduce is True, config.no_sync_func must be None; a custom no_sync_func is not supported when overlapping grad-reduce`.

**Cause:** the assert only triggers with `--overlap-grad-reduce` on. slime installs a custom `no_sync_func` for its multi-rollout loop; Megatron DDP rejects a custom `no_sync_func` under grad-reduce overlap. The second-rollout variant is the older form (the func set on the first call and not cleared); an idempotent-install patch fixes that variant only. Whether a given image tolerates overlap at all varies by build — inspect the image's Megatron source (`megatron/core/distributed/`) and slime's `no_sync_func` rather than assuming a tag carries a fix.

**Fix:** remove `--overlap-grad-reduce --overlap-param-gather` from the YAML (param-gather requires grad-reduce). Don't spend further slots on the overlap stack on that image (levers.md §d).

### Assertion: KeyError Float16Module

**Pattern:** `KeyError: Float16Module(...)` during `disable_forward_pre_hook` on re-entry.

**Observed cause:** when slime re-enters training for a second rollout, Megatron attempts to unregister a forward pre-hook that was never registered on the current `Float16Module` instance — a race between hook bookkeeping and module wrapping. An image that runs this path needs a guard that checks membership before removal; check the image's Megatron source (`megatron/core/distributed/`) for it.

**The guard is incomplete:** it protects the plain low-PP + intranode-EP + overlap-param-gather path, but **does NOT protect** when additional flags alter module wrapping. Known failure modes (all with overlap-param-gather, erroring at step 1):
- EP spanning more than one node.
- Several module-wrapping flags stacked (e.g. CUDA-graph + CE fusion together).
- PP≥4 with vpp — separate None-value variant below.

**Fix:** either (a) stay in a known-safe envelope (low PP + intranode EP + vpp∈{1,2}, small-win flags applied individually, not stacked), or (b) drop `--overlap-param-gather`. Dropping the flag removes forward pre-hook registration entirely, bypassing all assert variants at a small throughput cost. If you need a fix not yet in the base image, ask the operator for a patched image built FROM the baseline YAML's image (see the build section at the bottom) — never swap in a different fixed image.

### Assertion: remove_forward_pre_hook_handles[module] is not None  (PP+vpp variant)

**Pattern:** `AssertionError` in `megatron/core/distributed/distributed_data_parallel.py` — `assert self.remove_forward_pre_hook_handles[module] is not None`. Step 0 completes normally (single long compile), then the assert fires on re-entry into step 1 across all ranks.

**Trigger:** PP≥4 with vpp=2 + `--overlap-param-gather` + `--overlap-grad-reduce`. A PP=2 config with high vpp and the same overlap flags has run cleanly, so the difference is PP depth × vpp-per-stage. The KeyError-Float16Module guard covers the case where the *module key is missing* but NOT the case where the key is present with a `None` value (distinct code path in DDP).

**Cause hypothesis:** on PP>2 with vpp, there are multiple `model_chunk`s per rank, and each chunk's pre-hook handle bookkeeping is split. One chunk enters `disable_forward_pre_hook` with its handle slot already in the dict but unset (None) for this iteration — meaning the hook was never re-registered after the first training step ended. A membership-check guard returns early on missing keys, but lets None-valued keys fall through to the assert.

**Fix (immediate):** drop `--overlap-param-gather` from PP≥4-with-vpp configs. Keep `--overlap-grad-reduce` (no forward pre-hooks, not affected) if your image's `arguments.py` allows it without param-gather. The throughput cost is small, much less than a crash.

**Fix (durable):** extend the guard to also treat `handles[module] is None` as "nothing to remove" (no-op return), baked into a patched image (bottom of this file).

**Keep watching:** if a low-PP high-vpp config later hits the same assert at a different re-entry point, the diagnosis is wrong and it's a broader pre-hook bookkeeping issue, not PP-depth-specific.

### RayJob reports SUCCEEDED in <30 min (suspiciously fast) — verify it didn't crash at init

**Pattern:** `kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" get rayjob <run_id>` returns `jobStatus=SUCCEEDED, deploymentStatus=Complete` far faster than a real fixed-shape benchmark run takes on this recipe (a crash at init "completes" almost immediately). Driver pod is already gone, so logs must be pulled from the shared volume.

**Why this matters:** the Ray driver exits 0 whenever the Python entrypoint exits 0 — including when slime's training loop raises deep inside an actor and the error is swallowed during cluster teardown. The rayjob state says "SUCCEEDED" but `train.log` on the shared volume has a stack trace. `poll_run.sh` reports a SUCCEEDED run whose `latest_step` is below its target as `errored`, and one with no log as `done-unknown`; read the log before you trust any short SUCCEEDED run.

**Rule of thumb for triaging a short-duration SUCCEEDED:**
1. If run duration is far shorter than a normal run of this config (e.g. <30 min) → inspect the log before trusting.
2. `kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" exec "$AUTOTUNE_LOG_POD" -- tail -200 "$AUTOTUNE_EXPERIMENTS_DIR"/<run_id>/logs/<ts>/train.log`
3. Grep for `Error|Traceback|AssertionError|KeyError|TypeError|RuntimeError|OOM|RayTaskError` — short-duration SUCCEEDEDs virtually always have one.
4. If found, reclassify as `errored`, triage per the matching pattern below, and tear down (`teardown_run.sh <run_id>`).

### MoE expert-parallel overlap is incompatible with `recompute=full`

**Pattern:** `AssertionError: disable full recomputation when enabling overlap_moe_expert_parallel_comm`.

**Cause:** `--overlap-moe-expert-parallel-comm --delay-wgrad-compute` pipelines adjacent microbatches; full-layer recompute can't be reconstructed in that schedule.

**Fix:** you need *all three* of these together, not just the flag pair:
1. `--recompute-granularity selective`
2. `--recompute-modules moe_act layernorm mlp` (explicit list — **exclude `core_attn`**; observed that including it re-introduces the assert — verify the code path in the Megatron recompute/overlap source before assuming why)
3. `CUDA_DEVICE_MAX_CONNECTIONS=8` (set it in pod env, configmap, and runtime env to be safe — verify in slime source which of these it actually reads rather than assuming all three)

**Warning before even trying it:** this combo is memory-hungry because adjacent microbatches' activations co-exist; expect OOM at large mt. If your hypothesis is "unlock MoE overlap," pair the flag with higher PP OR a smaller mt — not just the recompute change. At PP>1 it also requires vpp (levers.md §b).

**Pre-flight check:** if your config has `recompute=full` AND you're adding `overlap-moe-expert-parallel-comm`, block the launch and require explicit recompute-modules to be set first.

### TE CUDA graphs also require selective recompute excluding core_attn

**Pattern:** `AssertionError` on init when `--cuda-graph-impl transformer_engine` is set with `recompute=full` (or `recompute=selective` without explicit `--recompute-modules`).

**Fix:** same trio as MoE overlap above:
1. `--recompute-granularity selective`
2. `--recompute-modules moe_act layernorm mlp` (exclude `core_attn` — the mechanism, "TE CUDA graphs capture attention and can't coexist with re-running it under autograd", is unverified; verify in the TE/CUDA-graph source before relying on it. The empirical action — exclude `core_attn` — stands.)

Note: `--cuda-graph-scope attn` (default implementation, not TE) was observed to work fine on `recompute=full`; the assert was specific to `--cuda-graph-impl transformer_engine`. Verify this compatibility split against the arg-validation source rather than treating the matrix as fixed.

### NCCL "unhandled cuda error" at init = GPU OOM, not a network issue

**Pattern:** `NCCL WARN Cuda failure 'out of memory'` / `NCCL WARN Failed to CUDA calloc <N> bytes` near the NCCL init sequence. Often preceded by a line like `mem_info: total_GB: <T>, free_GB: <near 0>, allocated_GB: <A>, reserved_GB: <close to T>`.

**Cause:** NCCL's cudaMalloc for its internal comm buffers fails because PyTorch has already reserved nearly all HBM for model weights + optimizer state. NCCL surfaces this as "unhandled cuda error" rather than OOM, which has historically led to false "network / EFA / NIC" diagnoses.

**Fix:** do NOT investigate NCCL config — treat exactly like a step-1 OOM:
1. First grep `Cuda failure 'out of memory'` and `Failed to CUDA calloc` in train.log to confirm.
2. Check `mem_info` line. `reserved_GB` close to the GPU's total HBM with `free_GB` near zero is a smoking gun.
3. Apply standard OOM remediation: add PP (shards model), reduce `max-tokens-per-gpu`, re-enable cpu-offload, or drop an unnecessary flag that consumes memory (FP8 scale tables, extra optimizer state).

**Common triggers:**
- Dropping EP without compensating PP (EP=16 → EP=8 doubles experts/rank → doubles MoE weight).
- Dropping `--optimizer-cpu-offload` at PP=1 with a large model.
- Enabling FP8 without reducing `max-tokens-per-gpu`.

### use-precision-aware-optimizer requires distributed optimizer

**Pattern:** `AssertionError: --use-precision-aware-optimizer only supported with distributed optimizer`.

**Cause:** `--use-distributed-optimizer` not set but `--use-precision-aware-optimizer` is.

**Fix:** add `--use-distributed-optimizer` explicitly to the config. (Some patched images relax this requirement.)

### Pickle / checkpoint load errors

**Pattern:** `UnpicklingError: pickle data was truncated` or `RuntimeError: ... checkpoint ... corrupt`.

**Cause:** incomplete checkpoint conversion, shared-filesystem inconsistency, or corrupted save during a prior crash.

**Fix:** (1) re-run the checkpoint conversion job, (2) or point `--ref-load` / `--load` at a known-good snapshot, (3) do not retry the same job with the same checkpoint path — you'll just fail again.

### Image pull backoff / pod scheduling

**Pattern:** pod stuck in `Init`, `ImagePullBackOff`, or `Pending` for >10 min.

**Cause:** infra — not a tuning failure.

**Fix:** check the events with `kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" describe pod <name>`. Common causes and their fixes:
1. **Registry credentials expired or missing on the nodes** → look for `no basic auth credentials`,
   `unauthorized` or `authorization token has expired` in pod events. Pull credentials belong to the cluster, so
   report the event to the operator and ask them to fix it; then tear the run down (`teardown_run.sh <run_id>`)
   and relaunch it once they confirm.
2. **Image registry rate-limiting** → wait a few minutes, pull retries automatically.
3. **Node-selector doesn't match available nodes** → verify capacity via `scripts/check_gpu_capacity.py` (in this skill).
4. **PVC not bound** → `kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" get pvc` to check the shared PVC state; report an unbound PVC to the operator.

### The step 10 is 1.5× slower than best — early kill

Not a failure per se — a proactive kill.

**Pattern:** run is training successfully but step-10 wall time is >1.5× the current best's step time.

**Fix:** `teardown_run.sh <run_id>`, update ledger with `status: "killed"`, record the prediction was wrong, and move on.

## Pattern: `ValueError: number of contiguous micro-batches in a virtual pipeline stage should range in [PP, M]`

**Observed:** arg-parse or first-step ValueError when the (bs, DP, mt) combination doesn't produce enough microbatches per rank for the interleaved-1F1B schedule.

**Cause:** Megatron's interleaved pipeline requires at least `PP` microbatches per DP rank per step. With dynamic batching:
```
samples_per_rank_per_step = global_batch_size / DP
microbatches_per_rank    ≈ samples_per_rank / (max_tokens_per_gpu / p50_seq_len)
```
If `microbatches_per_rank < PP`, the schedule can't fill the pipeline and Megatron refuses to start.

**Worked example (arithmetic):** world=32, `TP=1 PP=4 EP=8 vpp=2` at `bs=128`. DP = 32/(1·4·1) = 8, so 128/8 = 16 samples per rank. If `mt` packs ~7 short samples per microbatch, that is only ~2-3 microbatches per rank < PP=4. Fails.

**Fix options (pick whichever preserves throughput):**
- **Raise `--global-batch-size`** (more samples per rank). Cheapest if bs is flexible.
- **Lower `--max-tokens-per-gpu`** (more microbatches per rank, smaller kernels). Often worst — kills compute intensity.
- **Reduce PP** so the minimum-microbatch-count constraint relaxes.
- **Reduce DP** (by raising TP or EP, though those have their own constraints).

**Rule of thumb:** for Megatron interleaved-1F1B on short-seq data, need `(bs × p50_seq) / (DP × mt × PP) ≥ 1`. Run the math before trying low-bs + high-PP + big-mt combinations.

## Pattern: `AssertionError: --use-torch-fsdp2 is not supported with pipeline parallelism`

**Observed:** arg-parse-time AssertionError immediately at process start when `--use-torch-fsdp2` is set alongside `--pipeline-model-parallel-size > 1`. RayJob reports SUCCEEDED because driver exits 0 from the traceback capture, but `latest_step=-1`.

**Cause:** Megatron-LM's torch-FSDP2 path mutually excludes pipeline parallelism. This is a framework constraint, not a bug; asserted in `megatron/training/arguments.py`.

**Fix:** if you want FSDP2, you must drop to PP=1. On large MoE over EFA, PP=1 puts every layer on each rank and raises DP, which takes EP cross-node unless EDP > 1 holds it (levers.md §b), so FSDP2 is rarely viable. Retest only after PP=1 becomes viable on some new topology.

## Pattern: `ValueError: Not enough samples N for global_batch_size M` (slime SFT)

**Observed:** `ValueError: Not enough samples 256 for global_batch_size 1024` thrown from `slime/ray/rollout.py::_get_rollout_data` on step 1. Driver pod completes with `SUCCEEDED` but perf_lines_count=0.

**Cause:** `--rollout-batch-size` and `--global-batch-size` are independent flags. For SFT, rollout is tokenization/sample-loading only; if rollout produces fewer samples than the global batch requires, the trainer raises on step 1.

**Fix:** set `--rollout-batch-size` = `--global-batch-size` whenever you scale bs. If you only edit `--global-batch-size` (e.g. via `sed`), you'll ship this crash. Always grep both flags after YAML generation.

**Note:** this is a config-generation mistake, not a framework bug. Retry with matching values; no hypothesis cost.

## Pattern: `step_time_max` is many times `step_time_p50`, with `train_wait_time≈0`

A big step_time_max with a tight p50 looks like a rollout stall but usually isn't — it's JIT **compile-tail** (a new dynamic-batch microbatch shape triggering a one-time kernel recompile). The fixed-shape protocol permits startup compilation only in the discarded steps 0-4. A compile-tail signature in the measured window (steps 5-19 at the defaults) invalidates the run; diagnose an unexpected shape or runtime path instead of shifting the window or restoring a persistent cache. Distinguish stall vs compile-tail by reading `perf/train_wait_time`: it spikes = real rollout stall; it stays near zero while `actor_train_time` spikes = compile-tail. One-line check: `grep 'perf ' train.log | grep -oE "train_wait_time[=:] ?[0-9.]+" | sort -t: -k2 -n | tail`. Full diagnosis + the train_wait_time/actor_train_time table: `measurement.md`.

### `--overlap-param-gather` and `--overlap-grad-reduce` are coupled

**Pattern:** `AssertionError: Must use --overlap-param-gather with --overlap-grad-reduce` at Megatron init.

**Cause:** Megatron asserts both flags together. Can't ship one alone. Both also trip slime's `no_sync_func` assert (see "Assertion: config.no_sync_func must be None") unless the image carries a fix — and a runtime sed in `run-training.sh` (the entrypoint, which runs on the head pod) patches only the head pod's copy, not the worker pods' Ray actors.

**Fix:** ship both flags together, and only on an image whose Megatron/slime accept overlap with slime's `no_sync_func`. If it doesn't, drop both; a patched image (see "Building a patched image" at the bottom of this file) is rarely worth a marginal lever.

### `--manual-gc` loses throughput on slime regardless of interval

**Pattern:** a clear loss vs the same config without the flag, roughly independent of the GC interval.

**Cause:** `gc.disable()` interacts badly with the slime/Megatron CUDA caching allocator. Without refcycle GC, Python keeps cycles → memory pressure → slower allocator → slower steps. A small idle "bubble" that `analyze_profile.py` attributes to `ProfilerStep#N` is largely a profiler warmup artifact, not a real workload bubble; closing it with `--manual-gc` doesn't pay.

**Fix:** strike `--manual-gc` permanently on this stack. Don't trust idle-gap entries attributed to `ProfilerStep#N` as workload bubbles.

### Selective recompute OOM on hybrid linear-attention + many-expert MoE

**Pattern:** `torch.OutOfMemoryError` at step 1 in `chunk_gated_delta_rule` (file: `fla/ops/gated_delta_rule/chunk.py`). Allocator out of memory even with mt halved from a known-OK full-recompute config.

**Cause:** `--recompute-granularity selective` keeps linear-attention recurrent state + MoE expert activations live across forward. Hybrid linear attention combined with a many-expert, high-top-k MoE leaves too many activations live to fit even at small mt.

**Fix:** strike selective recompute on this model architecture at that GPU count. Use `full uniform`.

### TP=1 OOMs or stalls at the TP>1 leader's mt

**Pattern:** a TP=1 run OOMs (often on the last stage's fp32 logits buffer) or runs near the allocator-cleanup threshold at an mt that fits TP=2 comfortably.

**Cause:** TP=1 has no sequence parallel, so per-rank activations and the last stage's fp32 logits (`mt × vocab × 4 B / TP`) are unsharded: at the same mt they are TP times those of a TP>1 run.

**Fix:** rerun TP=1 at the TP>1 mt ÷ TP, or add PP to free memory. An OOM at the TP>1 leader's mt is not a TP=1 result; TP=1 can still be the winner at the right mt (levers.md §d).

### `--moe-pad-expert-input-to-capacity` deadlocks at first alltoall on slime/Megatron with many-expert EP

**Pattern:** `WorkNCCL ALLTOALL_BASE timeout 600s at SeqNum=2 (step 1)` on EXPERT_MODEL_PARALLEL_GROUP, independent of the capacity factor; `NumelIn` varies widely. Not a buffer-size issue.

**Cause:** fixed-shape pad-to-capacity dispatch in slime/Megatron's MoE path has deadlocked with NCCL/EFA on many-expert models. Dispatch metadata mismatch across ranks → infinite alltoall wait.

**Fix:** strike `--moe-pad-expert-input-to-capacity` on this stack. For MoE expert load imbalance, use `--moe-aux-loss-coeff` / `--moe-z-loss-coeff` (model-side, requires retraining for effect — not a perf-tuning flag).

**Workaround for measurement window:** if you observed straggler-driven bubble in an all-ranks profile, you have enough info; the lever to apply is at the model-architecture level (load balancing loss), not the dispatch-config level.

### `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a memory-relief lever, not a perf lever

**Pattern:** OOM at high mt is fixed by `expandable_segments`, but throughput drops noticeably.

**Cause:** VMM-backed segments have higher per-allocation overhead than the default caching allocator. `expandable_segments` defragments and lets bigger mt fit, but each subsequent compute step pays slower memory ops.

**Fix:** use `expandable_segments` ONLY when you have no alternative for memory. Don't use it as a knob to push mt past the OOM cliff for throughput — the cost outweighs the win.

## Building a patched image (when the baseline's image lacks a patch)

The training image must contain Megatron-LM and slime, pinned by digest, and for the levers you want to use:
slime's PyTorch profiler controls (for traces, SKILL.md gotchas), and, for the overlap stack, a Megatron/slime pair
that accepts `--overlap-grad-reduce` with slime's `no_sync_func` and guards the `Float16Module` pre-hook re-entry
(entries above). Check each in the image's source. Only if the baseline's image lacks a fix you need does anyone
build a patched image, and it must be built **FROM the exact image the baseline YAML references**, never by
swapping in some other fixed image. Ray actors load the image, not the head pod, so a runtime `sed` in
`run-training.sh` only patches the head; the patch must be baked in.

**Ask the operator before building or pushing any image**; they build it or approve your doing so. Either way:

1. **The base:** the baseline YAML's exact image reference, read off its `image:` line and pinned by digest
   (`crane digest <image>` or `skopeo inspect docker://<image>` resolves a tag).
2. **The patch:** a Dockerfile `FROM <that image>` that layers only the needed source changes.
3. **The tag:** a new, unique tag (e.g. `<registry>/<repo>:<username>-<short-desc>`); never push over an existing
   tag.

**Wire it in** once the image is pushed: set the new image's digest as the `image:` in the baseline YAML
(cloned runs inherit it; pin by digest, not tag). `make_run_yaml.py` does not rewrite image refs — this is a
one-time baseline edit, and a changed image makes a new baseline: rerun the baseline before ranking against it.

## What to do if triage doesn't match any pattern

1. Write the full error to the ledger under `what_i_learned` verbatim, including the last 50 lines of context from `train.log`.
2. Form a *best-guess* hypothesis based on what the error suggests (OOM → memory; timeout → network/deadlock; assertion → config mismatch). Don't skip this step — the hypothesis is the learning, even if wrong.
3. Try *one* reasoned fix based on the hypothesis, labeled clearly as a novel-failure response (include in the ledger entry's `next_experiment_suggestion`: "novel crash — trying X based on hypothesis Y").
4. If that fails too, record the pattern and what you learned in `RESULTS.md` (and, when it generalizes, as a lesson in `LESSONS.md`, SKILL.md L4(c); this file and `levers.md` stay read-only), and shift to a *different* experiment direction. Don't burn more attempts at the same crash.
5. Only escalate to "Blocked on human" in the project's `RESULTS.md` after you've accumulated **3 consecutive genuinely-novel crashes** that none of your reasoning unblocks. One crash is a puzzle; three is a pattern that needs human-level context.

The goal: **always have a next experiment to run**. A novel crash should produce both a ledger entry *and* a next experiment — never a stall.
