# Levers — Megatron / Slime training (depth reference)

This is the DEPTH reference behind SKILL.md. SKILL.md owns the loop and the objective-preserving Invariant; this file owns the
leverage/priority tiers and per-lever detail — it answers
**what each lever changes, when it helps/hurts, how it fails, and where the authoritative upstream
docs are.** Every rule is a mechanism plus the condition under which it was observed; magnitudes are
qualitative on purpose. The evidence behind a rule lives in a campaign's own `RESULTS.md` and ledger.

> **Objective-safety (SKILL.md Invariants):** a lever is objective-CHANGING until proven otherwise.
> Placement/scheduling changes (parallelism degree, sequence-parallel, cpu-offload, dtype-preserving
> overlap, recompute, fusion flags, microbatch packing/size, including under a per-microbatch MoE aux
> loss) are structural, and they still ship only through the final correctness check. cpu-offload is
> placement, but toggling it swaps the CPU optimizer for the fused GPU optimizer, so a small
> systematic loss offset is possible; the check decides. Kernel/backend swaps are NOT safe by
> construction (see `--attention-backend fused`, §d). Changes to the objective — `grad-reduce-in-bf16`, FP8,
> `global-batch-size`, capacity factor, aux-loss and the rest of the `objective` class of
> `scripts/correctness/rules/numerics_list.yaml` — never ship: layer 1 refuses them. Any other change touching
> gradient/optimizer/precision/loss/routing/batch **numerics** may ship ONLY after you (a) read its RESOLVED
> behavior in the running image's `arguments.py`/source (flags override each other and flip defaults across
> versions) and (b) pass the final correctness check (`scripts/correctness/PROTOCOL.md`). A passing loss check is
> necessary, not sufficient: `grad-reduce-in-bf16` can pass a loss A/B at rerun noise while disabling fp32 gradient
> accumulation in source, so it stays excluded on source grounds. Throughput guidance here says
> nothing about training quality.
>
> **Citation discipline:** a behavior claim here is TRUSTWORTHY only if it carries an explicit source
> (upstream file/function). Un-cited behavior notes are a *starting hypothesis to confirm against the
> image's source*, NOT ground truth — frameworks rename flags, flip defaults, and silently override
> each other between versions. Per SKILL.md loop step L2, grep the actual source before acting on any
> flag, and record the `file:line` you read in the ledger's `verified_from_source`.

Applies to both SFT and RL on this stack — most levers here are parallelism/packing/kernel knobs
that don't care whether a step's data came from a dataset (SFT) or a rollout (RL); where a lever
behaves differently under RL (e.g. rollout stalls, the generation phase), it's called out.
**Direction of a lever often does not transfer** across model size, GPU type/HBM, TP degree, or
sequence regime, and magnitude almost never does — every rule below is a prior to test with one run,
not a prediction.

## (a) Read these upstream sources FIRST

Frameworks ship every few weeks; 6-month-old blog posts may describe refactored behavior.
Form a working theory from these BEFORE launching (the `ETP × EP ≤ GPUs-per-node` rule in §b is
the kind of Tier-1 fact they surface that trial-and-error misses):

**Mandatory before any MoE tuning project:**
1. Megatron-LM MoE README + "Best Practices to achieve high performance on MoE training" —
   <https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe> — the
   upstream impact-tiered playbook.
2. Ultrascale Playbook (HF nanotron) — <https://huggingface.co/spaces/nanotron/ultrascale-playbook>
   — TP/PP/EP/CP mental model, PP-bubble math, alltoall topology rules, overlap mechanisms.
   Sanity-check framework: if a result disagrees with it, investigate before moving on.
3. Megatron-Core 0.16 parallelism guide —
   <https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/parallelism-guide.html>
   — official reference for every parallelism dim; version-qualify against the baseline image's
   Megatron version.
4. NeMo Megatron-Bridge parallelism guide —
   <https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html> — precise on
   `--expert-tensor-parallel-size` and `--num-layers-per-virtual-pipeline-stage` quirks + vpp.

**Model-recipe priors:**
- NVIDIA Megatron-Bridge model recipes —
  <https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/recipes> — find the
  exact or nearest-size model and GPU type; use its settings as topology, recompute, and kernel priors.
  Do not treat them as drop-in configs or throughput targets across different workloads, hardware, or
  versions: a public recipe for 80 GB-class GPUs often pins memory relief (cpu-offload, high recompute)
  that a larger-HBM GPU does not need (§d, CPU optimizer offload).

**Source-of-truth for flags/behavior:**
- `megatron/training/arguments.py` — every CLI flag, type, default, one-line help. Grep here
  before grepping the web.
- `megatron/core/transformer/transformer_config.py` — where flag-combination asserts live.
- `megatron/core/distributed/` — overlap/bucket/param-gather internals.
- Slime: `slime/backends/megatron_utils/` (how slime calls Megatron; flags may be gated),
  `slime/utils/profile_utils.py` (profiler controls), `slime/rollout/sft_rollout.py`
  (`--rollout-batch-size` vs `--global-batch-size`), `slime/utils/dp_schedule.py` (microbatch
  alignment under vpp, §d).
- Transformer Engine: `transformer_engine/pytorch/` (FP8 recipe, CUDA-graph impls).
- NCCL env-var reference — <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>
  — version-qualify. **Kernel names observed in a profile trace**
  (`ncclDevKernel_AllGather_RING_LL`, `*_NVLS_SIMPLE`) are ground truth for what algorithm actually
  ran (an observed artifact, not a code-behavior claim) — but confirm the *meaning* of a name against
  NCCL source for your version before acting on it.
- `CUDA_DEVICE_MAX_CONNECTIONS` is a CUDA env var; its stream-ordering effect is Megatron-specific.

**Principle:** version-qualify any claim that drives a decision. "Megatron distopt shards
optimizer state" is fine shorthand; the exact ZeRO tier ("pure ZeRO-1" vs "hybrid ZeRO-1+2") is a
structural claim that varies by version — **read `megatron/core/optimizer/distrib_optimizer.py` in
your image before relying on it**, don't quote a remembered tier. Cite `file:line` for structural behavior.

## (b) MoE Parallel Folding + the intranode-MoE rule (canonical)

**Megatron decouples attention-layer parallelism from MoE-layer parallelism.** Treating them
as a single 5D grid is wrong and leads to bad topology choices. They share only `PP`.

| Parallelism group | Attention layers | MoE (expert) layers |
|---|---|---|
| Dims | `TP × CP × DP × PP` | `ETP × EP × EDP × PP` |
| Shards | attention heads (Q/K/V), sequence, data-replication, pipeline stages | each expert's matmul, which experts per rank, expert-data-replication, pipeline stages |

**Controlling flags** (grep exact semantics/defaults in `arguments.py` before use):
`--tensor-model-parallel-size` (TP), `--pipeline-model-parallel-size` (PP, shared by attention
and MoE), `--context-parallel-size` (CP), `--expert-model-parallel-size` (EP),
`--expert-tensor-parallel-size` (ETP).

**Derived, not flags** — `DP`, `EDP`, experts-per-rank, and the world-size divisibility
constraint are all derivable from source. Compute them for your `(TP, CP, EP, PP, ETP)` and
`world_size` and verify against `arguments.py`/`parallel_state.py` rather than trusting a
remembered formula.

**The identity.** Both grids tile the same GPUs, so within each pipeline stage
`TP·CP·DP = ETP·EP·EDP` (equivalently `world = TP·PP·CP·DP = ETP·EP·EDP·PP`). The shorthand
`EP = TP·CP·DP` used below is the `ETP = EDP = 1` special case, **not** a general law — recompute for
your `(TP, CP, DP, ETP, EDP)` whenever ETP or EDP exceeds 1.

**Rank layout (default Megatron):** MoE layers fold as `[ETP][EP][EDP][PP]` (fastest-varying
first); attention layers fold as `[TP][CP][DP][PP]`. Rank `r` → node `r // GPUs-per-node`
under Ray/k8s co-located scheduling. *(Source: `megatron/core/parallel_state.py` — a separate
`expert_decoder_rank_generator` builds EP/ETP groups with divisor
`expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size`,
which is what makes EP ranks contiguous and grounds the `ETP×EP ≤ GPUs-per-node` intranode rule.
Fold `order` can change across versions — re-grep + confirm per-project with `print_topology.md`.)*

**Why folding matters (practical):** decoupling lets `EP` exceed `DP` (on 8-GPU nodes with
`DP=4` you can still run `EP=8` intranode) and lets attention keep a useful `TP` while MoE runs
`ETP=1`. Confirm the resolved group sizes for your config via `print_topology.md`, not by
assuming the mechanism.

**Coupling on this stack:** with the default folding and `ETP = EDP = 1`, `world = TP·PP·CP·DP` and
`EP = TP·CP·DP`, so **`EP = world/PP`**. Moving one axis forces another: raising PP lowers EP (and DP);
lowering TP raises DP. Reason about the coupling before ruling an axis out, then test it.

**Coupled moves that keep per-rank memory.** Expert weight bytes per rank scale as experts/(EP·ETP) × layers/PP,
so a fold that raises PP and lowers EP by the same factor (constant `EP·PP`) leaves them unchanged; an OOM
after such a fold comes from somewhere else (a heavier first stage, more in-flight activations, the
last-stage logits), not from "more experts per rank". Likewise lowering TP and raising PP by the same factor
keeps per-rank static memory (full width × fewer layers). Prose memory reasoning on exactly these moves is often
wrong and closes productive regions; do the arithmetic before ruling a fold out.
A fold to deeper PP with smaller EP at constant `EP·PP` fit where the smaller EP alone had OOMed at the
shallower PP, and it won only together with a larger mt and full recompute, which the freed memory paid
for; on its own it lost. Evaluate such a fold with its coupled partner, not one factor at a time.
On a single node, folding EP into PP trades the alltoall for a bubble: a deep fold that removed the
alltoall entirely won large on one memory-bound benchmark, and a shallow fold won large where it let the
model drop recompute (the single-stage layout had needed recompute to fit), while a shallow fold with no
memory to free lost or tied. Test the fold's endpoint, not only its first step.

**Folding × PP-divisibility lock.** Because `PP = world/EP` at `ETP = 1`, EP and PP are inverse-locked
at a fixed GPU count, and with even PP the stage count must divide the layer count. That quantizes the
reachable EP values: for some layer counts the intranode EP you want has no even-PP split at your GPU
count. Escapes, in order: (1) **uneven PP** via `--decoder-first/last-pipeline-num-layers` (supported on
current images; §d), (2) `ETP > 1`, which raises the fold denominator (`EP = TP·CP·DP/ETP`); the GPU count
itself is fixed. Check this lock before planning an intranode-EP
experiment, and include uneven splits in the candidate set (they are also a throughput lever in their own
right, §d "Uneven pipeline splits").

### THE RULE: EP alltoall is intranode iff `ETP × EP ≤ GPUs-per-node`

Independent of TP and PP (grounded in the contiguous EP-group layout cited above). If the
product fits one node the alltoall rides NVLink/NVSwitch; otherwise it crosses nodes via EFA.

- **The old `TP × EP` / `TP × CP × EP ≤ 8` mnemonic is WRONG** — it happens to give the right
  answer when `ETP=1` but for the wrong reason. Use `ETP × EP ≤ GPUs-per-node`.
- Keeping the EP group on NVLink is usually the single biggest topology win: intranode NVSwitch
  bandwidth ≫ inter-node EFA, and EFA adds per-message setup overhead that hurts small messages.
  Instances with the same GPU can differ in EFA geometry (NIC count × per-NIC rate) at the same
  aggregate bandwidth; check vendor specs rather than quoting a ratio. (Re-diagnose PP>1 failures
  from NCCL INFO logs, not from an assumed NIC-provisioning shortfall.)
- **Climbing PP to reach intranode EP can beat the added bubble.** Because `EP = world/PP`, a
  higher PP can fold a cross-node EP group into one node; on EFA that trade has won even when the
  PP bubble share rose substantially. Confirm with the profile (EP SendRecv share should drop).

**Placement examples (8 GPUs per node — arithmetic, not results):**

| Config | `ETP × EP` | EP group fits one node? | Implication |
|---|---|---|---|
| ETP=1 EP=8 (any TP, any PP) | 8 | **Yes** | Intranode alltoall. |
| ETP=1 EP=16 | 16 | No — 2 nodes/group | Cross-node alltoall; expect it to dominate comm. |
| ETP=2 EP=8 | 16 | No — 2 nodes/group | Cross-node alltoall; adds tensor-parallel comm per expert without compensating compute. |
| ETP=1 EP=4 | 4 | Yes | Doubles per-rank expert weights vs EP=8 — OOM risk on many-expert models; can fit on larger-HBM GPUs with TP1 and enough PP. |

**Consequence:** with `ETP=1` and `EP ≤ GPUs-per-node`, EP is intranode at *any* valid PP, so the
choice between PP values is NOT "which has intranode EP"; it's bubble vs DP-AllReduce placement
(see the PP entries in §d).

**When EP MUST be cross-node** (expert memory forces EP beyond one node), the alltoall dominates
and is hard to fix: such configs run comm-bound, with EP alltoall plus PP boundaries taking about half
of GPU kernel time, a small GEMM share, and low MFU. The direct attacks and their prerequisites:
- **`--overlap-moe-expert-parallel-comm --delay-wgrad-compute`** (merges FWD/BWD of adjacent
  microbatches to hide the A2A). Requirements asserted in `transformer_config.py`: EP>1;
  alltoall/flex dispatcher; bf16; **NOT full recompute** (selective OK — method/num_layers must be
  None); grouped-gemm (non-legacy) for delay_wgrad; and at PP>1 **`virtual_pipeline_model_parallel_size`
  MUST be set** (`AssertionError: virtual_pipeline_model_parallel_size must be specified`). So any
  VPP blocker (§d vpp) also blocks this lever. The Hopper path restricts it when
  `CUDA_DEVICE_MAX_CONNECTIONS==1` (which SP requires); on Blackwell (arch≥10) that restriction is
  skipped. Memory-hungry: adjacent microbatches' activations co-exist (see `failure_triage.md`). It
  has little field evidence on this stack; treat it as speculative and measure it.
- **Comm-overlap can HURT when compute is small.** When the a2a is cross-node-bound and GEMM is a
  small share of kernel time, there is little compute to hide behind and the reschedule/sync cost
  can exceed the savings (`--moe-shared-expert-overlap` lost on such a config). Measure; don't assume
  overlap helps.
- **DeepEP (`--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep`)** —
  requirements: TP×EP>1; `deep_ep` installed; NOT `moe_pad_expert_input_to_capacity`. No VPP and no
  recompute change — a clean single-variable swap from `alltoall` (routes same tokens → same
  experts; keep FP8 dispatch off for a dtype-preserving comparison, still checked at the end). The
  variable-sequence-length restriction applies to the `allgather` dispatcher, not flex.
  - **Cross-node DeepEP is infeasible on AWS EFA.** It needs NVSHMEM IBGDA (GPU-initiated IB RDMA)
    or IBRC over IB verbs; EFA is libfabric-based and provides neither. Signatures: `Buffer.__init__`
    fails with `nvshmemi_transport_init: init failed for transport: IBGDA` / "Peer GPU not
    accessible", then `ibv_poll_cq failed`, then ActorUnavailableError across the cluster; or every
    NIC logs `device rdmap* is not supported (expected HCA interface: mlx5)` and NCCL rendezvous
    collapses (`DistBackendError ... store->get got 0 bytes`). It failed this way on every EFA
    generation tried. Don't retry on EFA; it is an InfiniBand-cluster lever.
  - **Intranode DeepEP DOES work on EFA** when the EP group fits one node (`ETP×EP ≤ GPUs-per-node`).
    Its value is workload- and hardware-dependent (§d, Speculative).
- ⇒ On EFA, a forced-cross-node EP alltoall can be genuinely irreducible (overlap needs VPP, DeepEP
  needs IB, intranode EP needs a PP that may not fit at the fixed GPU count, or that host-OOMs under
  cpu-offload). Treat that comm floor as fixed and optimize the rest: the memory-minimal topology that
  keeps it, the least recompute that fits (selective where memory allows; full where it does not),
  and the best mt.

**Verify placement** via `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` — look for
`comm with N ranks and M nodes` (want `M=1` for EP groups), or the source-reading method in
`print_topology.md`. (The world-size divisibility constraint is derivable — compute and confirm
against source, don't quote it.)

**Cross-node EP hazard:** `TP>1` + cross-node EP has hung on the alltoall watchdog on one EFA
generation while the same topology completed on another. Root cause undiagnosed (EFA
version/config, NCCL version, TP↔EP collective concurrency). If a cross-node-EP config hangs
repeatedly, try a different instance pool before blaming the config.

## (c) Profile → bottleneck classification

Classify every ranked run before choosing the next lever: from its trace when it has one (SKILL.md
L4b), otherwise from its log scalars (below). **Don't skip
this — jumping to "try FP8" or "add recompute" without the bottleneck class is the #1 cause
of wasted runs.** Read every pipeline stage's trace (the default, one per stage); a stall or a straggler
local to one stage is invisible from the others. Cheap scalars (microbatch count, the analytical bubble,
per-stage free memory, the step-time split) are often as decisive as a trace.

| Profile signal | Bottleneck class | Where to go |
|---|---|---|
| NCCL >50%, AllGather-dominated | comm-bound (param gather) | overlap-param-gather / reduce DP |
| NCCL >50%, ReduceScatter/AllReduce-dominated | comm-bound (grad reduce) | overlap-grad-reduce (structural); grad-reduce-in-bf16 is objective-class and refused (§d) |
| NCCL >50%, SendRecv-dominated | comm-bound (PP or EP alltoall) | topology: reduce PP / verify EP intranode |
| PP>1 and GPU idle ≥~10% | possibly bubble-bound (PP schedule / stragglers) | first compare the idle to the analytical bubble (below) and look for a straggler stage (next row); if neither explains it, topology: PP↓, or vpp (if the microbatch-alignment rule in §d allows); bs↑ only if bs is unconstrained |
| PP>1 and one stage is the critical stage: its work time clearly above the other stages', which spend that difference waiting on PP p2p (analyzer `critical stage` and per-stage work / p2p-wait lines), with no allocator stall on it | stage compute imbalance (often the last stage, which also runs the output layer and loss; this is not the schedule bubble) | uneven split moving layers off the critical stage onto stages with slack; see "Uneven pipeline splits" (Tier topology) |
| Compute >50%, elementwise/norm significant | memory-bandwidth-bound | precision / fusions |
| Compute >50%, GEMM-dominant | compute-bound | precision (FP8 if GEMMs large) |
| OOM / near-OOM | memory-bound | recompute / offload / mt↓ |
| One PP stage spends far more host time in allocator cleanup (`clear_memory`-style helper, `empty_cache`, `gc.collect`, cudaFree) than the others; the analyzer prints `STAGE MEMORY-PRESSURE STALL` | stage memory-pressure stall (cleanup fires on that stage because its device free memory sits near the framework's threshold; the other stages then show it as PP p2p wait) | relieve that stage: uneven split moving layers off it (`--decoder-first/last-pipeline-num-layers`), or more recompute there; see "Stage memory-pressure stall" (Tier topology) |
| GPU busy low, idle spread over many short gaps, a very high kernel-launch count per step, kernel time well below step time | host-launch-bound (many small kernels), or dependency stall between small kernels | fewer, larger kernels: lower TP (TP>1 with SP splits every GEMM and adds collectives per layer), larger microbatches (mt); confirm the launch reading with one test (it held where lower TP doubled GPU busy; it did not where CUDA graphs left the idle unchanged) |
| Time inside `step_time` but outside `actor_train_time`, or one long idle gap per step at the optimizer step with the host CPU busy | CPU-offload bounce (the CPU optimizer step) or rollout stall | disable cpu-offload or lower its fraction; check rollout_time / dataloader |

**Read call count AND time together.** A comm kernel that dominates by time with a high call count
is overhead-bound (bigger microbatches amortize it); one that dominates with an already-low call
count is volume/bandwidth-bound (needs fewer bytes, a faster path, or intranode placement).

**Idle % alone does not separate bubble from comm-wait, and a low idle % does not mean the bubble
is intrinsic.** A PP>1 leader with a modest idle share still gained from vpp once it was tried;
conversely, about half the step idle on a cpu-offload config was comm-wait plus the offload stall
(changing PP and GPU count did not move it), not bubble. With EP-a2a or AllGather dominant, lowering
PP or adding vpp won't fix it. Test vpp directly when PP>1 rather than gating on an idle threshold.

**Check a mechanism with the run's own numbers before writing it down.** A wrong mechanism often has the
right sign and fails a one-line check:
- *Bubble:* the 1F1B bubble share is about `(p−1)/(v·m + p−1)` for `p` stages, `v` virtual stages and
  `m` microbatches per DP rank (on fixed rows, `m` = GBS / (DP × rows per microbatch)). Fit to one run, it
  predicted later interleave and microbatch changes within a few percent. Idle or p2p wait far above it
  is something else: a straggler stage, an offload stall, or comm-wait.
- *Memory:* per-stage free memory and allocator cleanup counts (memory lines, or the trace's memory
  management line), and the last stage's fp32 logits buffer of about `mt × vocab × 4 B / TP` per
  microbatch (failing allocations have matched it exactly).
- *Microbatches:* count them. A step of very many one-row microbatches is overhead-bound, whatever else
  the profile suggests.
- *Step split:* the microbatch loop versus the post-loop tail (optimizer step, offload copies,
  distributed-optimizer collectives). DP and mt shrink the loop; the tail did not shrink with DP, and
  only some offload changes moved it. Some per-step costs belong to the frozen data path (rollout wait,
  host-device copies of the batch); subtract them before bounding what a lever can recover.

**SendRecv time is not wire time.** When the per-call SendRecv time grows far faster than the bytes, or
NCCL time grows while the GPU idle share stays low, ranks are waiting inside the receive on a stalled peer
(a stage under memory pressure, a straggler). Find that stage before treating it as a bandwidth problem.

## (d) Lever catalog (organized by SKILL.md priority tiers)

Comm-bound large-MoE SFT typically runs at MFU far below dense-training bands regardless of tuning;
dense "healthy MFU" numbers are not a target here.

### Tier: topology (highest leverage, hardest to get right, most risk)

**Reduce PP when PP SendRecv dominates.**
- When: PP SendRecv >40%, bubble ratio `(stages−1)/microbatches > 0.3`.
- How: change `--pipeline-model-parallel-size` (and drop the vpp flag if going to PP=1); keep
  world-size divisibility (compute it, don't quote a formula).
- Hurts/failure mode: lowering PP raises EP and DP (`EP = world/PP`), which can push the EP group
  cross-node and shift cost to DP AllReduce / expert AllGather; PP=1 also puts all layers' memory
  on one rank. A PP decrease that pushed EP cross-node has lost about half the throughput. Check
  after: SendRecv should shrink; if AllGather/alltoall grew more than SendRecv shrank, **revert**.
- **Lower vs higher PP (both with intranode EP) is a bs/workload-regime choice, no uniquely-optimal PP:**
  - Lower PP: larger DP, smaller bubble. Tends to win when steps are long and rollout variance matters.
  - Higher PP: smaller DP (cheaper DP AllReduce), lower per-GPU activation memory, more
    microbatches/rank at high bs — better compute efficiency, and can win on short sequences on
    smaller-HBM GPUs. But in slime SFT it amplifies rollout-stall variance (larger `step_time_max`),
    which can erase the compute gain in wall-clock. For pretrain / stable RL without rollout stalls,
    prefer it.
  - Direction is workload/hardware-specific — sweep PP neighbors, don't assume.

**Increase PP when the all-layers rank is memory-pressured.**
- When: PP=1 OOMs even with full recompute, or activations dominate memory; or a lower TP (e.g.
  TP1) needs more layer-sharding to fit.
- How: raise PP; drop `--max-tokens-per-gpu` if needed; keep microbatch count high by raising
  `--global-batch-size` only if bs is not fixed by the benchmark contract.
- Hurts: more cross-node SendRecv hops; bubble worsens if microbatches don't grow proportionally;
  compile time at step 0 grows with stage count (see "Compile tail" below).

**TP — valid values and direction.**
- Constraint: `transformer_config.py` raises when `(num_query_groups % TP != 0) AND
  (TP % num_query_groups != 0)`. So TP is valid whenever it is a **multiple OR divisor** of
  `num_query_groups` — TP above the KV-head count is legal (Megatron replicates KV heads across the
  extra TP ranks: redundant KV compute, more attention-TP comm). Such TPs are extra valid (TP, DP)
  splits at the same EP (`EP = world/PP` at `ETP = 1`), so they widen the candidate set around an
  intranode-EP layout without moving EP. Look up the ACTUAL
  `--num-query-groups` in your baseline YAML / model `config.json`; don't trust remembered values.
  Attention with a per-head output gate needs a framework build that slices the gate per TP rank once
  TP exceeds the KV-head count; check the image's attention code before launching such a TP. The same
  holds for any per-head attention parameter outside the QKV and output projections (learnable sink or
  softmax-offset logits): where the framework does not split it across TP ranks, TP>1 fails at checkpoint
  load with a global shape mismatch. Megatron also asserts `CUDA_DEVICE_MAX_CONNECTIONS=1` for TP>1 with
  sequence parallelism, which an inherited recipe may set otherwise. Attention variants (sinks, sliding
  windows) can also lack context-parallel support; check before planning CP>1.
- How: `--tensor-model-parallel-size N`, paired with `--sequence-parallel` (verify the SP
  interaction in `arguments.py` before use).
- **TP1 removes all tensor-parallel comm** and often wins on comm-bound, narrow-expert MoE once
  CE-fusion removes the vocab-logits peak (the usual reason baselines pin TP2). Mechanics: TP1
  holds unsharded non-expert params/grads, so it may only fit with more PP (full-width × fewer
  layers ≈ same per-rank memory as TP2 at half the PP). Reach for TP1 + just-enough-PP-to-fit
  before assuming TP>1; the PP that wins may be optimal only because it is the smallest that fits TP1.
  "Fits" means headroom above the framework's allocator-cleanup threshold, not merely no OOM: the
  smallest PP that fit TP1 has run near that threshold and stalled, and the next PP up removed the stall.
  Check the loss path before counting on CE fusion: on one slime SFT loss path the loss was computed
  outside Megatron's fused CE, the full fp32 logits buffer stayed, and TP1 OOMed on exactly that buffer;
  on another the fp32 cast of the output ran before the CE (in the fp16 module wrapper) and slime cloned
  the logits for log-probs, so no CE implementation relieved it.
- **TP1 has no sequence parallel**, so activation memory per `mt` is unsharded: its mt ceiling is
  roughly the TP>1 mt ceiling ÷ TP, and it can go bimodal near the memory cliff
  (`failure_triage.md`). TP1 at high PP can also have pathological step-0 compile time (full hidden
  per GPU, SP off) — avoid that pairing.
- **Higher TP** shards attention and frees HBM (which can unlock bigger mt), but adds attention-TP
  AllGather; past the sweet spot it loses. Whether TP1 or a higher TP wins is hardware-dependent:
  on larger-HBM GPUs TP1 has won because comm dominates and attention-TP AllGather is pure overhead,
  while on 80 GB-class GPUs a TP above the KV-head count has won large on its own and freed the HBM
  that a bigger mt then used, with the next TP step losing to attention-TP comm. Re-measure TP direction per GPU type and workload.
  Two cautions. (1) A TP-up win can be a last-stage memory-pressure stall at the lower TP in disguise:
  relieving that stall with a lighter last stage at the lower TP has beaten the TP increase by a wide margin. Before crediting a TP increase, check the lower-TP run's per-stage allocator
  cleanup ("Stage memory-pressure stall"). (2) TP and mt are coupled (TP>1 with SP shards activations, so
  it admits a larger mt): compare TP values each at its own best mt. A TP>1 point measured at the TP1
  leader's mt understates it, and a one-factor-at-a-time walk can close a high-mt TP>1 region that a joint
  TP × mt search finds.
- Don't reach for TP>1 for the experts when the expert MLP is narrow — overhead dominates. Wide-expert
  MoEs may benefit — measure.

**EP — set placement deliberately** (see §b for the rule and coupling). Prefer
`ETP × EP ≤ GPUs-per-node` so the EP group fits one node. **ETP>1 is rarely a win at long fixed
sequences**: it trades PP bubble for expert AllGather and raises experts/rank memory pressure;
prefer `ETP=1` unless the profile says otherwise. Lever direction for ETP/PP folding is
regime-dependent (an ETP2 fold that won on short sequences lost on fixed-8K for the same model) —
don't inherit a short-seq winner. A model with biased MoE linear layers runs only at ETP=1: Megatron asserts
"Bias in Moe is only supported when ETP==1" (`transformer_config.py`), so ETP is not an axis there.

**CP (context parallelism).**
- When: sequences >4K common AND activation-memory-bound. How: `--context-parallel-size N`
  with `--sequence-parallel`. Won't help short-seq — ring-P2P overhead eats the benefit.
- **On comm-bound MoE with full recompute, REDUCE CP toward CP1** (raising DP to hold world/EP
  fixed). When activation memory is already handled by recompute, CP's only effect is added CP
  all-gather comm (expensive over EFA); lowering CP directly removes `--allgather-cp` volume, and each
  halving of CP has improved tok/s/GPU on such configs, even at 8K. CP only pays off when you're
  genuinely activation-bound and can't recompute your way out. Compare CP values at matched mt: a CP
  gap measured at different mt is an mt confound. At matched high mt, CP degrees have also tied: once
  mt amortizes the per-call cost the CP gap shrinks, so a CP win carried from another model or GPU is a prior to
  re-measure, not a target. Without activation pressure, CP>1 has lost clearly (ring KV exchange plus
  half-size kernels).

**vpp (virtual pipeline) is gated by per-rank microbatch count and layer divisibility, not by global batch size.**
- **Alignment rule (source):** slime's `dp_schedule.py` pads each step's microbatch count to a
  multiple of `DP × microbatch_group_size_per_vp_stage` when vpp>1, and Megatron defaults that group
  size to PP (`model_parallel_config.py`); read both in your image. On fixed-length rows at
  `mt = seq_len/CP` (1 seq/microbatch) the per-rank count is `m = GBS/DP` and cannot be split further,
  so **vpp works iff `m % PP == 0`** (at `mt = 2·seq_len/CP`, `m = GBS/(2·DP)`). A misaligned config
  fails at batch setup with a "needs N microbatches, only M available" alignment error; it is not a
  vpp/dynamic-batching incompatibility. `m = PP` (the minimum) works.
- When aligned, vpp has paid from a few percent up to mid-teens percent over the same config without
  it, including at the benchmark GBS under dynamic batching; at high PP with a large `m` the gain can
  shrink to a tie. On variable-length data the same rule shows up as vpp hurting at small GBS and
  helping at large GBS (microbatches per rank, not GBS itself).
- **Depth saturates.** The first interleave step carries most of the gain; the next has added a few
  percent or tied, and deeper still has tied or regressed (more chunk-boundary p2p calls, more launches).
  On one setup an odd depth dipped below both even neighbours. Sweep a few depths once vpp pays; stop at the peak.
  At a high per-rank microbatch count the bubble is already small, and vpp's extra chunking can lose.
- **vpp multiplies PP p2p messages and per-chunk kernel launches, and adds in-flight activations.** With
  PP boundaries crossing nodes and many small microbatches, it lost heavily (a quarter to two thirds) while
  the bubble it could recover was a few percent; on memory-tight stages it OOMed or tipped a stage into the
  cleanup stall. Compute the recoverable bubble (§c) before launching vpp.
- **Layer divisibility:** with `--num-virtual-stages-per-pipeline-rank N`, every stage's layer count
  (including `--decoder-first/last-pipeline-num-layers`) must be divisible by N. For other splits use
  an explicit `--pipeline-model-parallel-layout`, which cannot be combined with the decoder-first/last
  flags.
- **Step 0 can be very long** with vpp: an aligned vpp run killed inside step 0 with per-microbatch
  time still falling was never diagnosed as slow compile or a stall, and the same configuration later
  completed at several interleave depths on a newer stack. Budget step-0 time before
  killing. Other hazard: a
  `ReloadableProcessGroup` pickle bug at very high vpp on PP=2 (`failure_triage.md`).
- **vpp + DSA IndexShare (GLM-5.x):** index-share requires every pipeline-stage boundary to
  start on a *computing* layer, and vpp's virtual-chunk boundaries must too. The auto-split
  (`--num-virtual-stages-per-pipeline-rank N`) ignores this and asserts `DSA index-share pipeline
  split is invalid … top-k sharing can't cross PP boundaries`. A DSA model needs a hand
  `--pipeline-model-parallel-layout` with every chunk boundary on a computing layer (forcing uneven
  chunks, losing some bubble benefit); that layout is untested here. The assert is a correctness
  guardrail — it fails closed. Without it, a DSA model's bubble levers are microbatch count
  (GBS/DP/mt) and PP degree.

**Uneven pipeline splits (`--decoder-first/last-pipeline-num-layers`, or an explicit
`--pipeline-model-parallel-layout`).** Whenever PP > 1, splits belong in the PP candidate set. They have
three distinct uses; keep them apart when reading a result:
1. *Reachability:* PP degrees whose stage count does not divide the layer count become usable, which also
   opens EP folds (§b). "PP must divide the layer count" is false whenever the first/last flags are
   available; a deeper uneven PP ruled out this way can be among the best layouts.
2. *Memory relief:* moving layers off a stage that OOMs or sits near the framework's cleanup threshold
   ("Stage memory-pressure stall", next entry).
3. *Compute balance:* a throughput lever even when every stage fits. Under 1F1B the step is paced by the
   slowest stage. The end stages carry work beyond their decoder layers: the last runs the output layer
   and the loss, the first the embedding (cheap in compute, but that stage holds the most in-flight
   activations). With equal layer counts the last stage is often the critical stage, and the other
   stages spend the difference waiting on it in PP p2p.
   - *Detect:* a per-stage profile in which one stage's work time is clearly above the others' and the
     others show the difference as p2p wait (the analyzer's `critical stage` line, its per-stage work /
     p2p-wait table, and a top contrast giving one stage's compute share well above its peers'), with no
     allocator stall on that stage. Reading this as "bubble" is the common misfile: the schedule bubble is
     the analytical one (§c), a straggler is one stage. Without a trace, test a split on each side of the
     current one.
   - *Size:* the gain follows the end stage's extra work relative to one stage's share of layers. It
     mattered most with few layers per stage (deep PP) and an output layer that is large next to a decoder
     layer (large vocabulary, low TP): there the balanced split won high single digits over the even split,
     and a straggler split lost double digits against the balanced one. With tens of layers per stage,
     one-layer moves were within noise. Where it mattered the output layer was worth a few decoder layers;
     take that from per-stage work in a profile, not from parameter or FLOP counts.
   - *Both directions:* moving too many layers off an end stage makes the middle stages critical and
     loses; making the end stages heavier loses more. The optimum is a narrow plateau: neighbouring splits
     on it tied, and one more layer moved onto the middle stages or onto the first stage lost about a
     tenth. The balanced split can look uneven: the best splits have given the end stages one to a
     few layers fewer than the middle stages, never more off the first than off the last. The first stage
     is not free either: with its embedding it became the critical stage once a layer was added to it. At
     a deeper PP, heavier end stages collapsed throughput (mechanism not traced) while lighter ones won.
     Measure a split on each side before closing the axis; a single run in the wrong direction (layers
     moved onto an end stage) is not evidence against the lever.
   - *Granularity:* with the first/last flags the remaining layers are spread evenly over the other
     stages (`megatron/core/transformer/transformer_block.py`; setting only one flag spreads the rest over
     every other stage, the other end included), so a whole-layer move off one end lands on every middle
     stage at once and can overshoot. An explicit `--pipeline-model-parallel-layout` sets any per-stage
     (and per-chunk) count and cannot be combined with the first/last flags.
     `--account-for-embedding-in-pipeline-split` / `--account-for-loss-in-pipeline-split` count the
     embedding or the loss as one more layer in an even split; they need the enlarged layer count to
     divide by PP and do not combine with the first/last flags (`arguments.py`; verify in your image).
   - *Interactions:* under vpp every stage's (and chunk's) layer count must be divisible by the depth, and
     under uniform recompute by `--recompute-num-layers`; a split plus either may need an explicit
     layout. With PP2 and both end stages set there is no middle stage, and vpp then fails
     (`transformer_layer.py`, division by zero). Pick the stages that receive the moved layers by their
     work and free memory; the first stage is usually the tightest in memory. A move that pushed the
     receiving stages to their memory edge collapsed throughput by a large factor, not by a few percent
     (consistent with the allocator stall; not traced). The `--account-for-*-pipeline-split` divisibility
     assert says nothing about the first/last flags; do not read it as "uneven splits unsupported".
   - *Judge* it by the critical stage's work falling toward the others' and by tok/s/GPU. The balance
     point moves with TP, PP and mt: a split that won at one layout lost a few percent when stacked on a
     leader at another, so re-check it after any topology change.

Current images run uneven splits with and without vpp. Older image/fabric combinations once hung vpp>2
and uneven PP at NCCL init; the root cause was never confirmed (an `NCCL_PROTO=Simple` pin produced the
same signature) and it did not reproduce later. Treat it as a hazard to recognize, not a blocker.
Constraint: under uniform recompute, `--recompute-num-layers` must divide every stage's layer count
(Tier packing/recompute below).

**Stage memory-pressure stall → uneven split or more recompute on that stage.** Some training
frameworks guard low-level collective/p2p calls with a cleanup: when device free memory drops below a
fixed threshold they run `torch.cuda.synchronize()` + `gc.collect()` + `torch.cuda.empty_cache()`
before the call. The stage with the least free memory (often the last, which holds the output layer
and loss, or the first, which holds the embedding and the most in-flight activations under 1F1B) pays
that stall on every call, many times per step; the GPU idles meanwhile and every other stage waits
on it, so the stall shows up on the other stages as PP p2p wait, not as host time. It does not look
like OOM: peak allocated can sit well below device memory, because NCCL buffers and workspaces live
outside the allocator. Diagnose it from a per-stage profile (one trace per PP stage; a rank-0 trace
cannot see it): the analyzer's memory-management line shows that stage's cleanup time and call count
far above the others. Removing such a stall multiplied throughput on the configuration where it was
diagnosed. Fix it by relieving memory on that stage; a split helps here through the memory it frees,
and compute balance is the separate case in "Uneven pipeline splits" above:
- Move layers off it: `--decoder-last-pipeline-num-layers N` (last stage) and/or
  `--decoder-first-pipeline-num-layers N` (first stage), below `L/PP`. The middle stages share the rest,
  so `(L - first - last) % (PP - 2) == 0` must hold (Megatron raises a divisibility error otherwise).
  The winning split can look compute-imbalanced; judge it by the stall disappearing and tok/s/GPU.
- Or recompute more: go from selective/partial to full recompute, or, under full uniform recompute,
  raise `--recompute-num-layers` (fewer stored chunk inputs) subject to the divisibility rule below.
- Verify: re-profile per stage; the flagged stage's cleanup calls should drop to about zero. The same
  split can lose at another TP/PP layout where no stage is under pressure, so re-check after topology
  changes.
- A leader whose tightest stage ends its step only a little above the threshold is fragile: any change
  that adds memory there (a larger mt, a heavier last stage) can tip it into the stall and collapse
  throughput. Read that stage's free memory before stacking such a change.

**GPU count is fixed** at the recipe's own count (SKILL.md Invariants) and is not a lever: a tuned recipe runs on
the same GPUs as its baseline.

### Tier: microbatch packing / recompute (medium leverage, cheap to try)

**miles with `--qkv-format bshd` (required for DeepSeek-V4) has no dynamic batching**: miles asserts
`--use-dynamic-batch-size` off for bshd (miles/utils/arguments.py, v0.1.0). The packing lever is then
`--micro-batch-size k` (k fixed-length rows per microbatch); `mt` advice below applies to thd only.

**`--use-dynamic-batch-size --max-tokens-per-gpu N` is the real fixed-bs throughput knob.**
Packs variable-length samples into tight microbatches; raising `mt` amortizes fixed per-step
comm at the SAME bs. Start low (8K) and raise if memory allows; ceiling depends on recompute
granularity, PP depth, cpu-offload, TP/SP, and seq distribution — measure.
- Packing check: samples/microbatch ≈ `mt / mean_seq_len`. If `bs × mean_seq ≪ DP × mt`,
  packing is well-fed and `mt` can rise; if `≫`, microbatches already saturated — look upstream
  (topology/overlap) instead of more `mt`.
- **Fixed-length (8K ranking) workload → mt is a STEP function, not smooth.** Every sample is
  exactly `seq_len` tokens, and a sequence can't be fractionally packed (nor split without CP), so
  `samples/microbatch = floor(mt / (seq_len / CP))`, with at least one. At **CP1** every `mt` below
  `2·seq_len` gives **1 seq/microbatch**, so those values tie. To actually cut microbatch count you
  must cross a multiple of `seq_len/CP`; the only informative points are `mt = k·(seq_len/CP)`.
  Crossing UP is a real trade-off: bigger activation/microbatch (OOM risk) AND fewer microbatches
  can enlarge the 1F1B bubble. The optimum is not monotonic "bigger is better", and the same mt step
  has won on one model and lost badly on another on the same hardware.
- **mt is a call-count amortizer, NOT a volume fix.** Bigger microbatches reduce the NUMBER of
  EP-alltoall (and other per-microbatch) calls but not the bytes moved (alltoall volume = tokens ×
  hidden). Its gain comes from per-call overhead and better GEMM shapes, so it is biggest when
  microbatches are tiny (it has roughly doubled throughput there) and **saturates** once per-call
  overhead is small (a several-fold cut in call count then ties). If a comm kernel dominates by time
  with an already-low call count, mt↑ is done — pivot to a volume/bandwidth lever (intranode
  placement, DeepEP-intranode, fewer bytes).
- **mt direction is TP-dependent.** Higher TP (with SP) can sustain larger mt and may prefer it;
  TP1 often peaks at the smallest informative mt and OOMs or thrashes above it. Verify the mt
  direction per TP; don't carry it across the TP boundary.
- **mt interacts with the a2a cost.** With plain `alltoall` the same mt↑ lost; once the a2a was cheap
  (intranode DeepEP), it paid. Treat the interaction as measured and its mechanism as unverified (the
  loss fit a stage memory cliff better than the bubble model). Retest a lever that lost alone after a
  change that removes the reason it lost.
- **Past the peak, mt falls off a cliff, not a slope.** The analytical bubble grows only modestly as
  microbatches get fewer, so a sharp drop one mt step up points to memory: the tightest stage (usually
  the last, which also holds the fp32 logits of `mt × vocab × 4 B / TP`) reaching allocator cleanup or
  thrash. Check per-stage free memory before blaming the bubble. Throughput can also fall just before
  OOM (allocator pressure at the edge); back off to the last robust point.
- Bigger `mt` → fewer, fatter microbatches: use when matmul share is small and memory has headroom.
  Failure mode: OOM on the tightest PP stage (last stage holds LM head) — watch the last stage's
  memory lines in the log. On short-seq, mt-up can stall or hang at the rollout→train handoff at high
  PP. Not a free knob.
- Smaller `mt` → more microbatches: use when PP-bubble-bound and you can't raise bs further.
  Risk: smaller matmul kernels hurt compute MFU.
- **More than one row per microbatch at PP>1 keeps the pipeline shape exchange.** slime exchanges tensor
  shapes before every PP send/recv (a blocking send/recv plus a device sync per p2p call). Fixed p2p buffer
  shapes are possible only with one row per microbatch, so packing more rows keeps that exchange, which is
  part of the cost of mt↑ at PP>1. One packed run with an uneven split also desynced and hung.
- Slime already packs by default. If disabled, enable. Verify the packing path in
  `slime/backends/megatron_utils/` before assuming the mechanism.

**`--micro-batch-size` is the dominant lever on a memory-bound MoE without dynamic batching.**
Throughput rises roughly linearly with micro-batch size, because grad-accumulation microbatches
shrink in number and GEMMs grow, until the memory ceiling; the optimization reduces to "raise mbs to
the ceiling". Recompute, EP and TP matter there mainly through how much activation headroom they free
for a larger mbs (below). These MoE memory rules do not carry to a dense model: applying them to one
lost clearly against a pipeline split with recompute off.

**`--global-batch-size` is workload-coupled, NOT a default throughput knob.** It controls
gradient noise / convergence — sweeping it changes *what you're training* (objective-CHANGING;
see SKILL.md). The benchmark holds it fixed at the ranking standard (GBS=128; SKILL.md Invariants) — don't sweep it as a throughput knob. Only sweep when the user explicitly
provides candidates or says bs is unconstrained; then mark results "bs-coupled" so a bs winner
isn't promoted without convergence sign-off (throughput gaps across bs mostly measure bs-coupled
work growth, not a free knob). Raising bs also adds microbatches to cut PP bubble (use when PP>1
and bubble ratio >0.25); pair with `--rollout-batch-size` for SFT.

**Selective recompute** (`--recompute-granularity selective`; drop `--recompute-method` and
`--recompute-num-layers`).
- When: matmul is a meaningful share and memory has headroom. Public writeups report gains over
  full uniform (MoE FFN not recomputed) — treat as a prior, not a promise. On a memory-capped
  cross-node-EP MoE, full→selective has been the largest single lever once it fit; with ample headroom
  it has paid a mid-teens percent. It has also fit and lost (an offloaded, comm-bound configuration), so
  measure it.
- Recompute and mt compete for the same HBM, and TP sets how much there is: selective fit at a higher TP
  and stopped fitting when TP was halved (SP had been sharding the activations); at the lower TP the
  smaller mt that selective forced lost more than the recompute saved. Compare at each TP's best mt.
- Failure mode: selective holds MORE activations live than full → can OOM where full fits; the same
  model can fit selective at one GPU count and need full at another. Rule of thumb: drop `mt` ~25%
  concurrently. Static memory estimators generally do NOT model the recompute-granularity axis —
  don't trust an "it fits" estimate when the only change is `recompute-granularity`; dry-run with a
  real launch.
- **Many-expert, high-top-k MoE and hybrid linear-attention models** can OOM under selective at every
  topology where full fits: too many expert-intermediate / recurrent-state activations stay live,
  and full uniform compresses them. On such models default to `full uniform` and treat selective as
  a memory-headroom experiment (`failure_triage.md`).
- **Selective recompute has modules (`--recompute-modules`).** With the default (core attention) it frees
  little under flash or fused attention, which store little; `moe_act` frees the large expert
  intermediate at a modest cost and was load-bearing for memory on a many-expert model (dropping it OOMed);
  `layernorm` is cheap and frees little; the whole-`moe` module recomputes the expert path. Pick the
  modules by what dominates memory, and read the accepted values in `arguments.py`.
- For MoE memory, selective with the default modules ≈ no recompute (it recomputes attention; expert-FFN activations
  dominate), so on a memory-bound MoE full recompute is a memory enabler, not a throughput cost to
  remove: dropping it forces a lower microbatch size and loses.

**Full recompute (memory relief, always fits):**
`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`. Adds recompute
FLOPs (verify the cost model in source if it matters). On an MoE it re-runs the whole expert path in
backward, dispatch alltoall and grouped GEMMs included: where the step was compute-heavy, full recompute
alone cost about a sixth and full recompute paired with a doubled mt doubled the step. Its cost hid only
where the step had launch idle to absorb it (many tiny microbatches): there full recompute plus a doubled
mt won a few percent, and one mt step further, with the idle gone, the same pair lost a quarter or more. Raising `--recompute-num-layers` has usually cost
a few percent, and once gained a few percent over its parent on a no-offload leader (it was never
stacked): measure it rather than assume the sign. Across a wide sweep it stayed within a few percent until
one chunk spanned most of a stage, where it collapsed. Recompute granularity within full uniform is otherwise
close to throughput-neutral. Block/partial recompute (keeping some layers un-recomputed) has paid a few
percent where HBM headroom existed, and OOMs or thrashes once mbs or mt sits at the full-recompute ceiling.
- **Divisibility (uniform method).** Under `--recompute-method uniform`, every pipeline stage's local
  layer count (each virtual chunk under VPP) must be divisible by `--recompute-num-layers`. Megatron's
  uniform loop checkpoints chunks `[l, l + n)` without clamping, so a stage with `k % n != 0` layers
  indexes one past its last layer: `IndexError: index k is out of range` in `transformer_block._get_layer`
  on the first forward (k = that stage's layer count). With an uneven split, check the first, last and
  middle stages separately; `(L / PP) % n == 0` is the wrong check once first/last are set. Validate
  before launch; this failure costs a full launch and teaches nothing about throughput.
- **Full recompute is load-bearing when cpu-offload is dropped.** Dropping offload puts the optimizer
  on the GPU and consumes the HBM headroom that full recompute frees, so on a no-offload leader that
  sits near the HBM limit, full→selective OOMs. You cannot both drop offload and reduce recompute
  there; recompute reduction is available only with offload on, where it usually loses to no-offload.

### Tier: overlap / kernel flags (mixed impact — reshape schedule, not work)

No fixed order: which of these matters depends on which collective dominates the profile.

**`--grad-reduce-in-bf16` — OBJECTIVE-CHANGING. Excluded from shipped recipes, even after
a passing loss check.** It halves DP grad-reduce volume, so it helps only where DP
ReduceScatter/AllReduce is a meaningful share of the step; where grad-reduce is small, the bf16
conversion overhead can net negative. It was a large lever on variable-length data at small GBS
where grad-reduce dominated; at the fixed-8K benchmark GBS it is mostly a tie, with a few percent
either way depending on topology, and it does not transfer — re-measure at the topology winner.
**It is NOT a wire-dtype-only change** — it is easy to mistake it for one and call it "training-neutral".
In Megatron `megatron/training/arguments.py` (`validate_args`):
```
if args.grad_reduce_in_bf16:
    args.accumulate_allreduce_grads_in_fp32 = False   # force-nulls fp32 accumulation
```
So setting it makes gradients **accumulate across microbatches AND all-reduce in bf16** — and it
**silently overrides `--accumulate-allreduce-grads-in-fp32`** even if you also pass that flag (the
fp32-accum flag becomes a no-op). Megatron's own comment says bf16 training expects
accumulation+all-reduce in fp32; this departs from that. Low-precision accumulation loses small
gradient contributions (worse with more microbatches) → can degrade convergence. Loss A/Bs of
100 steps have passed at rerun noise while a stricter exact-repeat A/B failed: the loss check is too
insensitive to catch it, so the source classification is the guard. This is the canonical example of
why a flag's *effective* behavior must be read from source, not memory.

**`--moe-shared-expert-overlap` — situational; often net-negative.** Overlapping one small shared
expert with dispatch can cost more in reschedule/sync than it hides, especially at TP1 or when the
a2a is cross-node-dominant: removing it from an inherited recipe has tied or helped, and adding it to
a cross-node-EP config lost, as did adding it to an intranode-EP leader. It is alltoall-dispatcher-only
(a `ValueError` with flex). Verify per-topology; grep its behavior/preconditions in source before use.

**`--cross-entropy-loss-fusion --cross-entropy-fusion-impl te` — small, additive.** The `native` vs
`te` impl choice is throughput-neutral. Verify what it fuses / whether it removes the vocab peak in
source before relying on it enabling TP1: the flag only matters if the training loss goes through
Megatron's CE (TP entry above).

**`--moe-permute-fusion` / `--moe-grouped-gemm` — cheap; often already default.** Verify in the
profile whether they're visible top kernels before/after; if already fused, turning on is a no-op.
Removing permute fusion from a recipe that had it lost clearly; keep it. Turning either off is not a
same-math swap: unfused, the top-k combine sums expert outputs in bf16 and its error grew across layers
(a one-layer test could not see it), and without grouped GEMM the experts ran a path that clamped the
activation before adding the expert bias, which moved the loss on a model with biased, clamped experts.

**`--overlap-grad-reduce` + `--overlap-param-gather` — treat as a pair.** `--overlap-param-gather`
requires `--overlap-grad-reduce` (asserted in `arguments.py`), and `--align-param-gather`
auto-disables on a non-interleaved schedule. Where they ran, the pair was marginal (a tie); an
earlier "overlap stack" gain came mostly from grad-reduce-in-bf16 bundled with it, so isolate them.
On slime the pair is frequently **unavailable**:
- **slime's custom `no_sync_func`:** Megatron DDP asserts `When overlap_grad_reduce is True,
  config.no_sync_func must be None; a custom no_sync_func is not supported when overlapping
  grad-reduce` at first backward. slime installs a custom `no_sync_func` for its multi-rollout loop,
  so on builds without an accommodating patch the whole overlap stack is unusable regardless of
  offload. Check the image once (grep the assert + slime's `no_sync_func`); if it's there, don't
  spend a slot. Do not assume a newer image carries the patch. A runtime patch that makes slime's `train()`
  re-entrant (the fix proposed in slime PR #1857) clears that assert; with it loaded, the pair ran without
  offload and tied, because the exposed DP comm it could hide was small.
- **With CPU optimizer offload:** `--overlap-grad-reduce` + `--optimizer-cpu-offload` throws
  `AssertionError: len(self.per_param_grad_ready_counts) == len(self.params)` at first backward.
  Whenever cpu-offload is required, the distopt param-gather AllGather is an inherent, un-hideable cost.
- Where the overlap stack is blocked, the distopt AllGather/ReduceScatter is a fixed cost of the
  recipe; reduce it with DP (fewer shards to gather across), not with flags.
- Older builds also needed the `Float16Module` guard for PP>1+vpp step-1 re-entry
  (`failure_triage.md`). If your baseline image lacks the needed patches, an adhoc image is rarely
  worth a marginal lever.

**`--tp-comm-overlap` fails on EFA** with TransformerEngine userbuffers: `initialize_ub` aborts with
`Intra-node group must be on the same backend as the world group` (`comm_gemm_overlap.cpp`) when the
intra-node UB group backend differs from the NCCL world backend. Together with the distopt-overlap
block above, TP-comm and distopt overlap can both be unavailable on a slime + EFA stack.

**Usually throughput-neutral (low priority):** aux-loss coeff (a load-balance knob; its throughput
effect depends on dispatcher — raising it lost under `alltoall` and tied under flex), CE-fusion
te-vs-native, `--defer-embedding-wgrad-compute` (can OOM), `--moe-router-fusion`, `--moe-router-dtype`
(fp32 vs fp64 tie; bf16 was rejected — check accepted values in `arguments.py`), `--ddp-bucket-size`,
`CUDA_DEVICE_MAX_CONNECTIONS` (a tie, and a small loss at TP1), the data padding
multiple, slime's update-weights interval, high-priority stream settings. `--manual-gc` hurts on slime
(`failure_triage.md`). Not usable on this stack: `--overlap-p2p-communication-warmup-flush`
(PP p2p deadlock at the first step), `--ddp-average-in-collective` (incompatible with
`--calculate-per-token-loss`), `--moe-apply-probs-on-input` (needs top-k 1).

**Flag toggles rarely pay at all.** Fusion, kernel, dtype, overlap and padding toggles mostly produce ties
and a few losses; the large gains come from structural changes (topology, offload, packing, recompute
granularity, interleave, pipeline split). Before spending a launch on a toggle, bound its gain by the
share of step time its kernel category takes (a lever cannot remove more than that share).

**Speculative / research-grade (exhaust everything above first):**
- **DeepEP dispatcher, intranode** (`--moe-token-dispatcher-type flex` with
  `--moe-flex-dispatcher-backend deepep`, or `--moe-enable-deepep` on builds that use that flag; check
  `arguments.py`; cross-node is infeasible on EFA, §b). It cheapens the EP a2a, but it pays only
  where EP a2a is the *residual* bottleneck: on a PP-SendRecv-bound leader it ties. Its value has ranged from a large win when stacked with a larger
  mt (the cheap a2a let bigger microbatches pay), through a tie, to a large loss on the newest NVLink
  generation where DeepEP's kernels were slower than NCCL alltoall; it also lost on the previous
  generation at large microbatches, where NCCL alltoall was already efficient. DeepEP reserves SMs for
  its kernels, which compete with the GEMMs when compute is the larger share. A win did not transfer
  across GPU types or TP, and on some stacks the DeepEP path has segfaulted; measure per stack. Its
  correctness is shape-dependent too: one DeepEP build returned wrong values or hit an illegal memory
  access at one model's hidden size while exact at others, so a dispatcher swap needs a component test at
  the model's own shapes. (The `allgather` dispatcher is not an alternative on slime: `validate_args`
  rewrites it to `alltoall`, so the two run the same code and tie.)
- MoE EP comm overlap (`--overlap-moe-expert-parallel-comm` + `--delay-wgrad-compute`) —
  requires vpp at PP>1 and is memory-budget-incompatible in most attempts (see §b and
  `failure_triage.md`). Public writeups report a sizable share of EP alltoall recoverable; rarely fits.
- CUDA graphs (`--cuda-graph-scope attn`, `--cuda-graph-impl transformer_engine`) — small when it
  works. Attention-scope and whole-layer capture both ran and tied on an MoE whose step was half
  fine-grained idle (the idle was dependency stall, not CPU launch latency); the MoE scope needs
  drop-padding (capacity-bounded, token-dropping, objective-changing) routing, and the legacy graph flag with EP asserts
  on expert padding; recompute constraints in `failure_triage.md`.
- **`--attention-backend fused` (cuDNN) — numeric: it may ship, and the verdict lists it for review.** It has
  produced NaN grads on low-GQA + softmax-in-fp32, and where it ran cleanly and faster, its loss drifted
  upward late in the run, beyond rerun noise. The 20-step check cannot see a late drift: cite this evidence
  in the review, and confirm with a longer loss run before production training.
- **Plausible, not yet measured:** an uneven split stacked on a TP1 leader; a hand pipeline layout whose
  stage and chunk boundaries satisfy a cross-layer constraint (DSA index-share), which would unlock vpp and
  EP comm overlap there; `--overlap-moe-expert-parallel-comm` with vpp on a model without such constraints;
  the coupled TP>1 × high-mt × vpp region with structural levers only; a higher `--recompute-num-layers`
  at a no-offload leader.

### Tier: precision

**FP8** (`--fp8 hybrid` + TE env; grep co-flag requirements in `arguments.py`).
Objective-CHANGING for training math — needs user sign-off on convergence before shipping.
- When: baseline is genuinely compute-bound (matmul >25%), GEMMs large enough to saturate
  tensor cores, TE healthy. FP8 tensor cores roughly double bf16 throughput only on *large* GEMMs.
- **Narrow-hidden MoE does NOT benefit:** casting/setup overhead dominates small GEMMs (FP8 lost
  clearly against bf16 on a narrow-hidden, narrow-expert MoE), and FP8-blockwise can OOM. NVIDIA
  guidance: hidden ≥4096 for meaningful FP8 wins.
- **FP8 can lose on a grouped-GEMM MoE at small micro-batch size:** amax/scaling overhead outweighs
  the GEMM speedup, and FP8's extra buffers lower the mbs that fits, pushing it below the bf16
  winner's operating point.
- Failure mode: setup/casting overhead dominates at small microbatches; accuracy deltas;
  recompute-mode and amax-history interactions — verify against the image's TE/Megatron source.

**`--use-precision-aware-optimizer` — keep it.** It affects throughput, not just FP8 paths;
removing it from a recipe that has it has regressed clearly. Requires `--use-distributed-optimizer`.

**bf16 Adam state** (`--exp-avg-dtype bf16 --exp-avg-sq-dtype bf16`): mostly memory-saving;
throughput-neutral unless memory-bound. Objective class — layer 1 refuses it. It has
also OOMed on a temporary fp32 upcast of the state and wedged one run late in its window, so it is not
a reliable memory saver either.

**Expert load balance** (quality/stability knobs — objective-CHANGING, and a hot expert stalls
the whole EP alltoall): `--moe-aux-loss-coeff` (0.001-0.01 SFT range from the Switch Transformer and
GShard papers — a starting point), `--moe-router-z-loss-coeff` (~1e-3). Log per-expert token counts every
50-100 steps; <~1.5× mean is a common heuristic target.

**Expert capacity factor / token drop** (`--moe-expert-capacity-factor` + `--moe-token-drop-policy`)
— OBJECTIVE, refused by layer 1. Bounding expert-FFN compute gives a real throughput gain
(roughly a tenth at cap ≈ 1.0, more at lower caps), and the drop *policy* (probs vs position) is
throughput-neutral; the bound is what pays, and the gain saturates above cap ≈ 1.0. It drops tokens:
at cap 1.0 it has failed a loss-parity check by two orders of magnitude over rerun noise, one-sided.
Verify its token-dropping behavior in source; report it as a separate objective-changing win, never in the
headline.

### Tier: optimizer / memory relief

**Distributed optimizer (ZeRO-style) — mandatory for large models:** `--use-distributed-optimizer`
(verify the default in your image's `arguments.py`). Once on, the lever is *how many ranks you
AllGather across* — reduce DP, don't drop the flag. (Sharding scheme / exact ZeRO tier is
version-dependent — read `distrib_optimizer.py`; see the Principle note in §a.)
- **Per-GPU optimizer state ≈ total optimizer bytes / world size, independent of EP**: the distributed
  optimizer shards over the whole data-parallel world; EP changes weight/grad/activation replication,
  not the per-GPU optimizer share. A higher GPU count is therefore what shrinks it. PP does not change it
  either: raising PP halves the parameters per rank as DP halves. Reasoning that "PP>1 shrinks DP, so the
  optimizer shard per rank grows" is wrong and can close a winning PP degree.

**CPU optimizer offload — last-resort memory relief:** `--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d`. The CPU↔GPU bounce is a large per-step cost (it can leave the GPU
idle for a large share of the step). Only when the optimizer state won't fit otherwise. If the
profile shows time inside `step_time` but outside `actor_train_time`, cpu-offload is the suspect —
disable and see if it fits on-GPU.
- **Dropping cpu-offload is often the dominant lever when an inherited recipe targets smaller-HBM
  GPUs** (public recipes for 80 GB parts commonly pin it). On a larger-HBM GPU the distributed
  optimizer may fit on-GPU, making offload pure overhead; removing it has given gains that dwarfed
  every topology lever and collapsed GPU idle. **When porting any cpu-offload recipe to a bigger-HBM
  GPU, try dropping offload before topology tuning.**
- **The fit condition is model size vs HBM × world size, not the GPU alone.** Estimate on-GPU
  optimizer bytes/GPU ≈ total fp32 optimizer state / world, plus params, grads, and activations.
  For very large MoE, offload can stay mandatory at every reachable topology. On a small-world MoE where optimizer state would crowd out
  activations, offload with overlapped copies can be cheap (once per global batch) and is mandatory.
- **After a no-offload win** the profile typically flips to comm-bound (EP alltoall), so the next lever
  is intranode EP.
- **No-offload and intranode EP can look mutually exclusive** (no-offload wants many optimizer shards;
  intranode EP at a fixed count forces thin DP), but the per-GPU share depends on world size, not DP,
  and with enough HBM both have fit together. Test the pairing rather than ruling it out.
- **With cpu-offload on, configs that all keep EP intranode can plateau** at the same throughput
  across PP and GPU-count changes: the idle is offload stall plus comm-wait, and the plateau breaks
  only when offload is dropped.
- **Confirm the on-GPU optimizer actually FITS by dry-running the backward.** A no-offload config can
  pass model-build and forward, then die in the MoE backward with `CUBLAS_STATUS_ALLOC_FAILED`
  (cuBLAS handle allocation = effective OOM), even with `expandable_segments`.
- **`--optimizer-offload-fraction`** is the middle path when full-HBM OOMs but full-offload stalls:
  keep fraction f offloaded, (1−f) on GPU. The fitting window can be narrow when HBM is near-full —
  size f from the OOM gap, don't assume. Where the CPU optimizer step is on the critical path, its time
  fell roughly in proportion to the offloaded share, so lowering f step by step paid each time down to
  the memory edge. f competes with mt and recompute for the same HBM: the balanced point beat pushing
  either alone, and where the copies were already hidden, lowering f slowed the microbatch loop through
  memory pressure and lost. Just above the OOM edge throughput fell sharply (allocator pressure); ship
  the robust step below it.
- **CPU optimizer threads.** With offload on, the CPU AdamW step's default thread count was wrong for the
  host (diagnosed once as single-threaded, once as oversubscribed); one thread per physical core per
  actor gave a large gain, and using hyperthreads as well did not add to it. The thread count is an environment variable: where the
  canonical-manifest rule holds unlisted env vars fixed (`scripts/env_rules.py`), it is not a lever,
  and the gap is one more reason to lower f or drop offload.
- **Offload host RAM per rank scales ∝ 1/DP.** With several actors per node, low-DP (high TP×PP)
  topologies can exceed the node's memory limit and Ray kills a worker (often at `update_weights`);
  DP ≥ 4 has been the safe floor with several actors per node. Check node allocatable memory ×
  actors/node before launching a low-DP offload config; the profiler adds more host RAM on top (see
  "Profiler operability").

**EP on a single node is mainly a memory lever.** At fixed micro-batch size it is close to
throughput-neutral; higher EP holds fewer experts per rank and leaves more activation headroom, which
is what lets mbs rise, while too low an EP OOMs on expert weights. Likewise TP does not help a
single-node MoE already at its mbs ceiling (it shrinks DP into more microbatches, or adds intranode
comm at a matched count) unless it unlocks a higher mbs; check that it does before spending TP.

**Last-resort memory:** drop `--max-tokens-per-gpu` by 25% (most activation-sensitive lever).
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a fit enabler whose throughput cost has ranged
from none to large (`failure_triage.md`); measure it at the operating point rather than assume either. It
reclaims fragmentation (a few GiB of reserved but unallocated memory), which fit some configurations and
left others a few hundred MiB short; at saturated memory it thrashed and ran several times slower.

### Tier: env / NCCL (verify, don't tune blindly)

- **Don't set `NCCL_PROTO`** — let NCCL auto-select Simple/LL/LL128 per-collective by message
  size; a pin that helps at one bs can tie or lose at another, which the auto-selector already
  handles. **Worse, `NCCL_PROTO=Simple` on EFA can hang PP>1 at comm setup / step-1 watchdog**
  (easily misread as a hardware fault). Strip it from any inherited YAML.
- **`NCCL_NVLS_ENABLE` — leave at default `2` (auto).** Authoritative check: grep the trace for
  `*_NVLS_*` kernel names; none present → NVLS didn't run regardless of the env-var. It has been
  neutral across workloads; don't spend a slot. Remove any hardcoded `=0` in base YAMLs (a legacy
  HPC-recipe pattern).
- `NCCL_BUFFSIZE` / `NCCL_P2P_NET_CHUNKSIZE` — rarely move past defaults; not an independent lever.
- `CUDA_DEVICE_MAX_CONNECTIONS` — commonly pinned to `1` on this stack (SP requires it); some flags
  (FSDP, MoE EP overlap) want >1. Verify what your image's Megatron expects before changing it.
- `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,NET` — topology verification (`comm with N ranks
  and M nodes`). AWS OFI NCCL plugin loaded when INFO says `Loaded net plugin Libfabric`. One stock
  image silently fell back to the socket transport until the network and tuner plugins were named
  explicitly; if INFO shows no Libfabric plugin, the baseline itself is mis-wired, so report it rather
  than tune around it. Keep INFO to short diagnostic launches: on long many-rank runs its output has
  filled Ray's temporary storage.
- An intermittent hang at NCCL init on EFA with no watchdog (cross-node GDRDMA connect stall) is an
  infrastructure transient, not a config bug; see `failure_triage.md` and budget it as a free retry.

## (e) How to drive it

1. Establish the baseline first (mandatory): run the normalized baseline once through the
   fixed-shape protocol and record `aggregate_tok_per_s_per_gpu` over the measured window —
   the denominator for every reported end-to-end Δ%. Also use it as a smoke test for
   YAML/image/dataset/shared-volume wiring. See SKILL.md.
2. **Classify before choosing the next lever** (from a profile, or from the log scalars in §c) — the
   classification (§c) drives selection, not the tier order.
3. Change one knob at a time, measure, revert if no gain.
4. **Stop micro-optimizing a stage once it's clearly paying off** (>20% Δ, stable). Pivot
   rather than bisect — e.g. don't bisect between two mt values after the first already paid off.
   Anything inside the band (SKILL.md "Verdict") is a tie: document it, do not rerun it, and move to
   a new hypothesis.
5. Topology is high-leverage/high-risk, and it is where the large gains come from: try the leader's
   structural neighbours (each topology axis both ways, an uneven split, vpp, offload, recompute, mt)
   before flag toggles, which mostly tie ("Flag toggles rarely pay at all", §d). The largest lever can
   appear only once the PP neighbours of the leader are swept.
   You may launch a topology candidate concurrently once you have a stable winner to rebase on.
   On a recipe inherited from smaller-HBM GPUs, check cpu-offload first (Tier optimizer).
   The controlled 8K ranker represents typical SFT. The short/long distinctions below are useful
   when interpreting transfer to a different production workload:
   - **Short-seq** (p50<500, p99<2K, chat): dynamic-batch packing matters most; vpp starves;
     EP alltoall often dominant; low TP with higher PP tends to fit.
   - **Long-seq** (p50>2K, p99>4K): activation memory binds; CP and higher PP become useful;
     higher mt viable.
   For RL the "seq length" that sets the regime is prompt+response, not a static dataset — take
   the regime from the first rollout's `rollout/response_len/mean` (measurement.md §2), not only
   from a pre-tuning prompt scan.
6. **Don't stack tier priors to predict a compound gain** — compound stacks land far below the
   product of individual gains. Workload shape (seq length, model width, bs) caps the compound win;
   meaningful MFU gains need compute-bound shape + FP8-friendly GEMM sizes + hidden overlap
   together — not reachable by flag tuning alone on narrow-hidden MoE. The exception is a genuine
   **interaction**: a lever that lost alone can win once another change removes the reason it lost
   (cheap intranode a2a × larger mt is the canonical case). Stacked ties stay ties unless the
   combined delta clears the band.
7. One-off run: cap effort at a few candidates per tier and ship. Recurring production: the
   structural wins (FP8, DeepEP, kernel work) justify more effort.
8. **Keep each run's delta against its parent as well as against the leader.** With several runs in
   flight, a change that beat its own parent can still score "lose" against a newer leader, and such
   parent-positive changes are easy to leave unstacked. When the leader changes, re-test them on it.
9. **Cross-node results carry a placement term.** The same multi-node recipe ran markedly slower when its
   nodes straddled two network switches, and baselines spread far more across allocations than an
   intranode leader did. Treat a surprising cross-node win or loss whose placement may differ from the
   leader's as provisional before it closes an axis.

## Profiler operability on memory-tight + distributed configs

A profiled run can fail to write a trace by DEFAULT on cpu-offload / high-mt configs and on multi-pod
RayJobs. Independent failure modes, each fatal to the trace (FAILED or zero traces):

1. **Host-RAM OOM at the capture window.** slime `TrainProfiler` (older image default) profiles ALL
   ranks with `record_shapes=True, with_stack=True, profile_memory=True, with_flops=True`. On a
   cpu-offload config each actor already pins a large share of host RAM; the per-rank profiler host
   buffers (esp. `with_stack` = per-event call stacks, `profile_memory` = every alloc/dealloc) push the
   node over its cgroup limit → Ray kills a worker mid-capture. The measured window is already logged; the
   crash is purely a profiler artifact at rollout ≥ profile_step_start, and the run is unrankable.
   **Fix: profile one rank per pipeline stage (the skill's default, via `scripts/profile_rank_hook/`)
   and set `record_shapes/with_stack/profile_memory/with_flops = False`.** The analyzer needs neither
   stacks nor the memory timeline. Verify node RAM headroom (offload host RAM + profiler overhead)
   before assuming a trace will write. Upstream, THUDM/slime **PR #1879** makes all four options
   off-by-default (opt-in via `SLIME_PROFILE_*` env vars) and gates profiling to rank 0
   (`SLIME_PROFILE_ALL_RANKS=1` to override). On a newer slime the file is
   `slime/observability/profile_utils.py`; older images have it under `slime/utils/`, which is the
   module the rank hook patches — check the path in your image.
2. **The patch must reach the ACTOR pods, not the submitter.** A RayJob `entrypoint: bash run-training.sh`
   runs on the SUBMITTER pod only; `TrainProfiler` is instantiated inside `MegatronTrainRayActor` on the
   head+worker pods, each importing its OWN `/root/slime` (baked into the image, not shared). So an
   entrypoint `sed -i .../profile_utils.py` patches nothing that runs. **Deliver profiler patches by (a)
   baking them into the image, or (b) a hook in the hook chain on every pod** (`scripts/hook_chain.py`;
   the `profile_rank` hook is what `make_run_yaml.py --with-profiler` enables); never mount another
   `sitecustomize.py` in front of the chain. This same "entrypoint ≠ actor pods" trap applies to ANY
   runtime source patch on a multi-pod RayJob — confirm from the actor ranks' log lines that the patch
   took effect on the worker pods, never just the submitter/head.
3. **`--profile-target` wiring differs by slime build.** On some builds `megatron_utils/actor.py` wires
   the profiler ONLY via the `train_overall` path (`self.prof.on_init_end()` then
   `self.prof.step(rollout_id)` per training step); the `iterate_train_actor` /
   `iterate_train_log_probs` paths (targets `train_actor` / `train_log_probs`) are never called. Then
   `--profile-target train_actor` yields `use_pytorch_profiler=True` but **zero traces** (silently no
   trace). The skill defaults to `--use-pytorch-profiler --profile-target train_overall
   --profile-step-start 22 --profile-step-end 23` (at the default step counts, `scripts/workload.py`); that
   target's schedule counts training steps, so start/end remain step indices. Check `actor.py` once per image.
4. **A broken bash line-continuation drops trailing flags.** An empty conditional line after a `\`
   silently truncates the python command. Verify `use_pytorch_profiler` is actually True in the
   resolved-argument dump.

### Profiler trace size on large MoE
A `train_overall` trace captures the FULL training step (all microbatches × fwd/bwd/optim) for every
rank that writes. On large MoE at the benchmark GBS a multi-step window can produce traces of several
GB per stage. The per-step category mix is stable across the captured steps, and the analyzer reads
one step (the last complete one, `--step` to override), so one active step is sufficient for
`profile_top_categories`; the throughput window is independent of the capture window.
Run the analyzer with a long timeout or in the background (SKILL.md L4b). Also: the trace is still
being FLUSHED for a while after the job hits its last step — copy it only after its size STABILIZES
(or the job is SUCCEEDED), else the gzip is truncated and parsing raises EOFError. The trace lives on
the shared PVC, so it survives teardown — finalize (copy+analyze) can happen after the RayJob is gone.

### Compile tail at high PP
With `TORCHINDUCTOR_COMPILE_THREADS=1` (synchronous; avoids an async compile-pool deadlock) the
compile tail lands entirely in step 0 and grows with PP stage count (cold MoE grouped-GEMM/permute
kernels, serialized). The later steps then run at full speed and the measured window stays clean without
cache-seeding, even on large many-expert models. The cost is wall-clock only; add cache-seeding only
if high-PP compile time dominates the budget.
