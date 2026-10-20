# Levers — Megatron / Slime training (depth reference)

Depth reference behind SKILL.md (which owns the loop and Invariants): per lever, what it changes, when it helps or
hurts, how it fails, where the source is. A rule is a mechanism plus the condition it was seen under; magnitudes are
rough on purpose; evidence stays in each campaign's `RESULTS.md` and ledger. Direction often fails to transfer across
model size, GPU/HBM, TP or sequence regime, magnitude almost always: every rule is a prior to test with one run.
Measured on SFT and pretraining only; most of these knobs are expected to carry to RL's training phase, untested there.

This catalog holds what the campaign model does not reliably know, measured against Claude Opus 5.5; it is re-probed
when the campaign model changes.

**Scope tags** (*Scope: workload; model; GPUs*). **F8** fixed 8,192-token SFT rows, GBS 128 (the ranking workload);
**F2** fixed 2K packed SFT, GBS 128; **F4** fixed 4K pretraining, GBS 1,280; **F32** fixed 32K rows (one diagnostic
run); **V15** variable-length SFT (a public chat set and internal sets, one synthetic; rows ≤ ~15K; dynamic batching, mt
2K–24K; GBS 128–6,144); **Vchat** the chat set alone (rows ≤ ~4K); **V32** variable-length ≤ 32K; **MOCK** mock-data
pretraining on Pai-Megatron (4K/8K rows); **LC** long-context packing (rows of tens of thousands of tokens: a
micro-batch's activations and logits grow with the longest row and several rules reverse; only entries naming LC cover
it). Models: MoE (also when a tag omits the model) unless **D** (dense); **HLA** hybrid linear attention
(Gated-DeltaNet); **Mamba** hybrid Mamba MoE; **XL** a very large MoE. **1n** one 8-GPU node; EFA between nodes. Stack:
slime on Megatron-LM unless **miles** (v0.1.0). "Against:" = records the other way; **UNK** = untraced; untagged =
source, arithmetic or policy. mt = `--max-tokens-per-gpu`, mbs = `--micro-batch-size`, a2a = EP alltoall, CDMC =
`CUDA_DEVICE_MAX_CONNECTIONS`, offload = CPU optimizer offload.

> **Objective safety:** change classes and gates are SKILL.md's Invariants (structural, same-math swaps, the objective
> class refused by layer 1; every recipe ships through the final check). Toggling offload swaps CPU for fused GPU Adam
> (a small systematic loss offset is possible); a backend swap can move numerics (`--attention-backend fused`); a
> passing loss check is necessary, not sufficient (`--grad-reduce-in-bf16`). Throughput guidance says nothing about
> training quality. **Citations:** a behaviour claim is trustworthy only with an upstream file/function; otherwise
> confirm it in the image's source (flags rename, defaults flip, flags override each other) and record `file:line` in
> `verified_from_source` (SKILL.md L2).

**Index — bottleneck → entries.** *a2a:* §b THE RULE, Forced cross-node EP, DeepEP; EP; EP on a single node; mt; Reduce
PP. *TP/SP, param-gather, grad-reduce, CP comm:* TP; overlap pair; CP; Distributed optimizer. *PP SendRecv:* §c; Reduce
PP; Stage memory-pressure stall. *Memory/OOM:* §b Moves that keep weight bytes per rank; Increase PP; TP; recompute;
Stage memory-pressure stall; Distributed optimizer; CPU optimizer offload; Last-resort memory. *Compute-bound:*
Selective recompute, Full recompute (FP8 objective-class). *Bandwidth-bound:* fusions (`--moe-permute-fusion`); Flag
toggles rarely pay at all. *Launch-bound:* §c; TP; mt/mbs; CUDA graphs. *Bubble/straggler:* §c; vpp; Uneven pipeline
splits; smaller mt. *Long step 0:* Compile tail at high PP. *Optimizer-step gap:* CPU optimizer offload; §c step split.
*Data/rollout wait:* §c step split.

## (a) Read these upstream sources FIRST

Read these and form a theory before any MoE tuning project: the Megatron-LM MoE README and best practices
(<https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe>); the Ultrascale Playbook
(<https://huggingface.co/spaces/nanotron/ultrascale-playbook>; investigate results that disagree with it); the
Megatron-Core 0.16 parallelism guide
(<https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/parallelism-guide.html>; version-qualify
against the image); the Megatron-Bridge parallelism guide
(<https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html>).
- **Model recipes** (<https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/recipes>): nearest
  model/GPU as topology, recompute and kernel priors, not configs or throughput targets. *Scope: F8; MoE; B200, H200.*
- **Source paths beyond Megatron:** slime `backends/megatron_utils/` (flags may be gated), `utils/profile_utils.py`,
  `rollout/sft_rollout.py`, `utils/dp_schedule.py`; TE `transformer_engine/pytorch/`; the NCCL env reference
  (<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>).

## (b) MoE Parallel Folding + the intranode-MoE rule (canonical)

**Two grids share only PP:** attention `TP×CP×DP×PP`, experts `ETP×EP×EDP×PP`; per stage `TP·CP·DP = ETP·EP·EDP`. The
default fold keeps EP ranks contiguous (`parallel_state.py`, `expert_decoder_rank_generator`, order
`[ETP][EP][EDP][PP]`; rank r on node `r // GPUs-per-node` under Ray/k8s co-located scheduling); the fold order can
change across versions: confirm it with `print_topology.md`.
- **Moves that keep weight bytes per rank:** expert bytes per rank ∝ experts/(EP·ETP) × layers/PP. Do the arithmetic:
  prose reasoning here often closes good regions.
- **One node, EP folded into PP:** a deep fold removing the a2a won large on one memory-bound benchmark, a shallow one
  won large when it let recompute drop, lost or tied with no memory to free; test the endpoint. *Scope: F2, F8, MOCK;
  MoE; 1n H100.*
- **PP-divisibility lock** (`PP = world/EP` at `ETP = EDP = 1`): escapes, in order: uneven PP (§d), EDP > 1 (the same EP
  at a lower PP, more expert bytes), ETP > 1 (`EP = TP·CP·DP/ETP`); GPUs are fixed. Check this lock before planning an
  intranode-EP experiment.

### THE RULE: EP alltoall is intranode iff `ETP × EP ≤ GPUs-per-node`

The `TP × EP` (`TP × CP × EP ≤ 8`) mnemonic is wrong unless ETP = TP and CP = 1 (ETP defaults to TP, `arguments.py`):
set ETP. Verify with NCCL INFO (`comm with N ranks and M nodes`, M = 1) or `print_topology.md`. With EP intranode, PP
choice is bubble vs DP AllReduce.
- Diagnose PP>1 failures from NCCL INFO, not assumed NIC shortfalls (same-GPU instances differ in EFA geometry).
  *Scope: F8, V15; MoE; H100, H200, B200.* **Climbing PP** to fold a cross-node EP into a node won even with a much
  larger bubble. *Scope: F8; MoE; H100, B200.*
- **Forced cross-node EP** (expert memory): a2a + PP boundaries ≈ half the kernel time, small GEMM share, low MFU; on
  EFA possibly irreducible (overlap needs vpp, DeepEP needs IB, intranode EP needs a PP that may not fit or host-OOMs
  under offload): keep it; optimize the memory-minimal topology keeping it, the least recompute that fits, the best mt.
  *Scope: F8; XL; H100, H200, B200.* Overlap can hurt there (little GEMM to hide behind: shared-expert overlap lost on
  one such config, tied on another). *Scope: F8; XL; H200.*
- **`--overlap-moe-expert-parallel-comm --delay-wgrad-compute`** overlaps adjacent microbatches' FWD/BWD with the a2a.
  Prerequisites asserted in `transformer_config.py`: EP > 1, alltoall or flex dispatcher, bf16, no full recompute
  (selective OK, `--recompute-method`, `--recompute-num-layers` unset), non-legacy grouped GEMM for delay-wgrad, and vpp
  at PP>1 (`virtual_pipeline_model_parallel_size must be specified`), so vpp blockers block it. With TP>1 or CP>1 on
  Hopper it conflicts with CDMC=1 (Tier env). Memory-hungry (two microbatches' activations), rarely fit
  (`failure_triage.md`); public writeups report a sizable share of the a2a recoverable. Speculative. *Scope: V15 (H100,
  H200), F8 (XL; H200, B200); MoE.*
- **DeepEP** (`--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep`, or `--moe-enable-deepep` on
  builds that use that flag: check `arguments.py`; needs TP×EP > 1, `deep_ep`, no pad-to-capacity): a clean swap from
  alltoall (same routing; no vpp or recompute change; FP8 dispatch off); the variable-length limit is `allgather`'s.
  *Cross-node: infeasible on EFA* (needs NVSHMEM IBGDA or IB verbs; signatures: `Buffer.__init__` fails with
  `nvshmemi_transport_init: init failed for transport: IBGDA` or "Peer GPU not accessible", then `ibv_poll_cq failed`
  and ActorUnavailableError; or every NIC logs `device rdmap* is not supported (expected HCA interface: mlx5)`, then
  `store->get got 0 bytes`) on every EFA generation and build tried: an InfiniBand lever unless a build adds an EFA
  transport. *Scope: F8; cross-node EP; H100, H200, B200.* *Intranode* (works on EFA) it pays only where the a2a is the
  residual bottleneck (ties when PP SendRecv dominates): from a large win stacked with a larger mt, through ties, to a
  large loss on the newest NVLink (slower than NCCL) and a loss on the previous one at large microbatches; its SMs
  compete with GEMMs when compute is the larger share; wins did not transfer across GPUs or TP; some stacks segfaulted
  (measure per stack); one build was wrong or crashed at one hidden size — component-test at the model's shapes.
  `allgather` is no alternative: slime and miles v0.1.0 rewrite it to `alltoall` after `validate_args` printed the dump
  (the log says `allgather`; same code, a tie). *Scope: F8 (H100, H200, B200), F2, F4 (H100), V15 (H100, H200, B200);
  MoE.*

## (c) Profile → bottleneck classification

Classify every ranked run before the next lever, from its trace (SKILL.md L4b) or log scalars; skipping it ("try FP8",
"add recompute") is a common cause of wasted runs. *Scope: F8; MoE; H100, B200.*

| Signal | Class → go to |
|---|---|
| NCCL largest, AllGather-dominated | param gather → overlap-param-gather, DP↓ |
| NCCL largest, ReduceScatter- or AllReduce-dominated | grad reduce → overlap-grad-reduce (bf16 reduce refused) |
| NCCL largest, SendRecv-dominated | PP or a2a → PP↓, verify EP intranode |
| PP>1, idle ≥ ~10% | maybe bubble → compare the analytical bubble, look for a critical stage; then PP↓, aligned vpp, bs↑ only if GBS is unconstrained |
| compute largest, GEMM-dominated | compute-bound → less recompute (FP8 objective-class) |
| one stage's allocator cleanup far above the rest; `STAGE MEMORY-PRESSURE STALL` | stage stall → Stage memory-pressure stall |
| busy low, many short idle gaps and launches, kernel ≪ step | launch-bound or small-kernel dependency stall → lower TP (SP splits GEMMs, adds collectives), bigger microbatches; held where TP↓ doubled busy, not where CUDA graphs left idle |

*Scope: comm and compute rows routing heuristics (UNK); the rest F8, F4; MoE; H100, H200, B200.*
- **Test vpp rather than gate on idle %:** with a2a or AllGather dominant, PP↓/vpp won't fix it. *Scope: F8; MoE; B200,
  H200.*
- **Check mechanisms with the run's numbers** (a wrong mechanism often has the right sign and fails a one-line check).
  *Microbatches:* very many one-row microbatches = overhead-bound, whatever else the profile suggests (*Scope: F8; MoE;
  B200*). *Step split:* loop vs tail (optimizer step, offload copies, distopt collectives): DP and mt shrink the loop,
  the tail mostly not with DP, only some offload changes; subtract frozen data-path costs (rollout wait, batch copies)
  before bounding what a lever can recover (*Scope: F8, F4, F2; MoE; H100*).

## (d) Lever catalog (organized by SKILL.md priority tiers)

### Tier: topology (highest leverage, hardest to get right, most risk)

**Reduce PP** (at PP1 drop the vpp flag) — where EP rises with DP and crosses nodes it lost ~half (EDP > 1 keeps EP
intranode at more expert bytes). Revert if AllGather/a2a grew more than SendRecv shrank. *Scope: V15, F8; MoE; H200,
B200.* **Lower vs higher PP** (EP intranode): higher PP (more microbatches per rank at high bs) can win on short rows
and smaller HBM, but under dynamic batching it has a longer compile warm-up (more stages/chunks meet new shapes) that
can dominate a short run; past it, it tied or led. Sweep neighbours. *Scope: V15 at GBS 128–1,024; MoE; H100, H200.*

**Increase PP** — for weights, grads and optimizer state (or a lower TP needing layer sharding), not activations. Hurts:
more cross-node hops, a bigger bubble unless microbatches grow, a longer step 0. *Scope: F8 (B200, H100), V15 (H100,
H200); MoE.*

**TP — valid values and direction.** TP must divide or be a multiple of `num_query_groups` (`transformer_config.py`;
read the real value); above the KV heads they are replicated (redundant KV compute, more TP comm), adding (TP, DP)
splits at the same EP; a per-head output gate then needs per-rank slicing (check the image's attention code before
launching such a TP), and an unsplit per-head parameter (sinks, softmax-offset logits) fails TP>1 at checkpoint load
(global shape mismatch). Pair TP with SP; CDMC rules: Tier env. Sinks and sliding windows may lack CP: check before
planning CP>1.
- **TP1** removes all TP comm and often won on comm-bound narrow-expert MoE with EP intranode (where it lost: Higher
  TP); memory is what blocks it: unsharded non-expert params/grads (more PP: full width × fewer layers ≈ TP2 at half the
  PP) and unsharded logits (`mt × vocab × 4 B`). Try TP1 + just-enough PP first; the winning PP may only be the smallest
  that fits, and "fits" means headroom above the cleanup threshold (that PP stalled; one up cleared it). CE fusion frees
  no logits (its entry). *Scope: F8 (B200, H200), V15 at mt 4K–6K (H100, B200); large-vocab MoE. Against: TP2 beat TP1
  at PP1 on V15 (H200); TP1 lost to TP2 on a larger MoE on F8 (H200) and on V15 (H100); on F8 (B200) CE fusion freed
  nothing and TP1 fit only through more PP.* With no SP its mt ceiling is ≈ TP>1's ÷ TP (`failure_triage.md`). *Scope:
  F8; MoE, D HLA; H100, H200, B200.* On LC it reverses: a micro-batch holds the longest row (only CP splits it), its
  logits (tokens × vocab × 4 B / TP) grow with it, and TP1/TP2 ran out of memory where a higher TP with CP fit — size
  logits at the longest row. *Scope: LC; D; B200.*
- **Carry mt/TP, not mt:** at fixed mt/TP microbatches per rank are TP-independent: start each TP there (closing TP1 on
  a TP>1-mt OOM shipped several percent less). *Scope: F8, PP1–2; MoE, Mamba; H200 (miles), B200.*
- **Higher TP** frees HBM (bigger mt) but adds AllGather; past the sweet spot it loses. TP1 won with ample HBM and EP
  intranode (the TP AllGather is pure overhead), lost to TP2 under a cross-node a2a; on 80 GB GPUs TP1 often did not fit
  and a TP above the KV heads won large over a stalling lower TP (its HBM went to mt; the next step lost to TP comm).
  Re-measure per GPU and workload. (1) A TP-up win can be a last-stage stall at the lower TP (a lighter last stage beat
  it widely): check per-stage cleanup first. (2) Compare each TP at its own best mt (at the TP1 mt a TP>1 point is
  understated); one-factor walks close high-mt TP>1 regions a joint search finds. *Scope: F8; MoE; H100, H200, B200. TP1
  also lost on larger HBM (TP1's Against).*
- **Check that every token mixer is tensor-parallel:** slime and miles run Gated-DeltaNet layers through an HF wrapper
  that runs the whole mixer on every TP and CP rank (`slime_plugins/models/hf_attention.py`): with three of four such
  layers each TP halving won a fifth to a half, and no-recompute OOMed at TP>1. Read the plugin's forward. *Scope: F8; D
  HLA; H100, H200, B200.*

**EP — set placement deliberately** (§b). ETP > 1 lost at F8 on every MoE tried — it splits small expert GEMMs, adds
expert-TP collectives (the MoE README advises ETP1 for fine-grained MoE), and `ETP × EP` above a node crosses nodes:
prefer ETP1 unless the profile disagrees. ETP/PP folds are regime-dependent (don't inherit short-row winners): an ETP2
fold that won on short variable-length rows (lower EDP at fixed EP while optimizer AllGather/ReduceScatter dominated)
lost at F8 on the same model. Biased MoE linears force ETP1 (`transformer_config.py`). *Scope: losses F8, V15 (H100,
H200, B200); MoE, narrow experts. Against: ETP2 fold wins V15 (H100, H200).* Single node: Tier optimizer.

**CP:** with full recompute on comm-bound MoE CP only adds `--allgather-cp` comm (costly on EFA); CP1 won even at 8K,
not at every halving. Compare at matched mt; at high mt CP degrees tied (mt amortizes per-call cost), so a CP win from
elsewhere is a prior. *Scope: F8 (H100, H200, B200), V15 (H200); MoE.*

**vpp** is gated by per-rank microbatch count and layer divisibility, not GBS.
- **Alignment (source):** slime's (and miles') `dp_schedule.py` pads microbatches to a multiple of
  `DP × microbatch_group_size_per_vp_stage` (default PP, `model_parallel_config.py`). Fixed rows at `mt = seq_len/CP`
  give unsplittable `m = GBS/DP` (`GBS/(2·DP)` at `2·seq_len/CP`): vpp works iff `m % PP == 0`; else batch setup asserts
  ("could only produce M ... after maximal splitting; need N"), not a dynamic-batching conflict.
- **Aligned it paid** a few percent to mid-teens, also under dynamic batching; a tie at high PP with large m. On
  variable-length data it hurt at small GBS, helped at large (microbatches per rank; those losses included compile
  warm-up, which chunks lengthen). *Scope: F8 (H100, B200; misalignment H200); V15 (H200), tie/loss at GBS 512 (B200);
  MoE.*
- **Depth:** vpp multiplies p2p and launches and adds in-flight activations: cross-node boundaries with many small
  microbatches lost a quarter to two thirds against a few-percent bubble, and tight stages OOMed or stalled — compute
  the bubble first. *Scope: F4; MoE; H100.*
- **Divisibility:** step 0 runs long with vpp ("Compile tail"). **DSA index-share (GLM-5.x)** needs stage and chunk
  boundaries on computing layers; the auto-split ignores it and slime's GLM-5 model asserts `DSA index-share pipeline
  split is invalid` (`glm5.py`, fails closed); a conforming layout (uneven chunks, less bubble gain) is untested, so the
  bubble levers are microbatch count (GBS/DP/mt) and PP. **Mamba hybrids:** vpp asserts at model build in miles v0.1.0
  (Megatron-Bridge's Mamba provider; MCore `MambaStack`), closing EP comm overlap at PP>1; CP runs. *Untested:* that DSA
  layout (would unlock vpp and EP comm overlap), EP comm overlap with vpp elsewhere, the coupled TP>1 × high-mt × vpp
  region with structural levers only.

**Uneven pipeline splits** (`--decoder-first/last-pipeline-num-layers` or `--pipeline-model-parallel-layout`) belong in
every PP>1 candidate set (*Scope: F8 (H100, B200), F4 (H100), V15 (H100); MoE*). *Compute balance:* without a trace test
a split each way. With few layers per stage and a large output layer (big vocab, low TP) balanced beat even by high
single digits and a straggler split lost double digits; one-layer moves were noise at tens of layers per stage; weigh
the output layer (a few decoder layers there) by per-stage work, not parameter or FLOP counts. Too many layers off an
end makes the middle critical; heavier ends lose more; the plateau is narrow (neighbours tied; one more layer onto the
middle or first stage lost ~a tenth); the best gave the ends one to a few layers fewer, never more off the first than
off the last, and the first turned critical when given a layer. One wrong-direction run proves nothing; judge by the
critical stage's work and tok/s/GPU; the balance moves with TP, PP and mt (a winning split lost a few percent stacked
elsewhere) — re-check after topology changes. *Scope: mostly F4, large-vocab MoE at TP1 (H100); F8 (H100, B200).*
- *Mechanics:* a one-layer move with the flags lands on every middle stage (`transformer_block.py`) and can overshoot.
  The `--account-for-embedding-in-pipeline-split` / `--account-for-loss-in-pipeline-split` divisibility assert says
  nothing about uneven splits (`arguments.py`). Every stage's (chunk's) layers must divide by the vpp depth and
  `--recompute-num-layers` (uniform), so a split may need a layout; PP2 with both ends set breaks vpp
  (`transformer_layer.py`, division by zero). Pick receivers by work and free memory: receivers pushed to their memory
  edge collapsed throughput by a large factor (allocator stall; untraced) — the first stage is usually tightest.
- *Stack:* Megatron's transformer decoder runs splits with and without vpp; in the Megatron-LM of miles v0.1.0
  `MambaStack` ignores the split flags its attention/MoE layers follow (`megatron/core/ssm/mamba_block.py`), so a split
  loads wrong weights (step-0 loss several times the baseline's). A hang at NCCL init on an uneven layout matched an
  `NCCL_PROTO=Simple` pin; check for one first. *Scope: hang V15 (H200, older image); F8 (H100, B200).*

**Stage memory-pressure stall → uneven split or more recompute on that stage.** Below 3 GB free slime runs
`torch.cuda.synchronize()` + `gc.collect()` + `torch.cuda.empty_cache()` before collectives
(`slime/utils/reloadable_process_group.py`; recent versions skip p2p, alltoall, all-gather, reduce-scatter); miles
v0.1.0 clears once per step below a tenth free (`finalize_model_grads_with_empty_cache`). The least-free stage (often
last: output, loss; or first: embedding, activations) pays per call while the GPU idles and the others show p2p wait.
Not an OOM (peak allocated well below device memory; NCCL buffers sit outside the allocator). Per-stage traces show it
(rank 0 cannot): cleanup time and calls far above the rest. Removing it multiplied throughput where it was diagnosed.
Fix: move layers off it below L/PP, or recompute more (selective/partial → full, or a higher `--recompute-num-layers`
under full uniform, divisibility permitting) — a split helps here via memory, not balance; judge by cleanup calls → ~0
and tok/s/GPU (the split can lose where nothing is pressured). A leader just above the threshold is fragile (a larger mt
or heavier last stage tips it): read free memory before stacking. *Scope: F8; MoE; H100.*

### Tier: microbatch packing / recompute (medium leverage, cheap to try)

**mt (`--use-dynamic-batch-size --max-tokens-per-gpu N`):** start at 8K, raise while memory allows (ceiling: recompute,
PP, offload, TP/SP, length mix). *Scope: V15; MoE; H100, H200, B200.*
- **Packing check:** microbatches per DP rank ≈ `bs × mean_seq / (DP × CP × mt)`; mt↑ helps only while that is well
  above one — at about one, look upstream (topology, overlap).
- **Fixed rows: a step function** — rows per microbatch = `floor(mt / (seq_len/CP))` ≥ 1, so only `mt = k·seq_len/CP`
  informs (at CP1 all mt < 2·seq_len tie); the same step up won on one model and lost badly on another. *Scope: F8; MoE;
  H100, H200, B200.*
- **mt cuts calls, not volume:** biggest with tiny microbatches (about doubled throughput), saturating once per-call
  cost is small (a several-fold call cut tied). *Scope: F8; XL; B200.*
- **Past the peak mt falls off a cliff** — a sharp drop one step up is memory (the tightest, usually last, stage in
  cleanup or thrash; also just before OOM), not the bubble, which grows modestly; ship the last robust point. *Scope:
  F8; MoE; H100, B200.* **Bigger mt** — helps with large per-microbatch overhead and headroom — fails as a last-stage
  OOM (watch the last stage's memory lines in the log). *Scope: F8 (H100, H200), V15 (B200).* **Smaller mt** — a smaller
  bubble but smaller GEMMs and more calls — lost on a pipelined MoE (the overhead outgrew the bubble removed); bound the
  bubble first. *Scope: V15; MoE; H100, H200.*
- **PP>1 pays a shape exchange per p2p call:** slime and miles force `variable_seq_lengths` (`arguments.py`).
- **miles `--qkv-format bshd`** (DeepSeek-V4): miles v0.1.0 asserts dynamic batching off (`miles/utils/arguments.py`),
  so packing is mbs (k rows), mt advice is thd-only. On DeepSeek-V4-Flash k > 1 crashed in the first training step
  (`view size is not compatible with input tensor's size and stride`), leaving DP (lower TP or CP); retry k = 2 per
  image. *Scope: F8; a large MoE; H200; miles.*

**Without dynamic batching mbs is the packing lever** — fewer accumulation microbatches, bigger GEMMs: ~linear to the
ceiling on a memory-bound single-stage MoE, saturating after one step at deep PP; recompute, EP and TP matter via the
headroom they free. *Scope: MOCK 4K (1n H100), saturation F4 (H100); MoE.* The MoE rules do not carry to dense models,
where full recompute is a large share of the step: at 8K rows a split that let recompute drop beat these rules clearly
(*Scope: MOCK 8K (1n H100), F8 (H200); D*); the split does not shrink first-stage activations, so where they grow with
the row measure split vs recompute (*Scope: LC; D; B200*).

**GBS** is fixed (SKILL.md Invariants): sweep it only on the user's say, marked "bs-coupled" so a bs winner isn't
promoted without convergence sign-off.

**Selective recompute** (`--recompute-granularity selective`; drop `--recompute-method` and `--recompute-num-layers`) —
helps when matmul is a meaningful share and memory has headroom: the largest single lever on a memory-capped
cross-node-EP MoE once it fit, mid-teens with ample headroom; it fit and lost on an offloaded comm-bound config. *Scope:
F8 (H200, 1n H100), V15 (H200, B200); MoE.* It shares HBM with mt, and TP sets how much: it stopped fitting at half the
TP (SP had sharded activations), its forced smaller mt losing more than it saved — compare at each TP's best mt.
*Scope: F8; MoE; 1n H100.* It fails as an OOM (more live activations than full; fit at one GPU count, not another);
distrust "it fits" estimates for granularity changes (one undershot widely): dry-run. *Scope: V15 (H200), F8 (H100,
H200, B200); MoE.*
- `--recompute-modules` (`transformer_config.py` `recompute_modules`): `layernorm` is cheap but frees little. *Scope:
  F4; MoE; H100.* Default-module selective ≈ no recompute for MoE memory, so full recompute enables a memory-bound MoE
  (dropping it forced a lower mbs and lost). *Scope: MOCK 4K; MoE; 1n H100.*

**Full recompute** (`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`) always fits and
adds FLOPs; on MoE it re-runs the expert path (a2a, grouped GEMMs): ~a sixth alone on a compute-heavy step, and paired
with a doubled mt it doubled the step; it hid only in launch idle (tiny microbatches: + doubled mt won a few percent,
one mt step more lost a quarter+). *Scope: F2, F4 (H100); MoE.* A higher `--recompute-num-layers` usually tied or cost a
few percent, occasionally gained a few, and collapsed once a chunk spanned most of a stage; other full-uniform
granularity is near-neutral. *Scope: F8 (H100, B200), V15 (H100), F2, MOCK (1n H100); MoE.* Block/partial paid a few
percent with headroom; OOMed or thrashed at the full-recompute mbs/mt ceiling. *Scope: F8 (H100, B200), V15 (B200), MOCK
(1n H100); MoE.*
- **Uniform divisibility:** each stage's (chunk's) layers must divide by `--recompute-num-layers`, else
  `IndexError: index k is out of range` (`transformer_block._get_layer`) at the first forward; check every stage before
  launch.
- **Recompute can be load-bearing once offload is dropped:** near the HBM limit full→selective OOMs; with headroom
  dropping both offload and recompute won; if only one fits, measure (no-offload won on a large MoE, no recompute on a
  dense model). *Scope: F8; MoE, D; H200.*

### Tier: overlap / kernel flags (mixed impact — reshape schedule, not work)

The dominant collective decides which matters. **Flag toggles rarely pay at all:** fusion, kernel, dtype, overlap and
padding toggles mostly tie or lose a little; big gains are structural (topology, offload, packing, recompute,
interleave, splits). Bound a toggle by its kernel category's share of the step. *Scope: F8; MoE; H100, H200, B200.*

**`--grad-reduce-in-bf16` — objective-changing, never shipped even after a passing loss check:** gradients accumulate in
bf16 (`validate_args` forces `accumulate_allreduce_grads_in_fp32 = False`); 100-step loss A/Bs passed at rerun noise
while an exact-repeat A/B failed — the source classification is the guard. Gains did not transfer: a sixth to a half on
one variable-length workload where grad reduce was large, a few percent elsewhere, mostly a tie at F8. *Scope: V15
(H200), F8 (H100, H200, B200); MoE.*

**`--moe-shared-expert-overlap`** — often net-negative (reschedule/sync exceeds the hidden dispatch): removing it tied
or helped, adding it lost or tied with cross-node and intranode EP; alltoall only (`ValueError` with flex). *Scope: V15
(H100), F8 (H200, B200); MoE. Against: V15 (H200), tie or gain within noise.*

**`--cross-entropy-loss-fusion`** (`--cross-entropy-fusion-impl` `te` or `native`: a tie) — usually a tie; it frees no
logits memory in slime/miles SFT (losses take log-probs from the full logits, `sft_loss_function`; miles v0.1.0 calls
the model with `labels=None`, `megatron_utils/model.py`). *Scope: V15, F8, F4, F2; MoE.*

**`--moe-permute-fusion` / `--moe-grouped-gemm`** — cheap, often default (a no-op if already fused: check the profile);
removing permute fusion lost clearly. Neither off is same-math: unfused, the top-k combine sums in bf16 (error grows
over layers, unseen in one-layer tests); without grouped GEMM the per-expert MLP's fused bias-activation drops the
activation clamp (moved the loss on a clamped-GLU model). *Scope: F8 (H100); component tests at F2/F4 shapes (H100);
MoE.*

**`--overlap-grad-reduce` + `--overlap-param-gather`** (a pair: param-gather requires grad-reduce, `arguments.py`;
`--align-param-gather` needs interleaving) — tied where they ran; an "overlap stack" gain came from flags bundled with
it. *Scope: V15 (H200), F8 (H100, H200, B200); MoE.* Often blocked: slime's `train()` asserts
`config.no_sync_func must be None` under grad-reduce overlap, then sets it for good and runs per rollout, so grad-reduce
overlap fails at the second step (step 1; `slime/backends/megatron_utils/model.py`) and param-gather overlap on a
`KeyError` (`failure_triage.md`); miles v0.1.0's `model.py` has the same loop. Grep the image once for the assert; if
present, don't spend a slot, and don't assume a newer image is patched (an adhoc image is rarely worth it). A
re-entrancy patch (slime PR #1857) clears it; then the pair tied without offload (little exposed DP comm). With offload,
grad-reduce overlap asserts `len(self.per_param_grad_ready_counts) == len(self.params)`, so the distopt AllGather is
unhideable. Blocked, cut distopt AllGather/ReduceScatter by lowering DP (smaller shards; a small DP group stays in a
node).

**`--tp-comm-overlap`** aborts at TE userbuffers init on slime: `initialize_ub` raises `Intra-node group must be on the
same backend as the world group` (`comm_gemm_overlap.cpp`), as slime wraps non-gloo groups in a backend-less
`ReloadableProcessGroup` (`slime/utils/reloadable_process_group.py`; miles too). *Scope: F8, V15; MoE; H200.*

**Usually neutral:** aux-loss coefficient (objective), `--defer-embedding-wgrad-compute` (can OOM),
`--moe-router-fusion`, `--moe-router-dtype` (objective; fp32 ≈ fp64; bf16 rejected), `--ddp-bucket-size`, CDMC (a tie;
small loss at TP1), padding multiple, update-weights interval, high-priority streams; `--manual-gc` hurt or tied.
Unusable: `--ddp-average-in-collective` (asserts with `--calculate-per-token-loss`), `--moe-apply-probs-on-input` (top-k
1). *Scope: V15 (H100, H200), F8 (H100, H200, B200); MoE.*

**Speculative** (exhaust everything above first): intranode DeepEP, EP comm overlap (§b). **CUDA graphs**
(`--cuda-graph-scope attn`, TE impl) — small: attention and whole-layer capture tied where half the step was
dependency-stall idle (not launch latency); the legacy flag asserts with EP on expert padding; recompute limits in
`failure_triage.md`. *Scope: F4 (H100), V15 (H200); MoE.* **`--attention-backend fused`** (cuDNN) is numeric, may ship,
listed for review: NaN grads with low GQA + fp32 softmax; where clean and faster its loss drifted up late beyond rerun
noise, which the 20-step check misses — cite it, confirm with a longer loss run. *Scope: NaN V15 (H100, H200), drift F8
(B200); MoE.*

### Tier: precision

**FP8** (`--fp8 hybrid`; co-flags in `arguments.py`) — objective-changing, needs convergence sign-off — fails on narrow
hidden: casting dominates small GEMMs (on one narrow-hidden, narrow-expert MoE tensorwise tied and delayed-scaling
hybrid lost heavily; blockwise can OOM), as linear cost ∝ hidden² vs elementwise ∝ hidden (NeMo performance guide, "Low
Precision Training"). *Scope: V15; narrow-expert MoE; H100, H200.* It costs memory: the fitting mbs fell below the bf16
winner's (small MoE); blockwise and MXFP8 OOMed where bf16 fit (larger MoEs). *Scope: MOCK 4K (1n H100); V15 (H200), F8
(B200); MoE.* Also small-microbatch overhead, accuracy deltas, recompute-mode and amax-history interactions (check TE
source). *Scope: amax V15 (H200); rest UNK.* **`--use-precision-aware-optimizer`**: CPU optimizer offload needs it
(below).

**Objective class, refused by layer 1:** **bf16 Adam state** (`--exp-avg-dtype bf16 --exp-avg-sq-dtype bf16`) is
unreliable (OOMed on a temporary fp32 upcast; wedged a run late). *Scope: F8 (H100, B200), V15 (H200); MoE.* **Expert
load balance** (`--moe-aux-loss-coeff`, `--moe-router-z-loss-coeff`: objective class). **Capacity factor**
(`--moe-expert-capacity-factor`, drop policy neutral): pays ~a tenth at cap 1.0, more below, saturating above; objective
class (drops tokens: at cap 1.0 it failed loss parity by two orders of magnitude over rerun noise, one-sided); report it
apart from the headline. *Scope: MOCK (1n H100), V15 (H100), F8 (B200); MoE.*

### Tier: optimizer / memory relief

**Distributed optimizer** (`--use-distributed-optimizer`; check its default in `arguments.py`): the lever is how many
ranks gather (DP), not the flag. Per-GPU state ≈ total / world (PP↑ halves params per rank as DP halves), so only more
GPUs shrink it.

**CPU optimizer offload — last-resort memory relief** (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d`) — only
when the state won't fit: the CPU step and copies are a large per-step cost (GPU idle a large share). Signal: one long
idle gap per step at the optimizer step with the host CPU busy — drop it and see if it fits. slime and miles time
`train()`, optimizer step included, as `actor_train_time` (`megatron_utils/actor.py`): a small
`step_time − actor_train_time` gap clears nothing; the no-offload run is the test. Offload requires the precision-aware
optimizer (`arguments.py`), not the reverse: dropping offload keeps the flag (a miles v0.1.0 assert of the reverse
applies only with `--rematerialize-param-from-master-weight`). *Scope: F8, V15; MoE.*
- **Drop offload first** on a recipe from smaller-HBM GPUs: it dwarfed every topology lever on large MoEs and collapsed
  idle (some dense models gained more from TP). *Scope: F8 (B200, H200), V15 (H100, H200, B200); MoE.*
- **Fit:** for very large MoEs offload can stay mandatory at every reachable topology (*Scope: F8; XL; H100, H200,
  B200*). On a small world its share falls as tokens per GPU per step grow (~1% on long steps, a sixth to a third on
  short) (*Scope: MOCK 4K; MoE; 1n H100*).
- **After a no-offload win:** no-offload and intranode EP looked exclusive (shards vs thin DP) but the share depends on
  world: with enough HBM both fit. *Scope: F8; MoE; B200.*
- **Confirm the fit through backward and step 1:** `CUBLAS_STATUS_ALLOC_FAILED` in the MoE backward (despite
  `expandable_segments`) is an effective OOM (*Scope: F8; XL; H200*).
- **`--optimizer-offload-fraction` f** — the middle path when full HBM OOMs and full offload stalls; size f from the OOM
  gap (the window can be narrow). With the CPU step critical its time fell ∝ f, each step down paying to the memory
  edge; f shares HBM with mt and recompute (balance beat pushing one); with copies hidden, lowering f slowed the loop
  and lost; just above the OOM edge throughput fell sharply — ship the robust step. *Scope: V15 (1n H100), F8 (B200, 1n
  H100); MoE.*
- **CPU optimizer threads** (Ray sets `OMP_NUM_THREADS` to the actor's CPU request, 1 if none, `ray/_private/utils.py`):
  a thread per physical core per actor gave a large gain, hyperthreads nothing. Where `scripts/env_rules.py` freezes
  unlisted env vars it is no lever — another reason to lower f or drop offload. *Scope: F8; MoE; B200 hosts.*
- **Offload host RAM per node ≈ offloaded state ÷ nodes** (per rank total/world, not DP: DP1–DP8 used equal RAM per
  actor); past the node limit Ray kills a worker, often at `update_weights`: check allocatable memory, plus the
  profiler's ("Profiler operability"). *Scope: F8; XL; H200, B200.*

**EP on a single node** is mainly a memory lever (*Scope: MOCK 4K; MoE; 1n H100*). At fixed microbatch on F8 one EP step
moved a few percent: up with higher EP on a 128-expert Mamba MoE, up with lower EP on a 32-expert MoE (fewer, smaller
a2a), a tie at 64 experts — test one step each way. *Scope: F8; MoE; H200.*

**Last-resort memory:** drop mt (most activation-sensitive; on fixed rows only a whole-row drop frees memory, as an
over-mt row gets its own microbatch). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` enables fits at a cost from
none to large (`failure_triage.md`): it reclaims a few GiB of fragmentation, fitting some configs and leaving others a
few hundred MiB short; at saturated memory it thrashed several times slower. *Scope: F8 (1n H100), F4 (H100), V15
(H200); MoE.*

### Tier: env / NCCL (verify, don't tune blindly)

- **Don't set `NCCL_PROTO`:** NCCL picks per collective, and a pin that helps at one bs ties or loses at another;
  `Simple` hung PP>1 at setup / step-1 watchdog on one EFA generation (looks like hardware); strip pins. *Scope: V15 at
  GBS 256–1,024; MoE; H200.* **`NCCL_NVLS_ENABLE`:** keep the default (2; `*_NVLS_*` trace kernels are the check);
  neutral multi-node, where no NVLS kernel ran for training collectives; remove legacy `=0`. *Scope: V15; MoE; H200.*
- **CDMC:** FSDP asserts > 1; EP comm overlap prefers > 1 and only warns (`arguments.py`). Check the image first.
- **`NCCL_DEBUG=INFO`:** one stock image used sockets until `NCCL_NET_PLUGIN` and `NCCL_TUNER_PLUGIN` were set; keep
  INFO short (long many-rank runs filled Ray's temp storage). *Scope: V15; MoE; H100.*

## (e) How to drive it

1. Baseline first (SKILL.md B0-B3): every Δ%'s denominator and a smoke test of YAML, image, data and volume wiring.
2. Classify before choosing (§c): the class, not the tier order, picks the lever.
3. Large gains are topological: try the leader's structural neighbours (each axis both ways, uneven split, vpp, offload,
   recompute, mt) before toggles; the largest lever may appear only after the leader's PP neighbours. *Scope: V15
   (H100), F8 (H100, H200, B200); MoE.* You may launch topology candidates concurrently once a stable winner exists; on
   a smaller-HBM recipe check offload first. F8 is one regime (ETP, vpp, TP1 vs TP2, `grad-reduce-in-bf16` flipped
   between it and variable-length data): re-measure a winner at production lengths. *Short rows* (chat, median a few
   hundred tokens, dynamic batching): packing matters most, the a2a often dominates, vpp starves only when GBS leaves
   few microbatches per rank (*Scope: Vchat; MoE; H100, H200, B200*). *Long rows* (LC): activation memory binds and the
   longest row's micro-batch sets the floor (mt); other rules are hypotheses. RL: length = prompt + response from the
   first rollouts (prompt length plus `rollout/response_len/mean`, measurement.md §2), not a prompt scan alone.
4. Shape (length, width, bs) caps MFU: real MFU needs a compute-bound shape, FP8-friendly GEMMs and hidden overlap
   together, beyond flag tuning on narrow-hidden MoE. *Scope: V15 (H200), F8 (H100, H200); MoE.*
5. One-off: a few candidates per tier, ship; recurring production justifies deeper work (kernels, DeepEP, a
   convergence-signed FP8 recipe).
6. Cross-node results carry a placement term: a recipe ran markedly slower with its nodes on two switches, and
   baselines spread far more than an intranode leader. *Scope: F8; MoE; B200.*

## Profiler operability on memory-tight + distributed configs

Each loses the trace (FAILED or zero traces) on offload/high-mt configs or multi-pod RayJobs:
1. **Host-RAM OOM at capture** (slime's and miles' `TrainProfiler` profiles all ranks with `record_shapes`,
   `with_stack`, `profile_memory`, `with_flops`): use one rank per stage (default, `scripts/profile_rank_hook/`) with
   all four off (the analyzer needs neither); check RAM headroom. THUDM/slime PR #1879 (superseded by #2397, not merged)
   would make them opt-in (`SLIME_PROFILE_*`) and rank-0 only (`SLIME_PROFILE_ALL_RANKS=1`): check the image's
   `_create_torch_profiler`. Newer slime: `slime/observability/profile_utils.py`; older `slime/utils/` (the rank hook's
   target). *Scope: F8; MoE with offload; B200, H200.*
2. **Patch the actor pods.** The entrypoint runs on the head pod: Ray Jobs pins the driver to the head node unless the
   job sets entrypoint resources (`dashboard/modules/job/job_manager.py`), and KubeRay's submitter pod only runs
   `ray job submit`. `TrainProfiler` lives in `MegatronTrainRayActor` on head and workers, each with its baked-in
   `/root/slime`, so an entrypoint `sed` patches only the head pod's copy and the workers' actors run unpatched. Bake it
   in or use a chain hook on every pod (`scripts/hook_chain.py`; `make_run_yaml.py --with-profiler` enables
   `profile_rank`); never mount another `sitecustomize.py` ahead of the chain; confirm any runtime patch in actor logs.
3. **Only `--profile-target train_overall` traces:** slime's `megatron_utils/actor.py` (upstream, every image checked)
   wires it only there (`self.prof.on_init_end()`, `self.prof.step(rollout_id)`); `train_actor` / `train_log_probs` give
   `use_pytorch_profiler=True` and zero traces; miles v0.1.0 raises `NotImplementedError`. Default
   `--use-pytorch-profiler --profile-target train_overall --profile-step-start 22 --profile-step-end 23`
   (`scripts/workload.py`; training-step indices). Check `actor.py` per image.

### Profiler trace size on large MoE
`train_overall` captures whole steps per writing rank (several GB per stage per multi-step window). The category mix is
stable and the analyzer reads one step (last complete; `--step`), so one active step suffices for
`profile_top_categories`; capture is independent of the throughput window. Use a long timeout or the background
(SKILL.md L4b); copy a trace only once its size is stable or the job SUCCEEDED (else truncated gzip, EOFError);
shared-volume traces survive teardown. *Scope: F8; XL; B200.*

### Compile tail at high PP
On fixed rows `TORCHINDUCTOR_COMPILE_THREADS=1` (synchronous; avoids an async-pool deadlock) puts the compile in step 0,
and the window stays clean without cache seeding, even on large many-expert models (*Scope: F8 (H100), F32 (B200);
MoE*). Estimate step 0 as ~(PP+1) × the first microbatch before launching or killing (a slow, near-constant time per
microbatch for the first ~PP+1 microbatches, then the steady rate; one run was killed there as a stall) — ~10 min at PP8
on a dense hybrid, hours for a very large MoE at TP1/PP16; a kernel cache shared by all pods (Triton, Inductor, other
JIT caches on the shared volume) cut a deep-PP step 0 by more than half. *Scope: F8; MoE, D HLA (H100, H200, B200), XL
(H100); cache F8 (a large MoE; H200; miles).* With vpp too, falling per-microbatch times are warm-up, not a stall: judge
speed from step 1. *Scope: F8; MoE; B200.* FLA 0.4.1 recompiles per new token bucket (1,024 tokens for
`ShortConvolution`, finer for `FusedRMSNormGated`) on variable-length micro-batches. *Scope: V32; D HLA; H100.
Dynamic-batching recompiles also hit early steps on V15 (MoE; H200).*
