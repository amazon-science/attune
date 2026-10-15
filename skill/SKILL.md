---
name: autotune-training-throughput
description: Autonomously tune LLM training throughput by launching, monitoring, and iterating fixed-shape 8K experiments on a Kubernetes/Ray/Megatron stack. North-star KPI is tokens/sec/GPU (`perf/actor_train_tok_per_s / num_gpus`). Use when the user wants AI-driven performance tuning that runs unattended — launching experiment configs, reading training logs, forming hypotheses from bottleneck profiles, proposing the next experiment until the iteration budget is spent. Invoke for phrases like "keep tuning throughput", "run experiments overnight and pick winners", "autotune this training job", "keep improving tok/s".
---

# Autonomous Training Throughput Tuning

## Overview

Given a working-but-untuned Megatron/Ray/K8s training recipe, benchmark it once at its own GPU count
as the baseline, then iterate experiments to maximize
`aggregate_tok_per_s_per_gpu = perf/actor_train_tok_per_s / num_gpus`, unattended, until the
budget is spent or you judge the search has converged. You launch RayJobs, read `train.log` metrics, profile bottlenecks, and
pick the next recipe — keeping a durable ledger so a fresh session can resume.

Two phases: a one-time **Bootstrap** (establish the baseline) and the repeating **tuning loop** (whose
Invariants are the non-negotiable ranking rules). State contract and gotchas follow; depth (levers,
failure signatures, topology) lives in `references/` (indexed below).

## Usage

Pass `<baseline_yaml_path>` when invoking the skill. It is a **working but untuned**
RayJob+ConfigMap YAML and the source of the training objective. If invoked without a path, ask; don't
guess. Derive:
- `project_tag` = baseline basename minus `.yaml`, **lowercased** (K8s rejects uppercase names).
  Name it by **GPU type** (`h100`/`h200`/`b200`), not instance type — e.g. `mymodel-moe-sft-h200` —
  so priors are discoverable by GPU family across projects. Append the instance only when GPU type
  doesn't disambiguate (p5e and p5en are both H200 → `…-h200-p5e` / `…-h200-p5en`).
- `run_id` = `<prefix><project_tag>-<suffix>`: lowercase letters, digits and `-`, ≤47 chars, where `<prefix>` is
  `${ATTUNE_RUN_PREFIX:-$AUTOTUNE_USER-}` (`$AUTOTUNE_USER`, your owner-label value, is required). Every RayJob and
  ConfigMap name starts with that prefix: `make_run_yaml.py` refuses another run id and labels the run
  `<AUTOTUNE_OWNER_LABEL>=<AUTOTUNE_USER>`, and `teardown_run.sh` refuses a run without both.
- `project_folder` = `${AUTOTUNE_PROJECT_ROOT:-$PWD/attune-projects}/<project_tag>/` — all state lives here,
  including the ledger `AUTOTUNE_LEDGER.jsonl` (`$AUTOTUNE_LEDGER`, when set, names the ledger file instead; the
  harness Stop hook reads the same file).

## Bootstrap (once per project, before any tuning)

Run B0-B3 once to establish the baseline — the denominator every experiment is measured against.
Skip any step whose durable output exists; a valid `baseline` ledger row means bootstrap is done.

```
B0. Find authoritative recipes for the model (exact model, then nearest architecture/size; vendor/framework
    repos over blogs). With web access (`AUTOTUNE_ALLOW_WEB=1`; off by default) search the internet; without
    it, use the recipes and references available locally or provided by the operator (`references/levers.md`
    §a lists the upstream sources) and make no network requests. Record dated sources + GPU count/type,
    topology, batch, seq-len, precision, recompute, offload in `RESULTS.md`, noting what transfers.
B1. Swap in the ranking workload. The workload comes from the environment (`scripts/workload.py`):
    GBS `$AUTOTUNE_GBS` (default 128), rows of exactly `$AUTOTUNE_SEQ_LEN` tokens (default 8,192),
    dataset `$AUTOTUNE_DATASET`, and `$AUTOTUNE_NUM_STEPS` steps per unprofiled run (default 20; a profiled
    run adds the profiler's setup and warm-up steps and `$AUTOTUNE_PROFILE_CAPTURE` recorded steps, default 1, so 23). Without these
    set, the workload is fixed-8K at GBS 128: repoint the recipe (`--prompt-data`) at the model's
    fixed-8K dataset and set GBS=128 — its own dataset/GBS are NOT what we rank on. Fixed-8K
    datasets live under `$AUTOTUNE_DATA_ROOT/<model_tag>-sft-fixed8192-v1/train.parquet` (one per
    tokenizer; source corpus/revision in the manifest, not the name). If none exists for the target
    model, build it once (tokenize a pinned SFT corpus with THAT tokenizer + chat template, correct
    loss mask, pack/pad every row to exactly the workload's row length, frozen count/order/seed),
    e.g. as a K8s Job. Store `tokens` and `loss_mask` in each row's metadata so a pretokenized
    rollout adapter can feed them unchanged. Verify the artifact (every row exactly the row length
    in valid tokens, masks valid, order/seed frozen, hashes in the manifest) and persist path +
    hashes + GBS to `<project_folder>/data_insights.md`. `make_run_yaml.py --benchmark` stamps the
    same workload onto every run. Never rank different dataset bytes/tokenizers together (a
    real-data transfer run is a post-selection check, not a leaderboard row).
B2. GPU count = the recipe's own count, fixed for the whole campaign: the GPU count is not a tuning dimension
    (see Invariants). Dry-run to confirm the count is feasible: it must satisfy model-memory, topology/world-size, layer/expert
    divisibility, `global_batch_size >= DP`, and the image's ACTUAL expert rank-fold placement (verify
    from source/`print_topology.md`, never an `ETP × EP` mnemonic). A fit/feasibility failure is
    ledgered unranked (not a tuning slot).
B3. Run one unprofiled benchmark job (`$AUTOTUNE_NUM_STEPS` steps, default 20) at that count as the baseline (profile it in a separate run if you want a trace), applying only the operational fixes needed to
    run on the frozen data+count (drop checkpointing, strip a `NCCL_PROTO` pin, adjust topology to
    fit) — objective and GPU count unchanged. No tuning until it lands. Its tok/s/GPU is the
    denominator every Δ% is measured against AND the reference side of the final correctness check; every run keeps its GPU count.
    Its ledger row has `kind: "baseline"` and `status: "completed"`; every candidate's terminal row has
    `kind: "ranked"`, its `verdict` and the `metrics` JSON `read_metrics.sh` printed (`references/ledger.md` §1;
    these are the fields the Stop hook counts).
```

## The tuning loop

Repeats until a stop condition fires (below). It is a producer/consumer, not lockstep: L2-L3 (launch
to fill idle slots up to the cap) and L4 (harvest each run as it finishes) run concurrently. Never
block the whole loop waiting on one job — keep free slots full while others run, and finalize each
completion independently.

```
L1. Read ledger + RESULTS.md + data_insights.md + the skill's `past_tuning_projects/*_summary.md`
    (priors from prior projects, if any) + references/ledger.md (ledger schema, anti-patterns)
    → current best, what's tried, priors, and which topology family to favor.
    **Read `references/levers.md` IN FULL once per session before your first hypothesis** (it is the
    cross-project heuristic catalog — priority ladder, MoE folding, profile→lever table, per-lever
    mechanisms/failure-modes); re-read the relevant section after any edit to it. Do not form an L2
    hypothesis without having loaded it — a "remembered" lever is stale.
L2. Choose the next experiment(s) with your own tuning judgement — there is NO prescribed rule for
    which lever to pull. Under a fixed slot budget you optimize for the best recipe you can find by the
    last slot, never for coverage of the search space. Ground the decision in four sources of context,
    then pick the hypothesis with the highest expected `tok/s/GPU` gain × confidence:
      1. **Profile** — the latest bottleneck (L4), when you have one: what actually limits the current leader.
      2. **Best practice** — `references/levers.md` (priority ladder, MoE folding, profile→lever table).
      3. **History** — this project's ledger/RESULTS.md + `past_tuning_projects/*_summary.md`: what
         already won/tied/lost here and in prior projects (don't relearn a known dead end).
      4. **Real tuning space** — the flags actually present in the image's `arguments.py`/source.
    If you run out of ideas, mine the history and the source flags for an untried, plausible lever;
    only if STILL out of ideas, fall back to filling out topology combinations for completeness.
    Process discipline (not a choice-rule): one variable per run for clean attribution unless tightly
    coupled; state the hypothesis (change <X> → bottleneck <Y> shrinks via <Z>, expected Δ%) — no
    hypothesis, no launch; close an axis only by the two-point rule under Search discipline below. A candidate topology must be
    valid (divisibility + intranode-MoE placement — confirm against `megatron/core/parallel_state.py`
    / `print_topology.md`); a flag win rarely transfers across topologies, so re-verify it at the
    eventual topology winner. Fill idle slots up to the cap with your top hypotheses; queue the rest.
    **Topology is FIVE independent axes (TP, PP, EP, CP, DP) — sweep each before concluding anything about
    topology.** A win on one axis (e.g. CP) does NOT mean the others are explored, and "I tuned topology"
    is false if you only moved one dim. Before declaring a "topology/structural ceiling", confirm you have
    actually probed TP (incl. TP1/2), PP, and EP neighbors — not just CP/DP. These axes are coupled through
    `world = TP·PP·CP·DP` and `EP = TP·CP·DP` (so moving one forces another — e.g. TP↓ raises DP; PP↓ raises
    EP): reason about that coupling and the profile before ruling an axis out, but a coupling-driven
    prediction is a hypothesis to TEST with one run, not a reason to skip the axis.
    **Before ANY flag, verify it in the running image's `/root/Megatron-LM/megatron/training/arguments.py`
    /source** (never from memory/docs) — what it sets, asserts, and which other flags it silently
    overrides (Invariants: objective-preserving). Record the `file:line` as `verified_from_source`; no flag enters a run un-read.
L3. check_gpu_capacity.py (`--baseline <yaml>` counts the run's tolerations) → need free ≥ nodes_needed, else
    queue (don't shrink to fit).
    make_run_yaml.py --benchmark → stamps the fixed-shape contract onto the YAML (Invariants); verify
    the output, append a
    `tracked_jobs.jsonl` row with `target_steps` = the run's last step (19 unprofiled, 22 profiled, at the
    defaults) → `kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" apply -f <run.yaml>`
    (GPU-hour budget mode: then `run_gpuh.py <run_id> --launch --gpus N`). (`target_steps` is the
    completion marker: the LAST 0-indexed step a full run reaches — steps run 0…N-1, so
    `--num-rollout 20` ⇒ last step 19 — used to reject a SUCCEEDED-but-truncated run. `poll_run.sh` takes it
    from the run's own `--num-rollout` when the log or RayJob shows it, and prints the `target_step` it used.)
L4. On each finished run, do BOTH:
    (a) **Finalize** — in the exact "State & idempotency contract" order so an interrupted finalize
        resumes safely: read_metrics.sh → verdict (table below) → append ledger + rewrite RESULTS.md
        → teardown → drop the `tracked_jobs.jsonl` line.
    (b) **Diagnose the run, and decide whether the next one needs a trace.** Record `profiled` and
        `profile_reason` on every terminal ledger row (ledger.md). Profiling is per launch and the choice is
        yours: profile when the next decision needs a diagnosis you do not have — the baseline is the natural
        first candidate, and so is a run whose bottleneck you cannot name from its metrics — and skip it when
        you are sweeping values along a lever you have already diagnosed. A profiled launch is charged its full
        cost like any other.
        When a run was profiled, run `analyze_profile.py` (Scripts §) to record `profile_top_categories`, then
        classify the dominant bottleneck (`measurement.md` §6 + `levers.md` §c). Fetch every trace the run wrote
        (one per pipeline stage by default), not just rank 0, into one directory, and run the analyzer
        with a long Bash timeout (600000 ms) or in the background: a cold parse of several ~0.5 GB
        traces takes minutes. Read its `coverage:` line first; if it reports `MISSING` stages, fetch
        those traces and rerun before classifying, since a stall local to one stage is invisible
        from the others. Without a trace, classify what you can from `train.log` (`perf N:` fields,
        memory and timing lines) and say so in `profile_reason`.
    (c) **Write back conclusive insights.** If the run established a *generalizable* fact about a lever
        — its mechanism, when it helps or hurts, a failure mode, an interaction, or a saturation/quantization
        boundary (e.g. fixed-length mt step-function) — APPEND it to `references/levers.md` (the
        cross-project catalog) as a rule, in the same commit as the ledger row. `levers.md` holds guidance
        only: mechanism, condition, and direction, with a magnitude at most as a word ("large", "a
        tie") or a rough range; no exact throughputs, deltas, step times, memory figures, GPU counts,
        dates, model or cluster names, or run_ids. The numbers and evidence go in RESULTS.md and the
        ledger. When in doubt whether it generalizes, add it.
L5. Goto L1.
```

**Search discipline** (applies to every choice of the next experiment, however it is made):
- **Two points before an axis closes.** A structural axis (a parallelism degree, a pipeline split,
  interleave depth, recompute, offload, max-tokens) closes only after two measured points on it against
  the same parent, in both directions where it has two (a larger and a smaller value; layers moved off a
  stage and onto it), or after its feasible end point. One losing run is one point.
- **A new topology starts lean.** Launch a new topology from the leader without the memory-relief
  settings the leader needed (full recompute, CPU optimizer offload or its fraction) unless a memory
  estimate for the new topology shows they are still needed. Add them back after the lean run OOMs.
  A topology probe that carries relief it does not need measures the relief, not the topology.

**Invariants** (every launched run must satisfy these — they define a valid ranking; get one wrong and the result is invalid):

- **Fixed-shape ranking.** One benchmark job per recipe on frozen rows of exactly the workload's row
  length (`$AUTOTUNE_SEQ_LEN`, default 8,192 tokens), with identical dataset bytes/order, seeds,
  objective, GBS (`$AUTOTUNE_GBS`, default 128), hardware type, image, and code across candidates;
  `--num-rollout` N = `$AUTOTUNE_NUM_STEPS` (default 20, at least 20), or N+2+K (default 23) when the run is profiled
  with K recorded steps. **The GPU count is fixed at the baseline's count**: a tuned recipe runs on the recipe's own GPU
  count, since at a smaller count per-GPU throughput can rise while aggregate throughput falls. `make_run_yaml.py`
  refuses any other count (`$AUTOTUNE_NUM_GPUS`, else the base YAML's own count) unless
  `AUTOTUNE_ALLOW_GPU_COUNT_CHANGE=1`, an explicit opt-in. Everything else stays identical too.
- **Measurement window, the same for every run.** Discard warm-up steps 0-4, measure steps 5..N-1 (5-19, 15 samples,
  at the default N=20) via `read_metrics.sh <log> 5 19 <num_gpus>` — always pass the real `num_gpus` (a wrong count
  silently corrupts the headline); never rank an invalid/incomplete window. A profiled run measures the SAME
  window, then spends step N on the profiler's setup and step N+1 on its warm-up (slime steps the profiler inside
  the step's timer, so both are slower) and records K steps N+2..N+1+K (`--profile-step-start 22
  --profile-step-end 23` at the defaults, end exclusive); `make_run_yaml.py --profile-capture 2|3` records up to
  three steps when the step-time CV is high or something periodic is suspected, and refuses any profile range whose
  setup step falls inside steps 0..N-1 or that runs past `--num-rollout`. An unprofiled run carries no profiler
  flag. The last step is 19 unprofiled and 22 profiled (one recorded step); a run that ends earlier is incomplete.
  Benchmark runs never load the correctness probe: `make_run_yaml.py --benchmark` removes `parity_probe` from
  `$ATTUNE_HOOKS`.
- **Frozen arguments.** When `$AUTOTUNE_FROZEN_ARGS` names a frozen-arguments file
  (`scripts/correctness/frozen_args.py`), those arguments keep their frozen values in every run:
  `make_run_yaml.py` refuses a run that sets them otherwise, and correctness layer 1 fails any
  candidate whose resolved arguments violate them. Read the file before forming hypotheses.
- **Profiling is per launch, not a gate.** A completed run is rankable with or without a trace. When you asked
  for one (`--with-profiler`) and it is missing at `<tensorboard-dir>/*.pt.trace.json.gz`, say so in
  `profile_reason` and decide whether to relaunch with it; do not discard a valid measurement over a trace.
- **Objective-preserving only — a lever is objective-CHANGING until PROVEN otherwise.** Never trust a
  flag as safe from memory (e.g. `--grad-reduce-in-bf16` also force-sets
  `accumulate_allreduce_grads_in_fp32=False`, see `levers.md`).
  - *Structural (placement/scheduling only, no dtype/value change):* parallelism degrees
    (TP/PP/EP/ETP/CP/DP), `sequence-parallel`, dtype-preserving data-parallel comm overlap,
    recompute, and microbatch packing and size
    (`--max-tokens-per-gpu`, `--micro-batch-size`, dynamic batching). Packing and microbatch changes
    stay structural even when the MoE auxiliary load-balancing loss is computed per microbatch:
    regrouping rows shifts that term slightly, as TP/DP/PP changes shift reduction order and rounding,
    and the final correctness check (below) is what accepts or rejects the recipe. Structural does not
    mean exempt: every shipped recipe passes that check.
  - *Same-math swaps (ship only where the campaign's allowlist lists them):* a fusion, a kernel or
    dispatcher implementation, optimizer CPU offload, or any other different implementation of the same
    function. Correctness layer 1 admits one only where the setup's allowlist lists that argument and value,
    which happens only after the swap passed its component test. The allowlist is the `allowlist.yaml` beside
    the frozen-arguments file `$AUTOTUNE_FROZEN_ARGS` (by convention `rules/layer1/<setup>/`,
    `scripts/correctness/rules/README.md`), plus `rules/images/<digest prefix>/allowlist.yaml` for swaps
    tested on the image, which apply to every model on it when the pair's `images.json` (written by
    `pair_from_runs.py`) names that image, pinned by digest, for both runs. Every other swap is numeric by default and fails layer 1. Read both allowlists before
    you propose one.
  - *Numeric (different math or lower precision → never shipped):* anything changing the dtype/value of
    gradients, optimizer state, accumulation, precision (FP8), loss, expert routing/capacity,
    `global-batch-size`/`rollout-batch-size`, or LR/optimizer. Layer 1 fails a candidate that carries one,
    whatever the other layers show. You may measure one to learn from it; report it apart from the shipped
    recipe, and never call a change "training-neutral" without a recorded PASS.
- **Concurrency cap: 3 running RayJobs** (the baseline counts). Overflow → `QUEUE.md`. If a scheduler
  (e.g. Kueue) admits fewer, let it gate.

Poll in-flight runs every ~10 min with the active driver's persistent scheduler or goal mechanism,
reading `tracked_jobs.jsonl` and calling `poll_run.sh`. If none is available, poll in the foreground;
do not launch jobs and then end the session.

**Verdict** (aggregate vs the current leader; also retain Δ vs the original baseline). The band b is 3%, or, when
`$AUTOTUNE_CALIBRATION` names a calibration file, its band (`band`, at least 3%), and Δ vs baseline is then taken
against its `denominator`; `scripts/calibration.py` prints both and exits non-zero on a missing or invalid file. The
file is optional in both budget modes: without it the band is 3% and Δ vs baseline is taken against this campaign's
baseline run.

| Δ tok/s/GPU | verdict |
|---|---|
| > +b, validity gates pass | **win** — promote as the new leader |
| −b … +b | **tie** — document it, do not rerun, and move to a new experiment |
| < −b | **lose** |
| crashed before any `perf N:` | **error** (triage: `references/failure_triage.md`) |
| RayJob SUCCEEDED but `latest_step` below the run's target (19 unprofiled, 22 profiled) | **error** — incomplete benchmark or a swallowed actor crash; never trust SUCCEEDED alone |

Each recipe gets one ranked run; reruns are not part of search or convergence. The one other run a recipe can get is
a **confirm run**: when a recipe's ranked run was profiled and the recipe reaches the final correctness check, run it
once more unprofiled (`kind: "confirm"`, ledger.md), because a profiled run is never a correctness arm. A confirm
run gives the check its candidate side and nothing else: it takes no budget slot, does not count toward
convergence, and its throughput is recorded but never re-ranks the recipe. It costs GPU-hours like any launch. For
concurrent completions, compare against the leader at ledger-commit time and record that run ID.

**Budget accounting** (N-ranked mode, the default; GPU-hour mode: see When to stop). The N-slot budget counts only runs that produced a **valid ranked
`tok/s/GPU`** (win / tie / lose) — those are your real shots at the best recipe. A run that produces
no ranking does **not** consume a slot:
- **Recipe-controlled failure** (OOM, invalid divisibility/topology, bad flag, or SUCCEEDED-but-
  incomplete): log the ledger row and its wall-time/GPU-hours in cost accounting, but it does not
  spend a slot. It is still informative — **never relaunch the identical config**; the next launch
  must carry a diagnosed change. Most of these are preventable by the pre-launch feasibility check
  (memory estimate + divisibility/placement), so a pile-up signals mis-estimation, not progress.
- **External-infrastructure failure** (node/network/preemption, cluster unreachable): one retry of the
  identical candidate, on other nodes; log its cost. A second infra failure on disjoint nodes is a recipe failure
  (no third launch).

## State & idempotency contract

The **ledger is the single source of truth**, and every state change follows a fixed order so that a
crash between any two steps leaves consistent, resumable state — re-running the poll just continues.

Per-project state files:

| file | role | write rule |
|---|---|---|
| `AUTOTUNE_LEDGER.jsonl` | source of truth: one terminal row per finished run | append-only; never rewrite a prior row (schema: `references/ledger.md`) |
| `tracked_jobs.jsonl` | the in-flight runs being polled | one line per live run; shrinks as runs finalize (schema: `references/ledger.md`) |
| `RESULTS.md` | human-readable leaderboard | **REWRITE the headline leaderboard table after EVERY completed run** — regenerate it from the ledger (current leader + all ranked rows sorted by tok/s/GPU), don't just append notes below a stale table. A resuming session/human reads the top table to know the leader; a drifted headline (e.g. still showing the baseline as leader after a win) is a real failure. Sync it in the same commit as the ledger row (L4a). |
| `QUEUE.md` | runs deferred past the concurrency cap | append/trim as slots free |
| `gpuh.jsonl` | GPU-hour log, GPU-hour budget mode only | one line per launch, written at submit and completed at finalize by `run_gpuh.py` (path `$AUTOTUNE_GPUH_LOG`, else `./gpuh.jsonl`) |

**On each poll**, read `poll_run.sh`'s `status` and act by case. A finished run goes through the
commit sequence **in this exact order** — each step runs only if the previous one succeeded, so an
interruption is always safe to resume:

- **done / errored:**
  1. `read_metrics.sh` — validate. Exit 0 gives the ranked number. Exit 3 or 4 (no perf lines, an invalid or
     incomplete window) makes the run an `error`, never a rank. On any other non-zero exit (5: the log could not
     be read) **stop here**, keep the tracking line and retry on the next poll (never ledger unread metrics).
  2. append the ledger row — **the commit point.** Write the fields of `references/ledger.md` §1: `kind`,
     `status`, `verdict` and the `metrics` JSON unchanged. A `run_id` that already has a terminal row is never
     re-ledgered (idempotent), so a re-poll after a crash won't double-count.
  3. `teardown_run.sh` — free the cluster resources.
  4. remove the `tracked_jobs.jsonl` line — **only after** teardown exits 0 (removing it earlier
     orphans cluster resources).
- **api-error** (cluster unreachable): do nothing, keep the line, retry on the next poll. This is **not**
  absence — never treat it as a finished/gone run.
- **absent** (a *successful* query returned nothing): drop the line.
- **stuck** (no new perf line for 10 minutes): triage it as a hang (`failure_triage.md`, `measurement.md` §10); a
  confirmed hang is finalized as `errored` (above), which tears it down.
- **done-unknown** (RayJob SUCCEEDED, no log found): keep the line and find the log under
  `$AUTOTUNE_EXPERIMENTS_DIR/<run_id>/logs/` on the log-access pod; if it is lost, finalize the run as `errored`.
- **setup / running:** leave alone.

**Concurrency & drift.** If a cron and an interactive session might poll at once, guard mutations with
`flock <project>/.state.lock`. Periodically reconcile live RayJobs (project prefix) against
`tracked_jobs.jsonl`: a live job with no line — or a line with no live job and no terminal ledger row —
is drift; add the missing line or finalize/clean the orphan.

### When to stop

**Ranked-count mode** (no GPU-hour budget set). Keep launching batches until **converged** — BOTH: (1) a
baseline row exists; (2) ≥ N distinct recipes with a **valid ranked result** (default N=20; failures don't
count — see Budget accounting). Then stop and report the leader. A quiet poll, a finished batch, "good
enough", or <N ranked recipes are NOT stops. A persistent goal/runbook/budget may set N (record it in the
`RESULTS.md` header).

**GPU-hour budget mode.** When a GPU-hour budget B is set (`$AUTOTUNE_GPUH_BUDGET`, or a
`GPU-h budget: B` line in the `RESULTS.md` header; the env var wins), it replaces the N-ranked rule. B is a
fixed cap, not a target to spend:
- *Accounting.* Every launch costs `num_gpus x (admission -> job end)` GPU-h (admission = the Kueue
  Workload's admission time on a Kueue cluster, else the RayJob's start time): baseline, ranked
  runs, confirm runs, failures (OOM, crash, invalid, incomplete) and correctness-check arms. Queue time before
  admission is free; a launch rejected before admission costs 0. Log every launch in `gpuh.jsonl`:
  `scripts/run_gpuh.py <run_id> --launch --gpus N` right after `kubectl apply` (parity arms too), and
  `scripts/run_gpuh.py <run_id> --record --status <done|failed|oom|killed>` in finalize, before
  teardown (the Workload and RayJob go with it). Copy its `gpuh` and `cum_gpuh` to the ledger row's
  `gpuh` and `gpuh_cumulative` (ledger.md); `run_gpuh.py --total` prints the campaign total. Failures
  still consume no ranked slot, but they do consume GPU-h, so the pre-launch feasibility check matters
  more.
- *Stop.* The search ends at the first of two events, and there is no ranked-count requirement:
  (a) **the cap**: `gpuh_cumulative` reaches B (counting jobs still running at their elapsed cost); a job
  that crosses B may finish, but its result counts only if it ended before cumulative spend reached B; or
  (b) **your convergence declaration**: you judge the leader is the best throughput you can find. The
  declaration needs a baseline row; the cap ends the search with or without one. Every launch still states a
  hypothesis with an expected gain (no hypothesis, no launch), so never launch a run only to spend budget. A
  quiet poll or a finished batch is not convergence.
- *Declaring convergence.* Nothing may be in flight: tear down any live run first (`teardown_run.sh`) and
  let it finalize; it is charged as usual. Then write the **`STOP` record**: a ledger row with `kind:
  "stop"` and non-empty `reason`, `leader`, `neighbours` and `diagnosis` (ledger.md; the Stop hook accepts no
  other), and the same content as a `## STOP` section at the top of `RESULTS.md`: the reason, the leader, the
  structural neighbours of the leader you tested (their run_ids and the axis each moved) and what the last
  profile or log diagnosis showed (the remaining bottleneck, and why no untried lever is expected to clear the
  band). Every topology or feasibility claim in that evidence rests on this
  campaign's ledger: an infeasibility claim cites the run that showed it (an OOM claim needs that
  configuration's OOM run, or a per-stage memory estimate written into `RESULTS.md` that exceeds device
  memory by at least 10%; inside that margin, launch it), a divisibility claim also holds for uneven
  pipeline splits, and a predicted loss is a hypothesis to test, not evidence. The final correctness check then runs
  exactly as at the cap.
- *Cascade inside the budget.* The final correctness check reuses runs you already have, so it spends from B only
  when a candidate's ranked run was profiled and needs its confirm run. Keep a reserve of two such runs (costed
  from your own runs' admission -> end times) and start the cascade when the remaining budget reaches it or when you
  declare convergence. The shipped recipe is the best one that passes within B; if none passes within B, ship the
  baseline.
- Record B, the accounting rule and the running total in the `RESULTS.md` header. Without a budget
  set, the N-ranked rule above is the default. Under the unattended harness (`AUTOTUNE_AUTONOMOUS=1`)
  the Stop gate reads only `$AUTOTUNE_GPUH_BUDGET`, so a header budget there must equal it; it accepts the
  cap or a STOP row.

**Before declaring the goal complete — final correctness check (cascade).** Test the leaderboard
top-down with the correctness check (`scripts/correctness/PROTOCOL.md`), on runs you already have: the baseline's
unprofiled benchmark run is the reference side and the candidate's own unprofiled benchmark run is the candidate side
(same data, seed, benchmark GBS, image and `--num-rollout`, scored on steps 0-19). Copy both `train.log`s off the log
pod, make the pair with `scripts/correctness/pair_from_runs.py --base-log <log> --base-yaml <yaml> --win-log <log>
--win-yaml <yaml> --pair <dir>` (`--allow-image-change` only when the recipe itself changes the image), then run
`scripts/correctness/parity_verdict.py` on it. A candidate whose ranked run was profiled first gets its confirm run
(Verdict §). Probe arms (`scripts/correctness/make_parity_arm.py`) are for diagnosing a failing pair, not for the
verdict.
Layer 1: resolved-argument diff with no numeric change (by the frozen rule lists,
`scripts/correctness/PROTOCOL.md`) and no frozen-argument violation.
Layer 2: the candidate's per-step training loss over the 20 steps against the baseline's, by Megatron-Bridge's
convergence check (`scripts/correctness/mbridge_convergence.py`, with the fixed thresholds of
`scripts/correctness/PROTOCOL.md`); the step-0 loss and grad-norm differences are reported beside it (and the probe's
per-step counts when the runs carry the probe).
**Pass** (layers 1 and 2)
→ that recipe is the shipped winner; set `parity_checked:true`. **Fail** → mark it `parity_failed`
with the failing layer, drop it, and test the next-best ranked recipe against the same baseline run;
repeat until one passes. If **none** pass, ship the baseline and report "no promotable winner". This
applies to structural leaders too (TP/PP/EP/CP changes alter reduction order), so no recipe is
declared the winner without a recorded PASS. If capacity blocks a needed unprofiled run, queue it and wait: an unrun
check stays open and does not pass. The thresholds and the 20-step window are fixed; never change them
after seeing a result. Infrastructure failures are relaunched and give no verdict.

**Whenever you stop** — mandatory closeout so future projects inherit experience.
Write `<this-skill-dir>/past_tuning_projects/<project_tag>_summary.md`:
model + hardware, baseline → winner tok/s/GPU and Δ%, the biggest lever (and why), what won / tied /
lost with reasons, excluded objective-changing wins, gotchas hit, untried ideas, and a **reference to
the project artifacts path as an ABSOLUTE path**. Summaries are local operator records (they carry
your numbers, paths, and cluster details); keep them out of any shared copy of the skill and move
their generalizable rules into `levers.md` instead (`past_tuning_projects/README.md`).

## Environment gotchas (can't-derive; verify per cluster)

- **Always use an explicit Kubernetes context and namespace.** The skill scripts read
  `AUTOTUNE_KUBECTL_CONTEXT` and `KUBECTL_NS` and fail closed when either is absent (there is no default
  namespace); `poll_run.sh` also needs `$AUTOTUNE_LOG_POD`. Every manual command is
  `kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" ...`; never rely on or mutate the process-global
  current context.
- **Deletes go through `teardown_run.sh` only.** Never run `kubectl delete` yourself. `teardown_run.sh
  <run_id>` deletes a run's RayJob and ConfigMaps and refuses a run that lacks the operator's owner label
  (`<AUTOTUNE_OWNER_LABEL>=<AUTOTUNE_USER>`) or the run-id prefix. Never touch other users' jobs or pods.
- **`kubectl exec` and `kubectl cp` only against `$AUTOTUNE_LOG_POD`, with read-only commands** (`ls`,
  `cat`, `tail`, `grep`). Never exec into training pods or write inside a pod. When a step needs a write
  or an in-pod action (removing a stale directory, a stack dump, a node check), ask the operator.
- **`make_run_yaml.py` edits only run_id + training CLI flags** (`--set/--unset/--add`, `--benchmark`,
  `--with-profiler`), plus the hook chain, `AUTOTUNE_SEQ_LEN` and the structural env vars of
  `scripts/env_rules.py` (`--env NAME=VALUE`: `CUDA_DEVICE_MAX_CONNECTIONS`, `PYTORCH_CUDA_ALLOC_CONF`, `NCCL_ALGO`,
  `NCCL_PROTO`, debug, dump and `SLIME_PROFILE_*` profiler switches). It does NOT touch `image:` or `nodeAffinity`.
  - **The canonical-manifest rule.** A run may differ from its baseline manifest only in training CLI flag tokens
    and in those structural env vars. Keep other env vars (Transformer Engine `NVTE_*` backend selectors, TF32 and
    determinism switches, anything unclassified), script lines, modules, other pod-template or object changes (pod
    affinity included), data, checkpoint, seed and batch flags as the baseline has them, put no shell syntax in a
    flag token, and do not set flags the layer-1 rules call numeric or unknown. Layer 1 compares resolved arguments,
    not the environment: env vars a frozen-arguments file freezes are checked before launch (`make_run_yaml.py`),
    and the rest rely on this rule. Edit a baseline only to set up a new workload (env vars in the pod `env:` blocks, not
    `export`, or Ray actors don't see them; image pinned by digest).
  - **What scores a run.** A run is scored from its job's train.log over the frozen window, so tear down a running
    job only to abort it: a run torn down early is scored only if its log holds a complete window.
  - **`--benchmark` does NOT enable the profiler** (fixed-shape contract only): a benchmark run is unprofiled
    unless you add `--with-profiler`, which adds the profiler flags and the per-stage trace hook. Generate runs
    any other way and you must reproduce the contract yourself.
  - **`make_run_yaml.py` only edits flags embedded in the YAML entrypoint command.** If the baseline mounts
    a `run-training.sh` via ConfigMap, the tool has nothing to edit and silently applies neither
    `--benchmark` nor the profiler. So either inline the training command in the YAML entrypoint, or keep
    the mounted script but stamp `--num-rollout N` + the workload GBS/RBS (and the profiler flags when you want a
    trace) onto every generated run.
- **Strip any `NCCL_PROTO` pin from an inherited baseline** — auto-select beats it, and `NCCL_PROTO=Simple`
  on EFA hangs PP>1. (`references/ledger.md`, anti-patterns.)
- **No checkpoints in benchmark runs** — drop `--save/--load/--save-interval`; `--ref-load $TORCH_DIST`
  warm-starts. TB-scale MoE checkpoints fill the shared filesystem fast.
- **Pin the image by digest, not tag** — floating tags re-tag silently; a head/worker digest mismatch wedges
  workers at `Init:0/1`. If the baseline YAML names a tag, ask the operator to pin its digest
  (`crane digest <image>` or `skopeo inspect docker://<image>` resolves one). Source patches must be baked into
  the image (Dockerfile), not sed'd in run-training.sh (Ray actors load the image, not the head pod).
- **The image must carry slime's PyTorch profiler controls for a profiled run to write a trace.** Verify by
  reading the image's source, not by grepping for a flag literal — CLI flags are often dataclass-derived
  (`profile_step_start` → `--profile-step-start`), so a flag grep gives false negatives. Check the source
  instead (e.g. `use_pytorch_profiler` in slime's `profile_utils.py`, levers.md "Profiler operability").
  Genuinely absent ⇒ profiled runs on that image write no trace; ask the operator for an image that carries
  the controls (`failure_triage.md`, "Building a patched image").
- **Logs are read through the operator's log-access pod** `$AUTOTUNE_LOG_POD`, a long-lived pod with the
  shared volume mounted (`measurement.md` §1). If it is missing, ask the operator.
- **Cluster credentials** may expire mid-run: treat poll `api-error` as retry, not absence, and report a
  persistent one to the operator.

## Scripts (`--help` for full usage; the cluster-facing ones need `AUTOTUNE_KUBECTL_CONTEXT` and `KUBECTL_NS`)

Script paths are relative to this skill's directory, `<this-skill-dir>` (the base directory Claude Code reports
when it loads the skill): run them as `python3 <this-skill-dir>/scripts/<name>.py` and
`bash <this-skill-dir>/scripts/<name>.sh`, with arguments rather than inline `AUTOTUNE_*=` settings (the campaign's
environment comes from the launcher).

**Analysis tools are aids; scoring tools are frozen.** `analyze_profile.py`, `check_gpu_capacity.py` and
`references/print_topology.md` are there to help you diagnose a run: copy any of them (`analyze_profile.py` with
its `profile_analyzer/` package) into your campaign directory and change or replace them there if that serves the
next decision (your copy lives and dies with this campaign). Run your own helpers as `python3 <file>` or
`bash <file>` with the file inside the campaign directory (not `-c`, `-m` or stdin). What a recipe SCORES, and
whether it ships, never comes from your copy: `read_metrics.sh`, the correctness check
(`correctness/pair_from_runs.py`, `correctness/parity_verdict.py`, and `correctness/make_parity_arm.py` for
diagnostic arms), `run_gpuh.py` and `calibration.py` are frozen
and authoritative, so only their results decide the shipped recipe. Leave the skill
directory unchanged except for the two records you add to it, the only writes the unattended harness allows
there: lever rules appended to `references/levers.md` (L4c) and the closeout summary
`past_tuning_projects/<project_tag>_summary.md`.

Listed in the order you use them in a cycle — what each does and when to call it:

- **`check_gpu_capacity.py [instance-types…] [--baseline <yaml>]`** — reports GPU nodes
  free/occupied/broken/cordoned/tainted for your instance types (the arguments, else `$AUTOTUNE_INSTANCE_TYPES`;
  one of the two is required); `--baseline` treats the taints the run's pods tolerate as free. Call at **loop step
  L3** before launching: need free ≥ nodes_needed, else queue (don't shrink to fit).
- **`make_run_yaml.py --base <yaml> --run-id <id> --out <yaml> [--set/--unset/--add FLAG=VAL] [--benchmark] [--with-profiler]`**
  — clones the baseline YAML into a new run, editing only run_id + training CLI flags (never
  image/nodeAffinity; `--benchmark` adds `AUTOTUNE_SEQ_LEN` and the hook chain's mount, `PYTHONPATH` prefix and
  `ATTUNE_HOOKS` to the pods; `--env` sets a structural env var). Call at **loop
  step L3** to generate each experiment; `--benchmark` applies the fixed-shape contract from
  `scripts/workload.py` (GBS/RBS, `--num-rollout N`, dataset; the profiler's setup, warm-up and capture steps
  after the measured window only with `--with-profiler`). In-pod hooks
  (profiler rank selection, correctness probe) load through one sitecustomize chain
  (`scripts/hook_chain.py`). Another `sitecustomize.py` goes after the chain's directory on `PYTHONPATH`, never in
  front of it; the chain runs it once after its own hooks, and that file must then run the image's own
  `sitecustomize.py`. Never put an ATTUNE hook's source directory on `PYTHONPATH`.
- **`poll_run.sh <run_id>`** — reports a RayJob's status
  (`setup/running/stuck/errored/done/done-unknown/absent/api-error`), the newest `log_path` and a recommendation
  (`wait/triage/teardown/retry`). Call **every ~10 min** for each `tracked_jobs.jsonl` entry, then act on status
  per the state contract.
- **`read_metrics.sh <log> <start> <end> <num_gpus>`** — parses `train.log`, aggregates `tok/s/GPU`
  over the step window (default 5–19) with a validity gate (`valid:false` / non-zero exit on a bad or
  incomplete window). Call at **loop step L4** on completion to get the ranked number.
- **`analyze_profile.py --profile-dir <dir> [--top-n N] [--json]`** (an aid you may modify in your campaign
  dir) — per-rank step decomposition, per-stage pipeline and host-side memory-management (allocator cleanup)
  facts for the profiler traces in `<dir>`. Traces land at
  `$AUTOTUNE_EXPERIMENTS_DIR/<run_id>/profiles/*.pt.trace.json.gz`, one per pipeline stage; copy all of them off
  the log-access pod first (full fetch-and-run recipe: `measurement.md` §6). Call at **loop step L4** on a
  profiled run (`--json` fills the ledger's `profile_top_categories`). Descriptive only — map the dominant
  category to a lever via `levers.md` §c + `measurement.md` §6; it does not recommend flags.
- **`correctness/pair_from_runs.py --base-log <log> --base-yaml <yaml> --win-log <log> --win-yaml <yaml> --pair <dir>`** and
  **`correctness/parity_verdict.py <pair_dir>`** — make a correctness-check pair from two finished unprofiled
  benchmark runs and score it on layers 1 and 2. Call at the final cascade (When to stop); protocol in
  `scripts/correctness/PROTOCOL.md`. **`correctness/make_parity_arm.py`** makes a 20-step probe arm (count probe on
  every pod) when a failing pair needs diagnosing.
- **`calibration.py [--require]`** — the band b and the baseline denominator from `$AUTOTUNE_CALIBRATION`
  (Verdict §; the file is optional, and is refused unless its `num_steps` and `window` stamps match the campaign's
  workload); call it once before the first verdict. `calibration.py build` writes a stamped file from >= 5 baseline
  replicates; `calibration.py stamp` records the stamps of a file written before they existed.
- **`run_gpuh.py <run_id> [--launch --gpus N | --record --status S] | --total`** — GPU-h a run has
  consumed (`num_gpus x` admission -> job end: the Kueue Workload's admission time when the cluster
  runs Kueue, else the RayJob's start time; running jobs charged to now) and the campaign's
  `gpuh.jsonl`. `--launch` after submit, `--record` in finalize before teardown; required in GPU-hour
  budget mode.
- **`teardown_run.sh <run_id>`** — deletes the run's RayJob and ConfigMaps and frees cluster resources; the
  only delete path. It refuses a run without the operator's owner label or the run-id prefix. Call **after** a
  completed run's metrics validate and its ledger row is appended; only once it exits 0 do you remove
  the run's `tracked_jobs.jsonl` line (state contract).

## References (`references/<name>.md`)

One topic each:
- **`glossary.md`** — TP/PP/EP/ETP/CP/DP, mt, bs, vpp, MFU, IQR, slime. Read first if new to the terms.
- **`levers.md`** — **which flags to tune, what they do, when they help or hurt, and the authoritative
  upstream docs** (Megatron MoE playbook, Ultrascale, NeMo, arguments.py). Priority ladder, MoE
  parallel folding + intranode rule, profile→bottleneck table, lever catalog. Read before forming hypotheses.
- **`measurement.md`** — fixed-shape data contract, log locations + log-access pod, `perf N:` fields,
  the aggregate metric, sanity checks, profiling + NCCL decoder, compile-tail-vs-stall, hang diagnosis.
- **`ledger.md`** — ledger entry schema + anti-patterns checklist.
- **`print_topology.md`** — topology-placement verification from the Megatron source and NCCL logs.
- **`failure_triage.md`** — crash/hang signatures and fixes, and what a patched image needs (built FROM the baseline's image, with the operator's approval).
