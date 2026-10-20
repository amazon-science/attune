# State-file schemas & anti-patterns

## 1. Ledger entry schema (`<project_folder>/AUTOTUNE_LEDGER.jsonl`)

This is the canonical ledger schema; SKILL.md and the harness Stop hook (`harness/convergence.py`) follow it.
Append-only, one JSON object per line, one ledger per project; `scripts/ledger_append.py` appends a row. Under the
launcher a rewritten, truncated or deleted row fails the campaign's integrity check. The file is
`${AUTOTUNE_PROJECT_ROOT:-$PWD/attune-projects}/<project_tag>/AUTOTUNE_LEDGER.jsonl` (project_tag = baseline YAML
basename minus `.yaml`, lowercased), or `$AUTOTUNE_LEDGER` when that is set. `run_id` carries the full
`<prefix><project_tag>-<suffix>` name, identical in the ledger and in K8s — never a bare suffix. (The `exp-NNNN`
values below illustrate only the *suffix* part; a real run_id is e.g. `alice-mymodel-sft-h200-pp8-ep8`. All values
in the example are illustrative placeholders, not results.)

```json
{
  "run_id": "exp-0017-tp1-pp4",
  "kind": "ranked",
  "launched_at": "<ISO-8601 UTC>",
  "base_run": "exp-0014-tp2-pp2",
  "hypothesis": "TP2 to TP1 removes attention-TP AllGather; PP2 to PP4 keeps per-rank memory flat so it fits.",
  "prediction": "<expected delta range, e.g. a win above the band>",
  "config_diff": {"tensor-model-parallel-size": "2 -> 1", "pipeline-model-parallel-size": "2 -> 4"},
  "yaml_path": "<project_folder>/yamls/....yaml",
  "hardware": "<N>x <instance-type>",
  "parallelism_5d": "TP=1 DP=.. CP=1 EP=.. PP=4 vpp=1",
  "status": "completed",
  "objective_class": "structural",
  "verified_from_source": "<file:line per changed flag>",
  "metrics": {"aggregate_tok_per_s_per_gpu": 0.0, "window": [5, 19], "valid": true,
              "step_time_stats": {"mean_s": 0.0, "cv_pct": 0.0, "p50_s": 0.0, "min_s": 0.0, "max_s": 0.0}},
  "verdict": "win",
  "compared_leader_run_id": "exp-0014-tp2-pp2",
  "delta_vs_leader_pct": 0.0,
  "delta_vs_baseline_pct": 0.0,
  "what_i_learned": "<one sentence: did the bottleneck move as predicted?>",
  "profiled": false,
  "profile_reason": "<why this run was or was not profiled>",
  "profile_top_categories": {"NCCL Communication": 0.0, "GEMM/MatMul": 0.0, "Flash Attention": 0.0},
  "profile_top_kernels": [{"kernel": "SendRecv", "pct": 0.0}],
  "next_experiment_suggestion": "exp-0018-..."
}
```

**What the Stop hook reads.** Under the unattended harness the Stop hook decides from these fields:
- The **baseline row** is the row with `kind: "baseline"` and `status: "completed"`. Without it only the GPU-hour
  cap ends the campaign.
- A **ranked row** has `kind: "ranked"`, a `verdict` of `win`, `tie` or `lose`, and a number in
  `metrics.aggregate_tok_per_s_per_gpu`. Ranked-count mode counts each such `run_id` once; `error` and `killed`
  verdicts and every other `kind` count for nothing.
- In GPU-hour budget mode it also sums `gpuh` (or reads `gpuh_cumulative`) and the GPU-hour log `gpuh.jsonl`, and
  accepts a STOP row (below) once a baseline row exists. A STOP row counts only with non-empty `reason`, `leader`,
  `neighbours` and `diagnosis`.

**Field reference** (required unless noted):
- `run_id` — unique in project; suffix for all K8s names; ≤47 chars.
- `kind` — `baseline | ranked | confirm | parity | unranked | stop`. `baseline`: the baseline benchmark run (a failed
  attempt at it keeps `kind: "baseline"` with `status: "errored"`). `ranked`: a candidate benchmark run. `confirm`:
  an unprofiled run made only to give the final correctness check a side its ranked run cannot give: a recipe whose
  ranked run was profiled, or, when ranked runs are shorter than 20 steps, the baseline and each recipe the check tests
  (`make_run_yaml.py --benchmark --confirm`; SKILL.md "Verdict"); not ranked, no budget slot, counted in GPU-hours.
  `parity`: a diagnostic correctness-check arm (`make_parity_arm.py`). `unranked`: a diagnostic run that is not a
  candidate (a longer profile, a feasibility probe). `stop`: the STOP row below.
- `launched_at` — ISO-8601 UTC (ledger-entry time, not pod-up time).
- `base_run` — run_id this is diffed from; `null` for the baseline.
- `hypothesis` — full sentence: what you expect and why. `prediction` — numeric range.
- `config_diff` — only the flags changed vs `base_run`.
- `yaml_path`, `hardware` — provenance.
- `parallelism_5d` — `TP=.. DP=.. CP=.. EP=.. PP=.. vpp=..` (DP derived = `world/(TP×PP×CP)` but written explicitly; `vpp=1` = OFF; note `ETP` only if `ETP≠EP`).
- `status` — `launching | running | completed | killed | errored`.
- `metrics` *(required on a completed benchmark row)* — the JSON `read_metrics.sh` printed for the measured window,
  copied unchanged. `aggregate_tok_per_s_per_gpu` is the ranked number; `window` is the measurement window
  ([5, 19] at the defaults); `step_time_stats` and `means_in_window` carry the step-time distribution and
  the window means of the `perf N:` fields. Add `peak_gpu_mem_gb` when you read it from the log.
- `verdict` *(terminal, on `ranked` rows)* — `win | tie | lose | error | killed`, decided by the band of SKILL.md
  "Verdict" (no throughput rerun of a tie).
- `compared_leader_run_id`, `delta_vs_leader_pct` *(required for a ranked terminal row)* — the leader and delta used for the verdict at completion time. This matters when concurrent jobs finish after another result has already promoted a new leader.
- `delta_vs_baseline_pct` *(required for a ranked terminal row)* — end-to-end gain against the original fixed-allocation baseline.
- `gpuh`, `gpuh_cumulative` *(terminal; required in GPU-hour budget mode, SKILL.md "When to stop")* — this run's GPU-hours (`num_gpus x` admission -> job end, copied from the run's completed `gpuh.jsonl` line (`scripts/run_gpuh.py <run_id> --record`, before teardown; the log is append-only and a run's last line counts); 0 if never admitted) and the campaign total after it, summed over every row: baseline, ranked, confirm, failed, killed and correctness-check (`kind: parity`) runs. One row per run_id carries its `gpuh`.
- `objective_class` *(required on a WIN row)* — `structural` (placement/scheduling only, including packing / `--max-tokens-per-gpu` / `--micro-batch-size` changes, also with a per-microbatch MoE aux loss) `same_math` (a fusion, kernel or dispatcher implementation, optimizer offload or other same-function swap: layer 1 accepts it with a passed component test, refuses it on a failed test or a known stack defect, and otherwise lists it as unknown for review) `numeric` (another change of the math outside the `objective` class: unknown to layer 1, decided by the step-0 check and listed for review) or `objective` (changes an `objective`-class argument of `numerics_list.yaml`; see the objective-preserving Invariant in SKILL.md). An `objective` win is never shipped; any other ships only with `parity_checked:true`.
- `parity_checked` *(required before shipping any recipe)* — `true` only when the correctness check (`scripts/correctness/PROTOCOL.md`: the baseline's and this recipe's own unprofiled benchmark runs, paired by `pair_from_runs.py` and scored by `parity_verdict.py`) PASSes for this recipe vs the baseline. Record both run_ids (each side is its ranked run, or its `confirm` run when the ranked run was profiled or shorter than 20 steps) and the layer-1 and step-0 results with the verdict's review lines in `what_i_learned`. **Do not label a recipe/change "training-neutral" anywhere (recipe header, RESULTS.md, PR) unless this is true.**
- `verified_from_source` *(required whenever a run changes any flag)* — the `file:line`(s) in the running image's `arguments.py`/source where you confirmed each changed flag's resolved behavior (what it sets/asserts, and what it silently overrides). Per SKILL.md loop step L2, no flag is tried un-read; this field records that you read it. Guards against memory/hallucination and silent flag-override (e.g. grad-reduce-in-bf16).
- `profiled` *(required on a terminal row)* — `true` when the run was launched with the profiler (`--with-profiler`), else `false`. Profiling is per launch, so this records which runs carry a trace.
- `profile_reason` *(required on a terminal row)* — one line: why this run was profiled, or why it did not need a trace (e.g. "baseline: first diagnosis", "value sweep on an already diagnosed lever").
- `profile_top_categories` *(required when `profiled` is true and a trace was written)* — `analyze_profile.py --json` output.
- `what_i_learned`, `profile_top_kernels`, `gpu_busy_pct`, `gpu_idle_ms`, `top_bubble_cause`, `skill_updates`, `adhoc_scripts_written`, `next_experiment_suggestion` — *optional* enrichment.

**STOP row** (GPU-hour budget mode, SKILL.md "When to stop"): your convergence declaration, appended once,
after every launched run has its terminal row and nothing is in flight. It is not a run, so it has no
`run_id` and carries no metrics:

```json
{"kind": "stop", "declared_at": "<ISO-8601 UTC>", "stop_reason": "agent_stop", "gpuh_cumulative": 0.0,
 "reason": "<why further search is not expected to beat the leader>",
 "leader": "<run_id of the current leader>",
 "neighbours": [{"run_id": "<run_id>", "axis": "<the structural change vs the leader, e.g. PP4 -> PP2>"}],
 "diagnosis": "<the last profile or log diagnosis: the remaining bottleneck and the levers it rules out>"}
```

`neighbours` lists the structural neighbours of the leader you actually tested (one parallelism or
placement axis moved per run); `diagnosis` names the run it came from. Mirror the same content in a
`## STOP` section at the top of `RESULTS.md`. When the cap ends the search instead, no STOP row is
written; the RESULTS.md header records the cap.

**Terminal discipline:** a `completed|errored|killed` entry is terminal. Per SKILL.md's ordered
transition: validate metrics → append the ledger row with `ledger_append.py` (never overwrite a prior line for the
same run_id) and rewrite `RESULTS.md` → teardown → drop the `tracked_jobs.jsonl` line. Don't decouple the
ledger from `RESULTS.md`.

---

## 2. Anti-patterns to avoid

Read once when onboarding a project; re-read if a decision feels off. (Several of these also
appear as SKILL.md Invariants/gotchas — repeated here as a scan-able checklist.)

**Methodology**
1. **No hypothesis, no run.** "Try FP8" isn't one; "FP8 should help — matmul is now the dominant share of step time, vs negligible when it was last tried" is.
2. **One variable per run.** Changing topology + batch + recompute together tells you nothing about which lever moved the needle. Exception: tightly-coupled flags (call the coupling out).
3. **Don't kill on noise.** Legit early-kills: the benchmark step cap, diagnosed crash, confirmed hang, or a *decisive* loss once step-time has settled. Not "looks slow" hunches — a slow-but-stable run is still a data point.
4. **Failures are signal, not "flaky."** Look a crash or timeout signature up in `failure_triage.md` before relaunching; an NCCL error at init with no memory signal is often GPU OOM, not the network.

**Record-keeping**
5. **Keep `RESULTS.md` in sync with the ledger.** The ledger is the append-only audit trail; `RESULTS.md` is the human's morning view. Decoupling them loses runs between sessions.
6. **One project = one `(model, dataset, hardware)` folder.** Don't mix projects in one ledger — cross-project comparisons become garbage.

**Launching**
7. **Never collide K8s names** — suffix everything with run_id. **Always capacity-check** (`check_gpu_capacity.py`, `free ≥ nodes_needed`) before `kubectl apply`.
8. **Respect the cap** (default 3; the concurrency-cap Invariant in SKILL.md). Below the cap, if ≥3 are running AND free nodes are scarce, leave headroom for others.
9. **No checkpoints in benchmark runs** — drop `--save/--load/--save-interval` (not just `--no-save-optim`). MoE checkpoints are ~TB-scale; a sweep fills the shared filesystem and hangs mid-save.

**Status & measurement**
10. **Never trust `RayJob SUCCEEDED` alone** — confirm `latest_step` reached the run's `target_steps` (19 unprofiled, 22 profiled with one recorded step, at the defaults); a swallowed actor crash reports SUCCEEDED. (`poll_run.sh` flags this.)
11. **Never rerun a throughput tie.** Only a gain above the band (SKILL.md "Verdict") over the current leader is promoted. Record a tie and spend the next slot on a new hypothesis.
12. **Read `train_wait_time` before blaming rollout/topology for a big `step_time_max`** — a p50→max ratio of many times is usually JIT compile-tail, not a stall (misdiagnosing it can wrongly steer away from higher PP).
12b. **Never classify a flag as objective-preserving from memory — read the RESOLVED config in the image's `arguments.py`.** `--grad-reduce-in-bf16` is easy to call "training-neutral" on the belief it only changes the wire dtype; in Megatron (`megatron/training/arguments.py`: `if args.grad_reduce_in_bf16: args.accumulate_allreduce_grads_in_fp32 = False`) it force-nulls fp32 accumulation, silently overriding that flag even when also set → gradients accumulate in bf16 → a potential training-quality regression. Any numeric lever needs source-verification (`verified_from_source` file:line) + a PASS of the correctness check (`scripts/correctness/PROTOCOL.md`) before it's called neutral.

**Config-template hygiene**
13. **Strip `NCCL_PROTO` from inherited YAMLs** — auto-select beats a manual pin, and `NCCL_PROTO=Simple` on EFA hangs PP>1 (easily misread as a hardware fault).
14. **Intranode-MoE placement follows the EP/ETP rank-fold, not `TP × EP`** (the `TP × EP` mnemonic agrees with the rule only when ETP = TP, Megatron's default when ETP is unset). The rule + its source (`parallel_state.py` expert rank generator) is in levers.md §b; confirm the fold order for your image with `print_topology.md` rather than assuming.
15. **When moving a YAML between p5e↔p5en, update BOTH `instance-type` AND `vpc.amazonaws.com/efa`** (p5e=32, p5en=16 NICs) — a mismatch fails NCCL bootstrap with `NET/OFI Unexpected number of remote rails`.
16. **Don't port a winner topology across datasets without re-profiling** — short-seq vs long-seq favor different families (the swing can be large). Characterize the new data with `scripts/data/analyze_token_distribution.py` (SFT: the token-length distribution; RL: `--rl-prompts` on the prompt set, then the first rollouts' `response_len`) → refresh `data_insights.md` → re-sweep.

---


## 3. `tracked_jobs.jsonl` entry schema (`<project_folder>/tracked_jobs.jsonl`)

One JSON object per line, one per **in-flight** run. Append a line when you launch a run; remove it
only after that run's terminal ledger row is written **and** teardown succeeds (see SKILL.md state
contract). The file is transient scratch state — the ledger is the permanent record.

Only `run_id` is read by the scripts to poll and match a run. `target_steps` is a fallback: `poll_run.sh` takes a
run's completion step from its own `--num-rollout` and reads `target_steps` only when neither the log nor the RayJob
shows it (and `TARGET_STEPS` overrides both). Everything else is the agent's own bookkeeping, carried onto the ledger
row at completion.

| field | required | purpose |
|---|---|---|
| `run_id` | **yes** | RayJob name and poll key; `poll_run.sh` matches on it. Unique per project, ≤47 chars. |
| `target_steps` | no | Completion marker: the LAST 0-indexed step a full run reaches (= `--num-rollout − 1`, since steps run `0…N−1`), compared directly against `latest_step` to catch a truncated/false-SUCCEEDED run. At the default step counts (`scripts/workload.py`): 19 for an unprofiled run (benchmark, confirm or 20-step correctness-check arm) and 21+K for a profiled one with K recorded steps (22, 23 or 24). `poll_run.sh` reads the run's own `--num-rollout` first, so this is the fallback when the run's log and RayJob do not show it; env `TARGET_STEPS` overrides both. |
| `kind` | no | `baseline` / `ranked` / `confirm` / `parity` / `unranked`; copied to the ledger row's `kind`, which drives budget accounting (§1). |
| `yaml_path` | no | The run YAML applied (for reproduce / teardown). |
| `hypothesis`, `topology` | no | Agent notes; copied to the ledger row on completion. |
| `launched_at` | no | ISO-8601 launch time; helps spot stuck runs and reconcile drift. |

Extra keys are allowed and ignored by the scripts. Example line:
`{"run_id":"alice-mymodel-cp2","kind":"ranked","topology":"TP4/PP4/CP2/EP16/DP2","target_steps":19,"hypothesis":"CP4->CP2 removes CP all-gather volume"}`
