# Correctness check

A tuned recipe ships only if it trains the same objective as the baseline. Throughput runs say
nothing about that, so every recipe that ships, structural or numeric, is compared against its
baseline in a short paired A/B and must pass both layers below. The skill runs this check as
the final cascade (`SKILL.md`, "When to stop").

## Pair setup

- One baseline side and one candidate side, **20 training steps each** (steps 0-19).
- Same frozen benchmark data (bytes and order), same seed, same global batch size (the benchmark GBS), and the same
  training image for both sides.
- **The default reuses the benchmark runs.** Under the workload contract of `make_run_yaml.py --benchmark` the
  baseline's unprofiled benchmark run and the candidate's own unprofiled benchmark run already hold steps 0-19 on
  the same data, seed and GBS, so the check needs no new launch:
  `pair_from_runs.py --base-log <baseline train.log> --base-yaml <its yaml> --win-log <candidate train.log>
  --win-yaml <its yaml> --pair <dir>` checks both runs, then copies the two logs into a pair directory and writes
  each run's GPU count (`gpus.json`) and image (`images.json`). It refuses, before writing anything, a missing log or
  YAML, a profiled run, a run missing any loss of steps 0-19 or `num_rollout` in its argument dump, two runs with
  different `--num-rollout`, and two runs on different images unless `--allow-image-change` says the candidate's
  recipe changes the image. A profiled run is longer, and slime's LR and weight-decay schedules can depend on the
  run length, so it is never a side: a candidate whose ranked run was profiled gets one unprofiled confirm run
  (`SKILL.md`, Verdict) and that run is paired.
- **Probe arms, to diagnose a failing pair.** Make each arm from its ranked run's YAML:
  `make_parity_arm.py --base <run.yaml> --run-id <arm_run_id> --out <arm.yaml>`. It renames the run (the arm id
  follows the run-id and prefix rules of `make_run_yaml.py`, and the arm gets the owner label), sets `--num-rollout 20`, and enables the count probe (`parity_probe/sitecustomize.py`) on every pod
  through the hook chain (`../hook_chain.py`): ConfigMap keys, a read-only volume at `/opt/attune_hooks`, and a
  `PYTHONPATH` prefix (container env, `runtimeEnvYAML`, and any `export PYTHONPATH=` in the training script).
  Everything else stays as in the source YAML, so the two arms differ only by the recipe. `PARITY_PROBE=0`
  disables the probe; benchmark runs never load it (`make_run_yaml.py --benchmark` removes it from
  `$ATTUNE_HOOKS`).
- The chain's loader runs the next `sitecustomize.py` after its hooks (normally the image's own), so the probe
  combines with it.
- Both logs must contain Megatron's resolved-argument dump and the per-step `train/loss` and
  `train/grad_norm` lines (standard slime/Megatron output).
- One baseline run serves every candidate tested in the cascade.

The probe prints one line per training step:

```
PARITY_COUNTS rollout <r> step <s>: {"rows": R, "tokens": T, "loss_tokens": L, "microbatches": M, "basis": B}
```

Counts are summed over data-parallel replicas from pre-context-parallel-slicing data, on the first
pipeline stage, TP rank 0 and CP rank 0, so they do not depend on the parallel layout. The probe never
raises into training; on any internal error it prints `PARITY_PROBE_ERROR` once and disables itself
(its counts are then missing from the report).

## Layer 1: configuration

Diff the resolved arguments of the two runs (`layer1.py`). Every changed name is classified by the frozen rule
lists of the rules dir (`$AUTOTUNE_RULES_DIR`, else `rules/` beside these scripts; formats and an example in
`rules/README.md`): `numerics_list.yaml` (numeric, placement, packing, allowlisted, derived, ignored) and the
setup's directory, the one holding the frozen-arguments file (`layer1/<name>/frozen_args.yaml`), with its
`allowlist.yaml` (the swaps allowed on that setup and their values) and `ct_status.yaml` (its component-test
results), and the directory of the image both arms ran (`images/<digest prefix>/`, from the pair's `images.json`)
with the same two files for swaps tested on that image; a setup entry replaces the image's for its argument. Without
either directory no swap is admitted.

- **Fail** on any change to a numeric name; a name the list does not know is numeric.
- **Fail** on an allowlisted name that both allowlists omit, that moves to a value outside its list, or whose
  component test has not passed in the source that lists it. Layer 2 still applies to allowlisted swaps.
- **Fail** on a derived value (`padded_vocab_size`) that moves to anything the framework would not derive.
- **Placement and packing** changes never fail layer 1; layer 2 is their guard.
- **Missing or unreadable dumps and rules** fail layer 1: an empty resolved-argument dump in either log, two dumps
  that share fewer than 90% of their argument names, or an unreadable `numerics_list.yaml`.
- **Frozen arguments.** When `$AUTOTUNE_FROZEN_ARGS` names a frozen-arguments file
  (`frozen_args.py`), layer 1 also fails if either arm's resolved arguments violate it, its couplings
  (`couplings.py`) included, and an unreadable file fails layer 1. The same file is checked before launch
  (`make_run_yaml.py`).

## Layer 2: convergence

The candidate's per-step training loss over steps 0-19 (all 20 present in both arms) is compared with the
baseline's by NVIDIA Megatron-Bridge's `validate_convergence`, vendored verbatim in `mbridge_convergence.py`, with
the thresholds Megatron-Bridge's performance pipeline passes:

- the first 20% of steps are skipped;
- point-wise relative tolerance 10% / 5% / 2% for loss above 2.0 / 0.5-2.0 / below 0.5, with at most 10% of the
  compared points outside it;
- final-loss tolerance 5%, correlation at least 0.95, and a 3-sigma outlier check.

The check is tuned to avoid false rejects: it catches gross breakage and can miss subtle faults. The verdict also
reports, without gating, whether the probe's `rows`, `tokens` and `loss_tokens` are identical at every step (the
probe counts only chunk 0's `get_batch` calls, so counts are per microbatch and independent of VPP and PP), the
step-0 loss and grad-norm relative differences, and the 20-step mean loss gap.

## Decision rule

- A recipe ships only if it passes layers 1 and 2.
- On failure, test the next-best ranked recipe of the same campaign that is eligible to ship
  (`SKILL.md` Invariants) against the same baseline run. If none passes, ship the baseline.
- Infrastructure failures (hangs, NCCL init, node loss, image pull) are relaunched and give no
  verdict; any other failure is reported.
- Fix thresholds, windows and scope before looking at results, and never change them afterwards.

## Validating the check

Before relying on the gate for a new setup:

1. **Probe validation:** run the baseline twice. Counts must be identical at every step, and the
   step-0 loss must agree within the setting's rerun spread (step 0 can be nondeterministic).
2. **Negative controls:** run known-bad variants of the baseline and confirm the check flags each, for example `--grad-reduce-in-bf16`, an expert capacity factor that drops tokens,
   learning rate x10, per-token loss off, a larger aux-loss coefficient, and a global batch that
   drops rows.

## Running the verdict script

```
<pair>/base.log     # baseline side train.log
<pair>/win.log      # candidate side train.log
<pair>/gpus.json    # {"base": <num_gpus>, "win": <num_gpus>}
<pair>/images.json  # the image each side ran, read from its YAML's containers (never copied from the environment)
```

In a campaign, copy each run's `train.log` off the log-access pod, then let `pair_from_runs.py` build the pair:

```bash
kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" exec "$AUTOTUNE_LOG_POD" -- cat <baseline run log> > <dir>/base_run.log
kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" exec "$AUTOTUNE_LOG_POD" -- cat <candidate run log> > <dir>/win_run.log
python3 <this-skill-dir>/scripts/correctness/pair_from_runs.py --base-log <dir>/base_run.log --base-yaml <baseline yaml> \
    --win-log <dir>/win_run.log --win-yaml <candidate yaml> --pair <pair>
```

For diagnostic probe arms, `make_parity_arm.py --pair <pair> --arm base|win` records each arm's image instead.

Then score it (`<this-skill-dir>` is the skill's directory, `skill/` in this repository):

```bash
python3 <this-skill-dir>/scripts/correctness/parity_verdict.py <pair> [--json verdict.json]   # one pair
python3 <this-skill-dir>/scripts/correctness/parity_verdict.py <root> [--json verdicts.json]  # every pair under <root>
```

For each pair it prints PASS or FAIL with the layer-1 and layer-2 results, what tripped layer 1 (errors, numeric
changes, allowlist, derived and frozen violations), the failed layer-2 metrics, whether the probe counts are
identical (when the probe ran), the step-0 deltas and the 20-step gap. With `gpus.json`, throughput printed alongside
is tokens over logged step time per GPU on steps 5-19 (probe tokens when the probe ran, else the logged tokens), for
orientation only. It exits 0 when every pair passes, 1 when any pair fails
or cannot be scored, and 2 when it finds no pair.
