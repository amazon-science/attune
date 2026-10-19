# Correctness check

A tuned recipe ships only if it trains the same objective as the baseline. Throughput runs say
nothing about that, so every recipe that ships, structural or numeric, is compared against its
baseline in a short paired A/B and must pass layer 1 and the step-0 check below. The skill runs this
check as the final cascade (`SKILL.md`, "When to stop").

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
`rules/README.md`): `numerics_list.yaml` (objective, numeric, placement, packing, allowlisted, derived, ignored) and
the setup's directory, the one holding the frozen-arguments file (`layer1/<name>/frozen_args.yaml`), with its
`allowlist.yaml` (the swaps allowed on that setup and their values), `ct_status.yaml` (its component-test results)
and `defects.yaml` (known stack defects), and the directory of the image both arms ran (`images/<digest prefix>/`,
from the pair's `images.json`) with the same three files for swaps tested and defects found on that image. A setup
swap entry replaces the image's for its argument; the defects of both directories apply.

Each change is refused, accepted or unknown:

- **Refused** (layer 1 fails): a change to an `objective` name (data, sampling, batch, run length, learning-rate and
  weight-decay schedule, the optimizer and the parameters it trains, loss terms, dropout, precision format, token
  dropping, routing semantics); a candidate in a known stack defect state the baseline is not in, where the entry's
  argument takes a listed value and every `when` condition holds on the candidate's resolved arguments, whichever of
  those arguments changed (the verdict prints the entry's reason and evidence); an allowlisted swap whose component
  test is recorded `fail` in the setup's or the image's results; a derived value (`padded_vocab_size`, printed as
  integers) that moves to anything the framework would not derive.
- **Accepted**: placement, packing and ignored changes; a derived value the framework derives from the other
  changes; an allowlisted swap that an allowlist lists with both values and whose component test passed in the source
  that lists it.
- **Unknown** (layer 1 passes): a change to a `numeric` name or to a name the lists do not know, a name only one of
  the two dumps has (two builds; an objective-class one is flagged to check by hand), a derived value either dump does
  not print as an integer (cannot recompute), and an allowlisted swap that no allowlist lists, that moves outside its
  listed values, or whose component test has not passed. The step-0 check judges these, and the verdict lists each
  with its reason for human review.
- **Missing or unreadable dumps and rules** fail layer 1: an empty resolved-argument dump in either log, two dumps
  that share fewer than 90% of their argument names, an unreadable `numerics_list.yaml` or one with an unknown
  top-level key or a name in two classes, an `allowlist.yaml`, `ct_status.yaml` or `defects.yaml` that exists but is
  not a readable file or carries an unknown top-level key, a swap whose values are neither a list nor `any`, and a
  defect entry without its argument, values, reason or evidence. Allowlist
  and defect values and component-test statuses are compared after the same normalization as the dump values
  (case, whitespace, YAML booleans and null).
- **Frozen arguments.** When `$AUTOTUNE_FROZEN_ARGS` names a frozen-arguments file
  (`frozen_args.py`), layer 1 also fails if either arm's resolved arguments violate it, its couplings
  (`couplings.py`) included, and an unreadable file fails layer 1. The same file is checked before launch
  (`make_run_yaml.py`).

## Step-0 check

Both logs must hold the training loss of all 20 steps (0-19), each finite, and the candidate's step-0 loss must lie
within 5% of the baseline's: |loss_0(candidate) / loss_0(baseline) - 1| <= 0.05 (`STEP0_TOL` in
`parity_verdict.py`). Step 0 runs on the same initial weights and the same rows in both runs, so a change that keeps
the objective moves it only by rounding and reduction order, while a defect in weight loading, sharding or layer
mapping moves it far. On 40 ablation campaigns this rule accepted all 328 known-valid runs, which moved the step-0 loss
by at most 1.7%, and would have refused all 19 known defects of that kind, which moved it by +215% to +768%. A missing or
non-finite loss in either log fails the check. Step 0 sees only what changes the forward pass at initialization: a
kernel that rounds differently (a fused router that breaks exact BF16 ties differently) or a defect in the gradient or
optimizer path can pass it, so known defects are refused at layer 1 and the loss curve below is reported for review.

## Loss curve (review)

The verdict also compares the candidate's per-step training loss over steps 0-19 with the baseline's by NVIDIA
Megatron-Bridge's `validate_convergence`, vendored verbatim in `mbridge_convergence.py`, with the thresholds
Megatron-Bridge's performance pipeline passes:

- the first 20% of steps are skipped;
- point-wise relative tolerance 10% / 5% / 2% for loss above 2.0 / 0.5-2.0 / below 0.5, with at most 10% of the
  compared points outside it;
- final-loss tolerance 5%, correlation at least 0.95, and a 3-sigma outlier check.

It also reports the candidate's mean relative loss deviation from the baseline over steps 10-19. Both are for human
review and never decide: on GLM-4.7-Flash, whose loss spikes during its first ten steps, 151 of the 702 ordered pairs
of its 27 baseline-recipe runs fail the curve check against each other. Each verdict's `review` list holds every
unknown change of layer 1 (name, from, to, reason), a failed or unscored curve check, and an error in any review-only
computation, which never changes the verdict. The verdict also reports, without gating, whether the probe's `rows`,
`tokens` and `loss_tokens` are identical at every step (the probe counts only chunk 0's `get_batch` calls, so counts
are per microbatch and independent of VPP and PP), the step-0 grad-norm relative difference, and the 20-step mean
loss gap.

## Decision rule

- A recipe ships only if layer 1 refuses none of its changes and the step-0 check passes. Unknown changes and the
  loss curve are listed for human review and never refuse a recipe.
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

For each pair it prints one PASS or FAIL line with the layer-1 result and the step-0 shift, then what tripped layer 1
(errors, objective changes, known defects, failed component tests, derived and frozen violations) or the step-0
check, then the review lines (each unknown change with its reason, and the loss-curve result with the steps 10-19
deviation), whether the probe counts are identical (when the probe ran), the step-0 grad-norm difference and the
20-step gap. With `gpus.json`, throughput printed alongside
is tokens over logged step time per GPU on steps 5-19 (probe tokens when the probe ran, else the logged tokens), for
orientation only. It exits 0 when every pair passes, 1 when any pair fails
or cannot be scored, and 2 when it finds no pair. Under the launcher it first runs the campaign's integrity check
(`harness/integrity.py`), which also hashes the dataset again through the log pod; on a violation (a changed or
missing dataset included) it scores nothing, no recipe ships, and it exits 3. When the log pod cannot be reached it
scores nothing and exits 4: retry. A failure to write `--json` is reported on stderr and leaves the exit code as
scored.
