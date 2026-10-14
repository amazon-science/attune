# ATTUNE: Agentic Throughput Tuning for LLM Training

ATTUNE turns a coding agent into a training-throughput tuner. Given a working training recipe and a GPU-hour budget,
the agent improves the recipe's tokens/s/GPU for your model, cluster and data, unattended: it runs short benchmark
jobs, profiles them, finds the bottleneck, reads the framework source to learn what a flag does, and changes one
setting at a time. A correctness gate checks that a faster recipe still trains the same model before it ships.

This repository contains the skill (`skill/`): the tuning loop, measurement rules, lever catalog, helper scripts and
the correctness gate; and a harness (`harness/`) that keeps a campaign running unattended. The scripts and harness
target Megatron-LM + slime recipes run as RayJobs on Kubernetes and driven by Claude Code; other stacks and agent
harnesses work with adapted launch, metrics and hook scripts.

[Paper](https://arxiv.org/abs/2610.17528)

## Results

On four models held out from its development, each starting from the strongest public recipe:

- ATTUNE raises mean throughput by 6.7% on Llama-3.1-405B and 33.9% on GLM-4.5 (11.9% and 54.7% in its best
  campaigns), and keeps Mixtral-8x22B's recipe, where nothing it finds clears the noise band.
- Under a common GPU-hour budget it beats OptiCo, Bayesian optimization and random search on GLM-4.5 and MiniMax-M2,
  and on both models the tuning pays for itself within a day of training.
- The correctness gate rejects no legitimate run and catches 16 of 24 faulty runs.

## How it works

- **Search.** The agent proposes each candidate from a diagnosis of the current leader (exposed communication,
  pipeline bubbles, memory, kernel launches) and the lessons in the lever catalog, `skill/references/levers.md`, which
  it extends as a campaign learns.
- **Measurement.** Every candidate runs the same fixed-shape workload for 14 steps, scored on steps 4 to 13, and
  replaces the leader only if it beats it by more than the noise band.
- **Correctness.** A configuration check classes every changed argument as numeric, placement or an allowlisted
  same-math swap; a 20-step convergence check compares the candidate's loss with the baseline's using Megatron-Bridge's
  `validate_convergence` thresholds. `skill/scripts/correctness/PROTOCOL.md` has the rules.
- **Autonomy.** A Stop hook keeps the agent working until the budget is spent or it records a stop with evidence. A
  PreToolUse hook allows only read-only tools, the skill's scripts, the agent's own analysis scripts and scoped
  `kubectl` calls, and a ledger lets a crashed session resume.

## Requirements

Python 3.10 or later with numpy and PyYAML, and pytest for the tests. The row-building tools in `skill/scripts/data/`
also need pyarrow and transformers.

## Quick start

1. Run the tests, which need no cluster:
   `python3 -m pytest harness/tests skill/scripts/profile_analyzer/tests skill/scripts/tests`
2. Copy `config.example.env` to `config.env`, fill in your cluster, storage, image and identity settings, and load it
   with `set -a; . config.env; set +a`.
3. Prepare a working, untuned baseline recipe (`ATTUNE_BASELINE_YAML`), rows packed to the workload length
   (`skill/scripts/data/` builds and checks them), and a pod that can read run logs (`AUTOTUNE_LOG_POD`).
4. Class your stack's arguments in `skill/scripts/correctness/rules/numerics_list.yaml`; unlisted arguments count as
   numeric. `rules/README.md` documents the formats.
5. Launch: `harness/launch_campaign.sh <campaign_dir> [goal_prompt.txt]`, with `AUTOTUNE_GPUH_BUDGET` (GPU-hours) or
   `AUTOTUNE_BUDGET` (ranked runs). Rerun with `--session <id>` to resume. The project folder's `RESULTS.md` holds the
   leaderboard, and the final report names the shipped recipe.

The agent creates and deletes jobs on your cluster and spends GPU-hours until the budget is used or it stops. It
deletes only runs it owns, and the command guard limits what it may run and write, but the guard is a safeguard, not a
security boundary: use credentials scoped to one namespace and start with a small budget. A campaign's LLM cost was
about $60 to $110 at list price in our evaluation.

## Repository layout

```
skill/                  the skill
  SKILL.md                tuning loop, ranking rules, state contract, stop rule
  references/             lever catalog, measurement, failure triage, ledger schema, topology check
  scripts/                launch, poll, metrics, teardown, GPU-hour accounting, profile analysis
    correctness/            the correctness gate
    data/                   fixed-length SFT row builder, loss masks, rollout adapter
    pod_hooks/              in-pod hook loader and registry
  past_tuning_projects/   campaign summaries the skill writes and reads
harness/                unattended runner: launcher, Stop and PreToolUse hooks, command policy
config.example.env      every setting, one line each
```

## Contributing

Bug reports, fixes and new levers are welcome; see [CONTRIBUTING](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). For security issues, see
[CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications).

## License

Apache License 2.0; see `LICENSE` and `NOTICE`. Adapted third-party code is listed in `THIRD_PARTY_NOTICES`.
