# ATTUNE: Agentic Throughput Tuning for LLM Training

ATTUNE turns a coding agent into a training-throughput tuner. Give it a working training recipe and a GPU-hour
budget: it runs short benchmark jobs, profiles them, and changes one setting at a time to make the recipe faster. A
correctness gate checks that a faster recipe still trains the same model before it ships.

ATTUNE's recipes run in Amazon's production post-training, where they raise per-GPU throughput by 1.7× on average.
[Paper](https://arxiv.org/abs/2610.17528)

It works out of the box with Megatron-LM + slime recipes running as RayJobs on Kubernetes, driven by Claude Code.
Other stacks and agents need their own launch, metrics and hook scripts.

## Requirements

- Claude Code CLI (`claude`)
- Python 3.10 or later with numpy and PyYAML (pytest for the tests; pyarrow and transformers for the data tools)
- A Kubernetes cluster with KubeRay, and credentials scoped to one namespace

## Quick start

1. Run the tests (no cluster needed):
   `python3 -m pytest harness/tests skill/scripts/profile_analyzer/tests skill/scripts/tests`
2. Configure: `cp config.example.env config.env`, fill in your cluster, storage, image and run prefix, then
   `set -a; . config.env; set +a`.
3. Point ATTUNE at your baseline recipe (`ATTUNE_BASELINE_YAML`), your dataset (`AUTOTUNE_DATASET`, `AUTOTUNE_GBS`)
   and a read-only pod for run logs (`AUTOTUNE_LOG_POD`).
4. Launch a campaign: `AUTOTUNE_GPUH_BUDGET=100 harness/launch_campaign.sh <campaign_dir>`. Resume with
   `--session <id>`. The leaderboard is in `RESULTS.md`, and the final report names the shipped recipe.
   The header of `harness/launch_campaign.sh` lists every option.

To use the skill interactively instead, copy `skill/` to `.claude/skills/autotune-training-throughput/` in your
project and ask Claude Code to tune your training job.

On a different stack, list which of its arguments change training in
`skill/scripts/correctness/rules/numerics_list.yaml` (see `rules/README.md`), so the gate knows what to refuse.

## Before you run it

The agent creates and deletes jobs on your cluster and spends GPU-hours until the budget runs out. Use credentials
scoped to one namespace, keep storage the agent must not change read-only, and start with a small budget. The
built-in guard prevents accidents, not attacks: namespace-scoped credentials are the real boundary. A campaign's LLM
cost was about $60 to $110 at list price in our evaluation.

## How it works

- **Search.** The agent diagnoses each run (communication, pipeline bubbles, memory, kernel launches) and picks the
  next change with the help of the lever catalog, `skill/references/levers.md`.
- **Measurement.** Every candidate runs the same workload and replaces the leader only if it is faster by more than
  a 3% noise band.
- **Correctness.** A configuration check refuses changes to the training objective and known stack defects, and a
  paired run checks the step-0 loss (`skill/scripts/correctness/PROTOCOL.md`).
- **Autonomy.** A Stop hook keeps the agent working until the budget is spent, a ledger lets a crashed session
  resume, and a guard keeps `kubectl` inside the campaign's namespace.

## Repository layout

```
skill/                  the skill
  SKILL.md                tuning loop, ranking rules, state contract, stop rule
  references/             lever catalog, measurement, failure triage, ledger schema, topology check
  scripts/                launch, poll, metrics, teardown, GPU-hour accounting, profile analysis
    correctness/            the correctness gate
    data/                   fixed-length SFT row builder, loss masks, rollout adapter, token-length analysis
    pod_hooks/              in-pod hook loader and registry
  past_tuning_projects/   campaign summaries the skill writes and reads
harness/                unattended runner: launcher, integrity record, Stop and PreToolUse hooks, lesson export,
                        curation guide (CURATION.md)
config.example.env      every setting, one line each
```

## Contributing

Bug reports, fixes and new levers are welcome; see [CONTRIBUTING](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). For security issues, see
[CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications).

## Citation

```bibtex
@misc{fan2026attune,
  title         = {ATTUNE: Agentic Throughput Tuning for LLM Training},
  author        = {Leo Fan and Hongyang Wang and Wenbo Chen and Jingguo Liu and Xian Li and Xuan Wang and Ruochen Wen and
                   Yiming Yan and Hao Wu and Yichuan Li and Qian Qu and Zheshen Wang and Benoit Dumoulin},
  year          = {2026},
  eprint        = {2610.17528},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2610.17528}
}
```

## License

Apache License 2.0; see `LICENSE` and `NOTICE`. Adapted third-party code is listed in `THIRD_PARTY_NOTICES`.
