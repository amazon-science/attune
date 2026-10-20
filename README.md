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
  pipeline bubbles, memory, kernel launches) and the lessons in the lever catalog, `skill/references/levers.md`.
  Each measured lesson carries a scope tag naming the workload, model class and GPUs it was measured on. A campaign
  reads the catalog and never edits it: it writes the lessons it learns, with claim, scope, evidence and conflicts,
  to its own `LESSONS.md`. When a session ends, the launcher collects that file into the operator's inbox, if one is
  set and the campaign passed its integrity check. A curation pass, run when the operator chooses, admits the few
  that are correct and general as one catalog change, checked by an independent reviewer agent and approved by a
  human (`harness/CURATION.md`).
- **Measurement.** Every candidate runs the same fixed-shape workload for 20 steps, scored on steps 5 to 19, and
  replaces the leader only if it beats it by more than the noise band.
- **Correctness.** A configuration check refuses changes to the training objective (data, batch, schedule, optimizer,
  loss, precision, token dropping, routing), known stack defects and same-math swaps whose component test failed,
  accepts layout, packing and tested same-math swaps, and lists every other change for human review. A paired
  20-step run then requires finite losses and the candidate's step-0 loss within 5% of the baseline's;
  Megatron-Bridge's `validate_convergence` on the loss curve is reported for review.
  `skill/scripts/correctness/PROTOCOL.md` has the rules.
- **Autonomy.** A Stop hook keeps the agent working until the budget is spent or it records a stop with evidence,
  and a ledger lets a crashed session resume. The launcher gives each campaign read-only copies of the skill and
  harness and records the hashes of the files a campaign must keep: the lever catalog, the scoring, measurement and
  correctness tools, the campaign's settings and prompts, and the append-only ledger and GPU-hour log. A change to
  any of them fails the campaign, and no recipe ships from it. A small PreToolUse guard keeps `kubectl` inside the
  campaign's context, namespace and runs.

## Requirements

Python 3.10 or later with numpy and PyYAML, and pytest for the tests; the harness hooks also run on Python 3.7. The
row-building tools in `skill/scripts/data/` also need pyarrow and transformers.

## Quick start

1. Run the tests, which need no cluster:
   `python3 -m pytest harness/tests skill/scripts/profile_analyzer/tests skill/scripts/tests`
2. Copy `config.example.env` to `config.env`, fill in your cluster, storage, image and identity settings and a run
   prefix for this campaign (`ATTUNE_RUN_PREFIX=<user>-<tag>-`), and load it with `set -a; . config.env; set +a`.
3. Prepare a working, untuned baseline recipe (`ATTUNE_BASELINE_YAML`), rows packed to the workload length
   (`skill/scripts/data/` builds and checks them), and a pod that mounts the shared volume read-only to read run logs
   (`AUTOTUNE_LOG_POD`; `skill/references/measurement.md` §1).
4. Class your stack's arguments in `skill/scripts/correctness/rules/numerics_list.yaml`; an unlisted argument is
   unknown, so layer 1 passes it and lists it for review. `rules/README.md` documents the formats.
5. Launch: `harness/launch_campaign.sh <campaign_dir> [goal_prompt.txt]`, with `AUTOTUNE_GPUH_BUDGET` (GPU-hours) or
   `AUTOTUNE_BUDGET` (ranked runs). Rerun with `--session <id>` to resume; the campaign keeps the settings it started
   with. Edit `goal_prompt.txt` and `resume_prompt.txt` in the campaign directory only between sessions, before a
   relaunch: a change while a session runs fails the campaign. The project folder's `RESULTS.md` holds the
   leaderboard, and the final report names the shipped recipe. The launcher exits 3 when the campaign fails its
   integrity check (`<campaign_dir>/.attune/integrity_violation.json`), after tearing down the campaign's tracked runs.
   When a session ends, the launcher warns if this tree's `skill/` or `harness/` changed during it and, with
   `ATTUNE_LESSONS_INBOX` set, copies an intact campaign's `LESSONS.md` there for curation (`harness/CURATION.md`).

The agent creates and deletes jobs on your cluster and spends GPU-hours until the budget is used or it stops. The
guard prevents accidents to ATTUNE's own records and to the cluster scope: every `kubectl` call names the campaign's
context and namespace, changes only objects named with the campaign's run prefix, and reaches only the campaign's
pods and the log pod, and the protected files refuse edits. It is not a security boundary, since the agent's other
commands run unchecked; credentials scoped to one namespace are. The log pod mounts the shared volume read-only, but
commands the agent runs inside the campaign's own run pods can write to it unchecked. Scores stay safe: finalize
records each run's metrics in the ledger, and the dataset's `rows_sha256` is checked through the log pod at every
launch and before a recipe ships. Use namespace-scoped credentials, keep storage the campaign must not change
read-only to the agent, and start with a small budget. Web access is on by default; evaluation campaigns turn it off
with `AUTOTUNE_ALLOW_WEB=0`. A campaign's LLM cost was about $60 to $110 at list price in our evaluation.

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
harness/                unattended runner: launcher, integrity record, Stop and PreToolUse hooks, lesson export,
                        curation guide (CURATION.md)
config.example.env      every setting, one line each
```

## Contributing

Bug reports, fixes and new levers are welcome; see [CONTRIBUTING](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). For security issues, see
[CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications).

## License

Apache License 2.0; see `LICENSE` and `NOTICE`. Adapted third-party code is listed in `THIRD_PARTY_NOTICES`.
