# past_tuning_projects

The skill writes one `<project_tag>_summary.md` here when a tuning campaign stops (`SKILL.md`,
"Whenever you stop"), and reads every `*_summary.md` here as priors at loop step L1. This is how the
skill carries experience from one campaign to the next.

The directory ships empty. A summary records one campaign's model, hardware, baseline and winner
throughput, what won, tied and lost, gotchas, untried ideas, and the absolute path of the campaign's
artifacts. Those are operator-local facts: keep summaries out of any shared or published copy of the
skill, and move the generalizable rules they contain into `references/levers.md` (mechanism,
condition, direction; no exact numbers, model or cluster names, or run ids).
