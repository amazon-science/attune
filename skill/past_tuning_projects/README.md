# past_tuning_projects

The skill writes one `<project_tag>_<campaign_id>_summary.md` here when a tuning campaign stops (`SKILL.md`,
"Whenever you stop"; the campaign id is `$ATTUNE_CAMPAIGN_ID`, so two campaigns on one project never share a
name), and reads every `*_summary.md` here as priors at loop step L1. This is how the skill carries experience from
one campaign to the next.

The directory ships empty. A summary records one campaign's model, hardware, baseline and winner
throughput, what won, tied and lost, gotchas, untried ideas, and the absolute path of the campaign's
artifacts. Those are operator-local facts: keep summaries out of any shared or published copy of the
skill, and move the generalizable rules they contain into `references/levers.md` (mechanism,
condition, direction; no exact numbers, model or cluster names, or run ids).

Under the launcher this is the campaign's copy of the directory: it takes new files, and the files it held at
launch stay unchanged (the launcher's integrity check). Copy a campaign's summary back into your tree to keep it.
