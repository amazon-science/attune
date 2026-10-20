# Curating campaign lessons

Campaigns never edit the lever catalog, `skill/references/levers.md`. Each campaign writes the lessons it proposes
to `<project_folder>/LESSONS.md` (`skill/SKILL.md` L4(c)), and its copy of the catalog is read-only: any change to
it fails the campaign. When a session ends and the campaign passes its integrity check, `launch_campaign.sh` runs
`harness/lessons.py`, which copies each project folder's file to
`$ATTUNE_LESSONS_INBOX/<campaign id>/<folder>/LESSONS.md` and writes a `manifest.json` beside the folders: campaign
id, model, GPU, image and framework as the campaign's settings recorded them at its first launch, catalog version,
campaign directory, export time, and for each copy its folder, source path and sha256. A later session of the same
campaign replaces them. With `ATTUNE_LESSONS_INBOX` unset the lessons stay in the campaign folder, and so do those of
a campaign that failed its integrity check, whose records cannot be trusted. The inbox must be an absolute path
outside the campaign directory and outside the ATTUNE tree's `skill/` and `harness/`, which the launcher copies into
every campaign. The catalog changes only through the pass below.

## The curation pass

The operator decides when to run a pass; nothing triggers one. Its inputs are the inbox entries not yet curated,
each campaign's records at the manifest's `campaign_dir` (ledger, `RESULTS.md`, `data_insights.md`, profiles), the
current catalog in the ATTUNE tree, and the framework source the lessons cite. Keep a campaign's directory until
its lessons are curated.

1. **Curator.** A fresh agent session on the ATTUNE tree, outside the launcher and every campaign's skill copy,
   checks each lesson against its campaign's records, applies the admission rule, and proposes one diff to
   `levers.md` with a note per lesson: admitted, merged into an existing entry, or dropped, and why.
2. **Reviewer.** A separate agent session, given the diff, the notes, the inbox entries and the records but not the
   curator's session, checks every change against the records and the rule and lists what it would block. The
   curator revises until the reviewer has no blocking finding.
3. **Approval.** A human reads the diff and the review and approves it or sends it back.
4. **Landing.** The approved diff lands as one commit that changes `levers.md`; that commit, with the file's
   sha256, is the new catalog version. Its message names the campaign ids it curated. Move those entries out of the
   inbox (for example to `<inbox>/curated/<commit>/`), so the next pass sees only new ones; an entry that reappears
   after a resumed session is compared by sha256 with its curated copy.

## Admission rule

A lesson is dropped unless it meets every condition:

- **Correct on its records.** The runs it cites exist in the campaign's ledger and show what it claims.
- **General.** It holds across models or campaigns, or its mechanism is verified in the framework source or the
  logs and its condition is stated. A framework limit verified in source counts, scoped to the version.
- **Measured.** It rests on a clean measurement above the noise band, or on a verified hard failure (an OOM, an
  assertion, a crash with its cause).
- **Scoped.** Its workload, model class and GPUs are stated.
- **New.** The catalog does not already say it. A lesson that narrows or corrects an entry amends that entry
  rather than adding one.

Keep the catalog short. An admitted entry states the mechanism, condition and direction, with a magnitude at most
as a word or a rough range, and ends with the scope tag *Scope: workload; model class; GPUs.* It carries no model or
cluster names, run ids or exact numbers. The evidence stays with the review notes, which are operator records like
the lessons themselves and stay out of any shared copy of the tree.

## Re-probing when the campaign model changes

The catalog holds what the campaign model does not reliably know. When that model changes, probe the raw model, with
no system prompt, tools or catalog, using a neutral question for each catalog claim (one that does not state the
answer) and several samples per question. Grade each answer's key points against the claim, then remove only the
points every sample covers; text for a point no question asked, or that no answer actually covered, stays. Probe the
archived removals the same way and restore what the new model does not know. Archive each removal with its scores in
the review notes. The cut lands through the reviewer, approval and landing steps of a curation pass.

## Catalog versions

The manifest names the catalog version a campaign's lessons were written against, once for the whole file:
`catalog_version.levers_sha256`, the hash the launcher sealed for the campaign's `levers.md`, and, when `ATTUNE_HOME`
was a git checkout, its `commit` and `dirty`, true when its `skill/` or `harness/` had uncommitted changes at the
first launch. A lesson written against an older version is checked against the current catalog, since a later pass
may already have admitted, amended or contradicted what it says.
