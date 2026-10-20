#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# launch_campaign.sh — run one ATTUNE tuning campaign unattended (headless Claude Code + hooks).
#
# Usage:
#   set -a; source config.env; set +a
#   harness/launch_campaign.sh <campaign_dir> [goal_prompt_file] [--detach] [--session SESSION_ID]
#     --session SESSION_ID continues that Claude Code session (e.g. after an outage)
#
# The first launch in <campaign_dir>:
#   1. Checks the settings: the required ones are set, the workload settings and the verdict band are usable
#      (skill/scripts/workload.py, calibration.py), ATTUNE_RUN_PREFIX is campaign-specific (<user>-<tag>-), and
#      the log pod $AUTOTUNE_LOG_POD mounts the shared volume read-only (harness/log_pod.py). It prints the campaign
#      id (ATTUNE_CAMPAIGN_ID, which names the campaign's summary) and the effective web setting.
#   2. Copies $ATTUNE_HOME/skill and $ATTUNE_HOME/harness into the launcher's control directory
#      <campaign_dir>/.attune, writes the campaign's settings there as config.env, and links the skill copy as
#      <campaign_dir>/.claude/skills/autotune-training-throughput. The copies are read-only except
#      skill/past_tuning_projects/ (new summaries); the lever catalog references/levers.md is read-only too, and the
#      campaign proposes lessons in <project_folder>/LESSONS.md instead.
#   3. Renders the copy's settings.template.json into <campaign_dir>/.claude/settings.json (the environment, the
#      PreToolUse guard and Stop hook of the harness copy, the denied tools; WebSearch and WebFetch are denied only
#      with AUTOTUNE_ALLOW_WEB=0) and the goal prompt (${VAR} placeholders filled) into goal_prompt.rendered.txt.
#   4. Records the hashes of the protected files in .attune/integrity.json (harness/integrity.py seal), the source
#      tree with its git commit, whether it had uncommitted changes and the hashes of its skill/ and harness/, and the
#      dataset's rows_sha256, read through the log pod: $AUTOTUNE_DATASET, else the baseline's --prompt-data (its
#      $VAR and ${VAR} expanded from the settings), else $SFT_DATA. A changed or missing dataset fails the campaign
#      at a later launch and before a recipe ships; a log pod that cannot be reached only holds them back.
#   5. Runs `claude -p "<goal>"` in <campaign_dir> with only project/local settings, no MCP servers, no CLAUDE.md or
#      auto memory, no bundled skills (the skill copy is the only one), and, when ATTUNE_MODEL is set, that model on
#      every tier (main, subagents, Opus/Sonnet/Haiku/Fable defaults) with no fallback model. The session's
#      ATTUNE_HOME names the control directory, so a path built from it reaches the read-only copies, not the tree,
#      and it does not see ATTUNE_LESSONS_INBOX.
#   6. Checks the record again when the session ends; on a violation it tears down the runs in tracked_jobs.jsonl.
#      It then warns of any change to the source tree's skill/ and harness/ since launch (integrity.py source-check),
#      and, when the check passed, harness/lessons.py copies the campaign's LESSONS.md files to
#      $ATTUNE_LESSONS_INBOX/<campaign id>/ with a manifest; unset, or after a failed check, the lessons stay in the
#      campaign folder. The warning and a failed copy never change the session's status.
# A later launch in the same directory (--session after an outage, or a new session) checks the record and the
# dataset first, takes the settings from .attune/config.env (an environment value that disagrees is refused; an unset
# or empty one takes the campaign's), keeps the copies, and re-renders and re-records the settings and prompts and
# the source tree's hashes. ATTUNE_LESSONS_INBOX must be absolute and outside the campaign directory and the ATTUNE
# tree's skill/ and harness/.
# Edit goal_prompt.txt and resume_prompt.txt only between sessions: during one, a change fails the campaign.
# The Stop hook keeps the session going until the ledger holds $AUTOTUNE_BUDGET ranked results, or, when
# $AUTOTUNE_GPUH_BUDGET is set, until that many GPU-hours are spent or the agent records a STOP row; it ends the
# session at once when the integrity check fails, after tearing down the tracked runs.
#
# Exit status: 2 on a usage or settings error, 3 when the campaign fails its integrity check (the violation record
# is .attune/integrity_violation.json, with the teardown of the tracked runs), else that of `claude`. With --detach
# the launcher exits 0 once the session starts, and the session writes its own status (3 on a violation) and the
# teardown result to <campaign_dir>/run.exit when it ends.
#
# `claude -p` streams JSON events to run.out (--output-format stream-json); the session transcript is under
# CLAUDE_CONFIG_DIR. Follow progress in <campaign_dir>/<project_tag>/AUTOTUNE_LEDGER.jsonl and `kubectl get rayjobs`.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 2; }
failed() { echo "ERROR: $*" >&2; exit 3; }
USAGE="usage: launch_campaign.sh <campaign_dir> [goal_prompt_file] [--detach] [--session ID]"
[ $# -ge 1 ] || die "$USAGE"
CAMPAIGN_DIR=$1
shift
GOAL_SRC=""
DETACH=0
SESSION=""
while [ $# -gt 0 ]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --session)   # continue that Claude Code session
      [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#-}" = "$2" ] || die "--session needs a session id"
      SESSION=$2; shift 2 ;;
    -*) die "unknown option $1; $USAGE" ;;
    *) [ -z "$GOAL_SRC" ] || die "$USAGE"; GOAL_SRC=$1; shift ;;
  esac
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ATTUNE_HOME=${ATTUNE_HOME:-$(dirname "$HERE")}
mkdir -p "$CAMPAIGN_DIR"
CAMPAIGN_DIR=$(cd "$CAMPAIGN_DIR" && pwd -P)
CONTROL="$CAMPAIGN_DIR/.attune"
VIOLATION="$CONTROL/integrity_violation.json"
export ATTUNE_HOME CAMPAIGN_DIR ATTUNE_CONTROL_DIR="$CONTROL"

if [ -f "$CONTROL/integrity.json" ]; then   # a later launch: the campaign must be intact, its settings are its own
  FIRST=0
  SRC=$CONTROL
  rc=0
  python3 "$CONTROL/harness/integrity.py" check "$CONTROL" --where "launch_campaign.sh (a later launch)" --relaunch \
    --teardown --dataset || rc=$?
  [ "$rc" != 3 ] || failed "the campaign failed its integrity check ($VIOLATION); start a new campaign directory"
  [ "$rc" = 0 ] || die "the campaign's integrity or dataset cannot be verified now (above); retry"
  python3 "$CONTROL/skill/scripts/campaign.py" check-settings \
    || die "the environment disagrees with the campaign's settings (above); unset those values or match them"
  set -a; . "$CONTROL/config.env"; set +a
elif [ -e "$CONTROL" ]; then
  failed "$CONTROL exists without its integrity record, so the campaign cannot be verified;" \
         "start a new campaign directory"
else
  FIRST=1
  SRC=$ATTUNE_HOME
  [ -f "$SRC/harness/integrity.py" ] && [ -f "$SRC/skill/SKILL.md" ] \
    || die "ATTUNE_HOME ($ATTUNE_HOME) has no skill/ and harness/"
fi

for v in AUTOTUNE_USER AUTOTUNE_KUBECTL_CONTEXT KUBECTL_NS AUTOTUNE_NUM_GPUS AUTOTUNE_BUDGET ATTUNE_RUN_PREFIX \
         AUTOTUNE_LOG_POD; do
  [ -n "${!v:-}" ] || die "$v is not set (source your config.env first)"
done
is_count() { [[ $1 =~ ^[0-9]+$ && $1 =~ [1-9] ]]; }
is_number() { [[ $1 =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ && $1 =~ [1-9] ]]; }
for v in AUTOTUNE_NUM_GPUS AUTOTUNE_BUDGET AUTOTUNE_MAX_STALL; do
  [ -z "${!v:-}" ] || is_count "${!v}" || die "$v must be a positive integer, not '${!v}'"
done
[ -z "${AUTOTUNE_GPUH_BUDGET:-}" ] || is_number "$AUTOTUNE_GPUH_BUDGET" \
  || die "AUTOTUNE_GPUH_BUDGET must be a positive number of GPU-hours, not '$AUTOTUNE_GPUH_BUDGET'"
if [ -n "${ATTUNE_LESSONS_INBOX:-}" ] && [ -f "$SRC/harness/lessons.py" ]; then   # older copies export nothing
  python3 "$SRC/harness/lessons.py" check-inbox "$ATTUNE_LESSONS_INBOX" "$CAMPAIGN_DIR" "$ATTUNE_HOME" \
    || die "ATTUNE_LESSONS_INBOX must be an absolute path outside the campaign directory and the ATTUNE tree's" \
           "skill/ and harness/ (above)"
fi
case "${AUTOTUNE_ALLOW_WEB:-}" in
  ""|0|1) export AUTOTUNE_ALLOW_WEB="${AUTOTUNE_ALLOW_WEB:-1}" ;;   # unset or empty: on, recorded as 1
  *) die "AUTOTUNE_ALLOW_WEB is 0 (web access off) or 1 or unset (on), not '$AUTOTUNE_ALLOW_WEB'" ;;
esac
# The guard lets kubectl change only objects named with the run prefix, so it must belong to this campaign alone.
USER_TAG=$(printf '%s' "$AUTOTUNE_USER" | tr '[:upper:]' '[:lower:]')
TAG=${ATTUNE_RUN_PREFIX#"$USER_TAG-"}
if [[ ! $ATTUNE_RUN_PREFIX =~ ^[a-z0-9][-a-z0-9]*$ ]] || [ "$TAG" = "$ATTUNE_RUN_PREFIX" ] \
   || [[ ! $TAG =~ [a-z0-9] ]]; then
  die "ATTUNE_RUN_PREFIX must be campaign-specific: $USER_TAG-<campaign tag>- in lowercase letters, digits and '-'" \
      "(e.g. $USER_TAG-glm45-h100-), not '$ATTUNE_RUN_PREFIX'"
fi
CLAUDE_BIN="${ATTUNE_CLAUDE_BIN:-claude}"
command -v "$CLAUDE_BIN" >/dev/null || die "$CLAUDE_BIN (Claude Code CLI) not found"

# The guard lets any command run in the log pod, so its read-only mount keeps those commands from writing the shared
# volume. Commands in the campaign's own run pods can write it unchecked; scores stay safe because finalize ledgers
# each run's metrics, and the dataset hash is checked at every launch and before a recipe ships.
SHARED_MOUNT=${AUTOTUNE_SHARED_MOUNT:-/shared}
POD_JSON=$(kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" get pod "$AUTOTUNE_LOG_POD" -o json) \
  || die "cannot read the log pod $AUTOTUNE_LOG_POD in $AUTOTUNE_KUBECTL_CONTEXT/$KUBECTL_NS"
printf '%s' "$POD_JSON" | python3 "$SRC/harness/log_pod.py" "$SHARED_MOUNT" "the log pod $AUTOTUNE_LOG_POD" \
  || die "the log pod must mount the shared volume read-only (above)"

# Model pin: one id on every tier, no fallback (the user-level settings are not loaded either).
if [ -n "${ATTUNE_MODEL:-}" ]; then
  for t in ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL \
           ANTHROPIC_DEFAULT_FABLE_MODEL ANTHROPIC_SMALL_FAST_MODEL CLAUDE_CODE_SUBAGENT_MODEL; do
    export "$t=$ATTUNE_MODEL"
  done
fi
export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1
export AUTOTUNE_PROJECT_ROOT="$CAMPAIGN_DIR"
export AUTOTUNE_GPUH_LOG="${AUTOTUNE_GPUH_LOG:-$CAMPAIGN_DIR/gpuh.jsonl}"

# config.env seals the workload settings, so refuse now a value workload.py would refuse at every render.
python3 "$SRC/skill/scripts/workload.py" >/dev/null \
  || die "the workload settings (AUTOTUNE_NUM_STEPS, AUTOTUNE_WINDOW_START, AUTOTUNE_GBS, AUTOTUNE_SEQ_LEN) are not" \
         "usable (above)"
# The calibration file (band, baseline denominator) is optional; when it is named it must be usable. A set
# AUTOTUNE_BAND (default 0.03) must be a fraction either way; it is the band when there is no file.
if [ -n "${AUTOTUNE_CALIBRATION:-}" ] && [ -n "${AUTOTUNE_BAND:-}" ]; then
  BAND_SETTINGS="AUTOTUNE_CALIBRATION or AUTOTUNE_BAND"
elif [ -n "${AUTOTUNE_CALIBRATION:-}" ]; then
  BAND_SETTINGS=AUTOTUNE_CALIBRATION
else
  BAND_SETTINGS=AUTOTUNE_BAND
fi
CALIBRATION=$(python3 "$SRC/skill/scripts/calibration.py") || die "$BAND_SETTINGS is not usable (above)"
if [ -z "${AUTOTUNE_CALIBRATION:-}" ] && [ -n "${AUTOTUNE_GPUH_BUDGET:-}" ]; then   # the band the campaign's copy uses
  BAND=$(printf '%s' "$CALIBRATION" | python3 -c 'import json, sys; print(json.load(sys.stdin)["band"])' 2>/dev/null) \
    || BAND=unknown
  echo "WARNING: no AUTOTUNE_CALIBRATION; verdicts use the band $BAND and the campaign's own baseline run" >&2
fi
if [ "$FIRST" = 1 ]; then   # the dataset every run trains on; the record holds its hash
  DATASET=${AUTOTUNE_DATASET:-}
  if [ -z "$DATASET" ] && [ -f "${ATTUNE_BASELINE_YAML:-}" ]; then   # its --prompt-data, set $VAR/${VAR} expanded
    DATASET=$(python3 -c 'import os, re, sys
m = re.search(r"--prompt-data[= ]+[\x22\x27]?([^\s\x22\x27\\]+)", open(sys.argv[1]).read())
var = re.compile(r"\$(?:\{(\w+)\}|(\w+))")
print(var.sub(lambda v: os.environ.get(v.group(1) or v.group(2)) or v.group(0), m.group(1)) if m else "")' \
      "$ATTUNE_BASELINE_YAML")
  fi
  DATASET=${DATASET:-${SFT_DATA:-}}
  [ -n "$DATASET" ] || die "no dataset to record: set AUTOTUNE_DATASET (or --prompt-data in ATTUNE_BASELINE_YAML)"
  [[ $DATASET != *'$'* ]] || die "the dataset path $DATASET names a variable the settings do not set; set" \
                                 "AUTOTUNE_DATASET to the path"
  export ATTUNE_CAMPAIGN_ID="${ATTUNE_RUN_PREFIX%-}-$(date -u +%Y%m%d-%H%M%S)-$(printf '%04x' "$RANDOM")"
fi
WEB=on
[ "$AUTOTUNE_ALLOW_WEB" = 1 ] || WEB=off
echo "[launch_campaign] campaign id: ${ATTUNE_CAMPAIGN_ID:-}; web access: $WEB (AUTOTUNE_ALLOW_WEB=$AUTOTUNE_ALLOW_WEB)"
GOAL_SRC="${GOAL_SRC:-$CAMPAIGN_DIR/goal_prompt.txt}"
[ -f "$GOAL_SRC" ] || GOAL_SRC="$SRC/harness/goal_prompt.example.txt"

if [ "$FIRST" = 1 ]; then   # the campaign's copies; a failed setup leaves no half-made control directory
  trap 'chmod -R u+w "$CONTROL" 2>/dev/null; rm -rf "$CONTROL"' ERR
  mkdir "$CONTROL"
  for d in skill harness; do
    python3 -c 'import shutil, sys
shutil.copytree(sys.argv[1], sys.argv[2], ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))' \
      "$ATTUNE_HOME/$d" "$CONTROL/$d"
  done
fi
mkdir -p "$CAMPAIGN_DIR/.claude/skills"
ln -sfn "$CONTROL/skill" "$CAMPAIGN_DIR/.claude/skills/autotune-training-throughput"
rm -f "$CAMPAIGN_DIR/.claude/settings.json" "$CAMPAIGN_DIR/goal_prompt.rendered.txt"

# Fill ${VAR} from the environment: unset variables render as "" in settings (and the entry is dropped) and are an
# error in the goal; values in hook commands are shell-quoted. The first launch also writes config.env, every setting
# of the template's env block (empty ones included), which the skill scripts then hold the environment to.
CONFIG_OUT=""
[ "$FIRST" = 0 ] || CONFIG_OUT="$CONTROL/config.env"
python3 - "$CONTROL/harness/settings.template.json" "$CAMPAIGN_DIR/.claude/settings.json" "$GOAL_SRC" \
  "$CAMPAIGN_DIR/goal_prompt.rendered.txt" "$CONFIG_OUT" <<'PY'
import json, os, re, shlex, sys
tmpl, out, goal_src, goal_out, config_out = sys.argv[1:6]
pat = re.compile(r"\$\{([A-Z0-9_]+)\}")

def fill(s, strict=False, quote=False):
    def rep(m):
        v = os.environ.get(m.group(1))
        if v is None and strict:
            sys.exit(f"ERROR: goal prompt references ${{{m.group(1)}}} but it is not set")
        return shlex.quote(v or "") if quote else v or ""
    return pat.sub(rep, s)

def walk(x, key=None):
    if isinstance(x, dict):
        return {k: walk(v, k) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, list):
        return [walk(v, key) for v in x]
    return fill(x, quote=key == "command") if isinstance(x, str) else x

with open(tmpl) as f:
    raw = json.load(f)
if config_out:
    with open(config_out, "w") as f:
        f.write("# This campaign's settings, written by launch_campaign.sh. The skill scripts refuse an environment\n"
                "# that disagrees with them; the launcher's integrity record holds this file's hash.\n")
        for k, v in raw["env"].items():
            if k != "AUTOTUNE_AUTONOMOUS" and not k.startswith("BASH_"):
                f.write(f"{k}={shlex.quote(fill(v))}\n")
cfg = walk(raw)
cfg["env"] = {k: v for k, v in cfg["env"].items() if v != ""}
if os.environ.get("AUTOTUNE_ALLOW_WEB") == "0":
    cfg["permissions"]["deny"] = ["WebSearch", "WebFetch"] + cfg["permissions"]["deny"]
if os.environ.get("ATTUNE_MODEL"):
    cfg["model"] = os.environ["ATTUNE_MODEL"]
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
with open(goal_src) as f:
    goal = fill(f.read(), strict=True)
with open(goal_out, "w") as f:
    f.write(goal)
PY

if [ "$FIRST" = 1 ]; then
  set -a; . "$CONTROL/config.env"; set +a
  chmod -R a-w "$CONTROL/skill" "$CONTROL/harness" "$CONTROL/config.env"
  chmod u+w "$CONTROL/skill/past_tuning_projects"
  python3 "$CONTROL/harness/integrity.py" seal "$CONTROL" --dataset "$DATASET" --source "$ATTUNE_HOME"
  chmod a-w "$CONTROL"
  trap - ERR
else
  python3 "$CONTROL/harness/integrity.py" refresh "$CONTROL"
fi
chmod a-w "$CAMPAIGN_DIR/.claude/settings.json" "$CAMPAIGN_DIR/goal_prompt.rendered.txt"

FLAGS=(--setting-sources project,local --strict-mcp-config --dangerously-skip-permissions
       --output-format stream-json --verbose)
if [ "${AUTOTUNE_ALLOW_WEB:-}" = 0 ]; then
  FLAGS+=(--disallowedTools WebFetch WebSearch)
fi
if [ -n "${ATTUNE_MODEL:-}" ]; then
  FLAGS+=(--model "$ATTUNE_MODEL")
fi

echo "[launch_campaign] campaign: $CAMPAIGN_DIR"
echo "[launch_campaign] settings: $CAMPAIGN_DIR/.claude/settings.json; control directory: $CONTROL"
cd "$CAMPAIGN_DIR"
PROMPT=$(cat goal_prompt.rendered.txt)
if [ -n "$SESSION" ]; then   # the same session, its context intact; resume_prompt.txt, if present, says what happened
  FLAGS+=(--resume "$SESSION")
  if [ -f resume_prompt.txt ]; then
    PROMPT=$(cat resume_prompt.txt)
  else
    PROMPT="Continue the campaign from where it stopped."
  fi
fi
RUN=(env -u ATTUNE_LESSONS_INBOX ATTUNE_HOME="$CONTROL" AUTOTUNE_AUTONOMOUS=1 "$CLAUDE_BIN" -p "$PROMPT" "${FLAGS[@]}")
# The steps after the session-end integrity check ($1: its exit status); a campaign sealed before the lessons export
# has neither.
after_session() {
  [ -f "$ATTUNE_CONTROL_DIR/harness/lessons.py" ] || return 0
  python3 "$ATTUNE_CONTROL_DIR/harness/integrity.py" source-check "$ATTUNE_CONTROL_DIR" \
    --where "launch_campaign.sh (session end)" \
    || echo "WARNING: the source tree was not checked (above); the session status is unaffected" >&2
  if [ "$1" = 0 ]; then
    python3 "$ATTUNE_CONTROL_DIR/harness/lessons.py" export "$ATTUNE_CONTROL_DIR" \
      || echo "WARNING: the lessons were not exported (above); the session status is unaffected" >&2
  else
    echo "lessons: not exported, the campaign failed its integrity check; they stay in $CAMPAIGN_DIR/<project_folder>/"
  fi
}
if [ -n "$SESSION" ] && [ -e run.out ]; then mv run.out "run.out.before-resume-$(date -u +%Y%m%dT%H%M%SZ)"; fi
if [ "$DETACH" = 1 ]; then   # the detached session checks the record when it ends and writes its status to run.exit
  rm -f run.exit
  setsid bash -c "$(declare -f after_session)"'
"$@" > run.out 2>&1 < /dev/null; rc=$?; intact=0
python3 "$ATTUNE_CONTROL_DIR/harness/integrity.py" check "$ATTUNE_CONTROL_DIR" \
  --where "launch_campaign.sh (session end)" --teardown > run.exit 2>&1 || { rc=3; intact=3; }
after_session "$intact" >> run.exit 2>&1
echo "exit $rc" >> run.exit
exit $rc' attune-session "${RUN[@]}" > /dev/null 2>&1 < /dev/null &
  echo "[launch_campaign] detached (pid $!); output -> $CAMPAIGN_DIR/run.out, status -> $CAMPAIGN_DIR/run.exit"
  exit 0
fi
set +e
"${RUN[@]}" > run.out 2>&1 < /dev/null
rc=$?
set -e
echo "[launch_campaign] session ended (exit $rc); output -> $CAMPAIGN_DIR/run.out"
intact=0
python3 "$CONTROL/harness/integrity.py" check "$CONTROL" --where "launch_campaign.sh (session end)" --teardown \
  || intact=$?
after_session "$intact"
[ "$intact" = 0 ] || failed "the campaign failed its integrity check ($VIOLATION): no recipe ships (the teardown of" \
                            "its tracked runs is above and in the record)"
exit "$rc"
