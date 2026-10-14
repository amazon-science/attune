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
# What it does:
#   1. Renders harness/settings.template.json into <campaign_dir>/.claude/settings.json
#      (environment for the skill scripts, the PreToolUse guard and Stop gate hooks, denied tools).
#   2. Links $ATTUNE_HOME/skill into <campaign_dir>/.claude/skills/autotune-training-throughput.
#      Point ATTUNE_HOME at a copy of the tree, not a live checkout: the guard refuses writes to it except the lever
#      rules the campaign appends to skill/references/levers.md and its skill/past_tuning_projects/*_summary.md.
#   3. Renders the goal prompt (${VAR} placeholders filled from the environment) into
#      <campaign_dir>/goal_prompt.rendered.txt.
#   4. Runs `claude -p "<goal>"` in <campaign_dir> with only project/local settings, no MCP servers, no
#      CLAUDE.md or auto memory, no bundled skills (the skill above is the only one), WebFetch/WebSearch
#      disallowed unless AUTOTUNE_ALLOW_WEB=1, and, when ATTUNE_MODEL is set, that model on every tier
#      (main, subagents, Opus/Sonnet/Haiku/Fable defaults) with no fallback model.
#   The Stop hook blocks the session from ending until the ledger holds $AUTOTUNE_BUDGET ranked results, or,
#   when $AUTOTUNE_GPUH_BUDGET is set, until that many GPU-hours are spent or the agent records a STOP row.
#
# `claude -p` streams JSON events to run.out (--output-format stream-json); the session transcript is under
# CLAUDE_CONFIG_DIR. Follow progress in <campaign_dir>/<project_tag>/AUTOTUNE_LEDGER.jsonl and `kubectl get rayjobs`.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 2; }
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

for v in AUTOTUNE_USER AUTOTUNE_KUBECTL_CONTEXT KUBECTL_NS AUTOTUNE_NUM_GPUS AUTOTUNE_BUDGET; do
  [ -n "${!v:-}" ] || die "$v is not set (source your config.env first)"
done
is_count() { [[ $1 =~ ^[0-9]+$ && $1 =~ [1-9] ]]; }
is_number() { [[ $1 =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ && $1 =~ [1-9] ]]; }
for v in AUTOTUNE_NUM_GPUS AUTOTUNE_BUDGET AUTOTUNE_MAX_STALL; do
  [ -z "${!v:-}" ] || is_count "${!v}" || die "$v must be a positive integer, not '${!v}'"
done
[ -z "${AUTOTUNE_GPUH_BUDGET:-}" ] || is_number "$AUTOTUNE_GPUH_BUDGET" \
  || die "AUTOTUNE_GPUH_BUDGET must be a positive number of GPU-hours, not '$AUTOTUNE_GPUH_BUDGET'"
[ -n "${AUTOTUNE_PROTECTED_PATHS:-}${AUTOTUNE_SHARED_MOUNT:-}" ] \
  || echo "WARNING: AUTOTUNE_PROTECTED_PATHS is not set; the guard protects no shared storage" >&2
CLAUDE_BIN="${ATTUNE_CLAUDE_BIN:-claude}"
command -v "$CLAUDE_BIN" >/dev/null || die "$CLAUDE_BIN (Claude Code CLI) not found"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${ATTUNE_HOME:-}" ]; then
  ATTUNE_HOME=$(dirname "$HERE")
fi
export ATTUNE_HOME
[ -f "$ATTUNE_HOME/harness/hooks/pretooluse_guard.py" ] || die "ATTUNE_HOME ($ATTUNE_HOME) has no harness/hooks"
mkdir -p "$CAMPAIGN_DIR"
CAMPAIGN_DIR=$(cd "$CAMPAIGN_DIR" && pwd)
export CAMPAIGN_DIR
GOAL_SRC="${GOAL_SRC:-$CAMPAIGN_DIR/goal_prompt.txt}"
[ -f "$GOAL_SRC" ] || GOAL_SRC="$HERE/goal_prompt.example.txt"

# Model pin: one id on every tier, no fallback (the user-level settings are not loaded either).
if [ -n "${ATTUNE_MODEL:-}" ]; then
  for t in ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL \
           ANTHROPIC_DEFAULT_FABLE_MODEL ANTHROPIC_SMALL_FAST_MODEL CLAUDE_CODE_SUBAGENT_MODEL; do
    export "$t=$ATTUNE_MODEL"
  done
fi
export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1
export AUTOTUNE_GPUH_LOG="${AUTOTUNE_GPUH_LOG:-$CAMPAIGN_DIR/gpuh.jsonl}"

# The calibration file (band, baseline denominator) is optional; when it is named it must be usable.
if [ -n "${AUTOTUNE_CALIBRATION:-}" ]; then
  python3 "$ATTUNE_HOME/skill/scripts/calibration.py" >/dev/null || die "AUTOTUNE_CALIBRATION is not usable"
elif [ -n "${AUTOTUNE_GPUH_BUDGET:-}" ]; then
  echo "WARNING: no AUTOTUNE_CALIBRATION; verdicts use the default 3% band and the campaign's own baseline run" >&2
fi
mkdir -p "$CAMPAIGN_DIR/.claude/skills"
ln -sfn "$ATTUNE_HOME/skill" "$CAMPAIGN_DIR/.claude/skills/autotune-training-throughput"

# Fill ${VAR} from the environment: unset variables render as "" in settings (and the entry is dropped) and are an
# error in the goal; values in hook commands are shell-quoted. With AUTOTUNE_ALLOW_WEB=1 the web tools are not denied.
python3 - "$HERE/settings.template.json" "$CAMPAIGN_DIR/.claude/settings.json" "$GOAL_SRC" "$CAMPAIGN_DIR/goal_prompt.rendered.txt" <<'PY'
import json, os, re, shlex, sys
tmpl, out, goal_src, goal_out = sys.argv[1:5]
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
    cfg = walk(json.load(f))
cfg["env"] = {k: v for k, v in cfg["env"].items() if v != ""}
if os.environ.get("AUTOTUNE_ALLOW_WEB") == "1":
    cfg["permissions"]["deny"] = [r for r in cfg["permissions"]["deny"] if r not in ("WebFetch", "WebSearch")]
if os.environ.get("ATTUNE_MODEL"):
    cfg["model"] = os.environ["ATTUNE_MODEL"]
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
with open(goal_src) as f:
    goal = fill(f.read(), strict=True)
with open(goal_out, "w") as f:
    f.write(goal)
PY

FLAGS=(--setting-sources project,local --strict-mcp-config --dangerously-skip-permissions
       --output-format stream-json --verbose)
if [ "${AUTOTUNE_ALLOW_WEB:-}" != 1 ]; then
  FLAGS+=(--disallowedTools WebFetch WebSearch)
fi
if [ -n "${ATTUNE_MODEL:-}" ]; then
  FLAGS+=(--model "$ATTUNE_MODEL")
fi

echo "[launch_campaign] campaign: $CAMPAIGN_DIR"
echo "[launch_campaign] settings: $CAMPAIGN_DIR/.claude/settings.json"
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
RUN=(env AUTOTUNE_AUTONOMOUS=1 "$CLAUDE_BIN" -p "$PROMPT" "${FLAGS[@]}")
if [ -n "$SESSION" ] && [ -e run.out ]; then mv run.out "run.out.before-resume-$(date -u +%Y%m%dT%H%M%SZ)"; fi
if [ "$DETACH" = 1 ]; then
  setsid "${RUN[@]}" > run.out 2>&1 < /dev/null &
  echo "[launch_campaign] detached (pid $!); output -> $CAMPAIGN_DIR/run.out"
else
  "${RUN[@]}" > run.out 2>&1 < /dev/null
  echo "[launch_campaign] session ended; output -> $CAMPAIGN_DIR/run.out"
fi
