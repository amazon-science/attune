#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# poll_run.sh — classify a running experiment's current state.
#
# Usage:
#   poll_run.sh <run_id> [log_access_pod]
#
# Environment:
#   AUTOTUNE_KUBECTL_CONTEXT  required; every kubectl call passes --context explicitly
#   KUBECTL_NS                required; the namespace of the run and of the log-access pod
#   AUTOTUNE_LOG_POD          the log-access pod (a long-lived pod that mounts the shared experiments volume),
#                             unless given as argument 2; keep one per operator
#   AUTOTUNE_EXPERIMENTS_DIR  default /shared/experiments (path inside the log-access pod); the run's log is the
#                             newest <dir>/<run_id>/logs/<timestamp>/train.log
#   AUTOTUNE_PROJECT_ROOT     default $PWD/attune-projects; <root>/*/tracked_jobs.jsonl is searched for the run's
#                             `target_steps`
#   TARGET_STEPS              the last 0-indexed step of a complete run; overrides everything below
# Under the launcher the settings above except TARGET_STEPS come from the campaign's config.env, and a different value
# in the environment is refused (exit 2; campaign.py).
#
# The target step (the last 0-indexed step of a complete run) is TARGET_STEPS, else the run's own --num-rollout - 1
# (the log's resolved-argument dump, else the RayJob spec), else its tracked_jobs.jsonl `target_steps`, else
# AUTOTUNE_NUM_STEPS - 1. So a profiled run (23 steps at the defaults) is done at step 22, not at the window's end.
#
# Pods are matched exactly: the submitter Job's pods by the label job-name=<run_id>, and the Ray cluster's pods by
# ray.io/cluster=<the RayJob's status.rayClusterName>, so run ids that share a prefix never mix.
#
# Prints one JSON object:
#   run_id, status (setup|running|stuck|errored|done|done-unknown|absent|api-error), reason, driver_pod_phase,
#   rayjob_status, rayjob_deployment, log_path, perf_lines_count, latest_step, first_perf_at,
#   secs_since_last_perf, first_error_line, target_step, target_source, recommendation (wait|triage|teardown|retry)
# api-error means the cluster API could not be read (auth, context, timeout): retry, never treat it as absent or done.
# A run with no new perf line for 10 minutes is stuck (triage), however long the silence.

set -euo pipefail

case "${1:-}" in -h|--help) sed -n '4,/^$/s/^# \{0,1\}//p' "$0"; exit 0 ;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS=$(python3 "$HERE/campaign.py" check-settings --shell) || exit 2   # the campaign's config.env decides
eval "$SETTINGS"
RUN_ID="${1:?usage: poll_run.sh <run_id> [log_access_pod]}"
if [[ ! "$RUN_ID" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]] || [ "${#RUN_ID}" -gt 47 ]; then
  echo "poll_run.sh: invalid run id '$RUN_ID' (lowercase letters, digits and '-', at most 47 characters)" >&2
  exit 2
fi
: "${AUTOTUNE_KUBECTL_CONTEXT:?set AUTOTUNE_KUBECTL_CONTEXT to the target cluster context}"
: "${KUBECTL_NS:?set KUBECTL_NS to the namespace of the run}"
LOG_POD="${2:-${AUTOTUNE_LOG_POD:?set AUTOTUNE_LOG_POD (or pass the log-access pod as argument 2)}}"
EXP_DIR="${AUTOTUNE_EXPERIMENTS_DIR:-/shared/experiments}"
PROJECT_ROOT="${AUTOTUNE_PROJECT_ROOT:-$PWD/attune-projects}"
KUBECTL=(kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS")

RJ_JSON=$(mktemp); DRIVER_JSON=$(mktemp); CLUSTER_JSON=$(mktemp); LOG_LINES=$(mktemp); ERR=$(mktemp)
trap 'rm -f "$RJ_JSON" "$DRIVER_JSON" "$CLUSTER_JSON" "$LOG_LINES" "$ERR"' EXIT

# RayJob: ok, absent (a successful NotFound) or error:<message>
if "${KUBECTL[@]}" get rayjob "$RUN_ID" -o json >"$RJ_JSON" 2>"$ERR"; then
  RJ_STATE=ok
elif grep -q NotFound "$ERR"; then
  RJ_STATE=absent
else
  RJ_STATE="error:$(head -c 200 "$ERR")"
fi

PODS_STATE=ok
"${KUBECTL[@]}" get pods -l "job-name=$RUN_ID" -o json >"$DRIVER_JSON" 2>"$ERR" || PODS_STATE="error:$(head -c 200 "$ERR")"
CLUSTER=$(python3 -c 'import json, sys; print((json.load(sys.stdin).get("status") or {}).get("rayClusterName", ""))' \
  <"$RJ_JSON" 2>/dev/null || true)
echo '{"items": []}' >"$CLUSTER_JSON"
if [ -n "$CLUSTER" ]; then
  "${KUBECTL[@]}" get pods -l "ray.io/cluster=$CLUSTER" -o json >"$CLUSTER_JSON" 2>"$ERR" \
    || PODS_STATE="error:$(head -c 200 "$ERR")"
fi

# The newest log directory, then the perf, error and traceback lines of its train.log (values pass as arguments).
LOG_PATTERN='train_metric_utils.*perf |Traceback \(most recent call last\)|DistBackendError|OutOfMemoryError'
LOG_PATTERN+='|AssertionError|RuntimeError|St12length_error|ActorUnavailableError|store->get.*timeout'
LOG_PATTERN+='|wandb.*BrokenPipeError|BrokenPipeError.*wandb|wandb/sdk/lib|num_rollout \.{3,} '
LOG_PATH=""
LATEST=$("${KUBECTL[@]}" exec "$LOG_POD" -- ls -t -- "$EXP_DIR/$RUN_ID/logs" 2>/dev/null | head -1 || true)
if [ -n "$LATEST" ] && [[ "$LATEST" != */* ]]; then
  LOG_PATH="$EXP_DIR/$RUN_ID/logs/$LATEST/train.log"
  "${KUBECTL[@]}" exec "$LOG_POD" -- grep -E -e "$LOG_PATTERN" -- "$LOG_PATH" >"$LOG_LINES" 2>/dev/null || true
fi

python3 - "$HERE" "$RUN_ID" "$RJ_STATE" "$RJ_JSON" "$PODS_STATE" "$DRIVER_JSON" "$CLUSTER_JSON" "$LOG_PATH" \
  "$LOG_LINES" "$PROJECT_ROOT" <<'PY'
import datetime as dt
import glob
import json
import os
import re
import sys

here, run_id, rj_state, rj_json, pods_state, driver_json, cluster_json, log_path, log_lines, root = sys.argv[1:11]
sys.path.insert(0, here)
import workload  # noqa: E402

PERF = re.compile(r"train_metric_utils.*perf (\d+):")
REAL = re.compile(r"DistBackendError|OutOfMemoryError|AssertionError|RuntimeError|St12length_error|"
                  r"ActorUnavailableError|store->get.*timeout")
TRACEBACK = "Traceback (most recent call last)"
WANDB_PIPE = re.compile(r"wandb.*BrokenPipeError|BrokenPipeError.*wandb|wandb/sdk/lib")
STAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
NUM_ROLLOUT_ARG = re.compile(r"\bnum_rollout \.{3,} (\d+)\s*$")
NUM_ROLLOUT_FLAG = re.compile(r"--num-rollout[= ]+(\d+)\b")
STUCK_S = 600


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def own_num_rollout(lines, rj):
    """The run's --num-rollout: the log's resolved-argument dump, else one value in the RayJob spec; None if unknown."""
    for x in lines:
        m = NUM_ROLLOUT_ARG.search(x)
        if m:
            return int(m.group(1))
    found = set(NUM_ROLLOUT_FLAG.findall(json.dumps(rj.get("spec") or {})))
    return int(found.pop()) if len(found) == 1 else None


def target_step(lines, rj):
    """(target, source): TARGET_STEPS, else the run's own --num-rollout - 1, else its tracked_jobs.jsonl
    `target_steps`, else the unprofiled last step."""
    if os.environ.get("TARGET_STEPS", "").strip():
        try:
            return int(os.environ["TARGET_STEPS"]), "TARGET_STEPS"
        except ValueError:
            sys.exit("poll_run.sh: TARGET_STEPS=%r is not an integer" % os.environ["TARGET_STEPS"])
    n = own_num_rollout(lines, rj)
    if n is not None:
        return n - 1, "num_rollout"
    for path in sorted(glob.glob(os.path.join(root, "*", "tracked_jobs.jsonl"))):
        with open(path, errors="ignore") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if isinstance(e, dict) and e.get("run_id") == run_id and e.get("target_steps") is not None:
                    return int(e["target_steps"]), "tracked_jobs"
    return workload.resolve()["target_step"], "workload"


def first_error(lines):
    """The first error or traceback line, unless every traceback is wandb's shutdown BrokenPipe noise."""
    first = next((x for x in lines if REAL.search(x) or TRACEBACK in x), "")
    if TRACEBACK in first and not any(REAL.search(x) for x in lines):
        tracebacks = sum(TRACEBACK in x for x in lines)
        pipes = sum(bool(WANDB_PIPE.search(x)) for x in lines)
        if 0 < tracebacks <= pipes:
            return ""
    return first


def perf_facts(lines):
    perf = [x for x in lines if PERF.search(x)]
    steps = [int(PERF.search(x).group(1)) for x in perf]
    stamps = [STAMP.search(x).group(0) for x in perf if STAMP.search(x)]
    secs = -1
    if stamps:   # log timestamps are UTC
        last = dt.datetime.strptime(stamps[-1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        secs = int((dt.datetime.now(dt.timezone.utc) - last).total_seconds())
    return len(perf), max(steps, default=-1), (stamps[0] if stamps else None), secs


def classify(f, target):
    """(status, reason, recommendation) from the collected facts."""
    if f["api_error"]:
        return "api-error", "kubectl could not read the cluster API (auth/context/timeout), NOT absence: " + \
            f["api_error"], "retry"
    if f["absent"]:
        return "absent", "no RayJob and no driver pod for this run id (already torn down, or never applied)", "triage"
    if f["deployment"] == "Initializing":
        return "setup", "RayJob deployment Initializing (capacity-waiting or pre-training setup)", "wait"
    if f["latest"] >= target:
        return "done", "hit the target step (latest_step=%d, target=%d)" % (f["latest"], target), "teardown"
    if f["error"]:
        return "errored", "error pattern in the log: " + f["error"][:200], "triage"
    if f["rj_status"] == "FAILED":
        return "errored", "RayJob status=FAILED", "triage"
    if f["driver"] == "Failed":
        return "errored", "driver pod phase=Failed", "triage"
    if f["driver"] == "Succeeded" and f["perf"] == 0:
        return "errored", "driver pod Succeeded but no perf lines (training never reached step 0)", "triage"
    if f["rj_status"] == "SUCCEEDED" and not log_path:
        return "done-unknown", ("RayJob SUCCEEDED but no log at <AUTOTUNE_EXPERIMENTS_DIR>/<run_id>/logs/<ts>/train.log; "
                                "check the experiments dir and the log-access pod's volume"), "triage"
    if f["rj_status"] == "SUCCEEDED":
        return "errored", ("RayJob SUCCEEDED but latest_step=%d < %d: an actor crash swallowed at driver exit"
                           % (f["latest"], target)), "triage"
    if f["perf"] == 0:
        why = "%d pod(s) Pending, capacity-waiting" % f["pending"] if f["pending"] else "setup or checkpoint load"
        return "setup", "no perf lines yet; " + why, "wait"
    if f["secs"] > STUCK_S:
        return "stuck", "no new perf line in %d s (possible hang)" % f["secs"], "triage"
    return "running", "healthy: latest_step=%d, %d s since the last perf line" % (f["latest"], f["secs"]), "wait"


def main():
    rj = load(rj_json) if rj_state == "ok" else {}
    driver_pods = load(driver_json).get("items") or []
    pods = driver_pods + (load(cluster_json).get("items") or [])
    newest = max(driver_pods, key=lambda p: p.get("metadata", {}).get("creationTimestamp", ""), default=None)
    try:
        with open(log_lines, errors="ignore") as fh:
            lines = [x.rstrip("\n") for x in fh]
    except OSError:
        lines = []
    perf, latest, first_ts, secs = perf_facts(lines)
    errors = [s[len("error:"):] for s in (rj_state, pods_state) if s.startswith("error:")]
    f = {"api_error": "; ".join(errors), "absent": rj_state == "absent" and not driver_pods,
         "deployment": (rj.get("status") or {}).get("jobDeploymentStatus"),
         "rj_status": (rj.get("status") or {}).get("jobStatus"),
         "driver": ((newest or {}).get("status") or {}).get("phase"),
         "pending": sum((p.get("status") or {}).get("phase") == "Pending" for p in pods),
         "perf": perf, "latest": latest, "secs": secs, "error": first_error(lines)}
    target, source = target_step(lines, rj)
    status, reason, rec = classify(f, target)
    print(json.dumps({
        "run_id": run_id, "status": status, "reason": reason, "driver_pod_phase": f["driver"],
        "rayjob_status": f["rj_status"], "rayjob_deployment": f["deployment"], "log_path": log_path or None,
        "perf_lines_count": perf, "latest_step": latest, "first_perf_at": first_ts, "secs_since_last_perf": secs,
        "first_error_line": f["error"][:500] or None, "target_step": target, "target_source": source,
        "recommendation": rec,
    }, indent=2))


main()
PY
