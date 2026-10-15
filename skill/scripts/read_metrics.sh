#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# read_metrics.sh — extract training perf metrics from a slime train.log
#
# Usage:
#   read_metrics.sh <train_log_path> [step_start] [step_end] [num_gpus]
#
# Defaults: step_start=$AUTOTUNE_WINDOW_START (5), step_end=N-1 for an N-step unprofiled run ($AUTOTUNE_NUM_STEPS,
# default 20): the canonical fixed-shape window (discard the startup steps, measure the rest; see measurement.md).
# num_gpus is REQUIRED: pass it as arg 4 or export $AUTOTUNE_NUM_GPUS (a wrong or absent denominator silently corrupts
# the headline tok/s/GPU).
# Primary metric is aggregate_tok_per_s_per_gpu = total_tokens / total_step_time / num_gpus across the window, the
# normalized, cross-cluster-comparable headline; aggregate_tok_per_s is the raw sum. Both include every overhead
# (stragglers, slow steps), so a wall clock would measure the same number, and the script checks that it does: the
# window's summed step_time must equal the log's own timestamps (perf line of step_end minus that of step_start - 1)
# within max(2%, 2 s); a disagreement or a missing timestamp makes the window invalid.
#
# With $AUTOTUNE_LOG_POD set, the log is read through that pod (kubectl exec ... cat), which needs
# $AUTOTUNE_KUBECTL_CONTEXT and $KUBECTL_NS; with it unset or empty, the path is a local file.
# Under the launcher these settings come from the campaign's config.env, and a different value in the environment is
# refused (exit 2; campaign.py).
#
# Exit codes: 0 valid window; 2 usage; 3 no perf lines in the log; 4 invalid or incomplete window; 5 the log could
# not be read (kubectl or file error).
#
# Examples (64 GPUs):
#   read_metrics.sh <experiments_dir>/<run_id>/logs/<ts>/train.log 5 19 64   # through $AUTOTUNE_LOG_POD
#   read_metrics.sh /tmp/train.log 5 19 64                                   # a local copy, with AUTOTUNE_LOG_POD unset

set -euo pipefail

case "${1:-}" in -h|--help) sed -n '4,/^$/s/^# \{0,1\}//p' "$0"; exit 0 ;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS=$(python3 "$HERE/campaign.py" check-settings --shell) || exit 2   # the campaign's config.env decides
eval "$SETTINGS"
LOG_PATH="${1:?usage: read_metrics.sh <train_log_path> [step_start] [step_end] <num_gpus>}"
# Default window WINDOW_START..N-1 of the unprofiled layout (5-19 by default; see workload.py). A profiled run
# measures the same window, so the defaults apply to both.
_N_STEPS="${AUTOTUNE_NUM_STEPS:-20}"
STEP_START="${2:-${AUTOTUNE_WINDOW_START:-5}}"
STEP_END="${3:-$(( _N_STEPS - 1 ))}"
NUM_GPUS="${4:-${AUTOTUNE_NUM_GPUS:-}}"
if [ -z "$NUM_GPUS" ]; then
    echo '{"error": "num_gpus is required: pass it as arg 4 or export AUTOTUNE_NUM_GPUS (a wrong GPU count corrupts the metric)"}' >&2
    exit 2
fi
LOG_POD="${AUTOTUNE_LOG_POD:-}"
if [ -n "$LOG_POD" ]; then
    : "${AUTOTUNE_KUBECTL_CONTEXT:?set AUTOTUNE_KUBECTL_CONTEXT to read the log through AUTOTUNE_LOG_POD}"
    : "${KUBECTL_NS:?set KUBECTL_NS to read the log through AUTOTUNE_LOG_POD}"
fi

# Per-invocation temp files: concurrent polls must not share one.
PERF_TMP="$(mktemp "${TMPDIR:-/tmp}/autotune_perf.XXXXXX")"
ERR_TMP="$(mktemp "${TMPDIR:-/tmp}/autotune_perf_err.XXXXXX")"
trap 'rm -f "$PERF_TMP" "$ERR_TMP"' EXIT

read_log() {
    if [ -n "$LOG_POD" ]; then
        kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" exec "$LOG_POD" -- cat -- "$LOG_PATH"
    else
        cat -- "$LOG_PATH"
    fi
}

# Extract "perf N: {dict}" lines from train_metric_utils (actor training metrics, not the rollout perf line).
set +e
read_log 2>"$ERR_TMP" | grep "train_metric_utils.*perf " >"$PERF_TMP"
status=("${PIPESTATUS[@]}")
set -e
if [ "${status[0]}" -ne 0 ]; then
    python3 -c 'import json, sys; print(json.dumps({"error": "cannot read the log: " + sys.argv[1], "valid": False}))' \
        "$(head -c 300 "$ERR_TMP")" >&2
    exit 5
fi
if [ "${status[1]}" -ne 0 ]; then
    echo '{"error": "no perf lines found in log: training may not have started yet", "valid": false}' >&2
    exit 3
fi

python3 - "$PERF_TMP" "$STEP_START" "$STEP_END" "$NUM_GPUS" "$LOG_PATH" <<'PYEOF'
import datetime as dt
import json
import re
import statistics as st
import sys

perf_file, step_start, step_end, num_gpus, log_path = sys.argv[1:6]
step_start, step_end, num_gpus = int(step_start), int(step_end), int(num_gpus)
expected_n = step_end - step_start + 1

with open(perf_file) as f:
    lines = f.readlines()

# Parse lines of form: "... perf <N>: {'key': val, ...}"
pat_step = re.compile(r"perf (\d+): \{(.*)\}")
pat_kv = re.compile(r"'([^']+)': ([0-9.eE+-]+)")

pat_ts = re.compile(r"\[(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:\.\d+)?)(?: [^\]]*)?\]")   # slime; miles adds " <worker>"
steps = {}
stamps = {}           # step -> datetime of its perf line (the log's own clock)
all_step_ids = []
for line in lines:
    m = pat_step.search(line)
    if not m:
        continue
    step_id = int(m.group(1))
    all_step_ids.append(step_id)
    body = m.group(2)
    kvs = {k: float(v) for k, v in pat_kv.findall(body)}
    steps[step_id] = kvs
    t = pat_ts.search(line[:m.start()])
    if t:
        stamps[step_id] = dt.datetime.fromisoformat(t.group(1).replace("T", " "))

in_window = {sid: kv for sid, kv in steps.items() if step_start <= sid <= step_end}
latest_id = max(all_step_ids) if all_step_ids else -1

keys_to_summarize = [
    "perf/actor_train_tok_per_s",
    "perf/actor_train_tflops",
    "perf/actor_train_mfu_pct",
    "perf/step_time",
    "perf/actor_train_time",
    "perf/wait_time_ratio",
    "perf/train_wait_time",
]

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else None

summary = {
    "log_path": log_path,
    "latest_step": latest_id,
    "steps_seen_total": len(all_step_ids),
    "window": [step_start, step_end],
    "window_expected_n": expected_n,
    "window_n": len(in_window),
    "num_gpus": num_gpus,
    "means_in_window": {},
    "all_step_ids": sorted(all_step_ids),
}

for k in keys_to_summarize:
    vals = [kv.get(k) for kv in in_window.values()]
    vals = [v for v in vals if v is not None]
    summary["means_in_window"][k] = safe_mean(vals)

# === PRIMARY METRIC: aggregate tok/s over window ===
# The fair headline number: total tokens / total step time over the window
# (NOT mean of per-step tok/s — that double-discounts outliers). Tokens and
# time MUST come from the SAME steps, else the ratio mixes step sets and is
# meaningless. Only steps that have BOTH step_time and tokens are aggregated;
# any step missing either is recorded as incomplete.
paired = {}           # step_id -> (tokens, step_time)
missing_fields = []   # step_ids in window missing a needed field
for sid, kv in in_window.items():
    tok_per_s = kv.get("perf/actor_train_tok_per_s")
    train_time = kv.get("perf/actor_train_time")   # slime doesn't emit raw tokens; reconstruct
    step_time = kv.get("perf/step_time")
    if tok_per_s is not None and train_time is not None and step_time is not None:
        paired[sid] = (tok_per_s * train_time, step_time)
    else:
        missing_fields.append(sid)

tokens_sum = sum(t for t, _ in paired.values())
time_sum = sum(s for _, s in paired.values())
step_times = [s for _, s in paired.values()]

summary["window_steps_aggregated"] = len(paired)
summary["window_steps_missing_fields"] = sorted(missing_fields)

if paired and time_sum > 0:
    agg = tokens_sum / time_sum
    summary["aggregate_tok_per_s"] = round(agg, 1)
    summary["aggregate_tok_per_s_per_gpu"] = round(agg / num_gpus, 2)
    summary["window_total_tokens"] = int(tokens_sum)
    summary["window_total_step_time_s"] = round(time_sum, 2)
else:
    summary["aggregate_tok_per_s"] = None
    summary["aggregate_tok_per_s_per_gpu"] = None

# Step time stats — just the basics for sanity check
if step_times:
    mean_step_time = st.mean(step_times)
    std_step_time = st.stdev(step_times) if len(step_times) > 1 else 0.0
    summary["step_time_stats"] = {
        "mean_s": round(mean_step_time, 2),
        "stddev_s": round(std_step_time, 2),
        "cv_pct": round(std_step_time / mean_step_time * 100, 2) if mean_step_time > 0 else None,
        "p50_s": round(st.median(step_times), 2),
        "min_s": round(min(step_times), 2),
        "max_s": round(max(step_times), 2),
    }
else:
    summary["step_time_stats"] = None

# Also emit the aggregate over *last 10 steps seen* as a "current status" signal
recent = sorted(all_step_ids)[-10:]
recent_kvs = [steps[sid] for sid in recent]
r_times = [kv.get("perf/step_time") for kv in recent_kvs if kv.get("perf/step_time") is not None]
r_tokens = [kv.get("perf/actor_train_tok_per_s", 0) * kv.get("perf/actor_train_time", 0)
            for kv in recent_kvs
            if kv.get("perf/actor_train_tok_per_s") is not None and kv.get("perf/actor_train_time") is not None]
if r_times and r_tokens and sum(r_times) > 0:
    agg_r = sum(r_tokens) / sum(r_times)
    summary["aggregate_tok_per_s_last_10"] = round(agg_r, 1)
    summary["aggregate_tok_per_s_per_gpu_last_10"] = round(agg_r / num_gpus, 2)
summary["last_10_step_ids"] = recent

# === Completeness / validity gate ===
# The window metric is only trustworthy if the FULL window was captured. Dropped
# perf emission (stragglers, log truncation) silently skews the aggregate, so a
# short window is INVALID, not just "smaller n". Emit valid=false + reason and
# exit non-zero so the caller ranks on complete data only.
reasons = []
if summary["aggregate_tok_per_s_per_gpu"] is None:
    reasons.append("no step in the window had both tokens and step_time")
if len(paired) < expected_n:
    reasons.append(f"incomplete window: aggregated {len(paired)}/{expected_n} expected steps "
                   f"(missing/short-emitted: {sorted(set(range(step_start, step_end+1)) - set(paired))})")
# Timing cross-check: the summed step time must be the time the log itself says the window took.
TIMING_REL, TIMING_ABS_S = 0.02, 2.0   # 2%, with a 2 s floor for whole-second log timestamps
t0, t1 = stamps.get(step_start - 1), stamps.get(step_end)
if t0 is None or t1 is None:
    reasons.append("timing check: no log timestamp on the perf line of step %d or %d"
                   % (step_start - 1, step_end))
elif paired:
    elapsed = (t1 - t0).total_seconds()
    summary["window_log_elapsed_s"] = round(elapsed, 2)
    if abs(time_sum - elapsed) > max(TIMING_REL * elapsed, TIMING_ABS_S):
        reasons.append("timing check: summed step_time %.2f s differs from the log's timestamps %.2f s by more "
                       "than max(2%%, 2 s)" % (time_sum, elapsed))
summary["valid"] = not reasons
if reasons:
    summary["invalid_reason"] = "; ".join(reasons)

print(json.dumps(summary, indent=2))
sys.exit(0 if summary["valid"] else 4)
PYEOF
