#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# teardown_run.sh — delete one of your runs: its RayJob and its ConfigMaps (<run_id>, <run_id>-params, <run_id>-scripts).
#
# Usage:
#   teardown_run.sh <run_id>
#
# Environment (all required except where a default is given):
#   AUTOTUNE_KUBECTL_CONTEXT, KUBECTL_NS   every kubectl call passes --context and -n explicitly
#   AUTOTUNE_USER                          the operator; owner label value and default run-id prefix
#   AUTOTUNE_OWNER_LABEL                   owner label key (default: owner)
#   ATTUNE_RUN_PREFIX                      required run-id prefix (default: "$AUTOTUNE_USER-")
#
# It refuses (exit 2, nothing deleted) unless the run id is valid, starts with the prefix, and every object that
# exists carries <AUTOTUNE_OWNER_LABEL>=<AUTOTUNE_USER> exactly (make_run_yaml.py stamps it). Each delete selects by
# name and owner label, so it cannot remove an object someone else owns. Only a confirmed NotFound counts as already
# deleted: any other kubectl failure (auth, timeout, forbidden) exits 1, because the run may still hold its GPUs.

set -uo pipefail

case "${1:-}" in -h|--help) sed -n '4,/^$/s/^# \{0,1\}//p' "$0"; exit 0 ;; esac

RUN_ID="${1:?usage: teardown_run.sh <run_id>}"
: "${AUTOTUNE_KUBECTL_CONTEXT:?set AUTOTUNE_KUBECTL_CONTEXT to the target cluster context}"
: "${KUBECTL_NS:?set KUBECTL_NS to the namespace of your runs}"
: "${AUTOTUNE_USER:?set AUTOTUNE_USER (the owner label value of your runs)}"
OWNER_LABEL="${AUTOTUNE_OWNER_LABEL:-owner}"
PREFIX="${ATTUNE_RUN_PREFIX:-$AUTOTUNE_USER-}"
KUBECTL=(kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS")

refuse() { echo "[teardown_run.sh] REFUSED: $*" >&2; exit 2; }

if [[ ! "$RUN_ID" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]] || [ "${#RUN_ID}" -gt 47 ]; then
  refuse "run id '$RUN_ID' is not a valid run id (lowercase letters, digits and '-', at most 47 characters)"
fi
[[ "$RUN_ID" == "$PREFIX"* ]] || refuse "run id '$RUN_ID' does not start with '$PREFIX' (ATTUNE_RUN_PREFIX)"
if [[ ! "$AUTOTUNE_USER" =~ ^[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?$ ]]; then
  refuse "AUTOTUNE_USER '$AUTOTUNE_USER' is not a valid label value"
fi

# Prints the object's owner label value, or nothing when the object is absent; exits 1 on any other kubectl failure.
owner_of() {
  local kind="$1" name="$2" out rc
  out=$("${KUBECTL[@]}" get "$kind" "$name" -o json 2>&1); rc=$?
  if [ "$rc" -ne 0 ]; then
    if printf '%s' "$out" | grep -q "NotFound"; then echo "<absent>"; return 0; fi
    echo "[teardown_run.sh] ERROR reading $kind/$name: $(printf '%s' "$out" | head -c 300)" >&2
    return 1
  fi
  printf '%s' "$out" | python3 -c '
import json, sys
labels = json.load(sys.stdin).get("metadata", {}).get("labels") or {}
print(labels.get(sys.argv[1], "<none>"))' "$OWNER_LABEL"
}

OBJECTS=("rayjob $RUN_ID" "configmap $RUN_ID" "configmap $RUN_ID-params" "configmap $RUN_ID-scripts")
PRESENT=()
for obj in "${OBJECTS[@]}"; do
  read -r kind name <<<"$obj"
  got=$(owner_of "$kind" "$name") || exit 1
  [ "$got" = "<absent>" ] && continue
  [ "$got" = "$AUTOTUNE_USER" ] || refuse "$kind/$name has $OWNER_LABEL='$got', not '$AUTOTUNE_USER'; it is not your run"
  PRESENT+=("$obj")
done

echo "[teardown_run.sh] $RUN_ID in $AUTOTUNE_KUBECTL_CONTEXT/$KUBECTL_NS: ${#PRESENT[@]} object(s) to delete"
fail=0
for obj in "${PRESENT[@]}"; do
  read -r kind name <<<"$obj"
  if out=$("${KUBECTL[@]}" delete "$kind" -l "$OWNER_LABEL=$AUTOTUNE_USER" --field-selector "metadata.name=$name" 2>&1); then
    echo "  deleted $kind/$name"
  else
    echo "  ERROR deleting $kind/$name: $(printf '%s' "$out" | head -c 300)" >&2
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  echo "[teardown_run.sh] FAILED: a delete errored; the run may still be live (retry; keep tracking it)" >&2
  exit 1
fi
echo "[teardown_run.sh] done"
