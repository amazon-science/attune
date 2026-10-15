# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convergence rule of an unattended campaign, read from the skill's ledger (references/ledger.md).

The ledger is $AUTOTUNE_LEDGER, else the single <project_folder>/AUTOTUNE_LEDGER.jsonl under
$AUTOTUNE_PROJECT_ROOT. A campaign has converged
  - in ranked-count mode, when the ledger holds the baseline row and $AUTOTUNE_BUDGET (default 20) distinct
    runs with a ranked result;
  - in GPU-hour mode ($AUTOTUNE_GPUH_BUDGET set), when the GPU-hours spent reach the budget, or when the
    ledger holds the baseline row and the agent's STOP row with its evidence.
Spend is the largest of the ledger's per-run `gpuh` sum, its `gpuh_cumulative` and the total of the GPU-hour
log that scripts/run_gpuh.py writes ($AUTOTUNE_GPUH_LOG, else <project_root>/gpuh.jsonl; a run's last line counts).
Runs on Python 3.7+ (the Stop hook imports it).
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

LEDGER_NAME = "AUTOTUNE_LEDGER.jsonl"
RANKED_VERDICTS = frozenset({"win", "tie", "lose"})
STOP_FIELDS = ("reason", "leader", "neighbours", "diagnosis")  # the STOP row's evidence (ledger.md §1)
NO_BASELINE = 'no baseline yet (a ledger row with kind "baseline" and status "completed")'


class LedgerError(ValueError):
    """The ledger cannot be located unambiguously."""


@dataclass(frozen=True)
class Limits:
    slots: int = 20
    gpuh: float | None = None
    max_stall: int = 3

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Limits:
        return cls(slots=_positive(env, "AUTOTUNE_BUDGET", int, 20),
                   gpuh=_positive(env, "AUTOTUNE_GPUH_BUDGET", float, None),
                   max_stall=_positive(env, "AUTOTUNE_MAX_STALL", int, 3))


def _positive(env: Mapping[str, str], name: str, kind: type, default):
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = kind(raw)
    except ValueError:
        value = None
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name}={raw!r} is not a positive {'integer' if kind is int else 'number'}")
    return value


@dataclass(frozen=True)
class Eval:
    converged: bool
    reason: str
    slots_used: int
    progress: int  # advances with ranked results (count mode) or with spend (GPU-hour mode)


# ---- ledger rows -------------------------------------------------------------------------------------------

def is_baseline(row: dict) -> bool:
    return row.get("kind") == "baseline" and row.get("status") == "completed"


def is_ranked(row: dict) -> bool:
    metrics = row.get("metrics")
    score = metrics.get("aggregate_tok_per_s_per_gpu") if isinstance(metrics, dict) else None
    return (bool(row.get("run_id")) and row.get("kind") == "ranked" and row.get("verdict") in RANKED_VERDICTS
            and isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score))


def is_stop(row: dict) -> bool:
    return row.get("kind") == "stop" and all(row.get(k) for k in STOP_FIELDS)


def read_jsonl(path: str | None) -> list[dict]:
    """The JSON objects of a JSON-lines file (none when it is missing); malformed lines are skipped."""
    if not path:
        return []
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        return []
    rows = []
    with f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = None
            if isinstance(row, dict):
                rows.append(row)
            else:
                print(f"convergence: skipping malformed line {n} of {path}", file=sys.stderr)
    return rows


def find_ledger(env: Mapping[str, str]) -> str | None:
    """$AUTOTUNE_LEDGER, else the one project ledger under $AUTOTUNE_PROJECT_ROOT (None before the first row)."""
    if env.get("AUTOTUNE_LEDGER"):
        return env["AUTOTUNE_LEDGER"]
    root = env.get("AUTOTUNE_PROJECT_ROOT")
    if not root:
        raise LedgerError("neither AUTOTUNE_LEDGER nor AUTOTUNE_PROJECT_ROOT is set")
    found = sorted(glob.glob(os.path.join(glob.escape(root), "*", LEDGER_NAME)))
    if len(found) > 1:
        raise LedgerError(f"{len(found)} ledgers under {root} ({', '.join(found)}); a campaign keeps one project "
                          "folder, so move the others out or set AUTOTUNE_LEDGER")
    return found[0] if found else None


def gpuh_log(env: Mapping[str, str]) -> str:
    if env.get("AUTOTUNE_GPUH_LOG"):
        return env["AUTOTUNE_GPUH_LOG"]
    if not env.get("AUTOTUNE_PROJECT_ROOT"):
        raise LedgerError("neither AUTOTUNE_GPUH_LOG nor AUTOTUNE_PROJECT_ROOT is set")
    return os.path.join(env["AUTOTUNE_PROJECT_ROOT"], "gpuh.jsonl")


# ---- the rule --------------------------------------------------------------------------------------------------

def _hours(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def gpuh_spent(rows: list[dict], log_rows: list[dict]) -> float:
    """The largest of the ledger's per-run sum, its cumulative values and the log's total. A run counts its last
    measured value; a later row or line without one leaves it."""
    def latest(items, key):
        out = {}
        for n, r in enumerate(items):
            h = _hours(r.get("gpuh"))
            if h is not None:
                out[r.get(key) or "#%d" % n] = h
        return out
    per_run = latest([r for r in rows if r.get("run_id")], "run_id")
    cumulative = [h for h in (_hours(r.get("gpuh_cumulative")) for r in rows) if h is not None]
    return max([sum(per_run.values()), sum(latest(log_rows, "run").values())] + cumulative)


def evaluate(rows: list[dict], limits: Limits, log_rows: list[dict] = ()) -> Eval:
    slots = len({r["run_id"] for r in rows if is_ranked(r)})
    baseline = any(is_baseline(r) for r in rows)
    if limits.gpuh is None:
        progress = slots + baseline
        status = (f"{slots}/{limits.slots} ranked results (ledger rows with kind ranked, verdict win, tie or lose "
                  "and a number in metrics.aggregate_tok_per_s_per_gpu)")
        if not baseline:
            return Eval(False, NO_BASELINE, slots, progress)
        return Eval(slots >= limits.slots, status, slots, progress)
    spent = gpuh_spent(rows, list(log_rows))
    progress = int(spent * 10)
    status = f"{spent:.1f}/{limits.gpuh:g} GPU-h spent, {slots} ranked results"
    if spent >= limits.gpuh:
        return Eval(True, f"GPU-hour budget spent ({status})", slots, progress)
    if not baseline:
        return Eval(False, f"{NO_BASELINE}; {status}", slots, progress)
    if any(is_stop(r) for r in rows):
        return Eval(True, f"STOP declared ({status})", slots, progress)
    return Eval(False, status, slots, progress)


def status(env: Mapping[str, str], limits: Limits | None = None) -> Eval:
    """Evaluate the campaign described by the environment."""
    limits = limits or Limits.from_env(env)
    rows = read_jsonl(find_ledger(env))
    log_rows = read_jsonl(gpuh_log(env)) if limits.gpuh is not None else []
    return evaluate(rows, limits, log_rows)


def stop_declared(env: Mapping[str, str]) -> bool:
    """A baseline row and a STOP row with its evidence: the agent's declaration ends the campaign."""
    rows = read_jsonl(find_ledger(env))
    return any(is_baseline(r) for r in rows) and any(is_stop(r) for r in rows)


def continuation(limits: Limits) -> str:
    """What the blocked agent is told to do next."""
    stop = (", or, when no untried lever is expected to beat the leader, append a STOP row with its evidence "
            '(SKILL.md "When to stop")') if limits.gpuh is not None else ""
    return ("Continue the skill loop: pick the next highest-leverage experiment, launch it, read its metrics and "
            f"iterate{stop}. Do not stop or ask; assume defaults.")
