# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Stop hook allows a stop only on convergence, a STOP row or a stall, finds the ledger, and fails closed.
Run: python3 -m pytest harness/tests"""
import json
import os
import subprocess
import sys

import pytest

import convergence
import stop_gate

BASE = {"run_id": "alice-m-base", "kind": "baseline", "base_run": None, "status": "completed",
        "metrics": {"aggregate_tok_per_s_per_gpu": 100.0}}
STOP = {"kind": "stop", "declared_at": "2026-10-12T00:00:00Z", "stop_reason": "agent_stop", "reason": "no lever left",
        "leader": "alice-m-r1",
        "neighbours": [{"run_id": "alice-m-r2", "axis": "PP4 -> PP2"}], "diagnosis": "alice-m-r1: compute bound"}


def ranked(i: int, verdict: str = "win", **extra) -> dict:
    return {"run_id": f"alice-m-r{i}", "kind": "ranked", "base_run": "alice-m-base", "status": "completed",
            "verdict": verdict, "metrics": {"aggregate_tok_per_s_per_gpu": 100.0 + i}, **extra}


def write(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))


def blocked(out: str) -> bool:
    return bool(out) and json.loads(out)["decision"] == "block"


@pytest.fixture
def gate(campaign, monkeypatch):
    monkeypatch.setenv("AUTOTUNE_AUTONOMOUS", "1")
    monkeypatch.setenv("AUTOTUNE_BUDGET", "2")
    monkeypatch.setenv("AUTOTUNE_MAX_STALL", "50")
    campaign.ledger = campaign.root / "proj" / convergence.LEDGER_NAME
    return campaign


def test_inactive_without_autonomous_mode(campaign):
    assert stop_gate.respond("{}") == ""


def test_count_mode(gate):
    write(gate.ledger, [BASE, ranked(1), ranked(1), ranked(2, "error"), ranked(3, "tie", kind="parity"),
                        ranked(4, metrics={"valid": False}), ranked(5, metrics={"aggregate_tok_per_s_per_gpu": None})])
    out = stop_gate.respond("{}")
    assert blocked(out) and "1/2 ranked results" in out
    write(gate.ledger, [BASE, ranked(1), ranked(2, "lose")])
    assert stop_gate.respond("{}") == ""


def test_rows_follow_the_ledger_schema(gate):
    legacy = [{"run_id": "alice-m-base", "base_run": None, "status": "completed"},
              {"run_id": "alice-m-r1", "is_ranked": True, "verdict": "win"},
              {"run_id": "alice-m-r2", "is_ranked": True, "verdict": "tie"}]
    write(gate.ledger, legacy)
    out = stop_gate.respond("{}")
    assert blocked(out) and "no baseline yet" in out
    write(gate.ledger, [BASE, ranked(1, metrics={"aggregate_tok_per_s_per_gpu": True}), ranked(2)])
    assert "1/2 ranked results" in stop_gate.respond("{}")


def test_count_mode_needs_a_baseline(gate):
    write(gate.ledger, [ranked(1), ranked(2), dict(BASE, status="errored"), dict(BASE, kind="ranked")])
    out = stop_gate.respond("{}")
    assert blocked(out) and "no baseline yet" in out


def test_gpu_hour_mode(gate, monkeypatch):
    monkeypatch.setenv("AUTOTUNE_GPUH_BUDGET", "10")
    write(gate.ledger, [dict(BASE, gpuh=4.0), ranked(1, gpuh=5.0)])
    out = stop_gate.respond("{}")
    assert blocked(out) and "9.0/10 GPU-h" in out and "STOP row" in out
    write(gate.root / "gpuh.jsonl", [{"run": "alice-m-base", "gpuh": 4.0}, {"run": "alice-m-r1", "gpuh": 6.5}])
    assert stop_gate.respond("{}") == ""


def test_gpu_hour_cap_ends_the_campaign_even_without_a_baseline(gate, monkeypatch):
    monkeypatch.setenv("AUTOTUNE_GPUH_BUDGET", "10")
    write(gate.ledger, [ranked(1, gpuh=3.0, gpuh_cumulative=10.5)])
    assert stop_gate.respond("{}") == ""


def test_stop_row(gate, monkeypatch):
    write(gate.ledger, [BASE, STOP])
    assert blocked(stop_gate.respond("{}")), "a STOP row does not end ranked-count mode"
    monkeypatch.setenv("AUTOTUNE_GPUH_BUDGET", "100")
    assert stop_gate.respond("{}") == ""
    write(gate.ledger, [BASE, dict(STOP, diagnosis="")])
    assert blocked(stop_gate.respond("{}")), "a STOP row needs its evidence"
    write(gate.ledger, [STOP])
    assert blocked(stop_gate.respond("{}")), "a STOP row needs a baseline"


def test_stall_limit(gate, monkeypatch):
    monkeypatch.setenv("AUTOTUNE_MAX_STALL", "2")
    write(gate.ledger, [BASE])
    assert blocked(stop_gate.respond("{}"))
    assert blocked(stop_gate.respond("{}"))
    write(gate.ledger, [BASE, ranked(1)])
    assert blocked(stop_gate.respond("{}")), "progress resets the count"
    assert blocked(stop_gate.respond("{}"))
    assert stop_gate.respond("{}") == ""


def test_malformed_lines_are_skipped_with_a_warning(gate, capsys):
    write(gate.ledger, [BASE, "{not json", "[1, 2]", ranked(1), ranked(2)])
    assert stop_gate.respond("{}") == ""
    assert "skipping malformed line 2" in capsys.readouterr().err


def test_ledger_from_the_environment(gate, monkeypatch):
    elsewhere = gate.root.parent / "elsewhere" / convergence.LEDGER_NAME
    write(elsewhere, [BASE, ranked(1), ranked(2)])
    monkeypatch.setenv("AUTOTUNE_LEDGER", str(elsewhere))
    assert stop_gate.respond("{}") == ""


def test_ledger_resolution_by_project_folder(gate):
    out = stop_gate.respond("{}")
    assert blocked(out) and "no baseline yet" in out
    write(gate.ledger, [BASE, ranked(1), ranked(2)])
    assert stop_gate.respond("{}") == ""
    write(gate.root / "other" / convergence.LEDGER_NAME, [BASE])
    out = stop_gate.respond("{}")
    assert blocked(out) and "2 ledgers" in out


def test_bad_settings_block_unless_a_stop_is_declared(gate, monkeypatch):
    monkeypatch.setenv("AUTOTUNE_BUDGET", "twenty")
    write(gate.ledger, [BASE, ranked(1), ranked(2)])
    out = stop_gate.respond("{}")
    assert blocked(out) and "AUTOTUNE_BUDGET" in out
    write(gate.ledger, [BASE, STOP])
    assert stop_gate.respond("{}") == ""


@pytest.mark.parametrize("name,value", [("AUTOTUNE_GPUH_BUDGET", "0"), ("AUTOTUNE_GPUH_BUDGET", "nan"),
                                        ("AUTOTUNE_MAX_STALL", "1.5"), ("AUTOTUNE_BUDGET", "-3")])
def test_limits_reject_non_positive_values(name, value):
    with pytest.raises(ValueError, match=name):
        convergence.Limits.from_env({name: value})


def test_hook_script_end_to_end(gate):
    p = subprocess.run([sys.executable, stop_gate.__file__], input="{}", capture_output=True, text=True,
                       env=dict(os.environ), check=False)
    assert p.returncode == 0 and blocked(p.stdout)
