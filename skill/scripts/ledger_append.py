#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append one row to a project's ledger: the commit point of finalize (SKILL.md, "State & idempotency contract").

    ledger_append.py <project_folder>/AUTOTUNE_LEDGER.jsonl '<row JSON>'
    ledger_append.py <project_folder>/AUTOTUNE_LEDGER.jsonl - < row.json

The row is one JSON object (references/ledger.md). A run_id that already has a terminal row (status completed,
errored or killed) is not ledgered again, so a re-poll after a crash cannot double-count. Under the launcher the
campaign's integrity check runs first: on a violation nothing is appended, the campaign's tracked runs are torn down
and the script exits 3. The campaign has then failed and no recipe ships: stop.
Prints {"appended": true|false, "ledger": <path>, ...}; exit 0 when the row is in the ledger, 2 on bad input, 3 on
an integrity violation or any other error (nothing appended).
"""
import fcntl
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign  # noqa: E402

TERMINAL = frozenset({"completed", "errored", "killed"})


def _rows(text):
    out = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def append(path, row):
    """Append `row` unless its run_id already has a terminal row; returns whether it was appended."""
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        text = f.read()
        rid = row.get("run_id")
        if rid and any(r.get("run_id") == rid and r.get("status") in TERMINAL for r in _rows(text)):
            return False
        f.write(("\n" if text and not text.endswith("\n") else "") + json.dumps(row) + "\n")
        return True


def main(argv=None):
    import argparse

    campaign.require_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger", help="the project's AUTOTUNE_LEDGER.jsonl")
    ap.add_argument("row", help="the row as one JSON object, or - to read it from stdin")
    a = ap.parse_args(argv)
    try:
        row = json.loads(sys.stdin.read() if a.row == "-" else a.row)
    except ValueError as e:
        print(json.dumps({"appended": False, "error": "the row is not JSON: %s" % e}))
        return 2
    if not isinstance(row, dict):
        print(json.dumps({"appended": False, "error": "the row must be a JSON object"}))
        return 2
    where = "ledger_append.py (finalize %s)" % (row.get("run_id") or row.get("kind") or "a row")
    problems = campaign.integrity_check(where, teardown=True)
    if problems:
        print(json.dumps({"appended": False, "error": "INTEGRITY VIOLATION: the campaign has failed and no recipe "
                          "ships; stop", "violations": campaign.describe(problems),
                          "teardown": campaign.teardown_note()}))
        return 3
    done = append(a.ledger, row)
    campaign.integrity_check(where)  # records the appended row as seen
    out = {"appended": done, "ledger": os.path.abspath(a.ledger)}
    if not done:
        out["note"] = "%s already has a terminal row; not ledgered again" % row.get("run_id")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # any error: the row is not ledgered
        print(json.dumps({"appended": False, "error": "%s: %s" % (type(e).__name__, e)}))
        sys.exit(3)
