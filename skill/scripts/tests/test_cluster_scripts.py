# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""teardown_run.sh, poll_run.sh and read_metrics.sh against a fake kubectl on PATH (no cluster).

The fake records every call and answers from a JSON state file: `objects` ("kind/name" -> labels, spec and status) for
`get <kind> <name> -o json`, `pods` (label selector -> pods) for `get pods -l`, and `fail` (substring of the call ->
error) for injected failures. `exec <pod> -- <cmd>` runs <cmd> locally, so the test's files play the pod's volume.
Run: python3 -m pytest skill/scripts/tests"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(HERE))
import fixtures  # noqa: E402

CTX, NS = "test-ctx", "test-ns"
FAKE_KUBECTL = '''#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_KUBECTL_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
with open(os.environ["FAKE_KUBECTL_STATE"]) as f:
    state = json.load(f)
call = " ".join(args)
for needle, err in state.get("fail", {}).items():
    if needle in call:
        sys.exit(print(err, file=sys.stderr) or 1)
rest = args[4:]                                   # after --context CTX -n NS
if rest[:1] == ["exec"]:
    sys.exit(subprocess.run(rest[rest.index("--") + 1:]).returncode)
if rest[:1] == ["delete"]:
    sys.exit(print("%s deleted" % rest[1]) or 0)
if rest[:2] == ["get", "pods"]:
    sys.exit(print(json.dumps({"items": state.get("pods", {}).get(rest[rest.index("-l") + 1], [])})) or 0)
if rest[:1] == ["get"]:
    obj = state.get("objects", {}).get("%s/%s" % (rest[1], rest[2]))
    if obj is None:
        sys.exit(print('Error from server (NotFound): %s "%s" not found' % (rest[1], rest[2]), file=sys.stderr) or 1)
    sys.exit(print(json.dumps({"metadata": {"name": rest[2], "labels": obj.get("labels", {})},
                               "spec": obj.get("spec", {}), "status": obj.get("status", {})})) or 0)
sys.exit(print("fake kubectl: unexpected call " + call, file=sys.stderr) or 3)
'''


@pytest.fixture
def cluster(tmp_path):
    """run(script, *args, state=..., **env) -> (CompletedProcess, kubectl calls)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "kubectl").write_text(FAKE_KUBECTL)
    (bin_dir / "kubectl").chmod(0o755)
    calls_file, state_file = tmp_path / "calls.jsonl", tmp_path / "state.json"

    def run(script, *args, state=None, **env):
        state_file.write_text(json.dumps(state or {}))
        calls_file.write_text("")
        base = {k: v for k, v in os.environ.items() if not k.startswith(("AUTOTUNE_", "ATTUNE_", "KUBECTL_"))}
        base.update(PATH="%s:%s" % (bin_dir, os.environ["PATH"]), FAKE_KUBECTL_LOG=str(calls_file),
                    FAKE_KUBECTL_STATE=str(state_file), AUTOTUNE_KUBECTL_CONTEXT=CTX, KUBECTL_NS=NS,
                    AUTOTUNE_USER="alice", AUTOTUNE_PROJECT_ROOT=str(tmp_path / "projects"),
                    PYTHONDONTWRITEBYTECODE="1")
        base.update(env)
        p = subprocess.run(["bash", str(SCRIPTS / script), *args], capture_output=True, text=True, env=base,
                           cwd=tmp_path, timeout=60)
        calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
        assert all(c[:4] == ["--context", CTX, "-n", NS] for c in calls), calls
        return p, [c[4:] for c in calls]
    return run


def owned(user="alice"):
    return {"labels": {"owner": user}}


# ---------------- teardown_run.sh ----------------
def test_teardown_deletes_exactly_the_owned_objects(cluster):
    state = {"objects": {"rayjob/alice-r1": owned(), "configmap/alice-r1-scripts": owned()}}
    p, calls = cluster("teardown_run.sh", "alice-r1", state=state)
    assert p.returncode == 0, p.stderr
    deletes = [c for c in calls if c[0] == "delete"]
    t = "--request-timeout=30s"
    sel = ["-l", "owner=alice", "--field-selector"]
    assert deletes == [["delete", "rayjob", *sel, "metadata.name=alice-r1", t],
                       ["delete", "configmap", *sel, "metadata.name=alice-r1-scripts", t]]
    assert all(c[-1] == t for c in calls), "every kubectl call is bounded"


def test_teardown_of_a_run_already_gone_succeeds(cluster):
    p, calls = cluster("teardown_run.sh", "alice-r1", state={"objects": {}})
    assert p.returncode == 0 and "0 object(s) to delete" in p.stdout, p.stderr
    assert not [c for c in calls if c[0] == "delete"]


def test_teardown_refuses_another_owner(cluster):
    state = {"objects": {"rayjob/alice-r1": owned("bob"), "configmap/alice-r1": owned()}}
    p, calls = cluster("teardown_run.sh", "alice-r1", state=state)
    assert p.returncode == 2 and "REFUSED" in p.stderr and "owner='bob'" in p.stderr
    assert not [c for c in calls if c[0] == "delete"]
    state = {"objects": {"rayjob/alice-r1": {"labels": {}}}}
    p, calls = cluster("teardown_run.sh", "alice-r1", state=state)
    assert p.returncode == 2 and not [c for c in calls if c[0] == "delete"]


@pytest.mark.parametrize("run_id, env", [
    ("bob-r1", {}),                                    # another operator's prefix
    ("alice-r1", {"ATTUNE_RUN_PREFIX": "alice-tune-"}),
    ("alice-r1;kubectl", {}),
    ("Alice-r1", {}),
])
def test_teardown_refuses_a_bad_run_id(cluster, run_id, env):
    p, calls = cluster("teardown_run.sh", run_id, state={"objects": {"rayjob/%s" % run_id: owned()}}, **env)
    assert p.returncode == 2 and "REFUSED" in p.stderr and calls == []


def test_teardown_fails_closed_on_api_errors(cluster):
    p, calls = cluster("teardown_run.sh", "alice-r1", state={"fail": {"get rayjob": "error: Unauthorized"}})
    assert p.returncode == 1 and "Unauthorized" in p.stderr and not [c for c in calls if c[0] == "delete"]


def test_teardown_requires_context_and_namespace(cluster):
    p, calls = cluster("teardown_run.sh", "alice-r1", KUBECTL_NS="")
    assert p.returncode != 0 and "KUBECTL_NS" in p.stderr and calls == []


# ---------------- poll_run.sh ----------------
def _log(exp_dir, run_id, last_step, last_age_s, step_s=60, num_rollout=None):
    """A train.log whose perf line of last_step is last_age_s old, with the argument dump's num_rollout if given."""
    d = exp_dir / run_id / "logs" / "20261001-000000"
    d.mkdir(parents=True)
    t_last = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(seconds=last_age_s)
    lines = ["  num_rollout ........................ %d" % num_rollout] if num_rollout is not None else []
    for t in range(last_step + 1):
        stamp = (t_last - dt.timedelta(seconds=(last_step - t) * step_s)).strftime("%Y-%m-%d %H:%M:%S")
        lines.append("[%s] train_metric_utils.py:1 - perf %d: {'perf/step_time': %d.0}" % (stamp, t, step_s))
    (d / "train.log").write_text("\n".join(lines) + "\n")


def _poll(cluster, tmp_path, run_id, state, **env):
    p, calls = cluster("poll_run.sh", run_id, state=state, AUTOTUNE_LOG_POD="alice-log-access",
                       AUTOTUNE_EXPERIMENTS_DIR=str(tmp_path / "exp"), **env)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout), calls


def running(cluster_name="alice-r1-abcde"):
    return {"status": {"jobStatus": "RUNNING", "jobDeploymentStatus": "Running", "rayClusterName": cluster_name}}


def test_poll_rejects_an_injection_run_id(cluster, tmp_path):
    for bad in ("alice-r1'; touch pwned; '", "alice-r1$(touch pwned)", "../alice-r1"):
        p, calls = cluster("poll_run.sh", bad, AUTOTUNE_LOG_POD="alice-log-access")
        assert p.returncode == 2 and "invalid run id" in p.stderr and calls == []
    assert not (tmp_path / "pwned").exists()


def test_poll_running_done_and_stuck(cluster, tmp_path):
    exp = tmp_path / "exp"
    _log(exp, "alice-r1", last_step=6, last_age_s=30)
    r, calls = _poll(cluster, tmp_path, "alice-r1", {"objects": {"rayjob/alice-r1": running()}})
    assert r["status"] == "running" and r["latest_step"] == 6 and r["perf_lines_count"] == 7, r
    assert ["get", "pods", "-l", "job-name=alice-r1", "-o", "json"] in calls
    assert ["get", "pods", "-l", "ray.io/cluster=alice-r1-abcde", "-o", "json"] in calls
    _log(exp, "alice-r2", last_step=19, last_age_s=30)   # the last step of a default 20-step run
    r, _ = _poll(cluster, tmp_path, "alice-r2", {"objects": {"rayjob/alice-r2": running()}})
    assert r["status"] == "done" and r["recommendation"] == "teardown", r
    _log(exp, "alice-r3", last_step=5, last_age_s=7200)
    r, _ = _poll(cluster, tmp_path, "alice-r3", {"objects": {"rayjob/alice-r3": running()}})
    assert r["status"] == "stuck" and r["recommendation"] == "triage" and r["secs_since_last_perf"] >= 7200, r


def test_poll_target_follows_the_runs_own_num_rollout(cluster, tmp_path):
    exp = tmp_path / "exp"
    _log(exp, "alice-p1", last_step=19, last_age_s=30, num_rollout=23)   # profiled: setup 20, warm-up 21, record 22
    r, _ = _poll(cluster, tmp_path, "alice-p1", {"objects": {"rayjob/alice-p1": running()}})
    assert r["status"] == "running" and (r["target_step"], r["target_source"]) == (22, "num_rollout"), r
    _log(exp, "alice-p2", last_step=22, last_age_s=30, num_rollout=23)
    r, _ = _poll(cluster, tmp_path, "alice-p2", {"objects": {"rayjob/alice-p2": running()}})
    assert r["status"] == "done", r
    _log(exp, "alice-p3", last_step=19, last_age_s=30)                   # no dump yet: the RayJob spec
    job = dict(running(), spec={"entrypoint": "python3 train.py \\\n  --num-rollout 24 \\\n  --seed 1"})
    r, _ = _poll(cluster, tmp_path, "alice-p3", {"objects": {"rayjob/alice-p3": job}})
    assert r["status"] == "running" and (r["target_step"], r["target_source"]) == (23, "num_rollout"), r
    proj = tmp_path / "attune-projects" / "c1"
    proj.mkdir(parents=True)
    (proj / "tracked_jobs.jsonl").write_text('{"run_id": "alice-p3", "target_steps": 19}\n')
    r, _ = _poll(cluster, tmp_path, "alice-p3", {"objects": {"rayjob/alice-p3": job}},
                 AUTOTUNE_PROJECT_ROOT=str(tmp_path / "attune-projects"))
    assert r["status"] == "running" and r["target_step"] == 23, r    # the run itself outranks a stale record
    r, _ = _poll(cluster, tmp_path, "alice-p3", {"objects": {"rayjob/alice-p3": running()}},
                 AUTOTUNE_PROJECT_ROOT=str(tmp_path / "attune-projects"))
    assert r["status"] == "done" and r["target_source"] == "tracked_jobs", r
    r, _ = _poll(cluster, tmp_path, "alice-p1", {"objects": {"rayjob/alice-p1": running()}}, TARGET_STEPS="19")
    assert r["status"] == "done" and r["target_source"] == "TARGET_STEPS", r


def test_poll_matches_pods_exactly(cluster, tmp_path):
    _log(tmp_path / "exp", "alice-r1", last_step=3, last_age_s=30)
    state = {"objects": {"rayjob/alice-r1": running()},
             "pods": {"job-name=alice-r1-b": [{"metadata": {"name": "alice-r1-b-x1"}, "status": {"phase": "Failed"}}]}}
    r, _ = _poll(cluster, tmp_path, "alice-r1", state)
    assert r["status"] == "running" and r["driver_pod_phase"] is None, r


def test_poll_api_error_and_absent(cluster, tmp_path):
    r, _ = _poll(cluster, tmp_path, "alice-r1", {"fail": {"get rayjob": "error: You must be logged in"}})
    assert r["status"] == "api-error" and r["recommendation"] == "retry", r
    r, _ = _poll(cluster, tmp_path, "alice-r1", {})
    assert r["status"] == "absent", r


# ---------------- read_metrics.sh ----------------
def _metrics(cluster, tmp_path, text, *args, **env):
    log = tmp_path / "train.txt"
    log.write_text(text)
    p, calls = cluster("read_metrics.sh", str(log), *args, **env)
    return p, (json.loads(p.stdout) if p.stdout.strip() else None), calls


def test_read_metrics_window_and_validity(cluster, tmp_path):
    p, r, calls = _metrics(cluster, tmp_path, fixtures.train_log(steps=14), "4", "13", "16")
    assert p.returncode == 0 and r["valid"] and calls == [], p.stderr
    assert r["window_n"] == 10 and r["aggregate_tok_per_s"] == 900.0 and r["aggregate_tok_per_s_per_gpu"] == 56.25
    p, r, _ = _metrics(cluster, tmp_path, fixtures.train_log(steps=12), "4", "13", "16")
    assert p.returncode == 4 and not r["valid"] and "incomplete window" in r["invalid_reason"]
    slow = fixtures.train_log(steps=14).replace("perf 8: {'perf/actor_train_tok_per_s': 1000.0, "
                                                "'perf/actor_train_time': 9.0, 'perf/step_time': 10.0}",
                                                "perf 8: {'perf/actor_train_tok_per_s': 1000.0, "
                                                "'perf/actor_train_time': 9.0, 'perf/step_time': 15.0}")
    p, r, _ = _metrics(cluster, tmp_path, slow, "4", "13", "16")
    assert p.returncode == 4 and "timing check" in r["invalid_reason"]


def test_read_metrics_default_window(cluster, tmp_path):
    p, r, _ = _metrics(cluster, tmp_path, fixtures.train_log(steps=20), AUTOTUNE_NUM_GPUS="16")
    assert p.returncode == 0 and r["valid"], p.stderr
    assert r["window"] == [5, 19] and r["window_n"] == 15 and r["aggregate_tok_per_s_per_gpu"] == 56.25
    p, r, _ = _metrics(cluster, tmp_path, fixtures.train_log(steps=23), AUTOTUNE_NUM_GPUS="16")   # a profiled run
    assert p.returncode == 0 and r["window"] == [5, 19] and r["window_n"] == 15, p.stderr


def test_read_metrics_reads_miles_timestamps(cluster, tmp_path):
    """miles logs "[<time> <worker>] file.py:N - perf N: {...}"; the timing check must still find the time."""
    log = re.sub(r"^\[([^\]]+)\]", r"[\1 actor_cell0_rank0]", fixtures.train_log(steps=20), flags=re.M)
    assert "actor_cell0_rank0] train_metric_utils.py" in log
    p, r, _ = _metrics(cluster, tmp_path, log, "5", "19", "16")
    assert p.returncode == 0 and r["valid"], r.get("invalid_reason")


def test_read_metrics_through_the_log_pod(cluster, tmp_path):
    p, r, calls = _metrics(cluster, tmp_path, fixtures.train_log(steps=14), "4", "13", "16",
                           AUTOTUNE_LOG_POD="alice-log-access")
    assert p.returncode == 0 and r["valid"], p.stderr
    assert calls == [["exec", "alice-log-access", "--", "cat", "--", str(tmp_path / "train.txt")]]


def test_read_metrics_reports_read_failures_apart_from_missing_perf_lines(cluster, tmp_path):
    p, _, _ = _metrics(cluster, tmp_path, "no perf here\n", "4", "13", "16")
    assert p.returncode == 3 and "no perf lines" in p.stderr
    p, _, _ = _metrics(cluster, tmp_path, fixtures.train_log(steps=14), "4", "13", "16",
                       AUTOTUNE_LOG_POD="alice-log-access", state={"fail": {"exec": "error: pod not found"}})
    assert p.returncode == 5 and "cannot read the log" in p.stderr and "pod not found" in p.stderr
    p, _, _ = _metrics(cluster, tmp_path, "x\n", "4", "13")
    assert p.returncode == 2 and "num_gpus is required" in p.stderr
