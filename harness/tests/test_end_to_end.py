# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One baseline and one candidate through the documented loop, in a campaign launch_campaign.sh made, with a fake
kubectl and claude and no cluster.

The launcher sets the campaign up (copies, config.env, integrity record, dataset hash) and its fake session ends at
once. Each step after that is a tool call the skill leads the agent to make: the campaign's PreToolUse hook, run
through its subprocess entry point, must allow it, and a Bash command then runs for real in the campaign directory
with the campaign's settings, from the campaign's copy of the skill (Write and Edit are applied by the test). The
steps include a heredoc, a multi-line command, edits to RESULTS.md and the lever catalog, and a relaunch from an
environment with an empty AUTOTUNE_GPUH_LOG. The fake kubectl records every call, keeps applied RayJobs and ConfigMaps
in a state file (every RayJob has already succeeded), answers for a log pod that mounts the shared volume read-only,
and runs `exec <log pod> -- <cmd>` locally, so the test's files play the shared volume. The Stop hook blocks until
the agent's STOP row. A changed dataset at the end fails the campaign before shipping, tears down the live run, and
then fails it at the Stop hook and at ledger_append.py too.
Run: python3 -m pytest harness/tests"""
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import integrity
from conftest import unlock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "skill" / "scripts" / "tests"))
import fixtures  # noqa: E402

SKILL = ".claude/skills/autotune-training-throughput/scripts"
GPUS = 24  # fixtures.BASE_YAML: a head and two workers with 8 GPUs each
FAKE_KUBECTL = '''#!@PYTHON@
import json, os, subprocess, sys
import yaml
args = sys.argv[1:]
with open(os.environ["FAKE_KUBECTL_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
state_file = os.environ["FAKE_KUBECTL_STATE"]
state = json.load(open(state_file)) if os.path.exists(state_file) else {}
words, i = [], 0
while i < len(args):
    if args[i] in ("--context", "-n"):
        i += 2
    else:
        words.append(args[i])
        i += 1

def save():
    json.dump(state, open(state_file, "w"))

def done(out="", code=0, err=""):
    sys.stdout.write(out)
    sys.stderr.write(err)
    sys.exit(code)

if words[:3] == ["get", "pod", os.environ["AUTOTUNE_LOG_POD"]]:
    done(json.dumps({"spec": {"volumes": [{"name": "shared", "hostPath": {"path": "/x"}}], "containers": [
        {"name": "main", "volumeMounts": [{"name": "shared", "mountPath": "/shared", "readOnly": True}]}]}}))
if words[0] == "exec" and os.environ.get("FAKE_EXEC_FAIL") == "1":
    done(code=1, err="error: unable to upgrade connection\\n")
if words[0] == "exec":
    if words[1] != os.environ["AUTOTUNE_LOG_POD"]:
        done(code=1, err="fake kubectl: exec into %s\\n" % words[1])
    sys.exit(subprocess.run(words[words.index("--") + 1:]).returncode)
if words[0] == "apply":
    for d in yaml.safe_load_all(open(words[words.index("-f") + 1])):
        if d:
            d["status"] = {"jobStatus": "SUCCEEDED", "jobDeploymentStatus": "Complete",
                           "rayClusterName": d["metadata"]["name"] + "-raycluster",
                           "startTime": "2026-10-12T00:00:00Z", "endTime": "2026-10-12T00:30:00Z"}
            state["%s/%s" % (d["kind"].lower(), d["metadata"]["name"])] = d
    save()
    done("applied\\n")
if words[0] == "delete":
    key = "%s/%s" % (words[1], words[words.index("--field-selector") + 1].split("=", 1)[1])
    label, value = words[words.index("-l") + 1].split("=", 1)
    if key in state and state[key]["metadata"].get("labels", {}).get(label) == value:
        del state[key]
        save()
    done("%s deleted\\n" % words[1])
if words[:2] == ["get", "nodes"]:
    nodes = [{"metadata": {"name": "gpu-%d" % k, "labels": {"node.kubernetes.io/instance-type": "p5.48xlarge"}},
              "spec": {}, "status": {"allocatable": {"nvidia.com/gpu": "8"},
                                     "conditions": [{"type": "Ready", "status": "True"}]}} for k in range(3)]
    done(json.dumps({"items": nodes}))
if words[:2] == ["get", "pods"]:
    selector = words[words.index("-l") + 1] if "-l" in words else ""
    pods = []
    if selector.startswith("job-name="):
        pods = [{"metadata": {"name": selector[9:] + "-x", "creationTimestamp": "2026-10-12T00:00:00Z"},
                 "status": {"phase": "Succeeded"}}] if "rayjob/" + selector[9:] in state else []
    done(json.dumps({"items": pods}))
if words[:2] == ["get", "workloads"]:
    done(code=1, err="error: the server doesn't have a resource type workloads\\n")
if words[0] == "get":
    obj = state.get("%s/%s" % (words[1], words[2]))
    if obj is None:
        done(code=1, err='Error from server (NotFound): %s "%s" not found\\n' % (words[1], words[2]))
    done(json.dumps(obj))
done(code=3, err="fake kubectl: unexpected call %s\\n" % args)
'''


def test_one_campaign_through_the_documented_loop(campaign):
    root, exp = campaign.root, campaign.shared / "experiments"
    bin_dir = root.parent / "bin"
    bin_dir.mkdir()
    (bin_dir / "kubectl").write_text(FAKE_KUBECTL.replace("@PYTHON@", sys.executable))
    (bin_dir / "claude").write_text("#!/bin/sh\nexit 0\n")
    for name in ("kubectl", "claude"):
        (bin_dir / name).chmod(0o755)
    calls_file = root.parent / "kubectl_calls.jsonl"
    (root / "proj-h100.yaml").write_text(fixtures.BASE_YAML)
    (root / "proj-h100" / "yamls").mkdir(parents=True)
    dataset = campaign.shared / "perf-data" / "train.parquet"
    dataset.parent.mkdir()
    dataset.write_bytes(b"fixed rows")

    base = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(campaign.home),
            "FAKE_KUBECTL_LOG": str(calls_file), "FAKE_KUBECTL_STATE": str(root.parent / "kubectl_state.json")}
    settings = {"AUTOTUNE_KUBECTL_CONTEXT": "ctx", "KUBECTL_NS": "training", "AUTOTUNE_LOG_POD": "logpod",
                "AUTOTUNE_USER": "alice", "AUTOTUNE_OWNER_LABEL": "owner", "ATTUNE_RUN_PREFIX": "alice-tune-",
                "AUTOTUNE_NUM_GPUS": str(GPUS), "AUTOTUNE_INSTANCE_TYPES": "p5.48xlarge",
                "AUTOTUNE_EXPERIMENTS_DIR": str(exp), "AUTOTUNE_GPUH_LOG": str(root / "gpuh.jsonl"),
                "AUTOTUNE_BUDGET": "20", "AUTOTUNE_GPUH_BUDGET": "100", "AUTOTUNE_MAX_STALL": "3",
                "AUTOTUNE_DATASET": str(dataset), "TRAINING_IMAGE": "img@sha256:0", "ATTUNE_MODEL_NAME": "m",
                "ATTUNE_GPU_NAME": "H100", "ATTUNE_BASELINE_YAML": str(root / "proj-h100.yaml"),
                "SFT_DATA": str(dataset)}

    def launch(*args, **env):
        p = subprocess.run(["bash", str(REPO / "harness" / "launch_campaign.sh"), str(root), *args],
                           capture_output=True, text=True, env={**base, **settings, **env}, timeout=120)
        assert p.returncode == 0, p.stderr
        return p.stdout
    assert "web access: on" in launch()
    control = root / ".attune"
    env = dict(base, **integrity.read_config(control / "config.env"))   # the session's environment
    hook_env = dict(env, AUTOTUNE_AUTONOMOUS="1")
    calls_file.write_text("")

    def tool(name: str, **tool_input) -> str:
        """The campaign's PreToolUse hook, as Claude Code runs it: its decision for one tool call."""
        payload = json.dumps({"tool_name": name, "tool_input": tool_input, "cwd": str(root)})
        p = subprocess.run([sys.executable, str(control / "harness" / "hooks" / "pretooluse_guard.py")],
                           input=payload, env=hook_env, capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)["hookSpecificOutput"]["permissionDecision"] if p.stdout else "allow"

    def agent(cmd: str, code: int = 0) -> str:
        """A Bash tool call: the hook allows it, then it runs in the campaign directory."""
        assert tool("Bash", command=cmd) == "allow", cmd
        p = subprocess.run(["bash", "-c", cmd], cwd=root, env=env, capture_output=True, text=True, timeout=60)
        assert p.returncode == code, (cmd, p.stdout, p.stderr)
        return p.stdout if code == 0 else p.stdout + p.stderr

    def write(path: str, content: str) -> None:
        assert tool("Write", file_path=path, content=content) == "allow", path
        (root / path).write_text(content)

    def edit(path: str, old: str, new: str) -> None:
        assert tool("Edit", file_path=path, old_string=old, new_string=new) == "allow", path
        text = (root / path).read_text()
        (root / path).write_text(text.replace(old, new, 1))

    def stop_gate() -> str:
        hook = control / "harness" / "hooks" / "stop_gate.py"
        p = subprocess.run([sys.executable, str(hook)], input="{}", env=hook_env, capture_output=True, text=True,
                           timeout=60)
        assert p.returncode == 0, p.stderr
        return p.stdout

    logs = {}

    def run(run_id: str, kind: str, extra: str = "") -> dict:
        yaml_path = f"proj-h100/yamls/{run_id}.yaml"
        log = logs[run_id] = exp / run_id / "logs" / "20261012-000000" / "train.log"
        log.parent.mkdir(parents=True)
        log.write_text(fixtures.train_log(steps=20, counts=False, args={"num_rollout": "20"}))  # the job's tee
        agent(f"python3 {SKILL}/make_run_yaml.py --base proj-h100.yaml --run-id {run_id} --out {yaml_path} "
              f"--benchmark {extra}")
        assert "free=3" in agent(f"python3 {SKILL}/check_gpu_capacity.py --baseline {yaml_path}")
        track = json.dumps({"run_id": run_id, "kind": kind, "target_steps": 19})
        agent(f"echo {shlex.quote(track)} >> proj-h100/tracked_jobs.jsonl")
        agent(f'kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" \\\n  -n "$KUBECTL_NS" \\\n  apply -f {yaml_path}')
        agent(f"python3 {SKILL}/run_gpuh.py {run_id} --launch --gpus {GPUS}")
        return {"run_id": run_id, "kind": kind, "log": log}

    def harvest(launched: dict) -> dict:
        run_id, log = launched["run_id"], launched["log"]
        poll = json.loads(agent(f"bash {SKILL}/poll_run.sh {run_id}"))
        assert poll["status"] == "done" and poll["log_path"] == str(log), poll
        assert (poll["target_step"], poll["target_source"]) == (19, "num_rollout"), poll
        metrics = json.loads(agent(f"bash {SKILL}/read_metrics.sh {poll['log_path']} 5 19 {GPUS}"))
        assert metrics["valid"] and metrics["aggregate_tok_per_s_per_gpu"] == 37.5, metrics
        spent = json.loads(agent(f"python3 {SKILL}/run_gpuh.py {run_id} --record --status done"))
        assert spent["gpuh"] == 12.0, spent
        return {"run_id": run_id, "kind": launched["kind"], "status": "completed", "metrics": metrics,
                "profiled": False, "profile_reason": "e2e check", "gpuh": spent["gpuh"],
                "gpuh_cumulative": spent["cum_gpuh"], "hypothesis": "kubectl apply of the recipe; it's the base"}

    def finalize(row: dict) -> None:
        out = agent(f"python3 {SKILL}/ledger_append.py proj-h100/AUTOTUNE_LEDGER.jsonl - <<'EOF'\n"
                    f"{json.dumps(row)}\nEOF")
        assert json.loads(out)["appended"] is True, out
        agent(f"bash {SKILL}/teardown_run.sh {row['run_id']}")
        agent(f"grep -v '\"{row['run_id']}\"' proj-h100/tracked_jobs.jsonl > proj-h100/tracked.tmp; "
              "mv proj-h100/tracked.tmp proj-h100/tracked_jobs.jsonl")

    finalize(dict(harvest(run("alice-tune-proj-h100-base", "baseline")), base_run=None))
    write("proj-h100/RESULTS.md", "# proj-h100\n\n| run | tok/s/GPU |\n|---|---|\n| base | 37.5 |\n")
    assert "Campaign not converged" in stop_gate()
    ranked = harvest(run("alice-tune-proj-h100-tp1", "ranked", "--set tensor-model-parallel-size=1"))
    finalize(dict(ranked, base_run="alice-tune-proj-h100-base", config_diff={"tensor-model-parallel-size": "2 -> 1"},
                  verdict="tie", compared_leader_run_id="alice-tune-proj-h100-base", delta_vs_leader_pct=0.0,
                  delta_vs_baseline_pct=0.0))
    edit("proj-h100/RESULTS.md", "| base | 37.5 |\n", "| base | 37.5 |\n| tp1 | 37.5 (tie) |\n")
    levers = ".claude/skills/autotune-training-throughput/references/levers.md"
    last = (root / levers).read_text().splitlines(keepends=True)[-1]
    edit(levers, last, last + "- TP1 at this size ties TP2 when attention TP traffic is already overlapped.\n")
    assert tool("Edit", file_path=levers, old_string="# ", new_string="# (edited) ") == "deny"
    out = stop_gate()
    assert "Campaign not converged" in out and "24.0/100 GPU-h spent, 1 ranked results" in out, out

    # the session ends (an outage); the operator relaunches it from a config whose AUTOTUNE_GPUH_LOG is empty
    (root / "resume_prompt.txt").write_text("The session stopped; continue the campaign.\n")
    launch("--session", "abc", AUTOTUNE_GPUH_LOG="")

    # the final correctness check pairs the two unprofiled benchmark runs: no new launch
    agent("mkdir -p proj-h100/parity")
    for side, rid in (("base", "alice-tune-proj-h100-base"), ("win", "alice-tune-proj-h100-tp1")):
        agent(f'kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" exec "$AUTOTUNE_LOG_POD" -- '
              f"cat {logs[rid]} > proj-h100/parity/{side}_run.log")
    agent(f"python3 {SKILL}/correctness/pair_from_runs.py --base-log proj-h100/parity/base_run.log "
          "--base-yaml proj-h100/yamls/alice-tune-proj-h100-base.yaml --win-log proj-h100/parity/win_run.log "
          "--win-yaml proj-h100/yamls/alice-tune-proj-h100-tp1.yaml --pair proj-h100/parity/tp1")
    assert json.loads((root / "proj-h100" / "parity" / "tp1" / "gpus.json").read_text()) == {"base": GPUS, "win": GPUS}
    numpy = importlib.util.find_spec("numpy")
    if numpy:
        verdict = agent(f"python3 {SKILL}/correctness/parity_verdict.py proj-h100/parity/tp1")
        assert "tp1: PASS" in verdict and "probe counts off" in verdict, verdict
    stop = {"kind": "stop", "declared_at": "2026-10-12T01:00:00Z", "stop_reason": "agent_stop",
            "gpuh_cumulative": 24.0, "reason": "e2e check", "leader": "alice-tune-proj-h100-base",
            "neighbours": [{"run_id": "alice-tune-proj-h100-tp1", "axis": "TP2 -> TP1"}],
            "diagnosis": "alice-tune-proj-h100-tp1: a tie"}
    agent(f"python3 {SKILL}/ledger_append.py proj-h100/AUTOTUNE_LEDGER.jsonl {shlex.quote(json.dumps(stop))}")
    assert stop_gate() == ""

    assert (root / "proj-h100" / "tracked_jobs.jsonl").read_text() == ""
    assert json.loads((root.parent / "kubectl_state.json").read_text()) == {}, "teardown left objects behind"
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
    cluster_wide = [c for c in calls if c[2:4] == ["get", "nodes"] or (c[2:4] == ["get", "pods"] and "-A" in c)]
    assert len(cluster_wide) == 4  # check_gpu_capacity.py lists nodes and the pods of every namespace
    assert all(c[:2] == ["--context", "ctx"] for c in calls), calls
    assert all(c[2:4] == ["-n", "training"] for c in calls if c not in cluster_wide), calls
    assert all(c[4:6] == ["exec", "logpod"] for c in calls if "exec" in c)
    assert sum(c[4] == "delete" for c in calls if c not in cluster_wide) == 4  # RayJob and ConfigMap of each run
    dataset_reads = [c for c in calls if c[4:] == ["exec", "logpod", "-i", "--", "python3", "-", str(dataset)]]
    assert len(dataset_reads) == 1 + bool(numpy)   # the relaunch and the correctness check

    # an export in the agent's shell cannot change a campaign setting
    out = agent(f"KUBECTL_NS=default bash {SKILL}/teardown_run.sh alice-tune-proj-h100-tp1", code=2)
    assert "KUBECTL_NS" in out and "config.env" in out

    # a log pod that cannot read the dataset holds shipping back without failing the campaign
    hook_env["FAKE_EXEC_FAIL"] = env["FAKE_EXEC_FAIL"] = "1"
    if numpy:
        assert "cannot verify the campaign's dataset now" in agent(
            f"python3 {SKILL}/correctness/parity_verdict.py proj-h100/parity/tp1", code=4)
    assert not (control / "integrity_violation.json").exists()
    del hook_env["FAKE_EXEC_FAIL"], env["FAKE_EXEC_FAIL"]

    # a changed dataset fails the campaign before shipping, tears down the live run, and fails it everywhere after
    live = run("alice-tune-proj-h100-pp2", "ranked", "--set pipeline-model-parallel-size=2")
    assert "rayjob/alice-tune-proj-h100-pp2" in json.loads((root.parent / "kubectl_state.json").read_text())
    dataset.write_bytes(b"other rows")
    out = agent(f"python3 {SKILL}/correctness/parity_verdict.py proj-h100/parity/tp1", code=3)
    assert "no recipe ships" in out and "the dataset changed" in out and "torn down: alice-tune-proj-h100-pp2" in out
    assert json.loads((root.parent / "kubectl_state.json").read_text()) == {}, "the live run was not torn down"
    out = json.loads(stop_gate())
    assert out["continue"] is False and "the dataset changed" in out["stopReason"]
    killed = {"run_id": live["run_id"], "kind": "ranked", "status": "killed", "verdict": "killed"}
    out = agent(f"python3 {SKILL}/ledger_append.py proj-h100/AUTOTUNE_LEDGER.jsonl {shlex.quote(json.dumps(killed))}",
                code=3)
    assert "INTEGRITY VIOLATION" in out
    assert live["run_id"] not in (root / "proj-h100" / "AUTOTUNE_LEDGER.jsonl").read_text()
    unlock(root)
