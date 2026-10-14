# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One baseline and one candidate through the documented loop, with a fake kubectl and no cluster.

Each step is the command the skill tells the agent to type: the command guard must allow it under the launcher's
environment, and it then runs for real in the campaign directory. The fake kubectl records every call, keeps applied
RayJobs and ConfigMaps in a state file (every RayJob has already succeeded), and runs `exec <log pod> -- <cmd>`
locally, so the test's files play the shared volume. The Stop hook blocks until the agent's STOP row.
Run: python3 -m pytest harness/tests"""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import command_guard as guard

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


def test_one_campaign_through_the_documented_loop(campaign, monkeypatch):
    root, exp = campaign.root, campaign.shared / "experiments"
    bin_dir = root.parent / "bin"
    bin_dir.mkdir()
    (bin_dir / "kubectl").write_text(FAKE_KUBECTL.replace("@PYTHON@", sys.executable))
    (bin_dir / "kubectl").chmod(0o755)
    calls_file = root.parent / "kubectl_calls.jsonl"
    (root / "proj-h100.yaml").write_text(fixtures.BASE_YAML)
    (root / "proj-h100" / "yamls").mkdir(parents=True)

    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "PYTHONDONTWRITEBYTECODE": "1",
           "FAKE_KUBECTL_LOG": str(calls_file), "FAKE_KUBECTL_STATE": str(root.parent / "kubectl_state.json"),
           **campaign.env, "CAMPAIGN_DIR": str(root), "AUTOTUNE_KUBECTL_CONTEXT": "ctx", "KUBECTL_NS": "ns",
           "AUTOTUNE_USER": "alice", "AUTOTUNE_OWNER_LABEL": "owner", "ATTUNE_RUN_PREFIX": "alice-tune-",
           "AUTOTUNE_NUM_GPUS": str(GPUS), "AUTOTUNE_INSTANCE_TYPES": "p5.48xlarge",
           "AUTOTUNE_EXPERIMENTS_DIR": str(exp), "AUTOTUNE_GPUH_LOG": str(root / "gpuh.jsonl"),
           "AUTOTUNE_BUDGET": "20", "AUTOTUNE_GPUH_BUDGET": "100", "AUTOTUNE_MAX_STALL": "3"}
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    def agent(cmd: str) -> str:
        """The command as the agent's Bash tool would run it, after the PreToolUse guard."""
        decision = guard.classify_command(cmd, cwd=str(root))
        assert decision.allow, (cmd, decision.reason)
        p = subprocess.run(["bash", "-c", cmd], cwd=root, env=env, capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, (cmd, p.stdout, p.stderr)
        return p.stdout

    def stop_gate() -> str:
        hook = REPO / "harness" / "hooks" / "stop_gate.py"
        p = subprocess.run([sys.executable, str(hook)], input="{}", env=dict(env, AUTOTUNE_AUTONOMOUS="1"),
                           capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr
        return p.stdout

    def run(run_id: str, kind: str, extra: str = "") -> dict:
        yaml_path = f"proj-h100/yamls/{run_id}.yaml"
        log = exp / run_id / "logs" / "20261012-000000" / "train.log"
        log.parent.mkdir(parents=True)
        log.write_text(fixtures.train_log(steps=14, counts=False))  # what the job's tee writes
        agent(f"python3 {SKILL}/make_run_yaml.py --base proj-h100.yaml --run-id {run_id} --out {yaml_path} "
              f"--benchmark {extra}")
        assert "free=3" in agent(f"python3 {SKILL}/check_gpu_capacity.py --baseline {yaml_path}")
        track = json.dumps({"run_id": run_id, "kind": kind, "target_steps": 13})
        agent(f"echo {shlex.quote(track)} >> proj-h100/tracked_jobs.jsonl")
        agent(f'kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" apply -f {yaml_path}')
        agent(f"python3 {SKILL}/run_gpuh.py {run_id} --launch --gpus {GPUS}")
        poll = json.loads(agent(f"bash {SKILL}/poll_run.sh {run_id}"))
        assert poll["status"] == "done" and poll["log_path"] == str(log), poll
        metrics = json.loads(agent(f"bash {SKILL}/read_metrics.sh {poll['log_path']} 4 13 {GPUS}"))
        assert metrics["valid"] and metrics["aggregate_tok_per_s_per_gpu"] == 37.5, metrics
        spent = json.loads(agent(f"python3 {SKILL}/run_gpuh.py {run_id} --record --status done"))
        assert spent["gpuh"] == 12.0, spent
        return {"run_id": run_id, "kind": kind, "status": "completed", "metrics": metrics, "profiled": False,
                "profile_reason": "e2e check", "gpuh": spent["gpuh"], "gpuh_cumulative": spent["cum_gpuh"]}

    def finalize(row: dict) -> None:
        agent(f"echo {shlex.quote(json.dumps(row))} >> proj-h100/AUTOTUNE_LEDGER.jsonl")
        agent(f"bash {SKILL}/teardown_run.sh {row['run_id']}")
        agent(f"grep -v '\"{row['run_id']}\"' proj-h100/tracked_jobs.jsonl > proj-h100/tracked.tmp; "
              "mv proj-h100/tracked.tmp proj-h100/tracked_jobs.jsonl")

    finalize(dict(run("alice-tune-proj-h100-base", "baseline"), base_run=None))
    assert "Campaign not converged" in stop_gate()
    ranked = run("alice-tune-proj-h100-tp1", "ranked", "--set tensor-model-parallel-size=1")
    finalize(dict(ranked, base_run="alice-tune-proj-h100-base", config_diff={"tensor-model-parallel-size": "2 -> 1"},
                  verdict="tie", compared_leader_run_id="alice-tune-proj-h100-base", delta_vs_leader_pct=0.0,
                  delta_vs_baseline_pct=0.0))
    out = stop_gate()
    assert "Campaign not converged" in out and "24.0/100 GPU-h spent, 1 ranked results" in out, out
    stop = {"kind": "stop", "declared_at": "2026-10-12T01:00:00Z", "stop_reason": "agent_stop",
            "gpuh_cumulative": 24.0, "reason": "e2e check", "leader": "alice-tune-proj-h100-base",
            "neighbours": [{"run_id": "alice-tune-proj-h100-tp1", "axis": "TP2 -> TP1"}],
            "diagnosis": "alice-tune-proj-h100-tp1: a tie"}
    agent(f"echo {shlex.quote(json.dumps(stop))} >> proj-h100/AUTOTUNE_LEDGER.jsonl")
    assert stop_gate() == ""

    assert (root / "proj-h100" / "tracked_jobs.jsonl").read_text() == ""
    assert json.loads((root.parent / "kubectl_state.json").read_text()) == {}, "teardown left objects behind"
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
    cluster_wide = [c for c in calls if c[2:4] == ["get", "nodes"] or (c[2:4] == ["get", "pods"] and "-A" in c)]
    assert len(cluster_wide) == 4  # check_gpu_capacity.py lists nodes and the pods of every namespace
    assert all(c[:2] == ["--context", "ctx"] for c in calls), calls
    assert all(c[2:4] == ["-n", "ns"] for c in calls if c not in cluster_wide), calls
    assert all(c[4:6] == ["exec", "logpod"] for c in calls if "exec" in c)
    assert sum(c[4] == "delete" for c in calls if c not in cluster_wide) == 4  # RayJob and ConfigMap of each run
