# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""launch_campaign.sh with a fake claude and kubectl: it refuses a shared run prefix and a writable log pod, makes the
campaign's read-only copies, config.env and integrity record (with the dataset's hash), renders and records the web
switch, holds a later launch to the campaign's settings, relaunches from config.example.env, and exits 3 once the
campaign fails its integrity check, after tearing down its tracked runs.
Run: python3 -m pytest harness/tests"""
import json
import os
import subprocess
import sys
import time

import pytest
from conftest import REPO, unlock

LAUNCHER = REPO / "harness" / "launch_campaign.sh"
FAKE_KUBECTL = '''#!@PYTHON@
import json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_KUBECTL_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
want = ["--context", os.environ.get("FAKE_CTX", "ctx"), "-n", os.environ.get("FAKE_NS", "ns")]
assert args[:4] == want, args
words, pod = args[4:], os.environ.get("FAKE_LOG_POD", "logpod")
state_file = os.environ.get("FAKE_KUBECTL_STATE", "")
state = json.load(open(state_file)) if os.path.exists(state_file) else {}
if words == ["get", "pod", pod, "-o", "json"]:
    mount = {"name": "fsx", "mountPath": "/shared"}
    if os.environ.get("FAKE_POD_READ_ONLY", "1") == "1":
        mount["readOnly"] = True
    print(json.dumps({"spec": {"volumes": [{"name": "fsx", "persistentVolumeClaim": {"claimName": "fsx"}}],
                               "containers": [{"name": "main", "volumeMounts": [mount]}]}}))
    sys.exit(0)
if words[:2] == ["exec", pod] and os.environ.get("FAKE_EXEC_FAIL") == "1":
    sys.exit(print("error: unable to upgrade connection", file=sys.stderr) or 1)
if words[:2] == ["exec", pod]:
    cmd = words[words.index("--") + 1:]
    if cmd[0] == "python3" and os.environ.get("FAKE_NO_PYTHON") == "1":
        sys.exit(print("exec: python3: executable file not found in $PATH", file=sys.stderr) or 126)
    sys.exit(subprocess.run(cmd).returncode)
if words[0] == "get" and len(words) >= 3:
    obj = state.get("%s/%s" % (words[1], words[2]))
    if obj is None:
        sys.exit(print('Error from server (NotFound): %s "%s" not found' % (words[1], words[2]), file=sys.stderr) or 1)
    sys.exit(print(json.dumps(obj)) or 0)
if words[0] == "delete":
    name = words[words.index("--field-selector") + 1].split("=", 1)[1]
    state.pop("%s/%s" % (words[1], name), None)
    json.dump(state, open(state_file, "w"))
    sys.exit(print("deleted") or 0)
sys.exit("fake kubectl: unexpected call %s" % args)
'''
FAKE_CLAUDE = '''#!@PYTHON@
import json, os, sys
with open(os.environ["FAKE_CLAUDE_LOG"], "a") as f:
    f.write(json.dumps({"args": sys.argv[1:], "autonomous": os.environ.get("AUTOTUNE_AUTONOMOUS"),
                        "control": os.environ.get("ATTUNE_CONTROL_DIR")}) + "\\n")
action = os.environ.get("FAKE_CLAUDE_ACTION", "")
skill = os.path.join(os.environ["ATTUNE_CONTROL_DIR"], "skill")
os.makedirs("proj", exist_ok=True)
if action in ("append", "tamper"):
    with open("proj/tracked_jobs.jsonl", "a") as f:
        f.write('{"run_id": "alice-tune-r1", "target_steps": 19}\\n')
if action == "append":
    with open("proj/AUTOTUNE_LEDGER.jsonl", "a") as f:
        f.write('{"run_id": "alice-tune-base", "kind": "baseline", "status": "completed"}\\n')
    with open(os.path.join(skill, "references", "levers.md"), "a") as f:
        f.write("- a lever rule\\n")
    with open(os.path.join(skill, "past_tuning_projects", "proj_c1_summary.md"), "w") as f:
        f.write("# proj\\n")
elif action == "write-copy":
    try:
        open(os.path.join(skill, "scripts", "read_metrics.sh"), "a").write("# tuned\\n")
    except PermissionError:
        sys.exit(0)
    sys.exit("the copy was writable")
elif action == "tamper":
    path = os.path.join(skill, "scripts", "read_metrics.sh")
    os.chmod(path, 0o644)
    open(path, "a").write("# tuned\\n")
'''
RUN_OBJECT = {"metadata": {"name": "alice-tune-r1", "labels": {"owner": "alice"}}}


@pytest.fixture
def launch(tmp_path):
    """launch(*args, unset=(), **env) -> CompletedProcess of launch_campaign.sh <campaign> *args."""
    tmp = tmp_path.resolve()
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    for name, text in (("kubectl", FAKE_KUBECTL), ("claude", FAKE_CLAUDE)):
        (bin_dir / name).write_text(text.replace("@PYTHON@", sys.executable))
        (bin_dir / name).chmod(0o755)
    campaign, dataset = tmp / "campaign", tmp / "data" / "train.parquet"
    dataset.parent.mkdir()
    dataset.write_bytes(b"rows")
    base = {k: v for k, v in os.environ.items() if not k.startswith(("AUTOTUNE_", "ATTUNE_", "KUBECTL_"))}
    base.pop("CAMPAIGN_DIR", None)
    fakes = {"PATH": "%s:%s" % (bin_dir, os.environ["PATH"]), "PYTHONDONTWRITEBYTECODE": "1",
             "FAKE_CLAUDE_LOG": str(tmp / "claude.jsonl"), "FAKE_KUBECTL_LOG": str(tmp / "kubectl.jsonl"),
             "FAKE_KUBECTL_STATE": str(tmp / "state.json")}
    settings = {"AUTOTUNE_USER": "alice", "ATTUNE_RUN_PREFIX": "alice-tune-", "AUTOTUNE_KUBECTL_CONTEXT": "ctx",
                "KUBECTL_NS": "ns", "AUTOTUNE_LOG_POD": "logpod", "AUTOTUNE_NUM_GPUS": "8", "AUTOTUNE_BUDGET": "20",
                "AUTOTUNE_DATASET": str(dataset), "TRAINING_IMAGE": "img@sha256:0", "ATTUNE_MODEL_NAME": "m",
                "ATTUNE_GPU_NAME": "H100", "ATTUNE_BASELINE_YAML": "/x.yaml", "SFT_DATA": "/d.parquet"}

    def run(*args, unset=(), **env):
        full = {k: v for k, v in {**base, **fakes, **settings, **env}.items() if k not in unset}
        return subprocess.run(["bash", str(LAUNCHER), str(campaign), *args], capture_output=True, text=True,
                              env=full, timeout=120)
    log = tmp / "claude.jsonl"
    run.campaign, run.control, run.dataset, run.tmp, run.fakes, run.base = (
        campaign, campaign / ".attune", dataset, tmp, fakes, base)
    run.calls = lambda: [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    run.kubectl = lambda: [json.loads(x)[4:] for x in (tmp / "kubectl.jsonl").read_text().splitlines()]
    run.objects = lambda objs: (tmp / "state.json").write_text(json.dumps(objs))
    yield run
    unlock(tmp)


def test_a_first_launch_makes_a_sealed_campaign(launch):
    p = launch()
    assert p.returncode == 0, p.stderr
    assert "web access: on (AUTOTUNE_ALLOW_WEB=1)" in p.stdout and "campaign id: alice-tune-" in p.stdout
    control, campaign = launch.control, launch.campaign
    skill = control / "skill"
    assert not os.access(skill / "scripts" / "read_metrics.sh", os.W_OK)
    assert not os.access(skill / "scripts", os.W_OK) and not os.access(control / "harness", os.W_OK)
    assert not os.access(control, os.W_OK) and not os.access(control / "config.env", os.W_OK)
    assert os.access(skill / "references" / "levers.md", os.W_OK)
    assert os.access(skill / "past_tuning_projects", os.W_OK)
    assert not list(control.rglob("__pycache__"))
    assert (campaign / ".claude" / "skills" / "autotune-training-throughput").resolve() == skill
    config = (control / "config.env").read_text().splitlines()
    for line in ("ATTUNE_RUN_PREFIX=alice-tune-", "KUBECTL_NS=ns", "AUTOTUNE_GPUH_BUDGET=''",
                 "AUTOTUNE_ALLOW_WEB=1", f"CAMPAIGN_DIR={campaign}", f"ATTUNE_CONTROL_DIR={control}"):
        assert line in config, line
    assert any(x.startswith("ATTUNE_CAMPAIGN_ID=alice-tune-2") for x in config)
    settings = json.loads((campaign / ".claude" / "settings.json").read_text())
    hooks = [h["command"] for e in settings["hooks"].values() for m in e for h in m["hooks"]]
    assert all(c.startswith(f"python3 {control}/harness/hooks/") and c.endswith("|| exit 2") for c in hooks), hooks
    assert "WebSearch" not in settings["permissions"]["deny"], "web access is on by default"
    (call,) = launch.calls()
    assert call["autonomous"] == "1" and call["control"] == str(control)
    assert "--disallowedTools" not in call["args"]
    record = json.loads((control / "integrity.json").read_text())
    assert str(skill / "scripts" / "read_metrics.sh") in record["files"]
    assert record["dataset"]["path"] == str(launch.dataset) and len(record["dataset"]["parquet_sha256"]) == 64


def test_web_access_off(launch):
    p = launch(AUTOTUNE_ALLOW_WEB="0")
    assert p.returncode == 0 and "web access: off (AUTOTUNE_ALLOW_WEB=0)" in p.stdout, p.stderr
    deny = json.loads((launch.campaign / ".claude" / "settings.json").read_text())["permissions"]["deny"]
    assert {"WebSearch", "WebFetch"} <= set(deny)
    args = launch.calls()[0]["args"]
    assert args[args.index("--disallowedTools") + 1:][:2] == ["WebFetch", "WebSearch"]
    assert "AUTOTUNE_ALLOW_WEB=0" in (launch.control / "config.env").read_text().splitlines()
    p = launch("--session", "abc", AUTOTUNE_ALLOW_WEB="1")   # a reused config that turns it back on is refused
    assert p.returncode == 2 and "AUTOTUNE_ALLOW_WEB" in p.stderr


@pytest.mark.parametrize("env,message", [
    ({"ATTUNE_RUN_PREFIX": "alice-"}, "campaign-specific"), ({"ATTUNE_RUN_PREFIX": "alice"}, "campaign-specific"),
    ({"ATTUNE_RUN_PREFIX": "tune-"}, "campaign-specific"), ({"ATTUNE_RUN_PREFIX": "Alice-Tune-"}, "campaign-specific"),
    ({"ATTUNE_RUN_PREFIX": ""}, "ATTUNE_RUN_PREFIX is not set"), ({"FAKE_POD_READ_ONLY": "0"}, "readOnly: true"),
    ({"AUTOTUNE_ALLOW_WEB": "yes"}, "AUTOTUNE_ALLOW_WEB"),
    ({"AUTOTUNE_DATASET": "/no/such/file.parquet"}, "finds no dataset at /no/such/file.parquet"),
    ({"AUTOTUNE_DATASET": "", "SFT_DATA": "", "ATTUNE_BASELINE_YAML": "/x.yaml"}, "no dataset")])
def test_refusals(launch, env, message):
    p = launch(**env)
    assert p.returncode == 2 and message in p.stderr, p.stderr
    assert not launch.control.exists() and not launch.calls()


def test_the_dataset_comes_from_the_baseline_when_unset(launch):
    baseline = launch.tmp / "base.yaml"
    baseline.write_text(f"spec:\n  entrypoint: python3 train.py --prompt-data '{launch.dataset}' --seed 1\n")
    p = launch(AUTOTUNE_DATASET="", ATTUNE_BASELINE_YAML=str(baseline))
    assert p.returncode == 0, p.stderr
    assert json.loads((launch.control / "integrity.json").read_text())["dataset"]["path"] == str(launch.dataset)


def test_variables_in_the_baselines_prompt_data_are_expanded(launch):
    baseline = launch.tmp / "base.yaml"
    baseline.write_text("spec:\n  entrypoint: python3 train.py --prompt-data ${AUTOTUNE_DATA_ROOT}/train.parquet\n")
    p = launch(AUTOTUNE_DATASET="", SFT_DATA="", ATTUNE_BASELINE_YAML=str(baseline), AUTOTUNE_DATA_ROOT="")
    assert p.returncode == 2 and "names a variable the settings do not set" in p.stderr, p.stderr
    p = launch(AUTOTUNE_DATASET="", ATTUNE_BASELINE_YAML=str(baseline), AUTOTUNE_DATA_ROOT=str(launch.dataset.parent))
    assert p.returncode == 0, p.stderr
    assert json.loads((launch.control / "integrity.json").read_text())["dataset"]["path"] == str(launch.dataset)


def test_a_clean_session_and_a_later_launch(launch):
    assert launch(FAKE_CLAUDE_ACTION="append").returncode == 0
    (launch.campaign / "resume_prompt.txt").write_text("The cluster was down; continue.\n")
    p = launch("--session", "abc", FAKE_CLAUDE_ACTION="append")
    assert p.returncode == 0, p.stderr
    args = launch.calls()[-1]["args"]
    assert args[args.index("-p") + 1] == "The cluster was down; continue."
    assert args[args.index("--resume") + 1] == "abc"
    p = launch("--session", "abc", KUBECTL_NS="other")
    assert p.returncode == 2 and "KUBECTL_NS" in p.stderr and "config.env" in p.stderr
    p = launch("--session", "abc", unset=("KUBECTL_NS", "ATTUNE_RUN_PREFIX"), AUTOTUNE_LOG_POD="")
    assert p.returncode == 0, p.stderr   # unset and empty values come from the campaign's config.env
    assert len(launch.calls()) == 3


def test_a_relaunch_from_the_example_config(launch):
    """The operator's flow: source config.example.env (values filled in), launch, source it again, resume."""
    example = (REPO / "config.example.env").read_text()
    fills = {"AUTOTUNE_DATASET=": f"AUTOTUNE_DATASET={launch.dataset}"}
    for a, b in fills.items():
        example = example.replace(a + " ", b + " ", 1)
    config = launch.tmp / "config.env"
    config.write_text(example)
    env = {**os.environ, **launch.fakes, "FAKE_CTX": "my-gpu-cluster", "FAKE_NS": "training",
           "FAKE_LOG_POD": "alice-log-access"}

    def source_and_launch(*args):
        script = 'set -a; . "$1"; set +a; shift; exec bash "$@"'
        return subprocess.run(["bash", "-c", script, "x", str(config), str(LAUNCHER), str(launch.campaign), *args],
                              capture_output=True, text=True, env=env, timeout=120)
    p = source_and_launch()
    assert p.returncode == 0, p.stderr
    p = source_and_launch("--session", "abc")
    assert p.returncode == 0, p.stderr
    assert len(launch.calls()) == 2


def test_the_copies_refuse_accidental_writes(launch):
    p = launch(FAKE_CLAUDE_ACTION="write-copy")
    assert p.returncode == 0, p.stderr


def test_a_tampered_campaign_tears_down_and_exits_3_now_and_later(launch):
    launch.objects({"rayjob/alice-tune-r1": RUN_OBJECT})
    p = launch(FAKE_CLAUDE_ACTION="tamper")
    assert p.returncode == 3 and "integrity" in p.stderr and "live runs torn down: alice-tune-r1" in p.stderr, \
        p.stderr
    violation = json.loads((launch.control / "integrity_violation.json").read_text())
    assert violation["where"] == "launch_campaign.sh (session end)"
    assert violation["violations"][0]["path"].endswith("scripts/read_metrics.sh")
    assert violation["teardown"]["runs"] == {"alice-tune-r1": "torn down"}
    assert ["delete", "rayjob", "-l", "owner=alice", "--field-selector", "metadata.name=alice-tune-r1",
            "--request-timeout=30s"] \
        in launch.kubectl()
    p = launch("--session", "abc")
    assert p.returncode == 3 and "start a new campaign directory" in p.stderr
    assert len(launch.calls()) == 1, "no session after a violation"


@pytest.mark.parametrize("change,why", [("rewrite", "the dataset changed"), ("delete", "the dataset is gone"),
                                        ("move", "the dataset is gone")])
def test_a_changed_or_missing_dataset_fails_the_later_launch(launch, change, why):
    assert launch().returncode == 0
    if change == "rewrite":
        launch.dataset.write_bytes(b"other rows")
    elif change == "delete":
        launch.dataset.unlink()
    else:
        launch.dataset.rename(launch.dataset.with_name("moved.parquet"))
    p = launch("--session", "abc")
    assert p.returncode == 3 and why in p.stderr, p.stderr
    assert len(launch.calls()) == 1


def test_a_log_pod_without_python3_hashes_the_file(launch):
    assert launch(FAKE_NO_PYTHON="1").returncode == 0
    recorded = json.loads((launch.control / "integrity.json").read_text())["dataset"]
    assert set(recorded) == {"path", "parquet_sha256"}
    launch.dataset.unlink()
    p = launch("--session", "abc", FAKE_NO_PYTHON="1")
    assert p.returncode == 3 and "the dataset is gone" in p.stderr, p.stderr


def test_an_unreachable_log_pod_holds_a_later_launch_without_failing_it(launch):
    assert launch().returncode == 0
    p = launch("--session", "abc", FAKE_EXEC_FAIL="1")
    assert p.returncode == 2 and "cannot be verified now" in p.stderr, p.stderr
    assert not (launch.control / "integrity_violation.json").exists()
    assert launch("--session", "abc").returncode == 0


@pytest.mark.parametrize("action,status", [("append", "exit 0"), ("tamper", "exit 3")])
def test_a_detached_session_writes_its_status(launch, action, status):
    launch.objects({"rayjob/alice-tune-r1": RUN_OBJECT})
    p = launch("--detach", FAKE_CLAUDE_ACTION=action)
    assert p.returncode == 0 and "detached" in p.stdout, p.stderr
    exit_file = launch.campaign / "run.exit"
    for _ in range(400):
        if exit_file.exists() and "exit " in exit_file.read_text():
            break
        time.sleep(0.05)
    text = exit_file.read_text()
    assert text.splitlines()[-1] == status
    if status == "exit 3":
        assert "live runs torn down: alice-tune-r1" in text
