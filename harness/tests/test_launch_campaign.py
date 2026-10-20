# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""launch_campaign.sh with a fake claude and kubectl: it refuses a shared run prefix, a writable log pod and an inbox
inside the campaign or the tree's skill/ and harness/, makes the campaign's read-only copies (the lever catalog among
them), config.env (with the manifest's model, GPU, image and framework) and integrity record (with the dataset's hash
and the source tree), gives the session an ATTUNE_HOME that names the copies and no lessons inbox, renders and records
the web switch, holds a later launch to the campaign's settings, relaunches from config.example.env, warns at session
end of a write into the tree, copies LESSONS.md to the inbox at session end (or says it stays), and exits 3 once the
campaign fails its integrity check, after tearing down its tracked runs and without exporting its lessons, even when
the violation record cannot be written.
Run: python3 -m pytest harness/tests"""
import hashlib
import json
import os
import shutil
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
                        "control": os.environ.get("ATTUNE_CONTROL_DIR"), "home": os.environ.get("ATTUNE_HOME"),
                        "inbox": os.environ.get("ATTUNE_LESSONS_INBOX")}) + "\\n")
action = os.environ.get("FAKE_CLAUDE_ACTION", "")
skill = os.path.join(os.environ["ATTUNE_CONTROL_DIR"], "skill")
os.makedirs("proj", exist_ok=True)
if action in ("append", "tamper", "tamper-unrecorded"):
    with open("proj/tracked_jobs.jsonl", "a") as f:
        f.write('{"run_id": "alice-tune-r1", "target_steps": 19}\\n')
    with open("proj/LESSONS.md", "a") as f:
        f.write("## TP1 ties TP2 here\\n- Claim: ...\\n")
if action == "append":
    with open("proj/AUTOTUNE_LEDGER.jsonl", "a") as f:
        f.write('{"run_id": "alice-tune-base", "kind": "baseline", "status": "completed"}\\n')
    with open(os.path.join(skill, "past_tuning_projects", "proj_c1_summary.md"), "w") as f:
        f.write("# proj\\n")
elif action == "write-tree":   # a Bash heredoc into the tree the campaign was copied from
    with open(os.path.join(os.environ["FAKE_TREE"], "skill", "references", "levers.md"), "a") as f:
        f.write("- a rule\\n")
elif action in ("write-copy", "write-home"):
    path = (os.path.join(skill, "scripts", "read_metrics.sh") if action == "write-copy" else
            os.path.join(os.environ["ATTUNE_HOME"], "skill", "references", "levers.md"))
    try:
        open(path, "a").write("# tuned\\n")
    except PermissionError:
        sys.exit(0)
    sys.exit("%s was writable" % path)
elif action in ("tamper", "tamper-unrecorded"):
    path = os.path.join(skill, "scripts", "read_metrics.sh")
    os.chmod(path, 0o644)
    open(path, "a").write("# tuned\\n")
    if action == "tamper-unrecorded":   # the violation record cannot be written
        control = os.environ["ATTUNE_CONTROL_DIR"]
        os.chmod(control, 0o755)
        os.mkdir(os.path.join(control, "integrity_violation.json.tmp"))
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
    assert not os.access(skill / "references" / "levers.md", os.W_OK)
    assert os.access(skill / "past_tuning_projects", os.W_OK)
    assert not list(control.rglob("__pycache__"))
    assert (campaign / ".claude" / "skills" / "autotune-training-throughput").resolve() == skill
    config = (control / "config.env").read_text().splitlines()
    for line in ("ATTUNE_RUN_PREFIX=alice-tune-", "KUBECTL_NS=ns", "AUTOTUNE_GPUH_BUDGET=''",
                 "AUTOTUNE_ALLOW_WEB=1", f"CAMPAIGN_DIR={campaign}", f"ATTUNE_CONTROL_DIR={control}",
                 "ATTUNE_MODEL_NAME=m", "ATTUNE_GPU_NAME=H100", "TRAINING_IMAGE=img@sha256:0", "ATTUNE_FRAMEWORK=''"):
        assert line in config, line
    assert any(x.startswith("ATTUNE_CAMPAIGN_ID=alice-tune-2") for x in config)
    settings = json.loads((campaign / ".claude" / "settings.json").read_text())
    hooks = [h["command"] for e in settings["hooks"].values() for m in e for h in m["hooks"]]
    assert all(c.startswith(f"python3 {control}/harness/hooks/") and c.endswith("|| exit 2") for c in hooks), hooks
    assert "WebSearch" not in settings["permissions"]["deny"], "web access is on by default"
    (call,) = launch.calls()
    assert call["autonomous"] == "1" and call["control"] == str(control)
    assert call["home"] == str(control) and call["inbox"] is None
    assert "--disallowedTools" not in call["args"]
    record = json.loads((control / "integrity.json").read_text())
    assert str(skill / "scripts" / "read_metrics.sh") in record["files"]
    assert str(skill / "references" / "levers.md") in record["files"] and record["source"] == str(REPO)
    assert "skill/references/levers.md" in record["source_files"] and "harness/lessons.py" in record["source_files"]
    assert "lessons: none written" in p.stdout and "unchanged since launch" in p.stdout
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
    ({"AUTOTUNE_ALLOW_WEB": "yes"}, "AUTOTUNE_ALLOW_WEB"), ({"ATTUNE_LESSONS_INBOX": "inbox"}, "absolute path"),
    ({"AUTOTUNE_DATASET": "/no/such/file.parquet"}, "finds no dataset at /no/such/file.parquet"),
    ({"AUTOTUNE_DATASET": "", "SFT_DATA": "", "ATTUNE_BASELINE_YAML": "/x.yaml"}, "no dataset")])
def test_refusals(launch, env, message):
    p = launch(**env)
    assert p.returncode == 2 and message in p.stderr, p.stderr
    assert not launch.control.exists() and not launch.calls()


def _tree(launch, drop=()):
    """A copy of this repository's skill/ and harness/ to launch from (`drop`: files left out)."""
    tree = launch.tmp / "tree"
    for d in ("skill", "harness"):
        shutil.copytree(REPO / d, tree / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    for rel in drop:
        (tree / rel).unlink()
    return tree


@pytest.mark.parametrize("where", ["campaign", "inside", "linked", "skill", "harness"])
def test_an_inbox_in_the_campaign_or_the_trees_skill_or_harness_is_refused(launch, where):
    tree = _tree(launch)
    launch.campaign.mkdir()
    (launch.tmp / "link").symlink_to(launch.campaign)
    inbox = {"campaign": launch.campaign, "inside": launch.campaign / "proj", "linked": launch.tmp / "link" / "x",
             "skill": tree / "skill" / "inbox", "harness": tree / "harness"}[where]
    p = launch(ATTUNE_HOME=str(tree), ATTUNE_LESSONS_INBOX=str(inbox))
    assert p.returncode == 2 and "outside the campaign directory" in p.stderr and "refused" in p.stderr, p.stderr
    assert not launch.control.exists() and not launch.calls()


def test_a_later_launch_refuses_an_inbox_inside_the_campaign(launch):
    assert launch().returncode == 0
    p = launch("--session", "abc", ATTUNE_LESSONS_INBOX=str(launch.campaign / "inbox"))
    assert p.returncode == 2 and "refused" in p.stderr, p.stderr
    assert len(launch.calls()) == 1


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


def test_the_sessions_attune_home_reaches_the_copy_not_the_tree(launch):
    tree = launch.tmp / "tree"
    for d in ("skill", "harness"):
        shutil.copytree(REPO / d, tree / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    levers = tree / "skill" / "references" / "levers.md"
    before = levers.read_bytes()
    p = launch(FAKE_CLAUDE_ACTION="write-home", ATTUNE_HOME=str(tree))
    assert p.returncode == 0, p.stderr   # the session's $ATTUNE_HOME/skill is the read-only copy
    assert launch.calls()[0]["home"] == str(launch.control) and levers.read_bytes() == before
    assert json.loads((launch.control / "integrity.json").read_text())["source"] == str(tree)


def _campaign_id(launch):
    lines = (launch.control / "config.env").read_text().splitlines()
    (line,) = [x for x in lines if x.startswith("ATTUNE_CAMPAIGN_ID=")]
    return line.split("=", 1)[1]


def test_lessons_go_to_the_inbox_at_session_end(launch):
    inbox = launch.tmp / "inbox"
    p = launch(FAKE_CLAUDE_ACTION="append", ATTUNE_LESSONS_INBOX=str(inbox), ATTUNE_FRAMEWORK="slime 0.3")
    assert p.returncode == 0, p.stderr
    assert launch.calls()[0]["inbox"] is None, "the session does not see the inbox"
    cid, lessons = _campaign_id(launch), launch.campaign / "proj" / "LESSONS.md"
    out = inbox / cid
    assert (out / "proj" / "LESSONS.md").read_bytes() == lessons.read_bytes() and "lessons: exported" in p.stdout
    manifest = json.loads((out / "manifest.json").read_text())
    record = json.loads((launch.control / "integrity.json").read_text())
    levers = launch.control / "skill" / "references" / "levers.md"
    version = {"levers_sha256": hashlib.sha256(levers.read_bytes()).hexdigest(), "commit": record["source_commit"],
               "dirty": record["source_dirty"]}
    assert manifest["catalog_version"] == version
    assert (manifest["campaign_id"], manifest["model"], manifest["gpu"], manifest["framework"], manifest["image"]) == (
        cid, "m", "H100", "slime 0.3", "img@sha256:0")
    assert manifest["lessons"] == [{"folder": "proj", "source": str(lessons),
                                    "sha256": hashlib.sha256(lessons.read_bytes()).hexdigest()}]
    assert manifest["campaign_dir"] == str(launch.campaign)
    with open(lessons, "a") as f:   # a later session, from a shell without the four settings, replaces the copy
        f.write("## another\n")
    p = launch("--session", "abc", ATTUNE_LESSONS_INBOX=str(inbox),
               unset=("ATTUNE_MODEL_NAME", "ATTUNE_GPU_NAME", "TRAINING_IMAGE"))
    assert p.returncode == 0, p.stderr
    assert (out / "proj" / "LESSONS.md").read_bytes() == lessons.read_bytes()
    later = json.loads((out / "manifest.json").read_text())
    assert (later["model"], later["gpu"], later["framework"], later["image"], later["catalog_version"]) == (
        "m", "H100", "slime 0.3", "img@sha256:0", version)
    p = launch("--session", "abc", ATTUNE_FRAMEWORK="slime 0.4")   # a different value is refused
    assert p.returncode == 2 and "ATTUNE_FRAMEWORK" in p.stderr, p.stderr


def test_lessons_stay_in_the_campaign_without_an_inbox(launch):
    p = launch(FAKE_CLAUDE_ACTION="append")
    assert p.returncode == 0, p.stderr
    assert "ATTUNE_LESSONS_INBOX is unset, so they stay in the campaign folder" in p.stdout


def test_a_failed_copy_does_not_fail_the_session(launch):
    blocked = launch.tmp / "not-a-directory"
    blocked.write_text("")
    p = launch(FAKE_CLAUDE_ACTION="append", ATTUNE_LESSONS_INBOX=str(blocked))
    assert p.returncode == 0 and "lessons: not exported" in p.stderr and "WARNING" in p.stderr, p.stderr


@pytest.mark.parametrize("detach", [False, True])
def test_a_write_into_the_tree_is_reported_at_session_end(launch, detach):
    tree = _tree(launch)
    p = launch(*(["--detach"] if detach else []), ATTUNE_HOME=str(tree), FAKE_TREE=str(tree),
               FAKE_CLAUDE_ACTION="write-tree")
    assert p.returncode == 0, p.stderr
    out = _status(launch) if detach else p.stderr
    assert "WARNING: the source tree %s changed" % tree in out and "skill/references/levers.md: modified" in out, out
    if detach:
        assert out.splitlines()[-1] == "exit 0"
    record = json.loads((launch.control / "integrity.json").read_text())
    assert record["source_checks"][-1]["changes"] == [{"path": "skill/references/levers.md", "problem": "modified"}]
    assert not (launch.control / "integrity_violation.json").exists()


def test_a_campaign_without_the_lessons_export_skips_both_steps(launch):
    tree = _tree(launch, drop=("harness/lessons.py",))
    p = launch(ATTUNE_HOME=str(tree), ATTUNE_LESSONS_INBOX=str(launch.campaign))
    assert p.returncode == 0, p.stderr
    assert "lessons" not in p.stdout + p.stderr and "source tree" not in p.stdout + p.stderr


def test_the_export_follows_the_checks_result_not_the_violation_record(launch):
    inbox = launch.tmp / "inbox"
    p = launch(FAKE_CLAUDE_ACTION="tamper-unrecorded", ATTUNE_LESSONS_INBOX=str(inbox))
    assert p.returncode == 3 and "could not be written" in p.stderr, p.stderr
    assert not (launch.control / "integrity_violation.json").exists()
    assert (launch.campaign / "proj" / "LESSONS.md").exists()
    assert "lessons: not exported, the campaign failed its integrity check" in p.stdout and not inbox.exists()


def test_a_tampered_campaign_tears_down_and_exits_3_now_and_later(launch):
    launch.objects({"rayjob/alice-tune-r1": RUN_OBJECT})
    inbox = launch.tmp / "inbox"
    p = launch(FAKE_CLAUDE_ACTION="tamper", ATTUNE_LESSONS_INBOX=str(inbox))
    assert p.returncode == 3 and "integrity" in p.stderr and "live runs torn down: alice-tune-r1" in p.stderr, \
        p.stderr
    assert "not exported, the campaign failed its integrity check" in p.stdout and not inbox.exists()
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


def _status(launch):
    """run.exit once the detached session has written its status."""
    exit_file = launch.campaign / "run.exit"
    for _ in range(400):
        if exit_file.exists() and "exit " in exit_file.read_text():
            break
        time.sleep(0.05)
    return exit_file.read_text()


@pytest.mark.parametrize("action,status", [("append", "exit 0"), ("tamper", "exit 3"),
                                           ("tamper-unrecorded", "exit 3")])
def test_a_detached_session_writes_its_status(launch, action, status):
    launch.objects({"rayjob/alice-tune-r1": RUN_OBJECT})
    inbox = launch.tmp / "inbox"
    p = launch("--detach", FAKE_CLAUDE_ACTION=action, ATTUNE_LESSONS_INBOX=str(inbox))
    assert p.returncode == 0 and "detached" in p.stdout, p.stderr
    text = _status(launch)
    assert text.splitlines()[-1] == status
    assert (inbox / _campaign_id(launch) / "proj" / "LESSONS.md").exists() == (status == "exit 0"), text
    assert "unchanged since launch" in text
    if action == "tamper":
        assert "live runs torn down: alice-tune-r1" in text
    if status == "exit 3":
        assert "lessons: not exported, the campaign failed its integrity check" in text
