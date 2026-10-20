# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""make_run_yaml.py: token-level flag edits (other flags on a line survive, comments are ignored, --add works without a
`2>&1 | tee` anchor), run ownership, the workload contract and confirm runs. Run: python3 -m pytest skill/scripts/tests"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "correctness"))
import frozen_args  # noqa: E402
import make_run_yaml as mry  # noqa: E402

SCRIPT = """apiVersion: v1
kind: ConfigMap
metadata:
  name: alice-tune-base-scripts
data:
  run.sh: |
    exec > >(tee -a /shared/alice/train.log) 2>&1
    # --pipeline-model-parallel-size 9 was slower; see notes
    python3 -m trainer --async \\
      --tensor-model-parallel-size 3 --pipeline-model-parallel-size 5 --context-parallel-size 2 \\
      --recompute-modules mod_a mod_b \\
      --micro-batch-size=1 \\
      --hf-checkpoint "$HF_CKPT" \\
      --num-epoch 1 \\
      --seed 1234
    echo done
---
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: alice-tune-base
  labels: {app: alice-tune-base}
spec:
  entrypoint: bash /scripts/run.sh
  rayClusterSpec:
    headGroupSpec:
      template:
        spec:
          containers:
            - name: ray-head
              image: img@sha256:abc
              resources: {limits: {nvidia.com/gpu: 8}}
"""

# A host-memory preflight python command with its own `2>&1 | tee` before the training command.
TWO = SCRIPT.replace(
    "    exec > >(tee -a /shared/alice/train.log) 2>&1\n"
    "    # --pipeline-model-parallel-size 9 was slower; see notes\n"
    "    python3 -m trainer --async \\\n",
    '    python3 /scripts/all_nodes.py 2 "bash /scripts/hostmem_preflight.sh $LOG_DIR --sample" \\\n'
    '      2>&1 | tee "$LOG_DIR/hostmem_preflight.log"\n'
    '    python3 /scripts/check_nodes.py --seed 7 2>&1 | tee "$LOG_DIR/check.log"\n'
    '    MODEL_ARGS_LINE="$(python3 tools/model_args.py my-model-full)"\n'
    "    python3 train.py \\\n").replace(
    "      --seed 1234\n    echo done",
    '      --seed 1234 \\\n      2>&1 | tee "$LOG_DIR/train.log"\n    echo done')


def commands(text):
    """{program of each python3 command: its joined text} of the run.sh commands."""
    _, cmds = mry._commands(text)
    return {j.split("python3 ", 1)[1].split()[0]: j for _, _, j in cmds if j.lstrip().startswith("python3 ")}


def flags(text):
    return {k: v for k, v in frozen_args.manifest_flags(frozen_args.strip_comments(text)).items()}


def clean_env(**extra):
    """The test's environment without the operator's AUTOTUNE_/ATTUNE_ settings, as operator alice."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AUTOTUNE_", "ATTUNE_"))}
    return dict(env, AUTOTUNE_USER="alice", **extra)


def make(tmp_path, base_text, run_id, *extra, **env):
    base = tmp_path / "base.yaml"
    base.write_text(base_text)
    out = tmp_path / ("%s.yaml" % run_id)
    p = subprocess.run([sys.executable, str(SCRIPTS / "make_run_yaml.py"), "--base", str(base), "--run-id", run_id,
                        "--out", str(out), *extra], capture_output=True, text=True, env=clean_env(**env))
    return p, (out.read_text() if p.returncode == 0 else None)


def test_set_keeps_other_flags_on_the_line():
    out = mry.apply_set(SCRIPT, "tensor-model-parallel-size", "6")
    f = flags(out)
    assert f["tensor_model_parallel_size"] == ["6"] and f["pipeline_model_parallel_size"] == ["5"]
    assert f["context_parallel_size"] == ["2"]
    out = mry.apply_set(SCRIPT, "context-parallel-size", "3")   # not first on its line: edited in place
    assert "--pipeline-model-parallel-size 5 --context-parallel-size 3 \\" in out


def test_multi_word_and_equals_values():
    out = mry.apply_set(SCRIPT, "recompute-modules", "mod_c")
    assert "--recompute-modules mod_c \\" in out and "mod_b" not in out
    out = mry.apply_set(SCRIPT, "micro-batch-size", "2")
    assert "--micro-batch-size 2 \\" in out


def test_unset_token_and_last_line():
    out = mry.apply_unset(SCRIPT, "pipeline-model-parallel-size")
    assert "--tensor-model-parallel-size 3 --context-parallel-size 2 \\" in out
    assert "# --pipeline-model-parallel-size 9" in out          # the comment is left alone
    out = mry.apply_unset(SCRIPT, "seed")                     # last line of the command
    assert "--num-epoch 1\n    echo done" in out
    out = mry.apply_unset(SCRIPT, "recompute-modules")
    assert "recompute-modules" not in out and "--micro-batch-size=1 \\" in out


def test_add_without_tee_anchor():
    out = mry.apply_add(SCRIPT, "num-rollout", "14")
    assert "--seed 1234 \\\n      --num-rollout 14\n    echo done" in out
    with pytest.raises(RuntimeError, match="already exists"):
        mry.apply_add(SCRIPT, "seed", "1")
    piped = SCRIPT.replace("--seed 1234\n", "--seed 1234 2>&1 | tee x.log\n")
    out = mry.apply_add(piped, "num-rollout", "14")
    assert "--seed 1234 --num-rollout 14 2>&1 | tee x.log" in out


def test_benchmark_on_exec_tee_script(tmp_path):
    p, out = make(tmp_path, SCRIPT, "alice-tune-r2", "--benchmark", "--with-profiler", "--profile-ranks", "all",
                  "--set", "tensor-model-parallel-size=6")
    assert p.returncode == 0, p.stderr
    f = flags(out)
    assert f["tensor_model_parallel_size"] == ["6"] and f["pipeline_model_parallel_size"] == ["5"]
    # 20 unprofiled steps, then profiler setup (step 20), warm-up (21) and one recorded step (22)
    assert f["num_rollout"] == ["23"] and f["global_batch_size"] == ["128"] and f["profile_step_start"] == ["22"]
    assert f["profile_step_end"] == ["23"]


def test_benchmark_profile_capture(tmp_path):
    prof = ("--benchmark", "--with-profiler", "--profile-ranks", "all")
    for extra, env, (n, start, end) in ((("--profile-capture", "3"), {}, (25, 22, 25)),
                                        ((), {"AUTOTUNE_PROFILE_CAPTURE": "2"}, (24, 22, 24)),
                                        (("--profile-capture", "1"), {"AUTOTUNE_PROFILE_CAPTURE": "3"}, (23, 22, 23))):
        p, out = make(tmp_path, SCRIPT, "alice-tune-r3", *prof, *extra, **env)
        assert p.returncode == 0, p.stderr
        f = flags(out)
        assert (f["num_rollout"], f["profile_step_start"], f["profile_step_end"]) == ([str(n)], [str(start)], [str(end)])
    p, out = make(tmp_path, SCRIPT, "alice-tune-r4", "--benchmark", AUTOTUNE_PROFILE_CAPTURE="7")
    assert p.returncode == 0, p.stderr          # a bad capture setting never blocks an unprofiled render
    f = flags(out)
    assert f["num_rollout"] == ["20"] and "profile_step_start" not in f
    p, _ = make(tmp_path, SCRIPT, "alice-tune-r5", *prof, AUTOTUNE_PROFILE_CAPTURE="7")
    assert p.returncode == 1 and "outside 1..3" in p.stderr
    p, _ = make(tmp_path, SCRIPT, "alice-tune-r5", *prof, "--profile-capture", "4")
    assert p.returncode == 2 and "invalid choice" in p.stderr
    p, _ = make(tmp_path, SCRIPT, "alice-tune-r5", "--benchmark", "--profile-capture", "2")
    assert p.returncode == 1 and "--profile-capture needs --with-profiler" in p.stderr
    p, _ = make(tmp_path, SCRIPT, "alice-tune-r5", "--benchmark", AUTOTUNE_NUM_STEPS="7")
    assert p.returncode == 1 and "below the minimum 8" in p.stderr      # window 5-6 holds fewer than 3 steps


def test_short_runs_and_their_confirm_runs(tmp_path):
    """A 10-step campaign ranks 10-step runs; its confirm runs hold the check's 20 steps; a profiled run starts after
    the measured run."""
    short = {"AUTOTUNE_NUM_STEPS": "10"}
    p, out = make(tmp_path, SCRIPT, "alice-tune-s1", "--benchmark", **short)
    assert p.returncode == 0 and flags(out)["num_rollout"] == ["10"], p.stderr
    assert "10 steps (unprofiled; window 5-9, target step 9)" in p.stdout
    p, conf = make(tmp_path, SCRIPT, "alice-tune-s1", "--benchmark", "--confirm", **short)
    assert p.returncode == 0 and flags(conf)["num_rollout"] == ["20"], p.stderr
    assert "20 steps (confirm; window 5-9, target step 19)" in p.stdout
    assert {k for k in set(flags(out)) | set(flags(conf)) if flags(out).get(k) != flags(conf).get(k)} == {"num_rollout"}
    p, prof = make(tmp_path, SCRIPT, "alice-tune-s2", "--benchmark", "--with-profiler", "--profile-ranks", "all",
                   **short)
    f = flags(prof)
    assert p.returncode == 0 and (f["num_rollout"], f["profile_step_start"], f["profile_step_end"]) == \
        (["13"], ["12"], ["13"]), p.stderr
    p, long_conf = make(tmp_path, SCRIPT, "alice-tune-s3", "--benchmark", "--confirm", AUTOTUNE_NUM_STEPS="30")
    assert p.returncode == 0 and flags(long_conf)["num_rollout"] == ["30"], p.stderr   # pairs with 30-step runs
    _, plain = make(tmp_path, SCRIPT, "alice-tune-s4", "--benchmark")
    _, default_conf = make(tmp_path, SCRIPT, "alice-tune-s4", "--benchmark", "--confirm")
    assert default_conf == plain                # at N >= 20 a confirm run is an unprofiled ranked render
    p, _ = make(tmp_path, SCRIPT, "alice-tune-s5", "--confirm", **short)
    assert p.returncode == 1 and "--confirm needs --benchmark" in p.stderr
    p, _ = make(tmp_path, SCRIPT, "alice-tune-s5", "--benchmark", "--confirm", "--with-profiler", **short)
    assert p.returncode == 1 and "a confirm run is unprofiled" in p.stderr


def test_profile_range_stays_after_the_measured_steps(tmp_path):
    for steps, n_roll, why in (("15-17", "30", "setup step 13 falls inside steps 0-19"),
                               ("21-22", "30", "setup step 19 falls inside steps 0-19"),
                               ("22-26", "24", "needs --num-rollout >= 26")):
        p, _ = make(tmp_path, SCRIPT, "alice-tune-d1", "--with-profiler", "--profile-ranks", "all",
                    "--profile-steps", steps, "--set", "num-rollout=" + n_roll)
        assert p.returncode == 1 and why in p.stderr, (steps, p.stderr)
    p, out = make(tmp_path, SCRIPT, "alice-tune-d1", "--with-profiler", "--profile-ranks", "all",
                  "--profile-steps", "40-42", "--set", "num-rollout=42")
    assert p.returncode == 0 and flags(out)["profile_step_start"] == ["40"], p.stderr
    p, out = make(tmp_path, SCRIPT, "alice-tune-d2", "--benchmark", "--with-profiler", "--profile-steps", "22-24")
    assert p.returncode == 0 and flags(out)["num_rollout"] == ["24"], p.stderr   # an explicit range fixes K = 2
    p, _ = make(tmp_path, SCRIPT, "alice-tune-d2", "--benchmark", "--with-profiler", "--profile-steps", "21-23")
    assert p.returncode == 1 and "require --profile-steps 22-24" in p.stderr
    p, _ = make(tmp_path, SCRIPT, "alice-tune-d2", "--benchmark", "--with-profiler", "--profile-steps", "22-26")
    assert p.returncode == 1 and "record 1-3 steps" in p.stderr


def test_capture_setting_is_read_only_for_the_default_range(tmp_path):
    p, out = make(tmp_path, SCRIPT, "alice-tune-c1", "--benchmark", "--with-profiler", "--profile-ranks", "all",
                  "--profile-steps", "22-23", AUTOTUNE_PROFILE_CAPTURE="9")
    assert p.returncode == 0 and flags(out)["profile_step_end"] == ["23"], p.stderr
    p, _ = make(tmp_path, SCRIPT, "alice-tune-c2", "--with-profiler", "--profile-steps", "40-41",
                "--set", "num-rollout=41", AUTOTUNE_PROFILE_CAPTURE="9")
    assert p.returncode == 0, p.stderr
    p, _ = make(tmp_path, SCRIPT, "alice-tune-c3", "--benchmark", "--with-profiler", AUTOTUNE_PROFILE_CAPTURE="9")
    assert p.returncode == 1 and "outside 1..3" in p.stderr                    # the default range reads it
    p, _ = make(tmp_path, SCRIPT, "alice-tune-c4", "--benchmark", "--with-profiler", "--profile-steps", "22-23",
                "--profile-capture", "2")
    assert p.returncode == 1 and "disagrees with --profile-steps 22-23" in p.stderr
    p, _ = make(tmp_path, SCRIPT, "alice-tune-c5", "--benchmark", "--profile-steps", "22-23")
    assert p.returncode == 1 and "--profile-steps needs --with-profiler" in p.stderr


def test_profiled_render_needs_a_checkable_num_rollout(tmp_path):
    prof = ("--with-profiler", "--profile-ranks", "all", "--profile-steps", "22-23")
    p, _ = make(tmp_path, SCRIPT, "alice-tune-n1", *prof)                    # --num-epoch 1, no --num-rollout
    assert p.returncode == 1 and "not one literal integer" in p.stderr and "--unchecked-num-rollout" in p.stderr
    p, out = make(tmp_path, SCRIPT, "alice-tune-n1", *prof, "--unchecked-num-rollout")
    assert p.returncode == 0 and flags(out)["profile_step_start"] == ["22"], p.stderr
    variable = SCRIPT.replace("--num-epoch 1", "--num-rollout $NUM_ROLLOUT")
    p, _ = make(tmp_path, variable, "alice-tune-n2", *prof)
    assert p.returncode == 1 and "not one literal integer" in p.stderr
    quoted = SCRIPT.replace("--num-epoch 1", '--num-rollout "23"')
    p, _ = make(tmp_path, quoted, "alice-tune-n3", *prof)
    assert p.returncode == 0, p.stderr                                         # a quoted literal is checked
    p, _ = make(tmp_path, quoted, "alice-tune-n3", "--with-profiler", "--profile-ranks", "all", "--profile-steps", "22-25")
    assert p.returncode == 1 and "needs --num-rollout >= 25 (the run has 23 steps)" in p.stderr
    p, _ = make(tmp_path, SCRIPT, "alice-tune-n4", "--benchmark", "--unchecked-num-rollout")
    assert p.returncode == 1 and "--unchecked-num-rollout needs --with-profiler" in p.stderr


PROFILE_FLAGS = {"num_rollout", "use_pytorch_profiler", "profile_target", "profile_step_start", "profile_step_end",
                 "tensorboard_dir"}


def test_profiled_and_unprofiled_renders_differ_only_in_profile_flags(tmp_path):
    (tmp_path / "u").mkdir()
    (tmp_path / "p").mkdir()
    pu, unprof = make(tmp_path / "u", SCRIPT, "alice-tune-r6", "--benchmark", "--set", "seed=7")
    pp, prof = make(tmp_path / "p", SCRIPT, "alice-tune-r6", "--benchmark", "--set", "seed=7", "--with-profiler",
                    "--profile-ranks", "all")
    assert pu.returncode == 0 and pp.returncode == 0, pu.stderr + pp.stderr
    fu, fp = flags(unprof), flags(prof)
    assert {k for k in set(fu) | set(fp) if fu.get(k) != fp.get(k)} <= PROFILE_FLAGS
    assert fu["num_rollout"] == ["20"] and fp["num_rollout"] == ["23"]

    def rest(text):   # every document with the training script reduced to its non-flag lines
        docs = [d for d in yaml.safe_load_all(text) if d]
        for d in docs:
            if d.get("kind") == "ConfigMap" and "run.sh" in (d.get("data") or {}):
                d["data"]["run.sh"] = [x for x in d["data"]["run.sh"].splitlines() if "--" not in x]
        return docs
    assert rest(unprof) == rest(prof)


def test_benchmark_render_drops_the_parity_probe(tmp_path):
    probed = SCRIPT.replace("              image: img@sha256:abc\n",
                            "              image: img@sha256:abc\n"
                            "              env: [{name: ATTUNE_HOOKS, value: \"parity_probe,profile_rank\"}]\n")
    probed = probed.replace("  entrypoint: bash /scripts/run.sh\n", "  entrypoint: bash /scripts/run.sh\n"
                            "  runtimeEnvYAML: |\n    env_vars:\n      ATTUNE_HOOKS: parity_probe\n")
    p, out = make(tmp_path, probed, "alice-tune-r7", "--benchmark")
    assert p.returncode == 0, p.stderr
    docs = [d for d in yaml.safe_load_all(out) if d]
    job = next(d for d in docs if d["kind"] == "RayJob")
    env = job["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]["env"]
    assert next(e["value"] for e in env if e["name"] == "ATTUNE_HOOKS") == "profile_rank"
    assert yaml.safe_load(job["spec"]["runtimeEnvYAML"])["env_vars"]["ATTUNE_HOOKS"] == ""
    exported = SCRIPT.replace("    echo done\n", "    echo done\n    export ATTUNE_HOOKS=parity_probe\n")
    p, _ = make(tmp_path, exported, "alice-tune-r8", "--benchmark")
    assert p.returncode == 1 and "exports ATTUNE_HOOKS with parity_probe" in p.stderr


def test_frozen_check_ignores_comments():
    spec = {"path": "x", "args": {"pipeline_model_parallel_size": {"eq": 5}}, "products": [], "env": {}}
    assert frozen_args.check_manifest(spec, SCRIPT) == []    # the comment says 9
    assert frozen_args.check_manifest(spec, mry.apply_set(SCRIPT, "pipeline-model-parallel-size", "9"))


def test_tensorboard_dir_kept_when_per_run(tmp_path):
    for tb, kept in (('"$CAMPAIGN_DIR/$RUN_ID/profiles"', True), ("/shared/fixed/profiles", False)):
        text = SCRIPT.replace("--seed 1234\n", "--seed 1234 \\\n      --tensorboard-dir %s\n" % tb)
        p, out = make(tmp_path, text, "alice-tune-r3", "--benchmark", "--with-profiler", "--profile-ranks", "all")
        assert p.returncode == 0, p.stderr
        got = flags(out)["tensorboard_dir"]
        assert got == ([tb.strip('"')] if kept else ["/shared/experiments/alice-tune-r3/profiles"]), got


EMBEDDED = SCRIPT.replace("    echo done\n", "    echo done\n  adapter.py: |\n    \"\"\"Dataset: else --hf-checkpoint (a note) and --seed 7.\"\"\"\n"
                          "    x = 1  # --micro-batch-size 9\n  extra.json: |\n    {\"--seed\": 3}\n")


def test_embedded_files_are_not_edited():
    out = mry.apply_set(EMBEDDED, "seed", "99")                   # not "multiple occurrences"
    assert "--seed 99\n" in out and "and --seed 7." in out and '{"--seed": 3}' in out
    out = mry.apply_set(EMBEDDED, "hf-checkpoint", "/x")
    assert '--hf-checkpoint /x \\' in out and "else --hf-checkpoint (a note)" in out
    out = mry.apply_unset(EMBEDDED, "seed")
    assert "and --seed 7." in out
    with pytest.raises(RuntimeError, match="already exists"):
        mry.apply_add(EMBEDDED, "seed", "1")


def test_owner_label_on_rayjob_and_configmaps(tmp_path):
    p, out = make(tmp_path, SCRIPT, "alice-tune-r4", "--benchmark", AUTOTUNE_OWNER_LABEL="example.com/owner")
    assert p.returncode == 0, p.stderr
    docs = [d for d in yaml.safe_load_all(out) if d]
    assert {d["kind"] for d in docs} == {"RayJob", "ConfigMap"}
    for d in docs:
        assert d["metadata"]["labels"]["example.com/owner"] == "alice", d["metadata"]
    assert "alice-tune-base" not in out and any(d["metadata"]["name"] == "alice-tune-r4-scripts" for d in docs)


@pytest.mark.parametrize("run_id, env, why", [
    ("bob-tune-r1", {}, "must start with 'alice-'"),
    ("alice-r1", {"ATTUNE_RUN_PREFIX": "alice-tune-"}, "must start with 'alice-tune-'"),
    ("alice-Tune-r1", {}, "must match"),
    ("alice-" + "x" * 42, {}, "at most 47"),
    ("alice-tune-r1;rm", {}, "must match"),
])
def test_bad_run_id_is_refused(tmp_path, run_id, env, why):
    p, out = make(tmp_path, SCRIPT, run_id, **env)
    assert p.returncode == 1 and out is None and why in p.stderr, p.stderr


def test_unowned_objects_in_the_base_are_refused(tmp_path):
    p, _ = make(tmp_path, SCRIPT.replace("name: alice-tune-base-scripts", "name: shared-scripts"), "alice-tune-r5")
    assert p.returncode == 1 and "must be named alice-tune-base" in p.stderr
    p, _ = make(tmp_path, SCRIPT + "---\napiVersion: v1\nkind: Secret\nmetadata: {name: alice-tune-base}\n", "alice-tune-r5")
    assert p.returncode == 1 and "only the RayJob and its ConfigMaps" in p.stderr


def test_dataset_uses_the_global_dataset(tmp_path):
    text = SCRIPT.replace("--seed 1234\n", "--seed 1234 --disable-rollout-global-dataset\n")
    p, out = make(tmp_path, text, "alice-tune-r6", "--benchmark", AUTOTUNE_DATASET="/shared/data/rows.parquet",
                  AUTOTUNE_SEQ_LEN="4096")
    assert p.returncode == 0, p.stderr
    f = flags(out)
    assert f["prompt_data"] == ["/shared/data/rows.parquet"] and "disable_rollout_global_dataset" not in f
    job = next(d for d in yaml.safe_load_all(out) if d and d["kind"] == "RayJob")
    pod_env = job["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]["env"]
    assert {"name": "AUTOTUNE_SEQ_LEN", "value": "4096"} in pod_env


def test_rollout_adapter_honours_seq_len(monkeypatch):
    sys.path.insert(0, str(SCRIPTS / "data"))
    import types

    import pretokenized_sft_rollout as adapter

    def buffer(n):
        rows = [types.SimpleNamespace(metadata={"tokens": list(range(n)), "loss_mask": [1] * (n - 1)}) for _ in range(2)]
        return types.SimpleNamespace(get_samples=lambda k: [[r] for r in rows[:k]])

    args = types.SimpleNamespace(rollout_batch_size=2, rollout_global_dataset=True)
    monkeypatch.setenv("AUTOTUNE_SEQ_LEN", "16")
    out = adapter.generate_rollout(args, 0, buffer(16))
    assert [len(s.tokens) for (s,) in out] == [16, 16] and out[0][0].response_length == 15
    with pytest.raises(ValueError, match="Expected 16 tokens"):
        adapter.generate_rollout(args, 0, buffer(8192))
    monkeypatch.delenv("AUTOTUNE_SEQ_LEN")
    assert len(adapter.generate_rollout(args, 0, buffer(8192))[0][0].tokens) == 8192
    with pytest.raises(ValueError, match="disable-rollout-global-dataset"):
        adapter.generate_rollout(types.SimpleNamespace(rollout_batch_size=2, rollout_global_dataset=False), 0, buffer(16))


def test_flags_go_on_the_training_command_not_a_preflight(tmp_path):
    before = commands(TWO)
    assert set(before) == {"/scripts/all_nodes.py", "/scripts/check_nodes.py", "train.py"}
    out = mry.apply_add(TWO, "overlap-grad-reduce", "")
    out = mry.apply_set(out, "num-rollout", "14")                   # absent: appended to the training command
    out = mry.apply_set(out, "tensor-model-parallel-size", "6")    # present: edited in place
    out = mry.apply_set(out, "seed", "5")                          # the helper's own --seed is not touched
    out = mry.apply_unset(out, "hf-checkpoint")
    after = commands(out)
    assert after["/scripts/all_nodes.py"] == before["/scripts/all_nodes.py"]
    assert after["/scripts/check_nodes.py"] == before["/scripts/check_nodes.py"]
    train = after["train.py"]
    assert "--overlap-grad-reduce" in train and "--num-rollout 14" in train and "--seed 5" in train
    assert "--tensor-model-parallel-size 6" in train and "--hf-checkpoint" not in train
    assert train.rstrip().endswith('2>&1 | tee "$LOG_DIR/train.log"')
    p, rendered = make(tmp_path, TWO, "alice-tune-prof", "--benchmark", "--with-profiler", "--profile-ranks", "all",
                       "--add", "overlap-param-gather=")
    assert p.returncode == 0, p.stderr
    after = commands(rendered)
    assert after["/scripts/all_nodes.py"] == before["/scripts/all_nodes.py"]
    assert after["/scripts/check_nodes.py"] == before["/scripts/check_nodes.py"]
    for f in mry.PROFILER_FLAGS + ("tensorboard-dir", "overlap-param-gather", "num-rollout", "global-batch-size"):
        assert "--" + f in after["train.py"], f


@pytest.mark.parametrize("edit, error", [
    (lambda t: t.replace("python3 /scripts/check_nodes.py --seed 7", "python3 train.py --seed 7"), "found 2"),
    (lambda t: t.replace("python3 train.py", "python3 run.py").replace("--micro-batch-size=1", "--foo=1")
     .replace("--tensor-model-parallel-size 3 --pipeline-model-parallel-size 5 ", ""), "found 0"),
])
def test_no_single_training_command_is_an_error(edit, error):
    with pytest.raises(RuntimeError, match="cannot identify the single training command.*" + error):
        mry.apply_add(edit(TWO), "overlap-grad-reduce", "")


def test_a_training_command_in_a_quoted_yaml_string_is_refused():
    quoted = 'data:\n  run.sh: "python3 train.py --seed 1 2>&1 | tee x.log\\necho done"\n'
    with pytest.raises(RuntimeError, match="inside a quoted YAML string"):
        mry.apply_add(quoted, "overlap-grad-reduce", "")
