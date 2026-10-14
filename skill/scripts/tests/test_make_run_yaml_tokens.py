# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""make_run_yaml.py: token-level flag edits (other flags on a line survive, comments are ignored, --add works without a
`2>&1 | tee` anchor), run ownership and the fixed-shape dataset contract. Run: python3 -m pytest skill/scripts/tests"""
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
    assert f["num_rollout"] == ["17"] and f["global_batch_size"] == ["128"] and f["profile_step_start"] == ["15"]


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
