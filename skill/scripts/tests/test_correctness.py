# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layer-1 classification, the parity verdict end to end, and frozen-arguments loading.
Run: python3 -m pytest skill/scripts/tests"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CORRECTNESS = HERE.parent / "correctness"
EXAMPLE = CORRECTNESS / "rules" / "layer1" / "example"
sys.path.insert(0, str(CORRECTNESS))
sys.path.insert(0, str(HERE))
import fixtures  # noqa: E402
import frozen_args  # noqa: E402
import layer1  # noqa: E402

BASE = {"lr": "1e-05", "tensor_model_parallel_size": "2", "moe_grouped_gemm": "False", "seed": "1234"}


def _spec(tmp_path, ct="CT-grouped-gemm: pass"):
    """A setup directory like layer1/example/: frozen arguments that freeze nothing, its allowlist and test results."""
    shutil.copy(EXAMPLE / "allowlist.yaml", tmp_path / "allowlist.yaml")
    (tmp_path / "ct_status.yaml").write_text("tests:\n  %s\n" % ct)
    (tmp_path / "frozen_args.yaml").write_text("# freezes nothing\nargs: {}\n")
    return frozen_args.load(str(tmp_path / "frozen_args.yaml"))


def test_unlisted_name_is_numeric_and_fails():
    r = layer1.verdict(dict(BASE, brand_new_arg="1"), dict(BASE, brand_new_arg="2"))
    assert not r["L1"] and r["changes"]["numeric"] == {"brand_new_arg": ("1", "2")}


def test_numeric_change_fails_and_placement_passes():
    assert not layer1.verdict(BASE, dict(BASE, lr="2e-05"))["L1"]
    r = layer1.verdict(BASE, dict(BASE, tensor_model_parallel_size="4"))
    assert r["L1"] and "tensor_model_parallel_size" in r["changes"]["placement"]


def test_allowlisted_swap_needs_the_setup_allowlist_and_a_passed_test(tmp_path):
    cand = dict(BASE, moe_grouped_gemm="True")
    assert not layer1.verdict(BASE, cand)["L1"]                       # no setup directory: no swap admitted
    r = layer1.verdict(BASE, cand, _spec(tmp_path))
    assert r["L1"] and r["changes"]["allowlisted"] == {"moe_grouped_gemm": ("False", "True")}
    r = layer1.verdict(BASE, cand, _spec(tmp_path, ct="CT-grouped-gemm: fail"))
    assert not r["L1"] and "moe_grouped_gemm" in r["changes"]["allowlist_violation"]


PREFIX = "abcdef123456"
IMAGE = "reg/img@sha256:" + PREFIX + "0" * 52


def _image_rules(tmp_path, monkeypatch, ct="CT-grouped-gemm: pass", allowlist=None):
    """A rules dir whose images/<PREFIX>/ admits the example swaps (or `allowlist`) on IMAGE."""
    rules = tmp_path / "rules"
    idir = rules / "images" / PREFIX
    idir.mkdir(parents=True)
    shutil.copy(CORRECTNESS / "rules" / "numerics_list.yaml", rules / "numerics_list.yaml")
    if allowlist is None:
        shutil.copy(EXAMPLE / "allowlist.yaml", idir / "allowlist.yaml")
    else:
        (idir / "allowlist.yaml").write_text(allowlist)
    (idir / "ct_status.yaml").write_text("tests:\n  %s\n" % ct)
    monkeypatch.setattr(frozen_args, "RULES_DIR", str(rules))
    return rules


def test_image_rules_admit_a_tested_swap_only_for_that_pinned_image(tmp_path, monkeypatch):
    _image_rules(tmp_path, monkeypatch)
    cand = dict(BASE, moe_grouped_gemm="True")
    monkeypatch.setenv("TRAINING_IMAGE", IMAGE)
    assert not layer1.verdict(BASE, cand)["L1"]                                    # the environment is not read
    assert not layer1.verdict(BASE, cand, image="reg/img:latest")["L1"]            # not pinned by digest
    assert not layer1.verdict(BASE, cand, image="reg/img@sha256:" + "1" * 64)["L1"]  # another image
    assert not layer1.verdict(BASE, cand, image=IMAGE + "ff")["L1"]                # not a digest
    r = layer1.verdict(BASE, cand, image=IMAGE)                                    # no setup files needed
    assert r["L1"] and r["changes"]["allowlisted"] and r["image_dir"].endswith(PREFIX)


def test_setup_entry_is_judged_by_setup_tests_alone(tmp_path, monkeypatch):
    _image_rules(tmp_path, monkeypatch, allowlist="swaps:\n- {arg: moe_grouped_gemm, values: ['false'], "
                 "component_test: CT-grouped-gemm}\n")
    cand, setup = dict(BASE, moe_grouped_gemm="True"), tmp_path / "setup"
    setup.mkdir()
    (setup / "frozen_args.yaml").write_text("args: {}\n")
    (setup / "allowlist.yaml").write_text("swaps:\n- {arg: moe_grouped_gemm, values: any, component_test: CT-grouped-gemm}\n")
    (setup / "ct_status.yaml").write_text("tests: {}\n")
    spec = frozen_args.load(str(setup / "frozen_args.yaml"))
    assert not layer1.verdict(BASE, cand, spec, image=IMAGE)["L1"]   # the image's pass does not lend to the setup


def test_setup_failure_overrides_the_image_pass(tmp_path, monkeypatch):
    _image_rules(tmp_path, monkeypatch)
    setup = tmp_path / "setup"
    setup.mkdir()
    (setup / "frozen_args.yaml").write_text("args: {}\n")
    (setup / "ct_status.yaml").write_text("tests:\n  CT-grouped-gemm: fail\n")
    r = layer1.verdict(BASE, dict(BASE, moe_grouped_gemm="True"), frozen_args.load(str(setup / "frozen_args.yaml")),
                       image=IMAGE)
    assert not r["L1"] and "moe_grouped_gemm" in r["changes"]["allowlist_violation"]


def test_two_matching_image_directories_fail_layer1(tmp_path, monkeypatch):
    rules = _image_rules(tmp_path, monkeypatch)
    shutil.copytree(rules / "images" / PREFIX, rules / "images" / (PREFIX + "00"))
    r = layer1.verdict(BASE, dict(BASE, moe_grouped_gemm="True"), image=IMAGE)
    assert not r["L1"] and "more than one image directory" in r["errors"][0]


def test_pair_image_needs_one_pinned_image_for_both_arms(tmp_path):
    import parity_verdict
    assert parity_verdict.pair_image(str(tmp_path)) is None
    for refs, want in (({"base": IMAGE, "win": IMAGE}, IMAGE), ({"base": IMAGE, "win": "reg/img@sha256:" + "1" * 64}, None),
                       ({"base": "reg/img:latest", "win": "reg/img:latest"}, None), ([IMAGE], None)):
        (tmp_path / "images.json").write_text(json.dumps(refs))
        assert parity_verdict.pair_image(str(tmp_path)) == want


def test_make_parity_arm_records_the_arm_image(tmp_path):
    import make_parity_arm

    def pod(img):
        return {"containers": [{"name": "c", "image": img}], "initContainers": [{"name": "i", "image": "busybox"}]}
    docs = [{"kind": "RayJob", "spec": {"head": {"template": {"spec": pod(IMAGE)}},
                                        "workers": [{"template": {"spec": pod(IMAGE)}}]}}]
    assert make_parity_arm.arm_image(docs) == IMAGE                     # init containers aside
    docs[0]["spec"]["workers"][0]["template"]["spec"] = pod("reg/other@sha256:" + "1" * 64)
    assert make_parity_arm.arm_image(docs) is None                       # two images: none recorded
    make_parity_arm.record_image(tmp_path, "base", IMAGE)
    make_parity_arm.record_image(tmp_path, "win", IMAGE)
    import parity_verdict
    assert parity_verdict.pair_image(str(tmp_path)) == IMAGE


def test_empty_or_mismatched_dumps_fail():
    for base, cand in (({}, BASE), (BASE, {})):
        r = layer1.verdict(base, cand)
        assert not r["L1"] and "no resolved-argument dump" in r["errors"][0]
    other = {"%s_%d" % (k, i): v for i in range(3) for k, v in BASE.items()}
    r = layer1.verdict(dict(BASE, **other), BASE)
    assert not r["L1"] and "share" in r["errors"][0]


def test_frozen_args_reads_yaml_and_json(tmp_path):
    spec = frozen_args.load(str(EXAMPLE / "frozen_args.yaml"))         # JSON syntax
    assert spec["args"]["seed"] == {"eq": 1234} and spec["couplings"][0]["kind"] == "vpp_middle_stages"
    y = tmp_path / "frozen.yaml"
    y.write_text("# a comment\nargs:\n  seed: 1234\n  lr: {unchanged: true}\nenv:\n  NVTE_FUSED_ATTN: {absent: true}\n")
    spec = frozen_args.load(str(y))
    assert spec["args"] == {"seed": {"eq": 1234}, "lr": {"unchanged": True}}
    assert frozen_args.check_resolved(spec, {"seed": "1234", "lr": "2e-05"}, baseline={"lr": "1e-05"})[0]["arg"] == "lr"
    y.write_text("args: [1, 2]\n")
    with pytest.raises(frozen_args.SpecError):
        frozen_args.load(str(y))


def _write_pair(root, name, base_args=None, win_args=None, win_loss0=2.5):
    d = root / name
    d.mkdir(parents=True)
    (d / "base.log").write_text(fixtures.train_log(args=base_args))
    (d / "win.log").write_text(fixtures.train_log(args=win_args, loss0=win_loss0))
    (d / "gpus.json").write_text(json.dumps({"base": 16, "win": 16}))
    return d


def _verdict(path, tmp_path):
    pytest.importorskip("numpy")
    env = {k: v for k, v in os.environ.items() if k != "AUTOTUNE_FROZEN_ARGS"}
    out = tmp_path / "verdict.json"
    p = subprocess.run([sys.executable, str(CORRECTNESS / "parity_verdict.py"), str(path), "--json", str(out)],
                       capture_output=True, text=True, env=env)
    return p, (json.loads(out.read_text()) if out.exists() else None)


def test_verdict_pass_pair(tmp_path):
    d = _write_pair(tmp_path, "pass", win_args={"tensor_model_parallel_size": "4"})
    p, res = _verdict(d, tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert res[0]["PASS"] and res[0]["L1"] and res[0]["L2"] and res[0]["counts_identical"]
    assert p.stdout.startswith("pass: PASS")


def test_verdict_fail_pairs_exit_nonzero(tmp_path):
    _write_pair(tmp_path, "a-numeric", win_args={"moe_aux_loss_coeff": "0.05"})
    _write_pair(tmp_path, "b-loss", win_loss0=4.0)
    _write_pair(tmp_path, "c-pass")
    p, res = _verdict(tmp_path, tmp_path)
    assert p.returncode == 1, p.stdout + p.stderr
    by = {r["pair"]: r for r in res}
    assert not by["a-numeric"]["L1"] and by["a-numeric"]["L2"] and "moe_aux_loss_coeff" in by["a-numeric"]["L1_numeric"]
    assert by["b-loss"]["L1"] and not by["b-loss"]["L2"] and by["b-loss"]["convergence"]["failed_metrics"]
    assert by["c-pass"]["PASS"]
    assert "a-numeric: FAIL" in p.stdout and "L1 numeric changes: moe_aux_loss_coeff 0.02 -> 0.05" in p.stdout


def test_verdict_short_log_and_no_pairs(tmp_path):
    d = tmp_path / "short"
    d.mkdir()
    (d / "base.log").write_text(fixtures.train_log(steps=12))
    (d / "win.log").write_text(fixtures.train_log(steps=12))
    p, res = _verdict(d, tmp_path)
    assert p.returncode == 1 and not res[0]["L2"] and "12 of the 20 steps" in res[0]["L2_error"]
    empty = tmp_path / "empty"
    empty.mkdir()
    p, res = _verdict(empty, tmp_path / "x")
    assert p.returncode == 2 and res is None and "no pair directory" in p.stderr


def _pair_from_runs(tmp_path, base, win, *extra, base_yaml=fixtures.BASE_YAML, win_yaml=fixtures.BASE_YAML):
    """pair_from_runs.py on two run logs and their YAMLs (None = leave the file out)."""
    paths = {}
    for name, text in (("base_run.log", base), ("win_run.log", win), ("base_run.yaml", base_yaml),
                       ("win_run.yaml", win_yaml)):
        paths[name] = tmp_path / name
        if text is not None:
            paths[name].write_text(text)
    return subprocess.run([sys.executable, str(CORRECTNESS / "pair_from_runs.py"),
                           "--base-log", str(paths["base_run.log"]), "--base-yaml", str(paths["base_run.yaml"]),
                           "--win-log", str(paths["win_run.log"]), "--win-yaml", str(paths["win_run.yaml"]),
                           "--pair", str(tmp_path / "pair"), *extra], capture_output=True, text=True)


RUN = {"num_rollout": "20", "use_pytorch_profiler": "False"}


def test_pair_from_benchmark_runs_scores_without_new_arms(tmp_path):
    (tmp_path / "pair").mkdir()
    (tmp_path / "pair" / "gpus.json").write_text('{"base": 1, "win": 1}\n')   # stale, from an earlier pair
    p = _pair_from_runs(tmp_path, fixtures.train_log(args=RUN, counts=False),
                        fixtures.train_log(args=dict(RUN, tensor_model_parallel_size="4"), counts=False))
    assert p.returncode == 0, p.stderr
    pair = tmp_path / "pair"
    assert (pair / "base.log").exists() and (pair / "win.log").exists()
    assert json.loads((pair / "gpus.json").read_text()) == {"base": 24, "win": 24}
    assert json.loads((pair / "images.json").read_text()) == {"base": "registry.example.com/img@sha256:abc",
                                                               "win": "registry.example.com/img@sha256:abc"}
    q, res = _verdict(pair, tmp_path)
    assert q.returncode == 0 and res[0]["PASS"] and res[0]["count_steps"] == 0, q.stdout
    assert "probe counts off" in q.stdout
    assert res[0]["base_tok_s_gpu"] == 1000.0 * 9.0 / 10.0 / 24   # logged tokens / step time / GPUs over 5-19


def test_pair_from_runs_refuses_profiled_short_or_mismatched_runs(tmp_path):
    def refused(*a, **kw):
        d = tmp_path / ("case%d" % len(list(tmp_path.iterdir())))
        d.mkdir()
        p = _pair_from_runs(d, *a, **kw)
        assert p.returncode == 1 and not (d / "pair").exists(), p.stderr
        return p.stderr
    ok = fixtures.train_log(args=RUN)
    assert "win: the run is profiled" in refused(ok, fixtures.train_log(args=dict(RUN, use_pytorch_profiler="True")))
    assert "base: missing the training loss of steps [14" in refused(fixtures.train_log(args=RUN, steps=14), ok)
    assert "--num-rollout differs (20 vs 22)" in refused(ok, fixtures.train_log(args=dict(RUN, num_rollout="22")))
    assert "base: no log at" in refused(None, ok)
    assert "win: no run YAML at" in refused(ok, ok, win_yaml=None)
    no_rollout = fixtures.train_log(args={"use_pytorch_profiler": "False"})
    assert "win: the argument dump has no numeric num_rollout" in refused(ok, no_rollout)
    other = fixtures.BASE_YAML.replace("img@sha256:abc", "img@sha256:def")
    assert "the runs used different images" in refused(ok, ok, win_yaml=other)
    (tmp_path / "allowed").mkdir()
    p = _pair_from_runs(tmp_path / "allowed", ok, ok, "--allow-image-change", win_yaml=other)
    assert p.returncode == 0, p.stderr
    assert json.loads((tmp_path / "allowed" / "pair" / "images.json").read_text())["win"].endswith("sha256:def")


def _yaml_with(edit):
    """fixtures.BASE_YAML after edit(rayjob) on its parsed RayJob."""
    import yaml
    docs = [d for d in yaml.safe_load_all(fixtures.BASE_YAML) if d]
    edit(next(d for d in docs if d["kind"] == "RayJob"))
    return yaml.safe_dump_all(docs)


def _groups(job):
    rcs = job["spec"]["rayClusterSpec"]
    return [rcs["headGroupSpec"]] + rcs["workerGroupSpecs"]


def test_pair_from_runs_reads_the_ray_containers_image(tmp_path):
    ok, img = fixtures.train_log(args=RUN), "registry.example.com/img@sha256:abc"

    def sidecars(job):   # a log shipper beside Ray, and a submitter pod on another image
        _groups(job)[0]["template"]["spec"]["containers"].append({"name": "logs", "image": "fluent-bit:3"})
        job["spec"]["submitterPodTemplate"] = {"spec": {"containers": [{"name": "submit", "image": "ray-submit:1"}]}}

    def mixed(job):      # the worker group's Ray container runs another image
        _groups(job)[1]["template"]["spec"]["containers"][0]["image"] = "registry.example.com/img@sha256:def"
    cases = {}
    for name, extra, base_yaml in (("sidecar", (), _yaml_with(sidecars)), ("mixed", (), _yaml_with(mixed)),
                                   ("named", ("--image", img), _yaml_with(mixed)),
                                   ("wrong", ("--image", "registry.example.com/x:1"), _yaml_with(mixed)),
                                   ("unknown", ("--allow-image-change",), _yaml_with(mixed))):
        d = tmp_path / name
        d.mkdir()
        p = _pair_from_runs(d, ok, ok, *extra, base_yaml=base_yaml)
        images = json.loads((d / "pair" / "images.json").read_text()) if p.returncode == 0 else None
        cases[name] = (p.returncode, images, p.stderr)
    assert cases["sidecar"][:2] == (0, {"base": img, "win": img}), cases["sidecar"]
    assert cases["mixed"][0] == 1 and "runs several training images" in cases["mixed"][2]
    assert cases["named"][:2] == (0, {"base": img, "win": img}), cases["named"]
    assert cases["wrong"][0] == 1 and "--image" in cases["wrong"][2]
    assert cases["unknown"][:2] == (0, {"base": None, "win": img}) and "recorded as unknown" in cases["unknown"][2]


def test_couplings_fail_closed():
    import couplings
    require = {"id": "tp_overlap", "when": "tp_comm_overlap", "require": {"sequence_parallel": True}}
    get = {"tp_comm_overlap": "True", "sequence_parallel": None}.get
    assert couplings.violations({"couplings": [require]}, get) == ["tp_overlap: --sequence-parallel is required"]
    assert couplings.violations({"couplings": [require]}, {"tp_comm_overlap": "False"}.get) == []
    vpp = {"id": "uneven", "kind": "vpp_middle_stages"}
    get = {"num_virtual_stages_per_pipeline_rank": "2", "pipeline_model_parallel_size": "2",
           "decoder_first_pipeline_num_layers": "5", "decoder_last_pipeline_num_layers": "5"}.get
    assert couplings.violations({"couplings": [vpp]}, get)[0].startswith("uneven: virtual pipeline stages need")
    spec = {"path": "x", "args": {}, "products": [], "env": {}, "couplings": [{"id": "x", "kind": "nope"}],
            "raw": {"couplings": [{"id": "x", "kind": "nope"}]}}
    v = frozen_args.check_resolved(spec, {"seed": "1"})
    assert v and v[0]["arg"] == "couplings" and "unknown kind" in v[0]["why"]
