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
