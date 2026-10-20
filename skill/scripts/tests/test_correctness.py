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
import yaml

HERE = Path(__file__).resolve().parent
CORRECTNESS = HERE.parent / "correctness"
EXAMPLE = CORRECTNESS / "rules" / "layer1" / "example"
sys.path.insert(0, str(CORRECTNESS))
sys.path.insert(0, str(HERE))
import fixtures  # noqa: E402
import frozen_args  # noqa: E402
import layer1  # noqa: E402

BASE = {"lr": "1e-05", "tensor_model_parallel_size": "2", "moe_grouped_gemm": "False", "seed": "1234"}
OBJECTIVE = """prompt_data input_key metadata_key seed n_samples_per_prompt apply_chat_template rollout_shuffle
global_batch_size rollout_batch_size num_rollout seq_length lr min_lr lr_decay_style lr_decay_iters lr_warmup_iters
lr_warmup_fraction weight_decay start_weight_decay end_weight_decay weight_decay_incr_style adam_beta1 adam_beta2 adam_eps
clip_grad optimizer loss_type calculate_per_token_loss moe_aux_loss_coeff moe_z_loss_coeff moe_router_load_balancing_type
attention_dropout hidden_dropout bf16 fp16 fp8_recipe accumulate_allreduce_grads_in_fp32 attention_softmax_in_fp32
main_grads_dtype main_params_dtype exp_avg_dtype exp_avg_sq_dtype grad_reduce_in_bf16 params_dtype
fp8 fp4 moe_expert_capacity_factor moe_token_drop_policy moe_pad_expert_input_to_capacity moe_router_topk
moe_router_score_function moe_router_topk_scaling_factor moe_router_num_groups moe_router_group_topk
fp4_recipe fp4_param fp8_param fp8_wgrad fp8_margin fp8_amax_history_len
fp8_amax_compute_algo fp8_dot_product_attention fp8_multi_head_attention fp8_output_proj
first_last_layers_bf16 num_layers_at_start_in_bf16 num_layers_at_end_in_bf16 activation_func_fp8_input_store
fp16_lm_cross_entropy fp32_residual_connection disable_bf16_reduced_precision_matmul enable_autocast autocast_dtype
moe_router_dtype moe_activation_in_fp32 moe_combine_in_fp32 stream_optimizer_state_moment_dtype
mamba_training_ssm_states_dtype quant_recipe use_kitchen
moe_router_pre_softmax moe_router_topk_limited_devices moe_router_force_load_balancing moe_router_enable_expert_bias
moe_router_bias_update_rate moe_input_jitter_eps moe_apply_probs_on_input moe_token_dropping
moe_expert_rank_capacity_factor
loss_mask_type eod_mask_loss mtp_loss_scaling_factor dsa_indexer_loss_coeff logits_load_kd_loss_alpha entropy_coef
kl_coef kl_loss_coef kl_loss_type use_kl_loss use_unbiased_kl eps_clip eps_clip_high eps_clip_c value_clip
advantage_estimator gamma lambd normalize_advantages rewards_normalization use_tis tis_clip tis_clip_low
grpo_clamp_eps_lower grpo_clamp_eps_upper grpo_entropy_term_weight grpo_kl_beta grpo_std_normalization opd_kl_coef
data_path train_data_path split label_key apply_chat_template_kwargs chat_template chat_template_path
sft_tokenizer_prompt_format dataloader_type rollout_seed num_epoch train_iters train_samples rampup_batch_size
step_batch_size_schedule use_dynamic_global_batch_size num_steps_per_rollout grpo_group_size grpo_prompts_per_step
grpo_samples_per_iteration
decoupled_lr decoupled_min_lr head_lr_mult lr_warmup_init lr_warmup_samples lr_decay_samples lr_warmup_steps
lr_decay_steps lr_wsd_decay_iters lr_wsd_decay_samples lr_wsd_decay_style wsd_decay_steps wd_incr_steps
no_weight_decay_cond_type apply_wd_to_qk_layernorm sgd_momentum lion_beta1 lion_beta2 muon_momentum muon_use_nesterov
muon_nesterov muon_num_ns_steps muon_scale_mode muon_extra_scale_factor muon_coefficient_type muon_scalar_optimizer
muon_split_qkv muon_fp32_matmul_prec qk_clip qk_clip_alpha qk_clip_threshold critic_lr critic_lr_warmup_iters
freeze_all_layers freeze_params_name_list only_train_params_name_list freeze_e_score_correction_bias
moe_router_freeze_gate freeze_indexer debug_disable_optimizer lora_rank lora_alpha lora_dropout lora_type""".split()


def _spec(tmp_path, ct="CT-grouped-gemm: pass"):
    """A setup directory like layer1/example/: frozen arguments that freeze nothing, its allowlist and test results."""
    shutil.copy(EXAMPLE / "allowlist.yaml", tmp_path / "allowlist.yaml")
    (tmp_path / "ct_status.yaml").write_text("tests:\n  %s\n" % ct)
    (tmp_path / "frozen_args.yaml").write_text("# freezes nothing\nargs: {}\n")
    return frozen_args.load(str(tmp_path / "frozen_args.yaml"))


def test_numeric_and_unlisted_changes_pass_layer1_as_unknown():
    extra = {"brand_new_arg": "1", "use_precision_aware_optimizer": "False"}
    r = layer1.verdict(dict(BASE, **extra), dict(BASE, brand_new_arg="2", use_precision_aware_optimizer="True"))
    assert r["L1"] and r["changes"]["unknown"] == {"brand_new_arg": ("1", "2"),
                                                   "use_precision_aware_optimizer": ("False", "True")}
    assert r["unknown"] == [{"name": "brand_new_arg", "from": "1", "to": "2", "reason": "unlisted"},
                            {"name": "use_precision_aware_optimizer", "from": "False", "to": "True",
                             "reason": "numeric class"}]


def test_objective_class_holds_the_objective_names():
    nl = yaml.safe_load((CORRECTNESS / "rules" / "numerics_list.yaml").read_text())
    assert len(OBJECTIVE) == len(set(OBJECTIVE)) == 182 and sorted(nl["objective"]) == sorted(OBJECTIVE)
    for n in ("fp8_param_gather", "fp4_param_gather", "tp_only_amax_red", "fp8_interval",
              "megatron_fsdp_main_grads_dtype", "megatron_fsdp_main_params_dtype", "megatron_fsdp_grad_comm_dtype"):
        assert n in nl["numeric"], n


def test_an_unknown_numerics_list_key_fails_layer1(tmp_path, monkeypatch):
    (tmp_path / "numerics_list.yaml").write_text("objective: [lr]\nobjectives: [seed]\n")
    monkeypatch.setattr(frozen_args, "RULES_DIR", str(tmp_path))
    r = layer1.verdict(BASE, BASE)
    assert not r["L1"] and "unknown top-level key objectives" in r["errors"][0]


def test_a_name_in_two_classes_fails_layer1(tmp_path, monkeypatch):
    (tmp_path / "numerics_list.yaml").write_text("objective: [lr]\nnumeric: [lr]\n")
    monkeypatch.setattr(frozen_args, "RULES_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="lists lr in both objective and numeric"):
        layer1.load_classes()
    r = layer1.verdict(BASE, BASE)
    assert not r["L1"] and "lists lr in both" in r["errors"][0]


def test_a_name_only_one_dump_has_is_unknown():
    filler = {"log_x%d" % i: "1" for i in range(20)}
    base, cand = dict(BASE, **filler), dict(BASE, stream_optimizer_state_moment_dtype="float32", **filler)
    r = layer1.verdict(base, cand)                     # an objective name a newer build adds
    assert r["L1"] and r["unknown"] == [{"name": "stream_optimizer_state_moment_dtype", "from": None, "to": "float32",
                                         "reason": "objective-class name present in one dump only (the candidate's): "
                                                   "check by hand"}]
    r = layer1.verdict(dict(base, brand_new_arg="1"), cand)
    assert r["L1"] and r["changes"]["unknown_why"]["brand_new_arg"] == "only in the baseline's dump"


def test_numerics_list_classes_each_name_once_and_counts_match():
    import re

    text = (CORRECTNESS / "rules" / "numerics_list.yaml").read_text()
    nl = yaml.safe_load(text)
    names = [n for k in layer1.CLASSES for n in nl.get(k) or []]
    assert len(names) == len(set(names))
    for k in layer1.CLASSES:
        assert int(re.search(r"^%s:  # (\d+)$" % k, text, re.M).group(1)) == len(nl.get(k) or []), k


@pytest.mark.parametrize("name, cls", [("moe_deterministic_combine", "numeric"), ("moe_router_use_torch_mm", "numeric"),
                                       ("optimizer_state_offload_fraction", "allowlisted"), ("lr", "objective"),
                                       ("moe_router_fusion", "numeric"), ("brand_new_arg", "unlisted"),
                                       ("num_layers_in_first_pipeline_stage", "placement"),
                                       ("activation_offload_fraction", "placement"),
                                       ("tp_comm_overlap_disable_qkv", "allowlisted"),
                                       ("flight_recorder_dump_path", "ignored")])
def test_miles_bfdcff0e_arguments_are_classed(name, cls):
    """miles bfdcff0e and its Megatron-LM: new names classed, uncertain ones numeric."""
    assert layer1.classify(name, layer1.load_classes(CORRECTNESS / "rules")) == cls


VOCAB = {"make_vocab_size_divisible_by": "128", "tensor_model_parallel_size": "4"}


def test_an_unprinted_derived_value_is_unknown_when_tp_changes():
    base = dict(BASE, **VOCAB, padded_vocab_size="None")       # Llama-3.1-405B's slime stack prints None
    r = layer1.verdict(base, dict(base, tensor_model_parallel_size="8"))
    assert r["L1"] and "tensor_model_parallel_size" in r["changes"]["placement"]
    assert not r["changes"]["derived_violation"]
    assert r["unknown"] == [{"name": "padded_vocab_size", "from": "None", "to": "None",
                             "reason": "derived value not printed; cannot recompute"}]
    r = layer1.verdict(dict(base, padded_vocab_size="128256"), dict(base, tensor_model_parallel_size="8"))
    assert r["L1"] and r["changes"]["unknown_why"]["padded_vocab_size"] == "derived value not printed; cannot recompute"
    no_m = {k: v for k, v in dict(base, padded_vocab_size="128512").items() if k != "make_vocab_size_divisible_by"}
    r = layer1.verdict(no_m, dict(no_m, tensor_model_parallel_size="8"))
    assert r["L1"] and "cannot recompute" in r["changes"]["unknown_why"]["padded_vocab_size"]


def test_a_printed_derived_value_must_be_what_the_framework_derives():
    base = dict(BASE, **VOCAB, padded_vocab_size="128512")      # 128256 padded to a multiple of 128 x 4
    r = layer1.verdict(base, dict(base, tensor_model_parallel_size="8", padded_vocab_size="129024"))   # x 8
    assert r["L1"] and r["changes"]["derived"] == {"padded_vocab_size": ("128512", "129024")}
    r = layer1.verdict(base, dict(base, tensor_model_parallel_size="8"))         # unmoved, but 128512 % 1024 != 0
    assert not r["L1"] and "padded_vocab_size" in r["changes"]["derived_violation"]
    r = layer1.verdict(base, dict(base, padded_vocab_size="131072"))            # an arbitrary override
    assert not r["L1"] and "padded_vocab_size" in r["changes"]["derived_violation"]


def test_objective_change_fails_and_placement_passes():
    r = layer1.verdict(BASE, dict(BASE, lr="2e-05"))
    assert not r["L1"] and r["changes"]["objective"] == {"lr": ("1e-05", "2e-05")}
    r = layer1.verdict(BASE, dict(BASE, tensor_model_parallel_size="4"))
    assert r["L1"] and "tensor_model_parallel_size" in r["changes"]["placement"] and not r["unknown"]


def _swap(r, name="moe_grouped_gemm"):
    """How layer 1 judged one changed swap: its accepting or failing class, or the unknown reason."""
    for k in ("allowlisted", "ct_failed"):
        if name in r["changes"][k]:
            return k
    return r["changes"].get("unknown_why", {}).get(name)


def test_allowlisted_swap_passed_ct_accepted_failed_ct_refused_else_unknown(tmp_path):
    cand = dict(BASE, moe_grouped_gemm="True")
    r = layer1.verdict(BASE, cand)                                    # no setup directory: no swap listed
    assert r["L1"] and _swap(r) == "not in any allowlist"
    r = layer1.verdict(BASE, cand, _spec(tmp_path))
    assert r["L1"] and r["changes"]["allowlisted"] == {"moe_grouped_gemm": ("False", "True")} and not r["unknown"]
    r = layer1.verdict(BASE, cand, _spec(tmp_path, ct="CT-grouped-gemm: fail"))
    assert not r["L1"] and _swap(r) == "ct_failed"
    assert r["changes"]["ct_failed_why"] == {"moe_grouped_gemm": "component test CT-grouped-gemm recorded fail"}
    for ct in ("CT-other: pass", "CT-grouped-gemm: pending"):          # no CT recorded, or not run to a verdict
        r = layer1.verdict(BASE, cand, _spec(tmp_path, ct=ct))
        assert r["L1"] and _swap(r) == "component test CT-grouped-gemm not passed", ct
    r = layer1.verdict(BASE, dict(BASE, moe_grouped_gemm="sometimes"), _spec(tmp_path))
    assert r["L1"] and _swap(r) == "value outside its allowlist"
    r = layer1.verdict(BASE, cand, _spec(tmp_path, ct="CT-grouped-gemm: ' FAIL'"))   # statuses are normalized
    assert not r["L1"] and _swap(r) == "ct_failed"
    assert _swap(layer1.verdict(BASE, cand, _spec(tmp_path, ct="CT-grouped-gemm: Pass"))) == "allowlisted"


@pytest.mark.parametrize("swap", ["{arg: moe_grouped_gemm, values: 'true', component_test: CT-grouped-gemm}",
                                  "{values: any, component_test: CT-grouped-gemm}"])
def test_a_malformed_allowlist_swap_fails_layer1(tmp_path, swap):
    spec = _spec(tmp_path)
    (tmp_path / "allowlist.yaml").write_text("swaps:\n- %s\n" % swap)     # a plain string other than any; no arg
    r = layer1.verdict(BASE, dict(BASE, moe_grouped_gemm="True"), spec)
    assert not r["L1"] and "a swap needs arg and values" in r["errors"][0]


def test_allowlist_values_are_normalized_like_the_dump(tmp_path):
    spec = _spec(tmp_path)
    (tmp_path / "allowlist.yaml").write_text("swaps:\n- {arg: moe_grouped_gemm, values: [true, false, null], "
                                             "component_test: CT-grouped-gemm}\n")    # YAML booleans and null
    assert _swap(layer1.verdict(BASE, dict(BASE, moe_grouped_gemm="True"), spec)) == "allowlisted"
    assert _swap(layer1.verdict(dict(BASE, moe_grouped_gemm="None"), BASE, spec)) == "allowlisted"


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
    for image in (None,                                  # the environment is not read
                  "reg/img:latest",                      # not pinned by digest
                  "reg/img@sha256:" + "1" * 64,          # another image
                  IMAGE + "ff"):                         # not a digest
        assert _swap(layer1.verdict(BASE, cand, image=image)) == "not in any allowlist", image
    r = layer1.verdict(BASE, cand, image=IMAGE)          # no setup files needed
    assert r["L1"] and _swap(r) == "allowlisted" and r["image_dir"].endswith(PREFIX)


def test_setup_entry_is_judged_by_setup_tests_alone(tmp_path, monkeypatch):
    _image_rules(tmp_path, monkeypatch, allowlist="swaps:\n- {arg: moe_grouped_gemm, values: ['false'], "
                 "component_test: CT-grouped-gemm}\n")
    cand, setup = dict(BASE, moe_grouped_gemm="True"), tmp_path / "setup"
    setup.mkdir()
    (setup / "frozen_args.yaml").write_text("args: {}\n")
    (setup / "allowlist.yaml").write_text("swaps:\n- {arg: moe_grouped_gemm, values: any, component_test: CT-grouped-gemm}\n")
    (setup / "ct_status.yaml").write_text("tests: {}\n")
    spec = frozen_args.load(str(setup / "frozen_args.yaml"))
    r = layer1.verdict(BASE, cand, spec, image=IMAGE)     # the image's pass does not lend to the setup
    assert r["L1"] and _swap(r) == "component test CT-grouped-gemm not passed"


def test_setup_failure_overrides_the_image_pass(tmp_path, monkeypatch):
    _image_rules(tmp_path, monkeypatch)
    setup = tmp_path / "setup"
    setup.mkdir()
    (setup / "frozen_args.yaml").write_text("args: {}\n")
    (setup / "ct_status.yaml").write_text("tests:\n  CT-grouped-gemm: fail\n")
    r = layer1.verdict(BASE, dict(BASE, moe_grouped_gemm="True"), frozen_args.load(str(setup / "frozen_args.yaml")),
                       image=IMAGE)
    assert not r["L1"] and _swap(r) == "ct_failed"


def test_an_image_fail_refuses_a_swap_the_setup_allowlist_lists(tmp_path, monkeypatch):
    _image_rules(tmp_path, monkeypatch, ct="CT-grouped-gemm: fail")
    setup = tmp_path / "setup"
    setup.mkdir()
    spec = _spec(setup)                                   # the setup lists the swap and marks its test pass
    r = layer1.verdict(BASE, dict(BASE, moe_grouped_gemm="True"), spec, image=IMAGE)
    assert not r["L1"] and _swap(r) == "ct_failed"
    assert _swap(layer1.verdict(BASE, dict(BASE, moe_grouped_gemm="True"), spec)) == "allowlisted"  # other image


@pytest.mark.parametrize("make, error", [
    (lambda p: p.mkdir(), "exists but is not a readable file"),
    (lambda p: os.symlink(str(p) + ".missing", str(p)), "exists but is not a readable file"),
    (lambda p: p.write_text("defect:\n- {arg: lr, values: any, reason: r, evidence: e}\n"), "unknown top-level key"),
    (lambda p: p.write_text("defects: {arg: lr}\n"), "`defects` must be a list"),
])
def test_an_unreadable_or_malformed_defects_file_fails_layer1(tmp_path, monkeypatch, make, error):
    rules = _image_rules(tmp_path, monkeypatch)
    make(rules / "images" / PREFIX / "defects.yaml")
    r = layer1.verdict(BASE, BASE, image=IMAGE)
    assert not r["L1"] and error in r["errors"][0], r["errors"]


ROUTER = {"moe_router_fusion": "False", "moe_router_enable_expert_bias": "True"}
ROUTER_DEFECT = ("defects:\n- arg: moe_router_fusion\n  values: ['true']\n  when: {moe_router_enable_expert_bias: 'true'}\n"
                 "  reason: the fused router rounds small scores away when the router has an expert bias\n"
                 "  evidence: CT-router-fusion\n")


def _defect_rules(tmp_path, monkeypatch):
    """Image rules on IMAGE whose defects.yaml lists fused routing with an expert bias; a setup that freezes nothing."""
    rules = _image_rules(tmp_path, monkeypatch)
    (rules / "images" / PREFIX / "defects.yaml").write_text(ROUTER_DEFECT)
    setup = tmp_path / "setup"
    setup.mkdir()
    (setup / "frozen_args.yaml").write_text("args: {}\n")
    return frozen_args.load(str(setup / "frozen_args.yaml"))


def test_defect_entry_fails_only_when_its_condition_holds(tmp_path, monkeypatch):
    spec = _defect_rules(tmp_path, monkeypatch)
    base, fused = {**BASE, **ROUTER}, {**BASE, **ROUTER, "moe_router_fusion": "True"}
    r = layer1.verdict(base, fused, spec, image=IMAGE)
    why = r["changes"]["defect_why"]["moe_router_fusion"]
    assert not r["L1"] and r["changes"]["defect"] == {"moe_router_fusion": ("False", "True")}
    assert "expert bias" in why["reason"] and why["evidence"] == "CT-router-fusion"
    assert why["changed"] == ["moe_router_fusion"]
    no_bias = dict(base, moe_router_enable_expert_bias="False")
    r = layer1.verdict(no_bias, dict(no_bias, moe_router_fusion="True"), spec, image=IMAGE)   # no expert bias
    assert r["L1"] and r["changes"]["unknown_why"] == {"moe_router_fusion": "numeric class"}
    r = layer1.verdict(fused, base, spec, image=IMAGE)                            # fusion turned off
    assert r["L1"] and "moe_router_fusion" in r["changes"]["unknown"]
    assert layer1.verdict(base, fused, spec)["L1"]                                # the setup alone: no image rules
    r = layer1.verdict(fused, dict(fused, tensor_model_parallel_size="4"), spec, image=IMAGE)   # both in it
    assert r["L1"] and not r["changes"]["defect"]


def test_defect_judges_the_candidate_state_whichever_argument_changed(tmp_path, monkeypatch):
    spec = _defect_rules(tmp_path, monkeypatch)
    base = {**BASE, "moe_router_fusion": "True", "moe_router_enable_expert_bias": "False"}
    r = layer1.verdict(base, dict(base, moe_router_enable_expert_bias="True"), spec, image=IMAGE)  # bias on
    why = r["changes"]["defect_why"]["moe_router_fusion"]
    assert not r["L1"] and r["changes"]["defect"] == {"moe_router_fusion": ("True", "True")}
    assert why["changed"] == ["moe_router_enable_expert_bias"]
    assert r["changes"]["objective"] == {"moe_router_enable_expert_bias": ("False", "True")}


def test_image_defects_apply_and_a_malformed_entry_fails(tmp_path, monkeypatch):
    rules = _image_rules(tmp_path, monkeypatch)
    defects = rules / "images" / PREFIX / "defects.yaml"
    defects.write_text("defects:\n- {arg: brand_new_arg, values: any, reason: a known bug, evidence: report.md}\n")
    base, cand = dict(BASE, brand_new_arg="1"), dict(BASE, brand_new_arg="2")
    r = layer1.verdict(base, cand, image=IMAGE)
    assert not r["L1"] and r["changes"]["defect_why"]["brand_new_arg"]["reason"] == "a known bug"
    assert layer1.verdict(base, cand)["L1"]                                    # another image: unknown
    defects.write_text("defects:\n- {arg: brand_new_arg, values: any}\n")       # no reason or evidence
    r = layer1.verdict(base, cand, image=IMAGE)
    assert not r["L1"] and "a defect needs arg, values" in r["errors"][0]


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


def _write_pair(root, name, base_args=None, win_args=None, win_loss0=2.5, win_losses=None, base_losses=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "base.log").write_text(fixtures.train_log(args=base_args, losses=base_losses))
    (d / "win.log").write_text(fixtures.train_log(args=win_args, loss0=win_loss0, losses=win_losses))
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
    r = res[0]
    assert r["PASS"] and r["L1"] and r["STEP0"] and r["step0_shift"] == 0 and r["curve_check"]["passed"]
    assert r["counts_identical"] and r["review"] == [] and r["deviation_10_19"] == 0
    assert p.stdout.startswith("pass: PASS  (L1 pass, step-0 shift +0.000%, limit 5%")
    assert "review: loss curve pass; steps 10-19 mean deviation +0.000%" in p.stdout


def test_verdict_fail_pairs_exit_nonzero(tmp_path):
    _write_pair(tmp_path, "a-objective", win_args={"moe_aux_loss_coeff": "0.05"})
    _write_pair(tmp_path, "b-loss", win_loss0=4.0)
    _write_pair(tmp_path, "c-pass")
    p, res = _verdict(tmp_path, tmp_path)
    assert p.returncode == 1, p.stdout + p.stderr
    by = {r["pair"]: r for r in res}
    a, b = by["a-objective"], by["b-loss"]
    assert not a["L1"] and a["STEP0"] and a["L1_objective"] == {"moe_aux_loss_coeff": ["0.02", "0.05"]}
    assert b["L1"] and not b["STEP0"] and abs(b["step0_shift"] - 0.6) < 1e-9 and not b["PASS"]
    assert b["review"] == [{"kind": "curve_check", "error": None, "failed_metrics": b["curve_check"]["failed_metrics"]}]
    assert by["c-pass"]["PASS"]
    assert "a-objective: FAIL  (L1 FAIL" in p.stdout
    assert "L1 objective changes: moe_aux_loss_coeff 0.02 -> 0.05" in p.stdout
    assert "b-loss: FAIL  (L1 pass, step-0 shift +60.000%" in p.stdout


@pytest.mark.parametrize("shift, ok", [(0.04, True), (0.06, False), (-0.04, True), (-0.06, False)])
def test_step0_shift_within_five_percent_passes(tmp_path, shift, ok):
    d = _write_pair(tmp_path, "s", win_losses={0: 2.5 * (1 + shift)})
    p, res = _verdict(d, tmp_path)
    r = res[0]
    assert (p.returncode == 0) == ok and r["PASS"] == r["STEP0"] == ok and r["L1"], p.stdout
    assert abs(r["step0_shift"] - shift) < 1e-9 and r["curve_check"]["passed"]


def test_nan_loss_after_step0_fails(tmp_path):
    d = _write_pair(tmp_path, "nan", win_losses={7: float("nan")})
    p, res = _verdict(d, tmp_path)
    r = res[0]
    assert p.returncode == 1 and not r["PASS"] and not r["STEP0"] and r["step0_shift"] == 0, p.stdout
    assert r["step0_problems"] == ["candidate: training loss missing or not finite at steps 7"]
    assert r["curve_error"] == "loss not finite at steps 7" and r["review"][-1]["kind"] == "curve_check"


def test_nonfinite_step0_loss_on_either_side_fails(tmp_path):
    nan, inf = float("nan"), float("inf")
    cases = {"win-nan0": ({}, {0: nan}), "win-inf0": ({}, {0: inf}), "base-nan0": ({0: nan}, {}),
             "base-minf5": ({5: -inf}, {})}
    for name, (base_losses, win_losses) in cases.items():
        _write_pair(tmp_path, name, base_losses=base_losses, win_losses=win_losses)
    p, res = _verdict(tmp_path, tmp_path)
    assert p.returncode == 1, p.stdout + p.stderr
    by = {r["pair"]: r for r in res}
    assert all(not r["PASS"] and not r["STEP0"] and r["L1"] for r in res)
    assert by["win-nan0"]["step0_problems"] == ["candidate: training loss missing or not finite at steps 0"]
    assert by["win-inf0"]["step0_shift"] is None and by["base-nan0"]["step0_shift"] is None
    assert by["base-nan0"]["step0_problems"] == ["baseline: training loss missing or not finite at steps 0"]
    assert by["base-minf5"]["step0_problems"] == ["baseline: training loss missing or not finite at steps 5"]
    assert by["base-minf5"]["step0_shift"] == 0 and by["base-minf5"]["curve_error"] == "loss not finite at steps 5"


def test_read_steps_parses_nan_and_inf(tmp_path):
    import math

    import parity_verdict
    log = tmp_path / "x.log"
    log.write_text("".join("t step %d: {'train/loss': %s, 'train/grad_norm': %s}\n" % (i, v, g) for i, (v, g) in
                           enumerate((("nan", "1.0"), ("inf", "nan"), ("-inf", "2"), ("Infinity", "1e-3"),
                                      ("2.5e+00", "-"), ("garbled", "1")))))
    loss, _, _ = parity_verdict.read_steps(str(log))
    assert math.isnan(loss[0][0]) and loss[1][0] == math.inf and loss[2][0] == -math.inf and loss[3][0] == math.inf
    assert math.isnan(loss[1][1]) and loss[4] == (2.5, None) and 5 not in loss    # an unparseable loss is missing


def test_a_failed_json_write_keeps_the_exit_code(tmp_path, monkeypatch, capsys):
    import parity_verdict
    d = _write_pair(tmp_path, "p", win_losses={0: 3.0})          # a FAIL pair: exit 1
    monkeypatch.delenv("AUTOTUNE_FROZEN_ARGS", raising=False)
    monkeypatch.setattr(parity_verdict, "convergence", lambda *a: (None, "not needed here"))
    assert parity_verdict.main([str(d), "--json", str(tmp_path / "no" / "such" / "dir" / "v.json")]) == 1
    assert "cannot write" in capsys.readouterr().err
    good = _write_pair(tmp_path, "q")
    assert parity_verdict.main([str(good), "--json", str(tmp_path / "no" / "v.json")]) == 0


def test_review_errors_never_change_the_verdict(tmp_path, monkeypatch, capsys):
    import parity_verdict
    d = _write_pair(tmp_path, "p", win_args={"tensor_model_parallel_size": "4"})
    (d / "gpus.json").write_text("not json")

    def boom(*a, **k):
        raise RuntimeError("boom")
    for name in ("convergence", "deviation", "_tput"):
        monkeypatch.setattr(parity_verdict, name, boom)
    monkeypatch.delenv("AUTOTUNE_FROZEN_ARGS", raising=False)
    assert parity_verdict.main([str(d), "--json", str(tmp_path / "v.json")]) == 0
    r = json.loads((tmp_path / "v.json").read_text())[0]
    assert r["PASS"] and r["STEP0"] and r["curve_error"] == "the check raised RuntimeError: boom"
    errors = {u["what"] for u in r["review"] if u["kind"] == "review_error"}
    assert errors == {"gpus.json", "steps 10-19 deviation", "baseline throughput", "candidate throughput"}
    out = capsys.readouterr().out
    assert out.startswith("p: PASS") and "review: steps 10-19 deviation not computed (RuntimeError: boom)" in out


def test_curve_failure_and_unknown_flags_are_reviewed_without_failing(tmp_path):
    later = {t: 1.3 * (2.5 - 0.01 * t) for t in range(5, 20)}
    _write_pair(tmp_path, "curve", win_losses=later)
    _write_pair(tmp_path, "unknown", base_args={"use_precision_aware_optimizer": "False"},
                win_args={"use_precision_aware_optimizer": "True"})
    p, res = _verdict(tmp_path, tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    by = {r["pair"]: r for r in res}
    curve, unknown = by["curve"], by["unknown"]
    assert curve["PASS"] and curve["STEP0"] and not curve["curve_check"]["passed"]
    assert curve["review"] == [{"kind": "curve_check", "error": None,
                                "failed_metrics": curve["curve_check"]["failed_metrics"]}]
    assert abs(curve["deviation_10_19"] - 0.3) < 1e-9
    assert "review: loss curve FAIL (" in p.stdout and "steps 10-19 mean deviation +30.000%" in p.stdout
    assert unknown["PASS"] and unknown["review"] == [{"kind": "unknown", "name": "use_precision_aware_optimizer",
                                                      "from": "False", "to": "True", "reason": "numeric class"}]
    assert "review: unknown use_precision_aware_optimizer False -> True (numeric class)" in p.stdout


def test_verdict_short_log_and_no_pairs(tmp_path):
    d = tmp_path / "short"
    d.mkdir()
    (d / "base.log").write_text(fixtures.train_log(steps=12))
    (d / "win.log").write_text(fixtures.train_log(steps=12))
    p, res = _verdict(d, tmp_path)
    r = res[0]
    assert p.returncode == 1 and not r["STEP0"] and "12 of the 20 steps" in r["curve_error"]
    assert r["step0_problems"] == ["%s: training loss missing or not finite at steps 12-19" % s
                                   for s in ("baseline", "candidate")]
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


def test_pair_from_runs_pairs_longer_runs_on_steps_0_to_19(tmp_path):
    long_run = dict(RUN, num_rollout="30")
    p = _pair_from_runs(tmp_path, fixtures.train_log(args=long_run, steps=30, counts=False),
                        fixtures.train_log(args=dict(long_run, tensor_model_parallel_size="4"), steps=30, counts=False))
    assert p.returncode == 0, p.stderr
    q, res = _verdict(tmp_path / "pair", tmp_path)
    assert q.returncode == 0 and res[0]["PASS"], q.stdout


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
    short = fixtures.train_log(args=dict(RUN, num_rollout="10"), steps=10)   # a 10-step campaign's ranked run
    err = refused(short, short)
    assert "base: the run has 10 steps, fewer than the 20 the check scores; pair its 20-step confirm run" in err
    assert "make_run_yaml.py --benchmark --confirm" in err and "missing the training loss" not in err
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
