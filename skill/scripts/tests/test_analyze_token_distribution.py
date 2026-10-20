# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""skill/scripts/data/analyze_token_distribution.py: length statistics, fixed-length detection, regime boundaries,
micro-batch floors, slime's first-fit packing and its DP and virtual-pipeline alignment, pre-tokenized and rendered
rows, JSON and Markdown output. Synthetic data and stub tokenizers only. Run: python3 -m pytest skill/scripts/tests"""
import importlib.util
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "skill" / "scripts" / "data" / "analyze_token_distribution.py"
_spec = importlib.util.spec_from_file_location("analyze_token_distribution", SCRIPT)
atd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atd)

LOADER_PROMPT = [{"role": "user", "content": "fixed-length throughput benchmark"}]


def run(argv, capsys):
    """(exit status, stdout, stderr) of the script's main()."""
    try:
        code = atd.main([str(a) for a in argv])
    except SystemExit as e:   # argparse errors
        code = e.code
    out, err = capsys.readouterr()
    return code, out, err


def record(argv, capsys):
    code, out, err = run(list(argv) + ["--format", "json"], capsys)
    assert code == 0, err
    return json.loads(out)


@pytest.fixture
def env(monkeypatch):
    for k in ("AUTOTUNE_GBS", "AUTOTUNE_NUM_STEPS"):
        monkeypatch.delenv(k, raising=False)
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):   # restored after the script sets them
        monkeypatch.setenv(k, "0")
    return monkeypatch


def write_parquet(path, columns, schema=None):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.table(columns, schema=schema), path)
    return path


def builder_rows(path, lengths, supervised_every=4):
    """Rows laid out as generate_fixed_sft.py writes them: a loader prompt, and tokens + target mask in metadata."""
    pa = pytest.importorskip("pyarrow")
    schema = pa.schema([
        pa.field("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        pa.field("metadata", pa.struct([("tokens", pa.list_(pa.int32())), ("loss_mask", pa.list_(pa.uint8())),
                                        ("source_doc_count", pa.int32()), ("row_hash", pa.string())]))])
    md = [{"tokens": [5] * n, "loss_mask": [int(j % supervised_every == 0) for j in range(n - 1)],
           "source_doc_count": 1, "row_hash": "h%d" % i} for i, n in enumerate(lengths)]
    return write_parquet(path, {"messages": [LOADER_PROMPT] * len(lengths), "metadata": md}, schema)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ---- statistics and regimes -----------------------------------------------------------------------------------------

def test_nearest_rank_statistics_and_threshold_shares():
    st = atd.length_stats(list(range(100, 0, -1)))
    assert st == {"rows": 100, "total_tokens": 5050, "min": 1, "mean": 50.5, "p50": 50, "p90": 90, "p99": 99,
                  "max": 100}
    lengths = [4095, 4096, 8191, 8192, 16384, 32768]
    shares = atd.at_or_above(lengths, sum(lengths))
    assert [shares[k]["rows"] for k in ("4096", "8192", "16384", "32768")] == [5, 3, 2, 1]   # "at or above"
    assert shares["16384"]["token_share"] == pytest.approx((16384 + 32768) / sum(lengths))
    assert shares["4096"]["row_share"] == pytest.approx(5 / 6)


@pytest.mark.parametrize("lengths, regime", [
    ([40000] * 3, "fixed-length"),
    ([100, 300, 8192], "short variable"),                   # longest row exactly 8,192
    ([100, 300, 8193], "long-tail variable"),               # one row past 8,192
    ([16383, 16383, 20000], "long-tail variable"),          # median just below 16,384
    ([1000, 16384, 16384, 40000], "long-context"),          # median exactly 16,384 (nearest rank)
])
def test_regime_boundaries(lengths, regime):
    assert atd.classify(atd.length_stats(lengths))[0] == regime


# ---- slime's dynamic batching ---------------------------------------------------------------------------------------

def slime_first_fit_bins(total_lengths, max_tokens_per_bin):
    """slime's seqlen_balancing.first_fit_pack, the reference the fast version must match."""
    bins, bin_sums = [], []
    for idx, length in enumerate(total_lengths):
        for j in range(len(bins)):
            if bin_sums[j] + length <= max_tokens_per_bin:
                bins[j].append(idx)
                bin_sums[j] += length
                break
        else:
            bins.append([idx])
            bin_sums.append(length)
    return bins


def slime_first_fit_pack(total_lengths, max_tokens_per_bin):
    return [sum(total_lengths[i] for i in b) for b in slime_first_fit_bins(total_lengths, max_tokens_per_bin)]


def slime_split_bin_by_tokens(bin_indices, lengths):
    """slime's seqlen_balancing._split_bin_by_tokens."""
    halves, sums = [[], []], [0, 0]
    for idx in sorted(bin_indices, key=lambda i: -lengths[i]):
        h = 0 if sums[0] <= sums[1] else 1
        halves[h].append(idx)
        sums[h] += lengths[idx]
    return halves


def slime_expand_bins_by_splitting(bins, target_count, lengths):
    """slime's seqlen_balancing.expand_bins_by_splitting."""
    while len(bins) < target_count:
        candidates = [(sum(lengths[i] for i in b), idx) for idx, b in enumerate(bins) if len(b) > 1]
        if not candidates:
            break
        _, idx = max(candidates)
        left, right = slime_split_bin_by_tokens(bins[idx], lengths)
        bins[idx] = left
        bins.append(right)


def slime_step_microbatches(step_lengths, cap, dp, vpp=1, mb_group=1):
    """Micro-batches of one step in slime's dp_schedule.build_dp_schedule (dynamic batching), or None where its
    asserts fire."""
    align_to = dp * (mb_group if vpp > 1 else 1)
    if len(step_lengths) < dp:
        return None
    bins = slime_first_fit_bins(step_lengths, cap)
    target = max(((len(bins) + align_to - 1) // align_to) * align_to, align_to)
    if target != len(bins):
        slime_expand_bins_by_splitting(bins, target, step_lengths)
        if len(bins) != target:
            return None
    return len(bins)


def test_first_fit_matches_slime():
    assert atd.first_fit([100, 200, 300], 300) == [300, 300]     # slime's --use-dynamic-batch-size example
    assert atd.first_fit([500, 10, 20], 100) == [500, 30]        # an oversized row sits alone
    rng = random.Random(7)
    for trial in range(40):
        lengths = [rng.choice([0, 1, rng.randint(1, 5000), rng.randint(1, 40000)]) for _ in range(rng.randint(1, 300))]
        for cap in (1024, 4096, 8192, 32768):
            assert atd.first_fit(lengths, cap) == slime_first_fit_pack(lengths, cap), (trial, cap)


def test_dp_alignment():
    assert atd.aligned_count(3, 10, 2) == 4      # split up to a multiple of DP
    assert atd.aligned_count(4, 10, 2) == 4
    assert atd.aligned_count(1, 10, 4) == 4      # at least DP micro-batches
    assert atd.aligned_count(1, 3, 4) is None    # three rows cannot make four micro-batches: slime asserts


def test_virtual_pipeline_alignment():
    assert atd.alignment(dp=2, pp=4, vpp=1, group=None) == (1, 2)    # no virtual stages: DP
    assert atd.alignment(dp=2, pp=4, vpp=2, group=None) == (4, 8)    # Megatron's group defaults to PP
    assert atd.alignment(dp=2, pp=4, vpp=2, group=3) == (3, 6)       # --microbatch-group-size-per-vp-stage
    assert atd.alignment(dp=2, pp=4, vpp=1, group=3) == (1, 2)       # the group applies only with virtual stages


def test_alignment_matches_slime_splitting():
    rng = random.Random(11)
    for trial in range(400):
        rows = rng.randint(1, 80)
        lengths = [rng.choice([1, rng.randint(1, 4000), rng.randint(1, 40000)]) for _ in range(rows)]   # >= 1 token
        cap = rng.choice([2048, 8192, 32768])
        dp, pp, vpp = rng.choice([1, 2, 4, 8]), rng.choice([1, 2, 4]), rng.choice([1, 1, 2, 3])
        group = rng.choice([None, None, 1, 2, 3])
        _, align = atd.alignment(dp, pp, vpp, group)
        expected = slime_step_microbatches(lengths, cap, dp, vpp, group or pp)
        assert atd.aligned_count(len(atd.first_fit(lengths, cap)), rows, align) == expected, (trial, dp, pp, vpp)


# ---- pre-tokenized rows ---------------------------------------------------------------------------------------------

def test_fixed_length_builder_rows(tmp_path, env, capsys):
    ds = builder_rows(tmp_path / "train.parquet", [1024] * 64)
    rec = record([ds, "--gbs", 16, "--steps", 3, "--max-tokens-per-gpu", "512,1024,2048", "--dp", 2], capsys)
    assert rec["fixed_length"] is True and rec["regime"]["name"] == "fixed-length"
    assert rec["source"]["tokens_field"] == "metadata.tokens" and rec["source"]["loss_mask_length"] == "tokens - 1"
    assert rec["stats"]["total_tokens"] == 64 * 1024 and rec["sample"]["full_pass"] is True
    assert rec["supervised"]["tokens"] == 64 * 256      # every 4th of the 1,023 target positions
    imp = rec["implications"]
    assert imp["min_tokens_per_microbatch"] == {"1": 1024, "2": 512, "4": 256, "8": 128}
    assert imp["fp32_logits_gib_tp1"] is None           # no vocabulary given
    assert imp["steps"]["simulated"] == 3 and imp["steps"]["tokens_per_step"] == {"mean": 16384, "min": 16384,
                                                                                    "max": 16384}
    p = {x["max_tokens_per_gpu"]: x for x in imp["packing"]}
    assert p[512]["rows_above_cap"] == 64 and p[512]["microbatches_per_step"]["mean"] == 16
    assert p[1024]["rows_per_microbatch"] == 1 and p[1024]["microbatches_per_dp_rank"]["max"] == 8
    assert p[2048]["rows_per_microbatch"] == 2 and p[2048]["microbatches_per_dp_rank"] == {"mean": 4, "min": 4,
                                                                                            "max": 4}


def test_cp_and_vocabulary(tmp_path, env, capsys):
    ds = builder_rows(tmp_path / "train.parquet", [3000] * 8)
    rec = record([ds, "--gbs", 4, "--cp", 2, "--max-tokens-per-gpu", 1500, "--vocab-size", 1000], capsys)
    imp = rec["implications"]
    assert imp["min_tokens_per_microbatch"] == {"1": 3000, "2": 1500, "4": 750, "8": 375}
    assert imp["fp32_logits_gib_tp1"]["2"] == pytest.approx(1500 * 1000 * 4 / 2 ** 30)
    assert imp["packing"][0]["cap"] == 3000 and imp["packing"][0]["rows_above_cap"] == 0   # mt x CP holds a row


def test_variable_length_input_ids_without_mask(tmp_path, env, capsys):
    lengths = [70, 5000, 300, 12000, 32768, 900, 2500, 18000, 60, 7000, 400, 9000]
    ds = write_parquet(tmp_path / "v.parquet", {"input_ids": [[1] * n for n in lengths], "messages": ["x"] * 12})
    rec = record([ds, "--gbs", 4, "--max-tokens-per-gpu", 32768], capsys)
    assert rec["source"]["tokens_field"] == "input_ids" and rec["supervised"] is None
    assert rec["source"]["loss_mask_length"] is None
    assert rec["fixed_length"] is False and rec["regime"]["name"] == "long-tail variable"
    assert (rec["stats"]["min"], rec["stats"]["p50"], rec["stats"]["max"]) == (60, 2500, 32768)
    assert rec["at_or_above"]["16384"]["rows"] == 2
    steps = rec["implications"]["steps"]
    sums = [sum(lengths[i:i + 4]) for i in (0, 4, 8)]
    assert steps["simulated"] == 3 and steps["tokens_per_step"] == {"mean": sum(sums) / 3, "min": min(sums),
                                                                     "max": max(sums)}
    assert steps["longest_row_per_step"]["max"] == 32768
    ks = [len(slime_first_fit_pack(lengths[i:i + 4], 32768)) for i in (0, 4, 8)]
    assert rec["implications"]["packing"][0]["microbatches_per_step"]["max"] == max(ks)


def test_virtual_pipeline_packing(tmp_path, env, capsys):
    ds = builder_rows(tmp_path / "train.parquet", [1000] * 64)
    args = [ds, "--gbs", 16, "--steps", 2, "--max-tokens-per-gpu", 4000, "--dp", 2, "--pp", 4]
    plain = record(args, capsys)["implications"]["packing"][0]
    assert plain["align_to"] == 2 and plain["microbatches_per_dp_rank"]["max"] == 2    # 4 micro-batches over DP 2
    rec = record(args + ["--vpp", 2], capsys)
    p = rec["implications"]["packing"][0]
    assert (p["microbatch_group"], p["align_to"]) == (4, 8) and p["microbatches_per_dp_rank"]["max"] == 4
    md = atd.render(rec)
    assert "2 virtual pipeline stages per rank" in md and "multiple of DP x the micro-batch group 4 (8)" in md
    short = record([ds, "--gbs", 6, "--steps", 2, "--max-tokens-per-gpu", 8000, "--dp", 2, "--pp", 4, "--vpp", 2],
                   capsys)
    assert short["implications"]["packing"][0]["dp_alignment_fails"] == 2              # 6 rows cannot make 8
    assert "fails at 2 steps (fewer rows than DP x 4)" in atd.render(short)


def test_parquet_metadata_as_json_string(tmp_path, env, capsys):
    md = [json.dumps({"tokens": [1] * n, "loss_mask": [1] * (n - 1)}) for n in (10, 30, 20)]
    ds = write_parquet(tmp_path / "s.parquet", {"messages": ["x"] * 3, "metadata": md})
    rec = record([ds, "--gbs", 1, "--steps", 3], capsys)
    assert rec["source"]["tokens_field"] == "metadata.tokens" and rec["stats"]["total_tokens"] == 60
    assert rec["supervised"]["tokens"] == 57 and rec["source"]["loss_mask_length"] == "tokens - 1"
    assert record([ds, "--gbs", 1, "--steps", 3, "--sample-n", 2, "--seed", 1], capsys)["stats"]["rows"] == 2


def test_sample_is_seeded_and_short_data_wraps(tmp_path, env, capsys):
    lengths = list(range(10, 210))
    ds = write_parquet(tmp_path / "s.parquet", {"tokens": [[1] * n for n in lengths]})
    a = record([ds, "--sample-n", 50, "--seed", 3], capsys)
    b = record([ds, "--sample-n", 50, "--seed", 3], capsys)
    c = record([ds, "--sample-n", 50, "--seed", 4], capsys)
    assert a == b and a["stats"] != c["stats"]
    assert a["sample"] == {"rows_measured": 50, "full_pass": False, "sample_n": 50, "seed": 3, "render_errors": 0,
                           "first_errors": []}
    assert a["implications"]["steps"]["order"] == "sample" and a["implications"]["steps"]["simulated"] == 0
    assert "The 50 measured rows hold fewer than one step of 128 rows" in atd.render(a)
    full = record([ds, "--gbs", 64], capsys)          # 200 rows: three of the twenty steps
    assert full["implications"]["steps"]["simulated"] == 3
    md = atd.render(full)
    assert "A 20-step run needs 1,280 rows; the dataset has 200" in md


def test_jsonl_metadata_as_dict_or_string(tmp_path, env, capsys):
    rows = [{"metadata": {"tokens": [1] * 10, "loss_mask": [1] * 9}},
            {"metadata": json.dumps({"tokens": [1] * 30, "loss_mask": [0] * 29})}]
    rec = record([write_jsonl(tmp_path / "r.jsonl", rows), "--gbs", 1, "--steps", 2], capsys)
    assert rec["stats"]["total_tokens"] == 40 and rec["supervised"]["tokens"] == 9
    assert rec["source"]["loss_mask_length"] == "tokens - 1"


def test_env_defaults_for_batch_and_steps(tmp_path, env, capsys):
    ds = write_jsonl(tmp_path / "r.jsonl", [{"tokens": [1] * 8}] * 12)
    env.setenv("AUTOTUNE_GBS", "4")
    env.setenv("AUTOTUNE_NUM_STEPS", "2")
    code, out, err = run([ds, "--format", "json"], capsys)
    steps = json.loads(out)["implications"]["steps"]
    assert (steps["gbs"], steps["requested"], steps["simulated"], steps["defaulted"]) == (4, 2, 2, [])
    assert code == 0 and "WARNING" not in err
    env.setenv("AUTOTUNE_GBS", "many")
    assert run([ds], capsys)[0] == 2


def test_default_batch_is_announced(tmp_path, env, capsys):
    ds = write_jsonl(tmp_path / "r.jsonl", [{"tokens": [1] * 8}] * 300)
    code, out, err = run([ds, "--format", "json"], capsys)
    rec = json.loads(out)
    assert code == 0 and rec["implications"]["steps"]["defaulted"] == ["gbs", "steps"]
    assert "WARNING: GBS 128 is the default benchmark's" in err and "WARNING: 20 steps" in err
    md = atd.render(rec)
    assert "Warning: GBS 128 is the default benchmark's, used because neither --gbs nor $AUTOTUNE_GBS was set" in md
    code, _, err = run([ds, "--gbs", 128, "--steps", 2], capsys)
    assert code == 0 and "WARNING" not in err                # an explicit 128 is the run's own batch


# ---- rendered rows --------------------------------------------------------------------------------------------------

class StubTokenizer:
    """One token per whitespace-separated word; a chat template that marks each role and the generation prompt."""

    def __len__(self):
        return 500

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False):
        assert tokenize is False
        words = []
        for m in messages:
            words += ["<%s>" % m["role"], m["content"]]
        return " ".join(words + (["<tools>"] if tools else []) + (["<assistant>"] if add_generation_prompt else []))


def stub_transformers(monkeypatch):
    seen = {}

    def from_pretrained(path, trust_remote_code=False, local_files_only=False):
        seen.update(path=path, local_files_only=local_files_only, offline=os.environ.get("HF_HUB_OFFLINE"))
        return StubTokenizer()
    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained)))
    return seen


CHATS = [{"messages": [{"role": "user", "content": "a b c"}, {"role": "assistant", "content": "d e"}]},
         {"messages": [{"role": "user", "content": "f"}, {"role": "assistant", "content": "g h i j"}],
          "tools": [{"name": "t"}]},
         {"messages": [{"role": "user", "content": None}]},           # unrenderable: reported, left out
         {"messages": [{"role": "user", "content": "k l"}]}]


def test_messages_rendered_offline_with_chat_template(tmp_path, env, capsys):
    seen = stub_transformers(env)
    tok = tmp_path / "tok"
    tok.mkdir()
    rec = record([write_jsonl(tmp_path / "c.jsonl", CHATS), "--tokenizer", tok, "--gbs", 1, "--steps", 1], capsys)
    assert seen == {"path": str(tok), "local_files_only": True, "offline": "1"}
    assert rec["source"]["pretokenized"] is False and rec["source"]["chat_template"] is True
    assert rec["sample"]["render_errors"] == 1 and "row 2" in rec["sample"]["first_errors"][0]
    assert (rec["stats"]["min"], rec["stats"]["max"]) == (3, 8)          # "<user> k l"; the tools add a word
    assert rec["stats"]["total_tokens"] == 7 + 8 + 3 and rec["supervised"] is None
    assert rec["implications"]["vocab_size"] == 500
    rl = record([write_jsonl(tmp_path / "c.jsonl", CHATS), "--tokenizer", tok, "--rl-prompts",
                 "--max-response-len", 100], capsys)
    assert rl["stats"]["total_tokens"] == 7 + 8 + 3 + 3                # the generation prompt on every row
    assert rl["implications"]["longest_row"] == 9 + 100
    md = atd.render(rl)
    assert "These are RL prompts" in md and "100-token response cap" in md
    assert "The chat template alone can differ" in atd.render(rec) and "The chat template alone" not in md


def test_tools_as_json_string_or_column(tmp_path, env, capsys):
    stub_transformers(env)
    tok = tmp_path / "tok"
    tok.mkdir()
    msgs = [{"role": "user", "content": "a b"}]
    rows = [{"messages": msgs, "tools": json.dumps([{"name": "t"}])},       # parsed as slime's Dataset does
            {"messages": msgs, "tools": "not json"},                         # unparseable: reported, left out
            {"messages": msgs, "tools": json.dumps({"name": "t"})},          # not a list: slime asserts
            {"messages": msgs}]
    rec = record([write_jsonl(tmp_path / "t.jsonl", rows), "--tokenizer", tok, "--gbs", 1, "--steps", 1], capsys)
    assert rec["sample"]["render_errors"] == 2 and "tools must be a list" in rec["sample"]["first_errors"][1]
    assert rec["stats"]["rows"] == 2 and (rec["stats"]["min"], rec["stats"]["max"]) == (3, 4)   # "<tools>" added
    pq_rows = write_parquet(tmp_path / "t.parquet", {"messages": [msgs, msgs],
                                                     "tools": [json.dumps([{"name": "t"}]), None]})
    rec = record([pq_rows, "--tokenizer", tok, "--gbs", 1, "--steps", 1], capsys)
    assert rec["sample"]["render_errors"] == 0 and rec["stats"]["total_tokens"] == 4 + 3
    rec = record([pq_rows, "--tokenizer", tok, "--tool-key", "other", "--gbs", 1, "--steps", 1], capsys)
    assert rec["stats"]["total_tokens"] == 3 + 3                            # no such column: no tools


def test_text_rows_and_default_sample(tmp_path, env, capsys):
    stub_transformers(env)
    tok = tmp_path / "tok"
    tok.mkdir()
    ds = write_parquet(tmp_path / "t.parquet", {"text": ["w " * (i % 50 + 1) for i in range(12000)]})
    rec = record([ds, "--tokenizer", tok, "--no-hash"], capsys)
    assert rec["source"]["input_key"] == "text"
    assert rec["sample"]["sample_n"] == 10000 and rec["sample"]["rows_measured"] == 10000
    assert (rec["stats"]["min"], rec["stats"]["max"]) == (1, 50) and rec["regime"]["name"] == "short variable"
    md = atd.render(rec)
    assert "Steps 0-19 at GBS 128 rows (2,560 rows of the sample in file order, an estimate" in md
    assert "wrap into the next epoch" not in md          # 12,000 rows hold the 20 steps


def test_loss_mask_type_uses_slime_generator(tmp_path, env, capsys):
    stub_transformers(env)

    class MultiTurnLossMaskGenerator:
        def __init__(self, tokenizer, tokenizer_type):
            assert tokenizer_type == "qwen3_5"

        def get_loss_mask(self, messages, tools=None):
            tokens = [1] * (10 * len(messages))
            return tokens, [int(m["role"] == "assistant") for m in messages for _ in range(10)]
    pkg = types.ModuleType("slime")
    utils = types.ModuleType("slime.utils")
    mask_utils = types.ModuleType("slime.utils.mask_utils")
    mask_utils.MultiTurnLossMaskGenerator = MultiTurnLossMaskGenerator
    for name, mod in (("slime", pkg), ("slime.utils", utils), ("slime.utils.mask_utils", mask_utils)):
        env.setitem(sys.modules, name, mod)
    tok = tmp_path / "tok"
    tok.mkdir()
    rec = record([write_jsonl(tmp_path / "c.jsonl", CHATS), "--tokenizer", tok, "--loss-mask-type", "qwen3_5"],
                 capsys)
    assert rec["stats"]["total_tokens"] == 60 and rec["supervised"] == {"tokens": 20, "share": 20 / 60}
    assert rec["source"]["chat_template"] is False and rec["source"]["loss_mask_type"] == "qwen3_5"


# ---- output and errors ----------------------------------------------------------------------------------------------

def test_markdown_is_the_rendered_json(tmp_path, env, capsys):
    ds = builder_rows(tmp_path / "train.parquet", [600, 9000, 1200, 20000, 300, 4500, 16384, 800])
    args = [ds, "--gbs", 2, "--steps", 4, "--max-tokens-per-gpu", "8192,32768", "--dp", 2, "--vocab-size", 4096]
    rec = record(args, capsys)
    (tmp_path / "rec.json").write_text(json.dumps(rec))
    code, md, _ = run(args, capsys)
    assert code == 0 and run(["--render", tmp_path / "rec.json"], capsys)[1] == md == atd.render(rec)
    assert md.startswith("## Token-length distribution\n")
    assert "| 8 | 52,784 | 300 | 6,598 | 1,200 | 20,000 | 20,000 | 20,000 |" in md
    assert "Regime: long-tail variable (median 1,200 below 16,384, longest row 20,000 above 8,192)." in md
    assert "| tokens per GPU for the longest row | 20,000 | 10,000 | 5,000 | 2,500 |" in md
    assert "sha256 `%s`" % rec["dataset"]["sha256"] in md and len(rec["dataset"]["sha256"]) == 64
    assert "| 8,192 | 3 |" in md and "| 32,768 | 0 |" in md       # 9,000, 16,384 and 20,000 exceed 8,192
    assert record(args, capsys) == rec   # deterministic


def test_runs_from_stdin_as_in_the_log_pod(tmp_path, env):
    ds = write_jsonl(tmp_path / "r.jsonl", [{"tokens": [1] * n} for n in (5, 7, 9, 11)])
    with open(SCRIPT, "rb") as script:
        p = subprocess.run([sys.executable, "-", str(ds), "--gbs", "2", "--format", "json"], stdin=script,
                           capture_output=True, timeout=60, cwd=tmp_path,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert p.returncode == 0, p.stderr
    rec = json.loads(p.stdout)
    assert rec["stats"]["total_tokens"] == 32 and rec["implications"]["steps"]["simulated"] == 2
    assert sorted(os.listdir(tmp_path)) == ["r.jsonl"]   # wrote nothing


@pytest.mark.parametrize("make, extra, message", [
    (lambda d: d / "missing.parquet", [], "no dataset file"),
    (lambda d: write_jsonl(d / "x.csv", [{"tokens": [1]}]), [], "expected a .parquet file"),
    (lambda d: write_jsonl(d / "raw.jsonl", CHATS), [], "pass --tokenizer"),
    (lambda d: write_jsonl(d / "raw.jsonl", CHATS), ["--tokenizer", "/nonexistent/tok"], "is not a directory"),
    (lambda d: write_jsonl(d / "e.jsonl", []), [], "holds no JSON object rows"),
    (lambda d: write_jsonl(d / "k.jsonl", [{"other": 1}]), [], "no pre-tokenized field"),
    (lambda d: write_jsonl(d / "k.jsonl", [{"tokens": [1]}]), ["--tokens-key", "ids"], "no 'ids' column"),
    (lambda d: write_jsonl(d / "b.jsonl", [{"tokens": 3}]), [], "is not a list"),
    (lambda d: write_jsonl(d / "z.jsonl", [{"tokens": []}] * 3), [], "holds 0 tokens"),
])
def test_clear_errors(tmp_path, env, capsys, make, extra, message):
    code, _, err = run([make(tmp_path)] + extra, capsys)
    assert code == 2 and message in err and "Traceback" not in err


def test_argument_errors(tmp_path, env, capsys):
    ds = write_jsonl(tmp_path / "r.jsonl", [{"tokens": [1]}])
    for argv in ([], [ds, "--render", ds], [ds, "--max-tokens-per-gpu", "4k"], [ds, "--dp", 0], [ds, "--pp", 0],
                 [ds, "--vpp", 0], [ds, "--microbatch-group-size", 0],
                 [ds, "--sample-n", -1], [ds, "--rl-prompts", "--loss-mask-type", "qwen3_5"]):
        assert run(argv, capsys)[0] == 2, argv
    (tmp_path / "other.json").write_text("{}")
    code, _, err = run(["--render", tmp_path / "other.json"], capsys)
    assert code == 2 and "not an analyze_token_distribution record" in err


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK) and SCRIPT.read_text().startswith("#!/usr/bin/env python3\n")
