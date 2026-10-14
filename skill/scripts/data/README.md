# Fixed-length SFT rows

ATTUNE ranks recipes on rows of exactly 8,192 tokens, built once per tokenizer from a pinned SFT dataset (the Tulu 3
SFT mixture in the examples below). This directory holds what builds, checks and feeds those rows.

| File | Purpose |
|---|---|
| `generate_fixed_sft.py` | builds the rows from the source Parquet shards and writes `<output>.manifest.json` beside them |
| `patch_glm5_mask.py` | adds a glm5 loss mask to slime's `mask_utils.py`, in place; run it before a GLM-5.2 build |
| `patch_gptoss_mask.py` | adds a gpt_oss (harmony) loss mask to slime's `mask_utils.py`, in place; run it before a gpt-oss build |
| `pretokenized_sft_rollout.py` | the slime rollout adapter; it feeds each row's stored tokens and mask unchanged |
| `verify_rows.py` | checks a built file against a recorded dataset build |

## What the builder does

The builder shuffles the source conversations with `random.Random(seed)`. It renders each one with the tokenizer's
chat template and slime's `MultiTurnLossMaskGenerator`, then concatenates them with an EOS token (mask 0) between
conversations. The stream is cut into consecutive rows of `--sequence-length` tokens. A row with no supervised target
is skipped, and the remainder after the last full row is dropped, so no row is padded. Each row stores its tokens, its
target mask (the token mask shifted by one, `sequence-length - 1` entries) and a `row_hash` in `metadata`. Its
`messages` field holds a fixed one-turn prompt that lets slime's loader initialize. The builder then re-reads the file,
validates every row, and writes the manifest with the source, tokenizer and output hashes.

## Requirements

- Python 3.10 or later, `pyarrow` and `transformers`.
- A slime checkout on `PYTHONPATH`. The builder imports slime's `Dataset`, `MultiTurnLossMaskGenerator` and
  `load_processor`. Upstream slime (<https://github.com/THUDM/slime>) has the `qwen3_5` loss mask; GLM-5.2 needs
  `patch_glm5_mask.py` and gpt-oss `patch_gptoss_mask.py`. The `deepseek_v4` mask needs no slime generator; the
  loader check then uses miles when slime is absent.

## Build

You run the build yourself, once per tokenizer, outside a tuning campaign: it downloads data and writes outside the
campaign directory, which the campaign's command guard does not allow.

1. Download the source at the pinned revision:

   ```bash
   huggingface-cli download allenai/tulu-3-sft-mixture --repo-type dataset \
     --revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be --local-dir tulu-3-sft-mixture
   ```

   The six shards in `tulu-3-sft-mixture/data/` should have these sha256 values:

   | Shard | Rows | sha256 |
   |---|---:|---|
   | `train-00000-of-00006.parquet` | 156,558 | `41d4524345443ccca4202df2f01d84d84371afcde36f637178e0e2bcd14f79c7` |
   | `train-00001-of-00006.parquet` | 156,557 | `c612164eabfacb99e37de582bf91712f878602755cf87daaa514bdeebec71f56` |
   | `train-00002-of-00006.parquet` | 156,557 | `e15d19cb441578dacefa251e18023e2612497eaead1191c8b31acebf1a45260b` |
   | `train-00003-of-00006.parquet` | 156,557 | `b9e312d1a0e777a3a64450cb89d2e28116ef96b4f85875157deddc811599be79` |
   | `train-00004-of-00006.parquet` | 156,557 | `44ddad0bc45d272c4be32bd78a1473c0902e153d42178366414e0532ab29dd27` |
   | `train-00005-of-00006.parquet` | 156,557 | `244ec7ec0591fc67553ca0c9f42a0cba2b34a100438369bd323267a93b3b34ca` |

2. Use the model's Hugging Face checkpoint directory as the tokenizer. Its `tokenizer.json` sha256 should equal the
   one recorded for the build.

3. Build. For example, one set for the Qwen3.5 models, built with the Qwen3.5-122B-A10B tokenizer:

   ```bash
   export PYTHONPATH=<slime checkout>:$PYTHONPATH TOKENIZERS_PARALLELISM=false
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output qwen35-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <Qwen3.5-122B-A10B checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type qwen3_5 \
     --sequence-length 8192 --rows 6400 --seed 42 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

   The three GLM-5.2 entries use a second set. Patch slime first; the patch edits `slime/utils/mask_utils.py` in the
   checkout on `PYTHONPATH` and does nothing if a glm5 generator is already there:

   ```bash
   python3 -u patch_glm5_mask.py
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output glm52-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <GLM-5.2 checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type glm5 \
     --sequence-length 8192 --rows 12800 --seed 20260828 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

   DeepSeek-V4 ships no chat template: `--loss-mask-type deepseek_v4` (`dsv4_mask.py`) renders each conversation with
   the checkout's official `encoding/encoding_dsv4.py` in chat mode and supervises the assistant turns. It runs that
   file, so it needs `--trust-remote-code`; use a checkout of `deepseek-ai/DeepSeek-V4-Flash` (revision
   `60d8d70770c6776ff598c94bb586a859a38244f1`), which carries `encoding/`:

   ```bash
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output dsv4-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <DeepSeek-V4-Flash checkout> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --loss-mask-type deepseek_v4 --trust-remote-code \
     --sequence-length 8192 --rows 6400 --seed 42
   ```

   The gpt-oss entries use one set, built with the gpt-oss-20b tokenizer (revision
   `6cee5e81ee83917806bbde320786a8fb61efebee`), whose tokenizer files and chat template are byte-identical to
   gpt-oss-120b's. Patch slime first; the patch adds the `gpt_oss` mask, which supervises each assistant turn from
   after `<|start|>assistant` through its closing `<|end|>`, `<|return|>` or `<|call|>`, freezes the template's date to
   2026-10-01 and skips conversations the harmony template would not render faithfully:

   ```bash
   python3 -u patch_gptoss_mask.py
   python3 -u generate_fixed_sft.py \
     --input tulu-3-sft-mixture/data \
     --output gptoss-tulu3-fixed8192-v1/train.parquet \
     --tokenizer <gpt-oss-20b checkpoint> \
     --source-dataset allenai/tulu-3-sft-mixture \
     --source-revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
     --input-key messages --source-key source --loss-mask-type gpt_oss \
     --sequence-length 8192 --rows 6400 --seed 42 \
     --source-batch-size 2048 --write-batch-size 64 --min-token-entropy-bits 8.0
   ```

4. Verify against a recorded build:

   ```bash
   python3 verify_rows.py qwen35-tulu3-fixed8192-v1/train.parquet --entry <build record>.json
   ```

   `rows_sha256` is the sha256 of the rows' `row_hash` values in file order, one per line. It matches when every row
   matches, whatever Parquet writer produced the file. `parquet_sha256` matches only a byte-identical file, which also
   needs the same `pyarrow` version. The script exits non-zero on a row error or a `rows_sha256` mismatch.

## Using the rows

Put this directory on `PYTHONPATH` and pass these flags to slime:
`--prompt-data <train.parquet> --input-key messages --metadata-key metadata
--rollout-function-path pretokenized_sft_rollout.generate_rollout`. The adapter reads the rows through slime's
global dataset, so the command must not pass `--disable-rollout-global-dataset` (`make_run_yaml.py --benchmark`
removes it). Every row must hold exactly `$AUTOTUNE_SEQ_LEN` tokens (default 8,192), the row length
`make_run_yaml.py --benchmark` writes into the pod environment; the adapter refuses any other.

## Limits

- The library versions of the original builds (transformers, tokenizers, pyarrow, slime) were not recorded. A
  different chat template or tokenizer version can change the rows; `rows_sha256` detects it.
- The GLM-5.2 set was built with `patch_glm5_mask.py` applied to a slime that had no glm5 generator. With a slime that
  ships its own, the patch does nothing and the rows can differ.
- The recorded builds ran with the tokenizer's remote code allowed. The Qwen3.5 tokenizer has no custom code (its
  `tokenizer_config.json` has no `auto_map`), so the Qwen3.5 command needs no `--trust-remote-code`. If the GLM-5.2
  checkpoint's `tokenizer_config.json` has an `auto_map`, add `--trust-remote-code` to the GLM-5.2 command;
  `verify_rows.py` shows whether the rows match.
- The gpt-oss set was built with transformers 5.8.1, tokenizers 0.22.2 and pyarrow 23.0.1; rebuilt on two more
  clusters from the same inputs, it matched byte for byte.
- `row_hash` hashes the tokens as native 32-bit integers, so it assumes a little-endian host (x86-64 or ARM64).
