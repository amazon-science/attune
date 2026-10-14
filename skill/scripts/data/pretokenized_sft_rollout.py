# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""slime rollout adapter for pretokenized, fixed-length SFT rows (generate_fixed_sft.py).

slime calls generate_rollout(args, rollout_id, data_buffer, evaluation=False) once per rollout
(--rollout-function-path pretokenized_sft_rollout.generate_rollout). The rows come from slime's global dataset
(--prompt-data <rows.parquet> --input-key messages --metadata-key metadata, without --disable-rollout-global-dataset),
and each row's metadata holds its tokens and target mask, which the adapter passes on unchanged. Every row must hold
exactly $AUTOTUNE_SEQ_LEN tokens (default 8192), the row length make_run_yaml.py --benchmark writes into the pod env.
"""

import os

DEFAULT_SEQUENCE_LENGTH = 8192


def sequence_length():
    raw = os.environ.get("AUTOTUNE_SEQ_LEN", "").strip()
    return int(raw) if raw else DEFAULT_SEQUENCE_LENGTH


def _to_int_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("the fixed-length SFT adapter has no evaluation rollout")
    if not getattr(args, "rollout_global_dataset", True):
        raise ValueError("the rows are read through slime's global dataset: remove --disable-rollout-global-dataset")
    length = sequence_length()
    samples = data_buffer.get_samples(args.rollout_batch_size)
    for wrapped_sample in samples:
        (sample,) = wrapped_sample
        tokens = _to_int_list(sample.metadata["tokens"])
        loss_mask = _to_int_list(sample.metadata["loss_mask"])

        if len(tokens) != length:
            raise ValueError(f"Expected {length} tokens (AUTOTUNE_SEQ_LEN), got {len(tokens)}")
        if len(loss_mask) != length - 1:
            raise ValueError(f"Expected {length - 1} target-mask entries, got {len(loss_mask)}")
        if not any(loss_mask):
            raise ValueError("Fixed-length SFT row has no supervised target tokens")

        sample.tokens = tokens
        sample.response_length = len(tokens) - 1
        sample.loss_mask = loss_mask
        sample.reward = 0

    return samples
