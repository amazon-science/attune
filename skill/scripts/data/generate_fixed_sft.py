#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build exact-length, pretokenized SFT rows from a conversation Parquet (README.md here gives the build commands).

Needs pyarrow, transformers and a slime or miles checkout on PYTHONPATH. The tokenizer's own Python code runs only
with --trust-remote-code; --loss-mask-type deepseek_v4 also needs it, because it runs the checkout's official encoder
(dsv4_mask.py).
"""

import argparse
import bisect
import glob
import hashlib
import json
import math
import os
import random
import statistics
import time
from array import array
from collections import Counter
from pathlib import Path

LOADER_PROMPT = [{"role": "user", "content": "fixed-length throughput benchmark"}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        required=True,
        help="A Parquet file, a directory of Parquet shards, or a glob expression.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source-dataset")
    parser.add_argument("--source-revision")
    parser.add_argument("--input-key", default="messages")
    parser.add_argument("--tool-key", default="tools")
    parser.add_argument("--source-key", default="source")
    parser.add_argument("--loss-mask-type", default="qwen3_5",
                        help="a slime MultiTurnLossMaskGenerator type, or deepseek_v4 (dsv4_mask.py)")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=6400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-batch-size", type=int, default=2048)
    parser.add_argument("--write-batch-size", type=int, default=128)
    parser.add_argument("--min-token-entropy-bits", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="let transformers run Python code shipped with the tokenizer checkout (off by default)")
    return parser.parse_args()


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_digest(path: Path) -> tuple[str, dict[str, str]]:
    files = {}
    digest = hashlib.sha256()
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        candidate = path / name
        if not candidate.is_file():
            continue
        value = sha256_file(candidate)
        files[name] = value
        digest.update(name.encode())
        digest.update(value.encode())
    if not files:
        raise ValueError(f"No tokenizer files found under {path}")
    return digest.hexdigest(), files


def resolve_source_files(input_spec: str) -> list[Path]:
    source = Path(input_spec).expanduser()
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(source.rglob("*.parquet"))
    else:
        files = sorted(Path(match) for match in glob.glob(input_spec, recursive=True))
        files = [path for path in files if path.is_file() and path.suffix == ".parquet"]
    if not files:
        raise FileNotFoundError(f"No Parquet source files matched {input_spec!r}")
    return [path.resolve() for path in files]


def source_inventory(paths: list[Path]) -> tuple[str, list[dict[str, int | str]]]:
    import pyarrow.parquet as pq

    common_parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
    aggregate = hashlib.sha256()
    files: list[dict[str, int | str]] = []
    for path in paths:
        relative_path = str(path.relative_to(common_parent))
        file_hash = sha256_file(path)
        parquet = pq.ParquetFile(path)
        aggregate.update(relative_path.encode())
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode())
        aggregate.update(b"\0")
        files.append(
            {
                "path": relative_path,
                "sha256": file_hash,
                "bytes": path.stat().st_size,
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
            }
        )
    return aggregate.hexdigest(), files


def row_digest(tokens: list[int], loss_mask: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(array("i", tokens).tobytes())
    digest.update(bytes(loss_mask))
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def validate_dataset(
    output: Path,
    expected_rows: int,
    sequence_length: int,
    tokenizer_size: int,
) -> dict[str, int]:
    import pyarrow.parquet as pq

    seen_hashes: set[str] = set()
    rows = 0
    supervised_tokens = 0

    parquet = pq.ParquetFile(output)
    for batch in parquet.iter_batches(columns=["messages", "metadata"], batch_size=128):
        for record in batch.to_pylist():
            messages = record["messages"]
            metadata = record["metadata"]
            tokens = metadata["tokens"]
            loss_mask = metadata["loss_mask"]
            expected_hash = metadata["row_hash"]

            if messages != LOADER_PROMPT:
                raise ValueError(f"row {rows}: unexpected loader prompt {messages!r}")
            if len(tokens) != sequence_length:
                raise ValueError(f"row {rows}: token length {len(tokens)} != {sequence_length}")
            if len(loss_mask) != sequence_length - 1:
                raise ValueError(f"row {rows}: loss-mask length {len(loss_mask)} != {sequence_length - 1}")
            if any(token < 0 or token >= tokenizer_size for token in tokens):
                raise ValueError(f"row {rows}: token ID outside [0, {tokenizer_size})")
            if any(value not in (0, 1) for value in loss_mask):
                raise ValueError(f"row {rows}: loss mask is not binary")
            if not any(loss_mask):
                raise ValueError(f"row {rows}: no supervised target tokens")
            if row_digest(tokens, loss_mask) != expected_hash:
                raise ValueError(f"row {rows}: row hash mismatch")
            if expected_hash in seen_hashes:
                raise ValueError(f"row {rows}: duplicate row hash {expected_hash}")

            seen_hashes.add(expected_hash)
            supervised_tokens += sum(loss_mask)
            rows += 1

    if rows != expected_rows:
        raise ValueError(f"row count {rows} != {expected_rows}")
    return {"rows": rows, "supervised_tokens": supervised_tokens}


def validate_slime_loader(
    output: Path,
    tokenizer,
    tokenizer_path: str,
    sequence_length: int,
    trust_remote_code: bool,
) -> dict[str, int | bool]:
    import pyarrow.parquet as pq
    try:
        from slime.utils.data import Dataset
        from slime.utils.processing_utils import load_processor
    except ImportError:
        from miles.utils.data import Dataset
        from miles.utils.processing_utils import load_processor

    processor = load_processor(tokenizer_path, trust_remote_code=trust_remote_code)
    parquet = pq.ParquetFile(output)
    probe_path = output.with_name(f"{output.stem}.loader-probe.{os.getpid()}.parquet")
    probe_path.unlink(missing_ok=True)
    pq.write_table(parquet.read_row_group(0).slice(0, 1), probe_path)
    try:
        dataset = Dataset(
            str(probe_path),
            tokenizer=tokenizer,
            processor=processor,
            max_length=None,
            prompt_key="messages",
            metadata_key="metadata",
            seed=0,
        )
    finally:
        probe_path.unlink(missing_ok=True)
    if len(dataset) == 0:
        raise ValueError("Slime Dataset loaded no rows")

    sample = dataset.origin_samples[0]
    tokens = sample.metadata["tokens"]
    loss_mask = sample.metadata["loss_mask"]
    if sample.prompt != LOADER_PROMPT:
        raise ValueError(f"Slime Dataset changed loader prompt to {sample.prompt!r}")
    if len(tokens) != sequence_length or len(loss_mask) != sequence_length - 1:
        raise ValueError(
            "Slime Dataset changed pretokenized lengths: "
            f"tokens={len(tokens)}, loss_mask={len(loss_mask)}"
        )
    if sample.multimodal_inputs and any(value is not None for value in sample.multimodal_inputs.values()):
        raise ValueError(f"Loader prompt unexpectedly produced multimodal inputs: {sample.multimodal_inputs!r}")

    return {
        "source_rows": parquet.metadata.num_rows,
        "probe_rows": len(dataset),
        "processor_loaded": processor is not None,
        "first_row_tokens": len(tokens),
        "first_row_loss_mask": len(loss_mask),
    }


def main() -> None:
    args = parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be at least 2")
    if args.rows < 1:
        raise ValueError("--rows must be positive")
    if args.source_batch_size < 1:
        raise ValueError("--source-batch-size must be positive")

    source_files = resolve_source_files(args.input)
    output = Path(args.output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f"{output.stem}.tmp.{os.getpid()}{output.suffix}"
    )
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + f".tmp.{os.getpid()}")
    temporary_output.unlink(missing_ok=True)
    temporary_manifest.unlink(missing_ok=True)

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for assistant-mask alignment")
    tokenizer_size = len(tokenizer)
    if args.loss_mask_type == "deepseek_v4":
        if not args.trust_remote_code:
            raise ValueError("--loss-mask-type deepseek_v4 runs the checkout's encoding_dsv4.py: pass --trust-remote-code")
        from dsv4_mask import DeepSeekV4LossMaskGenerator, load_encoder

        mask_generator = DeepSeekV4LossMaskGenerator(tokenizer, load_encoder(args.tokenizer))
    else:
        from slime.utils.mask_utils import MultiTurnLossMaskGenerator

        mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
    tokenizer_hash, tokenizer_files = tokenizer_digest(Path(args.tokenizer))
    source_hash, source_file_inventory = source_inventory(source_files)

    source_parquets = [pq.ParquetFile(path) for path in source_files]
    available_columns_by_file: list[set[str]] = []
    for path, source_parquet in zip(source_files, source_parquets, strict=True):
        available_columns = set(source_parquet.schema_arrow.names)
        if args.input_key not in available_columns:
            raise KeyError(
                f"{args.input_key!r} not found in {sorted(available_columns)} for {path}"
            )
        available_columns_by_file.append(available_columns)

    columns = [args.input_key]
    if all(args.tool_key in available for available in available_columns_by_file):
        columns.append(args.tool_key)
    if all(args.source_key in available for available in available_columns_by_file):
        columns.append(args.source_key)

    source_tables = [pq.read_table(path, columns=columns) for path in source_files]
    source_ends: list[int] = []
    source_row_count = 0
    for table in source_tables:
        source_row_count += table.num_rows
        source_ends.append(source_row_count)

    schema = pa.schema(
        [
            pa.field(
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string(), nullable=False),
                            pa.field("content", pa.string(), nullable=False),
                        ]
                    )
                ),
                nullable=False,
            ),
            pa.field(
                "metadata",
                pa.struct(
                    [
                        pa.field("tokens", pa.list_(pa.int32()), nullable=False),
                        pa.field("loss_mask", pa.list_(pa.uint8()), nullable=False),
                        pa.field("source_doc_count", pa.int32(), nullable=False),
                        pa.field("row_hash", pa.string(), nullable=False),
                    ]
                ),
                nullable=False,
            ),
        ]
    )

    rng = random.Random(args.seed)
    source_order = list(range(source_row_count))
    rng.shuffle(source_order)

    writer = pq.ParquetWriter(temporary_output, schema=schema, compression="zstd")
    pending_tokens: list[int] = []
    pending_masks: list[int] = []
    pending_doc_ids: list[int] = []
    output_batch: list[dict] = []
    token_counts: Counter[int] = Counter()
    source_doc_counts: list[int] = []
    seen_hashes: set[str] = set()
    generated_rows = 0
    source_docs = 0
    source_component_counts: Counter[str] = Counter()
    malformed_docs = 0
    empty_mask_blocks = 0

    try:
        for batch_start in range(0, len(source_order), args.source_batch_size):
            global_indices = source_order[
                batch_start : batch_start + args.source_batch_size
            ]
            records: list[dict | None] = [None] * len(global_indices)
            indices_by_file: dict[int, list[tuple[int, int]]] = {}
            for output_position, global_index in enumerate(global_indices):
                file_index = bisect.bisect_right(source_ends, global_index)
                file_start = 0 if file_index == 0 else source_ends[file_index - 1]
                indices_by_file.setdefault(file_index, []).append(
                    (output_position, global_index - file_start)
                )
            for file_index, positions_and_indices in indices_by_file.items():
                selected = source_tables[file_index].take(
                    pa.array(
                        [local_index for _, local_index in positions_and_indices],
                        type=pa.int64(),
                    )
                )
                for (output_position, _), record in zip(
                    positions_and_indices,
                    selected.to_pylist(),
                    strict=True,
                ):
                    records[output_position] = record

            for record in records:
                if record is None:
                    raise RuntimeError("Failed to materialize a shuffled source record")
                messages = record.get(args.input_key)
                tools = record.get(args.tool_key)
                if not isinstance(messages, list):
                    malformed_docs += 1
                    continue

                try:
                    tokens, full_mask = mask_generator.get_loss_mask(messages, tools=tools)
                except Exception as error:
                    malformed_docs += 1
                    if malformed_docs <= 10:
                        print(f"Skipping malformed source row: {type(error).__name__}: {error}", flush=True)
                    continue
                if not tokens or len(tokens) != len(full_mask):
                    malformed_docs += 1
                    continue

                source_docs += 1
                source_component_counts[str(record.get(args.source_key, "unknown"))] += 1
                doc_id = source_docs
                if pending_tokens and tokenizer.eos_token_id is not None:
                    pending_tokens.append(int(tokenizer.eos_token_id))
                    pending_masks.append(0)
                    pending_doc_ids.append(doc_id)
                pending_tokens.extend(int(token) for token in tokens)
                pending_masks.extend(int(value) for value in full_mask)
                pending_doc_ids.extend([doc_id] * len(tokens))

                while len(pending_tokens) >= args.sequence_length and generated_rows < args.rows:
                    block_tokens = pending_tokens[: args.sequence_length]
                    block_full_mask = pending_masks[: args.sequence_length]
                    block_doc_ids = pending_doc_ids[: args.sequence_length]
                    del pending_tokens[: args.sequence_length]
                    del pending_masks[: args.sequence_length]
                    del pending_doc_ids[: args.sequence_length]

                    target_mask = block_full_mask[1:]
                    if not any(target_mask):
                        empty_mask_blocks += 1
                        continue

                    digest = row_digest(block_tokens, target_mask)
                    if digest in seen_hashes:
                        raise ValueError(f"Generated duplicate row hash {digest}")
                    seen_hashes.add(digest)
                    source_doc_count = len(set(block_doc_ids))
                    source_doc_counts.append(source_doc_count)
                    token_counts.update(block_tokens)
                    output_batch.append(
                        {
                            "messages": LOADER_PROMPT,
                            "metadata": {
                                "tokens": block_tokens,
                                "loss_mask": target_mask,
                                "source_doc_count": source_doc_count,
                                "row_hash": digest,
                            },
                        }
                    )
                    generated_rows += 1

                    if len(output_batch) >= args.write_batch_size:
                        writer.write_table(pa.Table.from_pylist(output_batch, schema=schema))
                        output_batch.clear()
                    if generated_rows % 500 == 0:
                        elapsed = time.time() - started
                        print(
                            f"generated={generated_rows}/{args.rows} source_docs={source_docs} "
                            f"elapsed_s={elapsed:.1f}",
                            flush=True,
                        )

                if generated_rows >= args.rows:
                    break
            if generated_rows >= args.rows:
                break

        if generated_rows != args.rows:
            raise RuntimeError(
                f"Source exhausted after producing {generated_rows}/{args.rows} rows "
                f"({len(pending_tokens)} trailing tokens)"
            )
        if output_batch:
            writer.write_table(pa.Table.from_pylist(output_batch, schema=schema))
    finally:
        writer.close()

    validation = validate_dataset(temporary_output, args.rows, args.sequence_length, tokenizer_size)
    loader_validation = validate_slime_loader(
        temporary_output,
        tokenizer,
        args.tokenizer,
        args.sequence_length,
        args.trust_remote_code,
    )

    total_tokens = args.rows * args.sequence_length
    entropy = -sum(
        (count / total_tokens) * math.log2(count / total_tokens) for count in token_counts.values()
    )
    if entropy < args.min_token_entropy_bits:
        raise ValueError(
            f"Token entropy {entropy:.4f} bits is below the required "
            f"{args.min_token_entropy_bits:.4f} bits"
        )
    manifest = {
        "format_version": 2,
        "source": {
            "input": args.input,
            "dataset": args.source_dataset,
            "revision": args.source_revision,
            "sha256": source_hash,
            "files": source_file_inventory,
            "rows": sum(source_parquet.metadata.num_rows for source_parquet in source_parquets),
            "source_docs_consumed": source_docs,
            "source_component_counts": dict(sorted(source_component_counts.items())),
            "malformed_docs_skipped": malformed_docs,
        },
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "tokenizer": {
            "path": args.tokenizer,
            "sha256": tokenizer_hash,
            "files": tokenizer_files,
            "size": tokenizer_size,
            "loss_mask_type": args.loss_mask_type,
            "trust_remote_code": args.trust_remote_code,
        },
        "generation": {
            "sequence_length": args.sequence_length,
            "rows": args.rows,
            "seed": args.seed,
            "shuffle_strategy": "seeded global source-row permutation",
            "source_batch_size": args.source_batch_size,
            "padding_tokens_added": 0,
            "trailing_tokens_dropped": len(pending_tokens),
            "empty_mask_blocks_skipped": empty_mask_blocks,
            "target_mask_alignment": "full_token_mask[1:]",
        },
        "loader_contract": {
            "input_key": "messages",
            "metadata_tokens_key": "tokens",
            "purpose": "The valid text-only conversation lets multimodal checkpoints initialize their processor; the rollout adapter ignores it and consumes pretokenized metadata.",
            "validation": loader_validation,
        },
        "statistics": {
            "total_tokens": total_tokens,
            "supervised_tokens": validation["supervised_tokens"],
            "supervised_token_ratio": validation["supervised_tokens"] / (args.rows * (args.sequence_length - 1)),
            "unique_token_ids": len(token_counts),
            "token_entropy_bits": entropy,
            "source_docs_per_row": {
                "min": min(source_doc_counts),
                "mean": statistics.fmean(source_doc_counts),
                "p50": percentile(source_doc_counts, 0.50),
                "p95": percentile(source_doc_counts, 0.95),
                "max": max(source_doc_counts),
            },
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(temporary_output),
            "bytes": temporary_output.stat().st_size,
        },
        "elapsed_seconds": time.time() - started,
    }
    with open(temporary_manifest, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    os.replace(temporary_output, output)
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
