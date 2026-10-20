#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token-length distribution of a training dataset and the micro-batch arithmetic it implies.

    analyze_token_distribution.py <dataset> [--gbs N] [--steps N] [--max-tokens-per-gpu MT[,MT...]] [--dp N]
        [--cp N] [--pp N --vpp N] [--vocab-size V] [--tokenizer DIR] [--loss-mask-type TYPE] [--input-key KEY]
        [--sample-n N] [--seed S] [--format markdown|json]
    analyze_token_distribution.py --render <record.json>

<dataset> is a Parquet or JSONL file. Pre-tokenized rows are counted exactly, over the whole file by default: a
`tokens` or `input_ids` column, or that field of the `metadata` struct the skill's row builder writes (or of a
`metadata` JSON string), with a `loss_mask` beside it when there is one. Other rows are rendered with a local tokenizer
directory (--tokenizer; no network access): `messages` through the tokenizer's chat template, or slime's loss-mask
generator with --loss-mask-type (the exact SFT tokens and mask, needs slime on PYTHONPATH; pass the recipe's value for
SFT rows, since the chat template alone can differ from them), and `text` through the tokenizer alone. A `tools`
column (--tool-key, slime's default) goes to the template, a JSON string parsed as slime parses it. Rendered rows are a
seeded sample of 10,000 by default. Images and video are not counted. For an RL prompt set pass --rl-prompts: prompts
render with the generation prompt, and since training sees prompt + response their lengths are a lower bound
(--max-response-len adds the response cap to every row in the implications).

Reported: rows, total tokens, min / mean / p50 / p90 / p99 / max (nearest rank), the share of rows and of tokens in
rows of at least 4K/8K/16K/32K tokens, whether every row has the same length, the supervised share when a loss mask
exists, and the regime: fixed-length (one length), short variable (longest row at most 8,192), long-context (median at
least 16,384) or long-tail variable (the rest). Implications: the tokens per GPU a micro-batch needs to hold the longest
row at CP 1/2/4/8 (with a vocabulary size, from --vocab-size or the tokenizer, its fp32 logits at TP1), and over the
run's first --steps steps of --gbs rows in file order, the tokens per step and the micro-batches slime's dynamic
batching makes at each --max-tokens-per-gpu (first fit into micro-batches of mt x CP tokens, then split to a multiple
of DP, or of DP x the micro-batch group with virtual pipeline stages, as slime's dp_schedule.py; --balance-by-flops
packing is not modeled). --gbs and --steps default to $AUTOTUNE_GBS and $AUTOTUNE_NUM_STEPS, else to the default
benchmark's 128 and 20 with a warning on stderr and in the output.

Output: the Markdown block for <project_folder>/data_insights.md on stdout, or with --format json the record that
--render turns into the same block. The script needs only the standard library (Python 3.8+), pyarrow for Parquet and
transformers for rendered rows, reads nothing but the dataset and the tokenizer, and writes nothing, so it runs in a
pod reading itself from stdin. Run it in the baseline run's head pod, whose training image has pyarrow, transformers
and slime and mounts the dataset and the tokenizer (KubeRay names it <run_id>-<5>-head-<5>; the script needs about
2 GiB of memory on a 134M-token set):

    kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" exec -i <head pod> -- \\
        python3 - <dataset> --gbs "${AUTOTUNE_GBS:-128}" --steps "${AUTOTUNE_NUM_STEPS:-20}" \\
        --dp <DP> --cp <CP> --max-tokens-per-gpu <mt> --vocab-size <V> --format json \\
        < analyze_token_distribution.py > token_distribution.json
    analyze_token_distribution.py --render token_distribution.json

The log pod ($AUTOTUNE_LOG_POD) takes the same command when its Python has pyarrow (JSON Lines needs none). Otherwise
read the file off it (`kubectl ... exec "$AUTOTUNE_LOG_POD" -- cat <dataset> > <copy>`), check the copy's sha256
against the dataset's manifest, or the parquet_sha256 the launcher recorded (`dataset` in the campaign's
.attune/integrity.json), and run the script on the copy where pyarrow is installed.

Exit status: 0 done, 2 bad arguments or an unreadable dataset.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys

VERSION = 1
THRESHOLDS = (4096, 8192, 16384, 32768)
CP_SIZES = (1, 2, 4, 8)
SHORT_MAX = 8192             # short variable: the longest row is at most this many tokens
LONG_CONTEXT_P50 = 16384     # long-context: the median row is at least this many tokens
DEFAULT_RENDER_SAMPLE = 10000
DEFAULT_MAX_TOKENS = (4096, 8192, 16384, 32768)
DEFAULT_GBS, DEFAULT_STEPS = 128, 20    # the default benchmark's (workload.py)
DEFAULTED = {
    "gbs": "GBS %d is the default benchmark's, used because neither --gbs nor $AUTOTUNE_GBS was set; pass the "
           "run's GBS if it differs" % DEFAULT_GBS,
    "steps": "%d steps is the default benchmark's run length, used because neither --steps nor $AUTOTUNE_NUM_STEPS "
             "was set" % DEFAULT_STEPS}
TOKEN_KEYS = ("tokens", "input_ids")
GIB = 1 << 30


class InputError(Exception):
    """A bad argument or an unreadable dataset (exit 2)."""


# ---- statistics -----------------------------------------------------------------------------------------------------

def percentile(ordered, q):
    """Nearest-rank percentile of an ascending list: the value at rank ceil(q * n)."""
    return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]


def length_stats(lengths):
    ordered = sorted(lengths)
    total = sum(ordered)
    return {"rows": len(ordered), "total_tokens": total, "min": ordered[0], "mean": total / len(ordered),
            "p50": percentile(ordered, 0.50), "p90": percentile(ordered, 0.90), "p99": percentile(ordered, 0.99),
            "max": ordered[-1]}


def at_or_above(lengths, total):
    out = {}
    for t in THRESHOLDS:
        long_rows = [n for n in lengths if n >= t]
        out[str(t)] = {"rows": len(long_rows), "row_share": len(long_rows) / len(lengths),
                       "token_share": sum(long_rows) / total if total else 0.0}
    return out


def classify(stats):
    """(regime, rule) from the length statistics; the boundaries are SHORT_MAX and LONG_CONTEXT_P50."""
    p50, longest = f"{stats['p50']:,}", f"{stats['max']:,}"
    if stats["min"] == stats["max"]:
        return "fixed-length", "every row is %s tokens" % longest
    if stats["max"] <= SHORT_MAX:
        return "short variable", "longest row %s, at most %s" % (longest, f"{SHORT_MAX:,}")
    if stats["p50"] >= LONG_CONTEXT_P50:
        return "long-context", "median %s, at least %s" % (p50, f"{LONG_CONTEXT_P50:,}")
    return "long-tail variable", "median %s below %s, longest row %s above %s" % (
        p50, f"{LONG_CONTEXT_P50:,}", longest, f"{SHORT_MAX:,}")


# ---- slime's dynamic batching (dp_schedule.py: pack first, distribute second) ----------------------------------------

def first_fit(lengths, cap):
    """Token totals of the micro-batches slime's first_fit_pack makes: each row, in order, joins the first
    micro-batch with room for it; a row longer than `cap` sits alone in its own micro-batch. A max-tree over the
    micro-batches' free room finds that first one in O(log n)."""
    size = 1
    while size < len(lengths):
        size *= 2
    room = [-1] * (2 * size)   # room[1] is the most free room of any micro-batch; -1 marks no micro-batch
    sums = []
    for n in lengths:
        if room[1] >= n:
            i = 1
            while i < size:
                i = 2 * i if room[2 * i] >= n else 2 * i + 1
            sums[i - size] += n
        else:
            i = size + len(sums)
            sums.append(n)
        room[i] = cap - sums[i - size]
        i //= 2
        while i:
            room[i] = max(room[2 * i], room[2 * i + 1])
            i //= 2
    return sums


def alignment(dp, pp, vpp, group):
    """(micro-batch group, align_to) of slime's dp_schedule.py: with virtual pipeline stages a step's micro-batch count
    is a multiple of DP x Megatron's microbatch_group_size_per_vp_stage, which defaults to PP; otherwise of DP."""
    g = (group or pp) if vpp > 1 else 1
    return g, dp * g


def aligned_count(k, rows, align):
    """Micro-batches of a step after slime splits the largest to reach a multiple of `align` (at least `align`); None
    when the step has too few rows to split that far (slime asserts)."""
    target = max(-(-k // align) * align, align)
    return target if target <= rows else None


def summary(values):
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)} if values else None


def implications(lengths, args, order, vocab, dataset_rows):
    rows = len(lengths)
    longest = max(lengths)
    floors = {str(cp): -(-longest // cp) for cp in CP_SIZES}
    out = {"longest_row": longest, "max_response_len": args.max_response_len,
           "min_tokens_per_microbatch": floors, "vocab_size": vocab,
           "fp32_logits_gib_tp1": {cp: f * vocab * 4 / GIB for cp, f in floors.items()} if vocab else None}
    n_steps = min(args.steps, rows // args.gbs)
    steps = [lengths[s * args.gbs:(s + 1) * args.gbs] for s in range(n_steps)]
    out["steps"] = {"gbs": args.gbs, "requested": args.steps, "simulated": n_steps, "order": order,
                    "rows_needed": args.steps * args.gbs, "rows_measured": rows, "dataset_rows": dataset_rows,
                    "defaulted": list(args.defaulted),
                    "tokens_per_step": summary([sum(s) for s in steps]),
                    "longest_row_per_step": summary([max(s) for s in steps])}
    mts = args.max_tokens_per_gpu or sorted(set(DEFAULT_MAX_TOKENS) | {1 << max(12, (longest - 1).bit_length())})
    group, align = alignment(args.dp, args.pp, args.vpp, args.microbatch_group_size)
    packing = []
    for mt in mts:
        cap = mt * args.cp
        per_step = [first_fit(s, cap) for s in steps]
        ks = [len(p) for p in per_step]
        aligned = [aligned_count(k, len(s), align) for k, s in zip(ks, steps)]
        packing.append({
            "max_tokens_per_gpu": mt, "cp": args.cp, "dp": args.dp, "pp": args.pp, "vpp": args.vpp,
            "microbatch_group": group, "align_to": align, "cap": cap,
            "rows_above_cap": sum(1 for n in lengths if n > cap),
            "rows_per_microbatch": sum(len(s) for s in steps) / sum(ks) if ks else None,
            "microbatches_per_step": summary(ks),
            "microbatches_per_dp_rank": (summary([a // args.dp for a in aligned])
                                         if aligned and None not in aligned else None),
            "dp_alignment_fails": sum(1 for a in aligned if a is None),
        })
    out["packing"] = packing
    return out


# ---- reading --------------------------------------------------------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_format(path):
    if not os.path.isfile(path):
        raise InputError("no dataset file at %s" % path)
    low = path.lower()
    if low.endswith(".parquet"):
        return "parquet"
    if low.endswith((".jsonl", ".json")):
        return "jsonl"
    raise InputError("%s: expected a .parquet file or a .jsonl (JSON Lines) file" % path)


def sample_indices(n_rows, sample_n, seed):
    """Sorted row indices to measure: every row, or a seeded sample of sample_n of them."""
    if not sample_n or sample_n >= n_rows:
        return None
    return sorted(random.Random(seed).sample(range(n_rows), sample_n))


def _metadata(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


class Fields:
    """Where the rows' tokens, loss mask and raw input live."""

    def __init__(self, names, metadata_names, args, metadata_json=False):
        self.tokens = self.mask = None
        self.metadata_key = args.metadata_key
        self.metadata_json = metadata_json   # a Parquet metadata column of JSON strings, read row by row
        want = (args.tokens_key,) if args.tokens_key else TOKEN_KEYS
        for k in want:
            if k in names:
                self.tokens = ("column", k)
                break
            if k in metadata_names:
                self.tokens = ("metadata", k)
                break
        if args.tokens_key and self.tokens is None:
            raise InputError("no %r column or %s.%s field" % (args.tokens_key, args.metadata_key, args.tokens_key))
        if self.tokens:   # the loss mask lives beside the tokens
            if args.loss_mask_key in (names if self.tokens[0] == "column" else metadata_names):
                self.mask = (self.tokens[0], args.loss_mask_key)
            self.input = None
            return
        keys = (args.input_key,) if args.input_key else ("messages", "text")
        self.input = next((k for k in keys if k in names), None)
        if self.input is None:
            raise InputError("no pre-tokenized field (%s, or %s.<one of them>) and no raw %s column; found %s" % (
                " / ".join(want), args.metadata_key, " / ".join(keys), sorted(names)))
        self.tools = args.tool_key if args.tool_key in names else None

    def label(self, which):
        f = getattr(self, which)
        return None if f is None else (f[1] if f[0] == "column" else "%s.%s" % (self.metadata_key, f[1]))


def read_parquet_pretokenized(pf, fields, keep):
    """(lengths, supervised tokens or None, loss-mask lengths relative to the tokens) over the kept rows."""
    import pyarrow as pa
    import pyarrow.compute as pc

    lengths, supervised, mask_rel, offset = [], 0 if fields.mask else None, set(), 0
    if fields.tokens[0] == "column":
        cols = [fields.tokens[1]] + ([fields.mask[1]] if fields.mask else [])
    else:
        cols = ["%s.%s" % (fields.metadata_key, f[1]) for f in (fields.tokens, fields.mask) if f]
    try:
        batches = pf.iter_batches(columns=cols, batch_size=4096)
        first = next(batches, None)
    except Exception:   # an older pyarrow selects no nested field: read the whole struct
        cols = sorted({c.split(".")[0] for c in cols})
        batches = pf.iter_batches(columns=cols, batch_size=4096)
        first = next(batches, None)
    keep_set = set(keep) if keep is not None else None

    def column(batch, f):
        if f[0] == "column":
            return batch.column(batch.schema.get_field_index(f[1]))
        md = batch.column(batch.schema.get_field_index(fields.metadata_key))
        return pc.struct_field(md, [md.type.get_field_index(f[1])])

    batch = first
    while batch is not None:
        toks = column(batch, fields.tokens)
        n = pc.list_value_length(toks)
        masks = column(batch, fields.mask) if fields.mask else None
        if keep_set is not None:
            idx = pa.array([i - offset for i in range(offset, offset + batch.num_rows) if i in keep_set],
                           type=pa.int64())
            n = n.take(idx)
            masks = masks.take(idx) if masks is not None else None
        if n.null_count:
            raise InputError("a row has no %s" % fields.label("tokens"))
        nl = n.to_pylist()
        lengths += nl
        if masks is not None:
            if masks.null_count:
                raise InputError("a row has no %s" % fields.label("mask"))
            supervised += int(pc.sum(pc.list_flatten(masks)).as_py() or 0)
            mask_rel |= {m - t for m, t in zip(pc.list_value_length(masks).to_pylist(), nl)}
        offset += batch.num_rows
        batch = next(batches, None)
    return lengths, supervised, mask_rel


def read_parquet_raw(pf, fields, keep):
    """(values, tools) of the kept rows' raw input."""
    import pyarrow as pa

    cols = [fields.input] + ([fields.tools] if fields.tools else [])
    keep_set = set(keep) if keep is not None else None
    values, tools, offset = [], [], 0
    for batch in pf.iter_batches(columns=cols, batch_size=1024):
        rows = batch.num_rows
        if keep_set is not None:
            batch = batch.take(pa.array([i - offset for i in range(offset, offset + rows) if i in keep_set],
                                        type=pa.int64()))
        values += batch.column(0).to_pylist()
        tools += batch.column(1).to_pylist() if fields.tools else [None] * batch.num_rows
        offset += rows
    return values, tools


def parquet_metadata_records(pf, key):
    """{key: value} records of a Parquet metadata column, for read_records."""
    for batch in pf.iter_batches(columns=[key], batch_size=4096):
        for value in batch.column(0).to_pylist():
            yield {key: value}


def parquet_rows(path, args):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise InputError("reading Parquet needs pyarrow: run the script where it is installed (the training "
                         "image has it), or on a copy of the file") from None
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        raise InputError("%s is not a readable Parquet file: %s" % (path, e)) from None
    schema = pf.schema_arrow
    names = set(schema.names)
    md_names, md_json = set(), False
    if args.metadata_key in names:
        t = schema.field(args.metadata_key).type
        if pa.types.is_struct(t):
            md_names = {t.field(i).name for i in range(t.num_fields)}
        elif pa.types.is_string(t) or pa.types.is_large_string(t):   # the first row names the fields, as for JSONL
            first = next(parquet_metadata_records(pf, args.metadata_key), {})
            md_names = set(_metadata(first.get(args.metadata_key)) or {})
            md_json = bool(md_names)
    fields = Fields(names, md_names, args, metadata_json=md_json)
    n_rows = pf.metadata.num_rows
    return pf, fields, n_rows


def jsonl_records(path):
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as e:
                    raise InputError("%s line %d is not a JSON object (JSON Lines expected): %s" % (path, no, e)) \
                        from None


def jsonl_rows(path, args):
    first = next(jsonl_records(path), None)
    if not isinstance(first, dict):
        raise InputError("%s holds no JSON object rows" % path)
    md = _metadata(first.get(args.metadata_key)) or {}
    fields = Fields(set(first), set(md), args)
    if fields.input:   # JSON Lines rows carry their own keys: any row may hold tools
        fields.tools = args.tool_key
    n_rows = sum(1 for _ in jsonl_records(path))
    return fields, n_rows


def read_records(records, fields, keep):
    """(lengths, supervised, mask lengths, raw values, tools) of the kept rows of an iterator of dict records."""
    keep_set = set(keep) if keep is not None else None
    lengths, supervised, mask_rel, values, tools = [], 0 if fields.mask else None, set(), [], []
    for i, rec in enumerate(records):
        if keep_set is not None and i not in keep_set:
            continue
        if fields.tokens is None:
            values.append(rec.get(fields.input))
            tools.append(rec.get(fields.tools) if fields.tools else None)
            continue

        def get(f):
            src = rec if f[0] == "column" else (_metadata(rec.get(fields.metadata_key)) or {})
            v = src.get(f[1])
            if not isinstance(v, list):
                raise InputError("row %d: %s is not a list" % (i, fields.label("tokens" if f is fields.tokens
                                                                                  else "mask")))
            return v
        toks = get(fields.tokens)
        lengths.append(len(toks))
        if fields.mask:
            m = get(fields.mask)
            supervised += sum(int(v) for v in m)
            mask_rel.add(len(m) - len(toks))
    return lengths, supervised, mask_rel, values, tools


# ---- rendering raw rows ---------------------------------------------------------------------------------------------

def load_tokenizer(path, trust_remote_code):
    if not path:
        raise InputError("the rows are not pre-tokenized: pass --tokenizer <the model's local tokenizer directory>")
    if not os.path.isdir(path):
        raise InputError("--tokenizer %s is not a directory: pass the model's local Hugging Face checkpoint or "
                         "tokenizer directory (the script makes no network requests)" % path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise InputError("rendering raw rows needs transformers: %s" % e) from None
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, local_files_only=True)
    except Exception as e:
        raise InputError("cannot load the tokenizer at %s: %s: %s" % (path, type(e).__name__, str(e)[:300])) from None


def vocab_size(tokenizer):
    try:
        return len(tokenizer)
    except TypeError:
        return getattr(tokenizer, "vocab_size", None)


def mask_generator(tokenizer, loss_mask_type):
    try:
        from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    except ImportError:
        try:
            from miles.utils.mask_utils import MultiTurnLossMaskGenerator
        except ImportError:
            raise InputError("--loss-mask-type needs slime (or miles) on PYTHONPATH") from None
    return MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=loss_mask_type)


def render_rows(values, tools, tokenizer, args):
    """(lengths, supervised tokens or None, errors) of raw rows: messages through the chat template or the loss-mask
    generator, text through the tokenizer."""
    gen = mask_generator(tokenizer, args.loss_mask_type) if args.loss_mask_type else None
    lengths, supervised, errors = [], 0 if gen else None, []
    for i, (value, tool) in enumerate(zip(values, tools)):
        try:
            sup = 0
            if isinstance(tool, str):   # slime's Dataset parses a JSON string and requires a list
                tool = json.loads(tool)
            if tool is not None and not isinstance(tool, list):
                raise ValueError("tools must be a list, got %s" % type(tool).__name__)
            if isinstance(value, str) and not gen:
                n = len(tokenizer(value, add_special_tokens=False)["input_ids"])
            elif not isinstance(value, list):
                raise ValueError("expected a list of messages%s, got %s" % ("" if gen else " or a text",
                                                                          type(value).__name__))
            elif gen:
                toks, mask = gen.get_loss_mask(value, tools=tool)
                n, sup = len(toks), sum(int(v) for v in mask)
            else:
                text = tokenizer.apply_chat_template(value, tools=tool, tokenize=False,
                                                     add_generation_prompt=args.rl_prompts)
                n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception as e:   # an unrenderable row is reported and left out
            errors.append("row %d: %s: %s" % (i, type(e).__name__, str(e)[:200]))
            continue
        lengths.append(n)
        if gen:
            supervised += sup
    return lengths, supervised, errors


# ---- the record -----------------------------------------------------------------------------------------------------

def mask_length(rel):
    """How long each row's loss mask is against its tokens: "tokens - 1" (the target mask the skill's rows store),
    "tokens", "other", "mixed", or None without a mask."""
    if not rel:
        return None
    if len(rel) > 1:
        return "mixed"
    return {-1: "tokens - 1", 0: "tokens"}.get(next(iter(rel)), "other")


def analyze(args):
    fmt = dataset_format(args.dataset)
    if fmt == "parquet":
        pf, fields, n_rows = parquet_rows(args.dataset, args)
    else:
        fields, n_rows = jsonl_rows(args.dataset, args)
    if n_rows == 0:
        raise InputError("%s holds no rows" % args.dataset)
    pretokenized = fields.tokens is not None
    sample_n = args.sample_n if args.sample_n is not None else (0 if pretokenized else DEFAULT_RENDER_SAMPLE)
    keep = sample_indices(n_rows, sample_n, args.seed)
    tokenizer, errors, mask_rel = None, [], set()
    if not pretokenized or args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer, args.trust_remote_code)
    try:
        if pretokenized and fmt == "parquet" and fields.tokens[0] == "metadata" and fields.metadata_json:
            lengths, supervised, mask_rel, _, _ = read_records(parquet_metadata_records(pf, fields.metadata_key),
                                                               fields, keep)
        elif pretokenized and fmt == "parquet":
            lengths, supervised, mask_rel = read_parquet_pretokenized(pf, fields, keep)
        elif pretokenized:
            lengths, supervised, mask_rel, _, _ = read_records(jsonl_records(args.dataset), fields, keep)
        elif fmt == "parquet":
            values, tools = read_parquet_raw(pf, fields, keep)
        else:
            _, _, _, values, tools = read_records(jsonl_records(args.dataset), fields, keep)
    except InputError:
        raise
    except Exception as e:   # a column of the wrong type, a damaged file
        raise InputError("cannot read %s: %s: %s" % (args.dataset, type(e).__name__, str(e)[:300])) from None
    if not pretokenized:
        lengths, supervised, errors = render_rows(values, tools, tokenizer, args)
    if not lengths:
        raise InputError("no row could be measured (%d errors; first: %s)" % (len(errors), errors[:1]))
    stats = length_stats(lengths)
    if not stats["total_tokens"]:
        raise InputError("every measured row of %s holds 0 tokens" % args.dataset)
    regime, rule = classify(stats)
    vocab = args.vocab_size or (vocab_size(tokenizer) if tokenizer is not None else None)
    impl_lengths = [n + args.max_response_len for n in lengths] if args.max_response_len else lengths
    return {
        "tool": "analyze_token_distribution", "version": VERSION,
        "dataset": {"path": args.dataset, "format": fmt, "rows": n_rows,
                    "sha256": None if args.no_hash else sha256_file(args.dataset)},
        "source": {"pretokenized": pretokenized, "tokens_field": fields.label("tokens"),
                   "loss_mask_field": fields.label("mask") if pretokenized else None,
                   "loss_mask_length": mask_length(mask_rel),
                   "input_key": None if pretokenized else fields.input,
                   "tokenizer": args.tokenizer, "loss_mask_type": None if pretokenized else args.loss_mask_type,
                   "chat_template": (not pretokenized and not args.loss_mask_type),
                   "rl_prompts": args.rl_prompts},
        "sample": {"rows_measured": len(lengths), "full_pass": keep is None, "sample_n": sample_n or None,
                   "seed": args.seed if keep is not None else None, "render_errors": len(errors),
                   "first_errors": errors[:5]},
        "stats": stats,
        "fixed_length": stats["min"] == stats["max"],
        "at_or_above": at_or_above(lengths, stats["total_tokens"]),
        "supervised": (None if supervised is None else
                       {"tokens": supervised, "share": supervised / stats["total_tokens"]}),
        "regime": {"name": regime, "rule": rule},
        "implications": implications(impl_lengths, args, "file" if keep is None else "sample", vocab, n_rows),
    }


# ---- Markdown -------------------------------------------------------------------------------------------------------

def _n(v):
    """A count with thousands separators; a float to one decimal, without a trailing .0."""
    if v is None:
        return "-"
    if isinstance(v, float):
        v = round(v, 1)
        return f"{int(v):,}" if v == int(v) else f"{v:,.1f}"
    return f"{v:,}"


def _pct(v):
    return f"{100 * v:.1f}%"


def _range(s):
    return "-" if not s else "%s (%s-%s)" % (_n(s["mean"]), _n(s["min"]), _n(s["max"]))


def render(rec):
    ds, src, smp, st, imp = rec["dataset"], rec["source"], rec["sample"], rec["stats"], rec["implications"]
    if src["pretokenized"]:
        how = "pre-tokenized `%s`" % src["tokens_field"]
        if src["loss_mask_field"]:
            how += ", loss mask `%s` (%s entries per row)" % (src["loss_mask_field"], src["loss_mask_length"])
    else:
        how = "`%s` rendered with the tokenizer at `%s`" % (src["input_key"], src["tokenizer"])
        how += (" and slime's `%s` loss mask" % src["loss_mask_type"] if src["loss_mask_type"]
                else " and its chat template" if src["input_key"] != "text" else "")
    scope = ("every row" if smp["full_pass"] else
             "a sample of %s rows, seed %s" % (_n(smp["rows_measured"] + smp["render_errors"]), smp["seed"]))
    out = ["## Token-length distribution", "",
           "Dataset `%s` (%s, %s rows%s): %s; %s." % (
               ds["path"], ds["format"], _n(ds["rows"]), ", sha256 `%s`" % ds["sha256"] if ds["sha256"] else "",
               how, scope)]
    if smp["render_errors"]:
        out.append("%s rows could not be rendered and are left out (first: %s)." % (
            _n(smp["render_errors"]), smp["first_errors"][0]))
    if src["chat_template"] and src["input_key"] != "text" and not src["rl_prompts"]:
        out.append("The chat template alone can differ from the SFT tokens slime trains on; `--loss-mask-type` with "
                   "the recipe's value renders those exactly.")
    out += ["", "| rows | total tokens | min | mean | p50 | p90 | p99 | max |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| %s |" % " | ".join(_n(st[k]) for k in ("rows", "total_tokens", "min", "mean", "p50", "p90", "p99",
                                                      "max")), "",
            "| rows of at least | %s |" % " | ".join(_n(int(t)) for t in rec["at_or_above"]),
            "|---|" + "---:|" * len(rec["at_or_above"]),
            "| share of rows | %s |" % " | ".join(_pct(v["row_share"]) for v in rec["at_or_above"].values()),
            "| share of tokens | %s |" % " | ".join(_pct(v["token_share"]) for v in rec["at_or_above"].values()), ""]
    out.append("Regime: %s (%s). Fixed-length: %s." % (rec["regime"]["name"], rec["regime"]["rule"],
                                                        "yes" if rec["fixed_length"] else "no"))
    if rec["supervised"]:
        out.append("Supervised tokens: %s, %s of all tokens." % (_n(rec["supervised"]["tokens"]),
                                                                 _pct(rec["supervised"]["share"])))
    if src["rl_prompts"]:
        out.append("These are RL prompts: training sees prompt + response, so every length here is a lower bound; "
                   "take the response length from the first rollouts (`rollout/response_len/mean`).")
    out += ["", "### Implications", ""]
    longest = "longest row (%s tokens" % _n(imp["longest_row"])
    if imp["max_response_len"]:
        longest += ", the longest prompt plus the %s-token response cap" % _n(imp["max_response_len"])
    out.append("The micro-batch that holds the %s) carries at least that row / CP tokens per GPU, whatever "
               "`--max-tokens-per-gpu` says: slime puts a row longer than `--max-tokens-per-gpu` x CP alone in its own "
               "micro-batch. Size activation and logits memory for it:" % longest)
    floors = imp["min_tokens_per_microbatch"]
    out += ["", "| CP | %s |" % " | ".join(floors), "|---|" + "---:|" * len(floors),
            "| tokens per GPU for the longest row | %s |" % " | ".join(_n(v) for v in floors.values())]
    if imp["fp32_logits_gib_tp1"]:
        out.append("| fp32 logits at that size, TP1 (GiB; / TP) | %s |" % " | ".join(
            f"{v:.1f}" for v in imp["fp32_logits_gib_tp1"].values()))
        out.append("")
        out.append("Logits use a vocabulary of %s (tokens x vocabulary x 4 B / TP)." % _n(imp["vocab_size"]))
    stp = imp["steps"]
    out.append("")
    for name in stp.get("defaulted") or ():
        out.append("Warning: %s." % DEFAULTED[name])
    if stp["dataset_rows"] < stp["rows_needed"]:
        out.append("A %d-step run needs %s rows; the dataset has %s, so later steps wrap into the next epoch." % (
            stp["requested"], _n(stp["rows_needed"]), _n(stp["dataset_rows"])))
    if not stp["simulated"]:
        out.append("The %s measured rows hold fewer than one step of %s rows: no per-step figures." % (
            _n(stp["rows_measured"]), _n(stp["gbs"])))
        return "\n".join(out) + "\n"
    rows = stp["simulated"] * stp["gbs"]
    where = ("rows 0-%s of %s in file order, as slime reads them without `--rollout-shuffle`" % (
        _n(rows - 1), _n(stp["dataset_rows"])) if stp["order"] == "file" else
        "%s rows of the sample in file order, an estimate of the run's steps" % _n(rows))
    out.append("Steps 0-%d at GBS %s rows (%s), mean (min-max): tokens per step %s, longest row per step %s." % (
        stp["simulated"] - 1, _n(stp["gbs"]), where, _range(stp["tokens_per_step"]),
        _range(stp["longest_row_per_step"])))
    pk = imp["packing"]
    dp, cp, vpp, group = pk[0]["dp"], pk[0]["cp"], pk[0].get("vpp", 1), pk[0].get("microbatch_group", 1)
    layout, multiple, short = "CP %d, DP %d" % (cp, dp), "DP", "DP"
    if vpp > 1:
        layout += ", %d virtual pipeline stages per rank" % vpp
        multiple = "DP x the micro-batch group %d (%d)" % (group, pk[0]["align_to"])
        short = "DP x %d" % group
    out += ["", "Dynamic batching at %s (first fit in row order into micro-batches of `--max-tokens-per-gpu` x CP "
                "tokens, then split to a multiple of %s, as slime's `dp_schedule.py`; `--balance-by-flops` packs by "
                "FLOPs instead and is not modeled):" % (layout, multiple),
            "", "| `--max-tokens-per-gpu` | rows above mt x CP (all rows) | rows per micro-batch | "
                "micro-batches per step | per DP rank |", "|---:|---:|---:|---:|---:|"]
    for p in pk:
        rank = _range(p["microbatches_per_dp_rank"]) if p["microbatches_per_dp_rank"] else (
            "fails at %d steps (fewer rows than %s)" % (p["dp_alignment_fails"], short))
        per_mb = "-" if p["rows_per_microbatch"] is None else f"{p['rows_per_microbatch']:.2f}"
        out.append("| %s | %s | %s | %s | %s |" % (_n(p["max_tokens_per_gpu"]), _n(p["rows_above_cap"]), per_mb,
                                                    _range(p["microbatches_per_step"]), rank))
    return "\n".join(out) + "\n"


# ---- command line ---------------------------------------------------------------------------------------------------

def _ints(text):
    try:
        vals = [int(v) for v in text.replace(" ", "").split(",") if v]
    except ValueError:
        raise argparse.ArgumentTypeError("expected integers separated by commas, got %r" % text) from None
    if not vals or min(vals) < 1:
        raise argparse.ArgumentTypeError("expected positive integers, got %r" % text)
    return vals


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit() or int(raw) < 1:
        raise InputError("%s=%r is not a positive integer" % (name, raw))
    return int(raw)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="analyze_token_distribution.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", help="Parquet or JSONL dataset")
    ap.add_argument("--render", metavar="JSON", help="print the Markdown block of a saved --format json record")
    ap.add_argument("--format", choices=("markdown", "json"), default="markdown", help="stdout format")
    g = ap.add_argument_group("rows")
    g.add_argument("--tokens-key", help="pre-tokenized field (default: tokens, then input_ids; a column or a "
                                        "field of --metadata-key)")
    g.add_argument("--loss-mask-key", default="loss_mask", help="loss mask beside the tokens (default loss_mask)")
    g.add_argument("--metadata-key", default="metadata", help="struct holding the row builder's fields "
                                                              "(default metadata)")
    g.add_argument("--input-key", help="raw input column (default: messages, then text)")
    g.add_argument("--tool-key", default="tools", help="tools column passed to the chat template, slime's --tool-key "
                                                       "(default tools, as slime's)")
    g.add_argument("--tokenizer", metavar="DIR", help="local tokenizer directory for raw rows (also the vocabulary)")
    g.add_argument("--loss-mask-type", help="render messages with slime's MultiTurnLossMaskGenerator of this type "
                                            "(e.g. qwen3_5): the SFT tokens and mask slime trains on")
    g.add_argument("--trust-remote-code", action="store_true", help="let the tokenizer run its own Python code")
    g.add_argument("--rl-prompts", action="store_true", help="RL prompt set: render with the generation prompt; "
                                                             "lengths are a lower bound")
    g.add_argument("--max-response-len", type=int, default=0, metavar="N",
                   help="RL: add the response cap to every row in the implications (default 0)")
    g.add_argument("--sample-n", type=int, default=None, metavar="N",
                   help="rows to measure, a seeded sample (default: all pre-tokenized rows, %d rendered; 0 = all)"
                        % DEFAULT_RENDER_SAMPLE)
    g.add_argument("--seed", type=int, default=0, help="sample seed (default 0)")
    g.add_argument("--no-hash", action="store_true", help="skip the dataset's sha256")
    g = ap.add_argument_group("implications")
    g.add_argument("--gbs", type=int, default=None,
                   help="rows per step, the run's global batch size (default $AUTOTUNE_GBS, else %d with a warning)"
                        % DEFAULT_GBS)
    g.add_argument("--steps", type=int, default=None,
                   help="steps of the run (default $AUTOTUNE_NUM_STEPS, else %d with a warning)" % DEFAULT_STEPS)
    g.add_argument("--max-tokens-per-gpu", type=_ints, default=None, metavar="MT[,MT...]",
                   help="values to pack at (default 4096,8192,16384,32768 and the power of two holding the longest "
                        "row)")
    g.add_argument("--cp", type=int, default=1, help="context-parallel size of the packing table (default 1)")
    g.add_argument("--dp", type=int, default=1, help="data-parallel size for micro-batches per rank (default 1)")
    g.add_argument("--pp", type=int, default=1, help="pipeline-parallel size, the default micro-batch group with "
                                                     "virtual pipeline stages (default 1)")
    g.add_argument("--vpp", type=int, default=1, help="virtual pipeline stages per pipeline rank; above 1, slime "
                                                      "aligns a step's micro-batches to DP x the group (default 1)")
    g.add_argument("--microbatch-group-size", type=int, default=None, metavar="N",
                   help="the recipe's --microbatch-group-size-per-virtual-pipeline-stage (default --pp, Megatron's "
                        "default)")
    g.add_argument("--vocab-size", type=int, default=None,
                   help="vocabulary for the logits size (default: the tokenizer's, when one is loaded)")
    a = ap.parse_args(argv)
    if bool(a.render) == bool(a.dataset):
        ap.error("pass a dataset, or --render <record.json>")
    a.defaulted = []
    try:
        for name, env, default in (("gbs", "AUTOTUNE_GBS", DEFAULT_GBS), ("steps", "AUTOTUNE_NUM_STEPS",
                                                                          DEFAULT_STEPS)):
            if getattr(a, name) is None:
                value = _env_int(env, None)
                if value is None:
                    value = default
                    a.defaulted.append(name)
                setattr(a, name, value)
    except InputError as e:
        ap.error(str(e))
    for name in ("gbs", "steps", "cp", "dp", "pp", "vpp"):
        if getattr(a, name) < 1:
            ap.error("--%s must be at least 1" % name)
    if a.microbatch_group_size is not None and a.microbatch_group_size < 1:
        ap.error("--microbatch-group-size must be at least 1")
    if a.sample_n is not None and a.sample_n < 0:
        ap.error("--sample-n must be 0 (all rows) or positive")
    if a.max_response_len < 0 or (a.vocab_size is not None and a.vocab_size < 1):
        ap.error("--max-response-len and --vocab-size must be positive")
    if a.loss_mask_type and a.rl_prompts:
        ap.error("--loss-mask-type renders SFT conversations; it does not apply to --rl-prompts")
    return a


def main(argv=None):
    a = parse_args(argv)
    try:
        if a.render:
            try:
                with open(a.render, encoding="utf-8") as f:
                    rec = json.load(f)
            except (OSError, ValueError) as e:
                raise InputError("cannot read the record %s: %s" % (a.render, e)) from None
            if rec.get("tool") != "analyze_token_distribution":
                raise InputError("%s is not an analyze_token_distribution record" % a.render)
            sys.stdout.write(render(rec))
            return 0
        for name in a.defaulted:
            print("WARNING: %s." % DEFAULTED[name], file=sys.stderr)
        rec = analyze(a)
    except InputError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(rec, indent=1) + "\n" if a.format == "json" else render(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
