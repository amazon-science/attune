#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""make_run_yaml.py: clone a baseline RayJob YAML into a new run and edit its training CLI flags.

    make_run_yaml.py --base <base_yaml> --run-id <run_id> --out <output_yaml> \\
        [--set FLAG=VALUE ...] [--unset FLAG ...] [--add FLAG=VALUE ...] [--env NAME=VALUE ...] \\
        [--benchmark] [--with-profiler]

What it changes:
  1. The run id. The base run id is the RayJob's metadata.name; every occurrence (K8s names, labels, paths) becomes
     the new run id. The baseline's ConfigMaps must be named <base>, <base>-params or <base>-scripts.
  2. Ownership. The new run id must match ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$, be at most 47 characters and start with
     $ATTUNE_RUN_PREFIX (default "$AUTOTUNE_USER-"). The RayJob and every ConfigMap get the label
     <$AUTOTUNE_OWNER_LABEL, default owner>=<$AUTOTUNE_USER>, which teardown_run.sh checks before it deletes.
  3. Training flags (--set/--unset/--add) on the `--flag value` words of the embedded shell command, the fixed-shape
     contract of --benchmark (workload.py), and the profiler flags of --with-profiler. A --benchmark run is
     unprofiled unless --with-profiler is given.
  4. The hook chain (hook_chain.py) on every --benchmark or profiled run: the registry's default hooks, plus
     profile_rank when the run is profiled per stage; --benchmark removes parity_probe from $ATTUNE_HOOKS and
     writes AUTOTUNE_SEQ_LEN into the pod env.
  5. Structural env vars (--env NAME=VALUE, only the names env_rules.py classes structural: CUDA_DEVICE_MAX_CONNECTIONS,
     PYTORCH_CUDA_ALLOC_CONF, NCCL_ALGO, NCCL_PROTO, debug, dump and SLIME_PROFILE_* profiler switches) on every pod
     and in runtimeEnvYAML.

The canonical-manifest rule: a run may differ from its baseline only in training CLI flag tokens and those
structural env vars. YAML structure (image, nodeAffinity, volumes, other env vars) is never edited. A run that
violates the frozen arguments ($AUTOTUNE_FROZEN_ARGS) or the fixed GPU count is refused.

Flag semantics:
  --set     Replace `--flag [value ...]`; VALUE may be empty for a boolean flag. Appends the flag when absent.
  --unset   Remove `--flag [value ...]`; the flag must exist.
  --add     Append `--flag value`; the flag must not exist.
Edits are token-level: other flags on the same line are kept, and flags in shell or YAML comments, in embedded
non-shell files (.py, .json, .yaml, ...) and in helper python commands (a preflight, a model-args generator) are
ignored. An appended flag goes on the training command: the one command that launches a training entry point
(train*.py, pretrain*.py, `python -m ...train...`), else the one command carrying the recipe's parallelism and batch
flags; before its `2>&1 | tee` line if it has one, else at its end. No single training command is an error.

Examples:
    make_run_yaml.py --base base.yaml --run-id alice-tune-bs4096 --out alice-tune-bs4096.yaml \\
        --set global-batch-size=4096 --set rollout-batch-size=4096
    make_run_yaml.py --base base.yaml --run-id alice-tune-pp1 --out alice-tune-pp1.yaml \\
        --set pipeline-model-parallel-size=1 --unset num-layers-per-virtual-pipeline-stage --add overlap-param-gather=
"""
import argparse
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "correctness")]
import hook_chain  # noqa: E402

RUN_ID_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
MAX_RUN_ID = 47  # KubeRay names pods <rayjob>-<5>-head-<5>, and pod names stop at 63 characters
LABEL_VALUE_RE = re.compile(r"^[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?$")
LABEL_KEY_RE = re.compile(r"^([a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?/)?[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?$")
CONFIGMAP_SUFFIXES = ("", "-params", "-scripts")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="baseline RayJob YAML to clone")
    ap.add_argument("--run-id", required=True, help="the new run's id (every K8s name and run path)")
    ap.add_argument("--out", required=True, help="output YAML path")
    ap.add_argument("--set", action="append", default=[], metavar="FLAG=VALUE",
                    help="replace `--FLAG VALUE` (empty VALUE for a boolean); appended when absent")
    ap.add_argument("--unset", action="append", default=[], metavar="FLAG", help="remove `--FLAG [VALUE]`")
    ap.add_argument("--add", action="append", default=[], metavar="FLAG=VALUE", help="append a new `--FLAG VALUE`")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                    help="set a structural env var (env_rules.py) on every pod and in runtimeEnvYAML; any other name "
                         "is refused (the canonical-manifest rule)")
    ap.add_argument("--with-profiler", action="store_true",
                    help="enable the PyTorch profiler: target train_overall, active steps START..END-1 (default "
                         "22-23), traces in $AUTOTUNE_EXPERIMENTS_DIR (default /shared/experiments)/<run_id>/profiles; "
                         "needs an image whose slime or miles has profiler support")
    ap.add_argument("--benchmark", action="store_true",
                    help="apply the fixed-shape ranking contract of workload.py: GBS and RBS $AUTOTUNE_GBS (default "
                         "128); --num-rollout $AUTOTUNE_NUM_STEPS (default 20: steps 0-4 warm-up, 5-19 measured), or, with "
                         "--with-profiler, the same window then the profiler's setup step 20, warm-up step 21 and "
                         "--profile-capture recorded steps (default 1: step 22, so 23 steps); --prompt-data $AUTOTUNE_DATASET when set (read through "
                         "slime's global dataset, so a --disable-rollout-global-dataset is removed); AUTOTUNE_SEQ_LEN "
                         "(default 8192) in every pod's env for the rollout adapter. Profiler flags the baseline "
                         "carries are removed unless --with-profiler is given")
    ap.add_argument("--profile-steps", default=None, metavar="START-END",
                    help="end-exclusive profile range for unranked diagnostics (default 22-23 with the default step "
                         "counts). slime's profiler setup lands in step START-2, so START-2 must come after the "
                         "$AUTOTUNE_NUM_STEPS steps of the measured window and the correctness check, and END may not "
                         "exceed --num-rollout; ranked --benchmark runs require the range --profile-capture implies")
    ap.add_argument("--profile-capture", type=int, choices=(1, 2, 3), default=None,
                    help="recorded profiler steps of a profiled run after its setup and warm-up steps (default "
                         "$AUTOTUNE_PROFILE_CAPTURE, else 1); ask for 2-3 when the step-time CV is high or something "
                         "periodic is suspected")
    ap.add_argument("--unchecked-num-rollout", action="store_true",
                    help="profile although the YAML's --num-rollout is not a literal integer (a shell variable, or "
                         "--num-epoch), so the profile range cannot be checked against the run length; you vouch that "
                         "the run reaches END")
    ap.add_argument("--profile-ranks", choices=("stage", "all"), default="stage",
                    help="which ranks write traces on a profiled run. stage (default): one rank per pipeline stage, "
                         "through the hook chain's profile_rank hook; all: stock slime, every rank traces")
    return ap.parse_args(argv)


def owner():
    """(label key, user): $AUTOTUNE_OWNER_LABEL (default owner) and $AUTOTUNE_USER, validated as a K8s label."""
    user = os.environ.get("AUTOTUNE_USER", "").strip()
    key = os.environ.get("AUTOTUNE_OWNER_LABEL", "").strip() or "owner"
    if not user:
        raise RuntimeError("set AUTOTUNE_USER: every run is labelled <AUTOTUNE_OWNER_LABEL>=<AUTOTUNE_USER>")
    if not LABEL_VALUE_RE.match(user):
        raise RuntimeError(f"AUTOTUNE_USER={user!r} is not a valid label value (letters, digits, -_. ; at most 63)")
    if not LABEL_KEY_RE.match(key):
        raise RuntimeError(f"AUTOTUNE_OWNER_LABEL={key!r} is not a valid label key")
    return key, user


def run_prefix():
    return os.environ.get("ATTUNE_RUN_PREFIX", "").strip() or owner()[1] + "-"


def check_run_id(run_id):
    """Refuse a run id that K8s or KubeRay would reject, or that lacks the operator's prefix."""
    if not RUN_ID_RE.match(run_id) or len(run_id) > MAX_RUN_ID:
        raise RuntimeError(f"run_id {run_id!r} must match {RUN_ID_RE.pattern} and have at most {MAX_RUN_ID} "
                           "characters (lowercase letters, digits and '-')")
    prefix = run_prefix()
    if not run_id.startswith(prefix):
        raise RuntimeError(f"run_id {run_id!r} must start with {prefix!r} ($ATTUNE_RUN_PREFIX, default $AUTOTUNE_USER-)")


def stamp_owner(docs):
    """Label the RayJob and every ConfigMap with the owner label (in place)."""
    key, user = owner()
    for d in docs:
        if isinstance(d, dict) and d.get("kind") in ("RayJob", "ConfigMap"):
            meta = d.setdefault("metadata", {})
            meta["labels"] = dict(meta.get("labels") or {}, **{key: user})
    return docs


def infer_base_run_id(yaml_text: str) -> str:
    """The base run id: the RayJob's metadata.name. The YAML may hold that one RayJob and ConfigMaps named <base>,
    <base>-params or <base>-scripts, so the renamed run owns all its objects and teardown_run.sh finds them."""
    import yaml
    docs = [d for d in yaml.safe_load_all(yaml_text) if isinstance(d, dict)]
    jobs = [d for d in docs if d.get("kind") == "RayJob"]
    if len(jobs) != 1 or not (jobs[0].get("metadata") or {}).get("name"):
        raise RuntimeError(f"the base YAML must hold exactly one named RayJob (found {len(jobs)})")
    base = jobs[0]["metadata"]["name"]
    others = [d.get("kind") for d in docs if d.get("kind") not in ("RayJob", "ConfigMap")]
    if others:
        raise RuntimeError(f"the base YAML may hold only the RayJob and its ConfigMaps; found {others}")
    names = [(d.get("metadata") or {}).get("name") for d in docs if d.get("kind") == "ConfigMap"]
    bad = [n for n in names if n not in [base + s for s in CONFIGMAP_SUFFIXES]]
    if bad:
        raise RuntimeError(f"ConfigMaps {bad} must be named {base}, {base}-params or {base}-scripts "
                           "(teardown_run.sh deletes exactly those)")
    return base


def apply_name_suffix(yaml_text: str, base_run_id: str, new_run_id: str) -> str:
    """Replace the base run id everywhere (names, labels, paths): a partial rename gets the RayJob rejected."""
    return yaml_text.replace(base_run_id, new_run_id)


# One shell word of a flag's value: a quoted string, or a run of characters that ends at whitespace, a line
# continuation or a shell operator.
_VAL_TOK = r"""(?:"[^"\n]*"|'[^'\n]*'|[^\s\\|;&<>'"]+)"""
# the profiler flags --with-profiler adds; --benchmark without it removes any the baseline carries
PROFILER_FLAGS = ("use-pytorch-profiler", "profile-target", "profile-step-start", "profile-step-end")


def _in_comment(text: str, pos: int) -> bool:
    """True when pos lies after a `#` that starts a shell or YAML comment on its line."""
    ls = text.rfind("\n", 0, pos) + 1
    return re.search(r"(^|\s)#", text[ls:pos]) is not None


# A ConfigMap value that is a file of another language (the rollout adapter, hook-chain files, configs): its text is
# never part of the training command, so flags it mentions are not edited.
_EMBEDDED_FILE = re.compile(r"^(?P<ind>[ \t]*)[\w.\-]+\.(?:py|json|ya?ml|txt|md):[ \t]*[|>][-+0-9]*[ \t]*\n"
                            r"(?:(?:(?P=ind)[ \t]+[^\n]*|[ \t]*)\n)*", re.M)


def _embedded_spans(text: str):
    return [(m.start(), m.end()) for m in _EMBEDDED_FILE.finditer(text)]


# The training command launches a training entry point; a helper command runs another python program.
_TRAIN_ENTRY = re.compile(r"\b(?:python3?(?:\.\d+)?|torchrun|deepspeed)\b[^|;&]*?"
                          r"(?:\s-m\s+[\w.]*train\w*\b|[\s/](?:pre)?train(?:_\w+)?\.py\b)")
_PROGRAM = re.compile(r"(?:^|[\s;&|(])(?:python3?(?:\.\d+)?|torchrun|deepspeed)\s")
TRAIN_MARKERS = ("tensor-model-parallel-size", "pipeline-model-parallel-size", "global-batch-size",
                 "micro-batch-size", "max-tokens-per-gpu", "num-rollout", "rollout-batch-size")


def _commands(text: str):
    """(lines, [(first, last, joined text)]) of every shell command outside embedded non-shell files and comment
    lines: a run of backslash-continued lines, joined without its continuations and trailing comments."""
    lines, spans, pos, out, i = text.split("\n"), _embedded_spans(text), [0], [], 0
    for line in lines:
        pos.append(pos[-1] + len(line) + 1)
    while i < len(lines):
        if not lines[i].strip() or lines[i].lstrip().startswith("#") or any(a <= pos[i] < b for a, b in spans):
            i += 1
            continue
        j = i
        while j + 1 < len(lines) and lines[j].rstrip().endswith("\\"):
            j += 1
        joined = " ".join(re.sub(r"(^|\s)#.*$", "", ln).rstrip().rstrip("\\") for ln in lines[i:j + 1])
        out.append((i, j, joined))
        i = j + 1
    return lines, out


def _has_marker(joined: str) -> bool:
    return any(re.search(rf"(?<!\S)--{f}(?=[\s=]|$)", joined) for f in TRAIN_MARKERS)


def training_command(text: str, flag: str):
    """(lines, first line, last line) of the training command; RuntimeError unless exactly one command qualifies."""
    lines, cmds = _commands(text)
    hits = [c for c in cmds if _TRAIN_ENTRY.search(c[2])] or [c for c in cmds if _has_marker(c[2])]
    if len(hits) != 1:
        found = "; ".join(repr(lines[c[0]].strip()[:70]) for c in hits) or "none"
        raise RuntimeError(f"--{flag}: cannot identify the single training command (one that launches train*.py or "
                           f"`python -m ...train...`, else one carrying --{' / --'.join(TRAIN_MARKERS[:3])}); "
                           f"found {len(hits)}: {found}")
    if re.match(r"""^\s*[\w.\-]+:\s*["']""", lines[hits[0][0]]):
        raise RuntimeError(f"--{flag}: the training command is inside a quoted YAML string; write that script as a "
                           f"literal block (`key: |`)")
    return lines, hits[0][0], hits[0][1]


def _helper_spans(text: str):
    """Character spans of helper commands: python programs that are neither a training entry point nor carry the
    recipe's training flags (a preflight, a model-args generator); their flags are never edited."""
    lines, cmds = _commands(text)
    pos = [0]
    for line in lines:
        pos.append(pos[-1] + len(line) + 1)
    return [(pos[a], pos[b + 1]) for a, b, j in cmds
            if _PROGRAM.search(j) and not _TRAIN_ENTRY.search(j) and not _has_marker(j)]


def _flag_matches(text: str, flag: str):
    """Every `--flag [value ...]` / `--flag=value` occurrence outside comments, embedded non-shell files and helper
    commands, as regex matches (token-level: other flags on the same line are untouched; all value words up to the
    next flag, a line continuation, an operator or newline)."""
    pat = re.compile(rf"(?<!\S)--{re.escape(flag)}(?:=\S+|(?:[ \t]+(?!--){_VAL_TOK})*)(?=\s|$)")
    spans = _embedded_spans(text) + _helper_spans(text)
    return [m for m in pat.finditer(text) if not _in_comment(text, m.start())
            and not any(a <= m.start() < b for a, b in spans)]


def _set_cli_token(yaml_text: str, flag: str, value: str):
    """Set `--flag [value]` in place, even when several flags share a line. None if the flag is absent."""
    ms = _flag_matches(yaml_text, flag)
    if not ms:
        return None
    if len(ms) > 1:
        raise RuntimeError(f"--{flag}: multiple occurrences, too risky to edit")
    new = f"--{flag} {value}" if value else f"--{flag}"
    return yaml_text[:ms[0].start()] + new + yaml_text[ms[0].end():]


def _append_cli(yaml_text: str, flag: str, value: str) -> str:
    """Append `--flag [value]` to the training command (training_command): before its `2>&1 | tee` line when it has
    one, else at its end (before any pipe or redirect on its last line)."""
    tok = f"--{flag} {value}" if value else f"--{flag}"
    lines, start, j = training_command(yaml_text, flag)
    tee = next((k for k in range(start + 1, j + 1) if re.match(r"^\s*2>&1 \| tee", lines[k])), None)
    if tee is not None:
        lines.insert(tee, re.match(r"^(\s*)", lines[tee]).group(1) + f"{tok} \\")
        return "\n".join(lines)
    op = re.search(r"(?<!\S)(?:\d?>>?|&>|\||;|&&)", lines[j])
    if op:  # keep the pipe/redirect last: the flag goes before it
        lines[j] = lines[j][:op.start()].rstrip() + f" {tok} " + lines[j][op.start():]
        return "\n".join(lines)
    indent = re.match(r"^(\s*)", lines[j]).group(1)
    lines[j] = lines[j].rstrip() + " \\"
    lines.insert(j + 1, f"{indent}{tok}")
    return "\n".join(lines)


def apply_set(yaml_text: str, flag: str, value: str) -> str:
    """Replace `--flag [value ...]` token-level; append the flag when absent (e.g. --set num-rollout=27 on a
    --num-epoch-only base; --num-rollout overrides --num-epoch)."""
    out = _set_cli_token(yaml_text, flag, value)
    return out if out is not None else _append_cli(yaml_text, flag, value)


def apply_unset(yaml_text: str, flag: str) -> str:
    """Remove `--flag [value ...]` token-level; a line left empty is dropped, keeping the continuation intact."""
    ms = _flag_matches(yaml_text, flag)
    if not ms:
        raise RuntimeError(f"--unset {flag}: flag not found in base YAML")
    if len(ms) > 1:
        raise RuntimeError(f"--unset {flag}: multiple occurrences, too risky")
    s, e = ms[0].start(), ms[0].end()
    while e < len(yaml_text) and yaml_text[e] in " \t":
        e += 1
    ls = yaml_text.rfind("\n", 0, s) + 1
    le = yaml_text.find("\n", e)
    le = len(yaml_text) if le == -1 else le
    if (yaml_text[ls:s] + yaml_text[e:le]).strip() not in ("", "\\"):
        return yaml_text[:s] + yaml_text[e:]
    continued = yaml_text[e:le].strip() == "\\"
    head, tail = yaml_text[:ls], yaml_text[le + 1:]
    if not continued:  # the flag ended the command: the line before must not continue into what follows
        head = re.sub(r"[ \t]*\\[ \t]*\n$", "\n", head)
    return head + tail


def apply_add(yaml_text: str, flag: str, value: str) -> str:
    if _flag_matches(yaml_text, flag):
        raise RuntimeError(f"--add {flag}: flag already exists (use --set)")
    return _append_cli(yaml_text, flag, value)


def install_hook_chain(yaml_text: str, run_id: str, enable=(), disable=()):
    """Install the hook chain (hook_chain.py) on every pod: the registry's default hooks, plus `enable` (profile_rank
    when this run is profiled per stage), minus `disable`. Every ranked run carries the chain, profiled or not."""
    import yaml
    docs = [d for d in yaml.safe_load_all(yaml_text) if d is not None]
    info = hook_chain.install(docs, run_id, enable=list(enable), disable=list(disable))
    msg = "hook chain %s in ConfigMap %s, %d containers" % (info["chain_sha256"][:12], info["configmap"], info["containers"])
    if "profile_rank" in enable:
        msg = "profiler ranks: one per pipeline stage (%s)" % msg
    return hook_chain.dump_yaml(docs), msg


def _rayjob_pod_specs(docs):
    for job in (d for d in docs if isinstance(d, dict) and d.get("kind") == "RayJob"):
        rcs = (job.get("spec") or {}).get("rayClusterSpec") or {}
        for g in [rcs.get("headGroupSpec") or {}] + list(rcs.get("workerGroupSpecs") or []):
            pod = (g.get("template") or {}).get("spec")
            if pod:
                yield job, g, pod


def inject_env(yaml_text: str, env: dict) -> str:
    """Set literal env vars on every RayJob container and in runtimeEnvYAML env_vars (when present)."""
    import yaml
    docs = [d for d in yaml.safe_load_all(yaml_text) if d is not None]
    n = 0
    for job, _, pod in _rayjob_pod_specs(docs):
        for c in pod.get("containers") or []:
            cenv = c.get("env") or []
            c["env"] = cenv
            for k, v in env.items():
                e = next((e for e in cenv if e.get("name") == k), None)
                if e is None:
                    cenv.append({"name": k, "value": str(v)})
                else:
                    e.pop("valueFrom", None)
                    e["value"] = str(v)
            n += 1
    for job in (d for d in docs if isinstance(d, dict) and d.get("kind") == "RayJob"):
        spec = job.get("spec") or {}
        if spec.get("runtimeEnvYAML"):
            r = yaml.safe_load(spec["runtimeEnvYAML"]) or {}
            r.setdefault("env_vars", {}).update({k: str(v) for k, v in env.items()})
            spec["runtimeEnvYAML"] = yaml.safe_dump(r, sort_keys=False)
    if n == 0:
        raise RuntimeError("no RayJob containers found to carry the env vars %s" % ", ".join(env))
    return hook_chain.dump_yaml(docs)


def count_gpus(yaml_text: str) -> int:
    """nvidia.com/gpu summed over every RayJob pod group (replicas x containers)."""
    import yaml
    docs = [d for d in yaml.safe_load_all(yaml_text) if d is not None]
    total = 0
    for _, g, pod in _rayjob_pod_specs(docs):
        reps = int(g.get("replicas", 1))
        for c in pod.get("containers") or []:
            r = c.get("resources") or {}
            total += reps * int((r.get("limits") or {}).get("nvidia.com/gpu")
                                or (r.get("requests") or {}).get("nvidia.com/gpu") or 0)
    return total


def fixed_gpu_count(base_count):
    """The GPU count a run must request, or None when count changes are allowed. The count is fixed by default, at
    $AUTOTUNE_NUM_GPUS or else at the base YAML's own count: a tuned recipe runs on the recipe's own GPU count.
    Only AUTOTUNE_ALLOW_GPU_COUNT_CHANGE=1 lifts that, and never together with AUTOTUNE_FIXED_GPUS=1."""
    if os.environ.get("AUTOTUNE_ALLOW_GPU_COUNT_CHANGE") == "1" and os.environ.get("AUTOTUNE_FIXED_GPUS") != "1":
        return None
    want = os.environ.get("AUTOTUNE_NUM_GPUS", "").strip()
    return int(want) if want else base_count


def check_launch_policy(yaml_text: str, base_count=None) -> list:
    """Frozen-argument and fixed-GPU-count checks on a generated run; returns messages (empty = ok)."""
    import frozen_args
    problems = []
    for v in frozen_args.check_manifest(frozen_args.load(), yaml_text):
        problems.append(f"frozen argument {v['arg']}={v['value']!r} ({v['why']}; want {v['want']})")
    want = fixed_gpu_count(base_count)
    if want is not None:
        got = count_gpus(yaml_text)
        if got != want:
            problems.append(f"GPU count is fixed at {want}; this run requests {got} (a count change needs "
                            "AUTOTUNE_ALLOW_GPU_COUNT_CHANGE=1)")
    return problems


def split_kv(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise RuntimeError(f"expected FLAG=VALUE, got {spec!r}")
    flag, value = spec.split("=", 1)
    return flag.strip(), value.strip()


def normalize_blocks(yaml_text: str) -> str:
    """Re-emit the manifest with every multi-line string as a literal block, so flag edits see one script line per
    text line (a quoted scalar escapes newlines and can wrap a flag across lines). A file already in block style
    keeps its content; YAML comments are dropped."""
    import yaml
    docs = [d for d in yaml.safe_load_all(yaml_text) if d is not None]
    needs = re.search(r'^[ \t]*[\w.\-]+:[ \t]*"(?:[^"\\]|\\.)*\\n', yaml_text, re.M)
    return hook_chain.dump_yaml(docs) if needs else yaml_text


def benchmark_sets(text, wl, profiled):
    """(text, --set specs) of the fixed-shape contract: an unprofiled run carries no profiler flag."""
    if not profiled:
        for f in PROFILER_FLAGS:
            if _flag_matches(text, f):
                text = apply_unset(text, f)
    steps = wl["num_steps_profiled"] if profiled else wl["num_steps"]
    sets = [f"global-batch-size={wl['gbs']}", f"rollout-batch-size={wl['gbs']}", f"num-rollout={steps}"]
    if wl["dataset"]:
        sets.append(f"prompt-data={wl['dataset']}")
        if _flag_matches(text, "disable-rollout-global-dataset"):   # the rollout adapter reads the global dataset
            text = apply_unset(text, "disable-rollout-global-dataset")
    return text, sets


def _flag_int(text, flag):
    """The integer value of the single `--flag` occurrence (quotes allowed), or None (absent, repeated, or a shell
    variable)."""
    ms = _flag_matches(text, flag)
    if len(ms) != 1:
        return None
    v = ms[0].group(0)[len(flag) + 2:].lstrip("= \t")
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    return int(v) if v.isdigit() else None


def parse_profile_steps(steps):
    """(start, end) of an end-exclusive --profile-steps START-END."""
    try:
        start, end = (int(x) for x in steps.split("-"))
    except ValueError:
        raise RuntimeError(f"--profile-steps must be START-END (got {steps!r})") from None
    if start < 1 or end <= start:
        raise RuntimeError("--profile-steps requires 1 <= START < END")
    return start, end


def profiler_flags(text, run_id, steps, wl, prof, unchecked=False):
    """(flags, trace dir, (start, end)) of --with-profiler. A tensorboard dir the baseline already sets per run (it names the run id
    or $RUN_ID) is kept, so traces of different runs never share a directory. `prof` is workload.profiled() on a
    --benchmark render, else None; `unchecked` accepts a --num-rollout that is not a literal integer."""
    start, end = parse_profile_steps(steps)
    if prof and (start, end) != (prof["profile_start"], prof["profile_end"]):
        raise RuntimeError(f"ranked --benchmark runs require --profile-steps {prof['profile_start']}-{prof['profile_end']}")
    if start - 2 < wl["num_steps"]:   # slime steps the profiler inside the step timer: setup in START-2, warm-up START-1
        raise RuntimeError(f"--profile-steps {start}-{end}: the profiler's setup step {start - 2} falls inside steps "
                           f"0-{wl['num_steps'] - 1} (measured window and correctness check); use START >= "
                           f"{wl['num_steps'] + 2}")
    n_roll = _flag_int(text, "num-rollout")
    if n_roll is None and not unchecked:
        raise RuntimeError(f"--profile-steps {start}-{end}: the YAML's --num-rollout is not one literal integer (a shell "
                           "variable, repeated, or absent under --num-epoch), so the range cannot be checked against "
                           "the run length; set it with --set num-rollout=N, or pass --unchecked-num-rollout")
    if n_roll is not None and end > n_roll:
        raise RuntimeError(f"--profile-steps {start}-{end} needs --num-rollout >= {end} (the run has {n_roll} steps)")
    exp_root = os.environ.get("AUTOTUNE_EXPERIMENTS_DIR") or "/shared/experiments"
    trace_dir = f"{exp_root}/{run_id}/profiles"
    tb = [m.group(0).split(None, 1)[-1].split("=", 1)[-1] for m in _flag_matches(text, "tensorboard-dir")]
    if len(tb) == 1 and (run_id in tb[0] or re.search(r"\$\{?(RUN_ID|PROF_DIR)\b", tb[0])):
        trace_dir = tb[0]
    # train_overall: TrainProfiler steps it once per rollout; train_actor is not wired in every slime
    return [("use-pytorch-profiler", ""), ("profile-target", "train_overall"), ("profile-step-start", str(start)),
            ("profile-step-end", str(end)), ("tensorboard-dir", trace_dir)], trace_dir, (start, end)


def apply_env(text, specs):
    import env_rules
    envs = dict(kv.split("=", 1) if "=" in kv else (kv, "") for kv in specs)
    bad = ["%s (%s)" % (k, env_rules.classify(k)) for k in envs if not env_rules.may_differ(k)]
    if bad:
        raise RuntimeError("--env: only the structural env vars of env_rules.py may be set: %s" % ", ".join(bad))
    return inject_env(text, envs)


def stamp_owner_text(text):
    import yaml
    return hook_chain.dump_yaml(stamp_owner([d for d in yaml.safe_load_all(text) if d is not None]))


def build(args, base_text, wl, prof=None):
    """(output YAML, summary lines). `prof` is workload.profiled() when the run is profiled."""
    notes = []
    sets = list(args.set)
    if args.benchmark:
        base_text, contract = benchmark_sets(base_text, dict(wl, **(prof or {})), args.with_profiler)
        sets += contract
    base_run_id = infer_base_run_id(base_text)
    text = apply_name_suffix(base_text, base_run_id, args.run_id)
    for spec in sets:
        text = apply_set(text, *split_kv(spec))
    for flag in args.unset:
        text = apply_unset(text, flag.strip())
    for spec in args.add:
        text = apply_add(text, *split_kv(spec))
    if args.with_profiler:
        flags, trace_dir, (start, end) = profiler_flags(text, args.run_id, args.profile_steps, wl,
                                                        prof if args.benchmark else None, args.unchecked_num_rollout)
        for flag, value in flags:
            text = apply_set(text, flag, value)
        notes.append(f"profiler on, setup step {start - 2}, warm-up step {start - 1}, active steps [{start},{end}), "
                     f"trace -> {trace_dir}")
    else:
        left = [f for f in PROFILER_FLAGS if _flag_matches(text, f)]
        if left:
            raise RuntimeError("profiler flags without --with-profiler: %s" % ", ".join("--" + f for f in left))
    if args.benchmark:
        text = inject_env(text, {"AUTOTUNE_SEQ_LEN": wl["seq_len"]})
    if args.env:
        text = apply_env(text, args.env)
    problems = check_launch_policy(text, count_gpus(base_text))
    if problems:
        raise RuntimeError("launch policy: " + "; ".join(problems))
    profiled = bool(_flag_matches(text, "use-pytorch-profiler"))
    enable = ["profile_rank"] if profiled and args.profile_ranks == "stage" else []
    if args.benchmark or enable:
        # a benchmark run never loads the parity probe, even when its baseline was a probed arm
        text, msg = install_hook_chain(text, args.run_id, enable, ["parity_probe"] if args.benchmark else [])
        notes.append(msg)
        if args.benchmark and re.search(r"ATTUNE_HOOKS=\S*parity_probe", text):
            raise RuntimeError("the baseline's script exports ATTUNE_HOOKS with parity_probe; a benchmark run is "
                               "unprobed")
    notes.insert(0, f"base_run_id {base_run_id}, diffs applied: {len(sets)} set, {len(args.unset)} unset, "
                    f"{len(args.add)} add")
    return stamp_owner_text(text), notes


def main(argv=None):
    import campaign

    campaign.require_settings()  # the campaign's config.env decides its settings
    args = parse_args(argv)
    check_run_id(args.run_id)
    import workload
    wl = workload.resolve()
    for given, name in ((args.profile_capture is not None, "--profile-capture"),
                        (args.profile_steps is not None, "--profile-steps"),
                        (args.unchecked_num_rollout, "--unchecked-num-rollout")):
        if given and not args.with_profiler:
            raise RuntimeError(f"{name} needs --with-profiler")
    prof = None
    if args.with_profiler and args.profile_steps is None:
        # the default range: K from --profile-capture, else $AUTOTUNE_PROFILE_CAPTURE (validated only here)
        prof = workload.profiled(wl, args.profile_capture)
        args.profile_steps = f"{prof['profile_start']}-{prof['profile_end']}"
    elif args.with_profiler:
        start, end = parse_profile_steps(args.profile_steps)
        if args.profile_capture is not None and args.profile_capture != end - start:
            raise RuntimeError(f"--profile-capture {args.profile_capture} disagrees with --profile-steps "
                               f"{args.profile_steps} ({end - start} recorded steps); pass one of them")
        if args.benchmark:   # a ranked run's explicit range fixes K
            try:
                prof = workload.profiled(wl, end - start)
            except ValueError:
                raise RuntimeError(f"ranked --benchmark runs record 1-{workload.MAX_PROFILE_CAPTURE} steps; "
                                   f"--profile-steps {args.profile_steps} records {end - start}") from None
    out_text, notes = build(args, normalize_blocks(Path(args.base).read_text()), wl, prof)
    Path(args.out).write_text(out_text)
    print(f"Wrote {args.out} (run {args.run_id}, owner label {'='.join(owner())})")
    if args.benchmark:
        steps = prof["num_steps_profiled"] if prof else wl["num_steps"]
        notes.append(f"workload: GBS {wl['gbs']}, rows of {wl['seq_len']} tokens, {steps} steps "
                     f"({'profiled' if args.with_profiler else 'unprofiled'}; window {wl['window_start']}-"
                     f"{wl['window_end']}, target step {steps - 1}), dataset {wl['dataset'] or '(baseline)'}")
    for n in notes:
        print(f"  {n}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
