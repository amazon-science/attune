#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layer 1 of the correctness check: every changed resolved argument, classified by the frozen rule lists.

    layer1.py <base train.log> <candidate train.log> [--frozen-args FILE] [--image REF]   # exit 0 pass, 1 fail

The rules dir (frozen_args.rules_dir(): $AUTOTUNE_RULES_DIR, else rules/ beside this file; rules/README.md gives the
formats) holds numerics_list.yaml, which classes every argument. The setup's directory is the one holding the
frozen-arguments file (--frozen-args, else $AUTOTUNE_FROZEN_ARGS; by convention rules/layer1/<setup>/frozen_args.yaml);
its allowlist.yaml admits same-math swaps and its ct_status.yaml records their component tests. The image's directory,
rules/images/<digest prefix>/ for the image both arms ran (--image; parity_verdict.py reads images.json), holds the
same two files for swaps whose component test passed on that image; they apply to every setup on it. A setup entry
replaces the image's for its argument and is judged by the setup's tests alone. Without either directory no swap is
admitted.

diff() compares two resolved-argument dumps (`  name ....... value`, Megatron arguments.py:1311-1321) and puts every
changed name in one class. Layer 1 FAILS when either dump is empty or the two share less than 90% of their names, a
numeric name changed (unlisted names are numeric), an allowlisted name moved outside its list or without a passed
component test, a derived value moved to something the framework would not derive, or either arm violates the
frozen arguments (frozen_args.py).
"""
import argparse
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))  # campaign.py
import frozen_args  # noqa: E402

CLASSES = ("numeric", "placement", "packing", "allowlisted", "derived", "ignored")
FAILING = ("numeric", "allowlist_violation", "derived_violation")
MIN_SHARED_NAMES = 0.9
# Derived values and the arguments the framework computes them from (numerics_list.yaml `derived`).
DERIVED_FROM = {"padded_vocab_size": ("make_vocab_size_divisible_by", "tensor_model_parallel_size")}


def load_classes(rules_dir=None):
    """name -> class from <rules dir>/numerics_list.yaml."""
    with open(os.path.join(rules_dir or frozen_args.rules_dir(), "numerics_list.yaml")) as f:
        nl = yaml.safe_load(f)
    return {n: k for k in CLASSES for n in nl.get(k) or []}


def classify(name, classes):
    if name in DERIVED_FROM:
        return "derived"
    c = classes.get(name, "numeric")          # unlisted -> numeric (fail closed)
    return "numeric" if c == "derived" else c  # a derived name layer 1 cannot recompute is numeric


def image_dir(image, rules_dir=None):
    """rules/images/<prefix>/ whose name (12-64 hex, a real directory) begins the image's sha256 digest; None for an
    image not pinned by digest or without a directory. More than one match raises ValueError."""
    m = re.search(r"@sha256:([0-9a-f]{64})$", image or "")
    root = os.path.join(rules_dir or frozen_args.rules_dir(), "images")
    if not m or not os.path.isdir(root):
        return None
    hits = [os.path.join(root, n) for n in sorted(os.listdir(root))
            if re.fullmatch(r"[0-9a-f]{12,64}", n) and m.group(1).startswith(n)
            and os.path.isdir(os.path.join(root, n)) and not os.path.islink(os.path.join(root, n))]
    if len(hits) > 1:
        raise ValueError("more than one image directory matches %s: %s" % (m.group(1)[:12], ", ".join(hits)))
    return hits[0] if hits else None


def setup_dir(spec):
    """The setup's layer-1 directory: the frozen-arguments file's directory, or None without one."""
    return os.path.dirname(os.path.abspath(spec["path"])) if spec else None


def _yaml(sdir, name):
    path = os.path.join(sdir, name) if sdir else None
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_allowlist(sdir):
    """arg -> swap of the allowlist.yaml in sdir, a setup or image directory ({} without one)."""
    return {s["arg"]: s for s in _yaml(sdir, "allowlist.yaml").get("swaps") or []}


def passed_tests(sdir):
    """Component tests the ct_status.yaml in sdir marks `pass` (a missing test is not passed)."""
    return {t for t, status in (_yaml(sdir, "ct_status.yaml").get("tests") or {}).items() if status == "pass"}


def swap_rules(sdir, idir):
    """arg -> (swap, tests passed for it). A setup entry replaces the image's and is judged by the setup's tests; an
    image entry by the image's tests minus any the setup marks other than `pass`."""
    setup_not_passed = {t for t, s in (_yaml(sdir, "ct_status.yaml").get("tests") or {}).items() if s != "pass"}
    rules = {a: (s, passed_tests(idir) - setup_not_passed) for a, s in load_allowlist(idir).items()}
    rules.update({a: (s, passed_tests(sdir)) for a, s in load_allowlist(sdir).items()})
    return rules


def _norm(v):
    s = str(v).strip().strip("'\"").lower()
    m = re.match(r"^[a-z_][a-z0-9_]*\.([a-z0-9_]+)$", s)      # AttnBackend.flash -> flash
    return m.group(1) if m else s


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def derived_ok(name, base, cand):
    """A derived value may move only as a consequence of arguments layer 1 already accepted, and must equal what the
    framework would compute. Returns (ok, why).

    padded_vocab_size = ceil(orig_vocab / m) * m with m = make_vocab_size_divisible_by * TP
    (megatron/training/tokenizer/tokenizer.py:110-123). orig_vocab comes from the tokenizer and is NOT a resolved
    argument (`vocab_size` prints None), and the tokenizer arguments are numeric, so orig_vocab is the same in both
    runs and is pinned by the baseline: orig in (padded_b - m_b, padded_b]. The candidate is accepted iff its value is
    a multiple of its own m and admits that same orig, i.e. the two half-open intervals intersect. With m unchanged
    this forces the value to be unchanged; an arbitrary override lands outside the baseline's interval and is refused.
    """
    pb, pc = _int(base.get(name)), _int(cand.get(name))
    if pb is None or pc is None:
        return False, f"{name}: not an integer ({base.get(name)!r} -> {cand.get(name)!r})"
    f = DERIVED_FROM[name]
    mb, mc = 1, 1
    for a in f:
        vb, vc = _int(base.get(a)), _int(cand.get(a))
        if not vb or not vc:
            return False, f"{name}: cannot recompute, {a} missing ({base.get(a)!r} -> {cand.get(a)!r})"
        mb, mc = mb * vb, mc * vc
    if pc % mc:
        return False, f"{name}: {pc} is not a multiple of {' x '.join(f)} = {mc}"
    lo, hi = max(pb - mb, pc - mc), min(pb, pc)          # (lo, hi] = the orig_vocab values both runs allow
    if lo >= hi:
        return False, (f"{name}: {pb} -> {pc} is not what {' x '.join(f)} = {mb} -> {mc} derives "
                       f"(no original vocab size gives both)")
    return True, f"{name}: {pb} -> {pc} follows {mb} -> {mc} (original vocab in ({lo}, {hi}])"


def _swap_class(name, a, b, rules):
    """allowlisted, or allowlist_violation when neither the setup's nor the image's allowlist lists the name, a value
    is outside its list, or its component test has not passed in the source that lists it."""
    if name not in rules:
        return "allowlist_violation"
    swap, passed = rules[name]
    vals, ct = swap["values"], swap.get("component_test", "none")
    if vals != "any" and (_norm(a) not in vals or _norm(b) not in vals):
        return "allowlist_violation"
    # the dispatcher swap needs its component test only when the flex dispatcher is involved
    needs = ct != "none" and not (name == "moe_token_dispatcher_type" and "flex" not in (_norm(a), _norm(b)))
    return "allowlist_violation" if needs and ct not in passed else "allowlisted"


def _unmoved_derived(base, cand, out, why, fixed):
    """A derived value that did not move while an argument it is derived from did must still be the value the
    framework computes, or one of the two logs does not describe its own run. A value the frozen arguments fix (`eq`)
    is set by flag, not derived (Megatron tokenizer.py:104-105 keeps an explicit --padded-vocab-size)."""
    for n, causes in DERIVED_FROM.items():
        if n in out["derived"] or n in out["derived_violation"] or n not in base or n not in cand:
            continue
        if n in fixed and _int(base.get(n)) == _int(cand.get(n)) == _int(fixed[n]):
            why[n] = "%s: fixed at %s by the setup's frozen arguments (an explicit flag, not derived)" % (n, fixed[n])
            continue
        if any(base.get(c) != cand.get(c) for c in causes):
            ok, why[n] = derived_ok(n, base, cand)
            if not ok:
                out["derived_violation"][n] = (base.get(n), cand.get(n))


def diff(base, cand, classes, spec=None, idir=None):
    """{class: {name: (base, cand)}} over every name whose value differs, judged with the setup's and image's rules."""
    rules = swap_rules(setup_dir(spec), idir)
    out = {k: {} for k in CLASSES + ("allowlist_violation", "derived_violation")}
    why = {}
    for n in sorted(set(base) | set(cand)):
        a, b = base.get(n), cand.get(n)
        if a == b:
            continue
        c = classify(n, classes)
        if c == "derived":
            ok, why[n] = derived_ok(n, base, cand)
            c = "derived" if ok else "derived_violation"
        elif c == "allowlisted":
            c = _swap_class(n, a, b, rules)
        out[c][n] = (a, b)
    fixed = {k: r["eq"] for k, r in (spec or {}).get("args", {}).items() if "eq" in r}
    _unmoved_derived(base, cand, out, why, fixed)
    if why:
        out["derived_why"] = why
    return out


def dump_problems(base, cand):
    """Reasons the two resolved-argument dumps cannot be compared."""
    out = ["no resolved-argument dump in the %s log" % n for n, d in (("baseline", base), ("candidate", cand)) if not d]
    if out:
        return out
    shared = len(set(base) & set(cand)) / len(set(base) | set(cand))
    if shared < MIN_SHARED_NAMES:
        only = sorted(set(base) ^ set(cand))
        out.append("the dumps share %.0f%% of their argument names (minimum %.0f%%): different builds or a truncated "
                   "dump; e.g. %s" % (100 * shared, 100 * MIN_SHARED_NAMES, ", ".join(only[:8])))
    return out


def verdict(base, cand, spec=None, image=None):
    """Layer-1 verdict of two resolved-argument dicts; spec: a loaded frozen-arguments file (frozen_args.load) or None;
    image: the image reference, pinned by digest, that both arms ran (None: no image rules)."""
    errors = dump_problems(base, cand)
    idir = None
    try:
        idir = image_dir(image)
        changes = diff(base, cand, load_classes(), spec, idir)
    except (OSError, ValueError, yaml.YAMLError, KeyError, TypeError, AttributeError) as e:
        changes = {}
        errors.append("cannot classify arguments with the rules in %s: %s: %s"
                      % (frozen_args.rules_dir(), type(e).__name__, e))
    frozen = {}
    if spec:
        frozen = {"base": frozen_args.check_resolved(spec, base),
                  "win": frozen_args.check_resolved(spec, cand, baseline=base)}
    ok = not errors and not any(changes.get(k) for k in FAILING) and not any(frozen.values())
    return {"L1": ok, "errors": errors, "changes": changes, "frozen": frozen, "setup_dir": setup_dir(spec),
            "image_dir": idir}


def main(argv=None):
    import campaign

    campaign.require_settings()  # the campaign's config.env decides its settings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_log")
    ap.add_argument("cand_log")
    ap.add_argument("--frozen-args", help="frozen-arguments file (default: $%s)" % frozen_args.ENV)
    ap.add_argument("--image", help="the image both arms ran, pinned by digest (repo@sha256:...)")
    a = ap.parse_args(argv)
    try:
        spec = frozen_args.load(a.frozen_args)
    except frozen_args.SpecError as e:
        print(json.dumps({"L1": False, "errors": [str(e)]}, indent=1))
        return 1
    r = verdict(frozen_args.resolved_args(a.base_log), frozen_args.resolved_args(a.cand_log), spec, a.image)
    print(json.dumps(r, indent=1, default=str))
    return 0 if r["L1"] else 1


if __name__ == "__main__":
    sys.exit(main())
