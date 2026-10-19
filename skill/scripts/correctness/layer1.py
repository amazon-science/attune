#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layer 1 of the correctness check: every changed resolved argument, classified by the frozen rule lists.

    layer1.py <base train.log> <candidate train.log> [--frozen-args FILE] [--image REF]   # exit 0 pass, 1 fail

The rules dir (frozen_args.rules_dir(): $AUTOTUNE_RULES_DIR, else rules/ beside this file; rules/README.md gives the
formats) holds numerics_list.yaml, which classes every argument. The setup's directory is the one holding the
frozen-arguments file (--frozen-args, else $AUTOTUNE_FROZEN_ARGS; by convention rules/layer1/<setup>/frozen_args.yaml);
its allowlist.yaml admits same-math swaps, its ct_status.yaml records their component tests and its defects.yaml lists
known stack defects. The image's directory, rules/images/<digest prefix>/ for the image both arms ran (--image;
parity_verdict.py reads images.json), holds the same three files for swaps tested and defects found on that image; they
apply to every setup on it. A setup swap entry replaces the image's for its argument and is judged by the setup's tests
alone; the defects of both directories apply. Without either directory no swap is admitted.

diff() compares two resolved-argument dumps (`  name ....... value`, Megatron arguments.py:1311-1321) and puts every
changed name in one class. Layer 1 FAILS when either dump is empty or the two share less than 90% of their names, an
objective name changed, the candidate is in a known defect state the baseline is not in (whichever of the entry's
arguments changed), an allowlisted swap's component test is recorded `fail` in the setup's or the image's results, a
derived value printed as integers moved to something the framework would not derive, either arm violates the frozen
arguments (frozen_args.py), or a rules file is unreadable or malformed. It accepts placement, packing, ignored and
derived changes and tested allowlisted swaps. Every other change is UNKNOWN and passes, with its reason, for the step-0
check to judge and a human to review: a numeric name, a name the lists do not know, a name only one dump has, a derived
value it cannot recompute, and an allowlisted name no allowlist lists, moved outside its values or whose component test
has not passed.
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

CLASSES = ("objective", "numeric", "placement", "packing", "allowlisted", "derived", "ignored")
FAILING = ("objective", "defect", "ct_failed", "derived_violation")
ACCEPTED = ("placement", "packing", "allowlisted", "derived", "ignored")
MIN_SHARED_NAMES = 0.9
# Derived values and the arguments the framework computes them from (numerics_list.yaml `derived`).
DERIVED_FROM = {"padded_vocab_size": ("make_vocab_size_divisible_by", "tensor_model_parallel_size")}


def load_classes(rules_dir=None):
    """name -> class from <rules dir>/numerics_list.yaml; an unknown top-level key or a name listed in two classes
    raises ValueError."""
    with open(os.path.join(rules_dir or frozen_args.rules_dir(), "numerics_list.yaml")) as f:
        nl = yaml.safe_load(f)
    if not isinstance(nl, dict):
        raise ValueError("numerics_list.yaml: expected a mapping at the top level")
    unknown = sorted(set(nl) - {"version", *CLASSES}, key=str)
    if unknown:
        raise ValueError("numerics_list.yaml: unknown top-level key %s (allowed: version, %s)"
                         % (", ".join(map(str, unknown)), ", ".join(CLASSES)))
    out = {}
    for k in CLASSES:
        for n in nl.get(k) or []:
            if n in out:
                raise ValueError("numerics_list.yaml lists %s in both %s and %s" % (n, out[n], k))
            out[n] = k
    return out


def classify(name, classes):
    if name in DERIVED_FROM:
        return "derived"
    c = classes.get(name, "unlisted")
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


RULE_KEYS = {"allowlist.yaml": ("version", "swaps"), "ct_status.yaml": ("version", "tests"),
             "defects.yaml": ("version", "defects")}


def _yaml(sdir, name):
    """The mapping in sdir/name ({} when absent). A path that exists but is not a readable file (a directory, a broken
    symlink), a non-mapping and an unknown top-level key raise ValueError."""
    path = os.path.join(sdir, name) if sdir else None
    if not path or not os.path.lexists(path):
        return {}
    if not os.path.isfile(path):
        raise ValueError("%s exists but is not a readable file" % path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("%s: expected a mapping at the top level" % path)
    unknown = sorted(set(data) - set(RULE_KEYS.get(name, data)))
    if unknown:
        raise ValueError("%s: unknown top-level key %s (allowed: %s)" % (path, ", ".join(map(str, unknown)),
                                                                         ", ".join(RULE_KEYS[name])))
    return data


def _any(vals):
    return isinstance(vals, str) and vals.strip().lower() == "any"


def _values(vals):
    """`any`, or the set of normalized values of an allowlist or defect entry (YAML booleans and null included)."""
    return "any" if _any(vals) else {_norm(v) for v in vals}


def load_allowlist(sdir):
    """arg -> swap of the allowlist.yaml in sdir, a setup or image directory ({} without one); a swap without an
    argument name or with `values` neither a list nor `any` raises ValueError."""
    path = os.path.join(sdir, "allowlist.yaml") if sdir else None
    swaps = _yaml(sdir, "allowlist.yaml").get("swaps") or []
    if not isinstance(swaps, list):
        raise ValueError("%s: `swaps` must be a list" % path)
    for sw in swaps:
        vals = sw.get("values") if isinstance(sw, dict) else None
        if not (isinstance(sw, dict) and isinstance(sw.get("arg"), str) and (_any(vals) or isinstance(vals, list))):
            raise ValueError("%s: a swap needs arg and values (a list or any): %r" % (path, sw))
    return {sw["arg"]: sw for sw in swaps}


def ct_status(sdir):
    """test -> normalized status (`pass`, `fail`, ...) of the ct_status.yaml in sdir ({} without one)."""
    tests = _yaml(sdir, "ct_status.yaml").get("tests") or {}
    return {str(t).strip(): str(s).strip().lower() for t, s in tests.items()}


def passed_tests(sdir):
    """Component tests the ct_status.yaml in sdir marks `pass` (a missing test is not passed)."""
    return {t for t, status in ct_status(sdir).items() if status == "pass"}


def swap_rules(sdir, idir):
    """arg -> (swap, test -> status that judges it). A setup entry replaces the image's and is judged by the setup's
    tests, an image entry by the image's tests with the setup's non-`pass` records over them; a `fail` in either
    directory holds for both."""
    setup, image = ct_status(sdir), ct_status(idir)
    fails = {t: "fail" for d in (setup, image) for t, s in d.items() if s == "fail"}
    image_view = dict(image)
    image_view.update({t: s for t, s in setup.items() if s != "pass"})
    image_view.update(fails)
    setup_view = dict(setup)
    setup_view.update(fails)
    rules = {a: (s, image_view) for a, s in load_allowlist(idir).items()}
    rules.update({a: (s, setup_view) for a, s in load_allowlist(sdir).items()})
    return rules


def load_defects(*dirs):
    """The known stack defects of the defects.yaml in each directory (setup and image), each with its source file."""
    out = []
    for d in dirs:
        entries = _yaml(d, "defects.yaml").get("defects") or []
        if not isinstance(entries, list):
            raise ValueError("%s: `defects` must be a list" % os.path.join(d, "defects.yaml"))
        for e in entries:
            vals, when = (e or {}).get("values"), (e or {}).get("when") or {}
            if not (isinstance(e, dict) and isinstance(e.get("arg"), str) and (_any(vals) or isinstance(vals, list))
                    and isinstance(when, dict) and e.get("reason") and e.get("evidence")):
                raise ValueError("%s: a defect needs arg, values (a list or any), reason and evidence, and `when` "
                                 "must be a mapping: %r" % (os.path.join(d, "defects.yaml"), e))
            out.append(dict(e, values=_values(vals), when=when, source=os.path.join(d, "defects.yaml")))
    return out


def in_defect_state(e, args, base):
    """True when `args` (a run's resolved arguments) are in the defect state of entry e: its arg takes a listed value
    (`any`: a value other than the baseline's) and every `when` condition holds."""
    a = e["arg"]
    hit = _norm(args.get(a)) != _norm(base.get(a)) if e["values"] == "any" else _norm(args.get(a)) in e["values"]
    return hit and all(_norm(args.get(k)) == _norm(v) for k, v in e["when"].items())


def fired_defects(base, cand, defects):
    """arg -> the first defect entry the candidate is in and the baseline is not, whichever arguments changed."""
    out = {}
    for e in defects:
        if e["arg"] not in out and in_defect_state(e, cand, base) and not in_defect_state(e, base, base):
            out[e["arg"]] = e
    return out


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
    framework would compute. Returns (ok, why): True, False when both values are integers the framework could not
    derive, None when a value or an argument it is derived from is not printed as an integer (cannot recompute).

    padded_vocab_size = ceil(orig_vocab / m) * m with m = make_vocab_size_divisible_by * TP
    (megatron/training/tokenizer/tokenizer.py:110-123). orig_vocab comes from the tokenizer and is NOT a resolved
    argument (`vocab_size` prints None). The candidate must train the baseline's vocabulary, so orig_vocab is pinned by
    the baseline: orig in (padded_b - m_b, padded_b]. The candidate is accepted iff its value is
    a multiple of its own m and admits that same orig, i.e. the two half-open intervals intersect. With m unchanged
    this forces the value to be unchanged; an arbitrary override lands outside the baseline's interval and is refused.
    """
    pb, pc = _int(base.get(name)), _int(cand.get(name))
    if pb is None or pc is None:
        return None, "derived value not printed; cannot recompute"
    f = DERIVED_FROM[name]
    mb, mc = 1, 1
    for a in f:
        vb, vc = _int(base.get(a)), _int(cand.get(a))
        if not vb or not vc:
            return None, f"{a} not printed as a positive integer ({base.get(a)!r} -> {cand.get(a)!r}); cannot recompute"
        mb, mc = mb * vb, mc * vc
    if pc % mc:
        return False, f"{name}: {pc} is not a multiple of {' x '.join(f)} = {mc}"
    lo, hi = max(pb - mb, pc - mc), min(pb, pc)          # (lo, hi] = the orig_vocab values both runs allow
    if lo >= hi:
        return False, (f"{name}: {pb} -> {pc} is not what {' x '.join(f)} = {mb} -> {mc} derives "
                       f"(no original vocab size gives both)")
    return True, f"{name}: {pb} -> {pc} follows {mb} -> {mc} (original vocab in ({lo}, {hi}])"


def _swap_class(name, a, b, rules):
    """(class, why): allowlisted when the setup's or the image's allowlist lists the name with both values and its
    component test passed in the source that lists it, ct_failed when that test is recorded `fail` in either
    directory, else unknown."""
    if name not in rules:
        return "unknown", "not in any allowlist"
    swap, status = rules[name]
    vals, ct = _values(swap["values"]), str(swap.get("component_test", "none")).strip()
    ct = "none" if ct.lower() == "none" else ct
    if vals != "any" and (_norm(a) not in vals or _norm(b) not in vals):
        return "unknown", "value outside its allowlist"
    # the dispatcher swap needs its component test only when the flex dispatcher is involved
    needs = ct != "none" and not (name == "moe_token_dispatcher_type" and "flex" not in (_norm(a), _norm(b)))
    if needs and status.get(ct) == "fail":
        return "ct_failed", "component test %s recorded fail" % ct
    if not needs or status.get(ct) == "pass":
        return "allowlisted", None
    return "unknown", "component test %s not passed" % ct


def _unmoved_derived(base, cand, out, why, fixed, unknown_why):
    """A derived value that did not move while an argument it is derived from did must still be the value the
    framework computes, or one of the two logs does not describe its own run; one that cannot be recomputed is
    unknown. A value the frozen arguments fix (`eq`) is set by flag, not derived (Megatron tokenizer.py:104-105 keeps
    an explicit --padded-vocab-size)."""
    for n, causes in DERIVED_FROM.items():
        if any(n in out[k] for k in ("derived", "derived_violation", "unknown")) or n not in base or n not in cand:
            continue
        if n in fixed and _int(base.get(n)) == _int(cand.get(n)) == _int(fixed[n]):
            why[n] = "%s: fixed at %s by the setup's frozen arguments (an explicit flag, not derived)" % (n, fixed[n])
            continue
        if any(base.get(c) != cand.get(c) for c in causes):
            ok, w = derived_ok(n, base, cand)
            if ok is None:
                out["unknown"][n], unknown_why[n] = (base.get(n), cand.get(n)), w
            else:
                why[n] = w
                if not ok:
                    out["derived_violation"][n] = (base.get(n), cand.get(n))


def diff(base, cand, classes, spec=None, idir=None):
    """{outcome: {name: (base, cand)}} over every name whose value differs, judged with the setup's and image's rules;
    the outcomes are FAILING, ACCEPTED and unknown. defect_why, ct_failed_why and unknown_why give each such name's
    reason, derived_why each derived value's."""
    sdir = setup_dir(spec)
    rules, defects = swap_rules(sdir, idir), load_defects(*(d for d in (sdir, idir) if d))
    out = {k: {} for k in FAILING + ACCEPTED + ("unknown",)}
    why, reasons = {}, {"defect": {}, "ct_failed": {}, "unknown": {}}
    for n in sorted(set(base) | set(cand)):
        a, b = base.get(n), cand.get(n)
        if a == b:
            continue
        c, r = classify(n, classes), None
        if c == "derived":
            ok, w = derived_ok(n, base, cand)
            if ok is None:
                c, r = "unknown", w
            else:
                why[n], c = w, "derived" if ok else "derived_violation"
        elif n not in base or n not in cand:   # one build lacks the argument: not a change the candidate made
            side = "baseline's" if n in base else "candidate's"
            c, r = "unknown", ("objective-class name present in one dump only (the %s): check by hand" % side
                               if c == "objective" else "only in the %s dump" % side)
        elif c == "allowlisted":
            c, r = _swap_class(n, a, b, rules)
        elif c in ("numeric", "unlisted"):
            c, r = "unknown", "numeric class" if c == "numeric" else "unlisted"
        out[c][n] = (a, b)
        if r:
            reasons[c][n] = r
    for n, e in fired_defects(base, cand, defects).items():
        for k in out:                          # the defect replaces the argument's own outcome
            out[k].pop(n, None)
        for k in reasons:
            reasons[k].pop(n, None)
        out["defect"][n] = (base.get(n), cand.get(n))
        moved = [k for k in [n] + list(e["when"]) if _norm(base.get(k)) != _norm(cand.get(k))]
        reasons["defect"][n] = dict({k: e[k] for k in ("reason", "evidence", "source")}, changed=moved)
    fixed = {k: r["eq"] for k, r in (spec or {}).get("args", {}).items() if "eq" in r}
    _unmoved_derived(base, cand, out, why, fixed, reasons["unknown"])
    if why:
        out["derived_why"] = why
    out.update({k + "_why": v for k, v in reasons.items() if v})
    return out


def unknown_flags(changes):
    """[{name, from, to, reason}] of the unknown changes, for review."""
    why = changes.get("unknown_why", {})
    return [{"name": n, "from": a, "to": b, "reason": why.get(n)} for n, (a, b) in changes.get("unknown", {}).items()]


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
    image: the image reference, pinned by digest, that both arms ran (None: no image rules). L1 is False on a FAILING
    change, an error or a frozen violation; unknown changes are listed in `unknown` and do not fail it."""
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
    return {"L1": ok, "errors": errors, "changes": changes, "unknown": unknown_flags(changes), "frozen": frozen,
            "setup_dir": setup_dir(spec), "image_dir": idir}


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
