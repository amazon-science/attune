#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-campaign frozen arguments: arguments no candidate may change.

The file named by $AUTOTUNE_FROZEN_ARGS (YAML; JSON is valid YAML) lists them:

    {"args": {"tensor_model_parallel_size": {"eq": 2, "default": 1},
              "optimizer": "adam",                                   # bare value = {"eq": value}
              "use_example_feature": {"eq": false, "default": false},
              "example_kernel": {"in": ["impl_a", "impl_b"]},
              "seed": {"unchanged": true}},                          # equal to the baseline arm's value
     "env": {"EXAMPLE_DEBUG": {"absent": true}, "EXAMPLE_MODE": {"eq": "2"}},
     "products": [{"args": ["max_tokens_per_gpu", "context_parallel_size"], "eq": 6144,
                   "defaults": {"context_parallel_size": 1}}]}

Names are Megatron/slime argument names (dest form, underscores). `default` is the value an argument
takes when a manifest does not pass its flag. `unchanged` freezes an argument at whatever the baseline arm
resolved (layer 1 compares the two arms). `env` freezes pod environment variables (`eq`, or `absent`), checked
pre-launch in container env, runtimeEnvYAML env_vars and `export NAME=` lines. `couplings` (conditional rules, e.g.
"when X is on, Y must be set") are evaluated by couplings.py beside this file (`violations(spec, get)`), both
pre-launch and at layer 1; a spec whose couplings cannot be evaluated fails closed. The file's directory is the
setup's layer-1 directory (rules/README.md).

Two checks share the spec:
  check_resolved(args)  layer 1 of the correctness check, on the resolved-argument dump of a run's log;
                        an argument missing from the dump with no default is a violation (fail closed).
  check_manifest(text)  pre-launch, on the `--flag value` tokens of a RayJob/ConfigMap manifest outside comments;
                        a flag absent from the manifest is checked at layer 1 instead, and a non-literal value
                        (shell variable) is a violation.

    frozen_args.py check-log <train.log> [--spec FILE]
    frozen_args.py check-manifest <manifest.yaml> [--spec FILE]
Exit 0 = no violation, 1 = violations (printed as JSON), 2 = usage or spec error.
"""
import argparse
import importlib.util
import json
import os
import re
import sys

ENV = "AUTOTUNE_FROZEN_ARGS"
# The layer-1 rule lists (numerics_list.yaml, layer1/<setup>/): $AUTOTUNE_RULES_DIR, else rules/ beside this file.
HERE = os.path.dirname(os.path.abspath(__file__))
RULES_DIR = os.environ.get("AUTOTUNE_RULES_DIR") or os.path.join(HERE, "rules")


class SpecError(Exception):
    pass


def load(path=None):
    """Return the normalized spec, or None when no frozen-args file is configured."""
    path = path or os.environ.get(ENV, "").strip()
    if not path:
        return None
    import yaml
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"cannot read frozen-args file {path}: {e}") from e
    if not isinstance(raw, dict):
        raise SpecError(f"frozen-args file {path}: expected a mapping at the top level")
    for key, kind in (("args", dict), ("products", list), ("env", dict), ("couplings", list)):
        if not isinstance(raw.get(key) or kind(), kind):
            raise SpecError(f"frozen-args file {path}: `{key}` must be a {'mapping' if kind is dict else 'list'}")
    args = {}
    for name, rule in (raw.get("args") or {}).items():
        if not isinstance(rule, dict):
            rule = {"eq": rule}
        if not ({"eq", "in", "unchanged"} & set(rule)):
            raise SpecError(f"frozen arg {name}: rule needs 'eq', 'in' or 'unchanged'")
        args[name.replace("-", "_")] = rule
    products = []
    for p in raw.get("products") or []:
        if not p.get("args") or "eq" not in p:
            raise SpecError("product rule needs 'args' and 'eq'")
        products.append({"args": [a.replace("-", "_") for a in p["args"]], "eq": p["eq"],
                         "defaults": {k.replace("-", "_"): v for k, v in (p.get("defaults") or {}).items()}})
    env = {}
    for name, rule in (raw.get("env") or {}).items():
        if not isinstance(rule, dict):
            rule = {"eq": rule}
        if not ({"eq", "absent"} & set(rule)):
            raise SpecError(f"frozen env {name}: rule needs 'eq' or 'absent'")
        env[name] = rule
    return {"path": path, "args": args, "products": products, "env": env, "couplings": raw.get("couplings") or [],
            "raw": raw}


def _couplings(spec, get):
    """Violations of the spec's couplings for a getter (dest name -> value or None); fail closed without an evaluator."""
    if not (spec or {}).get("couplings"):
        return []
    path = os.path.join(HERE, "couplings.py")
    try:
        m = importlib.util.spec_from_file_location("attune_rules_couplings", path)
        mod = importlib.util.module_from_spec(m)
        m.loader.exec_module(mod)
        msgs = mod.violations(spec["raw"], get)
    except Exception as e:
        return [{"arg": "couplings", "value": None, "want": "evaluable", "why": "cannot evaluate couplings (%s: %s)"
                 % (type(e).__name__, e)}]
    return [{"arg": "coupling:" + m.split(":", 1)[0], "value": None, "want": m.split(":", 1)[-1].strip(),
             "why": "frozen coupling violated"} for m in msgs]


def _norm(v):
    """Comparable form: number, bool, None, or a lowercased string with any Enum prefix removed."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().strip("'\"")
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z0-9_]+)$", s)  # AttnBackend.flash -> flash
    return (m.group(1) if m else s).lower()


def _ok(rule, value):
    v = _norm(value)
    if "eq" in rule and v != _norm(rule["eq"]):
        return False
    if "in" in rule and v not in {_norm(x) for x in rule["in"]}:
        return False
    return True


def _want(rule):
    if "eq" in rule:
        return "== %r" % rule["eq"]
    if "in" in rule:
        return "in %r" % list(rule["in"])
    return "unchanged from the baseline"


def check_resolved(spec, resolved, baseline=None):
    """Violations of the spec in a resolved-argument dict (name -> string). baseline: the baseline arm's dict,
    needed for `unchanged` rules (without it they are not checked)."""
    out = []
    if not spec:
        return out
    for name, rule in spec["args"].items():
        if name in resolved:
            val = resolved[name]
        elif "default" in rule:
            val = rule["default"]
        else:
            out.append({"arg": name, "value": None, "want": _want(rule), "why": "missing from the resolved dump"})
            continue
        if not _ok(rule, val):
            out.append({"arg": name, "value": val, "want": _want(rule), "why": "frozen argument changed"})
        elif rule.get("unchanged") and baseline is not None:
            bval = baseline.get(name, rule.get("default"))
            if _norm(bval) != _norm(val):
                out.append({"arg": name, "value": val, "want": "== %r (baseline)" % bval, "why": "frozen argument changed"})
    for p in spec["products"]:
        vals = []
        for a in p["args"]:
            v = resolved.get(a, p["defaults"].get(a))
            vals.append(_norm(v))
        if any(not isinstance(v, float) for v in vals):
            out.append({"arg": "*".join(p["args"]), "value": vals, "want": "== %r" % p["eq"],
                        "why": "product operand missing or non-numeric"})
            continue
        prod = 1.0
        for v in vals:
            prod *= v
        if prod != float(p["eq"]):
            out.append({"arg": "*".join(p["args"]), "value": prod, "want": "== %r" % p["eq"],
                        "why": "frozen product changed"})
    out += _couplings(spec, lambda n: None if resolved.get(n) in (None, "None", "") else resolved.get(n))
    return out


_FLAG = re.compile(r"(?<![\w-])--([A-Za-z0-9][A-Za-z0-9-]*)(?:[ \t=]+(?!--)([^\s\\]+))?")


def _manifest_strings(node, out):
    if isinstance(node, dict):
        for v in node.values():
            _manifest_strings(v, out)
    elif isinstance(node, list):
        for v in node:
            _manifest_strings(v, out)
    elif isinstance(node, str):
        out.append(node)
    return out


# ConfigMap entries that are data or source, never a launched script: the hook chain, adapters, embedded modules,
# config files and notes. Every other entry is treated as a script, whatever its name (`run`, `train.bash`, `run.sh`),
# because the entrypoint may run any of them.
NOT_SCRIPT = (".py", ".json", ".yaml", ".yml", ".md", ".txt")


def _scripts(docs):
    """Every ConfigMap script entry (the one that starts `python3 -m` last, so the run script's flags win), then every
    RayJob entrypoint."""
    parts = [v for d in docs if isinstance(d, dict) and d.get("kind") == "ConfigMap"
             for k, v in (d.get("data") or {}).items() if isinstance(v, str) and not str(k).endswith(NOT_SCRIPT)]
    parts = sorted(parts, key=lambda v: "python3 -m" in v)
    parts += [str((d.get("spec") or {}).get("entrypoint") or "") for d in docs
              if isinstance(d, dict) and d.get("kind") == "RayJob"]
    return "\n".join(p for p in parts if p)


def script_text(text):
    """The text whose `--flag value` tokens a run passes: _scripts() of a Kubernetes manifest (text or parsed docs),
    else the text itself (a bare script)."""
    if isinstance(text, list):
        return _scripts(text)
    try:
        import yaml
        docs = [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    except Exception:
        return text
    return _scripts(docs) if any(d.get("kind") in ("ConfigMap", "RayJob") for d in docs) else text


def manifest_flags(text):
    """Every `--flag [value]` occurrence in a manifest's text: dest name -> list of values (True = bare flag)."""
    flags = {}
    for m in _FLAG.finditer(text):
        name, val = m.group(1), m.group(2)
        dest = name.replace("-", "_")
        value = True if val is None else val.strip("'\"")
        if dest.startswith("no_") and val is None:
            flags.setdefault(dest[3:], []).append(False)
        flags.setdefault(dest, []).append(value)
    return flags


def manifest_env(text):
    """NAME -> values from `export NAME=value` / `NAME=value` script lines and YAML env entries in manifest text."""
    env = {}
    for m in re.finditer(r"(?:^|[\s;])(?:export\s+)?([A-Z][A-Z0-9_]*)=(\"[^\"]*\"|'[^']*'|[^\s;]*)", text):
        env.setdefault(m.group(1), []).append(m.group(2).strip("'\""))
    for m in re.finditer(r"name:\s*['\"]?([A-Z][A-Z0-9_]*)['\"]?\s*,?\s*\n?\s*value:\s*(\"[^\"]*\"|'[^']*'|[^\s,}]*)", text):
        env.setdefault(m.group(1), []).append(m.group(2).strip("'\""))
    for m in re.finditer(r"^\s*([A-Z][A-Z0-9_]*):\s*(\"[^\"]*\"|'[^']*'|\S+)\s*$", text, re.M):
        env.setdefault(m.group(1), []).append(m.group(2).strip("'\""))
    return env


def check_env(spec, env):
    """Violations of the spec's env rules in a NAME -> [values] map (manifest_env, or parsed pod env)."""
    out = []
    for name, rule in (spec or {}).get("env", {}).items():
        vals = env.get(name, [])
        if rule.get("absent") and vals:
            out.append({"arg": "env:" + name, "value": vals[0], "want": "absent", "why": "frozen environment variable set"})
        elif "eq" in rule and any(_norm(v) != _norm(rule["eq"]) for v in vals):
            out.append({"arg": "env:" + name, "value": vals, "want": "== %r" % rule["eq"],
                        "why": "frozen environment variable changed"})
    return out


def strip_comments(text):
    """Drop full-line comments and trailing ` # ...` comments (a `#` outside quotes after whitespace)."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"(?<=\s)#", line):
            before = line[:m.start()]
            if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                line = before.rstrip()
                break
        out.append(line)
    return "\n".join(out)


def check_manifest(spec, text, env=None):
    """Pre-launch violations: flags present in the manifest with a value the spec forbids, and frozen env vars.
    text: manifest YAML, parsed manifest docs, or a bare script. Flags come from script_text() only (the `.sh` entries
    and entrypoints); env from the whole text. Shell and YAML comments are ignored. env: NAME -> [values] from parsed
    pod specs (added to the text scan)."""
    out = []
    if not spec:
        return out
    flags = manifest_flags(strip_comments(script_text(text)))
    text = strip_comments("\n".join(_manifest_strings(text, [])) if isinstance(text, list) else text)
    menv = manifest_env(text)
    for k, v in (env or {}).items():
        menv.setdefault(k, []).extend(v)
    out += check_env(spec, menv)

    def literal(v):
        return not (isinstance(v, str) and ("$" in v or "`" in v))

    for name, rule in spec["args"].items():
        if not ({"eq", "in"} & set(rule)):
            continue  # `unchanged` needs the baseline arm: layer 1 checks it
        for v in flags.get(name, []):
            if not literal(v):
                out.append({"arg": name, "value": v, "want": _want(rule), "why": "non-literal value for a frozen argument"})
            elif not _ok(rule, v):
                out.append({"arg": name, "value": v, "want": _want(rule), "why": "frozen argument changed"})
    for p in spec["products"]:
        vals, known = [], True
        for a in p["args"]:
            vs = flags.get(a)
            if vs:
                v = vs[-1]
                if not literal(v):
                    out.append({"arg": a, "value": v, "want": "literal", "why": "non-literal value in a frozen product"})
                    known = False
                    break
                vals.append(_norm(v))
            elif a in p["defaults"]:
                vals.append(_norm(p["defaults"][a]))
            else:
                known = False  # operand comes from outside the manifest: layer 1 checks it
                break
        if known and all(isinstance(v, float) for v in vals):
            prod = 1.0
            for v in vals:
                prod *= v
            if prod != float(p["eq"]):
                out.append({"arg": "*".join(p["args"]), "value": prod, "want": "== %r" % p["eq"],
                            "why": "frozen product changed"})
    out += _couplings(spec, lambda n: (flags.get(n) or [None])[-1])
    return out


def resolved_args(log_path):
    """Megatron's resolved-argument dump (`  name ........ value`) of a train.log, first occurrence wins."""
    d = {}
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = re.match(r"^\s*(?:\S+\s+)?([a-z_0-9]+) \.{3,} (.*)$", line.rstrip())
            if m and m.group(1) not in d:
                d[m.group(1)] = m.group(2).strip()
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("check-log", "check-manifest"))
    ap.add_argument("path", help="train.log (check-log) or RayJob/ConfigMap manifest (check-manifest)")
    ap.add_argument("--spec", help="frozen-arguments file (default: $%s)" % ENV)
    a = ap.parse_args(argv)
    try:
        spec = load(a.spec)
    except SpecError as e:
        print(json.dumps({"error": str(e)}))
        return 2
    if spec is None:
        print(json.dumps({"violations": [], "note": f"no frozen-args file (${ENV} unset)"}))
        return 0
    if a.command == "check-log":
        v = check_resolved(spec, resolved_args(a.path))
    else:
        with open(a.path) as f:
            v = check_manifest(spec, f.read())
    print(json.dumps({"spec": spec["path"], "violations": v}, default=str))
    return 1 if v else 0


if __name__ == "__main__":
    sys.exit(main())
