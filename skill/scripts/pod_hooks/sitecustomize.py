# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATTUNE hook-chain loader: the first sitecustomize.py on a run's PYTHONPATH.

Python imports just the first sitecustomize.py it finds, so every in-pod hook (miles compatibility, profiler rank
selection, the parity probe) loads through this one file. hook_chain.py mounts it with
`attune_chain.json` and the hook files (`hook_<name>.py`) in one directory at the front of PYTHONPATH.

Hooks load in registry order. A hook loads when the registry marks it default or its name is in $ATTUNE_HOOKS
(comma-separated), and only if its sha256 matches the registry. One line per process records the outcome on stderr:
  ATTUNE_HOOKS chain=<sha12> loaded=[name@sha12,...] skipped=[...] unknown=[...] chained=<path or none>
Then the next sitecustomize.py runs, once: the first one on sys.path after this directory that is not one of ATTUNE's
own (a registered hook's sha256, a copy of this loader, or a file marked attune-hook-chain-member). That is the
image's own, or another loader placed after this directory on PYTHONPATH, which must then run the image's itself.
A sitecustomize.py in front of this directory shadows the chain.
Never raises into the host program.
"""
import hashlib
import importlib.util
import json
import os
import sys


# Files that belong to the chain itself and never run as "the next" sitecustomize.py, in any version.
OWN_MARKERS = (b"attune-hook-chain-member", b"ATTUNE hook-chain loader")


def _chain_next(here, skip_sha):
    """Run the first sitecustomize.py on sys.path after `here` that is not ATTUNE's own. Returns <path>@<sha12>."""
    f = "?"
    try:
        entries = [os.path.abspath(p) for p in sys.path if p]
        start = entries.index(here) + 1 if here in entries else 0
        for d in entries[start:]:
            f = os.path.join(d, "sitecustomize.py")
            if d == here or not os.path.isfile(f):
                continue
            with open(f, "rb") as fh:
                data = fh.read()
            sha = hashlib.sha256(data).hexdigest()
            if sha in skip_sha or any(m in data for m in OWN_MARKERS):
                continue
            exec(compile(data, f, "exec"), {"__name__": "_chained_sitecustomize", "__file__": f})
            return "%s@%s" % (f, sha[:12])
    except Exception as e:
        return "%s (raised %s)" % (f, type(e).__name__)
    return "none"


def _main():
    if getattr(sys, "_attune_hook_chain_loaded", False):
        return
    sys._attune_hook_chain_loaded = True
    here = os.path.dirname(os.path.abspath(globals().get("__file__") or ""))
    loaded, skipped = [], []
    try:
        with open(os.path.join(here, "attune_chain.json")) as fh:
            reg = json.load(fh)
    except Exception as e:
        reg = {"hooks": [], "chain_sha256": "none"}
        skipped.append("registry:%s" % type(e).__name__)
    want = {x.strip() for x in os.environ.get("ATTUNE_HOOKS", "").split(",") if x.strip()}
    names = set()
    for h in reg.get("hooks", []):
        name = h.get("name", "?")
        names.add(name)
        if not (h.get("default") or name in want):
            continue
        path = os.path.join(here, h.get("file", ""))
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            sha = hashlib.sha256(data).hexdigest()
            if sha != h.get("sha256"):
                skipped.append("%s:hash-mismatch" % name)
                continue
            spec = importlib.util.spec_from_file_location("attune_hook_" + name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            loaded.append("%s@%s" % (name, sha[:12]))
        except Exception as e:
            skipped.append("%s:%s" % (name, type(e).__name__))
    skip_sha = {h.get("sha256") for h in reg.get("hooks", [])}
    try:
        with open(os.path.abspath(globals().get("__file__") or ""), "rb") as fh:
            skip_sha.add(hashlib.sha256(fh.read()).hexdigest())
    except Exception:
        pass
    chained = _chain_next(here, skip_sha)
    try:
        sys.stderr.write("ATTUNE_HOOKS chain=%s loaded=%s skipped=%s unknown=%s chained=%s\n" % (
            str(reg.get("chain_sha256", "?"))[:12], loaded, skipped, sorted(want - names), chained))
        sys.stderr.flush()
    except Exception:
        pass


try:
    _main()
except Exception:
    pass
