#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One hashed sitecustomize chain for every in-pod hook.

Python imports only the first sitecustomize.py on PYTHONPATH, so hooks are never mounted individually.
`install()` puts the loader (pod_hooks/sitecustomize.py), the rendered registry (attune_chain.json) and every
registered hook (hook_<name>.py) into the run's ConfigMap, mounts them read-only at the registry's mount_dir
on every RayJob container, puts that directory first on PYTHONPATH (container env, runtimeEnvYAML, and any
`export PYTHONPATH=` in the training script), and adds the requested hooks to $ATTUNE_HOOKS (removing any it is
told to disable).
make_run_yaml.py and make_parity_arm.py call install(); the command line does the same for a single YAML:

    hook_chain.py install <in.yaml> <out.yaml> [--enable a,b] [--run-id ID]
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "pod_hooks" / "registry.json"
LOADER = HERE / "pod_hooks" / "sitecustomize.py"
CHAIN_JSON = "attune_chain.json"
HOOK_ENV = "ATTUNE_HOOKS"
_EXPORT_RE = re.compile(r"(\bexport\s+PYTHONPATH=)([\"']?)")


def _sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def registry(path=None):
    with open(path or REGISTRY) as f:
        reg = json.load(f)
    names = [h["name"] for h in reg["hooks"]]
    if len(set(names)) != len(names) or not all(re.match(r"^[a-z0-9_]+$", n) for n in names):
        raise ValueError("hook names must be unique [a-z0-9_]+")
    return reg


def render(reg=None):
    """(data, chain_sha256): ConfigMap key -> file content for the whole chain."""
    reg = reg or registry()
    hooks, data = [], {}
    for h in reg["hooks"]:
        text = (HERE / h["src"]).read_text()
        key = "hook_%s.py" % h["name"]
        data[key] = text
        hooks.append({"name": h["name"], "file": key, "sha256": _sha(text), "default": bool(h.get("default"))})
    loader = LOADER.read_text()
    chain = _sha(json.dumps({"loader": _sha(loader), "hooks": hooks}, sort_keys=True))
    data["sitecustomize.py"] = loader
    data[CHAIN_JSON] = json.dumps({"chain_sha256": chain, "hooks": hooks}, indent=1, sort_keys=True) + "\n"
    return data, chain


def _pod_specs(job):
    rcs = (job.get("spec") or {}).get("rayClusterSpec") or {}
    for g in [rcs.get("headGroupSpec") or {}] + list(rcs.get("workerGroupSpecs") or []):
        pod = (g.get("template") or {}).get("spec")
        if pod:
            yield pod


def _prefix(value, d):
    parts = str(value).split(":") if value else []
    return value if parts and parts[0] == d else (d + (":" + str(value) if value else ""))


def _walk_strings(node, fn):
    """Replace every string scalar of a parsed YAML tree with fn(string), in place."""
    items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
    for k, v in list(items):
        if isinstance(v, str):
            node[k] = fn(v)
        else:
            _walk_strings(v, fn)


def _merge_hooks(current, enable, disable=()):
    names = [x.strip() for x in str(current or "").split(",") if x.strip() and x.strip() not in disable]
    for e in enable:
        if e not in names:
            names.append(e)
    return ",".join(names)


def _chain_configmap(docs, run_id, job):
    """The manifest's first ConfigMap, or a new one named run_id with the RayJob's namespace and labels."""
    cm = next((d for d in docs if isinstance(d, dict) and d.get("kind") == "ConfigMap"), None)
    if cm is None:
        meta = job.get("metadata") or {}
        cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": run_id}, "data": {}}
        if meta.get("namespace"):
            cm["metadata"]["namespace"] = meta["namespace"]
        if meta.get("labels"):
            cm["metadata"]["labels"] = dict(meta["labels"])
        docs.insert(0, cm)
    cm["data"] = cm.get("data") or {}
    return cm


def _set_container(c, mdir, vol, enable, disable=()):
    mounts = c.setdefault("volumeMounts", [])
    mounts[:] = [m for m in mounts if m.get("mountPath") != mdir and m.get("name") != vol]
    mounts.append({"name": vol, "mountPath": mdir, "readOnly": True})
    env = c.setdefault("env", [])
    pp = next((e for e in env if e.get("name") == "PYTHONPATH"), None)
    if pp is None:
        env.append({"name": "PYTHONPATH", "value": mdir})
    elif "value" not in pp:
        raise RuntimeError("container %s: PYTHONPATH uses valueFrom; set a literal value" % c.get("name"))
    else:
        pp["value"] = _prefix(pp["value"], mdir)
    he = next((e for e in env if e.get("name") == HOOK_ENV), None)
    if he is None:
        env.append({"name": HOOK_ENV, "value": _merge_hooks("", enable)})
    else:
        he["value"] = _merge_hooks(he.get("value"), enable, disable)


def _set_runtime_env(job, mdir, enable, disable=()):
    import yaml
    spec = job.get("spec") or {}
    if not spec.get("runtimeEnvYAML"):
        return
    r = yaml.safe_load(spec["runtimeEnvYAML"]) or {}
    ev = r.setdefault("env_vars", {})
    if "PYTHONPATH" in ev:  # a job-level PYTHONPATH overrides the pod env for Ray workers
        ev["PYTHONPATH"] = _prefix(ev["PYTHONPATH"], mdir)
    ev[HOOK_ENV] = _merge_hooks(ev.get(HOOK_ENV), enable, disable)
    spec["runtimeEnvYAML"] = yaml.safe_dump(r, sort_keys=False)


def _prefix_exports(docs, cm, chain_keys, mdir):
    """Put mdir first in every `export PYTHONPATH=` of the manifest (the chain's own files excepted); returns the
    number of edits."""
    edits = []

    def rep(m, s):
        if s[m.end():].startswith(mdir):
            return m.group(0)
        edits.append(m.start())
        return m.group(1) + m.group(2) + mdir + ":"

    def fix(s):
        return _EXPORT_RE.sub(lambda m: rep(m, s), s)

    for d in docs:
        if d is not cm:
            _walk_strings(d, fix)
            continue
        for k, v in list(cm["data"].items()):
            if k not in chain_keys and isinstance(v, str):
                cm["data"][k] = fix(v)
    return len(edits)


def install(docs, run_id, enable=(), reg=None, disable=()):
    """Install the chain into parsed YAML docs (in place); `disable` names are removed from an $ATTUNE_HOOKS the
    manifest already carries. Returns a summary dict."""
    reg = reg or registry()
    known = {h["name"] for h in reg["hooks"]}
    bad = [e for e in list(enable) + list(disable) if e not in known]
    if bad:
        raise RuntimeError("unknown hook(s) %s; registered: %s" % (bad, sorted(known)))
    jobs = [d for d in docs if isinstance(d, dict) and d.get("kind") == "RayJob"]
    if not jobs:
        raise RuntimeError("hook chain: no RayJob in the YAML")
    data, chain = render(reg)
    mdir, vol = reg["mount_dir"], reg["volume"]
    cm = _chain_configmap(docs, run_id, jobs[0])
    cm["data"].update(data)
    items = [{"key": k, "path": k} for k in sorted(data)]
    n = 0
    for job in jobs:
        for pod in _pod_specs(job):
            vols = pod.setdefault("volumes", [])
            vols[:] = [v for v in vols if v.get("name") != vol]
            vols.append({"name": vol, "configMap": {"name": cm["metadata"]["name"], "items": items}})
            for c in pod.get("containers") or []:
                _set_container(c, mdir, vol, enable, disable)
                n += 1
        _set_runtime_env(job, mdir, enable, disable)
    if n == 0:
        raise RuntimeError("hook chain: no pod containers found in the RayJob")
    return {"chain_sha256": chain, "configmap": cm["metadata"]["name"], "containers": n,
            "export_edits": _prefix_exports(docs, cm, data, mdir), "enabled": list(enable)}


def dump_yaml(docs):
    """Multi-document YAML with every multi-line string as a literal block."""
    import yaml

    class _Dumper(yaml.SafeDumper):
        pass

    _Dumper.add_representer(str, lambda d, v: d.represent_scalar(
        "tag:yaml.org,2002:str", v, style="|" if "\n" in v else None))
    return yaml.dump_all(docs, Dumper=_Dumper, sort_keys=False, width=4096, default_flow_style=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("install", help="put the hook chain into a RayJob YAML")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--enable", default="", help="comma-separated hooks to enable besides the default ones")
    p.add_argument("--run-id", help="name of a ConfigMap to create when the YAML has none (default: the RayJob name)")
    a = ap.parse_args(argv)
    import yaml
    with open(a.input) as f:
        docs = [d for d in yaml.safe_load_all(f) if d is not None]
    rid = a.run_id or next((d["metadata"]["name"] for d in docs if isinstance(d, dict) and d.get("kind") == "RayJob"),
                           None)
    info = install(docs, rid, enable=[x for x in a.enable.split(",") if x])
    Path(a.output).write_text(dump_yaml(docs))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as e:
        sys.exit("ERROR: %s" % e)
