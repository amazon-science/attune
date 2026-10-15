#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the log pod mounts the shared volume read-only (launch_campaign.sh runs it before a campaign).

    kubectl --context CTX -n NS get pod <log pod> -o json | log_pod.py <shared mount> [<log pod>]

The agent reads run logs through the log pod with `kubectl exec`, and the guard lets any command run there, so the
pod's own mount is what keeps the shared volume safe. The shared volume is every volume mounted at, above or below
the shared mount path, and every other volume on the same PersistentVolumeClaim; each of its mounts in every
container must be read-only (the mount's readOnly, or the volume source's). Exit 0 when it is, 1 with the problems
and the mount to use otherwise, 2 on unreadable input.
"""
import json
import sys


def _covers(mount_path, shared):
    m, s = mount_path.rstrip("/") or "/", shared.rstrip("/") or "/"
    return s == m or s.startswith(m.rstrip("/") + "/") or m.startswith(s.rstrip("/") + "/")


def _claim(volume):
    return ((volume or {}).get("persistentVolumeClaim") or {}).get("claimName")


def _source_read_only(volume):
    return any(isinstance(v, dict) and v.get("readOnly") is True for k, v in (volume or {}).items() if k != "name")


def problems(pod, shared):
    """Why the pod's mounts of the shared volume are not all read-only ([] when they are)."""
    spec = pod.get("spec") or {}
    volumes = {v.get("name"): v for v in spec.get("volumes") or []}
    containers = list(spec.get("containers") or []) + list(spec.get("ephemeralContainers") or [])
    names = {m.get("name") for c in containers for m in c.get("volumeMounts") or [] if _covers(m.get("mountPath", ""),
                                                                                                shared)}
    if not names:
        return ["mounts nothing at %s" % shared]
    claims = {_claim(volumes.get(n)) for n in names} - {None}
    shared_volumes = names | {n for n, v in volumes.items() if _claim(v) in claims}
    out = []
    for c in containers:
        for m in c.get("volumeMounts") or []:
            if m.get("name") in shared_volumes and m.get("readOnly") is not True \
                    and not _source_read_only(volumes.get(m.get("name"))):
                out.append("container %s mounts volume %s at %s writable" % (c.get("name"), m.get("name"),
                                                                             m.get("mountPath")))
    return out


def fix(pod, shared):
    names = sorted({m.get("name") for c in (pod.get("spec") or {}).get("containers") or []
                    for m in c.get("volumeMounts") or [] if _covers(m.get("mountPath", ""), shared)}) or ["shared"]
    return ("Mount the shared volume read-only in the log pod's container spec:\n"
            "  volumeMounts:\n  - name: %s\n    mountPath: %s\n    readOnly: true" % (names[0], shared))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print("usage: " + __doc__.split("\n\n")[1].strip())
        return 0 if args else 2
    shared, name = args[0], args[1] if len(args) > 1 else "the log pod"
    try:
        pod = json.load(sys.stdin)
    except ValueError as e:
        print("log_pod.py: cannot read the pod JSON: %s" % e, file=sys.stderr)
        return 2
    found = problems(pod if isinstance(pod, dict) else {}, shared)
    if not found:
        return 0
    print("%s must mount the shared volume read-only: %s.\n%s" % (name, "; ".join(found), fix(pod, shared)),
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
