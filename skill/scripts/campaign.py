#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The campaign a skill script runs in, when harness/launch_campaign.sh made one.

A script runs in a campaign when its skill directory is a campaign's copy: <control>/skill, beside the launcher's
integrity record <control>/integrity.json. The campaign's settings are then its <control>/config.env.
`require_settings()` refuses (exit 2) an environment value that differs from them, so an `export` in the agent's shell
cannot change the budget, the kubectl context or namespace, the run prefix, the log pod, the campaign directory or any
other campaign setting; an unset or empty value takes the campaign's value. `integrity_check(where)` runs the
harness's integrity check of the campaign (harness/integrity.py). Outside a campaign both do nothing.

    campaign.py check-settings           # exit 2 with the differences
    campaign.py check-settings --shell   # also print `export NAME=value` for the settings the environment lacks
"""
import os
import shlex
import sys

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CONTROL = os.path.dirname(SKILL_DIR)
RECORD = os.path.join(CONTROL, "integrity.json")
CONFIG = os.path.join(CONTROL, "config.env")


def in_campaign():
    return os.path.isfile(RECORD)


def settings(path=CONFIG):
    """The KEY=value lines of the campaign's config.env (values shell-quoted)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, _, value = line.partition("=")
                words = shlex.split(value)
                out[key.strip()] = words[0] if words else ""
    return out


def resolve(env=None):
    """(differences [(name, campaign value, environment value)], {name: campaign value} the environment lacks)."""
    env = os.environ if env is None else env
    diffs, missing = [], {}
    for name, want in sorted(settings().items()):
        got = env.get(name) or ""
        if not got:
            if want:
                missing[name] = want
        elif got != want:
            diffs.append((name, want, got))
    return diffs, missing


def refusal(diffs):
    lines = ["  %s is %r in %s, but %r in the environment" % (n, w, CONFIG, g) for n, w, g in diffs]
    return ("refused: this campaign's settings come from its config.env, and the environment disagrees:\n%s\n"
            "Run the script without overriding them (an export or NAME=value prefix cannot change a campaign "
            "setting; an unset or empty value takes the campaign's)." % "\n".join(lines))


def require_settings():
    """Exit 2 when the environment disagrees with the campaign's config.env, else fill in the settings it lacks;
    no-op outside a campaign."""
    if not in_campaign():
        return
    try:
        diffs, missing = resolve()
    except (OSError, ValueError) as e:
        sys.exit("refused: cannot read the campaign's settings %s: %s" % (CONFIG, e))
    if diffs:
        print(refusal(diffs), file=sys.stderr)
        sys.exit(2)
    os.environ.update(missing)


def integrity_check(where, **kwargs):
    """The campaign's integrity problems ([] when intact or outside a campaign); kwargs go to integrity.check."""
    if not in_campaign():
        return []
    try:
        sys.path.insert(0, os.path.join(CONTROL, "harness"))
        import integrity
    except Exception as e:  # a campaign whose check cannot run has not passed it
        return [{"path": os.path.join(CONTROL, "harness", "integrity.py"), "problem": "cannot load: %s" % e}]
    return integrity.check(CONTROL, where, **kwargs)


def teardown_note():
    """The violation record's line on the teardown of the campaign's tracked runs."""
    try:
        sys.path.insert(0, os.path.join(CONTROL, "harness"))
        import integrity
        return integrity.teardown_note(CONTROL)
    except Exception as e:
        return "live runs: unknown (%s)" % e


def describe(problems):
    head = "; ".join("%s: %s" % (p.get("path"), p.get("problem")) for p in problems[:5])
    return head + ("; and %d more" % (len(problems) - 5) if len(problems) > 5 else "")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("check-settings",))
    ap.add_argument("--shell", action="store_true", help="print exports for the settings the environment lacks")
    a = ap.parse_args(argv)
    if not in_campaign():
        return 0
    try:
        diffs, missing = resolve()
    except (OSError, ValueError) as e:
        print("refused: cannot read the campaign's settings %s: %s" % (CONFIG, e), file=sys.stderr)
        return 2
    if diffs:
        print(refusal(diffs), file=sys.stderr)
        return 2
    if a.shell:
        print("".join("export %s=%s\n" % (n, shlex.quote(v)) for n, v in missing.items()), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
