# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The hook-chain loader runs the next sitecustomize.py on sys.path once, after its own hooks."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

LOADER = Path(__file__).resolve().parents[1] / "pod_hooks" / "sitecustomize.py"
MARK = 'import os\nopen(os.environ["MARK_FILE"], "a").write("%s\\n")\n'


def _chain(tmp_path, hooks=()):
    d = tmp_path / "chain"
    d.mkdir()
    shutil.copy(LOADER, d / "sitecustomize.py")
    (d / "attune_chain.json").write_text(json.dumps({"hooks": list(hooks), "chain_sha256": "test"}))
    return d


def _site(tmp_path, name, text):
    d = tmp_path / name
    d.mkdir()
    (d / "sitecustomize.py").write_text(text)
    return d


def _run(tmp_path, *dirs):
    mark = tmp_path / "mark.txt"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(d) for d in dirs), MARK_FILE=str(mark))
    r = subprocess.run([sys.executable, "-c", "pass"], env=env, capture_output=True, text=True)
    return (mark.read_text().split() if mark.exists() else []), r.stderr


def test_next_sitecustomize_after_the_chain_runs_once(tmp_path):
    marks, err = _run(tmp_path, _chain(tmp_path), _site(tmp_path, "other", MARK % "other"),
                      _site(tmp_path, "image", MARK % "image"))
    assert marks == ["other"] and "chained=" + str(tmp_path / "other" / "sitecustomize.py") + "@" in err


def test_a_copy_of_the_loader_is_skipped(tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    shutil.copy(LOADER, copy / "sitecustomize.py")
    marks, _ = _run(tmp_path, _chain(tmp_path), copy, _site(tmp_path, "image", MARK % "image"))
    assert marks == ["image"]


def test_a_registered_hook_mounted_as_sitecustomize_is_skipped(tmp_path):
    hook = MARK % "hook"
    sha = hashlib.sha256(hook.encode()).hexdigest()
    chain = _chain(tmp_path, hooks=[{"name": "h", "file": "h.py", "sha256": sha, "default": False}])
    marks, _ = _run(tmp_path, chain, _site(tmp_path, "stray", hook), _site(tmp_path, "image", MARK % "image"))
    assert marks == ["image"]


def test_a_sitecustomize_in_front_of_the_chain_shadows_it(tmp_path):
    marks, err = _run(tmp_path, _site(tmp_path, "front", MARK % "front"), _chain(tmp_path))
    assert marks == ["front"] and "ATTUNE_HOOKS" not in err


def test_a_stale_copy_of_a_hook_is_skipped_by_its_marker(tmp_path):
    stale = "# attune-hook-chain-member: an older version of a hook\n" + MARK % "stale"
    marks, _ = _run(tmp_path, _chain(tmp_path), _site(tmp_path, "stale", stale), _site(tmp_path, "image", MARK % "image"))
    assert marks == ["image"]


def test_a_chained_file_that_raises_is_reported_and_contained(tmp_path):
    marks, err = _run(tmp_path, _chain(tmp_path), _site(tmp_path, "bad", "raise ValueError('x')\n"))
    assert "(raised ValueError)" in err and "Traceback" not in err
