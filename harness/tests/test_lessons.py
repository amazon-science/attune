# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""lessons.py copies each project folder's LESSONS.md to <inbox>/<campaign id>/<folder>/ with a manifest whose
metadata comes from the campaign's config.env and integrity record, keeps the lessons in the campaign folder without
an inbox or after a violation, refuses an inbox inside the campaign or an ATTUNE tree's skill/ and harness/, and
drops a folder whose file is gone at the next export.
Run: python3 -m pytest harness/tests"""
import hashlib
import json

import pytest
from conftest import CAMPAIGN_ID, seal_campaign

import integrity
import lessons

KEYS = {"campaign_id", "model", "gpu", "image", "framework", "catalog_version", "campaign_dir", "lessons",
        "exported_at"}


def _export(sealed, inbox, **env):
    return lessons.export(str(sealed.control), {"ATTUNE_LESSONS_INBOX": str(inbox), **env})


def test_each_folder_is_copied_with_its_own_hash(sealed):
    one, two = b"## one\n- Claim: x\n", b"## two"
    (sealed.root / "proj" / "LESSONS.md").write_bytes(one)
    (sealed.root / "other").mkdir()
    (sealed.root / "other" / "LESSONS.md").write_bytes(two)
    inbox = sealed.tmp / "inbox"
    assert _export(sealed, inbox).startswith("lessons: exported")
    entry = inbox / sealed.campaign_id
    assert (entry / "proj" / "LESSONS.md").read_bytes() == one and (entry / "other" / "LESSONS.md").read_bytes() == two
    manifest = json.loads((entry / "manifest.json").read_text())
    assert set(manifest) == KEYS and manifest["campaign_dir"] == str(sealed.root)
    assert manifest["lessons"] == [
        {"folder": f, "source": str(sealed.root / f / "LESSONS.md"), "sha256": hashlib.sha256(t).hexdigest()}
        for f, t in (("other", two), ("proj", one))]
    assert not list(inbox.rglob("*.tmp"))


def test_the_manifest_comes_from_the_campaigns_settings_not_the_shell(campaign):
    settings = {"CAMPAIGN_DIR": str(campaign.root), "ATTUNE_CAMPAIGN_ID": CAMPAIGN_ID, "ATTUNE_MODEL_NAME": "m",
                "ATTUNE_GPU_NAME": "H100", "TRAINING_IMAGE": "img@sha256:0", "ATTUNE_FRAMEWORK": "slime 0.3"}
    control = seal_campaign(campaign.root, settings)
    (campaign.root / "proj" / "LESSONS.md").write_text("## one\n")
    inbox = campaign.tmp / "inbox"
    shell = {"ATTUNE_LESSONS_INBOX": str(inbox), "ATTUNE_MODEL_NAME": "other", "ATTUNE_GPU_NAME": "B200",
             "TRAINING_IMAGE": "img@sha256:1"}   # and no ATTUNE_FRAMEWORK
    lessons.export(str(control), shell)
    manifest = json.loads((inbox / CAMPAIGN_ID / "manifest.json").read_text())
    assert (manifest["model"], manifest["gpu"], manifest["image"], manifest["framework"]) == (
        "m", "H100", "img@sha256:0", "slime 0.3")
    sealed_levers = json.loads((control / integrity.RECORD).read_text())["files"][
        str(control / "skill" / "references" / "levers.md")]
    assert manifest["catalog_version"] == {"levers_sha256": sealed_levers, "commit": None, "dirty": None}


def test_unset_settings_are_null(sealed):
    (sealed.root / "proj" / "LESSONS.md").write_text("## one\n")
    _export(sealed, sealed.tmp / "inbox", ATTUNE_GPU_NAME="H100")
    manifest = json.loads((sealed.tmp / "inbox" / sealed.campaign_id / "manifest.json").read_text())
    assert manifest["model"] is None and manifest["gpu"] is None and manifest["framework"] is None


def test_a_folder_gone_since_the_last_export_is_dropped(sealed):
    (sealed.root / "proj" / "LESSONS.md").write_text("## one\n")
    (sealed.root / "other").mkdir()
    (sealed.root / "other" / "LESSONS.md").write_text("## two\n")
    inbox = sealed.tmp / "inbox"
    _export(sealed, inbox)
    (sealed.root / "other" / "LESSONS.md").unlink()
    (sealed.root / "proj" / "LESSONS.md").write_text("## one, revised\n")
    _export(sealed, inbox)
    entry = inbox / sealed.campaign_id
    assert not (entry / "other").exists() and (entry / "proj" / "LESSONS.md").read_text() == "## one, revised\n"
    assert [c["folder"] for c in json.loads((entry / "manifest.json").read_text())["lessons"]] == ["proj"]


def test_without_an_inbox_or_after_a_violation_nothing_leaves(sealed):
    assert lessons.export(str(sealed.control), {}).startswith("lessons: none written")
    (sealed.root / "proj" / "LESSONS.md").write_text("## one\n")
    assert "stay in the campaign folder" in lessons.export(str(sealed.control), {})
    (sealed.skill / "SKILL.md").chmod(0o644)
    (sealed.skill / "SKILL.md").write_text("changed\n")
    assert integrity.check(str(sealed.control))
    inbox = sealed.tmp / "inbox"
    assert "failed its integrity check" in _export(sealed, inbox)
    assert not inbox.exists()


@pytest.mark.parametrize("where", ["relative", "campaign", "inside", "linked", "skill", "harness", "recorded"])
def test_an_inbox_in_the_campaign_or_a_trees_skill_or_harness_is_refused(campaign, where):
    tree, recorded = campaign.tmp / "tree", campaign.tmp / "recorded"
    for t in (tree, recorded):
        for d in ("skill", "harness"):
            (t / d).mkdir(parents=True)
    control = seal_campaign(campaign.root, {"CAMPAIGN_DIR": str(campaign.root), "ATTUNE_CAMPAIGN_ID": CAMPAIGN_ID},
                            source=str(recorded))
    (campaign.tmp / "link").symlink_to(campaign.root)
    inbox = {"relative": "inbox", "campaign": campaign.root, "inside": campaign.root / "inbox",
             "linked": campaign.tmp / "link" / "inbox", "skill": tree / "skill" / "inbox",
             "harness": tree / "harness", "recorded": recorded / "skill" / "x"}[where]
    why = lessons.inbox_refusal(str(inbox), str(campaign.root), [str(tree)])
    assert why, where
    assert lessons.main(["check-inbox", str(inbox), str(campaign.root), str(tree)]) == 2
    if where in ("inside", "recorded"):   # the export refuses it too
        (campaign.root / "proj" / "LESSONS.md").write_text("## one\n")
        with pytest.raises(ValueError, match="refused"):
            lessons.export(str(control), {"ATTUNE_LESSONS_INBOX": str(inbox)})


def test_an_inbox_beside_them_is_accepted(campaign):
    tree = campaign.tmp / "tree"
    (tree / "skill").mkdir(parents=True)
    for inbox in (campaign.tmp / "inbox", tree / "inbox", tree / "skillz"):
        assert lessons.inbox_refusal(str(inbox), str(campaign.root), [str(tree)]) == "", inbox
        assert lessons.main(["check-inbox", str(inbox), str(campaign.root), str(tree)]) == 0


def test_a_failed_copy_exits_1(sealed, capsys, monkeypatch):
    (sealed.root / "proj" / "LESSONS.md").write_text("## one\n")
    blocked = sealed.tmp / "file"
    blocked.write_text("")
    monkeypatch.setenv("ATTUNE_LESSONS_INBOX", str(blocked))
    assert lessons.main(["export", str(sealed.control)]) == 1
    assert "not exported" in capsys.readouterr().err
    assert lessons.main([str(sealed.control)]) == 2
