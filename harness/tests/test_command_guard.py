# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""command_guard.classify_command denies every bypass found in review and allows the commands the skill needs.
Placeholders: {camp} campaign directory, {shared} protected volume, {skill} skill directory, {scripts} its scripts.
Run: python3 -m pytest harness/tests"""
import pytest

import command_guard as guard

CTX = "--context c -n ns"
DENIED = [
    # command chains, wrappers and interpreters
    "echo ok; kubectl delete deployment web --context c -n ns",
    "ls && kubectl delete rayjob bob-run --context c -n ns",
    "bash -c 'kubectl delete rayjob x --context c -n ns'",
    "find {shared} -delete",
    "ls {shared} | xargs rm -rf",
    "python3 -c \"import shutil; shutil.rmtree('{shared}/ckpt')\"",
    "awk 'BEGIN{{system(\"rm -rf {shared}\")}}'",
    "bash <<'EOF'\nrm -rf {shared}\nEOF",
    'echo x > "{shared}/data"',
    "echo x >| {shared}/data",
    "echo x | tee {shared}/data",
    "sudo rm -rf / # poll_run.sh",
    "git push origin main",
    "git clean -fdx",
    "git -c core.pager='rm -rf /' log",
    'cu""rl https://example.com',
    "/usr/bin/curl https://example.com",
    "wget -O /tmp/page https://example.com",
    "kubectl --context get delete rayjob x -n ns",
    "kubectl apply --prune --all -f {camp}/x.yaml --context c -n ns",
    # ownership by substring: every direct delete is denied
    "kubectl delete jobs -l owner=alicex,preemptible=true --context c -n ns",
    "kubectl delete rayjob x --context c -n ns # owner=alice",
    "kubectl delete jobs,deployments,pvc -l owner=alice --context c -n ns",
    "kubectl delete jobs --context c -n ns -l owner=alice # teardown_run.sh",
    "echo hi; rm -rf {shared}/experiments",
    # kubectl scope
    "kubectl get pods",
    "kubectl --context other -n ns get pods",
    "kubectl --context c -n other get pods",
    "kubectl --context c --context other -n ns get pods",
    f"kubectl {CTX} --kubeconfig /tmp/k get pods",
    "kubectl --context c -n ns patch rayjob x -p '{{}}'",
    f"kubectl {CTX} create configmap x --from-file=y",
    f"kubectl {CTX} apply -f -",
    f"kubectl {CTX} apply -k dir",
    f"kubectl {CTX} apply -f https://example.com/x.yaml",
    f"kubectl {CTX} apply --force -f x.yaml",
    f"kubectl {CTX} apply -f {{camp}}/proj",
    f"kubectl {CTX} exec otherpod -- cat /x",
    f"kubectl {CTX} exec -it logpod -- sh",
    f"kubectl {CTX} exec logpod -- rm -rf {{shared}}/x",
    f"kubectl {CTX} exec logpod -- sh -c 'cat x > {{shared}}/y'",
    f"kubectl {CTX} exec logpod -- sh -c 'ls; rm x'",
    f"kubectl {CTX} cp local.txt logpod:{{shared}}/x",
    f"kubectl {CTX} cp other/logpod:/x /tmp/y",
    f"kubectl {CTX} cp logpod:/x {{shared}}/y",
    f"xargs kubectl {CTX} delete pod",
    # rm and writes
    "rm -rf /tmp/x",
    "rm -rf {camp}",
    "rm {camp}/.claude/settings.json",
    "rm -rf {camp}/.cl*",
    "rm -rf $(echo {shared})",
    "rm -rf `echo {shared}`",
    "rm -rf ~",
    "cd {shared} && rm -rf x",
    "D={shared}; rm -rf $D",
    "D={camp}/x; bash -c 'rm -rf $D/y'",
    "echo x > {camp}/proj/AUTOTUNE_LEDGER.jsonl",
    "cp x {shared}/y",
    "cp -t {shared} x",
    "mv {shared}/a /tmp/b",
    "mkdir -p {shared}/new",
    "touch ~/.kube/config",
    "sort -o {shared}/x f",
    "tee {skill}/references/levers.md",
    "echo rule > {skill}/references/levers.md",
    "cp rules.md {skill}/references/levers.md",
    "echo x >> {skill}/references/measurement.md",
    "echo x >> {skill}/SKILL.md",
    "echo x > {skill}/past_tuning_projects/README.md",
    "mkdir -p {skill}/past_tuning_projects/sub && echo x > {skill}/past_tuning_projects/sub/p_summary.md",
    "echo x > {skill}/past_tuning_projects/_summary.md",
    "echo x 1<> {skill}/references/levers.md",
    "echo x 1<> {camp}/proj/AUTOTUNE_LEDGER.jsonl",
    "echo x > /etc/passwd",
    "cat x &> {shared}/y",
    # unsupported syntax and programs
    "eval ls", "source x.sh", ". x.sh", "exec ls", "f() {{ rm -rf /; }}; f", "cat <(ls)",
    "case x in a) ls;; esac", "echo 'unbalanced", "ls |",
    "sed -i s/a/b/ f", "sed 'w {shared}/x' f", "sed 's/a/b/e' f", "find . -exec rm {{}} ;",
    "awk '{{print > \"out\"}}' f", "rg --pre cat x",
    "python3", "python3 -", "python3 -m http.server", "python3 /tmp/evil.py", "bash /tmp/x.sh", "bash",
    "PATH=/tmp/evil:$PATH kubectl --context c -n ns get pods",
    "AUTOTUNE_KUBECTL_CONTEXT=other bash {scripts}/poll_run.sh r",
    "for PATH in /tmp; do ls; done",
    "ssh host ls", "rsync -a a b", "pip install x", "chmod 777 x",
]
ALLOWED = [
    "kubectl -n ns --context c get pods",
    "kubectl --context c -n ns logs pod/x",
    "cd /tmp && ls",
    "mkdir -p {camp}/x",
    "sleep 60",
    "bash {scripts}/poll_run.sh alice-m-r1",
    "python3 {scripts}/make_run_yaml.py --base a.yaml --run-id r --out b.yaml --benchmark",
    "kubectl --context c -n ns exec logpod -- tail -n 50 /path/log",
    "kubectl --context=c --namespace=ns get rayjobs -o json | jq -r '.items[].metadata.name'",
    'kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" get pods',
    "kubectl -nns --context c get pods",
    f"kubectl {CTX} logs -f pod/x --tail 100",
    f"kubectl {CTX} wait --for=condition=Ready pod/x --timeout=60s",
    f"kubectl {CTX} auth can-i create rayjobs",
    f"kubectl {CTX} apply -f {{camp}}/proj/r.yaml",
    f"python3 {{scripts}}/make_run_yaml.py --out x.yaml && kubectl {CTX} apply -f x.yaml",
    f'kubectl {CTX} exec logpod -- sh -c "ls {{shared}}/exp/*.pt.trace.json.gz"',
    f"kubectl {CTX} exec logpod -- find /x -name '*.gz'",
    f"kubectl {CTX} cp ns/logpod:{{shared}}/exp/r/x.gz /tmp/r/x.gz",
    (f"D={{shared}}/exp/r; mkdir -p /tmp/r\nfor f in $(kubectl {CTX} exec logpod -- sh -c \"ls $D/*.gz\"); do\n"
     f'  kubectl {CTX} exec logpod -- cat "$f" > /tmp/r/$(basename "$f")\ndone'),
    "echo '{{\"run_id\": \"r\"}}' >> {camp}/proj/AUTOTUNE_LEDGER.jsonl",
    "cd {camp} && cd proj && echo x >> AUTOTUNE_LEDGER.jsonl",
    "rm -f {camp}/proj/tmp.yaml",
    "rm -rf proj/scratch",
    "grep -v r1 proj/tracked_jobs.jsonl > proj/t.tmp && mv proj/t.tmp proj/tracked_jobs.jsonl",
    "grep -c perf log 2>/dev/null || true",
    'for f in a b; do echo "$f"; done',
    "if [ -f x ]; then cat x; elif true; then :; else echo n; fi",
    'while read -r l; do echo "$l"; done < file',
    "awk -F, '{{print $1}}' f", "sed -n '1,20p;/perf/p' f", "sed -E 's#a/b#c#g' f",
    "git status", "git log --oneline -5", "git -C /tmp diff HEAD~1",
    'D=/tmp/x; mkdir -p "$D"',
    "timeout 60 bash {scripts}/poll_run.sh r",
    "env TARGET_STEPS=19 bash {scripts}/poll_run.sh r",
    "TARGET_STEPS=19 bash {scripts}/poll_run.sh r",
    "bash .claude/skills/autotune-training-throughput/scripts/teardown_run.sh r",
    "flock {camp}/.state.lock bash {scripts}/poll_run.sh r",
    "nohup python3 {scripts}/analyze_profile.py --profile-dir /tmp/r > /tmp/a.log 2>&1 &",
    "ls | xargs -n1 basename",
    "date -u +%Y-%m-%dT%H:%M:%SZ",
    "echo $((1+2)) ${{HOME:-x}} | tee /dev/stderr",
    "set -euo pipefail; ls \\\n  -la  # a comment",
    "(cd proj && ls) > /tmp/out",
    "cat {shared}/exp/train.log | head",
    "echo '- a rule' >> {skill}/references/levers.md",
    "tee -a .claude/skills/autotune-training-throughput/references/levers.md < proj/rule.md",
    "cp proj/RESULTS.md {skill}/past_tuning_projects/proj-h100_summary.md",
    "echo x > .claude/skills/autotune-training-throughput/past_tuning_projects/proj-h100_summary.md",
]


def _fill(cmd: str, campaign) -> str:
    return cmd.format(camp=campaign.root, shared=campaign.shared, skill=guard.SKILL_DIR, scripts=guard.SCRIPTS_DIR)


@pytest.mark.parametrize("cmd", DENIED)
def test_denied(cmd, campaign):
    d = guard.classify_command(_fill(cmd, campaign), cwd=str(campaign.root))
    assert not d.allow, d.reason


@pytest.mark.parametrize("cmd", ALLOWED)
def test_allowed(cmd, campaign):
    d = guard.classify_command(_fill(cmd, campaign), cwd=str(campaign.root))
    assert d.allow, d.reason


def test_the_agent_runs_its_own_aids_from_the_campaign_directory(campaign):
    (campaign.root / "proj" / "aid.py").write_text("print(1)\n")
    (campaign.root / "proj" / "aid.sh").write_text("echo 1\n")
    (campaign.shared / "x.py").write_text("print(1)\n")
    cwd = str(campaign.root)
    for cmd in ("python3 proj/aid.py --top 5", f"python3 -u {campaign.root}/proj/aid.py", "bash proj/aid.sh",
                "cd proj && python3 aid.py", "timeout 600 python3 proj/aid.py > proj/out.txt 2>&1"):
        assert guard.classify_command(cmd, cwd=cwd).allow, cmd
    for cmd in ("python3 proj/missing.py", f"python3 {campaign.shared}/x.py", "python3 -m proj.aid",
                "python3 - < proj/aid.py", "python3 -c 'import os'", "bash -s < proj/aid.sh", "bash ../shared/x.py",
                "cd /tmp && python3 aid.py", "proj/aid.py"):
        d = guard.classify_command(cmd, cwd=cwd)
        assert not d.allow, cmd
    assert "campaign directory" in guard.classify_command("python3 /tmp/x.py", cwd=cwd).reason


def test_reasons_name_the_rule(campaign):
    assert "teardown_run.sh" in guard.classify_command("kubectl --context c -n ns delete rayjob x").reason
    assert "--context c" in guard.classify_command("kubectl -n ns get pods").reason
    assert "append-only" in guard.classify_command(f"echo > {campaign.root}/p/AUTOTUNE_LEDGER.jsonl").reason


def test_cd_in_a_subshell_or_after_a_failure_does_not_move_later_commands(campaign):
    cwd = str(campaign.shared)
    assert not guard.classify_command(f"(cd {campaign.root}); rm -rf x", cwd=cwd).allow
    assert not guard.classify_command(f"cd {campaign.root} || rm -rf x", cwd=cwd).allow
    assert not guard.classify_command(f"cd {campaign.root} && true; rm -rf x", cwd=cwd).allow
    assert guard.classify_command(f"cd {campaign.root} && rm -rf x", cwd=cwd).allow


def test_web_switch(campaign, monkeypatch):
    fetch = "curl -sSL https://example.com -o /tmp/page.html"
    assert not guard.classify_command(fetch).allow
    monkeypatch.setenv("AUTOTUNE_ALLOW_WEB", "1")
    assert guard.classify_command(fetch).allow
    assert guard.classify_command("wget -q -O - https://example.com").allow
    for cmd in ("curl -X POST https://example.com", "curl -d a=b https://example.com", "curl -T f https://x",
                f"curl -o {campaign.shared}/x https://example.com", "curl http://169.254.169.254/latest",
                "wget --post-data=a https://example.com -O -", "ssh host ls"):
        assert not guard.classify_command(cmd).allow, cmd


def test_campaign_directory_inside_a_protected_path(campaign, monkeypatch):
    monkeypatch.setenv("AUTOTUNE_PROTECTED_PATHS", str(campaign.root.parent))
    assert guard.classify_command(f"echo x >> {campaign.root}/proj/AUTOTUNE_LEDGER.jsonl").allow
    assert not guard.classify_command(f"echo x > {campaign.shared}/f").allow
    assert not guard.classify_command(f"echo x > {campaign.root}/.claude/settings.json").allow


def test_kubectl_needs_the_campaign_settings(campaign, monkeypatch):
    monkeypatch.delenv("AUTOTUNE_KUBECTL_CONTEXT")
    assert "AUTOTUNE_KUBECTL_CONTEXT" in guard.classify_command("kubectl --context c -n ns get pods").reason
    monkeypatch.setenv("AUTOTUNE_KUBECTL_CONTEXT", "c")
    monkeypatch.delenv("AUTOTUNE_LOG_POD")
    assert not guard.classify_command("kubectl --context c -n ns exec logpod -- ls").allow


def test_rm_needs_a_campaign_directory(campaign, monkeypatch):
    monkeypatch.delenv("AUTOTUNE_PROJECT_ROOT")
    assert not guard.classify_command("rm -f /tmp/x").allow


def test_an_internal_error_denies(campaign, monkeypatch):
    def boom(*_):
        raise RuntimeError("bug")
    monkeypatch.setattr(guard._Guard, "walk", boom)
    d = guard.classify_command("ls")
    assert not d.allow and "RuntimeError" in d.reason


def test_empty_command_denied():
    assert not guard.classify_command("  ").allow


def test_write_block_reason(campaign):
    assert guard.write_block_reason(f"{campaign.shared}/x") is not None
    assert guard.write_block_reason(f"{guard.SKILL_DIR}/SKILL.md") is not None
    assert guard.write_block_reason(".claude/skills/autotune-training-throughput/SKILL.md", str(campaign.root))
    assert guard.write_block_reason("~/.ssh/config") is not None
    assert guard.write_block_reason("proj/RESULTS.md", str(campaign.root)) is None
    assert guard.write_block_reason("proj/AUTOTUNE_LEDGER.jsonl", str(campaign.root)) is not None
    assert guard.write_block_reason("proj/AUTOTUNE_LEDGER.jsonl", str(campaign.root), append=True) is None
