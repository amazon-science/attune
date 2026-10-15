# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""command_guard.check_command: kubectl stays in the campaign's context, namespace and runs, curl and wget follow the
web switch, and every other command runs. Each denied command sits beside an allowed neighbour.
Placeholders: {camp} campaign directory, {other} a directory outside it.
Run: python3 -m pytest harness/tests"""
import pytest

import command_guard as guard

K = "kubectl --context c -n ns"
PAIRS = [  # (allowed, denied)
    # context and namespace on every invocation
    (f"{K} get pods", "kubectl get pods"),
    (f"{K} get pods", "kubectl --context other -n ns get pods"),
    (f"{K} get pods", "kubectl --context c -n other get pods"),
    (f"{K} get pods", "kubectl --context c --context other -n ns get pods"),
    ("kubectl -nns --context=c get pods", "kubectl -nother --context=c get pods"),
    ('kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n "$KUBECTL_NS" get pods',
     'kubectl --context "$CTX" -n "$KUBECTL_NS" get pods'),
    ('kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n ns get pods',
     'export AUTOTUNE_KUBECTL_CONTEXT=other; kubectl --context "$AUTOTUNE_KUBECTL_CONTEXT" -n ns get pods'),
    (f"{K} get pods -o json | jq .items", f"{K} --kubeconfig /tmp/k get pods"),
    (f"{K} get pods", f"{K} --token abc get pods"),
    (f"{K} get pods", f"{K} --as system:admin get pods"),
    (f"{K} version", "kubectl version --client"),
    # changing verbs: explicit names with the run prefix
    (f"{K} delete rayjob alice-tune-r1", f"{K} delete rayjob bob-r1"),
    (f"{K} delete rayjob/alice-tune-r1 configmap/alice-tune-r1", f"{K} delete rayjob/alice-tune-r1 configmap/web"),
    (f"{K} delete pod,service alice-tune-a alice-tune-b", f"{K} delete pod,service alice-tune-a web"),
    (f"{K} delete rayjob --field-selector metadata.name=alice-tune-r1 -l owner=alice",
     f"{K} delete rayjob -l owner=alice"),
    (f"{K} delete rayjob alice-tune-r1", f"{K} delete rayjob --all"),
    (f"{K} delete rayjob alice-tune-r1", f"{K} delete rayjobs -A"),
    (f'{K} delete rayjob "$ATTUNE_RUN_PREFIX"r1', f'{K} delete rayjob "$RUN"'),
    (f"{K} patch rayjob alice-tune-r1 --type merge -p '{{\"spec\": {{}}}}'", f"{K} patch rayjob bob -p '{{}}'"),
    (f"{K} label rayjob alice-tune-r1 tier=x", f"{K} label rayjob bob tier=x"),
    (f"{K} annotate rayjob alice-tune-r1 note-", f"{K} annotate rayjob bob note-"),
    (f"{K} scale deployment alice-tune-d --replicas 0", f"{K} scale deployment web --replicas 0"),
    (f"{K} rollout restart deployment/alice-tune-d", f"{K} rollout restart deployment/web"),
    (f"{K} set image deployment/alice-tune-d c=img", f"{K} set image deployment/web c=img"),
    (f"{K} edit rayjob alice-tune-r1", f"{K} edit rayjob bob"),
    (f"{K} replace -f {{camp}}/proj/run.yaml", f"{K} replace -f {{camp}}/proj/other.yaml"),
    (f"{K} create configmap alice-tune-cm --from-file=a=x", f"{K} create configmap web --from-file=a=x"),
    (f"{K} create secret generic alice-tune-s --from-literal=a=b", f"{K} create namespace alice-tune-ns"),
    (f"{K} run alice-tune-p --image=busybox -- sleep 1", f"{K} run p --image=busybox"),
    (f"{K} expose deployment alice-tune-d --port 80 --name alice-tune-svc",
     f"{K} expose deployment alice-tune-d --port 80 --name web"),
    (f"{K} debug alice-tune-r1-head -it --image=busybox", f"{K} debug node/gpu-1 -it --image=busybox"),
    (f"{K} get nodes", f"{K} cordon gpu-1"),
    (f"{K} describe node gpu-1", f"{K} drain gpu-1 --ignore-daemonsets"),
    (f"{K} get nodes", f"{K} taint nodes gpu-1 k=v:NoSchedule"),
    (f"{K} get clusterqueues", f"{K} delete clusterqueue alice-tune-q"),
    (f"{K} get pv", f"{K} delete pv alice-tune-pv"),
    (f"{K} delete rayjob alice-tune-r1", f"xargs {K} delete rayjob"),
    # -f: existing manifests under the campaign directory, named with the prefix, namespaced
    (f"{K} apply -f {{camp}}/proj/run.yaml", f"{K} apply -f {{camp}}/proj/missing.yaml"),
    (f"{K} apply -f proj/run.yaml", f"{K} apply -f {{other}}/run.yaml"),
    (f"{K} create -f {{camp}}/proj/run.yaml", f"{K} create -f {{camp}}/proj/other.yaml"),
    (f"{K} delete -f {{camp}}/proj/run.yaml", f"{K} delete -f {{camp}}/proj/cluster.yaml"),
    (f"{K} apply -f {{camp}}/proj/run.yaml", f"{K} apply -f {{camp}}/proj/elsewhere.yaml"),
    (f"{K} apply -f {{camp}}/proj/run.yaml", f"{K} apply -f {{camp}}/proj/list.yaml"),
    (f"{K} apply -f {{camp}}/proj/run.yaml", f"{K} apply -f {{camp}}/proj"),
    (f"{K} apply -f {{camp}}/proj/run.yaml", f"{K} apply -f -"),
    (f"{K} apply -f {{camp}}/proj/run.yaml", f"{K} apply -f https://example.com/x.yaml"),
    (f"{K} apply -f {{camp}}/proj/run.yaml", f"{K} apply -k {{camp}}/proj"),
    (f"{K} apply -f {{camp}}/proj/run.yaml", f"{K} apply -f {{camp}}/proj/run.yaml --prune -l a=b"),
    (f"{K} diff -f {{camp}}/proj/other.yaml", f"{K} apply -f {{camp}}/proj/other.yaml"),
    # exec, cp, attach, port-forward: run pods and the log pod
    (f"{K} exec logpod -- rm -rf /shared/x", f"{K} exec otherpod -- ls"),
    (f"{K} exec -it alice-tune-r1-head -- bash", f"{K} exec -it web-0 -- bash"),
    (f'{K} exec "$AUTOTUNE_LOG_POD" -- cat /x', f'{K} exec "$POD" -- cat /x'),
    (f"{K} exec pod/logpod -c main -- ls", f"{K} exec deploy/web -- ls"),
    (f"{K} cp logpod:/shared/x /tmp/x", f"{K} cp web-0:/x /tmp/x"),
    (f"{K} cp ns/logpod:/shared/x ./x", f"{K} cp other/logpod:/x ./x"),
    (f"{K} cp ./x alice-tune-r1-head:/tmp/x", f"{K} cp ./x web-0:/tmp/x"),
    (f"{K} attach alice-tune-r1-head", f"{K} attach web-0"),
    (f"{K} port-forward alice-tune-r1-head 8265:8265", f"{K} port-forward svc/web 8080:80"),
    (f"{K} logs web-0 --tail 100", f"{K} exec web-0 -- tail /log"),
    # kubectl config and plugins
    (f"{K} config view", f"{K} config use-context other"),
    (f"{K} config current-context", f"{K} config set-context c --namespace other"),
    (f"{K} ray get clusters", f"{K} ray delete cluster alice-tune-x"),
    (f"{K} ray logs alice-tune-x", f"{K} tree rayjob alice-tune-x"),
    ("kubectl-ray --context c -n ns get clusters", "kubectl-ray get clusters"),
    (f"{K} auth can-i create rayjobs", f"{K} auth reconcile -f {{camp}}/proj/cluster.yaml"),
    # finding the invocations
    (f"echo ok; {K} get deploy", f"echo ok; {K} delete deployment web"),
    (f"ls && {K} delete rayjob alice-tune-r1", f"ls && {K} delete rayjob bob-r1"),
    (f"false || {K} get pods", f"false || {K} delete pod web-0"),
    (f"{K} get pods | grep x", f"{K} get pods | {K} delete pod web-0"),
    (f"bash -c '{K} get pods'", f"bash -c '{K} delete rayjob bob'"),
    (f"sh -lc \"{K} get pods\"", "sh -lc \"kubectl get pods\""),
    (f"eval '{K} get pods'", "eval 'kubectl delete rayjob bob'"),
    (f"for p in $({K} get pods -o name); do echo $p; done", "for p in $(kubectl get pods -o name); do :; done"),
    (f'echo "$({K} get pods)"', f'echo "$({K} --context other get pods)"'),
    (f"echo `{K} get pods`", f"echo `{K} delete pod web-0`"),
    (f"(cd /tmp && {K} get pods)", f"(cd /tmp && {K} delete pod web-0)"),
    (f"if true; then {K} get pods; fi", f"if true; then {K} delete pod web-0; fi"),
    (f"timeout 60 {K} get pods", "timeout 60 kubectl get pods"),
    (f"env A=1 {K} get pods", "env -i A=1 kubectl get pods"),
    (f"watch -n 5 {K} get pods", "watch -n 5 'kubectl get pods'"),
    ("/usr/local/bin/kubectl --context c -n ns get pods", "/usr/local/bin/kubectl get pods"),
    ("ssh host true", "ssh host kubectl delete pod web-0"),
    (f'K="{K}"; $K get pods', f'K="{K}"; $K delete pod web-0'),
    (f"{K} get pods 2>&1 | head -n 5", f"{K} delete pod web-0 2>/dev/null"),
    (f"{K} get pods # then delete web-0", f"{K} delete pod web-0 # alice-tune-"),
    # a segment shlex cannot split: refused only for a changing verb or another context or namespace
    (f"echo 'unbalanced; {K} get pods", f"echo 'unbalanced; {K} delete rayjob alice-tune-r1"),
    (f"echo 'unbalanced; {K} get pods", "echo 'unbalanced; kubectl --context other -n ns get pods"),
    # heredoc bodies are data; a shell reading one as its script runs it, and an unquoted one runs its substitutions
    ("python3 s/ledger_append.py proj/AUTOTUNE_LEDGER.jsonl - <<'EOF'\n"
     '{"hypothesis": "kubectl delete rayjob bob frees its GPUs; don\'t wait"}\nEOF',
     "bash <<'EOF'\nkubectl --context c -n ns delete rayjob bob\nEOF"),
    ("cat > proj/RESULTS.md <<'EOF'\nran kubectl delete pod web-0 and curl https://x; it's done\nEOF",
     "cat > proj/RESULTS.md <<EOF\n$(kubectl --context c -n ns delete pod web-0)\nEOF"),
    ("cat <<-EOF > proj/notes.md\n\tkubectl delete x\n\tEOF\nls",
     f"cat <<-EOF > proj/notes.md\n\tkubectl delete x\n\tEOF\n{K} delete pod web-0"),
    ('cat > proj/a.md <<"END"\nkubectl --context other get pods\nEND',
     "sh <<END\nkubectl --context other get pods\nEND"),
    ("cat <<< 'kubectl delete pod web-0' > proj/note.md", "bash <<< 'kubectl --context c -n ns delete pod web-0'"),
    (f"bash <<< '{K} get pods'", f"sh <<<\"{K} delete rayjob bob\""),
    ("grep x <<< 'kubectl --context other get pods'", "bash -s <<< 'kubectl --context other get pods'"),
    # backslash-newline continuations
    ("kubectl --context c -n ns \\\n  exec logpod -- \\\n  tail -n 50 /x",
     "kubectl --context c -n ns \\\n  exec web-0 -- \\\n  tail -n 50 /x"),
    ("kubectl --context c -n ns apply \\\n  -f {camp}/proj/run.yaml",
     "kubectl --context c -n ns apply \\\n  -f {camp}/proj/other.yaml"),
    ("kubectl \\\n  --context c -n ns get pods", "kubectl \\\n  --context other -n ns get pods"),
    # mentions are not invocations
    ("grep -n 'kubectl delete' SKILL.md", "kubectl delete -f x"),
    ("echo kubectl --context x delete everything", "kubectl --context x delete everything"),
    ("which kubectl", "kubectl"),
]
ALWAYS_ALLOWED = [
    "ls -la", "rm -rf proj/scratch", "python3 -c 'import os'", "pip install x", "git status", "cat ~/.bashrc",
    "echo x > /tmp/y", "sed -i s/a/b/ proj/notes.md", "bash proj/aid.sh", "kill 1234", "cd /tmp && ls",
    "cat <<'EOF' > proj/notes.md\nsome notes\nEOF",
]


@pytest.fixture
def manifests(campaign):
    proj = campaign.root / "proj"
    (proj / "run.yaml").write_text("apiVersion: ray.io/v1\nkind: RayJob\nmetadata:\n  name: alice-tune-r1\n---\n"
                                   "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: alice-tune-r1-params\n"
                                   "  namespace: ns\n")
    (proj / "other.yaml").write_text("kind: RayJob\nmetadata:\n  name: bob-r1\n")
    (proj / "cluster.yaml").write_text("kind: ClusterRole\nmetadata:\n  name: alice-tune-role\n")
    (proj / "elsewhere.yaml").write_text("kind: RayJob\nmetadata:\n  name: alice-tune-r2\n  namespace: other\n")
    (proj / "list.yaml").write_text("kind: List\nitems:\n- kind: RayJob\n  metadata:\n    name: alice-tune-r3\n"
                                    "- kind: Namespace\n  metadata:\n    name: alice-tune-ns\n")
    other = campaign.tmp / "elsewhere"
    other.mkdir()
    (other / "run.yaml").write_text((proj / "run.yaml").read_text())
    return campaign


def _check(cmd, campaign, **env):
    cmd = cmd.replace("{camp}", str(campaign.root)).replace("{other}", str(campaign.tmp / "elsewhere"))
    return guard.check_command(cmd, dict(campaign.env, **env), str(campaign.root))


@pytest.mark.parametrize("allowed,denied", PAIRS)
def test_pairs(allowed, denied, manifests):
    assert _check(allowed, manifests) is None, allowed
    why = _check(denied, manifests)
    assert why, denied


@pytest.mark.parametrize("cmd", ALWAYS_ALLOWED)
def test_everything_else_runs(cmd, campaign):
    assert _check(cmd, campaign) is None


def test_reasons_name_the_rule(manifests):
    assert "--context c" in _check("kubectl -n ns get pods", manifests)
    assert "-n ns" in _check("kubectl --context c get pods", manifests)
    assert "alice-tune-" in _check(f"{K} delete rayjob bob", manifests)
    assert "cluster-scoped" in _check(f"{K} delete node gpu-1", manifests)
    assert "log pod logpod" in _check(f"{K} exec web-0 -- ls", manifests)
    assert "read-only subcommand" in _check(f"{K} ray delete cluster x", manifests)
    assert "does not start with alice-tune-" in _check(f"{K} apply -f {{camp}}/proj/other.yaml", manifests)


def test_web_access_is_on_unless_turned_off(campaign):
    for cmd in ("curl -sSL https://example.com -o /tmp/page.html", "wget -qO- https://example.com",
                "curl -X POST -d a=b https://example.com"):
        assert _check(cmd, campaign) is None, cmd
        assert _check(cmd, campaign, AUTOTUNE_ALLOW_WEB="1") is None, cmd
        assert "web access is off" in _check(cmd, campaign, AUTOTUNE_ALLOW_WEB="0"), cmd
    off = {"AUTOTUNE_ALLOW_WEB": "0"}
    for cmd in ("ls; curl https://x", "bash -c 'wget https://x'", "echo $(curl -s https://x)", "/usr/bin/curl x",
                'cu""rl https://x', "timeout 9 curl https://x", "echo 'unbalanced; curl https://x"):
        assert _check(cmd, campaign, **off), cmd
    for cmd in ("grep -n curl notes.md", "echo wget", "pip download x",
                "cat > proj/RESULTS.md <<'EOF'\nfetched with curl https://x\nEOF"):
        assert _check(cmd, campaign, **off) is None, cmd


def test_kubectl_needs_the_campaign_settings(campaign):
    env = dict(campaign.env)
    del env["AUTOTUNE_KUBECTL_CONTEXT"]
    assert "AUTOTUNE_KUBECTL_CONTEXT" in guard.check_command(f"{K} get pods", env, str(campaign.root))
    env = dict(campaign.env, ATTUNE_RUN_PREFIX="")
    assert "ATTUNE_RUN_PREFIX" in guard.check_command(f"{K} delete rayjob x", env, str(campaign.root))
    assert guard.check_command(f"{K} exec x -- ls", dict(campaign.env, AUTOTUNE_LOG_POD="", ATTUNE_RUN_PREFIX=""),
                               str(campaign.root))


def test_nesting_is_bounded(campaign):
    cmd = f"{K} get pods"
    for _ in range(12):
        cmd = "bash -c %s" % guard.shlex.quote(cmd)
    assert "nest" in _check(cmd, campaign)
