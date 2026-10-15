# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shell-command rules of the PreToolUse guard. They keep a campaign inside its cluster scope; they are not a
security boundary (namespace-scoped credentials are).

`check_command(command, env, cwd)` returns why a Bash command is refused, or None to allow it. Two rules:

(a) kubectl scope. Every kubectl invocation passes the campaign's --context and -n/--namespace, and no flag that
    changes the cluster or the identity (--kubeconfig, --server, --token, --as, ...). Changing verbs (apply, create,
    delete, patch, replace, scale, label, annotate, edit, set, rollout, cordon, drain, taint, and run, expose,
    autoscale, debug, uncordon, certificate) act only on objects named explicitly with names starting with
    $ATTUNE_RUN_PREFIX; with -f, only on existing manifest files under the campaign directory whose documents all have
    such a metadata.name, a namespaced kind and no other namespace. Cluster-scoped kinds are refused. exec, cp,
    attach and port-forward reach only pods named with the run prefix or $AUTOTUNE_LOG_POD. Mutating `kubectl config`
    subcommands are refused, and a plugin only runs with a read-only subcommand (get, list, logs, ...).
(b) curl and wget run only while web access is on: AUTOTUNE_ALLOW_WEB unset or 1 (the default); 0 turns it off.

Everything else is allowed. Invocations are found by joining backslash-newline continuations, splitting the line at
unquoted ;, |, &, &&, ||, newlines and parentheses, scanning each command substitution and each `bash -c` or `eval`
script as a line of its own, and tokenizing every segment with shlex. Heredoc bodies and here-strings are data and
are skipped, except the substitutions of an unquoted heredoc and the text a shell reads as its script (`bash <<EOF`,
`sh <<< '...'`). The program is the
first word after assignments, keywords and wrappers (env, sudo, timeout, xargs, watch, ...). A segment that mentions
kubectl but cannot be tokenized is refused only when it holds a changing verb or a context or namespace other than
the campaign's. Variables expand only for the campaign
settings ($AUTOTUNE_KUBECTL_CONTEXT, $KUBECTL_NS, $ATTUNE_RUN_PREFIX, $AUTOTUNE_LOG_POD) and only when the line does
not assign them. Runs on Python 3.7+ (the hook imports it).
"""

from __future__ import annotations

import os
import re
import shlex

SUB = "__ATTUNE_SUBSTITUTION__"   # stands in for a command substitution inside the words around it
MAX_DEPTH = 8

CHANGING = frozenset({"apply", "create", "delete", "patch", "replace", "scale", "label", "annotate", "edit", "set",
                      "rollout", "cordon", "uncordon", "drain", "taint", "run", "expose", "autoscale", "debug",
                      "certificate"})
POD_ACCESS = frozenset({"exec", "cp", "attach", "port-forward"})
READ = frozenset({"get", "describe", "logs", "top", "explain", "api-resources", "api-versions", "version",
                  "cluster-info", "events", "diff", "wait", "kustomize", "completion", "options", "help", "proxy",
                  "plugin"})
CONFIG_READ = frozenset({"view", "current-context", "get-contexts", "get-clusters", "get-users"})
PLUGIN_READ = frozenset({"get", "describe", "list", "ls", "log", "logs", "show", "status", "view", "top", "version",
                         "help"})
IDENTITY = frozenset({"--kubeconfig", "--cluster", "--user", "-s", "--server", "--token", "--as", "--as-group",
                      "--as-uid", "--certificate-authority", "--client-certificate", "--client-key",
                      "--tls-server-name", "--insecure-skip-tls-verify", "--username", "--password"})
# flags that take the next word as their value (the others are booleans or need --flag=value)
LONG_VALUES = frozenset({
    "--context", "--namespace", "--kubeconfig", "--cluster", "--user", "--server", "--token", "--as", "--as-group",
    "--as-uid", "--certificate-authority", "--client-certificate", "--client-key", "--tls-server-name",
    "--request-timeout", "--cache-dir", "--v", "--vmodule", "--log-file", "--log-file-max-size", "--username",
    "--password", "--profile", "--profile-output", "--filename", "--kustomize", "--selector", "--output",
    "--field-selector", "--sort-by", "--template", "--label-columns", "--chunk-size", "--raw", "--subresource",
    "--field-manager", "--container", "--patch", "--patch-file", "--type", "--timeout", "--for", "--tail", "--since",
    "--since-time", "--limit-bytes", "--max-log-requests", "--pod-running-timeout", "--replicas", "--current-replicas",
    "--resource-version", "--grace-period", "--image", "--port", "--env", "--labels", "--restart", "--overrides",
    "--target-port", "--protocol", "--name", "--external-ip", "--load-balancer-ip", "--session-affinity",
    "--cluster-ip", "--from", "--from-file", "--from-literal", "--from-env-file", "--docker-server",
    "--docker-username", "--docker-password", "--docker-email", "--cert", "--key", "--schedule", "--min", "--max",
    "--cpu-percent", "--to-revision", "--revision", "--address", "--retries", "--limits", "--requests", "--containers",
    "--serviceaccount", "--role", "--clusterrole", "--verb", "--resource", "--resource-name", "--group",
    "--pod-selector", "--target", "--copy-to", "--image-pull-policy", "--api-group", "--api-version", "--verbs",
    "--categories", "--prune-allowlist", "--max-age"})
SHORT_VALUES = "nsvfklocpL"
ROLLOUT_SUBS = frozenset({"history", "pause", "restart", "resume", "status", "undo"})
SET_SUBS = frozenset({"env", "image", "resources", "selector", "serviceaccount", "subject"})
CREATE_KINDS = {"configmap": "configmap", "cm": "configmap", "secret": "secret", "service": "service",
                "svc": "service", "deployment": "deployment", "deploy": "deployment", "job": "job",
                "cronjob": "cronjob", "cj": "cronjob", "namespace": "namespace", "ns": "namespace",
                "quota": "resourcequota", "resourcequota": "resourcequota", "role": "role",
                "rolebinding": "rolebinding", "clusterrole": "clusterrole", "clusterrolebinding": "clusterrolebinding",
                "poddisruptionbudget": "poddisruptionbudget", "pdb": "poddisruptionbudget",
                "priorityclass": "priorityclass", "pc": "priorityclass", "serviceaccount": "serviceaccount",
                "sa": "serviceaccount", "ingress": "ingress", "ing": "ingress", "token": "serviceaccount"}
CREATE_SUBTYPES = frozenset({"generic", "docker-registry", "tls", "clusterip", "nodeport", "loadbalancer",
                             "externalname"})
_CLUSTER = {  # cluster-scoped kinds (lowercase singular) and their short names
    "node": ("no",), "namespace": ("ns",), "persistentvolume": ("pv",), "storageclass": ("sc",),
    "clusterrole": (), "clusterrolebinding": (), "customresourcedefinition": ("crd", "crds"),
    "priorityclass": ("pc",), "certificatesigningrequest": ("csr",), "mutatingwebhookconfiguration": (),
    "validatingwebhookconfiguration": (), "validatingadmissionpolicy": (), "validatingadmissionpolicybinding": (),
    "mutatingadmissionpolicy": (), "mutatingadmissionpolicybinding": (), "apiservice": (), "runtimeclass": (),
    "ingressclass": (), "volumeattachment": (), "csidriver": (), "csinode": (), "componentstatus": ("cs",),
    "flowschema": (), "prioritylevelconfiguration": (), "clustertrustbundle": (), "ipaddress": (),
    "servicecidr": (), "deviceclass": (), "resourceslice": (), "storageversionmigration": (),
    "clusterqueue": ("cq",), "resourceflavor": ("rf", "flavor", "flavors"), "workloadpriorityclass": (),
    "cohort": (), "multikueuecluster": (), "multikueueconfig": (), "admissioncheck": (), "clusterpolicy": (),
}


def _forms(kind, short):
    plural = kind[:-1] + "ies" if kind.endswith("y") else kind + "es" if kind.endswith("s") else kind + "s"
    return {kind, plural} | set(short)


CLUSTER_SCOPED = frozenset(f for k, s in _CLUSTER.items() for f in _forms(k, s))
PINNED = ("AUTOTUNE_KUBECTL_CONTEXT", "KUBECTL_NS", "ATTUNE_RUN_PREFIX", "AUTOTUNE_LOG_POD")
SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})
KEYWORDS = frozenset({"if", "then", "elif", "else", "fi", "do", "done", "while", "until", "!", "{", "}", "time",
                      "for", "in", "case", "esac", "function", "select", "coproc"})
# programs whose arguments are data: a kubectl word among them is a mention, not a call
TEXT_TOOLS = frozenset({"echo", "printf", "grep", "egrep", "fgrep", "zgrep", "rg", "ag", "sed", "awk", "gawk", "cat",
                        "less", "more", "head", "tail", "wc", "which", "type", "man", "ls", "find", "git", "jq",
                        "yq", "sort", "uniq", "cut", "tr", "diff", "tee", "test", "[", "[[", "basename", "dirname",
                        "realpath", "readlink", "stat", "file", "touch", "mkdir", "rm", "cp", "mv", "ln", "chmod",
                        "true", "false", ":", "export", "unset", "local", "declare", "read", "cd", "pwd", "sleep",
                        "date", "python", "python3", "pip", "pip3"})
# wrapper -> its options that take a value; the program follows the options (and timeout's duration)
WRAPPERS = {"env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
            "sudo": {"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-T", "-U"},
            "command": set(), "builtin": set(), "exec": {"-a"}, "nohup": set(), "setsid": set(), "time": set(),
            "nice": {"-n", "--adjustment"}, "timeout": {"-s", "--signal", "-k", "--kill-after"},
            "stdbuf": {"-i", "-o", "-e"}, "unbuffer": set(), "ionice": {"-c", "-n", "-p"},
            "xargs": {"-n", "-P", "-L", "-s", "-d", "-E", "-I", "-a", "--max-args", "--max-procs", "--max-lines",
                      "--max-chars", "--delimiter", "--eof", "--replace", "--arg-file"},
            "flock": {"-w", "--timeout", "-E", "--conflict-exit-code"},
            "watch": {"-n", "--interval", "-q", "--equexit"}}
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")
_VAR = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_REDIR = re.compile(r"^\d*(?:&>>|&>|>>|>&|<&|<>|>\||<<<|<<-?|>|<)(.*)$", re.S)
_REMOTE = re.compile(r"^(?:([a-z0-9][-a-z0-9.]*)/)?([a-z0-9][-a-z0-9.]*):(.*)$", re.S)
_MENTION = re.compile(r"(?<![\w.-])(kubectl(?:-[\w-]+)?|curl|wget)(?![\w.-])")


class _Ctx:
    def __init__(self, env, command, cwd):
        self.context = env.get("AUTOTUNE_KUBECTL_CONTEXT", "")
        self.namespace = env.get("KUBECTL_NS", "")
        self.prefix = env.get("ATTUNE_RUN_PREFIX", "")
        self.log_pod = env.get("AUTOTUNE_LOG_POD", "")
        campaign = env.get("AUTOTUNE_PROJECT_ROOT") or env.get("CAMPAIGN_DIR")
        self.campaign = os.path.realpath(campaign) if campaign else None
        self.cwd = cwd or os.getcwd()
        self.web = env.get("AUTOTUNE_ALLOW_WEB", "") != "0"
        self.kubectl_line = bool(re.search(r"(?<![\w.-])kubectl", command))
        self.known = {n: env[n] for n in PINNED
                      if env.get(n) and not re.search(r"(?<![\w])%s\+?=" % n, command)}

    def value(self, word):
        """The word with the campaign settings expanded; None when another expansion remains."""
        unknown = []

        def rep(m):
            name = m.group(1) or m.group(2)
            if name in self.known:
                return self.known[name]
            unknown.append(name)
            return ""
        text = _VAR.sub(rep, word)
        if unknown or "$" in text or "`" in text or SUB in text:
            return None
        return text


# ---- splitting a line into segments --------------------------------------------------------------------------------

def _body(src, i):
    """(the text of the substitution starting at src[i], the index after it); ValueError when unbalanced."""
    if src[i] == "`":
        j = i + 1
        while j < len(src) and src[j] != "`":
            j += 2 if src[j] == "\\" else 1
        if j >= len(src):
            raise ValueError("unbalanced backtick")
        return src[i + 1:j], j + 1
    j, depth, quote = i + 2, 1, None
    while j < len(src):
        c = src[j]
        if quote == "'":
            if c == "'":
                quote = None
        elif c == "\\":
            j += 1
        elif c == '"':
            quote = None if quote == '"' else '"'
        elif quote is None and c == "'":
            quote = "'"
        elif quote is None and c == "(":
            depth += 1
        elif quote is None and c == ")":
            depth -= 1
            if depth == 0:
                return src[i + 2:j], j + 1
        j += 1
    raise ValueError("unbalanced $(")


_HEREDOC = re.compile(r"""<<(-?)[ \t]*(?:'([^'\n]*)'|"([^"\n]*)"|\\?([^\s;&|<>()'"]+))""")


def _scan(src):
    """(segments, substitution bodies, heredocs [(segment, body)]) of a shell line; ValueError on unbalanced quoting.
    A backslash-newline joins two lines; a heredoc body is not part of any segment."""
    segs, subs, docs, cur, pending = [], [], [], [], []
    i, n, quote = 0, len(src), None
    while i < n:
        c = src[i]
        if quote == "'":
            cur.append(c)
            quote = None if c == "'" else quote
            i += 1
        elif src.startswith("\\\n", i):
            i += 2
        elif c == "\\":
            cur.append(src[i:i + 2])
            i += 2
        elif c == "`" or (c == "$" and src.startswith("(", i + 1)) or (
                quote is None and c in "<>" and src.startswith("(", i + 1)):
            body, i = _body(src, i)
            subs.append(body)
            cur.append(SUB)
        elif quote is None and src.startswith("<<", i) and not src.startswith("<<<", i) and _HEREDOC.match(src, i):
            m = _HEREDOC.match(src, i)
            pending.append((m.group(2) if m.group(2) is not None else m.group(3) if m.group(3) is not None
                            else m.group(4), bool(m.group(1)), m.group(4) is not None))
            cur.append(m.group(0))
            i = m.end()
        elif c == '"':
            quote = None if quote == '"' else '"'
            cur.append(c)
            i += 1
        elif quote is None and c == "'":
            quote = "'"
            cur.append(c)
            i += 1
        elif quote is None and c == "#" and (not cur or cur[-1][-1:] in (" ", "\t", "\n")):
            while i < n and src[i] != "\n":
                i += 1
        elif quote is None and c == "&" and (src[i - 1:i] in ("<", ">") or src.startswith(">", i + 1)):
            cur.append(c)  # a redirection (>&, <&, &>), not a separator
            i += 1
        elif quote is None and c in ";&|\n()":
            segs.append("".join(cur))
            cur = []
            i += 1
            if c == "\n" and pending:   # the heredoc bodies start on the next line
                for delim, tabs, expands in pending:
                    body, i = _heredoc_body(src, i, delim, tabs)
                    docs.append((segs[-1], body))
                    if expands:   # an unquoted delimiter: $(...) and `...` in the body run
                        subs += _body_subs(body)
                pending = []
        else:
            cur.append(c)
            i += 1
    if quote:
        raise ValueError("unbalanced quote")
    segs.append("".join(cur))
    return [s for s in segs if s.strip()], subs, docs


def _heredoc_body(src, i, delim, tabs):
    """(the body from src[i] up to the line holding only `delim`, the index after that line)."""
    lines = []
    while i < len(src):
        end = src.find("\n", i)
        end = len(src) if end < 0 else end
        line = src[i:end]
        i = end + 1
        if (line.lstrip("\t") if tabs else line) == delim:
            break
        lines.append(line)
    return "\n".join(lines), i


def _body_subs(text):
    """The command substitutions in a heredoc body, where quotes are plain text."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "\\":
            i += 2
        elif text[i] == "`" or text.startswith("$(", i):
            try:
                body, i = _body(text, i)
            except ValueError:
                break
            out.append(body)
        else:
            i += 1
    return out


def _words(segment):
    """The segment's words without redirections; ValueError when shlex cannot split it."""
    segment = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"${\1}", segment)  # "$X"y keeps its boundary once unquoted
    words, out, skip = shlex.split(segment), [], False
    for w in words:
        if skip:
            skip = False
            continue
        m = _REDIR.match(w)
        if m:
            skip = not m.group(1)
            continue
        out.append(w)
    return out


def _program(words):
    """(index of the program word, or None; the command line a `watch` runs, or None)."""
    i = 0
    while i < len(words):
        w = words[i]
        name = os.path.basename(w)
        if _ASSIGN.match(w) or w in KEYWORDS:
            i += 1
        elif name in WRAPPERS:
            values, i = WRAPPERS[name], i + 1
            if name == "command" and i < len(words) and words[i] in ("-v", "-V"):
                return None, None
            while i < len(words) and (words[i].startswith("-") or (name == "env" and _ASSIGN.match(words[i]))):
                if words[i] == "--":
                    i += 1
                    break
                i += 2 if words[i] in values else 1
            if name == "timeout" and i < len(words):
                i += 1  # the duration
            if name == "flock" and i < len(words):
                i += 1  # the lock file
                if i < len(words) and words[i] in ("-c", "--command"):
                    return None, " ".join(words[i + 1:i + 2])
            if name == "watch":
                return None, " ".join(words[i:])  # watch runs its words joined by spaces with sh -c
        else:
            return i, None
    return None, None


# ---- the rules -------------------------------------------------------------------------------------------------------

def check_command(command, env=None, cwd=None):
    """Why the Bash `command` is refused, or None to allow it."""
    env = os.environ if env is None else env
    ctx = _Ctx(env, command or "", cwd)
    return _line(command or "", ctx, 0)


def _mentions(text):
    return bool(_MENTION.search(text.replace('"', "").replace("'", "")))


def _line(src, ctx, depth):
    if depth > MAX_DEPTH:
        return "commands nest too deeply to check"
    if not _mentions(src):
        return None
    try:
        segments, subs, docs = _scan(src)
    except ValueError:
        return _fallback(src, ctx)
    scripts = [body for seg, body in docs if _reads_script(seg)]
    scripts += [text for seg in segments if _reads_script(seg) for text in _herestrings(seg)]
    for text in subs + scripts:
        why = _line(text, ctx, depth + 1)
        if why:
            return why
    for seg in segments:
        why = _segment(seg, ctx, depth)
        if why:
            return why
    return None


def _herestrings(segment):
    """The here-string words of a segment (`<<< 'text'`, `<<<text`)."""
    try:
        words = shlex.split(re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"${\1}", segment))
    except ValueError:
        return []
    out = []
    for k, w in enumerate(words):
        m = re.match(r"^\d*<<<(.*)$", w, re.S)
        if m:
            out.append(m.group(1) or (words[k + 1] if k + 1 < len(words) else ""))
    return out


def _reads_script(segment):
    """True when the segment is a shell reading its script from stdin (`bash <<EOF`, `sh <<< '...'`): the heredoc
    body or here-string runs."""
    try:
        words = _words(segment)
    except ValueError:
        return False
    i, _ = _program(words)
    return i is not None and os.path.basename(words[i]) in SHELLS and not any(
        re.match(r"^-[a-z]*c[a-z]*$", w) for w in words[i + 1:])


def _segment(seg, ctx, depth):
    try:
        words = _words(seg)
    except ValueError:
        return _fallback(seg, ctx)
    i, watched = _program(words)
    if watched is not None:
        return _line(watched, ctx, depth + 1)
    if i is None:
        return None
    program, rest = words[i], words[i + 1:]
    if ctx.kubectl_line and ("$" in program or "`" in program or SUB in program):
        if any(w in CHANGING for w in rest):  # `$K delete ...` after K="kubectl ...": cannot be checked
            return "a changing verb through a variable or substitution cannot be checked; write kubectl plainly"
        return None
    if not _mentions(seg):
        return None
    name = os.path.basename(program)
    if name == "kubectl":
        return _kubectl(rest, ctx)
    if name.startswith("kubectl-"):
        return _kubectl(rest, ctx, plugin=name[len("kubectl-"):])
    if name in ("curl", "wget"):
        return None if ctx.web else "%s: web access is off (AUTOTUNE_ALLOW_WEB=0)" % name
    if name in SHELLS:
        for k, w in enumerate(rest):
            if re.match(r"^-[a-z]*c[a-z]*$", w) and k + 1 < len(rest):
                return _line(rest[k + 1], ctx, depth + 1)
        return None
    if name == "eval":
        return _line(" ".join(rest), ctx, depth + 1)
    if name in TEXT_TOOLS:
        return None
    for k, w in enumerate(rest):  # through a program the guard does not know: ssh host kubectl ..., retry kubectl ...
        if os.path.basename(w) in ("kubectl", "curl", "wget") or w.startswith("kubectl-"):
            return _segment(shlex_join(rest[k:]), ctx, depth + 1)
        if " " in w and _MENTION.search(w):
            why = _line(w, ctx, depth + 1)
            if why:
                return why
    return None


def shlex_join(words):
    return " ".join(shlex.quote(w) for w in words)


def _fallback(text, ctx):
    """A line or segment shlex cannot split: refuse only a changing verb, or a context or namespace other than the
    campaign's, after a kubectl word (and curl or wget while web access is off)."""
    if not ctx.web and re.search(r"(?<![\w.-])(curl|wget)(?![\w.-])", text):
        return "curl/wget: web access is off (AUTOTUNE_ALLOW_WEB=0)"
    for m in re.finditer(r"(?<![\w.-])kubectl(?![\w.-])([^;|&\n]*)", text):
        words = [w.strip("'\"") for w in m.group(1).split()]
        a = _parse(words)
        verb = a.pos[0] if a.pos else None
        if verb in CHANGING or any(w in CHANGING for w in words):
            return "kubectl: cannot parse this command, and it holds a changing verb; write it plainly"
        for flag, value in a.flags:
            want = ctx.context if flag == "--context" else ctx.namespace if flag in ("-n", "--namespace") else None
            if want is not None and value is not None and ctx.value(value) != want:
                return "kubectl: cannot parse this command, and it names %s %s, not the campaign's %s" % (
                    flag, value, want)
    return None


class _Args:
    def __init__(self):
        self.flags, self.pos, self.rest = [], [], None

    def values(self, *names):
        return [v for f, v in self.flags if f in names]

    def has(self, *names):
        return any(f in names for f, _ in self.flags)


def _parse(argv, bool_short=""):
    a, i = _Args(), 0
    while i < len(argv):
        w = argv[i]
        if w == "--":
            a.rest = argv[i + 1:]
            break
        if w.startswith("--") and len(w) > 2:
            name, eq, val = w.partition("=")
            if eq:
                a.flags.append((name, val))
            elif name in LONG_VALUES and i + 1 < len(argv):
                a.flags.append((name, argv[i + 1]))
                i += 1
            else:
                a.flags.append((name, None))
        elif w.startswith("-") and len(w) > 1:
            for j in range(1, len(w)):
                ch = w[j]
                if ch in SHORT_VALUES and ch not in bool_short:
                    val = w[j + 1:]
                    val = val[1:] if val.startswith("=") else val
                    if not val and i + 1 < len(argv):
                        val = argv[i + 1]
                        i += 1
                    a.flags.append(("-" + ch, val or None))
                    break
                a.flags.append(("-" + ch, None))
        else:
            a.pos.append(w)
        i += 1
    return a


def _kubectl(argv, ctx, plugin=None):
    a = _parse(argv)
    if a.pos[:1] == ["logs"]:
        a = _parse(argv, bool_short="p")  # logs -p is --previous
    for flag, _ in a.flags:
        if flag in IDENTITY:
            return "kubectl %s changes the cluster or the identity; use the campaign's --context" % flag
    if not ctx.context or not ctx.namespace:
        return "kubectl needs the campaign's AUTOTUNE_KUBECTL_CONTEXT and KUBECTL_NS"
    contexts = a.values("--context")
    if not contexts or any(ctx.value(v or "") != ctx.context for v in contexts):
        return "kubectl must pass --context %s (the campaign's context)" % ctx.context
    namespaces = a.values("-n", "--namespace")
    if not namespaces or any(ctx.value(v or "") != ctx.namespace for v in namespaces):
        return "kubectl must pass -n %s (the campaign's namespace)" % ctx.namespace
    if a.has("-h", "--help"):
        return None
    verb = plugin or (a.pos[0] if a.pos else None)
    ops = a.pos if plugin else a.pos[1:]
    if verb is None or verb in READ:
        return None
    if verb in CHANGING:
        return _changing(verb, ops, a, ctx)
    if verb in POD_ACCESS:
        return _pod_access(verb, ops, a, ctx)
    if verb == "auth":
        return _changing("auth reconcile", ops[1:], a, ctx) if ops[:1] == ["reconcile"] else None
    if verb == "config":
        if ops and ops[0] not in CONFIG_READ:
            return "kubectl config %s changes the kubeconfig; only %s are allowed" % (
                ops[0], ", ".join(sorted(CONFIG_READ)))
        return None
    sub = ops[0] if ops else None
    if sub in PLUGIN_READ and not any(o in CHANGING for o in ops):
        return None
    return "kubectl %s is a plugin or unknown verb; it runs only with a read-only subcommand (%s)" % (
        verb, ", ".join(sorted(PLUGIN_READ)))


def _cluster_scoped(kind):
    return kind.lower().split(".", 1)[0] in CLUSTER_SCOPED


def _is_assignment(word):
    return "=" in word or word.endswith("-") or ":" in word


def _targets(verb, ops):
    """(kind, name) pairs a changing verb names; kind may be None (from -f or create without a kind)."""
    if verb in ("cordon", "uncordon", "drain"):
        return [("node", o) for o in ops]
    if verb == "certificate":
        return [("certificatesigningrequest", o) for o in ops[1:]] or [("certificatesigningrequest", "")]
    ops = [o for o in ops if not _is_assignment(o)]
    if verb == "rollout" and ops[:1] and ops[0] in ROLLOUT_SUBS:
        ops = ops[1:]
    if verb == "set" and ops[:1] and ops[0] in SET_SUBS:
        ops = ops[1:]
    if verb == "run":
        return [("pod", ops[0])] if ops else []
    if verb == "debug":
        return [tuple(o.split("/", 1)) if "/" in o else ("pod", o) for o in ops[:1]]
    if verb == "create":
        kind = CREATE_KINDS.get(ops[0]) if ops else None
        rest = ops[1:]
        if rest[:1] and rest[0] in CREATE_SUBTYPES:
            rest = rest[1:]
        return [(kind, rest[0])] if kind and rest else [(kind or ops[0], "")] if ops else []
    out = [tuple(o.split("/", 1)) for o in ops if "/" in o]
    plain = [o for o in ops if "/" not in o]
    if plain:
        out += [(t, n) for n in plain[1:] for t in plain[0].split(",")] or [(t, "") for t in plain[0].split(",")]
    return out


def _changing(verb, ops, a, ctx):
    if not ctx.prefix:
        return "kubectl %s needs the campaign's ATTUNE_RUN_PREFIX" % verb
    if a.has("--all", "-A", "--all-namespaces", "--prune"):
        return "kubectl %s may not act on objects it does not name (--all, -A, --prune); name each object" % verb
    if a.has("-k", "--kustomize", "-R", "--recursive"):
        return "kubectl %s -k/-R is refused; pass each manifest file with -f" % verb
    files = a.values("-f", "--filename")
    if files:
        for f in files:
            why = _manifest(f, ctx)
            if why:
                return "kubectl %s -f: %s" % (verb, why)
        return None
    targets = [] if verb.startswith("auth ") else _targets(verb.split()[0], ops)
    selected = [v[len("metadata.name="):] for v in a.values("--field-selector")
                if v and v.startswith("metadata.name=") and "," not in v]
    if selected:
        kinds = {k for k, n in targets if not n} or {None}
        targets = [(k, n) for k, n in targets if n] + [(k, s) for k in kinds for s in selected]
    names = [(k, n) for k, n in targets if n]
    if not names:
        return ("kubectl %s must name its objects (TYPE NAME, TYPE/NAME or -f <manifest>); selectors alone are "
                "refused" % verb)
    for kind, name in names:
        if kind and _cluster_scoped(kind):
            return "kubectl %s on %s: cluster-scoped kinds are refused" % (verb, kind)
        value = ctx.value(name)
        if value is None or not value.startswith(ctx.prefix):
            return "kubectl %s acts only on names starting with %s (ATTUNE_RUN_PREFIX), not %s" % (
                verb, ctx.prefix, name)
    for extra in a.values("--name", "--copy-to"):
        value = ctx.value(extra or "")
        if value is None or not value.startswith(ctx.prefix):
            return "kubectl %s creates %s; names start with %s" % (verb, extra, ctx.prefix)
    return None


def _manifest(word, ctx):
    """Why the -f manifest is refused, or None."""
    path = ctx.value(word or "")
    if not path or path == "-" or "://" in path:
        return "needs a local manifest file under the campaign directory, not %s" % word
    full = os.path.realpath(os.path.join(ctx.cwd, os.path.expanduser(path)))
    if not ctx.campaign or not (full == ctx.campaign or full.startswith(ctx.campaign + "/")):
        return "%s is not under the campaign directory %s" % (path, ctx.campaign)
    if not os.path.isfile(full):
        return "%s is not an existing file" % path
    try:
        import yaml
    except ImportError:
        return "PyYAML is needed to read %s" % path
    try:
        with open(full, encoding="utf-8") as f:
            docs = [d for d in yaml.safe_load_all(f) if d is not None]
    except (OSError, ValueError, yaml.YAMLError) as e:
        return "cannot read %s: %s" % (path, e)
    items = []
    for d in docs:
        is_list = isinstance(d, dict) and str(d.get("kind", "")).endswith("List")
        items += (d.get("items") or []) if is_list else [d]
    if not items:
        return "%s holds no documents" % path
    for d in items:
        meta = d.get("metadata") if isinstance(d, dict) else None
        kind, name = (d.get("kind"), (meta or {}).get("name")) if isinstance(d, dict) else (None, None)
        if not isinstance(kind, str) or not kind:
            return "%s: a document has no kind" % path
        if _cluster_scoped(kind):
            return "%s: %s is cluster-scoped" % (path, kind)
        if not isinstance(name, str) or not name.startswith(ctx.prefix):
            return "%s: %s %r does not start with %s (ATTUNE_RUN_PREFIX)" % (path, kind, name, ctx.prefix)
        if (meta or {}).get("namespace") not in (None, ctx.namespace):
            return "%s: %s %s is in namespace %s, not %s" % (path, kind, name, meta["namespace"], ctx.namespace)
    return None


def _pod_name_ok(name, ctx):
    value = ctx.value(name)
    return value is not None and bool(value) and (value == ctx.log_pod or bool(ctx.prefix) and value.startswith(
        ctx.prefix))


def _pod_access(verb, ops, a, ctx):
    allowed = "pods named %s* or the log pod %s" % (ctx.prefix, ctx.log_pod or "(AUTOTUNE_LOG_POD unset)")
    if verb == "cp":
        for op in ops:
            text = ctx.value(op)
            if text is None:
                if ":" in op:
                    return "kubectl cp: cannot resolve %s; cp reaches only %s" % (op, allowed)
                continue
            m = _REMOTE.match(text)
            if not m or text.startswith(("/", ".", "~")):
                continue
            if m.group(1) and m.group(1) != ctx.namespace:
                return "kubectl cp: %s is in namespace %s, not %s" % (op, m.group(1), ctx.namespace)
            if not _pod_name_ok(m.group(2), ctx):
                return "kubectl cp reaches only %s, not %s" % (allowed, m.group(2))
        return None
    if not ops:
        return "kubectl %s must name the pod" % verb
    name = ops[0].split("/", 1)[1] if "/" in ops[0] else ops[0]
    if not _pod_name_ok(name, ctx):
        return "kubectl %s reaches only %s, not %s" % (verb, allowed, ops[0])
    return None
