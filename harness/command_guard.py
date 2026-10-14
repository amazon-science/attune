# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deny-by-default screen for the shell commands of an unattended campaign.

`classify_command(command, cwd)` splits a Bash command line into simple commands, recursing into
command substitutions, `bash -c` scripts and the wrappers `env`, `timeout`, `nice`, `nohup`, `time`,
`flock` and `xargs`. The line is allowed only when every simple command passes the allowlist:
read-only tools, the skill's scripts, `python3|bash <file>` for a file in the campaign directory (the agent's own
analysis aids), scoped `kubectl` reads, `kubectl apply|create -f`, read-only `kubectl exec|cp` on the log pod,
read-only `git`, and file writes outside the protected paths. The skill directory is protected except for the two
records a campaign adds to it: lever rules appended to references/levers.md and past_tuning_projects/*_summary.md.
Syntax it does not model (here-documents, process substitution, `eval`, `case`, functions) is denied.
`write_block_reason(path, cwd)` is the path rule shared with the file-edit tools.

The guard prevents accidents; it is not a security boundary.
"""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SKILL_DIR = os.path.join(ROOT, "skill")
HARNESS_DIR = os.path.join(ROOT, "harness")
SCRIPTS_DIR = os.path.join(SKILL_DIR, "scripts")
LEVERS = os.path.realpath(os.path.join(SKILL_DIR, "references", "levers.md"))
SUMMARIES_DIR = os.path.realpath(os.path.join(SKILL_DIR, "past_tuning_projects"))
LEDGER_NAME = "AUTOTUNE_LEDGER.jsonl"


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str


class Deny(Exception):
    """Raised anywhere in the classifier; becomes a DENY decision."""


# ---- lexer: words keep their quoting so that expansions and globs are known ----------------------------------

@dataclass(frozen=True)
class Lit:
    text: str
    quoted: bool = False


@dataclass(frozen=True)
class Var:
    name: str  # "" when the expansion has no static value ($1, ${x:-y}, $((...)), ~user)


@dataclass(frozen=True)
class Sub:
    tokens: tuple  # tokens of a command substitution


@dataclass(frozen=True)
class Word:
    parts: tuple
    raw: str

    @property
    def literal(self) -> str | None:
        """The word's value when it contains no expansion."""
        if all(isinstance(p, Lit) for p in self.parts):
            return "".join(p.text for p in self.parts)
        return None

    @property
    def prefix(self) -> str:
        """The literal text before the first expansion."""
        out = []
        for p in self.parts:
            if not isinstance(p, Lit):
                break
            out.append(p.text)
        return "".join(out)

    def drop(self, n: int) -> Word:
        """The word without its first n literal characters (n <= len(prefix))."""
        parts = list(self.parts)
        while n and parts:
            if len(parts[0].text) <= n:
                n -= len(parts.pop(0).text)
            else:
                parts[0], n = Lit(parts[0].text[n:], parts[0].quoted), 0
        return Word(tuple(parts), self.raw)

    def unquoted(self, chars: str) -> bool:
        return any(isinstance(p, Lit) and not p.quoted and any(c in p.text for c in chars) for p in self.parts)


@dataclass(frozen=True)
class Op:
    value: str


@dataclass(frozen=True)
class Redir:
    op: str
    fd: str


_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TILDE = re.compile(r"~([A-Za-z0-9._+-]*)")
_OPS = ("&&", "||", "|&", ";;", ";&", ";", "|", "&", "(", ")", "\n")
_REDIRS = ("&>>", "&>", ">>", ">|", ">&", "<&", "<>", ">", "<")
_WORD_END = " \t\n;&|()<>"


class _Lexer:
    """Splits Bash source into Word, Op and Redir tokens."""

    def __init__(self, src: str, pos: int = 0):
        self.s, self.i = src, pos
        self.tokens: list = []
        self.parts: list = []
        self.start: int | None = None  # start of the current word

    def run(self, nested: bool = False) -> list:
        """Tokens up to the end of input, or (nested) up to the `)` closing a `$(`."""
        depth = 0
        while self.i < len(self.s):
            c = self.s[self.i]
            if self.s.startswith("\\\n", self.i):
                self.i += 2
            elif c in " \t":
                self._end_word()
                self.i += 1
            elif c == "#" and self.start is None:
                while self.i < len(self.s) and self.s[self.i] != "\n":
                    self.i += 1
            elif c == ")" and nested and depth == 0:
                self._end_word()
                self.i += 1
                return self.tokens
            elif c in "<>" or self.s.startswith("&>", self.i):
                self._redirect()
            elif c in "\n;&|()":
                depth += self._op()
            else:
                self._word_char(c)
        if nested:
            raise Deny("unbalanced $(")
        self._end_word()
        return self.tokens

    def _end_word(self) -> None:
        if self.start is not None:
            self.tokens.append(Word(tuple(self.parts), self.s[self.start:self.i]))
            self.parts, self.start = [], None

    def _lit(self, text: str, quoted: bool) -> None:
        if self.parts and isinstance(self.parts[-1], Lit) and self.parts[-1].quoted == quoted:
            self.parts[-1] = Lit(self.parts[-1].text + text, quoted)
        else:
            self.parts.append(Lit(text, quoted))

    def _op(self) -> int:
        self._end_word()
        op = next(o for o in _OPS if self.s.startswith(o, self.i))
        if op in (";;", ";&"):
            raise Deny("case statements are not supported")
        self.tokens.append(Op(op))
        self.i += len(op)
        return (op == "(") - (op == ")")

    def _redirect(self) -> None:
        fd = ""
        if self.start is not None:
            word = Word(tuple(self.parts), "")
            if (word.literal or "").isdigit() and not any(p.quoted for p in self.parts):
                fd, self.parts, self.start = word.literal, [], None
            else:
                self._end_word()
        if self.s.startswith("<<", self.i):
            raise Deny("here-documents and here-strings are not supported")
        if self.s.startswith(("<(", ">("), self.i):
            raise Deny("process substitution is not supported")
        op = next(r for r in _REDIRS if self.s.startswith(r, self.i))
        self.tokens.append(Redir(op, fd))
        self.i += len(op)

    def _word_char(self, c: str) -> None:
        if self.start is None:
            self.start = self.i
            if c == "~":
                return self._tilde()
        if c == "'":
            self._single()
        elif c == '"':
            self._double()
        elif c == "$":
            self._dollar(quoted=False)
        elif c == "`":
            self._backtick()
        elif c == "\\":
            self._lit(self.s[self.i + 1:self.i + 2], True)
            self.i += 2
        else:
            self._lit(c, False)
            self.i += 1

    def _tilde(self) -> None:
        m = _TILDE.match(self.s, self.i)
        nxt = self.s[m.end():m.end() + 1]
        if nxt and nxt not in "/" + _WORD_END:
            self._lit("~", False)
            self.i += 1
            return
        self.parts.append(Var("" if m.group(1) else "HOME"))
        self.i = m.end()

    def _single(self) -> None:
        end = self.s.find("'", self.i + 1)
        if end < 0:
            raise Deny("unbalanced quote")
        self._lit(self.s[self.i + 1:end], True)
        self.i = end + 1

    def _double(self) -> None:
        self.i += 1
        self._lit("", True)
        while True:
            if self.i >= len(self.s):
                raise Deny("unbalanced quote")
            c = self.s[self.i]
            if c == '"':
                self.i += 1
                return
            if c == "\\" and self.s[self.i + 1:self.i + 2] in ('"', "\\", "$", "`", "\n"):
                self._lit("" if self.s[self.i + 1] == "\n" else self.s[self.i + 1], True)
                self.i += 2
            elif c == "$":
                self._dollar(quoted=True)
            elif c == "`":
                self._backtick()
            else:
                self._lit(c, True)
                self.i += 1

    def _dollar(self, quoted: bool) -> None:
        s, i = self.s, self.i
        end = self._arith_end(i + 3) if s.startswith("$((", i) else None
        if end is not None:
            if "$(" in s[i + 3:end] or "`" in s[i + 3:end]:
                raise Deny("command substitution inside $((...)) is not supported")
            self.parts.append(Var(""))
            self.i = end
        elif s.startswith("$(", i):
            inner = _Lexer(s, i + 2)
            self.parts.append(Sub(tuple(inner.run(nested=True))))
            self.i = inner.i
        elif s.startswith("${", i):
            close = s.find("}", i + 2)
            body = s[i + 2:close]
            if close < 0 or "$(" in body or "`" in body or "{" in body:
                raise Deny("unsupported ${...} expansion")
            self.parts.append(Var(body if _NAME.fullmatch(body) else ""))
            self.i = close + 1
        elif not quoted and s.startswith("$'", i):
            self._ansi_c()
        elif not quoted and s.startswith('$"', i):
            self.i += 1
            self._double()
        elif m := _NAME.match(s, i + 1):
            self.parts.append(Var(m.group()))
            self.i = m.end()
        elif s[i + 1:i + 2] and s[i + 1] in "0123456789@*#?$!-":
            self.parts.append(Var(""))
            self.i += 2
        else:
            self._lit("$", quoted)
            self.i += 1

    def _arith_end(self, j: int) -> int | None:
        """End of `$((...))` arithmetic, or None when the parentheses open a subshell instead."""
        depth = 2
        while j < len(self.s):
            if self.s[j] == "(":
                depth += 1
            elif self.s[j] == ")":
                depth -= 1
                if depth == 1:
                    return j + 2 if self.s[j + 1:j + 2] == ")" else None
            j += 1
        return None

    def _ansi_c(self) -> None:
        j = self.i + 2
        while j < len(self.s) and self.s[j] != "'":
            j += 2 if self.s[j] == "\\" else 1
        if j >= len(self.s):
            raise Deny("unbalanced quote")
        body = self.s[self.i + 2:j]
        if "\\" in body:  # escapes can spell any character, so the value is treated as unknown
            self.parts.append(Var(""))
        else:
            self._lit(body, True)
        self.i = j + 1

    def _backtick(self) -> None:
        j, buf = self.i + 1, []
        while True:
            if j >= len(self.s):
                raise Deny("unbalanced backtick")
            if self.s[j] == "`":
                break
            if self.s[j] == "\\" and self.s[j + 1:j + 2] in ("`", "\\", "$"):
                buf.append(self.s[j + 1])
                j += 2
            else:
                buf.append(self.s[j])
                j += 1
        self.parts.append(Sub(tuple(_Lexer("".join(buf)).run())))
        self.i = j + 1


# ---- parser: simple commands with the operators around them ---------------------------------------------------

@dataclass
class _Command:
    words: list
    redirs: list
    before: str = ";"
    after: str = ";"


_PUSH, _POP = object(), object()  # subshell markers


def _parse(tokens: list) -> list:
    steps, cur, prev, depth, i = [], _Command([], []), ";", 0, 0

    def flush(op: str) -> None:
        nonlocal cur, prev
        if cur.words or cur.redirs:
            cur.before, cur.after = prev, op
            steps.append(cur)
        cur, prev = _Command([], []), op

    while i < len(tokens):
        t = tokens[i]
        if isinstance(t, Word):
            cur.words.append(t)
        elif isinstance(t, Redir):
            if i + 1 >= len(tokens) or not isinstance(tokens[i + 1], Word):
                raise Deny("redirection without a target")
            cur.redirs.append((t, tokens[i + 1]))
            i += 1
        elif t.value == "(":
            if cur.words or cur.redirs:
                raise Deny("function definitions are not supported")
            depth += 1
            steps.append(_PUSH)
        elif t.value == ")":
            if depth == 0:
                raise Deny("unbalanced )")
            flush(")")
            depth -= 1
            steps.append(_POP)
        else:
            flush(t.value)
        i += 1
    if depth:
        raise Deny("unbalanced (")
    if prev in ("|", "|&", "&&", "||") and not (cur.words or cur.redirs):
        raise Deny(f"incomplete command after {prev}")
    flush(";")
    return steps


def _steps(source: str) -> list:
    return _parse(_Lexer(source).run())


# ---- option parsing shared by the program rules ---------------------------------------------------------------

@dataclass
class _Args:
    flags: list            # (flag, value Word or None)
    operands: list
    rest: list | None      # words after `--`

    def values(self, *names: str) -> list:
        return [v for f, v in self.flags if f in names]

    def has(self, *names: str) -> bool:
        return any(f in names for f, _ in self.flags)

    @property
    def all_operands(self) -> list:
        return self.operands + (self.rest or [])


def _opts(args: list, short_values: str = "", long_values=frozenset(), stop_at_operand: bool = False) -> _Args:
    """Split getopt-style arguments; `-xVALUE`, `-x VALUE`, `--name=VALUE` and `--name VALUE` are understood."""
    flags, operands, i = [], [], 0
    while i < len(args):
        w = args[i]
        head = w.prefix
        if w.literal == "--":
            return _Args(flags, operands, list(args[i + 1:]))
        if len(head) < 2 or not head.startswith("-"):
            if stop_at_operand:
                return _Args(flags, list(args[i:]), None)
            operands.append(w)
        elif head.startswith("--"):
            name, eq, _ = head.partition("=")
            if eq:
                flags.append((name, w.drop(len(name) + 1)))
            elif w.literal is None:
                raise Deny(f"cannot parse the option {w.raw}")
            elif name in long_values:
                if i + 1 >= len(args):
                    raise Deny(f"{name} needs a value")
                flags.append((name, args[i + 1]))
                i += 1
            else:
                flags.append((name, None))
        else:
            i = _short_cluster(w, head, args, i, short_values, flags)
        i += 1
    return _Args(flags, operands, None)


def _short_cluster(w: Word, head: str, args: list, i: int, short_values: str, flags: list) -> int:
    for k in range(1, len(head)):
        flag = "-" + head[k]
        if head[k] in short_values:
            value = w.drop(k + 1)
            if value.prefix.startswith("="):
                value = value.drop(1)
            if value.parts and value.literal != "":
                flags.append((flag, value))
                return i
            if i + 1 >= len(args):
                raise Deny(f"{flag} needs a value")
            flags.append((flag, args[i + 1]))
            return i + 1
        flags.append((flag, None))
    if w.literal is None:
        raise Deny(f"cannot parse the option {w.raw}")
    return i


def _assignment(w: Word) -> tuple[str, Word] | None:
    first = w.parts[0] if w.parts else None
    if isinstance(first, Lit) and not first.quoted and (m := re.match(r"([A-Za-z_][A-Za-z0-9_]*)\+?=", first.text)):
        return m.group(1), w.drop(m.end())
    return None


def _keyword(w: Word) -> str | None:
    return w.parts[0].text if len(w.parts) == 1 and isinstance(w.parts[0], Lit) and not w.parts[0].quoted else None


# ---- policy ---------------------------------------------------------------------------------------------------

_SYSTEM_DIRS = ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/sbin", "/sys", "/usr")
_DEVICES = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty")
_HOME_DIRS = (".kube", ".aws", ".ssh", ".config/gh", ".claude")
_BIN_DIRS = ("/bin", "/usr/bin", "/usr/local/bin")


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _skill_record(path: str, append: bool) -> bool:
    """An append to the lever catalog, or a campaign's closeout summary: the only writes into the skill."""
    if path == LEVERS:
        return append
    name = os.path.basename(path)
    return os.path.dirname(path) == SUMMARIES_DIR and name.endswith("_summary.md") and name != "_summary.md"


@dataclass(frozen=True)
class Policy:
    """The campaign settings the guard enforces, read from the environment."""

    env: Mapping[str, str]
    context: str
    namespace: str
    log_pod: str
    allow_web: bool
    campaign: str | None
    protected: tuple   # AUTOTUNE_PROTECTED_PATHS; writes inside the campaign directory are exempt
    fixed: tuple       # credentials, skill, harness, campaign settings and system directories

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Policy:
        env = os.environ if env is None else env
        home = env.get("HOME") or os.path.expanduser("~")
        campaign = env.get("CAMPAIGN_DIR") or env.get("AUTOTUNE_PROJECT_ROOT")
        campaign = os.path.realpath(campaign) if campaign else None
        spec = env.get("AUTOTUNE_PROTECTED_PATHS") or env.get("AUTOTUNE_SHARED_MOUNT") or ""
        protected = [os.path.join(home, p[2:]) if p.startswith("~/") else p for p in spec.split(":") if p]
        fixed = [os.path.join(home, d) for d in _HOME_DIRS] + [SKILL_DIR, HARNESS_DIR]
        fixed += [env["CLAUDE_CONFIG_DIR"]] if env.get("CLAUDE_CONFIG_DIR") else []
        fixed += [os.path.join(campaign, ".claude")] if campaign else []
        return cls(env, env.get("AUTOTUNE_KUBECTL_CONTEXT", ""), env.get("KUBECTL_NS", ""),
                   env.get("AUTOTUNE_LOG_POD", ""), env.get("AUTOTUNE_ALLOW_WEB") == "1", campaign,
                   tuple(os.path.realpath(p) for p in protected),
                   tuple(os.path.realpath(p) for p in fixed) + _SYSTEM_DIRS)

    def write_block(self, path: str, append: bool = False) -> str | None:
        """Why writing the resolved absolute `path` is refused, or None."""
        if path in _DEVICES or path.startswith("/dev/fd/"):
            return None
        roots = self.fixed
        if not (self.campaign and _under(path, self.campaign)):
            roots += self.protected
        if _skill_record(path, append):
            roots = tuple(r for r in roots if r != os.path.realpath(SKILL_DIR))
        for root in roots:
            if _under(path, root):
                return f"{path} is under the write-protected {root}"
        if not append and os.path.basename(path) == LEDGER_NAME:
            return f"{path} is append-only (use >>)"
        return None


def write_block_reason(path: str, cwd: str | None = None, *, append: bool = False) -> str | None:
    """Why the file-edit tools may not write `path` (relative to `cwd`), or None."""
    full = os.path.join(cwd or os.getcwd(), os.path.expanduser(path))
    return Policy.from_env().write_block(os.path.realpath(full), append)


# ---- checker --------------------------------------------------------------------------------------------------

_MAX_DEPTH = 8
_PIPES = ("|", "|&")
_LEADING = {"if", "then", "elif", "else", "do", "while", "until", "!", "{", "time"}
_CLOSING = {"fi", "done", "}"}
_UNSUPPORTED = {"case", "esac", "select", "function", "coproc", "in"}
_SCRIPT = "<skill script>"
_READ_ONLY = {
    "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "zgrep", "zcat", "wc", "cut", "tr", "jq", "du", "df",
    "stat", "sleep", "echo", "pwd", "true", "false", ":", "test", "[", "[[", "diff", "cmp", "md5sum", "sha256sum",
    "basename", "dirname", "realpath", "readlink", "seq", "tac", "paste", "column", "nl", "comm", "join", "which",
    "type", "wait", "exit", "set",
}
_POD_READ = {"cat", "ls", "head", "tail", "grep", "find", "du", "stat", "test", "[", "wc", "zcat", "md5sum",
             "sha256sum", "cd", "true", "false"}
_XARGS_OK = {"cat", "grep", "egrep", "fgrep", "zgrep", "head", "tail", "wc", "ls", "stat", "du", "md5sum",
             "sha256sum", "echo", "basename", "dirname", "realpath", "zcat", "jq"}
_SETTINGS_VARS = {"PATH", "HOME", "IFS", "ENV", "BASH_ENV", "SHELLOPTS", "BASHOPTS", "PROMPT_COMMAND", "CDPATH",
                  "KUBECONFIG", "KUBECTL_NS", "CAMPAIGN_DIR", "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE",
                  "PYTHONSTARTUP", "PYTHONPLATLIBDIR"}
_FIND_ACTIONS = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
_RM_FLAGS = {"-f", "-r", "-R", "-i", "-I", "-v", "-d", "--force", "--recursive", "--dir", "--verbose",
             "--interactive", "--one-file-system", "--preserve-root"}
_FLOCK_FLAGS = {"-s", "-x", "-e", "-n", "-o", "-u", "-F", "--shared", "--exclusive", "--nonblock", "--nb",
                "--close", "--unlock", "--no-fork", "--verbose"}
_GIT_READ = {"status", "log", "diff", "show", "rev-parse", "ls-files", "grep"}
_PY_FLAGS = {"-u", "-B", "-I", "-E", "-s", "-O", "-OO"}
_SED_FLAGS = {"-n", "--quiet", "--silent", "-E", "-r", "--regexp-extended", "-s", "--separate", "-u",
              "--unbuffered", "-z", "--null-data", "--posix", "--debug", "--sandbox", "-l", "--line-length",
              "-e", "--expression"}

_K_GLOBAL_VALUES = frozenset({
    "--context", "-n", "--namespace", "--kubeconfig", "--cluster", "--user", "-s", "--server", "--token", "--as",
    "--as-group", "--as-uid", "--certificate-authority", "--client-certificate", "--client-key", "--tls-server-name",
    "--request-timeout", "--cache-dir", "-v", "--v", "--vmodule", "--log-file", "--username", "--password",
    "--profile", "--profile-output"})
_K_GLOBAL_BOOLS = frozenset({"--insecure-skip-tls-verify", "--match-server-version", "--warnings-as-errors",
                             "--disable-compression"})
_K_IDENTITY = frozenset({
    "--kubeconfig", "--cluster", "--user", "-s", "--server", "--token", "--as", "--as-group", "--as-uid",
    "--certificate-authority", "--client-certificate", "--client-key", "--tls-server-name",
    "--insecure-skip-tls-verify", "--username", "--password"})
_K_READ = frozenset({"get", "describe", "logs", "top", "explain", "api-resources", "version", "auth", "wait"})
# Per verb: short and long flags that take a value, beyond the global ones.
_K_VALUES = {
    "get": ("lofLk", {"--selector", "--output", "--filename", "--label-columns", "--field-selector", "--sort-by",
                      "--template", "--chunk-size", "--raw", "--subresource", "--kustomize"}),
    "describe": ("lfk", {"--selector", "--filename", "--kustomize", "--chunk-size"}),
    "logs": ("cl", {"--container", "--selector", "--since", "--since-time", "--tail", "--limit-bytes",
                    "--max-log-requests", "--pod-running-timeout"}),
    "top": ("l", {"--selector", "--sort-by"}),
    "explain": ("o", {"--output", "--api-version"}),
    "api-resources": ("o", {"--output", "--api-group", "--verbs", "--sort-by", "--categories"}),
    "version": ("o", {"--output"}),
    "auth": ("o", {"--output", "--subresource"}),
    "wait": ("lfo", {"--selector", "--filename", "--for", "--timeout", "--output", "--field-selector"}),
    "apply": ("fo", {"--filename", "--output", "--field-manager"}),
    "create": ("fo", {"--filename", "--output", "--field-manager"}),
    "exec": ("cf", {"--container", "--filename", "--pod-running-timeout"}),
    "cp": ("c", {"--container", "--retries"}),
}
_K_SCOPE = {"--context", "-n", "--namespace", "--request-timeout", "-v", "--v"}
_K_APPLY_FLAGS = _K_SCOPE | {"-f", "--filename", "-o", "--output", "--dry-run", "--server-side", "--field-manager",
                             "--validate", "--save-config"}
_K_EXEC_FLAGS = _K_SCOPE | {"-c", "--container", "-q", "--quiet", "--pod-running-timeout"}
_K_CP_FLAGS = _K_SCOPE | {"-c", "--container", "--retries", "--no-preserve"}

_CURL_FLAGS = {"-s", "-S", "-L", "-f", "-I", "-G", "-k", "-v", "-i", "--silent", "--show-error", "--location",
               "--fail", "--fail-with-body", "--head", "--get", "--insecure", "--verbose", "--include",
               "--compressed", "--create-dirs", "-H", "--header", "-A", "--user-agent", "-e", "--referer", "-m",
               "--max-time", "--connect-timeout", "--retry", "--retry-delay", "--retry-max-time", "-w",
               "--write-out", "-r", "--range", "--url"}
_CURL_VALUES = {"--output", "--header", "--user-agent", "--referer", "--max-time", "--connect-timeout", "--retry",
                "--retry-delay", "--retry-max-time", "--write-out", "--range", "--request", "--url"}
_WGET_FLAGS = {"-q", "--quiet", "--no-verbose", "-T", "--timeout", "-t", "--tries", "--header", "-U",
               "--user-agent", "--no-check-certificate", "-S", "--server-response", "--spider"}

# sed commands that only read, print or edit the pattern space (no w, W, e or s///w|e)
_SED_SIMPLE = set("pdqQ=lnNgGhHxDPzF{}!")


def _sed_read_only(script: str) -> bool:
    i, n = 0, len(script)
    while i < n:
        c = script[i]
        if c in " \t\n;" or c.isdigit() or c in "$,~+":
            i += 1
        elif c in "/\\":
            delim = "/" if c == "/" else script[i + 1:i + 2]
            i = _sed_field(script, i + (1 if c == "/" else 2), delim)
            while i < n and script[i] in "IM":
                i += 1
        elif c in "sy":
            delim = script[i + 1:i + 2]
            if not delim or delim in "\\\n":
                return False
            i = _sed_field(script, _sed_field(script, i + 2, delim), delim)
            while i < n and script[i] in "gpiImM0123456789":
                i += 1
            if i < n and script[i] in "wWe":
                return False
        elif c in "aicrRbtT:#":  # text, file name or label: runs to the end of the line (labels: to ;)
            stop = "\n;" if c in "btT:" else "\n"
            while i < n and script[i] not in stop:
                i += 1
        elif c in _SED_SIMPLE:
            i += 1
        else:
            return False
    return True


def _sed_field(script: str, i: int, delim: str) -> int:
    """Index after the field that starts at i and ends at the next unescaped delim."""
    while i < len(script) and script[i] != delim:
        i += 2 if script[i] == "\\" else 1
    if i >= len(script):
        raise Deny("cannot parse the sed script")
    return i + 1


@dataclass
class _Ctx:
    cwds: frozenset           # possible working directories (None = unknown)
    vars: dict                # shell variables set on this line (None = unknown value)
    exported: frozenset = frozenset()
    pod: bool = False         # inside `kubectl exec`: the command runs in the log pod
    depth: int = 0

    def child(self, shell: bool = False, pod: bool = False) -> _Ctx:
        """Context for a command substitution, a new shell (`bash -c`) or a command in the log pod."""
        if self.depth >= _MAX_DEPTH:
            raise Deny("commands nest too deeply")
        if pod:
            return _Ctx(frozenset({None}), {}, frozenset(), True, self.depth + 1)
        variables = {k: v for k, v in self.vars.items() if k in self.exported} if shell else dict(self.vars)
        return replace(self, vars=variables, depth=self.depth + 1)


class _Guard:
    def __init__(self, policy: Policy):
        self.p = policy
        self.handlers = {
            "rm": self._rm, "mv": self._mv, "cp": self._cp, "mkdir": self._mkdir, "touch": self._touch,
            "tee": self._tee, "sort": self._sort, "uniq": self._uniq, "sed": self._sed, "awk": self._awk,
            "gawk": self._awk, "find": self._find, "rg": self._rg, "date": self._date, "printf": self._printf,
            "read": self._read, "export": self._export, "unset": self._unset, "git": self._git,
            "kubectl": self._kubectl, "python": self._python, "python3": self._python, "bash": self._shell,
            "sh": self._shell, "env": self._env, "timeout": self._timeout, "nice": self._nice,
            "nohup": self._nohup, "time": self._time, "xargs": self._xargs, "flock": self._flock,
            "curl": self._curl, "wget": self._wget,
        }

    # -- walking ----------------------------------------------------------------------------------------------

    def walk(self, steps: list, ctx: _Ctx) -> None:
        saved, fallback = [], frozenset()  # fallback: directories a failed `cd X &&` leaves later commands in
        for step in steps:
            if step is _PUSH:
                saved.append((ctx.cwds, dict(ctx.vars), ctx.exported, fallback))
                continue
            if step is _POP:
                ctx.cwds, ctx.vars, ctx.exported, fallback = saved.pop()
                continue
            if step.before not in ("&&",) + _PIPES:
                ctx.cwds, fallback = ctx.cwds | fallback, frozenset()
            target = self._command(step, ctx)
            if target is None or step.before in _PIPES or step.after in _PIPES + ("&",):
                continue
            if step.after == "&&":
                ctx.cwds, fallback = target, fallback | ctx.cwds
            else:
                ctx.cwds |= target

    def _command(self, cmd: _Command, ctx: _Ctx) -> frozenset | None:
        """Check one simple command; return the directories it moves to when it is `cd`."""
        for w in cmd.words + [t for _, t in cmd.redirs]:
            for part in w.parts:
                if isinstance(part, Sub):
                    self.walk(_parse(list(part.tokens)), ctx.child())
        for redir, target in cmd.redirs:
            self._redirect(redir, target, ctx)
        words = self._strip_keywords(cmd.words)
        if not words:
            return None
        if words[0].literal == "for":
            return self._for(words, ctx)
        assigns = []
        while words and (a := _assignment(words[0])):
            assigns.append(a)
            words = words[1:]
        for name, _ in assigns:
            self._check_name(name, ctx)
        if not words:
            ctx.vars.update({name: self._expand(value, ctx) for name, value in assigns})
            return None
        return self._argv(words, ctx)

    def _strip_keywords(self, words: list) -> list:
        while words and _keyword(words[0]) in _LEADING:
            words = words[1:]
        key = _keyword(words[0]) if words else None
        if key in _CLOSING:
            if len(words) > 1:
                raise Deny(f"unexpected words after {key}")
            return []
        if key in _UNSUPPORTED:
            raise Deny(f"{key} is not supported")
        return words

    def _for(self, words: list, ctx: _Ctx) -> None:
        name = words[1].literal if len(words) > 1 else None
        if not name or not _NAME.fullmatch(name) or (len(words) > 2 and words[2].literal != "in"):
            raise Deny("unsupported for loop")
        self._check_name(name, ctx)
        ctx.vars[name] = None

    def _argv(self, words: list, ctx: _Ctx) -> frozenset | None:
        name, args = self._program(words[0], ctx), words[1:]
        if ctx.pod:
            return self._pod_argv(name, args, ctx)
        if name == "cd":
            return self._cd(args, ctx)
        if name == _SCRIPT or name in _READ_ONLY:
            return None
        handler = self.handlers.get(name)
        if handler is None:
            raise Deny(f"{name} is not on the allowlist")
        handler(args, ctx)
        return None

    def _program(self, word: Word, ctx: _Ctx) -> str:
        text = word.literal if word.literal is not None else self._expand(word, ctx)
        if text is None or (word.literal is None and "/" not in text):
            raise Deny(f"the command name must be literal: {word.raw}")
        if "/" not in text:
            return text
        if os.path.dirname(text) in _BIN_DIRS:
            return os.path.basename(text)
        if not ctx.pod and self._is_script(text, ctx, (".sh", ".py")):
            return _SCRIPT
        raise Deny(f"{text} is not one of the skill's scripts")

    # -- values and paths -------------------------------------------------------------------------------------

    def _expand(self, word: Word, ctx: _Ctx) -> str | None:
        out = []
        for p in word.parts:
            value = p.text if isinstance(p, Lit) else self._lookup(p.name, ctx) if isinstance(p, Var) else None
            if value is None:
                return None
            out.append(value)
        return "".join(out)

    def _lookup(self, name: str, ctx: _Ctx) -> str | None:
        if not name or ctx.pod:
            return None
        if name in ctx.vars:
            return ctx.vars[name]
        if name == "PWD":
            return next(iter(ctx.cwds)) if len(ctx.cwds) == 1 else None
        return self.p.env.get(name)

    def _paths(self, word: Word, ctx: _Ctx, leaf_ok: bool = False) -> list[str]:
        """Every absolute path the word can name (glob matches included)."""
        text = self._expand(word, ctx)
        if text is None and leaf_ok:
            text = self._basename_leaf(word, ctx)
        if not text:
            raise Deny(f"cannot resolve the path {word.raw}")
        if word.unquoted("{"):
            raise Deny(f"brace expansion in a path is not supported: {word.raw}")
        if os.path.isabs(text):
            bases = [text]
        elif None in ctx.cwds:
            raise Deny(f"cannot resolve {word.raw}: unknown working directory")
        else:
            bases = [os.path.join(c, text) for c in ctx.cwds]
        out = []
        for base in bases:
            out.append(os.path.realpath(base))
            if word.unquoted("*?["):
                out += [os.path.realpath(m) for m in glob.glob(base)]
        return out

    def _basename_leaf(self, word: Word, ctx: _Ctx) -> str | None:
        """`dir/$(basename ...)` names one entry of dir, so dir/<leaf> stands in for it."""
        if not word.parts:
            return None
        *head, last = word.parts
        if not (isinstance(last, Sub) and last.tokens and isinstance(last.tokens[0], Word)
                and last.tokens[0].literal == "basename"):
            return None
        prefix = self._expand(Word(tuple(head), word.raw), ctx)
        return prefix + "leaf" if prefix and prefix.endswith("/") else None

    def _write(self, word: Word, ctx: _Ctx, append: bool = False, leaf_ok: bool = False) -> None:
        text = self._expand(word, ctx)
        if text in _DEVICES or (text or "").startswith("/dev/fd/"):  # before realpath, which follows /dev/stdout
            return
        for path in self._paths(word, ctx, leaf_ok):
            if why := self.p.write_block(path, append):
                raise Deny(why)

    def _is_script(self, text: str | None, ctx: _Ctx, suffixes: tuple | str) -> bool:
        return self._file_under(text, ctx, SCRIPTS_DIR, suffixes)

    def _file_under(self, text: str | None, ctx: _Ctx, root: str | None, suffixes: tuple | str = "") -> bool:
        """True when `text` names an existing file under `root` from every possible working directory."""
        if not text or not root:
            return False
        if os.path.isabs(text):
            candidates = [text]
        elif None in ctx.cwds:
            return False
        else:
            candidates = [os.path.join(c, text) for c in ctx.cwds]
        resolved = [os.path.realpath(c) for c in candidates]
        return all(_under(r, root) and r.endswith(suffixes) and os.path.isfile(r) for r in resolved)

    def _runnable(self, word: Word, ctx: _Ctx, suffix: str) -> bool:
        """A skill script with `suffix`, or any file in the campaign directory (the agent's own aids)."""
        text = self._expand(word, ctx)
        return self._is_script(text, ctx, suffix) or self._file_under(text, ctx, self.p.campaign)

    def _check_name(self, name: str, ctx: _Ctx) -> None:
        if ctx.pod:
            return
        if name in _SETTINGS_VARS or name.startswith(("LD_", "GIT_", "AUTOTUNE_", "ATTUNE_")):
            raise Deny(f"setting {name} is not allowed; the campaign's environment comes from the launcher")

    def _redirect(self, redir: Redir, target: Word, ctx: _Ctx) -> None:
        text = target.literal
        if redir.op in ("<", "<&") or (redir.op == ">&" and text is not None and (text.isdigit() or text == "-")):
            return
        if ctx.pod:
            if text != "/dev/null":
                raise Deny("commands in kubectl exec may not write files")
            return
        self._write(target, ctx, append=redir.op in (">>", "&>>"), leaf_ok=True)  # <> writes from offset 0

    def _cd(self, args: list, ctx: _Ctx) -> frozenset:
        words = [w for w in args if w.literal not in ("-L", "-P", "-e", "-@")]
        dest = self._expand(words[0], ctx) if words else self.p.env.get("HOME")
        if dest is None or dest == "-":
            return frozenset({None})
        if os.path.isabs(dest):
            return frozenset({os.path.realpath(dest)})
        return frozenset(None if c is None else os.path.realpath(os.path.join(c, dest)) for c in ctx.cwds)

    def _script(self, word: Word | None, ctx: _Ctx, what: str) -> None:
        """Check the shell script held by a word (`bash -c`, `flock -c`)."""
        source = self._expand(word, ctx) if word is not None else None
        if source is None:
            raise Deny(f"cannot resolve the {what} script")
        self.walk(_steps(source), ctx.child(shell=True))

    # -- file tools -------------------------------------------------------------------------------------------

    def _rm(self, args: list, ctx: _Ctx) -> None:
        parsed = _opts(args)
        for flag, _ in parsed.flags:
            if flag not in _RM_FLAGS:
                raise Deny(f"rm {flag} is not allowed")
        for w in parsed.all_operands:
            for path in self._paths(w, ctx):
                if not self.p.campaign or path == self.p.campaign or not _under(path, self.p.campaign):
                    raise Deny(f"rm is limited to files inside the campaign directory, not {path}")
                if why := self.p.write_block(path):
                    raise Deny(why)

    def _mv(self, args: list, ctx: _Ctx) -> None:
        parsed = _opts(args, "tS", {"--target-directory", "--suffix"})
        for w in parsed.all_operands + parsed.values("-t", "--target-directory"):
            self._write(w, ctx)

    def _cp(self, args: list, ctx: _Ctx) -> None:
        parsed = _opts(args, "tS", {"--target-directory", "--suffix"})
        targets = parsed.values("-t", "--target-directory")
        if not targets and len(parsed.all_operands) >= 2:
            targets = parsed.all_operands[-1:]
        for w in targets:
            self._write(w, ctx)

    def _mkdir(self, args: list, ctx: _Ctx) -> None:
        for w in _opts(args, "m", {"--mode", "--context"}).all_operands:
            self._write(w, ctx, append=True)

    def _touch(self, args: list, ctx: _Ctx) -> None:
        for w in _opts(args, "drt", {"--date", "--reference", "--time"}).all_operands:
            self._write(w, ctx, append=True)

    def _tee(self, args: list, ctx: _Ctx) -> None:
        parsed = _opts(args)
        for w in parsed.all_operands:
            self._write(w, ctx, append=parsed.has("-a", "--append"))

    def _sort(self, args: list, ctx: _Ctx) -> None:
        parsed = _opts(args, "otkSTy", {"--output", "--key", "--field-separator", "--buffer-size",
                                        "--temporary-directory", "--parallel", "--files0-from", "--random-source",
                                        "--batch-size", "--compress-program"})
        if parsed.has("--compress-program"):
            raise Deny("sort --compress-program is not allowed")
        for w in parsed.values("-o", "--output"):
            self._write(w, ctx)

    def _uniq(self, args: list, ctx: _Ctx) -> None:
        operands = _opts(args, "fsw", {"--skip-fields", "--skip-chars", "--check-chars"}).all_operands
        if len(operands) > 1:
            self._write(operands[1], ctx)

    # -- text tools -------------------------------------------------------------------------------------------

    def _sed(self, args: list, ctx: _Ctx) -> None:
        parsed = _opts(args, "el", {"--expression", "--line-length"})
        for flag, _ in parsed.flags:
            if flag not in _SED_FLAGS:
                raise Deny(f"sed {flag} is not allowed (no in-place edits or script files)")
        scripts = parsed.values("-e", "--expression") or parsed.all_operands[:1]
        for w in scripts:
            text = self._expand(w, ctx)
            if text is None or not _sed_read_only(text):
                raise Deny("sed scripts may print and substitute but not write or execute (w, W, e)")

    def _awk(self, args: list, ctx: _Ctx) -> None:
        parsed = _opts(args, "Fvfe", {"--field-separator", "--assign", "--file", "--source"}, stop_at_operand=True)
        for flag, _ in parsed.flags:
            if flag not in ("-F", "-v", "-e", "--field-separator", "--assign", "--source"):
                raise Deny(f"awk {flag} is not allowed")
        programs = parsed.values("-e", "--source") or parsed.all_operands[:1]
        if not programs:
            raise Deny("awk needs a program")
        for w in programs:
            text = self._expand(w, ctx)
            if text is None or "system(" in text or "|" in text or ">" in text:
                raise Deny("awk programs may not call system() or redirect (|, >)")

    def _find(self, args: list, ctx: _Ctx) -> None:
        for w in args:
            if (self._expand(w, ctx) or w.literal) in _FIND_ACTIONS:
                raise Deny(f"find {w.literal} is not allowed")

    def _rg(self, args: list, ctx: _Ctx) -> None:
        if any(w.prefix.startswith("--pre") for w in args):
            raise Deny("rg --pre is not allowed")

    def _date(self, args: list, ctx: _Ctx) -> None:
        for w in args:
            if w.prefix.startswith("--set") or re.match(r"-[uR]*s", w.prefix):
                raise Deny("date --set is not allowed")

    # -- shell builtins that set variables --------------------------------------------------------------------

    def _printf(self, args: list, ctx: _Ctx) -> None:
        if args and args[0].literal == "-v":
            self._set_unknown(args[1:2], ctx)

    def _read(self, args: list, ctx: _Ctx) -> None:
        parsed = _opts(args, "adinNptu")
        self._set_unknown(parsed.all_operands + parsed.values("-a"), ctx)

    def _unset(self, args: list, ctx: _Ctx) -> None:
        self._set_unknown([w for w in args if w.literal not in ("-v", "-f", "-n")], ctx)

    def _set_unknown(self, words: list, ctx: _Ctx) -> None:
        for w in words:
            name = w.literal
            if not name or not _NAME.fullmatch(name):
                raise Deny(f"cannot parse the variable name {w.raw}")
            self._check_name(name, ctx)
            ctx.vars[name] = None

    def _export(self, args: list, ctx: _Ctx) -> None:
        for w in args:
            if w.literal in ("-n", "-p"):
                continue
            a = _assignment(w)
            name = a[0] if a else w.literal
            if not name or not _NAME.fullmatch(name):
                raise Deny(f"cannot parse export {w.raw}")
            self._check_name(name, ctx)
            if a:
                ctx.vars[name] = self._expand(a[1], ctx)
            ctx.exported |= {name}

    # -- programs that run programs ---------------------------------------------------------------------------

    def _python(self, args: list, ctx: _Ctx) -> None:
        i = 0
        while i < len(args) and args[i].literal in _PY_FLAGS:
            i += 1
        if [w.literal for w in args[i:]] in (["-V"], ["--version"]):
            return
        if i < len(args) and self._runnable(args[i], ctx, ".py"):
            return
        raise Deny("python3 may run only the skill's scripts or a file in the campaign directory (no -c, -m or stdin)")

    def _shell(self, args: list, ctx: _Ctx) -> None:
        i = 0
        while i < len(args):
            t = args[i].literal or ""
            if re.fullmatch(r"[-+][eux]*o", t):
                i += 2
            elif re.fullmatch(r"[-+][eux]+", t):
                i += 1
            elif re.fullmatch(r"-[eux]*c", t):
                return self._script(args[i + 1] if i + 1 < len(args) else None, ctx, "bash -c")
            else:
                break
        if i < len(args) and self._runnable(args[i], ctx, ".sh"):
            return None
        raise Deny("bash may run only the skill's .sh scripts, a file in the campaign directory or a -c script")

    def _env(self, args: list, ctx: _Ctx) -> None:
        i = 0
        while i < len(args):
            t = args[i].literal or ""
            if t in ("-i", "--ignore-environment", "-0", "--null"):
                i += 1
            elif t in ("-u", "--unset"):
                self._set_unknown(args[i + 1:i + 2], ctx)
                i += 2
            elif t == "--":
                i += 1
                break
            elif t.startswith("-"):
                raise Deny(f"env {t} is not allowed")
            else:
                break
        while i < len(args) and (a := _assignment(args[i])):
            self._check_name(a[0], ctx)
            i += 1
        if i >= len(args):
            raise Deny("env needs a command (printing the environment is not allowed)")
        self._argv(args[i:], ctx)

    def _timeout(self, args: list, ctx: _Ctx) -> None:
        parsed = _opts(args, "sk", {"--signal", "--kill-after"}, stop_at_operand=True)
        if len(parsed.all_operands) < 2:
            raise Deny("timeout needs a duration and a command")
        self._argv(parsed.all_operands[1:], ctx)

    def _nice(self, args: list, ctx: _Ctx) -> None:
        i = 0
        if args and re.fullmatch(r"-\d+|--adjustment=.*", args[0].literal or ""):
            i = 1
        elif args and args[0].literal == "-n":
            i = 2
        if i < len(args):
            self._argv(args[i:], ctx)

    def _nohup(self, args: list, ctx: _Ctx) -> None:
        if not args:
            raise Deny("nohup needs a command")
        self._argv(args, ctx)

    def _time(self, args: list, ctx: _Ctx) -> None:
        if args and args[0].literal == "-p":
            args = args[1:]
        self._nohup(args, ctx)

    def _xargs(self, args: list, ctx: _Ctx) -> None:
        parsed = _opts(args, "nPLsdEIa", {"--max-args", "--max-procs", "--max-lines", "--max-chars", "--delimiter",
                                          "--eof", "--replace", "--arg-file"}, stop_at_operand=True)
        for flag, _ in parsed.flags:
            if flag in ("-p", "-o", "-i", "-e", "-l", "--interactive", "--open-tty"):
                raise Deny(f"xargs {flag} is not allowed")
        if not parsed.all_operands:
            return
        name = self._program(parsed.all_operands[0], ctx)
        if name not in _XARGS_OK:
            raise Deny(f"xargs may run only read-only tools, not {name}")
        self._argv(parsed.all_operands, ctx)

    def _flock(self, args: list, ctx: _Ctx) -> None:
        i = 0
        while i < len(args) and (args[i].literal or "").startswith("-"):
            t = args[i].literal
            if t in ("-w", "--timeout", "-E", "--conflict-exit-code"):
                i += 2
            elif t in _FLOCK_FLAGS:
                i += 1
            else:
                raise Deny(f"flock {t} is not allowed")
        if i >= len(args):
            raise Deny("flock needs a lock file")
        lock, rest = args[i], args[i + 1:]
        if not (lock.literal or "").isdigit():
            self._write(lock, ctx, append=True)
        if rest and rest[0].literal in ("-c", "--command"):
            return self._script(rest[1] if len(rest) > 1 else None, ctx, "flock -c")
        if rest:
            self._argv(rest, ctx)
        return None

    # -- git ----------------------------------------------------------------------------------------------------

    def _git(self, args: list, ctx: _Ctx) -> None:
        i = 0
        while i < len(args):
            t = args[i].literal or ""
            if t == "-C":
                i += 2
            elif t in ("--no-pager", "-P") or t.startswith(("--git-dir=", "--work-tree=")):
                i += 1
            elif t.startswith("-"):
                raise Deny(f"git {t} is not allowed")
            else:
                break
        sub = args[i].literal if i < len(args) else None
        if sub not in _GIT_READ:
            raise Deny(f"git {sub or ''} is not allowed; git is limited to {', '.join(sorted(_GIT_READ))}")
        for w in args[i + 1:]:
            if w.prefix.startswith(("--output", "--ext-diff", "-O", "--open-files-in-pager")):
                raise Deny(f"git {sub} {w.prefix} is not allowed")

    # -- kubectl ------------------------------------------------------------------------------------------------

    def _kubectl(self, args: list, ctx: _Ctx) -> None:
        if not self.p.context or not self.p.namespace:
            raise Deny("kubectl needs AUTOTUNE_KUBECTL_CONTEXT and KUBECTL_NS in the environment")
        pre = _opts(args, "nsv", _K_GLOBAL_VALUES, stop_at_operand=True)
        for flag, _ in pre.flags:
            if flag not in _K_GLOBAL_VALUES | _K_GLOBAL_BOOLS:
                raise Deny(f"kubectl {flag} before the verb is not supported")
        verb = pre.operands[0].literal if pre.operands else None
        if verb == "delete":
            raise Deny("kubectl delete is not allowed; delete a run with teardown_run.sh <run_id>")
        if verb not in _K_VALUES:
            raise Deny(f"kubectl {verb or '(no verb)'} is not allowed")
        short, long_ = _K_VALUES[verb]
        post = _opts(pre.operands[1:], "nsv" + short, _K_GLOBAL_VALUES | long_)
        flags = pre.flags + post.flags
        self._kubectl_scope(flags, ctx)
        if post.rest is not None and verb != "exec":
            raise Deny(f"kubectl {verb} does not take --")
        if verb in _K_READ:
            if verb == "auth" and (not post.operands or post.operands[0].literal not in ("can-i", "whoami")):
                raise Deny("kubectl auth is limited to can-i and whoami")
        elif verb in ("apply", "create"):
            self._kubectl_apply(verb, flags, post.operands, ctx)
        elif verb == "exec":
            self._kubectl_exec(flags, post.operands, post.rest, ctx)
        else:
            self._kubectl_cp(flags, post.operands, ctx)

    def _kubectl_scope(self, flags: list, ctx: _Ctx) -> None:
        for flag, _ in flags:
            if flag in _K_IDENTITY:
                raise Deny(f"kubectl {flag} is not allowed (it changes the cluster or the identity)")
        contexts = [self._expand(v, ctx) for f, v in flags if f == "--context"]
        if not contexts or any(c != self.p.context for c in contexts):
            raise Deny(f"kubectl must pass --context {self.p.context}")
        namespaces = [self._expand(v, ctx) for f, v in flags if f in ("-n", "--namespace")]
        if not namespaces or any(n != self.p.namespace for n in namespaces):
            raise Deny(f"kubectl must pass -n {self.p.namespace}")

    def _kubectl_apply(self, verb: str, flags: list, operands: list, ctx: _Ctx) -> None:
        for flag, _ in flags:
            if flag not in _K_APPLY_FLAGS:
                raise Deny(f"kubectl {verb} {flag} is not allowed")
        files = [v for f, v in flags if f in ("-f", "--filename")]
        if operands or not files:
            raise Deny(f"kubectl {verb} takes only -f <file>")
        for w in files:
            text = self._expand(w, ctx)
            if text is None or text == "-" or "://" in text or any(os.path.isdir(p) for p in self._paths(w, ctx)):
                raise Deny(f"kubectl {verb} -f needs one local file, not {w.raw}")

    def _kubectl_exec(self, flags: list, operands: list, rest: list | None, ctx: _Ctx) -> None:
        for flag, _ in flags:
            if flag not in _K_EXEC_FLAGS:
                raise Deny(f"kubectl exec {flag} is not allowed")
        if len(operands) != 1 or not rest:
            raise Deny("kubectl exec needs the form: exec <log pod> -- <command>")
        self._log_pod(self._expand(operands[0], ctx), "exec")
        self._exec_argv(rest, ctx)

    def _kubectl_cp(self, flags: list, operands: list, ctx: _Ctx) -> None:
        for flag, _ in flags:
            if flag not in _K_CP_FLAGS:
                raise Deny(f"kubectl cp {flag} is not allowed")
        if len(operands) != 2:
            raise Deny("kubectl cp needs the form: cp <log pod>:<path> <local path>")
        m = re.fullmatch(r"(?:([^/:]+)/)?([^/:]+):(.+)", self._expand(operands[0], ctx) or "")
        if not m or (m.group(1) and m.group(1) != self.p.namespace):
            raise Deny("kubectl cp copies only from the log pod to a local path")
        self._log_pod(m.group(2), "cp")
        if re.match(r"[^/]*:", self._expand(operands[1], ctx) or ":"):
            raise Deny("kubectl cp copies only from the log pod to a local path")
        self._write(operands[1], ctx)

    def _log_pod(self, name: str | None, verb: str) -> None:
        if not self.p.log_pod:
            raise Deny(f"kubectl {verb} needs AUTOTUNE_LOG_POD in the environment")
        if (name or "").removeprefix("pod/").removeprefix("pods/") != self.p.log_pod:
            raise Deny(f"kubectl {verb} is limited to the log pod {self.p.log_pod}")

    def _exec_argv(self, words: list, ctx: _Ctx) -> None:
        """The command `kubectl exec` runs in the log pod; host expansions are already applied."""
        values = [self._expand(w, ctx) for w in words]
        if values[0] is None:
            raise Deny("the kubectl exec command name must be literal")
        pod_words = [Word((Lit(v, True),), v) if v is not None else Word((Var(""),), w.raw)
                     for v, w in zip(values, words, strict=True)]
        self._pod_argv(values[0], pod_words[1:], ctx.child(pod=True))

    def _pod_argv(self, name: str, args: list, ctx: _Ctx) -> None:
        name = os.path.basename(name) if os.path.dirname(name) in _BIN_DIRS else name
        if name in ("sh", "bash"):
            if len(args) < 2 or not re.fullmatch(r"-[eux]*c", args[0].literal or "") or args[1].literal is None:
                raise Deny("kubectl exec runs sh only as sh -c '<read-only commands>'")
            return self.walk(_steps(args[1].literal), ctx)
        if name not in _POD_READ:
            raise Deny(f"kubectl exec runs only read-only commands, not {name}")
        if name == "find":
            self._find(args, ctx)
        return None

    # -- network --------------------------------------------------------------------------------------------------

    def _need_web(self, name: str) -> None:
        if not self.p.allow_web:
            raise Deny(f"{name}: network access is off (AUTOTUNE_ALLOW_WEB=1 allows GET requests)")

    def _urls(self, words: list, ctx: _Ctx, name: str) -> None:
        for w in words:
            url = self._expand(w, ctx) or ""
            if not url.startswith(("http://", "https://")) or "169.254.169.254" in url:
                raise Deny(f"{name} fetches only http(s) URLs, not {w.raw}")

    def _curl(self, args: list, ctx: _Ctx) -> None:
        self._need_web("curl")
        parsed = _opts(args, "oHAemrwX", _CURL_VALUES)
        for flag, value in parsed.flags:
            if flag in ("-X", "--request"):
                if self._expand(value, ctx) not in ("GET", "HEAD"):
                    raise Deny("curl is limited to GET requests")
            elif flag in ("-o", "--output"):
                if self._expand(value, ctx) != "-":
                    self._write(value, ctx)
            elif flag not in _CURL_FLAGS:
                raise Deny(f"curl {flag} is not allowed (GET requests only)")
        self._urls(parsed.all_operands + parsed.values("--url"), ctx, "curl")

    def _wget(self, args: list, ctx: _Ctx) -> None:
        self._need_web("wget")
        args = [Word((Lit("--no-verbose"),), w.raw) if w.literal == "-nv" else w for w in args]
        parsed = _opts(args, "OTtU", {"--output-document", "--timeout", "--tries", "--header", "--user-agent"})
        for flag, value in parsed.flags:
            if flag in ("-O", "--output-document"):
                if self._expand(value, ctx) != "-":
                    self._write(value, ctx)
            elif flag not in _WGET_FLAGS:
                raise Deny(f"wget {flag} is not allowed (GET requests only)")
        if not parsed.has("-O", "--output-document", "--spider"):
            raise Deny("wget needs -O <file> (or -O -)")
        self._urls(parsed.all_operands, ctx, "wget")


def classify_command(command: str, cwd: str | None = None) -> Decision:
    """ALLOW only when every simple command in `command` passes the policy; `cwd` resolves relative paths."""
    if not (command or "").strip():
        return Decision(False, "empty command")
    try:
        ctx = _Ctx(frozenset({os.path.realpath(cwd or os.getcwd())}), {})
        _Guard(Policy.from_env()).walk(_steps(command), ctx)
    except Deny as e:
        return Decision(False, str(e))
    except Exception as e:  # a parser bug must not let a command through
        return Decision(False, f"guard error: {type(e).__name__}: {e}")
    return Decision(True, "allowed")
