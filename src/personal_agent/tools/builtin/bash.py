"""Safe shell command execution — whitelist + sandbox + audit.

Layered defense:
  1. Command whitelist — unknown commands blocked
  2. Argument-level dangerous pattern detection
  3. Network isolation (curl/wget/pip blocked unless config allows)
  4. Working directory restricted to data dir
  5. Timeout (default 30s)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry

logger = logging.getLogger(__name__)

# ── sandbox config (set at startup) ──────────────────

_work_dir: Path = Path("./data").resolve()
_allow_network: bool = False
_restrict_paths: bool = True
_MAX_OUTPUT = 4000


def set_work_dir(path: Path) -> None:
    global _work_dir
    _work_dir = path.resolve()


def set_restrict_paths(restrict: bool) -> None:
    global _restrict_paths
    _restrict_paths = restrict


def set_allow_network(allowed: bool) -> None:
    global _allow_network
    _allow_network = allowed


# ── command whitelist ─────────────────────────────────
# Format: command_name → (arg_patterns, needs_network)
# arg_patterns: "*" = any args allowed; ["-n", "-l"] = only these flags

WHITELIST: dict[str, tuple[str | list[str], bool]] = {
    # File ops
    "ls":     ("*", False),   "dir":    ("*", False),
    "cat":    ("*", False),   "type":   ("*", False),
    "head":   ("*", False),   "tail":   ("*", False),
    "wc":     ("*", False),   "find":   ("*", False),
    "grep":   ("*", False),   "cp":     ("*", False),
    "mv":     ("*", False),   "mkdir":  ("*", False),
    "rmdir":  ("*", False),   "touch":  ("*", False),
    "rm":     ("*", False),   "tree":   ("*", False),
    # Git
    "git":    ("*", False),
    # Python
    "python": ("*", False),   "python3": ("*", False),
    "pip":    ("*", True),    "uv":      ("*", True),
    # Text processing
    "echo":   ("*", False),   "sed":    ("*", False),
    "awk":    ("*", False),   "sort":   ("*", False),
    "uniq":   ("*", False),   "cut":    ("*", False),
    "tr":     ("*", False),   "diff":   ("*", False),
    # System info (no destructive args)
    "whoami":  ([], False),   "pwd":    ([], False),
    "date":    ([], False),   "env":    ([], False),
    "uname":   ([], False),   "hostname": ([], False),
    "df":     ("*", False),   "du":     ("*", False),
    "ps":     ("*", False),   "which":  ("*", False),
    "where":  ("*", False),
    # Compilers / build
    "gcc":    ("*", False),   "g++":   ("*", False),
    "make":   ("*", False),   "cargo": ("*", True),
    "go":     ("*", False),   "rustc": ("*", False),
    # Network tools (only if _allow_network)
    "curl":   ("*", True),    "wget":  ("*", True),
    "npx":    ("*", True),    "npm":   ("*", True),
}

# Windows command aliases
_WINDOWS_ALIASES: dict[str, str] = {
    "dir": "dir", "type": "type", "findstr": "findstr",
    "where": "where",
}

# ── Hard blacklist — catastrophic commands, NEVER allowed ──
# These are checked BEFORE the whitelist and cannot be overridden.
# Even /allow bash does not bypass these.

_HARD_BLACKLIST: list[str] = [
    # Filesystem destruction (root paths)
    r'\brm\s+-rf\s+/', r'\brm\s+-rf\s+/\*',
    r'\brm\s+-rf\s+~', r'\brm\s+-rf\s+\$HOME',
    r'\brm\s+-rf\s+/(etc|boot|bin|sbin|lib|lib64|sys|proc|dev)\b',
    # Block device writes
    r'\bdd\s+.*\bof=/dev/[sh]da', r'\bdd\s+.*\bof=\\\\.\\',
    r'\bdd\s+.*\bof=/dev/(null|zero|random)',
    # Format / mkfs
    r'\bmkfs\.', r'\bmkfs\s', r'\bmke2fs\b',
    # Raw disk writes
    r'>\s*/dev/[sh]d[a-z]', r'>\s*\\\\.\\[A-Z]',
    # Fork bomb
    r':\(\)\s*\{', r'\)\(\)\s*\{',
    # System shutdown (anchored to command start, not args)
    r'(?:^|[\s;&|])(?:sudo\s+)?(?:shutdown|reboot|halt|poweroff|init\s+[06])\b',
    # chmod 777 on system dirs
    r'\bchmod\s+777\s+/',
    # Write to system config
    r'>\s*/etc/(passwd|shadow|sudoers|hosts)',
    r'>\s*C:\\Windows\\(System32|SysWOW64)',
    # Kernel module / sysctl tampering
    r'\b(modprobe|sysctl|kldload)\b.*\b(-[a-z]*r\b|write\b)',
]


# Dangerous argument patterns — blocked regardless of whitelist
_DANGEROUS_PATTERNS: list[str] = [
    r'>\s*\\\\.\\',                              # write to raw devices (Windows)
    r'>\s*/etc/', r'>\s*C:\\Windows',            # system config overwrite
    r'\|.*sh\b', r'`[^`]+`',                    # pipe to shell / backtick injection
    r'\$\([^)]+\)',                               # command substitution
    r'\bsudo\b.*\brm\b',                         # sudo rm (any target)
    r'\bgit\s+push\s+--force',                   # force push (potentially destructive)
]


def _check_command(cmd_line: str) -> str | None:
    """Validate command against hard blacklist → whitelist → patterns.

    Layer order:
      0. Hard blacklist — catastrophic, unconditional, NEVER bypassed
      1. Command chaining detection
      2. Whitelist check
      3. Network isolation
      4. Dangerous pattern detection
    """
    cmd_stripped = cmd_line.strip()

    # Extract base command (first word, handling quotes)
    parts = cmd_stripped.split()
    if not parts:
        return "Error: empty command"

    # ── 0. Hard blacklist (UNCONDITIONAL, even with /allow bash) ──
    cmd_lower = cmd_stripped.lower()
    for pattern in _HARD_BLACKLIST:
        if re.search(pattern, cmd_lower, re.IGNORECASE):
            return f"Error: catastrophic command blocked by hard blacklist — this cannot be overridden"

    # ── 1. Block command chaining ──
    _CHAIN_TOKENS = ("&&", "||", "|", ";")
    if any(tok in cmd_stripped for tok in _CHAIN_TOKENS):
        return "Error: command chaining (&& || | ;) is not allowed. Use one command per call."

    # ── 1.5. Path sandbox — no absolute paths, no traversal ──
    path_error = _check_path_sandbox(cmd_stripped)
    if path_error:
        return path_error

    base = parts[0].lower().replace("\\", "/").split("/")[-1]  # strip path

    # ── 2. Whitelist check ──
    if base not in WHITELIST:
        return (
            f"Error: command '{base}' is not in the allowed list. "
            f"Allowed commands: {', '.join(sorted(WHITELIST.keys()))}"
        )

    _, needs_network = WHITELIST[base]
    if needs_network and not _allow_network:
        return (
            f"Error: network access not allowed (blocked '{base}'). "
            f"Set bash_allow_network: true in config.yaml to enable."
        )

    # Check dangerous patterns (case-insensitive matching)
    cmd_normalized = cmd_stripped.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd_normalized, re.IGNORECASE):
            return f"Error: dangerous pattern detected"

    return None


# ── path sandbox ─────────────────────────────────────

# System-level escape patterns — paths that should never be accessible
# These go beyond sandbox roots: /etc, C:\Windows, ~, .. are dangerous
# regardless of configured roots.
_PATH_ESCAPE_PATTERNS: list[str] = [
    r'(?:^|\s)/(?:etc|var|tmp|home|root|proc|sys|dev|opt|usr|bin|sbin|boot)/',  # Unix system paths
    r'(?:^|\s)[A-Za-z]:[\\\\/](?:Windows|Program|Users|WINDOWS)',  # Windows system paths
    r'(?:^|\s)~(?:[/\s]|$)',     # ~/ home dir or bare ~
    r'(?:^|\s)\.\.(?:\s|$|/|\\)',  # parent dir traversal
]


def _glob_pattern_to_regex(glob_pat: str) -> str:
    """Convert a sandbox blocked glob (e.g. '**/.env') to a regex for command scanning.

    Examples:
      **/.env          -> \\.env
      **/.env.*        -> \\.env\\.[^/\\s]*
      **/.git/**       -> \\.git/
      **/id_rsa*       -> id_rsa[^/\\s]*
      **/data/auth/**  -> data/auth/
    """
    pat = glob_pat.strip()
    # Strip leading **/
    if pat.startswith("**/"):
        pat = pat[3:]
    # Strip trailing /**
    if pat.endswith("/**"):
        pat = pat[:-3] + "/"
    # Split on * wildcard, escape literal parts, rejoin with wildcard
    parts = pat.split("*")
    escaped = [re.escape(p) for p in parts]
    return "[^/\\\\\\s]*".join(escaped)


def _check_path_sandbox(cmd_line: str) -> str | None:
    """Block commands that access files outside the sandbox.

    Uses the unified sandbox for blocked patterns and roots.
    Additionally blocks system-level escape patterns when _restrict_paths is true.
    """
    from personal_agent.tools.sandbox import get_sandbox
    sandbox = get_sandbox()

    # ── 1. Blocked patterns (from sandbox) — NEVER allowed ──
    for pattern in sandbox.blocked:
        regex = _glob_pattern_to_regex(pattern)
        if re.search(regex, cmd_line, re.IGNORECASE):
            return (
                f"Error: sandbox blocked — '{pattern}' matches "
                f"protected files. Config and credential files are never readable via bash."
            )

    # ── 2. Restrict paths off? allow everything except blocked ──
    if not _restrict_paths:
        return None

    # ── 3. Check if command references a path under a sandbox root ──
    cmd_norm = cmd_line.replace("\\", "/")
    for root in sandbox.roots:
        rs = str(root).replace("\\", "/")
        # Match as full path component — avoids Desktop matching DesktopProjects
        escaped_root = re.escape(rs)
        if re.search(rf'(?:^|\s){escaped_root}(?:/|$)', cmd_norm):
            return None  # path is under a configured root, allow

    # ── 4. Escape patterns — system paths that bypass root check ──
    for pattern in _PATH_ESCAPE_PATTERNS:
        if re.search(pattern, cmd_line, re.IGNORECASE):
            return (
                f"Error: path sandbox blocked — absolute system path or "
                f"parent traversal detected. Use only relative paths within "
                f"the working directory, or absolute paths under configured "
                f"sandbox roots."
            )

    return None


def _audit(command: str, result: str, success: bool) -> None:
    """Write audit entry for every shell execution."""
    try:
        from personal_agent.tools.audit import audit_log
        audit_log("bash", command, result[:200], success)
    except Exception:
        pass


# ── handler ──────────────────────────────────────────

async def _bash(command: str, timeout: int = 30) -> str:
    error = _check_command(command)
    if error:
        _audit(command, error, False)
        return error

    try:
        from personal_agent.tools.env_filter import filter_env
        proc = await asyncio.create_subprocess_bash(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_work_dir),
            env=filter_env(),
        )
        try:
            deadline = time.time() + min(timeout, 60)
            stdout, stderr = b"", b""
            while time.time() < deadline:
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=1.0
                    )
                    break
                except asyncio.TimeoutError:
                    from personal_agent.tools.executor import is_interrupted
                    if is_interrupted():
                        proc.kill()
                        await proc.wait()
                        _audit(command, "interrupted by user", False)
                        return "Interrupted by user."
        except asyncio.TimeoutError:
            proc.kill()

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        result = out or err or "(no output)"
        if len(result) > _MAX_OUTPUT:
            result = result[:_MAX_OUTPUT] + f"\n...({len(result) - _MAX_OUTPUT} more chars)"

        _audit(command, result, proc.returncode == 0 if proc.returncode is not None else True)
        return result
    except Exception as e:
        _audit(command, str(e), False)
        return f"Error: {e}"


tool_registry.register(ToolEntry(
    name="bash",
    description="Execute a shell command in a restricted sandbox. "
                "Only whitelisted commands allowed (ls, cat, grep, git, python, etc.). "
                "Network tools (curl, pip) blocked unless bash_allow_network=true.",
    schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command, e.g. 'ls -la' or 'python --version'"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 60)"},
        },
        "required": ["command"],
    },
    handler=_bash,
    toolset="builtin",
    is_parallel_safe=False,
    is_destructive=False,  # whitelist constrains safety
))
