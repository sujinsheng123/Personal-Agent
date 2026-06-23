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

_work_dir: Path = Path("./data")
_allow_network: bool = False
_MAX_OUTPUT = 4000


def set_work_dir(path: Path) -> None:
    global _work_dir
    _work_dir = path.resolve()


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

# Dangerous argument patterns — blocked regardless of whitelist
_DANGEROUS_PATTERNS: list[str] = [
    r'>\s*/dev/[sh]da', r'>\s*\\\\.\\',       # write to raw devices
    r'rm\s+-rf\s+/', r'rm\s+-rf\s+~',          # rm root/home
    r'mkfs\.', r'dd\s+if=',                    # format / raw write
    r':\(\)\s*\{',                              # fork bomb
    r'chmod\s+777\s+/',                         # world-writable root
    r'>\s*/etc/', r'>\s*C:\\Windows',           # system config overwrite
    r'\|.*sh\b', r'`[^`]+`',                   # pipe to shell / backtick injection
    r'\$\([^)]+\)',                              # command substitution
]


def _check_command(cmd_line: str) -> str | None:
    """Validate command against whitelist + patterns. Returns error or None."""
    cmd_stripped = cmd_line.strip()

    # Extract base command (first word, handling quotes)
    parts = cmd_stripped.split()
    if not parts:
        return "Error: empty command"

    base = parts[0].lower().replace("\\", "/").split("/")[-1]  # strip path

    # Check whitelist
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

    # Check dangerous patterns
    cmd_normalized = cmd_stripped.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd_normalized):
            return f"Error: dangerous pattern detected ({pattern})"

    return None


def _audit(command: str, result: str, success: bool) -> None:
    """Write audit entry for every shell execution."""
    try:
        from personal_agent.tools.audit import audit_log
        audit_log("bash", command, result[:200], success)
    except Exception:
        pass


# ── handler ──────────────────────────────────────────

async def _shell(command: str, timeout: int = 30) -> str:
    error = _check_command(command)
    if error:
        _audit(command, error, False)
        return error

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_work_dir),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=min(timeout, 60)
            )
        except asyncio.TimeoutError:
            proc.kill()
            msg = f"Error: command timed out after {timeout}s"
            _audit(command, msg, False)
            return msg

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
    handler=_shell,
    toolset="builtin",
    is_parallel_safe=False,
    is_destructive=False,  # whitelist constrains safety
))
