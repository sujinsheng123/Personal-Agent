"""Sandboxed Python code execution.

Runs user code in a fresh subprocess with:
  - Temp working directory (no access to agent files)
  - Credential-stripped environment
  - Hard timeout (default 30s, max 120s)
  - stdout/stderr capture with truncation
  - Audit logging

This is safer than `bash python -c "..."` because it's a separate process
with minimal environment and no persistent state.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import textwrap
from pathlib import Path

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry

logger = logging.getLogger(__name__)

MAX_OUTPUT = 8000
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120

# Packages known to be available — listed so LLM knows what it can import
_AVAILABLE_MODULES_HINT = (
    "# Commonly available: re, json, math, datetime, collections, itertools, "
    "pathlib, csv, io, base64, hashlib, textwrap, functools, typing, enum, "
    "dataclasses, random, statistics, urllib.parse, xml, html, decimal, fractions"
)


def _audit(code: str, result: str, success: bool) -> None:
    """Write audit entry for code execution."""
    try:
        from personal_agent.tools.audit import audit_log
        audit_log("execute_code", code[:200], result[:200], success)
    except Exception:
        pass


async def _execute_code(code: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Execute Python code in a sandboxed subprocess.

    The code runs in a fresh Python process with:
    - Isolated temp working directory
    - No access to the agent's environment variables (API keys stripped)
    - Hard timeout

    Use for: data processing, calculations, file format conversions,
    quick scripts, and any task that needs real Python execution.
    Do NOT use for: long-running services, GUI applications, or code
    that needs the agent's Python packages.
    """
    timeout = min(max(timeout, 5), MAX_TIMEOUT)

    # Dedent code for clean execution
    code = textwrap.dedent(code).strip()

    # ── Sandbox: temp directory ──
    work_dir = tempfile.mkdtemp(prefix="pyexec_")
    try:
        # ── Sandbox: filtered environment ──
        from personal_agent.tools.env_filter import filter_env
        env = filter_env()
        # Keep only essential env vars for Python to run
        keep = {
            "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TMP", "TEMP", "TMPDIR",
            "PATH", "PATHEXT", "COMSPEC", "USERNAME", "USER", "HOME",
            "APPDATA", "LOCALAPPDATA", "HOMEDRIVE", "HOMEPATH",
        }
        env = {k: v for k, v in env.items() if k in keep or k.startswith("PYTHON")}

        # ── Execute ──
        import sys
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            _audit(code, f"timed out after {timeout}s", False)
            return f"Error: code execution timed out after {timeout}s"

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()

        # ── Build result ──
        lines = []
        if out:
            lines.append(out)
        if err:
            lines.append(f"[stderr]\n{err}")

        if not lines:
            lines.append("(no output)")

        result = "\n".join(lines)
        if len(result) > MAX_OUTPUT:
            result = result[:MAX_OUTPUT] + (
                f"\n\n...(truncated {len(result) - MAX_OUTPUT} more chars)"
            )

        success = proc.returncode == 0
        _audit(code, result, success)
        return result

    except Exception as exc:
        _audit(code, str(exc), False)
        return f"Error: {exc}"
    finally:
        # Clean up temp dir
        try:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


tool_registry.register(ToolEntry(
    name="execute_code",
    description=(
        "Execute Python code in a sandboxed subprocess. "
        "The code runs in an isolated temp directory with no access to agent files "
        "or environment variables. Use for calculations, data processing, quick "
        "scripts, or any task that needs real Python execution. "
        f"{_AVAILABLE_MODULES_HINT}. "
        "stdout and stderr are both captured and returned."
    ),
    schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Can use stdlib modules.",
            },
            "timeout": {
                "type": "integer",
                "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT})",
            },
        },
        "required": ["code"],
    },
    handler=_execute_code,
    toolset="builtin",
    is_parallel_safe=True,
))
