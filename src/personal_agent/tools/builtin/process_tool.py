"""Process manager — background process tracking.

Track subprocesses spawned by bash, allowing the agent to:
  - List running background processes
  - Kill a specific process
  - Wait for a process to finish and get its output
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry

logger = logging.getLogger(__name__)


@dataclass
class TrackedProcess:
    """A single tracked background process."""

    pid: int
    command: str
    proc: asyncio.subprocess.Process
    started_at: float = field(default_factory=time.time)
    finished: bool = False
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""


# In-memory registry — lost on restart, fine for a session
_processes: dict[int, TrackedProcess] = {}
_next_id = 0
_MAX_OUTPUT = 4000


def _register(proc: asyncio.subprocess.Process, command: str) -> int:
    """Register a new background process. Returns the internal PID."""
    global _next_id
    _next_id += 1
    pid = _next_id
    _processes[pid] = TrackedProcess(pid=pid, command=command, proc=proc)
    # Schedule background waiter
    asyncio.create_task(_waiter(pid, proc))
    return pid


async def _waiter(pid: int, proc: asyncio.subprocess.Process) -> None:
    """Wait for process completion and capture output."""
    try:
        stdout, stderr = await proc.communicate()
        tp = _processes.get(pid)
        if tp:
            tp.stdout = stdout.decode("utf-8", errors="replace")[:_MAX_OUTPUT]
            tp.stderr = stderr.decode("utf-8", errors="replace")[:_MAX_OUTPUT]
            tp.returncode = proc.returncode
            tp.finished = True
    except Exception as exc:
        logger.debug("Process %d waiter error: %s", pid, exc)


# ── tool handlers ──────────────────────────────────────


async def _process_list() -> str:
    """List all tracked background processes."""
    if not _processes:
        return "No background processes running."

    lines = ["Background processes:"]
    for pid, p in sorted(_processes.items()):
        status = "done" if p.finished else "running"
        runtime = time.time() - p.started_at
        lines.append(
            f"  [{pid}] {status} ({runtime:.0f}s) — {p.command[:80]}"
            + (f" → rc={p.returncode}" if p.finished else "")
        )
    return "\n".join(lines)


async def _process_kill(pid: int) -> str:
    """Kill a background process by its ID."""
    tp = _processes.get(pid)
    if tp is None:
        return f"Error: no process with ID {pid}"

    if tp.finished:
        return f"Process [{pid}] already finished (rc={tp.returncode})"

    try:
        tp.proc.kill()
        tp.finished = True
        tp.returncode = -9
        return f"Process [{pid}] killed."
    except Exception as e:
        return f"Error killing process [{pid}]: {e}"


async def _process_wait(pid: int, timeout: int = 30) -> str:
    """Wait for a process to finish and return its output."""
    tp = _processes.get(pid)
    if tp is None:
        return f"Error: no process with ID {pid}"

    if tp.finished:
        return _format_result(tp, "already finished")

    try:
        await asyncio.wait_for(tp.proc.wait(), timeout=min(timeout, 120))
        # Give waiter a moment to capture output
        await asyncio.sleep(0.1)
        tp = _processes.get(pid)  # refresh after waiter updated
        if tp and tp.finished:
            return _format_result(tp, "finished")
        return f"Process [{pid}] ended but output was not captured."
    except asyncio.TimeoutError:
        return f"Process [{pid}] still running after {timeout}s. Use process_list to check status."
    except Exception as e:
        return f"Error waiting for process [{pid}]: {e}"


def _format_result(tp: TrackedProcess, status: str) -> str:
    lines = [f"Process [{tp.pid}] {status} (rc={tp.returncode})"]
    if tp.stdout.strip():
        lines.append(f"stdout:\n{tp.stdout}")
    if tp.stderr.strip():
        lines.append(f"stderr:\n{tp.stderr}")
    return "\n".join(lines)


# ── registration ───────────────────────────────────────


tool_registry.register(ToolEntry(
    name="process_list",
    description="List all tracked background processes and their status (running/done).",
    schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    handler=_process_list,
    toolset="builtin",
))

tool_registry.register(ToolEntry(
    name="process_kill",
    description="Kill a running background process by its ID (from process_list).",
    schema={
        "type": "object",
        "properties": {
            "pid": {"type": "integer", "description": "Process ID from process_list"},
        },
        "required": ["pid"],
    },
    handler=_process_kill,
    toolset="builtin",
))

tool_registry.register(ToolEntry(
    name="process_wait",
    description="Wait for a background process to finish and return its output. Useful for long-running builds, installs, or data processing.",
    schema={
        "type": "object",
        "properties": {
            "pid": {"type": "integer", "description": "Process ID from process_list"},
            "timeout": {"type": "integer", "description": "Max seconds to wait (default 30, max 120)"},
        },
        "required": ["pid"],
    },
    handler=_process_wait,
    toolset="builtin",
))
