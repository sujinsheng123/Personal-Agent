"""Tool execution pipeline: scope gate → pre-hook → dispatch → post-process.
Parallel/serial execution with individual fault isolation.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time as _time_module
from pathlib import Path
from typing import Any

from personal_agent.tools.registry import tool_registry

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 8000


async def execute_tool_calls(
    tool_calls: list[dict],
    messages: list[dict],
    *,
    agent: Any = None,
    hooks: Any = None,
) -> None:
    """Execute all tool calls, append results to messages in original order.

    tool_calls: [{"id":..., "name":..., "input":{}}, ...]
    """
    parallel_safe: list[tuple[int, dict]] = []
    sequential: list[tuple[int, dict]] = []

    for i, tc in enumerate(tool_calls):
        entry = tool_registry.get(tc["name"])
        if entry and entry.is_parallel_safe:
            parallel_safe.append((i, tc))
        else:
            sequential.append((i, tc))

    results: dict[int, str] = {}

    # ── parallel group: thread pool (asyncio.to_thread), single fault isolation ──
    if parallel_safe:
        tasks = [asyncio.to_thread(_exec_one_sync, tc, agent, hooks) for _, tc in parallel_safe]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for i, (idx, _tc) in enumerate(parallel_safe):
            result = gathered[i]
            if isinstance(result, Exception):
                results[idx] = f"Error: {result}"
            else:
                results[idx] = result

    # ── sequential group: ordered, previous result may feed next ──
    for idx, tc in sequential:
        try:
            results[idx] = await _exec_one(tc, agent=agent, hooks=hooks)
        except Exception as exc:
            results[idx] = f"Error: {exc}"

    # ── append results in ORIGINAL order ──
    for i, tc in enumerate(tool_calls):
        result_text = results.get(i, "Error: tool execution skipped")
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": result_text,
            }],
        })


async def _exec_one(tc: dict, *, agent: Any = None, hooks: Any = None) -> str:
    """Execute a single tool call through: scope gate → pre-hook → dispatch → post-process."""

    entry = tool_registry.get(tc["name"])
    if entry is None:
        return f"Error: unknown tool '{tc['name']}'"

    # ── 0. scope gate ────────────────────────────────
    gate_error = _scope_gate(tc, entry, agent)
    if gate_error:
        return gate_error

    # ── 0.5. checkpoint (write tool only) ────────────
    if tc["name"] == "write":
        _checkpoint_file_write(tc)

    # ── 1. pre-hook ──────────────────────────────────
    if hooks:
        result = await hooks.fire("on_before_tool_exec", tc, entry)
        if result is None:
            return "Error: tool execution blocked"
        if isinstance(result, dict):
            tc = result

    # ── 2. dispatch ──────────────────────────────────
    try:
        result = await entry.handler(**tc["input"])
    except Exception as exc:
        logger.exception("Tool dispatch failed for '%s'", tc["name"])
        result = f"Error: {exc}"

    # ── 3. post-process ──────────────────────────────
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + f"\n\n...({len(result) - MAX_RESULT_CHARS} more chars truncated)"

    if hooks:
        modified = await hooks.fire("on_after_tool_exec", tc, result)
        if isinstance(modified, str):
            result = modified

    logger.debug("Tool '%s' done: %d chars", tc["name"], len(result))
    return result


def _exec_one_sync(tc: dict, agent: Any = None, hooks: Any = None) -> str:
    """Synchronous wrapper for thread-pool execution. Fresh event loop per thread.
    Handles Windows ProactorEventLoop quirks gracefully."""
    try:
        return asyncio.run(_exec_one(tc, agent=agent, hooks=hooks))
    except RuntimeError as e:
        if "event loop" in str(e).lower():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_exec_one(tc, agent=agent, hooks=hooks))
            finally:
                loop.close()
        raise


# ── scope gate ────────────────────────────────────────

def _tool_category(name: str) -> str:
    """Map tool name → destructive category for granular /allow."""
    _CATEGORY_MAP: dict[str, str] = {
        "write": "write",
        "edit": "write",
        "bash": "bash",
    }
    return _CATEGORY_MAP.get(name, "write")


def _scope_gate(tc: dict, entry, agent: Any) -> str | None:
    """Check if this tool call should be allowed. Returns error string or None."""

    # ① check_fn — runtime dependency check
    if entry.check_fn and not entry.check_fn():
        return f"Error: tool '{tc['name']}' is currently unavailable (dependency not met)"

    if agent is None:
        return None  # no guard checks without agent context

    # ② destructive guard — check category in allowed set
    if entry.is_destructive:
        category = _tool_category(tc["name"])
        if category not in agent._destructive_allowed and "all" not in agent._destructive_allowed:
            return (
                f"Error: destructive tool '{tc['name']}' requires authorization. "
                f"Send /allow {category} or /allow all to enable for this turn."
            )

    # ③ guardrail — per-turn call quota
    if agent._tool_calls_this_turn >= agent._max_tool_calls_per_turn:
        return (
            f"Error: tool call limit ({agent._max_tool_calls_per_turn}) reached. "
            f"Please summarize what has been done and stop."
        )

    agent._tool_calls_this_turn += 1
    return None


# ── checkpoint ────────────────────────────────────────

def _checkpoint_file_write(tc: dict) -> None:
    """Backup target file before a file_write dispatch. Best-effort — never blocks execution."""
    try:
        from personal_agent.tools.builtin.file_write import _allowed_base  # noqa: F401

        path = tc.get("input", {}).get("path", "")
        if not path:
            return

        full = (_allowed_base / path).resolve()
        if not full.exists():
            return  # new file, nothing to backup

        backup_dir = _allowed_base / "checkpoints"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _time_module.strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{full.name}.{timestamp}.bak"
        shutil.copy2(full, backup_path)
        logger.info("Checkpoint saved: %s → %s", path, backup_path.name)
    except Exception:
        logger.exception("Checkpoint failed for file_write — tool execution will proceed")
