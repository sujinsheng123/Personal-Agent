"""Tool execution pipeline: pre-hook → dispatch → post-process.
Parallel/serial execution with individual fault isolation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from personal_agent.tools.registry import tool_registry

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 8000


async def execute_tool_calls(
    tool_calls: list[dict],
    messages: list[dict],
    *,
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
        tasks = [asyncio.to_thread(_exec_one_sync, tc, hooks) for _, tc in parallel_safe]
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
            results[idx] = await _exec_one(tc, hooks)
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


async def _exec_one(tc: dict, hooks: Any = None) -> str:
    """Execute a single tool call through the 3-stage pipeline."""

    # ── 1. pre-hook ──────────────────────────────────
    if hooks:
        result = await hooks.fire("on_before_tool_exec", tc, tool_registry.get(tc["name"]))
        if result is None:
            return "Error: tool execution blocked"
        if isinstance(result, dict):
            tc = result

    entry = tool_registry.get(tc["name"])
    if entry is None:
        return f"Error: unknown tool '{tc['name']}'"

    if entry.is_destructive:
        logger.warning("Executing destructive tool: %s(%s)", tc["name"], tc["input"])

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


def _exec_one_sync(tc: dict, hooks: Any = None) -> str:
    """Synchronous wrapper for thread-pool execution. Runs async _exec_one in a new loop."""
    return asyncio.run(_exec_one(tc, hooks))
