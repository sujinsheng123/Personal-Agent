"""Delegate task — spawn a lightweight sub-agent for isolated subtasks.

Use cases:
  - "Research X and Y in parallel, then compare"
  - "Check these 3 files for bugs independently"
  - "Summarize this long text before we continue"

Each delegation runs in a fresh context (no history from main conversation),
so it's ideal for focused, independent work.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry

logger = logging.getLogger(__name__)

# Set by main.py at startup — the delegate tool needs LLM access
_delegate_call: Callable | None = None
_delegate_tools: list[dict] | None = None
_delegate_max_tokens: int = 4096


def setup_delegate(
    call_fn: Callable,
    tools: list[dict],
    max_tokens: int = 4096,
) -> None:
    """Configure delegate subsystem. Called once at startup."""
    global _delegate_call, _delegate_tools, _delegate_max_tokens
    _delegate_call = call_fn
    _delegate_tools = tools
    _delegate_max_tokens = max_tokens


async def _delegate_task(
    prompt: str,
    context: str = "",
    max_tokens: int = 2048,
) -> str:
    """Spawn a sub-agent to handle an isolated task.

    The sub-agent has:
    - No access to the main conversation history
    - Access to all the same tools as the main agent
    - A single-turn execution (no multi-turn loop)
    - Its own system prompt focused on task completion

    Args:
        prompt: The task for the sub-agent. Be specific about what you want.
        context: Optional background info / data the sub-agent needs.
        max_tokens: Max output tokens for the sub-agent (default 2048).
    """
    if _delegate_call is None:
        return (
            "Error: delegate subsystem not initialized. "
            "Set up delegate transport in main.py."
        )

    # ── Build messages ──
    messages: list[dict] = []

    if context:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"Context:\n{context}"}],
        })

    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    })

    system = (
        "You are a focused sub-agent. Complete the assigned task and return "
        "your result. Be concise and direct — no conversation, no questions. "
        "If you need external information, use the available tools. "
        "When done, provide your final answer in plain text."
    )

    # ── Call LLM ──
    try:
        response = await asyncio.wait_for(
            _delegate_call(
                messages=messages,
                system_prompt=system,
                tools=_delegate_tools,
                max_tokens=min(max_tokens, _delegate_max_tokens),
            ),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        return "Error: delegate task timed out (120s)"
    except Exception as exc:
        return f"Error: delegate task failed: {exc}"

    # ── Handle tool calls (single round) ──
    if response.tool_calls:
        from personal_agent.tools.executor import execute_tool_calls
        await execute_tool_calls(response.tool_calls, messages)

        # One follow-up to get final answer
        final_msg = {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Tools executed. Now provide your final answer based on these results."
            }],
        }
        messages.append(final_msg)

        try:
            response = await asyncio.wait_for(
                _delegate_call(
                    messages=messages,
                    system_prompt=system,
                    tools=[],  # no tools on final turn
                    max_tokens=min(max_tokens, _delegate_max_tokens),
                ),
                timeout=60.0,
            )
        except Exception as exc:
            return f"Error: delegate follow-up failed: {exc}"

    return response.text or "(sub-agent returned no text)"


tool_registry.register(ToolEntry(
    name="delegate_task",
    description=(
        "Spawn a focused sub-agent to handle an isolated task. "
        "The sub-agent runs in a fresh context (no main conversation history) "
        "and has access to the same tools. Use for: parallel research, "
        "independent code review, summarization, or any task that benefits "
        "from focused attention. One delegation = one task. "
        "Be specific about what you want the sub-agent to do."
    ),
    schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The task for the sub-agent. Be specific and include all needed details.",
            },
            "context": {
                "type": "string",
                "description": "Optional background info, data, or text the sub-agent needs.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens for the sub-agent (default 2048)",
            },
        },
        "required": ["prompt"],
    },
    handler=_delegate_task,
    toolset="builtin",
    is_parallel_safe=False,
))
