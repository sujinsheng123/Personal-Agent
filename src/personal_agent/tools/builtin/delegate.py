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
    max_iterations: int = 5,
) -> str:
    """Spawn a sub-agent to handle an isolated task with multi-turn execution.

    The sub-agent has:
    - No access to the main conversation history
    - Access to all the same tools as the main agent
    - Multi-turn execution (max 5 iterations) — enough for research tasks
    - Its own system prompt focused on task completion
    - Empty-response retry (max 1 nudge)

    Args:
        prompt: The task for the sub-agent. Be specific about what you want.
        context: Optional background info / data the sub-agent needs.
        max_tokens: Max output tokens per LLM call (default 2048).
        max_iterations: Max turns before forced summary (default 5, max 10).
    """
    if _delegate_call is None:
        return (
            "Error: delegate subsystem not initialized. "
            "Set up delegate transport in main.py."
        )

    # ── Build seed messages ──
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
        "You are a focused sub-agent running inside a while-loop. You have "
        "access to tools to fetch information and perform actions. "
        "Work step by step:\n"
        "1. Understand the task\n"
        "2. Use tools to gather needed information\n"
        "3. Synthesize your findings\n"
        "4. When done, provide your FINAL ANSWER in plain text.\n\n"
        "IMPORTANT: Do NOT ask questions or wait for user input. The user "
        "will NOT reply — you must complete the task autonomously. "
        "When you have the final result, stop calling tools and just output "
        "your answer. Be concise — no unnecessary commentary."
    )

    iterations = min(max_iterations, 10)
    empty_retries = 0

    # ── Multi-turn loop ──
    for turn in range(iterations):
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

        # ── Empty response retry ──
        if not response.text and not response.tool_calls:
            if empty_retries < 1:
                empty_retries += 1
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Your last response was empty. Please continue working on the task.",
                    }],
                })
                continue
            return "Error: sub-agent returned empty response twice"

        # ── No tool calls → final answer ──
        if not response.tool_calls:
            return response.text or "(sub-agent returned no text)"

        # ── Execute tools and continue ──
        from personal_agent.tools.executor import execute_tool_calls
        assistant_blocks = []
        if response.text:
            assistant_blocks.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            assistant_blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": tc["input"],
            })
        messages.append({"role": "assistant", "content": assistant_blocks})
        await execute_tool_calls(response.tool_calls, messages)

        # ── Last turn: force summary ──
        if turn == iterations - 1:
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "This was your last turn. Summarize your findings as the FINAL ANSWER now. Do NOT call any more tools.",
                }],
            })
            try:
                response = await asyncio.wait_for(
                    _delegate_call(
                        messages=messages,
                        system_prompt=system,
                        tools=[],  # no tools on forced-final turn
                        max_tokens=min(max_tokens, _delegate_max_tokens),
                    ),
                    timeout=60.0,
                )
                return response.text or "(sub-agent returned no text)"
            except Exception as exc:
                return f"Error: delegate final summary failed: {exc}"

    return "(sub-agent exhausted all turns without result)"


tool_registry.register(ToolEntry(
    name="delegate_task",
    description=(
        "Spawn a focused sub-agent to handle an isolated task. "
        "The sub-agent runs in a fresh context (no main conversation history), "
        "has access to the same tools, and can execute multiple turns (max 5). "
        "Use for: research tasks, independent code review, multi-step data "
        "gathering, or any task that benefits from focused attention. "
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
                "description": "Max output tokens per turn (default 2048)",
            },
            "max_iterations": {
                "type": "integer",
                "description": "Max tool-calling turns before forced summary (default 5, max 10). Increase for complex research.",
            },
        },
        "required": ["prompt"],
    },
    handler=_delegate_task,
    toolset="builtin",
    is_parallel_safe=False,
))
