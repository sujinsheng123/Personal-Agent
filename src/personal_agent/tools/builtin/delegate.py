"""Sub-agent tools — CC-style multi-agent primitives.

sub_agent:     Spawn one sub-agent for a focused task (parallel-safe)
sub_parallel:  Run multiple sub-agents concurrently, wait for all
sub_pipeline:  Run items through stages independently (no barrier)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry

logger = logging.getLogger(__name__)

_delegate_call: Callable | None = None
_delegate_tools: list[dict] | None = None
_delegate_max_tokens: int = 4096


def setup_delegate(call_fn, tools, max_tokens=4096):
    global _delegate_call, _delegate_tools, _delegate_max_tokens
    _delegate_call = call_fn
    _delegate_tools = tools
    _delegate_max_tokens = max_tokens


async def _run_agent(prompt, system_prompt="", schema="", max_tokens=2048):
    if _delegate_call is None:
        return "Error: sub-agent system not initialized"

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    sys = system_prompt or "You are a focused sub-agent. Complete the task and return your result concisely."

    # Sub-agents get read-only tools + safe utilities
    _SUB_BLOCKED = {
        "sub_agent", "sub_parallel", "sub_pipeline", "workflow_run",
        "clarify", "confirm", "memory", "memory_ingest",
        "write", "edit", "bash", "execute_code", "delegate_task",
        "process_kill",
    }
    safe_tools = [t for t in (_delegate_tools or []) if t.get("name") not in _SUB_BLOCKED]

    try:
        response = await asyncio.wait_for(
            _delegate_call(messages=messages, system_prompt=sys, tools=safe_tools,
                          max_tokens=min(max_tokens, _delegate_max_tokens)),
            timeout=180.0)
    except asyncio.TimeoutError:
        return "Error: sub-agent timed out"
    except Exception as e:
        return f"Error: sub-agent failed: {e}"

    if response.tool_calls:
        from personal_agent.tools.executor import execute_tool_calls

        # Build a restricted agent context so sub-agent tools go through
        # the full execution pipeline (scope gate + hooks + dispatch).
        # Sub-agents get NO destructive privileges — write/edit/bash blocked.
        class _SubAgentCtx:
            _destructive_allowed: set = set()
            _tool_calls_this_turn: int = 0
            _max_tool_calls_per_turn: int = 10

        _sub_ctx = _SubAgentCtx()
        blocks = []
        if response.text:
            blocks.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]})
        messages.append({"role": "assistant", "content": blocks})
        await execute_tool_calls(response.tool_calls, messages, agent=_sub_ctx)
        messages.append({"role": "user", "content": [{"type": "text", "text": "Tools done. Now give your final answer."}]})
        try:
            response = await asyncio.wait_for(
                _delegate_call(messages=messages, system_prompt=sys, tools=[], max_tokens=max_tokens),
                timeout=120.0)
        except Exception as e:
            return f"Error: follow-up failed: {e}"

    text = (response.text or "").strip()

    if schema:
        try:
            schema_obj = json.loads(schema)
            result = _extract_json(text, schema_obj)
            if result is not None:
                return json.dumps(result, indent=2, ensure_ascii=False)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            messages.append({"role": "user", "content": [{"type": "text", "text": "Return ONLY valid JSON matching the schema."}]})
            try:
                r2 = await asyncio.wait_for(
                    _delegate_call(messages=messages, system_prompt=sys, tools=[], max_tokens=max_tokens),
                    timeout=60.0)
                result = _extract_json((r2.text or "").strip(), schema_obj)
                if result is not None:
                    return json.dumps(result, indent=2, ensure_ascii=False)
            except Exception:
                pass
            return f"Error: could not produce valid JSON. Raw: {text[:500]}"
        except json.JSONDecodeError:
            pass
    return text


def _extract_json(text, schema):
    import re
    try:
        obj = json.loads(text)
        if _validate(obj, schema):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if _validate(obj, schema):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _validate(obj, schema):
    if schema.get("type") != "object":
        return True
    for key in schema.get("required", []):
        if key not in obj:
            return False
    return True


async def _sub_agent(prompt, system_prompt="", schema="", max_tokens=2048):
    return await _run_agent(prompt, system_prompt, schema, max_tokens)


async def _sub_parallel(tasks_json):
    try:
        tasks = json.loads(tasks_json)
    except json.JSONDecodeError as e:
        return f"Error: invalid tasks JSON: {e}"
    if not isinstance(tasks, list) or not tasks:
        return "Error: tasks must be a non-empty JSON array"

    async def _one(task):
        return await _run_agent(
            prompt=task.get("prompt", ""),
            system_prompt=task.get("system_prompt", ""),
            schema=task.get("schema", ""),
            max_tokens=task.get("max_tokens", 2048))

    results = await asyncio.gather(*[_one(t) for t in tasks], return_exceptions=True)
    lines = []
    for i, r in enumerate(results):
        label = tasks[i].get("prompt", f"Task {i}")[:60]
        rt = str(r) if not isinstance(r, BaseException) else f"Error: {r}"
        lines.append(f"## {label}\n{rt}")
    return "\n\n".join(lines)


async def _sub_pipeline(items_json, stage_prompt, stage_system_prompt=""):
    try:
        items = json.loads(items_json)
    except json.JSONDecodeError as e:
        return f"Error: invalid items JSON: {e}"
    if not isinstance(items, list) or not items:
        return "Error: items must be a non-empty JSON array"

    async def _process(item, index):
        item_str = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
        prompt = stage_prompt.replace("{item}", item_str)
        return await _run_agent(prompt=prompt, system_prompt=stage_system_prompt, max_tokens=2048)

    results = await asyncio.gather(*[_process(item, i) for i, item in enumerate(items)], return_exceptions=True)
    lines = []
    for i, r in enumerate(results):
        label = str(items[i])[:60]
        rt = str(r) if not isinstance(r, BaseException) else f"Error: {r}"
        lines.append(f"## [{i}] {label}\n{rt}")
    return "\n\n".join(lines)


tool_registry.register(ToolEntry(
    name="sub_agent",
    description="Spawn a focused sub-agent. Call MULTIPLE TIMES in one turn to run in PARALLEL. Optional system_prompt for role, schema for structured output.",
    schema={"type": "object", "properties": {
        "prompt": {"type": "string", "description": "Task prompt."},
        "system_prompt": {"type": "string", "description": "Optional role/persona."},
        "schema": {"type": "string", "description": "Optional JSON schema for structured output."},
        "max_tokens": {"type": "integer", "description": "Max output tokens (default 2048)."},
    }, "required": ["prompt"]},
    handler=_sub_agent, toolset="builtin", is_parallel_safe=True))

tool_registry.register(ToolEntry(
    name="sub_parallel",
    description="Run multiple sub-agents concurrently, wait for ALL. Tasks JSON: [{\"prompt\": \"...\"}, ...]. Total time = slowest.",
    schema={"type": "object", "properties": {
        "tasks_json": {"type": "string", "description": "JSON array of {prompt, system_prompt?, schema?}"},
    }, "required": ["tasks_json"]},
    handler=_sub_parallel, toolset="builtin", is_parallel_safe=False))

tool_registry.register(ToolEntry(
    name="sub_pipeline",
    description="Process items through a stage independently. No barrier — each item flows immediately. Use {item} as placeholder in stage_prompt.",
    schema={"type": "object", "properties": {
        "items_json": {"type": "string", "description": "JSON array of items."},
        "stage_prompt": {"type": "string", "description": "Prompt template with {item} placeholder."},
        "stage_system_prompt": {"type": "string", "description": "Optional system prompt."},
    }, "required": ["items_json", "stage_prompt"]},
    handler=_sub_pipeline, toolset="builtin", is_parallel_safe=False))
