"""Bridge tools — registered on import as normal tools.
When LLM calls tool_search/describe/call, these handlers manage deferrable tools.

Flow: tool_search (discover) → tool_describe (get schema) → LLM calls tool directly
tool_call is a fallback for tools that can't be surfaced mid-turn — but destructive
tools are NOT callable through it (they must go through the full executor pipeline).
"""

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import (
    tool_registry,
    dispatch_tool_search,
    dispatch_tool_describe,
)

MAX_RESULT_CHARS = 8000


async def _tool_search(query: str) -> str:
    return await dispatch_tool_search(query)


async def _tool_describe(name: str) -> str:
    """Return full schema for a tool. After the LLM sees this, it should
    call the tool directly by name in the next iteration — NOT via tool_call.
    """
    return await dispatch_tool_describe(name)


async def _tool_call(name: str, arguments: dict) -> str:
    """Fallback: execute a deferrable tool by name. Only works for safe tools —
    destructive tools are blocked and must be called directly with /allow."""
    from personal_agent.tools.registry import tool_registry as _tr

    entry = _tr.get(name)
    if entry is None:
        return f"Error: unknown tool '{name}'"

    # Destructive tools must go through the full executor pipeline
    # (scope gate + checkpoint + hooks), not through this shortcut
    if entry.is_destructive:
        return (
            f"Error: destructive tool '{name}' cannot be called via tool_call. "
            f"Send /allow to authorize it, then call '{name}' directly in your next response."
        )

    # Safe tools: execute with truncation matching executor post-processing
    try:
        result = await entry.handler(**arguments)
    except Exception as exc:
        result = f"Error: {exc}"

    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + f"\n\n...({len(result) - MAX_RESULT_CHARS} more chars truncated)"

    return result


tool_registry.register(ToolEntry(
    name="tool_search",
    description="Search for tools by keyword. Returns matching tools with name, description, and "
                "full input_schema. After searching, call the matched tool DIRECTLY by name — "
                "you already have the schema and can construct the call immediately.",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to search for in tool names and descriptions"},
        },
        "required": ["query"],
    },
    handler=_tool_search,
    toolset="system",
    is_parallel_safe=True,
))

tool_registry.register(ToolEntry(
    name="tool_describe",
    description="Get the full parameter schema for a specific tool. "
                "After calling this, call the tool directly by name.",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact tool name from tool_search results"},
        },
        "required": ["name"],
    },
    handler=_tool_describe,
    toolset="system",
    is_parallel_safe=True,
))

tool_registry.register(ToolEntry(
    name="tool_call",
    description="Execute a safe, non-destructive tool by name. "
                "If the tool is destructive, it will be blocked — call it directly after /allow instead.",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Tool name to execute"},
            "arguments": {"type": "object", "description": "Tool arguments as a JSON object"},
        },
        "required": ["name", "arguments"],
    },
    handler=_tool_call,
    toolset="system",
    is_parallel_safe=False,
))
