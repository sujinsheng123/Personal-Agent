"""Token estimation. char/4 fast-path for context window awareness."""

import logging

logger = logging.getLogger(__name__)

_CHAR_PER_TOKEN = 4
_DEFAULT_CONTEXT_LIMIT = 64_000  # DeepSeek


def estimate_tokens(text: str) -> int:
    """Fast token count estimate."""
    if not text:
        return 0
    return max(1, len(text) // _CHAR_PER_TOKEN)


def count_messages_tokens(messages: list[dict], system_prompt: str = "") -> int:
    """Sum token estimates across messages + system prompt."""
    total = estimate_tokens(system_prompt)
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += estimate_tokens(block.get("text", ""))
                    if block.get("type") == "tool_use":
                        total += estimate_tokens(str(block.get("input", {})))
                    if block.get("type") == "tool_result":
                        total += estimate_tokens(str(block.get("content", "")))
    return total


def count_tools_tokens(tools: list[dict]) -> int:
    """Token estimate for tool definitions."""
    total = 0
    for tool in tools:
        total += estimate_tokens(tool.get("name", ""))
        total += estimate_tokens(tool.get("description", ""))
        total += estimate_tokens(str(tool.get("input_schema", tool.get("parameters", {}))))
    return total


def context_usage(messages: list[dict], system_prompt: str = "",
                  tools: list[dict] | None = None,
                  context_limit: int = _DEFAULT_CONTEXT_LIMIT) -> dict:
    """Estimate context window usage. Returns dict with keys:
    used, limit, percent, system, messages, tools — all in tokens."""
    sys_tok = estimate_tokens(system_prompt)
    msg_tok = count_messages_tokens(messages)
    tool_tok = count_tools_tokens(tools or [])
    used = sys_tok + msg_tok + tool_tok
    return {
        "used": used,
        "limit": context_limit,
        "remaining": max(0, context_limit - used),
        "percent": round(used / max(context_limit, 1) * 100, 1),
        "system": sys_tok,
        "messages": msg_tok,
        "tools": tool_tok,
    }
