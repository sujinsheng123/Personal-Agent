"""Token estimation. char/4 fast-path + tiktoken fallback."""

import logging

logger = logging.getLogger(__name__)

# Rough fallback: 1 token ≈ 4 characters (OK for English, conservative for Chinese)
_CHAR_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Fast token count estimate. For MVP truncation decisions."""
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
                    # tool_use input
                    if block.get("type") == "tool_use":
                        total += estimate_tokens(str(block.get("input", {})))
                    # tool_result content
                    if block.get("type") == "tool_result":
                        total += estimate_tokens(str(block.get("content", "")))
    return total


def count_tools_tokens(tools: list[dict]) -> int:
    """Token estimate for tool definitions sent in system prompt."""
    total = 0
    for tool in tools:
        total += estimate_tokens(tool.get("name", ""))
        total += estimate_tokens(tool.get("description", ""))
        total += estimate_tokens(str(tool.get("input_schema", tool.get("parameters", {}))))
    return total
