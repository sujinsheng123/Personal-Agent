"""Clarify tool — ask the user a structured question when more info is needed.

Unlike most tools, this doesn't just return a result — it pauses the agent
and prompts the user for input. The user's next message is the answer.
"""

from __future__ import annotations

import json

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry


async def _clarify(
    question: str,
    choices: list[str] | None = None,
    allow_freeform: bool = True,
) -> str:
    """Ask the user a clarifying question. The conversation pauses here
    and the user's response becomes the answer.

    Args:
        question: The question to ask the user
        choices: Optional list of predefined choices (max 6)
        allow_freeform: Whether user can type a free-text answer (default True)
    """
    lines = [f"[CLARIFY] {question}"]

    if choices:
        lines.append("")
        for i, c in enumerate(choices[:6], 1):
            lines.append(f"  {i}. {c}")
        if allow_freeform:
            lines.append(f"  (or type your own answer)")
        lines.append("")
        lines.append("Please reply with your choice number or free-text answer.")
    else:
        lines.append("")
        lines.append("Please reply with your answer.")

    return "\n".join(lines)


tool_registry.register(ToolEntry(
    name="clarify",
    description=(
        "Ask the user a clarifying question when you need more information to proceed. "
        "Use when: you're unsure about what the user wants, there are multiple valid "
        "ways to interpret the request, or you need the user to make a choice. "
        "Provide a clear question and optional predefined choices."
    ),
    schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The clarifying question to ask the user",
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of predefined choices (max 6)",
            },
            "allow_freeform": {
                "type": "boolean",
                "description": "Whether the user can type a free-text answer instead of picking a choice",
            },
        },
        "required": ["question"],
    },
    handler=_clarify,
    toolset="builtin",
))
