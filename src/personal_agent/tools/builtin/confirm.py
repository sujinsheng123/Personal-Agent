"""Confirm tool — Agent asks user for permission before acting.

Unlike scope gate (passive "you got blocked"), this is proactive:
  Agent: "I'm about to delete these 3 files. Confirm?"
  User:   "yes" / "no"
  Agent:  proceeds or aborts

The tool itself just returns a prompt. The user's next message is the answer.
"""

from __future__ import annotations

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry


async def _confirm(
    action: str,
    detail: str = "",
    risk: str = "low",
) -> str:
    """Ask the user to confirm an action before proceeding.

    Call this BEFORE executing any action that:
    - Modifies or deletes files
    - Pushes code / makes irreversible changes
    - Affects multiple items (batch operations)
    - The user might want to review

    The user's response will be in their next message.
    If the user says 'yes'/'ok'/'confirm'/'proceed', execute the action.
    If the user says 'no'/'cancel'/'stop', abort and explain why.

    Args:
        action: What you want to do (e.g., "Delete 3 files: a.txt, b.txt, c.txt")
        detail: Optional extra context (e.g., "These files are old backups from 2024")
        risk: "low" | "medium" | "high" — helps the user gauge urgency
    """
    risk_label = {"low": "[Risk: Low]", "medium": "[Risk: Medium]", "high": "[Risk: High ⚠]"}
    label = risk_label.get(risk, f"[Risk: {risk}]")

    lines = [f"{label} Confirm action:"]
    lines.append(f"  {action}")
    if detail:
        lines.append(f"")
        lines.append(f"  {detail}")
    lines.append(f"")
    lines.append(f"Reply 'yes' to proceed, 'no' to cancel.")

    return "\n".join(lines)


tool_registry.register(ToolEntry(
    name="confirm",
    description=(
        "Ask the user to confirm an action BEFORE executing it. "
        "Use proactively before: deleting/modifying files, pushing code, "
        "batch operations, or any irreversible action. "
        "The user will reply yes/no. If yes, proceed. If no, stop. "
        "DO NOT call this for trivial/read-only operations — use your judgment."
    ),
    schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Describe what action you want to take (e.g., 'Delete 3 old backup files')",
            },
            "detail": {
                "type": "string",
                "description": "Optional: extra context about why or what files are affected",
            },
            "risk": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Risk level: low (reversible), medium (some impact), high (irreversible/destructive)",
            },
        },
        "required": ["action"],
    },
    handler=_confirm,
    toolset="builtin",
))
