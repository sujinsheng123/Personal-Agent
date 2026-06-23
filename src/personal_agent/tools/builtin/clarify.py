"""Clarify tool — structured questions when the user's intent is unclear.

Modeled after Claude Code's AskUserQuestion: each question has a header,
options with label + description, optional previews, and multi-select.
An "Other" free-text option is always available.

Use when:
  - The user's request is ambiguous and could go multiple ways
  - There's a design/architecture choice to make
  - You need the user to pick between approaches before proceeding
"""

from __future__ import annotations

import json

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry


async def _clarify(questions: str) -> str:
    """Present structured clarifying questions to the user.

    questions: JSON array of question objects:
      [
        {
          "header": "Auth method",        // short label (max 12 chars)
          "question": "Which auth library should we use?",
          "options": [
            {
              "label": "JWT (Recommended)",  // short (1-5 words)
              "description": "Stateless, good for microservices"  // why/when
            },
            {
              "label": "Session",
              "description": "Server-side sessions, simpler to revoke"
            }
          ],
          "multiSelect": false              // allow multiple answers?
        }
      ]

    The user will see each question with its options, pick one (or type
    their own answer via the always-available "Other" option), and their
    response will appear in the next conversation turn.
    """
    try:
        items = json.loads(questions)
    except json.JSONDecodeError as e:
        return f"Error: invalid questions JSON: {e}"

    if not isinstance(items, list) or not items:
        return "Error: questions must be a non-empty JSON array"

    lines: list[str] = []

    for i, q in enumerate(items):
        if not isinstance(q, dict):
            continue

        header = q.get("header", f"Question {i + 1}")
        question = q.get("question", "")
        options = q.get("options", [])
        multi = q.get("multiSelect", False)

        # ── Question block ──
        if i > 0:
            lines.append("")
            lines.append("---")
            lines.append("")
        lines.append(f"## {header}")
        lines.append(f"{question}")
        lines.append("")

        # ── Options ──
        for j, opt in enumerate(options):
            if not isinstance(opt, dict):
                continue
            label = opt.get("label", f"Option {j + 1}")
            desc = opt.get("description", "")
            lines.append(f"{j + 1}. **{label}**")
            if desc:
                lines.append(f"   {desc}")

        lines.append(f"{len(options) + 1}. **Other** — (type your own answer)")

        if multi:
            lines.append(f"_You can select multiple options (e.g., '1, 3')_")

    lines.append("")
    lines.append("Reply with your choice(s).")

    return "\n".join(lines)


tool_registry.register(ToolEntry(
    name="clarify",
    description=(
        "Ask the user clarifying questions when their intent is unclear. "
        "Each question has a short header, a clear question text, and options "
        "with label + description. An 'Other' free-text choice is always available. "
        "Use when: the user's request is ambiguous, there are multiple valid "
        "approaches, or you need the user to make a design decision. "
        "Format questions as a JSON array — see the tool's input schema for details."
    ),
    schema={
        "type": "object",
        "properties": {
            "questions": {
                "type": "string",
                "description": (
                    "JSON array of question objects. Each object has:\n"
                    '  "header": short label like "Auth method" (max 12 chars)\n'
                    '  "question": the full question text\n'
                    '  "options": array of {label, description} — label is 1-5 words, description explains the choice\n'
                    '  "multiSelect": true/false (default false) — allow picking multiple options\n'
                    "An 'Other' free-text option is always added automatically."
                ),
            },
        },
        "required": ["questions"],
    },
    handler=_clarify,
    toolset="builtin",
))
