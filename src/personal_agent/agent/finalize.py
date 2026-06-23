"""finalize_turn — persist new messages to DB, update session counters."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def unpack_message(msg: dict) -> tuple[str, str, list | None, str | None, str | None]:
    """Unpack an Anthropic-format message dict into flat fields for DB storage.

    Returns: (role, content, tool_calls, tool_name, tool_call_id)

    Used by finalize_turn, session_store.save_transcript, and
    session_store.create_compressed_session — single source of truth.
    """
    role = msg.get("role", "user")
    content = ""
    tool_calls = None
    tool_name = None
    tool_call_id = None

    if isinstance(msg.get("content"), list):
        for block in msg["content"]:
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls = tool_calls or []
                tool_calls.append({
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                })
                tool_name = block.get("name")
            elif block.get("type") == "tool_result":
                content = str(block.get("content", ""))
                tool_call_id = block.get("tool_use_id", "")
    elif isinstance(msg.get("content"), str):
        content = msg["content"]

    return role, content, tool_calls, tool_name, tool_call_id


async def finalize_turn(db, session_id: str, ctx, previous_message_count: int) -> None:
    """Persist new messages added during this turn."""
    new_messages = ctx.messages[previous_message_count:]
    if not new_messages:
        return

    for msg in new_messages:
        role, content, tool_calls, tool_name, tool_call_id = unpack_message(msg)
        await db.save_message(
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )

    await db.update_last_active(session_id, increment_message=True)
    logger.debug("Persisted %d messages for session %s", len(new_messages), session_id)
