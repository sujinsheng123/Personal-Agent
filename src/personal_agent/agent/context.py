"""build_turn_context — assemble messages, check tokens, apply compression."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

from personal_agent.llm.token_counter import count_messages_tokens, count_tools_tokens

logger = logging.getLogger(__name__)

CONTEXT_LIMIT = 64000    # DeepSeek context window
THRESHOLD = 0.6          # compress at 60% usage
PROTECT_FIRST = 2        # messages at head to protect
PROTECT_LAST = 6         # messages at tail to protect


@dataclass
class TurnContext:
    user_message: str
    original_user_message: str
    messages: list[dict]              # working copy, persisted
    conversation_history: list[dict]  # read-only original from DB
    active_system_prompt: str
    turn_id: str = ""
    current_turn_user_idx: int = 0
    should_review_memory: bool = False
    was_compressed: bool = False            # True if compression ran this turn
    pre_compress_message_count: int = 0     # message count before compression


def build_turn_context(
    agent,
    user_message: str,
    history: list[dict] | None = None,
) -> TurnContext:
    """Prepare messages for a conversation turn.
    Does NOT build api_messages — that happens inside the while loop.
    """
    import time
    import uuid

    # Reset per-turn state
    agent._iteration_budget = agent.max_iterations
    agent._retry.reset()
    agent._interrupt_requested = False
    agent._tool_calls_this_turn = 0
    agent._destructive_allowed.clear()

    # Refresh tools (if registry changed)
    from personal_agent.agent.agent import _refresh_tools, _build_system_prompt
    _refresh_tools(agent)
    if agent._cached_system_prompt is None:
        _build_system_prompt(agent)

    # Copy history
    conversation_history = list(history or [])
    messages = copy.deepcopy(conversation_history)

    # Append current user message
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": user_message}],
    })
    user_idx = len(messages) - 1

    # Token check + compression
    pre_count = len(messages)
    messages = _check_and_compress(agent, messages)
    was_compressed = len(messages) != pre_count

    turn_id = f"{uuid.uuid4().hex[:8]}"

    return TurnContext(
        user_message=user_message,
        original_user_message=user_message,
        messages=messages,
        conversation_history=conversation_history,
        active_system_prompt=agent._cached_system_prompt or "",
        turn_id=turn_id,
        current_turn_user_idx=user_idx,
        was_compressed=was_compressed,
        pre_compress_message_count=pre_count,
    )


def _check_and_compress(agent, messages: list[dict]) -> list[dict]:
    """If estimated tokens exceed threshold, compress via ContextEngine."""
    if agent._compressor is None:
        return messages

    total = (
        count_messages_tokens(messages)
        + count_messages_tokens([], agent._cached_system_prompt or "")
        + count_tools_tokens(agent.tools)
    )

    if not agent._compressor.should_compress(total, messages):
        return messages

    logger.info("Compressing: %d tokens > %d limit", total, agent._compressor.threshold_tokens)
    try:
        result = agent._compressor.compress(
            messages,
            agent._cached_system_prompt or "",
            agent._transport,
        )
        return result
    except Exception:
        logger.exception("Compression failed, falling back to truncation")
        return _truncate(messages, agent._compressor.protect_head, agent._compressor.protect_tail)


def _truncate(messages: list[dict], head: int = 2, tail: int = 6) -> list[dict]:
    """Fallback: drop oldest messages except protected ones."""
    if len(messages) <= head + tail:
        return messages
    return messages[:head] + messages[-tail:]
