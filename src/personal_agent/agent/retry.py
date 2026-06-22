"""Retry counters and logic for Agent loop defects.

Per-turn: reset in build_turn_context().
Corresponds to 6 specific LLM response defects (Hermes pattern).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetryState:
    empty_content_retries: int = 0           # LLM returned no text, no tool_calls
    invalid_tool_retries: int = 0            # tool_call JSON parse failed
    invalid_json_retries: int = 0            # response body JSON malformed
    incomplete_scratchpad_retries: int = 0   # Anthropic scratchpad truncated (reserved)
    thinking_prefill_retries: int = 0        # thinking block prefill failed (reserved)
    post_tool_empty_retried: bool = False    # after tools ran, LLM returned empty

    MAX_EMPTY_CONTENT = 2
    MAX_INVALID_TOOL = 2
    MAX_INVALID_JSON = 2
    MAX_INCOMPLETE_SCRATCHPAD = 1
    MAX_THINKING_PREFILL = 1
    MAX_POST_TOOL_EMPTY = 1

    def reset(self) -> None:
        self.empty_content_retries = 0
        self.invalid_tool_retries = 0
        self.invalid_json_retries = 0
        self.incomplete_scratchpad_retries = 0
        self.thinking_prefill_retries = 0
        self.post_tool_empty_retried = False
