"""Anthropic Messages API transport — stream → NormalizedResponse."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from personal_agent.llm.base import BaseTransport
from personal_agent.llm.client import call_anthropic
from personal_agent.llm.provider import ProviderProfile
from personal_agent.models.messages import NormalizedResponse

logger = logging.getLogger(__name__)


class AnthropicMessagesTransport(BaseTransport):
    """Implements Anthropic Messages API wire format.
    Also compatible with DeepSeek's /anthropic endpoint.
    """

    def __init__(self, provider: ProviderProfile) -> None:
        self._provider = provider

    # ── build_request ──────────────────────────────────

    def build_request(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        max_tokens: int,
    ) -> dict:
        body: dict = {
            "model": self._provider.model,
            "max_tokens": max_tokens or self._provider.max_tokens,
            "messages": self.convert_messages(messages),
        }
        if system_prompt:
            body["system"] = system_prompt
        if tools:
            body["tools"] = self.convert_tool_definitions(tools)

        if self._provider.request_hook:
            body = self._provider.request_hook(body)
        return body

    # ── parse_stream ───────────────────────────────────

    async def parse_stream(self, stream: AsyncIterator[dict]) -> NormalizedResponse:
        """Parse stream events (SSE or single non-streaming JSON) → NormalizedResponse."""
        text_parts: list[str] = []
        tool_use_blocks: dict[int, dict] = {}
        usage = {"input_tokens": 0, "output_tokens": 0}
        stop_reason = ""
        model = self._provider.model
        seen_message = False  # tracks if we got the full message event (non-streaming)

        async for event in stream:
            etype = event.get("type", "")

            # Non-streaming: DeepSeek returns a single "message" event
            if etype == "message":
                seen_message = True
                model = event.get("model", model)
                stop_reason = event.get("stop_reason", "")
                usage_in = event.get("usage", {})
                usage["input_tokens"] = usage_in.get("input_tokens", 0)
                usage["output_tokens"] = usage_in.get("output_tokens", 0)

                for block in event.get("content", []):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_use_blocks[len(tool_use_blocks)] = {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": block.get("input", {}),
                        }
                break  # single event, done

            # Streaming SSE events
            if etype == "message_start":
                msg = event.get("message", {})
                usage["input_tokens"] = msg.get("usage", {}).get("input_tokens", 0)
                model = msg.get("model", model)

            elif etype == "content_block_start":
                block = event.get("content_block", {})
                idx = event.get("index", 0)
                if block.get("type") == "tool_use":
                    tool_use_blocks[idx] = {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input_json": "",
                    }

            elif etype == "content_block_delta":
                delta = event.get("delta", {})
                idx = event.get("index", 0)
                if delta.get("type") == "text_delta":
                    text_parts.append(delta.get("text", ""))
                elif delta.get("type") == "input_json_delta":
                    if idx in tool_use_blocks:
                        tool_use_blocks[idx]["input_json"] += delta.get("partial_json", "")

            elif etype == "message_delta":
                usage["output_tokens"] = event.get("usage", {}).get("output_tokens", 0)
                stop_reason = event.get("delta", {}).get("stop_reason", "") or stop_reason

            elif etype == "message_stop":
                pass

        # Reassemble tool calls
        tool_calls = []
        for idx in sorted(tool_use_blocks.keys()):
            block = tool_use_blocks[idx]
            if "input" in block:
                inp = block["input"]  # non-streaming: already parsed
            else:
                try:
                    inp = json.loads(block.get("input_json", "")) if block.get("input_json") else {}
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool input JSON for %s", block.get("name"))
                    inp = {}
            tool_calls.append({
                "id": block["id"],
                "name": block["name"],
                "input": inp,
            })

        finish_reason = _map_stop_reason(stop_reason, bool(tool_calls))

        normalized = NormalizedResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            model=model,
        )

        if self._provider.response_hook:
            normalized = self._provider.response_hook(normalized)

        return normalized

    # ── format conversions ─────────────────────────────

    def convert_tool_definitions(self, tools: list[dict]) -> list[dict]:
        """Internal tool dicts → Anthropic tool schema."""
        result = []
        for tool in tools:
            entry = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", tool.get("parameters", {})),
            }
            result.append(entry)
        return result

    def convert_messages(self, messages: list[dict]) -> list[dict]:
        """Already in Anthropic format — pass through (fast path)."""
        return messages

    # ── convenience call ───────────────────────────────

    async def call(
        self,
        messages: list[dict],
        system_prompt: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        stream: bool = False,  # DeepSeek doesnʼt support SSE streaming
    ) -> NormalizedResponse:
        """Build request, stream, parse — all in one call."""
        body = self.build_request(
            messages, system_prompt, tools or [], max_tokens
        )
        event_stream = call_anthropic(
            base_url=self._provider.base_url,
            api_key=self._provider.api_key,
            body=body,
            stream=stream,
            extra_headers=self._provider.extra_headers,
        )
        return await self.parse_stream(event_stream)


def _map_stop_reason(stop_reason: str, has_tool_calls: bool) -> str:
    """Map Anthropic stop_reason to our finish_reason."""
    if stop_reason == "end_turn":
        return "end_turn"
    if stop_reason == "tool_use":
        return "tool_use"
    if stop_reason == "max_tokens":
        return "max_tokens"
    if stop_reason == "stop_sequence":
        return "stop"
    # Fallback
    if has_tool_calls:
        return "tool_use"
    return "end_turn"
