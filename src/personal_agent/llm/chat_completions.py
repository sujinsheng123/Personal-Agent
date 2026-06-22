"""OpenAI Chat Completions API transport — stream → NormalizedResponse."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from personal_agent.llm.base import BaseTransport
from personal_agent.llm.client import call_chat_completions
from personal_agent.llm.provider import ProviderProfile
from personal_agent.models.messages import NormalizedResponse

logger = logging.getLogger(__name__)


class ChatCompletionsTransport(BaseTransport):
    """Implements OpenAI /v1/chat/completions wire format.

    Handles both streaming SSE and non-streaming JSON responses.
    Converts internal Anthropic-format messages to OpenAI format.
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
            "messages": self.convert_messages(messages, system_prompt),
        }
        if tools:
            body["tools"] = self.convert_tool_definitions(tools)

        if self._provider.request_hook:
            body = self._provider.request_hook(body)
        return body

    # ── parse_stream ───────────────────────────────────

    async def parse_stream(self, stream: AsyncIterator[dict]) -> NormalizedResponse:
        """Parse OpenAI SSE stream → NormalizedResponse."""
        text_parts: list[str] = []
        tool_call_deltas: dict[int, dict] = {}
        usage = {"input_tokens": 0, "output_tokens": 0}
        finish_reason = ""
        model = self._provider.model

        async for event in stream:
            choices = event.get("choices", [])
            if not choices:
                u = event.get("usage")
                if u:
                    usage["input_tokens"] = u.get("prompt_tokens", 0)
                    usage["output_tokens"] = u.get("completion_tokens", 0)
                continue

            choice = choices[0]
            finish_reason = choice.get("finish_reason", "") or finish_reason

            delta = choice.get("delta", {}) or choice.get("message", {}) or {}

            if delta.get("content"):
                text_parts.append(delta["content"])

            tc_list = delta.get("tool_calls", [])
            for tc in tc_list:
                idx = tc.get("index", 0)
                if idx not in tool_call_deltas:
                    tool_call_deltas[idx] = {"id": "", "name": "", "arguments_json": ""}
                entry = tool_call_deltas[idx]
                if tc.get("id"):
                    entry["id"] = tc["id"]
                func = tc.get("function", {})
                if func.get("name"):
                    entry["name"] = func["name"]
                if func.get("arguments"):
                    entry["arguments_json"] += func["arguments"]

        # Reassemble tool calls
        tool_calls = []
        for idx in sorted(tool_call_deltas.keys()):
            block = tool_call_deltas[idx]
            try:
                inp = json.loads(block["arguments_json"]) if block["arguments_json"] else {}
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool call arguments for %s", block.get("name"))
                inp = {}
            tool_calls.append({
                "id": block["id"],
                "name": block["name"],
                "input": inp,
            })

        has_tool_calls = bool(tool_calls)
        normalized = NormalizedResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason or ("tool_calls" if has_tool_calls else "stop"),
            stop_reason=finish_reason,
            model=model,
        )

        if self._provider.response_hook:
            normalized = self._provider.response_hook(normalized)

        return normalized

    # ── format conversions ─────────────────────────────

    def convert_tool_definitions(self, tools: list[dict]) -> list[dict]:
        """Anthropic tool schema → OpenAI function schema."""
        result = []
        for tool in tools:
            entry = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {})),
                },
            }
            result.append(entry)
        return result

    def convert_messages(self, messages: list[dict], system_prompt: str = "") -> list[dict]:
        """Convert internal (Anthropic-format) messages → OpenAI format."""
        result = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                result.append({"role": role, "content": str(content)})
                continue

            text_blocks = []
            tool_calls = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    text_blocks.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        },
                    })
                elif btype == "tool_result":
                    result.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": block.get("content", ""),
                    })

            if tool_calls:
                assistant_msg = {"role": "assistant", "content": "\n".join(text_blocks) or None}
                assistant_msg["tool_calls"] = tool_calls
                result.append(assistant_msg)
            elif text_blocks:
                result.append({"role": role, "content": "\n".join(text_blocks)})

        return result

    # ── convenience call ───────────────────────────────

    async def call(
        self,
        messages: list[dict],
        system_prompt: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> NormalizedResponse:
        body = self.build_request(messages, system_prompt, tools or [], max_tokens)
        event_stream = call_chat_completions(
            base_url=self._provider.base_url,
            api_key=self._provider.api_key,
            body=body,
            stream=stream,
            extra_headers=self._provider.extra_headers,
        )
        return await self.parse_stream(event_stream)
