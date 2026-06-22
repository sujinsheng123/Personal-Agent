# Provider + Transport + Retry 系统改造 — 执行文档

## 背景

当前 Personal Agent 有三个硬伤需要修复：

1. **Provider 硬编码**：Gateway 直接手写 `ProviderProfile(...)`，没有注册表。`response_hook` 有字段但从未调用
2. **Transport 单一**：只有 `AnthropicMessagesTransport`，无法对接 OpenAI 兼容 API
3. **Retry 简陋**：只有 2 种重试，缺少 JSON 解析失败等关键 retry

**你的任务**：按以下文件清单，从上到下实现。每完成一个文件，确认 import 不报错再继续。

---

## 文件 1：`src/personal_agent/agent/retry.py` — 扩展 RetryState

**改动类型**：修改现有文件

当前只有 2 种 retry + post_tool_empty。扩展到 6 种（2 个预留）。完整替换文件内容：

```python
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
```

---

## 文件 2：`src/personal_agent/llm/provider.py` — 新增 ProviderRegistry

**改动类型**：修改现有文件，在 `ProviderProfile` 下方追加 `ProviderRegistry` + 单例 + 内置 provider 注册

`ProviderProfile` 类**保持不变**。在文件末尾追加以下内容：

```python
# ── Provider Registry ──────────────────────────────

class ProviderRegistry:
    """Global registry of LLM providers. Each provider registers its default
    profile factory so Gateway can resolve providers by name at runtime.
    """

    def __init__(self) -> None:
        self._factories: dict[str, callable] = {}

    def register(self, name: str, factory: callable) -> None:
        """Register a provider factory.
        
        factory signature: (config: Settings) -> ProviderProfile
        """
        self._factories[name] = factory

    def get(self, name: str, config) -> ProviderProfile:
        """Create a ProviderProfile from the registered factory."""
        if name not in self._factories:
            raise KeyError(f"Unknown provider: {name}. Registered: {list(self._factories)}")
        return self._factories[name](config)

    def list(self) -> list[str]:
        return list(self._factories.keys())

    @staticmethod
    def detect_api_mode(base_url: str, provider_name: str) -> str:
        """Infer api_mode from base_url (Hermes module-2 pattern).
        
        Returns one of: 'anthropic_messages' | 'chat_completions'
        """
        import os
        # Explicit config override wins
        explicit = os.getenv("LLM_API_MODE", "auto")
        if explicit != "auto":
            return explicit

        url_lower = base_url.lower()
        if "anthropic" in url_lower:
            return "anthropic_messages"
        if "openai" in url_lower:
            return "chat_completions"
        # DeepSeek's /anthropic endpoint
        if provider_name == "deepseek" and "anthropic" in url_lower:
            return "anthropic_messages"
        # Default
        return "chat_completions"


# Module-level singleton
provider_registry = ProviderRegistry()


# ── Builtin provider factories ─────────────────────

def _deepseek_factory(config) -> ProviderProfile:
    return ProviderProfile(
        name="deepseek",
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
    )


def _openai_factory(config) -> ProviderProfile:
    return ProviderProfile(
        name="openai",
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
    )


def _anthropic_factory(config) -> ProviderProfile:
    return ProviderProfile(
        name="anthropic",
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
    )


provider_registry.register("deepseek", _deepseek_factory)
provider_registry.register("openai", _openai_factory)
provider_registry.register("anthropic", _anthropic_factory)
```

**注意**：文件顶部的 import 需要保持不变（`from dataclasses import dataclass, field` 等），`ProviderProfile` 类定义不修改。

---

## 文件 3：`src/personal_agent/llm/transport_registry.py` — 新建 Transport 注册表

**改动类型**：新建文件

```python
"""Transport registry — maps api_mode → Transport class. Self-registering."""

from __future__ import annotations

from collections.abc import Callable

from personal_agent.llm.base import BaseTransport
from personal_agent.llm.provider import ProviderProfile


class TransportRegistry:
    """Maps api_mode strings to Transport factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[ProviderProfile], BaseTransport]] = {}

    def register(self, api_mode: str, factory: Callable[[ProviderProfile], BaseTransport]) -> None:
        self._factories[api_mode] = factory

    def get(self, api_mode: str, provider: ProviderProfile) -> BaseTransport:
        if api_mode not in self._factories:
            raise KeyError(f"Unknown api_mode: {api_mode}. Registered: {list(self._factories)}")
        return self._factories[api_mode](provider)

    def list_modes(self) -> list[str]:
        return list(self._factories.keys())


# Module-level singleton
transport_registry = TransportRegistry()
```

---

## 文件 4：`src/personal_agent/llm/client.py` — 新增 `call_chat_completions()`

**改动类型**：修改现有文件，在 `call_anthropic()` 下方追加新函数

保留 `call_anthropic()` 不变。在文件末尾追加：

```python
async def call_chat_completions(
    base_url: str,
    api_key: str,
    body: dict,
    *,
    timeout: float = 120.0,
    stream: bool = True,
    extra_headers: dict[str, str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """POST to OpenAI /v1/chat/completions.

    When stream=True: yields parsed SSE events (one per chunk).
    When stream=False: yields a single complete response dict.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    body_with_stream = {**body, "stream": stream}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **(extra_headers or {}),
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        if stream:
            async with client.stream("POST", url, json=body_with_stream, headers=headers) as response:
                if response.status_code == 429:
                    raise RateLimitError("429 rate limited")
                if response.status_code >= 400:
                    body_text = await response.aread()
                    raise StreamError(f"HTTP {response.status_code}: {body_text[:500]}")
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.debug("Unparsable SSE line: %s", line[:100])
                        continue
        else:
            response = await client.post(url, json=body_with_stream, headers=headers)
            if response.status_code == 429:
                raise RateLimitError("429 rate limited")
            if response.status_code >= 400:
                raise StreamError(f"HTTP {response.status_code}: {response.text[:500]}")
            yield response.json()
```

**注意**：`StreamError`、`RateLimitError` 已在文件顶部定义，不需重复。`httpx`、`json`、`logger` 已 import。

---

## 文件 5：`src/personal_agent/llm/chat_completions.py` — 新建 ChatCompletionsTransport

**改动类型**：新建文件

这是最核心的新代码。实现 OpenAI Chat Completions 协议的 Transport。

```python
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
        tool_call_deltas: dict[int, dict] = {}  # index → {id, name, arguments_json}
        usage = {"input_tokens": 0, "output_tokens": 0}
        finish_reason = ""
        model = self._provider.model

        async for event in stream:
            choices = event.get("choices", [])
            if not choices:
                # Usage info sometimes comes in a separate chunk
                u = event.get("usage")
                if u:
                    usage["input_tokens"] = u.get("prompt_tokens", 0)
                    usage["output_tokens"] = u.get("completion_tokens", 0)
                continue

            choice = choices[0]
            finish_reason = choice.get("finish_reason", "") or finish_reason

            delta = choice.get("delta", {}) or choice.get("message", {}) or {}

            # Text
            if delta.get("content"):
                text_parts.append(delta["content"])

            # Tool calls in delta (streaming) or message (non-streaming)
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
        """Convert internal (Anthropic-format) messages → OpenAI format.
        
        - Fast scan: if no _-prefixed fields → shallow copy
        - System prompt → first message with role="system"  
        - tool_use content blocks → assistant message with tool_calls
        - tool_result → tool role message
        """
        result = []

        # System prompt as first message
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Simple text content
            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            # Content is a list of blocks (Anthropic format)
            if not isinstance(content, list):
                result.append({"role": role, "content": str(content)})
                continue

            # Check for tool_use blocks → assistant with tool_calls
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
                    # tool_result → tool role
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
```

---

## 文件 6：`src/personal_agent/llm/__init__.py` — 触发运输注册

**改动类型**：修改现有文件（当前为空 `__init__.py`，只有 package marker）

```python
"""LLM layer — transports, providers, HTTP client."""

# Import triggers: register all transports with transport_registry
from personal_agent.llm.anthropic import AnthropicMessagesTransport
from personal_agent.llm.chat_completions import ChatCompletionsTransport
from personal_agent.llm.transport_registry import transport_registry
from personal_agent.llm.provider import provider_registry  # noqa: F401

# Self-register transports
def _register_transports():
    transport_registry.register(
        "anthropic_messages",
        lambda p: AnthropicMessagesTransport(p),
    )
    transport_registry.register(
        "chat_completions", 
        lambda p: ChatCompletionsTransport(p),
    )

_register_transports()
```

---

## 文件 7：`src/personal_agent/agent/loop.py` — 接入新 retry

**改动类型**：修改现有文件

在 `run_conversation()` 中的 LLM 响应处理部分，增加 `invalid_tool_retries` 和 `invalid_json_retries` 的处理。

**具体改动**：在现有 "retry: empty response" 代码块（约第 61-75 行）之后、"no tool_calls → done" 之前，插入以下逻辑：

```python
        # ── retry: invalid JSON in tool calls ──
        # Check if any tool call has empty input (JSON parse failed in transport)
        invalid_tools = [
            tc for tc in (response.tool_calls or [])
            if not tc.get("input") and tc.get("name")
        ]
        if invalid_tools and response.tool_calls:
            if agent._retry.invalid_tool_retries < agent._retry.MAX_INVALID_TOOL:
                agent._retry.invalid_tool_retries += 1
                bad_names = ", ".join(tc["name"] for tc in invalid_tools)
                ctx.messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text":
                        f"Your previous tool call(s) had invalid JSON arguments: {bad_names}. "
                        f"Please retry with valid JSON arguments."}],
                })
                logger.debug("Invalid tool retry %d: %s", agent._retry.invalid_tool_retries, bad_names)
                continue
```

**另一个改动**：在 tool 执行完成后（`await execute_tool_calls(...)` 之后），增加 post-tool empty 处理。将现有第 100-107 行替换为：

```python
        # ── retry: post-tool empty (LLM returned no text after tools ran) ──
        # The next iteration will call LLM with tool results. If LLM returns
        # empty again, trigger post_tool_empty retry.
        # This is handled at the TOP of the loop when we detect empty response
        # AFTER tool results were already in messages.
        agent._iteration_budget -= 1
        if agent._iteration_budget <= 0:
            ctx.messages.append({
                "role": "user",
                "content": [{"type": "text", "text": "请总结一下已完成的操作。"}],
            })
            break
```

**注意**：post-tool empty 的实际触发逻辑已在上面的 "retry: empty response" 中处理——如果 messages 里已有 tool 结果但 LLM 返回空，`empty_content_retries` 会触发。现有逻辑基本正确，保持即可。

---

## 文件 8：`src/personal_agent/gateway/gateway.py` — 用 registry 替代硬编码

**改动类型**：修改现有文件

找到 `_create_agent` 方法（约第 138 行），替换为使用 registry：

```python
    def _create_agent(self, session_key: str):
        from personal_agent.agent.agent import init_agent
        from personal_agent.llm.provider import provider_registry
        from personal_agent.llm.transport_registry import transport_registry
        from personal_agent.compression.simple import SimpleCompressor

        # Resolve provider via registry
        provider_name = self.config.llm_provider
        provider = provider_registry.get(provider_name, self.config)

        # Detect api_mode and get transport
        api_mode = provider_registry.detect_api_mode(
            self.config.llm_base_url, provider_name
        )
        transport = transport_registry.get(api_mode, provider)
        logger.debug("Agent transport: provider=%s api_mode=%s", provider_name, api_mode)

        compressor = SimpleCompressor() if self.config.compressor_engine == "simple" else None

        agent = init_agent(
            transport, provider,
            memory_manager=self._memory_manager,
            compressor=compressor,
            max_iterations=self.config.max_iterations,
            system_prompt_template=self._system_prompt_template,
        )
        # LRU eviction if cache too large
        if len(self._agent_cache) >= 128:
            from collections import OrderedDict
            self._agent_cache.popitem(last=False)
        self._agent_cache[session_key] = agent
        return agent
```

同时确保文件顶部有 `import logging`（已有），`logger = logging.getLogger(__name__)`（已有）。

---

## 文件 9：`src/personal_agent/llm/anthropic.py` — 调用 response_hook

**改动类型**：修改现有文件，在 `parse_stream()` 末尾加 `response_hook` 调用

在 `parse_stream()` 方法的 `return NormalizedResponse(...)` 之前，加上：

```python
        # Apply provider response_hook (vendor-specific post-processing)
        if self._provider.response_hook:
            normalized = self._provider.response_hook(normalized)
```

实际改动：在 `parse_stream()` 末尾，`finish_reason = _map_stop_reason(...)` 之后，`return` 之前，将 `NormalizedResponse(...)` 先赋给变量，调 hook，再 return。

具体：在第 132-141 行区域，改为：

```python
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
```

---

## 文件 10：`src/personal_agent/main.py` — 触发 LLM 模块 import

**改动类型**：修改现有文件，在 boot() 的 import triggers 区域加上 llm 模块

在 `import personal_agent.adapters.feishu` 附近加一行：

```python
    import personal_agent.llm              # noqa: F401 — trigger transport/provider registration
```

放在 `import personal_agent.adapters.feishu` 之前即可。

---

## 验证步骤

按顺序执行：

```bash
# 1. 检查 import 不报错
cd "c:/Users/MR/Desktop/Personal Agent"
.venv/Scripts/python -c "from personal_agent.llm.provider import provider_registry; print('Providers:', provider_registry.list())"
.venv/Scripts/python -c "from personal_agent.llm.transport_registry import transport_registry; print('Transports:', transport_registry.list_modes())"
.venv/Scripts/python -c "from personal_agent.llm.provider import provider_registry; print('api_mode:', provider_registry.detect_api_mode('https://api.deepseek.com/anthropic', 'deepseek'))"

# 2. 启动 agent，飞书发消息，确认正常回复
.venv/Scripts/python -m personal_agent
```

期望结果：
- Provider list: `['deepseek', 'openai', 'anthropic']`
- Transport list: `['anthropic_messages', 'chat_completions']`
- api_mode 检测: `anthropic_messages`（DeepSeek 的 /anthropic 端点）
- 飞书消息正常回复
