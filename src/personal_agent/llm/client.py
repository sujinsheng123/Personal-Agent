"""Low-level HTTP client for Anthropic-compatible LLM APIs.
Handles both streaming (SSE) and non-streaming (JSON) responses.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class StreamError(Exception):
    """Non-retryable stream error."""


class RateLimitError(Exception):
    """429 — retry after delay."""


async def call_anthropic(
    base_url: str,
    api_key: str,
    body: dict,
    *,
    timeout: float = 120.0,
    stream: bool = True,
    extra_headers: dict[str, str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """POST to Anthropic Messages API.

    When stream=True: yields parsed SSE events.
    When stream=False (DeepSeek): yields a single full response dict.
    """
    url = f"{base_url.rstrip('/')}/messages"
    body_with_stream = {**body, "stream": stream}

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        **(extra_headers or {}),
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        if stream:
            # Streaming SSE path
            async with client.stream("POST", url, json=body_with_stream, headers=headers) as response:
                if response.status_code == 429:
                    raise RateLimitError("429 rate limited")
                if response.status_code >= 400:
                    body_text = await response.aread()
                    raise StreamError(f"HTTP {response.status_code}: {body_text[:500]}")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if not data_str:
                            continue
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.debug("Unparsable SSE line: %s", line[:100])
                            continue
        else:
            # Non-streaming JSON path (DeepSeek compatible)
            response = await client.post(url, json=body_with_stream, headers=headers)
            if response.status_code == 429:
                raise RateLimitError("429 rate limited")
            if response.status_code >= 400:
                raise StreamError(f"HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            yield data
