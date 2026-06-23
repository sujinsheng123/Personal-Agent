"""Low-level HTTP client for Anthropic / OpenAI compatible LLM APIs.

Retry strategy:
  - 429 Rate Limit: parse Retry-After, max 3 retries with backoff
  - 5xx Server Error: exponential backoff (1s → 2s → 4s + jitter), max 3 retries
  - Connection errors: same backoff
  - 4xx (except 429): no retry — client error
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 30.0


class StreamError(Exception):
    """Non-retryable stream error."""


class RateLimitError(Exception):
    """429 — retryable."""


def _backoff(attempt: int, retry_after: float | None = None) -> float:
    """Calculate delay: Retry-After header or exponential backoff + jitter."""
    if retry_after is not None and 0 < retry_after <= MAX_DELAY:
        return retry_after + random.uniform(0, 0.5)
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    return delay + random.uniform(0, delay * 0.3)


def _is_retryable(status_code: int | None, error: Exception | None) -> bool:
    """Decide whether to retry based on status code or exception type."""
    if status_code is not None:
        return status_code == 429 or status_code >= 500
    if error is not None:
        if isinstance(error, httpx.TimeoutException):
            return True
        if isinstance(error, (httpx.ConnectError, httpx.RemoteProtocolError)):
            return True
        return False
    return False


async def _call_with_retry(
    url: str,
    headers: dict,
    body: dict,
    *,
    timeout: float = 120.0,
    stream: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """POST with retry. Yields parsed events (SSE chunks or single response)."""
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                if stream:
                    async with client.stream("POST", url, json=body, headers=headers) as response:
                        if _is_retryable(response.status_code, None):
                            if attempt < MAX_RETRIES:
                                retry_after = _parse_retry_after(response.headers)
                                delay = _backoff(attempt, retry_after)
                                logger.warning(
                                    "HTTP %d (attempt %d/%d), retrying in %.1fs",
                                    response.status_code, attempt + 1, MAX_RETRIES, delay,
                                )
                                await asyncio.sleep(delay)
                                continue
                            raise RateLimitError(f"{response.status_code} after {MAX_RETRIES} retries")

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
                        return  # stream completed successfully
                else:
                    # Non-streaming JSON path
                    response = await client.post(url, json=body, headers=headers)

                    if _is_retryable(response.status_code, None):
                        if attempt < MAX_RETRIES:
                            retry_after = _parse_retry_after(response.headers)
                            delay = _backoff(attempt, retry_after)
                            logger.warning(
                                "HTTP %d (attempt %d/%d), retrying in %.1fs",
                                response.status_code, attempt + 1, MAX_RETRIES, delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise RateLimitError(f"{response.status_code} after {MAX_RETRIES} retries")

                    if response.status_code >= 400:
                        raise StreamError(f"HTTP {response.status_code}: {response.text[:500]}")

                    yield response.json()
                    return  # non-streaming request completed successfully

        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            if attempt < MAX_RETRIES:
                delay = _backoff(attempt)
                logger.warning(
                    "Connection error (attempt %d/%d): %s, retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, exc, delay,
                )
                await asyncio.sleep(delay)
                last_error = exc
                continue
            raise StreamError(f"Connection failed after {MAX_RETRIES} retries: {exc}") from exc

        except (RateLimitError, StreamError):
            raise  # non-retryable or exhausted retries


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    """Extract Retry-After header value in seconds."""
    value = headers.get("retry-after", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ── Public API ────────────────────────────────────────

async def call_anthropic(
    base_url: str, api_key: str, body: dict, *,
    timeout: float = 120.0, stream: bool = True,
    extra_headers: dict[str, str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """POST to Anthropic Messages API with retry."""
    url = f"{base_url.rstrip('/')}/messages"
    body = {**body, "stream": stream}
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        **(extra_headers or {}),
    }
    async for event in _call_with_retry(url, headers, body, timeout=timeout, stream=stream):
        yield event


async def call_chat_completions(
    base_url: str, api_key: str, body: dict, *,
    timeout: float = 120.0, stream: bool = True,
    extra_headers: dict[str, str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """POST to OpenAI /v1/chat/completions with retry."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = {**body, "stream": stream}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **(extra_headers or {}),
    }
    async for event in _call_with_retry(url, headers, body, timeout=timeout, stream=stream):
        yield event
