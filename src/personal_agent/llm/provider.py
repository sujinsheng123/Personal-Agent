"""Provider profile — 'who to talk to' vs Transport's 'how to talk'."""

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any


@dataclass
class ProviderProfile:
    """Same ChatCompletionsTransport can serve 16+ OpenAI-compatible vendors.
    Differences live in request_hook / response_hook.
    """
    name: str                              # "deepseek", "openai", "anthropic"
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 4096

    # Hooks to patch vendor quirks (e.g., a vendor doesn't support temperature)
    request_hook: Callable[[dict], dict] | None = None
    response_hook: Callable[[dict], dict] | None = None

    extra_headers: dict[str, str] = field(default_factory=dict)
