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
    context_window: int = 0                # 0 = auto-detect from model name

    # Hooks to patch vendor quirks (e.g., a vendor doesn't support temperature)
    request_hook: Callable[[dict], dict] | None = None
    response_hook: Callable[[dict], dict] | None = None

    extra_headers: dict[str, str] = field(default_factory=dict)


def _detect_context_window(model: str) -> int:
    """Infer context window from model name. Conservative estimates."""
    m = model.lower()
    if "1m" in m or "1.0m" in m:
        return 1_000_000
    if "200k" in m or "claude" in m:
        return 200_000
    if "128k" in m or "gpt-4" in m or "gpt-4o" in m:
        return 128_000
    if "32k" in m:
        return 32_000
    if "deepseek" in m:  # all DeepSeek models = 1M (v4, r1, chat)
        return 1_000_000
    return 64_000


# ── Provider Registry ──────────────────────────────────

class ProviderRegistry:
    """Global registry of LLM providers. Each provider registers its default
    profile factory so Gateway can resolve providers by name at runtime.
    """

    def __init__(self) -> None:
        self._factories: dict[str, callable] = {}

    def register(self, name: str, factory: callable) -> None:
        self._factories[name] = factory

    def get(self, name: str, config) -> ProviderProfile:
        if name not in self._factories:
            raise KeyError(f"Unknown provider: {name}. Registered: {list(self._factories)}")
        return self._factories[name](config)

    def list(self) -> list[str]:
        return list(self._factories.keys())

    @staticmethod
    def detect_api_mode(base_url: str, provider_name: str) -> str:
        """Infer api_mode from base_url. Returns 'anthropic_messages' | 'chat_completions'."""
        import os
        explicit = os.getenv("LLM_API_MODE", "auto")
        if explicit != "auto":
            return explicit

        url_lower = base_url.lower()
        if "anthropic" in url_lower:
            return "anthropic_messages"
        if "openai" in url_lower or "openrouter" in url_lower:
            return "chat_completions"
        if provider_name == "deepseek" and "anthropic" in url_lower:
            return "anthropic_messages"
        return "chat_completions"


# Module-level singleton
provider_registry = ProviderRegistry()


# ── Builtin provider factories ─────────────────────────

def _build_context_window(config) -> int:
    if config.llm_context_window > 0:
        return config.llm_context_window
    return _detect_context_window(config.llm_model)


def _deepseek_factory(config) -> ProviderProfile:
    return ProviderProfile(
        name="deepseek", base_url=config.llm_base_url, api_key=config.llm_api_key,
        model=config.llm_model, max_tokens=config.llm_max_tokens,
        context_window=_build_context_window(config),
    )

def _openai_factory(config) -> ProviderProfile:
    return ProviderProfile(
        name="openai", base_url=config.llm_base_url, api_key=config.llm_api_key,
        model=config.llm_model, max_tokens=config.llm_max_tokens,
        context_window=_build_context_window(config),
    )

def _anthropic_factory(config) -> ProviderProfile:
    return ProviderProfile(
        name="anthropic", base_url=config.llm_base_url, api_key=config.llm_api_key,
        model=config.llm_model, max_tokens=config.llm_max_tokens,
        context_window=_build_context_window(config),
    )

def _openrouter_factory(config) -> ProviderProfile:
    return ProviderProfile(
        name="openrouter",
        base_url=config.llm_base_url if "openrouter" in (config.llm_base_url or "").lower()
                 else "https://openrouter.ai/api/v1",
        api_key=config.llm_api_key,
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
        context_window=_build_context_window(config),
        extra_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Personal Agent",
        },
    )


provider_registry.register("deepseek", _deepseek_factory)
provider_registry.register("openai", _openai_factory)
provider_registry.register("anthropic", _anthropic_factory)
provider_registry.register("openrouter", _openrouter_factory)
