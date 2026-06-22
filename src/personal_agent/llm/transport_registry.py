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
