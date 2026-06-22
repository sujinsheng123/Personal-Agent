"""LLM layer — transports, providers, HTTP client."""

from personal_agent.llm.anthropic import AnthropicMessagesTransport
from personal_agent.llm.chat_completions import ChatCompletionsTransport
from personal_agent.llm.transport_registry import transport_registry
from personal_agent.llm.provider import provider_registry  # noqa: F401


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
