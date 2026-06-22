"""Base transport abstraction — all Provider transports implement this."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from personal_agent.models.messages import NormalizedResponse


class BaseTransport(ABC):
    """Strategy: handle protocol differences (Anthropic / OpenAI / etc.).
    The Agent loop only consumes NormalizedResponse.
    """

    @abstractmethod
    def build_request(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        max_tokens: int,
    ) -> dict:
        """Build the API request body in the target format."""
        ...

    @abstractmethod
    async def parse_stream(self, stream: AsyncIterator[bytes]) -> NormalizedResponse:
        """Parse streaming SSE events into a unified NormalizedResponse."""
        ...

    @abstractmethod
    def convert_tool_definitions(self, tools: list[dict]) -> list[dict]:
        """Convert internal tool schemas to target API format."""
        ...

    @abstractmethod
    def convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert internal message format to target API format."""
        ...

    async def close(self) -> None:
        """Optional cleanup."""
        pass
