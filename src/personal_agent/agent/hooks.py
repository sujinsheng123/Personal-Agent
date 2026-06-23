"""Simple hook system — async callback lists, called in registration order."""

from collections.abc import Callable, Awaitable
from typing import Any

Hook = Callable[..., Awaitable[Any]]


class Hooks:
    """Holds lists of async callbacks for each mount point."""

    on_message_received: list[Hook]
    on_before_llm_call: list[Hook]
    on_after_llm_call: list[Hook]
    on_before_tool_exec: list[Hook]
    on_after_tool_exec: list[Hook]
    on_before_send: list[Hook]

    def __init__(self) -> None:
        self.on_message_received = []
        self.on_before_llm_call = []
        self.on_after_llm_call = []
        self.on_before_tool_exec = []
        self.on_after_tool_exec = []
        self.on_before_send = []

    async def fire(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call all registered hooks in order. Returns last non-None result.

        When no hooks are registered, returns the first positional arg
        (pass-through) — None means "nothing changed", not "blocked".
        """
        result = None
        hooks_list = getattr(self, name, [])
        if not hooks_list:
            return args[0] if args else None
        for hook in hooks_list:
            r = await hook(*args, **kwargs)
            if r is not None:
                result = r
        return result
