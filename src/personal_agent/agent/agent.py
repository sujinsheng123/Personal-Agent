"""Agent dataclass — flat runtime state container. init_agent() does the wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from personal_agent.agent.hooks import Hooks
from personal_agent.agent.retry import RetryState
from personal_agent.llm.provider import ProviderProfile
from personal_agent.tools.registry import tool_registry


@dataclass
class Agent:
    # ── identity (set by init_agent, never changes) ──
    model: str = ""
    max_iterations: int = 30

    # ── transport & provider ──
    _transport: Any = None                 # BaseTransport instance
    _provider: ProviderProfile | None = None

    # ── tools ──
    tools: list[dict] = field(default_factory=list)
    enabled_toolsets: list[str] | None = None   # None = all tools
    _tools_generation: int = -1

    # ── system prompt ──
    _cached_system_prompt: str | None = None  # None=not built, ""=empty, str=present

    # ── memory ──
    _memory_manager: Any = None

    # ── compressor ──
    _compressor: Any = None

    # ── hooks ──
    hooks: Hooks = field(default_factory=Hooks)

    # ── per-session counters (accumulate across turns) ──
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_api_calls: int = 0

    # ── per-turn state (reset each build_turn_context) ──
    _iteration_budget: int = 0
    _retry: RetryState = field(default_factory=RetryState)
    _interrupt_requested: bool = False
    _tool_calls_this_turn: int = 0
    _destructive_allowed: set[str] = field(default_factory=set)  # {"write", "shell", "all"}
    _max_tool_calls_per_turn: int = 20
    _pending_skill_injection: str | None = None  # set by Gateway, consumed by context

    # ── memory review (Hermes-style background nudge) ──
    _turns_since_memory: int = 0
    _memory_review_interval: int = 10  # nudge every N turns, 0=disabled

    # ── pool split (same pool for MVP, separate later) ──
    _llm_pool: Any = None
    _tool_pool: Any = None


def init_agent(
    transport,
    provider: ProviderProfile,
    *,
    memory_manager=None,
    compressor=None,
    max_iterations: int = 30,
    max_tool_calls_per_turn: int = 20,
    memory_review_interval: int = 10,
    system_prompt_template: str = "",
    enabled_toolsets: list[str] | None = None,
) -> Agent:
    """Wire an Agent instance. Flat initialization — no 1700-line magic."""
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=8)
    agent = Agent(
        model=provider.model,
        max_iterations=max_iterations,
        _max_tool_calls_per_turn=max_tool_calls_per_turn,
        _memory_review_interval=memory_review_interval,
        _transport=transport,
        _provider=provider,
        _memory_manager=memory_manager,
        _compressor=compressor,
        enabled_toolsets=enabled_toolsets,
        _llm_pool=pool,
        _tool_pool=pool,  # shared pool for MVP, separate later
    )
    _refresh_tools(agent)
    _build_system_prompt(agent, system_prompt_template)
    _register_default_hooks(agent)
    return agent


def _register_default_hooks(agent: Agent) -> None:
    """Non-restrictive default hooks for observability."""
    import logging as _logging
    _log = _logging.getLogger("personal_agent.hooks")

    async def _log_llm_usage(response, usage):
        _log.info("LLM call: in=%d out=%d",
                  usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        return response

    agent.hooks.on_after_llm_call.append(_log_llm_usage)


def _refresh_tools(agent: Agent) -> None:
    """Sync agent.tools with current registry state, respecting enabled_toolsets."""
    gen = tool_registry.generation
    if agent._tools_generation != gen:
        agent.tools = tool_registry.get_definitions(
            enabled_toolsets=agent.enabled_toolsets,
            quiet_mode=True,
        )
        agent._tools_generation = gen
        agent._cached_system_prompt = None  # invalidate


def _build_system_prompt(agent: Agent, template: str = "") -> str:
    """Build or refresh cached system prompt."""
    parts = []
    if template:
        parts.append(template)

    # Tool list (sorted for deterministic byte stream → cache hits)
    if agent.tools:
        tool_lines = ["可用工具："]
        for t in sorted(agent.tools, key=lambda t: t["name"]):
            tool_lines.append(f"- {t['name']}: {t['description']}")
        parts.append("\n".join(tool_lines))

    # Memory
    if agent._memory_manager:
        mem_text = agent._memory_manager.get_system_prompt_text()
        if mem_text:
            parts.append(mem_text)

    agent._cached_system_prompt = "\n\n".join(parts)
    return agent._cached_system_prompt
