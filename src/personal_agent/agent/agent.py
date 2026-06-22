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
    tools: list[dict] = field(default_factory=list)     # Anthropic tool schemas
    _tools_generation: int = -1                         # detect staleness

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
    system_prompt_template: str = "",
) -> Agent:
    """Wire an Agent instance. Flat initialization — no 1700-line magic."""
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=8)
    agent = Agent(
        model=provider.model,
        max_iterations=max_iterations,
        _transport=transport,
        _provider=provider,
        _memory_manager=memory_manager,
        _compressor=compressor,
        _llm_pool=pool,
        _tool_pool=pool,  # shared pool for MVP, separate later
    )
    _refresh_tools(agent)
    _build_system_prompt(agent, system_prompt_template)
    return agent


def _refresh_tools(agent: Agent) -> None:
    """Sync agent.tools with current registry state."""
    gen = tool_registry.generation
    if agent._tools_generation != gen:
        agent.tools = tool_registry.get_definitions()
        agent._tools_generation = gen
        agent._cached_system_prompt = None  # invalidate


def _build_system_prompt(agent: Agent, template: str = "") -> str:
    """Build or refresh cached system prompt."""
    parts = []
    if template:
        parts.append(template)

    # Tool list
    if agent.tools:
        tool_lines = ["可用工具："]
        for t in agent.tools:
            tool_lines.append(f"- {t['name']}: {t['description']}")
        parts.append("\n".join(tool_lines))

    # Memory
    if agent._memory_manager:
        mem_text = agent._memory_manager.get_system_prompt_text()
        if mem_text:
            parts.append(mem_text)

    agent._cached_system_prompt = "\n\n".join(parts)
    return agent._cached_system_prompt
