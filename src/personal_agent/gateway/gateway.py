"""Gateway — central orchestrator: adapters, routing, session management, agent dispatch."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

from personal_agent.adapters.base import platform_registry
from personal_agent.agent.hooks import Hooks
from personal_agent.gateway.session_store import SessionStore

logger = logging.getLogger(__name__)


class Gateway:
    def __init__(self, config, db, memory_manager, system_prompt_template: str = "") -> None:
        self.config = config
        self.db = db
        self._memory_manager = memory_manager
        self._system_prompt_template = system_prompt_template
        self._session_store = SessionStore(db, config.agent_data_dir)
        self._adapters: list = []
        self._running_agents: dict[str, bool] = {}
        self._agent_cache: OrderedDict[str, object] = OrderedDict()
        self.hooks = Hooks()
        self._shutdown_event = asyncio.Event()

    # ── lifecycle ─────────────────────────────────────

    async def start(self) -> None:
        await self._session_store.initialize()

        for entry in platform_registry.list():
            if entry.check_fn(self.config):
                adapter = entry.factory(self.config, self.db)
                adapter.set_message_handler(self._handle_message)
                try:
                    await adapter.connect()
                except Exception:
                    logger.exception("Platform '%s' connect failed", entry.name)
                    continue
                self._adapters.append(adapter)
                logger.info("Platform '%s' connected", entry.name)
            else:
                logger.warning("Platform '%s' skipped: check_fn returned False", entry.name)

        logger.info("Gateway started with %d platform(s)", len(self._adapters))

    async def stop(self) -> None:
        for adapter in self._adapters:
            try:
                await adapter.disconnect()
            except Exception:
                logger.exception("Error disconnecting adapter")
        self._shutdown_event.set()

    async def wait_for_shutdown(self) -> None:
        await self._shutdown_event.wait()

    # ── message handling ──────────────────────────────

    async def _handle_message(self, event) -> str | None:
        """Gateway callback from adapter. Returns response text."""
        session_key = f"{event.source.platform}:{event.source.chat_id}:{event.source.user_id}"

        # 1. Hook: on_message_received (only if hooks registered)
        if self.hooks.on_message_received:
            hook_result = await self.hooks.fire("on_message_received", event)
            if hook_result is None:
                return None  # dropped
            if hook_result is not event:
                event = hook_result

        # 2. Authorization (skip internal events)
        if not event.internal:
            if not self._authorize(event):
                return "抱歉，你没有权限使用此服务。"

        # 3. Command detection
        if event.text.startswith("/"):
            return await self._handle_command(event, session_key)

        # 4. Busy check
        if session_key in self._running_agents:
            return "我正在处理你上一条消息，请稍候..."

        # 5. Mark running → process → cleanup
        self._running_agents[session_key] = True
        try:
            return await self._handle_message_with_agent(event, session_key)
        finally:
            self._running_agents.pop(session_key, None)

    # ── agent dispatch ────────────────────────────────

    async def _handle_message_with_agent(self, event, session_key: str) -> str:
        session = await self._session_store.get_or_create(session_key, event.source)
        history = await self._session_store.load_history(session.session_id)
        previous_count = len(history)

        agent = self._get_or_create_agent(session_key)

        from personal_agent.agent.context import build_turn_context
        from personal_agent.agent.loop import run_conversation

        ctx = build_turn_context(agent, event.text, history)
        result = await run_conversation(agent, ctx)

        # Persist
        if not result.get("context_overflow"):
            await self._session_store.save_transcript(
                session.session_id, result["messages"], previous_count
            )

        # Hook: on_before_send
        final = result.get("final_response", "")
        hook_result = await self.hooks.fire("on_before_send", final, event.source)
        if isinstance(hook_result, str):
            final = hook_result

        return final or "..."

    def _get_or_create_agent(self, session_key: str):
        """Return cached Agent if available, otherwise create and cache."""
        if session_key in self._agent_cache:
            agent = self._agent_cache[session_key]
            # Check if tools stale (registry generation changed)
            from personal_agent.tools.registry import tool_registry
            if agent._tools_generation == tool_registry.generation:
                return agent
            # Tools changed — evict stale cache entry
            del self._agent_cache[session_key]

        return self._create_agent(session_key)

    def _create_agent(self, session_key: str):
        from personal_agent.agent.agent import init_agent
        from personal_agent.llm.provider import provider_registry
        from personal_agent.llm.transport_registry import transport_registry
        from personal_agent.compression.simple import SimpleCompressor

        # Resolve provider via registry
        provider_name = self.config.llm_provider
        provider = provider_registry.get(provider_name, self.config)

        # Detect api_mode and get transport
        api_mode = provider_registry.detect_api_mode(
            self.config.llm_base_url, provider_name
        )
        transport = transport_registry.get(api_mode, provider)
        logger.debug("Agent transport: provider=%s api_mode=%s", provider_name, api_mode)

        compressor = SimpleCompressor() if self.config.compressor_engine == "simple" else None

        agent = init_agent(
            transport, provider,
            memory_manager=self._memory_manager,
            compressor=compressor,
            max_iterations=self.config.max_iterations,
            system_prompt_template=self._system_prompt_template,
        )
        # LRU eviction if cache too large
        if len(self._agent_cache) >= 128:
            from collections import OrderedDict
            self._agent_cache.popitem(last=False)
        self._agent_cache[session_key] = agent
        return agent


    # ── commands ──────────────────────────────────────

    async def _handle_command(self, event, session_key: str) -> str | None:
        text = event.text.strip()

        if text.startswith("/new"):
            await self._session_store.delete_session(session_key)
            return "会话已重置。开始新的对话吧。"

        if text.startswith("/stop"):
            # Interrupt running agent (if any in _running_agents)
            # For now, just clear pending
            return "已停止。有新消息时重新开始。"

        if text.startswith("/help"):
            return (
                "可用命令:\n"
                "/new - 重置对话\n"
                "/stop - 停止当前处理\n"
                "/help - 显示此帮助\n"
                "/<skill-name> - 加载技能（如果可用）"
            )

        # Skill command: /skill-name
        skill_name = text[1:].split()[0]
        if skill_name:
            try:
                from personal_agent.skills.registry import skill_registry
                content = skill_registry.load(skill_name)
                if content:
                    # Inject as skill prompt — the agent loop handles this
                    return f"[SKILL:{skill_name}]\n{content}\n\n请按照以上技能的指导处理用户消息。"
            except Exception:
                pass

        return None  # unknown command → pass to agent

    # ── auth ──────────────────────────────────────────

    def _authorize(self, event) -> bool:
        """Simple auth: MVP allows all. Extend with allowlists later."""
        return True
