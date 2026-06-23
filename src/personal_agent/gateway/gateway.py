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
        from personal_agent.gateway.compression_chain import CompressionChain
        from personal_agent.gateway.auth import AuthManager
        self._compression_chain = CompressionChain(config.agent_data_dir / "compression_chain.json")
        self._auth_manager = AuthManager(config, config.agent_data_dir)
        self._session_store = SessionStore(db, config.agent_data_dir, chain=self._compression_chain)
        self._adapters: list = []
        self._running_agents: dict[str, bool] = {}
        self._agent_cache: OrderedDict[str, object] = OrderedDict()
        self._session_override: dict[str, str] = {}  # platform:user → custom chat_id
        self._cron_scheduler = None
        self.hooks = Hooks()
        self._shutdown_event = asyncio.Event()
        self._mcp_manager = None  # set by main.py after MCPManager.start()

    # ── lifecycle ─────────────────────────────────────

    async def start(self) -> None:
        self._compression_chain.load()
        self._session_override.update(self.config.session_override)  # config defaults
        await self._session_store.initialize()
        await self._session_store.expire_sessions(self.config.session_expire_days)

        # Seed and start cron if enabled
        if self.config.enable_cron:
            from personal_agent.cron.store import CronStore
            from personal_agent.cron.scheduler import CronScheduler
            cron_store = CronStore(self.config.agent_data_dir / "cron" / "jobs.json")
            cron_store.seed_defaults()
            self._cron_scheduler = CronScheduler(cron_store, self)
            self._cron_scheduler.start()
        else:
            self._cron_scheduler = None

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
        if self._cron_scheduler:
            self._cron_scheduler.stop()
        mcp = getattr(self, '_mcp_manager', None)
        if mcp is not None:
            try:
                await mcp.stop()
            except Exception:
                logger.exception("Error stopping MCP manager")
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
        from personal_agent.trace import trace_id, set_trace
        token = set_trace(f"{event.source.platform}:{event.source.user_id[:8]}")
        try:
            return await self._handle_message_inner(event)
        finally:
            trace_id.reset(token)

    async def _handle_message_inner(self, event) -> str | None:
        session_key = f"{event.source.platform}:{event.source.chat_id}:{event.source.user_id}"
        # Apply session override if set (via /session command)
        override = self._session_override.get(session_key)
        if override:
            session_key = override

        # 1. Hook: on_message_received (only if hooks registered)
        if self.hooks.on_message_received:
            hook_result = await self.hooks.fire("on_message_received", event)
            if hook_result is None:
                return None  # dropped
            if hook_result is not event:
                event = hook_result

        # 2. Authorization (skip internal/cron events)
        if not event.internal and event.source.user_id != "cron":
            allowed, response = self._auth_manager.check(
                event.source.user_id, event.text
            )
            if not allowed:
                return response or "抱歉，你没有权限使用此服务。"
            # Auth passed with a message (e.g. pairing success greeting)
            if allowed and response is not None:
                return response

        # 3. Command detection
        if event.text.startswith("/"):
            cmd_result = await self._handle_command(event, session_key)
            if cmd_result is not None:
                return cmd_result
            # cmd_result is None → continue to agent (skill injection, etc.)

        # 4. Busy check
        if session_key in self._running_agents:
            return "我正在处理你上一条消息，请稍候..."

        # 5. Mark running → process → cleanup
        self._running_agents[session_key] = True
        try:
            from personal_agent.memory.file_store import set_current_session
            set_current_session(session_key)
            return await self._handle_message_with_agent(event, session_key)
        finally:
            self._running_agents.pop(session_key, None)

    # ── agent dispatch ────────────────────────────────

    async def _handle_message_with_agent(self, event, session_key: str) -> str:
        session = await self._session_store.get_or_create(session_key, event.source)

        # Walk chain to find the latest (uncompressed) session
        current_id = self._compression_chain.resolve(session.session_id)
        history = await self._session_store.load_history(current_id)
        previous_count = len(history)

        agent = self._get_or_create_agent(session_key)

        from personal_agent.agent.context import build_turn_context
        from personal_agent.agent.loop import run_conversation

        ctx = build_turn_context(agent, event.text, history)
        result = await run_conversation(agent, ctx)

        # If compression ran, create new session for compressed messages
        target_session_id = current_id
        if ctx.was_compressed and not result.get("context_overflow"):
            target_session_id = await self._session_store.create_compressed_session(
                session_key, event.source, result["messages"]
            )
        elif not result.get("context_overflow"):
            await self._session_store.save_transcript(
                target_session_id, result["messages"], previous_count
            )

        # Hook: on_before_send
        final = result.get("final_response", "")
        hook_result = await self.hooks.fire("on_before_send", final, event.source)
        if isinstance(hook_result, str):
            final = hook_result

        # Background memory review (Hermes-style nudge)
        if result.get("should_review_memory") and final:
            self._spawn_memory_review(agent, ctx.messages)

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
        from personal_agent.compression.simple import ContextCompressor

        # Resolve provider via registry
        provider_name = self.config.llm_provider
        provider = provider_registry.get(provider_name, self.config)

        # Detect api_mode and get transport
        api_mode = provider_registry.detect_api_mode(
            self.config.llm_base_url, provider_name
        )
        transport = transport_registry.get(api_mode, provider)
        logger.debug("Agent transport: provider=%s api_mode=%s", provider_name, api_mode)

        # Compressor: optionally use a separate cheap model for compression
        compressor = None
        if self.config.compressor_engine in ("simple", "compressor"):
            compressor_transport = None
            if self.config.compressor_model:
                from personal_agent.llm.provider import ProviderProfile
                comp_provider = ProviderProfile(
                    name="compressor",
                    base_url=self.config.llm_base_url,
                    api_key=self.config.llm_api_key,
                    model=self.config.compressor_model,
                    max_tokens=512,
                )
                compressor_transport = transport_registry.get(api_mode, comp_provider)

            compressor = ContextCompressor(
                context_length=64000,
                threshold_ratio=0.6,
                tail_token_budget=self.config.tail_token_budget,
                max_summary_tokens=self.config.compressor_max_tokens,
                compressor_transport=compressor_transport,
            )

        agent = init_agent(
            transport, provider,
            memory_manager=self._memory_manager,
            compressor=compressor,
            max_iterations=self.config.max_iterations,
            max_tool_calls_per_turn=self.config.max_tool_calls_per_turn,
            memory_review_interval=self.config.memory_review_interval,
            system_prompt_template=self._system_prompt_template,
            enabled_toolsets=self.config.enabled_toolsets,
        )
        # Wire delegate subsystem (so delegate_task tool can call LLM)
        from personal_agent.tools.builtin.delegate import setup_delegate
        setup_delegate(
            call_fn=transport.call,
            tools=agent.tools,
            max_tokens=provider.max_tokens,
        )

        # Wire workflow engine (so workflow_run tool can call LLM)
        from personal_agent.workflow.engine import setup_engine
        setup_engine(
            call_fn=transport.call,
            tools=agent.tools,
            max_tokens=provider.max_tokens,
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

        if text.startswith("/session"):
            parts = text.split()
            base_key = f"{event.source.platform}:{event.source.chat_id}:{event.source.user_id}"
            key_parts = base_key.split(":", 2)
            platform = key_parts[0]
            user_id = key_parts[2] if len(key_parts) > 2 else ""

            if len(parts) < 2 or parts[1] == "list":
                current = self._session_override.get(base_key, base_key)
                # Show all sessions belonging to this platform+user
                sessions = await self._session_store.list_user_sessions(platform, user_id)
                lines = [f"当前会话: {current}", "你的会话列表:"]
                for s in sessions[:10]:
                    marker = " ←" if s["session_key"] == current else ""
                    lines.append(f"  {s['session_key']}{marker} ({s.get('message_count', 0)} 条消息)")
                return "\n".join(lines)

            new_name = parts[1]
            # Compute base key from source (ignoring any existing override)
            base_key = f"{event.source.platform}:{event.source.chat_id}:{event.source.user_id}"
            key_parts = base_key.split(":", 2)
            platform = key_parts[0]
            user_id = key_parts[2] if len(key_parts) > 2 else ""
            new_key = f"{platform}:{new_name}:{user_id}"
            self._session_override[base_key] = new_key
            await self._session_store.get_or_create(new_key, event.source)
            return f"会话已切换: {new_key}"

        if text.startswith("/new"):
            await self._session_store.delete_session(session_key)
            return "会话已重置。开始新的对话吧。"

        if text.startswith("/allow"):
            # Granular: /allow write, /allow shell, /allow all
            parts = text.split()
            category = parts[1] if len(parts) > 1 else "write"
            valid = {"write", "bash", "all"}
            if category not in valid:
                return f"用法: /allow [write|bash|all]，当前有效类别: {', '.join(sorted(valid))}"
            for agent in self._agent_cache.values():
                if hasattr(agent, '_destructive_allowed'):
                    agent._destructive_allowed.add(category)
            return f"✅ 已授权 {category} 操作，本轮对话内有效。"

        if text.startswith("/stop"):
            # Set interrupt flag on all cached agents
            for agent in self._agent_cache.values():
                if hasattr(agent, '_interrupt_requested'):
                    agent._interrupt_requested = True
            # Also trigger tool-level interrupt for running bash/execute_code
            from personal_agent.tools.executor import set_interrupted
            set_interrupted()
            return "已停止。"

        if text.startswith("/usage"):
            # Show token usage + context stats for this session
            agent = self._agent_cache.get(session_key)
            if agent is None:
                return "暂无会话数据。"
            from personal_agent.llm.token_counter import context_usage
            ctx_win = agent._provider.context_window if agent._provider and agent._provider.context_window else 64_000
            cu = context_usage([], agent._cached_system_prompt or "", agent.tools,
                               context_limit=ctx_win)
            return (
                f"📊 会话用量\n"
                f"API 调用: {agent.session_api_calls} 次\n"
                f"输入 tokens: {agent.session_prompt_tokens:,} (API 报告)\n"
                f"输出 tokens: {agent.session_completion_tokens:,} (API 报告)\n"
                f"\n📐 上下文窗口 (估算)\n"
                f"已用: {cu['used']:,} / {cu['limit']:,} tokens ({cu['percent']}%)\n"
                f"  system prompt: {cu['system']:,}\n"
                f"  工具定义: {cu['tools']:,}\n"
                f"剩余: {cu['remaining']:,} tokens\n"
                f"\n🔧 本轮工具调用: {agent._tool_calls_this_turn} / {agent._max_tool_calls_per_turn}"
            )

        if text.startswith("/help"):
            return (
                "可用命令:\n"
                "/new - 重置对话\n"
                "/session <name> - 切换会话\n"
                "/stop - 停止当前处理\n"
                "/allow - 授权危险操作（如写文件）\n"
                "/help - 显示此帮助\n"
                "/<skill-name> - 加载技能（如果可用）"
            )

        # Skill command: /skill-name [message]
        skill_name = text[1:].split()[0]
        if skill_name:
            try:
                from personal_agent.skills.registry import skill_registry
                content = skill_registry.load(skill_name)
                if content:
                    # Inject skill via agent field → TurnContext → api_messages
                    # NOT into ctx.messages (not persisted)
                    agent = self._get_or_create_agent(session_key)
                    agent._pending_skill_injection = (
                        f"[技能: {skill_name}]\n\n{content}"
                    )
                    parts = text.split(None, 1)
                    event.text = parts[1] if len(parts) > 1 else "你好"
                    return None  # flow to agent with clean user message
            except Exception:
                pass
                pass

        return None  # unknown command → pass to agent

    # ── memory review ────────────────────────────────

    _MEMORY_REVIEW_PROMPT = (
        "Review this conversation and save anything worth remembering.\n\n"
        "Focus on:\n"
        "1. Has the user revealed personal details, preferences, or facts worth keeping?\n"
        "2. Has the user expressed expectations about how you should behave?\n\n"
        "If something stands out, call the memory tool to save it. "
        "Use target='user' for preferences, target='memory' for facts.\n"
        "If nothing is worth saving, just reply 'Nothing to save.' and stop."
    )

    def _spawn_memory_review(self, agent, messages: list[dict]) -> None:
        """Spawn a lightweight background review to extract memories."""
        import threading

        def _run():
            import asyncio as _asyncio
            _asyncio.run(self._do_memory_review(agent, list(messages)))

        t = threading.Thread(target=_run, daemon=True, name="mem-review")
        t.start()
        logger.debug("Memory review spawned")

    async def _do_memory_review(self, agent, messages: list[dict]) -> None:
        """Run a quick LLM call to review conversation and save memories."""
        try:
            review_messages = list(messages[-12:])  # last 12 messages only
            review_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": self._MEMORY_REVIEW_PROMPT}],
            })
            response = await agent._transport.call(
                messages=review_messages,
                system_prompt="你是一个记忆管理助手。判断对话中是否有值得保存的信息。",
                tools=agent.tools,
                max_tokens=512,
            )
            if response.tool_calls:
                from personal_agent.tools.executor import execute_tool_calls
                await execute_tool_calls(response.tool_calls, review_messages, agent=agent)
                logger.info("Memory review: %d memories saved", len(response.tool_calls))
        except Exception:
            pass  # best-effort, never block the turn

    # ── auth ──────────────────────────────────────────
    # Auth is now handled by AuthManager — see gateway/auth.py
