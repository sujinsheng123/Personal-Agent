"""Entry point — bootstrap and run the agent system."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from personal_agent.config import Settings
from personal_agent.db.database import Database
from personal_agent.gateway.gateway import Gateway
from personal_agent.memory.file_store import FileMemoryProvider, set_memory_path
from personal_agent.memory.manager import MemoryManager
from personal_agent.tools.builtin.file_read import set_allowed_base as set_file_base
from personal_agent.tools.builtin.file_write import set_allowed_base as set_file_write_base, set_max_write_bytes
from personal_agent.tools.builtin.todo import set_todos_path
from personal_agent.tools.builtin.shell import set_allow_network
from personal_agent.tools.audit import set_audit_path

logger = logging.getLogger("personal_agent")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def boot() -> None:
    # ── 1. Config ─────────────────────────────────────
    settings = Settings()
    setup_logging(settings.log_level)
    logger.info("Personal Agent starting...")

    # ── 2. Data dirs ──────────────────────────────────
    data_dir = settings.agent_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. Import triggers (self-registration) ────────
    import personal_agent.tools.builtin.calculator       # noqa
    import personal_agent.tools.builtin.web_search        # noqa
    import personal_agent.tools.builtin.web_fetch         # noqa
    import personal_agent.tools.builtin.datetime_tool     # noqa
    import personal_agent.tools.builtin.file_read         # noqa
    import personal_agent.tools.builtin.file_write        # noqa
    import personal_agent.tools.builtin.todo              # noqa
    import personal_agent.tools.builtin.weather           # noqa
    import personal_agent.tools.builtin.shell             # noqa
    import personal_agent.tools.builtin.random_tool       # noqa
    import personal_agent.tools.builtin.timer             # noqa
    import personal_agent.tools.builtin.json_tool         # noqa
    import personal_agent.tools.builtin.skill_tools       # noqa: skill_search, skill_load
    import personal_agent.tools.bridge                    # noqa: bridge tools (tool_search/describe/call)
    # memory tool is auto-registered in file_store.py

    import personal_agent.llm                # noqa: trigger transport/provider registration
    import personal_agent.adapters.feishu     # noqa
    import personal_agent.adapters.telegram   # noqa
    import personal_agent.skills.builtin      # noqa: triggers skill registration

    # ── 4. Database ────────────────────────────────────
    db = Database(data_dir / "state.db")
    await db.initialize()

    # ── 5. Memory ──────────────────────────────────────
    memory_path = data_dir / "memory" / "SYSTEM.md"
    set_memory_path(memory_path)
    memory_store = FileMemoryProvider(memory_path)   # system prompt material

    external_store = None
    if settings.memory_external_provider == "embedding":
        from personal_agent.memory.embedding_store import EmbeddingMemoryProvider, set_external_instance
        external_store = EmbeddingMemoryProvider(data_dir / "memory")
        set_external_instance(external_store)
        logger.info("External memory: embedding (BAAI/bge-small-zh-v1.5)")

    memory_manager = MemoryManager(builtin=memory_store, external=external_store)

    # ── 6. File tool sandbox ───────────────────────────
    set_file_base(data_dir)
    set_file_write_base(data_dir)
    set_max_write_bytes(settings.file_max_write_bytes)
    set_todos_path(data_dir / "todos.json")
    set_allow_network(settings.bash_allow_network)
    if settings.audit_enabled:
        set_audit_path(data_dir / "audit.log")

    # ── 7. Gateway ─────────────────────────────────────
    system_prompt = (
        "你是一个智能个人助理。你有以下能力：\n"
        "- 使用工具获取实时信息（日期、搜索、网页抓取等）\n"
        "- 执行计算、管理待办事项、读写文件\n"
        "- 管理用户记忆\n\n"
        "重要规则：\n"
        "1. 涉及实时数据（日期、天气、搜索）时，必须调用工具，不要凭记忆回答\n"
        "2. 用户要求计算时，使用 calculator 工具\n"
        "3. 用中文回复，保持简洁有条理\n"
        "4. 工具返回的结果要如实转述，不要编造"
    )
    gateway = Gateway(settings, db, memory_manager, system_prompt_template=system_prompt)

    # ── 8. Start ───────────────────────────────────────
    await gateway.start()

    # ── 9. Wait for shutdown ──────────────────────────
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(gateway)))
        except NotImplementedError:
            pass

    logger.info("Personal Agent running. Press Ctrl+C to stop.")
    # Windows: poll with sleep so KeyboardInterrupt can interrupt
    try:
        while not hasattr(gateway, '_shutdown_event') or not gateway._shutdown_event.is_set():
            await asyncio.sleep(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Interrupted, shutting down...")
    finally:
        await gateway.stop()


async def _shutdown(gateway: Gateway) -> None:
    logger.info("Shutting down...")
    await gateway.stop()


def main() -> None:
    """CLI entry: python -m personal_agent"""
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        _run_cli(sys.argv[2] if len(sys.argv) > 2 else "Hello")
    else:
        try:
            asyncio.run(boot())
        except KeyboardInterrupt:
            pass


def _run_cli(message: str) -> None:
    """Interactive CLI mode for debugging without a platform."""
    import asyncio
    from personal_agent.agent.agent import init_agent
    from personal_agent.agent.context import build_turn_context
    from personal_agent.agent.loop import run_conversation
    from personal_agent.compression.simple import ContextCompressor
    from personal_agent.tools.builtin import calculator, datetime_tool, todo, web_search, web_fetch  # noqa
    from personal_agent.memory.file_store import FileMemoryProvider
    from personal_agent.memory.manager import MemoryManager

    async def _run():
        settings = Settings()
        from personal_agent.llm.provider import ProviderProfile, provider_registry
        from personal_agent.llm.transport_registry import transport_registry
        provider = provider_registry.get(settings.llm_provider, settings)
        api_mode = provider_registry.detect_api_mode(settings.llm_base_url, settings.llm_provider)
        transport = transport_registry.get(api_mode, provider)
        memory = FileMemoryProvider(settings.agent_data_dir / "memory" / "MEMORY.md")
        memory_manager = MemoryManager(builtin=memory)
        agent = init_agent(transport, provider, memory_manager=memory_manager,
                          compressor=ContextCompressor(), max_iterations=settings.max_iterations,
                          max_tool_calls_per_turn=settings.max_tool_calls_per_turn,
                          enabled_toolsets=settings.enabled_toolsets,
                          system_prompt_template='你是一个智能助手。优先使用工具获取实时信息和执行操作，不要凭记忆编造。用中文回复。')

        ctx = build_turn_context(agent, message)
        result = await run_conversation(agent, ctx)
        # Use sys.stdout with encoding fix for Windows console
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(result["final_response"])

    asyncio.run(_run())


if __name__ == "__main__":
    main()
