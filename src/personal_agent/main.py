"""Entry point — bootstrap and run the agent system."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from personal_agent.config import Settings
from personal_agent.db.database import Database
from personal_agent.gateway.gateway import Gateway
from personal_agent.memory.file_store import FileMemoryProvider, set_system_dir
from personal_agent.memory.manager import MemoryManager
from personal_agent.tools.builtin.file_read import set_allowed_base as set_file_base
from personal_agent.tools.builtin.file_edit import set_allowed_base as set_file_edit_base
from personal_agent.tools.builtin.file_write import set_allowed_base as set_file_write_base, set_max_write_bytes
from personal_agent.tools.builtin.grep_tool import set_workspace as set_grep_workspace
from personal_agent.tools.builtin.glob_tool import set_workspace as set_glob_workspace
from personal_agent.tools.builtin.bash import set_allow_network
from personal_agent.tools.audit import set_audit_path

logger = logging.getLogger("personal_agent")


def setup_logging(level: str = "INFO") -> None:
    class ColorFormatter(logging.Formatter):
        """Colorize log level + highlight key events."""

        COLORS = {
            "DEBUG": "\033[90m",     # grey
            "INFO": "\033[37m",      # white
            "WARNING": "\033[93m",   # yellow
            "ERROR": "\033[91m",     # red
            "CRITICAL": "\033[91;1m",  # bold red
        }
        RESET = "\033[0m"
        GREEN = "\033[92m"
        CYAN = "\033[96m"

        def format(self, record):
            msg = super().format(record)
            color = self.COLORS.get(record.levelname, "")
            if color:
                msg = msg.replace(f"[{record.levelname}]", f"{color}[{record.levelname}]{self.RESET}", 1)

            # Highlight key events
            if record.levelname in ("WARNING", "ERROR"):
                return msg
            text = record.getMessage()
            if "connected" in text and ("Platform" in text or "connected" in text):
                msg = msg.replace(text, f"{self.GREEN}{text}{self.RESET}")
            elif "inbound" in text and "user=" in text:
                msg = msg.replace(text, f"{self.CYAN}{text}{self.RESET}")
            elif "Auth:" in text:
                msg = msg.replace(text, f"{self.CYAN}{text}{self.RESET}")
            elif "HTTP Request:" in text and "200" in text:
                msg = msg.replace(text, f"{self.GREEN}{text}{self.RESET}")
            elif "HTTP Request:" in text and ("4" in text or "5" in text):
                msg = msg.replace(text, f"{self.COLORS['ERROR']}{text}{self.RESET}")
            return msg

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(getattr(logging, level.upper(), logging.INFO))


def _ensure_system_files(system_dir: Path) -> None:
    """Create default system prompt files if they don't exist."""
    system_dir.mkdir(parents=True, exist_ok=True)
    defaults = {
        "SOUL.md": "# 角色与人格\n\n- 你是一个智能个人助理，名字叫小助\n- 你擅长编程、问题分析和技术支持\n- 回复风格：简洁、直接、有条理\n",
        "AGENT.md": "# 行为规则\n\n- 涉及实时数据时必须调用工具，不要凭记忆回答\n- 使用中文回复\n- 工具返回的结果要如实转述，不要编造\n- 优先使用工具而不是猜测\n",
        "USER.md": "# 用户偏好\n\n- 用户偏好从这里开始记录\n",
        "MEMORY.md": "# 用户画像\n\n- 从这里开始记录用户的重要信息\n",
    }
    for name, content in defaults.items():
        f = system_dir / name
        if not f.exists():
            f.write_text(content, encoding="utf-8")


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
    import personal_agent.tools.builtin.file_edit         # noqa
    import personal_agent.tools.builtin.grep_tool         # noqa
    import personal_agent.tools.builtin.glob_tool         # noqa
    import personal_agent.tools.builtin.todo              # noqa
    import personal_agent.tools.builtin.task              # noqa
    import personal_agent.tools.builtin.weather           # noqa
    import personal_agent.tools.builtin.bash              # noqa
    import personal_agent.tools.builtin.random_tool       # noqa
    import personal_agent.tools.builtin.timer             # noqa
    import personal_agent.tools.builtin.json_tool         # noqa
    import personal_agent.tools.builtin.skill_tools       # noqa: skill_search, skill_load
    import personal_agent.tools.builtin.clarify           # noqa
    import personal_agent.tools.builtin.process_tool      # noqa: process_list/kill/wait
    import personal_agent.tools.builtin.execute_code      # noqa
    import personal_agent.tools.builtin.delegate          # noqa: delegate_task
    import personal_agent.tools.builtin.confirm           # noqa
    import personal_agent.tools.bridge                    # noqa: bridge tools (tool_search/describe/call)
    # memory tool is auto-registered in file_store.py

    import personal_agent.llm                # noqa: trigger transport/provider registration
    import personal_agent.adapters.feishu     # noqa
    import personal_agent.adapters.telegram   # noqa
    import personal_agent.adapters.wechat     # noqa
    import personal_agent.skills.builtin      # noqa: triggers skill registration
    from personal_agent.skills.registry import discover_skills
    discover_skills(data_dir / "skills")       # auto-discover user-provided skills

    # ── 3.5. MCP servers ──────────────────────────
    mcp_manager = None
    if settings.mcp_enabled and settings.mcp_servers:
        from personal_agent.mcp.manager import MCPManager
        mcp_manager = MCPManager(settings.mcp_servers)
        await mcp_manager.start()

    # ── 4. Database ────────────────────────────────────
    db = Database(data_dir / "state.db")
    await db.initialize()

    # ── 5. Memory ──────────────────────────────────────
    system_dir = data_dir / "system"
    set_system_dir(system_dir)
    _ensure_system_files(system_dir)
    memory_store = FileMemoryProvider(system_dir)   # system prompt material

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
    set_file_edit_base(data_dir)
    set_max_write_bytes(settings.file_max_write_bytes)
    set_grep_workspace(data_dir)
    set_glob_workspace(data_dir)
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
    gateway._mcp_manager = mcp_manager  # for shutdown cleanup

    # ── 7.5. Default hooks — non-restrictive utility hooks ──

    async def _norm_message(event):
        """Normalize inbound text: strip whitespace, collapse blank lines."""
        event.text = event.text.strip()
        return event

    async def _truncate_response(text, source):
        """Truncate very long responses (>4000 chars) with a note."""
        if len(text) > 4000:
            return text[:4000] + f"\n\n…(截断 {len(text) - 4000} 字符)"
        return text

    async def _log_usage(response, usage):
        """Log per-call token usage for observability."""
        logger.info("LLM usage: in=%d out=%d total=%d",
                     usage.get("input_tokens", 0),
                     usage.get("output_tokens", 0),
                     usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
        return response

    gateway.hooks.on_message_received.append(_norm_message)
    gateway.hooks.on_before_send.append(_truncate_response)

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
    elif len(sys.argv) > 1 and sys.argv[1] == "--wechat-login":
        _run_wechat_login()
    elif len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        _run_ingest(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        try:
            asyncio.run(boot())
        except KeyboardInterrupt:
            pass


def _run_wechat_login() -> None:
    """CLI: QR login for WeChat."""
    import asyncio
    from pathlib import Path
    from personal_agent.adapters.wechat.adapter import wechat_qr_login

    settings = Settings()
    base_url = settings.weixin_base_url
    asyncio.run(wechat_qr_login(settings.agent_data_dir / "wechat", base_url))


def _run_ingest(file_path: str) -> None:
    """CLI: ingest a file into external memory."""
    import asyncio
    from pathlib import Path
    from personal_agent.memory.embedding_store import EmbeddingMemoryProvider

    async def _run():
        settings = Settings()
        path = Path(file_path)
        if not path.exists():
            print(f"Error: file not found: {file_path}")
            return
        ext = EmbeddingMemoryProvider(settings.agent_data_dir / "memory")
        try:
            count = await ext.ingest_file(str(path.resolve()))
            print(f"Ingested {path.name}: {count} chunks stored.")
        except ValueError as e:
            print(f"Error: {e}")

    asyncio.run(_run())


def _run_cli(message: str) -> None:
    """Interactive CLI mode for debugging without a platform."""
    import asyncio
    from personal_agent.agent.agent import init_agent
    from personal_agent.agent.context import build_turn_context
    from personal_agent.agent.loop import run_conversation
    from personal_agent.compression.simple import ContextCompressor
    from personal_agent.tools.builtin import calculator, datetime_tool, todo, web_search, web_fetch, file_edit  # noqa
    from personal_agent.tools.builtin import grep_tool, glob_tool  # noqa
    from personal_agent.memory.file_store import FileMemoryProvider
    from personal_agent.memory.manager import MemoryManager

    async def _run():
        settings = Settings()
        from personal_agent.llm.provider import ProviderProfile, provider_registry
        from personal_agent.llm.transport_registry import transport_registry
        provider = provider_registry.get(settings.llm_provider, settings)
        api_mode = provider_registry.detect_api_mode(settings.llm_base_url, settings.llm_provider)
        transport = transport_registry.get(api_mode, provider)
        memory = FileMemoryProvider(settings.agent_data_dir / "system")
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
