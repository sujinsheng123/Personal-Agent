"""FileMemoryProvider — reads/writes data/memory/MEMORY.md, auto-registers as tool."""

from __future__ import annotations

import logging
from pathlib import Path

from personal_agent.memory.base import MemoryProvider

logger = logging.getLogger(__name__)

_MEMORY_FILE = Path("./data/memory/SYSTEM.md")


def set_memory_path(path: Path) -> None:
    global _MEMORY_FILE
    _MEMORY_FILE = path


class FileMemoryProvider(MemoryProvider):
    """Memory backed by a single Markdown file. Also registers as 'memory' tool."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _MEMORY_FILE

    # ── MemoryProvider interface ─────────────────────

    async def prefetch(self, user_message: str) -> list[dict]:
        """File-based: no prefetch needed (all memories in system prompt)."""
        return []

    async def save(self, content: str) -> None:
        self._ensure_file()
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(f"\n- {content}\n")
        logger.info("Memory saved: %s", content[:80])

    async def search(self, query: str) -> list[str]:
        entries = self._read_entries()
        query_lower = query.lower()
        return [e for e in entries if query_lower in e.lower()]

    async def load_all(self) -> list[str]:
        return self._read_entries()

    def get_system_prompt_text(self) -> str:
        """Hand-curated system material injected into system prompt. Stable, small."""
        entries = self._read_entries()
        if not entries:
            return ""
        lines = ["系统提示补充："]
        for e in entries:
            lines.append(f"- {e}")
        return "\n".join(lines)

    # ── internals ────────────────────────────────────

    def _read_entries(self) -> list[str]:
        if not self._path.exists():
            return []
        entries: list[str] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("- "):
                entries.append(line[2:])
        return entries

    def _ensure_file(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("# Memory\n", encoding="utf-8")


# Register as tool
from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry

_default_store = FileMemoryProvider()


def _get_ext_store():
    try:
        from personal_agent.memory.embedding_store import get_external_instance
        return get_external_instance()
    except Exception:
        return None


async def _memory_tool(action: str, content: str = "", query: str = "") -> str:
    # External provider is the primary store (semantic, scalable).
    # Builtin MEMORY.md is hand-curated system material — never auto-written.
    ext = _get_ext_store()
    store = ext if ext else _default_store

    if action == "add":
        await store.save(content)
        return f"Memory saved: {content}"
    elif action == "search":
        results = await store.search(query)
        return "\n".join(results) if results else "No matching memories."
    elif action == "list":
        entries = await store.load_all()
        return "\n".join(entries) if entries else "No memories yet."
    return f"Unknown action: {action}"


tool_registry.register(ToolEntry(
    name="memory",
    description="Manage persistent user memories. Actions: add (save a fact), search (find by keyword), list (show all).",
    schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "search", "list"]},
            "content": {"type": "string", "description": "Memory content to save (for 'add')"},
            "query": {"type": "string", "description": "Search keyword (for 'search')"},
        },
        "required": ["action"],
    },
    handler=_memory_tool,
    toolset="builtin",
))
