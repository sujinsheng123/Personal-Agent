"""Read files within allowed data directory."""

from pathlib import Path

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry

# Set at startup — overwritten by main.py
_allowed_base: Path = Path("./data")

MAX_READ_BYTES = 50_000


def set_allowed_base(path: Path) -> None:
    global _allowed_base
    _allowed_base = path.resolve()


async def _file_read(path: str) -> str:
    try:
        full = (_allowed_base / path).resolve()
        if not str(full).startswith(str(_allowed_base)):
            return f"Error: path traversal denied — '{path}' is outside allowed directory"
        if not full.exists():
            return f"Error: file not found: {path}"
        if full.is_dir():
            return f"Error: '{path}' is a directory"
        content = full.read_text(encoding="utf-8", errors="replace")
        if len(content) > MAX_READ_BYTES:
            content = content[:MAX_READ_BYTES] + f"\n\n...(truncated {len(content) - MAX_READ_BYTES} bytes)"
        from personal_agent.tools.audit import audit_log
        audit_log("file_read", path, f"{len(content)} bytes read", True)
        return content
    except Exception as e:
        from personal_agent.tools.audit import audit_log
        audit_log("file_read", path, str(e), False)
        return f"Error: {e}"


tool_registry.register(ToolEntry(
    name="file_read",
    description="Read a file from the agent's data directory. Path is relative to data dir. Use for reading saved notes, code, or data files.",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to file, e.g. 'notes/ideas.txt'"},
        },
        "required": ["path"],
    },
    handler=_file_read,
    toolset="builtin",
))
