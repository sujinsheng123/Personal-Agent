"""Write files within allowed data directory — destructive tool.

Safety:
  - Extension whitelist (no .exe/.bat/.sh etc.)
  - Max file size (default 100KB)
  - Path traversal prevention
  - Audit logging
"""

from pathlib import Path

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry

# Shared with file_read — set at startup
_allowed_base: Path = Path("./data")

# Only these extensions are writable (and their uppercase variants)
_ALLOWED_EXTENSIONS: set[str] = {
    ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".svg",
    ".csv", ".log", ".xml", ".rst", ".tex", ".bib",
    ".sh", ".bat", ".ps1", ".env", ".gitignore", ".dockerignore",
}
_MAX_WRITE_BYTES = 100_000


def set_allowed_base(path: Path) -> None:
    global _allowed_base
    _allowed_base = path.resolve()


def set_max_write_bytes(max_bytes: int) -> None:
    global _MAX_WRITE_BYTES
    _MAX_WRITE_BYTES = max_bytes


def _check_extension(path: str) -> str | None:
    suffix = Path(path).suffix
    if suffix and suffix.lower() not in _ALLOWED_EXTENSIONS:
        return (
            f"Error: file extension '{suffix}' is not allowed. "
            f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )
    return None


async def _file_write(path: str, content: str) -> str:
    ext_error = _check_extension(path)
    if ext_error:
        return ext_error

    if len(content) > _MAX_WRITE_BYTES:
        return f"Error: content too large ({len(content)} bytes, max {_MAX_WRITE_BYTES})"

    try:
        full = (_allowed_base / path).resolve()
        if not str(full).startswith(str(_allowed_base)):
            return f"Error: path traversal denied — '{path}' is outside allowed directory"
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        msg = f"Written {len(content)} bytes to {path}"

        from personal_agent.tools.audit import audit_log
        audit_log("file_write", path, msg, True)
        return msg
    except Exception as e:
        from personal_agent.tools.audit import audit_log
        audit_log("file_write", path, str(e), False)
        return f"Error: {e}"


tool_registry.register(ToolEntry(
    name="file_write",
    description="Write content to a file in the agent's data directory. Path is relative. "
                f"Allowed extensions: {', '.join(sorted(_ALLOWED_EXTENSIONS))}. Max {_MAX_WRITE_BYTES // 1000}KB.",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to file, e.g. 'output/report.md'"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    },
    handler=_file_write,
    toolset="builtin",
    is_parallel_safe=False,
    is_destructive=True,
))
