"""Audit log — records all file I/O and shell executions for traceability."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_AUDIT_PATH: Path = Path("./data/audit.log")
_AUDIT_LOCK = None  # lazy init


def set_audit_path(path: Path) -> None:
    global _AUDIT_PATH
    _AUDIT_PATH = path


def _get_lock():
    global _AUDIT_LOCK
    if _AUDIT_LOCK is None:
        import threading
        _AUDIT_LOCK = threading.Lock()
    return _AUDIT_LOCK


def audit_log(tool: str, detail: str, result_snippet: str, success: bool) -> None:
    """Append one JSON line to the audit log. Non-blocking — errors are suppressed."""
    try:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tool": tool,
            "detail": detail[:500],
            "result": result_snippet[:200],
            "success": success,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _get_lock():
            with open(_AUDIT_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass  # audit failure never blocks operations
