"""Todo list management — persisted to SQLite for cross-session durability."""

from __future__ import annotations

import aiosqlite
import logging
import time
from pathlib import Path

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry

logger = logging.getLogger(__name__)

_db_path: Path = Path("./data/todos.db")


def set_todos_path(path: Path) -> None:
    global _db_path
    _db_path = path


async def _get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(_db_path))
    await db.execute(
        "CREATE TABLE IF NOT EXISTS todos ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  title TEXT NOT NULL,"
        "  status TEXT DEFAULT 'pending',"
        "  created_at REAL"
        ")"
    )
    await db.commit()
    return db


async def _todo(action: str, title: str = "", id: int = 0, status: str = "pending") -> str:
    db = await _get_db()
    try:
        if action == "add":
            await db.execute(
                "INSERT INTO todos (title, status, created_at) VALUES (?, 'pending', ?)",
                (title, time.time()),
            )
            await db.commit()
            cursor = await db.execute("SELECT last_insert_rowid()")
            row = await cursor.fetchone()
            new_id = row[0] if row else 0
            return f"Todo #{new_id} added: {title}"

        if action == "list":
            cursor = await db.execute(
                "SELECT id, title, status FROM todos ORDER BY id"
            )
            rows = await cursor.fetchall()
            if not rows:
                return "No todos."
            lines = []
            for r in rows:
                mark = "[x]" if r[2] == "done" else "[ ]"
                lines.append(f"#{r[0]} {mark} {r[1]}")
            return "\n".join(lines)

        if action == "update":
            updates = []
            params = []
            if title:
                updates.append("title = ?")
                params.append(title)
            if status in ("pending", "done", "cancelled"):
                updates.append("status = ?")
                params.append(status)
            if not updates:
                return "Error: nothing to update (provide title or status)"
            params.append(id)
            cursor = await db.execute(
                f"UPDATE todos SET {', '.join(updates)} WHERE id = ?", params
            )
            await db.commit()
            if cursor.rowcount == 0:
                return f"Todo #{id} not found."
            return f"Todo #{id} updated."

        if action == "delete":
            cursor = await db.execute("DELETE FROM todos WHERE id = ?", (id,))
            await db.commit()
            if cursor.rowcount == 0:
                return f"Todo #{id} not found."
            return f"Todo #{id} deleted."

        return f"Unknown action: {action}"
    finally:
        await db.close()


tool_registry.register(ToolEntry(
    name="todo",
    description="Manage a todo list (persisted cross-session). Actions: add (create), list (show all), update (modify title/status), delete (remove). Status: pending, done, cancelled.",
    schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "update", "delete"]},
            "title": {"type": "string", "description": "Todo title (for add/update)"},
            "id": {"type": "integer", "description": "Todo ID (for update/delete)"},
            "status": {"type": "string", "description": "New status: pending, done, cancelled"},
        },
        "required": ["action"],
    },
    handler=_todo,
    toolset="builtin",
))
