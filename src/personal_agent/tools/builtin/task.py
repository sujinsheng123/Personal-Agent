"""Task tool — persistent cross-session task management.

Unlike todo (session-scoped, lost on restart), tasks are durable.
Use task for things that span multiple sessions: reminders, projects,
long-running work, or anything you want to remember across restarts.

Compare with:
  - todo: session-only progress tracking (in-memory)
  - cron: scheduled recurring jobs (data/cron/)
  - task: persistent one-time tasks (SQLite) ← this
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import aiosqlite

from personal_agent.tools.entry import ToolEntry
from personal_agent.tools.registry import tool_registry

logger = logging.getLogger(__name__)

_db_path: Path = Path("./data/tasks.db")


def set_tasks_path(path: Path) -> None:
    global _db_path
    _db_path = path


async def _get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(_db_path))
    await db.execute(
        "CREATE TABLE IF NOT EXISTS tasks ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  title TEXT NOT NULL,"
        "  description TEXT DEFAULT '',"
        "  status TEXT DEFAULT 'pending',"
        "  priority TEXT DEFAULT 'medium',"
        "  due_date TEXT DEFAULT '',"
        "  created_at REAL,"
        "  updated_at REAL"
        ")"
    )
    await db.commit()
    return db


STATUS_EMOJI = {
    "pending": "⬜",
    "in_progress": "🔄",
    "completed": "✅",
    "cancelled": "❌",
}
PRIORITY_SORT = {"high": 0, "medium": 1, "low": 2}


async def _task_list(status: str = "", priority: str = "", search: str = "") -> str:
    """List tasks with optional filtering."""
    db = await _get_db()
    try:
        where = []
        params = []
        if status and status != "all":
            where.append("status = ?")
            params.append(status)
        if priority and priority != "all":
            where.append("priority = ?")
            params.append(priority)

        sql = "SELECT id, title, status, priority, due_date FROM tasks"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY status = 'completed', status = 'cancelled', id DESC"

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()

        if not rows:
            return _filter_summary(status, priority, 0)

        lines = []
        for r in rows:
            id_, title, st, pri, due = r
            emoji = STATUS_EMOJI.get(st, "")
            flags = []
            if pri != "medium":
                flags.append(pri)
            if due:
                flags.append(f"due:{due}")
            suffix = f" ({', '.join(flags)})" if flags else ""
            if st == "completed":
                lines.append(f"  #{id_} {emoji} ~~{title}~~{suffix}")
            elif st == "cancelled":
                lines.append(f"  #{id_} {emoji} ~~{title}~~{suffix}")
            else:
                lines.append(f"  #{id_} {emoji} {title}{suffix}")

        return _filter_summary(status, priority, len(rows)) + "\n" + "\n".join(lines)
    finally:
        await db.close()


async def _task_add(
    title: str,
    description: str = "",
    priority: str = "medium",
    due_date: str = "",
) -> str:
    """Add a new task."""
    if priority not in ("low", "medium", "high"):
        priority = "medium"

    db = await _get_db()
    try:
        now = time.time()
        cursor = await db.execute(
            "INSERT INTO tasks (title, description, status, priority, due_date, created_at, updated_at) "
            "VALUES (?, ?, 'pending', ?, ?, ?, ?)",
            (title.strip(), description.strip(), priority, due_date.strip(), now, now),
        )
        await db.commit()
        new_id = cursor.lastrowid
        return f"Task #{new_id} added: {title}" + (f" (priority: {priority})" if priority != "medium" else "")
    finally:
        await db.close()


async def _task_update(
    id: int,
    title: str = "",
    description: str = "",
    status: str = "",
    priority: str = "",
    due_date: str = "",
) -> str:
    """Update a task's fields."""
    db = await _get_db()
    try:
        # Check exists
        cursor = await db.execute("SELECT id FROM tasks WHERE id = ?", (id,))
        if not await cursor.fetchone():
            return f"Task #{id} not found."

        updates = []
        params = []
        if title:
            updates.append("title = ?")
            params.append(title.strip())
        if description:
            updates.append("description = ?")
            params.append(description.strip())
        if status and status in ("pending", "in_progress", "completed", "cancelled"):
            updates.append("status = ?")
            params.append(status)
        if priority and priority in ("low", "medium", "high"):
            updates.append("priority = ?")
            params.append(priority)
        if due_date:
            updates.append("due_date = ?")
            params.append(due_date.strip())

        if not updates:
            return "Error: nothing to update"

        updates.append("updated_at = ?")
        params.append(time.time())
        params.append(id)

        await db.execute(
            f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?", params
        )
        await db.commit()
        return f"Task #{id} updated."
    finally:
        await db.close()


async def _task_delete(id: int) -> str:
    """Delete a task."""
    db = await _get_db()
    try:
        cursor = await db.execute("DELETE FROM tasks WHERE id = ?", (id,))
        await db.commit()
        if cursor.rowcount == 0:
            return f"Task #{id} not found."
        return f"Task #{id} deleted."
    finally:
        await db.close()


async def _task_get(id: int) -> str:
    """Get full details of a single task."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT id, title, description, status, priority, due_date, created_at, updated_at "
            "FROM tasks WHERE id = ?",
            (id,),
        )
        row = await cursor.fetchone()
        if not row:
            return f"Task #{id} not found."

        id_, title, desc, st, pri, due, created, updated = row
        emoji = STATUS_EMOJI.get(st, "")
        lines = [
            f"#{id_} {emoji} {title}",
            f"  Status: {st}",
            f"  Priority: {pri}",
        ]
        if desc:
            lines.append(f"  Description: {desc}")
        if due:
            lines.append(f"  Due: {due}")
        lines.append(f"  Created: {time.strftime('%Y-%m-%d %H:%M', time.localtime(created))}")
        lines.append(f"  Updated: {time.strftime('%Y-%m-%d %H:%M', time.localtime(updated))}")
        return "\n".join(lines)
    finally:
        await db.close()


def _filter_summary(status: str, priority: str, count: int) -> str:
    parts = [f"{count} task(s)"]
    if status and status != "all":
        parts.append(f"status={status}")
    if priority and priority != "all":
        parts.append(f"priority={priority}")
    return ", ".join(parts) + (":" if count > 0 else ". No tasks match.")


# ── dispatch ───────────────────────────────────────────


async def _task(
    action: str,
    title: str = "",
    description: str = "",
    status: str = "",
    priority: str = "",
    due_date: str = "",
    id: int = 0,
    search: str = "",
) -> str:
    """Persistent task manager. Actions: list, add, update, delete, get."""
    if action == "list":
        return await _task_list(status=status, priority=priority, search=search)
    elif action == "add":
        if not title.strip():
            return "Error: title is required for add"
        return await _task_add(title=title, description=description, priority=priority, due_date=due_date)
    elif action == "update":
        if not id:
            return "Error: id is required for update"
        return await _task_update(id=id, title=title, description=description, status=status, priority=priority, due_date=due_date)
    elif action == "delete":
        if not id:
            return "Error: id is required for delete"
        return await _task_delete(id=id)
    elif action == "get":
        if not id:
            return "Error: id is required for get"
        return await _task_get(id=id)
    else:
        return f"Unknown action: {action}. Use list, add, update, delete, or get."


tool_registry.register(ToolEntry(
    name="task",
    description=(
        "Manage persistent tasks (cross-session, SQLite-backed). "
        "Use for: reminders, projects, long-running work, things to remember across restarts. "
        "NOT for: session-only progress (use todo), scheduled jobs (use cron). "
        "Actions: list (with status/priority filters), add, update, delete, get (full details). "
        "Priority: low, medium (default), high. "
        "Status: pending, in_progress, completed, cancelled."
    ),
    schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "add", "update", "delete", "get"],
                "description": "What to do",
            },
            "title": {"type": "string", "description": "Task title (for add/update)"},
            "description": {"type": "string", "description": "Optional details (for add/update)"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "cancelled"],
                "description": "Task status (for update, or filter for list)",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Priority (for add/update, or filter for list)",
            },
            "due_date": {"type": "string", "description": "Due date in YYYY-MM-DD format (for add/update)"},
            "id": {"type": "integer", "description": "Task ID (for update/delete/get)"},
            "search": {"type": "string", "description": "Search keyword (for list)"},
        },
        "required": ["action"],
    },
    handler=_task,
    toolset="builtin",
))
