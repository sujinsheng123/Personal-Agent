"""SQLite persistence via aiosqlite. Serializes writes with an asyncio.Lock."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    session_key  TEXT NOT NULL,
    platform     TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    user_name    TEXT DEFAULT '',
    chat_id      TEXT DEFAULT '',
    chat_type    TEXT DEFAULT 'dm',
    created_at   REAL NOT NULL,
    last_active_at REAL NOT NULL,
    message_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_key ON sessions(session_key);

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    role          TEXT NOT NULL,
    content       TEXT DEFAULT '',
    tool_calls    TEXT DEFAULT NULL,
    tool_name     TEXT DEFAULT NULL,
    tool_call_id  TEXT DEFAULT NULL,
    timestamp     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._write_lock = asyncio.Lock()
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        logger.info("Database initialized at %s", self._path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ── sessions ──────────────────────────────────────

    async def create_session_direct(self, session_id: str, session_key: str) -> None:
        """Minimal session creation for tests. Skips the full SessionEntry object."""
        import time
        async with self._write_lock:
            await self._conn.execute(
                "INSERT INTO sessions (session_id, session_key, platform, user_id, created_at, last_active_at) "
                "VALUES (?, ?, 'test', '', ?, ?)",
                (session_id, session_key, time.time(), time.time()),
            )
            await self._conn.commit()

    async def create_session(self, entry) -> None:
        async with self._write_lock:
            await self._conn.execute(
                """INSERT INTO sessions (session_id, session_key, platform, user_id,
                   user_name, chat_id, chat_type, created_at, last_active_at, message_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry.session_id, entry.session_key, entry.platform, entry.user_id,
                 entry.user_name, entry.chat_id, entry.chat_type,
                 entry.created_at, entry.last_active_at, entry.message_count),
            )
            await self._conn.commit()

    async def update_last_active(self, session_id: str, increment_message: bool = True) -> None:
        import time
        async with self._write_lock:
            if increment_message:
                await self._conn.execute(
                    "UPDATE sessions SET last_active_at = ?, message_count = message_count + 1 WHERE session_id = ?",
                    (time.time(), session_id),
                )
            else:
                await self._conn.execute(
                    "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
                    (time.time(), session_id),
                )
            await self._conn.commit()

    async def delete_session(self, session_id: str) -> None:
        async with self._write_lock:
            await self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            await self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await self._conn.commit()

    # ── messages ──────────────────────────────────────

    async def save_message(
        self, session_id: str, role: str, content: str = "",
        tool_calls: list[dict] | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        import time
        tc_json = json.dumps(tool_calls) if tool_calls else None
        async with self._write_lock:
            await self._conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_calls, tool_name, tool_call_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, role, content, tc_json, tool_name, tool_call_id, time.time()),
            )
            await self._conn.commit()

    async def load_history(self, session_id: str) -> list[dict]:
        """Load conversation history as a list of Anthropic-format message dicts."""
        rows = await self._conn.execute(
            "SELECT role, content, tool_calls, tool_name, tool_call_id FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        messages: list[dict] = []
        async for row in rows:
            role = row["role"]
            if role == "user":
                if row["tool_call_id"]:
                    # This is a tool_result message
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": row["tool_call_id"], "content": row["content"]}],
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": row["content"]}],
                    })
            elif role == "assistant":
                content_blocks = []
                if row["content"]:
                    content_blocks.append({"type": "text", "text": row["content"]})
                if row["tool_calls"]:
                    for tc in json.loads(row["tool_calls"]):
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("name", ""),
                            "input": tc.get("input", {}),
                        })
                messages.append({"role": "assistant", "content": content_blocks})
        return messages

    async def get_message_count(self, session_id: str) -> int:
        row = await self._conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?", (session_id,)
        )
        async for r in row:
            return r["cnt"]
        return 0

    async def export_jsonl(self, session_id: str, output_path: str) -> int:
        """Export session as JSONL — user/assistant text only, no tool calls.

        Each line: {"role": "user|assistant", "content": "text"}
        Returns the number of messages exported.
        """
        rows = await self._conn.execute(
            "SELECT role, content, tool_calls, tool_name, tool_call_id "
            "FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            async for row in rows:
                role = row["role"]
                # Skip tool messages
                if row["tool_call_id"] or row["tool_name"]:
                    continue
                # Extract text from content blocks
                try:
                    blocks = json.loads(row["content"])
                    if isinstance(blocks, list):
                        texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
                        content = " ".join(texts)
                    else:
                        content = str(blocks)
                except (json.JSONDecodeError, TypeError):
                    content = str(row["content"])
                if not content.strip():
                    continue
                f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")
                count += 1
        return count
