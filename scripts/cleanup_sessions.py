"""Clean tool messages from session database.

Removes any message that contains tool_use or tool_result content blocks,
plus any message with non-null tool_calls/tool_name/tool_call_id columns.

Usage:
  uv run python scripts/cleanup_sessions.py [--db path/to/state.db]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiosqlite


async def cleanup(db_path: str) -> int:
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row

    # Count before
    cur = await db.execute("SELECT COUNT(*) as c FROM messages")
    before = (await cur.fetchone())["c"]

    # Pass 1: messages with tool metadata columns
    await db.execute("DELETE FROM messages WHERE tool_calls IS NOT NULL")
    await db.execute("DELETE FROM messages WHERE tool_name IS NOT NULL")
    await db.execute("DELETE FROM messages WHERE tool_call_id IS NOT NULL")

    # Pass 2: messages with tool_use/tool_result content blocks
    cur = await db.execute("SELECT id, content FROM messages")
    to_delete = []
    async for row in cur:
        try:
            blocks = json.loads(row["content"])
            if isinstance(blocks, list):
                for b in blocks:
                    if b.get("type") in ("tool_use", "tool_result"):
                        to_delete.append(row["id"])
                        break
        except (json.JSONDecodeError, TypeError):
            pass

    for mid in to_delete:
        await db.execute("DELETE FROM messages WHERE id = ?", (mid,))

    await db.commit()

    # Count after
    cur = await db.execute("SELECT COUNT(*) as c FROM messages")
    after = (await cur.fetchone())["c"]

    # Update session message counts
    cur = await db.execute(
        "SELECT session_id, COUNT(*) as c FROM messages GROUP BY session_id"
    )
    async for row in cur:
        await db.execute(
            "UPDATE sessions SET message_count = ? WHERE session_id = ?",
            (row["c"], row["session_id"]),
        )
    await db.commit()

    removed = before - after
    print(f"Cleaned: {before} → {after} ({removed} tool messages removed)")
    await db.close()
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Remove tool messages from session database"
    )
    parser.add_argument(
        "--db", default="./data/state.db", help="Path to SQLite database"
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Error: database not found: {args.db}")
        sys.exit(1)

    asyncio.run(cleanup(args.db))


if __name__ == "__main__":
    main()
