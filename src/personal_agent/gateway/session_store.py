"""SessionStore — two-layer: JSON index + SQLite messages."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from personal_agent.models.session import SessionEntry

logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, db, data_dir: Path, chain=None) -> None:
        self._db = db
        self._index_path = data_dir / "sessions.json"
        self._index: dict[str, SessionEntry] = {}
        self._chain = chain  # CompressionChain, optional

    async def initialize(self) -> None:
        self._load_index()

    # ── chain-aware session resolution ────────────────

    def resolve_session_id(self, session_key: str) -> str | None:
        """Walk the chain to find the latest session_id for this key."""
        entry = self._index.get(session_key)
        if entry is None:
            return None
        if self._chain:
            return self._chain.resolve(entry.session_id)
        return entry.session_id

    # ── CRUD ──────────────────────────────────────────

    async def get_or_create(self, session_key: str, source) -> SessionEntry:
        if session_key in self._index:
            return self._index[session_key]

        entry = SessionEntry(
            session_id=str(uuid.uuid4()),
            session_key=session_key,
            platform=source.platform,
            user_id=source.user_id,
            user_name=getattr(source, "user_name", ""),
            chat_id=getattr(source, "chat_id", ""),
            chat_type=getattr(source, "chat_type", "dm"),
        )
        await self._db.create_session(entry)
        self._index[session_key] = entry
        self._save_index()
        logger.info("New session: %s → %s", session_key, entry.session_id)
        return entry

    async def load_history(self, session_id: str) -> list[dict]:
        return await self._db.load_history(session_id)

    async def save_transcript(self, session_id: str, messages: list[dict],
                              previous_count: int = 0) -> None:
        """Save new messages only (those after previous_count)."""
        new_msgs = messages[previous_count:]
        if not new_msgs:
            return
        from personal_agent.agent.finalize import unpack_message
        for msg in new_msgs:
            role, content, tool_calls, tool_name, tool_call_id = unpack_message(msg)
            await self._db.save_message(session_id, role, content, tool_calls, tool_name, tool_call_id)

        await self._db.update_last_active(session_id, increment_message=True)

    async def delete_session(self, session_key: str) -> str | None:
        entry = self._index.pop(session_key, None)
        if entry:
            await self._db.delete_session(entry.session_id)
            self._save_index()
            new_id = str(uuid.uuid4())
            logger.info("Session deleted: %s, new session will get ID %s", session_key, new_id)
            return new_id
        return None

    async def create_compressed_session(self, session_key: str, source,
                                         compressed_messages: list[dict]) -> str:
        """Create a new session holding compressed messages, link to old via chain."""
        old_entry = self._index.get(session_key)
        if old_entry is None:
            return ""

        new_id = str(uuid.uuid4())
        entry = SessionEntry(
            session_id=new_id,
            session_key=session_key,
            platform=old_entry.platform,
            user_id=old_entry.user_id,
            user_name=old_entry.user_name,
            chat_id=old_entry.chat_id,
            chat_type=old_entry.chat_type,
            message_count=len(compressed_messages),
        )
        await self._db.create_session(entry)

        # Persist compressed messages to new session
        from personal_agent.agent.finalize import unpack_message
        for msg in compressed_messages:
            role, content, tool_calls, tool_name, tool_call_id = unpack_message(msg)
            await self._db.save_message(new_id, role, content, tool_calls, tool_name, tool_call_id)

        # Link chain
        if self._chain:
            self._chain.link(old_entry.session_id, new_id)

        logger.info("Compressed session: %s → %s (%d messages)",
                     old_entry.session_id[:8], new_id[:8], len(compressed_messages))
        return new_id

    def get(self, session_key: str) -> SessionEntry | None:
        return self._index.get(session_key)

    def get_current_session_id(self, session_key: str) -> str | None:
        """Return the latest (uncompressed) session_id for this key."""
        entry = self._index.get(session_key)
        if entry is None:
            return None
        if self._chain:
            return self._chain.resolve(entry.session_id)
        return entry.session_id

    async def list_user_sessions(self, platform: str, user_id: str) -> list[dict]:
        """Return sessions matching platform + user_id, sorted by last active."""
        results = []
        for key, entry in self._index.items():
            if entry.platform == platform and entry.user_id == user_id:
                results.append({
                    "session_key": key,
                    "session_id": entry.session_id[:8],
                    "message_count": entry.message_count,
                    "last_active": entry.last_active_at,
                })
        results.sort(key=lambda x: x.get("last_active", ""), reverse=True)
        return results

    async def export(self, session_id: str, output_path: str) -> int:
        """Export session as JSONL — user/assistant text only."""
        return await self._db.export_jsonl(session_id, output_path)

    async def expire_sessions(self, max_age_days: int = 30) -> int:
        """Remove sessions inactive for > max_age_days. Returns count removed."""
        import time
        cutoff = time.time() - (max_age_days * 86400)
        expired: list[str] = []

        for key, entry in list(self._index.items()):
            if entry.last_active_at < cutoff:
                expired.append(key)

        for key in expired:
            entry = self._index.pop(key, None)
            if entry:
                await self._db.delete_session(entry.session_id)

        if expired:
            self._save_index()
            # Follow chain to also clean compressed descendants
            if self._chain:
                for key in expired:
                    old_id = entry.session_id if (entry := self._index.get(key)) else None  # already popped
                # Note: chain cleanup is deferred — old sessions without index entry
                # are orphaned but not automatically deleted from DB

        logger.info("Expired %d sessions (>%d days)", len(expired), max_age_days)
        return len(expired)

    # ── persistence ───────────────────────────────────

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            for key, val in data.items():
                self._index[key] = SessionEntry(**val)
            logger.info("Loaded %d sessions from index", len(self._index))
        except Exception:
            logger.exception("Failed to load sessions.json")

    def _save_index(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: {
            "session_id": v.session_id, "session_key": v.session_key,
            "platform": v.platform, "user_id": v.user_id, "user_name": v.user_name,
            "chat_id": v.chat_id, "chat_type": v.chat_type,
            "created_at": v.created_at, "last_active_at": v.last_active_at,
            "message_count": v.message_count,
        } for k, v in self._index.items()}
        self._index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
