"""Auth system — static allowlist + pairing challenge.

Flow:
  admin/superuser → always allowed
  in allowlist   → allowed
  in pending     → check if reply matches code:
      match      → move to allowlist, greet
      mismatch   → retry (max 3), then kick
  unknown user   → generate code, store pending, send challenge
"""

from __future__ import annotations

import json
import logging
import random
import string
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHALLENGE_EXPIRE = 300    # 5 minutes
MAX_ATTEMPTS = 3


class AuthManager:
    def __init__(self, config: Any, data_dir: Path) -> None:
        self._enabled: bool = getattr(config, "auth_enabled", False)
        self._admins: set[str] = set(getattr(config, "auth_admins", []) or [])
        self._allowlist: set[str] = set()
        self._pending: dict[str, dict] = {}  # user_id → {code, expires_at, attempts}

        self._allowlist_path = data_dir / "auth" / "allowlist.json"
        self._pending_path = data_dir / "auth" / "pending.json"

        if self._enabled:
            self._load()

    def _load(self) -> None:
        self._allowlist_path.parent.mkdir(parents=True, exist_ok=True)
        if self._allowlist_path.exists():
            try:
                data = json.loads(self._allowlist_path.read_text())
                self._allowlist = set(data.get("users", []))
                logger.info("Auth: loaded %d allowed users", len(self._allowlist))
            except Exception:
                logger.exception("Failed to load allowlist")

        if self._pending_path.exists():
            try:
                data = json.loads(self._pending_path.read_text())
                self._pending = data.get("pending", {})
                # Clean expired
                now = time.time()
                self._pending = {k: v for k, v in self._pending.items()
                                 if v.get("expires_at", 0) > now}
                logger.info("Auth: loaded %d pending challenges", len(self._pending))
            except Exception:
                logger.exception("Failed to load pending")

    def _save_allowlist(self) -> None:
        self._allowlist_path.parent.mkdir(parents=True, exist_ok=True)
        self._allowlist_path.write_text(json.dumps(
            {"users": sorted(self._allowlist)}, indent=2, ensure_ascii=False
        ))

    def _save_pending(self) -> None:
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending_path.write_text(json.dumps(
            {"pending": self._pending}, indent=2, ensure_ascii=False
        ))

    # ── main entry ────────────────────────────────────

    def check(self, user_id: str, message_text: str = "") -> tuple[bool, str | None]:
        """Returns (allowed, response_if_not).
        response_if_not is None if user is allowed.
        """
        if not self._enabled:
            return True, None

        # Admin always passes
        if user_id in self._admins:
            return True, None

        # Already in allowlist
        if user_id in self._allowlist:
            return True, None

        # Clean expired pending entries
        self._clean_expired()

        # User has a pending challenge — check their reply
        if user_id in self._pending:
            return self._check_challenge(user_id, message_text.strip())

        # Unknown user — issue challenge
        return self._issue_challenge(user_id)

    # ── challenge logic ───────────────────────────────

    def _issue_challenge(self, user_id: str) -> tuple[bool, str | None]:
        code = "".join(random.choices(string.digits, k=6))
        self._pending[user_id] = {
            "code": code,
            "expires_at": time.time() + CHALLENGE_EXPIRE,
            "attempts": 0,
        }
        self._save_pending()
        logger.info("Auth: issued challenge to %s (code=%s)", user_id[:12], code)
        return False, (
            f"🔐 检测到新用户，请完成验证。\n\n"
            f"请回复以下 6 位数字验证码完成配对：\n\n"
            f"    {code}\n\n"
            f"验证码 {CHALLENGE_EXPIRE // 60} 分钟内有效。"
        )

    def _check_challenge(self, user_id: str, reply: str) -> tuple[bool, str | None]:
        pending = self._pending[user_id]
        pending["attempts"] += 1

        if reply == pending["code"]:
            # Success!
            del self._pending[user_id]
            self._allowlist.add(user_id)
            self._save_allowlist()
            self._save_pending()
            logger.info("Auth: user %s verified via pairing", user_id[:12])
            return True, None  # Let the original message through to Agent

        # Wrong code
        remaining = MAX_ATTEMPTS - pending["attempts"]
        if remaining <= 0:
            del self._pending[user_id]
            self._save_pending()
            logger.warning("Auth: user %s exceeded max attempts", user_id[:12])
            return False, "🚫 验证码错误次数过多，请联系管理员。"

        self._save_pending()
        return False, f"❌ 验证码错误，你还有 {remaining} 次机会。请回复正确的 6 位数字。"

    def _clean_expired(self) -> None:
        now = time.time()
        stale = [k for k, v in self._pending.items() if v.get("expires_at", 0) < now]
        if stale:
            for k in stale:
                del self._pending[k]
            self._save_pending()

    # ── management ─────────────────────────────────────

    def add_user(self, user_id: str) -> None:
        self._allowlist.add(user_id)
        self._save_allowlist()
