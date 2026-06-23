"""Telegram adapter — python-telegram-bot integration.

Bridge pattern same as Feishu: PTB runs its own event loop in a background
thread → run_coroutine_threadsafe → main loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from personal_agent.adapters.base import BasePlatformAdapter, ChatInfo, SendResult
from personal_agent.models.messages import MessageEvent, SessionSource

logger = logging.getLogger(__name__)


class TelegramAdapter(BasePlatformAdapter):
    def __init__(self, config, db) -> None:
        super().__init__(config, db)
        self._application = None
        self._bot = None
        self._token = getattr(config, "telegram_bot_token", "")

    # ── connect / disconnect ──────────────────────────

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        if not self._token:
            logger.warning("Telegram bot token not configured, skipping")
            return

        from telegram.ext import Application, MessageHandler, filters

        self._application = Application.builder().token(self._token).build()
        self._application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )
        self._application.add_handler(
            MessageHandler(filters.COMMAND, self._on_command)
        )

        await self._application.initialize()
        await self._application.start()
        self._bot = self._application.bot
        await self.hooks.fire("on_connect")
        logger.info("Telegram adapter connected")

    async def disconnect(self) -> None:
        await self.hooks.fire("on_disconnect")
        if self._application:
            try:
                await self._application.stop()
                await self._application.shutdown()
            except Exception:
                logger.exception("Telegram disconnect error")
            self._application = None
            self._bot = None
        logger.info("Telegram adapter disconnected")

    # ── send ──────────────────────────────────────────

    async def send(self, chat_id: str, content: str) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="Bot not connected")
        try:
            msg = await self._bot.send_message(
                chat_id=chat_id,
                text=content,
                parse_mode="Markdown",
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as exc:
            # Retry without Markdown formatting on parse error
            if "parse" in str(exc).lower():
                try:
                    msg = await self._bot.send_message(
                        chat_id=chat_id, text=content,
                    )
                    return SendResult(success=True, message_id=str(msg.message_id))
                except Exception as exc2:
                    return SendResult(success=False, error=str(exc2))
            return SendResult(success=False, error=str(exc))

    # ── get_chat_info ─────────────────────────────────

    async def get_chat_info(self, chat_id: str) -> ChatInfo:
        if not self._bot:
            return ChatInfo(chat_id=chat_id, chat_type="dm")
        try:
            chat = await self._bot.get_chat(chat_id=int(chat_id))
            return ChatInfo(
                chat_id=str(chat.id),
                chat_type="group" if chat.type == "group" else "dm",
                chat_name=chat.title or chat.username or "",
            )
        except Exception:
            return ChatInfo(chat_id=chat_id, chat_type="dm")

    # ── PTB callbacks (run in PTB's event loop thread) ──

    async def _on_text(self, update, context) -> None:
        self._bridge_update(update, "text")

    async def _on_command(self, update, context) -> None:
        self._bridge_update(update, "command")

    def _bridge_update(self, update, message_type: str) -> None:
        """PTB callback thread → main loop via run_coroutine_threadsafe."""
        import asyncio as _asyncio
        try:
            msg = update.message
            if msg is None or not msg.text:
                return

            source = SessionSource(
                platform="telegram",
                user_id=str(msg.from_user.id) if msg.from_user else "",
                user_name=msg.from_user.username or msg.from_user.first_name or "" if msg.from_user else "",
                chat_id=str(msg.chat_id),
                chat_type="group" if msg.chat.type == "group" else "dm",
            )

            event = MessageEvent(
                text=msg.text,
                message_type=message_type,
                source=source,
                raw_message=update,
                message_id=str(msg.message_id),
                timestamp=msg.date.timestamp() if msg.date else time.time(),
            )
            _asyncio.run_coroutine_threadsafe(
                self._handle_telegram_event(event, update), self._loop
            )
        except Exception:
            logger.exception("Telegram bridge failed")

    async def _handle_telegram_event(self, event: MessageEvent, raw_update=None) -> None:
        """Called on main loop — runs parse hooks then enters the base pipeline."""
        # ── platform hook: on_before_parse ──
        if raw_update is not None:
            modified = await self.hooks.fire("on_before_parse", raw_update)
            if modified is not None:
                raw_update = modified

        # ── platform hook: on_after_parse ──
        modified = await self.hooks.fire("on_after_parse", raw_update, event)
        if modified is not None:
            event = modified

        self.handle_message(event)

    # ── typing indicator ──────────────────────────────

    async def _send_typing(self, chat_id: str) -> None:
        if self._bot:
            try:
                await self._bot.send_chat_action(chat_id=chat_id, action="typing")
            except Exception:
                pass
