"""Feishu (Lark) adapter — WebSocket long connection via lark-oapi."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict

from personal_agent.adapters.base import BasePlatformAdapter, ChatInfo, SendResult
from personal_agent.models.messages import MessageEvent, SessionSource

logger = logging.getLogger(__name__)


class FeishuAdapter(BasePlatformAdapter):
    def __init__(self, config, db) -> None:
        super().__init__(config, db)
        self._ws_client = None
        self._lark_client = None  # Reused API client
        self._app_id = config.feishu_app_id
        self._app_secret = config.feishu_app_secret
        # Dedup + debounce
        self._seen_event_ids: OrderedDict[str, float] = OrderedDict()
        self._debounce_buffers: dict[str, dict] = {}  # chat_id → {timer, messages, ...}
        self._DEBOUNCE_WINDOW = 2.0  # seconds
        self._DEDUP_MAXSIZE = 1000

    # ── connect / disconnect ──────────────────────────

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        logger.info("Feishu adapter connecting (app_id=%s...)", self._app_id[:8])

        # Create reusable API client (used by send / get_chat_info)
        import lark_oapi as lark
        self._lark_client = lark.Client.builder() \
            .app_id(self._app_id) \
            .app_secret(self._app_secret) \
            .build()

        # Event to signal WS thread status
        import threading
        self._stop_event = threading.Event()
        self._ws_ready = threading.Event()
        self._ws_connected = threading.Event()  # tracks live connection status

        def _run_ws():
            # WS client module captures the event loop at import time.
            # Give it a fresh event loop for this daemon thread (avoids
            # "event loop already running" / cross-thread event loop errors).
            import lark_oapi.ws.client as ws_client_module
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            ws_client_module.loop = new_loop

            from lark_oapi.ws import Client as WsClient
            from lark_oapi.event.dispatcher_handler import EventDispatcherHandlerBuilder

            def on_message(event_data):
                logger.debug("Feishu WS raw event received: type=%s", type(event_data).__name__)
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._handle_feishu_event(event_data), self._loop
                    )
                except Exception:
                    logger.exception("Feishu WS run_coroutine_threadsafe failed")

            handler = EventDispatcherHandlerBuilder("", "") \
                .register_p2_im_message_receive_v1(on_message) \
                .build()

            try:
                client = WsClient(
                    app_id=self._app_id,
                    app_secret=self._app_secret,
                    event_handler=handler,
                    auto_reconnect=True,  # lark-oapi built-in reconnect
                )
                self._ws_client = client
                self._ws_connected.set()
                self._ws_ready.set()
                logger.info("Feishu WS client starting (v2 SDK, auto_reconnect=True)")
                client.start()
            except Exception:
                logger.exception("Feishu WS client failed to start")
                self._ws_connected.clear()
                self._ws_ready.set()

        self._ws_thread = threading.Thread(target=_run_ws, daemon=True, name="feishu-ws")
        self._ws_thread.start()

        if not self._ws_ready.wait(timeout=10):
            logger.warning("Feishu WS connection timed out after 10s")
        else:
            await self.hooks.fire("on_connect")
            logger.info("Feishu adapter connected")

        # Health check watcher: periodically verify WS is alive, reconnect if not
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def disconnect(self) -> None:
        await self.hooks.fire("on_disconnect")
        if hasattr(self, '_health_check_task'):
            self._health_check_task.cancel()
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        if self._ws_client:
            try:
                # lark-oapi WS Client doesn't have stop() — use internal _disconnect
                if hasattr(self._ws_client, '_disconnect'):
                    await self._ws_client._disconnect()
            except Exception:
                pass
            self._ws_client = None
        self._lark_client = None
        self._ws_connected.clear()
        logger.info("Feishu adapter disconnected")

    # ── health check ──────────────────────────────────

    async def _health_check_loop(self) -> None:
        """Every 30s, log WS status. If disconnected > 90s, force reconnect."""
        await asyncio.sleep(60)
        disconnected_since = 0.0
        while not (hasattr(self, '_stop_event') and self._stop_event.is_set()):
            await asyncio.sleep(30)
            if not self._ws_connected.is_set():
                if disconnected_since == 0:
                    disconnected_since = time.time()
                    logger.warning("Feishu WS appears disconnected — SDK auto_reconnect should handle it")
                elif time.time() - disconnected_since > 90:
                    logger.error("Feishu WS disconnected for 90s, forcing reconnect")
                    try:
                        if self._ws_client:
                            self._ws_client.stop()
                    except Exception:
                        pass
                    self._ws_client = None
                    self._ws_ready.clear()
                    await self.connect()
                    return
            else:
                disconnected_since = 0.0

    # ── send ──────────────────────────────────────────

    async def send(self, chat_id: str, content: str) -> SendResult:
        try:
            import lark_oapi as lark

            if chat_id.startswith("oc_"):
                req = lark.im.v1.CreateMessageRequest.builder() \
                    .receive_id_type("chat_id") \
                    .request_body(
                        lark.im.v1.CreateMessageRequestBody.builder()
                        .receive_id(chat_id)
                        .msg_type("text")
                        .content(json.dumps({"text": content}))
                        .build()
                    ).build()
            else:
                req = lark.im.v1.CreateMessageRequest.builder() \
                    .receive_id_type("open_id") \
                    .request_body(
                        lark.im.v1.CreateMessageRequestBody.builder()
                        .receive_id(chat_id)
                        .msg_type("text")
                        .content(json.dumps({"text": content}))
                        .build()
                    ).build()

            resp = self._lark_client.im.v1.message.create(req)
            if resp.success():
                return SendResult(success=True, message_id=resp.data.message_id)
            return SendResult(success=False, error=f"Feishu API error: {resp.code} {resp.msg}")
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    # ── get_chat_info ─────────────────────────────────

    async def get_chat_info(self, chat_id: str) -> ChatInfo:
        try:
            if chat_id.startswith("oc_"):
                import lark_oapi as lark
                req = lark.im.v1.GetChatRequest.builder().chat_id(chat_id).build()
                resp = self._lark_client.im.v1.chat.get(req)
                if resp.success():
                    return ChatInfo(
                        chat_id=chat_id,
                        chat_type="group" if resp.data.chat_type == "group" else "dm",
                        chat_name=resp.data.name or "",
                        member_count=resp.data.member_count or 0,
                    )
        except Exception:
            logger.exception("get_chat_info failed")
        return ChatInfo(chat_id=chat_id, chat_type="dm")

    # ── message parsing ───────────────────────────────

    async def _handle_feishu_event(self, event_data) -> None:
        """Parse Feishu v2 event (P2ImMessageReceiveV1) → MessageEvent → pipeline."""
        try:
            # ── platform hook: on_before_parse ──
            modified = await self.hooks.fire("on_before_parse", event_data)
            if modified is not None:
                event_data = modified

            inner = event_data.event
            if inner is None:
                logger.debug("Feishu event dropped: inner is None, event_data=%s", type(event_data).__name__)
                return

            msg = inner.message
            if msg is None:
                logger.debug("Feishu event dropped: msg is None, inner=%s", type(inner).__name__)
                return

            content_raw = msg.content or "{}"
            try:
                content_obj = json.loads(content_raw)
                text = content_obj.get("text", "")
            except (json.JSONDecodeError, TypeError):
                text = str(content_raw)

            if not text:
                logger.debug("Feishu event dropped: empty text, msg_type=%s chat_id=%s",
                           getattr(msg, "message_type", "?"), getattr(msg, "chat_id", "?"))
                return

            chat_type = msg.chat_type or "dm"
            # ── @ detection: only respond to @mentions in group chats ──
            if chat_type == "group":
                mentions = getattr(msg, "mentions", None) or []
                mentioned = any(
                    getattr(m, "name", "") == self._app_id for m in mentions
                )
                # Also check for @all mentions
                if not mentioned and not any(
                    getattr(m, "name", "") == "all" for m in mentions
                ):
                    # Fallback: check raw text for @
                    if "@" not in text:
                        logger.debug("Feishu event dropped: not @mentioned in group chat")
                        return

            # sender_id is a UserId object with open_id/union_id/user_id attrs
            sender = inner.sender
            user_id = ""
            if sender and sender.sender_id:
                uid = sender.sender_id
                user_id = uid.open_id or uid.union_id or uid.user_id or ""

            event_id = msg.message_id or ""

            # ── dedup ──
            if event_id and self._is_duplicate(event_id):
                logger.debug("Feishu event skipped (duplicate): %s", event_id)
                return

            # ── debounce ──
            chat_id = msg.chat_id or ""
            if chat_id:
                merged = self._debounce(chat_id, text, source_info={
                    "user_id": user_id, "user_name": "",
                    "chat_id": chat_id, "chat_type": chat_type,
                })
                if merged is not None:
                    if merged == "":
                        logger.debug("Feishu event absorbed by debounce: chat=%s", chat_id[:16])
                        return
                    text = merged  # accumulated text from burst

            logger.info("Feishu inbound: user=%s chat=%s type=%s text=%s",
                       user_id[:12] if user_id else "?",
                       chat_id[:16], chat_type,
                       text[:60])

            source = SessionSource(
                platform="feishu",
                user_id=user_id,
                user_name="",
                chat_id=msg.chat_id or "",
                chat_type=chat_type,
            )

            event = MessageEvent(
                text=text,
                message_type="command" if text.startswith("/") else "text",
                source=source,
                raw_message=event_data,
                message_id=msg.message_id,
                timestamp=float(msg.create_time or time.time()),
            )
            # ── platform hook: on_after_parse ──
            modified = await self.hooks.fire("on_after_parse", event_data, event)
            if modified is not None:
                event = modified
            self.handle_message(event)
        except Exception:
            logger.exception("_handle_feishu_event failed")

    # ── dedup + debounce ──────────────────────────────

    def _is_duplicate(self, event_id: str) -> bool:
        """Check if this event has already been processed. LRU-bounded set."""
        now = time.time()
        if event_id in self._seen_event_ids:
            return True
        self._seen_event_ids[event_id] = now
        # Evict oldest entries if too large
        while len(self._seen_event_ids) > self._DEDUP_MAXSIZE:
            self._seen_event_ids.popitem(last=False)
        # Evict entries older than 5 minutes (longer than any retry window)
        stale = [k for k, v in self._seen_event_ids.items() if now - v > 300]
        for k in stale:
            self._seen_event_ids.pop(k, None)
        return False

    def _debounce(self, chat_id: str, text: str, source_info: dict | None = None) -> str | None:
        """Merge rapid-fire messages from same chat within DEBOUNCE_WINDOW seconds.
        First message passes through immediately; subsequent ones accumulate.
        When window expires, the accumulated text is sent as one merged message.
        """
        now = time.time()
        buf = self._debounce_buffers.get(chat_id)

        if buf is None:
            # First message in potential burst — pass through, schedule flush
            self._debounce_buffers[chat_id] = {
                "texts": [text],
                "first_at": now,
                "source_info": source_info or {},
            }
            return text  # Process immediately

        elapsed = now - buf["first_at"]
        if elapsed < self._DEBOUNCE_WINDOW:
            # Still within window — accumulate, don't fire
            buf["texts"].append(text)
            return ""  # Absorbed into buffer

        # Window elapsed since first message — flush old buffer, start new one
        accumulated = "\n".join(buf.pop("texts", [text]))
        buf["texts"] = [text]
        buf["first_at"] = now
        # Fire accumulated text as a synthetic MessageEvent
        loop = asyncio.get_running_loop()
        loop.call_later(0, lambda a=accumulated, s=buf.get("source_info", {}):
            asyncio.ensure_future(self._fire_debounced(a, s)))
        return text  # Current message proceeds normally

    async def _fire_debounced(self, text: str, source_info: dict) -> None:
        """Create a MessageEvent from accumulated debounce buffer and feed to pipeline."""
        from personal_agent.models.messages import MessageEvent, SessionSource
        source = SessionSource(
            platform="feishu",
            user_id=source_info.get("user_id", ""),
            user_name=source_info.get("user_name", ""),
            chat_id=source_info.get("chat_id", ""),
            chat_type=source_info.get("chat_type", "dm"),
        )
        event = MessageEvent(
            text=text,
            message_type="text",
            source=source,
            raw_message=None,
            )
        logger.debug("Debounce flush: chat=%s delivering accumulated %d chars",
                      source_info.get("chat_id", "")[:16], len(text))
        self.handle_message(event)

    # ── typing indicator ──────────────────────────────

    async def _send_typing(self, chat_id: str) -> None:
        """Feishu doesn't support typing indicators via bot API — no-op."""
        pass
