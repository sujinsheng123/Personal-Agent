"""Feishu (Lark) adapter — WebSocket long connection via lark-oapi."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from personal_agent.adapters.base import BasePlatformAdapter, ChatInfo, SendResult
from personal_agent.models.messages import MessageEvent, SessionSource

logger = logging.getLogger(__name__)


class FeishuAdapter(BasePlatformAdapter):
    def __init__(self, config, db) -> None:
        super().__init__(config, db)
        self._ws_client = None
        self._app_id = config.feishu_app_id
        self._app_secret = config.feishu_app_secret

    # ── connect / disconnect ──────────────────────────

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        logger.info("Feishu adapter connecting (app_id=%s...)", self._app_id[:8])

        # Start WS client in a background thread
        import threading
        self._stop_event = threading.Event()

        def _run_ws():
            from lark_oapi.ws import Client as WsClient
            from lark_oapi.event.dispatcher_handler import EventDispatcherHandlerBuilder

            def on_message(event_data):
                # event_data is a CustomizedEvent with the message payload
                try:
                    msg = event_data.event
                    asyncio.run_coroutine_threadsafe(
                        self._handle_feishu_event(msg), self._loop
                    )
                except Exception:
                    logger.exception("Feishu WS message parse failed")

            # encrypt_key and verification_token can be empty strings for WS mode
            handler = EventDispatcherHandlerBuilder("", "") \
                .register_p1_customized_event("im.message.receive_v1", on_message) \
                .build()

            client = WsClient(
                app_id=self._app_id,
                app_secret=self._app_secret,
                event_handler=handler,
            )
            self._ws_client = client
            logger.info("Feishu WS client starting")
            client.start()

        self._ws_thread = threading.Thread(target=_run_ws, daemon=True, name="feishu-ws")
        self._ws_thread.start()
        await asyncio.sleep(1)  # Give WS time to connect
        logger.info("Feishu adapter connected")

    async def disconnect(self) -> None:
        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception:
                pass
            self._ws_client = None
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        logger.info("Feishu adapter disconnected")

    # ── send ──────────────────────────────────────────

    async def send(self, chat_id: str, content: str) -> SendResult:
        try:
            import lark_oapi as lark
            client = lark.Client.builder().app_id(self._app_id).app_secret(self._app_secret).build()

            # Parse chat_id to determine message type
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

            resp = client.im.v1.message.create(req)
            if resp.success():
                return SendResult(success=True, message_id=resp.data.message_id)
            return SendResult(success=False, error=f"Feishu API error: {resp.code} {resp.msg}")
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    # ── get_chat_info ─────────────────────────────────

    async def get_chat_info(self, chat_id: str) -> ChatInfo:
        try:
            import lark_oapi as lark
            client = lark.Client.builder().app_id(self._app_id).app_secret(self._app_secret).build()

            if chat_id.startswith("oc_"):
                req = lark.im.v1.GetChatRequest.builder().chat_id(chat_id).build()
                resp = client.im.v1.chat.get(req)
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

    async def _handle_feishu_event(self, event_data: dict) -> None:
        """Parse Feishu event → MessageEvent → pipeline."""
        try:
            event_type = event_data.get("type", "")
            if event_type == "message":
                msg_data = event_data.get("message", event_data)
                content_raw = msg_data.get("content", "{}")

                # Parse message content (JSON string in Feishu)
                try:
                    content_obj = json.loads(content_raw)
                    text = content_obj.get("text", "")
                except (json.JSONDecodeError, TypeError):
                    text = str(content_raw)

                if not text:
                    return

                source = SessionSource(
                    platform="feishu",
                    user_id=msg_data.get("open_id", ""),
                    user_name=msg_data.get("sender_name", ""),
                    chat_id=msg_data.get("chat_id", ""),
                    chat_type=msg_data.get("chat_type", "dm"),
                )

                event = MessageEvent(
                    text=text,
                    message_type="command" if text.startswith("/") else "text",
                    source=source,
                    raw_message=event_data,
                    message_id=msg_data.get("message_id"),
                    timestamp=msg_data.get("create_time", time.time()),
                )
                self.handle_message(event)
        except Exception:
            logger.exception("_handle_feishu_event failed")

    # ── typing indicator ──────────────────────────────

    async def _send_typing(self, chat_id: str) -> None:
        """Feishu doesn't support typing indicators via bot API — no-op."""
        pass
