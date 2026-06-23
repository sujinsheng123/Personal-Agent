"""WeChat adapter — personal WeChat via Tencent iLink Bot API.

QR login → long-poll getupdates → sendmessage.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import struct
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import aiohttp

from personal_agent.adapters.base import BasePlatformAdapter, ChatInfo, SendResult
from personal_agent.models.messages import MessageEvent, SessionSource

logger = logging.getLogger(__name__)

API_BASE = "https://ilinkai.weixin.qq.com"
ILINK_APP_ID = "bot"
CHANNEL_VERSION = "2.2.0"
ILINK_APP_CLIENT_VERSION = (2 << 16) | (2 << 8) | 0

POLL_TIMEOUT_MS = 35_000
API_TIMEOUT_MS = 15_000
DEDUP_MAXSIZE = 1000
DEDUP_TTL = 300


def _random_wechat_uin() -> str:
    value = struct.unpack(">I", secrets.token_bytes(4))[0]
    return base64.b64encode(str(value).encode()).decode()


def _headers(token: str | None, body: str) -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Content-Length": str(len(body.encode("utf-8"))),
        "X-WECHAT-UIN": _random_wechat_uin(),
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


class WeChatAdapter(BasePlatformAdapter):
    supports_code_blocks = True

    def __init__(self, config, db) -> None:
        super().__init__(config, db)
        self._token: str = getattr(config, "weixin_token", "") or ""
        self._account_id: str = getattr(config, "weixin_account_id", "") or ""
        self._user_id: str = getattr(config, "weixin_user_id", "") or ""
        self._base_url: str = getattr(config, "weixin_base_url", "") or API_BASE

        self._state_dir = config.agent_data_dir / "wechat"
        self._state_dir.mkdir(parents=True, exist_ok=True)

        self._poll_session: aiohttp.ClientSession | None = None
        self._send_session: aiohttp.ClientSession | None = None
        self._poll_task: asyncio.Task | None = None
        self._sync_buf = ""
        self._seen_ids: OrderedDict[str, float] = OrderedDict()

    # ── connect / disconnect ──────────────────────────

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._load_creds()

        if not self._token or not self._account_id:
            logger.warning("WeChat not logged in. Run: uv run python -m personal_agent --wechat-login")
            return

        t = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=None, sock_read=None)
        self._poll_session = aiohttp.ClientSession(trust_env=True, timeout=t)
        self._send_session = aiohttp.ClientSession(trust_env=True, timeout=t)
        self._load_sync_buf()
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("✅ WeChat connected — account=%s..., polling started", self._account_id[:12])

    async def disconnect(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None
        if self._poll_session:
            await self._poll_session.close()
            self._poll_session = None
        if self._send_session:
            await self._send_session.close()
            self._send_session = None
        logger.info("WeChat adapter disconnected")

    # ── send ──────────────────────────────────────────

    async def send(self, chat_id: str, content: str) -> SendResult:
        if not self._send_session:
            return SendResult(success=False, error="Not connected")
        try:
            client_id = f"pa-weixin-{uuid.uuid4().hex[:12]}"
            payload = {
                "base_info": {"channel_version": CHANNEL_VERSION},
                "msg": {
                    "from_user_id": "",
                    "to_user_id": chat_id,
                    "client_id": client_id,
                    "message_type": 2,   # MSG_TYPE_BOT
                    "message_state": 2,   # MSG_STATE_FINISH
                    "item_list": [{"type": 1, "text_item": {"text": content}}],
                },
            }
            result = await self._api("ilink/bot/sendmessage", payload, self._send_session, API_TIMEOUT_MS)
            logger.debug("WeChat send result: %s", {k: result.get(k) for k in ("ret", "errcode", "errmsg") if k in result})
            if result.get("errcode") in (0, None) and result.get("ret") in (0, None):
                return SendResult(success=True, message_id=client_id)
            return SendResult(success=False,
                error=f"WeChat error ret={result.get('ret')} errcode={result.get('errcode')}")
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def get_chat_info(self, chat_id: str) -> ChatInfo:
        return ChatInfo(chat_id=chat_id, chat_type="dm")

    # ── long-poll loop ────────────────────────────────

    async def _poll_loop(self) -> None:
        timeout_ms = POLL_TIMEOUT_MS
        failures = 0
        while True:
            try:
                result = await self._api("ilink/bot/getupdates",
                    {"base_info": {"channel_version": CHANNEL_VERSION},
                     "get_updates_buf": self._sync_buf},
                    self._poll_session, timeout_ms)
                t = result.get("longpolling_timeout_ms")
                if isinstance(t, int) and t > 0:
                    timeout_ms = t
                if result.get("ret") in (0, None) and result.get("errcode") in (0, None):
                    failures = 0
                    self._sync_buf = result.get("get_updates_buf", self._sync_buf)
                    self._save_sync_buf()
                    msgs = result.get("msgs") or []
                    if msgs:
                        logger.info("WeChat poll: %d message(s) received", len(msgs))
                    for msg in msgs:
                        asyncio.create_task(self._process_message(msg))
                else:
                    failures += 1
                    logger.warning("WeChat poll error: ret=%s errcode=%s errmsg=%s",
                                   result.get("ret"), result.get("errcode"), result.get("errmsg", ""))
                    errcode = result.get("errcode")
                    if errcode == -14:
                        logger.warning("WeChat session expired, pausing 10min")
                        await asyncio.sleep(600)
                        failures = 0
                    elif failures > 3:
                        await asyncio.sleep(30)
                    else:
                        await asyncio.sleep(2)
            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                pass
            except Exception:
                failures += 1
                logger.exception("WeChat poll error (%d)", failures)
                await asyncio.sleep(min(2 ** failures, 30))

    # ── message processing ────────────────────────────

    async def _process_message(self, msg: dict) -> None:
        try:
            sender = str(msg.get("from_user_id") or "")
            if sender == self._account_id:
                return

            msg_id = str(msg.get("message_id") or "").strip()
            if msg_id and self._is_duplicate(msg_id):
                return

            text = ""
            media = []
            for item in (msg.get("item_list") or []):
                itype = item.get("type") or item.get("msg_type")
                if itype == 1 or itype == "text":  # ITEM_TEXT
                    ti = item.get("text_item") or {}
                    text += ti.get("text", "") or item.get("content", "")
                elif itype in (2, 3, 4, "image", "video", "file", "voice"):
                    media.append(f"[{itype}]")

            combined = text.strip()
            if not combined:
                combined = " ".join(media)
            if not combined:
                return

            source = SessionSource(
                platform="wechat",
                user_id=sender,
                user_name=msg.get("from_user_name", ""),
                chat_id=sender,
                chat_type="dm",
            )
            event = MessageEvent(
                text=combined,
                message_type="command" if combined.startswith("/") else "text",
                source=source,
                raw_message=msg,
                message_id=msg_id,
                timestamp=msg.get("create_time", time.time()),
            )
            logger.info("WeChat inbound: user=%s text=%s",
                       sender[:12] if sender else "?", combined[:60])
            self.handle_message(event)
        except Exception:
            logger.exception("WeChat message processing failed")

    # ── dedup ─────────────────────────────────────────

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        if msg_id in self._seen_ids:
            return True
        self._seen_ids[msg_id] = now
        while len(self._seen_ids) > DEDUP_MAXSIZE:
            self._seen_ids.popitem(last=False)
        stale = [k for k, v in self._seen_ids.items() if now - v > DEDUP_TTL]
        for k in stale:
            self._seen_ids.pop(k, None)
        return False

    # ── API helper ────────────────────────────────────

    async def _api(self, path: str, payload: dict,
                   session: aiohttp.ClientSession, timeout_ms: int) -> dict:
        body = json.dumps(payload, ensure_ascii=False)
        url = f"{self._base_url.rstrip('/')}/{path}"
        headers = _headers(self._token, body)

        async def _do():
            async with session.post(url, data=body, headers=headers) as resp:
                raw = await resp.text()
                if not resp.ok:
                    raise RuntimeError(f"iLink {path} HTTP {resp.status}: {raw[:200]}")
                return json.loads(raw)

        return await asyncio.wait_for(_do(), timeout=timeout_ms / 1000)

    # ── persistence ───────────────────────────────────

    def _load_creds(self) -> None:
        path = self._state_dir / "creds.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._token = data.get("token", self._token)
            self._account_id = data.get("account_id", self._account_id)
            self._user_id = data.get("user_id", self._user_id)
        except Exception:
            pass

    def _save_creds(self) -> None:
        (self._state_dir / "creds.json").write_text(json.dumps({
            "token": self._token, "account_id": self._account_id, "user_id": self._user_id,
        }, indent=2))

    def _load_sync_buf(self) -> None:
        path = self._state_dir / "sync.json"
        if path.exists():
            try:
                self._sync_buf = json.loads(path.read_text()).get("get_updates_buf", "")
            except Exception:
                pass

    def _save_sync_buf(self) -> None:
        (self._state_dir / "sync.json").write_text(
            json.dumps({"get_updates_buf": self._sync_buf}))


# ── QR Login (CLI) ────────────────────────────────────

async def wechat_qr_login(state_dir: Path, base_url: str = API_BASE) -> dict | None:
    """Interactive QR login. Returns creds or None."""
    state_dir.mkdir(parents=True, exist_ok=True)
    t = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=None, sock_read=None)

    async with aiohttp.ClientSession(trust_env=True, timeout=t) as session:
        # 1. Get QR (GET, not POST)
        async with session.get(
            f"{base_url}/ilink/bot/get_bot_qrcode?bot_type=3",
            headers={"iLink-App-Id": ILINK_APP_ID, "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION)},
        ) as resp:
            qr_data = json.loads(await resp.text())

        qr_value = str(qr_data.get("qrcode") or "")
        qr_url = str(qr_data.get("qrcode_img_content") or "")
        qr_scan = qr_url or qr_value
        if not qr_value:
            print("❌ 获取二维码失败。")
            return None

        print("\n请用微信扫描以下二维码登录：\n")
        try:
            import qrcode
            qr = qrcode.QRCode()
            qr.add_data(qr_scan)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except Exception as e:
            print(f"链接: {qr_url or qr_value}\n(渲染失败: {e})")

        # 2. Poll
        for _ in range(480):
            await asyncio.sleep(1)
            async with session.get(
                f"{base_url}/ilink/bot/get_qrcode_status?qrcode={qr_value}",
                headers={"iLink-App-Id": ILINK_APP_ID, "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION)},
            ) as resp:
                status = json.loads(await resp.text())

            state = str(status.get("status") or status.get("state") or "")
            if state == "confirmed":
                creds = {
                    "token": str(status.get("bot_token") or ""),
                    "account_id": str(status.get("ilink_bot_id") or ""),
                    "user_id": str(status.get("ilink_user_id") or ""),
                }
                if not creds["token"] or not creds["account_id"]:
                    print("\n❌ 登录失败：凭证不完整。")
                    return None
                (state_dir / "creds.json").write_text(json.dumps(creds, indent=2))
                print(f"\n✅ 登录成功！Account: {creds['account_id'][:12]}...")
                return creds
            elif state == "expired":
                print("\n❌ 二维码已过期，请重试。")
                return None
            elif state == "scaned":
                print("  已扫描，请在手机上确认...")
            # "wait" → continue polling

        print("\n❌ 登录超时（8分钟）。")
        return None
