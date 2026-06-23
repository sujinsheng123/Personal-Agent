"""WeChat adapter — self-registers on import."""

from personal_agent.adapters.base import PlatformEntry, platform_registry
from personal_agent.adapters.wechat.adapter import WeChatAdapter


def _factory(config, db):
    return WeChatAdapter(config, db)


def _check(config):
    """Only enable if creds exist or token is in env (QR login was done)."""
    if config.weixin_token and config.weixin_account_id:
        return True
    creds_path = config.agent_data_dir / "wechat" / "creds.json"
    return creds_path.exists()


platform_registry.register(PlatformEntry(
    name="wechat",
    factory=_factory,
    check_fn=_check,
))
