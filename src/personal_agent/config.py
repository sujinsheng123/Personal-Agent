"""Central config — .env for LLM/secrets, config.yaml for behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str = "config.yaml") -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_env(path: str = ".env") -> dict[str, str]:
    from dotenv import dotenv_values
    return {k: v or "" for k, v in dotenv_values(path).items()}


class Settings:
    def __init__(self) -> None:
        yaml_cfg = _load_yaml()
        env = _load_env()

        # ── LLM (from .env) ──
        self.llm_api_key: str = env.get("LLM_API_KEY", "")
        self.llm_base_url: str = env.get("LLM_BASE_URL", "")
        self.llm_model: str = env.get("LLM_MODEL", "deepseek-chat")
        self.llm_api_mode: str = env.get("LLM_API_MODE", "auto")
        self.llm_provider: str = env.get("LLM_PROVIDER", "deepseek")
        self.llm_max_tokens: int = int(env.get("LLM_MAX_TOKENS", "4096"))

        # ── Platforms (from .env) ──
        self.feishu_app_id: str = env.get("FEISHU_APP_ID", "")
        self.feishu_app_secret: str = env.get("FEISHU_APP_SECRET", "")
        self.telegram_bot_token: str = env.get("TELEGRAM_BOT_TOKEN", "")
        self.weixin_token: str = env.get("WEIXIN_TOKEN", "")
        self.weixin_account_id: str = env.get("WEIXIN_ACCOUNT_ID", "")
        self.weixin_user_id: str = env.get("WEIXIN_USER_ID", "")
        self.weixin_base_url: str = env.get("WEIXIN_BASE_URL", "https://ilinkai.weixin.qq.com")

        # ── Agent (from config.yaml) ──
        agent = yaml_cfg.get("agent", {})
        self.max_iterations: int = agent.get("max_iterations", 30)
        self.max_tool_calls_per_turn: int = agent.get("max_tool_calls_per_turn", 20)

        # ── Storage (from config.yaml) ──
        storage = yaml_cfg.get("storage", {})
        self.agent_data_dir: Path = Path(storage.get("data_dir", "./data"))
        self.log_level: str = storage.get("log_level", "INFO")

        # ── Toolsets (from config.yaml) ──
        toolsets = yaml_cfg.get("toolsets", {})
        self.enabled_toolsets: list[str] | None = toolsets.get("enabled", ["all"])

        # ── Compression (from config.yaml) ──
        comp = yaml_cfg.get("compression", {})
        self.compressor_engine: str = comp.get("engine", "compressor")
        self.compressor_model: str = comp.get("model", "")
        self.compressor_max_tokens: int = comp.get("max_tokens", 500)
        self.tail_token_budget: int = comp.get("tail_token_budget", 20000)
        self.compression_threshold_ratio: float = comp.get("threshold_ratio", 0.6)

        # ── Memory (from config.yaml) ──
        memory = yaml_cfg.get("memory", {})
        self.memory_provider: str = memory.get("provider", "file")
        self.memory_external_provider: str = memory.get("external_provider", "none")

        # ── Cron (from config.yaml) ──
        cron = yaml_cfg.get("cron", {})
        self.enable_cron: bool = cron.get("enabled", False)
        self.cron_jobs_path: Path = Path("data/cron")

        # ── Security (from config.yaml) ──
        security = yaml_cfg.get("security", {})
        self.bash_allow_network: bool = security.get("bash_allow_network", False)
        self.file_max_write_bytes: int = security.get("file_max_write_bytes", 100000)
        self.audit_enabled: bool = security.get("audit_enabled", True)

        # ── Session (from config.yaml) ──
        session = yaml_cfg.get("session", {})
        self.session_expire_days: int = session.get("expire_days", 30)
        self.session_override: dict[str, str] = session.get("override", {})

        # ── Auth (from config.yaml) ──
        auth = yaml_cfg.get("auth", {})
        self.auth_enabled: bool = auth.get("enabled", False)
        self.auth_admins: list[str] = auth.get("admins", [])
        self.auth_allowed_users: list[str] = auth.get("allowed_users", [])
