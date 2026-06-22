"""Central config via pydantic-settings. Reads from .env / env vars."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM ──
    llm_api_key: str
    llm_base_url: str = "https://api.deepseek.com/anthropic"
    llm_model: str = "deepseek-chat"
    llm_api_mode: str = "auto"  # auto | anthropic | chat_completions
    llm_provider: str = "deepseek"
    llm_max_tokens: int = 4096
    max_iterations: int = 30

    # ── Platforms ──
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    telegram_bot_token: str = ""

    # ── Storage ──
    agent_data_dir: Path = Path("./data")
    log_level: str = "INFO"

    # ── Strategy engines ──
    compressor_engine: str = "simple"
    compressor_model: str = ""        # empty = use main model; set = cheap model for compression
    compressor_max_tokens: int = 500  # max tokens for generated summary
    tail_token_budget: int = 20000    # tail protection token budget
    memory_provider: str = "file"

    # ── Cron ──
    enable_cron: bool = False
    cron_jobs_path: Path = Path("data/cron")

    # ── Session ──
    session_expire_days: int = 30  # auto-clean sessions inactive > N days
