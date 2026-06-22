"""CronEntry — scheduled task definition."""

from dataclasses import dataclass, field
import time


@dataclass
class CronEntry:
    job_id: str              # UUID
    name: str                # display name
    schedule: str            # "*/5 * * * *" 5-field cron
    prompt: str              # prompt to send to Agent
    session_key: str         # which session identity to use
    platform: str = "feishu"
    chat_id: str = ""        # where to deliver results
    enabled: bool = True
    last_run: float = 0
    next_run: float = 0
    max_retries: int = 3
    timeout: int = 300
    created_at: float = field(default_factory=time.time)
