"""CronStore — jobs.json persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from personal_agent.cron.entry import CronEntry

logger = logging.getLogger(__name__)


class CronStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load_all(self) -> list[CronEntry]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            return [CronEntry(**item) for item in data]
        except Exception:
            logger.exception("Failed to load cron jobs")
            return []

    def save_all(self, jobs: list[CronEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {k: v for k, v in j.__dict__.items()}
            for j in jobs
        ]
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def seed_defaults(self) -> list[CronEntry]:
        """Create default jobs if jobs.json doesn't exist."""
        if self._path.exists():
            return self.load_all()

        import uuid
        jobs = [
            CronEntry(
                job_id=str(uuid.uuid4()),
                name="daily-summary",
                schedule="0 21 * * *",
                prompt="请帮我总结今天发生的事情，包括对话历史中的关键信息。用中文，200字以内。",
                session_key="feishu::",
                platform="feishu",
                chat_id="",
                next_run=0,  # will be calculated on first tick
            ),
            CronEntry(
                job_id=str(uuid.uuid4()),
                name="morning-brief",
                schedule="0 8 * * *",
                prompt="早上好！请根据我的记忆和最近对话，告诉我今天可能需要注意的事项。用中文，简洁。",
                session_key="feishu::",
                platform="feishu",
                chat_id="",
                next_run=0,
            ),
        ]
        self.save_all(jobs)
        logger.info("Seeded %d default cron jobs", len(jobs))
        return jobs
