"""SkillRegistry — module-level singleton. Skills self-register on import.

Skill loading pipeline:
  1. Lookup: name → SkillEntry
  2. Path validation: resolve, prevent traversal outside SKILLS_DIR
  3. Extension check: .md only
  4. Size limit: max 50KB
  5. Read content
  6. Audit log
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from personal_agent.skills.entry import SkillEntry

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent / "builtin"
MAX_SKILL_BYTES = 50_000


class SkillRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, SkillEntry] = {}

    def register(self, entry: SkillEntry) -> None:
        self._entries[entry.name] = entry
        logger.debug("Skill registered: %s", entry.name)

    def get(self, name: str) -> SkillEntry | None:
        return self._entries.get(name)

    def list(self) -> list[SkillEntry]:
        return list(self._entries.values())

    def get_summaries(self) -> str:
        """Tier 1: name + one-line description for system prompt."""
        if not self._entries:
            return ""
        lines = ["可用技能："]
        for entry in self._entries.values():
            lines.append(f"- {entry.name}: {entry.description}")
        return "\n".join(lines)

    def load(self, name: str) -> str | None:
        """Tier 2: load skill content through security pipeline. Returns content or None."""
        entry = self._entries.get(name)
        if entry is None:
            return None

        try:
            # ── 1. Path resolution + traversal prevention ──
            path = Path(entry.path)
            if not path.is_absolute():
                path = SKILLS_DIR / path
            path = path.resolve()

            if not str(path).startswith(str(SKILLS_DIR.resolve())):
                logger.warning("Skill path traversal blocked: %s → %s", name, path)
                return None

            # ── 2. Extension check ──
            if path.suffix.lower() != ".md":
                logger.warning("Skill extension blocked: %s (%s)", name, path.suffix)
                return None

            # ── 3. Existence + size check ──
            if not path.exists():
                logger.warning("Skill file not found: %s → %s", name, path)
                return None

            file_size = path.stat().st_size
            if file_size > MAX_SKILL_BYTES:
                logger.warning("Skill too large: %s (%d bytes, max %d)", name, file_size, MAX_SKILL_BYTES)
                return None

            # ── 4. Read ──
            content = path.read_text(encoding="utf-8")
            logger.debug("Skill loaded: %s (%d bytes)", name, len(content))

            # ── 5. Audit ──
            from personal_agent.tools.audit import audit_log
            audit_log("skill_load", name, f"{len(content)} bytes", True)

            return content
        except Exception:
            logger.exception("Failed to load skill: %s", name)
            return None


skill_registry = SkillRegistry()
