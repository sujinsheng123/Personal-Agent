"""Comprehensive tests for skill loading pipeline.

Covers: registry, security pipeline (path traversal, extension, size, existence),
skill_search/skill_load tools, Gateway /command integration, end-to-end flow,
and audit logging.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Helpers ────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _ensure_builtin_skills_registered():
    """Make sure builtin skills are registered (idempotent)."""
    import personal_agent.skills.builtin  # noqa: F401


def _make_skill_content(size: int) -> str:
    return "# Test\n\n" + "x" * max(0, size - 8)


# ── Registry tests ─────────────────────────────────────


def test_register_and_get():
    from personal_agent.skills.registry import skill_registry

    assert skill_registry.get("python-expert") is not None
    assert skill_registry.get("git-workflow") is not None
    assert skill_registry.get("shell-guide") is not None
    assert skill_registry.get("nonexistent") is None


def test_list_entries():
    from personal_agent.skills.registry import skill_registry

    entries = skill_registry.list()
    names = {e.name for e in entries}
    assert "python-expert" in names
    assert "git-workflow" in names
    assert "shell-guide" in names


def test_get_summaries():
    from personal_agent.skills.registry import skill_registry

    summary = skill_registry.get_summaries()
    assert "python-expert" in summary
    assert "git-workflow" in summary
    assert "可用技能" in summary


def test_duplicate_registration_overwrites():
    from personal_agent.skills.registry import skill_registry
    from personal_agent.skills.entry import SkillEntry

    original = skill_registry.get("python-expert")
    original_desc = original.description
    original_triggers = list(original.triggers)

    try:
        entry = SkillEntry(name="python-expert", description="Updated", path="python_expert.md")
        skill_registry.register(entry)
        assert skill_registry.get("python-expert").description == "Updated"
    finally:
        # Restore original so downstream tests aren't polluted
        skill_registry.register(SkillEntry(
            name="python-expert",
            description=original_desc,
            path="python_expert.md",
            triggers=original_triggers,
        ))


# ── Security pipeline tests ────────────────────────────


def test_load_valid_skill():
    from personal_agent.skills.registry import skill_registry

    content = skill_registry.load("python-expert")
    assert content is not None
    assert len(content) > 0
    assert "Python" in content or "python" in content


def test_load_nonexistent_skill():
    from personal_agent.skills.registry import skill_registry

    assert skill_registry.load("no-such-skill") is None


def test_load_path_traversal_blocked():
    from personal_agent.skills.registry import skill_registry, SKILLS_DIR
    from personal_agent.skills.entry import SkillEntry

    entry = SkillEntry(
        name="escape-attempt",
        description="Tries to escape",
        path="../../../etc/passwd",
    )
    skill_registry.register(entry)

    content = skill_registry.load("escape-attempt")
    assert content is None  # blocked by traversal check


def test_load_absolute_path_outside_skills_dir():
    from personal_agent.skills.registry import skill_registry
    from personal_agent.skills.entry import SkillEntry

    entry = SkillEntry(
        name="abs-escape",
        description="Absolute path",
        path="C:/Windows/System32/drivers/etc/hosts",
    )
    skill_registry.register(entry)

    content = skill_registry.load("abs-escape")
    assert content is None  # blocked


def test_load_non_md_extension_blocked():
    from personal_agent.skills.registry import skill_registry, SKILLS_DIR
    from personal_agent.skills.entry import SkillEntry

    # Create a .txt file inside SKILLS_DIR to bypass traversal check
    txt_path = SKILLS_DIR / "test_skill.txt"
    txt_path.write_text("hello", encoding="utf-8")

    try:
        entry = SkillEntry(
            name="txt-skill",
            description="Text file",
            path="test_skill.txt",
        )
        skill_registry.register(entry)

        content = skill_registry.load("txt-skill")
        assert content is None  # blocked by extension check
    finally:
        txt_path.unlink(missing_ok=True)


def test_load_nonexistent_file():
    from personal_agent.skills.registry import skill_registry, SKILLS_DIR
    from personal_agent.skills.entry import SkillEntry

    entry = SkillEntry(
        name="ghost-skill",
        description="Doesn't exist",
        path="ghost.md",
    )
    skill_registry.register(entry)

    content = skill_registry.load("ghost-skill")
    assert content is None  # file not found


def test_load_too_large_file():
    from personal_agent.skills.registry import skill_registry, SKILLS_DIR, MAX_SKILL_BYTES
    from personal_agent.skills.entry import SkillEntry

    # Create a large .md file inside SKILLS_DIR
    large_path = SKILLS_DIR / "huge.md"
    large_path.write_text(_make_skill_content(MAX_SKILL_BYTES + 1), encoding="utf-8")

    try:
        entry = SkillEntry(
            name="huge-skill",
            description="Too big",
            path="huge.md",
        )
        skill_registry.register(entry)

        content = skill_registry.load("huge-skill")
        assert content is None  # blocked by size check
    finally:
        large_path.unlink(missing_ok=True)


def test_load_exactly_max_size():
    from personal_agent.skills.registry import skill_registry, SKILLS_DIR, MAX_SKILL_BYTES
    from personal_agent.skills.entry import SkillEntry

    # Use write_bytes to avoid Windows \n → \r\n expansion
    content_bytes = b"# Test\n\n" + b"x" * (MAX_SKILL_BYTES - 8)
    path = SKILLS_DIR / "max_size.md"
    path.write_bytes(content_bytes)

    try:
        entry = SkillEntry(
            name="max-size-skill",
            description="Exactly max",
            path="max_size.md",
        )
        skill_registry.register(entry)

        content = skill_registry.load("max-size-skill")
        assert content is not None, f"Expected pass but was blocked (file: {len(content_bytes)} bytes)"
        assert len(content_bytes) == MAX_SKILL_BYTES
    finally:
        path.unlink(missing_ok=True)


# ── Audit logging ──────────────────────────────────────


def test_load_writes_audit(tmp_path: Path):
    from personal_agent.tools.audit import set_audit_path
    from personal_agent.skills.registry import skill_registry

    audit_path = tmp_path / "audit.log"
    set_audit_path(audit_path)

    content = skill_registry.load("python-expert")
    assert content is not None

    # Audit writes happen async — give it a moment
    import time
    time.sleep(0.3)

    assert audit_path.exists()
    log_content = audit_path.read_text(encoding="utf-8")
    assert "skill_load" in log_content
    assert "python-expert" in log_content


# ── skill_search tool ──────────────────────────────────


@pytest.mark.asyncio
async def test_skill_search_exact_name():
    from personal_agent.tools.builtin.skill_tools import _skill_search

    result = await _skill_search("python-expert")
    assert "python-expert" in result
    assert "Python coding" in result


@pytest.mark.asyncio
async def test_skill_search_partial():
    from personal_agent.tools.builtin.skill_tools import _skill_search

    result = await _skill_search("python")
    assert "python-expert" in result


@pytest.mark.asyncio
async def test_skill_search_no_match():
    from personal_agent.tools.builtin.skill_tools import _skill_search

    result = await _skill_search("zzz-nonexistent-query")
    assert "No matching" in result or "Available:" in result


@pytest.mark.asyncio
async def test_skill_search_prompts_load():
    from personal_agent.tools.builtin.skill_tools import _skill_search

    result = await _skill_search("git")
    assert "skill_load" in result


# ── skill_load tool ────────────────────────────────────


@pytest.mark.asyncio
async def test_skill_load_existing():
    from personal_agent.tools.builtin.skill_tools import _skill_load

    content = await _skill_load("python-expert")
    assert len(content) > 0


@pytest.mark.asyncio
async def test_skill_load_nonexistent():
    from personal_agent.tools.builtin.skill_tools import _skill_load

    result = await _skill_load("no-such-skill")
    assert "not found" in result.lower()


# ── End-to-end: LLM flow simulation ────────────────────


@pytest.mark.asyncio
async def test_e2e_llm_skill_flow():
    """Simulate: LLM calls skill_search → gets results → calls skill_load → gets content."""
    from personal_agent.tools.builtin.skill_tools import _skill_search, _skill_load

    # Step 1: LLM asks "I need help with git"
    search_result = await _skill_search("git")
    assert "git-workflow" in search_result
    assert "skill_load" in search_result

    # Step 2: LLM loads the matched skill
    content = await _skill_load("git-workflow")
    assert len(content) > 0
    assert "git" in content.lower()


@pytest.mark.asyncio
async def test_e2e_user_slash_command():
    """Simulate: User types /python, Gateway resolves skill."""
    from personal_agent.skills.registry import skill_registry

    # Gateway does: skill_registry.load(skill_name)
    content = skill_registry.load("python-expert")
    assert content is not None

    # Gateway injects into api_messages as system instruction
    injected = f"[SKILL:python-expert]\n{content}\n\n请按照以上技能的指导处理用户消息。"
    assert "Python" in injected
    assert "SKILL" in injected


# ── Edge cases ─────────────────────────────────────────


def test_load_empty_file():
    from personal_agent.skills.registry import skill_registry, SKILLS_DIR
    from personal_agent.skills.entry import SkillEntry

    empty_path = SKILLS_DIR / "empty.md"
    empty_path.write_text("", encoding="utf-8")

    try:
        entry = SkillEntry(name="empty-skill", description="Empty", path="empty.md")
        skill_registry.register(entry)

        content = skill_registry.load("empty-skill")
        assert content == ""  # empty is valid — just no content
    finally:
        empty_path.unlink(missing_ok=True)


def test_trigger_metadata():
    from personal_agent.skills.registry import skill_registry

    python = skill_registry.get("python-expert")
    assert "/python" in python.triggers
    assert "/py" in python.triggers

    git = skill_registry.get("git-workflow")
    assert "/git" in git.triggers

    shell = skill_registry.get("shell-guide")
    assert "/shell" in shell.triggers
    assert "/bash" in shell.triggers


def test_skill_tools_registered():
    from personal_agent.tools.registry import tool_registry

    search = tool_registry.get("skill_search")
    assert search is not None
    assert not search.is_destructive
    assert search.is_parallel_safe

    load = tool_registry.get("skill_load")
    assert load is not None
    assert not load.is_destructive
    assert load.is_parallel_safe


def test_builtin_skills_readable():
    """Verify all 3 builtin .md files are readable and non-empty."""
    from personal_agent.skills.registry import skill_registry

    for name in ["python-expert", "git-workflow", "shell-guide"]:
        content = skill_registry.load(name)
        assert content is not None, f"{name} should be loadable"
        assert len(content) > 0, f"{name} should be non-empty"
        # Verify basic structure
        assert "#" in content or len(content) > 50, f"{name} should have content"
