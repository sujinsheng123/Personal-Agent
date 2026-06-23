"""Targeted tests for the new tool execution pipeline: scope gate, shell whitelist,
file_write safety checks, bridge destructive blocking, checkpoint, audit."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ── Shell command whitelist ────────────────────────────


def test_shell_allowed_commands():
    from personal_agent.tools.builtin.bash import _check_command

    # Safe commands should pass
    for cmd in ["ls -la", "cat file.txt", "grep pattern file", "git status",
                "python --version", "echo hello", "whoami", "pwd", "date"]:
        assert _check_command(cmd) is None, f"'{cmd}' should be allowed"

    # Command chaining (&&, ;, |) should be caught by dangerous patterns
    # Known: whoami is whitelisted, and '&& ls' doesn't match current dangerous patterns
    # This is a real security gap — see test_shell_command_chaining_bypass
    for cmd in ["nmap -sP 192.168.1.1", "nc -lvp 4444", "evil_command"]:
        result = _check_command(cmd)
        assert result is not None, f"'{cmd}' should be blocked"


def test_shell_command_chaining_bypass():
    """Command chaining (&& || | ;) is now blocked — one command per call."""
    from personal_agent.tools.builtin.bash import _check_command

    # All chain operators are blocked
    assert _check_command("whoami && ls") is not None     # blocked
    assert _check_command("whoami && rm -rf /") is not None  # blocked
    assert _check_command("ls || echo fail") is not None  # blocked
    assert _check_command("cat file | grep x") is not None  # blocked
    assert _check_command("echo hello; ls") is not None   # blocked

    # Single commands still work
    assert _check_command("whoami") is None               # allowed
    assert _check_command("ls -la") is None               # allowed


def test_shell_network_blocked():
    from personal_agent.tools.builtin.bash import _check_command, _allow_network

    assert _allow_network is False  # default should be false

    for cmd in ["curl http://example.com", "wget http://example.com",
                "pip install requests", "npm install lodash"]:
        result = _check_command(cmd)
        assert result is not None, f"'{cmd}' should be blocked (network)"
        assert "network" in result.lower()


def test_shell_dangerous_patterns():
    from personal_agent.tools.builtin.bash import _check_command

    dangerous = [
        "rm -rf /",
        "dd if=/dev/zero of=/dev/sda",
        "chmod 777 /etc/passwd",
    ]
    for cmd in dangerous:
        result = _check_command(cmd)
        assert result is not None, f"'{cmd}' should match dangerous pattern"


# ── File write safety checks ───────────────────────────


def test_file_write_extension_whitelist():
    from personal_agent.tools.builtin.file_write import _check_extension

    # Allowed extensions
    for ext in [".txt", ".md", ".json", ".py", ".js", ".html", ".css", ".csv",
                ".yaml", ".yml", ".toml", ".log", ".xml", ".sh", ".bat"]:
        assert _check_extension(f"test{ext}") is None, f"'{ext}' should be allowed"

    # Blocked extensions
    for ext in [".exe", ".dll", ".so", ".bin", ".com", ".msi", ".scr"]:
        result = _check_extension(f"test{ext}")
        assert result is not None, f"'{ext}' should be blocked"


def test_file_write_max_size():
    from personal_agent.tools.builtin.file_write import _MAX_WRITE_BYTES

    assert _MAX_WRITE_BYTES == 100_000  # default 100KB


@pytest.mark.asyncio
async def test_file_write_large_content():
    from personal_agent.tools.builtin.file_write import _file_write

    large = "x" * 200_000
    result = await _file_write("large.txt", large)
    assert "too large" in result.lower()


@pytest.mark.asyncio
async def test_file_write_path_traversal():
    from personal_agent.tools.builtin.file_write import _file_write

    result = await _file_write("../../../etc/passwd", "hello")
    assert "path traversal" in result.lower() or "outside" in result.lower()


# ── Bridge tool_call blocking destructive tools ────────


@pytest.mark.asyncio
async def test_bridge_tool_call_blocks_destructive():
    from personal_agent.tools.bridge import _tool_call

    # file_write is destructive — should be blocked via tool_call
    result = await _tool_call("write", {"path": "test.txt", "content": "hello"})
    assert "destructive" in result.lower() or "cannot be called" in result.lower()


@pytest.mark.asyncio
async def test_bridge_tool_call_allows_safe():
    from personal_agent.tools.bridge import _tool_call

    # tool_search is safe — should work
    result = await _tool_call("tool_search", {"query": "search"})
    assert "destructive" not in result.lower()


# ── Executor scope gate ────────────────────────────────


class MockAgent:
    def __init__(self):
        self._destructive_allowed: set[str] = set()
        self._tool_calls_this_turn: int = 0
        self._max_tool_calls_per_turn: int = 20


def test_scope_gate_destructive_blocked_by_default():
    from personal_agent.tools.executor import _scope_gate
    from personal_agent.tools.entry import ToolEntry

    agent = MockAgent()
    tc = {"name": "write", "input": {"path": "x.txt", "content": "hi"}}
    entry = ToolEntry(
        name="write",
        description="Write file",
        schema={},
        handler=lambda **kw: "ok",
        toolset="builtin",
        is_destructive=True,
    )

    result = _scope_gate(tc, entry, agent)
    assert result is not None
    assert "authorization" in result.lower() or "allow" in result.lower()


def test_scope_gate_destructive_allowed_after_allow():
    from personal_agent.tools.executor import _scope_gate
    from personal_agent.tools.entry import ToolEntry

    agent = MockAgent()
    agent._destructive_allowed.add("write")

    tc = {"name": "write", "input": {"path": "x.txt", "content": "hi"}}
    entry = ToolEntry(
        name="write",
        description="Write",
        schema={},
        handler=lambda **kw: "ok",
        toolset="builtin",
        is_destructive=True,
    )

    result = _scope_gate(tc, entry, agent)
    assert result is None  # allowed now


def test_scope_gate_allow_all():
    from personal_agent.tools.executor import _scope_gate
    from personal_agent.tools.entry import ToolEntry

    agent = MockAgent()
    agent._destructive_allowed.add("all")

    tc = {"name": "write", "input": {"path": "x.txt", "content": "hi"}}
    entry = ToolEntry(
        name="write",
        description="Write",
        schema={},
        handler=lambda **kw: "ok",
        toolset="builtin",
        is_destructive=True,
    )

    result = _scope_gate(tc, entry, agent)
    assert result is None  # "all" bypasses everything


def test_scope_gate_non_destructive_passes():
    from personal_agent.tools.executor import _scope_gate
    from personal_agent.tools.entry import ToolEntry

    agent = MockAgent()
    tc = {"name": "calculator", "input": {"expression": "2+2"}}
    entry = ToolEntry(
        name="calculator",
        description="Calculate",
        schema={},
        handler=lambda **kw: "4",
        toolset="utility",
        is_destructive=False,
    )

    result = _scope_gate(tc, entry, agent)
    assert result is None  # safe tool, always passes


def test_scope_gate_tool_call_quota():
    from personal_agent.tools.executor import _scope_gate
    from personal_agent.tools.entry import ToolEntry

    agent = MockAgent()
    agent._tool_calls_this_turn = 20  # already at limit
    agent._max_tool_calls_per_turn = 20

    tc = {"name": "calculator", "input": {"expression": "1+1"}}
    entry = ToolEntry(
        name="calculator",
        description="Calc",
        schema={},
        handler=lambda **kw: "2",
        toolset="utility",
        is_destructive=False,
    )

    result = _scope_gate(tc, entry, agent)
    assert result is not None
    assert "limit" in result.lower()


# ── Executor _exec_one integration ─────────────────────


@pytest.mark.asyncio
async def test_exec_one_blocks_destructive_without_allow():
    from personal_agent.tools.executor import _exec_one

    agent = MockAgent()
    tc = {"name": "write", "input": {"path": "test.txt", "content": "hello"}}

    result = await _exec_one(tc, agent=agent)
    assert "authorization" in result.lower() or "allow" in result.lower()


@pytest.mark.asyncio
async def test_exec_one_unknown_tool():
    from personal_agent.tools.executor import _exec_one

    tc = {"name": "nonexistent_tool_xyz", "input": {}}
    result = await _exec_one(tc)
    assert "unknown" in result.lower()


# ── Audit module ───────────────────────────────────────


def test_audit_imports():
    """Verify audit module exists and has expected API."""
    from personal_agent.tools.audit import audit_log, set_audit_path
    assert callable(audit_log)
    assert callable(set_audit_path)


@pytest.mark.asyncio
async def test_audit_writes_log():
    """Verify audit_log actually writes to the configured path."""
    from personal_agent.tools.audit import audit_log, set_audit_path

    with tempfile.TemporaryDirectory() as tmpdir:
        audit_path = Path(tmpdir) / "audit.log"
        set_audit_path(audit_path)

        audit_log("test_tool", "test_target", "test result", True)
        audit_log("test_tool", "test_target", "error message", False)

        # Give async writer a moment
        await asyncio.sleep(0.2)

        assert audit_path.exists()
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert "test_tool" in lines[0]
        assert "test result" in lines[0]
        assert "test_tool" in lines[1]
        assert "error message" in lines[1]


# ── Checkpoint (file_write backup) ─────────────────────


def test_checkpoint_creates_backup(tmp_path: Path):
    from personal_agent.tools.builtin.file_write import set_allowed_base, _allowed_base
    from personal_agent.tools.executor import _checkpoint_file_write

    # Temporarily redirect the file_write sandbox to tmp_path
    orig = _allowed_base
    try:
        set_allowed_base(tmp_path)

        # Create a file to be modified
        target = tmp_path / "test.txt"
        target.write_text("original content")

        tc = {"name": "write", "input": {"path": "test.txt", "content": "new"}}
        _checkpoint_file_write(tc)

        # Verify backup exists
        checkpoints = tmp_path / "checkpoints"
        assert checkpoints.exists()
        backups = list(checkpoints.glob("test.txt.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "original content"
    finally:
        set_allowed_base(orig)


def test_checkpoint_noop_for_new_file(tmp_path: Path):
    from personal_agent.tools.builtin.file_write import set_allowed_base
    from personal_agent.tools.executor import _checkpoint_file_write

    set_allowed_base(tmp_path)
    tc = {"name": "write", "input": {"path": "new_file.txt", "content": "new"}}
    _checkpoint_file_write(tc)

    # No backup should be created for new file
    checkpoints = tmp_path / "checkpoints"
    assert not checkpoints.exists() or len(list(checkpoints.glob("*.bak"))) == 0


# ── BM25 search now returns input_schema ───────────────


def test_bm25_search_returns_schema():
    from personal_agent.tools.registry import _bm25_search

    catalog = [
        {"name": "weather", "description": "Get weather forecast", "input_schema": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]
        }},
        {"name": "calculator", "description": "Calculate math expression", "input_schema": {
            "type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]
        }},
    ]

    results = _bm25_search(catalog, "weather")
    assert len(results) > 0
    assert "input_schema" in results[0]
    assert results[0]["input_schema"] is not None
    assert "city" in str(results[0]["input_schema"])
