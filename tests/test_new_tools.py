"""Tests for new tools: clarify, process, execute_code, delegate."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


# ── clarify ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clarify_question_only():
    from personal_agent.tools.builtin.clarify import _clarify
    import json

    q = json.dumps([{
        "header": "Continue",
        "question": "Do you want to continue?",
        "options": [],
    }])
    result = await _clarify(q)
    assert "Do you want to continue?" in result
    assert "Other" in result


@pytest.mark.asyncio
async def test_clarify_with_choices():
    from personal_agent.tools.builtin.clarify import _clarify
    import json

    q = json.dumps([{
        "header": "Language",
        "question": "What language?",
        "options": [
            {"label": "Python", "description": "Great for AI"},
            {"label": "Rust", "description": "Fast and safe"},
        ],
    }])
    result = await _clarify(q)
    assert "1. **Python**" in result
    assert "2. **Rust**" in result


@pytest.mark.asyncio
async def test_clarify_multi_question():
    from personal_agent.tools.builtin.clarify import _clarify
    import json

    q = json.dumps([
        {"header": "A", "question": "First?", "options": [{"label": "X", "description": ""}]},
        {"header": "B", "question": "Second?", "options": [{"label": "Y", "description": ""}]},
    ])
    result = await _clarify(q)
    assert "## A" in result
    assert "## B" in result
    assert "---" in result


@pytest.mark.asyncio
async def test_clarify_multi_select():
    from personal_agent.tools.builtin.clarify import _clarify
    import json

    q = json.dumps([{
        "header": "Features",
        "question": "Which features?",
        "options": [{"label": "A", "description": ""}, {"label": "B", "description": ""}],
        "multiSelect": True,
    }])
    result = await _clarify(q)
    assert "multiple options" in result.lower()


@pytest.mark.asyncio
async def test_clarify_invalid_json():
    from personal_agent.tools.builtin.clarify import _clarify

    result = await _clarify("not json")
    assert "Error" in result
    assert "invalid" in result.lower()


# ── process ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_list_empty():
    from personal_agent.tools.builtin.process_tool import _process_list

    result = await _process_list()
    assert "No background processes" in result or "running" not in result.lower()


@pytest.mark.asyncio
async def test_process_kill_nonexistent():
    from personal_agent.tools.builtin.process_tool import _process_kill

    result = await _process_kill(99999)
    assert "no process" in result.lower()


@pytest.mark.asyncio
async def test_process_wait_nonexistent():
    from personal_agent.tools.builtin.process_tool import _process_wait

    result = await _process_wait(99999)
    assert "no process" in result.lower()


@pytest.mark.asyncio
async def test_process_lifecycle():
    """Spawn a real background process, list it, wait for it, verify."""
    from personal_agent.tools.builtin.process_tool import (
        _process_list, _process_wait, _process_kill, _register,
    )

    # Spawn a real background process
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(0.3); print('done')",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pid = _register(proc, "echo test")

    # List — should appear
    result = await _process_list()
    assert str(pid) in result
    assert "echo test" in result

    # Wait for it
    result = await _process_wait(pid, timeout=5)
    assert "rc=0" in result or "finished" in result.lower()

    # Kill after completion — should say already finished
    result = await _process_kill(pid)
    assert "already finished" in result.lower()


# ── execute_code ────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_code_basic():
    from personal_agent.tools.builtin.execute_code import _execute_code

    result = await _execute_code("print('hello world')")
    assert "hello world" in result


@pytest.mark.asyncio
async def test_execute_code_math():
    from personal_agent.tools.builtin.execute_code import _execute_code

    result = await _execute_code("print(2 ** 10)")
    assert "1024" in result


@pytest.mark.asyncio
async def test_execute_code_stderr():
    from personal_agent.tools.builtin.execute_code import _execute_code

    result = await _execute_code("import sys; print('ok', file=sys.stderr)")
    assert "[stderr]" in result
    assert "ok" in result


@pytest.mark.asyncio
async def test_execute_code_exception():
    from personal_agent.tools.builtin.execute_code import _execute_code

    result = await _execute_code("raise RuntimeError('boom')")
    assert "RuntimeError" in result
    assert "boom" in result


@pytest.mark.asyncio
async def test_execute_code_imports():
    from personal_agent.tools.builtin.execute_code import _execute_code

    result = await _execute_code(
        "import json, math, datetime, collections; "
        "print(json.dumps({'sqrt': math.sqrt(16), 'now': str(datetime.date.today())}))"
    )
    assert "4.0" in result


@pytest.mark.asyncio
async def test_execute_code_timeout():
    from personal_agent.tools.builtin.execute_code import _execute_code

    result = await _execute_code("import time; time.sleep(120)", timeout=5)
    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_execute_code_sandbox_env():
    """API keys should NOT be available in the sandbox."""
    from personal_agent.tools.builtin.execute_code import _execute_code

    result = await _execute_code(
        "import os; print('LLM_API_KEY' in os.environ)"
    )
    assert "False" in result


@pytest.mark.asyncio
async def test_execute_code_isolated_cwd():
    """Sandbox should run in a temp directory, not the agent's directory."""
    from personal_agent.tools.builtin.execute_code import _execute_code

    result = await _execute_code("import os; print(os.getcwd())")
    # Should be a temp dir, not the project dir
    assert "Personal Agent" not in result
    assert "Temp" in result or "tmp" in result.lower()


@pytest.mark.asyncio
async def test_execute_code_no_output():
    from personal_agent.tools.builtin.execute_code import _execute_code

    result = await _execute_code("x = 1 + 1")
    assert "no output" in result.lower()


# ── delegate_task ───────────────────────────────────────


@pytest.mark.asyncio
async def test_delegate_not_initialized():
    """Without setup, delegate should return a clear error."""
    from personal_agent.tools.builtin.delegate import _delegate_task

    result = await _delegate_task("test prompt")
    assert "not initialized" in result.lower()


# ── tools are registered ────────────────────────────────


def test_all_new_tools_registered():
    from personal_agent.tools.registry import tool_registry

    expected = [
        "clarify", "execute_code", "delegate_task",
        "process_list", "process_kill", "process_wait",
    ]
    for name in expected:
        entry = tool_registry.get(name)
        assert entry is not None, f"Tool '{name}' not registered"
        assert entry.toolset in ("builtin",)
