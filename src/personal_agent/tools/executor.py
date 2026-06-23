"""Tool execution pipeline: scope gate → pre-hook → dispatch → post-process.
Parallel/serial execution with individual fault isolation.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time as _time_module
from pathlib import Path
from typing import Any

from personal_agent.tools.registry import tool_registry

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 8000

# ── Interrupt support ────────────────────────────────
# Long-running tools (bash, execute_code) check this to abort early.
# Set by Gateway on /stop, cleared each turn.
_interrupted: bool = False


def set_interrupted() -> None:
    global _interrupted
    _interrupted = True


def clear_interrupted() -> None:
    global _interrupted
    _interrupted = False


def is_interrupted() -> bool:
    return _interrupted


async def execute_tool_calls(
    tool_calls: list[dict],
    messages: list[dict],
    *,
    agent: Any = None,
    hooks: Any = None,
) -> None:
    clear_interrupted()
    """Execute all tool calls, append results to messages in original order.

    Adjacent parallel-safe tools run concurrently; sequential tools act as
    barriers that preserve LLM ordering. Example: [bash, grep, write] →
    bash first (barrier), then grep+write concurrently.
    """
    results: dict[int, str] = {}

    i = 0
    while i < len(tool_calls):
        entry = tool_registry.get(tool_calls[i]["name"])

        if entry and entry.is_parallel_safe:
            # Collect adjacent parallel-safe tools into a batch
            batch: list[tuple[int, dict]] = []
            while i < len(tool_calls):
                e = tool_registry.get(tool_calls[i]["name"])
                if e and e.is_parallel_safe:
                    batch.append((i, tool_calls[i]))
                    i += 1
                else:
                    break

            # Split batch: destructive first (write before read in same batch)
            destructive_batch = [(idx, tc) for idx, tc in batch
                                 if tool_registry.get(tc["name"]) and tool_registry.get(tc["name"]).is_destructive]
            safe_batch = [(idx, tc) for idx, tc in batch
                          if not (tool_registry.get(tc["name"]) and tool_registry.get(tc["name"]).is_destructive)]

            async def _run_batch(items):
                if len(items) == 1:
                    idx, tc = items[0]
                    try:
                        results[idx] = await _exec_one(tc, agent=agent, hooks=hooks)
                    except Exception as exc:
                        results[idx] = f"Error: {exc}"
                elif items:
                    tasks = [asyncio.to_thread(_exec_one_sync, tc, agent, hooks) for _, tc in items]
                    gathered = await asyncio.gather(*tasks, return_exceptions=True)
                    for j, (idx, _tc) in enumerate(items):
                        r = gathered[j]
                        results[idx] = f"Error: {r}" if isinstance(r, Exception) else r

            await _run_batch(destructive_batch)  # writes first
            await _run_batch(safe_batch)          # reads after
        else:
            idx, tc = i, tool_calls[i]
            try:
                results[idx] = await _exec_one(tc, agent=agent, hooks=hooks)
            except Exception as exc:
                results[idx] = f"Error: {exc}"
            i += 1

    # ── append ALL results as ONE user message (Anthropic requires this) ──
    result_blocks = []
    for i, tc in enumerate(tool_calls):
        result_text = results.get(i, "Error: tool execution skipped")
        result_blocks.append({
            "type": "tool_result",
            "tool_use_id": tc["id"],
            "content": result_text,
        })
    messages.append({"role": "user", "content": result_blocks})


async def _exec_one(tc: dict, *, agent: Any = None, hooks: Any = None) -> str:
    """Execute a single tool call through the security pipeline.

    Order matters — hard rejections first, then user-facing gates:
      ① pre-check — hard blocks (never ask user): bash whitelist, ext, SSRF...
      ② scope gate — may ask user: /allow for destructive tools
      ③ checkpoint — backup before destructive write
      ④ pre-hook → dispatch → post-process
    """

    entry = tool_registry.get(tc["name"])
    if entry is None:
        return f"Error: unknown tool '{tc['name']}'"

    # ── ① pre-check: hard rejections, NEVER ask user ──
    pre_error = _pre_check(tc, entry)
    if pre_error:
        return pre_error

    # ── ② scope gate: may ask user (/allow) ────────────
    gate_error = _scope_gate(tc, entry, agent)
    if gate_error:
        return gate_error

    # ── ③ checkpoint (destructive file tools) ──────────
    if tc["name"] in ("write", "edit"):
        _checkpoint_file_write(tc)

    # ── 1. pre-hook ──────────────────────────────────
    if hooks:
        result = await hooks.fire("on_before_tool_exec", tc, entry)
        if result is None:
            return "Error: tool execution blocked"
        if isinstance(result, dict):
            tc = result

    # ── 2. dispatch (with retry for idempotent tools) ──
    max_attempts = 2 if not entry.is_destructive else 1  # retry safe tools once
    last_exc = None
    for attempt in range(max_attempts):
        try:
            result = await entry.handler(**tc["input"])
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1 and _is_retryable(exc):
                logger.warning("Tool '%s' failed (attempt %d/2): %s", tc["name"], attempt + 1, exc)
                await asyncio.sleep(0.5 * (attempt + 1))  # brief backoff
                continue
            logger.exception("Tool dispatch failed for '%s'", tc["name"])
    else:
        # All attempts failed
        result = f"Error: {last_exc}"

    # ── 3. post-process ──────────────────────────────
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + f"\n\n...({len(result) - MAX_RESULT_CHARS} more chars truncated)"

    if hooks:
        modified = await hooks.fire("on_after_tool_exec", tc, result)
        if isinstance(modified, str):
            result = modified

    logger.debug("Tool '%s' done: %d chars", tc["name"], len(result))
    return result


def _exec_one_sync(tc: dict, agent: Any = None, hooks: Any = None) -> str:
    """Synchronous wrapper for thread-pool execution. Fresh event loop per thread.
    Handles Windows ProactorEventLoop quirks gracefully."""
    try:
        return asyncio.run(_exec_one(tc, agent=agent, hooks=hooks))
    except RuntimeError as e:
        if "event loop" in str(e).lower():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_exec_one(tc, agent=agent, hooks=hooks))
            finally:
                loop.close()
        raise


# ── pre-check: hard blocks (NEVER ask user) ─────────


def _pre_check(tc: dict, entry) -> str | None:
    """Hard security checks that NEVER result in user interaction.

    Run BEFORE scope gate — these are unconditional rejections
    that no amount of /allow can override.
    """
    name = tc["name"]
    inp = tc.get("input", {})

    # ── bash: hard blacklist + whitelist + dangerous patterns + chaining + network ──
    if name == "bash":
        from personal_agent.tools.builtin.bash import _check_command
        cmd = inp.get("command", "")
        if cmd:
            err = _check_command(cmd)
            if err:
                return err

    # ── write: extension whitelist + path traversal ──
    elif name == "write":
        path = inp.get("path", "")
        if path:
            from personal_agent.tools.builtin.file_write import _check_extension, _allowed_base
            ext_err = _check_extension(path)
            if ext_err:
                return ext_err
            full = (_allowed_base / path).resolve()
            if not str(full).startswith(str(_allowed_base)):
                return f"Error: path traversal denied — '{path}' is outside allowed directory"
            content = inp.get("content", "")
            if len(content) > 100_000:
                return f"Error: content too large ({len(content)} bytes, max 100000)"

    # ── edit: same path check as write ──
    elif name == "edit":
        path = inp.get("path", "")
        if path:
            from personal_agent.tools.builtin.file_write import _allowed_base
            full = (_allowed_base / path).resolve()
            if not str(full).startswith(str(_allowed_base)):
                return f"Error: path traversal denied — '{path}' is outside allowed directory"

    # ── read: sensitive file blocklist ──
    elif name == "read":
        path = inp.get("path", "")
        if path:
            from personal_agent.tools.builtin.file_read import _allowed_base as _read_base, _check_sensitive
            full = (_read_base / path).resolve()
            if not str(full).startswith(str(_read_base)):
                return f"Error: path traversal denied — '{path}' is outside allowed directory"
            sensitive_err = _check_sensitive(full)
            if sensitive_err:
                return sensitive_err

    # ── web_fetch: SSRF prevention ──
    elif name == "web_fetch":
        url = inp.get("url", "")
        if url:
            from personal_agent.tools.url_safety import check_url
            ssrf_err = check_url(url)
            if ssrf_err:
                return ssrf_err

    return None


# ── scope gate ────────────────────────────────────────

def _tool_category(name: str) -> str:
    """Map tool name → destructive category for granular /allow."""
    _CATEGORY_MAP: dict[str, str] = {
        "write": "write",
        "edit": "write",
        "bash": "bash",
    }
    return _CATEGORY_MAP.get(name, "write")


def _scope_gate(tc: dict, entry, agent: Any) -> str | None:
    """Check if this tool call should be allowed. Returns error string or None."""

    # ① check_fn — runtime dependency check
    if entry.check_fn and not entry.check_fn():
        return f"Error: tool '{tc['name']}' is currently unavailable (dependency not met)"

    if agent is None:
        return None  # no guard checks without agent context

    # ② destructive guard — check category in allowed set
    if entry.is_destructive:
        category = _tool_category(tc["name"])
        if category not in agent._destructive_allowed and "all" not in agent._destructive_allowed:
            return (
                f"Error: destructive tool '{tc['name']}' requires authorization. "
                f"Send /allow {category} or /allow all to enable for this turn."
            )

    # ③ guardrail — per-turn call quota
    if agent._tool_calls_this_turn >= agent._max_tool_calls_per_turn:
        return (
            f"Error: tool call limit ({agent._max_tool_calls_per_turn}) reached. "
            f"Please summarize what has been done and stop."
        )

    # ③b — destructive quota (stricter, default 3 per turn)
    if entry.is_destructive:
        max_destructive = getattr(agent, '_max_destructive_per_turn', 3)
        destructive_count = getattr(agent, '_destructive_calls_this_turn', 0)
        if destructive_count >= max_destructive:
            return (
                f"Error: destructive tool limit ({max_destructive}) reached. "
                f"Please summarize or request /allow for more."
            )
        agent._destructive_calls_this_turn = destructive_count + 1

    agent._tool_calls_this_turn += 1
    return None


# ── checkpoint ────────────────────────────────────────

def _checkpoint_file_write(tc: dict) -> None:
    """Backup target file before a file_write dispatch. Best-effort — never blocks execution."""
    try:
        from personal_agent.tools.builtin.file_write import _allowed_base  # noqa: F401

        path = tc.get("input", {}).get("path", "")
        if not path:
            return

        full = (_allowed_base / path).resolve()
        if not full.exists():
            return  # new file, nothing to backup

        backup_dir = _allowed_base / "checkpoints"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _time_module.strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{full.name}.{timestamp}.bak"
        shutil.copy2(full, backup_path)
        logger.info("Checkpoint saved: %s → %s", path, backup_path.name)
    except Exception:
        logger.exception("Checkpoint failed for file_write — tool execution will proceed")


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is likely transient (worth retrying once)."""
    msg = str(exc).lower()
    transient = (
        "timeout", "connection", "reset", "refused", "temporary",
        "network", "dns", "unreachable", "429", "503", "502", "504",
    )
    return any(k in msg for k in transient)
