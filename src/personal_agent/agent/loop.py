"""run_conversation — the core while loop: LLM call → parse → tools → continue."""

from __future__ import annotations

import logging

from personal_agent.tools.executor import execute_tool_calls

logger = logging.getLogger(__name__)


async def run_conversation(agent, ctx) -> dict:
    """Execute the agent while loop. Returns final result dict."""
    while agent._iteration_budget > 0:
        if agent._interrupt_requested:
            logger.info("Agent interrupted by user")
            break

        # ── build api_messages (injections, NOT persisted) ──
        api_messages = await _build_api_messages(agent, ctx)

        # ── refresh system prompt if tools changed ──
        system_prompt = agent._cached_system_prompt or ""

        # ── hooks: on_before_llm_call ──
        hook_result = await agent.hooks.fire(
            "on_before_llm_call",
            api_messages, system_prompt, agent.tools,
        )
        if isinstance(hook_result, dict):
            api_messages = hook_result.get("messages", api_messages)
            system_prompt = hook_result.get("system_prompt", system_prompt)

        # ── LLM call ──
        try:
            response = await agent._transport.call(
                messages=api_messages,
                system_prompt=system_prompt,
                tools=agent.tools,
                max_tokens=agent._provider.max_tokens,
            )
        except Exception as exc:
            logger.exception("LLM call failed")
            return {
                "final_response": f"抱歉，模型调用出错了：{exc}",
                "messages": ctx.messages,
                "api_calls": agent.session_api_calls,
                "completed": False,
            }

        agent.session_prompt_tokens += response.usage.get("input_tokens", 0)
        agent.session_completion_tokens += response.usage.get("output_tokens", 0)
        agent.session_api_calls += 1

        # ── hooks: on_after_llm_call ──
        modified = await agent.hooks.fire("on_after_llm_call", response, response.usage)
        if isinstance(modified, dict):
            response.text = modified.get("text", response.text)

        # ── retry: empty response ──
        if not response.text and not response.tool_calls:
            if agent._retry.empty_content_retries < agent._retry.MAX_EMPTY_CONTENT:
                agent._retry.empty_content_retries += 1
                ctx.messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "请继续。"}],
                })
                logger.debug("Empty response retry %d", agent._retry.empty_content_retries)
                continue
            return {
                "final_response": "(empty response from model)",
                "messages": ctx.messages,
                "api_calls": agent.session_api_calls,
                "completed": True,
            }

        # ── retry: invalid JSON in tool calls ──
        invalid_tools = [
            tc for tc in (response.tool_calls or [])
            if not tc.get("input") and tc.get("name")
        ]
        if invalid_tools and response.tool_calls:
            if agent._retry.invalid_tool_retries < agent._retry.MAX_INVALID_TOOL:
                agent._retry.invalid_tool_retries += 1
                bad_names = ", ".join(tc["name"] for tc in invalid_tools)
                ctx.messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text":
                        f"Your previous tool call(s) had invalid JSON arguments: {bad_names}. "
                        f"Please retry with valid JSON arguments."}],
                })
                logger.debug("Invalid tool retry %d: %s", agent._retry.invalid_tool_retries, bad_names)
                continue

        # ── no tool_calls → done ──
        if not response.tool_calls:
            ctx.messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response.text}],
            })
            break

        # ── has tool_calls → execute ──
        assistant_blocks = []
        if response.text:
            assistant_blocks.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            assistant_blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": tc["input"],
            })
        ctx.messages.append({"role": "assistant", "content": assistant_blocks})

        await execute_tool_calls(response.tool_calls, ctx.messages, hooks=agent.hooks)

        # ── retry: post-tool empty ──
        agent._iteration_budget -= 1
        if agent._iteration_budget <= 0:
            ctx.messages.append({
                "role": "user",
                "content": [{"type": "text", "text": "请总结一下已完成的操作。"}],
            })
            break

    # ── final response ──
    final_text = ""
    if ctx.messages and ctx.messages[-1]["role"] == "assistant":
        for block in ctx.messages[-1].get("content", []):
            if block.get("type") == "text":
                final_text += block["text"]

    return {
        "final_response": final_text,
        "messages": ctx.messages,
        "api_calls": agent.session_api_calls,
        "completed": True,
    }


async def _build_api_messages(agent, ctx) -> list[dict]:
    """Build messages for LLM: messages + injections. NOT persisted."""
    msgs = list(ctx.messages)

    # Memory prefetch: external provider results injected as prefix
    if agent._memory_manager:
        try:
            prefetched = await agent._memory_manager.prefetch(ctx.original_user_message)
            for p in prefetched:
                msgs.insert(0, p)
        except Exception:
            pass  # prefetch failure never blocks the turn

    return msgs
