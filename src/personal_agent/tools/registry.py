"""ToolRegistry — module-level singleton. Tools self-register on import.

get_definitions(enabled_toolsets, quiet_mode): resolves toolsets, returns
Anthropic-format schemas. LRU cache (8 entries) when quiet_mode=True.
On bridge mode: deferrable tools replaced with tool_search/describe/call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from personal_agent.tools.entry import ToolEntry

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ToolEntry] = {}
        self._generation: int = 0
        self._defs_cache: dict[tuple, list[dict]] = {}  # key → cached result
        self._cache_maxsize = 8

    # ── registration ──────────────────────────────────

    def register(self, entry: ToolEntry) -> None:
        self._entries[entry.name] = entry
        self._generation += 1

    def unregister(self, name: str) -> None:
        if name in self._entries:
            del self._entries[name]
            self._generation += 1

    def get(self, name: str) -> ToolEntry | None:
        return self._entries.get(name)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def all_names(self) -> set[str]:
        return set(self._entries.keys())

    # ── definitions ────────────────────────────────────

    def get_definitions(
        self,
        enabled_toolsets: list[str] | None = None,
        *,
        quiet_mode: bool = True,
        skip_bridge: bool = False,
    ) -> list[dict]:
        """Return Anthropic-format tool schemas for resolved toolsets.

        quiet_mode=True → cache result (8-entry LRU). Used by Agent init.
        skip_bridge=True → return ALL schemas (used by tool_search index).
        """
        from personal_agent.tools.toolsets import resolve_toolsets, is_core_tool

        # Cache key: toolsets + generation + skip_bridge
        cache_key = (
            frozenset(enabled_toolsets or []),
            self._generation,
            skip_bridge,
        )

        if quiet_mode and cache_key in self._defs_cache:
            return list(self._defs_cache[cache_key])

        # Resolve which tools to include
        resolved = resolve_toolsets(enabled_toolsets, self.all_names)

        # Check dependencies
        active: list[ToolEntry] = []
        for name in sorted(resolved):
            entry = self._entries.get(name)
            if entry is None:
                continue
            if entry.check_fn and not entry.check_fn():
                logger.debug("Tool '%s' skipped: check_fn returned False", name)
                continue
            active.append(entry)

        # Build schemas
        if skip_bridge:
            result = [_entry_to_schema(e) for e in active]
        else:
            result = _assemble_with_bridge(active, is_core_tool)

        # Cache if quiet
        if quiet_mode:
            if len(self._defs_cache) >= self._cache_maxsize:
                oldest = next(iter(self._defs_cache))
                del self._defs_cache[oldest]
            self._defs_cache[cache_key] = result

        return result

    # ── dispatch ───────────────────────────────────────

    async def dispatch(self, name: str, args: dict) -> str:
        entry = self._entries.get(name)
        if entry is None:
            return f"Error: unknown tool '{name}'"
        try:
            return await entry.handler(**args)
        except Exception as exc:
            logger.exception("Tool '%s' failed", name)
            return f"Error: {exc}"


tool_registry = ToolRegistry()


# ── helpers ────────────────────────────────────────────

def _entry_to_schema(entry: ToolEntry) -> dict:
    return {
        "name": entry.name,
        "description": entry.description,
        "input_schema": entry.schema,
    }


def _assemble_with_bridge(active: list[ToolEntry], is_core) -> list[dict]:
    """Split active tools: core → full schema, deferrable → bridge tools."""
    core: list[dict] = []
    deferrable: list[dict] = []

    for entry in active:
        if is_core(entry.name):
            core.append(_entry_to_schema(entry))
        else:
            deferrable.append(_entry_to_schema(entry))

    result = list(core)

    # Only add bridge tools if there are deferrable tools
    if deferrable:
        result.append({
            "name": "tool_search",
            "description": "Search for tools by keyword. Returns matching tools with name, description, and "
                          "full input_schema. After searching, call the matched tool DIRECTLY by name.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for in tool names and descriptions"},
                },
                "required": ["query"],
            },
        })
        result.append({
            "name": "tool_describe",
            "description": "Get the full parameter schema for a specific tool. Use after tool_search if you need more detail.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact tool name from tool_search results"},
                },
                "required": ["name"],
            },
        })
        result.append({
            "name": "tool_call",
            "description": "Execute a safe tool by name with arguments. Destructive tools are blocked — call them directly after /allow.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Tool name to execute"},
                    "arguments": {"type": "object", "description": "Tool arguments as a JSON object"},
                },
                "required": ["name", "arguments"],
            },
        })

    return result


# ── bridge tool dispatch ──────────────────────────────

async def dispatch_tool_search(query: str) -> str:
    """BM25 search over deferrable tool catalog."""
    import json
    from personal_agent.tools.toolsets import is_core_tool

    # Get ALL tool schemas (skip_bridge to avoid recursion)
    all_defs = tool_registry.get_definitions(
        enabled_toolsets=None, quiet_mode=False, skip_bridge=True
    )

    # Build catalog: only deferrable tools
    catalog = [d for d in all_defs if not is_core_tool(d["name"])]
    if not catalog:
        return json.dumps({"hits": [], "message": "No deferrable tools available."}, ensure_ascii=False)

    hits = _bm25_search(catalog, query)
    return json.dumps({"hits": hits}, ensure_ascii=False)


async def dispatch_tool_describe(name: str) -> str:
    """Return full schema for a specific tool."""
    import json
    all_defs = tool_registry.get_definitions(
        enabled_toolsets=None, quiet_mode=False, skip_bridge=True
    )
    for d in all_defs:
        if d["name"] == name:
            return json.dumps(d, ensure_ascii=False)
    return json.dumps({"error": f"Tool not found: {name}"}, ensure_ascii=False)


async def dispatch_tool_call(name: str, arguments: dict) -> str:
    """Execute a tool by name."""
    return await tool_registry.dispatch(name, arguments)


# ── BM25 ──────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple 2-char n-gram tokenizer for CJK + space-split for Latin."""
    import re
    tokens = []
    # Split on whitespace for Latin text
    for word in text.lower().split():
        if re.search(r'[一-鿿]', word):
            # CJK: 2-char ngrams
            for i in range(len(word) - 1):
                tokens.append(word[i:i + 2])
            tokens.append(word[-1])  # last char solo
        else:
            tokens.append(word)
    return tokens


def _bm25_search(catalog: list[dict], query: str, top_k: int = 5) -> list[dict]:
    """BM25 search over catalog."""
    if not catalog:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return catalog[:top_k]

    # Build per-doc token frequencies
    docs_tokens = [_tokenize(d["name"] + " " + d["description"]) for d in catalog]

    k1, b = 1.5, 0.75
    avgdl = sum(len(t) for t in docs_tokens) / max(len(docs_tokens), 1)
    N = len(catalog)

    # IDF
    df: dict[str, int] = {}
    for tokens in docs_tokens:
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1

    scores = []
    for i, doc_tokens in enumerate(docs_tokens):
        score = 0.0
        doc_len = len(doc_tokens)
        tf: dict[str, int] = {}
        for tok in doc_tokens:
            tf[tok] = tf.get(tok, 0) + 1

        for tok in query_tokens:
            if tok not in tf:
                continue
            idf = max(0, __import__("math").log((N - df.get(tok, 0) + 0.5) / (df.get(tok, 0) + 0.5)) + 1)
            numerator = tf[tok] * (k1 + 1)
            denominator = tf[tok] + k1 * (1 - b + b * doc_len / max(avgdl, 1))
            score += idf * numerator / max(denominator, 0.001)

        scores.append({
            "name": catalog[i]["name"],
            "description": catalog[i]["description"],
            "input_schema": catalog[i].get("input_schema", {}),
            "score": round(score, 3),
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return [{"name": s["name"], "description": s["description"],
             "input_schema": s["input_schema"]}
            for s in scores[:top_k] if s["score"] > 0]
