from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TOOL_NAME_ALIASES: dict[str, str] = {
    "open_file": "read_file",
    "find": "grep",
    "find_file": "glob",
    "advance_web_search_tool": "advanced_web_search_tool",
    "web_search": "advanced_web_search_tool",
    "search_web": "advanced_web_search_tool",
    "core_web_search": "advanced_web_search_tool",
    "search": "advanced_web_search_tool",
    "todos": "write_todos",
    "tink": "think",
}


SEARCH_TOOL_NAMES = {"advanced_web_search_tool", "paper_search_tool"}


def normalize_tool_name(name: str) -> str:
    cleaned = name.strip()
    if "<|channel|>" in cleaned:
        cleaned = cleaned.split("<|channel|>", maxsplit=1)[0]
    if "." in cleaned:
        base = cleaned.split(".", maxsplit=1)[0]
        if base:
            cleaned = base
    if "-" in cleaned:
        candidate = cleaned.replace("-", "_")
        if candidate in TOOL_NAME_ALIASES.values() or candidate in SEARCH_TOOL_NAMES:
            cleaned = candidate
    return TOOL_NAME_ALIASES.get(cleaned, TOOL_NAME_ALIASES.get(cleaned.lower(), cleaned))


def normalize_tool_arguments(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    normalized_name = normalize_tool_name(tool_name)
    normalized = dict(arguments)
    if (
        normalized_name in SEARCH_TOOL_NAMES
        and "query" not in normalized
        and "question" in normalized
    ):
        normalized["query"] = normalized["question"]
    return normalized


def truncate_search_query(query: str, *, max_chars: int = 400) -> str:
    stripped = " ".join(query.split())
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3].rstrip() + "..."


def think(_: str | None = None) -> str:
    return "Thought recorded."
