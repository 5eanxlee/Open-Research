from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from open_research.tools.normalization import normalize_tool_arguments, normalize_tool_name

SEARCH_TOOL_NAMES = {"advanced_web_search_tool", "paper_search_tool"}
VALID_TODO_STATUSES = {"pending", "in_progress", "completed"}
TODO_STATUS_ALIASES = {
    "todo": "pending",
    "not_started": "pending",
    "not-started": "pending",
    "open": "pending",
    "doing": "in_progress",
    "in-progress": "in_progress",
    "in progress": "in_progress",
    "started": "in_progress",
    "active": "in_progress",
    "done": "completed",
    "complete": "completed",
    "finished": "completed",
    "closed": "completed",
}


class EmptyContentFixMiddleware(AgentMiddleware):
    """Repair empty message content before provider validation."""

    tools: Sequence[BaseTool] = ()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request.override(messages=_repair_empty_messages(request.messages)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request.override(messages=_repair_empty_messages(request.messages)))


class ToolNameSanitizationMiddleware(AgentMiddleware):
    """Normalize common malformed tool names and search argument aliases."""

    tools: Sequence[BaseTool] = ()

    def __init__(self, *, tools: Sequence[BaseTool]) -> None:
        super().__init__()
        self._tools_by_name = {tool.name: tool for tool in tools}

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._rewrite_request(request))

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await handler(self._rewrite_request(request))

    def _rewrite_request(self, request: Any) -> Any:
        tool_call = dict(request.tool_call)
        raw_name = str(tool_call.get("name") or "")
        normalized_name = normalize_tool_name(raw_name)
        raw_args = tool_call.get("args") or {}
        if not isinstance(raw_args, dict):
            raw_args = {"query": str(raw_args)}
        normalized_args = normalize_tool_arguments(normalized_name, raw_args)
        if normalized_name == raw_name and normalized_args == raw_args:
            return request

        updated_tool_call = {**tool_call, "name": normalized_name, "args": normalized_args}
        overrides: dict[str, Any] = {"tool_call": updated_tool_call}
        mapped_tool = self._tools_by_name.get(normalized_name)
        if mapped_tool is not None:
            overrides["tool"] = mapped_tool
        return request.override(**overrides)


class TodoSanitizationMiddleware(AgentMiddleware):
    """Normalize malformed `write_todos` payloads before validation."""

    tools: Sequence[BaseTool] = ()

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._rewrite_request(request))

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await handler(self._rewrite_request(request))

    @staticmethod
    def _rewrite_request(request: Any) -> Any:
        tool_call = dict(request.tool_call)
        if normalize_tool_name(str(tool_call.get("name") or "")) != "write_todos":
            return request
        args = tool_call.get("args") or {}
        if not isinstance(args, dict):
            args = {"todos": args}
        sanitized = {**args, "todos": sanitize_todos(args.get("todos", args))}
        return request.override(tool_call={**tool_call, "args": sanitized})


class SearchToolCallLimitMiddleware(ToolCallLimitMiddleware):
    """Enforce a combined run limit across all search tools."""

    def __init__(
        self,
        *,
        run_limit: int,
        tool_names: set[str] | None = None,
        exit_behavior: str = "continue",
    ) -> None:
        super().__init__(tool_name=None, run_limit=run_limit, exit_behavior=exit_behavior)
        self.tool_names = tool_names or set(SEARCH_TOOL_NAMES)

    @property
    def name(self) -> str:
        return "SearchToolCallLimitMiddleware"

    def _matches_tool_filter(self, tool_call: Any) -> bool:
        return normalize_tool_name(str(tool_call.get("name") or "")) in self.tool_names


def sanitize_todos(raw_todos: Any) -> list[dict[str, str]]:
    todos = _coerce_todo_list(raw_todos)
    sanitized: list[dict[str, str]] = []
    for item in todos:
        todo = _coerce_single_todo(item)
        if todo is not None:
            sanitized.append(todo)
    return sanitized or [{"content": "Continue the research task.", "status": "in_progress"}]


def _repair_empty_messages(messages: Sequence[Any]) -> list[Any]:
    return [_repair_empty_message(message) for message in messages]


def _repair_empty_message(message: Any) -> Any:
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    )
    if not _is_empty_content(content):
        return message
    replacement = "empty content received."
    if isinstance(message, dict):
        return {**message, "content": replacement}
    if isinstance(message, BaseMessage):
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"content": replacement})
        return message.copy(update={"content": replacement})
    return ToolMessage(
        content=replacement,
        tool_call_id=getattr(message, "tool_call_id", "unknown"),
    )


def _is_empty_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return len(content) == 0
    return False


def _coerce_todo_list(raw_todos: Any) -> list[Any]:
    value = raw_todos
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
    if isinstance(value, dict):
        for key in ("todos", "tasks", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [value]
    if isinstance(value, list):
        return value
    return [value]


def _coerce_single_todo(item: Any) -> dict[str, str] | None:
    if isinstance(item, str):
        content = item.strip()
        return {"content": content, "status": "pending"} if content else None
    if not isinstance(item, dict):
        content = str(item).strip()
        return {"content": content, "status": "pending"} if content else None

    content = str(
        item.get("content")
        or item.get("task")
        or item.get("title")
        or item.get("description")
        or ""
    ).strip()
    if not content:
        return None
    raw_status = str(item.get("status") or "pending").strip().lower()
    normalized_status = TODO_STATUS_ALIASES.get(raw_status, raw_status)
    if normalized_status not in VALID_TODO_STATUSES:
        normalized_status = "pending"
    return {"content": content, "status": normalized_status}
