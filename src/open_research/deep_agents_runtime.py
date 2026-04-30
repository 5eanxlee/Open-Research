from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html import unescape
from importlib import resources
from importlib.util import find_spec
from typing import Any

from jinja2 import Environment, StrictUndefined

from .agents.middleware import (
    EmptyContentFixMiddleware,
    SearchToolCallLimitMiddleware,
    TodoSanitizationMiddleware,
    ToolNameSanitizationMiddleware,
)
from .citations import build_citation_key
from .config import Settings
from .custom_responses import (
    build_custom_research_report,
    create_run_request_from_research_options,
    evaluate_completion_gate,
)
from .domain import (
    BudgetPolicy,
    CitationRecord,
    CitationSupportLabel,
    CompletionGateResult,
    FinalReport,
    ResearchOptions,
    ResearchReport,
    RunStatus,
    SearchResult,
    resolve_model_config,
)
from .providers import ProviderError, SearchProvider
from .tool_registry import PAPER_SEARCH_TOOL
from .tools import AdvancedWebSearchTool, PaperSearchTool
from .tools import think as record_thought
from .utils import normalize_url


@dataclass(frozen=True, slots=True)
class CustomResponsesDeepAgentStatus:
    available: bool
    backend: str
    missing_dependencies: tuple[str, ...]
    model: str
    search_backend: str
    prompt_templates: dict[str, str]
    middleware: tuple[str, ...]
    tools: tuple[str, ...]
    native_openai_web_search: bool


def inspect_custom_responses_deep_agent(settings: Settings) -> CustomResponsesDeepAgentStatus:
    required = ("deepagents", "langchain", "langchain_openai")
    missing = tuple(name for name in required if find_spec(name) is None)
    return CustomResponsesDeepAgentStatus(
        available=not missing
        and settings.resolved_custom_responses_runtime_backend == "deepagents"
        and settings.resolved_llm_backend == "openai"
        and settings.openai_api_key is not None,
        backend=settings.resolved_custom_responses_runtime_backend,
        missing_dependencies=missing,
        model=settings.lead_model,
        search_backend=settings.resolved_search_backend,
        prompt_templates={
            "orchestrator": "prompts/custom_responses/orchestrator.j2",
            "planner-agent": "prompts/custom_responses/planner.j2",
            "researcher-agent": "prompts/custom_responses/researcher.j2",
        },
        middleware=(
            "TodoListMiddleware",
            "FilesystemMiddleware(StateBackend)",
            "SubAgentMiddleware",
            "EmptyContentFixMiddleware",
            "ToolNameSanitizationMiddleware",
            "TodoSanitizationMiddleware",
            "ModelRetryMiddleware",
            "ToolRetryMiddleware",
            "ToolCallLimitMiddleware",
            "SearchToolCallLimitMiddleware",
        ),
        tools=tuple(available_deep_agent_tool_names(settings)),
        native_openai_web_search=settings.resolved_search_backend == "openai",
    )


async def run_custom_responses_deep_agent_research(
    *,
    runtime: Any,
    prompt: str,
    options: ResearchOptions | None,
) -> ResearchReport:
    status = inspect_custom_responses_deep_agent(runtime.settings)
    if not status.available:
        reason = (
            "Custom Responses DeepAgents runtime is unavailable. "
            f"backend={status.backend}, llm={runtime.settings.resolved_llm_backend}, "
            f"missing={list(status.missing_dependencies)}"
        )
        raise ValueError(reason)

    request = create_run_request_from_research_options(prompt, options)
    budget = request.budget or runtime.default_budget()
    model_config = resolve_model_config(
        request.model_config_override,
        defaults=runtime.default_model_config(),
    )
    profile_id = request.profile_id or "default"
    source_selection = runtime.resolve_source_selection(request.source_selection)
    metadata = {
        **dict(request.metadata),
        "runtime_executor": "deepagents",
        "custom_responses_backend": status.backend,
        "model_config": model_config.model_dump(mode="json"),
        "profile_id": profile_id,
        "effective_budget": budget.model_dump(mode="json"),
        "source_selection": source_selection,
    }
    run = await runtime.store.create_run(
        prompt,
        budget,
        profile_id=profile_id,
        project_id=request.project_id,
        metadata=metadata,
    )
    run_id = run.id
    await runtime.store.set_run_execution_context(
        run_id,
        worker_id=runtime.worker_id,
        workflow_backend=runtime.workflow_backend_name,
    )
    await runtime.store.update_run_status(run_id, RunStatus.RESEARCHING)
    await runtime.events.publish(
        run_id,
        "custom_responses.deepagent.started",
        {
            "model": model_config.lead_model,
            "search_backend": runtime.settings.resolved_search_backend,
            "native_openai_web_search": status.native_openai_web_search,
        },
    )

    try:
        markdown, gate = await _invoke_deep_agent_until_complete(
            runtime=runtime,
            prompt=prompt,
            run_id=run_id,
            budget=budget,
            lead_model=model_config.lead_model,
            planner_model=model_config.planner_model,
            researcher_model=model_config.worker_model,
            source_selection=source_selection,
        )
        source_records = _extract_sources_from_markdown(markdown)
        citations = _build_citation_records(source_records)
        await runtime.store.register_source_registry_entries(
            run_id,
            [
                {
                    "source_id": citation.source_id if index <= len(citations) else None,
                    "url": record["url"],
                    "canonical_url": record["normalized_url"],
                    "normalized_url": record["normalized_url"],
                    "citation_key": record.get("citation_key"),
                    "title": record.get("title"),
                    "provider": runtime.settings.resolved_search_backend,
                    "discovered_via": "custom_responses.deepagents.final_report",
                    "metadata": {
                        "runtime_executor": "deepagents",
                        "citation_number": record.get("citation_number"),
                    },
                }
                for index, record in enumerate(source_records, start=1)
                for citation in [citations[index - 1] if index <= len(citations) else None]
            ],
        )
        registry_entries = await runtime.store.list_source_registry_entries(run_id)
        citation_warnings = _citation_reconciliation_warnings(
            final_sources=source_records,
            registry_entries=registry_entries,
        )
        if citation_warnings:
            await _store_run_warnings(runtime=runtime, run_id=run_id, warnings=citation_warnings)
        final_report = FinalReport(
            markdown=markdown,
            citations=citations,
            unsupported_claims=[] if gate.passed else list(gate.reasons),
            confidence=0.86 if gate.passed else 0.55,
        )
        await runtime.store.update_run_status(
            run_id,
            RunStatus.COMPLETED,
            final_report=final_report,
            terminal_reason="custom_responses_deepagent_completed",
        )
        await runtime.events.publish(
            run_id,
            "completion_gate.completed",
            gate.model_dump(mode="json"),
        )
    except Exception as exc:
        await runtime.store.update_run_status(
            run_id,
            RunStatus.FAILED,
            error_message=str(exc),
            terminal_reason="custom_responses_deepagent_failed",
        )
        await runtime.events.publish(run_id, "run.failed", {"error": str(exc)})
        raise

    detail = await runtime.get_run_detail(run_id)
    if detail is None:
        raise RuntimeError(f"Research run {run_id} could not be loaded after completion.")
    budget_events = await runtime.list_budget_events(run_id)
    return build_custom_research_report(
        detail=detail,
        budget_events=budget_events,
        settings=runtime.settings,
    )


async def _invoke_deep_agent_until_complete(
    *,
    runtime: Any,
    prompt: str,
    run_id: str,
    budget: BudgetPolicy,
    lead_model: str,
    planner_model: str,
    researcher_model: str,
    source_selection: Sequence[str],
) -> tuple[str, CompletionGateResult]:
    agent = _build_deep_agent(
        settings=runtime.settings,
        search_provider=runtime.orchestrator.worker.search_provider,
        run_id=run_id,
        record_budget_event=runtime.store.record_budget_event,
        publish_event=runtime.events.publish,
        register_source_registry_entries=runtime.store.register_source_registry_entries,
        lead_model=lead_model,
        planner_model=planner_model,
        researcher_model=researcher_model,
        max_results=min(5, budget.max_results_per_query),
        source_selection=source_selection,
    )
    state: dict[str, Any] = {"messages": [{"role": "user", "content": prompt}]}
    final_markdown = ""
    final_gate = evaluate_completion_gate(
        "",
        min_chars=runtime.settings.completion_gate_min_chars,
        min_headings=runtime.settings.completion_gate_min_headings,
    )
    for attempt in range(1, runtime.settings.completion_gate_max_attempts + 1):
        previous_message_count = len(list(state.get("messages", [])))
        result = await agent.ainvoke(state)
        await _record_langchain_usage(
            runtime=runtime,
            run_id=run_id,
            result=result,
            phase=f"deepagents_attempt_{attempt}",
            model=lead_model,
            previous_message_count=previous_message_count,
        )
        await runtime.store.update_run_metadata(
            run_id,
            {"custom_responses_response_state": _extract_response_state(result)},
        )
        final_markdown = _extract_last_message_text(result)
        final_gate = evaluate_completion_gate(
            final_markdown,
            min_chars=runtime.settings.completion_gate_min_chars,
            min_headings=runtime.settings.completion_gate_min_headings,
        )
        contract_warnings = await _runtime_contract_warnings(runtime=runtime, run_id=run_id)
        if contract_warnings:
            final_gate = final_gate.model_copy(
                update={
                    "passed": False,
                    "reasons": [*final_gate.reasons, *contract_warnings],
                }
            )
        await runtime.events.publish(
            run_id,
            "completion_gate.evaluated",
            {"attempt": attempt, **final_gate.model_dump(mode="json")},
        )
        if final_gate.passed:
            return final_markdown, final_gate
        if attempt >= runtime.settings.completion_gate_max_attempts:
            break
        continuation = _completion_continuation_prompt(final_gate)
        state = dict(result)
        state["messages"] = [
            *list(result.get("messages", [])),
            {"role": "user", "content": continuation},
        ]
        await runtime.events.publish(
            run_id,
            "completion_gate.continuation_requested",
            {"attempt": attempt, "reasons": list(final_gate.reasons)},
        )
    return final_markdown, final_gate


def _build_deep_agent(
    *,
    settings: Settings,
    search_provider: SearchProvider,
    run_id: str,
    record_budget_event: Any,
    publish_event: Any,
    register_source_registry_entries: Any,
    lead_model: str,
    planner_model: str,
    researcher_model: str,
    max_results: int,
    source_selection: Sequence[str],
) -> Any:
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain.agents.middleware import TodoListMiddleware, ToolCallLimitMiddleware
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    web_search_tool = AdvancedWebSearchTool(search_provider, max_results=max_results)
    paper_enabled = _paper_search_enabled(settings=settings, source_selection=source_selection)
    paper_tool = (
        PaperSearchTool(
            settings.serper_api_key.get_secret_value(),
            timeout=settings.http_timeout_seconds,
            max_results=max_results,
        )
        if paper_enabled
        else None
    )

    def make_advanced_web_search_tool(agent_role: str):
        @tool("advanced_web_search_tool")
        async def advanced_web_search_tool(query: str) -> str:
            """Search the public web and return normalized document blocks with source URLs."""
            await _publish_tool_event(
                publish_event=publish_event,
                run_id=run_id,
                event_type="deepagents.tool.started",
                payload={
                    "provider": search_provider.provider_name,
                    "tool": "advanced_web_search_tool",
                    "agent_role": agent_role,
                    "query_hash": _hash_text(query),
                },
            )
            await record_budget_event(
                run_id,
                "search_calls",
                1,
                {
                    "provider": search_provider.provider_name,
                    "tool": "advanced_web_search_tool",
                    "agent_role": agent_role,
                },
            )
            try:
                output = await web_search_tool.search(query)
            except ProviderError as exc:
                await _publish_tool_event(
                    publish_event=publish_event,
                    run_id=run_id,
                    event_type="deepagents.tool.failed",
                    payload={
                        "provider": search_provider.provider_name,
                        "tool": "advanced_web_search_tool",
                        "agent_role": agent_role,
                        "query_hash": _hash_text(query),
                        "error": str(exc),
                    },
                )
                output = f"Search failed: {exc}"
            source_count = await _register_tool_output_sources(
                run_id=run_id,
                register_source_registry_entries=register_source_registry_entries,
                output=output,
                tool_name="advanced_web_search_tool",
                provider=search_provider.provider_name,
                agent_role=agent_role,
                query=query,
            )
            event_payload = {
                "provider": search_provider.provider_name,
                "tool": "advanced_web_search_tool",
                "agent_role": agent_role,
                "query": query,
                "query_hash": _hash_text(query),
                "source_count": source_count,
            }
            await _publish_tool_event(
                publish_event=publish_event,
                run_id=run_id,
                event_type="search.performed",
                payload=event_payload,
            )
            await _publish_tool_event(
                publish_event=publish_event,
                run_id=run_id,
                event_type="deepagents.tool.completed",
                payload=event_payload,
            )
            return output

        return advanced_web_search_tool

    def make_paper_search_tool(agent_role: str):
        if paper_tool is None:
            return None

        @tool("paper_search_tool")
        async def paper_search_tool(query: str, year: str | int | None = None) -> str:
            """Search scholarly papers for academic, empirical, or technical evidence."""
            await _publish_tool_event(
                publish_event=publish_event,
                run_id=run_id,
                event_type="deepagents.tool.started",
                payload={
                    "provider": "serper_scholar",
                    "tool": "paper_search_tool",
                    "agent_role": agent_role,
                    "query_hash": _hash_text(query),
                    "year": year,
                },
            )
            await record_budget_event(
                run_id,
                "search_calls",
                1,
                {
                    "provider": "serper_scholar",
                    "tool": "paper_search_tool",
                    "agent_role": agent_role,
                },
            )
            output = await paper_tool.search(query, year=year)
            source_count = await _register_tool_output_sources(
                run_id=run_id,
                register_source_registry_entries=register_source_registry_entries,
                output=output,
                tool_name="paper_search_tool",
                provider="serper_scholar",
                agent_role=agent_role,
                query=query,
            )
            event_payload = {
                "provider": "serper_scholar",
                "tool": "paper_search_tool",
                "agent_role": agent_role,
                "query": query,
                "query_hash": _hash_text(query),
                "source_count": source_count,
                "year": year,
            }
            await _publish_tool_event(
                publish_event=publish_event,
                run_id=run_id,
                event_type="search.performed",
                payload=event_payload,
            )
            await _publish_tool_event(
                publish_event=publish_event,
                run_id=run_id,
                event_type="deepagents.tool.completed",
                payload=event_payload,
            )
            return output

        return paper_search_tool

    @tool("think")
    def think(thought: str) -> str:
        """Record a concise private research or synthesis thought."""
        return record_thought(thought)

    orchestrator_web_search = make_advanced_web_search_tool("orchestrator")
    planner_web_search = make_advanced_web_search_tool("planner-agent")
    researcher_web_search = make_advanced_web_search_tool("researcher-agent")
    orchestrator_paper_search = make_paper_search_tool("orchestrator")
    researcher_paper_search = make_paper_search_tool("researcher-agent")

    model = ChatOpenAI(
        model=lead_model,
        api_key=api_key,
        base_url=settings.openai_base_url,
        timeout=settings.http_timeout_seconds,
        max_retries=settings.provider_retry_attempts,
        reasoning_effort=settings.llm_reasoning_effort,
        use_responses_api=True,
        use_previous_response_id=True,
        stream_usage=True,
        store=True,
    )
    planner = ChatOpenAI(
        model=planner_model,
        api_key=api_key,
        base_url=settings.openai_base_url,
        timeout=settings.http_timeout_seconds,
        max_retries=settings.provider_retry_attempts,
        reasoning_effort=settings.openai_web_search_reasoning_effort,
        use_responses_api=True,
        use_previous_response_id=True,
        stream_usage=True,
        store=True,
    )
    researcher = ChatOpenAI(
        model=researcher_model,
        api_key=api_key,
        base_url=settings.openai_base_url,
        timeout=settings.http_timeout_seconds,
        max_retries=settings.provider_retry_attempts,
        reasoning_effort=settings.openai_web_search_reasoning_effort,
        use_responses_api=True,
        use_previous_response_id=True,
        stream_usage=True,
        store=True,
    )
    tool_list = [
        tool
        for tool in [orchestrator_web_search, orchestrator_paper_search, think]
        if tool is not None
    ]
    planner_tools = [planner_web_search, think]
    researcher_tools = [
        tool
        for tool in [researcher_web_search, researcher_paper_search, think]
        if tool is not None
    ]
    repair_tools = [
        *tool_list,
        *planner_tools,
        *researcher_tools,
        *TodoListMiddleware().tools,
        *FilesystemMiddleware(backend=StateBackend()).tools,
    ]
    reliability_middleware = _build_reliability_middleware(
        settings=settings,
        repair_tools=repair_tools,
    )
    planner_middleware = _build_subagent_middleware(
        settings=settings,
        repair_tools=repair_tools,
        search_run_limit=max(settings.planner_min_discovery_queries + 8, 18),
    )
    researcher_middleware = _build_subagent_middleware(
        settings=settings,
        repair_tools=repair_tools,
        search_run_limit=8,
    )
    context = {
        "current_datetime": datetime.now(UTC).isoformat(),
        "tools": [
            {"name": name}
            for name in available_deep_agent_tool_names(
                settings,
                source_selection=source_selection,
            )
        ],
    }
    middleware: list[Any] = [*reliability_middleware]
    for tool_name in ["advanced_web_search_tool", *([PAPER_SEARCH_TOOL] if paper_tool else [])]:
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name=tool_name,
                run_limit=settings.max_search_tool_calls_per_run,
                exit_behavior="continue",
            )
        )
    return create_deep_agent(
        model=model,
        tools=tool_list,
        system_prompt=_render_prompt_template("orchestrator", context),
        subagents=[
            {
                "name": "planner-agent",
                "description": (
                    "Creates evidence-grounded research plans, tables of contents, "
                    "constraints, and executable queries."
                ),
                "system_prompt": _render_prompt_template("planner", context),
                "tools": planner_tools,
                "model": planner,
                "middleware": planner_middleware,
            },
            {
                "name": "researcher-agent",
                "description": (
                    "Performs delegated evidence gathering and synthesizes findings "
                    "with source URLs."
                ),
                "system_prompt": _render_prompt_template("researcher", context),
                "tools": researcher_tools,
                "model": researcher,
                "middleware": researcher_middleware,
            },
        ],
        backend=StateBackend(),
        middleware=middleware,
        name="custom-responses-orchestrator",
    )


def available_deep_agent_tool_names(
    settings: Settings,
    *,
    source_selection: Sequence[str] | None = None,
) -> list[str]:
    names = [
        "advanced_web_search_tool",
        "think",
        "write_todos",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "ls",
        "execute",
    ]
    if _paper_search_enabled(settings=settings, source_selection=source_selection):
        names.insert(1, PAPER_SEARCH_TOOL)
    return names


def _paper_search_enabled(
    *,
    settings: Settings,
    source_selection: Sequence[str] | None,
) -> bool:
    if settings.serper_api_key is None:
        return False
    if source_selection is None:
        return True
    return "serper-scholar" in set(source_selection)


def _build_reliability_middleware(*, settings: Settings, repair_tools: Sequence[Any]) -> list[Any]:
    from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

    return [
        EmptyContentFixMiddleware(),
        ToolNameSanitizationMiddleware(tools=repair_tools),
        TodoSanitizationMiddleware(),
        ModelRetryMiddleware(
            max_retries=settings.provider_retry_attempts,
            backoff_factor=2.0,
            initial_delay=settings.provider_retry_base_seconds,
            max_delay=settings.provider_retry_max_seconds,
        ),
        ToolRetryMiddleware(
            max_retries=2,
            tools=["advanced_web_search_tool", "paper_search_tool"],
            retry_on=(TimeoutError, ConnectionError, ProviderError),
            backoff_factor=2.0,
            initial_delay=settings.provider_retry_base_seconds,
            max_delay=settings.provider_retry_max_seconds,
        ),
    ]


def _build_subagent_middleware(
    *,
    settings: Settings,
    repair_tools: Sequence[Any],
    search_run_limit: int,
) -> list[Any]:
    return [
        *_build_reliability_middleware(settings=settings, repair_tools=repair_tools),
        SearchToolCallLimitMiddleware(run_limit=search_run_limit, exit_behavior="continue"),
    ]


def _render_prompt_template(name: str, context: dict[str, Any]) -> str:
    template = (
        resources.files("open_research")
        .joinpath("prompts", "custom_responses", f"{name}.j2")
        .read_text(encoding="utf-8")
    )
    environment = Environment(undefined=StrictUndefined, autoescape=False)
    return environment.from_string(template).render(**context)


def _completion_continuation_prompt(gate: CompletionGateResult) -> str:
    return (
        "Continue the same research run and revise the final answer into a complete, "
        "publication-ready report. Do not ask for confirmation or mention internal workflow. "
        "Fix these completion gate failures: "
        + "; ".join(gate.reasons)
        + ". Preserve inline numeric citations and include a Sources section."
    )


def _extract_last_message_text(result: dict[str, Any]) -> str:
    messages = list(result.get("messages", []))
    if not messages:
        return ""
    content = getattr(
        messages[-1],
        "content",
        messages[-1].get("content") if isinstance(messages[-1], dict) else "",
    )
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


async def _record_langchain_usage(
    *,
    runtime: Any,
    run_id: str,
    result: dict[str, Any],
    phase: str,
    model: str,
    previous_message_count: int = 0,
) -> None:
    messages = list(result.get("messages", []))
    new_messages = messages[previous_message_count:] if previous_message_count > 0 else messages
    usage_by_model: dict[str, dict[str, int]] = {}
    for message in new_messages:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        message_model = _message_model_name(message) or model
        totals = usage_by_model.setdefault(
            message_model,
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
    for message_model, totals in usage_by_model.items():
        if totals["total_tokens"] <= 0:
            continue
        await runtime.store.record_budget_event(
            run_id,
            "llm_tokens",
            totals["total_tokens"],
            {
                "model": message_model,
                "phase": phase,
                "input_tokens": totals["input_tokens"],
                "output_tokens": totals["output_tokens"],
                "total_tokens": totals["total_tokens"],
            },
        )


def _message_model_name(message: Any) -> str | None:
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("model_name", "model", "model_id"):
            value = response_metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _extract_response_state(result: dict[str, Any]) -> dict[str, Any]:
    response_ids: list[str] = []
    seen: set[str] = set()
    for message in list(result.get("messages", [])):
        for candidate in _walk_response_id_candidates(message):
            if candidate in seen:
                continue
            seen.add(candidate)
            response_ids.append(candidate)
    return {
        "response_ids": response_ids,
        "last_response_id": response_ids[-1] if response_ids else None,
        "provider_items_retained": True,
    }


def _walk_response_id_candidates(value: Any) -> list[str]:
    candidates: list[str] = []
    stack = [
        getattr(value, "response_metadata", None),
        getattr(value, "additional_kwargs", None),
        getattr(value, "id", None),
    ]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            if current.startswith(("resp_", "response_")):
                candidates.append(current)
            continue
        if isinstance(current, dict):
            for key, nested in current.items():
                if key in {"id", "response_id", "previous_response_id"} and isinstance(nested, str):
                    if nested.startswith(("resp_", "response_")):
                        candidates.append(nested)
                else:
                    stack.append(nested)
            continue
        if isinstance(current, list):
            stack.extend(current)
    return candidates


_DOCUMENT_BLOCK_RE = re.compile(
    r'<Document\s+href="(?P<url>[^"]+)">\s*<title>\s*(?P<title>.*?)\s*</title>\s*(?P<snippet>.*?)\s*</Document>',
    re.IGNORECASE | re.DOTALL,
)
_PAPER_BLOCK_RE = re.compile(
    r"^\s*\d+\.\s+\*\*(?P<title>.+?)\*\*\s+\((?P<year>.*?)\)\s*"
    r"(?P<body>.*?)(?=^\s*\d+\.\s+\*\*|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_PAPER_LINK_RE = re.compile(r"^\s*-\s+\*\*Link\*\*:\s*(?P<url>https?://\S+)", re.MULTILINE)
_SOURCE_LINE_RE = re.compile(
    r"^\s*\[(?P<number>\d+)\]\s*(?P<title>[^:\n]+?)?\s*:?\s*(?P<url>https?://\S+)",
    re.IGNORECASE | re.MULTILINE,
)
_URL_RE = re.compile(r"https?://[^\s<>)\]]+")


async def _register_tool_output_sources(
    *,
    run_id: str,
    register_source_registry_entries: Any,
    output: str,
    tool_name: str,
    provider: str,
    agent_role: str,
    query: str,
) -> int:
    entries = [
        _source_candidate_to_registry_entry(
            candidate,
            tool_name=tool_name,
            provider=provider,
            agent_role=agent_role,
            query=query,
        )
        for candidate in _extract_tool_source_candidates(output, tool_name=tool_name)
    ]
    if entries:
        await register_source_registry_entries(run_id, entries)
    return len(entries)


async def _publish_tool_event(
    *,
    publish_event: Any,
    run_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    try:
        await publish_event(run_id, event_type, payload)
    except Exception:
        # Tool observability should never turn a successful search into a failed run.
        return


def _extract_tool_source_candidates(output: str, *, tool_name: str) -> list[dict[str, Any]]:
    if tool_name == "paper_search_tool":
        return _extract_paper_source_candidates(output)
    return _extract_web_source_candidates(output)


def _extract_web_source_candidates(output: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _DOCUMENT_BLOCK_RE.finditer(output):
        url = _clean_url(unescape(match.group("url")))
        normalized = _safe_normalize(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        snippet = _clean_text_block(unescape(match.group("snippet")))
        candidates.append(
            {
                "url": url,
                "normalized_url": normalized,
                "title": _clean_text_block(unescape(match.group("title"))) or url,
                "snippet": snippet,
                "snippet_hash": _hash_text(snippet),
            }
        )
    return candidates


def _extract_paper_source_candidates(output: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _PAPER_BLOCK_RE.finditer(output):
        link_match = _PAPER_LINK_RE.search(match.group("body"))
        if link_match is None:
            continue
        url = _clean_url(link_match.group("url"))
        normalized = _safe_normalize(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        body = _clean_text_block(match.group("body"))
        candidates.append(
            {
                "url": url,
                "normalized_url": normalized,
                "title": _clean_text_block(match.group("title")) or url,
                "snippet": body,
                "snippet_hash": _hash_text(body),
                "year": _clean_text_block(match.group("year")),
            }
        )
    return candidates


def _source_candidate_to_registry_entry(
    candidate: dict[str, Any],
    *,
    tool_name: str,
    provider: str,
    agent_role: str,
    query: str,
) -> dict[str, Any]:
    url = candidate["url"]
    title = candidate.get("title")
    normalized_url = candidate["normalized_url"]
    return {
        "url": url,
        "canonical_url": normalized_url,
        "normalized_url": normalized_url,
        "citation_key": build_citation_key(title, normalized_url),
        "title": title,
        "provider": provider,
        "discovered_via": f"custom_responses.deepagents.{agent_role}.{tool_name}",
        "metadata": {
            "query": query,
            "agent_role": agent_role,
            "tool": tool_name,
            "snippet_hash": candidate.get("snippet_hash"),
            "snippet": str(candidate.get("snippet") or "")[:1000],
            "year": candidate.get("year"),
        },
    }


def _clean_text_block(text: str) -> str:
    return " ".join(text.split())


def _hash_text(text: str) -> str | None:
    cleaned = _clean_text_block(text)
    if not cleaned:
        return None
    return sha256(cleaned.encode("utf-8")).hexdigest()


async def _runtime_contract_warnings(*, runtime: Any, run_id: str) -> list[str]:
    budget_events = await runtime.list_budget_events(run_id)
    planner_web_searches = 0
    researcher_task_overages = 0
    for event in budget_events:
        if event.get("category") != "search_calls":
            continue
        metadata = dict(event.get("metadata") or {})
        if (
            metadata.get("agent_role") == "planner-agent"
            and metadata.get("tool") == "advanced_web_search_tool"
        ):
            planner_web_searches += int(event.get("delta") or 0)
        if metadata.get("agent_role") == "researcher-agent" and int(event.get("delta") or 0) > 8:
            researcher_task_overages += 1

    warnings: list[str] = []
    min_queries = runtime.settings.planner_min_discovery_queries
    if planner_web_searches < min_queries:
        warnings.append(
            "Planner discovery is below contract: "
            f"{planner_web_searches}/{min_queries} web searches."
        )
    if researcher_task_overages:
        warnings.append("A researcher task exceeded the configured search-call budget.")
    return warnings


async def _store_run_warnings(*, runtime: Any, run_id: str, warnings: Sequence[str]) -> None:
    detail = await runtime.get_run_detail(run_id)
    existing: list[str] = []
    if detail is not None:
        raw_existing = detail.metadata.get("custom_response_warnings")
        if isinstance(raw_existing, list):
            existing = [str(warning) for warning in raw_existing if warning]
    merged = list(dict.fromkeys([*existing, *(str(warning) for warning in warnings if warning)]))
    await runtime.store.update_run_metadata(run_id, {"custom_response_warnings": merged})


def _citation_reconciliation_warnings(
    *,
    final_sources: Sequence[dict[str, Any]],
    registry_entries: Sequence[Any],
) -> list[str]:
    tool_seen_urls = {
        entry.normalized_url
        for entry in registry_entries
        if str(entry.discovered_via).startswith("custom_responses.deepagents.")
        and ".final_report" not in str(entry.discovered_via)
    }
    final_urls = {str(source["normalized_url"]) for source in final_sources}
    if not final_urls:
        return ["Final report did not expose a parseable Sources section."]
    unseen = sorted(final_urls - tool_seen_urls)
    if not unseen:
        return []
    return [
        "Final report cited sources that were not captured from tool output: "
        + ", ".join(unseen[:5])
        + ("." if len(unseen) <= 5 else f", and {len(unseen) - 5} more.")
    ]


def _extract_sources_from_markdown(markdown: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _SOURCE_LINE_RE.finditer(markdown):
        url = _clean_url(match.group("url"))
        normalized = _safe_normalize(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        number = int(match.group("number"))
        title = (match.group("title") or f"Source {number}").strip(" -:")
        records.append(
            {
                "citation_number": number,
                "citation_key": str(number),
                "title": title or f"Source {number}",
                "url": url,
                "normalized_url": normalized,
            }
        )
    for index, raw_url in enumerate(_URL_RE.findall(markdown), start=1):
        url = _clean_url(raw_url)
        normalized = _safe_normalize(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        records.append(
            {
                "citation_number": None,
                "citation_key": None,
                "title": f"Source {index}",
                "url": url,
                "normalized_url": normalized,
            }
        )
    return records


def _build_citation_records(records: Sequence[dict[str, Any]]) -> list[CitationRecord]:
    citations: list[CitationRecord] = []
    for index, record in enumerate(records, start=1):
        citation_number = record.get("citation_number") or index
        citations.append(
            CitationRecord(
                claim=f"Final report source {citation_number}",
                support_label=CitationSupportLabel.SUPPORTED,
                source_id=f"deepagents-source-{citation_number}",
                source_title=str(record.get("title") or f"Source {citation_number}"),
                source_url=record["url"],
                citation_key=str(citation_number),
                passage_index=0,
                quote=str(record.get("title") or record["url"]),
                confidence=0.75,
            )
        )
    return citations


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:)]}\"'")


def _safe_normalize(url: str) -> str:
    try:
        return normalize_url(url)
    except Exception:
        return url


def search_results_to_markdown_sources(results: Sequence[SearchResult]) -> str:
    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(f"[{index}] {result.title}: {result.url}")
    return "\n".join(lines)
