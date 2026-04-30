from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from .config import Settings
from .domain import (
    CompletionGateResult,
    CreateRunRequest,
    ResearchOptions,
    ResearchReport,
    ResearchReportSource,
    RunDetail,
    RunStatus,
    SourceRegistryEntry,
)
from .tool_registry import contract_tool_names
from .utils import normalize_url

FORBIDDEN_COMPLETION_PHRASES = (
    "please confirm",
    "do you want me to",
    "should i proceed",
    "choose one",
    "option 1",
    "option 2",
    "option (1",
    "option (2",
    "allow me to",
    "i need your permission",
    "i can't produce",
    "i cannot produce",
    "what i need from you",
)


def create_run_request_from_research_options(
    prompt: str,
    options: ResearchOptions | None,
) -> CreateRunRequest:
    resolved = options or ResearchOptions()
    metadata = dict(resolved.metadata)
    metadata.update(
        {
            "runtime_contract": "custom_responses_research",
            "runtime_contract_version": "2026-04-30.1",
            "completion_gate_enabled": True,
        }
    )
    return CreateRunRequest(
        question=prompt,
        budget=resolved.budget,
        agent_config=resolved.agent_config,
        model_config_override=resolved.model_config_override,
        profile_id=resolved.profile_id,
        project_id=resolved.project_id,
        memory_policy_override=resolved.memory_policy_override,
        execution_mode=resolved.execution_mode,
        require_plan_approval=resolved.require_plan_approval,
        source_selection=resolved.source_selection,
        input_assets=resolved.input_assets,
        staged_asset_ids=resolved.staged_asset_ids,
        async_submit=False,
        metadata=metadata,
    )


async def wait_for_terminal_detail(
    *,
    runtime: Any,
    run_id: str,
    timeout_seconds: float | None,
    poll_seconds: float = 0.2,
) -> RunDetail:
    loop = asyncio.get_running_loop()
    deadline = None if timeout_seconds is None else loop.time() + timeout_seconds
    while True:
        detail = await runtime.get_run_detail(run_id)
        if detail is not None and detail.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            return detail
        if deadline is not None and loop.time() >= deadline:
            raise TimeoutError(f"Research run {run_id} did not complete before timeout.")
        await asyncio.sleep(poll_seconds)


def evaluate_completion_gate(
    markdown: str,
    *,
    min_chars: int,
    min_headings: int,
    forbidden_phrases: Sequence[str] = FORBIDDEN_COMPLETION_PHRASES,
) -> CompletionGateResult:
    stripped = markdown.strip()
    lowered = stripped.lower()
    heading_count = sum(1 for line in stripped.splitlines() if line.startswith("## "))
    matched_forbidden = [phrase for phrase in forbidden_phrases if phrase in lowered]
    reasons: list[str] = []
    if len(stripped) < min_chars:
        reasons.append(f"Final report is shorter than {min_chars} characters.")
    if heading_count < min_headings:
        reasons.append(f"Final report has fewer than {min_headings} second-level headings.")
    if matched_forbidden:
        reasons.append("Final report contains a handoff or clarification phrase.")
    return CompletionGateResult(
        passed=not reasons,
        char_count=len(stripped),
        heading_count=heading_count,
        forbidden_phrases=matched_forbidden,
        reasons=reasons,
    )


def build_custom_research_report(
    *,
    detail: RunDetail,
    budget_events: Iterable[dict[str, Any]],
    settings: Settings,
) -> ResearchReport:
    if detail.final_report is None:
        raise ValueError(f"Run {detail.id} does not have a final report.")

    gate = evaluate_completion_gate(
        detail.final_report.markdown,
        min_chars=settings.completion_gate_min_chars,
        min_headings=settings.completion_gate_min_headings,
    )
    warnings: list[str] = []
    if detail.asset_processing_errors:
        warnings.extend(detail.asset_processing_errors)
    runtime_warnings = detail.metadata.get("custom_response_warnings")
    if isinstance(runtime_warnings, list):
        warnings.extend(str(warning) for warning in runtime_warnings if warning)
    if detail.final_report.unsupported_claims:
        warnings.append(
            f"{len(detail.final_report.unsupported_claims)} claims were not fully supported."
        )
    if not gate.passed:
        warnings.extend(f"Completion gate: {reason}" for reason in gate.reasons)

    budget_event_list = list(budget_events)
    return ResearchReport(
        report_markdown=detail.final_report.markdown,
        sources=build_source_ledger(detail.source_registry_entries, detail=detail),
        run_id=detail.id,
        model_usage=summarize_model_usage(
            budget_event_list,
            estimated_cost=detail.estimated_cost_usd,
        ),
        tool_usage=summarize_tool_usage(
            budget_event_list,
            events=[event.model_dump(mode="json") for event in detail.events],
            settings=settings,
        ),
        warnings=warnings,
        trace_url=_trace_url(detail.metadata),
        completion_gate=gate,
    )


def build_source_ledger(
    registry_entries: Sequence[SourceRegistryEntry],
    *,
    detail: RunDetail,
) -> list[ResearchReportSource]:
    citation_by_url: dict[str, int] = {}
    citation_by_key: dict[str, int] = {}
    if detail.final_report is not None:
        for index, citation in enumerate(detail.final_report.citations, start=1):
            normalized = _safe_normalize(str(citation.source_url))
            citation_by_url.setdefault(normalized, index)
            if citation.citation_key:
                citation_by_key.setdefault(citation.citation_key, index)

    ledger: list[ResearchReportSource] = []
    seen_urls: set[str] = set()
    for entry in registry_entries:
        normalized = entry.normalized_url or _safe_normalize(entry.canonical_url or entry.url)
        if normalized in seen_urls:
            continue
        seen_urls.add(normalized)
        citation_number = citation_by_url.get(normalized)
        if citation_number is None and entry.citation_key is not None:
            citation_number = citation_by_key.get(entry.citation_key)
        if citation_number is None and entry.citation_key is not None:
            citation_number = _parse_citation_number(entry.citation_key)
        if citation_number is None:
            citation_number = _parse_citation_number(
                str(entry.metadata.get("citation_number") or "")
            )
        ledger.append(
            ResearchReportSource(
                citation_number=citation_number,
                title=entry.title,
                original_url=entry.url,
                normalized_url=normalized,
                provider=entry.provider,
                first_seen_agent=_first_seen_agent(entry.discovered_via),
                first_seen_at=entry.created_at,
                metadata={
                    "source_id": entry.source_id,
                    "citation_key": entry.citation_key,
                    "discovered_via": entry.discovered_via,
                    **dict(entry.metadata),
                },
            )
        )
    return sorted(
        ledger,
        key=lambda source: (
            source.citation_number is None,
            source.citation_number or 10**9,
            source.first_seen_at.isoformat() if source.first_seen_at else "",
        ),
    )


def summarize_model_usage(
    budget_events: Iterable[dict[str, Any]],
    *,
    estimated_cost: float,
) -> dict[str, Any]:
    total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": estimated_cost,
    }
    by_model: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    )
    by_phase: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    )
    for event in budget_events:
        if event.get("category") != "llm_tokens":
            continue
        metadata = dict(event.get("metadata") or {})
        model = str(metadata.get("model") or "unknown")
        phase = str(metadata.get("phase") or "unknown")
        values = {
            "input_tokens": int(metadata.get("input_tokens") or 0),
            "output_tokens": int(metadata.get("output_tokens") or 0),
            "reasoning_tokens": int(metadata.get("reasoning_tokens") or 0),
            "total_tokens": int(event.get("delta") or metadata.get("total_tokens") or 0),
            "estimated_cost_usd": float(metadata.get("estimated_cost_usd") or 0.0),
        }
        for key, value in values.items():
            total[key] = total[key] + value
            by_model[model][key] = by_model[model][key] + value
            by_phase[phase][key] = by_phase[phase][key] + value
    return {
        **total,
        "by_model": dict(by_model),
        "by_phase": dict(by_phase),
    }


def summarize_tool_usage(
    budget_events: Iterable[dict[str, Any]],
    *,
    events: Iterable[dict[str, Any]],
    settings: Settings,
) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    search_calls = 0
    fetch_calls = 0
    embedding_calls = 0
    for event in budget_events:
        category = str(event.get("category") or "unknown")
        if category == "llm_tokens":
            continue
        delta = int(event.get("delta") or 0)
        categories[category] += delta
        metadata = dict(event.get("metadata") or {})
        provider = str(metadata.get("provider") or "").strip()
        if provider:
            providers[provider] += delta
        for provider_name in metadata.get("result_providers") or []:
            providers[str(provider_name)] += 1
        if "search" in category:
            search_calls += delta
        if "fetch" in category:
            fetch_calls += delta
        if "embed" in category:
            embedding_calls += delta

    event_counts = Counter(str(event.get("event_type") or "unknown") for event in events)
    return {
        "search_calls": search_calls,
        "fetch_calls": fetch_calls,
        "embedding_calls": embedding_calls,
        "categories": dict(categories),
        "providers": dict(providers),
        "event_counts": dict(event_counts),
        "contract_tools": contract_tool_names(settings),
    }


def _first_seen_agent(discovered_via: str) -> str:
    lowered = discovered_via.lower()
    if "planning" in lowered:
        return "planner-agent"
    if "claim_repair" in lowered or "citation" in lowered:
        return "orchestrator"
    if "user_input" in lowered:
        return "user"
    return "researcher-agent"


def _safe_normalize(url: str) -> str:
    try:
        return normalize_url(url)
    except Exception:
        return url


def _parse_citation_number(value: str) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _trace_url(metadata: dict[str, Any]) -> str | None:
    for key in ("trace_url", "langsmith_trace_url", "langfuse_trace_url", "phoenix_trace_url"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
