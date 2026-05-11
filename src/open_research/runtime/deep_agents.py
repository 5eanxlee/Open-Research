from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html import unescape
from importlib import resources
from importlib.util import find_spec
from typing import Any

from jinja2 import Environment, StrictUndefined

from open_research.agents.middleware import (
    EmptyContentFixMiddleware,
    SearchToolCallLimitMiddleware,
    SubAgentTraceMiddleware,
    TodoSanitizationMiddleware,
    TodoSyncMiddleware,
    ToolNameSanitizationMiddleware,
)
from open_research.core.citations import (
    CitationCandidate,
    audit_citation_candidates,
    build_citation_key,
)
from open_research.core.config import Settings
from open_research.core.domain import (
    BudgetPolicy,
    CitationAuditDecision,
    CitationRecord,
    CitationSupportLabel,
    ClaimVerification,
    CompletionGateResult,
    DraftReport,
    ExecutionMode,
    FinalReport,
    PlanApprovalStatus,
    PlanningArtifact,
    PlanningDiscoveryRecord,
    PlanningStage,
    PlanPreview,
    RecommendedBudget,
    ReportSection,
    ResearchOptions,
    ResearchPlan,
    ResearchReport,
    ResearchStreamPlan,
    RunStatus,
    SearchResult,
    TaskStatus,
    resolve_model_config,
)
from open_research.core.utils import (
    chunk_text,
    clean_text,
    dedupe_preserve_order,
    derive_conversation_topic,
    derive_report_title,
    extract_sentences,
    normalize_url,
    tokenize,
)
from open_research.integrations.providers import ProviderError, SearchProvider
from open_research.runtime.custom_responses import (
    build_custom_research_report,
    create_run_request_from_research_options,
    evaluate_completion_gate,
)
from open_research.runtime.pipeline import (
    _absence_claim_contradicted_by_passage,
    _claim_repair_should_skip_search,
)
from open_research.runtime.tool_registry import FETCH_TOOL, PAPER_SEARCH_TOOL
from open_research.storage.artifacts import ArtifactPayload
from open_research.tools import AdvancedWebSearchTool, PaperSearchTool
from open_research.tools import think as record_thought

DEEP_RESEARCH_AGENT_TOPOLOGY: tuple[str, ...] = (
    "research-orchestrator",
    "planner-agent",
    "researcher-agent",
    "scholar-agent",
    "source-auditor-agent",
    "critic-agent",
    "synthesis-agent",
    "citation-agent",
)

PLANNER_CONTRACT_VERSION = "deepagents-planner-v1"
GROUNDING_CONTRACT_VERSION = "deepagents-grounding-v2"


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
            "scholar-agent": "prompts/custom_responses/scholar.j2",
            "source-auditor-agent": "prompts/custom_responses/source_auditor.j2",
            "critic-agent": "prompts/custom_responses/critic.j2",
            "synthesis-agent": "prompts/custom_responses/synthesis.j2",
            "citation-agent": "prompts/custom_responses/citation.j2",
        },
        middleware=(
            "TodoListMiddleware",
            "FilesystemMiddleware(StateBackend)",
            "SubAgentMiddleware",
            "EmptyContentFixMiddleware",
            "ToolNameSanitizationMiddleware",
            "TodoSanitizationMiddleware",
            "TodoSyncMiddleware",
            "SubAgentTraceMiddleware",
            "ModelRetryMiddleware",
            "ToolRetryMiddleware",
            "ToolCallLimitMiddleware",
            "SearchToolCallLimitMiddleware",
        ),
        tools=tuple(available_deep_agent_tool_names(settings)),
        native_openai_web_search=settings.resolved_search_backend == "openai",
    )


def inspect_research_deep_agent(settings: Settings) -> CustomResponsesDeepAgentStatus:
    status = inspect_custom_responses_deep_agent(settings)
    backend = settings.resolved_research_runtime_backend
    return CustomResponsesDeepAgentStatus(
        available=not status.missing_dependencies
        and backend in {"deepagents", "hybrid"}
        and settings.resolved_llm_backend == "openai"
        and settings.openai_api_key is not None,
        backend=backend,
        missing_dependencies=status.missing_dependencies,
        model=status.model,
        search_backend=status.search_backend,
        prompt_templates=status.prompt_templates,
        middleware=status.middleware,
        tools=status.tools,
        native_openai_web_search=status.native_openai_web_search,
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
        "deepagents_agent_topology": list(DEEP_RESEARCH_AGENT_TOPOLOGY),
        "deepagents_prompt_templates": status.prompt_templates,
        "planner_contract_version": PLANNER_CONTRACT_VERSION,
        "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
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
        await _finalize_deep_agent_run(
            runtime=runtime,
            run_id=run_id,
            question=prompt,
            markdown=markdown,
            gate=gate,
            final_report_discovered_via="custom_responses.deepagents.final_report",
            terminal_reason="custom_responses_deepagent_completed",
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


async def run_deep_agent_research_for_existing_run(
    *,
    runtime: Any,
    run_id: str,
    question: str,
    budget: BudgetPolicy,
) -> None:
    status = inspect_research_deep_agent(runtime.settings)
    if not status.available:
        reason = (
            "DeepAgents research runtime is unavailable. "
            f"backend={status.backend}, llm={runtime.settings.resolved_llm_backend}, "
            f"missing={list(status.missing_dependencies)}"
        )
        raise ValueError(reason)

    state = await runtime.store.get_run_execution_state(run_id)
    model_config = resolve_model_config(
        state.metadata.get("model_config") if state is not None else None,
        defaults=runtime.default_model_config(),
    )
    source_selection = runtime.resolve_source_selection(
        (state.metadata.get("source_selection") if state is not None else None) or None
    )
    await runtime.store.update_run_metadata(
        run_id,
        {
            "runtime_executor": "deepagents",
            "research_runtime_backend": status.backend,
            "deepagents_agent_topology": list(DEEP_RESEARCH_AGENT_TOPOLOGY),
            "deepagents_prompt_templates": status.prompt_templates,
            "planner_contract_version": PLANNER_CONTRACT_VERSION,
            "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
        },
    )
    await runtime.store.update_run_status(run_id, RunStatus.RESEARCHING)
    await runtime.events.publish(
        run_id,
        "deepagents.runtime.started",
        {
            "model": model_config.lead_model,
            "planner_model": model_config.planner_model,
            "researcher_model": model_config.worker_model,
            "search_backend": runtime.settings.resolved_search_backend,
            "fetch_backend": runtime.settings.resolved_fetch_backend,
            "native_openai_web_search": status.native_openai_web_search,
            "agent_topology": list(DEEP_RESEARCH_AGENT_TOPOLOGY),
            "planner_contract_version": PLANNER_CONTRACT_VERSION,
            "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
        },
    )
    markdown, gate = await _invoke_deep_agent_until_complete(
        runtime=runtime,
        prompt=question,
        run_id=run_id,
        budget=budget,
        lead_model=model_config.lead_model,
        planner_model=model_config.planner_model,
        researcher_model=model_config.worker_model,
        source_selection=source_selection,
    )
    await _finalize_deep_agent_run(
        runtime=runtime,
        run_id=run_id,
        question=question,
        markdown=markdown,
        gate=gate,
        final_report_discovered_via="deepagents.final_report",
        terminal_reason="deepagents_completed",
    )


async def _finalize_deep_agent_run(
    *,
    runtime: Any,
    run_id: str,
    question: str,
    markdown: str,
    gate: CompletionGateResult,
    final_report_discovered_via: str,
    terminal_reason: str,
) -> FinalReport:
    await _persist_deep_agent_artifact(
        runtime=runtime,
        run_id=run_id,
        kind="deepagents-final-report",
        content=markdown,
        metadata={"stage": "final_report"},
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
                "discovered_via": final_report_discovered_via,
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
    if runtime.settings.deepagents_grounding_enabled:
        grounded = await _ground_deep_agent_report(
            runtime=runtime,
            run_id=run_id,
            question=question,
            markdown=markdown,
            gate=gate,
            citation_warnings=citation_warnings,
            terminal_reason=terminal_reason,
        )
        if grounded is not None:
            return grounded
    candidates = [
        CitationCandidate(
            section_title="Sources",
            ordinal=index,
            claim=citation.claim,
            citation=citation,
        )
        for index, citation in enumerate(citations, start=1)
    ]
    audit_result = audit_citation_candidates(
        run_id=run_id,
        candidates=candidates,
        registry_entries=registry_entries,
    )
    await runtime.store.replace_citation_audits(run_id, audit_result.audits)
    for audit in audit_result.audits:
        if audit.decision != CitationAuditDecision.REMOVED:
            continue
        await runtime.events.publish(
            run_id,
            "citation.removed",
            {
                "section_title": audit.section_title,
                "ordinal": audit.ordinal,
                "claim": audit.claim,
                "reasons": [reason.value for reason in audit.reasons],
            },
        )
    kept_citations = [
        candidate.citation for candidate in audit_result.kept if candidate.citation is not None
    ]
    unsupported_claims = [
        *([] if gate.passed else list(gate.reasons)),
        *citation_warnings,
        *[candidate.claim for candidate in audit_result.removed if candidate.citation is not None],
    ]
    final_report = FinalReport(
        markdown=markdown,
        citations=kept_citations,
        unsupported_claims=list(dict.fromkeys(unsupported_claims)),
        confidence=0.86 if gate.passed and not audit_result.removed else 0.55,
        title=derive_report_title(question),
        conversation_topic=derive_report_title(question),
    )
    await runtime.events.publish(
        run_id,
        "citation.audit.completed",
        {
            "kept": len(audit_result.kept),
            "removed": len(audit_result.removed),
            "runtime_executor": "deepagents",
        },
    )
    await runtime.events.publish(
        run_id,
        "completion_gate.completed",
        gate.model_dump(mode="json"),
    )
    await runtime.store.update_run_status(
        run_id,
        RunStatus.COMPLETED,
        final_report=final_report,
        terminal_reason=terminal_reason,
    )
    await runtime.events.publish(
        run_id,
        "report.completed",
        {
            "citation_count": len(kept_citations),
            "unsupported_claim_count": len(final_report.unsupported_claims),
            "runtime_executor": "deepagents",
        },
    )
    return final_report


async def _ground_deep_agent_report(
    *,
    runtime: Any,
    run_id: str,
    question: str,
    markdown: str,
    gate: CompletionGateResult,
    citation_warnings: Sequence[str],
    terminal_reason: str,
) -> FinalReport | None:
    await runtime.events.publish(
        run_id,
        "deepagents.grounding.started",
        {
            "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
            "completion_gate_passed": gate.passed,
        },
    )
    try:
        passages = await runtime.store.list_passages(run_id)
        if not passages:
            raise RuntimeError("No fetched source passages are available for claim grounding.")
        final_report = await _ground_deep_agent_markdown_preserving_structure(
            runtime=runtime,
            run_id=run_id,
            question=question,
            markdown=markdown,
            gate=gate,
            citation_warnings=citation_warnings,
        )
        if citation_warnings:
            final_report = final_report.model_copy(
                update={
                    "unsupported_claims": list(
                        dict.fromkeys([*final_report.unsupported_claims, *citation_warnings])
                    )
                }
            )
            await runtime.store.update_run_status(
                run_id,
                RunStatus.COMPLETED,
                final_report=final_report,
                terminal_reason=terminal_reason,
            )
        else:
            await runtime.store.update_run_status(
                run_id,
                RunStatus.COMPLETED,
                final_report=final_report,
                terminal_reason=terminal_reason,
            )
        await runtime.store.update_run_metadata(
            run_id,
            {
                "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
                "deepagents_grounding_preserved_markdown": True,
            },
        )
        await runtime.events.publish(
            run_id,
            "completion_gate.completed",
            gate.model_dump(mode="json"),
        )
        await runtime.events.publish(
            run_id,
            "deepagents.grounding.completed",
            {
                "citation_count": len(final_report.citations),
                "unsupported_claim_count": len(final_report.unsupported_claims),
                "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
                "markdown_preserved": True,
            },
        )
        return final_report
    except Exception as exc:
        await runtime.events.publish(
            run_id,
            "deepagents.grounding.failed",
            {
                "error": str(exc),
                "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
            },
        )
        if runtime.settings.deepagents_grounding_strict:
            raise
        return None


async def _ground_deep_agent_markdown_preserving_structure(
    *,
    runtime: Any,
    run_id: str,
    question: str,
    markdown: str,
    gate: CompletionGateResult,
    citation_warnings: Sequence[str],
) -> FinalReport:
    orchestrator = runtime.orchestrator
    agent_config = await orchestrator._get_agent_config(run_id)
    model_config = await orchestrator._get_model_config(run_id)
    await orchestrator._ensure_run_active(run_id)
    await runtime.store.update_run_status(run_id, RunStatus.GROUNDING)

    draft = _draft_report_from_markdown(question=question, markdown=markdown)
    claim_rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    citation_candidates: list[CitationCandidate] = []
    verified_claims = 0
    max_verified_claims = max(1, runtime.settings.grounding_max_claims_per_run)

    for section in draft.sections:
        for ordinal, claim in enumerate(section.claims, start=1):
            await orchestrator._ensure_run_active(run_id)
            claim_key = (section.title, ordinal)
            if verified_claims >= max_verified_claims:
                claim_rows_by_key[claim_key] = {
                    "section_title": section.title,
                    "ordinal": ordinal,
                    "claim_text": claim,
                    "support_label": CitationSupportLabel.UNSUPPORTED.value,
                    "confidence": 0.0,
                }
                citation_candidates.append(
                    CitationCandidate(
                        section_title=section.title,
                        ordinal=ordinal,
                        claim=claim,
                        citation=None,
                    )
                )
                await runtime.events.publish(
                    run_id,
                    "citation.verification.skipped",
                    {
                        "section_title": section.title,
                        "claim": claim,
                        "reason": "grounding_max_claims_per_run",
                        "limit": max_verified_claims,
                        "runtime_executor": "deepagents",
                    },
                )
                continue

            verified_claims += 1
            candidates = await orchestrator._retrieve_supporting_passages(
                run_id=run_id,
                claim=claim,
            )
            verification_result = await orchestrator.verifier.verify(
                claim=claim,
                candidates=candidates,
                agent_config=agent_config,
                model_config=model_config,
            )
            verification = verification_result.value
            await orchestrator._record_llm_usage(
                run_id=run_id,
                phase="deepagents_claim_verify",
                model=model_config.verifier_model,
                usage=verification_result.usage,
                metadata={
                    "section_title": section.title,
                    **(verification_result.metadata or {}),
                },
            )

            repair_attempts = 0
            if (
                verification.support_label == CitationSupportLabel.UNSUPPORTED
                and _claim_repair_should_skip_search(claim)
            ):
                await runtime.events.publish(
                    run_id,
                    "claim.repair.skipped",
                    {
                        "section_title": section.title,
                        "claim": claim,
                        "reason": "uncertainty_or_absence_claim",
                        "runtime_executor": "deepagents",
                    },
                )
            else:
                while (
                    verification.support_label == CitationSupportLabel.UNSUPPORTED
                    and repair_attempts < runtime.settings.max_claim_repairs
                ):
                    repair_attempts += 1
                    new_passages = await orchestrator.worker.collect_supporting_passages(
                        run_id=run_id,
                        claim=claim,
                        section_title=section.title,
                    )
                    if not new_passages:
                        break
                    candidates = await orchestrator._retrieve_supporting_passages(
                        run_id=run_id,
                        claim=claim,
                    )
                    verification_result = await orchestrator.verifier.verify(
                        claim=claim,
                        candidates=candidates,
                        agent_config=agent_config,
                        model_config=model_config,
                    )
                    verification = verification_result.value
                    await orchestrator._record_llm_usage(
                        run_id=run_id,
                        phase="deepagents_claim_reverify",
                        model=model_config.verifier_model,
                        usage=verification_result.usage,
                        metadata={
                            "section_title": section.title,
                            "repair_attempt": repair_attempts,
                            **(verification_result.metadata or {}),
                        },
                    )

            await runtime.events.publish(
                run_id,
                "citation.verified",
                {
                    "section_title": section.title,
                    "claim": claim,
                    "support_label": verification.support_label.value,
                    "repair_attempts": repair_attempts,
                    "runtime_executor": "deepagents",
                    "prompt_template_version": (verification_result.metadata or {}).get(
                        "prompt_template_version"
                    ),
                },
            )
            claim_rows_by_key[claim_key] = {
                "section_title": section.title,
                "ordinal": ordinal,
                "claim_text": claim,
                "support_label": verification.support_label.value,
                "confidence": verification.confidence,
            }

            chosen = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.source_id == verification.selected_source_id
                    and candidate.passage_index == verification.selected_passage_index
                ),
                candidates[0] if candidates else None,
            )
            if (
                chosen is not None
                and verification.support_label != CitationSupportLabel.UNSUPPORTED
                and _absence_claim_contradicted_by_passage(claim, chosen.text)
            ):
                verification = ClaimVerification(
                    support_label=CitationSupportLabel.CONTRADICTED,
                    reason=(
                        "The selected passage contains concrete benchmark, metric, or "
                        "ablation evidence that contradicts the claim's absence-of-evidence "
                        "wording."
                    ),
                    selected_source_id=chosen.source_id,
                    selected_passage_index=chosen.passage_index,
                    quote=verification.quote or chosen.text[:240],
                    confidence=max(verification.confidence, 0.85),
                )
                claim_rows_by_key[claim_key]["support_label"] = verification.support_label.value
                claim_rows_by_key[claim_key]["confidence"] = verification.confidence
                await runtime.events.publish(
                    run_id,
                    "citation.contradicted",
                    {
                        "section_title": section.title,
                        "claim": claim,
                        "source_id": chosen.source_id,
                        "passage_index": chosen.passage_index,
                        "reason": "absence_claim_counterevidence",
                        "runtime_executor": "deepagents",
                    },
                )
            if (
                verification.support_label
                in {CitationSupportLabel.UNSUPPORTED, CitationSupportLabel.CONTRADICTED}
                or chosen is None
            ):
                citation_candidates.append(
                    CitationCandidate(
                        section_title=section.title,
                        ordinal=ordinal,
                        claim=claim,
                        citation=None,
                    )
                )
                continue

            citation_candidates.append(
                CitationCandidate(
                    section_title=section.title,
                    ordinal=ordinal,
                    claim=claim,
                    citation=CitationRecord(
                        claim=claim,
                        support_label=verification.support_label,
                        source_id=chosen.source_id,
                        source_title=chosen.source_title,
                        source_url=chosen.source_url,
                        citation_key=build_citation_key(
                            chosen.source_title,
                            str(chosen.source_url),
                        ),
                        passage_index=chosen.passage_index,
                        quote=verification.quote or chosen.text[:240],
                        confidence=verification.confidence,
                    ),
                )
            )

    registry_entries = await runtime.store.list_source_registry_entries(run_id)
    audit_result = audit_citation_candidates(
        run_id=run_id,
        candidates=citation_candidates,
        registry_entries=registry_entries,
    )
    unsupported_claims: list[str] = []
    citations: list[CitationRecord] = []
    citation_rows: list[dict[str, Any]] = []
    confidence_values: list[float] = []
    source_registry_annotations: list[dict[str, Any]] = []

    for audit in audit_result.audits:
        if audit.decision != CitationAuditDecision.REMOVED:
            continue
        await runtime.events.publish(
            run_id,
            "citation.removed",
            {
                "section_title": audit.section_title,
                "ordinal": audit.ordinal,
                "claim": audit.claim,
                "reasons": [reason.value for reason in audit.reasons],
                "runtime_executor": "deepagents",
            },
        )
        claim_key = (audit.section_title, audit.ordinal)
        claim_row = claim_rows_by_key[claim_key]
        claim_row["support_label"] = CitationSupportLabel.UNSUPPORTED.value
        unsupported_claims.append(audit.claim)
        source_registry_annotations.append(
            {
                "source_id": audit.source_id,
                "normalized_url": audit.normalized_url,
                "citation_key": audit.citation_key,
                "metadata": {
                    "survived_final_citation": False,
                    "removed_in_audit": True,
                    "audit_reasons": [reason.value for reason in audit.reasons],
                    "audit_removed_claim": audit.claim,
                },
            }
        )

    for candidate in audit_result.kept:
        if candidate.citation is None:
            unsupported_claims.append(candidate.claim)
            continue
        citation = candidate.citation
        citations.append(citation)
        citation_rows.append(
            {
                "section_title": candidate.section_title,
                "ordinal": candidate.ordinal,
                "source_id": citation.source_id,
                "passage_index": citation.passage_index,
                "quote": citation.quote,
                "support_label": citation.support_label.value,
                "confidence": citation.confidence,
            }
        )
        confidence_values.append(citation.confidence)
        source_registry_annotations.append(
            {
                "source_id": citation.source_id,
                "normalized_url": normalize_url(str(citation.source_url)),
                "citation_key": citation.citation_key,
                "metadata": {
                    "survived_final_citation": True,
                    "removed_in_audit": False,
                    "final_citation_section": candidate.section_title,
                    "final_citation_ordinal": candidate.ordinal,
                    "final_citation_support_label": citation.support_label.value,
                },
            }
        )

    unsupported_claims = dedupe_preserve_order(
        [
            *unsupported_claims,
            *([] if gate.passed else list(gate.reasons)),
            *citation_warnings,
        ]
    )
    await runtime.store.replace_claims_and_citations(
        run_id,
        list(claim_rows_by_key.values()),
        citation_rows,
    )
    await runtime.store.annotate_source_registry_entries(run_id, source_registry_annotations)
    await runtime.store.replace_citation_audits(run_id, audit_result.audits)
    await runtime.events.publish(
        run_id,
        "citation.audit.completed",
        {
            "kept": len(audit_result.kept),
            "removed": len(audit_result.removed),
            "runtime_executor": "deepagents",
            "markdown_preserved": True,
        },
    )
    await runtime.events.publish(
        run_id,
        "report.sanitized",
        {
            "citation_count": len(citations),
            "removed_citations": len(audit_result.removed),
            "runtime_executor": "deepagents",
            "markdown_preserved": True,
        },
    )

    confidence = round(sum(confidence_values) / max(len(confidence_values), 1), 3)
    if not confidence_values and not unsupported_claims and gate.passed:
        confidence = 0.75
    return FinalReport(
        markdown=markdown,
        citations=citations,
        unsupported_claims=unsupported_claims,
        confidence=confidence,
        title=draft.title or derive_report_title(question),
        conversation_topic=draft.conversation_topic or derive_conversation_topic(question),
    )


def _draft_report_from_markdown(*, question: str, markdown: str) -> DraftReport:
    title = derive_report_title(question)
    lines = markdown.splitlines()
    sections: list[ReportSection] = []
    current_title: str | None = None
    current_lines: list[str] = []
    preface: list[str] = []

    for line in lines:
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            if current_title is not None:
                sections.append(_section_from_markdown(current_title, "\n".join(current_lines)))
            elif current_lines:
                preface.extend(current_lines)
            current_title = clean_text(heading.group(1))
            current_lines = []
            continue
        title_match = re.match(r"^#\s+(.+?)\s*$", line)
        if title_match and title == derive_report_title(question):
            title = clean_text(title_match.group(1)) or title
            continue
        current_lines.append(line)

    if current_title is not None:
        sections.append(_section_from_markdown(current_title, "\n".join(current_lines)))
    elif current_lines:
        preface.extend(current_lines)

    if not sections:
        sections = [_section_from_markdown("Findings", markdown)]
    executive_summary = clean_text("\n".join(preface)) or sections[0].overview
    open_questions = [
        sentence
        for section in sections
        for sentence in section.claims
        if sentence.rstrip().endswith("?")
    ][:6]
    return DraftReport(
        title=title,
        conversation_topic=derive_conversation_topic(question),
        executive_summary=executive_summary[:4000],
        sections=sections,
        open_questions=open_questions,
    )


def _section_from_markdown(title: str, body: str) -> ReportSection:
    cleaned_body = re.sub(r"(?im)^#{1,6}\s+sources\s*$[\s\S]*", "", body).strip()
    overview = clean_text(cleaned_body)[:4000] or "No section overview was available."
    sentences = [
        sentence
        for sentence in extract_sentences(cleaned_body, max_sentences=8)
        if len(sentence.split()) >= 6 and not sentence.lower().startswith("sources")
    ]
    claims = dedupe_preserve_order(sentences)[:5] or [overview[:500]]
    return ReportSection(title=title, overview=overview, claims=claims)


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
        runtime=runtime,
        settings=runtime.settings,
        search_provider=runtime.orchestrator.worker.search_provider,
        run_id=run_id,
        question=prompt,
        record_budget_event=runtime.store.record_budget_event,
        publish_event=runtime.events.publish,
        register_source_registry_entries=runtime.store.register_source_registry_entries,
        budget=budget,
        lead_model=lead_model,
        planner_model=planner_model,
        researcher_model=researcher_model,
        max_results=min(5, budget.max_results_per_query),
        source_selection=source_selection,
    )
    initial_prompt = await _compose_deep_agent_prompt(runtime, run_id, prompt)
    state: dict[str, Any] = {"messages": [{"role": "user", "content": initial_prompt}]}
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


async def _compose_deep_agent_prompt(runtime: Any, run_id: str, prompt: str) -> str:
    state = await runtime.store.get_run_execution_state(run_id)
    if state is None:
        return prompt
    metadata = dict(state.metadata or {})
    if metadata.get("approval_status") != PlanApprovalStatus.APPROVED.value:
        return prompt
    preview_raw = metadata.get("plan_preview")
    if not preview_raw:
        return prompt
    try:
        preview = PlanPreview.model_validate(preview_raw)
    except Exception:
        return prompt
    return (
        f"{prompt}\n\n"
        "Approved research plan context. Treat this plan as fixed execution guidance; "
        "do not replace it unless tool evidence proves it is impossible.\n\n"
        f"{preview.plan.model_dump_json(indent=2)}"
    )


def _build_deep_agent(
    *,
    runtime: Any,
    settings: Settings,
    search_provider: SearchProvider,
    run_id: str,
    question: str,
    record_budget_event: Any,
    publish_event: Any,
    register_source_registry_entries: Any,
    budget: BudgetPolicy,
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
    worker = runtime.orchestrator.worker

    async def sync_todos(agent_role: str, todos: list[dict[str, str]]) -> None:
        await _sync_deep_agent_todos(
            runtime=runtime,
            run_id=run_id,
            agent_role=agent_role,
            todos=todos,
            model=_model_for_agent_role(
                agent_role,
                lead_model=lead_model,
                planner_model=planner_model,
                researcher_model=researcher_model,
            ),
        )

    async def publish_agent_lifecycle(event_type: str, payload: dict[str, Any]) -> None:
        await _publish_tool_event(
            publish_event=publish_event,
            run_id=run_id,
            event_type=event_type,
            payload=payload,
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
    async def think(thought: str) -> str:
        """Record a concise private research or synthesis thought."""
        await _publish_tool_event(
            publish_event=publish_event,
            run_id=run_id,
            event_type="deepagents.think.recorded",
            payload={"thought": thought[:2000], "thought_hash": _hash_text(thought)},
        )
        return record_thought(thought)

    @tool("fetch_page")
    async def fetch_page(url: str) -> str:
        """Fetch, normalize, store, chunk, and register a discovered source URL."""
        await _publish_tool_event(
            publish_event=publish_event,
            run_id=run_id,
            event_type="deepagents.tool.started",
            payload={"tool": "fetch_page", "url": url},
        )
        await record_budget_event(
            run_id,
            "source_fetch",
            1,
            {"provider": worker.fetch_provider.provider_name, "tool": "fetch_page", "url": url},
        )
        try:
            _, allowed_fetch = await worker._get_source_selection(run_id)
            document = await worker._fetch(
                run_id=run_id,
                url=url,
                allowed_fetch=allowed_fetch,
            )
            document = worker._annotate_document_with_trust(document)
            document, artifact_payloads = worker._extract_artifact_payloads(document)
            source_id, is_new = await runtime.store.save_source(run_id, None, document)
            await worker._register_source_document(
                run_id=run_id,
                source_id=source_id,
                document=document,
                discovered_via="deepagents.fetch_page",
            )
            if is_new:
                artifacts = await worker._persist_source_artifacts(
                    run_id=run_id,
                    source_id=source_id,
                    document=document,
                    artifact_payloads=artifact_payloads,
                )
                if artifacts:
                    await runtime.store.update_source_metadata(source_id, {"artifacts": artifacts})
                passages = [
                    {
                        "passage_index": index,
                        "text": chunk["text"],
                        "start_offset": chunk["start_offset"],
                        "end_offset": chunk["end_offset"],
                        "token_count": len(tokenize(chunk["text"])),
                    }
                    for index, chunk in enumerate(chunk_text(document.content))
                ]
                passages = await worker._embed_passages(
                    run_id=run_id,
                    stream_id=None,
                    source_id=source_id,
                    passages=passages,
                )
                await runtime.store.save_passages(run_id, source_id, passages)
            payload = {
                "tool": "fetch_page",
                "source_id": source_id,
                "url": str(document.canonical_url),
                "title": document.title,
                "is_new": is_new,
                "content_chars": len(document.content),
            }
            await _publish_tool_event(
                publish_event=publish_event,
                run_id=run_id,
                event_type="source.fetched",
                payload=payload,
            )
            await _publish_tool_event(
                publish_event=publish_event,
                run_id=run_id,
                event_type="deepagents.tool.completed",
                payload=payload,
            )
            return _format_fetched_document_for_agent(source_id=source_id, document=document)
        except Exception as exc:
            await _publish_tool_event(
                publish_event=publish_event,
                run_id=run_id,
                event_type="deepagents.tool.failed",
                payload={"tool": "fetch_page", "url": url, "error": str(exc)},
            )
            return f"Fetch failed for {url}: {exc}"

    @tool("save_research_artifact")
    async def save_research_artifact(name: str, content: str, kind: str = "deepagents-note") -> str:
        """Persist an intermediate research artifact for backend/UI inspection."""
        safe_kind = re.sub(r"[^a-zA-Z0-9_-]+", "-", kind or name).strip("-").lower()
        saved = await _persist_deep_agent_artifact(
            runtime=runtime,
            run_id=run_id,
            kind=safe_kind[:80] or "deepagents-note",
            content=content,
            metadata={"name": name, "tool": "save_research_artifact"},
        )
        if saved is None:
            return "Artifact storage is disabled or content was empty."
        if safe_kind.startswith("deepagents-planner-plan"):
            await _persist_deep_agent_plan(
                runtime=runtime,
                run_id=run_id,
                question=question,
                content=content,
                budget=budget,
                planner_model=planner_model,
                researcher_model=researcher_model,
                source_selection=source_selection,
            )
        if _artifact_kind_is_note(safe_kind):
            await _persist_deep_agent_note(
                runtime=runtime,
                run_id=run_id,
                kind=safe_kind,
                name=name,
                content=content,
                model=researcher_model,
            )
        return f"Artifact saved: {saved['uri']}"

    @tool("source_audit_tool")
    async def source_audit_tool(observation: str) -> str:
        """Record source quality, provenance, conflict, or diversity audit observations."""
        await _publish_tool_event(
            publish_event=publish_event,
            run_id=run_id,
            event_type="deepagents.source_audit.recorded",
            payload={
                "observation": observation[:3000],
                "observation_hash": _hash_text(observation),
            },
        )
        return "Source audit observation recorded."

    @tool("citation_reconciliation_tool")
    async def citation_reconciliation_tool(markdown: str) -> str:
        """Compare a draft Sources section against URLs observed by DeepAgents tools."""
        source_records = _extract_sources_from_markdown(markdown)
        registry_entries = await runtime.store.list_source_registry_entries(run_id)
        warnings = _citation_reconciliation_warnings(
            final_sources=source_records,
            registry_entries=registry_entries,
        )
        await _publish_tool_event(
            publish_event=publish_event,
            run_id=run_id,
            event_type="deepagents.citation.reconciled",
            payload={
                "source_count": len(source_records),
                "warning_count": len(warnings),
                "warnings": warnings,
            },
        )
        if not warnings:
            return f"Citation reconciliation passed for {len(source_records)} sources."
        return "Citation reconciliation warnings:\n" + "\n".join(
            f"- {warning}" for warning in warnings
        )

    orchestrator_web_search = make_advanced_web_search_tool("orchestrator")
    planner_web_search = make_advanced_web_search_tool("planner-agent")
    researcher_web_search = make_advanced_web_search_tool("researcher-agent")
    scholar_web_search = make_advanced_web_search_tool("scholar-agent")
    auditor_web_search = make_advanced_web_search_tool("source-auditor-agent")
    orchestrator_paper_search = make_paper_search_tool("orchestrator")
    researcher_paper_search = make_paper_search_tool("researcher-agent")
    scholar_paper_search = make_paper_search_tool("scholar-agent")

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
    critic = ChatOpenAI(
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
    tool_list = [
        tool
        for tool in [
            orchestrator_web_search,
            orchestrator_paper_search,
            fetch_page,
            think,
            save_research_artifact,
            source_audit_tool,
            citation_reconciliation_tool,
        ]
        if tool is not None
    ]
    planner_tools = [planner_web_search, think, save_research_artifact]
    researcher_tools = [
        tool
        for tool in [
            researcher_web_search,
            researcher_paper_search,
            fetch_page,
            think,
            save_research_artifact,
        ]
        if tool is not None
    ]
    scholar_tools = [
        tool
        for tool in [
            scholar_paper_search,
            scholar_web_search,
            fetch_page,
            think,
            save_research_artifact,
        ]
        if tool is not None
    ]
    auditor_tools = [
        tool
        for tool in [
            auditor_web_search,
            fetch_page,
            source_audit_tool,
            citation_reconciliation_tool,
            think,
        ]
        if tool is not None
    ]
    critic_tools = [
        tool
        for tool in [think, source_audit_tool, citation_reconciliation_tool, save_research_artifact]
        if tool is not None
    ]
    synthesis_tools = [
        tool
        for tool in [think, save_research_artifact, citation_reconciliation_tool]
        if tool is not None
    ]
    citation_tools = [
        tool
        for tool in [think, citation_reconciliation_tool, source_audit_tool]
        if tool is not None
    ]
    repair_tools = [
        *tool_list,
        *planner_tools,
        *researcher_tools,
        *scholar_tools,
        *auditor_tools,
        *critic_tools,
        *synthesis_tools,
        *citation_tools,
        *TodoListMiddleware().tools,
        *FilesystemMiddleware(backend=StateBackend()).tools,
    ]
    reliability_middleware = _build_reliability_middleware(
        settings=settings,
        repair_tools=repair_tools,
        agent_role="research-orchestrator",
        todo_sync=sync_todos,
    )
    planner_middleware = _build_subagent_middleware(
        settings=settings,
        repair_tools=repair_tools,
        agent_role="planner-agent",
        todo_sync=sync_todos,
        search_run_limit=max(settings.planner_min_discovery_queries + 8, 18),
    )
    researcher_middleware = _build_subagent_middleware(
        settings=settings,
        repair_tools=repair_tools,
        agent_role="researcher-agent",
        todo_sync=sync_todos,
        search_run_limit=8,
    )
    scholar_middleware = _build_subagent_middleware(
        settings=settings,
        repair_tools=repair_tools,
        agent_role="scholar-agent",
        todo_sync=sync_todos,
        search_run_limit=8,
    )
    lightweight_middleware = _build_reliability_middleware(
        settings=settings,
        repair_tools=repair_tools,
        agent_role="deepagents-specialist",
        todo_sync=sync_todos,
    )
    context = {
        "current_datetime": datetime.now(UTC).isoformat(),
        "agent_topology": list(DEEP_RESEARCH_AGENT_TOPOLOGY),
        "max_research_batches": settings.deepagents_max_research_batches,
        "require_critic_pass": settings.deepagents_require_critic_pass,
        "require_source_audit_pass": settings.deepagents_require_source_audit_pass,
        "require_citation_pass": settings.deepagents_require_citation_pass,
        "tools": [
            {"name": name}
            for name in available_deep_agent_tool_names(
                settings,
                source_selection=source_selection,
            )
        ],
    }
    middleware: list[Any] = [
        SubAgentTraceMiddleware(
            parent_role="research-orchestrator",
            on_event=publish_agent_lifecycle,
        ),
        *reliability_middleware,
    ]
    for tool_name in [
        "advanced_web_search_tool",
        FETCH_TOOL,
        *([PAPER_SEARCH_TOOL] if paper_tool else []),
    ]:
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name=tool_name,
                run_limit=(
                    settings.max_fetch_tool_calls_per_run
                    if tool_name == FETCH_TOOL
                    else settings.max_search_tool_calls_per_run
                ),
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
            {
                "name": "scholar-agent",
                "description": (
                    "Performs paper-first investigation for empirical, academic, "
                    "scientific, technical, medical, or historical evidence."
                ),
                "system_prompt": _render_prompt_template("scholar", context),
                "tools": scholar_tools,
                "model": researcher,
                "middleware": scholar_middleware,
            },
            {
                "name": "source-auditor-agent",
                "description": (
                    "Audits source quality, provenance, recency, conflicts, and "
                    "diversity before synthesis."
                ),
                "system_prompt": _render_prompt_template("source_auditor", context),
                "tools": auditor_tools,
                "model": researcher,
                "middleware": lightweight_middleware,
            },
            {
                "name": "critic-agent",
                "description": (
                    "Reviews constraint satisfaction, evidence gaps, unsupported "
                    "claims, and missing perspectives before final drafting."
                ),
                "system_prompt": _render_prompt_template("critic", context),
                "tools": critic_tools,
                "model": critic,
                "middleware": lightweight_middleware,
            },
            {
                "name": "synthesis-agent",
                "description": (
                    "Drafts long-form reports from accumulated notes, source maps, "
                    "and critic guidance."
                ),
                "system_prompt": _render_prompt_template("synthesis", context),
                "tools": synthesis_tools,
                "model": critic,
                "middleware": lightweight_middleware,
            },
            {
                "name": "citation-agent",
                "description": (
                    "Reconciles final citations against observed URLs and prepares "
                    "the report for deterministic backend audit."
                ),
                "system_prompt": _render_prompt_template("citation", context),
                "tools": citation_tools,
                "model": critic,
                "middleware": lightweight_middleware,
            },
        ],
        backend=StateBackend(),
        middleware=middleware,
        name="research-orchestrator",
    )


def available_deep_agent_tool_names(
    settings: Settings,
    *,
    source_selection: Sequence[str] | None = None,
) -> list[str]:
    names = [
        "advanced_web_search_tool",
        "fetch_page",
        "save_research_artifact",
        "source_audit_tool",
        "citation_reconciliation_tool",
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


def _build_reliability_middleware(
    *,
    settings: Settings,
    repair_tools: Sequence[Any],
    agent_role: str,
    todo_sync: Any,
) -> list[Any]:
    from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

    return [
        EmptyContentFixMiddleware(),
        ToolNameSanitizationMiddleware(tools=repair_tools),
        TodoSanitizationMiddleware(),
        TodoSyncMiddleware(agent_role=agent_role, on_sync=todo_sync),
        ModelRetryMiddleware(
            max_retries=settings.provider_retry_attempts,
            backoff_factor=2.0,
            initial_delay=settings.provider_retry_base_seconds,
            max_delay=settings.provider_retry_max_seconds,
        ),
        ToolRetryMiddleware(
            max_retries=2,
            tools=["advanced_web_search_tool", "paper_search_tool", "fetch_page"],
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
    agent_role: str,
    todo_sync: Any,
    search_run_limit: int,
) -> list[Any]:
    return [
        *_build_reliability_middleware(
            settings=settings,
            repair_tools=repair_tools,
            agent_role=agent_role,
            todo_sync=todo_sync,
        ),
        SearchToolCallLimitMiddleware(run_limit=search_run_limit, exit_behavior="continue"),
    ]


async def _sync_deep_agent_todos(
    *,
    runtime: Any,
    run_id: str,
    agent_role: str,
    todos: Sequence[dict[str, str]],
    model: str,
) -> None:
    for todo in todos:
        content = clean_text(str(todo.get("content") or "Continue the research task."))
        if not content:
            continue
        status = _task_status_from_todo(str(todo.get("status") or "pending"))
        stable_key = _hash_text(f"{agent_role}:{content}") or f"{agent_role}:{content}"
        task_id = await runtime.store.upsert_deep_agent_task(
            run_id,
            stable_key=stable_key,
            objective=content,
            status=status,
            agent_role=agent_role,
            model=model,
            metadata={
                "todo_status": str(todo.get("status") or "pending"),
                "runtime_executor": "deepagents",
            },
        )
        await runtime.events.publish(
            run_id,
            "deepagents.todo.synced",
            {
                "task_id": task_id,
                "agent_role": agent_role,
                "status": status.value,
                "objective": content,
                "stable_key": stable_key,
            },
        )


def _task_status_from_todo(status: str) -> TaskStatus:
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "completed":
        return TaskStatus.COMPLETED
    if normalized == "in_progress":
        return TaskStatus.RUNNING
    return TaskStatus.QUEUED


def _model_for_agent_role(
    agent_role: str,
    *,
    lead_model: str,
    planner_model: str,
    researcher_model: str,
) -> str:
    if agent_role == "planner-agent":
        return planner_model
    if agent_role in {"researcher-agent", "scholar-agent", "source-auditor-agent"}:
        return researcher_model
    return lead_model


async def _persist_deep_agent_plan(
    *,
    runtime: Any,
    run_id: str,
    question: str,
    content: str,
    budget: BudgetPolicy,
    planner_model: str,
    researcher_model: str,
    source_selection: Sequence[str],
) -> ResearchPlan | None:
    content_hash = _hash_text(content)
    detail = await runtime.get_run_detail(run_id)
    metadata = dict(detail.metadata if detail is not None else {})
    if content_hash and metadata.get("deepagents_planner_artifact_hash") == content_hash:
        return detail.latest_plan if detail is not None else None

    try:
        plan = _parse_deep_agent_planner_plan(
            content,
            question=question,
            budget=budget,
            planner_model=planner_model,
            researcher_model=researcher_model,
            source_selection=source_selection,
        )
    except Exception as exc:
        warning = f"DeepAgents planner JSON could not be parsed: {exc}"
        await _store_run_warnings(runtime=runtime, run_id=run_id, warnings=[warning])
        await runtime.events.publish(
            run_id,
            "deepagents.plan.parse_failed",
            {"error": str(exc), "content_hash": content_hash},
        )
        return None

    version = await runtime.store.get_next_plan_version(run_id)
    snapshot_id, stream_ids = await runtime.store.save_plan(run_id, plan, version)
    recommended_budget = plan.recommended_budget or RecommendedBudget(
        **budget.model_dump(mode="json"),
        rationale_summary="DeepAgents planner-derived budget equals the active run budget.",
    )
    preview = PlanPreview(
        version=version,
        summary=plan.summary,
        hypothesis=plan.hypothesis,
        plan=plan,
        requested_budget=budget,
        recommended_budget=recommended_budget,
        effective_budget=budget,
        budget_decision_reason="DeepAgents planner parsed during runtime execution.",
        approval_required=False,
        recommended_execution_mode=plan.recommended_execution_mode or ExecutionMode.DEEP,
        source_selection=list(source_selection),
    )
    metadata_update: dict[str, Any] = {
        "deepagents_planner_artifact_hash": content_hash,
        "deepagents_plan_snapshot_id": snapshot_id,
        "deepagents_plan_stream_ids": stream_ids,
        "deepagents_plan_preview": preview.model_dump(mode="json"),
        "planner_contract_version": PLANNER_CONTRACT_VERSION,
    }
    if metadata.get("plan_preview") is None:
        metadata_update["plan_preview"] = preview.model_dump(mode="json")
    await runtime.store.update_run_metadata(run_id, metadata_update)
    await runtime.events.publish(
        run_id,
        "deepagents.plan.parsed",
        {
            "version": version,
            "snapshot_id": snapshot_id,
            "stream_count": len(plan.streams),
            "query_count": sum(len(stream.queries) for stream in plan.streams),
            "constraint_count": len(plan.planning_artifact.constraints)
            if plan.planning_artifact is not None
            else 0,
            "planner_contract_version": PLANNER_CONTRACT_VERSION,
        },
    )
    return plan


def _parse_deep_agent_planner_plan(
    content: str,
    *,
    question: str,
    budget: BudgetPolicy,
    planner_model: str,
    researcher_model: str,
    source_selection: Sequence[str],
) -> ResearchPlan:
    raw = _extract_json_object(content)
    task_analysis = _coerce_dict(raw.get("task_analysis"))
    query_packages = [
        item if isinstance(item, dict) else {"query": str(item)}
        for item in _coerce_list(raw.get("queries"))
    ]
    query_strings = dedupe_preserve_order(
        [
            clean_text(str(package.get("query") or ""))
            for package in query_packages
            if clean_text(str(package.get("query") or ""))
        ]
    )
    if not query_strings:
        query_strings = [question]

    toc_titles = _flatten_toc_titles(raw.get("report_toc"))
    constraint_texts = _constraint_texts(raw.get("constraints"))
    validation_checks = _string_list(raw.get("validation_checks"))
    streams = [
        ResearchStreamPlan(
            name=_stream_name_for_query(package, index=index),
            objective=clean_text(str(package.get("rationale") or package.get("query") or question)),
            queries=[clean_text(str(package.get("query") or question))],
            model=researcher_model,
        )
        for index, package in enumerate(query_packages[: budget.max_streams], start=1)
        if clean_text(str(package.get("query") or ""))
    ]
    if not streams:
        streams = [
            ResearchStreamPlan(
                name="DeepAgents Research",
                objective=question,
                queries=query_strings[: budget.max_queries_per_stream],
                model=researcher_model,
            )
        ]

    discovery_records = [
        PlanningDiscoveryRecord(query=query, provider="deepagents-planner")
        for query in query_strings[: max(budget.max_queries_per_stream, 1)]
    ]
    artifact = PlanningArtifact(
        stage=PlanningStage.EXECUTION,
        task_breakdown=json.dumps(task_analysis, ensure_ascii=True, sort_keys=True)
        if task_analysis
        else None,
        table_of_contents=toc_titles,
        constraints=constraint_texts,
        planned_deliverables=validation_checks,
        key_questions=query_strings[:16],
        discovery_queries=query_strings,
        discovery_records=discovery_records,
        source_selection=list(source_selection),
        min_total_sources_retrieved=0,
        min_total_cited_sources=0,
        validation_checks=validation_checks,
        validation_notes=_string_list(raw.get("source_strategy", {}).get("diversity_requirements"))
        if isinstance(raw.get("source_strategy"), dict)
        else [],
    )
    title = clean_text(str(raw.get("report_title") or derive_report_title(question)))
    summary = clean_text(str(task_analysis.get("user_intent") or title or question))
    hypothesis = clean_text(
        str(task_analysis.get("audience_and_depth") or raw.get("report_title") or question)
    )
    return ResearchPlan(
        summary=summary,
        hypothesis=hypothesis,
        streams=streams,
        success_criteria=validation_checks or constraint_texts[:8],
        planning_artifact=artifact,
        recommended_budget=RecommendedBudget(
            **budget.model_dump(mode="json"),
            rationale_summary="DeepAgents planner output parsed into the active budget.",
        ),
        recommended_execution_mode=ExecutionMode.DEEP,
        approval_required=False,
        complexity_factors=[
            *(_string_list(task_analysis.get("implicit_requirements"))[:5]),
            *(_string_list(task_analysis.get("explicit_requirements"))[:5]),
        ],
    )


async def _persist_deep_agent_note(
    *,
    runtime: Any,
    run_id: str,
    kind: str,
    name: str,
    content: str,
    model: str,
) -> None:
    stream = await runtime.store.get_or_create_deep_agent_stream(run_id, model=model)
    summary, key_facts, open_questions = _note_fields_from_artifact(name=name, content=content)
    note_id = await runtime.store.save_note(
        run_id,
        stream.id,
        None,
        summary=summary,
        key_facts=key_facts,
        open_questions=open_questions,
        confidence=0.72,
    )
    await runtime.events.publish(
        run_id,
        "deepagents.note.saved",
        {
            "note_id": note_id,
            "stream_id": stream.id,
            "kind": kind,
            "summary": summary,
        },
    )


def _artifact_kind_is_note(kind: str) -> bool:
    return kind in {
        "deepagents-research-note",
        "deepagents-scholar-note",
        "deepagents-source-audit",
        "deepagents-critic-review",
        "deepagents-draft-report",
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("planner artifact root is not a JSON object")
    return value


def _coerce_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _string_list(value: Any) -> list[str]:
    return [clean_text(str(item)) for item in _coerce_list(value) if clean_text(str(item))]


def _flatten_toc_titles(value: Any) -> list[str]:
    titles: list[str] = []
    for item in _coerce_list(value):
        if isinstance(item, str):
            titles.append(item)
            continue
        if not isinstance(item, dict):
            continue
        title = clean_text(str(item.get("title") or ""))
        if title:
            titles.append(title)
        for subsection in _coerce_list(item.get("subsections")):
            if isinstance(subsection, dict):
                subtitle = clean_text(str(subsection.get("title") or ""))
            else:
                subtitle = clean_text(str(subsection))
            if subtitle:
                titles.append(subtitle)
    return dedupe_preserve_order(titles)[:16]


def _constraint_texts(value: Any) -> list[str]:
    constraints: list[str] = []
    for item in _coerce_list(value):
        if isinstance(item, str):
            constraints.append(item)
        elif isinstance(item, dict):
            text = item.get("constraint") or item.get("verification") or item.get("rationale")
            if text:
                constraints.append(str(text))
    return dedupe_preserve_order(clean_text(item) for item in constraints if clean_text(item))[:32]


def _stream_name_for_query(package: dict[str, Any], *, index: int) -> str:
    targets = _string_list(package.get("target_sections"))
    if targets:
        return targets[0][:120]
    query = clean_text(str(package.get("query") or ""))
    if query:
        return query[:120]
    return f"DeepAgents Stream {index}"


def _note_fields_from_artifact(*, name: str, content: str) -> tuple[str, list[str], list[str]]:
    stripped = clean_text(content)
    sentences = extract_sentences(stripped, max_sentences=10)
    summary = clean_text(name) or (sentences[0] if sentences else "DeepAgents note")
    if sentences:
        summary = sentences[0][:800]
    bullet_facts = [
        clean_text(line.lstrip("-*0123456789. "))
        for line in content.splitlines()
        if line.strip().startswith(("-", "*")) and clean_text(line.lstrip("-* "))
    ]
    key_facts = dedupe_preserve_order([*bullet_facts, *sentences[1:]])[:10]
    open_questions = dedupe_preserve_order(
        sentence for sentence in sentences if sentence.rstrip().endswith("?")
    )[:6]
    return summary, key_facts, open_questions


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


def _format_fetched_document_for_agent(*, source_id: str, document: Any) -> str:
    content = clean_text(str(document.content or ""))
    excerpt = content[:8000]
    if len(content) > len(excerpt):
        excerpt = excerpt.rstrip() + "\n\n[Content truncated for agent context.]"
    return "\n".join(
        [
            f'<FetchedDocument source_id="{source_id}" href="{document.canonical_url}">',
            "<title>",
            clean_text(str(document.title or document.canonical_url)),
            "</title>",
            "<metadata>",
            f"retrieval_method={document.retrieval_method}",
            f"source_kind={document.source_kind}",
            f"trust_tier={document.metadata.get('trust_tier', 'unknown')}",
            "</metadata>",
            "<content>",
            excerpt,
            "</content>",
            "</FetchedDocument>",
        ]
    )


def _hash_text(text: str) -> str | None:
    cleaned = _clean_text_block(text)
    if not cleaned:
        return None
    return sha256(cleaned.encode("utf-8")).hexdigest()


async def _runtime_contract_warnings(*, runtime: Any, run_id: str) -> list[str]:
    budget_events = await runtime.list_budget_events(run_id)
    search_calls_by_role: Counter[str] = Counter()
    for event in budget_events:
        if event.get("category") != "search_calls":
            continue
        metadata = dict(event.get("metadata") or {})
        role = str(metadata.get("agent_role") or "unknown")
        search_calls_by_role[role] += int(event.get("delta") or 0)

    warnings: list[str] = []
    min_queries = runtime.settings.planner_min_discovery_queries
    if search_calls_by_role["planner-agent"] < min_queries:
        warnings.append(
            "Planner discovery is below contract: "
            f"{search_calls_by_role['planner-agent']}/{min_queries} web searches."
        )
    if search_calls_by_role["researcher-agent"] <= 0 and search_calls_by_role["scholar-agent"] <= 0:
        warnings.append("No researcher or scholar subagent search activity was recorded.")
    registry_entries = await runtime.store.list_source_registry_entries(run_id)
    tool_seen = [
        entry
        for entry in registry_entries
        if str(entry.discovered_via).startswith(("deepagents.", "custom_responses.deepagents."))
        and ".final_report" not in str(entry.discovered_via)
    ]
    if len(tool_seen) < runtime.settings.planner_min_total_sources_retrieved:
        warnings.append(
            "Observed source count is below contract: "
            f"{len(tool_seen)}/{runtime.settings.planner_min_total_sources_retrieved} sources."
        )
    events = await runtime.list_events(run_id)
    think_count = sum(1 for event in events if event.event_type == "deepagents.think.recorded")
    if think_count <= 0:
        warnings.append("No DeepAgents think checkpoints were recorded.")
    completed_roles = {
        str(event.payload.get("agent_role") or "")
        for event in events
        if event.event_type == "deepagents.agent.completed"
    }
    if runtime.settings.deepagents_require_critic_pass and "critic-agent" not in completed_roles:
        warnings.append("DeepAgents critic-agent pass was not observed.")
    if (
        runtime.settings.deepagents_require_source_audit_pass
        and "source-auditor-agent" not in completed_roles
        and not any(event.event_type == "deepagents.source_audit.recorded" for event in events)
    ):
        warnings.append("DeepAgents source audit pass was not observed.")
    if runtime.settings.deepagents_require_citation_pass and not any(
        event.event_type == "deepagents.citation.reconciled" for event in events
    ):
        warnings.append("DeepAgents citation reconciliation pass was not observed.")
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


async def _persist_deep_agent_artifact(
    *,
    runtime: Any,
    run_id: str,
    kind: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    artifact_store = getattr(runtime.orchestrator.worker, "artifact_store", None)
    if artifact_store is None or not content.strip():
        return None
    payload = ArtifactPayload(
        kind=kind,
        extension="md",
        content_type="text/markdown",
        data=content.encode("utf-8"),
    )
    reference = await artifact_store.save_artifact(
        run_id=run_id,
        source_id="deepagents-runtime",
        payload=payload,
    )
    await runtime.store.save_artifact_record(
        run_id=run_id,
        source_id="deepagents-runtime",
        kind=reference.kind,
        uri=reference.uri,
        content_type=reference.content_type,
        size_bytes=reference.size_bytes,
        sha256_digest=reference.sha256,
    )
    event_payload = {
        "kind": kind,
        "uri": reference.uri,
        "size_bytes": reference.size_bytes,
        **(metadata or {}),
    }
    await runtime.events.publish(run_id, "deepagents.file.saved", event_payload)
    return reference.as_metadata()


def _citation_reconciliation_warnings(
    *,
    final_sources: Sequence[dict[str, Any]],
    registry_entries: Sequence[Any],
) -> list[str]:
    tool_seen_urls = {
        entry.normalized_url
        for entry in registry_entries
        if str(entry.discovered_via).startswith(("deepagents.", "custom_responses.deepagents."))
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
