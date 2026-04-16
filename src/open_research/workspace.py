from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .domain import (
    ApprovalDecision,
    CitationAuditDecision,
    CitationSupportLabel,
    ResearchAssetRecord,
    ResearchAssetUsage,
    ResearchPlan,
    RunDetail,
    RunEvent,
    RunNoteRecord,
    RunWorkspaceSnapshot,
    SourceRegistryEntry,
    SourceTrustTier,
    StreamStatus,
    TaskStatus,
    WorkspaceCitationView,
    WorkspaceConnectionState,
    WorkspaceConnectionTransport,
    WorkspaceDecisionCategory,
    WorkspaceDecisionView,
    WorkspacePhaseKey,
    WorkspacePhaseState,
    WorkspacePhaseStatus,
    WorkspacePlanView,
    WorkspaceReportClaimView,
    WorkspaceReportSectionView,
    WorkspaceSourceOrigin,
    WorkspaceSourceState,
    WorkspaceSourceView,
    WorkspaceStreamView,
    WorkspaceTaskView,
)


PHASE_ORDER: list[WorkspacePhaseKey] = [
    WorkspacePhaseKey.INTAKE,
    WorkspacePhaseKey.CLARIFY,
    WorkspacePhaseKey.PLAN,
    WorkspacePhaseKey.EXECUTE,
    WorkspacePhaseKey.GROUND,
    WorkspacePhaseKey.AUDIT,
    WorkspacePhaseKey.DELIVER,
]

PHASE_LABELS: dict[WorkspacePhaseKey, str] = {
    WorkspacePhaseKey.INTAKE: "Intake",
    WorkspacePhaseKey.CLARIFY: "Clarify",
    WorkspacePhaseKey.PLAN: "Plan",
    WorkspacePhaseKey.EXECUTE: "Execute",
    WorkspacePhaseKey.GROUND: "Ground",
    WorkspacePhaseKey.AUDIT: "Audit",
    WorkspacePhaseKey.DELIVER: "Deliver",
}

EVENT_PHASES: dict[str, WorkspacePhaseKey] = {
    "run.started": WorkspacePhaseKey.INTAKE,
    "prompt.profile.applied": WorkspacePhaseKey.INTAKE,
    "job.created": WorkspacePhaseKey.INTAKE,
    "clarification.required": WorkspacePhaseKey.CLARIFY,
    "clarification.answered": WorkspacePhaseKey.CLARIFY,
    "plan.preview.created": WorkspacePhaseKey.PLAN,
    "plan.preview.approved": WorkspacePhaseKey.PLAN,
    "plan.preview.rejected": WorkspacePhaseKey.PLAN,
    "plan.preview.changes_requested": WorkspacePhaseKey.PLAN,
    "plan.created": WorkspacePhaseKey.PLAN,
    "stream.created": WorkspacePhaseKey.EXECUTE,
    "task.started": WorkspacePhaseKey.EXECUTE,
    "search.performed": WorkspacePhaseKey.EXECUTE,
    "source.fetched": WorkspacePhaseKey.EXECUTE,
    "note.saved": WorkspacePhaseKey.EXECUTE,
    "input_assets.ingested": WorkspacePhaseKey.EXECUTE,
    "gap.detected": WorkspacePhaseKey.EXECUTE,
    "replan.started": WorkspacePhaseKey.EXECUTE,
    "passages.reranked": WorkspacePhaseKey.GROUND,
    "citation.verified": WorkspacePhaseKey.GROUND,
    "claim.repair.started": WorkspacePhaseKey.GROUND,
    "claim.repair.search_performed": WorkspacePhaseKey.GROUND,
    "claim.repair.source_fetched": WorkspacePhaseKey.GROUND,
    "claim.repair.completed": WorkspacePhaseKey.GROUND,
    "citation.removed": WorkspacePhaseKey.AUDIT,
    "citation.audit.completed": WorkspacePhaseKey.AUDIT,
    "report.sanitized": WorkspacePhaseKey.AUDIT,
    "report.drafted": WorkspacePhaseKey.DELIVER,
    "report.completed": WorkspacePhaseKey.DELIVER,
    "conversation.message.added": WorkspacePhaseKey.DELIVER,
}


@dataclass
class _SectionParseResult:
    sections: list[WorkspaceReportSectionView]
    section_text: dict[str, str]


def _extract_source_origin(entry: SourceRegistryEntry) -> WorkspaceSourceOrigin:
    if entry.asset_origin == "project":
        if entry.user_supplied and entry.discovered_via.startswith("user_input"):
            return WorkspaceSourceOrigin.PROJECT_CORPUS
        return WorkspaceSourceOrigin.PROJECT_CORPUS
    if entry.asset_origin == "run":
        if entry.discovered_via.startswith("user_input"):
            return WorkspaceSourceOrigin.RUN_ATTACHMENT
        return WorkspaceSourceOrigin.RUN_ATTACHMENT
    if entry.user_supplied:
        return WorkspaceSourceOrigin.USER_REFERENCE
    return WorkspaceSourceOrigin.WEB_DISCOVERED


def _extract_source_state(entry: SourceRegistryEntry) -> WorkspaceSourceState:
    if entry.removed_in_audit:
        return WorkspaceSourceState.REMOVED
    if entry.survived_final_citation:
        return WorkspaceSourceState.CITED
    fetched_stage = str(entry.metadata.get("fetched_stage") or "")
    if entry.metadata.get("retrieved_for_grounding"):
        return WorkspaceSourceState.RETRIEVED
    if fetched_stage in {"claim_repair_fetch", "fetch", "user_input_url", "user_input_file"}:
        return WorkspaceSourceState.FETCHED
    if entry.discovered_via.startswith("user_input") or entry.discovered_via == "search":
        return WorkspaceSourceState.DISCOVERED
    return WorkspaceSourceState.CHUNKED


def _extract_support_label(value: str | None) -> CitationSupportLabel | None:
    if value is None:
        return None
    try:
        return CitationSupportLabel(value)
    except ValueError:
        return None


def _extract_trust_tier(value: Any) -> SourceTrustTier | None:
    if not value:
        return None
    try:
        return SourceTrustTier(str(value))
    except ValueError:
        return None


def _current_phase(detail: RunDetail, events: list[RunEvent]) -> WorkspacePhaseKey:
    if detail.status == "clarifying":
        return WorkspacePhaseKey.CLARIFY
    if detail.status == "awaiting_plan_approval":
        return WorkspacePhaseKey.PLAN
    if detail.status == "planning":
        return WorkspacePhaseKey.PLAN
    if detail.status == "researching":
        return WorkspacePhaseKey.EXECUTE
    if detail.status == "grounding":
        if any(event.event_type == "citation.audit.completed" for event in events):
            return WorkspacePhaseKey.AUDIT
        return WorkspacePhaseKey.GROUND
    if detail.status == "completed":
        return WorkspacePhaseKey.DELIVER
    if detail.status in {"failed", "cancelled"}:
        for phase in reversed(PHASE_ORDER):
            if any(EVENT_PHASES.get(event.event_type) == phase for event in events):
                return phase
        return WorkspacePhaseKey.INTAKE
    return WorkspacePhaseKey.INTAKE


def _build_phases(detail: RunDetail, events: list[RunEvent]) -> list[WorkspacePhaseState]:
    current = _current_phase(detail, events)
    first_by_phase: dict[WorkspacePhaseKey, datetime] = {}
    last_by_phase: dict[WorkspacePhaseKey, datetime] = {}
    counts: dict[WorkspacePhaseKey, int] = defaultdict(int)
    for event in events:
        phase = EVENT_PHASES.get(event.event_type)
        if phase is None:
            continue
        counts[phase] += 1
        first_by_phase.setdefault(phase, event.created_at)
        last_by_phase[phase] = event.created_at

    blocked_reason: str | None = None
    if detail.status == "clarifying":
        blocked_reason = "Waiting for clarification response."
    elif detail.status == "awaiting_plan_approval":
        blocked_reason = "Waiting for plan approval."
    elif detail.status == "failed":
        blocked_reason = detail.error_message or detail.terminal_reason or "Run failed."
    elif detail.status == "cancelled":
        blocked_reason = detail.terminal_reason or "Run cancelled."

    current_index = PHASE_ORDER.index(current)
    phase_states: list[WorkspacePhaseState] = []
    for index, phase in enumerate(PHASE_ORDER):
        if detail.status in {"failed", "cancelled"} and phase == current:
            status = WorkspacePhaseStatus.FAILED
        elif index < current_index:
            status = WorkspacePhaseStatus.COMPLETE
        elif phase == current:
            status = (
                WorkspacePhaseStatus.BLOCKED
                if detail.status in {"clarifying", "awaiting_plan_approval"}
                else WorkspacePhaseStatus.ACTIVE
            )
            if detail.status == "completed":
                status = WorkspacePhaseStatus.COMPLETE
        else:
            status = WorkspacePhaseStatus.IDLE
        phase_states.append(
            WorkspacePhaseState(
                key=phase,
                label=PHASE_LABELS[phase],
                status=status,
                started_at=first_by_phase.get(phase),
                completed_at=last_by_phase.get(phase)
                if status in {WorkspacePhaseStatus.COMPLETE, WorkspacePhaseStatus.FAILED}
                else None,
                blocked_reason=blocked_reason if phase == current and status in {WorkspacePhaseStatus.BLOCKED, WorkspacePhaseStatus.FAILED} else None,
                event_count=counts.get(phase, 0),
            )
        )
    return phase_states


def _build_approval_history(events: list[RunEvent], latest: ApprovalDecision | None) -> list[ApprovalDecision]:
    history: list[ApprovalDecision] = []
    for event in events:
        decision = None
        if event.event_type == "plan.preview.approved":
            decision = "approve"
        elif event.event_type == "plan.preview.rejected":
            decision = "reject"
        elif event.event_type == "plan.preview.changes_requested":
            decision = "request_changes"
        if decision is None:
            continue
        history.append(
            ApprovalDecision(
                decision=decision,
                note=str(event.payload.get("note") or "") or None,
                actor=None,
                created_at=event.created_at,
            )
        )
    if latest is not None and not any(item.created_at == latest.created_at and item.decision == latest.decision for item in history):
        history.append(latest)
    return sorted(history, key=lambda item: item.created_at)


def _build_tasks(
    tasks: list[dict[str, Any]],
    streams_by_id: dict[str, Any],
    notes: list[RunNoteRecord],
    events: list[RunEvent],
) -> tuple[list[WorkspaceTaskView], dict[str, list[WorkspaceTaskView]]]:
    notes_by_stream: dict[str, list[RunNoteRecord]] = defaultdict(list)
    for note in notes:
        notes_by_stream[note.stream_id].append(note)

    events_by_stream: dict[str, list[RunEvent]] = defaultdict(list)
    for event in events:
        stream_id = event.payload.get("stream_id")
        if isinstance(stream_id, str):
            events_by_stream[stream_id].append(event)

    views: list[WorkspaceTaskView] = []
    by_stream: dict[str, list[WorkspaceTaskView]] = defaultdict(list)
    for task in tasks:
        stream_id = task["stream_id"]
        stream_events = events_by_stream.get(stream_id, [])
        search_events = [event for event in stream_events if event.event_type == "search.performed"]
        source_events = [event for event in stream_events if event.event_type == "source.fetched"]
        latest_source_titles = [
            str(event.payload.get("title") or "Untitled source") for event in source_events[-3:]
        ]
        latest_tool_call = source_events[-1].event_type if source_events else (search_events[-1].event_type if search_events else None)
        latest_note = (notes_by_stream.get(stream_id) or [None])[-1]
        last_decision = None
        for event in reversed(stream_events):
            if event.event_type in {
                "gap.detected",
                "replan.started",
                "citation.verified",
                "claim.repair.completed",
                "citation.removed",
            }:
                last_decision = event.event_type.replace(".", " ")
                break
        status = task["status"]
        blocker_reason = None
        if status == TaskStatus.FAILED.value:
            blocker_reason = str((task.get("output_json") or {}).get("error") or "Task failed.")
        view = WorkspaceTaskView(
            id=task["id"],
            stream_id=stream_id,
            stream_name=task.get("stream_name"),
            task_type=task["kind"],
            objective=task["objective"],
            status=status,
            query_count=len((task.get("input_json") or {}).get("queries") or []) or len(search_events),
            selected_source_count=len(source_events),
            notes_produced=len(notes_by_stream.get(stream_id) or []),
            elapsed_ms=(
                max(int((streams_by_id.get(stream_id).elapsed_ms if stream_id in streams_by_id else 0) or 0), 0)
                if stream_id in streams_by_id
                else None
            ),
            started_at=task.get("created_at"),
            completed_at=task.get("updated_at") if status in {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value} else None,
            next_action=(
                "Awaiting more sources"
                if status == TaskStatus.RUNNING.value and not source_events
                else "Ground claims"
                if status == TaskStatus.RUNNING.value and source_events
                else None
            ),
            blocker_reason=blocker_reason,
            latest_sources=latest_source_titles,
            latest_note_summary=latest_note.summary if latest_note is not None else None,
            last_tool_call=latest_tool_call,
            last_decision=last_decision,
            metadata={
                "attempt_count": task.get("attempt_count", 0),
            },
        )
        views.append(view)
        by_stream[stream_id].append(view)
    return views, by_stream


def _build_streams(
    detail: RunDetail,
    tasks_by_stream: dict[str, list[WorkspaceTaskView]],
    notes: list[RunNoteRecord],
    events: list[RunEvent],
) -> list[WorkspaceStreamView]:
    notes_by_stream: dict[str, list[RunNoteRecord]] = defaultdict(list)
    for note in notes:
        notes_by_stream[note.stream_id].append(note)
    source_titles_by_stream: dict[str, list[str]] = defaultdict(list)
    query_count_by_stream: dict[str, int] = defaultdict(int)
    for event in events:
        stream_id = event.payload.get("stream_id")
        if not isinstance(stream_id, str):
            continue
        if event.event_type == "source.fetched":
            source_titles_by_stream[stream_id].append(str(event.payload.get("title") or "Untitled source"))
        elif event.event_type == "search.performed":
            query_count_by_stream[stream_id] += 1

    streams: list[WorkspaceStreamView] = []
    for stream in detail.streams:
        stream_notes = notes_by_stream.get(stream.id) or []
        tasks = tasks_by_stream.get(stream.id) or []
        query_count = query_count_by_stream.get(stream.id) or max(
            (task.query_count for task in tasks),
            default=0,
        )
        streams.append(
            WorkspaceStreamView(
                id=stream.id,
                name=stream.name,
                objective=stream.objective,
                model=stream.model,
                status=stream.status,
                query_count=query_count,
                selected_source_count=len(source_titles_by_stream.get(stream.id) or []),
                note_count=len(stream_notes),
                latest_source_titles=(source_titles_by_stream.get(stream.id) or [])[-3:],
                latest_note_summary=stream_notes[-1].summary if stream_notes else None,
                tasks=tasks,
                confidence=stream.confidence,
                elapsed_ms=stream.elapsed_ms,
                cost_so_far=stream.cost_so_far,
            )
        )
    return streams


def _build_decisions(events: list[RunEvent], streams_by_id: dict[str, Any]) -> list[WorkspaceDecisionView]:
    decisions: list[WorkspaceDecisionView] = []
    for event in events:
        category: WorkspaceDecisionCategory | None = None
        title: str | None = None
        rationale: str | None = None
        if event.event_type == "plan.created":
            category = WorkspaceDecisionCategory.PLANNING
            title = "Research plan created"
            rationale = str(event.payload.get("summary") or "Planner generated the run plan.")
        elif event.event_type == "clarification.required":
            category = WorkspaceDecisionCategory.CLARIFICATION
            title = "Clarification required"
            rationale = str(event.payload.get("rationale") or "The run paused for clarification.")
        elif event.event_type == "clarification.answered":
            category = WorkspaceDecisionCategory.CLARIFICATION
            title = "Clarification answered"
            rationale = "Clarification response was received and the preview regenerated."
        elif event.event_type == "stream.created":
            category = WorkspaceDecisionCategory.STREAM_LAUNCH
            title = "Research stream launched"
            rationale = str(event.payload.get("objective") or "A stream was added to execute the plan.")
        elif event.event_type == "replan.started":
            category = WorkspaceDecisionCategory.REPLAN
            title = "Replan triggered"
            rationale = str(event.payload.get("rationale") or "Coverage gaps triggered a replan.")
        elif event.event_type == "citation.verified":
            category = WorkspaceDecisionCategory.VERIFICATION
            title = "Claim verified"
            rationale = str(event.payload.get("reason") or "Claim support was verified against retrieved evidence.")
        elif event.event_type == "claim.repair.completed":
            category = WorkspaceDecisionCategory.CLAIM_REPAIR
            title = "Claim repair completed"
            rationale = str(event.payload.get("reason") or "Claim repair cycle finished.")
        elif event.event_type == "citation.removed":
            category = WorkspaceDecisionCategory.AUDIT
            title = "Citation removed"
            reasons = event.payload.get("reasons") or []
            rationale = ", ".join(str(reason) for reason in reasons) or "Citation audit removed a citation."
        if category is None or title is None or rationale is None:
            continue
        stream_id = event.payload.get("stream_id")
        affected_stream_name = (
            streams_by_id[stream_id].name
            if isinstance(stream_id, str) and stream_id in streams_by_id
            else None
        )
        support_label = _extract_support_label(
            str(event.payload.get("support_label")) if event.payload.get("support_label") else None
        )
        decisions.append(
            WorkspaceDecisionView(
                id=f"decision-{event.id}",
                category=category,
                title=title,
                rationale=rationale,
                affected_stream_id=stream_id if isinstance(stream_id, str) else None,
                affected_stream_name=affected_stream_name,
                affected_section=(
                    str(event.payload.get("section_title"))
                    if event.payload.get("section_title")
                    else None
                ),
                affected_claim=str(event.payload.get("claim")) if event.payload.get("claim") else None,
                confidence=(
                    float(event.payload["confidence"])
                    if isinstance(event.payload.get("confidence"), (int, float))
                    else None
                ),
                uncertainty=None if support_label != CitationSupportLabel.UNSUPPORTED else "Unsupported claim",
                supporting_evidence_count=int(event.payload.get("supporting_evidence_count") or 0),
                timestamp=event.created_at,
                metadata=event.payload,
            )
        )
    return list(reversed(decisions[-40:]))


def _parse_report_sections(markdown: str | None) -> _SectionParseResult:
    if not markdown:
        return _SectionParseResult(sections=[], section_text={})
    pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))
    sections: list[WorkspaceReportSectionView] = []
    section_text: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        if title in {"Citations", "Remaining Uncertainty"}:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        section_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or f"section-{index}"
        sections.append(
            WorkspaceReportSectionView(
                id=section_id,
                title=title,
                body_markdown=body,
                draft_status="completed",
            )
        )
        section_text[title] = body
    return _SectionParseResult(sections=sections, section_text=section_text)


def _build_report_sections(
    detail: RunDetail,
    events: list[RunEvent],
) -> tuple[list[WorkspaceReportSectionView], list[WorkspaceCitationView]]:
    parsed = _parse_report_sections(detail.final_report_markdown)
    claim_views_by_section: dict[str, dict[tuple[str, int], WorkspaceReportClaimView]] = defaultdict(dict)
    citations_by_key: dict[tuple[str, str], WorkspaceCitationView] = {}

    for event in events:
        if event.event_type != "citation.verified":
            continue
        section_title = str(event.payload.get("section_title") or "Overview")
        claim = str(event.payload.get("claim") or "")
        ordinal = int(event.payload.get("ordinal") or len(claim_views_by_section[section_title]) + 1)
        support_label = _extract_support_label(
            str(event.payload.get("support_label")) if event.payload.get("support_label") else None
        )
        claim_views_by_section[section_title][(claim, ordinal)] = WorkspaceReportClaimView(
            section_title=section_title,
            ordinal=ordinal,
            claim=claim,
            support_label=support_label,
            confidence=(
                float(event.payload["confidence"])
                if isinstance(event.payload.get("confidence"), (int, float))
                else None
            ),
            citation_count=1,
            claim_repair_ran=bool(event.payload.get("repair_attempts")),
        )
        citations_by_key[(section_title, claim)] = WorkspaceCitationView(
            id=f"citation-{event.id}",
            section_title=section_title,
            claim=claim,
            status="surviving",
            source_id=str(event.payload.get("source_id")) if event.payload.get("source_id") else None,
            source_title=str(event.payload.get("source_title")) if event.payload.get("source_title") else None,
            source_url=str(event.payload.get("source_url")) if event.payload.get("source_url") else None,
            citation_key=str(event.payload.get("citation_key")) if event.payload.get("citation_key") else None,
            support_label=support_label,
            quote=str(event.payload.get("quote")) if event.payload.get("quote") else None,
            confidence=(
                float(event.payload["confidence"])
                if isinstance(event.payload.get("confidence"), (int, float))
                else None
            ),
            trust_tier=_extract_trust_tier(event.payload.get("trust_tier")),
            metadata=event.payload,
        )

    for audit in detail.citation_audits:
        if audit.decision != CitationAuditDecision.REMOVED:
            continue
        key = (audit.section_title, audit.claim)
        existing = citations_by_key.get(key)
        citations_by_key[key] = WorkspaceCitationView(
            id=f"audit-{audit.id}",
            section_title=audit.section_title,
            claim=audit.claim,
            status="removed",
            source_id=audit.source_id,
            source_title=existing.source_title if existing else None,
            source_url=audit.source_url,
            citation_key=audit.citation_key,
            support_label=existing.support_label if existing else None,
            quote=existing.quote if existing else None,
            confidence=existing.confidence if existing else None,
            trust_tier=existing.trust_tier if existing else None,
            audit_status=audit.decision,
            audit_reasons=[reason.value for reason in audit.reasons],
            metadata=audit.metadata,
        )
        section_claims = claim_views_by_section[audit.section_title]
        existing_claim = next((claim_view for claim_view in section_claims.values() if claim_view.claim == audit.claim), None)
        if existing_claim is not None:
            existing_claim.removed_citation_count += 1
        else:
            section_claims[(audit.claim, audit.ordinal)] = WorkspaceReportClaimView(
                section_title=audit.section_title,
                ordinal=audit.ordinal,
                claim=audit.claim,
                support_label=CitationSupportLabel.UNSUPPORTED,
                citation_count=0,
                removed_citation_count=1,
            )

    section_map = {section.title: section for section in parsed.sections}
    for section_title, claims in claim_views_by_section.items():
        section = section_map.get(section_title)
        if section is None:
            section = WorkspaceReportSectionView(
                id=re.sub(r"[^a-z0-9]+", "-", section_title.lower()).strip("-") or "section",
                title=section_title,
                body_markdown=parsed.section_text.get(section_title, ""),
                draft_status="completed" if detail.status == "completed" else "in_progress",
            )
            parsed.sections.append(section)
            section_map[section_title] = section
        ordered_claims = sorted(claims.values(), key=lambda claim: claim.ordinal)
        section.claims = ordered_claims
        section.grounded_claim_count = sum(1 for claim in ordered_claims if claim.support_label not in {None, CitationSupportLabel.UNSUPPORTED})
        section.unsupported_claim_count = sum(
            1
            for claim in ordered_claims
            if claim.support_label in {None, CitationSupportLabel.UNSUPPORTED}
        )
        section.citation_count = sum(claim.citation_count for claim in ordered_claims)
        section.removed_citation_count = sum(claim.removed_citation_count for claim in ordered_claims)
        section.claim_repair_count = sum(1 for claim in ordered_claims if claim.claim_repair_ran)

    return parsed.sections, sorted(citations_by_key.values(), key=lambda item: (item.section_title, item.claim, item.status))


def _build_sources(
    detail: RunDetail,
    notes: list[RunNoteRecord],
    passages: list[dict[str, Any]],
    citations: list[WorkspaceCitationView],
) -> list[WorkspaceSourceView]:
    notes_by_source: dict[str, list[RunNoteRecord]] = defaultdict(list)
    for note in notes:
        if note.source_id:
            notes_by_source[note.source_id].append(note)
    passages_by_source: dict[str, int] = defaultdict(int)
    for passage in passages:
        source_id = passage.get("source_id")
        if isinstance(source_id, str):
            passages_by_source[source_id] += 1

    citation_sections_by_source: dict[str, set[str]] = defaultdict(set)
    citation_status_by_source: dict[str, str] = {}
    for citation in citations:
        if citation.source_id:
            citation_sections_by_source[citation.source_id].add(citation.section_title)
            citation_status_by_source[citation.source_id] = citation.status

    stream_name_by_id = {stream.id: stream.name for stream in detail.streams}
    views: list[WorkspaceSourceView] = []
    for entry in detail.source_registry_entries:
        metadata = entry.metadata or {}
        source_id = entry.source_id
        note_summaries = [note.summary for note in notes_by_source.get(source_id or "", [])][:3]
        stream_ids = []
        for note in notes_by_source.get(source_id or "", []):
            if note.stream_id not in stream_ids:
                stream_ids.append(note.stream_id)
        views.append(
            WorkspaceSourceView(
                id=entry.id,
                source_id=source_id,
                asset_id=entry.asset_id,
                title=entry.title,
                url=entry.canonical_url or entry.url,
                origin=_extract_source_origin(entry),
                state=_extract_source_state(entry),
                provider=entry.provider,
                trust_tier=_extract_trust_tier(metadata.get("trust_tier")),
                stream_ids=stream_ids,
                stream_names=[stream_name_by_id[stream_id] for stream_id in stream_ids if stream_id in stream_name_by_id],
                report_sections=sorted(citation_sections_by_source.get(source_id or "", set())),
                citation_status=citation_status_by_source.get(source_id or ""),
                survived_final_citation=entry.survived_final_citation,
                removed_in_audit=entry.removed_in_audit,
                audit_reasons=entry.audit_reasons,
                note_summaries=note_summaries,
                passages_used=passages_by_source.get(source_id or "", 0),
                metadata=metadata,
            )
        )
    return views


def build_run_workspace_snapshot(
    detail: RunDetail,
    *,
    events: list[RunEvent],
    tasks: list[dict[str, Any]],
    notes: list[RunNoteRecord],
    passages: list[dict[str, Any]],
) -> RunWorkspaceSnapshot:
    phases = _build_phases(detail, events)
    current_phase = _current_phase(detail, events)
    streams_by_id = {stream.id: stream for stream in detail.streams}
    _, tasks_by_stream = _build_tasks(tasks, streams_by_id, notes, events)
    streams = _build_streams(detail, tasks_by_stream, notes, events)
    decisions = _build_decisions(events, streams_by_id)
    report_sections, citations = _build_report_sections(detail, events)
    sources = _build_sources(detail, notes, passages, citations)
    connection = WorkspaceConnectionState(
        transport=(
            WorkspaceConnectionTransport.TERMINAL
            if detail.status in {"completed", "failed", "cancelled"}
            else WorkspaceConnectionTransport.IDLE
        ),
        stream_mode="persisted",
        backend_mode="run_events",
        workflow_backend=detail.workflow_backend,
        last_event_id=events[-1].id if events else 0,
        event_count=len(events),
        replay_lag=0,
        reconnect_state=None,
        last_event_at=events[-1].created_at if events else None,
    )
    plan = WorkspacePlanView(
        clarification_session=detail.clarification_session,
        plan_preview=detail.plan_preview,
        approved_plan=detail.latest_plan,
        requested_budget=detail.requested_budget,
        recommended_budget=detail.recommended_budget,
        effective_budget=detail.effective_budget or detail.budget,
        budget_decision_reason=detail.budget_decision_reason,
        planning_assets=detail.planning_assets_used,
        project_assets=detail.project_assets_used,
        reference_assets=detail.reference_assets_used,
        approval_history=_build_approval_history(events, detail.latest_approval_decision),
    )
    return RunWorkspaceSnapshot(
        run_id=detail.id,
        question=detail.question,
        project_id=detail.project_id,
        status=detail.status,
        execution_mode=detail.execution_mode,
        approval_status=detail.approval_status,
        current_phase=current_phase,
        phases=phases,
        plan=plan,
        streams=streams,
        decisions=decisions,
        sources=sources,
        citations=citations,
        report_sections=report_sections,
        connection=connection,
        source_selection=detail.source_selection,
        project_assets_available=detail.project_assets_used,
        run_assets_available=detail.run_assets_used,
        asset_processing_errors=detail.asset_processing_errors,
        final_report_markdown=detail.final_report_markdown,
        estimated_cost_usd=detail.estimated_cost_usd,
        created_at=detail.created_at,
        updated_at=detail.updated_at,
        job=detail.job,
    )
