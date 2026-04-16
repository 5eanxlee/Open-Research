from __future__ import annotations

from collections import Counter
from datetime import timedelta
from uuid import uuid4

from .db import ResearchStore
from .domain import (
    AssessmentKind,
    AssessmentSource,
    BehaviorAssessment,
    CitationAuditDecision,
    ContextFragment,
    ContextFragmentKind,
    ContextPack,
    ContextPhase,
    MemoryInfluencePolicy,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    ProfileFeedback,
    ProfilePreferences,
    SourceTrustTier,
    utc_now,
)
from .events import RunEventService
from .utils import clean_text, dedupe_preserve_order, domain_for_url, tokenize


def render_context_pack(pack: ContextPack | None) -> str:
    if pack is None or not pack.fragments:
        return "No prior memory fragments were selected."
    blocks = [f"Context phase: {pack.phase.value}", f"Summary: {pack.summary}"]
    for fragment in pack.fragments:
        blocks.append(
            "\n".join(
                [
                    f"- Kind: {fragment.kind.value}",
                    f"- Title: {fragment.title}",
                    f"- Reason: {fragment.selected_reason}",
                    f"- Content: {fragment.content}",
                ]
            )
        )
    return "\n\n".join(blocks)


class ContextAssembler:
    def __init__(self, *, store: ResearchStore, events: RunEventService) -> None:
        self.store = store
        self.events = events

    async def assemble(
        self,
        *,
        run_id: str,
        question: str,
        profile_id: str,
        phase: ContextPhase,
        memory_policy: MemoryInfluencePolicy,
    ) -> ContextPack:
        preferences_record = await self.store.get_profile(profile_id)
        preferences = (
            preferences_record.preferences
            if preferences_record is not None
            else ProfilePreferences()
        )
        token_budget = memory_policy.budget_for_phase(phase)

        fragments: list[ContextFragment] = []
        dropped: list[ContextFragment] = []
        memories = await self.store.search_memories(
            question,
            profile_id=profile_id,
            phase=phase,
            kinds=self._kinds_for_phase(phase, memory_policy),
            limit=memory_policy.retrieval_limit,
        )

        if memory_policy.allow_preference_for_phase(phase):
            preference_content = self._render_preferences(preferences)
            if preference_content:
                fragments.append(
                    ContextFragment(
                        id=str(uuid4()),
                        kind=ContextFragmentKind.PREFERENCE,
                        phase=phase,
                        memory_id=None,
                        title="Profile preferences",
                        content=preference_content,
                        token_estimate=len(tokenize(preference_content)),
                        score=1.0,
                        freshness_score=1.0,
                        trust_score=1.0,
                        selected_reason="Stable profile preferences for this run.",
                        metadata={"profile_id": profile_id},
                    )
                )

        for memory in memories:
            fragment_kind = (
                ContextFragmentKind.PREFERENCE
                if memory.kind == MemoryKind.PREFERENCE
                else ContextFragmentKind.OPERATIONAL
            )
            freshness_score = 1.0
            freshness_expires_at = memory.freshness_expires_at
            if freshness_expires_at is not None and freshness_expires_at.tzinfo is None:
                freshness_expires_at = freshness_expires_at.replace(tzinfo=utc_now().tzinfo)
            if freshness_expires_at is not None and freshness_expires_at < utc_now():
                freshness_score = max(0.0, 1.0 - memory_policy.stale_penalty)
            trust_score = 0.8
            if memory.trust_tier == SourceTrustTier.PRIMARY:
                trust_score = 1.0
            elif memory.trust_tier == SourceTrustTier.LOW:
                trust_score = 0.4
            fragment = ContextFragment(
                id=str(uuid4()),
                kind=fragment_kind,
                phase=phase,
                memory_id=memory.id,
                title=memory.summary,
                content=memory.content,
                token_estimate=len(tokenize(memory.content)),
                score=round(memory.usefulness_score + freshness_score + trust_score, 3),
                freshness_score=round(freshness_score, 3),
                trust_score=round(trust_score, 3),
                selected_reason=f"{memory.kind.value} memory matched {phase.value}.",
                metadata={
                    "kind": memory.kind.value,
                    "tags": memory.tags,
                    "source_run_id": memory.source_run_id,
                    **memory.metadata,
                },
            )
            fragments.append(fragment)
            await self.events.publish(
                run_id,
                "memory.retrieved",
                {
                    "phase": phase.value,
                    "memory_id": memory.id,
                    "memory_kind": memory.kind.value,
                    "title": memory.summary,
                    "usefulness_score": memory.usefulness_score,
                    "tags": memory.tags,
                },
            )

        selected: list[ContextFragment] = []
        used_tokens = 0
        seen_titles: set[str] = set()
        for fragment in sorted(fragments, key=lambda item: item.score, reverse=True):
            duplicate = fragment.title in seen_titles
            over_budget = token_budget > 0 and used_tokens + fragment.token_estimate > token_budget
            if duplicate or over_budget:
                dropped_reason = "duplicate fragment" if duplicate else "token budget exhausted"
                dropped.append(fragment.model_copy(update={"dropped_reason": dropped_reason}))
                await self.events.publish(
                    run_id,
                    "context.fragment.dropped",
                    {
                        "phase": phase.value,
                        "title": fragment.title,
                        "reason": dropped_reason,
                        "kind": fragment.kind.value,
                        "token_estimate": fragment.token_estimate,
                    },
                )
                continue
            selected.append(fragment)
            used_tokens += fragment.token_estimate
            seen_titles.add(fragment.title)

        pack = ContextPack(
            id=str(uuid4()),
            run_id=run_id,
            profile_id=profile_id,
            phase=phase,
            summary=f"{len(selected)} fragments selected for {phase.value}.",
            token_budget=token_budget,
            used_tokens=used_tokens,
            fragments=selected,
            dropped_fragments=dropped,
        )
        await self.store.save_context_pack(pack)
        await self.events.publish(
            run_id,
            "context.pack.created",
            {
                "context_pack_id": pack.id,
                "phase": phase.value,
                "fragment_count": len(pack.fragments),
                "dropped_count": len(pack.dropped_fragments),
                "used_tokens": pack.used_tokens,
                "token_budget": pack.token_budget,
                "summary": pack.summary,
            },
        )
        return pack

    @staticmethod
    def _kinds_for_phase(
        phase: ContextPhase,
        memory_policy: MemoryInfluencePolicy,
    ) -> list[MemoryKind]:
        if phase == ContextPhase.GROUND:
            return [MemoryKind.PROCEDURAL] if memory_policy.grounding_budget_tokens > 0 else []
        kinds = [MemoryKind.EPISODIC, MemoryKind.PROCEDURAL, MemoryKind.FAILURE]
        if memory_policy.allow_preference_for_phase(phase):
            kinds.append(MemoryKind.PREFERENCE)
        return kinds

    @staticmethod
    def _render_preferences(preferences: ProfilePreferences) -> str:
        lines: list[str] = []
        if preferences.preferred_source_patterns:
            lines.append(
                "Prefer sources matching: " + ", ".join(preferences.preferred_source_patterns[:5])
            )
        if preferences.avoided_source_patterns:
            lines.append(
                "Avoid or de-prioritize: " + ", ".join(preferences.avoided_source_patterns[:5])
            )
        if preferences.answer_style_bias is not None:
            lines.append(f"Bias report style toward {preferences.answer_style_bias.value}.")
        if preferences.recency_bias is not None:
            lines.append(f"Bias recency handling toward {preferences.recency_bias.value}.")
        if preferences.source_trust_floor_bias is not None:
            lines.append(
                f"Treat {preferences.source_trust_floor_bias.value} as the preferred trust floor."
            )
        if preferences.include_counterevidence_bias is not None:
            lines.append(
                "Prioritize counterevidence."
                if preferences.include_counterevidence_bias
                else "Keep the report concise and avoid excess counterevidence."
            )
        return "\n".join(lines)


class MemoryCompiler:
    def __init__(self, *, store: ResearchStore, events: RunEventService) -> None:
        self.store = store
        self.events = events

    async def compile_stream(
        self,
        *,
        run_id: str,
        profile_id: str,
        stream_name: str,
        stream_objective: str,
        queries: list[str],
        providers: list[str],
        sources_examined: int,
        notes_written: int,
        confidence: float,
    ) -> list[MemoryRecord]:
        memories = [
            MemoryRecord(
                id=str(uuid4()),
                profile_id=profile_id,
                kind=MemoryKind.EPISODIC,
                scope=MemoryScope.PROFILE,
                phase_hints=[ContextPhase.PLAN, ContextPhase.REPLAN, ContextPhase.RESEARCH],
                summary=f"Stream pattern: {stream_name}",
                content=(
                    f"Objective: {stream_objective}. Queries that were used successfully: "
                    f"{', '.join(queries[:3])}."
                ),
                source_run_id=run_id,
                tags=dedupe_preserve_order(
                    [stream_name.lower(), *[provider.lower() for provider in providers]]
                ),
                usefulness_score=min(1.0, 0.35 + (0.1 * notes_written) + (0.2 * confidence)),
                freshness_expires_at=utc_now() + timedelta(days=14),
                metadata={
                    "stream_name": stream_name,
                    "stream_objective": stream_objective,
                    "suggested_queries": queries[:3],
                    "providers": providers,
                    "sources_examined": sources_examined,
                    "notes_written": notes_written,
                },
            ),
            MemoryRecord(
                id=str(uuid4()),
                profile_id=profile_id,
                kind=MemoryKind.PROCEDURAL,
                scope=MemoryScope.PROFILE,
                phase_hints=[ContextPhase.PLAN, ContextPhase.RESEARCH],
                summary=f"Operational lesson from {stream_name}",
                content=(
                    f"Provider mix {', '.join(providers) or 'unknown'} produced "
                    f"{notes_written} notes from {sources_examined} sources."
                ),
                source_run_id=run_id,
                tags=["procedural", *[provider.lower() for provider in providers]],
                usefulness_score=min(1.0, 0.4 + (0.15 * confidence)),
                freshness_expires_at=utc_now() + timedelta(days=30),
                metadata={
                    "providers": providers,
                    "yield_ratio": round(notes_written / max(sources_examined, 1), 3),
                    "suggested_queries": queries[:2],
                },
            ),
        ]
        for memory in memories:
            await self.store.save_memory_record(memory)
            await self.events.publish(
                run_id,
                "memory.compiled",
                {
                    "memory_id": memory.id,
                    "memory_kind": memory.kind.value,
                    "summary": memory.summary,
                    "source_run_id": run_id,
                },
            )
        return memories

    async def compile_run(self, *, run_id: str, profile_id: str) -> list[MemoryRecord]:
        detail = await self.store.get_run_detail(run_id)
        if detail is None:
            return []
        memories: list[MemoryRecord] = []
        if detail.citation_audits:
            removed = [
                audit
                for audit in detail.citation_audits
                if audit.decision == CitationAuditDecision.REMOVED
            ]
            if removed:
                memories.append(
                    MemoryRecord(
                        id=str(uuid4()),
                        profile_id=profile_id,
                        kind=MemoryKind.FAILURE,
                        scope=MemoryScope.PROFILE,
                        phase_hints=[ContextPhase.PLAN, ContextPhase.SYNTHESIZE],
                        summary="Citation audit removed claims",
                        content=(
                            "Prior runs lost citations for unsupported or mismatched claims. "
                            "Prefer narrower claims and stronger primary evidence."
                        ),
                        source_run_id=run_id,
                        tags=["grounding", "citation-audit"],
                        usefulness_score=0.7,
                        freshness_expires_at=utc_now() + timedelta(days=21),
                        metadata={
                            "removed_citations": len(removed),
                            "reasons": dedupe_preserve_order(
                                reason.value for audit in removed for reason in audit.reasons
                            ),
                        },
                    )
                )
        if detail.final_report is not None:
            memories.append(
                MemoryRecord(
                    id=str(uuid4()),
                    profile_id=profile_id,
                    kind=MemoryKind.PROCEDURAL,
                    scope=MemoryScope.PROFILE,
                    phase_hints=[ContextPhase.SYNTHESIZE, ContextPhase.PLAN],
                    summary="Completed run synthesis pattern",
                    content=(
                        f"The run completed with "
                        f"{len(detail.final_report.citations)} citations and "
                        f"{len(detail.final_report.unsupported_claims)} unsupported claims."
                    ),
                    source_run_id=run_id,
                    tags=["completed-run", detail.status.value],
                    usefulness_score=max(
                        0.2,
                        1.0 - (0.1 * len(detail.final_report.unsupported_claims)),
                    ),
                    freshness_expires_at=utc_now() + timedelta(days=30),
                    metadata={
                        "citation_count": len(detail.final_report.citations),
                        "unsupported_claims": len(detail.final_report.unsupported_claims),
                    },
                )
            )
        for memory in memories:
            await self.store.save_memory_record(memory)
        return memories

    async def compile_feedback(self, feedback: ProfileFeedback) -> list[MemoryRecord]:
        memories: list[MemoryRecord] = []
        preference_parts: list[str] = []
        if feedback.preferred_source_patterns:
            preference_parts.append("Prefer " + ", ".join(feedback.preferred_source_patterns[:5]))
        if feedback.avoided_source_patterns:
            preference_parts.append("Avoid " + ", ".join(feedback.avoided_source_patterns[:5]))
        if feedback.answer_style_bias is not None:
            preference_parts.append(f"Use {feedback.answer_style_bias.value} style.")
        if feedback.include_counterevidence_bias is not None:
            preference_parts.append(
                "Include counterevidence."
                if feedback.include_counterevidence_bias
                else "Keep synthesis focused."
            )
        if feedback.correction:
            preference_parts.append(clean_text(feedback.correction))
        if not preference_parts:
            return memories
        memory = MemoryRecord(
            id=str(uuid4()),
            profile_id=feedback.profile_id,
            kind=MemoryKind.PREFERENCE,
            scope=MemoryScope.PROFILE,
            phase_hints=[ContextPhase.PLAN, ContextPhase.RESEARCH, ContextPhase.SYNTHESIZE],
            summary="Explicit profile feedback",
            content=" ".join(preference_parts),
            source_run_id=feedback.run_id,
            tags=["explicit-feedback"],
            usefulness_score=0.95,
            metadata={
                "source_fit": feedback.source_fit,
                "style_fit": feedback.style_fit,
                "usefulness": feedback.usefulness,
            },
        )
        await self.store.save_memory_record(memory)
        memories.append(memory)
        return memories


class BehaviorJudge:
    def __init__(self, *, store: ResearchStore, events: RunEventService) -> None:
        self.store = store
        self.events = events

    async def assess_run(self, *, run_id: str, profile_id: str) -> list[BehaviorAssessment]:
        detail = await self.store.get_run_detail(run_id)
        profile = await self.store.get_profile(profile_id)
        preferences = profile.preferences if profile is not None else ProfilePreferences()
        if detail is None:
            return []

        source_score, source_rationale = self._source_fit(detail, preferences)
        style_score, style_rationale = self._style_fit(detail, preferences)
        operational_score, operational_rationale = self._operational_fit(detail)
        memory_score, memory_rationale = self._memory_usefulness(detail)
        assessments = [
            BehaviorAssessment(
                id=str(uuid4()),
                run_id=run_id,
                profile_id=profile_id,
                kind=AssessmentKind.SOURCE_FIT,
                source=AssessmentSource.SYSTEM,
                score=source_score,
                rationale=source_rationale,
            ),
            BehaviorAssessment(
                id=str(uuid4()),
                run_id=run_id,
                profile_id=profile_id,
                kind=AssessmentKind.STYLE_FIT,
                source=AssessmentSource.SYSTEM,
                score=style_score,
                rationale=style_rationale,
            ),
            BehaviorAssessment(
                id=str(uuid4()),
                run_id=run_id,
                profile_id=profile_id,
                kind=AssessmentKind.OPERATIONAL,
                source=AssessmentSource.SYSTEM,
                score=operational_score,
                rationale=operational_rationale,
            ),
            BehaviorAssessment(
                id=str(uuid4()),
                run_id=run_id,
                profile_id=profile_id,
                kind=AssessmentKind.MEMORY_USEFULNESS,
                source=AssessmentSource.SYSTEM,
                score=memory_score,
                rationale=memory_rationale,
            ),
        ]
        await self.store.save_behavior_assessments(run_id, assessments)
        return assessments

    async def assess_feedback(self, feedback: ProfileFeedback) -> list[BehaviorAssessment]:
        assessments: list[BehaviorAssessment] = []
        for kind, score in (
            (AssessmentKind.SOURCE_FIT, feedback.source_fit),
            (AssessmentKind.STYLE_FIT, feedback.style_fit),
            (AssessmentKind.MEMORY_USEFULNESS, feedback.usefulness),
        ):
            if score is None or feedback.run_id is None:
                continue
            assessments.append(
                BehaviorAssessment(
                    id=str(uuid4()),
                    run_id=feedback.run_id,
                    profile_id=feedback.profile_id,
                    kind=kind,
                    source=AssessmentSource.USER,
                    score=score,
                    rationale=feedback.correction or "Recorded explicit user feedback.",
                )
            )
        if feedback.run_id is not None and assessments:
            await self.store.save_behavior_assessments(feedback.run_id, assessments)
        return assessments

    @staticmethod
    def _source_fit(detail, preferences: ProfilePreferences) -> tuple[float, str]:
        if detail.final_report is None or not detail.final_report.citations:
            return (
                0.4,
                "No citations were available to compare against profile source preferences.",
            )
        domains = [
            domain_for_url(str(citation.source_url)) for citation in detail.final_report.citations
        ]
        matched = 0
        preferred = [pattern.lower() for pattern in preferences.preferred_source_patterns]
        avoided = [pattern.lower() for pattern in preferences.avoided_source_patterns]
        for domain in domains:
            lowered = domain.lower()
            if preferred and any(pattern in lowered for pattern in preferred):
                matched += 1
            if avoided and any(pattern in lowered for pattern in avoided):
                matched -= 1
        score = (
            0.5
            if not preferred and not avoided
            else max(0.0, min(1.0, 0.5 + (matched / max(len(domains), 1))))
        )
        return (
            round(score, 3),
            "Compared citation domains against profile source preferences.",
        )

    @staticmethod
    def _style_fit(detail, preferences: ProfilePreferences) -> tuple[float, str]:
        if preferences.answer_style_bias is None:
            return 0.6, "No profile answer-style bias was configured."
        style = detail.agent_config.answer_style if detail.agent_config is not None else None
        score = 1.0 if style == preferences.answer_style_bias else 0.35
        return (
            score,
            "Run answer style was "
            f"{style or 'unknown'} versus profile preference "
            f"{preferences.answer_style_bias.value}.",
        )

    @staticmethod
    def _operational_fit(detail) -> tuple[float, str]:
        if detail.final_report is None:
            return 0.25, "Run did not reach a final grounded report."
        removed = sum(
            1 for audit in detail.citation_audits if audit.decision == CitationAuditDecision.REMOVED
        )
        score = max(
            0.0, 1.0 - (0.08 * len(detail.final_report.unsupported_claims)) - (0.05 * removed)
        )
        return round(
            score, 3
        ), "Operational quality penalizes unsupported claims and citation removals."

    @staticmethod
    def _memory_usefulness(detail) -> tuple[float, str]:
        if not detail.active_context_pack_ids:
            return 0.5, "No context packs were used for this run."
        score = 0.8 if detail.final_report is not None else 0.4
        return (
            score,
            f"{len(detail.active_context_pack_ids)} context packs were active during the run.",
        )


def merge_feedback_into_preferences(
    existing: ProfilePreferences,
    feedback: ProfileFeedback,
) -> ProfilePreferences:
    preferred = dedupe_preserve_order(
        [*existing.preferred_source_patterns, *feedback.preferred_source_patterns]
    )
    avoided = dedupe_preserve_order(
        [*existing.avoided_source_patterns, *feedback.avoided_source_patterns]
    )
    return existing.model_copy(
        update={
            "preferred_source_patterns": preferred,
            "avoided_source_patterns": avoided,
            "answer_style_bias": feedback.answer_style_bias or existing.answer_style_bias,
            "include_counterevidence_bias": (
                feedback.include_counterevidence_bias
                if feedback.include_counterevidence_bias is not None
                else existing.include_counterevidence_bias
            ),
        }
    )


def query_hints_from_pack(pack: ContextPack | None) -> list[str]:
    if pack is None:
        return []
    hints: list[str] = []
    for fragment in pack.fragments:
        suggested = fragment.metadata.get("suggested_queries")
        if isinstance(suggested, list):
            hints.extend(str(item) for item in suggested if item)
    return dedupe_preserve_order(hints)


def provider_hints_from_pack(pack: ContextPack | None) -> list[str]:
    if pack is None:
        return []
    providers: list[str] = []
    for fragment in pack.fragments:
        suggested = fragment.metadata.get("providers")
        if isinstance(suggested, list):
            providers.extend(str(item) for item in suggested if item)
    counts = Counter(providers)
    return [provider for provider, _ in counts.most_common()]
