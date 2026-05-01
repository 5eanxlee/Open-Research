from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import orjson
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from .artifacts import ArtifactPayload, ArtifactStore
from .citations import (
    CitationCandidate,
    audit_citation_candidates,
    build_citation_key,
)
from .config import Settings
from .db import ResearchStore
from .domain import (
    AgentConfig,
    BudgetPolicy,
    BudgetRecommendationRationale,
    CitationAuditDecision,
    CitationRecord,
    CitationSupportLabel,
    ClaimVerification,
    ContextPack,
    ContextPhase,
    DraftReport,
    ExecutionMode,
    FetchedDocument,
    FinalReport,
    GapAnalysis,
    MemoryInfluencePolicy,
    ModelConfig,
    NoteDraft,
    PlanningArtifact,
    PlanningDiscoveryRecord,
    PlanningStage,
    PlanPreview,
    RecommendedBudget,
    ReportSection,
    ResearchAssetRecord,
    ResearchAssetType,
    ResearchAssetUsage,
    ResearchPlan,
    ResearchStreamPlan,
    ResearchStreamView,
    RetrievalMethod,
    RetrievedPassage,
    RunStatus,
    SourceKind,
    SourceTrustTier,
    StreamStatus,
    TaskStatus,
    resolve_model_config,
)
from .events import RunEventService
from .memory import (
    BehaviorJudge,
    ContextAssembler,
    MemoryCompiler,
    query_hints_from_pack,
    render_context_pack,
)
from .observability import ResearchTelemetry
from .prompting import (
    SOURCE_TRUST_POLICY_VERSION,
    assess_source_trust,
    build_source_prompt_context,
    claim_verifier_system_prompt,
    note_writer_system_prompt,
    planner_system_prompt,
    report_writer_system_prompt,
    resolve_agent_config,
)
from .providers import (
    EmbeddingProvider,
    FetchProvider,
    GenerationResult,
    OpenAIJsonClient,
    PassageReranker,
    SearchProvider,
    UsageInfo,
)
from .tool_registry import (
    FETCH_BUDGET_CATEGORIES,
    FETCH_TOOL,
    SEARCH_BUDGET_CATEGORIES,
    SEARCH_TOOL,
    assert_tool_budget_available,
)
from .utils import (
    chunk_text,
    clean_text,
    dedupe_preserve_order,
    derive_conversation_topic,
    derive_report_title,
    domain_for_url,
    extract_sentences,
    normalize_url,
    sanitize_source_snippet_for_url,
    strip_markdown_fences,
    tokenize,
)


class Planner(ABC):
    @abstractmethod
    async def create_plan(
        self,
        question: str,
        budget: BudgetPolicy,
        *,
        planning_stage: PlanningStage = PlanningStage.EXECUTION,
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
        prior_notes: Sequence[dict[str, Any]] | None = None,
        context_pack: ContextPack | None = None,
        replan_count: int = 0,
        approved_plan: ResearchPlan | None = None,
        available_documents: Sequence[ResearchAssetRecord] | None = None,
        source_selection: Sequence[str] | None = None,
        min_total_sources_retrieved: int = 0,
        min_total_cited_sources: int = 0,
    ) -> GenerationResult[ResearchPlan]:
        raise NotImplementedError


class NoteWriter(ABC):
    @abstractmethod
    async def write_note(
        self,
        *,
        question: str,
        stream: ResearchStreamPlan,
        document: FetchedDocument,
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
        context_pack: ContextPack | None = None,
    ) -> GenerationResult[NoteDraft]:
        raise NotImplementedError


class GapAnalyzer(ABC):
    @abstractmethod
    async def analyze(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        notes: Sequence[dict[str, Any]],
        budget: BudgetPolicy,
        replan_count: int,
        model_config: ModelConfig | None = None,
    ) -> GenerationResult[GapAnalysis]:
        raise NotImplementedError


class ReportWriter(ABC):
    @abstractmethod
    async def write_report(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        notes: Sequence[dict[str, Any]],
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
        context_pack: ContextPack | None = None,
        output_contract: dict[str, Any] | None = None,
    ) -> GenerationResult[DraftReport]:
        raise NotImplementedError


class ClaimVerifier(ABC):
    @abstractmethod
    async def verify(
        self,
        *,
        claim: str,
        candidates: Sequence[RetrievedPassage],
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
    ) -> GenerationResult[ClaimVerification]:
        raise NotImplementedError


class RunCancelledError(RuntimeError):
    pass


_LOW_SIGNAL_QUERY_TOKENS = {
    "about",
    "best",
    "between",
    "compare",
    "concise",
    "collect",
    "does",
    "establish",
    "explain",
    "find",
    "focus",
    "identify",
    "keep",
    "local",
    "practical",
    "research",
    "rules",
    "show",
    "single",
    "short",
    "should",
    "systems",
    "user",
    "what",
    "when",
    "with",
}
_TRUST_DOMAIN_HINTS = (
    "docs.",
    ".gov",
    ".edu",
    "arxiv.org",
    "ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "journals.plos.org",
)
_PRIMARY_RECORD_DOMAIN_HINTS = (
    "arxiv.org/abs/",
    "openreview.net/forum",
    "doi.org/",
)
_SOFT_PENALTY_HINTS = ("community", "newsletter", "weekly", "blog", "dev.to")
_OFF_TOPIC_PENALTY_HINTS = (
    "advertisement",
    "adverse conditions",
    "clawcloud",
    "content marketing",
    "disaster",
    "ecommerce",
    "faq",
    "marketing",
    "openclaw",
    "on-site search",
    "revenue",
    "seo",
    "search service resilience",
    "storm",
    "weather",
)
_BROAD_QUERY_TOKEN_EXCLUSIONS = {
    "question",
    "report",
    "research",
    "search",
    "source",
    "system",
}


def _default_model_config(settings: Settings) -> ModelConfig:
    return ModelConfig(
        lead_model=settings.lead_model,
        planner_model=settings.planner_model,
        worker_model=settings.worker_model,
        verifier_model=settings.verifier_model,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )


def _ordered_informative_query_tokens(
    text: str,
    *,
    allow_low_signal: bool = False,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_token in re.findall(r"[A-Za-z0-9]{3,}", text):
        token = raw_token.lower()
        short_but_informative = raw_token.isupper() or any(char.isdigit() for char in raw_token)
        if len(token) < 4 and not short_but_informative:
            continue
        if (not allow_low_signal and token in _LOW_SIGNAL_QUERY_TOKENS) or (
            len(token) < 4 and not short_but_informative
        ):
            continue
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _informative_query_tokens(text: str) -> set[str]:
    return set(_ordered_informative_query_tokens(text))


def _priority_query_tokens(text: str) -> set[str]:
    return {
        token
        for token in _informative_query_tokens(text)
        if (len(token) >= 5 or token in {"fail", "fetch"})
        and token not in _BROAD_QUERY_TOKEN_EXCLUSIONS
    }


def _query_hint_overlap(question: str, hint: str) -> float:
    question_tokens = _informative_query_tokens(question)
    hint_tokens = _informative_query_tokens(hint)
    if not question_tokens or not hint_tokens:
        return 0.0
    return len(question_tokens & hint_tokens) / max(1, min(len(question_tokens), len(hint_tokens)))


def _query_scope_relevance(
    *,
    question: str,
    query: str,
    objective: str | None = None,
) -> float:
    scope_text = " ".join(part for part in (question, objective or "") if part)
    scope_tokens = _informative_query_tokens(scope_text)
    query_tokens = _informative_query_tokens(query)
    if not scope_tokens or not query_tokens:
        return 0.0
    overlap = len(scope_tokens & query_tokens) / max(1, min(len(scope_tokens), len(query_tokens)))
    scope_priority_tokens = _priority_query_tokens(scope_text)
    priority_overlap = len(scope_priority_tokens & query_tokens)
    exact_bonus = 0.15 if clean_text(question).lower() in clean_text(query).lower() else 0.0
    off_topic_penalty = (
        -0.28 if any(hint in clean_text(query).lower() for hint in _OFF_TOPIC_PENALTY_HINTS) else 0.0
    )
    return round(
        overlap + min(0.25, 0.1 * priority_overlap) + exact_bonus + off_topic_penalty,
        4,
    )


def _compact_search_query(
    text: str,
    *,
    objective: str | None = None,
    extra_terms: Sequence[str] = (),
    max_tokens: int = 12,
    blend_objective_tokens: bool = True,
) -> str:
    candidate_tokens = _ordered_informative_query_tokens(text)
    if not candidate_tokens and objective:
        candidate_tokens = _ordered_informative_query_tokens(objective)
    objective_tokens: list[str] = []
    if blend_objective_tokens:
        objective_tokens = [
            token
            for token in _ordered_informative_query_tokens(objective or "")
            if token not in candidate_tokens
        ]
    extra_tokens = dedupe_preserve_order(
        token
        for term in extra_terms
        for token in _ordered_informative_query_tokens(
            clean_text(term).lower(),
            allow_low_signal=True,
        )
    )
    ordered: list[str] = []
    seen: set[str] = set()

    def append_tokens(
        tokens: Sequence[str],
        *,
        allow_low_signal: bool = False,
        limit: int | None = None,
    ) -> None:
        added = 0
        for token in tokens:
            if len(ordered) >= max_tokens:
                return
            if not token:
                continue
            if not allow_low_signal and token in _LOW_SIGNAL_QUERY_TOKENS:
                continue
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
            added += 1
            if limit is not None and added >= limit:
                return

    reserved_extra_slots = 0
    if extra_tokens:
        reserved_extra_slots = min(len(extra_tokens), max(1, min(3, max_tokens // 3 or 1)))
    base_limit = max(1, max_tokens - reserved_extra_slots)
    append_tokens(candidate_tokens, limit=base_limit)
    append_tokens(extra_tokens, allow_low_signal=True, limit=reserved_extra_slots or None)
    append_tokens(extra_tokens, allow_low_signal=True)
    append_tokens(objective_tokens)
    append_tokens(candidate_tokens)
    if not ordered:
        fallback = clean_text(" ".join(part for part in (text, objective or "") if part) or "research")
        return fallback[:120].strip()
    return " ".join(ordered)


def _relevant_query_hints(question: str, context_pack: ContextPack | None) -> list[str]:
    hints = query_hints_from_pack(context_pack)
    if not hints:
        return []
    return [
        hint
        for hint in hints
        if _query_scope_relevance(question=question, query=hint) >= 0.42
        or clean_text(question).lower() in clean_text(hint).lower()
    ]


def _default_stream_queries(
    *,
    question: str,
    objective: str,
    stream_name: str,
    max_queries: int,
) -> list[str]:
    seed = _compact_search_query(question, objective=objective)
    if stream_name == "Core facts":
        defaults = [
            seed,
            _compact_search_query(
                question,
                objective=objective,
                extra_terms=("official", "documentation"),
            ),
            _compact_search_query(
                question,
                objective=objective,
                extra_terms=("primary", "source"),
            ),
        ]
    elif stream_name == "Recent developments":
        defaults = [
            _compact_search_query(
                question,
                objective=objective,
                extra_terms=("recent", "changes"),
            ),
            _compact_search_query(
                question,
                objective=objective,
                extra_terms=("release", "notes"),
            ),
            _compact_search_query(
                question,
                objective=objective,
                extra_terms=("latest", "official", "update"),
            ),
        ]
    elif stream_name == "Cross-checks":
        defaults = [
            _compact_search_query(
                question,
                objective=objective,
                extra_terms=("comparison", "analysis"),
            ),
            _compact_search_query(
                question,
                objective=objective,
                extra_terms=("technical", "blog"),
            ),
            _compact_search_query(
                question,
                objective=objective,
                extra_terms=("evidence", "corroboration"),
            ),
        ]
    else:
        defaults = [
            seed,
            _compact_search_query(
                objective,
                extra_terms=("official", "documentation"),
            ),
            _compact_search_query(
                objective,
                extra_terms=("evidence",),
            ),
        ]
    return dedupe_preserve_order(defaults)[:max_queries]


def _normalize_scope_queries(
    *,
    question: str,
    objective: str,
    queries: Sequence[str],
    stream_name: str,
    max_queries: int,
) -> list[str]:
    normalized: list[str] = []
    minimum_relevance = (
        0.42 if stream_name in {"Core facts", "Recent developments", "Cross-checks"} else 0.3
    )
    for query in queries:
        compact = _compact_search_query(
            query,
            objective=objective,
            blend_objective_tokens=False,
        )
        if not compact:
            continue
        if (
            _query_scope_relevance(
                question=question,
                query=compact,
                objective=objective,
            )
            < minimum_relevance
        ):
            continue
        normalized.append(compact)
    normalized = dedupe_preserve_order(normalized)
    if len(normalized) >= max_queries:
        return normalized[:max_queries]
    for fallback in _default_stream_queries(
        question=question,
        objective=objective,
        stream_name=stream_name,
        max_queries=max_queries,
    ):
        if fallback not in normalized:
            normalized.append(fallback)
        if len(normalized) >= max_queries:
            break
    return normalized[:max_queries]


def _search_result_relevance(
    *,
    query: str,
    result: Any,
    stream_name: str | None = None,
) -> float:
    query_tokens = _informative_query_tokens(query)
    if not query_tokens:
        return 0.0
    priority_tokens = _priority_query_tokens(query)
    title = clean_text(str(getattr(result, "title", "") or ""))
    url = str(getattr(result, "url", "") or "")
    snippet = sanitize_source_snippet_for_url(
        url=url,
        snippet=str(getattr(result, "snippet", "") or ""),
    )
    title_tokens = set(tokenize(title))
    snippet_tokens = set(tokenize(snippet))
    url_tokens = set(tokenize(url))
    combined_tokens = title_tokens | snippet_tokens | url_tokens
    overlap = len(query_tokens & combined_tokens) / max(1, len(query_tokens))
    title_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
    snippet_overlap = len(query_tokens & snippet_tokens) / max(1, len(query_tokens))
    priority_overlap = len(priority_tokens & combined_tokens)
    title_priority_overlap = len(priority_tokens & title_tokens)
    provider_score = max(0.0, min(1.0, float(getattr(result, "score", 0.0) or 0.0) / 5.0))
    lowered_title = title.lower()
    lowered_snippet = snippet.lower()
    lowered_url = url.lower()
    lowered_combined = " ".join(
        part for part in (lowered_title, lowered_snippet, lowered_url) if part
    )
    trust_bonus = 0.2 if any(hint in lowered_url for hint in _TRUST_DOMAIN_HINTS) else 0.0
    is_primary_record = any(hint in lowered_url for hint in _PRIMARY_RECORD_DOMAIN_HINTS)
    asks_for_primary_record = bool(
        query_tokens & {"arxiv", "official", "paper", "papers", "primary", "source"}
    )
    if is_primary_record and asks_for_primary_record and provider_score >= 0.6:
        trust_bonus += 0.3
    if "official" in lowered_title or "/docs" in lowered_url or "documentation" in lowered_title:
        trust_bonus += 0.08
    soft_penalty = 0.0
    if any(hint in lowered_title or hint in lowered_url for hint in _SOFT_PENALTY_HINTS):
        soft_penalty -= 0.18
    if stream_name == "Core facts" and ("weekly" in lowered_title or "newsletter" in lowered_title):
        soft_penalty -= 0.12
    if any(hint in lowered_combined for hint in _OFF_TOPIC_PENALTY_HINTS):
        soft_penalty -= 0.22
    if priority_tokens:
        if priority_overlap == 0:
            soft_penalty -= 0.08 if is_primary_record and asks_for_primary_record else 0.24
        elif priority_overlap == 1:
            trust_bonus += 0.04
        else:
            trust_bonus += min(0.16, 0.07 * priority_overlap)
        if title_priority_overlap == 0 and priority_overlap > 0:
            soft_penalty -= 0.04
    return round(
        (0.48 * overlap)
        + (0.24 * title_overlap)
        + (0.12 * snippet_overlap)
        + (0.08 * provider_score)
        + trust_bonus
        + soft_penalty,
        4,
    )


def _interleave_results_by_query(
    results: Sequence[Any],
    result_query_order: Mapping[str, int],
) -> list[Any]:
    buckets: dict[int, list[Any]] = {}
    fallback_order = max(result_query_order.values(), default=-1) + 1
    for result in results:
        canonical_url = normalize_url(str(getattr(result, "url", "") or ""))
        query_order = result_query_order.get(canonical_url, fallback_order)
        buckets.setdefault(query_order, []).append(result)
    if len(buckets) <= 1:
        return list(results)
    interleaved: list[Any] = []
    max_bucket_size = max(len(bucket) for bucket in buckets.values())
    for offset in range(max_bucket_size):
        for query_order in sorted(buckets):
            bucket = buckets[query_order]
            if offset < len(bucket):
                interleaved.append(bucket[offset])
    return interleaved


def _select_results_for_fetch(
    results: Sequence[Any],
    result_query_order: Mapping[str, int],
    *,
    max_sources: int,
    per_domain_limit: int,
) -> list[Any]:
    buckets: dict[int, list[Any]] = {}
    fallback_order = max(result_query_order.values(), default=-1) + 1
    for result in results:
        canonical_url = normalize_url(str(getattr(result, "url", "") or ""))
        query_order = result_query_order.get(canonical_url, fallback_order)
        buckets.setdefault(query_order, []).append(result)
    if not buckets:
        return []

    selected: list[Any] = []
    seen_urls: set[str] = set()
    domain_counts: Counter[str] = Counter()
    positions = {query_order: 0 for query_order in buckets}
    max_sources = max(0, max_sources)
    per_domain_limit = max(1, per_domain_limit)
    target_query_coverage = min(len(buckets), max_sources)
    covered_queries: set[int] = set()

    while len(selected) < max_sources:
        progressed = False
        for query_order in sorted(buckets):
            bucket = buckets[query_order]
            while positions[query_order] < len(bucket):
                candidate = bucket[positions[query_order]]
                positions[query_order] += 1
                canonical_url = normalize_url(str(getattr(candidate, "url", "") or ""))
                if not canonical_url or canonical_url in seen_urls:
                    continue
                domain = domain_for_url(canonical_url)
                needs_query_coverage = (
                    query_order not in covered_queries
                    and len(covered_queries) < target_query_coverage
                )
                if domain_counts[domain] >= per_domain_limit and not needs_query_coverage:
                    continue
                seen_urls.add(canonical_url)
                domain_counts[domain] += 1
                covered_queries.add(query_order)
                selected.append(candidate)
                progressed = True
                break
            if len(selected) >= max_sources:
                break
        if not progressed:
            break
    return selected


def _claim_repair_should_skip_search(claim: str) -> bool:
    lowered = clean_text(claim).lower()
    uncertainty_markers = (
        "does not establish",
        "do not establish",
        "does not verify",
        "do not verify",
        "insufficient evidence",
        "not established",
        "not verified",
        "remain adoption risks",
        "remains unresolved",
        "remain unresolved",
        "the supplied evidence does not",
        "the retrieved evidence does not",
        "the available evidence does not",
        "the notes do not",
    )
    return any(marker in lowered for marker in uncertainty_markers)


def _absence_claim_contradicted_by_passage(claim: str, passage: str) -> bool:
    lowered_claim = clean_text(claim).lower()
    if not any(
        marker in lowered_claim
        for marker in (
            "no benchmark",
            "no quantitative",
            "no ablation",
            "no evaluation",
            "no metric",
            "no retrieved fragment provides",
            "does not include",
            "does not provide",
            "do not include",
            "do not provide",
            "not include",
            "not provide",
        )
    ):
        return False
    if not any(
        subject in lowered_claim
        for subject in (
            "benchmark",
            "quantitative",
            "ablation",
            "metric",
            "evaluation",
            "result",
            "score",
        )
    ):
        return False
    lowered_passage = clean_text(passage).lower()
    counterevidence_terms = (
        "benchmark",
        "ablation",
        "baseline",
        "dataset",
        "accuracy",
        "score",
        "improvement",
        "latency",
        "token-cost",
        "cost savings",
        "locomo",
        "hotpotqa",
        "wikimultihop",
        "longmemevals",
    )
    if not any(term in lowered_passage for term in counterevidence_terms):
        return False
    return bool(re.search(r"\d+(?:\.\d+)?\s*%|\b\d+\.\d+\b|\b\d+\s*x\b", lowered_passage))


def _derive_recommended_budget(
    *,
    question: str,
    budget: BudgetPolicy,
    stream_count: int,
    replan_count: int,
    prior_notes: Sequence[dict[str, Any]] | None = None,
) -> tuple[RecommendedBudget, BudgetRecommendationRationale, list[str], ExecutionMode]:
    lowered = clean_text(question).lower()
    complexity_factors: list[str] = []
    if any(token in lowered for token in ("compare", "vs", "versus", "tradeoff", "benchmark")):
        complexity_factors.append("comparative_analysis")
    if any(token in lowered for token in ("latest", "recent", "today", "current", "news")):
        complexity_factors.append("recency_sensitive")
    if any(token in lowered for token in ("risk", "failure", "security", "compliance", "audit")):
        complexity_factors.append("adversarial_checks")
    if any(token in lowered for token in ("global", "regional", "enterprise", "stakeholder")):
        complexity_factors.append("multi_axis_scope")
    if len(lowered.split()) > 18:
        complexity_factors.append("broad_question")
    unresolved = [
        item
        for note in (prior_notes or [])
        for item in note.get("open_questions", [])
        if item
    ]
    if unresolved:
        complexity_factors.append("open_gaps")

    recommended_streams = min(
        budget.max_streams,
        max(stream_count, 3 + len(complexity_factors)),
    )
    recommended_queries = min(
        budget.max_queries_per_stream,
        max(3, 3 + min(len(complexity_factors), 4)),
    )
    recommended_sources = min(
        budget.max_sources_per_stream,
        max(3, 2 + min(len(complexity_factors), 3)),
    )
    execution_mode = (
        ExecutionMode.DEEP if len(complexity_factors) >= 2 or recommended_streams >= 6 else ExecutionMode.STANDARD
    )
    rationale = BudgetRecommendationRationale(
        summary=(
            "Recommended budget expands when the question needs comparison, recency, "
            "adversarial checks, or wider source diversity."
        ),
        coverage_axes=list(complexity_factors),
        evidence_gaps=dedupe_preserve_order(unresolved)[:5],
        source_diversity_reasoning=(
            "Multiple streams and higher source caps are warranted when disagreement or "
            "multi-axis coverage is likely."
        ),
        grounding_difficulty=(
            "Higher complexity increases the chance that claims need narrower evidence and "
            "more retrieval candidates."
        ),
    )
    recommended_budget = RecommendedBudget(
        max_streams=recommended_streams,
        max_replans=min(budget.max_replans, max(1 if unresolved else 0, replan_count)),
        max_queries_per_stream=recommended_queries,
        max_results_per_query=budget.max_results_per_query,
        max_sources_per_stream=recommended_sources,
        per_domain_limit=min(budget.per_domain_limit, 3),
        rationale_summary=rationale.summary,
    )
    return recommended_budget, rationale, complexity_factors, execution_mode


def _summarize_available_documents(
    assets: Sequence[ResearchAssetRecord] | None,
) -> list[str]:
    summaries: list[str] = []
    for asset in assets or []:
        location = "project corpus" if asset.project_id else "run attachment"
        descriptor = f"{asset.label} ({location}, {asset.usage.value})"
        if asset.description:
            descriptor += f": {asset.description}"
        elif asset.preview_excerpt:
            descriptor += f": {asset.preview_excerpt}"
        summaries.append(descriptor)
    return summaries


def _build_discovery_queries(
    *,
    question: str,
    budget: BudgetPolicy,
    approved_plan: ResearchPlan | None,
    available_documents: Sequence[ResearchAssetRecord] | None,
    max_queries: int,
) -> list[str]:
    queries = [_compact_search_query(question)]
    if approved_plan is not None:
        for stream in approved_plan.streams:
            compact_objective = _compact_search_query(stream.objective, objective=question)
            if compact_objective and compact_objective not in queries:
                queries.append(compact_objective)
            for query in stream.queries:
                compact_query = _compact_search_query(
                    query,
                    objective=stream.objective,
                    blend_objective_tokens=False,
                )
                if (
                    compact_query
                    and _query_scope_relevance(
                        question=question,
                        query=compact_query,
                        objective=stream.objective,
                    )
                    >= 0.42
                    and compact_query not in queries
                ):
                    queries.append(compact_query)
    document_terms = [
        asset.label
        for asset in available_documents or []
        if asset.usage == ResearchAssetUsage.PLANNING_CONTEXT
    ]
    if document_terms:
        queries.append(
            _compact_search_query(
                question,
                extra_terms=document_terms[:3],
            )
        )
    discovery_angles = [
        ("official", "documentation", "source"),
        ("recent", "developments", "current"),
        ("disagreement", "analysis", "critique"),
        ("comparison", "alternatives", "tradeoffs"),
        ("evidence", "data", "benchmark"),
        ("risks", "limitations", "failure"),
        ("implementation", "architecture", "production"),
        ("case", "study", "deployment"),
        ("metrics", "evaluation", "quality"),
        ("standards", "guidance", "policy"),
        ("roadmap", "changelog", "updates"),
        ("expert", "review", "consensus"),
    ]
    queries.extend(
        _compact_search_query(question, extra_terms=angle)
        for angle in discovery_angles
    )
    return dedupe_preserve_order(queries)[: max(1, max_queries)]


def _derive_source_floor_targets(
    *,
    budget: BudgetPolicy,
    stream_count: int,
    minimum_retrieved: int,
    minimum_cited: int,
) -> tuple[int, int]:
    max_possible_sources = max(1, budget.max_streams * budget.max_sources_per_stream)
    retrieved_target = max(stream_count * 2, minimum_retrieved)
    retrieved_floor = min(max_possible_sources, max(1, retrieved_target))
    cited_floor = min(
        retrieved_floor,
        max(1, max(minimum_cited, min(max(stream_count, 3), retrieved_floor))),
    )
    return retrieved_floor, cited_floor


def _build_planning_artifact(
    *,
    stage: PlanningStage,
    question: str,
    plan: ResearchPlan,
    available_documents: Sequence[ResearchAssetRecord] | None,
    discovery_records: Sequence[PlanningDiscoveryRecord],
    source_selection: Sequence[str] | None,
    min_total_sources_retrieved: int,
    min_total_cited_sources: int,
    approved_preview_version: int | None = None,
) -> PlanningArtifact:
    toc = dedupe_preserve_order(
        [stream.name for stream in plan.streams]
        + list(plan.success_criteria)
    )[:8]
    constraints = dedupe_preserve_order(
        [
            "Respect runtime budget caps and source selection.",
            "Use uploaded planning context to shape stream priorities.",
            "Maintain enough source diversity for later grounding and citation audit.",
            *(
                plan.budget_rationale.coverage_axes
                if plan.budget_rationale is not None
                else []
            ),
        ]
    )[:10]
    deliverables = dedupe_preserve_order(
        [
            "Validated execution plan",
            "Parallel research streams",
            "Grounded report sections",
            "Audited citations",
            *plan.success_criteria,
        ]
    )[:10]
    key_questions = dedupe_preserve_order(
        [question, *plan.complexity_factors]
        + [stream.objective for stream in plan.streams]
    )[:12]
    validation_checks = [
        "Each stream has a distinct objective and at least one executable query.",
        "Planning artifact reflects uploaded documents and source constraints.",
        "Discovery evidence supports the chosen coverage axes.",
        "Recommended budget is consistent with stream breadth and query depth.",
        "Source floors are feasible under runtime caps.",
    ]
    return PlanningArtifact(
        stage=stage,
        approved_preview_version=approved_preview_version,
        task_breakdown=(
            "Search-before-plan discovery, stream design, query packaging, and validation "
            "before research execution."
        ),
        table_of_contents=toc,
        constraints=constraints,
        planned_deliverables=deliverables,
        key_questions=key_questions,
        available_documents=_summarize_available_documents(available_documents),
        discovery_queries=[record.query for record in discovery_records],
        discovery_records=list(discovery_records),
        source_selection=list(source_selection or []),
        min_total_sources_retrieved=min_total_sources_retrieved,
        min_total_cited_sources=min_total_cited_sources,
        validation_checks=validation_checks,
    )


def _bound_streams_to_budget(
    streams: Sequence[ResearchStreamPlan],
    *,
    question: str,
    budget: BudgetPolicy,
    worker_model: str,
    stream_limit: int | None = None,
) -> list[ResearchStreamPlan]:
    bounded_streams: list[ResearchStreamPlan] = []
    max_streams = min(budget.max_streams, stream_limit or budget.max_streams)
    for stream in streams[:max_streams]:
        queries = dedupe_preserve_order([query.strip() for query in stream.queries if query.strip()])
        bounded_queries = _normalize_scope_queries(
            question=question,
            objective=stream.objective,
            queries=queries,
            stream_name=stream.name,
            max_queries=budget.max_queries_per_stream,
        )
        if not bounded_queries:
            fallback_query = clean_text(stream.objective) or clean_text(stream.name) or "research"
            bounded_queries = [fallback_query]
        bounded_streams.append(
            stream.model_copy(
                update={
                    "queries": bounded_queries,
                    "model": stream.model or worker_model,
                }
            )
        )
    return bounded_streams


def _validate_research_plan(
    *,
    plan: ResearchPlan,
    budget: BudgetPolicy,
    stage: PlanningStage,
    min_total_sources_retrieved: int,
    min_total_cited_sources: int,
) -> list[str]:
    issues: list[str] = []
    if not plan.summary.strip():
        issues.append("Plan summary is empty.")
    if not plan.hypothesis.strip():
        issues.append("Plan hypothesis is empty.")
    if not plan.streams:
        issues.append("Plan must contain at least one stream.")
    seen_stream_names: set[str] = set()
    for stream in plan.streams:
        if not stream.name.strip():
            issues.append("Every stream must have a name.")
        if stream.name in seen_stream_names:
            issues.append(f"Duplicate stream name: {stream.name}")
        seen_stream_names.add(stream.name)
        if not stream.objective.strip():
            issues.append(f"Stream {stream.name!r} is missing an objective.")
        if not stream.queries:
            issues.append(f"Stream {stream.name!r} must include at least one query.")
        if len(stream.queries) > budget.max_queries_per_stream:
            issues.append(
                f"Stream {stream.name!r} exceeds max_queries_per_stream={budget.max_queries_per_stream}."
            )
    if len(plan.streams) > budget.max_streams:
        issues.append(f"Plan exceeds max_streams={budget.max_streams}.")
    if len(plan.success_criteria) < 2:
        issues.append("Plan should declare at least two success criteria.")
    artifact = plan.planning_artifact
    if artifact is None:
        issues.append("Plan is missing a planning artifact.")
    else:
        if artifact.stage != stage:
            issues.append(
                f"Planning artifact stage {artifact.stage.value!r} does not match {stage.value!r}."
            )
        if stage in {PlanningStage.EXECUTION, PlanningStage.REPLAN}:
            if not artifact.discovery_queries:
                issues.append("Execution planning must record discovery queries.")
            if not artifact.discovery_records:
                issues.append("Execution planning must record discovery findings.")
            if not artifact.constraints:
                issues.append("Execution planning must include constraints.")
            if not artifact.table_of_contents:
                issues.append("Execution planning must include a table of contents.")
        if artifact.min_total_sources_retrieved < min_total_sources_retrieved:
            issues.append("Planning artifact undershoots the minimum retrieved-source floor.")
        if artifact.min_total_cited_sources < min_total_cited_sources:
            issues.append("Planning artifact undershoots the minimum cited-source floor.")
    return dedupe_preserve_order(issues)


class HeuristicPlanner(Planner):
    def __init__(self, *, worker_model: str, max_streams: int) -> None:
        self.worker_model = worker_model
        self.max_streams = max_streams

    async def create_plan(
        self,
        question: str,
        budget: BudgetPolicy,
        *,
        planning_stage: PlanningStage = PlanningStage.EXECUTION,
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
        prior_notes: Sequence[dict[str, Any]] | None = None,
        context_pack: ContextPack | None = None,
        replan_count: int = 0,
        approved_plan: ResearchPlan | None = None,
        available_documents: Sequence[ResearchAssetRecord] | None = None,
        source_selection: Sequence[str] | None = None,
        min_total_sources_retrieved: int = 0,
        min_total_cited_sources: int = 0,
    ) -> GenerationResult[ResearchPlan]:
        prior_notes = prior_notes or []
        core = clean_text(question.rstrip("?"))
        worker_model = model_config.worker_model if model_config is not None else self.worker_model
        streams: list[ResearchStreamPlan] = [
            ResearchStreamPlan(
                name="Core facts",
                objective=(f"Establish the baseline facts and definitions relevant to: {core}."),
                queries=_default_stream_queries(
                    question=question,
                    objective=f"Establish the baseline facts and definitions relevant to: {core}.",
                    stream_name="Core facts",
                    max_queries=budget.max_queries_per_stream,
                ),
                model=worker_model,
            ),
            ResearchStreamPlan(
                name="Recent developments",
                objective=(
                    f"Find recent developments, updates, or changed assumptions about: {core}."
                ),
                queries=_default_stream_queries(
                    question=question,
                    objective=f"Find recent developments, updates, or changed assumptions about: {core}.",
                    stream_name="Recent developments",
                    max_queries=budget.max_queries_per_stream,
                ),
                model=worker_model,
            ),
            ResearchStreamPlan(
                name="Cross-checks",
                objective=(
                    "Collect corroborating sources and identify disagreements or open "
                    f"questions for: {core}."
                ),
                queries=_default_stream_queries(
                    question=question,
                    objective=(
                        "Collect corroborating sources and identify disagreements or open "
                        f"questions for: {core}."
                    ),
                    stream_name="Cross-checks",
                    max_queries=budget.max_queries_per_stream,
                ),
                model=worker_model,
            ),
        ]

        unresolved = dedupe_preserve_order(
            question
            for note in prior_notes
            for question in note.get("open_questions", [])
            if question
        )
        if unresolved and replan_count < budget.max_replans:
            streams.append(
                ResearchStreamPlan(
                    name=f"Gap closure {replan_count + 1}",
                    objective=(
                        "Close the highest-value unresolved questions from the first research pass."
                    ),
                    queries=unresolved[: budget.max_queries_per_stream],
                    model=worker_model,
                )
            )

        streams = _bound_streams_to_budget(
            streams,
            question=question,
            budget=budget,
            worker_model=worker_model,
            stream_limit=self.max_streams,
        )
        recommended_budget, budget_rationale, complexity_factors, execution_mode = (
            _derive_recommended_budget(
                question=question,
                budget=budget,
                stream_count=len(streams),
                replan_count=replan_count,
                prior_notes=prior_notes,
            )
        )
        retrieved_floor, cited_floor = _derive_source_floor_targets(
            budget=budget,
            stream_count=len(streams),
            minimum_retrieved=max(4, min_total_sources_retrieved),
            minimum_cited=max(3, min_total_cited_sources),
        )
        artifact = _build_planning_artifact(
            stage=planning_stage,
            question=question,
            plan=ResearchPlan(
                summary="placeholder",
                hypothesis="placeholder",
                streams=streams,
                success_criteria=[
                    "Every section should be supported by multiple grounded notes.",
                    "The final report should surface remaining uncertainty explicitly.",
                    "Recent or changed facts should be isolated from background context.",
                ],
                recommended_budget=recommended_budget,
                budget_rationale=budget_rationale,
                recommended_execution_mode=execution_mode,
                approval_required=execution_mode == ExecutionMode.DEEP,
                complexity_factors=complexity_factors,
            ),
            available_documents=available_documents,
            discovery_records=[
                PlanningDiscoveryRecord(
                    query=query,
                    provider="heuristic",
                    result_count=0,
                    summary="Heuristic planner seeded this query for later execution.",
                )
                for query in _build_discovery_queries(
                    question=question,
                    budget=budget,
                    approved_plan=approved_plan,
                    available_documents=available_documents,
                    max_queries=max(10, min(16, budget.max_queries_per_stream)),
                )
            ]
            if planning_stage != PlanningStage.PREVIEW
            else [],
            source_selection=source_selection,
            min_total_sources_retrieved=retrieved_floor,
            min_total_cited_sources=cited_floor,
            approved_preview_version=approved_plan.planning_artifact.approved_preview_version
            if approved_plan and approved_plan.planning_artifact
            else None,
        )
        success_criteria = [
            "Every section should be supported by multiple grounded notes.",
            "The final report should surface remaining uncertainty explicitly.",
            "Recent or changed facts should be isolated from background context.",
        ]
        return GenerationResult(
            value=ResearchPlan(
                summary=(
                    f"Research {core} using parallel source streams with explicit cross-checking."
                ),
                hypothesis=(
                    f"A reliable answer about {core} requires both baseline facts "
                    "and current evidence."
                ),
                streams=streams,
                success_criteria=success_criteria,
                planning_artifact=artifact,
                recommended_budget=recommended_budget,
                budget_rationale=budget_rationale,
                recommended_execution_mode=execution_mode,
                approval_required=execution_mode == ExecutionMode.DEEP,
                complexity_factors=complexity_factors,
            ),
            usage=UsageInfo(),
    )


def _clip_local_prompt(value: str | None, *, limit: int) -> str:
    if not value:
        return ""
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1].rstrip() + "…"


_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_HEADING_BREAK_RE = re.compile(r"\s+(#{1,6}\s+)")
_INLINE_CHROME_BREAK_RE = re.compile(
    r"\s+(Home›|Home ›|Sitemap|Open in app|Sign up|Sign in|Member-only story|Share on Twitter|Share on LinkedIn|Share on Mastodon|Read\s+\*\*|The Lead|Key Takeaways|Quick Take)",
    re.IGNORECASE,
)
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")
_NAVIGATION_LINE_HINTS = (
    "add to collection",
    "all rights reserved",
    "artificial intelligence guides",
    "breadcrumb",
    "cookie policy",
    "copyright",
    "get started",
    "home\") /",
    "home›",
    "member-only story",
    "navigation",
    "open in app",
    "privacy policy",
    "share on linkedin",
    "share on mastodon",
    "share on twitter",
    "skip to content",
    "skip to main content",
    "sitemap",
    "table of contents",
    "tools & plugins",
)
_BOILERPLATE_PREFIXES = (
    "advertisement",
    "copyright",
    "cookie policy",
    "navigation",
    "published",
    "skip to",
    "table of contents",
)
_MODEL_SOURCE_MIN_CHARS = 500
_MODEL_SOURCE_MAX_CHARS = 12000
_LOW_VALUE_DOCUMENT_TITLE_HINTS = (
    "this page has moved",
    "page not found",
    "reload",
)
_LOW_VALUE_DOCUMENT_TEXT_HINTS = (
    "please reload this page",
    "reload this page to refresh your session",
    "this page has moved",
    "you switched accounts on another tab or window",
)


def _minimum_result_relevance(*, stream_name: str | None = None) -> float:
    if stream_name == "Core facts":
        return 0.24
    if stream_name == "Recent developments":
        return 0.18
    if stream_name == "Cross-checks":
        return 0.17
    return 0.16


_SOURCE_TRUST_ORDER = {
    SourceTrustTier.UNKNOWN: 0,
    SourceTrustTier.LOW: 1,
    SourceTrustTier.STANDARD: 2,
    SourceTrustTier.HIGH: 3,
    SourceTrustTier.PRIMARY: 4,
}


def _document_below_stream_trust_floor(
    document: FetchedDocument,
    *,
    agent_config: AgentConfig,
    stream_name: str,
) -> bool:
    if (
        document.retrieval_method == RetrievalMethod.MOCK
        or bool(document.metadata.get("synthetic"))
    ):
        return False
    trust_tier_raw = str(document.metadata.get("trust_tier", "") or "").strip().lower()
    if not trust_tier_raw:
        return False
    try:
        trust_tier = SourceTrustTier(trust_tier_raw)
    except ValueError:
        return False
    return _SOURCE_TRUST_ORDER[trust_tier] < _SOURCE_TRUST_ORDER[agent_config.source_trust_floor]


def _sanitize_heuristic_source_text(content: str) -> str:
    text = content or ""
    text = _FRONTMATTER_RE.sub("", text, count=1)
    text = _MARKDOWN_IMAGE_RE.sub(" ", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MARKDOWN_HEADING_BREAK_RE.sub(r"\n\1", text)
    text = _INLINE_CHROME_BREAK_RE.sub(r"\n\1", text)
    text = text.replace("**", " ")
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        is_long_collapsed_line = len(line) >= 500
        if lower.startswith(("```", "---", "| ---", "[skip to content]", "skip to content")):
            continue
        if lower.startswith(_BOILERPLATE_PREFIXES):
            continue
        if not is_long_collapsed_line and any(token in lower for token in _NAVIGATION_LINE_HINTS):
            continue
        if not is_long_collapsed_line and (lower.count(" | ") >= 2 or lower.count(" / ") >= 3):
            continue
        if not is_long_collapsed_line and (line.count("[") >= 3 or line.count("]") >= 3):
            continue
        line = line.lstrip("#>-* ").strip()
        line = re.sub(r"^read\s+", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^(the lead|key takeaways|quick take)\s+", "", line, flags=re.IGNORECASE)
        line = _WHITESPACE_RE.sub(" ", line)
        if len(line) < 35:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _sanitize_model_source_text(content: str) -> str:
    sanitized = _sanitize_heuristic_source_text(content)
    if len(sanitized) >= _MODEL_SOURCE_MIN_CHARS:
        return sanitized[:_MODEL_SOURCE_MAX_CHARS]

    text = content or ""
    text = _FRONTMATTER_RE.sub("", text, count=1)
    text = _MARKDOWN_IMAGE_RE.sub(" ", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MARKDOWN_HEADING_BREAK_RE.sub(r"\n\1", text)
    text = _INLINE_CHROME_BREAK_RE.sub(r"\n\1", text)
    text = text.replace("**", " ")

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        is_long_collapsed_line = len(line) >= 500
        if lower.startswith(("```", "---", "| ---", "[skip to content]", "skip to content")):
            continue
        if lower.startswith(_BOILERPLATE_PREFIXES):
            continue
        if not is_long_collapsed_line and any(token in lower for token in _NAVIGATION_LINE_HINTS):
            continue
        if any(token in lower for token in _OFF_TOPIC_PENALTY_HINTS):
            continue
        if not is_long_collapsed_line and (lower.count(" / ") >= 3 or lower.count(" | ") >= 3):
            continue
        line = line.lstrip("#>-* ").strip()
        line = re.sub(r"^read\s+", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^(the lead|key takeaways|quick take)\s+", "", line, flags=re.IGNORECASE)
        line = _WHITESPACE_RE.sub(" ", line)
        if len(line) < 24:
            continue
        cleaned_lines.append(line)
    fallback = "\n".join(cleaned_lines)
    if fallback:
        return fallback[:_MODEL_SOURCE_MAX_CHARS]
    return clean_text(text)[:_MODEL_SOURCE_MAX_CHARS]


def _prepare_document_for_corpus(document: FetchedDocument) -> FetchedDocument | None:
    cleaned_title = clean_text(document.title)
    lowered_title = cleaned_title.lower()
    if any(hint in lowered_title for hint in _LOW_VALUE_DOCUMENT_TITLE_HINTS):
        return None

    sanitized_content = _sanitize_model_source_text(document.content)
    lowered_content = sanitized_content.lower()
    is_mock_document = document.retrieval_method == RetrievalMethod.MOCK or bool(
        document.metadata.get("synthetic")
    )
    if len(sanitized_content) < 180:
        return None
    if (
        not is_mock_document
        and any(hint in lowered_content for hint in _LOW_VALUE_DOCUMENT_TEXT_HINTS)
        and len(sanitized_content) < 650
    ):
        return None

    metadata = dict(document.metadata)
    metadata["raw_title"] = document.title
    metadata["content_sanitized"] = True
    return document.model_copy(
        update={
            "title": cleaned_title or document.title,
            "content": sanitized_content,
            "metadata": metadata,
        }
    )


def _looks_like_fact(sentence: str) -> bool:
    candidate = clean_text(sentence)
    if len(candidate) < 45 or len(candidate) > 360:
        return False
    lowered = candidate.lower()
    if lowered.count("http") or lowered.count("www."):
        return False
    if sum(1 for char in candidate if char in "[]{}|`") > 2:
        return False
    if candidate.count(" - ") >= 3:
        return False
    if lowered.endswith("vs.") or lowered.endswith("vs"):
        return False
    if ":" in candidate and candidate[-1] not in ".!?":
        return False
    if ":" in candidate and not any(
        f" {token} " in f" {lowered} "
        for token in ("is", "are", "was", "were", "can", "should", "has", "have")
    ):
        return False
    bad_prefixes = (
        "published",
        "updated",
        "latest articles",
        "content and translation",
        "add to collection",
        "search all of the site's content",
        "table of contents",
        "what can we help you find",
        "home /",
    )
    if lowered.startswith(bad_prefixes):
        return False
    pronoun_starts = (
        "let me ",
        "i use ",
        "i run ",
        "my project ",
        "we use ",
    )
    if lowered.startswith(pronoun_starts):
        return False
    if "copy link" in lowered and ("linkedin" in lowered or "email print" in lowered):
        return False
    if any(token in lowered for token in _NAVIGATION_LINE_HINTS) and (
        len(candidate) < 140 or candidate.count(" - ") >= 3
    ):
        return False
    return True


def _extract_heuristic_facts(document: FetchedDocument) -> tuple[str, list[str]]:
    sanitized = _sanitize_heuristic_source_text(document.content)
    candidate_corpus = sanitized or _sanitize_model_source_text(document.content)
    sentences: list[str] = []
    for line in candidate_corpus.splitlines():
        if len(sentences) >= 36:
            break
        line_sentences = extract_sentences(line, max_sentences=max(1, 36 - len(sentences)))
        if line_sentences:
            sentences.extend(line_sentences)
            continue
        cleaned_line = clean_text(line)
        if cleaned_line:
            sentences.append(cleaned_line)
    facts = dedupe_preserve_order(
        clean_text(sentence)
        for sentence in sentences
        if _looks_like_fact(sentence)
    )[:4]
    if not facts and candidate_corpus:
        fallback_lines = dedupe_preserve_order(
            clean_text(line)
            for line in candidate_corpus.splitlines()
            if _looks_like_fact(line)
        )
        facts = fallback_lines[:4]
    plausible_summary = next(
        (
            clean_text(sentence)
            for sentence in sentences
            if len(clean_text(sentence)) >= 45
            and "http" not in clean_text(sentence).lower()
            and not clean_text(sentence).lower().startswith(_BOILERPLATE_PREFIXES)
        ),
        "",
    )
    summary = facts[0] if facts else plausible_summary or clean_text(document.title or candidate_corpus[:220])
    return summary, facts


def _clean_generated_note_text(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    lowered = cleaned.lower()
    if lowered.startswith(_BOILERPLATE_PREFIXES):
        return ""
    if any(token in lowered for token in _NAVIGATION_LINE_HINTS):
        return ""
    if any(hint in lowered for hint in _OFF_TOPIC_PENALTY_HINTS):
        return ""
    return cleaned


_REPORT_FACT_NOISE_HINTS = (
    " | by ",
    " follow ",
    " min read",
    " press enter or click",
    " share ",
    " close menu",
    " distributed by ",
    " created by ",
    "* * *",
)


def _score_presentable_report_sentence(sentence: str) -> int:
    cleaned = clean_text(sentence)
    if not cleaned:
        return -10
    lowered = cleaned.lower()
    score = 0
    if _looks_like_fact(cleaned):
        score += 3
    if not any(hint in lowered for hint in _REPORT_FACT_NOISE_HINTS):
        score += 2
    if any(token in f" {lowered} " for token in (" is ", " are ", " can ", " should ", " uses ", " use ")):
        score += 1
    if ":" in cleaned or "—" in cleaned or "|" in cleaned:
        score -= 1
    if len(cleaned) > 220:
        score -= 1
    return score


def _select_presentable_report_sentence(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    candidates = extract_sentences(cleaned, max_sentences=4)
    if not candidates:
        candidates = [cleaned]
    ranked = sorted(
        ((candidate, _score_presentable_report_sentence(candidate)) for candidate in candidates),
        key=lambda item: item[1],
        reverse=True,
    )
    best_sentence, best_score = ranked[0]
    if best_score >= 2:
        return clean_text(best_sentence)
    return cleaned


def _presentable_report_claims(
    stream_notes: Sequence[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[str]:
    candidate_pool: list[tuple[str, int, int]] = []
    position = 0
    for note in stream_notes:
        for raw_candidate in note.get("key_facts", []):
            cleaned = _sanitize_report_claim_text(
                _select_presentable_report_sentence(str(raw_candidate or ""))
            )
            score = _score_presentable_report_sentence(cleaned)
            if score < 2:
                continue
            candidate_pool.append((cleaned, score, position))
            position += 1
        summary_candidate = _sanitize_report_claim_text(
            _select_presentable_report_sentence(str(note.get("summary", "") or ""))
        )
        summary_score = _score_presentable_report_sentence(summary_candidate)
        if summary_score < 2:
            continue
        candidate_pool.append((summary_candidate, summary_score, position))
        position += 1
    claims: list[str] = []
    for cleaned, _score, _position in sorted(
        candidate_pool,
        key=lambda item: (-item[1], item[2]),
    ):
        if not cleaned or cleaned in claims:
            continue
        claims.append(cleaned)
        if len(claims) >= limit:
            break
    return claims


def _sanitize_report_claim_text(claim: str) -> str:
    """Remove provenance fragments that belong in citation UI, not report prose."""

    cleaned = clean_text(claim)
    cleaned = re.sub(
        r"\b(This (?:draft|report) uses only [^.]*?retrieved (?:arxiv\s+)?records?/?excerpts?):\s*[^.]+\.",
        "This report is based on retrieved source excerpts.",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(The retrieved record set contains[^.:;\n]*?)\s*:\s*[^;\n]+;",
        r"\1;",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:at|on|from)\s+arXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\barXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\barXiv\s+records?/?excerpts?",
        "source excerpts",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\barXiv\s+record\b",
        "source record",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\barXiv\s+records\b",
        "source records",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*\bSources?:\s*(?:https?://\S+\s*(?:[;,]\s*)?)+(?:\s*\((?:partial|full)\s+support\))?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*\bSource:\s*(?:https?://\S+\s*(?:[;,]\s*)?)+(?:\s*\((?:partial|full)\s+support\))?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*\bSources?:\s*(?:[;,.]\s*)+(?:\((?:partial|full)\s+support\))?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*\bSource:\s*(?:[;,.]\s*)+(?:\((?:partial|full)\s+support\))?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*\bSources?:\s*[^.?!]*(?:\((?:partial|full)\s+support\))?$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*\bSource:\s*[^.?!]*(?:\((?:partial|full)\s+support\))?$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"([.;:]){2,}", r"\1", cleaned)
    return cleaned.strip()


def _parse_local_json_payload(text: str) -> dict[str, Any]:
    candidate = strip_markdown_fences(text)
    try:
        parsed = orjson.loads(candidate)
    except orjson.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = orjson.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise orjson.JSONDecodeError("Planner output was not a JSON object", candidate, 0)
    return parsed


class _CompactBudgetRationale(BaseModel):
    summary: str | None = None
    coverage_axes: list[str] = []
    source_diversity_reasoning: str | None = None
    grounding_difficulty: str | None = None


class _CompactResearchStreamPlan(BaseModel):
    name: str
    objective: str
    queries: list[str] = []


class _CompactResearchPlan(BaseModel):
    summary: str
    hypothesis: str
    streams: list[_CompactResearchStreamPlan]
    success_criteria: list[str] = []
    recommended_budget: RecommendedBudget | None = None
    budget_rationale: _CompactBudgetRationale | None = None
    recommended_execution_mode: ExecutionMode | None = None
    approval_required: bool = False
    complexity_factors: list[str] = []


def _expand_compact_plan(
    compact: _CompactResearchPlan,
    *,
    worker_model: str,
    planning_stage: PlanningStage,
    question: str,
    available_documents: Sequence[ResearchAssetRecord] | None,
    discovery_records: Sequence[PlanningDiscoveryRecord],
    source_selection: Sequence[str] | None,
    min_total_sources_retrieved: int,
    min_total_cited_sources: int,
    approved_plan: ResearchPlan | None,
) -> ResearchPlan:
    streams = [
        ResearchStreamPlan(
            name=stream.name,
            objective=stream.objective,
            queries=stream.queries,
            model=worker_model,
        )
        for stream in compact.streams
    ]
    rationale = (
        BudgetRecommendationRationale(
            summary=compact.budget_rationale.summary or "Compact planner rationale.",
            coverage_axes=list(compact.budget_rationale.coverage_axes),
            evidence_gaps=[],
            source_diversity_reasoning=(
                compact.budget_rationale.source_diversity_reasoning
                or "Local compact planner requested source diversity."
            ),
            grounding_difficulty=(
                compact.budget_rationale.grounding_difficulty
                or "Local compact planner expects standard downstream verification."
            ),
        )
        if compact.budget_rationale is not None
        else None
    )
    return ResearchPlan(
        summary=compact.summary,
        hypothesis=compact.hypothesis,
        streams=streams,
        success_criteria=list(compact.success_criteria),
        planning_artifact=None,
        recommended_budget=compact.recommended_budget,
        budget_rationale=rationale,
        recommended_execution_mode=compact.recommended_execution_mode,
        approval_required=compact.approval_required,
        complexity_factors=list(compact.complexity_factors),
    )


class OpenAIPlanner(Planner):
    def __init__(
        self,
        client: OpenAIJsonClient,
        *,
        planner_model: str,
        worker_model: str,
        search_provider: SearchProvider,
        settings: Settings,
    ) -> None:
        self.client = client
        self.planner_model = planner_model
        self.worker_model = worker_model
        self.search_provider = search_provider
        self.settings = settings

    def _use_compact_contract(self) -> bool:
        return (
            self.settings.resolved_llm_backend == "openai_compatible"
            and self.settings.resolved_llm_model_family in {"generic", "glm", "qwen", "deepseek"}
        )

    async def _planning_discovery(
        self,
        *,
        question: str,
        budget: BudgetPolicy,
        approved_plan: ResearchPlan | None,
        available_documents: Sequence[ResearchAssetRecord] | None,
        source_selection: Sequence[str] | None,
    ) -> list[PlanningDiscoveryRecord]:
        selected = set(source_selection or [])
        allowed_search = selected or None
        max_discovery_queries = self.settings.planner_max_discovery_queries
        if self.settings.resolved_search_backend == "openai":
            max_discovery_queries = min(
                max_discovery_queries,
                budget.max_queries_per_stream,
            )
        max_discovery_queries = max(1, max_discovery_queries)
        min_discovery_queries = min(
            self.settings.planner_min_discovery_queries,
            max_discovery_queries,
        )
        queries = _build_discovery_queries(
            question=question,
            budget=budget,
            approved_plan=approved_plan,
            available_documents=available_documents,
            max_queries=max_discovery_queries,
        )
        queries = queries[: max(min_discovery_queries, len(queries))]
        providers = getattr(self.search_provider, "providers", None)
        bounded_queries = queries[:max_discovery_queries]
        semaphore = asyncio.Semaphore(max(1, self.settings.planner_discovery_concurrency))

        async def discover(query: str) -> PlanningDiscoveryRecord:
            collected = []
            try:
                async with semaphore:
                    if providers:
                        seen_urls: set[str] = set()
                        for provider in providers:
                            if allowed_search is not None and provider.provider_name not in allowed_search:
                                continue
                            results = await provider.search(
                                query,
                                max_results=min(5, budget.max_results_per_query),
                            )
                            for result in results:
                                normalized = normalize_url(str(result.url))
                                if normalized in seen_urls:
                                    continue
                                seen_urls.add(normalized)
                                collected.append(result)
                            if len(collected) >= min(5, budget.max_results_per_query):
                                break
                    elif (
                        allowed_search is None
                        or self.search_provider.provider_name in allowed_search
                    ):
                        collected = await self.search_provider.search(
                            query,
                            max_results=min(5, budget.max_results_per_query),
                        )
            except Exception as exc:
                return PlanningDiscoveryRecord(
                    query=query,
                    provider=None,
                    result_count=0,
                    titles=[],
                    urls=[],
                    summary=f"Discovery search failed: {exc}",
                )
            if collected:
                collected = sorted(
                    collected,
                    key=lambda result: _search_result_relevance(query=query, result=result),
                    reverse=True,
                )[: min(5, budget.max_results_per_query)]
            return PlanningDiscoveryRecord(
                query=query,
                provider=(
                    ",".join(
                        dedupe_preserve_order(
                            result.provider for result in collected if result.provider
                        )
                    )
                    if collected
                    else None
                ),
                result_count=len(collected),
                titles=[result.title for result in collected[:3]],
                urls=[str(result.url) for result in collected[:3]],
                summary=(
                    "No search results were retrieved during planner discovery."
                    if not collected
                    else " | ".join(
                        f"{result.title} ({result.provider})" for result in collected[:3]
                    )
                ),
            )

        return list(await asyncio.gather(*(discover(query) for query in bounded_queries)))

    async def create_plan(
        self,
        question: str,
        budget: BudgetPolicy,
        *,
        planning_stage: PlanningStage = PlanningStage.EXECUTION,
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
        prior_notes: Sequence[dict[str, Any]] | None = None,
        context_pack: ContextPack | None = None,
        replan_count: int = 0,
        approved_plan: ResearchPlan | None = None,
        available_documents: Sequence[ResearchAssetRecord] | None = None,
        source_selection: Sequence[str] | None = None,
        min_total_sources_retrieved: int = 0,
        min_total_cited_sources: int = 0,
    ) -> GenerationResult[ResearchPlan]:
        effective_model_config = model_config or _default_model_config(self.settings)
        effective_worker_model = effective_model_config.worker_model
        effective_planner_model = effective_model_config.planner_model
        compact_contract = self._use_compact_contract()
        note_context = _render_notes(prior_notes or [])
        discovery_records = (
            await self._planning_discovery(
                question=question,
                budget=budget,
                approved_plan=approved_plan,
                available_documents=available_documents,
                source_selection=source_selection,
            )
            if planning_stage != PlanningStage.PREVIEW
            else []
        )
        retrieved_floor, cited_floor = _derive_source_floor_targets(
            budget=budget,
            stream_count=max(len(approved_plan.streams), 1) if approved_plan is not None else 1,
            minimum_retrieved=max(
                self.settings.planner_min_total_sources_retrieved,
                min_total_sources_retrieved,
            ),
            minimum_cited=max(
                self.settings.planner_min_total_cited_sources,
                min_total_cited_sources,
            ),
        )
        prompt_bundle = planner_system_prompt(
            settings=self.settings,
            planner_model=effective_planner_model,
            worker_model=effective_worker_model,
            planning_stage=planning_stage,
            available_documents=_summarize_available_documents(available_documents),
            discovery_digest="\n".join(
                f"- {record.query}: {record.summary or 'No summary'}"
                for record in discovery_records
            ),
            source_selection=list(source_selection or []),
            min_total_sources_retrieved=retrieved_floor,
            min_total_cited_sources=cited_floor,
            approved_plan_summary=(
                approved_plan.model_dump_json(indent=2)
                if approved_plan is not None
                else None
            ),
            agent_config=agent_config,
        )
        compact_system_prompt = (
            "You are the lead planner for a deep research system.\n"
            "Return only valid JSON matching the requested schema.\n"
            "Plan the minimum effective work needed to answer the question well.\n"
            "Use short stream names, direct objectives, and compact keyword-style search queries.\n"
            "Treat the budget as an upper bound, not a target.\n"
            "Respect selected sources, uploaded documents, discovery findings, and approved-plan context.\n"
            "Preserve uncertainty and avoid invented claims or sources."
        )
        available_documents_text = (
            "\n".join(_summarize_available_documents(available_documents)) or "- none"
        )
        discovery_text = (
            "\n".join(
                f"- {record.query}: {record.summary or 'No summary'}"
                for record in discovery_records
            )
            or "- no discovery performed"
        )
        approved_plan_text = (
            approved_plan.model_dump_json(indent=2) if approved_plan is not None else "None"
        )
        compact_approved_plan_text = (
            (
                f"Summary: {approved_plan.summary}\n"
                f"Hypothesis: {approved_plan.hypothesis}\n"
                f"Streams: {', '.join(stream.name for stream in approved_plan.streams)}"
            )
            if approved_plan is not None
            else "None"
        )
        prompt = (
            (
                f"Question:\n{question}\n\n"
                f"Budget:\n{budget.model_dump_json(indent=2)}\n\n"
                f"Prior notes:\n{_clip_local_prompt(note_context, limit=1600)}\n\n"
                f"Retrieved memory:\n{_clip_local_prompt(render_context_pack(context_pack), limit=1800)}\n\n"
                f"Planning stage: {planning_stage.value}\n\n"
                f"Selected sources: {list(source_selection or [])}\n\n"
                f"Available documents:\n{available_documents_text}\n\n"
                f"Planner discovery:\n{_clip_local_prompt(discovery_text, limit=1200)}\n\n"
                f"Minimum source floors:\n"
                f"- Retrieved: {retrieved_floor}\n"
                f"- Cited: {cited_floor}\n\n"
                f"Approved plan context:\n{compact_approved_plan_text}\n\n"
                f"Replan count: {replan_count}\n"
                "Compact local-model contract: return the essential execution plan fields only. "
                f"Use worker model `{effective_worker_model}` for every stream."
            )
            if compact_contract
            else (
                f"Question:\n{question}\n\n"
                f"Budget:\n{budget.model_dump_json(indent=2)}\n\n"
                f"Prior notes:\n{note_context}\n\n"
                f"Retrieved memory:\n{render_context_pack(context_pack)}\n\n"
                f"Planning stage: {planning_stage.value}\n\n"
                f"Selected sources: {list(source_selection or [])}\n\n"
                f"Available documents:\n{available_documents_text}\n\n"
                f"Planner discovery:\n{discovery_text}\n\n"
                f"Minimum source floors:\n"
                f"- Retrieved: {retrieved_floor}\n"
                f"- Cited: {cited_floor}\n\n"
                f"Approved plan context:\n{approved_plan_text}\n\n"
                f"Replan count: {replan_count}\n"
                f"Use worker model `{effective_worker_model}` for every stream."
            )
        )
        if compact_contract:
            compact_json_contract = (
                "Return only a JSON object with these keys:\n"
                "{\n"
                '  "summary": string,\n'
                '  "hypothesis": string,\n'
                '  "streams": [{"name": string, "objective": string, "queries": [string]}],\n'
                '  "success_criteria": [string],\n'
                '  "complexity_factors": [string]\n'
                "}\n"
                "Keep arrays short, do not use markdown fences, and never include extra keys. "
                "Do not include budget or artifact fields."
            )
            raw_text = await self.client.generate_text(
                model=effective_planner_model,
                system_prompt=compact_system_prompt,
                user_prompt=f"{prompt}\n\n{compact_json_contract}",
                reasoning_effort=self.settings.llm_reasoning_effort,
                temperature=0,
            )
            raw_plan = _CompactResearchPlan.model_validate(_parse_local_json_payload(raw_text))
            plan_result = GenerationResult(value=raw_plan, usage=UsageInfo())
        else:
            plan_result = await self.client.generate_json(
                model=effective_planner_model,
                system_prompt=prompt_bundle.system_prompt,
                user_prompt=prompt,
                schema_model=ResearchPlan,
                reasoning_effort=self.settings.llm_reasoning_effort,
            )
            raw_plan = plan_result.value
        plan = (
            _expand_compact_plan(
                raw_plan,
                worker_model=effective_worker_model,
                planning_stage=planning_stage,
                question=question,
                available_documents=available_documents,
                discovery_records=discovery_records,
                source_selection=source_selection,
                min_total_sources_retrieved=retrieved_floor,
                min_total_cited_sources=cited_floor,
                approved_plan=approved_plan,
            )
            if compact_contract
            else raw_plan
        )
        bounded_streams = _bound_streams_to_budget(
            plan.streams,
            question=question,
            budget=budget,
            worker_model=effective_worker_model,
        )
        recommended_budget = plan.recommended_budget or RecommendedBudget(
            max_streams=min(
                budget.max_streams,
                max(len(bounded_streams), min(len(bounded_streams) + 2, budget.max_streams)),
            ),
            max_replans=budget.max_replans,
            max_queries_per_stream=min(
                budget.max_queries_per_stream,
                max((max(len(stream.queries) for stream in bounded_streams) if bounded_streams else 1), 3),
            ),
            max_results_per_query=budget.max_results_per_query,
            max_sources_per_stream=budget.max_sources_per_stream,
            per_domain_limit=budget.per_domain_limit,
            rationale_summary="Planner-sized execution budget derived from the approved plan.",
        )
        budget_rationale = plan.budget_rationale or BudgetRecommendationRationale(
            summary="Planner-sized execution budget derived from the approved plan.",
            coverage_axes=plan.complexity_factors or ["plan_scoped_coverage"],
            evidence_gaps=[],
            source_diversity_reasoning="Stream count and query depth follow the plan breadth.",
            grounding_difficulty="Claims should remain narrow enough for downstream verification.",
        )
        recommended_execution_mode = plan.recommended_execution_mode or (
            ExecutionMode.DEEP
            if len(bounded_streams) >= 6 or recommended_budget.max_queries_per_stream >= 6
            else ExecutionMode.STANDARD
        )
        artifact = (
            plan.planning_artifact.model_copy(
                update={
                    "stage": planning_stage,
                    "approved_preview_version": (
                        approved_plan.planning_artifact.approved_preview_version
                        if approved_plan and approved_plan.planning_artifact is not None
                        else plan.planning_artifact.approved_preview_version
                    ),
                    "available_documents": (
                        plan.planning_artifact.available_documents
                        or _summarize_available_documents(available_documents)
                    ),
                    "discovery_queries": (
                        plan.planning_artifact.discovery_queries
                        or [record.query for record in discovery_records]
                    ),
                    "discovery_records": (
                        plan.planning_artifact.discovery_records or list(discovery_records)
                    ),
                    "source_selection": (
                        plan.planning_artifact.source_selection or list(source_selection or [])
                    ),
                    "min_total_sources_retrieved": max(
                        plan.planning_artifact.min_total_sources_retrieved,
                        retrieved_floor,
                    ),
                    "min_total_cited_sources": max(
                        plan.planning_artifact.min_total_cited_sources,
                        cited_floor,
                    ),
                }
            )
            if plan.planning_artifact is not None
            else _build_planning_artifact(
                stage=planning_stage,
                question=question,
                plan=plan.model_copy(update={"streams": bounded_streams}),
                available_documents=available_documents,
                discovery_records=discovery_records,
                source_selection=source_selection,
                min_total_sources_retrieved=retrieved_floor,
                min_total_cited_sources=cited_floor,
                approved_preview_version=(
                    approved_plan.planning_artifact.approved_preview_version
                    if approved_plan and approved_plan.planning_artifact is not None
                    else None
                ),
            )
        )
        return GenerationResult(
            value=plan.model_copy(
                update={
                    "streams": bounded_streams,
                    "planning_artifact": artifact,
                    "recommended_budget": recommended_budget,
                    "budget_rationale": budget_rationale,
                    "recommended_execution_mode": recommended_execution_mode,
                    "approval_required": plan.approval_required
                    or recommended_execution_mode == ExecutionMode.DEEP,
                    "complexity_factors": plan.complexity_factors
                    or ["multi_stream_plan" if len(bounded_streams) > 1 else "single_stream_plan"],
                }
            ),
            usage=plan_result.usage,
            metadata={
                **prompt_bundle.metadata(),
                "planning_stage": planning_stage.value,
                "discovery_query_count": len(discovery_records),
            },
        )


class HeuristicNoteWriter(NoteWriter):
    async def write_note(
        self,
        *,
        question: str,
        stream: ResearchStreamPlan,
        document: FetchedDocument,
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
        context_pack: ContextPack | None = None,
    ) -> GenerationResult[NoteDraft]:
        summary, key_facts = _extract_heuristic_facts(document)
        open_questions: list[str] = []
        if len(key_facts) < 2:
            open_questions.append(f"Find stronger corroboration for {stream.objective.lower()}")
        return GenerationResult(
            value=NoteDraft(
                summary=summary,
                key_facts=key_facts,
                open_questions=open_questions,
                confidence=min(0.9, 0.55 + (0.1 * len(key_facts))),
            ),
            usage=UsageInfo(),
        )


class OpenAINoteWriter(NoteWriter):
    def __init__(self, client: OpenAIJsonClient, *, worker_model: str, settings: Settings) -> None:
        self.client = client
        self.worker_model = worker_model
        self.settings = settings

    async def write_note(
        self,
        *,
        question: str,
        stream: ResearchStreamPlan,
        document: FetchedDocument,
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
        context_pack: ContextPack | None = None,
    ) -> GenerationResult[NoteDraft]:
        effective_worker_model = (
            stream.model
            or (model_config.worker_model if model_config is not None else self.worker_model)
        )
        source_context = build_source_prompt_context(document)
        prompt_bundle = note_writer_system_prompt(
            settings=self.settings,
            agent_config=agent_config,
            source_context=source_context,
        )
        prompt = (
            f"Research question:\n{question}\n\n"
            f"Stream objective:\n{stream.objective}\n\n"
            f"Operational context:\n{render_context_pack(context_pack)}\n\n"
            f"Source title: {document.title}\n"
            f"Source URL: {document.canonical_url}\n\n"
            f"Source content:\n{_sanitize_model_source_text(document.content)}"
        )
        note_result = await self.client.generate_json(
            model=effective_worker_model,
            system_prompt=prompt_bundle.system_prompt,
            user_prompt=prompt,
            schema_model=NoteDraft,
            reasoning_effort=self.settings.llm_reasoning_effort,
        )
        note = note_result.value
        return GenerationResult(
            value=note.model_copy(
                update={
                    "key_facts": dedupe_preserve_order(note.key_facts)[:5],
                    "open_questions": dedupe_preserve_order(note.open_questions)[:5],
                }
            ),
            usage=note_result.usage,
            metadata=prompt_bundle.metadata(),
        )


class HeuristicGapAnalyzer(GapAnalyzer):
    def __init__(self, *, worker_model: str) -> None:
        self.worker_model = worker_model

    async def analyze(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        notes: Sequence[dict[str, Any]],
        budget: BudgetPolicy,
        replan_count: int,
        model_config: ModelConfig | None = None,
    ) -> GenerationResult[GapAnalysis]:
        worker_model = model_config.worker_model if model_config is not None else self.worker_model
        notes_by_stream: dict[str, int] = Counter(
            note["stream_name"] for note in notes if note.get("stream_name")
        )
        unresolved = dedupe_preserve_order(
            question for note in notes for question in note.get("open_questions", []) if question
        )
        if replan_count >= budget.max_replans:
            return GenerationResult(
                value=GapAnalysis(should_replan=False, rationale="Replan budget exhausted."),
                usage=UsageInfo(),
            )

        if not notes:
            return GenerationResult(
                value=GapAnalysis(
                    should_replan=True,
                    rationale="The first pass did not produce any notes.",
                    additional_streams=[
                        ResearchStreamPlan(
                            name=f"Gap closure {replan_count + 1}",
                            objective=(
                                "Recover from a weak first pass with broader evidence gathering."
                            ),
                            queries=[
                                question,
                                f"{question} official source",
                                f"{question} documentation",
                            ][: budget.max_queries_per_stream],
                            model=worker_model,
                        )
                    ],
                ),
                usage=UsageInfo(),
            )

        weak_streams = []
        for stream in plan.streams:
            if notes_by_stream.get(stream.name, 0) == 0:
                weak_streams.append(stream.objective)

        if weak_streams or unresolved:
            queries = unresolved[: budget.max_queries_per_stream]
            if not queries:
                queries = [f"{question} unresolved risks", f"{question} edge cases"]
            return GenerationResult(
                value=GapAnalysis(
                    should_replan=True,
                    rationale="Some streams produced weak coverage or left unresolved questions.",
                    additional_streams=[
                        ResearchStreamPlan(
                            name=f"Gap closure {replan_count + 1}",
                            objective=(
                                "Resolve the most important missing evidence from the first pass."
                            ),
                            queries=queries,
                            model=worker_model,
                        )
                    ],
                ),
                usage=UsageInfo(),
            )

        return GenerationResult(
            value=GapAnalysis(
                should_replan=False,
                rationale="Coverage looks sufficient for synthesis.",
            ),
            usage=UsageInfo(),
        )


class HeuristicReportWriter(ReportWriter):
    async def write_report(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        notes: Sequence[dict[str, Any]],
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
        context_pack: ContextPack | None = None,
        output_contract: dict[str, Any] | None = None,
    ) -> GenerationResult[DraftReport]:
        notes_by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for note in notes:
            notes_by_stream[note.get("stream_name") or "Findings"].append(note)

        sections: list[ReportSection] = []
        all_facts: list[str] = []
        open_questions: list[str] = []
        for stream in plan.streams:
            stream_notes = notes_by_stream.get(stream.name, [])
            presentable_claims = _presentable_report_claims(stream_notes)
            if not presentable_claims:
                continue
            overview = presentable_claims[0]
            claims = presentable_claims[1:5] or presentable_claims[:4]
            all_facts.extend([overview, *claims])
            open_questions.extend(
                question for note in stream_notes for question in note.get("open_questions", [])
            )
            sections.append(
                ReportSection(
                    title=stream.name,
                    overview=overview,
                    claims=claims,
                )
            )

        known_streams = {stream.name for stream in plan.streams}
        for stream_name, stream_notes in notes_by_stream.items():
            if stream_name in known_streams:
                continue
            presentable_claims = _presentable_report_claims(stream_notes)
            if not presentable_claims:
                continue
            overview = presentable_claims[0]
            claims = presentable_claims[1:5] or presentable_claims[:4]
            sections.append(
                ReportSection(
                    title=stream_name,
                    overview=overview,
                    claims=claims,
                )
            )
            all_facts.extend([overview, *claims])
            open_questions.extend(
                question for note in stream_notes for question in note.get("open_questions", [])
            )

        if not sections:
            fallback_claims = [f"No grounded findings were extracted for: {question}."]
            open_questions.extend(
                question for note in notes for question in note.get("open_questions", []) if question
            )
            sections.append(
                ReportSection(
                    title="Findings",
                    overview=f"The run completed, but the heuristic synthesis stage did not extract grounded findings for {question}.",
                    claims=fallback_claims,
                )
            )
            all_facts.extend(fallback_claims)

        executive_summary = " ".join(dedupe_preserve_order(all_facts)[:3]) or sections[0].overview
        if context_pack is not None:
            preferred_style = next(
                (
                    fragment.metadata.get("profile_id")
                    for fragment in context_pack.fragments
                    if fragment.kind.value == "preference"
                ),
                None,
            )
            if preferred_style:
                executive_summary = f"Profile {preferred_style}: {executive_summary}"
        return GenerationResult(
            value=DraftReport(
                title=derive_report_title(question),
                conversation_topic=derive_conversation_topic(question),
                executive_summary=executive_summary,
                sections=sections,
                open_questions=dedupe_preserve_order(open_questions)[:6],
            ),
            usage=UsageInfo(),
        )


class OpenAIReportWriter(ReportWriter):
    def __init__(self, client: OpenAIJsonClient, *, lead_model: str, settings: Settings) -> None:
        self.client = client
        self.lead_model = lead_model
        self.settings = settings

    async def write_report(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        notes: Sequence[dict[str, Any]],
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
        context_pack: ContextPack | None = None,
        output_contract: dict[str, Any] | None = None,
    ) -> GenerationResult[DraftReport]:
        effective_lead_model = (
            model_config.lead_model if model_config is not None else self.lead_model
        )
        prompt_bundle = report_writer_system_prompt(
            settings=self.settings,
            agent_config=agent_config,
        )
        prompt = (
            f"Question:\n{question}\n\n"
            f"Plan:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Retrieved memory:\n{render_context_pack(context_pack)}\n\n"
            f"Output contract:\n{_render_output_contract(output_contract)}\n\n"
            f"Notes:\n{_render_notes(notes)}"
        )
        report_result = await self.client.generate_json(
            model=effective_lead_model,
            system_prompt=prompt_bundle.system_prompt,
            user_prompt=prompt,
            schema_model=DraftReport,
            reasoning_effort=self.settings.llm_reasoning_effort,
        )
        report = report_result.value
        report_title = report.title or derive_report_title(question)
        conversation_topic = report.conversation_topic or derive_conversation_topic(question)
        claim_budget = max(1, self.settings.grounding_max_claims_per_run)
        bounded_sections = []
        for section in report.sections[:4]:
            if claim_budget <= 0:
                claims = []
            else:
                claims = [
                    claim
                    for claim in (
                        _sanitize_report_claim_text(raw_claim)
                        for raw_claim in dedupe_preserve_order(section.claims)
                    )
                    if claim
                ][: min(3, claim_budget)]
                claim_budget -= len(claims)
            bounded_sections.append(
                section.model_copy(
                    update={
                        "overview": _sanitize_report_claim_text(section.overview),
                        "claims": claims,
                    }
                )
            )
        return GenerationResult(
            value=report.model_copy(
                update={
                    "title": report_title,
                    "conversation_topic": conversation_topic,
                    "sections": bounded_sections,
                    "open_questions": dedupe_preserve_order(report.open_questions)[:8],
                }
            ),
            usage=report_result.usage,
            metadata=prompt_bundle.metadata(),
        )


class HeuristicClaimVerifier(ClaimVerifier):
    async def verify(
        self,
        *,
        claim: str,
        candidates: Sequence[RetrievedPassage],
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
    ) -> GenerationResult[ClaimVerification]:
        if not candidates:
            return GenerationResult(
                value=ClaimVerification(
                    support_label=CitationSupportLabel.UNSUPPORTED,
                    reason="No candidate passages were retrieved for this claim.",
                    confidence=0.0,
                ),
                usage=UsageInfo(),
            )
        candidate = candidates[0]
        if candidate.score >= 0.55:
            label = CitationSupportLabel.SUPPORTED
        elif candidate.score >= 0.3:
            label = CitationSupportLabel.PARTIAL
        else:
            label = CitationSupportLabel.UNSUPPORTED
        return GenerationResult(
            value=ClaimVerification(
                support_label=label,
                reason=f"Top lexical overlap score: {candidate.score:.2f}",
                selected_source_id=candidate.source_id,
                selected_passage_index=candidate.passage_index,
                quote=candidate.text[:240],
                confidence=min(0.99, candidate.score),
            ),
            usage=UsageInfo(),
        )


class _ClaimVerificationSchema(BaseModel):
    support_label: CitationSupportLabel
    reason: str
    selected_source_id: str | None = None
    selected_passage_index: int | None = None
    quote: str | None = None
    confidence: float


class OpenAIClaimVerifier(ClaimVerifier):
    def __init__(
        self,
        client: OpenAIJsonClient,
        *,
        verifier_model: str,
        settings: Settings,
    ) -> None:
        self.client = client
        self.verifier_model = verifier_model
        self.settings = settings

    async def verify(
        self,
        *,
        claim: str,
        candidates: Sequence[RetrievedPassage],
        agent_config: AgentConfig | None = None,
        model_config: ModelConfig | None = None,
    ) -> GenerationResult[ClaimVerification]:
        if not candidates:
            return GenerationResult(
                value=ClaimVerification(
                    support_label=CitationSupportLabel.UNSUPPORTED,
                    reason="No candidate passages were available.",
                    confidence=0.0,
                ),
                usage=UsageInfo(),
            )
        rendered_candidates = []
        for candidate in candidates[:5]:
            rendered_candidates.append(
                {
                    "source_id": candidate.source_id,
                    "source_title": candidate.source_title,
                    "source_url": str(candidate.source_url),
                    "passage_index": candidate.passage_index,
                    "score": candidate.score,
                    "source_kind": (
                        candidate.source_kind.value if candidate.source_kind is not None else None
                    ),
                    "retrieval_method": (
                        candidate.retrieval_method.value
                        if candidate.retrieval_method is not None
                        else None
                    ),
                    "trust_tier": (
                        candidate.trust_tier.value if candidate.trust_tier is not None else None
                    ),
                    "trust_rationale": candidate.trust_rationale,
                    "text": candidate.text[:900],
                }
            )
        prompt_bundle = claim_verifier_system_prompt(
            settings=self.settings,
            agent_config=agent_config,
        )
        effective_verifier_model = (
            model_config.verifier_model if model_config is not None else self.verifier_model
        )
        result = await self.client.generate_json(
            model=effective_verifier_model,
            system_prompt=prompt_bundle.system_prompt,
            user_prompt=f"Claim:\n{claim}\n\nCandidates:\n{rendered_candidates}",
            schema_model=_ClaimVerificationSchema,
            reasoning_effort=self.settings.llm_reasoning_effort,
        )
        return GenerationResult(
            value=ClaimVerification.model_validate(result.value.model_dump(mode="json")),
            usage=result.usage,
            metadata=prompt_bundle.metadata(),
        )


class PassageRetriever:
    def retrieve(
        self,
        claim: str,
        passages: Sequence[dict[str, Any]],
        *,
        top_k: int = 5,
    ) -> list[RetrievedPassage]:
        claim_tokens = set(tokenize(claim))
        if not claim_tokens:
            return []
        ranked: list[RetrievedPassage] = []
        for passage in passages:
            passage_tokens = set(tokenize(passage["text"]))
            if not passage_tokens:
                continue
            overlap = len(claim_tokens & passage_tokens) / max(len(claim_tokens), 1)
            density = len(claim_tokens & passage_tokens) / max(len(passage_tokens), 1)
            exact_bonus = 0.2 if clean_text(claim).lower() in passage["text"].lower() else 0.0
            score = round((0.75 * overlap) + (0.25 * density) + exact_bonus, 4)
            if score <= 0:
                continue
            ranked.append(
                RetrievedPassage(
                    source_id=passage["source_id"],
                    source_title=passage["source_title"],
                    source_url=passage["source_url"],
                    passage_index=passage["passage_index"],
                    text=passage["text"],
                    score=score,
                )
            )
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]


class ResearchWorker:
    def __init__(
        self,
        *,
        store: ResearchStore,
        events: RunEventService,
        search_provider: SearchProvider,
        fetch_provider: FetchProvider,
        note_writer: NoteWriter,
        artifact_store: ArtifactStore | None,
        embedding_provider: EmbeddingProvider | None,
        settings: Settings,
        telemetry: ResearchTelemetry | None = None,
        context_assembler: ContextAssembler | None = None,
        memory_compiler: MemoryCompiler | None = None,
    ) -> None:
        self.store = store
        self.events = events
        self.search_provider = search_provider
        self.fetch_provider = fetch_provider
        self.note_writer = note_writer
        self.artifact_store = artifact_store
        self.embedding_provider = embedding_provider
        self.settings = settings
        self.telemetry = telemetry
        self.context_assembler = context_assembler
        self.memory_compiler = memory_compiler

    async def _get_agent_config(self, run_id: str) -> AgentConfig:
        state = await self.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        return resolve_agent_config(state.agent_config or state.metadata.get("agent_config"))

    async def _get_model_config(self, run_id: str) -> ModelConfig:
        state = await self.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        return resolve_model_config(
            state.metadata.get("model_config") or state.metadata.get("model_config_override"),
            defaults=_default_model_config(self.settings),
        )

    async def _get_run_profile(self, run_id: str) -> tuple[str, MemoryInfluencePolicy]:
        state = await self.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        profile_id = state.profile_id or str(state.metadata.get("profile_id", "default"))
        profile = await self.store.get_profile(profile_id)
        base_policy = (
            profile.preferences.memory_policy if profile is not None else MemoryInfluencePolicy()
        )
        override = state.metadata.get("memory_policy_override")
        if override is not None:
            return profile_id, MemoryInfluencePolicy.model_validate(override)
        return profile_id, base_policy

    async def _get_source_selection(
        self,
        run_id: str,
    ) -> tuple[set[str] | None, set[str] | None]:
        state = await self.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        selected = set(state.metadata.get("source_selection") or [])
        if not selected:
            return None, None
        search_allowed = {
            provider.provider_name
            for provider in getattr(self.search_provider, "providers", [])
            if provider.provider_name in selected
        }
        if not search_allowed and self.search_provider.provider_name in selected:
            search_allowed = {self.search_provider.provider_name}
        fetch_allowed = {
            provider.provider_name
            for provider in getattr(self.fetch_provider, "providers", [])
            if provider.provider_name in selected
        }
        if not fetch_allowed and self.fetch_provider.provider_name in selected:
            fetch_allowed = {self.fetch_provider.provider_name}
        return search_allowed, fetch_allowed

    async def _search(
        self,
        *,
        run_id: str,
        query: str,
        max_results: int,
        allowed_search: set[str] | None,
    ) -> list[Any]:
        if self.settings.tool_registry_enabled:
            await self._assert_tool_budget_available(
                run_id=run_id,
                tool_name=SEARCH_TOOL,
                categories=SEARCH_BUDGET_CATEGORIES,
                limit=self.settings.max_search_tool_calls_per_run,
            )
        providers = getattr(self.search_provider, "providers", None)
        if providers:
            aggregated: list[Any] = []
            seen_urls: set[str] = set()
            for provider in providers:
                if allowed_search is not None and provider.provider_name not in allowed_search:
                    continue
                results = await provider.search(query, max_results=max_results)
                for result in results:
                    normalized = normalize_url(str(result.url))
                    if normalized in seen_urls:
                        continue
                    seen_urls.add(normalized)
                    aggregated.append(result)
                    if len(aggregated) >= max_results:
                        return aggregated
            return aggregated
        if allowed_search is not None and self.search_provider.provider_name not in allowed_search:
            raise RuntimeError("Active search provider is not enabled for this run.")
        return await self.search_provider.search(query, max_results=max_results)

    async def _fetch(
        self,
        *,
        run_id: str,
        url: str,
        allowed_fetch: set[str] | None,
    ) -> FetchedDocument:
        if self.settings.tool_registry_enabled:
            await self._assert_tool_budget_available(
                run_id=run_id,
                tool_name=FETCH_TOOL,
                categories=FETCH_BUDGET_CATEGORIES,
                limit=self.settings.max_fetch_tool_calls_per_run,
            )
        providers = getattr(self.fetch_provider, "providers", None)
        if providers:
            errors: list[str] = []
            for provider in providers:
                if allowed_fetch is not None and provider.provider_name not in allowed_fetch:
                    continue
                try:
                    return await provider.fetch(url)
                except Exception as exc:
                    errors.append(f"{provider.provider_name}: {exc}")
                    continue
            raise RuntimeError("; ".join(errors) or "No fetch providers enabled for this run.")
        if allowed_fetch is not None and self.fetch_provider.provider_name not in allowed_fetch:
            raise RuntimeError("Active fetch provider is not enabled for this run.")
        return await self.fetch_provider.fetch(url)

    def _fallback_document_from_search_result(
        self,
        result: Any,
        *,
        fetch_error: str,
        discovered_via: str,
    ) -> FetchedDocument | None:
        snippet = sanitize_source_snippet_for_url(
            url=str(getattr(result, "url", "") or ""),
            snippet=str(getattr(result, "snippet", "") or ""),
        )
        if len(snippet) < 40:
            return None
        url = normalize_url(str(result.url))
        title = clean_text(str(getattr(result, "title", "") or url))[:200]
        provider = clean_text(str(getattr(result, "provider", "search") or "search"))
        content = clean_text(f"{title}. {snippet}")
        return FetchedDocument(
            url=url,
            canonical_url=url,
            title=title,
            content=content,
            source_kind=SourceKind.WEB,
            retrieval_method=RetrievalMethod.API_NATIVE,
            metadata={
                "provider": provider,
                "fetch_fallback": "search_result_snippet",
                "fetch_error": fetch_error[:500],
                "discovered_via": discovered_via,
                "search_score": float(getattr(result, "score", 0.0) or 0.0),
            },
        )

    async def _fetch_search_result_document(
        self,
        *,
        run_id: str,
        stream_id: str | None,
        result: Any,
        allowed_fetch: set[str] | None,
        discovered_via: str,
    ) -> tuple[FetchedDocument, bool]:
        try:
            return (
                await self._fetch(
                    run_id=run_id,
                    url=str(result.url),
                    allowed_fetch=allowed_fetch,
                ),
                False,
            )
        except Exception as exc:
            await self.events.publish(
                run_id,
                "source.fetch_failed",
                {
                    "stream_id": stream_id,
                    "title": getattr(result, "title", None),
                    "url": str(result.url),
                    "search_provider": getattr(result, "provider", None),
                    "error": str(exc),
                    "discovered_via": discovered_via,
                },
            )
            fallback = self._fallback_document_from_search_result(
                result,
                fetch_error=str(exc),
                discovered_via=discovered_via,
            )
            if fallback is None:
                raise
            await self.events.publish(
                run_id,
                "source.fallback_document.created",
                {
                    "stream_id": stream_id,
                    "title": fallback.title,
                    "url": str(fallback.canonical_url),
                    "provider": fallback.metadata.get("provider"),
                    "discovered_via": discovered_via,
                },
            )
            return fallback, True

    async def _assert_tool_budget_available(
        self,
        *,
        run_id: str,
        tool_name: str,
        categories: Sequence[str],
        limit: int,
    ) -> None:
        budget_events = await self.store.list_budget_events(run_id)
        remaining = assert_tool_budget_available(
            tool_name=tool_name,
            budget_events=budget_events,
            categories=categories,
            limit=limit,
        )
        if remaining <= max(1, limit // 20):
            await self.events.publish(
                run_id,
                "tool.budget.low",
                {
                    "tool_name": tool_name,
                    "remaining_calls": remaining,
                    "limit": limit,
                    "categories": list(categories),
                },
            )

    def _annotate_document_with_trust(self, document: FetchedDocument) -> FetchedDocument:
        assessment = assess_source_trust(
            url=str(document.canonical_url),
            source_kind=document.source_kind,
            title=document.title,
            metadata=dict(document.metadata),
        )
        metadata = dict(document.metadata)
        metadata["trust_tier"] = assessment.tier.value
        metadata["trust_rationale"] = assessment.rationale
        metadata["source_trust_policy_version"] = SOURCE_TRUST_POLICY_VERSION
        return document.model_copy(update={"metadata": metadata})

    async def _ensure_run_active(self, run_id: str) -> None:
        state = await self.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        if state.cancel_requested:
            raise RunCancelledError("Run cancellation was requested.")

    async def _record_request_cost(
        self,
        *,
        run_id: str,
        category: str,
        delta: int,
        amount_usd: float,
        metadata: dict[str, Any],
        stream_id: str | None = None,
    ) -> None:
        event_metadata = dict(metadata)
        if amount_usd:
            event_metadata["estimated_cost_usd"] = amount_usd
        await self.store.record_budget_event(run_id, category, delta, event_metadata)
        await self.store.add_run_cost(run_id, amount_usd)
        if stream_id is not None:
            await self.store.add_stream_cost(stream_id, amount_usd)

    async def _record_llm_usage(
        self,
        *,
        run_id: str,
        stream_id: str | None,
        phase: str,
        model: str,
        usage: UsageInfo,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if usage.total_tokens:
            event_metadata = {
                "phase": phase,
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "estimated_cost_usd": usage.estimated_cost_usd,
            }
            if metadata:
                event_metadata.update(metadata)
            await self.store.record_budget_event(
                run_id,
                "llm_tokens",
                usage.total_tokens,
                event_metadata,
            )
        await self.store.add_run_cost(run_id, usage.estimated_cost_usd)
        if stream_id is not None:
            await self.store.add_stream_cost(stream_id, usage.estimated_cost_usd)

    async def _embed_passages(
        self,
        *,
        run_id: str,
        stream_id: str | None,
        source_id: str | None,
        passages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not passages or self.embedding_provider is None:
            return passages
        embedding_result = await self.embedding_provider.embed_texts(
            [passage["text"] for passage in passages]
        )
        await self._record_llm_usage(
            run_id=run_id,
            stream_id=stream_id,
            phase="passage_embed",
            model=getattr(
                self.embedding_provider,
                "model",
                self.embedding_provider.provider_name,
            ),
            usage=embedding_result.usage,
            metadata={"source_id": source_id},
        )
        vectors = embedding_result.value
        if len(vectors) != len(passages):
            raise RuntimeError("Embedding provider returned the wrong number of vectors.")
        return [
            {**passage, "embedding_vector": vector}
            for passage, vector in zip(passages, vectors, strict=True)
        ]

    def _extract_artifact_payloads(
        self,
        document: FetchedDocument,
    ) -> tuple[FetchedDocument, list[ArtifactPayload]]:
        metadata = dict(document.metadata)
        payloads = metadata.pop("_artifact_payloads", [])
        artifact_payloads: list[ArtifactPayload] = [
            payload for payload in payloads if isinstance(payload, ArtifactPayload)
        ]
        sanitized = document.model_copy(update={"metadata": metadata})
        return sanitized, artifact_payloads

    async def _register_search_results(
        self,
        *,
        run_id: str,
        query: str,
        results: Sequence[Any],
    ) -> None:
        entries = [
            {
                "url": str(result.url),
                "canonical_url": normalize_url(str(result.url)),
                "normalized_url": normalize_url(str(result.url)),
                "citation_key": build_citation_key(result.title, str(result.url)),
                "title": result.title,
                "provider": getattr(result, "provider", self.search_provider.provider_name),
                "discovered_via": "search",
                "metadata": {
                    "query": query,
                    "snippet": sanitize_source_snippet_for_url(
                        url=str(result.url),
                        snippet=str(getattr(result, "snippet", "") or ""),
                    ),
                    "score": getattr(result, "score", 0.0),
                    "discovered_stage": "search",
                },
            }
            for result in results
        ]
        await self.store.register_source_registry_entries(run_id, entries)

    async def _register_source_document(
        self,
        *,
        run_id: str,
        source_id: str,
        document: FetchedDocument,
        discovered_via: str,
        asset_id: str | None = None,
        asset_origin: str | None = None,
    ) -> None:
        await self.store.register_source_registry_entries(
            run_id,
            [
                {
                    "source_id": source_id,
                    "url": str(document.url),
                    "canonical_url": str(document.canonical_url),
                    "normalized_url": normalize_url(str(document.canonical_url)),
                    "citation_key": build_citation_key(document.title, str(document.canonical_url)),
                    "title": document.title,
                    "provider": str(document.metadata.get("provider", "")) or None,
                    "discovered_via": discovered_via,
                    "metadata": {
                        **dict(document.metadata),
                        "fetched_stage": discovered_via,
                        "asset_id": asset_id,
                        "asset_origin": asset_origin,
                        "user_supplied": asset_id is not None,
                    },
                }
            ],
        )

    async def _persist_source_artifacts(
        self,
        *,
        run_id: str,
        source_id: str,
        document: FetchedDocument,
        artifact_payloads: Sequence[ArtifactPayload],
    ) -> list[dict[str, Any]]:
        if self.artifact_store is None:
            return []
        payloads = [
            ArtifactPayload(
                kind="normalized-text",
                extension="txt",
                content_type="text/plain",
                data=document.content.encode("utf-8"),
            ),
            *artifact_payloads,
        ]
        artifacts: list[dict[str, Any]] = []
        for payload in payloads:
            reference = await self.artifact_store.save_artifact(
                run_id=run_id,
                source_id=source_id,
                payload=payload,
            )
            await self.store.save_artifact_record(
                run_id=run_id,
                source_id=source_id,
                kind=reference.kind,
                uri=reference.uri,
                content_type=reference.content_type,
                size_bytes=reference.size_bytes,
                sha256_digest=reference.sha256,
            )
            artifacts.append(reference.as_metadata())
        return artifacts

    @staticmethod
    def _document_from_source_snapshot(snapshot: dict[str, Any]) -> FetchedDocument:
        return FetchedDocument(
            url=str(snapshot["url"]),
            canonical_url=str(snapshot["canonical_url"]),
            title=str(snapshot["title"]),
            content=str(snapshot["content"]),
            source_kind=SourceKind(str(snapshot["source_kind"])),
            retrieval_method=RetrievalMethod(str(snapshot["retrieval_method"])),
            metadata=dict(snapshot.get("metadata") or {}),
        )

    @staticmethod
    def _passages_from_source_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        document = ResearchWorker._document_from_source_snapshot(snapshot)
        return [
            {
                "source_id": str(snapshot["id"]),
                "source_title": document.title,
                "source_url": str(document.canonical_url),
                "passage_index": index,
                "text": chunk["text"],
                "start_offset": chunk["start_offset"],
                "end_offset": chunk["end_offset"],
                "token_count": len(tokenize(chunk["text"])),
                "source_kind": document.source_kind.value,
                "retrieval_method": document.retrieval_method.value,
                "trust_tier": str(document.metadata.get("trust_tier", "")) or None,
                "trust_rationale": str(document.metadata.get("trust_rationale", "")) or None,
            }
            for index, chunk in enumerate(chunk_text(document.content))
        ]

    async def _write_note_for_document(
        self,
        *,
        run_id: str,
        question: str,
        stream_view: ResearchStreamView,
        stream_plan: ResearchStreamPlan,
        source_id: str,
        document: FetchedDocument,
        agent_config: AgentConfig,
        model_config: ModelConfig,
        context_pack: ContextPack | None = None,
        discovered_via: str,
        artifacts: Sequence[dict[str, Any]] | None = None,
        reused_existing: bool = False,
    ) -> dict[str, Any]:
        note_result = await self.note_writer.write_note(
            question=question,
            stream=stream_plan,
            document=document,
            agent_config=agent_config,
            model_config=model_config,
            context_pack=context_pack,
        )
        note = note_result.value
        await self._record_llm_usage(
            run_id=run_id,
            stream_id=stream_view.id,
            phase="note_write",
            model=stream_plan.model,
            usage=note_result.usage,
            metadata={
                "source_id": source_id,
                "reused_existing_source": reused_existing,
                **(note_result.metadata or {}),
            },
        )
        await self.store.save_note(
            run_id,
            stream_view.id,
            source_id,
            summary=note.summary,
            key_facts=note.key_facts,
            open_questions=note.open_questions,
            confidence=note.confidence,
        )
        await self.events.publish(
            run_id,
            "source.fetched",
            {
                "stream_id": stream_view.id,
                "stream_name": stream_view.name,
                "source_id": source_id,
                "title": document.title,
                "url": str(document.canonical_url),
                "provider": str(document.metadata.get("provider", self.fetch_provider.provider_name)),
                "trust_tier": str(document.metadata.get("trust_tier", "")) or None,
                "artifact_count": len(list(artifacts or [])),
                "discovered_via": discovered_via,
                "reused_existing": reused_existing,
                "search_result_fallback": document.metadata.get("fetch_fallback")
                == "search_result_snippet",
            },
        )
        await self.events.publish(
            run_id,
            "note.saved",
            {
                "stream_id": stream_view.id,
                "source_id": source_id,
                "summary": note.summary,
                "confidence": note.confidence,
                "trust_tier": str(document.metadata.get("trust_tier", "")) or None,
                "reused_existing": reused_existing,
            },
        )
        return {
            "source_id": source_id,
            "confidence": note.confidence,
            "artifacts": list(artifacts or []),
            "reused_existing": reused_existing,
        }

    async def _ingest_document_into_stream(
        self,
        *,
        run_id: str,
        question: str,
        stream_view: ResearchStreamView,
        stream_plan: ResearchStreamPlan,
        document: FetchedDocument,
        agent_config: AgentConfig,
        model_config: ModelConfig,
        context_pack: ContextPack | None = None,
        discovered_via: str,
        asset_id: str | None = None,
        asset_origin: str | None = None,
    ) -> dict[str, Any] | None:
        document, artifact_payloads = self._extract_artifact_payloads(document)
        document = self._annotate_document_with_trust(document)
        document_provider = str(document.metadata.get("provider", self.fetch_provider.provider_name))
        await self._record_request_cost(
            run_id=run_id,
            category="source_fetch",
            delta=1,
            amount_usd=self.settings.fetch_request_cost_usd,
            metadata={
                "provider": document_provider,
                "url": str(document.canonical_url),
                "discovered_via": discovered_via,
            },
            stream_id=stream_view.id,
        )
        source_id, is_new = await self.store.save_source(run_id, stream_view.id, document)
        if not is_new:
            return await self._write_note_for_document(
                run_id=run_id,
                question=question,
                stream_view=stream_view,
                stream_plan=stream_plan,
                source_id=source_id,
                document=document,
                agent_config=agent_config,
                model_config=model_config,
                context_pack=context_pack,
                discovered_via=discovered_via,
                reused_existing=True,
            )
        await self._register_source_document(
            run_id=run_id,
            source_id=source_id,
            document=document,
            discovered_via=discovered_via,
            asset_id=asset_id,
            asset_origin=asset_origin,
        )
        artifacts = await self._persist_source_artifacts(
            run_id=run_id,
            source_id=source_id,
            document=document,
            artifact_payloads=artifact_payloads,
        )
        if artifacts:
            await self.store.update_source_metadata(source_id, {"artifacts": artifacts})
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
        passages = await self._embed_passages(
            run_id=run_id,
            stream_id=stream_view.id,
            source_id=source_id,
            passages=passages,
        )
        await self.store.save_passages(run_id, source_id, passages)
        return await self._write_note_for_document(
            run_id=run_id,
            question=question,
            stream_view=stream_view,
            stream_plan=stream_plan,
            source_id=source_id,
            document=document,
            agent_config=agent_config,
            model_config=model_config,
            context_pack=context_pack,
            discovered_via=discovered_via,
            artifacts=artifacts,
        )

    async def ingest_input_assets(
        self,
        *,
        run_id: str,
        question: str,
        assets: Sequence[ResearchAssetRecord],
    ) -> None:
        reference_assets = [
            asset for asset in assets if asset.usage == ResearchAssetUsage.REFERENCE_SOURCE
        ]
        if not reference_assets:
            return
        agent_config = await self._get_agent_config(run_id)
        model_config = await self._get_model_config(run_id)
        _, allowed_fetch = await self._get_source_selection(run_id)
        started = perf_counter()
        stream_plan = ResearchStreamPlan(
            name="Provided references",
            objective="Read and incorporate user-provided reference materials.",
            queries=[asset.label for asset in reference_assets],
            model=model_config.worker_model,
        )
        stream_view = await self.store.create_manual_stream(
            run_id=run_id,
            name=stream_plan.name,
            objective=stream_plan.objective,
            model=stream_plan.model,
            queries=stream_plan.queries,
        )
        ingested = 0
        total_confidence = 0.0
        for asset in reference_assets:
            await self._ensure_run_active(run_id)
            if asset.source_type == ResearchAssetType.URL:
                if asset.url is None:
                    continue
                fetched_document = await self._fetch(
                    run_id=run_id,
                    url=str(asset.url),
                    allowed_fetch=allowed_fetch,
                )
                result = await self._ingest_document_into_stream(
                    run_id=run_id,
                    question=question,
                    stream_view=stream_view,
                    stream_plan=stream_plan,
                    document=fetched_document,
                    agent_config=agent_config,
                    model_config=model_config,
                    discovered_via="user_input_url",
                    asset_id=asset.id,
                    asset_origin="project" if asset.project_id else "run",
                )
            else:
                extracted_text = asset.extracted_text or asset.content_text
                if not extracted_text:
                    continue
                safe_name = (asset.file_name or f"{asset.label}.txt").replace(" ", "-")
                synthetic_url = f"https://open-research.local/user-input/{run_id}/{safe_name}"
                document = FetchedDocument(
                    url=synthetic_url,
                    canonical_url=synthetic_url,
                    title=asset.label,
                    content=extracted_text,
                    source_kind=SourceKind.DOCS,
                    retrieval_method=RetrievalMethod.API_NATIVE,
                    metadata={
                        "provider": "user-upload",
                        "content_type": asset.content_type,
                        "file_name": safe_name,
                    },
                )
                result = await self._ingest_document_into_stream(
                    run_id=run_id,
                    question=question,
                    stream_view=stream_view,
                    stream_plan=stream_plan,
                    document=document,
                    agent_config=agent_config,
                    model_config=model_config,
                    discovered_via="user_input_file",
                    asset_id=asset.id,
                    asset_origin="project" if asset.project_id else "run",
                )
            if result is None:
                continue
            ingested += 1
            total_confidence += float(result["confidence"])
        await self.store.update_stream(
            stream_view.id,
            status=StreamStatus.COMPLETED,
            sources_examined=ingested,
            elapsed_ms=int((perf_counter() - started) * 1000),
            confidence=round(total_confidence / max(ingested, 1), 3) if ingested else 0.0,
        )
        task = await self.store.get_task_for_stream(stream_view.id)
        if task is not None:
            await self.store.update_task_status(
                task["id"],
                TaskStatus.COMPLETED,
                output_json={"notes_written": ingested, "sources_examined": ingested},
            )
        await self.events.publish(
            run_id,
            "input_assets.ingested",
            {"count": ingested, "stream_id": stream_view.id},
        )

    async def execute_stream(
        self,
        *,
        run_id: str,
        question: str,
        stream_view: ResearchStreamView,
        stream_plan: ResearchStreamPlan,
        budget: BudgetPolicy,
        context_pack: ContextPack | None = None,
    ) -> dict[str, Any]:
        telemetry_context = (
            self.telemetry.span(
                "stream.execute",
                run_id=run_id,
                stream_id=stream_view.id,
                stream_name=stream_view.name,
            )
            if self.telemetry is not None
            else None
        )
        if telemetry_context is None:
            return await self._execute_stream_impl(
                run_id=run_id,
                question=question,
                stream_view=stream_view,
                stream_plan=stream_plan,
                budget=budget,
                context_pack=context_pack,
            )
        with telemetry_context:
            return await self._execute_stream_impl(
                run_id=run_id,
                question=question,
                stream_view=stream_view,
                stream_plan=stream_plan,
                budget=budget,
                context_pack=context_pack,
            )

    async def _execute_stream_impl(
        self,
        *,
        run_id: str,
        question: str,
        stream_view: ResearchStreamView,
        stream_plan: ResearchStreamPlan,
        budget: BudgetPolicy,
        context_pack: ContextPack | None = None,
    ) -> dict[str, Any]:
        task = await self.store.get_task_for_stream(stream_view.id)
        if task is None:
            raise KeyError(f"No task exists for stream {stream_view.id}")

        await self.store.update_stream(stream_view.id, status=StreamStatus.RUNNING)
        await self.store.update_task_status(task["id"], TaskStatus.RUNNING)
        await self.events.publish(
            run_id,
            "task.started",
            {
                "stream_id": stream_view.id,
                "stream_name": stream_view.name,
                "objective": stream_view.objective,
            },
        )

        attempt_id = await self.store.create_task_attempt(
            task["id"],
            provider=f"{self.search_provider.provider_name}+{self.fetch_provider.provider_name}",
        )

        selected_results: list[Any] = []
        result_query_order: dict[str, int] = {}
        seen_urls: set[str] = set()
        query_domain_counts: defaultdict[int, Counter[str]] = defaultdict(Counter)
        selection_target = min(
            budget.max_queries_per_stream * budget.max_results_per_query,
            max(budget.max_sources_per_stream, budget.max_sources_per_stream * 3),
        )
        started = perf_counter()
        notes_written = 0
        sources_examined = 0
        confidence_total = 0.0
        result_providers_seen: list[str] = []

        await self._ensure_run_active(run_id)
        agent_config = await self._get_agent_config(run_id)
        model_config = await self._get_model_config(run_id)
        profile_id, _ = await self._get_run_profile(run_id)
        allowed_search, allowed_fetch = await self._get_source_selection(run_id)
        merged_queries = _normalize_scope_queries(
            question=question,
            objective=stream_plan.objective,
            queries=list(task["input_json"].get("queries", [])),
            stream_name=stream_plan.name,
            max_queries=budget.max_queries_per_stream,
        )
        query_batch = merged_queries[: budget.max_queries_per_stream]
        minimum_queries_to_run = min(len(query_batch), max(1, min(3, budget.max_queries_per_stream)))

        try:
            for query_index, query in enumerate(query_batch):
                await self._ensure_run_active(run_id)
                results = await self._search(
                    run_id=run_id,
                    query=query,
                    max_results=budget.max_results_per_query,
                    allowed_search=allowed_search,
                )
                await self._register_search_results(run_id=run_id, query=query, results=results)
                result_providers = dedupe_preserve_order(result.provider for result in results)
                result_providers_seen.extend(result_providers)
                await self._record_request_cost(
                    run_id=run_id,
                    category="search_query",
                    delta=1,
                    amount_usd=self.settings.search_request_cost_usd,
                    metadata={
                        "query": query,
                        "provider": self.search_provider.provider_name,
                        "result_providers": result_providers,
                    },
                    stream_id=stream_view.id,
                )
                await self.events.publish(
                    run_id,
                    "search.performed",
                    {
                        "stream_id": stream_view.id,
                        "query": query,
                        "provider": self.search_provider.provider_name,
                        "result_providers": result_providers,
                        "result_count": len(results),
                    },
                )
                ranked_results = sorted(
                    results,
                    key=lambda result: _search_result_relevance(
                        query=query,
                        result=result,
                        stream_name=stream_view.name,
                    ),
                    reverse=True,
                )
                for result in ranked_results:
                    relevance = _search_result_relevance(
                        query=query,
                        result=result,
                        stream_name=stream_view.name,
                    )
                    canonical_url = normalize_url(str(result.url))
                    domain = domain_for_url(canonical_url)
                    minimum_relevance = _minimum_result_relevance(stream_name=stream_view.name)
                    if canonical_url in seen_urls:
                        continue
                    if relevance < minimum_relevance:
                        continue
                    if query_domain_counts[query_index][domain] >= budget.per_domain_limit:
                        continue
                    seen_urls.add(canonical_url)
                    result_query_order[canonical_url] = query_index
                    query_domain_counts[query_index][domain] += 1
                    selected_results.append(result)
                    if len(selected_results) >= selection_target:
                        break
                if (
                    len(selected_results) >= selection_target
                    and query_index + 1 >= minimum_queries_to_run
                ):
                    break

            selected_results = _select_results_for_fetch(
                selected_results,
                result_query_order,
                max_sources=selection_target,
                per_domain_limit=budget.per_domain_limit,
            )
            await self.events.publish(
                run_id,
                "source.selection.finalized",
                {
                    "stream_id": stream_view.id,
                    "stream_name": stream_view.name,
                    "candidate_count": len(result_query_order),
                    "selected_count": len(selected_results),
                    "per_domain_limit": budget.per_domain_limit,
                    "selected": [
                        {
                            "title": clean_text(str(getattr(result, "title", "") or ""))[:200],
                            "url": normalize_url(str(getattr(result, "url", "") or "")),
                            "provider": str(getattr(result, "provider", "") or ""),
                            "query_order": result_query_order.get(
                                normalize_url(str(getattr(result, "url", "") or ""))
                            ),
                        }
                        for result in selected_results
                    ],
                },
            )
            reusable_sources: list[dict[str, Any]] = []
            reusable_source_ids: set[str] = set()
            for result in selected_results:
                if sources_examined >= budget.max_sources_per_stream:
                    break
                await self._ensure_run_active(run_id)
                selected_canonical_url = normalize_url(str(result.url))
                existing_source = await self.store.get_run_source_snapshot(
                    run_id,
                    selected_canonical_url,
                )
                if existing_source is not None:
                    await self.events.publish(
                        run_id,
                        "source.cache.hit",
                        {
                            "stream_id": stream_view.id,
                            "stream_name": stream_view.name,
                            "source_id": existing_source["id"],
                            "url": selected_canonical_url,
                            "stage": "stream_fetch",
                        },
                    )
                    source_id = str(existing_source["id"])
                    if source_id not in reusable_source_ids:
                        reusable_source_ids.add(source_id)
                        reusable_sources.append(existing_source)
                    continue
                fetch_started = perf_counter()
                try:
                    fetched_document, used_search_fallback = await (
                        self._fetch_search_result_document(
                            run_id=run_id,
                            stream_id=stream_view.id,
                            result=result,
                            allowed_fetch=allowed_fetch,
                            discovered_via="fetch",
                        )
                    )
                except Exception as exc:
                    await self.events.publish(
                        run_id,
                        "source.skipped",
                        {
                            "stream_id": stream_view.id,
                            "stream_name": stream_view.name,
                            "title": getattr(result, "title", None),
                            "url": str(getattr(result, "url", "")),
                            "reason": "fetch_failed_without_fallback",
                            "error": str(exc)[:500],
                        },
                    )
                    continue
                if self.telemetry is not None:
                    fetch_provider = str(
                        fetched_document.metadata.get("provider", self.fetch_provider.provider_name)
                    )
                    self.telemetry.record_fetch_latency(
                        provider=fetch_provider,
                        seconds=perf_counter() - fetch_started,
                    )
                document, artifact_payloads = self._extract_artifact_payloads(fetched_document)
                document = self._annotate_document_with_trust(document)
                prepared_document = _prepare_document_for_corpus(document)
                if prepared_document is None:
                    await self.events.publish(
                        run_id,
                        "source.skipped",
                        {
                            "stream_id": stream_view.id,
                            "stream_name": stream_view.name,
                            "title": document.title,
                            "url": str(document.canonical_url),
                            "reason": "low_value_document",
                        },
                    )
                    continue
                document = prepared_document
                if _document_below_stream_trust_floor(
                    document,
                    agent_config=agent_config,
                    stream_name=stream_view.name,
                ):
                    await self.events.publish(
                        run_id,
                        "source.skipped",
                        {
                            "stream_id": stream_view.id,
                            "stream_name": stream_view.name,
                            "title": document.title,
                            "url": str(document.canonical_url),
                            "reason": "below_trust_floor",
                            "trust_tier": str(document.metadata.get("trust_tier", "")) or None,
                            "trust_floor": agent_config.source_trust_floor.value,
                        },
                    )
                    continue
                document_provider = str(
                    document.metadata.get("provider", self.fetch_provider.provider_name)
                )
                await self._record_request_cost(
                    run_id=run_id,
                    category="source_fetch",
                    delta=1,
                    amount_usd=self.settings.fetch_request_cost_usd,
                    metadata={
                        "provider": document_provider,
                        "url": str(document.canonical_url),
                        "search_result_fallback": used_search_fallback,
                    },
                    stream_id=stream_view.id,
                )
                source_id, is_new = await self.store.save_source(run_id, stream_view.id, document)
                if not is_new:
                    source_snapshot = await self.store.get_source_snapshot(source_id)
                    if source_id not in reusable_source_ids:
                        reusable_source_ids.add(source_id)
                        reusable_sources.append(source_snapshot)
                    continue
                await self._register_source_document(
                    run_id=run_id,
                    source_id=source_id,
                    document=document,
                    discovered_via="fetch",
                )
                artifacts = await self._persist_source_artifacts(
                    run_id=run_id,
                    source_id=source_id,
                    document=document,
                    artifact_payloads=artifact_payloads,
                )
                if artifacts:
                    await self.store.update_source_metadata(source_id, {"artifacts": artifacts})
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
                passages = await self._embed_passages(
                    run_id=run_id,
                    stream_id=stream_view.id,
                    source_id=source_id,
                    passages=passages,
                )
                await self.store.save_passages(run_id, source_id, passages)
                note_payload = await self._write_note_for_document(
                    run_id=run_id,
                    question=question,
                    stream_view=stream_view,
                    stream_plan=stream_plan,
                    source_id=source_id,
                    document=document,
                    agent_config=agent_config,
                    model_config=model_config,
                    context_pack=context_pack,
                    discovered_via="fetch",
                    artifacts=artifacts,
                )
                notes_written += 1
                sources_examined += 1
                confidence_total += float(note_payload["confidence"])

            for source_snapshot in reusable_sources:
                if sources_examined >= budget.max_sources_per_stream:
                    break
                reused_document = self._document_from_source_snapshot(source_snapshot)
                if _document_below_stream_trust_floor(
                    reused_document,
                    agent_config=agent_config,
                    stream_name=stream_view.name,
                ):
                    continue
                note_payload = await self._write_note_for_document(
                    run_id=run_id,
                    question=question,
                    stream_view=stream_view,
                    stream_plan=stream_plan,
                    source_id=str(source_snapshot["id"]),
                    document=reused_document,
                    agent_config=agent_config,
                    model_config=model_config,
                    context_pack=context_pack,
                    discovered_via="run_source_reuse",
                    artifacts=(source_snapshot.get("metadata") or {}).get("artifacts") or [],
                    reused_existing=True,
                )
                notes_written += 1
                sources_examined += 1
                confidence_total += float(note_payload["confidence"])

            elapsed_ms = int((perf_counter() - started) * 1000)
            confidence = (
                round(confidence_total / max(notes_written, 1), 3) if notes_written else 0.0
            )
            await self.store.update_stream(
                stream_view.id,
                status=StreamStatus.COMPLETED,
                sources_examined=sources_examined,
                elapsed_ms=elapsed_ms,
                confidence=confidence,
            )
            await self.store.update_task_status(
                task["id"],
                TaskStatus.COMPLETED,
                output_json={
                    "notes_written": notes_written,
                    "sources_examined": sources_examined,
                    "elapsed_ms": elapsed_ms,
                },
            )
            await self.store.finish_task_attempt(
                attempt_id,
                TaskStatus.COMPLETED,
                metadata={"sources_examined": sources_examined, "notes_written": notes_written},
            )
            if self.memory_compiler is not None and self.settings.profile_memory_enabled:
                await self.memory_compiler.compile_stream(
                    run_id=run_id,
                    profile_id=profile_id,
                    stream_name=stream_view.name,
                    stream_objective=stream_view.objective,
                    queries=query_batch,
                    providers=dedupe_preserve_order(result_providers_seen),
                    sources_examined=sources_examined,
                    notes_written=notes_written,
                    confidence=confidence,
                )
            return {
                "stream_id": stream_view.id,
                "sources_examined": sources_examined,
                "notes_written": notes_written,
                "confidence": confidence,
            }
        except RunCancelledError as exc:
            await self.store.update_stream(stream_view.id, status=StreamStatus.QUEUED)
            await self.store.update_task_status(task["id"], TaskStatus.QUEUED)
            await self.store.finish_task_attempt(
                attempt_id,
                TaskStatus.FAILED,
                error_message=str(exc),
            )
            await self.events.publish(
                run_id,
                "stream.cancelled",
                {
                    "stream_id": stream_view.id,
                    "stream_name": stream_view.name,
                },
            )
            raise
        except Exception as exc:
            await self.store.update_stream(stream_view.id, status=StreamStatus.FAILED)
            await self.store.update_task_status(task["id"], TaskStatus.FAILED)
            await self.store.finish_task_attempt(
                attempt_id,
                TaskStatus.FAILED,
                error_message=str(exc),
            )
            await self.events.publish(
                run_id,
                "stream.failed",
                {
                    "stream_id": stream_view.id,
                    "stream_name": stream_view.name,
                    "error": str(exc),
                },
            )
            return {
                "stream_id": stream_view.id,
                "sources_examined": 0,
                "notes_written": 0,
                "confidence": 0.0,
            }

    async def collect_supporting_passages(
        self,
        *,
        run_id: str,
        claim: str,
        section_title: str,
    ) -> list[dict[str, Any]]:
        telemetry_context = (
            self.telemetry.span(
                "claim.repair",
                run_id=run_id,
                section_title=section_title,
            )
            if self.telemetry is not None
            else None
        )
        if telemetry_context is None:
            return await self._collect_supporting_passages_impl(
                run_id=run_id,
                claim=claim,
                section_title=section_title,
            )
        with telemetry_context:
            return await self._collect_supporting_passages_impl(
                run_id=run_id,
                claim=claim,
                section_title=section_title,
            )

    async def _collect_supporting_passages_impl(
        self,
        *,
        run_id: str,
        claim: str,
        section_title: str,
    ) -> list[dict[str, Any]]:
        await self._ensure_run_active(run_id)
        queries = dedupe_preserve_order(
            [
                claim,
                f"{section_title} {claim}",
                f"{claim} official source",
            ]
        )
        selected_results: list[Any] = []
        seen_urls: set[str] = set()
        domain_counts: Counter[str] = Counter()
        max_results = max(1, self.settings.claim_repair_max_results)
        allowed_search, allowed_fetch = await self._get_source_selection(run_id)

        await self.events.publish(
            run_id,
            "claim.repair.started",
            {"section_title": section_title, "claim": claim},
        )

        for query in queries[:2]:
            await self._ensure_run_active(run_id)
            results = await self._search(
                run_id=run_id,
                query=query,
                max_results=max_results,
                allowed_search=allowed_search,
            )
            await self._register_search_results(run_id=run_id, query=query, results=results)
            result_providers = dedupe_preserve_order(result.provider for result in results)
            await self._record_request_cost(
                run_id=run_id,
                category="claim_repair_search",
                delta=1,
                amount_usd=self.settings.search_request_cost_usd,
                metadata={
                    "query": query,
                    "provider": self.search_provider.provider_name,
                    "result_providers": result_providers,
                },
            )
            await self.events.publish(
                run_id,
                "claim.repair.search_performed",
                {
                    "claim": claim,
                    "query": query,
                    "provider": self.search_provider.provider_name,
                    "result_providers": result_providers,
                    "result_count": len(results),
                },
            )
            for result in results:
                canonical_url = normalize_url(str(result.url))
                domain = domain_for_url(canonical_url)
                if canonical_url in seen_urls:
                    continue
                if domain_counts[domain] >= 1:
                    continue
                seen_urls.add(canonical_url)
                domain_counts[domain] += 1
                selected_results.append(result)
                if len(selected_results) >= max_results:
                    break
            if len(selected_results) >= max_results:
                break

        new_passages: list[dict[str, Any]] = []
        for result in selected_results:
            await self._ensure_run_active(run_id)
            selected_canonical_url = normalize_url(str(result.url))
            existing_source = await self.store.get_run_source_snapshot(run_id, selected_canonical_url)
            if existing_source is not None:
                await self.events.publish(
                    run_id,
                    "source.cache.hit",
                    {
                        "source_id": existing_source["id"],
                        "url": selected_canonical_url,
                        "stage": "claim_repair",
                        "claim": claim,
                    },
                )
                new_passages.extend(
                    self._passages_from_source_snapshot(existing_source)[
                        : self.settings.grounding_candidate_limit
                    ]
                )
                continue
            fetch_started = perf_counter()
            fetched_document, used_search_fallback = await self._fetch_search_result_document(
                run_id=run_id,
                stream_id=None,
                result=result,
                allowed_fetch=allowed_fetch,
                discovered_via="claim_repair_fetch",
            )
            if self.telemetry is not None:
                fetch_provider = str(
                    fetched_document.metadata.get("provider", self.fetch_provider.provider_name)
                )
                self.telemetry.record_fetch_latency(
                    provider=fetch_provider,
                    seconds=perf_counter() - fetch_started,
                )
            document, artifact_payloads = self._extract_artifact_payloads(fetched_document)
            document = self._annotate_document_with_trust(document)
            document_provider = str(
                document.metadata.get("provider", self.fetch_provider.provider_name)
            )
            await self._record_request_cost(
                run_id=run_id,
                category="claim_repair_fetch",
                delta=1,
                amount_usd=self.settings.fetch_request_cost_usd,
                metadata={
                    "provider": document_provider,
                    "url": str(document.canonical_url),
                    "search_result_fallback": used_search_fallback,
                },
            )
            source_id, is_new = await self.store.save_source(run_id, None, document)
            if not is_new:
                continue
            await self._register_source_document(
                run_id=run_id,
                source_id=source_id,
                document=document,
                discovered_via="claim_repair_fetch",
            )
            artifacts = await self._persist_source_artifacts(
                run_id=run_id,
                source_id=source_id,
                document=document,
                artifact_payloads=artifact_payloads,
            )
            if artifacts:
                await self.store.update_source_metadata(source_id, {"artifacts": artifacts})
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
            passages = await self._embed_passages(
                run_id=run_id,
                stream_id=None,
                source_id=source_id,
                passages=passages,
            )
            await self.store.save_passages(run_id, source_id, passages)
            new_passages.extend(
                [
                    {
                        "source_id": source_id,
                        "source_title": document.title,
                        "source_url": str(document.canonical_url),
                        "passage_index": passage["passage_index"],
                        "text": passage["text"],
                        "start_offset": passage["start_offset"],
                        "end_offset": passage["end_offset"],
                        "token_count": passage["token_count"],
                        "source_kind": document.source_kind.value,
                        "retrieval_method": document.retrieval_method.value,
                        "trust_tier": str(document.metadata.get("trust_tier", "")) or None,
                        "trust_rationale": (
                            str(document.metadata.get("trust_rationale", "")) or None
                        ),
                    }
                    for passage in passages
                ]
            )
            await self.events.publish(
                run_id,
                "claim.repair.source_fetched",
                {
                    "claim": claim,
                    "source_id": source_id,
                    "title": document.title,
                    "url": str(document.canonical_url),
                    "provider": document_provider,
                    "trust_tier": str(document.metadata.get("trust_tier", "")) or None,
                    "artifact_count": len(artifacts),
                    "search_result_fallback": document.metadata.get("fetch_fallback")
                    == "search_result_snippet",
                },
            )

        await self.events.publish(
            run_id,
            "claim.repair.completed",
            {"claim": claim, "new_passages": len(new_passages)},
        )
        return new_passages


class ResearchGraphState(TypedDict):
    run_id: str
    question: str
    replan_count: int
    should_replan: bool
    budget: dict[str, Any]
    resume_from: str


class ResearchOrchestrator:
    def __init__(
        self,
        *,
        store: ResearchStore,
        events: RunEventService,
        planner: Planner,
        gap_analyzer: GapAnalyzer,
        report_writer: ReportWriter,
        verifier: ClaimVerifier,
        worker: ResearchWorker,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: PassageReranker | None = None,
        telemetry: ResearchTelemetry | None = None,
        context_assembler: ContextAssembler | None = None,
        memory_compiler: MemoryCompiler | None = None,
        behavior_judge: BehaviorJudge | None = None,
    ) -> None:
        self.store = store
        self.events = events
        self.planner = planner
        self.gap_analyzer = gap_analyzer
        self.report_writer = report_writer
        self.verifier = verifier
        self.worker = worker
        self.settings = worker.settings
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.telemetry = telemetry
        self.context_assembler = context_assembler
        self.memory_compiler = memory_compiler
        self.behavior_judge = behavior_judge
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ResearchGraphState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("research", self._research_node)
        graph.add_node("assess", self._assess_node)
        graph.add_node("replan", self._replan_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("ground", self._ground_node)
        graph.add_conditional_edges(
            START,
            self._route_from_start,
            {
                "plan": "plan",
                "research": "research",
                "assess": "assess",
                "ground": "ground",
            },
        )
        graph.add_edge("plan", "research")
        graph.add_edge("research", "assess")
        graph.add_conditional_edges(
            "assess",
            self._route_after_assess,
            {
                "replan": "replan",
                "synthesize": "synthesize",
            },
        )
        graph.add_edge("replan", "research")
        graph.add_edge("synthesize", "ground")
        graph.add_edge("ground", END)
        return graph.compile()

    async def execute(self, *, run_id: str, question: str, budget: BudgetPolicy) -> None:
        state = await self.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        await self._ensure_run_active(run_id)
        if state.has_final_report:
            return
        telemetry_context = (
            self.telemetry.span("graph.execute", run_id=run_id)
            if self.telemetry is not None
            else None
        )
        if telemetry_context is None:
            await self.graph.ainvoke(
                {
                    "run_id": run_id,
                    "question": question,
                    "replan_count": max(state.latest_plan_version - 1, 0),
                    "should_replan": False,
                    "budget": budget.model_dump(mode="json"),
                    "resume_from": self._determine_start_node(state),
                }
            )
            return
        with telemetry_context:
            await self.graph.ainvoke(
                {
                    "run_id": run_id,
                    "question": question,
                    "replan_count": max(state.latest_plan_version - 1, 0),
                    "should_replan": False,
                    "budget": budget.model_dump(mode="json"),
                    "resume_from": self._determine_start_node(state),
                }
            )

    async def _ensure_run_active(self, run_id: str) -> None:
        state = await self.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        if state.cancel_requested:
            raise RunCancelledError("Run cancellation was requested.")

    async def _get_agent_config(self, run_id: str) -> AgentConfig:
        state = await self.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        return resolve_agent_config(state.agent_config or state.metadata.get("agent_config"))

    async def _get_model_config(self, run_id: str) -> ModelConfig:
        state = await self.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        return resolve_model_config(
            state.metadata.get("model_config") or state.metadata.get("model_config_override"),
            defaults=_default_model_config(self.worker.settings),
        )

    async def _get_run_profile(self, run_id: str) -> tuple[str, MemoryInfluencePolicy]:
        state = await self.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        profile_id = state.profile_id or str(state.metadata.get("profile_id", "default"))
        profile = await self.store.get_profile(profile_id)
        base_policy = (
            profile.preferences.memory_policy if profile is not None else MemoryInfluencePolicy()
        )
        override = state.metadata.get("memory_policy_override")
        if override is not None:
            return profile_id, MemoryInfluencePolicy.model_validate(override)
        return profile_id, base_policy

    async def _record_llm_usage(
        self,
        *,
        run_id: str,
        phase: str,
        model: str,
        usage: UsageInfo,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if usage.total_tokens:
            event_metadata = {
                "phase": phase,
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "estimated_cost_usd": usage.estimated_cost_usd,
            }
            if metadata:
                event_metadata.update(metadata)
            await self.store.record_budget_event(
                run_id,
                "llm_tokens",
                usage.total_tokens,
                event_metadata,
            )
        await self.store.add_run_cost(run_id, usage.estimated_cost_usd)

    async def _retrieve_supporting_passages(
        self,
        *,
        run_id: str,
        claim: str,
        limit: int = 5,
    ) -> list[RetrievedPassage]:
        telemetry_context = (
            self.telemetry.span("passages.retrieve", run_id=run_id)
            if self.telemetry is not None
            else None
        )
        if telemetry_context is None:
            return await self._retrieve_supporting_passages_impl(
                run_id=run_id,
                claim=claim,
                limit=limit,
            )
        with telemetry_context:
            return await self._retrieve_supporting_passages_impl(
                run_id=run_id,
                claim=claim,
                limit=limit,
            )

    async def _retrieve_supporting_passages_impl(
        self,
        *,
        run_id: str,
        claim: str,
        limit: int = 5,
    ) -> list[RetrievedPassage]:
        query_embedding: list[float] | None = None
        if self.embedding_provider is not None:
            embedding_result = await self.embedding_provider.embed_texts([claim])
            await self._record_llm_usage(
                run_id=run_id,
                phase="query_embed",
                model=getattr(
                    self.embedding_provider,
                    "model",
                    self.embedding_provider.provider_name,
                ),
                usage=embedding_result.usage,
                metadata={"claim": claim[:200]},
            )
            if embedding_result.value:
                query_embedding = embedding_result.value[0]

        matches = await self.store.search_passages(
            run_id,
            claim,
            limit=max(limit, self.worker.settings.grounding_candidate_limit),
            query_embedding=query_embedding,
        )
        candidates = [
            RetrievedPassage(
                source_id=match["source_id"],
                source_title=match["source_title"],
                source_url=match["source_url"],
                passage_index=match["passage_index"],
                text=match["text"],
                score=float(match.get("score", 0.0)),
                source_kind=match.get("source_kind"),
                retrieval_method=match.get("retrieval_method"),
                trust_tier=match.get("trust_tier"),
                trust_rationale=match.get("trust_rationale"),
            )
            for match in matches
        ]
        if self.reranker is None:
            return candidates[:limit]
        reranked = await self.reranker.rerank(
            query=claim,
            passages=candidates,
            top_k=limit,
        )
        await self.events.publish(
            run_id,
            "passages.reranked",
            {
                "claim": claim,
                "provider": self.reranker.provider_name,
                "candidate_count": len(candidates),
                "returned_count": len(reranked),
            },
        )
        return reranked

    @staticmethod
    def _determine_start_node(state) -> str:
        if state.latest_plan_version == 0:
            return "plan"
        if state.has_draft_report:
            return "ground"
        if state.queued_streams > 0 or state.active_streams > 0 or state.failed_streams > 0:
            return "research"
        return "assess"

    def _route_from_start(self, state: ResearchGraphState) -> str:
        return state["resume_from"]

    async def _plan_node(self, state: ResearchGraphState) -> dict[str, Any]:
        run_id = state["run_id"]
        budget = BudgetPolicy.model_validate(state["budget"])
        replan_count = state["replan_count"]
        agent_config = await self._get_agent_config(run_id)
        model_config = await self._get_model_config(run_id)
        profile_id, memory_policy = await self._get_run_profile(run_id)
        await self._ensure_run_active(run_id)
        await self.store.update_run_status(run_id, RunStatus.PLANNING)
        run_state = await self.store.get_run_execution_state(run_id)
        preview_raw = run_state.metadata.get("plan_preview") if run_state is not None else None
        approval_status = (
            run_state.metadata.get("approval_status") if run_state is not None else None
        )
        prior_notes = await self.store.list_notes(run_id)
        context_pack = (
            await self.context_assembler.assemble(
                run_id=run_id,
                question=state["question"],
                profile_id=profile_id,
                phase=ContextPhase.PLAN,
                memory_policy=memory_policy,
            )
            if self.context_assembler is not None and memory_policy.enabled
            else None
        )
        project_assets = (
            await self.store.list_research_assets(project_id=run_state.project_id)
            if run_state is not None and run_state.project_id is not None
            else []
        )
        run_assets = await self.store.list_research_assets(run_id=run_id)
        available_documents = [
            asset
            for asset in [*project_assets, *run_assets]
            if asset.processing_status.value == "ready"
        ]
        source_selection = list((run_state.metadata.get("source_selection") or []) if run_state else [])
        approved_plan = None
        planning_stage = PlanningStage.EXECUTION
        if preview_raw is not None and approval_status == "approved":
            preview = PlanPreview.model_validate(preview_raw)
            approved_plan = preview.plan
            await self.events.publish(
                run_id,
                "planning.execution.started",
                {
                    "preview_version": preview.version,
                    "source_selection": source_selection,
                },
            )
        elif replan_count > 0:
            planning_stage = PlanningStage.REPLAN

        min_total_sources_retrieved, min_total_cited_sources = _derive_source_floor_targets(
            budget=budget,
            stream_count=len(approved_plan.streams) if approved_plan is not None else max(run_state.latest_plan_version, 1),
            minimum_retrieved=self.settings.planner_min_total_sources_retrieved,
            minimum_cited=self.settings.planner_min_total_cited_sources,
        )
        attempts = max(1, self.settings.planner_max_validation_retries + 1)
        plan_result = None
        plan = None
        validation_issues: list[str] = []
        for attempt in range(1, attempts + 1):
            plan_result = await self.planner.create_plan(
                state["question"],
                budget,
                planning_stage=planning_stage,
                agent_config=agent_config,
                model_config=model_config,
                prior_notes=prior_notes,
                context_pack=context_pack,
                replan_count=replan_count,
                approved_plan=approved_plan,
                available_documents=available_documents,
                source_selection=source_selection,
                min_total_sources_retrieved=min_total_sources_retrieved,
                min_total_cited_sources=min_total_cited_sources,
            )
            plan = plan_result.value
            validation_issues = (
                _validate_research_plan(
                    plan=plan,
                    budget=budget,
                    stage=planning_stage,
                    min_total_sources_retrieved=min_total_sources_retrieved,
                    min_total_cited_sources=min_total_cited_sources,
                )
                if self.settings.planner_validation_enabled
                else []
            )
            if not validation_issues:
                break
            await self.events.publish(
                run_id,
                "planning.validation.failed",
                {
                    "attempt": attempt,
                    "issues": validation_issues,
                    "planning_stage": planning_stage.value,
                },
            )
        if plan is None or plan_result is None:
            raise RuntimeError("Planner did not return a plan.")
        if validation_issues:
            raise RuntimeError(
                "Execution planning failed validation: " + "; ".join(validation_issues)
            )
        if preview_raw is not None and approval_status == "approved" and plan.planning_artifact is not None:
            preview = PlanPreview.model_validate(preview_raw)
            plan = plan.model_copy(
                update={
                    "planning_artifact": plan.planning_artifact.model_copy(
                        update={"approved_preview_version": preview.version}
                    )
                }
            )
        version = await self.store.get_next_plan_version(run_id)
        _, stream_ids = await self.store.save_plan(run_id, plan, version)
        await self._record_llm_usage(
            run_id=run_id,
            phase="plan_create",
            model=model_config.planner_model,
            usage=plan_result.usage,
            metadata={
                "version": version,
                "planning_stage": planning_stage.value,
                "approved_preview_version": (
                    PlanPreview.model_validate(preview_raw).version if preview_raw is not None else None
                ),
                **(plan_result.metadata or {}),
            },
        )
        if run_state is not None:
            await self.store.update_run_metadata(
                run_id,
                {
                    "execution_plan_version": version,
                    "planning_stage": planning_stage.value,
                },
            )
        await self.events.publish(
            run_id,
            "planning.validation.passed",
            {
                "planning_stage": planning_stage.value,
                "validation_checks": (
                    plan.planning_artifact.validation_checks
                    if plan.planning_artifact is not None
                    else []
                ),
            },
        )
        await self.events.publish(
            run_id,
            "plan.created",
            {
                "version": version,
                "summary": plan.summary,
                "stream_count": len(plan.streams),
                "planning_stage": planning_stage.value,
                "prompt_template_version": (plan_result.metadata or {}).get(
                    "prompt_template_version"
                ),
            },
        )
        if plan.planning_artifact is not None:
            for record in plan.planning_artifact.discovery_records:
                await self.events.publish(
                    run_id,
                    "planning.discovery.recorded",
                    {
                        "planning_stage": planning_stage.value,
                        "query": record.query,
                        "provider": record.provider,
                        "result_count": record.result_count,
                        "titles": record.titles,
                    },
                )
        for stream, stream_id in zip(plan.streams, stream_ids, strict=True):
            await self.events.publish(
                run_id,
                "stream.created",
                {
                    "stream_id": stream_id,
                    "name": stream.name,
                    "objective": stream.objective,
                    "model": stream.model,
                },
            )
        if preview_raw is not None and approval_status == "approved":
            await self.events.publish(
                run_id,
                "planning.execution.completed",
                {
                    "preview_version": PlanPreview.model_validate(preview_raw).version,
                    "plan_version": version,
                },
            )
        return {}

    async def _research_node(self, state: ResearchGraphState) -> dict[str, Any]:
        run_id = state["run_id"]
        budget = BudgetPolicy.model_validate(state["budget"])
        profile_id, memory_policy = await self._get_run_profile(run_id)
        await self._ensure_run_active(run_id)
        await self.store.update_run_status(run_id, RunStatus.RESEARCHING)
        plan = await self.store.get_latest_plan(run_id)
        if plan is None:
            raise RuntimeError("Cannot research without a plan")
        stream_views = await self.store.list_queued_streams(run_id)
        if not stream_views:
            return {}
        context_pack = (
            await self.context_assembler.assemble(
                run_id=run_id,
                question=state["question"],
                profile_id=profile_id,
                phase=ContextPhase.RESEARCH,
                memory_policy=memory_policy,
            )
            if self.context_assembler is not None and memory_policy.enabled
            else None
        )

        plan_by_name = {stream.name: stream for stream in plan.streams}
        if self.settings.database_url.startswith("sqlite+aiosqlite://"):
            for stream_view in stream_views:
                await self.worker.execute_stream(
                    run_id=run_id,
                    question=state["question"],
                    stream_view=stream_view,
                    stream_plan=plan_by_name.get(stream_view.name, plan.streams[0]),
                    budget=budget,
                    context_pack=context_pack,
                )
            return {}
        tasks = [
            asyncio.create_task(
                self.worker.execute_stream(
                    run_id=run_id,
                    question=state["question"],
                    stream_view=stream_view,
                    stream_plan=plan_by_name.get(stream_view.name, plan.streams[0]),
                    budget=budget,
                    context_pack=context_pack,
                ),
                name=f"stream-{stream_view.id}",
            )
            for stream_view in stream_views
        ]
        try:
            await asyncio.gather(*tasks)
        except RunCancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return {}

    async def _assess_node(self, state: ResearchGraphState) -> dict[str, Any]:
        run_id = state["run_id"]
        await self._ensure_run_active(run_id)
        plan = await self.store.get_latest_plan(run_id)
        if plan is None:
            raise RuntimeError("Cannot assess without a plan")
        notes = await self.store.list_notes(run_id)
        budget = BudgetPolicy.model_validate(state["budget"])
        model_config = await self._get_model_config(run_id)
        analysis_result = await self.gap_analyzer.analyze(
            question=state["question"],
            plan=plan,
            notes=notes,
            budget=budget,
            replan_count=state["replan_count"],
            model_config=model_config,
        )
        analysis = analysis_result.value
        await self._record_llm_usage(
            run_id=run_id,
            phase="gap_assess",
            model=model_config.worker_model,
            usage=analysis_result.usage,
        )
        if analysis.should_replan:
            await self.events.publish(
                run_id=run_id,
                event_type="gap.detected",
                payload={
                    "rationale": analysis.rationale,
                    "additional_streams": len(analysis.additional_streams),
                },
            )
        return {"should_replan": analysis.should_replan}

    def _route_after_assess(self, state: ResearchGraphState) -> str:
        return "replan" if state["should_replan"] else "synthesize"

    async def _replan_node(self, state: ResearchGraphState) -> dict[str, Any]:
        run_id = state["run_id"]
        agent_config = await self._get_agent_config(run_id)
        model_config = await self._get_model_config(run_id)
        profile_id, memory_policy = await self._get_run_profile(run_id)
        await self._ensure_run_active(run_id)
        notes = await self.store.list_notes(run_id)
        budget = BudgetPolicy.model_validate(state["budget"])
        replan_count = state["replan_count"] + 1
        await self.events.publish(run_id, "replan.started", {"replan_count": replan_count})
        context_pack = (
            await self.context_assembler.assemble(
                run_id=run_id,
                question=state["question"],
                profile_id=profile_id,
                phase=ContextPhase.REPLAN,
                memory_policy=memory_policy,
            )
            if self.context_assembler is not None and memory_policy.enabled
            else None
        )
        run_state = await self.store.get_run_execution_state(run_id)
        project_assets = (
            await self.store.list_research_assets(project_id=run_state.project_id)
            if run_state is not None and run_state.project_id is not None
            else []
        )
        run_assets = await self.store.list_research_assets(run_id=run_id)
        available_documents = [
            asset
            for asset in [*project_assets, *run_assets]
            if asset.processing_status.value == "ready"
        ]
        source_selection = list((run_state.metadata.get("source_selection") or []) if run_state else [])
        prior_plan = await self.store.get_latest_plan(run_id)
        min_total_sources_retrieved, min_total_cited_sources = _derive_source_floor_targets(
            budget=budget,
            stream_count=len(prior_plan.streams) if prior_plan is not None else 1,
            minimum_retrieved=self.settings.planner_min_total_sources_retrieved,
            minimum_cited=self.settings.planner_min_total_cited_sources,
        )
        plan_result = await self.planner.create_plan(
            state["question"],
            budget,
            planning_stage=PlanningStage.REPLAN,
            agent_config=agent_config,
            model_config=model_config,
            prior_notes=notes,
            context_pack=context_pack,
            replan_count=replan_count,
            approved_plan=prior_plan,
            available_documents=available_documents,
            source_selection=source_selection,
            min_total_sources_retrieved=min_total_sources_retrieved,
            min_total_cited_sources=min_total_cited_sources,
        )
        plan = plan_result.value
        validation_issues = (
            _validate_research_plan(
                plan=plan,
                budget=budget,
                stage=PlanningStage.REPLAN,
                min_total_sources_retrieved=min_total_sources_retrieved,
                min_total_cited_sources=min_total_cited_sources,
            )
            if self.settings.planner_validation_enabled
            else []
        )
        if validation_issues:
            await self.events.publish(
                run_id,
                "planning.validation.failed",
                {
                    "planning_stage": PlanningStage.REPLAN.value,
                    "issues": validation_issues,
                    "attempt": 1,
                },
            )
            raise RuntimeError("Replan failed validation: " + "; ".join(validation_issues))
        existing_stream_names = {stream.name for stream in await self.store.list_streams(run_id)}
        new_streams = [
            stream for stream in plan.streams if stream.name not in existing_stream_names
        ]
        version = await self.store.get_next_plan_version(run_id)
        await self._record_llm_usage(
            run_id=run_id,
            phase="plan_recreate",
            model=model_config.planner_model,
            usage=plan_result.usage,
            metadata={
                "version": version,
                "replan_count": replan_count,
                **(plan_result.metadata or {}),
            },
        )
        if not new_streams:
            return {"replan_count": replan_count, "should_replan": False}
        await self.events.publish(
            run_id,
            "planning.validation.passed",
            {
                "planning_stage": PlanningStage.REPLAN.value,
                "validation_checks": (
                    plan.planning_artifact.validation_checks
                    if plan.planning_artifact is not None
                    else []
                ),
            },
        )
        _, stream_ids = await self.store.save_plan(
            run_id,
            plan,
            version,
            streams_to_queue=new_streams,
        )
        await self.events.publish(
            run_id,
            "plan.created",
            {
                "version": version,
                "summary": plan.summary,
                "stream_count": len(new_streams),
                "planning_stage": PlanningStage.REPLAN.value,
                "prompt_template_version": (plan_result.metadata or {}).get(
                    "prompt_template_version"
                ),
            },
        )
        if plan.planning_artifact is not None:
            for record in plan.planning_artifact.discovery_records:
                await self.events.publish(
                    run_id,
                    "planning.discovery.recorded",
                    {
                        "planning_stage": PlanningStage.REPLAN.value,
                        "query": record.query,
                        "provider": record.provider,
                        "result_count": record.result_count,
                        "titles": record.titles,
                    },
                )
        for stream, stream_id in zip(new_streams, stream_ids, strict=True):
            await self.events.publish(
                run_id,
                "stream.created",
                {
                    "stream_id": stream_id,
                    "name": stream.name,
                    "objective": stream.objective,
                    "model": stream.model,
                },
            )
        return {"replan_count": replan_count, "should_replan": False}

    async def _synthesize_node(self, state: ResearchGraphState) -> dict[str, Any]:
        run_id = state["run_id"]
        run_state = await self.store.get_run_execution_state(run_id)
        agent_config = await self._get_agent_config(run_id)
        model_config = await self._get_model_config(run_id)
        profile_id, memory_policy = await self._get_run_profile(run_id)
        await self._ensure_run_active(run_id)
        plan = await self.store.get_latest_plan(run_id)
        if plan is None:
            raise RuntimeError("Cannot synthesize without a plan")
        notes = await self.store.list_notes(run_id)
        context_pack = (
            await self.context_assembler.assemble(
                run_id=run_id,
                question=state["question"],
                profile_id=profile_id,
                phase=ContextPhase.SYNTHESIZE,
                memory_policy=memory_policy,
            )
            if self.context_assembler is not None and memory_policy.enabled
            else None
        )
        report_result = await self.report_writer.write_report(
            question=state["question"],
            plan=plan,
            notes=notes,
            agent_config=agent_config,
            model_config=model_config,
            context_pack=context_pack,
            output_contract=_output_contract_from_metadata(
                run_state.metadata if run_state is not None else {}
            ),
        )
        draft_report = report_result.value
        await self.store.save_draft_report(run_id, draft_report.model_dump(mode="json"))
        await self._record_llm_usage(
            run_id=run_id,
            phase="report_synthesize",
            model=model_config.lead_model,
            usage=report_result.usage,
            metadata=report_result.metadata,
        )
        await self.events.publish(
            run_id,
            "report.drafted",
            {
                "section_count": len(draft_report.sections),
                "open_questions": len(draft_report.open_questions),
                "prompt_template_version": (report_result.metadata or {}).get(
                    "prompt_template_version"
                ),
            },
        )
        return {}

    async def _ground_node(self, state: ResearchGraphState) -> dict[str, Any]:
        run_id = state["run_id"]
        telemetry_context = (
            self.telemetry.span("report.ground", run_id=run_id)
            if self.telemetry is not None
            else None
        )
        if telemetry_context is None:
            return await self._ground_node_impl(state)
        with telemetry_context:
            return await self._ground_node_impl(state)

    async def _ground_node_impl(self, state: ResearchGraphState) -> dict[str, Any]:
        run_id = state["run_id"]
        agent_config = await self._get_agent_config(run_id)
        model_config = await self._get_model_config(run_id)
        await self._ensure_run_active(run_id)
        await self.store.update_run_status(run_id, RunStatus.GROUNDING)
        raw_draft = await self.store.get_draft_report(run_id)
        if raw_draft is None:
            raise RuntimeError("Cannot ground a report that does not exist")
        draft = DraftReport.model_validate(raw_draft)

        claim_rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        citation_candidates: list[CitationCandidate] = []
        verified_claims = 0
        max_verified_claims = max(1, self.worker.settings.grounding_max_claims_per_run)

        for section in draft.sections:
            for ordinal, claim in enumerate(section.claims, start=1):
                await self._ensure_run_active(run_id)
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
                    await self.events.publish(
                        run_id,
                        "citation.verification.skipped",
                        {
                            "section_title": section.title,
                            "claim": claim,
                            "reason": "grounding_max_claims_per_run",
                            "limit": max_verified_claims,
                        },
                    )
                    continue
                verified_claims += 1
                candidates = await self._retrieve_supporting_passages(run_id=run_id, claim=claim)
                verification_result = await self.verifier.verify(
                    claim=claim,
                    candidates=candidates,
                    agent_config=agent_config,
                    model_config=model_config,
                )
                verification = verification_result.value
                await self._record_llm_usage(
                    run_id=run_id,
                    phase="claim_verify",
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
                    await self.events.publish(
                        run_id,
                        "claim.repair.skipped",
                        {
                            "section_title": section.title,
                            "claim": claim,
                            "reason": "uncertainty_or_absence_claim",
                        },
                    )
                else:
                    while (
                        verification.support_label == CitationSupportLabel.UNSUPPORTED
                        and repair_attempts < self.worker.settings.max_claim_repairs
                    ):
                        repair_attempts += 1
                        new_passages = await self.worker.collect_supporting_passages(
                            run_id=run_id,
                            claim=claim,
                            section_title=section.title,
                        )
                        if not new_passages:
                            break
                        candidates = await self._retrieve_supporting_passages(
                            run_id=run_id,
                            claim=claim,
                        )
                        verification_result = await self.verifier.verify(
                            claim=claim,
                            candidates=candidates,
                            agent_config=agent_config,
                            model_config=model_config,
                        )
                        verification = verification_result.value
                        await self._record_llm_usage(
                            run_id=run_id,
                            phase="claim_reverify",
                            model=model_config.verifier_model,
                            usage=verification_result.usage,
                            metadata={
                                "section_title": section.title,
                                "repair_attempt": repair_attempts,
                                **(verification_result.metadata or {}),
                            },
                        )

                await self.events.publish(
                    run_id,
                    "citation.verified",
                    {
                        "section_title": section.title,
                        "claim": claim,
                        "support_label": verification.support_label.value,
                        "repair_attempts": repair_attempts,
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
                        quote=(verification.quote or chosen.text[:240]),
                        confidence=max(verification.confidence, 0.85),
                    )
                    claim_rows_by_key[claim_key]["support_label"] = verification.support_label.value
                    claim_rows_by_key[claim_key]["confidence"] = verification.confidence
                    await self.events.publish(
                        run_id,
                        "citation.contradicted",
                        {
                            "section_title": section.title,
                            "claim": claim,
                            "source_id": chosen.source_id,
                            "passage_index": chosen.passage_index,
                            "reason": "absence_claim_counterevidence",
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

        registry_entries = await self.store.list_source_registry_entries(run_id)
        audit_result = audit_citation_candidates(
            run_id=run_id,
            candidates=citation_candidates,
            registry_entries=registry_entries,
        )
        kept_keys = {
            (candidate.section_title, candidate.ordinal)
            for candidate in audit_result.kept
            if candidate.citation is not None
        }
        unsupported_claims: list[str] = []
        citations: list[CitationRecord] = []
        citation_rows: list[dict[str, Any]] = []
        confidence_values: list[float] = []
        source_registry_annotations: list[dict[str, Any]] = []

        for audit in audit_result.audits:
            if audit.decision != CitationAuditDecision.REMOVED:
                continue
            for reason in audit.reasons:
                if self.telemetry is not None:
                    self.telemetry.record_citation_removed(reason=reason.value)
            await self.events.publish(
                run_id,
                "citation.removed",
                {
                    "section_title": audit.section_title,
                    "ordinal": audit.ordinal,
                    "claim": audit.claim,
                    "reasons": [reason.value for reason in audit.reasons],
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

        unsupported_claims = dedupe_preserve_order(unsupported_claims)
        if self.telemetry is not None and unsupported_claims:
            self.telemetry.record_unsupported_claim(len(unsupported_claims))

        final_report = FinalReport(
            markdown=self._render_final_markdown(
                draft=draft,
                kept_keys=kept_keys,
                citations=citations,
                unsupported_claims=unsupported_claims,
            ),
            citations=citations,
            unsupported_claims=unsupported_claims,
            confidence=round(sum(confidence_values) / max(len(confidence_values), 1), 3),
            title=draft.title or derive_report_title(state["question"]),
            conversation_topic=(
                draft.conversation_topic or derive_conversation_topic(state["question"])
            ),
        )
        await self.store.update_run_metadata(
            run_id,
            {
                "report_title": final_report.title,
                "conversation_topic": final_report.conversation_topic,
            },
        )
        await self.store.replace_claims_and_citations(
            run_id,
            list(claim_rows_by_key.values()),
            citation_rows,
        )
        await self.store.annotate_source_registry_entries(run_id, source_registry_annotations)
        await self.store.replace_citation_audits(run_id, audit_result.audits)
        await self.events.publish(
            run_id,
            "citation.audit.completed",
            {
                "kept": len(audit_result.kept),
                "removed": len(audit_result.removed),
            },
        )
        await self.events.publish(
            run_id,
            "report.sanitized",
            {
                "citation_count": len(citations),
                "removed_citations": len(audit_result.removed),
            },
        )
        await self.store.update_run_status(run_id, RunStatus.COMPLETED, final_report=final_report)
        await self.events.publish(
            run_id,
            "report.completed",
            {
                "citation_count": len(citations),
                "unsupported_claims": len(unsupported_claims),
                "confidence": final_report.confidence,
            },
        )
        return {}

    def _render_final_markdown(
        self,
        *,
        draft: DraftReport,
        kept_keys: set[tuple[str, int]],
        citations: Sequence[CitationRecord],
        unsupported_claims: Sequence[str],
    ) -> str:
        citation_numbers = _citation_numbers_by_record(citations)
        report_title = clean_text(draft.title or "Research Report")
        markdown_lines = [
            f"# {report_title}",
            "",
            _sanitize_report_claim_text(draft.executive_summary),
            "",
        ]
        citation_index = 0
        for section in draft.sections:
            markdown_lines.extend(
                [f"## {section.title}", "", _sanitize_report_claim_text(section.overview), ""]
            )
            for ordinal, claim in enumerate(section.claims, start=1):
                if (section.title, ordinal) not in kept_keys:
                    continue
                if citation_index >= len(citations):
                    continue
                citation = citations[citation_index]
                support_suffix = (
                    " (partial support)"
                    if citation.support_label == CitationSupportLabel.PARTIAL
                    else ""
                )
                citation_number = citation_numbers[id(citation)]
                claim_text = _sanitize_report_claim_text(claim).rstrip(".")
                if not claim_text:
                    citation_index += 1
                    continue
                markdown_lines.append(f"{claim_text}{support_suffix}. [{citation_number}]")
                citation_index += 1
            markdown_lines.append("")

        if unsupported_claims:
            markdown_lines.extend(["## Remaining Uncertainty", ""])
            for claim in unsupported_claims:
                markdown_lines.append(f"- {claim}")
            markdown_lines.append("")

        open_questions = [
            question.strip()
            for question in dedupe_preserve_order(draft.open_questions)
            if question and question.strip()
        ][:8]
        if open_questions:
            markdown_lines.extend(["## Open Questions", ""])
            for question in open_questions:
                question_text = question.rstrip(".")
                if not question_text.endswith("?"):
                    question_text = f"{question_text}."
                markdown_lines.append(f"- {question_text}")
            markdown_lines.append("")

        return "\n".join(markdown_lines).strip()


def _citation_numbers_by_record(citations: Sequence[CitationRecord]) -> dict[int, int]:
    by_url: dict[str, int] = {}
    by_record: dict[int, int] = {}
    next_number = 1
    for citation in citations:
        key = normalize_url(str(citation.source_url))
        number = by_url.get(key)
        if number is None:
            number = next_number
            by_url[key] = number
            next_number += 1
        by_record[id(citation)] = number
    return by_record


def _render_notes(notes: Sequence[dict[str, Any]]) -> str:
    if not notes:
        return "No prior notes."
    rendered = []
    for note in notes:
        rendered.append(
            {
                "stream_name": note.get("stream_name"),
                "source_title": note.get("source_title"),
                "source_url": note.get("source_url"),
                "source_kind": (
                    note.get("source_kind").value if note.get("source_kind") is not None else None
                ),
                "retrieval_method": (
                    note.get("retrieval_method").value
                    if note.get("retrieval_method") is not None
                    else None
                ),
                "trust_tier": (
                    note.get("trust_tier").value if note.get("trust_tier") is not None else None
                ),
                "trust_rationale": note.get("trust_rationale"),
                "summary": note.get("summary"),
                "key_facts": note.get("key_facts", []),
                "open_questions": note.get("open_questions", []),
                "confidence": note.get("confidence"),
            }
        )
    return str(rendered)


def _output_contract_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = metadata.get("output_contract")
    if not isinstance(raw, Mapping):
        return None
    contract: dict[str, Any] = {}
    min_words = _coerce_positive_int(raw.get("report_min_words"))
    max_words = _coerce_positive_int(raw.get("report_max_words"))
    if min_words is not None:
        contract["report_min_words"] = min_words
    if max_words is not None:
        contract["report_max_words"] = max(max_words, min_words or 0)
    return contract or None


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _render_output_contract(output_contract: Mapping[str, Any] | None) -> str:
    if not output_contract:
        return "- Use the default report length and sectioning contract."
    lines = []
    min_words = output_contract.get("report_min_words")
    max_words = output_contract.get("report_max_words")
    if min_words or max_words:
        if min_words and max_words:
            lines.append(
                "- Target final report length: "
                f"{min_words}-{max_words} words before source excerpts. Treat the "
                "minimum as a hard floor unless the notes are genuinely too sparse."
            )
        elif min_words:
            lines.append(
                f"- Target final report length: at least {min_words} words. Treat this "
                "minimum as a hard floor unless the notes are genuinely too sparse."
            )
        elif max_words:
            lines.append(f"- Target final report length: no more than {max_words} words.")
        lines.append(
            "- To meet the length target, expand the executive summary and section "
            "overviews with supported analysis, caveats, comparison logic, and evidence "
            "gaps; do not add unsupported claims or filler."
        )
        lines.append(
            "- Put unresolved verification gaps in open_questions so the final report can "
            "surface them without presenting them as supported claims."
        )
        lines.append(
            "- Preserve citation support and uncertainty handling over hitting the "
            "length target exactly."
        )
    return (
        "\n".join(lines)
        if lines
        else "- Use the default report length and sectioning contract."
    )
