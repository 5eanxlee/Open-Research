from __future__ import annotations

import asyncio
import importlib.util
import socket
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from sqlalchemy.exc import OperationalError

from .asset_ingestion import extract_uploaded_file
from .artifacts import build_artifact_store
from .config import Settings, get_settings
from .db import ResearchStore, create_engine_and_sessionmaker
from .domain import (
    AgentConfig,
    ApprovalDecision,
    ApprovalDecisionKind,
    AssetProcessingStatus,
    AsyncJob,
    BehaviorAssessment,
    BudgetPolicy,
    ClarificationQuestion,
    ClarificationTurn,
    ClarificationSession,
    ClarifierConfig,
    CreateProjectRequest,
    CreateRunRequest,
    ExecutionMode,
    ModelConfig,
    ModelConfigOverride,
    PlanApprovalStatus,
    PlanningStage,
    PlanPreview,
    ProfileFeedback,
    ProfilePreferences,
    ProjectDetail,
    ProjectSummary,
    ProfileRecord,
    PublicRuntimeConfig,
    RecommendedBudget,
    RunConversationMessage,
    RunConversationReply,
    RunConversationRequest,
    RunConversationRole,
    ResearchAssetRecord,
    ResearchAssetUsage,
    ResearchInputAsset,
    RunDetail,
    RunNoteRecord,
    RunStatus,
    RunSummary,
    RunWorkspaceSnapshot,
    StagedAssetRecord,
    SourceCatalogEntry,
    is_terminal_run_status,
    resolve_model_config,
)
from .events import EventBroker, EventStreamService, RunEventService
from .memory import (
    BehaviorJudge,
    ContextAssembler,
    MemoryCompiler,
    merge_feedback_into_preferences,
)
from .observability import ResearchTelemetry
from .pipeline import (
    HeuristicClaimVerifier,
    HeuristicGapAnalyzer,
    HeuristicNoteWriter,
    HeuristicPlanner,
    HeuristicReportWriter,
    OpenAIClaimVerifier,
    OpenAINoteWriter,
    OpenAIPlanner,
    OpenAIReportWriter,
    ResearchOrchestrator,
    ResearchWorker,
    RunCancelledError,
)
from .prompting import (
    PROMPT_PROFILE_VERSION,
    SOURCE_TRUST_POLICY_VERSION,
    conversation_system_prompt,
    prompt_profile_metadata,
    resolve_agent_config,
)
from .providers import (
    BraveSearchProvider,
    BrowserbaseFetchProvider,
    BrowserbaseSessionFetchProvider,
    EmbeddingProvider,
    ExaSearchProvider,
    FetchPipeline,
    FirecrawlFetchProvider,
    HeuristicPassageReranker,
    MockEmbeddingProvider,
    MockFetchProvider,
    MockSearchProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OpenAIJsonClient,
    PassageReranker,
    PlaywrightFetchProvider,
    ProviderHooks,
    ProviderRetryNotice,
    RetriedEmbeddingProvider,
    RetriedFetchProvider,
    RetriedSearchProvider,
    SearchPipeline,
    SentenceTransformersReranker,
    TavilySearchProvider,
    provider_hooks_scope,
)
from .workspace import build_run_workspace_snapshot


class WorkflowEngine(Protocol):
    backend_name: str

    async def init(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def start_run(
        self,
        *,
        run_id: str,
        question: str,
        budget: BudgetPolicy,
    ) -> None: ...

    async def cancel_run(self, run_id: str) -> None: ...

    async def describe_run(self, run_id: str) -> dict[str, str] | None: ...

    async def is_healthy(self) -> bool: ...


def _require_secret(secret, *, setting_name: str) -> str:
    if secret is None:
        raise ValueError(f"{setting_name} must be configured for this backend.")
    return secret.get_secret_value()


def _retry_kwargs(settings: Settings) -> dict[str, float | int]:
    return {
        "max_attempts": settings.provider_retry_attempts,
        "base_delay_seconds": settings.provider_retry_base_seconds,
        "max_delay_seconds": settings.provider_retry_max_seconds,
        "cooldown_failures": settings.provider_cooldown_failures,
        "cooldown_seconds": settings.provider_cooldown_seconds,
    }


def _budget_limits() -> dict[str, dict[str, int]]:
    return {
        "max_streams": {"min": 1, "max": 30},
        "max_replans": {"min": 0, "max": 5},
        "max_queries_per_stream": {"min": 1, "max": 25},
        "max_results_per_query": {"min": 1, "max": 20},
        "max_sources_per_stream": {"min": 1, "max": 20},
        "per_domain_limit": {"min": 1, "max": 10},
    }


def _memory_policy_limits() -> dict[str, dict[str, float]]:
    return {
        "retrieval_limit": {"min": 1, "max": 50},
        "planning_budget_tokens": {"min": 0, "max": 6000},
        "research_budget_tokens": {"min": 0, "max": 4000},
        "synthesis_budget_tokens": {"min": 0, "max": 6000},
        "grounding_budget_tokens": {"min": 0, "max": 1000},
        "stale_penalty": {"min": 0.0, "max": 2.0},
        "conflict_penalty": {"min": 0.0, "max": 2.0},
    }


def _system_budget_caps() -> BudgetPolicy:
    return BudgetPolicy(
        max_streams=30,
        max_replans=5,
        max_queries_per_stream=25,
        max_results_per_query=20,
        max_sources_per_stream=20,
        per_domain_limit=10,
    )


def _clamp_budget(
    requested_budget: BudgetPolicy,
    *,
    recommended_budget: RecommendedBudget | BudgetPolicy | None = None,
) -> tuple[BudgetPolicy, str]:
    caps = _system_budget_caps()
    recommended = recommended_budget or requested_budget
    effective = BudgetPolicy(
        max_streams=min(caps.max_streams, requested_budget.max_streams, recommended.max_streams),
        max_replans=min(caps.max_replans, requested_budget.max_replans, recommended.max_replans),
        max_queries_per_stream=min(
            caps.max_queries_per_stream,
            requested_budget.max_queries_per_stream,
            recommended.max_queries_per_stream,
        ),
        max_results_per_query=min(
            caps.max_results_per_query,
            requested_budget.max_results_per_query,
            recommended.max_results_per_query,
        ),
        max_sources_per_stream=min(
            caps.max_sources_per_stream,
            requested_budget.max_sources_per_stream,
            recommended.max_sources_per_stream,
        ),
        per_domain_limit=min(
            caps.per_domain_limit,
            requested_budget.per_domain_limit,
            recommended.per_domain_limit,
        ),
    )
    reason = (
        "Effective budget clamps the planner recommendation against the explicit request "
        "and system caps."
    )
    return effective, reason


def _question_ambiguity_score(question: str) -> float:
    lowered = question.lower()
    score = 0.0
    if len(question.split()) < 10:
        score += 0.35
    if any(token in lowered for token in ("this", "that", "it", "best approach", "what about")):
        score += 0.25
    if "compare" not in lowered and any(token in lowered for token in ("vs", "versus")):
        score += 0.15
    if ":" not in question and "," not in question:
        score += 0.05
    return min(score, 1.0)


def _compose_clarified_question(question: str, session: ClarificationSession | None) -> str:
    if session is None or not session.turns:
        return question
    clarification_lines = [
        f"- {turn.prompt.strip()} Answer: {turn.response.strip()}"
        for turn in session.turns
        if turn.response.strip()
    ]
    if not clarification_lines:
        return question
    return f"{question}\n\nClarifications:\n" + "\n".join(clarification_lines)


def _render_asset_block(title: str, assets: list[ResearchAssetRecord], *, include_content: bool) -> str:
    if not assets:
        return ""
    lines = [title]
    for index, asset in enumerate(assets, start=1):
        lines.append(f"{index}. {asset.label}")
        if asset.project_id:
            lines.append("   Origin: project corpus")
        elif asset.run_id:
            lines.append("   Origin: run attachment")
        if asset.description:
            lines.append(f"   Description: {asset.description}")
        if asset.url:
            lines.append(f"   URL: {asset.url}")
        content = asset.extracted_text or asset.content_text
        if include_content and content:
            lines.append("   Content:")
            lines.extend(f"   {line}" for line in content.strip().splitlines() if line.strip())
    return "\n".join(lines)


def _augment_question_with_assets(
    question: str,
    assets: list[ResearchAssetRecord],
    *,
    include_reference_context: bool = False,
) -> str:
    ready_assets = [
        asset for asset in assets if asset.processing_status == AssetProcessingStatus.READY
    ]
    planning_assets = [
        asset for asset in ready_assets if asset.usage == ResearchAssetUsage.PLANNING_CONTEXT
    ]
    reference_assets = [
        asset for asset in ready_assets if asset.usage == ResearchAssetUsage.REFERENCE_SOURCE
    ]
    sections = [question]
    planning_block = _render_asset_block(
        "User-provided planning context that must shape the research plan and approval decision:",
        planning_assets,
        include_content=True,
    )
    if planning_block:
        sections.append(planning_block)
    if include_reference_context:
        reference_block = _render_asset_block(
            "User-provided reference materials that should be incorporated during research:",
            reference_assets,
            include_content=False,
        )
        if reference_block:
            sections.append(reference_block)
    return "\n\n".join(section for section in sections if section.strip())


def _asset_error_messages(assets: list[ResearchAssetRecord]) -> list[str]:
    return [
        f"{asset.label}: {asset.processing_error}"
        for asset in assets
        if asset.processing_status == AssetProcessingStatus.FAILED and asset.processing_error
    ]


def _clip_text(value: str | None, *, limit: int = 1200) -> str:
    if not value:
        return ""
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1].rstrip() + "…"


def _conversation_references(detail: RunDetail, passages: list[dict[str, object]]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        title = str(passage.get("source_title") or "").strip()
        url = str(passage.get("source_url") or "").strip()
        label = title or url
        if label and label not in seen:
            seen.add(label)
            refs.append(label)
    for citation in (detail.final_report.citations if detail.final_report else [])[:4]:
        label = citation.source_title or str(citation.source_url)
        if label not in seen:
            seen.add(label)
            refs.append(label)
    return refs[:8]


def _build_uploaded_asset(
    *,
    usage: ResearchAssetUsage,
    file_name: str,
    content_type: str | None,
    data: bytes,
    settings: Settings,
) -> tuple[ResearchInputAsset, dict[str, object]]:
    extracted = extract_uploaded_file(
        settings=settings,
        file_name=file_name,
        content_type=content_type,
        data=data,
    )
    label = Path(file_name).name
    asset = ResearchInputAsset(
        source_type="file",
        usage=usage,
        label=label,
        content_text=extracted.extracted_text,
        file_name=file_name,
        content_type=extracted.content_type,
    )
    return asset, extracted.metadata()


def _build_source_catalog(settings: Settings) -> list[SourceCatalogEntry]:
    def entry(
        *,
        id: str,
        name: str,
        description: str,
        backend_kind: str,
        configured: bool,
        auth_required: bool,
        supports_search: bool = False,
        supports_fetch: bool = False,
        supports_primary_sources: bool = False,
        supports_advanced_search: bool = False,
    ) -> SourceCatalogEntry:
        return SourceCatalogEntry(
            id=id,
            name=name,
            description=description,
            backend_kind=backend_kind,
            default_enabled=configured,
            configured=configured,
            auth_required=auth_required,
            supports_search=supports_search,
            supports_fetch=supports_fetch,
            supports_internal_docs=supports_primary_sources,
            supports_primary_sources=supports_primary_sources,
            supports_advanced_search=supports_advanced_search,
            status_reason=None if configured else "Not configured in this deployment.",
        )

    return [
        entry(
            id="brave",
            name="Brave Search",
            description="Web search provider focused on public-web coverage.",
            backend_kind="search",
            configured=settings.brave_api_key is not None or settings.search_backend in {"mock", "auto"},
            auth_required=True,
            supports_search=True,
            supports_advanced_search=False,
        ),
        entry(
            id="exa",
            name="Exa Search",
            description="Semantic search provider useful for docs and technical content.",
            backend_kind="search",
            configured=settings.exa_api_key is not None or settings.search_backend in {"mock", "auto"},
            auth_required=True,
            supports_search=True,
            supports_primary_sources=True,
            supports_advanced_search=True,
        ),
        entry(
            id="tavily",
            name="Tavily Search",
            description="General search API with quick public-web retrieval.",
            backend_kind="search",
            configured=settings.tavily_api_key is not None or settings.search_backend in {"mock", "auto"},
            auth_required=True,
            supports_search=True,
        ),
        entry(
            id="mock",
            name="Mock Search/Fetch",
            description="Synthetic fallback path used in local demo mode.",
            backend_kind="search_fetch",
            configured=True,
            auth_required=False,
            supports_search=True,
            supports_fetch=True,
        ),
        entry(
            id="firecrawl",
            name="Firecrawl Fetch",
            description="Fetches and normalizes public pages into main-content markdown.",
            backend_kind="fetch",
            configured=settings.firecrawl_api_key is not None or settings.fetch_backend in {"mock", "auto"},
            auth_required=True,
            supports_fetch=True,
            supports_primary_sources=True,
        ),
        entry(
            id="browserbase",
            name="Browserbase Fetch",
            description="Browser-backed fetch path for difficult pages.",
            backend_kind="fetch",
            configured=settings.browserbase_api_key is not None or settings.fetch_backend in {"mock", "auto"},
            auth_required=True,
            supports_fetch=True,
        ),
        entry(
            id="browserbase-session",
            name="Browserbase Session",
            description="Longer-lived browser sessions for harder interactive pages.",
            backend_kind="fetch",
            configured=settings.browserbase_api_key is not None or settings.fetch_backend in {"mock", "auto"},
            auth_required=True,
            supports_fetch=True,
        ),
        entry(
            id="playwright",
            name="Playwright Fetch",
            description="Local browser fallback for public pages.",
            backend_kind="fetch",
            configured=True,
            auth_required=False,
            supports_fetch=True,
        ),
    ]


def _resolve_oss_api_key(secret) -> str:
    if secret is None:
        return "EMPTY"
    return secret.get_secret_value()


def _build_search_provider(settings: Settings):
    if settings.search_backend == "auto":
        providers = []
        if settings.brave_api_key is not None:
            providers.append(
                RetriedSearchProvider(
                    BraveSearchProvider(
                        _require_secret(settings.brave_api_key, setting_name="BRAVE_API_KEY"),
                        timeout=settings.http_timeout_seconds,
                    ),
                    **_retry_kwargs(settings),
                )
            )
        if settings.exa_api_key is not None:
            providers.append(
                RetriedSearchProvider(
                    ExaSearchProvider(
                        _require_secret(settings.exa_api_key, setting_name="EXA_API_KEY"),
                        timeout=settings.http_timeout_seconds,
                    ),
                    **_retry_kwargs(settings),
                )
            )
        if settings.tavily_api_key is not None:
            providers.append(
                RetriedSearchProvider(
                    TavilySearchProvider(
                        _require_secret(settings.tavily_api_key, setting_name="TAVILY_API_KEY"),
                        timeout=settings.http_timeout_seconds,
                    ),
                    **_retry_kwargs(settings),
                )
            )
        if not providers:
            providers.append(MockSearchProvider())
        return SearchPipeline(providers)

    if settings.search_backend == "mock":
        return MockSearchProvider()
    if settings.search_backend == "brave":
        return RetriedSearchProvider(
            BraveSearchProvider(
                _require_secret(settings.brave_api_key, setting_name="BRAVE_API_KEY"),
                timeout=settings.http_timeout_seconds,
            ),
            **_retry_kwargs(settings),
        )
    if settings.search_backend == "exa":
        return RetriedSearchProvider(
            ExaSearchProvider(
                _require_secret(settings.exa_api_key, setting_name="EXA_API_KEY"),
                timeout=settings.http_timeout_seconds,
            ),
            **_retry_kwargs(settings),
        )
    return RetriedSearchProvider(
        TavilySearchProvider(
            _require_secret(settings.tavily_api_key, setting_name="TAVILY_API_KEY"),
            timeout=settings.http_timeout_seconds,
        ),
        **_retry_kwargs(settings),
    )


def _build_fetch_provider(settings: Settings):
    if settings.fetch_backend == "auto":
        providers = []
        if settings.firecrawl_api_key is not None:
            providers.append(
                RetriedFetchProvider(
                    FirecrawlFetchProvider(
                        _require_secret(
                            settings.firecrawl_api_key,
                            setting_name="FIRECRAWL_API_KEY",
                        ),
                        timeout=settings.http_timeout_seconds,
                    ),
                    **_retry_kwargs(settings),
                )
            )
        if settings.browserbase_api_key is not None:
            providers.append(
                RetriedFetchProvider(
                    BrowserbaseFetchProvider(
                        _require_secret(
                            settings.browserbase_api_key,
                            setting_name="BROWSERBASE_API_KEY",
                        ),
                        timeout=settings.http_timeout_seconds,
                        use_proxies=settings.browserbase_use_proxies,
                    ),
                    **_retry_kwargs(settings),
                )
            )
            providers.append(
                RetriedFetchProvider(
                    BrowserbaseSessionFetchProvider(
                        _require_secret(
                            settings.browserbase_api_key,
                            setting_name="BROWSERBASE_API_KEY",
                        ),
                        project_id=settings.browserbase_project_id,
                        timeout=settings.http_timeout_seconds,
                        use_proxies=settings.browserbase_use_proxies,
                        keep_alive=settings.browserbase_session_keep_alive,
                    ),
                    **_retry_kwargs(settings),
                )
            )
        if not providers:
            providers.append(MockFetchProvider())
        return FetchPipeline(providers)

    if settings.fetch_backend == "mock":
        return MockFetchProvider()
    if settings.fetch_backend == "firecrawl":
        return RetriedFetchProvider(
            FirecrawlFetchProvider(
                _require_secret(settings.firecrawl_api_key, setting_name="FIRECRAWL_API_KEY"),
                timeout=settings.http_timeout_seconds,
            ),
            **_retry_kwargs(settings),
        )
    if settings.fetch_backend == "browserbase":
        return RetriedFetchProvider(
            BrowserbaseFetchProvider(
                _require_secret(settings.browserbase_api_key, setting_name="BROWSERBASE_API_KEY"),
                timeout=settings.http_timeout_seconds,
                use_proxies=settings.browserbase_use_proxies,
            ),
            **_retry_kwargs(settings),
        )
    if settings.fetch_backend == "browserbase_session":
        return RetriedFetchProvider(
            BrowserbaseSessionFetchProvider(
                _require_secret(settings.browserbase_api_key, setting_name="BROWSERBASE_API_KEY"),
                project_id=settings.browserbase_project_id,
                timeout=settings.http_timeout_seconds,
                use_proxies=settings.browserbase_use_proxies,
                keep_alive=settings.browserbase_session_keep_alive,
            ),
            **_retry_kwargs(settings),
        )
    return RetriedFetchProvider(
        PlaywrightFetchProvider(timeout=settings.playwright_timeout_seconds),
        **_retry_kwargs(settings),
    )


def _build_embedding_provider(settings: Settings) -> EmbeddingProvider | None:
    if settings.resolved_embedding_backend == "disabled":
        return None
    if settings.resolved_embedding_backend == "openai":
        return RetriedEmbeddingProvider(
            OpenAIEmbeddingProvider(
                api_key=_require_secret(settings.openai_api_key, setting_name="OPENAI_API_KEY"),
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                base_url=settings.openai_base_url,
                timeout=settings.http_timeout_seconds,
                estimated_request_cost_usd=settings.embedding_request_cost_usd,
            ),
            **_retry_kwargs(settings),
        )
    if settings.resolved_embedding_backend == "openai_compatible":
        base_url = settings.embedding_base_url or settings.llm_base_url
        if base_url is None:
            raise ValueError(
                "OPEN_RESEARCH_EMBEDDING_BASE_URL or OPEN_RESEARCH_LLM_BASE_URL must be "
                "configured for openai_compatible embeddings."
            )
        api_key = _resolve_oss_api_key(settings.embedding_api_key or settings.llm_api_key)
        return RetriedEmbeddingProvider(
            OpenAICompatibleEmbeddingProvider(
                api_key=api_key,
                model=settings.embedding_model,
                dimensions=None,
                base_url=base_url,
                timeout=settings.http_timeout_seconds,
                estimated_request_cost_usd=settings.embedding_request_cost_usd,
            ),
            **_retry_kwargs(settings),
        )
    return RetriedEmbeddingProvider(
        MockEmbeddingProvider(dimensions=settings.embedding_dimensions),
        **_retry_kwargs(settings),
    )


def _build_reranker(settings: Settings) -> PassageReranker | None:
    if settings.resolved_reranker_backend == "disabled":
        return None
    if settings.resolved_reranker_backend == "sentence_transformers":
        if importlib.util.find_spec("sentence_transformers") is None:
            raise ValueError(
                "sentence-transformers is not installed. Install the grounding extra to "
                "enable the sentence_transformers reranker backend."
            )
        return SentenceTransformersReranker(model_name=settings.reranker_model)
    if (
        settings.reranker_backend == "auto"
        and importlib.util.find_spec("sentence_transformers") is not None
    ):
        return SentenceTransformersReranker(model_name=settings.reranker_model)
    return HeuristicPassageReranker()


class RunCoordinator:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self.runtime = runtime

    async def start_run(self, request: CreateRunRequest) -> RunSummary:
        requested_budget = request.budget or self.runtime.default_budget()
        agent_config = resolve_agent_config(
            request.agent_config or request.metadata.get("agent_config")
        )
        model_config = resolve_model_config(
            request.model_config_override or request.metadata.get("model_config_override"),
            defaults=self.runtime.default_model_config(),
        )
        profile_id = request.profile_id or "default"
        existing_profile = await self.runtime.store.get_profile(profile_id)
        if existing_profile is None:
            await self.runtime.store.upsert_profile_preferences(profile_id, ProfilePreferences())
        if request.project_id is not None:
            project = await self.runtime.store.get_project(request.project_id)
            if project is None:
                raise ValueError(f"Project {request.project_id} not found.")
        if request.staged_asset_ids:
            staged_assets = await self.runtime.store.list_staged_assets(request.staged_asset_ids)
            if len(staged_assets) != len(request.staged_asset_ids):
                found_ids = {asset.id for asset in staged_assets}
                missing = [asset_id for asset_id in request.staged_asset_ids if asset_id not in found_ids]
                raise ValueError(
                    "Some staged assets could not be found: " + ", ".join(sorted(missing))
                )
        source_selection = self.runtime.resolve_source_selection(request.source_selection)
        execution_mode = request.execution_mode
        requires_approval = self.runtime.should_require_plan_approval(
            execution_mode=execution_mode,
            requested_budget=requested_budget,
            explicit=request.require_plan_approval,
            source_selection=source_selection,
            model_config=model_config,
            user_supplied_budget=request.budget is not None,
            user_supplied_sources=request.source_selection is not None,
            user_supplied_model_override=request.model_config_override is not None,
        )
        metadata = dict(request.metadata)
        metadata["profile_id"] = profile_id
        metadata["requested_budget"] = requested_budget.model_dump(mode="json")
        metadata["effective_budget"] = requested_budget.model_dump(mode="json")
        metadata["execution_mode"] = execution_mode.value
        metadata["source_selection"] = source_selection
        metadata["approval_status"] = (
            PlanApprovalStatus.PENDING_CLARIFICATION.value
            if requires_approval
            else PlanApprovalStatus.NOT_REQUIRED.value
        )
        metadata["clarifier_config"] = (
            (request.clarifier_config or ClarifierConfig()).model_dump(mode="json")
        )
        metadata["model_config"] = model_config.model_dump(mode="json")
        if request.model_config_override is not None:
            metadata["model_config_override"] = request.model_config_override.model_dump(
                mode="json",
                exclude_none=True,
            )
        if request.memory_policy_override is not None:
            metadata["memory_policy_override"] = request.memory_policy_override.model_dump(
                mode="json"
            )
        metadata.update(
            prompt_profile_metadata(
                settings=self.runtime.settings,
                agent_config=agent_config,
            )
        )
        run = await self.runtime.store.create_run(
            request.question,
            requested_budget,
            profile_id=profile_id,
            project_id=request.project_id,
            metadata=metadata,
        )
        input_assets: list[ResearchAssetRecord] = []
        for asset in request.input_assets:
            input_assets.append(
                await self.runtime.store.save_research_asset(asset, run_id=run.id)
            )
        if request.staged_asset_ids:
            input_assets.extend(
                await self.runtime.store.materialize_staged_assets_to_run(
                    run_id=run.id,
                    staged_asset_ids=request.staged_asset_ids,
                )
            )
        effective_assets = await self.runtime.resolve_effective_assets(
            run_id=run.id,
            project_id=request.project_id,
        )
        ready_assets = [
            asset for asset in effective_assets if asset.processing_status == AssetProcessingStatus.READY
        ]
        effective_question = _augment_question_with_assets(
            request.question,
            ready_assets,
            include_reference_context=True,
        )
        await self.runtime.store.update_run_metadata(
            run.id,
            {
                "project_id": request.project_id,
                "input_asset_ids": [asset.id for asset in input_assets],
                "staged_asset_ids": list(request.staged_asset_ids),
                "effective_question": effective_question,
                "planning_context_asset_count": sum(
                    1 for asset in ready_assets if asset.usage == ResearchAssetUsage.PLANNING_CONTEXT
                ),
                "reference_asset_count": sum(
                    1 for asset in ready_assets if asset.usage == ResearchAssetUsage.REFERENCE_SOURCE
                ),
                "asset_processing_errors": _asset_error_messages(effective_assets),
            },
        )
        await self.runtime.store.set_run_execution_context(
            run.id,
            workflow_backend=self.runtime.workflow_backend_name,
        )
        await self.runtime.events.publish(
            run.id,
            "run.started",
            {
                "question": request.question,
                "budget": requested_budget.model_dump(mode="json"),
                "llm_backend": self.runtime.settings.resolved_llm_backend,
                "search_backend": self.runtime.settings.resolved_search_backend,
                "fetch_backend": self.runtime.settings.resolved_fetch_backend,
                "workflow_backend": self.runtime.workflow_backend_name,
                "prompt_profile_version": metadata["prompt_profile_version"],
                "prompt_model_family": metadata["prompt_model_family"],
                "profile_id": profile_id,
                "model_config": model_config.model_dump(mode="json"),
                "memory_policy_override": metadata.get("memory_policy_override"),
                "agent_config": agent_config.model_dump(mode="json"),
                "execution_mode": execution_mode.value,
                "source_selection": source_selection,
                "approval_status": metadata["approval_status"],
                "project_id": request.project_id,
                "input_asset_count": len(input_assets),
                "asset_processing_errors": _asset_error_messages(effective_assets),
            },
        )
        await self.runtime.events.publish(
            run.id,
            "prompt.profile.applied",
            {
                "prompt_profile_version": metadata["prompt_profile_version"],
                "source_trust_policy_version": metadata["source_trust_policy_version"],
                "prompt_model_family": metadata["prompt_model_family"],
                "prompt_templates": metadata["prompt_templates"],
            },
        )
        if requires_approval:
            await self.runtime.prepare_run_for_approval(
                run_id=run.id,
                question=effective_question,
                requested_budget=requested_budget,
            )
        else:
            await self.runtime.executor.launch_run(run.id, effective_question, requested_budget)
        return run

    async def cancel_run(self, run_id: str) -> RunDetail:
        state = await self.runtime.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        if is_terminal_run_status(state.status):
            raise ValueError("Run is already in a terminal state.")

        await self.runtime.store.request_run_cancel(run_id)
        cancellation_requested_at = datetime.now(UTC)
        await self.runtime.events.publish(
            run_id,
            "run.cancel_requested",
            {"status": state.status.value},
        )
        await self.runtime.events.publish(
            run_id,
            "run.cancellation_requested",
            {"status": state.status.value},
        )

        task = self.runtime._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
        elif self.runtime.workflow_engine is not None:
            await self.runtime.workflow_engine.cancel_run(run_id)
            await self.runtime.executor.finalize_cancelled_run(
                run_id,
                reason="Run cancelled by user request.",
            )
        else:
            await self.runtime.executor.finalize_cancelled_run(
                run_id,
                reason="Run cancelled before active execution.",
            )

        detail = await self.runtime.get_run_detail(run_id)
        if detail is None:
            raise KeyError(f"Run {run_id} not found")
        if detail.status == RunStatus.CANCELLED:
            updated_at = detail.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            latency = (updated_at - cancellation_requested_at).total_seconds()
            self.runtime.telemetry.record_cancellation_latency(latency)
        return detail

    async def resume_run(self, run_id: str) -> RunSummary:
        state = await self.runtime.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        if state.has_final_report or state.status == RunStatus.COMPLETED:
            raise ValueError("Completed runs cannot be resumed.")
        if state.status not in {RunStatus.CANCELLED, RunStatus.FAILED}:
            raise ValueError("Only failed or cancelled runs can be resumed.")
        active_task = self.runtime._tasks.get(run_id)
        if active_task is not None:
            if active_task.done():
                self.runtime._tasks.pop(run_id, None)
            elif is_terminal_run_status(state.status):
                active_task.cancel()
                self.runtime._tasks.pop(run_id, None)
            else:
                raise ValueError("Run is already active.")

        resumed_state = await self.runtime.store.prepare_run_for_resume(run_id)
        await self.runtime.events.publish(
            run_id,
            "run.resumed",
            {"prior_status": state.status.value},
        )
        approval_status = PlanApprovalStatus(
            resumed_state.metadata.get("approval_status", PlanApprovalStatus.NOT_REQUIRED.value)
        )
        if approval_status in {
            PlanApprovalStatus.PENDING_CLARIFICATION,
            PlanApprovalStatus.PENDING_APPROVAL,
            PlanApprovalStatus.CHANGES_REQUESTED,
        }:
            await self.runtime.prepare_run_for_approval(
                run_id=resumed_state.id,
                question=str(
                    resumed_state.metadata.get("effective_question", resumed_state.question)
                ),
                requested_budget=resumed_state.requested_budget or resumed_state.budget,
            )
        else:
            await self.runtime.executor.launch_run(
                resumed_state.id,
                str(resumed_state.metadata.get("effective_question", resumed_state.question)),
                resumed_state.effective_budget or resumed_state.budget,
            )
        run = await self.runtime.store.get_run(run_id)
        if run is None:
            raise KeyError(f"Run {run_id} not found")
        return run

    async def retry_run(self, run_id: str) -> RunSummary:
        state = await self.runtime.store.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        if not is_terminal_run_status(state.status):
            raise ValueError("Only terminal runs can be retried.")
        if run_id in self.runtime._tasks and not self.runtime._tasks[run_id].done():
            raise ValueError("Active runs cannot be retried.")

        metadata = dict(state.metadata)
        metadata["retry_of_run_id"] = run_id
        detail = await self.runtime.get_run_detail(run_id)
        if detail is None:
            raise KeyError(f"Run {run_id} not found")
        retry_run = await self.start_run(
            CreateRunRequest(
                question=state.question,
                budget=state.requested_budget or state.budget,
                project_id=state.project_id,
                execution_mode=state.execution_mode,
                input_assets=[
                    ResearchInputAsset(
                        source_type=asset.source_type,
                        usage=asset.usage,
                        label=asset.label,
                        description=asset.description,
                        url=asset.url,
                        content_text=asset.content_text,
                        content_type=asset.content_type,
                        file_name=asset.file_name,
                    )
                    for asset in detail.input_assets
                ],
                metadata=metadata,
            )
        )
        await self.runtime.events.publish(
            run_id,
            "run.retried",
            {"new_run_id": retry_run.id},
        )
        return retry_run


class WorkerExecutor:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self.runtime = runtime

    async def _retry_on_db_lock(self, operation) -> None:
        for attempt in range(5):
            try:
                await operation()
                return
            except OperationalError as exc:
                if "database is locked" not in str(exc).lower() or attempt == 4:
                    raise
                await asyncio.sleep(0.05 * (attempt + 1))

    async def launch_run(
        self,
        run_id: str,
        question: str,
        budget: BudgetPolicy,
        *,
        raise_on_error: bool = True,
    ) -> bool:
        try:
            if self.runtime.workflow_engine is None:
                scheduled = self.schedule_run(run_id, question, budget)
                if not scheduled:
                    raise ValueError("Run is already active.")
                return True
            await self.runtime.workflow_engine.start_run(
                run_id=run_id,
                question=question,
                budget=budget,
            )
            return True
        except Exception as exc:
            error_message = f"Failed to dispatch run for execution: {exc}"
            await self.runtime.store.update_run_status(
                run_id,
                RunStatus.FAILED,
                error_message=error_message,
                terminal_reason=error_message,
            )
            await self.runtime.events.publish(run_id, "run.failed", {"error": error_message})
            if raise_on_error:
                raise
            return False

    def schedule_run(self, run_id: str, question: str, budget: BudgetPolicy) -> bool:
        existing = self.runtime._tasks.get(run_id)
        if existing is not None and not existing.done():
            return False
        task = asyncio.create_task(
            self.execute_run(run_id, question, budget),
            name=f"run-{run_id}",
        )
        self.runtime._tasks[run_id] = task
        task.add_done_callback(lambda finished: self.handle_task_done(run_id, finished))
        return True

    async def execute_run(self, run_id: str, question: str, budget: BudgetPolicy) -> None:
        stop_heartbeat = asyncio.Event()
        heartbeat_task: asyncio.Task[None] | None = None
        prompt_profile_version = None
        prompt_model_family = None
        try:
            await self.runtime.store.set_run_execution_context(
                run_id,
                worker_id=self.runtime.worker_id,
                workflow_backend=self.runtime.workflow_backend_name,
            )
            state = await self.runtime.store.get_run_execution_state(run_id)
            if state is not None:
                prompt_profile_version = state.metadata.get("prompt_profile_version")
                prompt_model_family = state.metadata.get("prompt_model_family")
                if state.metadata.get("job_id"):
                    await self.runtime.store.update_job(
                        state.metadata["job_id"],
                        status=state.status.value,
                        started_at=datetime.now(UTC),
                    )
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(run_id, stop_heartbeat),
                name=f"heartbeat-{run_id}",
            )

            async def on_retry(notice: ProviderRetryNotice) -> None:
                self.runtime.telemetry.record_provider_retry(
                    category=notice.category,
                    provider=notice.provider_name,
                )
                await self.runtime.events.publish(
                    run_id,
                    "provider.retry",
                    {
                        "provider": notice.provider_name,
                        "category": notice.category,
                        "attempt": notice.attempt,
                        "max_attempts": notice.max_attempts,
                        "error": notice.error,
                        "in_cooldown": notice.in_cooldown,
                        "cooldown_seconds": notice.cooldown_seconds,
                    },
                )

            async def on_error(notice: ProviderRetryNotice) -> None:
                self.runtime.telemetry.record_provider_error(
                    category=notice.category,
                    provider=notice.provider_name,
                )

            with (
                self.runtime.telemetry.span(
                    "run.execute",
                    run_id=run_id,
                    workflow_backend=self.runtime.workflow_backend_name,
                    prompt_profile_version=prompt_profile_version,
                    prompt_model_family=prompt_model_family,
                ),
                provider_hooks_scope(ProviderHooks(on_retry=on_retry, on_error=on_error)),
            ):
                state = await self.runtime.store.get_run_execution_state(run_id)
                effective_assets = await self.runtime.resolve_effective_assets(
                    run_id=run_id,
                    project_id=(state.project_id if state is not None else None),
                )
                ready_reference_assets = [
                    asset
                    for asset in effective_assets
                    if asset.processing_status == AssetProcessingStatus.READY
                    and asset.usage == ResearchAssetUsage.REFERENCE_SOURCE
                ]
                if ready_reference_assets:
                    await self.runtime.orchestrator.worker.ingest_input_assets(
                        run_id=run_id,
                        question=question,
                        assets=ready_reference_assets,
                    )
                await self.runtime.orchestrator.execute(
                    run_id=run_id,
                    question=question,
                    budget=budget,
                )
        except RunCancelledError as exc:
            await self.finalize_cancelled_run(run_id, reason=str(exc))
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None:
                current_task.uncancel()
            state = await self.runtime.store.get_run_execution_state(run_id)
            if state is not None and state.status == RunStatus.CANCELLED:
                return
            if state is not None and state.cancel_requested:
                await asyncio.shield(
                    self.finalize_cancelled_run(
                        run_id,
                        reason="Run cancelled by user request.",
                    )
                )
                return
            if self.runtime._shutting_down:
                await self.runtime.events.publish(
                    run_id,
                    "run.shutdown",
                    {"reason": "runtime shutdown requested"},
                )
                return
            await self.runtime.store.requeue_inflight_work(run_id, include_failed=False)
            error_message = "Run execution was interrupted unexpectedly."
            await self.runtime.store.update_run_status(
                run_id,
                RunStatus.FAILED,
                error_message=error_message,
                terminal_reason=error_message,
            )
            await self.runtime.events.publish(run_id, "run.failed", {"error": error_message})
        except Exception as exc:
            error_message = str(exc)
            await self.runtime.store.update_run_status(
                run_id,
                RunStatus.FAILED,
                error_message=error_message,
                terminal_reason=error_message,
            )
            await self.runtime.events.publish(run_id, "run.failed", {"error": error_message})
        finally:
            if heartbeat_task is not None:
                stop_heartbeat.set()
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            detail = await self.runtime.get_run_detail(run_id)
            if detail is not None and is_terminal_run_status(detail.status):
                if detail.job_id is not None:
                    await self.runtime.store.update_job(
                        detail.job_id,
                        status=detail.status.value,
                        ended_at=detail.updated_at,
                        last_heartbeat_at=detail.last_heartbeat_at,
                    )
                if self.runtime.settings.profile_memory_enabled:
                    await self.runtime.memory_compiler.compile_run(
                        run_id=run_id,
                        profile_id=detail.profile_id,
                    )
                if self.runtime.settings.behavior_assessment_enabled:
                    await self.runtime.behavior_judge.assess_run(
                        run_id=run_id,
                        profile_id=detail.profile_id,
                    )
                self.runtime.telemetry.record_run_terminal(
                    status=detail.status.value,
                    cost_usd=detail.estimated_cost_usd,
                )

    async def _heartbeat_loop(self, run_id: str, stop_event: asyncio.Event) -> None:
        interval = max(1.0, self.runtime.settings.run_heartbeat_seconds)
        try:
            while not stop_event.is_set():
                heartbeat_at = await self.runtime.store.touch_run_heartbeat(
                    run_id,
                    worker_id=self.runtime.worker_id,
                    workflow_backend=self.runtime.workflow_backend_name,
                )
                state = await self.runtime.store.get_run_execution_state(run_id)
                if state is not None and state.metadata.get("job_id"):
                    await self.runtime.store.update_job(
                        state.metadata["job_id"],
                        status=state.status.value,
                        last_heartbeat_at=heartbeat_at,
                    )
                await self.runtime.events.publish(
                    run_id,
                    "run.heartbeat",
                    {
                        "worker_id": self.runtime.worker_id,
                        "workflow_backend": self.runtime.workflow_backend_name,
                        "heartbeat_at": heartbeat_at.isoformat(),
                    },
                )
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            return

    def handle_task_done(self, run_id: str, task: asyncio.Task[None]) -> None:
        self.runtime._tasks.pop(run_id, None)
        if task.cancelled():
            background_task = asyncio.create_task(self.finalize_cancelled_task(run_id))
            self.runtime._background_tasks.add(background_task)
            background_task.add_done_callback(self.runtime._background_tasks.discard)

    async def finalize_cancelled_task(self, run_id: str) -> None:
        if self.runtime._shutting_down:
            return
        state = await self.runtime.store.get_run_execution_state(run_id)
        if state is None or is_terminal_run_status(state.status):
            return
        if state.cancel_requested:
            await self.finalize_cancelled_run(run_id, reason="Run cancelled by user request.")
            return

        await self.runtime.store.requeue_inflight_work(run_id, include_failed=False)
        error_message = "Run execution was interrupted unexpectedly."
        await self.runtime.store.update_run_status(
            run_id,
            RunStatus.FAILED,
            error_message=error_message,
            terminal_reason=error_message,
        )
        await self.runtime.events.publish(run_id, "run.failed", {"error": error_message})

    async def finalize_cancelled_run(self, run_id: str, *, reason: str) -> None:
        await self._retry_on_db_lock(
            lambda: self.runtime.store.requeue_inflight_work(run_id, include_failed=False)
        )
        await self._retry_on_db_lock(
            lambda: self.runtime.store.update_run_status(
                run_id,
                RunStatus.CANCELLED,
                terminal_reason=reason,
            )
        )
        await self.runtime.events.publish(run_id, "run.cancelled", {"reason": reason})

    async def recover_incomplete_runs(self) -> None:
        recoverable_runs = await self.runtime.store.list_recoverable_runs()
        for state in recoverable_runs:
            await self.runtime.store.requeue_inflight_work(state.id, include_failed=True)
            if state.cancel_requested:
                await self.finalize_cancelled_run(
                    state.id,
                    reason="Recovered pending cancellation request.",
                )
                continue
            scheduled = await self.launch_run(
                state.id,
                state.question,
                state.budget,
                raise_on_error=False,
            )
            if scheduled:
                await self.runtime.events.publish(
                    state.id,
                    "run.recovered",
                    {"prior_status": state.status.value},
                )


class RunReconciler:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self.runtime = runtime
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._loop(), name="run-reconciler")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                await self.run_once()
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=max(1.0, self.runtime.settings.reconciler_interval_seconds),
                    )
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            return

    async def run_once(self) -> None:
        stale_before = datetime.now(UTC) - timedelta(
            seconds=max(1.0, self.runtime.settings.stale_run_timeout_seconds)
        )
        stale_runs = await self.runtime.store.list_stale_runs(stale_before=stale_before)
        self.runtime.telemetry.record_heartbeat_overdue(len(stale_runs))

        for state in stale_runs:
            if is_terminal_run_status(state.status):
                continue
            active_task = self.runtime._tasks.get(state.id)
            if active_task is not None and not active_task.done():
                continue

            workflow_state = await self.runtime.describe_workflow_run(state.id)
            if workflow_state is not None and workflow_state.get("status") == "running":
                continue
            if workflow_state is not None and workflow_state.get("status") == "cancelled":
                await self.runtime.executor.finalize_cancelled_run(
                    state.id,
                    reason="Workflow backend reported cancellation.",
                )
                continue
            if workflow_state is not None and workflow_state.get("status") == "failed":
                reason = workflow_state.get("reason", "Workflow backend reported failure.")
                await self.runtime.store.update_run_status(
                    state.id,
                    RunStatus.FAILED,
                    error_message=reason,
                    terminal_reason=reason,
                )
                await self.runtime.events.publish(state.id, "run.failed", {"error": reason})
                continue
            if state.cancel_requested:
                await self.runtime.executor.finalize_cancelled_run(
                    state.id,
                    reason="Recovered pending cancellation request.",
                )
                continue

            await self.runtime.store.requeue_inflight_work(state.id, include_failed=True)
            scheduled = await self.runtime.executor.launch_run(
                state.id,
                state.question,
                state.budget,
                raise_on_error=False,
            )
            if scheduled:
                await self.runtime.events.publish(
                    state.id,
                    "run.recovered",
                    {"prior_status": state.status.value, "reason": "stale_run_reconciled"},
                )


class ResearchRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        store: ResearchStore,
        events: RunEventService,
        stream_service: EventStreamService,
        orchestrator: ResearchOrchestrator,
        telemetry: ResearchTelemetry,
        memory_compiler: MemoryCompiler,
        context_assembler: ContextAssembler,
        behavior_judge: BehaviorJudge,
        conversation_client: OpenAIJsonClient | None = None,
        workflow_engine: WorkflowEngine | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.events = events
        self.stream_service = stream_service
        self.orchestrator = orchestrator
        self.telemetry = telemetry
        self.memory_compiler = memory_compiler
        self.context_assembler = context_assembler
        self.behavior_judge = behavior_judge
        self.conversation_client = conversation_client
        self.workflow_engine = workflow_engine
        self.worker_id = settings.worker_id or f"{socket.gethostname()}-{uuid4().hex[:8]}"
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._shutting_down = False
        self.coordinator = RunCoordinator(self)
        self.executor = WorkerExecutor(self)
        self.reconciler = RunReconciler(self)

    @property
    def workflow_backend_name(self) -> str:
        return self.workflow_engine.backend_name if self.workflow_engine else "local"

    @classmethod
    def build(cls, settings: Settings | None = None) -> ResearchRuntime:
        settings = settings or get_settings()
        engine, session_factory = create_engine_and_sessionmaker(settings.database_url)
        store = ResearchStore(engine, session_factory)
        broker = EventBroker()
        events = RunEventService(store, broker)
        stream_service = EventStreamService(
            events=events,
            keepalive_seconds=settings.sse_keepalive_seconds,
            mode=settings.event_pubsub_mode,
        )
        telemetry = ResearchTelemetry(settings)
        memory_compiler = MemoryCompiler(store=store, events=events)
        context_assembler = ContextAssembler(store=store, events=events)
        behavior_judge = BehaviorJudge(store=store, events=events)
        search_provider = _build_search_provider(settings)
        fetch_provider = _build_fetch_provider(settings)
        artifact_store = build_artifact_store(settings)
        embedding_provider = _build_embedding_provider(settings)
        reranker = _build_reranker(settings)
        conversation_client: OpenAIJsonClient | None = None

        if settings.resolved_llm_backend in {"openai", "openai_compatible"}:
            if settings.resolved_llm_backend == "openai":
                assert settings.openai_api_key is not None
                llm_api_key = settings.openai_api_key.get_secret_value()
                llm_base_url = settings.openai_base_url
            else:
                llm_api_key = _resolve_oss_api_key(settings.llm_api_key)
                llm_base_url = settings.llm_base_url
                if llm_base_url is None:
                    raise ValueError(
                        "OPEN_RESEARCH_LLM_BASE_URL must be configured for the "
                        "openai_compatible backend."
                    )

            openai_client = OpenAIJsonClient(
                api_key=llm_api_key,
                base_url=llm_base_url,
                timeout=settings.http_timeout_seconds * 2,
                api_style=settings.resolved_llm_api_style,
                structured_output_mode=settings.resolved_llm_structured_output_mode,
                supports_reasoning_effort=settings.resolved_llm_supports_reasoning_effort,
            )
            conversation_client = openai_client
            planner = OpenAIPlanner(
                openai_client,
                planner_model=settings.planner_model,
                worker_model=settings.worker_model,
                search_provider=search_provider,
                settings=settings,
            )
            note_writer = OpenAINoteWriter(
                openai_client,
                worker_model=settings.worker_model,
                settings=settings,
            )
            report_writer = OpenAIReportWriter(
                openai_client,
                lead_model=settings.lead_model,
                settings=settings,
            )
            verifier = OpenAIClaimVerifier(
                openai_client,
                verifier_model=settings.verifier_model,
                settings=settings,
            )
        else:
            planner = HeuristicPlanner(
                worker_model=settings.worker_model,
                max_streams=settings.max_streams,
            )
            note_writer = HeuristicNoteWriter()
            report_writer = HeuristicReportWriter()
            verifier = HeuristicClaimVerifier()

        gap_analyzer = HeuristicGapAnalyzer(worker_model=settings.worker_model)
        worker = ResearchWorker(
            store=store,
            events=events,
            search_provider=search_provider,
            fetch_provider=fetch_provider,
            note_writer=note_writer,
            artifact_store=artifact_store,
            embedding_provider=embedding_provider,
            settings=settings,
            telemetry=telemetry,
            context_assembler=context_assembler,
            memory_compiler=memory_compiler,
        )
        orchestrator = ResearchOrchestrator(
            store=store,
            events=events,
            planner=planner,
            gap_analyzer=gap_analyzer,
            report_writer=report_writer,
            verifier=verifier,
            worker=worker,
            embedding_provider=embedding_provider,
            reranker=reranker,
            telemetry=telemetry,
            context_assembler=context_assembler,
            memory_compiler=memory_compiler,
            behavior_judge=behavior_judge,
        )
        runtime = cls(
            settings=settings,
            store=store,
            events=events,
            stream_service=stream_service,
            orchestrator=orchestrator,
            telemetry=telemetry,
            memory_compiler=memory_compiler,
            context_assembler=context_assembler,
            behavior_judge=behavior_judge,
            conversation_client=conversation_client,
        )
        if settings.resolved_workflow_backend == "temporal":
            if settings.temporal_target_url is None:
                raise ValueError("TEMPORAL_TARGET_URL must be configured for Temporal mode.")
            try:
                from .temporal_backend import TemporalWorkflowBackend
            except ImportError as exc:
                raise ValueError(
                    "Temporal dependencies are not installed. Install the temporal extra to "
                    "enable this workflow backend."
                ) from exc

            workflow_settings = settings.model_copy(
                update={
                    "temporal_start_worker": (
                        settings.temporal_start_worker and settings.process_role != "api"
                    )
                }
            )
            runtime.workflow_engine = TemporalWorkflowBackend(
                settings=workflow_settings,
                execute_run=runtime._execute_run,
            )
        return runtime

    async def init(self) -> None:
        self._shutting_down = False
        self._background_tasks = set()
        await self.store.init_db(bootstrap_mode=self.settings.resolved_database_bootstrap_mode)
        if self.workflow_engine is not None:
            await self.workflow_engine.init()
        if (
            self.settings.automatic_run_recovery
            and self.settings.process_role != "api"
            and self.workflow_engine is None
        ):
            await self.executor.recover_incomplete_runs()
        if self.settings.process_role != "api":
            await self.reconciler.start()

    async def shutdown(self) -> None:
        self._shutting_down = True
        await self.reconciler.stop()
        if self.workflow_engine is None:
            for run_id in list(self._tasks):
                with suppress(Exception):
                    await self.events.publish(
                        run_id,
                        "run.shutdown",
                        {"reason": "runtime shutdown requested"},
                    )
                await self.store.requeue_inflight_work(run_id)
        else:
            await self.workflow_engine.shutdown()
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        await self.store.close()

    async def start_run(self, request: CreateRunRequest) -> RunSummary:
        return await self.coordinator.start_run(request)

    async def _execute_run(self, run_id: str, question: str, budget: BudgetPolicy) -> None:
        await self.executor.execute_run(run_id, question, budget)

    async def get_run_detail(self, run_id: str) -> RunDetail | None:
        return await self.store.get_run_detail(run_id)

    async def get_run_workspace(self, run_id: str) -> RunWorkspaceSnapshot | None:
        detail = await self.get_run_detail(run_id)
        if detail is None:
            return None
        return build_run_workspace_snapshot(
            detail,
            events=await self.list_events(run_id),
            tasks=await self.store.list_tasks(run_id),
            notes=[
                RunNoteRecord.model_validate(note)
                for note in await self.list_notes(run_id)
            ],
            passages=await self.list_passages(run_id),
        )

    async def list_run_conversation_messages(self, run_id: str) -> list[RunConversationMessage]:
        detail = await self.get_run_detail(run_id)
        if detail is None:
            raise KeyError(f"Run {run_id} not found")
        return detail.conversation_messages

    async def send_run_conversation_message(
        self,
        run_id: str,
        request: RunConversationRequest,
    ) -> RunConversationReply:
        detail = await self.get_run_detail(run_id)
        if detail is None:
            raise KeyError(f"Run {run_id} not found")
        if detail.final_report is None or not detail.final_report.markdown.strip():
            raise ValueError("Conversation is only available after the research run has a final report.")

        user_message = await self.store.save_conversation_message(
            run_id=run_id,
            role=RunConversationRole.USER,
            content=request.message.strip(),
        )
        assistant_text, references = await self._generate_run_conversation_reply(
            detail=detail,
            user_message=user_message,
        )
        model_config = resolve_model_config(
            detail.metadata.get("model_config"),
            defaults=self.default_model_config(),
        )
        assistant_message = await self.store.save_conversation_message(
            run_id=run_id,
            role=RunConversationRole.ASSISTANT,
            content=assistant_text,
            model=model_config.lead_model if self.conversation_client is not None else "heuristic",
            references=references,
        )
        await self.events.publish(
            run_id,
            "conversation.message.added",
            {"role": "user", "message_id": user_message.id},
        )
        await self.events.publish(
            run_id,
            "conversation.message.added",
            {"role": "assistant", "message_id": assistant_message.id, "references": references},
        )
        return RunConversationReply(
            user_message=user_message,
            assistant_message=assistant_message,
        )

    async def get_final_report(self, run_id: str):
        return await self.store.get_final_report(run_id)

    async def list_artifacts(self, run_id: str):
        return await self.store.list_artifacts(run_id)

    async def list_citation_audits(self, run_id: str):
        return await self.store.list_citation_audits(run_id)

    async def list_notes(self, run_id: str):
        return await self.store.list_notes(run_id)

    async def list_passages(self, run_id: str):
        return await self.store.list_passages(run_id)

    async def list_context_packs(self, run_id: str):
        return await self.store.list_context_packs(run_id)

    async def list_behavior_assessments(self, run_id: str):
        assessments = await self.store.list_behavior_assessments(run_id)
        if assessments:
            return assessments
        detail = await self.get_run_detail(run_id)
        if (
            detail is not None
            and is_terminal_run_status(detail.status)
            and self.settings.behavior_assessment_enabled
        ):
            return await self.behavior_judge.assess_run(
                run_id=run_id,
                profile_id=detail.profile_id,
            )
        return assessments

    async def _generate_run_conversation_reply(
        self,
        *,
        detail: RunDetail,
        user_message: RunConversationMessage,
    ) -> tuple[str, list[str]]:
        passages = await self.store.search_passages(detail.id, user_message.content, limit=6)
        notes = await self.store.list_notes(detail.id)
        recent_messages = detail.conversation_messages[-8:]
        plan_summary = detail.latest_plan.summary if detail.latest_plan is not None else "No saved plan."
        source_lines = [
            f"- {passage['source_title']}: {str(passage['text'])[:500]}"
            for passage in passages[:6]
        ]
        note_lines = [
            f"- {note['stream_name']}: {note['summary']}"
            for note in notes[:6]
        ]
        references = _conversation_references(detail, passages)

        if self.conversation_client is None:
            answer_parts = [
                "Here’s the best answer from the completed research run.",
                _clip_text(detail.final_report.markdown, limit=1400) if detail.final_report else "",
            ]
            if source_lines:
                answer_parts.append("Relevant retrieved passages:\n" + "\n".join(source_lines[:3]))
            return "\n\n".join(part for part in answer_parts if part), references

        agent_config = resolve_agent_config(detail.metadata.get("agent_config"))
        model_config = resolve_model_config(
            detail.metadata.get("model_config"),
            defaults=self.default_model_config(),
        )
        prompt = conversation_system_prompt(
            settings=self.settings,
            agent_config=agent_config,
        )
        conversation_history = "\n".join(
            f"{message.role.value}: {message.content}" for message in recent_messages
        ) or "No prior follow-up conversation."
        user_prompt = (
            f"Original research question:\n{detail.question}\n\n"
            f"Approved / latest plan summary:\n{plan_summary}\n\n"
            f"Final report:\n{_clip_text(detail.final_report.markdown, limit=8000) if detail.final_report else ''}\n\n"
            f"Recent follow-up conversation:\n{conversation_history}\n\n"
            f"Relevant note summaries:\n{chr(10).join(note_lines) or '- none'}\n\n"
            f"Retrieved passages for this follow-up:\n{chr(10).join(source_lines) or '- none'}\n\n"
            f"User follow-up:\n{user_message.content}"
        )
        response = await self.conversation_client.generate_text(
            model=model_config.lead_model,
            system_prompt=prompt.system_prompt,
            user_prompt=user_prompt,
            reasoning_effort="minimal",
            temperature=0.3,
        )
        return response.strip(), references

    async def get_profile_preferences(self, profile_id: str) -> ProfileRecord | None:
        return await self.store.get_profile(profile_id)

    async def update_profile_preferences(
        self,
        profile_id: str,
        preferences: ProfilePreferences,
    ) -> ProfileRecord:
        return await self.store.upsert_profile_preferences(profile_id, preferences)

    async def record_profile_feedback(self, feedback: ProfileFeedback) -> list[BehaviorAssessment]:
        await self.store.save_profile_feedback(feedback)
        profile = await self.store.get_profile(feedback.profile_id)
        existing_preferences = profile.preferences if profile is not None else ProfilePreferences()
        merged_preferences = merge_feedback_into_preferences(existing_preferences, feedback)
        await self.store.upsert_profile_preferences(feedback.profile_id, merged_preferences)
        if self.settings.profile_memory_enabled:
            await self.memory_compiler.compile_feedback(feedback)
        assessments = await self.behavior_judge.assess_feedback(feedback)
        if feedback.run_id is not None:
            await self.events.publish(
                feedback.run_id,
                "profile.feedback.recorded",
                {"profile_id": feedback.profile_id},
            )
        return assessments

    async def list_runs(
        self,
        *,
        limit: int = 50,
        status: RunStatus | None = None,
        project_id: str | None = None,
    ) -> list[RunSummary]:
        return await self.store.list_runs(limit=limit, status=status, project_id=project_id)

    async def create_project(self, request: CreateProjectRequest) -> ProjectSummary:
        return await self.store.create_project(request.name, description=request.description)

    async def list_projects(self) -> list[ProjectSummary]:
        return await self.store.list_projects()

    async def get_project_detail(self, project_id: str) -> ProjectDetail | None:
        return await self.store.get_project_detail(project_id)

    async def save_project_assets(
        self,
        project_id: str,
        assets: list[ResearchInputAsset],
    ) -> list[ResearchAssetRecord]:
        project = await self.store.get_project(project_id)
        if project is None:
            raise KeyError(f"Project {project_id} not found")
        records: list[ResearchAssetRecord] = []
        for asset in assets:
            records.append(await self.store.save_research_asset(asset, project_id=project_id))
        return records

    async def upload_project_files(
        self,
        *,
        project_id: str,
        usage: ResearchAssetUsage,
        files: list[tuple[str, str | None, bytes]],
        description: str | None = None,
    ) -> list[ResearchAssetRecord]:
        if len(files) > self.settings.max_upload_files_per_batch:
            raise ValueError(
                f"Upload batches are limited to {self.settings.max_upload_files_per_batch} files."
            )
        project = await self.store.get_project(project_id)
        if project is None:
            raise KeyError(f"Project {project_id} not found")
        records: list[ResearchAssetRecord] = []
        for file_name, content_type, data in files:
            asset, metadata = _build_uploaded_asset(
                usage=usage,
                file_name=file_name,
                content_type=content_type,
                data=data,
                settings=self.settings,
            )
            if description:
                asset = asset.model_copy(update={"description": description})
            records.append(
                await self.store.save_research_asset(asset, project_id=project_id, metadata=metadata)
            )
        return records

    async def stage_uploaded_files(
        self,
        *,
        usage: ResearchAssetUsage,
        files: list[tuple[str, str | None, bytes]],
        description: str | None = None,
    ) -> list[StagedAssetRecord]:
        if len(files) > self.settings.max_upload_files_per_batch:
            raise ValueError(
                f"Upload batches are limited to {self.settings.max_upload_files_per_batch} files."
            )
        records: list[StagedAssetRecord] = []
        for file_name, content_type, data in files:
            asset, metadata = _build_uploaded_asset(
                usage=usage,
                file_name=file_name,
                content_type=content_type,
                data=data,
                settings=self.settings,
            )
            if description:
                asset = asset.model_copy(update={"description": description})
            records.append(await self.store.create_staged_asset(asset, metadata=metadata))
        return records

    async def get_staged_asset(self, asset_id: str) -> StagedAssetRecord | None:
        return await self.store.get_staged_asset(asset_id)

    async def delete_staged_asset(self, asset_id: str) -> bool:
        return await self.store.delete_staged_asset(asset_id)

    async def delete_project_asset(self, project_id: str, asset_id: str) -> bool:
        return await self.store.delete_project_asset(project_id, asset_id)

    async def promote_run_asset(
        self,
        *,
        run_id: str,
        asset_id: str,
        project_id: str,
    ) -> ResearchAssetRecord:
        return await self.store.promote_run_asset_to_project(
            run_id=run_id,
            asset_id=asset_id,
            project_id=project_id,
        )

    async def resolve_effective_assets(
        self,
        *,
        run_id: str | None = None,
        project_id: str | None = None,
    ) -> list[ResearchAssetRecord]:
        project_assets = (
            await self.store.list_research_assets(project_id=project_id)
            if project_id is not None
            else []
        )
        run_assets = (
            await self.store.list_research_assets(run_id=run_id)
            if run_id is not None
            else []
        )
        return [*project_assets, *run_assets]

    async def submit_job(
        self,
        request: CreateRunRequest,
        *,
        owner_id: str | None = None,
    ) -> AsyncJob:
        return await self.create_job(request, owner_id=owner_id)

    async def list_events(self, run_id: str, *, after_id: int = 0):
        return await self.events.replay(run_id, after_id=after_id)

    async def cancel_run(self, run_id: str) -> RunDetail:
        return await self.coordinator.cancel_run(run_id)

    async def resume_run(self, run_id: str) -> RunSummary:
        return await self.coordinator.resume_run(run_id)

    async def retry_run(self, run_id: str) -> RunSummary:
        return await self.coordinator.retry_run(run_id)

    async def get_clarification(self, run_id: str) -> ClarificationSession | None:
        detail = await self.get_run_detail(run_id)
        return None if detail is None else detail.clarification_session

    async def answer_run_clarification(self, run_id: str, response: str) -> RunDetail:
        return await self.answer_clarification(run_id, response)

    async def get_plan_preview(self, run_id: str) -> PlanPreview | None:
        detail = await self.get_run_detail(run_id)
        return None if detail is None else detail.plan_preview

    async def approve_run_plan(
        self,
        run_id: str,
        *,
        decision: ApprovalDecisionKind,
        note: str | None = None,
        actor: str | None = None,
    ) -> RunDetail:
        return await self.approve_plan_preview(run_id, decision=decision, note=note, actor=actor)

    async def list_source_registry_entries(self, run_id: str):
        return await self.store.list_source_registry_entries(run_id)

    def default_budget(self) -> BudgetPolicy:
        return BudgetPolicy(
            max_streams=self.settings.max_streams,
            max_replans=self.settings.max_replans,
            max_queries_per_stream=self.settings.max_queries_per_stream,
            max_results_per_query=self.settings.max_results_per_query,
            max_sources_per_stream=self.settings.max_sources_per_stream,
            per_domain_limit=self.settings.per_domain_limit,
        )

    def default_agent_config(self) -> AgentConfig:
        return AgentConfig()

    def default_model_config(self) -> ModelConfig:
        return ModelConfig(
            lead_model=self.settings.lead_model,
            planner_model=self.settings.planner_model,
            worker_model=self.settings.worker_model,
            verifier_model=self.settings.verifier_model,
            embedding_model=self.settings.embedding_model,
            reranker_model=self.settings.reranker_model,
        )

    def available_sources(self) -> list[SourceCatalogEntry]:
        return _build_source_catalog(self.settings)

    def default_source_selection(self) -> list[str]:
        return [entry.id for entry in self.available_sources() if entry.default_enabled]

    def resolve_source_selection(self, source_selection: list[str] | None) -> list[str]:
        available = {entry.id: entry for entry in self.available_sources()}
        selected = source_selection or self.default_source_selection()
        unknown = [entry_id for entry_id in selected if entry_id not in available]
        if unknown:
            raise ValueError(f"Unknown source selection entries: {', '.join(sorted(unknown))}")
        unavailable = [
            entry_id
            for entry_id in selected
            if not available[entry_id].configured and entry_id != "mock"
        ]
        if unavailable:
            raise ValueError(
                "Selected sources are not configured for this deployment: "
                + ", ".join(sorted(unavailable))
            )
        return selected

    def should_require_plan_approval(
        self,
        *,
        execution_mode: ExecutionMode,
        requested_budget: BudgetPolicy,
        explicit: bool | None,
        source_selection: list[str],
        model_config: ModelConfig,
        user_supplied_budget: bool,
        user_supplied_sources: bool,
        user_supplied_model_override: bool,
    ) -> bool:
        if explicit is not None:
            return explicit
        if not self.settings.deep_plan_approval_enabled:
            return False
        if execution_mode == ExecutionMode.HITL:
            return True
        if execution_mode == ExecutionMode.DEEP:
            return True
        if execution_mode != ExecutionMode.STANDARD:
            return False
        source_selection_is_custom = (
            user_supplied_sources
            and sorted(source_selection) != sorted(self.default_source_selection())
        )
        if not any((user_supplied_budget, source_selection_is_custom, user_supplied_model_override)):
            return False
        return (
            requested_budget.max_streams >= 8
            or requested_budget.max_queries_per_stream >= 8
            or (source_selection_is_custom and len(source_selection) >= 4)
            or model_config.lead_model != self.settings.lead_model
            or model_config.planner_model != self.settings.planner_model
            or model_config.worker_model != self.settings.worker_model
            or model_config.verifier_model != self.settings.verifier_model
        )

    def _clarifier_config_from_metadata(self, metadata: dict[str, object]) -> ClarifierConfig:
        raw = metadata.get("clarifier_config")
        if raw is None:
            return ClarifierConfig()
        return ClarifierConfig.model_validate(raw)

    async def prepare_run_for_approval(
        self,
        *,
        run_id: str,
        question: str,
        requested_budget: BudgetPolicy,
    ) -> RunDetail | None:
        detail = await self.get_run_detail(run_id)
        if detail is None:
            raise KeyError(f"Run {run_id} not found")
        metadata = dict(detail.metadata)
        clarifier_config = self._clarifier_config_from_metadata(metadata)
        ambiguity_score = _question_ambiguity_score(question)
        needs_question = clarifier_config.enabled and (
            clarifier_config.require_response_for_deep
            or ambiguity_score >= clarifier_config.ambiguity_threshold
        )
        if needs_question:
            session = ClarificationSession(
                status=PlanApprovalStatus.PENDING_CLARIFICATION,
                rationale=(
                    "The run is gated for deep research and needs clarification before "
                    "committing the plan."
                ),
                questions=[
                    ClarificationQuestion(
                        id=str(uuid4()),
                        prompt=(
                            "What outcome, must-cover angles, or hard constraints should the "
                            "deep plan optimize for?"
                        ),
                        rationale="Clarification changes plan shape, stream count, and search depth.",
                        required=True,
                    )
                ],
                iteration_count=1,
            )
            metadata["clarification_session"] = session.model_dump(mode="json")
            metadata["approval_status"] = PlanApprovalStatus.PENDING_CLARIFICATION.value
            await self.store.update_run_metadata(
                run_id,
                {
                    "clarification_session": metadata["clarification_session"],
                    "approval_status": metadata["approval_status"],
                },
            )
            await self.store.update_run_status(run_id, RunStatus.CLARIFYING)
            await self.events.publish(
                run_id,
                "clarification.required",
                {
                    "question_count": len(session.questions),
                    "rationale": session.rationale,
                },
            )
            return await self.get_run_detail(run_id)
        await self._generate_plan_preview(run_id=run_id, question=question, requested_budget=requested_budget)
        return await self.get_run_detail(run_id)

    async def _generate_plan_preview(
        self,
        *,
        run_id: str,
        question: str,
        requested_budget: BudgetPolicy,
    ) -> PlanPreview:
        detail = await self.get_run_detail(run_id)
        if detail is None:
            raise KeyError(f"Run {run_id} not found")
        metadata = dict(detail.metadata)
        clarified_question = _compose_clarified_question(question, detail.clarification_session)
        agent_config = resolve_agent_config(metadata.get("agent_config"))
        model_config = resolve_model_config(
            metadata.get("model_config"),
            defaults=self.default_model_config(),
        )
        effective_assets = await self.resolve_effective_assets(
            run_id=run_id,
            project_id=detail.project_id,
        )
        ready_assets = [
            asset for asset in effective_assets if asset.processing_status == AssetProcessingStatus.READY
        ]
        plan_result = await self.orchestrator.planner.create_plan(
            clarified_question,
            requested_budget,
            planning_stage=PlanningStage.PREVIEW,
            agent_config=agent_config,
            model_config=model_config,
            available_documents=ready_assets,
            source_selection=list(metadata.get("source_selection") or []),
            min_total_sources_retrieved=self.settings.planner_min_total_sources_retrieved,
            min_total_cited_sources=self.settings.planner_min_total_cited_sources,
        )
        plan = plan_result.value
        recommended_budget = plan.recommended_budget or RecommendedBudget(
            **requested_budget.model_dump(mode="json"),
            rationale_summary="Fallback recommendation equals the requested run budget.",
        )
        effective_budget, decision_reason = _clamp_budget(
            requested_budget,
            recommended_budget=recommended_budget,
        )
        preview = PlanPreview(
            version=(detail.plan_preview.version + 1) if detail.plan_preview is not None else 1,
            summary=plan.summary,
            hypothesis=plan.hypothesis,
            plan=plan.model_copy(
                update={
                    "recommended_budget": recommended_budget,
                    "approval_required": True,
                }
            ),
            requested_budget=requested_budget,
            recommended_budget=recommended_budget,
            effective_budget=effective_budget,
            budget_decision_reason=decision_reason,
            approval_required=True,
            recommended_execution_mode=plan.recommended_execution_mode or ExecutionMode.DEEP,
            source_selection=list(metadata.get("source_selection") or []),
            clarification_summary=(
                detail.clarification_session.turns[-1].response
                if detail.clarification_session and detail.clarification_session.turns
                else None
            ),
        )
        await self.store.update_run_budget(run_id, effective_budget)
        await self.store.update_run_metadata(
            run_id,
            {
                "requested_budget": requested_budget.model_dump(mode="json"),
                "recommended_budget": recommended_budget.model_dump(mode="json"),
                "effective_budget": effective_budget.model_dump(mode="json"),
                "budget_decision_reason": decision_reason,
                "plan_preview": preview.model_dump(mode="json"),
                "approval_status": PlanApprovalStatus.PENDING_APPROVAL.value,
            },
        )
        await self.store.update_run_status(run_id, RunStatus.AWAITING_PLAN_APPROVAL)
        await self.events.publish(
            run_id,
            "plan.preview.created",
            {
                "version": preview.version,
                "stream_count": len(preview.plan.streams),
                "recommended_budget": recommended_budget.model_dump(mode="json"),
                "effective_budget": effective_budget.model_dump(mode="json"),
            },
        )
        if plan_result.metadata is not None:
            await self.events.publish(
                run_id,
                "planning.preview.generated",
                {
                    "planning_stage": PlanningStage.PREVIEW.value,
                    "prompt_template_version": plan_result.metadata.get("prompt_template_version"),
                    "discovery_query_count": plan_result.metadata.get("discovery_query_count", 0),
                },
            )
        return preview

    async def answer_clarification(self, run_id: str, response: str) -> RunDetail:
        detail = await self.get_run_detail(run_id)
        if detail is None:
            raise KeyError(f"Run {run_id} not found")
        if detail.status != RunStatus.CLARIFYING or detail.clarification_session is None:
            raise ValueError("Run is not waiting for clarification.")
        unanswered = [
            question
            for question in detail.clarification_session.questions
            if question.id not in {turn.question_id for turn in detail.clarification_session.turns}
        ]
        if not unanswered:
            raise ValueError("There are no outstanding clarification questions.")
        session = detail.clarification_session.model_copy(
            update={
                "turns": [
                    *detail.clarification_session.turns,
                    ClarificationTurn(
                        question_id=unanswered[0].id,
                        prompt=unanswered[0].prompt,
                        response=response,
                    ),
                ],
                "updated_at": datetime.now(UTC),
            }
        )
        await self.store.update_run_metadata(
            run_id,
            {"clarification_session": session.model_dump(mode="json")},
        )
        await self.events.publish(
            run_id,
            "clarification.answered",
            {"question_id": unanswered[0].id},
        )
        await self._generate_plan_preview(
            run_id=run_id,
            question=str(detail.metadata.get("effective_question", detail.question)),
            requested_budget=detail.requested_budget or detail.budget or self.default_budget(),
        )
        updated = await self.get_run_detail(run_id)
        if updated is None:
            raise KeyError(f"Run {run_id} not found")
        return updated

    async def approve_plan_preview(
        self,
        run_id: str,
        *,
        decision: ApprovalDecisionKind,
        note: str | None = None,
        actor: str | None = None,
    ) -> RunDetail:
        detail = await self.get_run_detail(run_id)
        if detail is None:
            raise KeyError(f"Run {run_id} not found")
        if detail.plan_preview is None:
            raise ValueError("Run does not have a pending plan preview.")
        if detail.status != RunStatus.AWAITING_PLAN_APPROVAL:
            raise ValueError("Run is not waiting for plan approval.")
        decision_record = ApprovalDecision(decision=decision, note=note, actor=actor)
        metadata_update = {"latest_approval_decision": decision_record.model_dump(mode="json")}
        if decision == ApprovalDecisionKind.APPROVE:
            metadata_update["approval_status"] = PlanApprovalStatus.APPROVED.value
            await self.store.update_run_metadata(run_id, metadata_update)
            await self.store.update_run_status(run_id, RunStatus.QUEUED)
            await self.events.publish(
                run_id,
                "plan.preview.approved",
                {"version": detail.plan_preview.version, "note": note},
            )
            await self.executor.launch_run(
                run_id,
                str(detail.metadata.get("effective_question", detail.question)),
                detail.effective_budget or detail.plan_preview.effective_budget,
            )
        elif decision == ApprovalDecisionKind.REJECT:
            metadata_update["approval_status"] = PlanApprovalStatus.REJECTED.value
            await self.store.update_run_metadata(run_id, metadata_update)
            await self.store.update_run_status(
                run_id,
                RunStatus.CANCELLED,
                terminal_reason=note or "Plan preview rejected.",
            )
            await self.events.publish(run_id, "plan.preview.rejected", {"note": note})
        else:
            next_question = ClarificationQuestion(
                id=str(uuid4()),
                prompt="What should change in the plan before execution starts?",
                rationale="Requested changes should explicitly alter the preview plan.",
                required=True,
            )
            existing_session = detail.clarification_session or ClarificationSession()
            session = existing_session.model_copy(
                update={
                    "status": PlanApprovalStatus.CHANGES_REQUESTED,
                    "questions": [*existing_session.questions, next_question],
                    "updated_at": datetime.now(UTC),
                    "iteration_count": existing_session.iteration_count + 1,
                }
            )
            metadata_update["approval_status"] = PlanApprovalStatus.CHANGES_REQUESTED.value
            metadata_update["clarification_session"] = session.model_dump(mode="json")
            await self.store.update_run_metadata(run_id, metadata_update)
            await self.store.update_run_status(run_id, RunStatus.CLARIFYING)
            await self.events.publish(run_id, "plan.preview.changes_requested", {"note": note})
        updated = await self.get_run_detail(run_id)
        if updated is None:
            raise KeyError(f"Run {run_id} not found")
        return updated

    async def create_job(
        self,
        request: CreateRunRequest,
        *,
        owner_id: str | None = None,
    ) -> AsyncJob:
        run = await self.start_run(request)
        job = await self.store.create_job(run_id=run.id, owner_id=owner_id)
        await self.store.update_run_metadata(run.id, {"job_id": job.job_id})
        await self.events.publish(run.id, "job.created", {"job_id": job.job_id})
        return job

    async def get_job(self, job_id: str) -> AsyncJob | None:
        job = await self.store.get_job(job_id)
        if job is None:
            return None
        detail = await self.get_run_detail(job.run_id)
        if detail is None:
            return job
        await self.store.update_job(
            job_id,
            status=detail.status.value,
            last_heartbeat_at=detail.last_heartbeat_at,
            started_at=detail.created_at if detail.status != RunStatus.QUEUED else None,
            ended_at=detail.updated_at if is_terminal_run_status(detail.status) else None,
        )
        refreshed = await self.store.get_job(job_id)
        return refreshed

    async def get_job_workspace(self, job_id: str) -> RunWorkspaceSnapshot | None:
        job = await self.get_job(job_id)
        if job is None:
            return None
        return await self.get_run_workspace(job.run_id)

    def public_config(self) -> PublicRuntimeConfig:
        available_sources = self.available_sources()
        return PublicRuntimeConfig(
            app_name=self.settings.app_name,
            environment=self.settings.environment,
            prompt_profile_version=PROMPT_PROFILE_VERSION,
            source_trust_policy_version=SOURCE_TRUST_POLICY_VERSION,
            default_budget=self.default_budget(),
            default_agent_config=self.default_agent_config(),
            backends={
                "llm": self.settings.resolved_llm_backend,
                "search": self.settings.resolved_search_backend,
                "fetch": self.settings.resolved_fetch_backend,
                "workflow": self.workflow_backend_name,
                "embedding": self.settings.resolved_embedding_backend,
                "reranker": self.settings.resolved_reranker_backend,
            },
            models=self.default_model_config(),
            prompt_mode=self.settings.prompt_mode,
            available_sources=available_sources,
            default_source_selection=[entry.id for entry in available_sources if entry.default_enabled],
            capabilities={
                "supports_temporal": self.settings.resolved_workflow_backend == "temporal",
                "supports_artifacts": self.settings.resolved_artifact_store_backend != "disabled",
                "supports_embeddings": self.settings.resolved_embedding_backend != "disabled",
                "supports_metrics": self.settings.metrics_enabled,
                "supports_prompt_profiles": True,
                "supports_replayable_sse": True,
                "supports_profiles": True,
                "supports_memory_harness": self.settings.profile_memory_enabled,
                "supports_behavior_assessment": self.settings.behavior_assessment_enabled,
                "supports_deep_approval": self.settings.deep_plan_approval_enabled,
                "supports_async_jobs": self.settings.async_jobs_enabled,
                "supports_source_registry": self.settings.source_registry_ui_enabled,
                "supports_debug_console": self.settings.debug_console_enabled,
                "supports_projects": True,
                "asset_upload_limits": {
                    "max_file_size_bytes": self.settings.max_upload_file_size_bytes,
                    "max_files_per_batch": self.settings.max_upload_files_per_batch,
                    "max_ocr_pdf_pages": self.settings.max_ocr_pdf_pages,
                },
                "budget_limits": _budget_limits(),
                "memory_policy_limits": _memory_policy_limits(),
            },
        )

    async def describe_workflow_run(self, run_id: str) -> dict[str, str] | None:
        if self.workflow_engine is None or not hasattr(self.workflow_engine, "describe_run"):
            return None
        try:
            return await self.workflow_engine.describe_run(run_id)
        except Exception:
            return None

    async def readiness(self) -> dict[str, bool | str]:
        database_ready = workflow_ready = events_ready = True
        database_error = workflow_error = ""
        try:
            await self.store.ping()
        except Exception as exc:
            database_ready = False
            database_error = str(exc)
        if self.workflow_engine is not None and hasattr(self.workflow_engine, "is_healthy"):
            try:
                workflow_ready = await self.workflow_engine.is_healthy()
            except Exception as exc:
                workflow_ready = False
                workflow_error = str(exc)
        events_ready = await self.stream_service.is_ready()
        return {
            "ready": database_ready and workflow_ready and events_ready,
            "database_ready": database_ready,
            "database_error": database_error,
            "workflow_ready": workflow_ready,
            "workflow_error": workflow_error,
            "events_ready": events_ready,
        }
