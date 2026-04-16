from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RunStatus(StrEnum):
    QUEUED = "queued"
    CLARIFYING = "clarifying"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    PLANNING = "planning"
    RESEARCHING = "researching"
    GROUNDING = "grounding"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def is_terminal_run_status(status: RunStatus) -> bool:
    return status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}


class StreamStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceKind(StrEnum):
    WEB = "web"
    BLOG = "blog"
    DOCS = "docs"
    PRICING = "pricing"
    PDF = "pdf"
    SYNTHETIC = "synthetic"


class RetrievalMethod(StrEnum):
    FIRECRAWL = "firecrawl"
    MOCK = "mock"
    API_NATIVE = "api_native"


class CitationSupportLabel(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class CitationMatchStrategy(StrEnum):
    EXACT = "exact"
    TRUNCATION = "truncation"
    PREFIX = "prefix"
    CHILD_PATH = "child_path"
    QUERY_SUBSET = "query_subset"


class CitationAuditDecision(StrEnum):
    KEPT = "kept"
    REMOVED = "removed"


class CitationAuditReason(StrEnum):
    URL_NOT_IN_REGISTRY = "url_not_in_registry"
    CITATION_KEY_NOT_IN_REGISTRY = "citation_key_not_in_registry"
    UNSAFE_URL = "unsafe_url"
    TRUNCATED_URL = "truncated_url"
    SHORTENED_URL = "shortened_url"
    IP_ADDRESS_URL = "ip_address_url"
    UNVERIFIABLE = "unverifiable"


class SourceTrustTier(StrEnum):
    PRIMARY = "primary"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"
    UNKNOWN = "unknown"


class ResearchProfile(StrEnum):
    BALANCED = "balanced"
    OFFICIAL_FIRST = "official_first"
    WIDE_NET = "wide_net"


class RecencyPolicy(StrEnum):
    AUTO = "auto"
    RECENT_FIRST = "recent_first"
    EVERGREEN = "evergreen"


class AnswerStyle(StrEnum):
    ANALYST = "analyst"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"


class CitationDiscipline(StrEnum):
    STRICT = "strict"
    BALANCED = "balanced"


class ClaimGranularity(StrEnum):
    ATOMIC = "atomic"
    BALANCED = "balanced"


class ExecutionMode(StrEnum):
    STANDARD = "standard"
    DEEP = "deep"
    HITL = "hitl"


class PlanApprovalStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING_CLARIFICATION = "pending_clarification"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class ApprovalDecisionKind(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class PromptMode(StrEnum):
    CODE = "code"
    TEMPLATE = "template"
    HYBRID = "hybrid"


class PlanningStage(StrEnum):
    PREVIEW = "preview"
    EXECUTION = "execution"
    REPLAN = "replan"


class ResearchAssetUsage(StrEnum):
    REFERENCE_SOURCE = "reference_source"
    PLANNING_CONTEXT = "planning_context"


class ResearchAssetType(StrEnum):
    URL = "url"
    FILE = "file"


class AssetProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class AssetExtractionMethod(StrEnum):
    URL = "url"
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF_TEXT = "pdf_text"
    PDF_OCR = "pdf_ocr"
    DOCX = "docx"
    IMAGE_OCR = "image_ocr"
    UNKNOWN = "unknown"


class BudgetPolicy(StrictModel):
    max_streams: int = Field(default=30, ge=1, le=30)
    max_replans: int = Field(default=1, ge=0, le=5)
    max_queries_per_stream: int = Field(default=25, ge=1, le=25)
    max_results_per_query: int = Field(default=5, ge=1, le=20)
    max_sources_per_stream: int = Field(default=3, ge=1, le=20)
    per_domain_limit: int = Field(default=2, ge=1, le=10)


class ModelConfig(StrictModel):
    lead_model: str
    planner_model: str
    worker_model: str
    verifier_model: str
    embedding_model: str
    reranker_model: str


class ModelConfigOverride(StrictModel):
    lead_model: str | None = None
    planner_model: str | None = None
    worker_model: str | None = None
    verifier_model: str | None = None
    embedding_model: str | None = None
    reranker_model: str | None = None


def resolve_model_config(
    model_config: ModelConfig | ModelConfigOverride | dict[str, Any] | None,
    *,
    defaults: ModelConfig,
) -> ModelConfig:
    if model_config is None:
        return defaults
    if isinstance(model_config, ModelConfig):
        return model_config
    if isinstance(model_config, ModelConfigOverride):
        override = model_config
    else:
        try:
            return ModelConfig.model_validate(model_config)
        except Exception:
            override = ModelConfigOverride.model_validate(model_config)
    return ModelConfig(
        lead_model=override.lead_model or defaults.lead_model,
        planner_model=override.planner_model or defaults.planner_model,
        worker_model=override.worker_model or defaults.worker_model,
        verifier_model=override.verifier_model or defaults.verifier_model,
        embedding_model=override.embedding_model or defaults.embedding_model,
        reranker_model=override.reranker_model or defaults.reranker_model,
    )


class AgentConfig(StrictModel):
    research_profile: ResearchProfile = ResearchProfile.BALANCED
    recency_policy: RecencyPolicy = RecencyPolicy.AUTO
    answer_style: AnswerStyle = AnswerStyle.ANALYST
    citation_discipline: CitationDiscipline = CitationDiscipline.STRICT
    claim_granularity: ClaimGranularity = ClaimGranularity.ATOMIC
    source_trust_floor: SourceTrustTier = SourceTrustTier.STANDARD
    include_counterevidence: bool = True


class ClarifierConfig(StrictModel):
    enabled: bool = True
    max_questions: int = Field(default=2, ge=0, le=5)
    ambiguity_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    require_response_for_deep: bool = False


class ClarificationQuestion(StrictModel):
    id: str
    prompt: str
    rationale: str
    required: bool = True
    created_at: datetime = Field(default_factory=utc_now)


class ClarificationTurn(StrictModel):
    question_id: str
    prompt: str
    response: str
    created_at: datetime = Field(default_factory=utc_now)


class ClarificationResult(StrictModel):
    needs_clarification: bool
    rationale: str
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    resolved_question: str | None = None


class ClarificationSession(StrictModel):
    status: PlanApprovalStatus = PlanApprovalStatus.PENDING_CLARIFICATION
    rationale: str = ""
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    turns: list[ClarificationTurn] = Field(default_factory=list)
    iteration_count: int = 0
    resolved_question: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class BudgetRecommendationRationale(StrictModel):
    summary: str
    coverage_axes: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    source_diversity_reasoning: str | None = None
    grounding_difficulty: str | None = None


class RecommendedBudget(BudgetPolicy):
    rationale_summary: str | None = None


class ExecutionBudgetDecision(StrictModel):
    requested_budget: BudgetPolicy
    recommended_budget: RecommendedBudget | None = None
    effective_budget: BudgetPolicy
    decision_reason: str


class ApprovalDecision(StrictModel):
    decision: ApprovalDecisionKind
    note: str | None = None
    actor: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ClarificationResponseRequest(StrictModel):
    response: str = Field(min_length=1, max_length=4000)


class PlanApprovalRequest(StrictModel):
    note: str | None = Field(default=None, max_length=4000)
    actor: str | None = Field(default=None, max_length=256)


class SourceCatalogEntry(StrictModel):
    id: str
    name: str
    description: str
    backend_kind: str
    default_enabled: bool = True
    configured: bool = False
    auth_required: bool = False
    supports_search: bool = False
    supports_fetch: bool = False
    supports_internal_docs: bool = False
    supports_recency: bool = True
    supports_primary_sources: bool = False
    supports_advanced_search: bool = False
    status_reason: str | None = None


class AsyncJob(StrictModel):
    job_id: str
    run_id: str
    status: RunStatus | PlanApprovalStatus | str
    submission_mode: str = "async"
    submitted_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    owner_id: str | None = None
    resume_cursor: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryScope(StrEnum):
    PROFILE = "profile"
    GLOBAL = "global"


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    FAILURE = "failure"
    PREFERENCE = "preference"
    SEMANTIC = "semantic"


class ContextPhase(StrEnum):
    PLAN = "plan"
    REPLAN = "replan"
    RESEARCH = "research"
    SYNTHESIZE = "synthesize"
    GROUND = "ground"


class ContextFragmentKind(StrEnum):
    OPERATIONAL = "operational"
    PREFERENCE = "preference"
    EVIDENCE_ADJACENT = "evidence_adjacent"
    SYSTEM = "system"


class AssessmentKind(StrEnum):
    SOURCE_FIT = "source_fit"
    STYLE_FIT = "style_fit"
    OPERATIONAL = "operational"
    MEMORY_USEFULNESS = "memory_usefulness"


class AssessmentSource(StrEnum):
    SYSTEM = "system"
    USER = "user"


class MemoryInfluencePolicy(StrictModel):
    enabled: bool = True
    retrieval_limit: int = Field(default=10, ge=1, le=50)
    planning_budget_tokens: int = Field(default=1200, ge=0, le=6000)
    research_budget_tokens: int = Field(default=450, ge=0, le=4000)
    synthesis_budget_tokens: int = Field(default=900, ge=0, le=6000)
    grounding_budget_tokens: int = Field(default=0, ge=0, le=1000)
    allow_preference_in_planning: bool = True
    allow_preference_in_research: bool = True
    allow_preference_in_synthesis: bool = True
    allow_preference_in_grounding: bool = False
    stale_penalty: float = Field(default=0.2, ge=0.0, le=2.0)
    conflict_penalty: float = Field(default=0.4, ge=0.0, le=2.0)

    def budget_for_phase(self, phase: ContextPhase) -> int:
        if phase in {ContextPhase.PLAN, ContextPhase.REPLAN}:
            return self.planning_budget_tokens
        if phase == ContextPhase.RESEARCH:
            return self.research_budget_tokens
        if phase == ContextPhase.SYNTHESIZE:
            return self.synthesis_budget_tokens
        return self.grounding_budget_tokens

    def allow_preference_for_phase(self, phase: ContextPhase) -> bool:
        if phase in {ContextPhase.PLAN, ContextPhase.REPLAN}:
            return self.allow_preference_in_planning
        if phase == ContextPhase.RESEARCH:
            return self.allow_preference_in_research
        if phase == ContextPhase.SYNTHESIZE:
            return self.allow_preference_in_synthesis
        return self.allow_preference_in_grounding


class ProfilePreferences(StrictModel):
    preferred_source_patterns: list[str] = Field(default_factory=list)
    avoided_source_patterns: list[str] = Field(default_factory=list)
    answer_style_bias: AnswerStyle | None = None
    recency_bias: RecencyPolicy | None = None
    source_trust_floor_bias: SourceTrustTier | None = None
    include_counterevidence_bias: bool | None = None
    memory_policy: MemoryInfluencePolicy = Field(default_factory=MemoryInfluencePolicy)


class ProfileRecord(StrictModel):
    profile_id: str
    preferences: ProfilePreferences = Field(default_factory=ProfilePreferences)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ProfileFeedback(StrictModel):
    profile_id: str = "default"
    run_id: str | None = None
    source_fit: float | None = Field(default=None, ge=0.0, le=1.0)
    style_fit: float | None = Field(default=None, ge=0.0, le=1.0)
    usefulness: float | None = Field(default=None, ge=0.0, le=1.0)
    correction: str | None = None
    preferred_source_patterns: list[str] = Field(default_factory=list)
    avoided_source_patterns: list[str] = Field(default_factory=list)
    answer_style_bias: AnswerStyle | None = None
    include_counterevidence_bias: bool | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ResearchInputAsset(StrictModel):
    source_type: ResearchAssetType
    usage: ResearchAssetUsage
    label: str = Field(min_length=1, max_length=255)
    description: str | None = None
    url: HttpUrl | None = None
    content_text: str | None = None
    content_type: str | None = None
    file_name: str | None = None


class ProjectSummary(StrictModel):
    id: str
    name: str
    description: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CreateProjectRequest(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2_000)


class ResearchAssetBatchRequest(StrictModel):
    assets: list[ResearchInputAsset] = Field(default_factory=list)


class PromoteAssetRequest(StrictModel):
    project_id: str | None = Field(default=None, min_length=1, max_length=36)


class CreateRunRequest(StrictModel):
    question: str = Field(min_length=12, max_length=10_000)
    budget: BudgetPolicy | None = None
    agent_config: AgentConfig | None = None
    model_config_override: ModelConfigOverride | None = None
    profile_id: str = Field(default="default", min_length=1, max_length=128)
    project_id: str | None = Field(default=None, min_length=1, max_length=36)
    memory_policy_override: MemoryInfluencePolicy | None = None
    execution_mode: ExecutionMode = ExecutionMode.STANDARD
    require_plan_approval: bool | None = None
    clarifier_config: ClarifierConfig | None = None
    source_selection: list[str] | None = None
    input_assets: list[ResearchInputAsset] = Field(default_factory=list)
    staged_asset_ids: list[str] = Field(default_factory=list)
    async_submit: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(StrictModel):
    title: str
    url: HttpUrl
    snippet: str = ""
    provider: str
    score: float = 0.0


class ResearchStreamPlan(StrictModel):
    name: str
    objective: str
    queries: list[str] = Field(default_factory=list)
    model: str


class PlanningDiscoveryRecord(StrictModel):
    query: str
    provider: str | None = None
    result_count: int = 0
    titles: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    summary: str | None = None


class PlanningArtifact(StrictModel):
    stage: PlanningStage = PlanningStage.EXECUTION
    approved_preview_version: int | None = None
    task_breakdown: str | None = None
    table_of_contents: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    planned_deliverables: list[str] = Field(default_factory=list)
    key_questions: list[str] = Field(default_factory=list)
    available_documents: list[str] = Field(default_factory=list)
    discovery_queries: list[str] = Field(default_factory=list)
    discovery_records: list[PlanningDiscoveryRecord] = Field(default_factory=list)
    source_selection: list[str] = Field(default_factory=list)
    min_total_sources_retrieved: int = 0
    min_total_cited_sources: int = 0
    validation_checks: list[str] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)


class ResearchPlan(StrictModel):
    summary: str
    hypothesis: str
    streams: list[ResearchStreamPlan]
    success_criteria: list[str] = Field(default_factory=list)
    planning_artifact: PlanningArtifact | None = None
    recommended_budget: RecommendedBudget | None = None
    budget_rationale: BudgetRecommendationRationale | None = None
    recommended_execution_mode: ExecutionMode | None = None
    approval_required: bool = False
    complexity_factors: list[str] = Field(default_factory=list)


class PlanPreview(StrictModel):
    version: int = 1
    summary: str
    hypothesis: str
    plan: ResearchPlan
    requested_budget: BudgetPolicy
    recommended_budget: RecommendedBudget | None = None
    effective_budget: BudgetPolicy
    budget_decision_reason: str
    approval_required: bool
    recommended_execution_mode: ExecutionMode
    source_selection: list[str] = Field(default_factory=list)
    clarification_summary: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class FetchedDocument(StrictModel):
    url: HttpUrl
    title: str
    content: str
    canonical_url: HttpUrl
    source_kind: SourceKind
    retrieval_method: RetrievalMethod
    metadata: dict[str, Any] = Field(default_factory=dict)


class PassageRecord(StrictModel):
    source_id: str
    passage_index: int
    text: str
    start_offset: int = 0
    end_offset: int = 0
    token_count: int = 0


class NoteDraft(StrictModel):
    summary: str
    key_facts: list[str]
    open_questions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RunNoteRecord(StrictModel):
    id: str
    stream_id: str
    stream_name: str
    stream_objective: str
    source_id: str | None = None
    source_title: str | None = None
    source_url: HttpUrl | None = None
    source_kind: SourceKind | None = None
    retrieval_method: RetrievalMethod | None = None
    trust_tier: SourceTrustTier | None = None
    trust_rationale: str | None = None
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PassageInspectionRecord(StrictModel):
    source_id: str
    source_title: str
    source_url: HttpUrl
    passage_index: int
    text: str
    search_document: str | None = None
    start_offset: int = 0
    end_offset: int = 0
    token_count: int = 0
    source_kind: SourceKind | None = None
    retrieval_method: RetrievalMethod | None = None
    trust_tier: SourceTrustTier | None = None
    trust_rationale: str | None = None


class GapAnalysis(StrictModel):
    should_replan: bool
    rationale: str
    additional_streams: list[ResearchStreamPlan] = Field(default_factory=list)


class ReportSection(StrictModel):
    title: str
    overview: str
    claims: list[str]


class DraftReport(StrictModel):
    executive_summary: str
    sections: list[ReportSection]
    open_questions: list[str] = Field(default_factory=list)


class RetrievedPassage(StrictModel):
    source_id: str
    source_title: str
    source_url: HttpUrl
    passage_index: int
    text: str
    score: float
    source_kind: SourceKind | None = None
    retrieval_method: RetrievalMethod | None = None
    trust_tier: SourceTrustTier | None = None
    trust_rationale: str | None = None


class ClaimVerification(StrictModel):
    support_label: CitationSupportLabel
    reason: str
    selected_source_id: str | None = None
    selected_passage_index: int | None = None
    quote: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CitationRecord(StrictModel):
    claim: str
    support_label: CitationSupportLabel
    source_id: str
    source_title: str
    source_url: HttpUrl
    citation_key: str | None = None
    passage_index: int
    quote: str
    confidence: float


class FinalReport(StrictModel):
    markdown: str
    citations: list[CitationRecord]
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RunEvent(StrictModel):
    id: int
    run_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RunConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class RunConversationMessage(StrictModel):
    id: str
    run_id: str
    role: RunConversationRole
    content: str
    model: str | None = None
    references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RunConversationRequest(StrictModel):
    message: str = Field(min_length=1, max_length=12000)


class RunConversationReply(StrictModel):
    user_message: RunConversationMessage
    assistant_message: RunConversationMessage


class ArtifactRecord(StrictModel):
    id: str
    run_id: str
    source_id: str
    kind: str
    uri: str
    content_type: str
    size_bytes: int
    sha256: str
    created_at: datetime = Field(default_factory=utc_now)


class ResearchAssetRecord(StrictModel):
    id: str
    project_id: str | None = None
    run_id: str | None = None
    source_type: ResearchAssetType
    usage: ResearchAssetUsage
    label: str
    description: str | None = None
    url: str | None = None
    content_text: str | None = None
    extracted_text: str | None = None
    content_type: str | None = None
    file_name: str | None = None
    processing_status: AssetProcessingStatus = AssetProcessingStatus.READY
    extraction_method: AssetExtractionMethod = AssetExtractionMethod.UNKNOWN
    ocr_used: bool = False
    page_count: int | None = None
    file_size_bytes: int | None = None
    sha256: str | None = None
    warnings: list[str] = Field(default_factory=list)
    preview_excerpt: str | None = None
    processing_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class StagedAssetRecord(StrictModel):
    id: str
    source_type: ResearchAssetType
    usage: ResearchAssetUsage
    label: str
    description: str | None = None
    url: str | None = None
    extracted_text: str | None = None
    content_type: str | None = None
    file_name: str | None = None
    processing_status: AssetProcessingStatus = AssetProcessingStatus.READY
    extraction_method: AssetExtractionMethod = AssetExtractionMethod.UNKNOWN
    ocr_used: bool = False
    page_count: int | None = None
    file_size_bytes: int | None = None
    sha256: str | None = None
    warnings: list[str] = Field(default_factory=list)
    preview_excerpt: str | None = None
    processing_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SourceRegistryEntry(StrictModel):
    id: str
    run_id: str
    source_id: str | None = None
    asset_id: str | None = None
    asset_origin: str | None = None
    user_supplied: bool = False
    url: str
    canonical_url: str
    normalized_url: str
    citation_key: str | None = None
    title: str | None = None
    provider: str | None = None
    discovered_via: str
    survived_final_citation: bool | None = None
    removed_in_audit: bool | None = None
    audit_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class CitationAuditRecord(StrictModel):
    id: str
    run_id: str
    section_title: str
    ordinal: int
    claim: str
    decision: CitationAuditDecision
    reasons: list[CitationAuditReason] = Field(default_factory=list)
    source_id: str | None = None
    citation_key: str | None = None
    source_url: str | None = None
    normalized_url: str | None = None
    matched_strategy: CitationMatchStrategy | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryRecord(StrictModel):
    id: str
    profile_id: str | None = None
    kind: MemoryKind
    scope: MemoryScope
    phase_hints: list[ContextPhase] = Field(default_factory=list)
    summary: str
    content: str
    source_run_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    trust_tier: SourceTrustTier | None = None
    freshness_expires_at: datetime | None = None
    usefulness_score: float = Field(default=0.5, ge=0.0, le=1.0)
    invalidated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ContextFragment(StrictModel):
    id: str
    kind: ContextFragmentKind
    phase: ContextPhase
    memory_id: str | None = None
    title: str
    content: str
    token_estimate: int = Field(default=0, ge=0)
    score: float = 0.0
    freshness_score: float = 0.0
    trust_score: float = 0.0
    selected_reason: str = ""
    dropped_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPack(StrictModel):
    id: str
    run_id: str
    profile_id: str | None = None
    phase: ContextPhase
    summary: str
    token_budget: int = Field(default=0, ge=0)
    used_tokens: int = Field(default=0, ge=0)
    fragments: list[ContextFragment] = Field(default_factory=list)
    dropped_fragments: list[ContextFragment] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class BehaviorAssessment(StrictModel):
    id: str
    run_id: str
    profile_id: str | None = None
    kind: AssessmentKind
    source: AssessmentSource = AssessmentSource.SYSTEM
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class PublicRuntimeConfig(StrictModel):
    app_name: str
    environment: str
    prompt_profile_version: str
    source_trust_policy_version: str
    default_budget: BudgetPolicy
    default_agent_config: AgentConfig
    backends: dict[str, str]
    models: ModelConfig
    prompt_mode: PromptMode = PromptMode.CODE
    available_sources: list[SourceCatalogEntry] = Field(default_factory=list)
    default_source_selection: list[str] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)


class RunSummary(StrictModel):
    id: str
    question: str
    profile_id: str = "default"
    project_id: str | None = None
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    estimated_cost_usd: float = 0.0
    final_report_markdown: str | None = None
    error_message: str | None = None
    worker_id: str | None = None
    workflow_backend: str | None = None
    last_heartbeat_at: datetime | None = None
    terminal_reason: str | None = None
    approval_status: PlanApprovalStatus | None = None
    job_id: str | None = None


class ResearchStreamView(StrictModel):
    id: str
    name: str
    objective: str
    model: str
    status: StreamStatus
    sources_examined: int
    elapsed_ms: int
    cost_so_far: float
    confidence: float | None = None


class RunDetail(RunSummary):
    budget: BudgetPolicy | None = None
    requested_budget: BudgetPolicy | None = None
    recommended_budget: RecommendedBudget | None = None
    effective_budget: BudgetPolicy | None = None
    budget_decision_reason: str | None = None
    agent_config: AgentConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    cancel_requested: bool = False
    execution_mode: ExecutionMode = ExecutionMode.STANDARD
    source_selection: list[str] = Field(default_factory=list)
    input_assets: list[ResearchAssetRecord] = Field(default_factory=list)
    project_assets_used: list[ResearchAssetRecord] = Field(default_factory=list)
    run_assets_used: list[ResearchAssetRecord] = Field(default_factory=list)
    planning_assets_used: list[ResearchAssetRecord] = Field(default_factory=list)
    reference_assets_used: list[ResearchAssetRecord] = Field(default_factory=list)
    asset_processing_errors: list[str] = Field(default_factory=list)
    clarification_session: ClarificationSession | None = None
    plan_preview: PlanPreview | None = None
    latest_approval_decision: ApprovalDecision | None = None
    source_registry_entries: list[SourceRegistryEntry] = Field(default_factory=list)
    job: AsyncJob | None = None
    streams: list[ResearchStreamView] = Field(default_factory=list)
    latest_plan: ResearchPlan | None = None
    final_report: FinalReport | None = None
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    citation_audits: list[CitationAuditRecord] = Field(default_factory=list)
    active_context_pack_ids: list[str] = Field(default_factory=list)
    behavior_assessments: list[BehaviorAssessment] = Field(default_factory=list)
    conversation_messages: list[RunConversationMessage] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list)


class WorkspacePhaseKey(StrEnum):
    INTAKE = "intake"
    CLARIFY = "clarify"
    PLAN = "plan"
    EXECUTE = "execute"
    GROUND = "ground"
    AUDIT = "audit"
    DELIVER = "deliver"


class WorkspacePhaseStatus(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    FAILED = "failed"


class WorkspaceDecisionCategory(StrEnum):
    PLANNING = "planning"
    CLARIFICATION = "clarification"
    STREAM_LAUNCH = "stream_launch"
    REPLAN = "replan"
    SOURCE_SELECTION = "source_selection"
    VERIFICATION = "verification"
    CLAIM_REPAIR = "claim_repair"
    AUDIT = "audit"


class WorkspaceSourceOrigin(StrEnum):
    PROJECT_CORPUS = "project_corpus"
    RUN_ATTACHMENT = "run_attachment"
    USER_REFERENCE = "user_reference"
    WEB_DISCOVERED = "web_discovered"


class WorkspaceSourceState(StrEnum):
    DISCOVERED = "discovered"
    FETCHED = "fetched"
    CHUNKED = "chunked"
    RETRIEVED = "retrieved"
    CITED = "cited"
    REMOVED = "removed"


class WorkspaceConnectionTransport(StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    REPLAY = "replay"
    LIVE = "live"
    TERMINAL = "terminal"
    ERROR = "error"


class WorkspacePhaseState(StrictModel):
    key: WorkspacePhaseKey
    label: str
    status: WorkspacePhaseStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    blocked_reason: str | None = None
    event_count: int = 0


class WorkspaceTaskView(StrictModel):
    id: str
    stream_id: str | None = None
    stream_name: str | None = None
    task_type: str
    objective: str
    status: TaskStatus | str
    query_count: int = 0
    selected_source_count: int = 0
    notes_produced: int = 0
    elapsed_ms: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    next_action: str | None = None
    blocker_reason: str | None = None
    latest_sources: list[str] = Field(default_factory=list)
    latest_note_summary: str | None = None
    last_tool_call: str | None = None
    last_decision: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceStreamView(StrictModel):
    id: str
    name: str
    objective: str
    model: str
    status: StreamStatus
    query_count: int = 0
    selected_source_count: int = 0
    note_count: int = 0
    latest_source_titles: list[str] = Field(default_factory=list)
    latest_note_summary: str | None = None
    tasks: list[WorkspaceTaskView] = Field(default_factory=list)
    confidence: float | None = None
    elapsed_ms: int | None = None
    cost_so_far: float = 0.0


class WorkspaceDecisionView(StrictModel):
    id: str
    category: WorkspaceDecisionCategory
    title: str
    rationale: str
    affected_stream_id: str | None = None
    affected_stream_name: str | None = None
    affected_section: str | None = None
    affected_claim: str | None = None
    confidence: float | None = None
    uncertainty: str | None = None
    supporting_evidence_count: int = 0
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceSourceView(StrictModel):
    id: str
    source_id: str | None = None
    asset_id: str | None = None
    title: str | None = None
    url: str
    origin: WorkspaceSourceOrigin
    state: WorkspaceSourceState
    provider: str | None = None
    trust_tier: SourceTrustTier | None = None
    stream_ids: list[str] = Field(default_factory=list)
    stream_names: list[str] = Field(default_factory=list)
    report_sections: list[str] = Field(default_factory=list)
    citation_status: str | None = None
    survived_final_citation: bool | None = None
    removed_in_audit: bool | None = None
    audit_reasons: list[str] = Field(default_factory=list)
    note_summaries: list[str] = Field(default_factory=list)
    passages_used: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceCitationView(StrictModel):
    id: str
    section_title: str
    claim: str
    status: str
    source_id: str | None = None
    source_title: str | None = None
    source_url: str | None = None
    citation_key: str | None = None
    support_label: CitationSupportLabel | None = None
    quote: str | None = None
    confidence: float | None = None
    trust_tier: SourceTrustTier | None = None
    audit_status: CitationAuditDecision | None = None
    audit_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceReportClaimView(StrictModel):
    section_title: str
    ordinal: int
    claim: str
    support_label: CitationSupportLabel | None = None
    confidence: float | None = None
    citation_count: int = 0
    removed_citation_count: int = 0
    claim_repair_ran: bool = False


class WorkspaceReportSectionView(StrictModel):
    id: str
    title: str
    body_markdown: str
    draft_status: str
    grounded_claim_count: int = 0
    unsupported_claim_count: int = 0
    citation_count: int = 0
    removed_citation_count: int = 0
    claim_repair_count: int = 0
    claims: list[WorkspaceReportClaimView] = Field(default_factory=list)


class WorkspaceConnectionState(StrictModel):
    transport: WorkspaceConnectionTransport = WorkspaceConnectionTransport.IDLE
    stream_mode: str | None = None
    backend_mode: str | None = None
    workflow_backend: str | None = None
    last_event_id: int = 0
    event_count: int = 0
    replay_lag: int = 0
    reconnect_state: str | None = None
    last_event_at: datetime | None = None


class WorkspacePlanView(StrictModel):
    clarification_session: ClarificationSession | None = None
    plan_preview: PlanPreview | None = None
    approved_plan: ResearchPlan | None = None
    requested_budget: BudgetPolicy | None = None
    recommended_budget: RecommendedBudget | None = None
    effective_budget: BudgetPolicy | None = None
    budget_decision_reason: str | None = None
    planning_assets: list[ResearchAssetRecord] = Field(default_factory=list)
    project_assets: list[ResearchAssetRecord] = Field(default_factory=list)
    reference_assets: list[ResearchAssetRecord] = Field(default_factory=list)
    approval_history: list[ApprovalDecision] = Field(default_factory=list)


class RunWorkspaceSnapshot(StrictModel):
    run_id: str
    question: str
    project_id: str | None = None
    status: RunStatus
    execution_mode: ExecutionMode = ExecutionMode.STANDARD
    approval_status: PlanApprovalStatus | None = None
    current_phase: WorkspacePhaseKey
    phases: list[WorkspacePhaseState] = Field(default_factory=list)
    plan: WorkspacePlanView
    streams: list[WorkspaceStreamView] = Field(default_factory=list)
    decisions: list[WorkspaceDecisionView] = Field(default_factory=list)
    sources: list[WorkspaceSourceView] = Field(default_factory=list)
    citations: list[WorkspaceCitationView] = Field(default_factory=list)
    report_sections: list[WorkspaceReportSectionView] = Field(default_factory=list)
    connection: WorkspaceConnectionState = Field(default_factory=WorkspaceConnectionState)
    source_selection: list[str] = Field(default_factory=list)
    project_assets_available: list[ResearchAssetRecord] = Field(default_factory=list)
    run_assets_available: list[ResearchAssetRecord] = Field(default_factory=list)
    asset_processing_errors: list[str] = Field(default_factory=list)
    final_report_markdown: str | None = None
    estimated_cost_usd: float = 0.0
    created_at: datetime
    updated_at: datetime
    generated_at: datetime = Field(default_factory=utc_now)
    job: AsyncJob | None = None


class RunWorkspaceDelta(StrictModel):
    run_id: str
    event_id: int
    changed: list[str] = Field(default_factory=list)
    reason: str | None = None
    generated_at: datetime = Field(default_factory=utc_now)


class ProjectDetail(ProjectSummary):
    assets: list[ResearchAssetRecord] = Field(default_factory=list)
    runs: list[RunSummary] = Field(default_factory=list)


class RunExecutionState(StrictModel):
    id: str
    question: str
    profile_id: str = "default"
    project_id: str | None = None
    budget: BudgetPolicy
    requested_budget: BudgetPolicy | None = None
    recommended_budget: RecommendedBudget | None = None
    effective_budget: BudgetPolicy | None = None
    agent_config: AgentConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: RunStatus
    execution_mode: ExecutionMode = ExecutionMode.STANDARD
    approval_status: PlanApprovalStatus | None = None
    latest_plan_version: int = 0
    has_draft_report: bool = False
    has_final_report: bool = False
    queued_streams: int = 0
    active_streams: int = 0
    completed_streams: int = 0
    failed_streams: int = 0
    cancel_requested: bool = False
    estimated_cost_usd: float = 0.0
    worker_id: str | None = None
    workflow_backend: str | None = None
    last_heartbeat_at: datetime | None = None
    terminal_reason: str | None = None
