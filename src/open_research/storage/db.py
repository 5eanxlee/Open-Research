from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

import orjson
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    delete,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator

from open_research.core.domain import (
    AgentConfig,
    AnswerStyle,
    ApprovalDecision,
    ArtifactRecord,
    AssessmentKind,
    AssessmentSource,
    AssetExtractionMethod,
    AssetProcessingStatus,
    AsyncJob,
    BehaviorAssessment,
    BudgetPolicy,
    CitationAuditDecision,
    CitationAuditReason,
    CitationAuditRecord,
    CitationMatchStrategy,
    ClarificationSession,
    ContextFragment,
    ContextPack,
    ContextPhase,
    ExecutionMode,
    FetchedDocument,
    FinalReport,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    PlanApprovalStatus,
    PlanPreview,
    ProfileFeedback,
    ProfilePreferences,
    ProfileRecord,
    ProjectDetail,
    ProjectSummary,
    RecommendedBudget,
    ResearchAssetRecord,
    ResearchAssetType,
    ResearchAssetUsage,
    ResearchInputAsset,
    ResearchPlan,
    ResearchStreamView,
    RetrievalMethod,
    RunConversationMessage,
    RunConversationRole,
    RunDetail,
    RunEvent,
    RunExecutionState,
    RunStatus,
    RunSummary,
    SourceKind,
    SourceRegistryEntry,
    SourceTrustTier,
    StagedAssetRecord,
    StreamStatus,
    TaskStatus,
)
from open_research.core.utils import clean_text, cosine_similarity, tokenize

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - optional postgres dependency
    Vector = None


def utc_now() -> datetime:
    return datetime.now(UTC)


def _extract_agent_config(metadata: dict[str, Any] | None) -> AgentConfig | None:
    if not metadata:
        return None
    raw = metadata.get("agent_config")
    if raw is None:
        return None
    return AgentConfig.model_validate(raw)


def _extract_budget(metadata: dict[str, Any] | None, key: str) -> BudgetPolicy | None:
    if not metadata:
        return None
    raw = metadata.get(key)
    if raw is None:
        return None
    return BudgetPolicy.model_validate(raw)


def _extract_recommended_budget(
    metadata: dict[str, Any] | None,
    key: str = "recommended_budget",
) -> RecommendedBudget | None:
    if not metadata:
        return None
    raw = metadata.get(key)
    if raw is None:
        return None
    return RecommendedBudget.model_validate(raw)


def _extract_clarification_session(
    metadata: dict[str, Any] | None,
) -> ClarificationSession | None:
    if not metadata:
        return None
    raw = metadata.get("clarification_session")
    if raw is None:
        return None
    return ClarificationSession.model_validate(raw)


def _extract_plan_preview(metadata: dict[str, Any] | None) -> PlanPreview | None:
    if not metadata:
        return None
    raw = metadata.get("plan_preview")
    if raw is None:
        return None
    return PlanPreview.model_validate(raw)


def _extract_approval_decision(metadata: dict[str, Any] | None) -> ApprovalDecision | None:
    if not metadata:
        return None
    raw = metadata.get("latest_approval_decision")
    if raw is None:
        return None
    return ApprovalDecision.model_validate(raw)


class EmbeddingVectorType(TypeDecorator):
    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int = 1536) -> None:
        super().__init__()
        self.dimensions = dimensions

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and Vector is not None:
            return dialect.type_descriptor(Vector(self.dimensions))
        return dialect.type_descriptor(JSON())


class SearchVectorType(TypeDecorator):
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(TSVECTOR())
        return dialect.type_descriptor(Text())


class Base(AsyncAttrs, DeclarativeBase):
    pass


class RunORM(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    profile_id: Mapped[str] = mapped_column(String(128), default="default", index=True)
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    budget_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    draft_report_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    final_report_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    final_report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    workflow_backend: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    terminal_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class JobORM(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    submission_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="async")
    owner_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resume_cursor: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ProjectORM(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class ResearchAssetORM(Base):
    __tablename__ = "research_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    usage: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class StagedAssetORM(Base):
    __tablename__ = "staged_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    usage: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class RunConversationMessageORM(Base):
    __tablename__ = "run_conversation_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    references_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )


class PlanSnapshotORM(Base):
    __tablename__ = "plan_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    plan_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ResearchStreamORM(Base):
    __tablename__ = "research_streams"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    plan_snapshot_id: Mapped[str] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    sources_examined: Mapped[int] = mapped_column(Integer, default=0)
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0)
    cost_so_far: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class TaskORM(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    stream_id: Mapped[str] = mapped_column(String(36), index=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class TaskAttemptORM(Base):
    __tablename__ = "task_attempts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class NoteORM(Base):
    __tablename__ = "notes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    stream_id: Mapped[str] = mapped_column(String(36), index=True)
    source_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    key_facts_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    open_questions_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SourceORM(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    stream_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    retrieval_method: Mapped[str] = mapped_column(String(32), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SourcePassageORM(Base):
    __tablename__ = "source_passages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    source_id: Mapped[str] = mapped_column(String(36), index=True)
    passage_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    search_document: Mapped[str] = mapped_column(Text, default="")
    search_vector: Mapped[str | None] = mapped_column(SearchVectorType(), nullable=True)
    embedding_vector: Mapped[list[float] | None] = mapped_column(
        EmbeddingVectorType(),
        nullable=True,
    )
    start_offset: Mapped[int] = mapped_column(Integer, default=0)
    end_offset: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)


class ClaimORM(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    section_title: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    support_label: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class CitationORM(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    claim_id: Mapped[str] = mapped_column(String(36), index=True)
    source_id: Mapped[str] = mapped_column(String(36), index=True)
    passage_index: Mapped[int] = mapped_column(Integer, nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    support_label: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ArtifactORM(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    source_id: Mapped[str] = mapped_column(String(36), index=True)
    kind: Mapped[str] = mapped_column(String(128), nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SourceRegistryEntryORM(Base):
    __tablename__ = "source_registry_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    source_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_url: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    citation_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    discovered_via: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class CitationAuditORM(Base):
    __tablename__ = "citation_audits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    section_title: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reasons_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    citation_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    matched_strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class BudgetEventORM(Base):
    __tablename__ = "budget_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    delta: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RunEventORM(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ProfileORM(Base):
    __tablename__ = "profiles"

    profile_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    preferences_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class ProfileFeedbackORM(Base):
    __tablename__ = "profile_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[str] = mapped_column(String(128), index=True)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    source_fit: Mapped[float | None] = mapped_column(Float, nullable=True)
    style_fit: Mapped[float | None] = mapped_column(Float, nullable=True)
    usefulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    correction: Mapped[str | None] = mapped_column(Text, nullable=True)
    preferred_source_patterns_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    avoided_source_patterns_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    answer_style_bias: Mapped[str | None] = mapped_column(String(32), nullable=True)
    include_counterevidence_bias: Mapped[bool | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MemoryRecordORM(Base):
    __tablename__ = "memory_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    scope: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    phase_hints_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    search_document: Mapped[str] = mapped_column(Text, default="")
    source_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    source_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    tags_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    trust_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)
    freshness_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    usefulness_score: Mapped[float] = mapped_column(Float, default=0.5)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class ContextPackORM(Base):
    __tablename__ = "context_packs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    profile_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    phase: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    token_budget: Mapped[int] = mapped_column(Integer, default=0)
    used_tokens: Mapped[int] = mapped_column(Integer, default=0)
    fragments_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    dropped_fragments_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class BehaviorAssessmentORM(Base):
    __tablename__ = "behavior_assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    profile_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, default=0.5)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


def create_engine_and_sessionmaker(
    database_url: str,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine_kwargs: dict[str, Any] = {
        "echo": False,
        "json_serializer": lambda value: orjson.dumps(value).decode(),
        "json_deserializer": orjson.loads,
    }
    if database_url.startswith("sqlite+aiosqlite://"):
        engine_kwargs["connect_args"] = {"timeout": 60}
    if database_url.startswith("sqlite+aiosqlite:///:memory:"):
        engine_kwargs["poolclass"] = StaticPool
    engine = create_async_engine(database_url, **engine_kwargs)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


class ResearchStore:
    def __init__(
        self,
        engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self.engine = engine
        self.session_factory = session_factory

    async def init_db(self, *, bootstrap_mode: str = "create_all") -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.run_sync(self._apply_schema_patches)

    async def close(self) -> None:
        await self.engine.dispose()

    async def ping(self) -> None:
        async with self.session_factory() as session:
            await session.execute(select(1))

    async def _retry_on_locked_sqlite(self, operation):
        for attempt in range(5):
            try:
                return await operation()
            except OperationalError as exc:
                if "database is locked" not in str(exc).lower() or attempt == 4:
                    raise
                await asyncio.sleep(0.05 * (attempt + 1))

    @staticmethod
    def _apply_schema_patches(connection) -> None:
        if connection.dialect.name == "sqlite":
            connection.exec_driver_sql("PRAGMA journal_mode=WAL")
            connection.exec_driver_sql("PRAGMA synchronous=NORMAL")
            connection.exec_driver_sql("PRAGMA busy_timeout=60000")
        inspector = inspect(connection)
        existing_tables = set(inspector.get_table_names())
        for table_name in (
            "artifacts",
            "source_registry_entries",
            "citation_audits",
            "profiles",
            "profile_feedback",
            "memory_records",
            "context_packs",
            "behavior_assessments",
            "projects",
            "research_assets",
            "staged_assets",
            "run_conversation_messages",
        ):
            if table_name not in existing_tables:
                Base.metadata.tables[table_name].create(connection, checkfirst=True)
        inspector = inspect(connection)
        existing_columns = {
            table_name: {column["name"] for column in inspector.get_columns(table_name)}
            for table_name in inspector.get_table_names()
        }
        migrations = {
            "runs": {
                "profile_id": (
                    "ALTER TABLE runs ADD COLUMN profile_id VARCHAR(128) NOT NULL DEFAULT 'default'"
                ),
                "project_id": "ALTER TABLE runs ADD COLUMN project_id VARCHAR(36) NULL",
                "estimated_cost_usd": (
                    "ALTER TABLE runs ADD COLUMN estimated_cost_usd FLOAT NOT NULL DEFAULT 0.0"
                ),
                "cancel_requested_at": (
                    "ALTER TABLE runs ADD COLUMN cancel_requested_at DATETIME NULL"
                ),
                "worker_id": "ALTER TABLE runs ADD COLUMN worker_id VARCHAR(128) NULL",
                "workflow_backend": (
                    "ALTER TABLE runs ADD COLUMN workflow_backend VARCHAR(64) NULL"
                ),
                "last_heartbeat_at": (
                    "ALTER TABLE runs ADD COLUMN last_heartbeat_at DATETIME NULL"
                ),
                "terminal_reason": "ALTER TABLE runs ADD COLUMN terminal_reason TEXT NULL",
            },
            "source_passages": {
                "search_document": (
                    "ALTER TABLE source_passages ADD COLUMN "
                    "search_document TEXT NOT NULL DEFAULT ''"
                ),
                "search_vector": ("ALTER TABLE source_passages ADD COLUMN search_vector TEXT NULL"),
                "embedding_vector": (
                    "ALTER TABLE source_passages ADD COLUMN embedding_vector JSON NULL"
                ),
            },
        }
        for table_name, columns in migrations.items():
            present_columns = existing_columns.get(table_name, set())
            for column_name, statement in columns.items():
                if column_name not in present_columns:
                    connection.execute(text(statement))

    async def create_run(
        self,
        question: str,
        budget: BudgetPolicy,
        *,
        profile_id: str = "default",
        project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunSummary:
        run = RunORM(
            id=str(uuid4()),
            question=question,
            profile_id=profile_id,
            project_id=project_id,
            status=RunStatus.QUEUED.value,
            budget_json=budget.model_dump(mode="json"),
            metadata_json=metadata or {},
        )
        async with self.session_factory() as session, session.begin():
            session.add(run)
        return self._run_to_summary(run)

    async def update_run_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        async with self.session_factory() as session, session.begin():
            run = await session.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            merged = dict(run.metadata_json or {})
            merged.update(metadata)
            run.metadata_json = merged

    async def replace_run_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        async with self.session_factory() as session, session.begin():
            run = await session.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.metadata_json = dict(metadata)

    async def update_run_budget(self, run_id: str, budget: BudgetPolicy) -> None:
        async with self.session_factory() as session, session.begin():
            run = await session.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.budget_json = budget.model_dump(mode="json")

    async def create_job(
        self,
        *,
        run_id: str,
        submission_mode: str = "async",
        owner_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncJob:
        job = JobORM(
            job_id=str(uuid4()),
            run_id=run_id,
            status=RunStatus.QUEUED.value,
            submission_mode=submission_mode,
            owner_id=owner_id,
            metadata_json=metadata or {},
        )
        async with self.session_factory() as session, session.begin():
            session.add(job)
        return self._job_to_record(job)

    async def get_job(self, job_id: str) -> AsyncJob | None:
        async with self.session_factory() as session:
            job = await session.get(JobORM, job_id)
            if job is None:
                return None
            return self._job_to_record(job)

    async def get_job_by_run(self, run_id: str) -> AsyncJob | None:
        async with self.session_factory() as session:
            stmt = select(JobORM).where(JobORM.run_id == run_id).limit(1)
            job = (await session.execute(stmt)).scalar_one_or_none()
            return self._job_to_record(job) if job is not None else None

    async def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        resume_cursor: int | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        last_heartbeat_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            job = await session.get(JobORM, job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found")
            if status is not None:
                job.status = status
            if resume_cursor is not None:
                job.resume_cursor = resume_cursor
            if started_at is not None:
                job.started_at = started_at
            if ended_at is not None:
                job.ended_at = ended_at
            if last_heartbeat_at is not None:
                job.last_heartbeat_at = last_heartbeat_at
            if metadata is not None:
                merged = dict(job.metadata_json or {})
                merged.update(metadata)
                job.metadata_json = merged

    async def get_run(self, run_id: str) -> RunSummary | None:
        async def _get() -> RunSummary | None:
            async with self.session_factory() as session:
                run = await session.get(RunORM, run_id)
                return self._run_to_summary(run) if run else None

        return await self._retry_on_locked_sqlite(_get)

    async def list_runs(
        self,
        *,
        limit: int = 50,
        status: RunStatus | None = None,
        project_id: str | None = None,
    ) -> list[RunSummary]:
        async def _list() -> list[RunSummary]:
            async with self.session_factory() as session:
                stmt = select(RunORM).order_by(RunORM.created_at.desc()).limit(limit)
                if status is not None:
                    stmt = stmt.where(RunORM.status == status.value)
                if project_id is not None:
                    stmt = stmt.where(RunORM.project_id == project_id)
                runs = (await session.execute(stmt)).scalars().all()
                return [self._run_to_summary(run) for run in runs]

        return await self._retry_on_locked_sqlite(_list)

    async def create_project(
        self,
        name: str,
        *,
        description: str | None = None,
    ) -> ProjectSummary:
        project = ProjectORM(
            id=str(uuid4()),
            name=name,
            description=description,
        )
        async with self.session_factory() as session, session.begin():
            session.add(project)
            await session.flush()
        return self._project_to_summary(project)

    async def list_projects(self) -> list[ProjectSummary]:
        async with self.session_factory() as session:
            stmt = select(ProjectORM).order_by(
                ProjectORM.updated_at.desc(), ProjectORM.created_at.desc()
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [self._project_to_summary(row) for row in rows]

    async def get_project(self, project_id: str) -> ProjectSummary | None:
        async with self.session_factory() as session:
            row = await session.get(ProjectORM, project_id)
            return self._project_to_summary(row) if row is not None else None

    async def save_research_asset(
        self,
        asset: ResearchInputAsset,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchAssetRecord:
        record = ResearchAssetORM(
            id=str(uuid4()),
            project_id=project_id,
            run_id=run_id,
            source_type=asset.source_type.value,
            usage=asset.usage.value,
            label=asset.label,
            description=asset.description,
            url=str(asset.url) if asset.url is not None else None,
            content_text=asset.content_text,
            content_type=asset.content_type,
            file_name=asset.file_name,
            metadata_json=metadata or {},
        )
        async with self.session_factory() as session, session.begin():
            session.add(record)
            await session.flush()
            if project_id is not None:
                project = await session.get(ProjectORM, project_id)
                if project is not None:
                    project.updated_at = utc_now()
        return self._research_asset_to_record(record)

    async def list_research_assets(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
    ) -> list[ResearchAssetRecord]:
        async with self.session_factory() as session:
            stmt = select(ResearchAssetORM)
            if project_id is not None:
                stmt = stmt.where(ResearchAssetORM.project_id == project_id)
            if run_id is not None:
                stmt = stmt.where(ResearchAssetORM.run_id == run_id)
            stmt = stmt.order_by(ResearchAssetORM.created_at.asc())
            rows = (await session.execute(stmt)).scalars().all()
            return [self._research_asset_to_record(row) for row in rows]

    async def get_research_asset(self, asset_id: str) -> ResearchAssetRecord | None:
        async with self.session_factory() as session:
            row = await session.get(ResearchAssetORM, asset_id)
            return self._research_asset_to_record(row) if row is not None else None

    async def delete_project_asset(self, project_id: str, asset_id: str) -> bool:
        async with self.session_factory() as session, session.begin():
            asset = await session.get(ResearchAssetORM, asset_id)
            if asset is None or asset.project_id != project_id:
                return False
            await session.delete(asset)
            project = await session.get(ProjectORM, project_id)
            if project is not None:
                project.updated_at = utc_now()
            return True

    async def create_staged_asset(
        self,
        asset: ResearchInputAsset,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> StagedAssetRecord:
        record = StagedAssetORM(
            id=str(uuid4()),
            source_type=asset.source_type.value,
            usage=asset.usage.value,
            label=asset.label,
            description=asset.description,
            url=str(asset.url) if asset.url is not None else None,
            content_text=asset.content_text,
            content_type=asset.content_type,
            file_name=asset.file_name,
            metadata_json=metadata or {},
        )
        async with self.session_factory() as session, session.begin():
            session.add(record)
            await session.flush()
        return self._staged_asset_to_record(record)

    async def get_staged_asset(self, asset_id: str) -> StagedAssetRecord | None:
        async with self.session_factory() as session:
            row = await session.get(StagedAssetORM, asset_id)
            return self._staged_asset_to_record(row) if row is not None else None

    async def list_staged_assets(self, asset_ids: Sequence[str]) -> list[StagedAssetRecord]:
        if not asset_ids:
            return []
        async with self.session_factory() as session:
            stmt = (
                select(StagedAssetORM)
                .where(StagedAssetORM.id.in_(list(asset_ids)))
                .order_by(StagedAssetORM.created_at.asc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [self._staged_asset_to_record(row) for row in rows]

    async def delete_staged_asset(self, asset_id: str) -> bool:
        async with self.session_factory() as session, session.begin():
            asset = await session.get(StagedAssetORM, asset_id)
            if asset is None:
                return False
            await session.delete(asset)
            return True

    async def materialize_staged_assets_to_run(
        self,
        *,
        run_id: str,
        staged_asset_ids: Sequence[str],
    ) -> list[ResearchAssetRecord]:
        if not staged_asset_ids:
            return []
        async with self.session_factory() as session, session.begin():
            stmt = (
                select(StagedAssetORM)
                .where(StagedAssetORM.id.in_(list(staged_asset_ids)))
                .order_by(StagedAssetORM.created_at.asc())
            )
            staged_assets = (await session.execute(stmt)).scalars().all()
            records: list[ResearchAssetRecord] = []
            for staged in staged_assets:
                record = ResearchAssetORM(
                    id=str(uuid4()),
                    run_id=run_id,
                    source_type=staged.source_type,
                    usage=staged.usage,
                    label=staged.label,
                    description=staged.description,
                    url=staged.url,
                    content_text=staged.content_text,
                    content_type=staged.content_type,
                    file_name=staged.file_name,
                    metadata_json=dict(staged.metadata_json or {}),
                )
                session.add(record)
                await session.flush()
                records.append(self._research_asset_to_record(record))
                await session.delete(staged)
            return records

    async def promote_run_asset_to_project(
        self,
        *,
        run_id: str,
        asset_id: str,
        project_id: str,
    ) -> ResearchAssetRecord:
        async with self.session_factory() as session, session.begin():
            asset = await session.get(ResearchAssetORM, asset_id)
            if asset is None or asset.run_id != run_id:
                raise KeyError(f"Run asset {asset_id} not found")
            project = await session.get(ProjectORM, project_id)
            if project is None:
                raise KeyError(f"Project {project_id} not found")
            promoted = ResearchAssetORM(
                id=str(uuid4()),
                project_id=project_id,
                source_type=asset.source_type,
                usage=asset.usage,
                label=asset.label,
                description=asset.description,
                url=asset.url,
                content_text=asset.content_text,
                content_type=asset.content_type,
                file_name=asset.file_name,
                metadata_json={
                    **dict(asset.metadata_json or {}),
                    "promoted_from_asset_id": asset.id,
                    "promoted_from_run_id": run_id,
                },
            )
            session.add(promoted)
            project.updated_at = utc_now()
            await session.flush()
            return self._research_asset_to_record(promoted)

    async def get_project_detail(
        self, project_id: str, *, run_limit: int = 100
    ) -> ProjectDetail | None:
        async with self.session_factory() as session:
            project = await session.get(ProjectORM, project_id)
            if project is None:
                return None
            asset_stmt = (
                select(ResearchAssetORM)
                .where(ResearchAssetORM.project_id == project_id)
                .order_by(ResearchAssetORM.created_at.asc())
            )
            run_stmt = (
                select(RunORM)
                .where(RunORM.project_id == project_id)
                .order_by(RunORM.created_at.desc())
                .limit(run_limit)
            )
            assets = (await session.execute(asset_stmt)).scalars().all()
            runs = (await session.execute(run_stmt)).scalars().all()
            return ProjectDetail(
                **self._project_to_summary(project).model_dump(mode="json"),
                assets=[self._research_asset_to_record(row) for row in assets],
                runs=[self._run_to_summary(row) for row in runs],
            )

    async def get_profile(self, profile_id: str) -> ProfileRecord | None:
        async with self.session_factory() as session:
            profile = await session.get(ProfileORM, profile_id)
            if profile is None:
                return None
            return self._profile_to_record(profile)

    async def upsert_profile_preferences(
        self,
        profile_id: str,
        preferences: ProfilePreferences,
    ) -> ProfileRecord:
        async with self.session_factory() as session, session.begin():
            profile = await session.get(ProfileORM, profile_id)
            if profile is None:
                profile = ProfileORM(
                    profile_id=profile_id,
                    preferences_json=preferences.model_dump(mode="json"),
                )
                session.add(profile)
            else:
                profile.preferences_json = preferences.model_dump(mode="json")
            await session.flush()
            return self._profile_to_record(profile)

    async def save_profile_feedback(self, feedback: ProfileFeedback) -> ProfileFeedback:
        async with self.session_factory() as session, session.begin():
            session.add(
                ProfileFeedbackORM(
                    id=str(uuid4()),
                    profile_id=feedback.profile_id,
                    run_id=feedback.run_id,
                    source_fit=feedback.source_fit,
                    style_fit=feedback.style_fit,
                    usefulness=feedback.usefulness,
                    correction=feedback.correction,
                    preferred_source_patterns_json=feedback.preferred_source_patterns,
                    avoided_source_patterns_json=feedback.avoided_source_patterns,
                    answer_style_bias=(
                        feedback.answer_style_bias.value
                        if feedback.answer_style_bias is not None
                        else None
                    ),
                    include_counterevidence_bias=feedback.include_counterevidence_bias,
                    created_at=feedback.created_at,
                )
            )
        return feedback

    async def list_profile_feedback(
        self,
        profile_id: str,
        *,
        run_id: str | None = None,
    ) -> list[ProfileFeedback]:
        async with self.session_factory() as session:
            stmt = (
                select(ProfileFeedbackORM)
                .where(ProfileFeedbackORM.profile_id == profile_id)
                .order_by(ProfileFeedbackORM.created_at.asc())
            )
            if run_id is not None:
                stmt = stmt.where(ProfileFeedbackORM.run_id == run_id)
            rows = (await session.execute(stmt)).scalars().all()
            return [self._profile_feedback_to_record(row) for row in rows]

    async def save_memory_record(
        self,
        memory: MemoryRecord,
    ) -> MemoryRecord:
        async with self.session_factory() as session, session.begin():
            session.add(
                MemoryRecordORM(
                    id=memory.id,
                    profile_id=memory.profile_id,
                    kind=memory.kind.value,
                    scope=memory.scope.value,
                    phase_hints_json=[phase.value for phase in memory.phase_hints],
                    summary=memory.summary,
                    content=memory.content,
                    search_document=clean_text(f"{memory.summary} {memory.content}"),
                    source_run_id=memory.source_run_id,
                    source_ids_json=memory.source_ids,
                    tags_json=memory.tags,
                    trust_tier=memory.trust_tier.value if memory.trust_tier is not None else None,
                    freshness_expires_at=memory.freshness_expires_at,
                    usefulness_score=memory.usefulness_score,
                    invalidated_at=memory.invalidated_at,
                    metadata_json=memory.metadata,
                    created_at=memory.created_at,
                    updated_at=memory.updated_at,
                )
            )
        return memory

    async def search_memories(
        self,
        query: str,
        *,
        profile_id: str | None,
        phase: ContextPhase,
        kinds: Sequence[MemoryKind] | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        query_tokens = set(tokenize(clean_text(query)))

        async with self.session_factory() as session:
            stmt = select(MemoryRecordORM).where(MemoryRecordORM.invalidated_at.is_(None))
            if kinds:
                stmt = stmt.where(MemoryRecordORM.kind.in_([kind.value for kind in kinds]))
            rows = (await session.execute(stmt)).scalars().all()

        ranked: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            if row.profile_id not in {None, profile_id}:
                continue
            memory = self._memory_to_record(row)
            content_tokens = set(
                tokenize(row.search_document or f"{memory.summary} {memory.content}")
            )
            lexical = 0.0
            if query_tokens and content_tokens:
                lexical = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
            phase_score = 0.2 if phase in memory.phase_hints else 0.0
            profile_score = (
                0.15 if memory.profile_id == profile_id and profile_id is not None else 0.0
            )
            freshness_penalty = 0.0
            freshness_expires_at = memory.freshness_expires_at
            if freshness_expires_at is not None and freshness_expires_at.tzinfo is None:
                freshness_expires_at = freshness_expires_at.replace(tzinfo=UTC)
            if freshness_expires_at is not None and freshness_expires_at < utc_now():
                freshness_penalty = 0.2
            score = (
                lexical
                + phase_score
                + profile_score
                + (0.35 * memory.usefulness_score)
                - freshness_penalty
            )
            ranked.append((score, memory))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [memory for score, memory in ranked if score > 0][:limit]

    async def invalidate_memory(self, memory_id: str) -> None:
        async with self.session_factory() as session, session.begin():
            record = await session.get(MemoryRecordORM, memory_id)
            if record is None:
                raise KeyError(f"Memory record {memory_id} not found")
            record.invalidated_at = utc_now()

    async def save_context_pack(self, pack: ContextPack) -> ContextPack:
        async with self.session_factory() as session, session.begin():
            session.add(
                ContextPackORM(
                    id=pack.id,
                    run_id=pack.run_id,
                    profile_id=pack.profile_id,
                    phase=pack.phase.value,
                    summary=pack.summary,
                    token_budget=pack.token_budget,
                    used_tokens=pack.used_tokens,
                    fragments_json=[
                        fragment.model_dump(mode="json") for fragment in pack.fragments
                    ],
                    dropped_fragments_json=[
                        fragment.model_dump(mode="json") for fragment in pack.dropped_fragments
                    ],
                    created_at=pack.created_at,
                )
            )
        return pack

    async def list_context_packs(self, run_id: str) -> list[ContextPack]:
        async with self.session_factory() as session:
            stmt = (
                select(ContextPackORM)
                .where(ContextPackORM.run_id == run_id)
                .order_by(ContextPackORM.created_at.asc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [self._context_pack_to_record(row) for row in rows]

    async def save_behavior_assessments(
        self,
        run_id: str,
        assessments: Sequence[BehaviorAssessment],
    ) -> list[BehaviorAssessment]:
        async with self.session_factory() as session, session.begin():
            for assessment in assessments:
                session.add(
                    BehaviorAssessmentORM(
                        id=assessment.id,
                        run_id=run_id,
                        profile_id=assessment.profile_id,
                        kind=assessment.kind.value,
                        source=assessment.source.value,
                        score=assessment.score,
                        rationale=assessment.rationale,
                        metadata_json=assessment.metadata,
                        created_at=assessment.created_at,
                    )
                )
        return list(assessments)

    async def list_behavior_assessments(self, run_id: str) -> list[BehaviorAssessment]:
        async with self.session_factory() as session:
            stmt = (
                select(BehaviorAssessmentORM)
                .where(BehaviorAssessmentORM.run_id == run_id)
                .order_by(BehaviorAssessmentORM.created_at.asc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [self._behavior_assessment_to_record(row) for row in rows]

    async def get_run_execution_state(self, run_id: str) -> RunExecutionState | None:
        async def _get() -> RunExecutionState | None:
            async with self.session_factory() as session:
                run = await session.get(RunORM, run_id)
                if run is None:
                    return None

                latest_plan_version_stmt = select(func.max(PlanSnapshotORM.version)).where(
                    PlanSnapshotORM.run_id == run_id
                )
                latest_plan_version = (await session.execute(latest_plan_version_stmt)).scalar_one()

                stream_status_stmt = select(ResearchStreamORM.status).where(
                    ResearchStreamORM.run_id == run_id
                )
                stream_statuses = (await session.execute(stream_status_stmt)).scalars().all()
                stream_counter = Counter(stream_statuses)

                return RunExecutionState(
                    id=run.id,
                    question=run.question,
                    profile_id=run.profile_id,
                    project_id=run.project_id,
                    budget=BudgetPolicy.model_validate(run.budget_json),
                    requested_budget=_extract_budget(
                        dict(run.metadata_json or {}), "requested_budget"
                    ),
                    recommended_budget=_extract_recommended_budget(dict(run.metadata_json or {})),
                    effective_budget=BudgetPolicy.model_validate(run.budget_json),
                    agent_config=_extract_agent_config(dict(run.metadata_json or {})),
                    metadata=dict(run.metadata_json or {}),
                    status=RunStatus(run.status),
                    execution_mode=ExecutionMode(
                        dict(run.metadata_json or {}).get(
                            "execution_mode", ExecutionMode.STANDARD.value
                        )
                    ),
                    approval_status=PlanApprovalStatus(
                        dict(run.metadata_json or {}).get(
                            "approval_status", PlanApprovalStatus.NOT_REQUIRED.value
                        )
                    ),
                    latest_plan_version=int(latest_plan_version or 0),
                    has_draft_report=run.draft_report_json is not None,
                    has_final_report=run.final_report_json is not None,
                    queued_streams=stream_counter.get(StreamStatus.QUEUED.value, 0),
                    active_streams=stream_counter.get(StreamStatus.RUNNING.value, 0),
                    completed_streams=stream_counter.get(StreamStatus.COMPLETED.value, 0),
                    failed_streams=stream_counter.get(StreamStatus.FAILED.value, 0),
                    cancel_requested=run.cancel_requested_at is not None,
                    estimated_cost_usd=run.estimated_cost_usd,
                    worker_id=run.worker_id,
                    workflow_backend=run.workflow_backend,
                    last_heartbeat_at=run.last_heartbeat_at,
                    terminal_reason=run.terminal_reason,
                )

        return await self._retry_on_locked_sqlite(_get)

    async def list_recoverable_runs(self) -> list[RunExecutionState]:
        async with self.session_factory() as session:
            stmt = (
                select(RunORM.id)
                .where(
                    RunORM.status.in_(
                        [
                            RunStatus.QUEUED.value,
                            RunStatus.PLANNING.value,
                            RunStatus.RESEARCHING.value,
                            RunStatus.GROUNDING.value,
                        ]
                    )
                )
                .order_by(RunORM.created_at.asc())
            )
            run_ids = list((await session.execute(stmt)).scalars().all())

        states: list[RunExecutionState] = []
        for run_id in run_ids:
            state = await self.get_run_execution_state(run_id)
            if state is not None:
                states.append(state)
        return states

    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_message: str | None = None,
        draft_report_json: dict[str, Any] | None = None,
        final_report: FinalReport | None = None,
        estimated_cost_usd: float | None = None,
        terminal_reason: str | None = None,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            run = await session.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.status = status.value
            if run.started_at is None and status != RunStatus.QUEUED:
                run.started_at = utc_now()
            if status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                run.completed_at = None
                run.terminal_reason = None
            if status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                run.completed_at = utc_now()
                if terminal_reason is not None:
                    run.terminal_reason = terminal_reason
            if error_message is not None:
                run.error_message = error_message
            elif status != RunStatus.FAILED:
                run.error_message = None
            if draft_report_json is not None:
                run.draft_report_json = draft_report_json
            if final_report is not None:
                run.final_report_json = final_report.model_dump(mode="json")
                run.final_report_markdown = final_report.markdown
            if estimated_cost_usd is not None:
                run.estimated_cost_usd = estimated_cost_usd
            if status == RunStatus.CANCELLED:
                run.cancel_requested_at = None

    async def set_run_execution_context(
        self,
        run_id: str,
        *,
        worker_id: str | None = None,
        workflow_backend: str | None = None,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            run = await session.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            if worker_id is not None:
                run.worker_id = worker_id
            if workflow_backend is not None:
                run.workflow_backend = workflow_backend

    async def touch_run_heartbeat(
        self,
        run_id: str,
        *,
        worker_id: str,
        workflow_backend: str,
    ) -> datetime:
        heartbeat_at = utc_now()
        async with self.session_factory() as session, session.begin():
            run = await session.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.worker_id = worker_id
            run.workflow_backend = workflow_backend
            run.last_heartbeat_at = heartbeat_at
        return heartbeat_at

    async def request_run_cancel(self, run_id: str) -> RunExecutionState:
        async with self.session_factory() as session, session.begin():
            run = await session.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.cancel_requested_at = run.cancel_requested_at or utc_now()
        state = await self.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        return state

    async def clear_run_cancel_request(self, run_id: str) -> None:
        async with self.session_factory() as session, session.begin():
            run = await session.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.cancel_requested_at = None

    async def add_run_cost(self, run_id: str, amount_usd: float) -> None:
        if amount_usd == 0:
            return
        async with self.session_factory() as session, session.begin():
            run = await session.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.estimated_cost_usd = round(run.estimated_cost_usd + amount_usd, 6)

    async def add_stream_cost(self, stream_id: str, amount_usd: float) -> None:
        if amount_usd == 0:
            return
        async with self.session_factory() as session, session.begin():
            stream = await session.get(ResearchStreamORM, stream_id)
            if stream is None:
                raise KeyError(f"Stream {stream_id} not found")
            stream.cost_so_far = round(stream.cost_so_far + amount_usd, 6)

    async def list_stale_runs(self, *, stale_before: datetime) -> list[RunExecutionState]:
        async with self.session_factory() as session:
            stmt = (
                select(RunORM.id)
                .where(
                    RunORM.status.in_(
                        [
                            RunStatus.QUEUED.value,
                            RunStatus.PLANNING.value,
                            RunStatus.RESEARCHING.value,
                            RunStatus.GROUNDING.value,
                        ]
                    ),
                    (
                        (RunORM.last_heartbeat_at.is_(None) & (RunORM.updated_at < stale_before))
                        | (RunORM.last_heartbeat_at < stale_before)
                    ),
                )
                .order_by(RunORM.updated_at.asc())
            )
            run_ids = list((await session.execute(stmt)).scalars().all())

        states: list[RunExecutionState] = []
        for run_id in run_ids:
            state = await self.get_run_execution_state(run_id)
            if state is not None:
                states.append(state)
        return states

    async def requeue_inflight_work(
        self,
        run_id: str,
        *,
        include_failed: bool = False,
    ) -> None:
        stream_statuses = [StreamStatus.RUNNING.value]
        task_statuses = [TaskStatus.RUNNING.value]
        if include_failed:
            stream_statuses.append(StreamStatus.FAILED.value)
            task_statuses.append(TaskStatus.FAILED.value)

        async with self.session_factory() as session, session.begin():
            stream_stmt = select(ResearchStreamORM).where(
                ResearchStreamORM.run_id == run_id,
                ResearchStreamORM.status.in_(stream_statuses),
            )
            streams = (await session.execute(stream_stmt)).scalars().all()
            for stream in streams:
                stream.status = StreamStatus.QUEUED.value

            task_stmt = select(TaskORM).where(
                TaskORM.run_id == run_id,
                TaskORM.status.in_(task_statuses),
            )
            tasks = (await session.execute(task_stmt)).scalars().all()
            for task in tasks:
                task.status = TaskStatus.QUEUED.value
                task.output_json = None

    async def prepare_run_for_resume(
        self,
        run_id: str,
        *,
        clear_cancel_request: bool = True,
    ) -> RunExecutionState:
        await self._retry_on_locked_sqlite(
            lambda: self.requeue_inflight_work(run_id, include_failed=True)
        )

        async def _prepare() -> None:
            async with self.session_factory() as session, session.begin():
                run = await session.get(RunORM, run_id)
                if run is None:
                    raise KeyError(f"Run {run_id} not found")
                run.completed_at = None
                run.error_message = None
                run.terminal_reason = None
                if clear_cancel_request:
                    run.cancel_requested_at = None
                if run.status in {RunStatus.FAILED.value, RunStatus.CANCELLED.value}:
                    run.status = RunStatus.QUEUED.value

        await self._retry_on_locked_sqlite(_prepare)
        state = await self.get_run_execution_state(run_id)
        if state is None:
            raise KeyError(f"Run {run_id} not found")
        return state

    async def save_plan(
        self,
        run_id: str,
        plan: ResearchPlan,
        version: int,
        *,
        streams_to_queue: Sequence[Any] | None = None,
    ) -> tuple[str, list[str]]:
        snapshot = PlanSnapshotORM(
            id=str(uuid4()),
            run_id=run_id,
            version=version,
            plan_json=plan.model_dump(mode="json"),
        )
        queued_identity = None
        if streams_to_queue is not None:
            queued_identity = {
                (stream.name, stream.objective, tuple(stream.queries))
                for stream in streams_to_queue
            }
        stream_ids: list[str] = []
        async with self.session_factory() as session, session.begin():
            session.add(snapshot)
            for stream in plan.streams:
                stream_identity = (stream.name, stream.objective, tuple(stream.queries))
                if queued_identity is not None and stream_identity not in queued_identity:
                    continue
                stream_id = str(uuid4())
                stream_ids.append(stream_id)
                session.add(
                    ResearchStreamORM(
                        id=stream_id,
                        run_id=run_id,
                        plan_snapshot_id=snapshot.id,
                        name=stream.name,
                        objective=stream.objective,
                        model=stream.model,
                        status=StreamStatus.QUEUED.value,
                    )
                )
                session.add(
                    TaskORM(
                        id=str(uuid4()),
                        run_id=run_id,
                        stream_id=stream_id,
                        kind="research",
                        objective=stream.objective,
                        status=TaskStatus.QUEUED.value,
                        input_json={"queries": stream.queries},
                    )
                )
        return snapshot.id, stream_ids

    async def create_manual_stream(
        self,
        *,
        run_id: str,
        name: str,
        objective: str,
        model: str,
        queries: Sequence[str] | None = None,
    ) -> ResearchStreamView:
        stream_plan = {
            "summary": "User-provided research inputs.",
            "hypothesis": objective,
            "streams": [
                {
                    "name": name,
                    "objective": objective,
                    "queries": list(queries or []),
                    "model": model,
                }
            ],
            "success_criteria": ["Ingest and summarize provided inputs."],
        }
        snapshot = PlanSnapshotORM(
            id=str(uuid4()),
            run_id=run_id,
            version=0,
            plan_json=stream_plan,
        )
        stream = ResearchStreamORM(
            id=str(uuid4()),
            run_id=run_id,
            plan_snapshot_id=snapshot.id,
            name=name,
            objective=objective,
            model=model,
            status=StreamStatus.RUNNING.value,
        )
        task = TaskORM(
            id=str(uuid4()),
            run_id=run_id,
            stream_id=stream.id,
            kind="provided_inputs",
            objective=objective,
            status=TaskStatus.RUNNING.value,
            input_json={"queries": list(queries or [])},
        )
        async with self.session_factory() as session, session.begin():
            session.add(snapshot)
            session.add(stream)
            session.add(task)
            await session.flush()
        return self._stream_to_view(stream)

    async def get_latest_plan(self, run_id: str) -> ResearchPlan | None:
        async with self.session_factory() as session:
            stmt = (
                select(PlanSnapshotORM)
                .where(PlanSnapshotORM.run_id == run_id)
                .order_by(PlanSnapshotORM.version.desc())
                .limit(1)
            )
            snapshot = (await session.execute(stmt)).scalar_one_or_none()
            if snapshot is None:
                return None
            return ResearchPlan.model_validate(snapshot.plan_json)

    async def get_next_plan_version(self, run_id: str) -> int:
        async with self.session_factory() as session:
            stmt = select(func.max(PlanSnapshotORM.version)).where(PlanSnapshotORM.run_id == run_id)
            current = (await session.execute(stmt)).scalar_one_or_none()
            return (current or 0) + 1

    async def list_streams(self, run_id: str) -> list[ResearchStreamView]:
        async with self.session_factory() as session:
            stmt = (
                select(ResearchStreamORM)
                .where(ResearchStreamORM.run_id == run_id)
                .order_by(ResearchStreamORM.created_at.asc())
            )
            streams = (await session.execute(stmt)).scalars().all()
            return [self._stream_to_view(stream) for stream in streams]

    async def list_queued_streams(self, run_id: str) -> list[ResearchStreamView]:
        async with self.session_factory() as session:
            stmt = (
                select(ResearchStreamORM)
                .where(
                    ResearchStreamORM.run_id == run_id,
                    ResearchStreamORM.status == StreamStatus.QUEUED.value,
                )
                .order_by(ResearchStreamORM.created_at.asc())
            )
            streams = (await session.execute(stmt)).scalars().all()
            return [self._stream_to_view(stream) for stream in streams]

    async def update_stream(
        self,
        stream_id: str,
        *,
        status: StreamStatus | None = None,
        sources_examined: int | None = None,
        elapsed_ms: int | None = None,
        confidence: float | None = None,
        cost_so_far: float | None = None,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            stream = await session.get(ResearchStreamORM, stream_id)
            if stream is None:
                raise KeyError(f"Stream {stream_id} not found")
            if status is not None:
                stream.status = status.value
            if sources_examined is not None:
                stream.sources_examined = sources_examined
            if elapsed_ms is not None:
                stream.elapsed_ms = elapsed_ms
            if confidence is not None:
                stream.confidence = confidence
            if cost_so_far is not None:
                stream.cost_so_far = cost_so_far

    async def get_task_for_stream(self, stream_id: str) -> dict[str, Any] | None:
        async with self.session_factory() as session:
            stmt = select(TaskORM).where(TaskORM.stream_id == stream_id).limit(1)
            task = (await session.execute(stmt)).scalar_one_or_none()
            if task is None:
                return None
            return {
                "id": task.id,
                "run_id": task.run_id,
                "stream_id": task.stream_id,
                "kind": task.kind,
                "objective": task.objective,
                "status": task.status,
                "input_json": task.input_json,
                "output_json": task.output_json,
                "attempt_count": task.attempt_count,
            }

    async def list_tasks(self, run_id: str) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            stmt = (
                select(TaskORM, ResearchStreamORM)
                .join(ResearchStreamORM, ResearchStreamORM.id == TaskORM.stream_id)
                .where(TaskORM.run_id == run_id)
                .order_by(TaskORM.created_at.asc())
            )
            rows = (await session.execute(stmt)).all()
            return [
                {
                    "id": task.id,
                    "run_id": task.run_id,
                    "stream_id": task.stream_id,
                    "stream_name": stream.name,
                    "kind": task.kind,
                    "objective": task.objective,
                    "status": task.status,
                    "input_json": task.input_json,
                    "output_json": task.output_json,
                    "attempt_count": task.attempt_count,
                    "created_at": task.created_at,
                    "updated_at": task.updated_at,
                }
                for task, stream in rows
            ]

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        output_json: dict[str, Any] | None = None,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            task = await session.get(TaskORM, task_id)
            if task is None:
                raise KeyError(f"Task {task_id} not found")
            task.status = status.value
            if output_json is not None:
                task.output_json = output_json

    async def get_or_create_deep_agent_stream(
        self,
        run_id: str,
        *,
        model: str,
    ) -> ResearchStreamView:
        async with self.session_factory() as session, session.begin():
            stream = await self._get_or_create_deep_agent_stream_orm(
                session,
                run_id=run_id,
                model=model,
            )
            await session.flush()
            return self._stream_to_view(stream)

    async def upsert_deep_agent_task(
        self,
        run_id: str,
        *,
        stable_key: str,
        objective: str,
        status: TaskStatus,
        agent_role: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        async with self.session_factory() as session, session.begin():
            stream = await self._get_or_create_deep_agent_stream_orm(
                session,
                run_id=run_id,
                model=model,
            )
            stmt = select(TaskORM).where(
                TaskORM.run_id == run_id,
                TaskORM.kind == "deepagents_todo",
            )
            tasks = (await session.execute(stmt)).scalars().all()
            input_json = {
                "deepagents_todo_key": stable_key,
                "agent_role": agent_role,
                **(metadata or {}),
            }
            for task in tasks:
                raw_input = dict(task.input_json or {})
                if raw_input.get("deepagents_todo_key") != stable_key:
                    continue
                task.objective = objective
                task.status = status.value
                task.input_json = {**raw_input, **input_json}
                return task.id

            task_id = str(uuid4())
            session.add(
                TaskORM(
                    id=task_id,
                    run_id=run_id,
                    stream_id=stream.id,
                    kind="deepagents_todo",
                    objective=objective,
                    status=status.value,
                    input_json=input_json,
                )
            )
            return task_id

    async def _get_or_create_deep_agent_stream_orm(
        self,
        session: AsyncSession,
        *,
        run_id: str,
        model: str,
    ) -> ResearchStreamORM:
        stmt = (
            select(ResearchStreamORM)
            .where(
                ResearchStreamORM.run_id == run_id,
                ResearchStreamORM.name == "DeepAgents Runtime",
            )
            .limit(1)
        )
        stream = (await session.execute(stmt)).scalar_one_or_none()
        if stream is not None:
            return stream

        snapshot = PlanSnapshotORM(
            id=str(uuid4()),
            run_id=run_id,
            version=0,
            plan_json={
                "summary": "DeepAgents runtime working stream.",
                "hypothesis": "DeepAgents orchestration state is persisted as tasks and notes.",
                "streams": [
                    {
                        "name": "DeepAgents Runtime",
                        "objective": (
                            "Persist DeepAgents todos, subagent traces, notes, and artifacts."
                        ),
                        "queries": [],
                        "model": model,
                    }
                ],
                "success_criteria": ["DeepAgents activity is visible in durable workspace state."],
            },
        )
        stream = ResearchStreamORM(
            id=str(uuid4()),
            run_id=run_id,
            plan_snapshot_id=snapshot.id,
            name="DeepAgents Runtime",
            objective="Persist DeepAgents todos, subagent traces, notes, and artifacts.",
            model=model,
            status=StreamStatus.RUNNING.value,
        )
        session.add(snapshot)
        session.add(stream)
        return stream

    async def create_task_attempt(self, task_id: str, provider: str) -> str:
        attempt_id = str(uuid4())
        async with self.session_factory() as session, session.begin():
            task = await session.get(TaskORM, task_id)
            if task is None:
                raise KeyError(f"Task {task_id} not found")
            task.attempt_count += 1
            session.add(
                TaskAttemptORM(
                    id=attempt_id,
                    task_id=task_id,
                    attempt_number=task.attempt_count,
                    provider=provider,
                    status=TaskStatus.RUNNING.value,
                )
            )
        return attempt_id

    async def finish_task_attempt(
        self,
        attempt_id: str,
        status: TaskStatus,
        *,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            attempt = await session.get(TaskAttemptORM, attempt_id)
            if attempt is None:
                raise KeyError(f"Task attempt {attempt_id} not found")
            attempt.status = status.value
            attempt.completed_at = utc_now()
            if error_message:
                attempt.error_message = error_message
            if metadata is not None:
                attempt.metadata_json = metadata

    async def record_budget_event(
        self,
        run_id: str,
        category: str,
        delta: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            session.add(
                BudgetEventORM(
                    id=str(uuid4()),
                    run_id=run_id,
                    category=category,
                    delta=delta,
                    metadata_json=metadata or {},
                )
            )

    async def list_budget_events(self, run_id: str) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            stmt = (
                select(BudgetEventORM)
                .where(BudgetEventORM.run_id == run_id)
                .order_by(BudgetEventORM.created_at.asc())
            )
            events = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": event.id,
                    "run_id": event.run_id,
                    "category": event.category,
                    "delta": event.delta,
                    "metadata": dict(event.metadata_json or {}),
                    "created_at": event.created_at,
                }
                for event in events
            ]

    async def has_source(self, run_id: str, canonical_url: str) -> bool:
        async with self.session_factory() as session:
            stmt = select(SourceORM.id).where(
                SourceORM.run_id == run_id,
                SourceORM.canonical_url == canonical_url,
            )
            return (await session.execute(stmt)).scalars().first() is not None

    async def save_source(
        self,
        run_id: str,
        stream_id: str | None,
        document: FetchedDocument,
    ) -> tuple[str, bool]:
        content_hash = sha256(document.content.encode("utf-8")).hexdigest()
        async with self.session_factory() as session, session.begin():
            stmt = select(SourceORM).where(
                SourceORM.run_id == run_id,
                SourceORM.canonical_url == str(document.canonical_url),
                SourceORM.content_hash == content_hash,
            )
            existing = (await session.execute(stmt)).scalars().first()
            if existing is not None:
                return existing.id, False
            source_id = str(uuid4())
            session.add(
                SourceORM(
                    id=source_id,
                    run_id=run_id,
                    stream_id=stream_id,
                    url=str(document.url),
                    canonical_url=str(document.canonical_url),
                    title=document.title,
                    source_kind=document.source_kind.value,
                    retrieval_method=document.retrieval_method.value,
                    content_hash=content_hash,
                    content=document.content,
                    metadata_json=document.metadata,
                )
            )
            return source_id, True

    async def register_source_registry_entries(
        self,
        run_id: str,
        entries: Sequence[dict[str, Any]],
    ) -> list[SourceRegistryEntry]:
        if not entries:
            return []
        records: list[SourceRegistryEntry] = []
        async with self.session_factory() as session, session.begin():
            for raw_entry in entries:
                normalized_url = str(raw_entry["normalized_url"])
                citation_key = raw_entry.get("citation_key")
                stmt = select(SourceRegistryEntryORM).where(
                    SourceRegistryEntryORM.run_id == run_id,
                    SourceRegistryEntryORM.normalized_url == normalized_url,
                    SourceRegistryEntryORM.discovered_via == raw_entry["discovered_via"],
                )
                if citation_key is None:
                    stmt = stmt.where(SourceRegistryEntryORM.citation_key.is_(None))
                else:
                    stmt = stmt.where(SourceRegistryEntryORM.citation_key == citation_key)
                existing = (await session.execute(stmt)).scalars().first()
                if existing is not None:
                    if raw_entry.get("source_id") and existing.source_id is None:
                        existing.source_id = raw_entry["source_id"]
                    merged_metadata = dict(existing.metadata_json or {})
                    merged_metadata.update(raw_entry.get("metadata", {}))
                    existing.metadata_json = merged_metadata
                    records.append(self._registry_entry_to_record(existing))
                    continue
                entry = SourceRegistryEntryORM(
                    id=str(uuid4()),
                    run_id=run_id,
                    source_id=raw_entry.get("source_id"),
                    url=raw_entry["url"],
                    canonical_url=raw_entry["canonical_url"],
                    normalized_url=normalized_url,
                    citation_key=citation_key,
                    title=raw_entry.get("title"),
                    provider=raw_entry.get("provider"),
                    discovered_via=raw_entry["discovered_via"],
                    metadata_json=raw_entry.get("metadata", {}),
                )
                session.add(entry)
                await session.flush()
                records.append(self._registry_entry_to_record(entry))
        return records

    async def list_source_registry_entries(self, run_id: str) -> list[SourceRegistryEntry]:
        async with self.session_factory() as session:
            stmt = (
                select(SourceRegistryEntryORM)
                .where(SourceRegistryEntryORM.run_id == run_id)
                .order_by(SourceRegistryEntryORM.created_at.asc())
            )
            entries = (await session.execute(stmt)).scalars().all()
            return [self._registry_entry_to_record(entry) for entry in entries]

    async def annotate_source_registry_entries(
        self,
        run_id: str,
        annotations: Sequence[dict[str, Any]],
    ) -> None:
        if not annotations:
            return
        async with self.session_factory() as session, session.begin():
            for annotation in annotations:
                stmt = select(SourceRegistryEntryORM).where(SourceRegistryEntryORM.run_id == run_id)
                source_id = annotation.get("source_id")
                normalized_url = annotation.get("normalized_url")
                citation_key = annotation.get("citation_key")
                if source_id:
                    stmt = stmt.where(SourceRegistryEntryORM.source_id == source_id)
                elif normalized_url:
                    stmt = stmt.where(SourceRegistryEntryORM.normalized_url == normalized_url)
                if citation_key is None:
                    stmt = stmt.where(SourceRegistryEntryORM.citation_key.is_(None))
                elif citation_key:
                    stmt = stmt.where(SourceRegistryEntryORM.citation_key == citation_key)
                matches = (await session.execute(stmt)).scalars().all()
                for match in matches:
                    metadata = dict(match.metadata_json or {})
                    metadata.update(annotation.get("metadata", {}))
                    match.metadata_json = metadata

    async def get_source_snapshot(self, source_id: str) -> dict[str, Any]:
        async with self.session_factory() as session:
            source = await session.get(SourceORM, source_id)
            if source is None:
                raise KeyError(f"Source {source_id} not found")
            return {
                "id": source.id,
                "title": source.title,
                "url": source.url,
                "canonical_url": source.canonical_url,
                "content": source.content,
                "source_kind": source.source_kind,
                "retrieval_method": source.retrieval_method,
                "metadata": dict(source.metadata_json or {}),
            }

    async def get_run_source_snapshot(
        self,
        run_id: str,
        canonical_url: str,
    ) -> dict[str, Any] | None:
        async with self.session_factory() as session:
            stmt = (
                select(SourceORM)
                .where(
                    SourceORM.run_id == run_id,
                    SourceORM.canonical_url == canonical_url,
                )
                .order_by(SourceORM.created_at.asc())
                .limit(1)
            )
            source = (await session.execute(stmt)).scalars().first()
            if source is None:
                return None
            return {
                "id": source.id,
                "title": source.title,
                "url": source.url,
                "canonical_url": source.canonical_url,
                "content": source.content,
                "source_kind": source.source_kind,
                "retrieval_method": source.retrieval_method,
                "metadata": dict(source.metadata_json or {}),
            }

    async def update_source_metadata(self, source_id: str, metadata: dict[str, Any]) -> None:
        async with self.session_factory() as session, session.begin():
            source = await session.get(SourceORM, source_id)
            if source is None:
                raise KeyError(f"Source {source_id} not found")
            merged = dict(source.metadata_json or {})
            merged.update(metadata)
            source.metadata_json = merged

    async def save_artifact_record(
        self,
        *,
        run_id: str,
        source_id: str,
        kind: str,
        uri: str,
        content_type: str,
        size_bytes: int,
        sha256_digest: str,
    ) -> ArtifactRecord:
        artifact = ArtifactORM(
            id=str(uuid4()),
            run_id=run_id,
            source_id=source_id,
            kind=kind,
            uri=uri,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256_digest,
        )
        async with self.session_factory() as session, session.begin():
            session.add(artifact)
            await session.flush()
        return self._artifact_to_record(artifact)

    async def list_artifacts(
        self,
        run_id: str,
        *,
        source_id: str | None = None,
    ) -> list[ArtifactRecord]:
        async with self.session_factory() as session:
            stmt = select(ArtifactORM).where(ArtifactORM.run_id == run_id)
            if source_id is not None:
                stmt = stmt.where(ArtifactORM.source_id == source_id)
            stmt = stmt.order_by(ArtifactORM.created_at.asc())
            artifacts = (await session.execute(stmt)).scalars().all()
            return [self._artifact_to_record(artifact) for artifact in artifacts]

    async def save_passages(
        self,
        run_id: str,
        source_id: str,
        passages: Sequence[dict[str, Any]],
    ) -> None:
        if not passages:
            return
        async with self.session_factory() as session, session.begin():
            for passage in passages:
                session.add(
                    SourcePassageORM(
                        id=str(uuid4()),
                        run_id=run_id,
                        source_id=source_id,
                        passage_index=passage["passage_index"],
                        text=passage["text"],
                        search_document=clean_text(passage["text"]),
                        search_vector=passage.get("search_vector"),
                        embedding_vector=passage.get("embedding_vector"),
                        start_offset=passage["start_offset"],
                        end_offset=passage["end_offset"],
                        token_count=passage["token_count"],
                    )
                )

    async def save_note(
        self,
        run_id: str,
        stream_id: str,
        source_id: str | None,
        *,
        summary: str,
        key_facts: list[str],
        open_questions: list[str],
        confidence: float,
    ) -> str:
        note_id = str(uuid4())
        async with self.session_factory() as session, session.begin():
            session.add(
                NoteORM(
                    id=note_id,
                    run_id=run_id,
                    stream_id=stream_id,
                    source_id=source_id,
                    summary=summary,
                    key_facts_json=key_facts,
                    open_questions_json=open_questions,
                    confidence=confidence,
                )
            )
        return note_id

    async def list_notes(self, run_id: str) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            stmt = (
                select(NoteORM, ResearchStreamORM, SourceORM)
                .join(ResearchStreamORM, ResearchStreamORM.id == NoteORM.stream_id)
                .outerjoin(SourceORM, SourceORM.id == NoteORM.source_id)
                .where(NoteORM.run_id == run_id)
                .order_by(NoteORM.created_at.asc())
            )
            rows = (await session.execute(stmt)).all()
            return [
                {
                    "id": note.id,
                    "stream_id": note.stream_id,
                    "stream_name": stream.name,
                    "stream_objective": stream.objective,
                    "source_id": note.source_id,
                    "source_title": source.title if source is not None else None,
                    "source_url": source.canonical_url if source is not None else None,
                    "source_kind": (SourceKind(source.source_kind) if source is not None else None),
                    "retrieval_method": (
                        RetrievalMethod(source.retrieval_method) if source is not None else None
                    ),
                    "trust_tier": (
                        SourceTrustTier((source.metadata_json or {}).get("trust_tier"))
                        if source is not None and (source.metadata_json or {}).get("trust_tier")
                        else None
                    ),
                    "trust_rationale": (
                        (source.metadata_json or {}).get("trust_rationale")
                        if source is not None
                        else None
                    ),
                    "summary": note.summary,
                    "key_facts": list(note.key_facts_json),
                    "open_questions": list(note.open_questions_json),
                    "confidence": note.confidence,
                }
                for note, stream, source in rows
            ]

    async def list_passages(self, run_id: str) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            stmt = (
                select(SourcePassageORM, SourceORM)
                .join(SourceORM, SourceORM.id == SourcePassageORM.source_id)
                .where(SourcePassageORM.run_id == run_id)
                .order_by(SourceORM.created_at.asc(), SourcePassageORM.passage_index.asc())
            )
            rows = (await session.execute(stmt)).all()
            return [
                {
                    "source_id": source.id,
                    "source_title": source.title,
                    "source_url": source.canonical_url,
                    "passage_index": passage.passage_index,
                    "text": passage.text,
                    "search_document": passage.search_document,
                    "embedding_vector": passage.embedding_vector,
                    "start_offset": passage.start_offset,
                    "end_offset": passage.end_offset,
                    "token_count": passage.token_count,
                    "source_kind": SourceKind(source.source_kind),
                    "retrieval_method": RetrievalMethod(source.retrieval_method),
                    "trust_tier": (
                        SourceTrustTier((source.metadata_json or {}).get("trust_tier"))
                        if (source.metadata_json or {}).get("trust_tier")
                        else None
                    ),
                    "trust_rationale": (source.metadata_json or {}).get("trust_rationale"),
                }
                for passage, source in rows
            ]

    async def search_passages(
        self,
        run_id: str,
        query: str,
        *,
        limit: int = 5,
        query_embedding: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_query = clean_text(query)
        if not normalized_query:
            return []

        passages = await self.list_passages(run_id)
        query_tokens = set(tokenize(normalized_query))
        if not query_tokens and not query_embedding:
            return []

        ranked: list[dict[str, Any]] = []
        for passage in passages:
            passage_tokens = set(tokenize(passage["text"]))
            lexical_score = 0.0
            if query_tokens and passage_tokens:
                overlap = len(query_tokens & passage_tokens) / max(len(query_tokens), 1)
                density = len(query_tokens & passage_tokens) / max(len(passage_tokens), 1)
                exact_bonus = 0.2 if normalized_query.lower() in passage["text"].lower() else 0.0
                lexical_score = (0.75 * overlap) + (0.25 * density) + exact_bonus

            vector_score = 0.0
            embedding_vector = passage.get("embedding_vector")
            if query_embedding and embedding_vector:
                vector_score = max(
                    0.0,
                    cosine_similarity(
                        list(query_embedding),
                        list(embedding_vector),
                    ),
                )

            score = round((0.55 * lexical_score) + (0.45 * vector_score), 4)
            if query_embedding and lexical_score == 0.0:
                score = round(vector_score, 4)
            if score <= 0.0:
                continue
            ranked.append(
                {
                    **passage,
                    "score": score,
                    "lexical_score": round(lexical_score, 4),
                    "vector_score": round(vector_score, 4),
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    async def save_draft_report(self, run_id: str, draft_report_json: dict[str, Any]) -> None:
        async with self.session_factory() as session, session.begin():
            run = await session.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.draft_report_json = draft_report_json

    async def get_draft_report(self, run_id: str) -> dict[str, Any] | None:
        async with self.session_factory() as session:
            run = await session.get(RunORM, run_id)
            if run is None:
                return None
            return run.draft_report_json

    async def get_final_report(self, run_id: str) -> FinalReport | None:
        async with self.session_factory() as session:
            run = await session.get(RunORM, run_id)
            if run is None or run.final_report_json is None:
                return None
            return FinalReport.model_validate(run.final_report_json)

    async def replace_claims_and_citations(
        self,
        run_id: str,
        claims: Sequence[dict[str, Any]],
        citations: Sequence[dict[str, Any]],
    ) -> None:
        async with self.session_factory() as session, session.begin():
            existing_claim_ids = (
                (await session.execute(select(ClaimORM.id).where(ClaimORM.run_id == run_id)))
                .scalars()
                .all()
            )
            if existing_claim_ids:
                await session.execute(
                    delete(CitationORM).where(CitationORM.claim_id.in_(existing_claim_ids))
                )
            await session.execute(delete(ClaimORM).where(ClaimORM.run_id == run_id))

            claim_id_map: dict[tuple[str, int], str] = {}
            for claim in claims:
                claim_id = str(uuid4())
                claim_id_map[(claim["section_title"], claim["ordinal"])] = claim_id
                session.add(
                    ClaimORM(
                        id=claim_id,
                        run_id=run_id,
                        section_title=claim["section_title"],
                        ordinal=claim["ordinal"],
                        claim_text=claim["claim_text"],
                        support_label=claim["support_label"],
                        confidence=claim["confidence"],
                    )
                )
            for citation in citations:
                session.add(
                    CitationORM(
                        id=str(uuid4()),
                        claim_id=claim_id_map[(citation["section_title"], citation["ordinal"])],
                        source_id=citation["source_id"],
                        passage_index=citation["passage_index"],
                        quote=citation["quote"],
                        support_label=citation["support_label"],
                        confidence=citation["confidence"],
                    )
                )

    async def replace_citation_audits(
        self,
        run_id: str,
        audits: Sequence[CitationAuditRecord],
    ) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(delete(CitationAuditORM).where(CitationAuditORM.run_id == run_id))
            for audit in audits:
                session.add(
                    CitationAuditORM(
                        id=audit.id,
                        run_id=run_id,
                        section_title=audit.section_title,
                        ordinal=audit.ordinal,
                        claim_text=audit.claim,
                        decision=audit.decision.value,
                        reasons_json=[reason.value for reason in audit.reasons],
                        source_id=audit.source_id,
                        citation_key=audit.citation_key,
                        source_url=audit.source_url,
                        normalized_url=audit.normalized_url,
                        matched_strategy=(
                            audit.matched_strategy.value
                            if audit.matched_strategy is not None
                            else None
                        ),
                        metadata_json=audit.metadata,
                    )
                )

    async def list_citation_audits(self, run_id: str) -> list[CitationAuditRecord]:
        async with self.session_factory() as session:
            stmt = (
                select(CitationAuditORM)
                .where(CitationAuditORM.run_id == run_id)
                .order_by(CitationAuditORM.ordinal.asc(), CitationAuditORM.created_at.asc())
            )
            audits = (await session.execute(stmt)).scalars().all()
            return [self._audit_to_record(audit) for audit in audits]

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        async def _append() -> RunEvent:
            event = RunEventORM(run_id=run_id, event_type=event_type, payload_json=payload or {})
            async with self.session_factory() as session, session.begin():
                session.add(event)
                await session.flush()
                return RunEvent(
                    id=event.id,
                    run_id=run_id,
                    event_type=event.event_type,
                    payload=event.payload_json,
                    created_at=event.created_at,
                )

        return await self._retry_on_locked_sqlite(_append)

    async def list_events(self, run_id: str, after_id: int = 0) -> list[RunEvent]:
        async def _list() -> list[RunEvent]:
            async with self.session_factory() as session:
                stmt = (
                    select(RunEventORM)
                    .where(RunEventORM.run_id == run_id, RunEventORM.id > after_id)
                    .order_by(RunEventORM.id.asc())
                )
                events = (await session.execute(stmt)).scalars().all()
                return [
                    RunEvent(
                        id=event.id,
                        run_id=event.run_id,
                        event_type=event.event_type,
                        payload=event.payload_json,
                        created_at=event.created_at,
                    )
                    for event in events
                ]

        return await self._retry_on_locked_sqlite(_list)

    async def save_conversation_message(
        self,
        *,
        run_id: str,
        role: RunConversationRole,
        content: str,
        model: str | None = None,
        references: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunConversationMessage:
        async def _save() -> RunConversationMessage:
            message = RunConversationMessageORM(
                id=str(uuid4()),
                run_id=run_id,
                role=role.value,
                content=content,
                model=model,
                references_json=list(references or []),
                metadata_json=dict(metadata or {}),
            )
            async with self.session_factory() as session, session.begin():
                session.add(message)
                await session.flush()
                return self._conversation_message_to_record(message)

        return await self._retry_on_locked_sqlite(_save)

    async def list_conversation_messages(self, run_id: str) -> list[RunConversationMessage]:
        async def _list() -> list[RunConversationMessage]:
            async with self.session_factory() as session:
                stmt = (
                    select(RunConversationMessageORM)
                    .where(RunConversationMessageORM.run_id == run_id)
                    .order_by(RunConversationMessageORM.created_at.asc())
                )
                rows = (await session.execute(stmt)).scalars().all()
                return [self._conversation_message_to_record(row) for row in rows]

        return await self._retry_on_locked_sqlite(_list)

    async def get_run_detail(self, run_id: str, *, event_limit: int = 200) -> RunDetail | None:
        async def _get() -> RunDetail | None:
            async with self.session_factory() as session:
                run = await session.get(RunORM, run_id)
                if run is None:
                    return None

                latest_plan_stmt = (
                    select(PlanSnapshotORM)
                    .where(PlanSnapshotORM.run_id == run_id)
                    .order_by(PlanSnapshotORM.version.desc())
                    .limit(1)
                )
                latest_plan = (await session.execute(latest_plan_stmt)).scalar_one_or_none()

                streams_stmt = (
                    select(ResearchStreamORM)
                    .where(ResearchStreamORM.run_id == run_id)
                    .order_by(ResearchStreamORM.created_at.asc())
                )
                streams = (await session.execute(streams_stmt)).scalars().all()

                events_stmt = (
                    select(RunEventORM)
                    .where(RunEventORM.run_id == run_id)
                    .order_by(RunEventORM.id.desc())
                    .limit(event_limit)
                )
                events = list(reversed((await session.execute(events_stmt)).scalars().all()))
                conversation_stmt = (
                    select(RunConversationMessageORM)
                    .where(RunConversationMessageORM.run_id == run_id)
                    .order_by(RunConversationMessageORM.created_at.asc())
                )
                conversation_messages = (await session.execute(conversation_stmt)).scalars().all()
                artifacts_stmt = (
                    select(ArtifactORM)
                    .where(ArtifactORM.run_id == run_id)
                    .order_by(ArtifactORM.created_at.asc())
                )
                artifacts = (await session.execute(artifacts_stmt)).scalars().all()
                audits_stmt = (
                    select(CitationAuditORM)
                    .where(CitationAuditORM.run_id == run_id)
                    .order_by(CitationAuditORM.ordinal.asc(), CitationAuditORM.created_at.asc())
                )
                audits = (await session.execute(audits_stmt)).scalars().all()
                context_packs_stmt = (
                    select(ContextPackORM)
                    .where(ContextPackORM.run_id == run_id)
                    .order_by(ContextPackORM.created_at.asc())
                )
                context_packs = (await session.execute(context_packs_stmt)).scalars().all()
                assessments_stmt = (
                    select(BehaviorAssessmentORM)
                    .where(BehaviorAssessmentORM.run_id == run_id)
                    .order_by(BehaviorAssessmentORM.created_at.asc())
                )
                assessments = (await session.execute(assessments_stmt)).scalars().all()
                registry_stmt = (
                    select(SourceRegistryEntryORM)
                    .where(SourceRegistryEntryORM.run_id == run_id)
                    .order_by(SourceRegistryEntryORM.created_at.asc())
                )
                registry_entries = (await session.execute(registry_stmt)).scalars().all()
                assets_stmt = (
                    select(ResearchAssetORM)
                    .where(ResearchAssetORM.run_id == run_id)
                    .order_by(ResearchAssetORM.created_at.asc())
                )
                input_assets = (await session.execute(assets_stmt)).scalars().all()
                project_assets: list[ResearchAssetORM] = []
                if run.project_id is not None:
                    project_assets_stmt = (
                        select(ResearchAssetORM)
                        .where(ResearchAssetORM.project_id == run.project_id)
                        .order_by(ResearchAssetORM.created_at.asc())
                    )
                    project_assets = (await session.execute(project_assets_stmt)).scalars().all()
                job_stmt = select(JobORM).where(JobORM.run_id == run_id).limit(1)
                job = (await session.execute(job_stmt)).scalar_one_or_none()
                metadata = dict(run.metadata_json or {})
                run_asset_records = [
                    self._research_asset_to_record(asset) for asset in input_assets
                ]
                project_asset_records = [
                    self._research_asset_to_record(asset) for asset in project_assets
                ]
                effective_assets = [*project_asset_records, *run_asset_records]
                ready_assets = [
                    asset
                    for asset in effective_assets
                    if asset.processing_status == AssetProcessingStatus.READY
                ]
                asset_processing_errors = [
                    f"{asset.label}: {asset.processing_error}"
                    for asset in effective_assets
                    if asset.processing_status == AssetProcessingStatus.FAILED
                    and asset.processing_error
                ]

                return RunDetail(
                    **self._run_to_summary(run).model_dump(mode="json"),
                    budget=BudgetPolicy.model_validate(run.budget_json),
                    requested_budget=_extract_budget(metadata, "requested_budget"),
                    recommended_budget=_extract_recommended_budget(metadata),
                    effective_budget=BudgetPolicy.model_validate(run.budget_json),
                    budget_decision_reason=metadata.get("budget_decision_reason"),
                    agent_config=_extract_agent_config(metadata),
                    metadata=metadata,
                    cancel_requested=run.cancel_requested_at is not None,
                    execution_mode=ExecutionMode(
                        metadata.get("execution_mode", ExecutionMode.STANDARD.value)
                    ),
                    source_selection=list(metadata.get("source_selection") or []),
                    input_assets=run_asset_records,
                    project_assets_used=project_asset_records,
                    run_assets_used=run_asset_records,
                    planning_assets_used=[
                        asset
                        for asset in ready_assets
                        if asset.usage == ResearchAssetUsage.PLANNING_CONTEXT
                    ],
                    reference_assets_used=[
                        asset
                        for asset in ready_assets
                        if asset.usage == ResearchAssetUsage.REFERENCE_SOURCE
                    ],
                    asset_processing_errors=asset_processing_errors,
                    clarification_session=_extract_clarification_session(metadata),
                    plan_preview=_extract_plan_preview(metadata),
                    latest_approval_decision=_extract_approval_decision(metadata),
                    source_registry_entries=[
                        self._registry_entry_to_record(entry) for entry in registry_entries
                    ],
                    job=self._job_to_record(job) if job is not None else None,
                    streams=[self._stream_to_view(stream) for stream in streams],
                    latest_plan=(
                        ResearchPlan.model_validate(latest_plan.plan_json)
                        if latest_plan is not None
                        else None
                    ),
                    final_report=(
                        FinalReport.model_validate(run.final_report_json)
                        if run.final_report_json is not None
                        else None
                    ),
                    artifacts=[self._artifact_to_record(artifact) for artifact in artifacts],
                    citation_audits=[self._audit_to_record(audit) for audit in audits],
                    active_context_pack_ids=[pack.id for pack in context_packs],
                    behavior_assessments=[
                        self._behavior_assessment_to_record(assessment)
                        for assessment in assessments
                    ],
                    conversation_messages=[
                        self._conversation_message_to_record(message)
                        for message in conversation_messages
                    ],
                    events=[
                        RunEvent(
                            id=event.id,
                            run_id=event.run_id,
                            event_type=event.event_type,
                            payload=event.payload_json,
                            created_at=event.created_at,
                        )
                        for event in events
                    ],
                )

        return await self._retry_on_locked_sqlite(_get)

    @staticmethod
    def _run_to_summary(run: RunORM) -> RunSummary:
        metadata = dict(run.metadata_json or {})
        return RunSummary(
            id=run.id,
            question=run.question,
            conversation_topic=metadata.get("conversation_topic"),
            report_title=metadata.get("report_title"),
            profile_id=run.profile_id,
            project_id=run.project_id,
            status=RunStatus(run.status),
            created_at=run.created_at,
            updated_at=run.updated_at,
            estimated_cost_usd=run.estimated_cost_usd,
            final_report_markdown=run.final_report_markdown,
            error_message=run.error_message,
            worker_id=run.worker_id,
            workflow_backend=run.workflow_backend,
            last_heartbeat_at=run.last_heartbeat_at,
            terminal_reason=run.terminal_reason,
            approval_status=PlanApprovalStatus(
                metadata.get("approval_status", PlanApprovalStatus.NOT_REQUIRED.value)
            ),
            job_id=metadata.get("job_id"),
        )

    @staticmethod
    def _conversation_message_to_record(
        message: RunConversationMessageORM,
    ) -> RunConversationMessage:
        return RunConversationMessage(
            id=message.id,
            run_id=message.run_id,
            role=RunConversationRole(message.role),
            content=message.content,
            model=message.model,
            references=list(message.references_json or []),
            metadata=dict(message.metadata_json or {}),
            created_at=message.created_at,
        )

    @staticmethod
    def _project_to_summary(project: ProjectORM) -> ProjectSummary:
        return ProjectSummary(
            id=project.id,
            name=project.name,
            description=project.description,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )

    @staticmethod
    def _research_asset_to_record(asset: ResearchAssetORM) -> ResearchAssetRecord:
        metadata = dict(asset.metadata_json or {})
        return ResearchAssetRecord(
            id=asset.id,
            project_id=asset.project_id,
            run_id=asset.run_id,
            source_type=ResearchAssetType(asset.source_type),
            usage=ResearchAssetUsage(asset.usage),
            label=asset.label,
            description=asset.description,
            url=asset.url,
            content_text=asset.content_text,
            extracted_text=metadata.get("extracted_text", asset.content_text),
            content_type=asset.content_type,
            file_name=asset.file_name,
            processing_status=AssetProcessingStatus(
                metadata.get("processing_status", AssetProcessingStatus.READY.value)
            ),
            extraction_method=AssetExtractionMethod(
                metadata.get("extraction_method", AssetExtractionMethod.UNKNOWN.value)
            ),
            ocr_used=bool(metadata.get("ocr_used", False)),
            page_count=metadata.get("page_count"),
            file_size_bytes=metadata.get("file_size_bytes"),
            sha256=metadata.get("sha256"),
            warnings=list(metadata.get("warnings") or []),
            preview_excerpt=metadata.get("preview_excerpt"),
            processing_error=metadata.get("processing_error"),
            metadata=metadata,
            created_at=asset.created_at,
            updated_at=asset.updated_at,
        )

    @staticmethod
    def _staged_asset_to_record(asset: StagedAssetORM) -> StagedAssetRecord:
        metadata = dict(asset.metadata_json or {})
        return StagedAssetRecord(
            id=asset.id,
            source_type=ResearchAssetType(asset.source_type),
            usage=ResearchAssetUsage(asset.usage),
            label=asset.label,
            description=asset.description,
            url=asset.url,
            extracted_text=metadata.get("extracted_text", asset.content_text),
            content_type=asset.content_type,
            file_name=asset.file_name,
            processing_status=AssetProcessingStatus(
                metadata.get("processing_status", AssetProcessingStatus.READY.value)
            ),
            extraction_method=AssetExtractionMethod(
                metadata.get("extraction_method", AssetExtractionMethod.UNKNOWN.value)
            ),
            ocr_used=bool(metadata.get("ocr_used", False)),
            page_count=metadata.get("page_count"),
            file_size_bytes=metadata.get("file_size_bytes"),
            sha256=metadata.get("sha256"),
            warnings=list(metadata.get("warnings") or []),
            preview_excerpt=metadata.get("preview_excerpt"),
            processing_error=metadata.get("processing_error"),
            metadata=metadata,
            created_at=asset.created_at,
            updated_at=asset.updated_at,
        )

    @staticmethod
    def _stream_to_view(stream: ResearchStreamORM) -> ResearchStreamView:
        return ResearchStreamView(
            id=stream.id,
            name=stream.name,
            objective=stream.objective,
            model=stream.model,
            status=StreamStatus(stream.status),
            sources_examined=stream.sources_examined,
            elapsed_ms=stream.elapsed_ms,
            cost_so_far=stream.cost_so_far,
            confidence=stream.confidence,
        )

    @staticmethod
    def _artifact_to_record(artifact: ArtifactORM) -> ArtifactRecord:
        return ArtifactRecord(
            id=artifact.id,
            run_id=artifact.run_id,
            source_id=artifact.source_id,
            kind=artifact.kind,
            uri=artifact.uri,
            content_type=artifact.content_type,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            created_at=artifact.created_at,
        )

    @staticmethod
    def _registry_entry_to_record(entry: SourceRegistryEntryORM) -> SourceRegistryEntry:
        metadata = dict(entry.metadata_json or {})
        return SourceRegistryEntry(
            id=entry.id,
            run_id=entry.run_id,
            source_id=entry.source_id,
            asset_id=metadata.get("asset_id"),
            asset_origin=metadata.get("asset_origin"),
            user_supplied=bool(metadata.get("user_supplied", False)),
            url=entry.url,
            canonical_url=entry.canonical_url,
            normalized_url=entry.normalized_url,
            citation_key=entry.citation_key,
            title=entry.title,
            provider=entry.provider,
            discovered_via=entry.discovered_via,
            survived_final_citation=metadata.get("survived_final_citation"),
            removed_in_audit=metadata.get("removed_in_audit"),
            audit_reasons=list(metadata.get("audit_reasons") or []),
            metadata=metadata,
            created_at=entry.created_at,
        )

    @staticmethod
    def _job_to_record(job: JobORM) -> AsyncJob:
        return AsyncJob(
            job_id=job.job_id,
            run_id=job.run_id,
            status=job.status,
            submission_mode=job.submission_mode,
            submitted_at=job.submitted_at,
            started_at=job.started_at,
            ended_at=job.ended_at,
            last_heartbeat_at=job.last_heartbeat_at,
            owner_id=job.owner_id,
            resume_cursor=job.resume_cursor,
            metadata=dict(job.metadata_json or {}),
        )

    @staticmethod
    def _audit_to_record(audit: CitationAuditORM) -> CitationAuditRecord:
        return CitationAuditRecord(
            id=audit.id,
            run_id=audit.run_id,
            section_title=audit.section_title,
            ordinal=audit.ordinal,
            claim=audit.claim_text,
            decision=CitationAuditDecision(audit.decision),
            reasons=[CitationAuditReason(reason) for reason in audit.reasons_json],
            source_id=audit.source_id,
            citation_key=audit.citation_key,
            source_url=audit.source_url,
            normalized_url=audit.normalized_url,
            matched_strategy=(
                CitationMatchStrategy(audit.matched_strategy)
                if audit.matched_strategy is not None
                else None
            ),
            metadata=dict(audit.metadata_json or {}),
            created_at=audit.created_at,
        )

    @staticmethod
    def _profile_to_record(profile: ProfileORM) -> ProfileRecord:
        return ProfileRecord(
            profile_id=profile.profile_id,
            preferences=ProfilePreferences.model_validate(profile.preferences_json or {}),
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )

    @staticmethod
    def _profile_feedback_to_record(feedback: ProfileFeedbackORM) -> ProfileFeedback:
        return ProfileFeedback(
            profile_id=feedback.profile_id,
            run_id=feedback.run_id,
            source_fit=feedback.source_fit,
            style_fit=feedback.style_fit,
            usefulness=feedback.usefulness,
            correction=feedback.correction,
            preferred_source_patterns=list(feedback.preferred_source_patterns_json or []),
            avoided_source_patterns=list(feedback.avoided_source_patterns_json or []),
            answer_style_bias=(
                AnswerStyle(feedback.answer_style_bias)
                if feedback.answer_style_bias is not None
                else None
            ),
            include_counterevidence_bias=feedback.include_counterevidence_bias,
            created_at=feedback.created_at,
        )

    @staticmethod
    def _memory_to_record(record: MemoryRecordORM) -> MemoryRecord:
        return MemoryRecord(
            id=record.id,
            profile_id=record.profile_id,
            kind=MemoryKind(record.kind),
            scope=MemoryScope(record.scope),
            phase_hints=[ContextPhase(phase) for phase in (record.phase_hints_json or [])],
            summary=record.summary,
            content=record.content,
            source_run_id=record.source_run_id,
            source_ids=list(record.source_ids_json or []),
            tags=list(record.tags_json or []),
            trust_tier=(
                SourceTrustTier(record.trust_tier) if record.trust_tier is not None else None
            ),
            freshness_expires_at=record.freshness_expires_at,
            usefulness_score=record.usefulness_score,
            invalidated_at=record.invalidated_at,
            metadata=dict(record.metadata_json or {}),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _context_pack_to_record(pack: ContextPackORM) -> ContextPack:
        return ContextPack(
            id=pack.id,
            run_id=pack.run_id,
            profile_id=pack.profile_id,
            phase=ContextPhase(pack.phase),
            summary=pack.summary,
            token_budget=pack.token_budget,
            used_tokens=pack.used_tokens,
            fragments=[
                ContextFragment.model_validate(fragment) for fragment in pack.fragments_json
            ],
            dropped_fragments=[
                ContextFragment.model_validate(fragment) for fragment in pack.dropped_fragments_json
            ],
            created_at=pack.created_at,
        )

    @staticmethod
    def _behavior_assessment_to_record(
        assessment: BehaviorAssessmentORM,
    ) -> BehaviorAssessment:
        return BehaviorAssessment(
            id=assessment.id,
            run_id=assessment.run_id,
            profile_id=assessment.profile_id,
            kind=AssessmentKind(assessment.kind),
            source=AssessmentSource(assessment.source),
            score=assessment.score,
            rationale=assessment.rationale,
            metadata=dict(assessment.metadata_json or {}),
            created_at=assessment.created_at,
        )
