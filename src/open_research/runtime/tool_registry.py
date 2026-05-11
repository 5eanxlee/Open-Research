from __future__ import annotations

from collections.abc import Iterable, Sequence

from open_research.core.config import Settings
from open_research.core.domain import ToolCatalogEntry

SEARCH_TOOL = "advanced_web_search_tool"
PAPER_SEARCH_TOOL = "paper_search_tool"
FETCH_TOOL = "fetch_page"
THINK_TOOL = "think"
WRITE_TODOS_TOOL = "write_todos"
SAVE_RESEARCH_ARTIFACT_TOOL = "save_research_artifact"
SOURCE_AUDIT_TOOL = "source_audit_tool"
CITATION_RECONCILIATION_TOOL = "citation_reconciliation_tool"

SEARCH_BUDGET_CATEGORIES = ("search_query", "claim_repair_search")
FETCH_BUDGET_CATEGORIES = ("source_fetch", "claim_repair_fetch")
EMBEDDING_BUDGET_CATEGORIES = ("embedding", "passage_embed", "query_embed")


def build_tool_catalog(settings: Settings) -> list[ToolCatalogEntry]:
    search_backend = settings.resolved_search_backend
    fetch_backend = settings.resolved_fetch_backend
    return [
        ToolCatalogEntry(
            name=SEARCH_TOOL,
            display_name="Advanced Web Search",
            category="search",
            owner="researcher-agent",
            description="Searches public web sources through the configured search backend.",
            enabled=search_backend != "mock",
            backend="search",
            provider=search_backend,
            budget_categories=list(SEARCH_BUDGET_CATEGORIES),
            per_run_limit=settings.max_search_tool_calls_per_run,
            risk="network",
            requires_auth=search_backend != "mock",
            failure_mode="provider_fallback_or_stream_failure",
        ),
        ToolCatalogEntry(
            name=PAPER_SEARCH_TOOL,
            display_name="Paper Search",
            category="search",
            owner="researcher-agent",
            description="Searches scholarly sources when a Serper Scholar key is configured.",
            enabled=settings.serper_api_key is not None,
            backend="search",
            provider="serper-scholar" if settings.serper_api_key is not None else None,
            budget_categories=["paper_search"],
            per_run_limit=settings.max_search_tool_calls_per_run,
            risk="network",
            requires_auth=True,
            failure_mode="surface_error",
        ),
        ToolCatalogEntry(
            name=FETCH_TOOL,
            display_name="Fetch Source",
            category="fetch",
            owner="researcher-agent",
            description="Fetches, normalizes, stores, and chunks discovered sources.",
            enabled=fetch_backend != "mock",
            backend="fetch",
            provider=fetch_backend,
            budget_categories=list(FETCH_BUDGET_CATEGORIES),
            per_run_limit=settings.max_fetch_tool_calls_per_run,
            risk="network",
            requires_auth=fetch_backend not in {"mock", "playwright"},
            failure_mode="search_result_fallback_or_skip",
        ),
        ToolCatalogEntry(
            name=THINK_TOOL,
            display_name="Think",
            category="reasoning",
            owner="orchestrator",
            description="Records internal progress checkpoints between delegated research steps.",
            enabled=True,
            budget_categories=[],
            per_run_limit=None,
            risk="low",
            requires_auth=False,
            failure_mode="no_op",
        ),
        ToolCatalogEntry(
            name=WRITE_TODOS_TOOL,
            display_name="Write Todos",
            category="planning",
            owner="orchestrator",
            description="Tracks ordered research subtasks for long-running investigations.",
            enabled=True,
            budget_categories=[],
            per_run_limit=None,
            risk="low",
            requires_auth=False,
            failure_mode="surface_error",
        ),
        ToolCatalogEntry(
            name=SAVE_RESEARCH_ARTIFACT_TOOL,
            display_name="Save Research Artifact",
            category="workspace",
            owner="orchestrator",
            description=(
                "Persists DeepAgents intermediate plans, notes, audits, drafts, and final reports."
            ),
            enabled=settings.resolved_artifact_store_backend != "disabled",
            budget_categories=[],
            per_run_limit=None,
            risk="filesystem",
            requires_auth=False,
            failure_mode="artifact_storage_disabled_or_surface_error",
        ),
        ToolCatalogEntry(
            name=SOURCE_AUDIT_TOOL,
            display_name="Source Audit",
            category="verification",
            owner="source-auditor-agent",
            description="Records source quality, provenance, conflict, and diversity observations.",
            enabled=True,
            budget_categories=[],
            per_run_limit=None,
            risk="low",
            requires_auth=False,
            failure_mode="no_op",
        ),
        ToolCatalogEntry(
            name=CITATION_RECONCILIATION_TOOL,
            display_name="Citation Reconciliation",
            category="verification",
            owner="citation-agent",
            description=(
                "Checks final draft citations against URLs observed during DeepAgents tool use."
            ),
            enabled=True,
            budget_categories=[],
            per_run_limit=None,
            risk="low",
            requires_auth=False,
            failure_mode="surface_warning",
        ),
        *[
            ToolCatalogEntry(
                name=name,
                display_name=name.replace("_", " ").title(),
                category="workspace",
                owner="orchestrator",
                description="Reserved workspace tool from the custom research contract.",
                enabled=False,
                budget_categories=[],
                per_run_limit=None,
                risk="filesystem",
                requires_auth=False,
                failure_mode="disabled_until_runtime_supports_workspace_tools",
            )
            for name in ("read_file", "write_file", "edit_file", "glob", "grep", "ls", "execute")
        ],
    ]


def enabled_tool_names(settings: Settings) -> list[str]:
    return [entry.name for entry in build_tool_catalog(settings) if entry.enabled]


def contract_tool_names(settings: Settings) -> list[str]:
    return [entry.name for entry in build_tool_catalog(settings)]


def count_budget_events(
    budget_events: Iterable[dict[str, object]],
    *,
    categories: Sequence[str],
) -> int:
    category_set = set(categories)
    total = 0
    for event in budget_events:
        if event.get("category") not in category_set:
            continue
        total += int(event.get("delta") or 0)
    return total


def assert_tool_budget_available(
    *,
    tool_name: str,
    budget_events: Iterable[dict[str, object]],
    categories: Sequence[str],
    limit: int,
) -> int:
    used = count_budget_events(budget_events, categories=categories)
    if used >= limit:
        category_label = ", ".join(categories)
        raise RuntimeError(
            f"Tool budget exhausted for {tool_name}: {used}/{limit} calls used "
            f"across {category_label}."
        )
    return limit - used
