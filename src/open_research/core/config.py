from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _looks_like_openai_key(secret: SecretStr | None) -> bool:
    if secret is None:
        return False
    value = secret.get_secret_value().strip()
    # OpenAI project/user keys are currently issued with an sk- prefix.
    return value.startswith("sk-")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OPEN_RESEARCH_",
        populate_by_name=True,
        env_ignore_empty=True,
        extra="ignore",
    )

    app_name: str = "Open Research"
    environment: Literal["development", "test", "production"] = "development"
    database_url: str = Field(
        default="sqlite+aiosqlite:///./openresearch.db",
        validation_alias=AliasChoices("DATABASE_URL", "OPEN_RESEARCH_DATABASE_URL"),
    )

    llm_backend: Literal["auto", "heuristic", "openai", "openai_compatible"] = "auto"
    search_backend: Literal["auto", "mock", "openai", "brave", "exa", "tavily"] = "auto"
    fetch_backend: Literal[
        "auto",
        "mock",
        "firecrawl",
        "browserbase",
        "browserbase_session",
        "playwright",
    ] = "auto"
    workflow_backend: Literal["auto", "local", "temporal"] = "auto"
    database_bootstrap_mode: Literal["auto", "create_all"] = "auto"
    artifact_store_backend: Literal["auto", "disabled", "local", "s3"] = "auto"
    embedding_backend: Literal["auto", "disabled", "mock", "openai", "openai_compatible"] = "auto"
    reranker_backend: Literal["auto", "disabled", "heuristic", "sentence_transformers"] = "auto"
    process_role: Literal["all", "api", "worker"] = "all"
    event_pubsub_mode: Literal["memory", "database"] = "database"

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "OPEN_RESEARCH_OPENAI_API_KEY"),
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_BASE_URL", "OPEN_RESEARCH_OPENAI_BASE_URL"),
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "OPEN_RESEARCH_LLM_API_KEY"),
    )
    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_BASE_URL", "OPEN_RESEARCH_LLM_BASE_URL"),
    )
    embedding_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "EMBEDDING_API_KEY",
            "OPEN_RESEARCH_EMBEDDING_API_KEY",
        ),
    )
    embedding_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "EMBEDDING_BASE_URL",
            "OPEN_RESEARCH_EMBEDDING_BASE_URL",
        ),
    )
    brave_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("BRAVE_API_KEY", "OPEN_RESEARCH_BRAVE_API_KEY"),
    )
    exa_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("EXA_API_KEY", "OPEN_RESEARCH_EXA_API_KEY"),
    )
    tavily_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TAVILY_API_KEY", "OPEN_RESEARCH_TAVILY_API_KEY"),
    )
    serper_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SERPER_API_KEY", "OPEN_RESEARCH_SERPER_API_KEY"),
    )
    firecrawl_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("FIRECRAWL_API_KEY", "OPEN_RESEARCH_FIRECRAWL_API_KEY"),
    )
    browserbase_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BROWSERBASE_API_KEY",
            "OPEN_RESEARCH_BROWSERBASE_API_KEY",
        ),
    )
    browserbase_project_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BROWSERBASE_PROJECT_ID",
            "OPEN_RESEARCH_BROWSERBASE_PROJECT_ID",
        ),
    )
    worker_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("WORKER_ID", "OPEN_RESEARCH_WORKER_ID"),
    )

    temporal_target_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "TEMPORAL_TARGET_URL",
            "OPEN_RESEARCH_TEMPORAL_TARGET_URL",
        ),
    )
    temporal_namespace: str = Field(
        default="default",
        validation_alias=AliasChoices(
            "TEMPORAL_NAMESPACE",
            "OPEN_RESEARCH_TEMPORAL_NAMESPACE",
        ),
    )
    temporal_task_queue: str = Field(
        default="open-research",
        validation_alias=AliasChoices(
            "TEMPORAL_TASK_QUEUE",
            "OPEN_RESEARCH_TEMPORAL_TASK_QUEUE",
        ),
    )
    temporal_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "TEMPORAL_API_KEY",
            "OPEN_RESEARCH_TEMPORAL_API_KEY",
        ),
    )
    temporal_tls: bool = Field(
        default=True,
        validation_alias=AliasChoices("TEMPORAL_TLS", "OPEN_RESEARCH_TEMPORAL_TLS"),
    )

    lead_model: str = "gpt-5.5"
    planner_model: str = "gpt-5.5"
    worker_model: str = "gpt-5.5"
    verifier_model: str = "gpt-5.5"
    embedding_model: str = "text-embedding-3-large"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    openai_web_search_model: str | None = None
    openai_web_search_context_size: Literal["low", "medium", "high"] = "medium"
    openai_web_search_reasoning_effort: Literal["low", "medium", "high"] = "high"
    openai_web_search_external_web_access: bool = True
    openai_web_search_max_output_tokens: int = 4096
    openai_web_search_timeout_seconds: float = 180.0
    llm_reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] = "high"
    llm_api_style: Literal["auto", "responses", "chat_completions"] = "auto"
    llm_structured_output_mode: Literal["auto", "parse", "json_schema", "prompted"] = "auto"
    llm_supports_reasoning_effort: bool | None = None
    llm_model_family: Literal["auto", "generic", "openai", "glm", "qwen", "deepseek"] = "auto"
    prompt_mode: Literal["code", "template", "hybrid"] = "hybrid"
    prompt_template_root: str | None = None
    prompt_template_strict: bool = False
    prompt_template_family_fallback: bool = True
    prompt_reload_in_dev: bool = True
    prompt_externalization_enabled: bool = True
    custom_responses_runtime_backend: Literal["auto", "pipeline", "deepagents"] = "auto"
    research_runtime_backend: Literal["auto", "langgraph", "deepagents", "hybrid"] = "auto"
    deep_plan_approval_enabled: bool = True
    source_registry_ui_enabled: bool = True
    async_jobs_enabled: bool = True
    debug_console_enabled: bool = True
    enforce_maximal_research_path: bool = True

    max_streams: int = 30
    max_replans: int = 1
    max_queries_per_stream: int = 25
    max_results_per_query: int = 5
    max_sources_per_stream: int = 3
    per_domain_limit: int = 2
    planner_min_discovery_queries: int = 10
    planner_max_discovery_queries: int = 16
    planner_min_total_sources_retrieved: int = 8
    planner_min_total_cited_sources: int = 4
    planner_validation_enabled: bool = True
    planner_max_validation_retries: int = 2
    planner_discovery_concurrency: int = 3
    deepagents_max_research_batches: int = 6
    deepagents_require_critic_pass: bool = True
    deepagents_require_source_audit_pass: bool = True
    deepagents_require_citation_pass: bool = True
    deepagents_grounding_enabled: bool = True
    deepagents_grounding_strict: bool = False
    completion_gate_min_chars: int = 2000
    completion_gate_min_headings: int = 2
    completion_gate_max_attempts: int = 5
    tool_registry_enabled: bool = True
    max_search_tool_calls_per_run: int = 200
    max_fetch_tool_calls_per_run: int = 200
    max_embedding_tool_calls_per_run: int = 500
    max_upload_file_size_bytes: int = 50 * 1024 * 1024
    max_upload_files_per_batch: int = 20
    max_ocr_pdf_pages: int = 200
    pdf_text_extraction_min_chars: int = 80

    http_timeout_seconds: float = 30.0
    sse_keepalive_seconds: float = 10.0
    automatic_run_recovery: bool = True
    max_claim_repairs: int = 1
    claim_repair_max_results: int = 2
    search_request_cost_usd: float = 0.0
    fetch_request_cost_usd: float = 0.0
    embedding_request_cost_usd: float = 0.0
    browserbase_use_proxies: bool = False
    browserbase_session_keep_alive: bool = False
    playwright_timeout_seconds: float = 15.0
    temporal_activity_timeout_seconds: int = 5400
    temporal_heartbeat_seconds: int = 10
    temporal_start_worker: bool = True
    temporal_worker_shutdown_seconds: float = 5.0
    run_heartbeat_seconds: float = 10.0
    stale_run_timeout_seconds: float = 120.0
    reconciler_interval_seconds: float = 15.0
    artifact_store_path: str = "./artifacts"
    artifact_store_s3_bucket: str | None = None
    artifact_store_s3_prefix: str = "open-research"
    artifact_store_s3_endpoint_url: str | None = None
    embedding_dimensions: int = 1536
    grounding_candidate_limit: int = 12
    grounding_max_claims_per_run: int = 4
    provider_retry_attempts: int = 3
    provider_retry_base_seconds: float = 0.5
    provider_retry_max_seconds: float = 4.0
    provider_cooldown_failures: int = 3
    provider_cooldown_seconds: float = 30.0
    metrics_enabled: bool = True
    profile_memory_enabled: bool = True
    behavior_assessment_enabled: bool = True
    otlp_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OTLP_ENDPOINT", "OPEN_RESEARCH_OTLP_ENDPOINT"),
    )
    otlp_headers: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OTLP_HEADERS", "OPEN_RESEARCH_OTLP_HEADERS"),
    )
    langfuse_public_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGFUSE_PUBLIC_KEY",
            "OPEN_RESEARCH_LANGFUSE_PUBLIC_KEY",
        ),
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGFUSE_SECRET_KEY",
            "OPEN_RESEARCH_LANGFUSE_SECRET_KEY",
        ),
    )
    langfuse_host: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_HOST", "OPEN_RESEARCH_LANGFUSE_HOST"),
    )
    trace_redaction_fields: str = (
        "authorization,cookie,set-cookie,api_key,token,password,secret,prompt,content,headers,body"
    )

    @property
    def resolved_llm_backend(self) -> Literal["heuristic", "openai", "openai_compatible"]:
        if self.llm_backend != "auto":
            return self.llm_backend
        if _looks_like_openai_key(self.openai_api_key):
            return "openai"
        if self.llm_base_url or self.llm_api_key:
            return "openai_compatible"
        return "heuristic"

    @property
    def resolved_llm_api_style(self) -> Literal["responses", "chat_completions"]:
        if self.llm_api_style != "auto":
            return self.llm_api_style
        if self.resolved_llm_backend == "openai":
            return "responses"
        return "chat_completions"

    @property
    def resolved_llm_structured_output_mode(
        self,
    ) -> Literal["parse", "json_schema", "prompted"]:
        if self.llm_structured_output_mode != "auto":
            return self.llm_structured_output_mode
        if self.resolved_llm_backend == "openai":
            return "parse"
        return "json_schema"

    @property
    def resolved_llm_supports_reasoning_effort(self) -> bool:
        if self.llm_supports_reasoning_effort is not None:
            return self.llm_supports_reasoning_effort
        return self.resolved_llm_backend == "openai"

    @property
    def resolved_llm_model_family(self) -> Literal["generic", "openai", "glm", "qwen", "deepseek"]:
        if self.llm_model_family != "auto":
            return self.llm_model_family
        model_names = " ".join(
            [self.lead_model, self.planner_model, self.worker_model, self.verifier_model]
        ).lower()
        if "gpt" in model_names:
            return "openai"
        if "glm" in model_names:
            return "glm"
        if "qwen" in model_names:
            return "qwen"
        if "deepseek" in model_names:
            return "deepseek"
        return "generic"

    @property
    def resolved_custom_responses_runtime_backend(self) -> Literal["pipeline", "deepagents"]:
        if self.custom_responses_runtime_backend != "auto":
            return self.custom_responses_runtime_backend
        if (
            self.resolved_llm_backend == "openai"
            and _looks_like_openai_key(self.openai_api_key)
            and importlib.util.find_spec("deepagents") is not None
            and importlib.util.find_spec("langchain_openai") is not None
        ):
            return "deepagents"
        return "pipeline"

    @property
    def resolved_research_runtime_backend(self) -> Literal["langgraph", "deepagents", "hybrid"]:
        if self.research_runtime_backend != "auto":
            return self.research_runtime_backend
        if (
            self.resolved_llm_backend == "openai"
            and _looks_like_openai_key(self.openai_api_key)
            and importlib.util.find_spec("deepagents") is not None
            and importlib.util.find_spec("langchain_openai") is not None
        ):
            return "hybrid"
        return "langgraph"

    @property
    def resolved_search_backend(self) -> Literal["mock", "openai", "brave", "exa", "tavily"]:
        if self.search_backend != "auto":
            return self.search_backend
        if self.brave_api_key:
            return "brave"
        if self.exa_api_key:
            return "exa"
        if self.tavily_api_key:
            return "tavily"
        if _looks_like_openai_key(self.openai_api_key):
            return "openai"
        return "mock"

    @property
    def resolved_fetch_backend(
        self,
    ) -> Literal["mock", "firecrawl", "browserbase", "browserbase_session", "playwright"]:
        if self.fetch_backend != "auto":
            return self.fetch_backend
        if self.firecrawl_api_key:
            return "firecrawl"
        if self.browserbase_api_key:
            return "browserbase"
        return "mock"

    @property
    def resolved_workflow_backend(self) -> Literal["local", "temporal"]:
        if self.workflow_backend != "auto":
            return self.workflow_backend
        return "temporal" if self.temporal_target_url else "local"

    @property
    def resolved_database_bootstrap_mode(self) -> Literal["create_all"]:
        if self.database_bootstrap_mode != "auto":
            return self.database_bootstrap_mode
        return "create_all"

    @property
    def resolved_artifact_store_backend(self) -> Literal["disabled", "local", "s3"]:
        if self.artifact_store_backend != "auto":
            return self.artifact_store_backend
        if self.artifact_store_s3_bucket:
            return "s3"
        if self.environment == "test":
            return "disabled"
        return "local"

    @property
    def resolved_embedding_backend(
        self,
    ) -> Literal["disabled", "mock", "openai", "openai_compatible"]:
        if self.embedding_backend != "auto":
            return self.embedding_backend
        if _looks_like_openai_key(self.openai_api_key):
            return "openai"
        if (
            self.embedding_base_url
            or self.embedding_api_key
            or self.llm_base_url
            or self.llm_api_key
        ):
            return "openai_compatible"
        return "mock"

    @property
    def resolved_reranker_backend(
        self,
    ) -> Literal["disabled", "heuristic", "sentence_transformers"]:
        if self.reranker_backend != "auto":
            return self.reranker_backend
        return "heuristic"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
