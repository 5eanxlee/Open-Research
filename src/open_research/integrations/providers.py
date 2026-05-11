from __future__ import annotations

import asyncio
import base64
import binascii
import contextvars
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from urllib.parse import urlparse

import httpx
import orjson
from openai import AsyncOpenAI
from openai.types.responses import ParsedResponse
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from open_research.core.domain import (
    FetchedDocument,
    RetrievalMethod,
    RetrievedPassage,
    SearchResult,
    SourceKind,
)
from open_research.core.utils import (
    clean_text,
    normalize_url,
    sanitize_source_snippet_for_url,
    slugify,
    strip_markdown_fences,
    tokenize,
)
from open_research.storage.artifacts import ArtifactPayload


@dataclass(slots=True)
class UsageInfo:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass(slots=True)
class GenerationResult[T]:
    value: T
    usage: UsageInfo
    metadata: dict[str, Any] | None = None


class ProviderError(RuntimeError):
    pass


@dataclass(slots=True)
class ProviderRetryNotice:
    provider_name: str
    category: str
    attempt: int
    max_attempts: int
    error: str
    in_cooldown: bool = False
    cooldown_seconds: float | None = None


@dataclass(slots=True)
class ProviderCallNotice:
    provider_name: str
    category: str
    attempt: int
    query: str | None = None
    url: str | None = None
    max_results: int | None = None
    result_count: int | None = None
    elapsed_seconds: float | None = None


@dataclass(slots=True)
class ProviderHooks:
    on_start: Any | None = None
    on_success: Any | None = None
    on_retry: Any | None = None
    on_error: Any | None = None


_provider_hooks_var: contextvars.ContextVar[ProviderHooks | None] = contextvars.ContextVar(
    "open_research_provider_hooks",
    default=None,
)


@contextmanager
def provider_hooks_scope(hooks: ProviderHooks):
    token = _provider_hooks_var.set(hooks)
    try:
        yield
    finally:
        _provider_hooks_var.reset(token)


async def _emit_provider_retry(notice: ProviderRetryNotice) -> None:
    hooks = _provider_hooks_var.get()
    if hooks is None or hooks.on_retry is None:
        return
    await hooks.on_retry(notice)


async def _emit_provider_start(notice: ProviderCallNotice) -> None:
    hooks = _provider_hooks_var.get()
    if hooks is None or hooks.on_start is None:
        return
    await hooks.on_start(notice)


async def _emit_provider_success(notice: ProviderCallNotice) -> None:
    hooks = _provider_hooks_var.get()
    if hooks is None or hooks.on_success is None:
        return
    await hooks.on_success(notice)


async def _emit_provider_error(notice: ProviderRetryNotice) -> None:
    hooks = _provider_hooks_var.get()
    if hooks is None or hooks.on_error is None:
        return
    await hooks.on_error(notice)


class SearchProvider(ABC):
    provider_name: str

    @abstractmethod
    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        raise NotImplementedError


class FetchProvider(ABC):
    provider_name: str

    @abstractmethod
    async def fetch(self, url: str) -> FetchedDocument:
        raise NotImplementedError


class EmbeddingProvider(ABC):
    provider_name: str

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> GenerationResult[list[list[float]]]:
        raise NotImplementedError


class PassageReranker(ABC):
    provider_name: str

    @abstractmethod
    async def rerank(
        self,
        *,
        query: str,
        passages: Sequence[RetrievedPassage],
        top_k: int,
    ) -> list[RetrievedPassage]:
        raise NotImplementedError


class _RetriedProviderMixin:
    category: str

    def __init__(
        self,
        *,
        provider_name: str,
        max_attempts: int,
        base_delay_seconds: float,
        max_delay_seconds: float,
        cooldown_failures: int,
        cooldown_seconds: float,
    ) -> None:
        self.provider_name = provider_name
        self.max_attempts = max(1, max_attempts)
        self.base_delay_seconds = max(0.0, base_delay_seconds)
        self.max_delay_seconds = max(self.base_delay_seconds, max_delay_seconds)
        self.cooldown_failures = max(1, cooldown_failures)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._consecutive_failures = 0
        self._cooldown_until = 0.0

    def _in_cooldown(self) -> float:
        remaining = self._cooldown_until - time.monotonic()
        return remaining if remaining > 0 else 0.0

    async def _before_call(self) -> None:
        remaining = self._in_cooldown()
        if remaining <= 0:
            return
        notice = ProviderRetryNotice(
            provider_name=self.provider_name,
            category=self.category,
            attempt=0,
            max_attempts=self.max_attempts,
            error="provider cooldown active",
            in_cooldown=True,
            cooldown_seconds=round(remaining, 3),
        )
        await _emit_provider_error(notice)
        raise ProviderError(f"{self.provider_name} is in cooldown for {remaining:.2f}s")

    async def _mark_success(self) -> None:
        self._consecutive_failures = 0
        self._cooldown_until = 0.0

    async def _mark_failure(self, *, attempt: int, error: str) -> None:
        if attempt < self.max_attempts:
            await _emit_provider_retry(
                ProviderRetryNotice(
                    provider_name=self.provider_name,
                    category=self.category,
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                    error=error,
                )
            )
            await asyncio.sleep(
                min(self.max_delay_seconds, self.base_delay_seconds * (2 ** max(attempt - 1, 0)))
            )
            return

        self._consecutive_failures += 1
        cooldown_seconds: float | None = None
        if self._consecutive_failures >= self.cooldown_failures:
            self._cooldown_until = time.monotonic() + self.cooldown_seconds
            cooldown_seconds = self.cooldown_seconds
        await _emit_provider_error(
            ProviderRetryNotice(
                provider_name=self.provider_name,
                category=self.category,
                attempt=attempt,
                max_attempts=self.max_attempts,
                error=error,
                in_cooldown=cooldown_seconds is not None,
                cooldown_seconds=cooldown_seconds,
            )
        )


class RetriedSearchProvider(_RetriedProviderMixin, SearchProvider):
    category = "search"

    def __init__(self, provider: SearchProvider, **kwargs: Any) -> None:
        self.provider = provider
        super().__init__(provider_name=provider.provider_name, **kwargs)

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        await self._before_call()
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            await _emit_provider_start(
                ProviderCallNotice(
                    provider_name=self.provider_name,
                    category=self.category,
                    attempt=attempt,
                    query=query,
                    max_results=max_results,
                )
            )
            try:
                results = await self.provider.search(query, max_results=max_results)
            except ProviderError as exc:
                await self._mark_failure(attempt=attempt, error=str(exc))
                if attempt >= self.max_attempts:
                    raise
                continue
            await self._mark_success()
            await _emit_provider_success(
                ProviderCallNotice(
                    provider_name=self.provider_name,
                    category=self.category,
                    attempt=attempt,
                    query=query,
                    max_results=max_results,
                    result_count=len(results),
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
            )
            return results
        raise ProviderError(f"{self.provider_name} exhausted retries")


class RetriedFetchProvider(_RetriedProviderMixin, FetchProvider):
    category = "fetch"

    def __init__(self, provider: FetchProvider, **kwargs: Any) -> None:
        self.provider = provider
        super().__init__(provider_name=provider.provider_name, **kwargs)

    async def fetch(self, url: str) -> FetchedDocument:
        await self._before_call()
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            await _emit_provider_start(
                ProviderCallNotice(
                    provider_name=self.provider_name,
                    category=self.category,
                    attempt=attempt,
                    url=url,
                )
            )
            try:
                document = await self.provider.fetch(url)
            except ProviderError as exc:
                await self._mark_failure(attempt=attempt, error=str(exc))
                if attempt >= self.max_attempts:
                    raise
                continue
            await self._mark_success()
            await _emit_provider_success(
                ProviderCallNotice(
                    provider_name=self.provider_name,
                    category=self.category,
                    attempt=attempt,
                    url=url,
                    result_count=1,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
            )
            return document
        raise ProviderError(f"{self.provider_name} exhausted retries")


class RetriedEmbeddingProvider(_RetriedProviderMixin, EmbeddingProvider):
    category = "embedding"

    def __init__(self, provider: EmbeddingProvider, **kwargs: Any) -> None:
        self.provider = provider
        super().__init__(provider_name=provider.provider_name, **kwargs)

    async def embed_texts(self, texts: Sequence[str]) -> GenerationResult[list[list[float]]]:
        await self._before_call()
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await self.provider.embed_texts(texts)
            except ProviderError as exc:
                await self._mark_failure(attempt=attempt, error=str(exc))
                if attempt >= self.max_attempts:
                    raise
                continue
            await self._mark_success()
            return result
        raise ProviderError(f"{self.provider_name} exhausted retries")


class MockSearchProvider(SearchProvider):
    provider_name = "mock"

    def __init__(self, result_map: Mapping[str, Sequence[SearchResult]] | None = None) -> None:
        self.result_map = {key: list(value) for key, value in (result_map or {}).items()}

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        if query in self.result_map:
            return list(self.result_map[query])[:max_results]
        synthetic_url = f"https://example.com/{slugify(query)}"
        return [
            SearchResult(
                title=f"Synthetic result for {query}",
                url=synthetic_url,
                snippet=f"Synthetic evidence generated locally for query: {query}",
                provider=self.provider_name,
                score=0.5,
            )
        ]


class BraveSearchProvider(SearchProvider):
    provider_name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": query, "count": max_results}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.endpoint, headers=headers, params=params)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Brave search request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"Brave search failed with {response.status_code}: {response.text}")
        payload = response.json()
        items = payload.get("web", {}).get("results", [])
        results: list[SearchResult] = []
        for index, item in enumerate(items):
            url = item.get("url")
            title = item.get("title")
            if not url or not title:
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=item.get("description", "") or "",
                    provider=self.provider_name,
                    score=float(max_results - index),
                )
            )
        return results


class ExaSearchProvider(SearchProvider):
    provider_name = "exa"
    endpoint = "https://api.exa.ai/search"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        payload = {
            "query": query,
            "numResults": max_results,
            "type": "auto",
            "text": True,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Exa search request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"Exa search failed with {response.status_code}: {response.text}")
        items = response.json().get("results", [])
        results: list[SearchResult] = []
        for index, item in enumerate(items):
            url = item.get("url")
            title = item.get("title")
            if not url or not title:
                continue
            snippet = clean_text(item.get("text") or item.get("summary") or "")[:500]
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    provider=self.provider_name,
                    score=float(max_results - index),
                )
            )
        return results


class TavilySearchProvider(SearchProvider):
    provider_name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "query": query,
            "max_results": max_results,
            "include_raw_content": False,
            "include_answer": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Tavily search request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"Tavily search failed with {response.status_code}: {response.text}"
            )
        items = response.json().get("results", [])
        results: list[SearchResult] = []
        for index, item in enumerate(items):
            url = item.get("url")
            title = item.get("title")
            if not url or not title:
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=clean_text(item.get("content") or "")[:500],
                    provider=self.provider_name,
                    score=float(max_results - index),
                )
            )
        return results


_OPENAI_WEB_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"'`*]+")
_OPENAI_SUMMARY_SOURCE_MARKER_RE = re.compile(
    r"(^|\s)[-*]\s+(?:(?:\*\*)?(?:source\s+url|url):(?:\*\*)?\s*)?(?:`|\*\*)?https?://",
    re.IGNORECASE,
)
_OPENAI_SUMMARY_BULLET_MARKER_RE = re.compile(r"(^|\s)[-*]\s+")


class OpenAIWebSearchProvider(SearchProvider):
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        search_context_size: str = "medium",
        reasoning_effort: str = "low",
        external_web_access: bool = True,
        max_output_tokens: int = 4096,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.search_context_size = search_context_size
        self.reasoning_effort = reasoning_effort
        self.external_web_access = external_web_access
        self.max_output_tokens = max(256, max_output_tokens)

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        if max_results <= 0:
            return []
        tool: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": self.search_context_size,
            "external_web_access": self.external_web_access,
        }
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": (
                "You are a web search adapter. Use web search for the query, prefer "
                "primary or official sources, open at most three pages when useful, "
                "then stop searching and write a concise source digest with citations. "
                "For broad technical surveys, diversify results across source types: "
                "include at least one paper or benchmark record, one official project "
                "or documentation page, and one limitations or evaluation source when "
                "available. Avoid returning more than two URLs from the same organization "
                "unless the query is specifically about that organization."
            ),
            "input": (
                "Find current, citable web sources for this research query. Return "
                f"up to {max_results} bullets, each with the source URL and the "
                "specific evidence it supports. Avoid generic summaries.\n\n"
                f"Query: {query}"
            ),
            "tools": [tool],
            "tool_choice": "auto",
            "include": ["web_search_call.action.sources"],
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self._model_supports_reasoning(self.model):
            request_kwargs["reasoning"] = {"effort": self.reasoning_effort}

        try:
            response = await self.client.responses.create(**request_kwargs)
        except Exception as exc:
            raise ProviderError(f"OpenAI web search request failed: {exc}") from exc

        payload = _response_model_dump(response)
        summary = _extract_response_text(response, payload)
        results = _extract_openai_web_search_results(
            payload=payload,
            summary=summary,
            max_results=max_results,
            provider_name=self.provider_name,
        )
        if not results:
            raise ProviderError("OpenAI web search returned no cited URLs.")
        return results

    @staticmethod
    def _model_supports_reasoning(model: str) -> bool:
        lowered = model.lower()
        return lowered.startswith("gpt-5") or lowered.startswith("o")


class SearchPipeline(SearchProvider):
    provider_name = "search-broker"

    def __init__(self, providers: Sequence[SearchProvider]) -> None:
        self.providers = list(providers)

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        aggregated: list[SearchResult] = []
        seen_urls: set[str] = set()
        errors: list[str] = []
        for provider in self._ordered_providers(query):
            try:
                results = await provider.search(query, max_results=max_results)
            except ProviderError as exc:
                errors.append(f"{provider.provider_name}: {exc}")
                continue
            for result in results:
                normalized = normalize_url(str(result.url))
                if normalized in seen_urls:
                    continue
                seen_urls.add(normalized)
                aggregated.append(result)
                if len(aggregated) >= max_results:
                    return aggregated
        if aggregated:
            return aggregated
        error_message = "; ".join(errors) if errors else "No search providers configured"
        raise ProviderError(error_message)

    def _ordered_providers(self, query: str) -> list[SearchProvider]:
        lowered = clean_text(query).lower()
        semantic_priority = any(
            hint in lowered
            for hint in (
                "docs",
                "documentation",
                "reference",
                "api",
                "changelog",
                "release notes",
                "paper",
                "arxiv",
                "blog",
                "guide",
            )
        )
        preferred = (
            ["exa", "brave", "openai", "tavily", "mock"]
            if semantic_priority
            else ["brave", "exa", "openai", "tavily", "mock"]
        )
        by_name = {provider.provider_name: provider for provider in self.providers}
        ordered: list[SearchProvider] = []
        for name in preferred:
            provider = by_name.get(name)
            if provider is not None:
                ordered.append(provider)
        for provider in self.providers:
            if provider not in ordered:
                ordered.append(provider)
        return ordered


class MockFetchProvider(FetchProvider):
    provider_name = "mock"

    def __init__(self, document_map: Mapping[str, FetchedDocument] | None = None) -> None:
        self.document_map = dict(document_map or {})

    async def fetch(self, url: str) -> FetchedDocument:
        normalized = normalize_url(url)
        if normalized in self.document_map:
            return self.document_map[normalized]
        title = (
            normalized.rsplit("/", maxsplit=1)[-1].replace("-", " ").title() or "Synthetic Document"
        )
        content = (
            f"{title}. This is a synthetic fallback document for {normalized}. "
            "It exists so the system can still execute its pipeline in local test mode. "
            "Treat the content as placeholder evidence rather than real web research."
        )
        return FetchedDocument(
            url=normalized,
            canonical_url=normalized,
            title=title,
            content=content,
            source_kind=SourceKind.SYNTHETIC,
            retrieval_method=RetrievalMethod.MOCK,
            metadata={"synthetic": True, "provider": self.provider_name},
        )


class FirecrawlFetchProvider(FetchProvider):
    provider_name = "firecrawl"
    endpoint = "https://api.firecrawl.dev/v1/scrape"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def fetch(self, url: str) -> FetchedDocument:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"url": url, "formats": ["markdown"], "onlyMainContent": True}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Firecrawl fetch request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"Firecrawl fetch failed with {response.status_code}: {response.text}"
            )
        body = response.json()
        if not body.get("success", True):
            raise ProviderError(f"Firecrawl reported failure: {body}")
        data = body.get("data", body)
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        content = clean_text(data.get("markdown") or data.get("content") or "")
        if not content:
            raise ProviderError(f"Firecrawl returned no content for {url}")
        canonical_url = normalize_url(metadata.get("sourceURL") or url)
        title = metadata.get("title") or canonical_url
        return FetchedDocument(
            url=normalize_url(url),
            canonical_url=canonical_url,
            title=title,
            content=content,
            source_kind=_infer_source_kind(canonical_url, metadata),
            retrieval_method=RetrievalMethod.FIRECRAWL,
            metadata={**metadata, "provider": self.provider_name},
        )


class BrowserbaseFetchProvider(FetchProvider):
    provider_name = "browserbase"
    endpoint = "https://api.browserbase.com/v1/fetch"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        use_proxies: bool = False,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.use_proxies = use_proxies

    async def fetch(self, url: str) -> FetchedDocument:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-BB-API-Key": self.api_key,
        }
        payload = {"url": url, "proxies": self.use_proxies}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Browserbase fetch request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"Browserbase fetch failed with {response.status_code}: {response.text}"
            )
        body = response.json()
        content = body.get("content") or ""
        if not content:
            raise ProviderError(f"Browserbase returned no content for {url}")
        if body.get("encoding") == "base64":
            try:
                content = base64.b64decode(content).decode("utf-8", errors="replace")
            except (binascii.Error, ValueError) as exc:
                raise ProviderError(f"Browserbase returned invalid base64 content: {exc}") from exc
        content = clean_text(content)
        canonical_url = normalize_url(url)
        return FetchedDocument(
            url=canonical_url,
            canonical_url=canonical_url,
            title=_infer_title_from_content(content, canonical_url),
            content=content,
            source_kind=_infer_source_kind(canonical_url, {"contentType": body.get("contentType")}),
            retrieval_method=RetrievalMethod.API_NATIVE,
            metadata={
                "statusCode": body.get("statusCode"),
                "headers": body.get("headers", {}),
                "contentType": body.get("contentType"),
                "encoding": body.get("encoding"),
                "provider": self.provider_name,
            },
        )


class BrowserbaseSessionFetchProvider(FetchProvider):
    provider_name = "browserbase-session"
    endpoint = "https://api.browserbase.com/v1/sessions"

    def __init__(
        self,
        api_key: str,
        *,
        project_id: str | None = None,
        timeout: float = 30.0,
        use_proxies: bool = False,
        keep_alive: bool = False,
    ) -> None:
        self.api_key = api_key
        self.project_id = project_id
        self.timeout = timeout
        self.use_proxies = use_proxies
        self.keep_alive = keep_alive

    async def fetch(self, url: str) -> FetchedDocument:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise ProviderError(
                "Playwright is required for Browserbase session fetches. "
                "Install the browser extra to enable this fetcher."
            ) from exc

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-BB-API-Key": self.api_key,
        }
        payload: dict[str, Any] = {
            "keepAlive": self.keep_alive,
            "proxies": self.use_proxies,
        }
        if self.project_id:
            payload["projectId"] = self.project_id

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Browserbase session request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"Browserbase session failed with {response.status_code}: {response.text}"
            )

        session = response.json()
        connect_url = (
            session.get("connectUrl")
            or session.get("connect_url")
            or session.get("wsUrl")
            or session.get("ws_url")
        )
        if not connect_url:
            raise ProviderError("Browserbase session did not return a connect URL.")
        session_id = str(session.get("id") or session.get("sessionId") or "")

        async with async_playwright() as playwright:
            browser = await playwright.chromium.connect_over_cdp(connect_url)
            try:
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = context.pages[0] if context.pages else await context.new_page()
                await page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
                title = await page.title()
                body_text = await page.text_content("body")
                html = await page.content()
                screenshot = await page.screenshot(full_page=True, type="png")
            except Exception as exc:  # pragma: no cover - environment specific
                raise ProviderError(f"Browserbase session fetch failed for {url}: {exc}") from exc
            finally:
                await browser.close()

        content = clean_text(body_text or html)
        if not content:
            raise ProviderError(f"Browserbase session returned no content for {url}")

        artifact_payloads = [
            ArtifactPayload(
                kind="rendered-html",
                extension="html",
                content_type="text/html",
                data=html.encode("utf-8"),
            ),
            ArtifactPayload(
                kind="screenshot",
                extension="png",
                content_type="image/png",
                data=screenshot,
            ),
        ]
        canonical_url = normalize_url(url)
        return FetchedDocument(
            url=canonical_url,
            canonical_url=canonical_url,
            title=title or _infer_title_from_content(content, canonical_url),
            content=content,
            source_kind=_infer_source_kind(canonical_url),
            retrieval_method=RetrievalMethod.API_NATIVE,
            metadata={
                "provider": self.provider_name,
                "browser_session_id": session_id,
                "_artifact_payloads": artifact_payloads,
            },
        )


class PlaywrightFetchProvider(FetchProvider):
    provider_name = "playwright"

    def __init__(self, *, timeout: float = 15.0) -> None:
        self.timeout = timeout

    async def fetch(self, url: str) -> FetchedDocument:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise ProviderError(
                "Playwright is not installed. Install the browser extra to enable this fetcher."
            ) from exc

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
                title = await page.title()
                body_text = await page.text_content("body")
                html = await page.content()
                screenshot = await page.screenshot(full_page=True, type="png")
                content = clean_text(body_text or html)
            except Exception as exc:  # pragma: no cover - environment specific
                raise ProviderError(f"Playwright fetch failed for {url}: {exc}") from exc
            finally:
                await browser.close()

        canonical_url = normalize_url(url)
        if not content:
            raise ProviderError(f"Playwright returned no content for {url}")
        return FetchedDocument(
            url=canonical_url,
            canonical_url=canonical_url,
            title=title or _infer_title_from_content(content, canonical_url),
            content=content,
            source_kind=_infer_source_kind(canonical_url),
            retrieval_method=RetrievalMethod.API_NATIVE,
            metadata={
                "rendered_with": self.provider_name,
                "provider": self.provider_name,
                "_artifact_payloads": [
                    ArtifactPayload(
                        kind="rendered-html",
                        extension="html",
                        content_type="text/html",
                        data=html.encode("utf-8"),
                    ),
                    ArtifactPayload(
                        kind="screenshot",
                        extension="png",
                        content_type="image/png",
                        data=screenshot,
                    ),
                ],
            },
        )


class FetchPipeline(FetchProvider):
    provider_name = "fetch-ladder"

    def __init__(self, providers: Sequence[FetchProvider]) -> None:
        self.providers = list(providers)

    async def fetch(self, url: str) -> FetchedDocument:
        errors: list[str] = []
        for provider in self.providers:
            try:
                document = await provider.fetch(url)
                metadata = dict(document.metadata)
                metadata.setdefault("provider", provider.provider_name)
                return document.model_copy(update={"metadata": metadata})
            except ProviderError as exc:
                errors.append(f"{provider.provider_name}: {exc}")
        error_message = "; ".join(errors) if errors else "No fetch providers configured"
        raise ProviderError(error_message)


class MockEmbeddingProvider(EmbeddingProvider):
    provider_name = "mock-embeddings"

    def __init__(self, *, dimensions: int = 1536) -> None:
        self.dimensions = dimensions

    async def embed_texts(self, texts: Sequence[str]) -> GenerationResult[list[list[float]]]:
        vectors = [self._embed_text(text) for text in texts]
        return GenerationResult(value=vectors, usage=UsageInfo())

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], byteorder="big") % self.dimensions
            vector[slot] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0.0:
            return vector
        return [round(value / norm, 6) for value in vector]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    provider_name = "openai-embeddings"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int | None,
        base_url: str | None = None,
        timeout: float = 60.0,
        estimated_request_cost_usd: float = 0.0,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.dimensions = dimensions
        self.estimated_request_cost_usd = estimated_request_cost_usd

    async def embed_texts(self, texts: Sequence[str]) -> GenerationResult[list[list[float]]]:
        if not texts:
            return GenerationResult(value=[], usage=UsageInfo())
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": list(texts),
            }
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            response = await self.client.embeddings.create(**kwargs)
        except Exception as exc:
            raise ProviderError(f"OpenAI embeddings request failed: {exc}") from exc

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens) or input_tokens)
        vectors = [list(item.embedding) for item in getattr(response, "data", [])]
        return GenerationResult(
            value=vectors,
            usage=UsageInfo(
                input_tokens=input_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=self.estimated_request_cost_usd,
            ),
        )


class OpenAICompatibleEmbeddingProvider(OpenAIEmbeddingProvider):
    provider_name = "openai-compatible-embeddings"


class HeuristicPassageReranker(PassageReranker):
    provider_name = "heuristic-reranker"

    async def rerank(
        self,
        *,
        query: str,
        passages: Sequence[RetrievedPassage],
        top_k: int,
    ) -> list[RetrievedPassage]:
        if not passages:
            return []
        query_tokens = set(tokenize(query))
        rescored: list[RetrievedPassage] = []
        for passage in passages:
            passage_tokens = set(tokenize(passage.text))
            if not passage_tokens:
                rescored.append(passage)
                continue
            overlap = len(query_tokens & passage_tokens) / max(len(query_tokens), 1)
            rerank_score = round((0.7 * passage.score) + (0.3 * overlap), 4)
            rescored.append(passage.model_copy(update={"score": rerank_score}))
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]


class SentenceTransformersReranker(PassageReranker):
    provider_name = "sentence-transformers"

    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    async def rerank(
        self,
        *,
        query: str,
        passages: Sequence[RetrievedPassage],
        top_k: int,
    ) -> list[RetrievedPassage]:
        if not passages:
            return []
        scores = await asyncio.to_thread(
            self._predict_scores,
            query,
            [passage.text for passage in passages],
        )
        reranked = [
            passage.model_copy(update={"score": round(float(score), 4)})
            for passage, score in zip(passages, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]

    def _predict_scores(self, query: str, passages: list[str]) -> list[float]:
        model = self._get_model()
        pairs = [(query, passage) for passage in passages]
        scores = model.predict(pairs)
        return [float(score) for score in scores]

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderError(
                "sentence-transformers is not installed. Install the grounding extra to "
                "enable cross-encoder reranking."
            ) from exc
        self._model = CrossEncoder(self.model_name)
        return self._model


class OpenAIJsonClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        api_style: str = "responses",
        structured_output_mode: str = "parse",
        supports_reasoning_effort: bool = True,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.api_style = api_style
        self.structured_output_mode = structured_output_mode
        self.supports_reasoning_effort = supports_reasoning_effort

    def _completion_messages(self, *, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _reasoning_kwargs(self, reasoning_effort: str) -> dict[str, Any]:
        if not self.supports_reasoning_effort:
            return {}
        return {"reasoning": {"effort": reasoning_effort}}

    def _temperature_kwargs(
        self,
        *,
        model: str,
        reasoning_effort: str,
        temperature: float,
    ) -> dict[str, Any]:
        if model.startswith("gpt-5") and reasoning_effort != "none":
            return {}
        return {"temperature": temperature}

    async def generate_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        reasoning_effort: str = "minimal",
        temperature: float = 0.2,
    ) -> str:
        if self.api_style == "chat_completions":
            response = await self.client.chat.completions.create(
                model=model,
                messages=self._completion_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ),
                **self._temperature_kwargs(
                    model=model,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                ),
                **self._reasoning_kwargs(reasoning_effort),
            )
            return _extract_chat_completion_text(response).strip()
        response = await self.client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            **self._reasoning_kwargs(reasoning_effort),
            **self._temperature_kwargs(
                model=model,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            ),
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()
        return _extract_output_text(response).strip()

    async def generate_json[SchemaT: BaseModel](
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[SchemaT],
        reasoning_effort: str = "minimal",
        temperature: float = 0.2,
    ) -> GenerationResult[SchemaT]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((ProviderError, ValidationError, orjson.JSONDecodeError)),
            reraise=True,
        ):
            with attempt:
                last_error: Exception | None = None
                for strategy in self._json_generation_strategies():
                    try:
                        return await strategy(
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            schema_model=schema_model,
                            reasoning_effort=reasoning_effort,
                            temperature=temperature,
                        )
                    except (
                        TypeError,
                        ProviderError,
                        ValidationError,
                        orjson.JSONDecodeError,
                    ) as exc:
                        last_error = exc
                        continue
                if last_error is not None:
                    raise last_error

        raise ProviderError("OpenAI JSON generation exhausted retries")

    def _json_generation_strategies(self):
        if self.api_style == "responses":
            return [
                self._generate_json_with_responses_parse,
                self._generate_json_fallback,
            ]
        if self.structured_output_mode == "parse":
            return [
                self._generate_json_with_chat_parse,
                self._generate_json_with_chat_json_schema,
                self._generate_json_fallback,
            ]
        if self.structured_output_mode == "json_schema":
            return [
                self._generate_json_with_chat_json_schema,
                self._generate_json_fallback,
            ]
        return [self._generate_json_fallback]

    async def _generate_json_with_responses_parse[SchemaT: BaseModel](
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[SchemaT],
        reasoning_effort: str,
        temperature: float,
    ) -> GenerationResult[SchemaT]:
        response = await self.client.responses.parse(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            **self._reasoning_kwargs(reasoning_effort),
            **self._temperature_kwargs(
                model=model,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            ),
            text_format=schema_model,
            store=False,
        )
        value = _extract_parsed_output(response, schema_model=schema_model)
        return GenerationResult(
            value=value,
            usage=_extract_usage(response, model=model),
        )

    async def _generate_json_with_chat_parse[SchemaT: BaseModel](
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[SchemaT],
        reasoning_effort: str,
        temperature: float,
    ) -> GenerationResult[SchemaT]:
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=self._completion_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ),
            response_format=schema_model,
            **self._temperature_kwargs(
                model=model,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            ),
            **self._reasoning_kwargs(reasoning_effort),
        )
        value = _extract_chat_parsed_output(response, schema_model=schema_model)
        return GenerationResult(
            value=value,
            usage=_extract_usage(response, model=model),
        )

    async def _generate_json_with_chat_json_schema[SchemaT: BaseModel](
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[SchemaT],
        reasoning_effort: str,
        temperature: float,
    ) -> GenerationResult[SchemaT]:
        response = await self.client.chat.completions.create(
            model=model,
            messages=self._completion_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_model.__name__.lower(),
                    "schema": schema_model.model_json_schema(),
                },
            },
            **self._temperature_kwargs(
                model=model,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            ),
            **self._reasoning_kwargs(reasoning_effort),
        )
        text = _extract_chat_completion_text(response)
        payload = _parse_json_payload(text)
        return GenerationResult(
            value=schema_model.model_validate(payload),
            usage=_extract_usage(response, model=model),
        )

    async def _generate_json_fallback[SchemaT: BaseModel](
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[SchemaT],
        reasoning_effort: str,
        temperature: float,
    ) -> GenerationResult[SchemaT]:
        schema_json = json.dumps(schema_model.model_json_schema(), indent=2, sort_keys=True)
        instructions = (
            f"{system_prompt}\n\n"
            "Return only valid JSON. Do not wrap the output in Markdown fences.\n"
            f"The JSON must satisfy this schema:\n{schema_json}"
        )
        if self.api_style == "chat_completions":
            response = await self.client.chat.completions.create(
                model=model,
                messages=self._completion_messages(
                    system_prompt=instructions,
                    user_prompt=user_prompt,
                ),
                **self._temperature_kwargs(
                    model=model,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                ),
                **self._reasoning_kwargs(reasoning_effort),
            )
            text = _extract_chat_completion_text(response)
        else:
            response = await self.client.responses.create(
                model=model,
                instructions=instructions,
                input=user_prompt,
                **self._reasoning_kwargs(reasoning_effort),
                **self._temperature_kwargs(
                    model=model,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                ),
                store=False,
            )
            text = getattr(response, "output_text", None) or _extract_output_text(response)
        payload = _parse_json_payload(text)
        return GenerationResult(
            value=schema_model.model_validate(payload),
            usage=_extract_usage(response, model=model),
        )


MODEL_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.00),
}


def _response_model_dump(response: Any) -> Mapping[str, Any]:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if isinstance(response, Mapping):
        return response
    return {}


def _extract_response_text(response: Any, payload: Mapping[str, Any]) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return clean_text(output_text)
    object_text = _extract_output_text(response)
    if object_text.strip():
        return clean_text(object_text)
    parts: list[str] = []
    for item in _walk_values(payload):
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if item.get("type") in {"output_text", "text"} and isinstance(text, str):
            parts.append(text)
    return clean_text("\n".join(parts))


def _extract_openai_web_search_results(
    *,
    payload: Mapping[str, Any],
    summary: str,
    max_results: int,
    provider_name: str,
) -> list[SearchResult]:
    sources: dict[str, dict[str, str]] = {}

    def add_source(
        raw_url: Any,
        *,
        title: Any = None,
        snippet: Any = None,
        prefer_snippet: bool = False,
    ) -> None:
        normalized = _clean_http_source_url(raw_url)
        if normalized is None:
            return
        record = sources.setdefault(normalized, {"url": normalized})
        if isinstance(title, str) and title.strip() and not record.get("title"):
            record["title"] = clean_text(title)[:200]
        if (
            isinstance(snippet, str)
            and snippet.strip()
            and (prefer_snippet or not record.get("snippet"))
        ):
            sanitized_snippet = sanitize_source_snippet_for_url(
                url=normalized,
                snippet=snippet,
            )
            if sanitized_snippet:
                record["snippet"] = sanitized_snippet[:500]

    for url, snippet in _extract_summary_url_contexts(summary):
        add_source(url, snippet=snippet, prefer_snippet=True)

    for item in _walk_values(payload):
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type == "url_citation":
            add_source(item.get("url"), title=item.get("title"), snippet=summary)
            continue
        if item_type == "url":
            add_source(item.get("url"), title=item.get("title"))
            continue
        if "url" in item and item_type in {
            "search",
            "open_page",
            "find_in_page",
            "source",
            "webpage",
            None,
        }:
            add_source(item.get("url"), title=item.get("title"))

    results: list[SearchResult] = []
    for index, record in enumerate(sources.values()):
        if len(results) >= max_results:
            break
        url = record["url"]
        snippet = record.get("snippet") or sanitize_source_snippet_for_url(
            url=url,
            snippet=summary[:500],
        )
        try:
            results.append(
                SearchResult(
                    title=record.get("title") or _title_from_url(url),
                    url=url,
                    snippet=snippet,
                    provider=provider_name,
                    score=float(max_results - index),
                )
            )
        except ValueError:
            continue
    return results


def _extract_summary_url_contexts(summary: str) -> list[tuple[str, str]]:
    contexts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _OPENAI_WEB_URL_RE.finditer(summary):
        normalized = _clean_http_source_url(match.group(0))
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        contexts.append(
            (normalized, _summary_context_for_match(summary, match.start(), match.end()))
        )
    return contexts


def _summary_context_for_match(summary: str, start: int, end: int) -> str:
    marker_positions = _summary_source_marker_positions(summary)
    context_start = max((position for position in marker_positions if position <= start), default=0)
    context_end = min(
        (position for position in marker_positions if position > start),
        default=min(len(summary), end + 700),
    )
    return clean_text(summary[context_start:context_end])[:500]


def _summary_source_marker_positions(summary: str) -> list[int]:
    positions = {0}
    for marker in _OPENAI_SUMMARY_SOURCE_MARKER_RE.finditer(summary):
        prefix = marker.group(1) or ""
        positions.add(marker.start() + len(prefix))
    for marker in _OPENAI_SUMMARY_BULLET_MARKER_RE.finditer(summary):
        prefix = marker.group(1) or ""
        position = marker.start() + len(prefix)
        if _OPENAI_WEB_URL_RE.search(summary[position : position + 500]):
            positions.add(position)
    return sorted(positions)


def _walk_values(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for child in value:
            yield from _walk_values(child)


def _clean_http_source_url(raw_url: Any) -> str | None:
    if not isinstance(raw_url, str):
        return None
    candidate = raw_url.strip().rstrip(".,;:!?)\\]}\"'`*")
    if not candidate.startswith(("http://", "https://")):
        return None
    parsed = urlparse(candidate)
    if not parsed.netloc:
        return None
    return normalize_url(candidate)


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    path_tail = parsed.path.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if path_tail:
        return path_tail.replace("-", " ").replace("_", " ").title()
    return parsed.netloc or url


def _infer_source_kind(url: str, metadata: Mapping[str, Any] | None = None) -> SourceKind:
    metadata = metadata or {}
    lowered = url.lower()
    if lowered.endswith(".pdf") or metadata.get("contentType") == "application/pdf":
        return SourceKind.PDF
    if "/pricing" in lowered:
        return SourceKind.PRICING
    if "/docs" in lowered or lowered.endswith("/documentation"):
        return SourceKind.DOCS
    if "/blog" in lowered or "/news" in lowered:
        return SourceKind.BLOG
    return SourceKind.WEB


def _infer_title_from_content(content: str, fallback_url: str) -> str:
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    if first_line:
        return clean_text(first_line[:200])
    return fallback_url


def _extract_parsed_output[SchemaT: BaseModel](
    response: ParsedResponse[Any],
    *,
    schema_model: type[SchemaT],
) -> SchemaT:
    if getattr(response, "error", None) is not None:
        raise ProviderError(str(response.error))
    for output in response.output:
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []) or []:
            if getattr(item, "type", None) == "refusal":
                refusal = getattr(item, "refusal", "OpenAI refused to return structured output.")
                raise ProviderError(refusal)
            parsed = getattr(item, "parsed", None)
            if parsed is None:
                continue
            if isinstance(parsed, schema_model):
                return parsed
            return schema_model.model_validate(parsed)

    text = getattr(response, "output_text", None) or _extract_output_text(response)
    if text:
        payload = _parse_json_payload(text)
        return schema_model.model_validate(payload)
    raise ProviderError("OpenAI returned no parseable structured output.")


def _extract_output_text(response: Any) -> str:
    output_items = getattr(response, "output", None) or []
    text_parts: list[str] = []
    for item in output_items:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                text_parts.append(getattr(content, "text", ""))
    return "\n".join(part for part in text_parts if part)


def _extract_chat_completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ProviderError("OpenAI-compatible server returned no completion choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ProviderError("OpenAI-compatible server returned an empty message.")
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ProviderError(str(refusal))
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"output_text", "text"}:
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    raise ProviderError("OpenAI-compatible server returned no text content.")


def _extract_chat_parsed_output[SchemaT: BaseModel](
    response: Any,
    *,
    schema_model: type[SchemaT],
) -> SchemaT:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ProviderError("OpenAI-compatible server returned no completion choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ProviderError("OpenAI-compatible server returned an empty message.")
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ProviderError(str(refusal))
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, schema_model):
            return parsed
        return schema_model.model_validate(parsed)
    text = _extract_chat_completion_text(response)
    payload = _parse_json_payload(text)
    return schema_model.model_validate(payload)


def _parse_json_payload(text: str) -> Any:
    candidate = strip_markdown_fences(text)
    try:
        return orjson.loads(candidate)
    except orjson.JSONDecodeError:
        start = min(
            [index for index in [candidate.find("{"), candidate.find("[")] if index >= 0],
            default=-1,
        )
        if start < 0:
            raise
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if end < 0:
            raise
        return orjson.loads(candidate[start : end + 1])


def _extract_usage(response: Any, *, model: str) -> UsageInfo:
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageInfo()
    output_details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)
    input_tokens = int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0)
    output_tokens = int(
        getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0
    )
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
    return UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=_estimate_openai_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _estimate_openai_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    pricing = _resolve_model_pricing(model)
    if pricing is None:
        return 0.0
    input_rate, output_rate = pricing
    estimated = ((input_tokens / 1_000_000) * input_rate) + (
        (output_tokens / 1_000_000) * output_rate
    )
    return round(estimated, 6)


def _resolve_model_pricing(model: str) -> tuple[float, float] | None:
    if model in MODEL_PRICING_USD_PER_1M:
        return MODEL_PRICING_USD_PER_1M[model]
    for prefix, pricing in MODEL_PRICING_USD_PER_1M.items():
        if model.startswith(f"{prefix}-"):
            return pricing
    return None
