from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

import httpx

from .normalization import truncate_search_query

SERPER_SCHOLAR_URL = "https://google.serper.dev/scholar"


class PaperSearchTool:
    name = "paper_search_tool"
    description = (
        "Searches scholarly papers through a Serper Google Scholar compatible endpoint. "
        "Use this first for scientific, academic, technical, or empirical questions."
    )

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        max_results: int = 5,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_results = max(1, min(max_results, 50))

    async def search(self, query: str, year: str | int | None = None) -> str:
        if not query.strip():
            return "Error: 'query' argument is required"
        bounded_query = truncate_search_query(query)
        try:
            records = await self._search_serper(bounded_query, year=year)
        except TimeoutError:
            return f"Paper search timed out after {self.timeout}s for query: {bounded_query}"
        except Exception as exc:
            return f"Paper search failed: {exc}"
        return format_paper_results(records)

    async def __call__(self, query: str, year: str | int | None = None) -> str:
        return await self.search(query, year=year)

    async def _search_serper(
        self,
        query: str,
        *,
        year: str | int | None = None,
    ) -> list[dict[str, Any]]:
        start_year, end_year = _parse_year_range(year)
        page_size = 10
        page_count = math.ceil(self.max_results / page_size)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._fetch_page(
                    client=client,
                    query=query,
                    num=min(page_size, self.max_results - (page * page_size)),
                    offset=page * page_size,
                    start_year=start_year,
                    end_year=end_year,
                )
                for page in range(page_count)
                if self.max_results - (page * page_size) > 0
            ]
            pages = await asyncio.gather(*tasks)
        records: list[dict[str, Any]] = []
        for page in pages:
            organic = page.get("organic") or []
            if isinstance(organic, list):
                records.extend(record for record in organic if isinstance(record, dict))
        return records[: self.max_results]

    async def _fetch_page(
        self,
        *,
        client: httpx.AsyncClient,
        query: str,
        num: int,
        offset: int,
        start_year: str | None,
        end_year: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "q": query,
            "num": min(num, 20),
            "start": offset,
        }
        if start_year:
            payload["as_ylo"] = start_year
        if end_year:
            payload["as_yhi"] = end_year
        response = await client.post(
            SERPER_SCHOLAR_URL,
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Serper API error: {response.status_code} - {response.text}")
        return response.json()


def format_paper_results(results: Sequence[dict[str, Any]]) -> str:
    if not results:
        return "No papers found via Google Scholar."
    formatted: list[str] = []
    for index, paper in enumerate(results, start=1):
        title = str(paper.get("title") or "Unknown Title").strip()
        year = str(paper.get("year") or "Unknown Year").strip()
        snippet = str(paper.get("snippet") or "").strip()
        link = str(paper.get("link") or "").strip()
        publication = str(paper.get("publicationInfo") or "").strip()
        citations = paper.get("citedBy", 0)
        formatted.append(
            "\n".join(
                [
                    f"{index}. **{title}** ({year})",
                    f"   - **Publication**: {publication}",
                    f"   - **Citations**: {citations}",
                    f"   - **Snippet**: {snippet}",
                    f"   - **Link**: {link}",
                ]
            )
        )
    return "\n\n".join(formatted)


def _parse_year_range(year: str | int | None) -> tuple[str | None, str | None]:
    if year is None:
        return None, None
    value = str(year).strip()
    if not value:
        return None, None
    if "-" not in value:
        return value, value
    start, end = (part.strip() for part in value.split("-", maxsplit=1))
    return start or None, end or None
