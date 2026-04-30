from __future__ import annotations

from collections.abc import Sequence
from html import escape

from open_research.domain import SearchResult
from open_research.providers import ProviderError, SearchProvider

from .normalization import truncate_search_query


class AdvancedWebSearchTool:
    name = "advanced_web_search_tool"
    description = (
        "Retrieves relevant public web contexts for a query. The query is truncated to "
        "400 characters and results are returned as normalized document blocks."
    )

    def __init__(self, provider: SearchProvider, *, max_results: int = 2) -> None:
        self.provider = provider
        self.max_results = max(1, max_results)

    async def search(self, query: str) -> str:
        if not query.strip():
            return "Error: 'query' argument is required"
        bounded_query = truncate_search_query(query)
        try:
            results = await self.provider.search(bounded_query, max_results=self.max_results)
        except ProviderError as exc:
            return f"Search failed: {exc}"
        return format_web_search_results(results)

    async def __call__(self, query: str) -> str:
        return await self.search(query)


def format_web_search_results(
    results: Sequence[SearchResult],
    *,
    answer: str | None = None,
) -> str:
    blocks: list[str] = []
    if answer:
        blocks.append(f"<Answer>\n{escape(answer.strip())}\n</Answer>")
    for result in results:
        snippet = result.snippet.strip() or "No snippet returned."
        blocks.append(
            "\n".join(
                [
                    f'<Document href="{escape(str(result.url), quote=True)}">',
                    "<title>",
                    escape(result.title.strip() or str(result.url)),
                    "</title>",
                    escape(snippet),
                    "</Document>",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks) if blocks else "Search returned no results"
