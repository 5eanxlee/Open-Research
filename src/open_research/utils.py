from __future__ import annotations

import math
import re
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if not key.lower().startswith("utm_")
    ]
    query = urlencode(sorted(query_pairs))
    return urlunsplit((scheme, netloc, path.rstrip("/") or "/", query, ""))


def domain_for_url(url: str) -> str:
    return urlsplit(url).netloc.lower()


def extract_sentences(text: str, *, max_sentences: int | None = None) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", clean_text(text))
    sentences = [piece.strip() for piece in pieces if len(piece.strip()) > 20]
    if max_sentences is not None:
        return sentences[:max_sentences]
    return sentences


def chunk_text(
    text: str,
    *,
    max_chars: int = 700,
    overlap_sentences: int = 1,
) -> list[dict[str, int | str]]:
    sentences = extract_sentences(text)
    if not sentences:
        fallback = clean_text(text)
        if not fallback:
            return []
        return [
            {
                "text": fallback[:max_chars],
                "start_offset": 0,
                "end_offset": min(len(fallback), max_chars),
            }
        ]

    chunks: list[dict[str, int | str]] = []
    buffer: list[str] = []
    start_offset = 0
    current_offset = 0

    for sentence in sentences:
        next_size = len(" ".join([*buffer, sentence]))
        if buffer and next_size > max_chars:
            chunk_text_value = " ".join(buffer).strip()
            chunks.append(
                {
                    "text": chunk_text_value,
                    "start_offset": start_offset,
                    "end_offset": start_offset + len(chunk_text_value),
                }
            )
            carry = buffer[-overlap_sentences:] if overlap_sentences else []
            buffer = list(carry)
            start_offset = max(current_offset - len(" ".join(buffer)), 0)
        if not buffer:
            start_offset = current_offset
        buffer.append(sentence)
        current_offset += len(sentence) + 1

    if buffer:
        chunk_text_value = " ".join(buffer).strip()
        chunks.append(
            {
                "text": chunk_text_value,
                "start_offset": start_offset,
                "end_offset": start_offset + len(chunk_text_value),
            }
        )

    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def cosine_similarity(
    left: list[float] | tuple[float, ...],
    right: list[float] | tuple[float, ...],
) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
