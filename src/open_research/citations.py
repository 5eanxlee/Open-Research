from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlparse
from uuid import uuid4

from .domain import (
    CitationAuditDecision,
    CitationAuditReason,
    CitationAuditRecord,
    CitationMatchStrategy,
    CitationRecord,
    SourceRegistryEntry,
)
from .utils import normalize_url, slugify

SHORTENER_HOSTS = {
    "bit.ly",
    "t.co",
    "tinyurl.com",
    "goo.gl",
    "ow.ly",
    "buff.ly",
    "s.id",
    "shorturl.at",
    "tiny.cc",
}


@dataclass(slots=True)
class CitationCandidate:
    section_title: str
    ordinal: int
    claim: str
    citation: CitationRecord | None


@dataclass(slots=True)
class CitationAuditResult:
    kept: list[CitationCandidate]
    removed: list[CitationCandidate]
    audits: list[CitationAuditRecord]


def build_citation_key(title: str | None, url: str) -> str:
    host = urlparse(url).netloc.lower()
    title_slug = slugify(title or host or url)[:80]
    host_slug = slugify(host)[:40]
    return f"{host_slug}-{title_slug}".strip("-") or slugify(url)


def audit_citation_candidates(
    *,
    run_id: str,
    candidates: list[CitationCandidate],
    registry_entries: list[SourceRegistryEntry],
) -> CitationAuditResult:
    kept: list[CitationCandidate] = []
    removed: list[CitationCandidate] = []
    audits: list[CitationAuditRecord] = []
    for candidate in candidates:
        audit = audit_single_citation(
            run_id=run_id,
            section_title=candidate.section_title,
            ordinal=candidate.ordinal,
            claim=candidate.claim,
            citation=candidate.citation,
            registry_entries=registry_entries,
        )
        audits.append(audit)
        if audit.decision == CitationAuditDecision.KEPT:
            kept.append(candidate)
        else:
            removed.append(candidate)
    return CitationAuditResult(kept=kept, removed=removed, audits=audits)


def audit_single_citation(
    *,
    run_id: str,
    section_title: str,
    ordinal: int,
    claim: str,
    citation: CitationRecord | None,
    registry_entries: list[SourceRegistryEntry],
) -> CitationAuditRecord:
    reasons: list[CitationAuditReason] = []
    matched_strategy: CitationMatchStrategy | None = None
    matched_entry: SourceRegistryEntry | None = None
    normalized_url: str | None = None
    citation_key: str | None = None
    source_id: str | None = None
    raw_url: str | None = None

    if citation is None:
        reasons.append(CitationAuditReason.UNVERIFIABLE)
    else:
        source_id = citation.source_id
        raw_url = str(citation.source_url)
        citation_key = citation.citation_key or build_citation_key(citation.source_title, raw_url)
        if is_unsafe_url(raw_url):
            reasons.append(CitationAuditReason.UNSAFE_URL)
        if is_truncated_url(raw_url):
            reasons.append(CitationAuditReason.TRUNCATED_URL)
        if is_shortened_url(raw_url):
            reasons.append(CitationAuditReason.SHORTENED_URL)
        if is_ip_address_url(raw_url):
            reasons.append(CitationAuditReason.IP_ADDRESS_URL)
        normalized_url = normalize_citation_url(raw_url)
        matched_entry, matched_strategy = match_registry_entry(
            raw_url=raw_url,
            normalized_url=normalized_url,
            citation_key=citation_key,
            registry_entries=registry_entries,
        )
        if matched_entry is None:
            reasons.append(CitationAuditReason.URL_NOT_IN_REGISTRY)
        if citation_key is not None and not registry_contains_citation_key(
            citation_key,
            registry_entries,
        ):
            reasons.append(CitationAuditReason.CITATION_KEY_NOT_IN_REGISTRY)
        if normalized_url is None:
            reasons.append(CitationAuditReason.UNVERIFIABLE)

    decision = CitationAuditDecision.KEPT if not reasons else CitationAuditDecision.REMOVED
    return CitationAuditRecord(
        id=str(uuid4()),
        run_id=run_id,
        section_title=section_title,
        ordinal=ordinal,
        claim=claim,
        decision=decision,
        reasons=reasons,
        source_id=source_id,
        citation_key=citation_key,
        source_url=raw_url,
        normalized_url=normalized_url,
        matched_strategy=matched_strategy,
        metadata={
            "matched_registry_entry_id": matched_entry.id if matched_entry is not None else None,
        },
    )


def normalize_citation_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return normalize_url(url)
    except Exception:
        return None


def is_truncated_url(url: str) -> bool:
    return "..." in url or "…" in url or any(char.isspace() for char in url)


def is_unsafe_url(url: str) -> bool:
    scheme = urlparse(url).scheme.lower()
    return scheme not in {"http", "https"}


def is_shortened_url(url: str) -> bool:
    return urlparse(url).netloc.lower() in SHORTENER_HOSTS


def is_ip_address_url(url: str) -> bool:
    host = urlparse(url).hostname
    if host is None:
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def registry_contains_citation_key(
    citation_key: str,
    registry_entries: list[SourceRegistryEntry],
) -> bool:
    return any(entry.citation_key == citation_key for entry in registry_entries)


def match_registry_entry(
    *,
    raw_url: str,
    normalized_url: str | None,
    citation_key: str | None,
    registry_entries: list[SourceRegistryEntry],
) -> tuple[SourceRegistryEntry | None, CitationMatchStrategy | None]:
    if normalized_url is None:
        return None, None

    exact = next(
        (entry for entry in registry_entries if entry.normalized_url == normalized_url),
        None,
    )
    if exact is not None:
        return exact, CitationMatchStrategy.EXACT

    if is_truncated_url(raw_url):
        prefix = raw_url.replace("...", "").replace("…", "")
        match = next(
            (entry for entry in registry_entries if entry.normalized_url.startswith(prefix)),
            None,
        )
        if match is not None:
            return match, CitationMatchStrategy.TRUNCATION

    match = next(
        (
            entry
            for entry in registry_entries
            if _same_origin_and_child_path(normalized_url, entry.normalized_url)
        ),
        None,
    )
    if match is not None:
        return match, CitationMatchStrategy.CHILD_PATH

    match = next(
        (
            entry
            for entry in registry_entries
            if _query_subset_match(normalized_url, entry.normalized_url)
        ),
        None,
    )
    if match is not None:
        return match, CitationMatchStrategy.QUERY_SUBSET

    match = next(
        (
            entry
            for entry in registry_entries
            if entry.normalized_url.startswith(normalized_url)
            or normalized_url.startswith(entry.normalized_url)
        ),
        None,
    )
    if match is not None:
        return match, CitationMatchStrategy.PREFIX

    match = next(
        (
            entry
            for entry in registry_entries
            if citation_key is not None and entry.citation_key == citation_key
        ),
        None,
    )
    if match is not None:
        return match, CitationMatchStrategy.PREFIX

    return None, None


def _same_origin_and_child_path(left: str, right: str) -> bool:
    left_parts = urlparse(left)
    right_parts = urlparse(right)
    if (
        left_parts.scheme != right_parts.scheme
        or left_parts.netloc != right_parts.netloc
        or left_parts.path == right_parts.path
    ):
        return False
    return left_parts.path.startswith(
        right_parts.path.rstrip("/") + "/"
    ) or right_parts.path.startswith(
        left_parts.path.rstrip("/") + "/"
    )


def _query_subset_match(left: str, right: str) -> bool:
    left_parts = urlparse(left)
    right_parts = urlparse(right)
    if (
        left_parts.scheme != right_parts.scheme
        or left_parts.netloc != right_parts.netloc
        or left_parts.path != right_parts.path
    ):
        return False
    left_query = set(parse_qsl(left_parts.query, keep_blank_values=True))
    right_query = set(parse_qsl(right_parts.query, keep_blank_values=True))
    return bool(left_query) and left_query.issubset(right_query)
