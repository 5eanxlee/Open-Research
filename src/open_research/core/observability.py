from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .config import Settings

try:  # pragma: no cover - optional dependency
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
except ImportError:  # pragma: no cover - optional dependency
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    CollectorRegistry = Counter = Gauge = Histogram = None

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except ImportError:  # pragma: no cover - optional dependency
    trace = None
    OTLPSpanExporter = Resource = TracerProvider = BatchSpanProcessor = None


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_payload = getattr(record, "payload", None)
        if isinstance(extra_payload, dict):
            payload.update(extra_payload)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


class TraceRedactor:
    def __init__(self, *, sensitive_fields: set[str]) -> None:
        self.sensitive_fields = {field.lower() for field in sensitive_fields if field}
        self._secret_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"sk-[a-z0-9]{10,}",
                r"bearer\s+[a-z0-9._-]{10,}",
                r"api[_-]?key",
                r"authorization",
                r"set-cookie",
                r"cookie",
                r"password",
                r"secret",
                r"token",
            )
        ]

    def redact_value(self, value: Any, *, key: str | None = None) -> Any:
        if key is not None and key.lower() in self.sensitive_fields:
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {
                nested_key: self.redact_value(nested_value, key=str(nested_key))
                for nested_key, nested_value in value.items()
            }
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, tuple):
            return [self.redact_value(item) for item in value]
        if isinstance(value, str):
            redacted = value
            for pattern in self._secret_patterns:
                redacted = pattern.sub("[REDACTED]", redacted)
            return redacted
        return value


@dataclass(slots=True)
class MetricHandles:
    run_completion_total: Any | None = None
    cancellation_latency_seconds: Any | None = None
    replay_lag_events: Any | None = None
    worker_heartbeat_overdue: Any | None = None
    provider_errors_total: Any | None = None
    provider_retries_total: Any | None = None
    fetch_latency_seconds: Any | None = None
    unsupported_claims_total: Any | None = None
    citation_removals_total: Any | None = None
    run_cost_usd: Any | None = None


class ResearchTelemetry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        sensitive_fields = {
            field.strip().lower()
            for field in settings.trace_redaction_fields.split(",")
            if field.strip()
        }
        self.redactor = TraceRedactor(sensitive_fields=sensitive_fields)
        self.logger = logging.getLogger("open_research")
        self._configure_logging()
        self.registry = (
            CollectorRegistry() if settings.metrics_enabled and CollectorRegistry else None
        )
        self.metrics = self._build_metrics()
        self._tracer = self._build_tracer()

    def _configure_logging(self) -> None:
        if any(isinstance(handler.formatter, JsonLogFormatter) for handler in self.logger.handlers):
            return
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        self.logger.handlers = [handler]
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def _build_metrics(self) -> MetricHandles:
        if self.registry is None or Counter is None or Gauge is None or Histogram is None:
            return MetricHandles()
        return MetricHandles(
            run_completion_total=Counter(
                "open_research_run_completion_total",
                "Runs reaching terminal state by status.",
                labelnames=("status",),
                registry=self.registry,
            ),
            cancellation_latency_seconds=Histogram(
                "open_research_cancellation_latency_seconds",
                "Observed cancellation latency in seconds.",
                registry=self.registry,
            ),
            replay_lag_events=Gauge(
                "open_research_replay_lag_events",
                "Replay lag between requested cursor and latest event id.",
                registry=self.registry,
            ),
            worker_heartbeat_overdue=Gauge(
                "open_research_worker_heartbeat_overdue",
                "Count of runs whose heartbeat is overdue.",
                registry=self.registry,
            ),
            provider_errors_total=Counter(
                "open_research_provider_errors_total",
                "Provider errors by category and provider.",
                labelnames=("category", "provider"),
                registry=self.registry,
            ),
            provider_retries_total=Counter(
                "open_research_provider_retries_total",
                "Provider retries by category and provider.",
                labelnames=("category", "provider"),
                registry=self.registry,
            ),
            fetch_latency_seconds=Histogram(
                "open_research_fetch_latency_seconds",
                "Source fetch latency by provider.",
                labelnames=("provider",),
                registry=self.registry,
            ),
            unsupported_claims_total=Counter(
                "open_research_unsupported_claims_total",
                "Unsupported claims emitted by grounding.",
                registry=self.registry,
            ),
            citation_removals_total=Counter(
                "open_research_citation_removals_total",
                "Removed citations by reason.",
                labelnames=("reason",),
                registry=self.registry,
            ),
            run_cost_usd=Histogram(
                "open_research_run_cost_usd",
                "Estimated cost per completed run.",
                registry=self.registry,
            ),
        )

    def _build_tracer(self):
        if (
            trace is None
            or TracerProvider is None
            or self.settings.otlp_endpoint is None
            or Resource is None
            or OTLPSpanExporter is None
            or BatchSpanProcessor is None
        ):
            return None
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": "open-research",
                    "service.version": "0.1.0",
                    "deployment.environment": self.settings.environment,
                }
            )
        )
        exporter = OTLPSpanExporter(
            endpoint=self.settings.otlp_endpoint,
            headers=self._otlp_headers(),
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return trace.get_tracer("open_research")

    def _otlp_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.settings.otlp_headers:
            for item in self.settings.otlp_headers.split(","):
                if "=" not in item:
                    continue
                key, value = item.split("=", maxsplit=1)
                headers[key.strip()] = value.strip()
        if (
            self.settings.langfuse_public_key is not None
            and self.settings.langfuse_secret_key is not None
        ):
            headers.setdefault(
                "x-langfuse-public-key",
                self.settings.langfuse_public_key.get_secret_value(),
            )
            headers.setdefault(
                "x-langfuse-secret-key",
                self.settings.langfuse_secret_key.get_secret_value(),
            )
        return headers

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        if self._tracer is None:
            yield
            return
        sanitized = self.redactor.redact_value(attributes)
        with self._tracer.start_as_current_span(name) as span:
            for key, value in sanitized.items():
                span.set_attribute(key, value)
            yield

    def log(self, message: str, **payload: Any) -> None:
        sanitized = self.redactor.redact_value(payload)
        self.logger.info(message, extra={"payload": sanitized})

    def render_metrics(self) -> tuple[str, bytes]:
        if self.registry is None or generate_latest is None:
            return CONTENT_TYPE_LATEST, b""
        return CONTENT_TYPE_LATEST, generate_latest(self.registry)

    def record_run_terminal(self, *, status: str, cost_usd: float) -> None:
        if self.metrics.run_completion_total is not None:
            self.metrics.run_completion_total.labels(status=status).inc()
        if self.metrics.run_cost_usd is not None:
            self.metrics.run_cost_usd.observe(max(cost_usd, 0.0))

    def record_cancellation_latency(self, seconds: float) -> None:
        if self.metrics.cancellation_latency_seconds is not None:
            self.metrics.cancellation_latency_seconds.observe(max(seconds, 0.0))

    def record_replay_lag(self, lag: int) -> None:
        if self.metrics.replay_lag_events is not None:
            self.metrics.replay_lag_events.set(max(lag, 0))

    def record_heartbeat_overdue(self, count: int) -> None:
        if self.metrics.worker_heartbeat_overdue is not None:
            self.metrics.worker_heartbeat_overdue.set(max(count, 0))

    def record_provider_error(self, *, category: str, provider: str) -> None:
        if self.metrics.provider_errors_total is not None:
            self.metrics.provider_errors_total.labels(category=category, provider=provider).inc()

    def record_provider_retry(self, *, category: str, provider: str) -> None:
        if self.metrics.provider_retries_total is not None:
            self.metrics.provider_retries_total.labels(category=category, provider=provider).inc()

    def record_fetch_latency(self, *, provider: str, seconds: float) -> None:
        if self.metrics.fetch_latency_seconds is not None:
            self.metrics.fetch_latency_seconds.labels(provider=provider).observe(max(seconds, 0.0))

    def record_unsupported_claim(self, count: int = 1) -> None:
        if self.metrics.unsupported_claims_total is not None:
            self.metrics.unsupported_claims_total.inc(max(count, 0))

    def record_citation_removed(self, *, reason: str, count: int = 1) -> None:
        if self.metrics.citation_removals_total is not None:
            self.metrics.citation_removals_total.labels(reason=reason).inc(max(count, 0))


@contextmanager
def observe_latency(callback, *args, **kwargs) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        callback(*args, seconds=time.perf_counter() - started, **kwargs)
