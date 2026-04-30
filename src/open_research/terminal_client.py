from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .domain import (
    AgentConfig,
    ArtifactRecord,
    AsyncJob,
    BehaviorAssessment,
    BudgetPolicy,
    CitationAuditRecord,
    ClarificationResponseRequest,
    ClarificationSession,
    ContextPack,
    CreateRunRequest,
    ExecutionMode,
    FinalReport,
    MemoryInfluencePolicy,
    ModelConfigOverride,
    PassageInspectionRecord,
    PlanApprovalRequest,
    PlanPreview,
    ProfileFeedback,
    ProfilePreferences,
    ProfileRecord,
    PublicRuntimeConfig,
    RunDetail,
    RunEvent,
    RunNoteRecord,
    RunStatus,
    RunSummary,
)


class TerminalClientError(RuntimeError):
    pass


@dataclass(slots=True)
class StreamEnvelope:
    id: int | None
    run_id: str | None
    event_type: str
    payload: dict[str, Any]
    created_at: str

    @property
    def is_terminal(self) -> bool:
        return self.event_type in {"report.completed", "run.failed", "run.cancelled"}


def build_request(
    *,
    question: str,
    budget: BudgetPolicy,
    agent_config: AgentConfig,
    profile_id: str = "default",
    memory_policy_override: MemoryInfluencePolicy | None = None,
    execution_mode: ExecutionMode = ExecutionMode.STANDARD,
    require_plan_approval: bool | None = False,
    source_selection: list[str] | None = None,
    model_config_override: ModelConfigOverride | None = None,
    metadata: dict[str, Any] | None = None,
) -> CreateRunRequest:
    return CreateRunRequest(
        question=question,
        budget=budget,
        agent_config=agent_config,
        profile_id=profile_id,
        memory_policy_override=memory_policy_override,
        execution_mode=execution_mode,
        require_plan_approval=require_plan_approval,
        source_selection=source_selection,
        model_config_override=model_config_override,
        metadata=metadata or {},
    )


def parse_sse_lines(lines: list[str]) -> list[StreamEnvelope]:
    events: list[StreamEnvelope] = []
    event_id: int | None = None
    event_type = "message"
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_id, event_type, data_lines
        if not data_lines:
            event_id = None
            event_type = "message"
            return
        payload = json.loads("\n".join(data_lines))
        events.append(
            StreamEnvelope(
                id=payload.get("id", event_id),
                run_id=payload.get("run_id"),
                event_type=payload.get("event_type", event_type),
                payload=payload.get("payload", {}),
                created_at=payload.get("created_at", ""),
            )
        )
        event_id = None
        event_type = "message"
        data_lines = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("id:"):
            raw_id = line.partition(":")[2].strip()
            try:
                event_id = int(raw_id)
            except ValueError:
                event_id = None
            continue
        if line.startswith("event:"):
            event_type = line.partition(":")[2].strip() or event_type
            continue
        if line.startswith("data:"):
            data_lines.append(line.partition(":")[2].strip())
    flush()
    return events


class ResearchTerminalClient:
    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owned_client = client is None
        self.client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)
        self.poll_interval_seconds = poll_interval_seconds

    async def aclose(self) -> None:
        if self._owned_client:
            await self.client.aclose()

    async def public_config(self) -> PublicRuntimeConfig:
        return await self._request_json("GET", "/config/public", model=PublicRuntimeConfig)

    async def list_runs(
        self,
        *,
        limit: int = 50,
        status: RunStatus | None = None,
    ) -> list[RunSummary]:
        params: dict[str, str | int] = {"limit": limit}
        if status is not None:
            params["status"] = status.value
        payload = await self._request_json("GET", "/runs", params=params, model=None)
        return [RunSummary.model_validate(item) for item in payload]

    async def get_run_detail(self, run_id: str) -> RunDetail:
        return await self._request_json("GET", f"/runs/{run_id}", model=RunDetail)

    async def get_report(self, run_id: str) -> FinalReport:
        return await self._request_json("GET", f"/runs/{run_id}/report", model=FinalReport)

    async def get_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        payload = await self._request_json("GET", f"/runs/{run_id}/artifacts", model=None)
        return [ArtifactRecord.model_validate(item) for item in payload]

    async def get_audit(self, run_id: str) -> list[CitationAuditRecord]:
        payload = await self._request_json("GET", f"/runs/{run_id}/audit", model=None)
        return [CitationAuditRecord.model_validate(item) for item in payload]

    async def get_notes(self, run_id: str) -> list[RunNoteRecord]:
        payload = await self._request_json("GET", f"/runs/{run_id}/notes", model=None)
        return [RunNoteRecord.model_validate(item) for item in payload]

    async def get_passages(self, run_id: str) -> list[PassageInspectionRecord]:
        payload = await self._request_json("GET", f"/runs/{run_id}/passages", model=None)
        return [PassageInspectionRecord.model_validate(item) for item in payload]

    async def list_events(self, run_id: str, *, after_id: int = 0) -> list[RunEvent]:
        payload = await self._request_json(
            "GET",
            f"/runs/{run_id}/events",
            params={"after_id": after_id},
            model=None,
        )
        return [RunEvent.model_validate(item) for item in payload]

    async def create_run(
        self,
        *,
        question: str,
        budget: BudgetPolicy,
        agent_config: AgentConfig,
        profile_id: str = "default",
        memory_policy_override: MemoryInfluencePolicy | None = None,
        execution_mode: ExecutionMode = ExecutionMode.STANDARD,
        require_plan_approval: bool | None = False,
        source_selection: list[str] | None = None,
        model_config_override: ModelConfigOverride | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunSummary:
        request = build_request(
            question=question,
            budget=budget,
            agent_config=agent_config,
            profile_id=profile_id,
            memory_policy_override=memory_policy_override,
            execution_mode=execution_mode,
            require_plan_approval=require_plan_approval,
            source_selection=source_selection,
            model_config_override=model_config_override,
            metadata=metadata,
        )
        return await self._request_json(
            "POST",
            "/runs",
            json=request.model_dump(mode="json"),
            model=RunSummary,
        )

    async def create_job(
        self,
        *,
        request: CreateRunRequest,
    ) -> AsyncJob:
        return await self._request_json(
            "POST",
            "/jobs",
            json=request.model_dump(mode="json"),
            model=AsyncJob,
        )

    async def get_job(self, job_id: str) -> AsyncJob:
        return await self._request_json("GET", f"/jobs/{job_id}", model=AsyncJob)

    async def get_clarification(self, run_id: str) -> ClarificationSession:
        return await self._request_json(
            "GET",
            f"/runs/{run_id}/clarification",
            model=ClarificationSession,
        )

    async def answer_clarification(self, run_id: str, response: str) -> RunDetail:
        payload = ClarificationResponseRequest(response=response)
        return await self._request_json(
            "POST",
            f"/runs/{run_id}/clarification/respond",
            json=payload.model_dump(mode="json"),
            model=RunDetail,
        )

    async def get_plan_preview(self, run_id: str) -> PlanPreview:
        return await self._request_json(
            "GET",
            f"/runs/{run_id}/plan-preview",
            model=PlanPreview,
        )

    async def approve_plan(self, run_id: str, note: str | None = None) -> RunDetail:
        payload = PlanApprovalRequest(note=note)
        return await self._request_json(
            "POST",
            f"/runs/{run_id}/plan-preview/approve",
            json=payload.model_dump(mode="json"),
            model=RunDetail,
        )

    async def reject_plan(self, run_id: str, note: str | None = None) -> RunDetail:
        payload = PlanApprovalRequest(note=note)
        return await self._request_json(
            "POST",
            f"/runs/{run_id}/plan-preview/reject",
            json=payload.model_dump(mode="json"),
            model=RunDetail,
        )

    async def request_plan_changes(self, run_id: str, note: str | None = None) -> RunDetail:
        payload = PlanApprovalRequest(note=note)
        return await self._request_json(
            "POST",
            f"/runs/{run_id}/plan-preview/request-changes",
            json=payload.model_dump(mode="json"),
            model=RunDetail,
        )

    async def cancel_run(self, run_id: str) -> RunDetail:
        return await self._request_json("POST", f"/runs/{run_id}/cancel", json={}, model=RunDetail)

    async def resume_run(self, run_id: str) -> RunSummary:
        return await self._request_json("POST", f"/runs/{run_id}/resume", json={}, model=RunSummary)

    async def retry_run(self, run_id: str) -> RunSummary:
        return await self._request_json("POST", f"/runs/{run_id}/retry", json={}, model=RunSummary)

    async def get_profile_preferences(self, profile_id: str) -> ProfileRecord:
        return await self._request_json(
            "GET",
            f"/profiles/{profile_id}/preferences",
            model=ProfileRecord,
        )

    async def update_profile_preferences(
        self,
        profile_id: str,
        preferences: ProfilePreferences,
    ) -> ProfileRecord:
        return await self._request_json(
            "PUT",
            f"/profiles/{profile_id}/preferences",
            json=preferences.model_dump(mode="json"),
            model=ProfileRecord,
        )

    async def post_profile_feedback(
        self,
        profile_id: str,
        feedback: ProfileFeedback,
    ) -> list[BehaviorAssessment]:
        payload = await self._request_json(
            "POST",
            f"/profiles/{profile_id}/feedback",
            json=feedback.model_dump(mode="json"),
            model=None,
        )
        return [BehaviorAssessment.model_validate(item) for item in payload]

    async def get_context_packs(self, run_id: str) -> list[ContextPack]:
        payload = await self._request_json("GET", f"/runs/{run_id}/context-packs", model=None)
        return [ContextPack.model_validate(item) for item in payload]

    async def get_assessments(self, run_id: str) -> list[BehaviorAssessment]:
        payload = await self._request_json("GET", f"/runs/{run_id}/assessments", model=None)
        return [BehaviorAssessment.model_validate(item) for item in payload]

    async def wait_for_terminal_run(
        self,
        run_id: str,
        *,
        timeout_seconds: float = 120.0,
    ) -> RunDetail:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            detail = await self.get_run_detail(run_id)
            if detail.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                return detail
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"Run {run_id} did not reach a terminal state in time")
            await asyncio.sleep(0.2)

    async def stream_run_events(
        self,
        run_id: str,
        *,
        after_id: int = 0,
    ) -> AsyncIterator[StreamEnvelope]:
        cursor = after_id
        try:
            async for envelope in self._stream_events_via_sse(run_id, after_id=after_id):
                if envelope.id is not None:
                    cursor = max(cursor, envelope.id)
                yield envelope
                if envelope.is_terminal:
                    return
        except Exception:
            pass

        while True:
            events = await self.list_events(run_id, after_id=cursor)
            if not events:
                detail = await self.get_run_detail(run_id)
                if detail.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                    return
                await asyncio.sleep(self.poll_interval_seconds)
                continue
            for event in events:
                cursor = max(cursor, event.id)
                envelope = StreamEnvelope(
                    id=event.id,
                    run_id=event.run_id,
                    event_type=event.event_type,
                    payload=event.payload,
                    created_at=event.created_at.isoformat(),
                )
                yield envelope
                if envelope.is_terminal:
                    return

    async def _stream_events_via_sse(
        self,
        run_id: str,
        *,
        after_id: int = 0,
    ) -> AsyncIterator[StreamEnvelope]:
        path = f"/runs/{run_id}/stream/{after_id}" if after_id > 0 else f"/runs/{run_id}/stream"
        async with self.client.stream(
            "GET",
            path,
            headers={"accept": "text/event-stream"},
        ) as response:
            self._raise_for_status(response)
            buffered: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    for envelope in parse_sse_lines([*buffered, ""]):
                        yield envelope
                    buffered = []
                else:
                    buffered.append(line)
            if buffered:
                for envelope in parse_sse_lines([*buffered, ""]):
                    yield envelope

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        model: type[Any] | None,
    ) -> Any:
        response = await self.client.request(method, path, params=params, json=json)
        self._raise_for_status(response)
        payload = response.json()
        if model is None:
            return payload
        return model.model_validate(payload)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = response.text
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict) and "detail" in payload:
            detail_value = payload["detail"]
            if isinstance(detail_value, str):
                detail = detail_value
            else:
                detail = json.dumps(detail_value)
        raise TerminalClientError(detail or f"{response.status_code} {response.reason_phrase}")
