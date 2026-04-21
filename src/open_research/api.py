from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .config import Settings, get_settings
from .domain import (
    ApprovalDecisionKind,
    AsyncJob,
    ArtifactRecord,
    BehaviorAssessment,
    CitationAuditRecord,
    ClarificationResponseRequest,
    ClarificationSession,
    ContextPack,
    CreateProjectRequest,
    CreateRunRequest,
    FinalReport,
    PassageInspectionRecord,
    PlanApprovalRequest,
    PlanPreview,
    PromoteAssetRequest,
    ProfileFeedback,
    ProfilePreferences,
    ProfileRecord,
    ProjectDetail,
    ProjectSummary,
    PublicRuntimeConfig,
    ResearchAssetBatchRequest,
    ResearchAssetRecord,
    ResearchAssetUsage,
    RunConversationMessage,
    RunConversationReply,
    RunConversationRequest,
    RunDetail,
    RunEvent,
    RunNoteRecord,
    RunStatus,
    RunSummary,
    RunWorkspaceSnapshot,
    StagedAssetRecord,
)
from .runtime import ResearchRuntime


def create_app(
    *,
    settings: Settings | None = None,
    runtime: ResearchRuntime | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    runtime = runtime or ResearchRuntime.build(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.init()
        try:
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runtime = runtime
    app.state.settings = settings

    @app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readycheck(http_request: Request) -> dict[str, bool | str]:
        runtime = _get_runtime(http_request)
        readiness = await runtime.readiness()
        if not readiness["ready"]:
            raise HTTPException(status_code=503, detail=readiness)
        return readiness

    @app.get("/metrics")
    async def metrics(http_request: Request) -> Response:
        runtime = _get_runtime(http_request)
        content_type, payload = runtime.telemetry.render_metrics()
        return Response(content=payload, media_type=content_type)

    @app.get("/config/public", response_model=PublicRuntimeConfig)
    async def public_config(http_request: Request) -> PublicRuntimeConfig:
        runtime = _get_runtime(http_request)
        return runtime.public_config()

    @app.post("/runs", response_model=RunSummary, status_code=202)
    async def create_run(request: CreateRunRequest, http_request: Request) -> RunSummary:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.start_run(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/jobs", response_model=AsyncJob, status_code=202)
    async def create_job(request: CreateRunRequest, http_request: Request) -> AsyncJob:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.submit_job(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/projects", response_model=list[ProjectSummary])
    async def list_projects(http_request: Request) -> list[ProjectSummary]:
        runtime = _get_runtime(http_request)
        return await runtime.list_projects()

    @app.post("/projects", response_model=ProjectSummary, status_code=201)
    async def create_project(
        payload: CreateProjectRequest,
        http_request: Request,
    ) -> ProjectSummary:
        runtime = _get_runtime(http_request)
        return await runtime.create_project(payload)

    @app.get("/projects/{project_id}", response_model=ProjectDetail)
    async def get_project(project_id: str, http_request: Request) -> ProjectDetail:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_project_detail(project_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return detail

    @app.post("/projects/{project_id}/assets", response_model=list[ResearchAssetRecord], status_code=201)
    async def add_project_assets(
        project_id: str,
        payload: ResearchAssetBatchRequest,
        http_request: Request,
    ) -> list[ResearchAssetRecord]:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.save_project_assets(project_id, payload.assets)
        except KeyError:
            raise HTTPException(status_code=404, detail="Project not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/projects/{project_id}/assets/upload",
        response_model=list[ResearchAssetRecord],
        status_code=201,
    )
    async def upload_project_assets(
        project_id: str,
        http_request: Request,
        usage: str = Form(...),
        description: str | None = Form(default=None),
        files: list[UploadFile] = File(...),
    ) -> list[ResearchAssetRecord]:
        runtime = _get_runtime(http_request)
        try:
            usage_enum = ResearchAssetUsage(usage)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        file_payloads = [(upload.filename or "upload", upload.content_type, await upload.read()) for upload in files]
        try:
            return await runtime.upload_project_files(
                project_id=project_id,
                usage=usage_enum,
                files=file_payloads,
                description=description,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Project not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/projects/{project_id}/assets/{asset_id}", status_code=204)
    async def delete_project_asset(
        project_id: str,
        asset_id: str,
        http_request: Request,
    ) -> Response:
        runtime = _get_runtime(http_request)
        deleted = await runtime.delete_project_asset(project_id, asset_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Project asset not found")
        return Response(status_code=204)

    @app.post("/assets/staged", response_model=list[StagedAssetRecord], status_code=201)
    async def stage_assets(
        http_request: Request,
        usage: str = Form(...),
        description: str | None = Form(default=None),
        files: list[UploadFile] = File(...),
    ) -> list[StagedAssetRecord]:
        runtime = _get_runtime(http_request)
        try:
            usage_enum = ResearchAssetUsage(usage)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        file_payloads = [(upload.filename or "upload", upload.content_type, await upload.read()) for upload in files]
        try:
            return await runtime.stage_uploaded_files(
                usage=usage_enum,
                files=file_payloads,
                description=description,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/assets/staged/{asset_id}", response_model=StagedAssetRecord)
    async def get_staged_asset(asset_id: str, http_request: Request) -> StagedAssetRecord:
        runtime = _get_runtime(http_request)
        asset = await runtime.get_staged_asset(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="Staged asset not found")
        return asset

    @app.delete("/assets/staged/{asset_id}", status_code=204)
    async def delete_staged_asset(asset_id: str, http_request: Request) -> Response:
        runtime = _get_runtime(http_request)
        deleted = await runtime.delete_staged_asset(asset_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Staged asset not found")
        return Response(status_code=204)

    @app.get("/jobs/{job_id}", response_model=AsyncJob)
    async def get_job(job_id: str, http_request: Request) -> AsyncJob:
        runtime = _get_runtime(http_request)
        job = await runtime.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/jobs/{job_id}/state", response_model=AsyncJob)
    async def get_job_state(job_id: str, http_request: Request) -> AsyncJob:
        return await get_job(job_id, http_request)

    @app.get("/jobs/{job_id}/report", response_model=FinalReport)
    async def get_job_report(job_id: str, http_request: Request) -> FinalReport:
        runtime = _get_runtime(http_request)
        job = await runtime.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return await get_report(job.run_id, http_request)

    @app.get("/jobs/{job_id}/workspace", response_model=RunWorkspaceSnapshot)
    async def get_job_workspace(job_id: str, http_request: Request) -> RunWorkspaceSnapshot:
        runtime = _get_runtime(http_request)
        workspace = await runtime.get_job_workspace(job_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return workspace

    @app.post("/jobs/{job_id}/cancel", response_model=RunDetail, status_code=202)
    async def cancel_job(job_id: str, http_request: Request) -> RunDetail:
        runtime = _get_runtime(http_request)
        job = await runtime.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return await cancel_run(job.run_id, http_request)

    @app.get("/profiles/{profile_id}/preferences", response_model=ProfileRecord)
    async def get_profile_preferences(
        profile_id: str,
        http_request: Request,
    ) -> ProfileRecord:
        runtime = _get_runtime(http_request)
        profile = await runtime.get_profile_preferences(profile_id)
        if profile is None:
            return await runtime.update_profile_preferences(profile_id, ProfilePreferences())
        return profile

    @app.put("/profiles/{profile_id}/preferences", response_model=ProfileRecord)
    async def put_profile_preferences(
        profile_id: str,
        preferences: ProfilePreferences,
        http_request: Request,
    ) -> ProfileRecord:
        runtime = _get_runtime(http_request)
        return await runtime.update_profile_preferences(profile_id, preferences)

    @app.post("/profiles/{profile_id}/feedback", response_model=list[BehaviorAssessment])
    async def post_profile_feedback(
        profile_id: str,
        feedback: ProfileFeedback,
        http_request: Request,
    ) -> list[BehaviorAssessment]:
        runtime = _get_runtime(http_request)
        normalized = feedback.model_copy(update={"profile_id": profile_id})
        return await runtime.record_profile_feedback(normalized)

    @app.get("/runs", response_model=list[RunSummary])
    async def list_runs(
        http_request: Request,
        limit: int = 50,
        status: RunStatus | None = None,
        project_id: str | None = None,
    ) -> list[RunSummary]:
        runtime = _get_runtime(http_request)
        return await runtime.list_runs(limit=limit, status=status, project_id=project_id)

    @app.get("/runs/{run_id}", response_model=RunDetail)
    async def get_run(run_id: str, http_request: Request) -> RunDetail:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return detail

    @app.get("/runs/{run_id}/messages", response_model=list[RunConversationMessage])
    async def get_run_messages(run_id: str, http_request: Request) -> list[RunConversationMessage]:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.list_run_conversation_messages(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None

    @app.post("/runs/{run_id}/messages", response_model=RunConversationReply, status_code=201)
    async def create_run_message(
        run_id: str,
        payload: RunConversationRequest,
        http_request: Request,
    ) -> RunConversationReply:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.send_run_conversation_message(run_id, payload)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/workspace", response_model=RunWorkspaceSnapshot)
    async def get_run_workspace(run_id: str, http_request: Request) -> RunWorkspaceSnapshot:
        runtime = _get_runtime(http_request)
        workspace = await runtime.get_run_workspace(run_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return workspace

    @app.get("/runs/{run_id}/clarification", response_model=ClarificationSession)
    async def get_clarification(run_id: str, http_request: Request) -> ClarificationSession:
        runtime = _get_runtime(http_request)
        session = await runtime.get_clarification(run_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Clarification session not found")
        return session

    @app.post("/runs/{run_id}/clarification/respond", response_model=RunDetail)
    async def respond_clarification(
        run_id: str,
        payload: ClarificationResponseRequest,
        http_request: Request,
    ) -> RunDetail:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.answer_run_clarification(run_id, payload.response)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/plan-preview", response_model=PlanPreview)
    async def get_plan_preview(run_id: str, http_request: Request) -> PlanPreview:
        runtime = _get_runtime(http_request)
        preview = await runtime.get_plan_preview(run_id)
        if preview is None:
            raise HTTPException(status_code=404, detail="Plan preview not found")
        return preview

    @app.post("/runs/{run_id}/plan-preview/approve", response_model=RunDetail)
    async def approve_plan_preview(
        run_id: str,
        payload: PlanApprovalRequest,
        http_request: Request,
    ) -> RunDetail:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.approve_run_plan(
                run_id,
                decision=ApprovalDecisionKind.APPROVE,
                note=payload.note,
                actor=payload.actor,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/plan-preview/reject", response_model=RunDetail)
    async def reject_plan_preview(
        run_id: str,
        payload: PlanApprovalRequest,
        http_request: Request,
    ) -> RunDetail:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.approve_run_plan(
                run_id,
                decision=ApprovalDecisionKind.REJECT,
                note=payload.note,
                actor=payload.actor,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/plan-preview/request-changes", response_model=RunDetail)
    async def request_plan_changes(
        run_id: str,
        payload: PlanApprovalRequest,
        http_request: Request,
    ) -> RunDetail:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.approve_run_plan(
                run_id,
                decision=ApprovalDecisionKind.REQUEST_CHANGES,
                note=payload.note,
                actor=payload.actor,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/assets/{asset_id}/promote", response_model=ResearchAssetRecord)
    async def promote_run_asset(
        run_id: str,
        asset_id: str,
        payload: PromoteAssetRequest,
        http_request: Request,
    ) -> ResearchAssetRecord:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        project_id = payload.project_id or detail.project_id
        if not project_id:
            raise HTTPException(
                status_code=400,
                detail="Run is not attached to a project; provide a project_id to promote the asset.",
            )
        try:
            return await runtime.promote_run_asset(
                run_id=run_id,
                asset_id=asset_id,
                project_id=project_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/report", response_model=FinalReport)
    async def get_report(run_id: str, http_request: Request) -> FinalReport:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        report = await runtime.get_final_report(run_id)
        if report is None:
            raise HTTPException(status_code=409, detail="Run does not have a final report yet")
        return report

    @app.get("/runs/{run_id}/artifacts", response_model=list[ArtifactRecord])
    async def get_artifacts(run_id: str, http_request: Request) -> list[ArtifactRecord]:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return await runtime.list_artifacts(run_id)

    @app.get("/runs/{run_id}/audit", response_model=list[CitationAuditRecord])
    async def get_audit(run_id: str, http_request: Request) -> list[CitationAuditRecord]:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return await runtime.list_citation_audits(run_id)

    @app.get("/runs/{run_id}/notes", response_model=list[RunNoteRecord])
    async def get_notes(run_id: str, http_request: Request) -> list[RunNoteRecord]:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return await runtime.list_notes(run_id)

    @app.get("/runs/{run_id}/passages", response_model=list[PassageInspectionRecord])
    async def get_passages(
        run_id: str,
        http_request: Request,
    ) -> list[PassageInspectionRecord]:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return await runtime.list_passages(run_id)

    @app.get("/runs/{run_id}/context-packs", response_model=list[ContextPack])
    async def get_context_packs(run_id: str, http_request: Request) -> list[ContextPack]:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return await runtime.list_context_packs(run_id)

    @app.get("/runs/{run_id}/assessments", response_model=list[BehaviorAssessment])
    async def get_assessments(run_id: str, http_request: Request) -> list[BehaviorAssessment]:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return await runtime.list_behavior_assessments(run_id)

    @app.post("/runs/{run_id}/cancel", response_model=RunDetail, status_code=202)
    async def cancel_run(run_id: str, http_request: Request) -> RunDetail:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.cancel_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/resume", response_model=RunSummary, status_code=202)
    async def resume_run(run_id: str, http_request: Request) -> RunSummary:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.resume_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/retry", response_model=RunSummary, status_code=202)
    async def retry_run(run_id: str, http_request: Request) -> RunSummary:
        runtime = _get_runtime(http_request)
        try:
            return await runtime.retry_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/events", response_model=list[RunEvent])
    async def get_events(run_id: str, http_request: Request, after_id: int = 0) -> list[RunEvent]:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return await runtime.list_events(run_id, after_id=after_id)

    @app.get("/runs/{run_id}/stream")
    async def stream_events(
        run_id: str,
        http_request: Request,
        after_id: int = 0,
    ) -> StreamingResponse:
        return await _stream_run_events(run_id, http_request, after_id=after_id)

    @app.get("/runs/{run_id}/stream/{last_event_id}")
    async def stream_events_from_cursor(
        run_id: str,
        last_event_id: int,
        http_request: Request,
    ) -> StreamingResponse:
        return await _stream_run_events(run_id, http_request, after_id=last_event_id)

    @app.get("/jobs/{job_id}/stream")
    async def stream_job_events(
        job_id: str,
        http_request: Request,
        after_id: int = 0,
    ) -> StreamingResponse:
        runtime = _get_runtime(http_request)
        job = await runtime.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return await _stream_run_events(job.run_id, http_request, after_id=after_id)

    async def _stream_run_events(
        run_id: str,
        http_request: Request,
        *,
        after_id: int,
    ) -> StreamingResponse:
        runtime = _get_runtime(http_request)
        detail = await runtime.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")

        try:
            last_event_id = int(http_request.headers.get("last-event-id", str(after_id)))
        except ValueError:
            last_event_id = after_id

        async def event_stream() -> AsyncIterator[str]:
            latest_events = await runtime.list_events(run_id)
            if latest_events:
                runtime.telemetry.record_replay_lag(max(latest_events[-1].id - last_event_id, 0))
            async for chunk in runtime.stream_service.stream(
                run_id=run_id,
                after_id=last_event_id,
                is_disconnected=http_request.is_disconnected,
            ):
                yield chunk

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _get_runtime(request: Request) -> ResearchRuntime:
    return request.app.state.runtime  # type: ignore[return-value]
