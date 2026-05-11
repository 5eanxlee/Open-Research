from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy, WorkflowIDReusePolicy
from temporalio.worker import Worker

from open_research.core.config import Settings
from open_research.core.domain import BudgetPolicy


@dataclass(slots=True)
class RunWorkflowInput:
    run_id: str
    question: str
    budget: dict[str, object]
    activity_timeout_seconds: int
    heartbeat_timeout_seconds: int


@workflow.defn(name="open-research-run")
class ResearchRunWorkflow:
    @workflow.run
    async def run(self, payload: RunWorkflowInput) -> None:
        await workflow.execute_activity(
            "execute_research_run",
            payload,
            start_to_close_timeout=timedelta(seconds=payload.activity_timeout_seconds),
            heartbeat_timeout=timedelta(seconds=payload.heartbeat_timeout_seconds),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )


class TemporalWorkflowBackend:
    backend_name = "temporal"

    def __init__(
        self,
        *,
        settings: Settings,
        execute_run: Callable[[str, str, BudgetPolicy], Awaitable[None]],
    ) -> None:
        self.settings = settings
        self._execute_run = execute_run
        self._client: Client | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._worker_stop = asyncio.Event()

    async def init(self) -> None:
        if self.settings.temporal_target_url is None:
            raise RuntimeError("TEMPORAL_TARGET_URL must be configured for Temporal mode.")
        if self._client is None:
            self._client = await Client.connect(
                self.settings.temporal_target_url,
                namespace=self.settings.temporal_namespace,
                api_key=(
                    self.settings.temporal_api_key.get_secret_value()
                    if self.settings.temporal_api_key is not None
                    else None
                ),
                tls=self.settings.temporal_tls,
            )
        if self.settings.temporal_start_worker and self._worker_task is None:
            self._worker_stop = asyncio.Event()
            self._worker_task = asyncio.create_task(
                self._run_worker(),
                name="temporal-open-research-worker",
            )

    async def shutdown(self) -> None:
        if self._worker_task is None:
            return
        self._worker_stop.set()
        try:
            await asyncio.wait_for(
                self._worker_task,
                timeout=self.settings.temporal_worker_shutdown_seconds,
            )
        except TimeoutError:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
        finally:
            self._worker_task = None

    async def start_run(
        self,
        *,
        run_id: str,
        question: str,
        budget: BudgetPolicy,
    ) -> None:
        client = self._require_client()
        await client.start_workflow(
            ResearchRunWorkflow.run,
            RunWorkflowInput(
                run_id=run_id,
                question=question,
                budget=budget.model_dump(mode="json"),
                activity_timeout_seconds=self.settings.temporal_activity_timeout_seconds,
                heartbeat_timeout_seconds=max(1, self.settings.temporal_heartbeat_seconds * 3),
            ),
            id=self.workflow_id_for_run(run_id),
            task_queue=self.settings.temporal_task_queue,
            id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        )

    async def cancel_run(self, run_id: str) -> None:
        client = self._require_client()
        handle = client.get_workflow_handle(self.workflow_id_for_run(run_id))
        await handle.cancel()

    async def describe_run(self, run_id: str) -> dict[str, str] | None:
        client = self._require_client()
        handle = client.get_workflow_handle(self.workflow_id_for_run(run_id))
        try:
            description = await handle.describe()
        except Exception:
            return None

        raw_status = getattr(getattr(description, "status", None), "name", None)
        if raw_status is None:
            return None
        status = str(raw_status).lower()
        return {"status": status}

    async def is_healthy(self) -> bool:
        client = self._require_client()
        try:
            await client.workflow_service.get_system_info(
                namespace=self.settings.temporal_namespace
            )
        except Exception:
            return False
        return True

    async def _run_worker(self) -> None:
        client = self._require_client()
        async with Worker(
            client,
            task_queue=self.settings.temporal_task_queue,
            workflows=[ResearchRunWorkflow],
            activities=[self.execute_research_run],
        ):
            await self._worker_stop.wait()

    @activity.defn(name="execute_research_run")
    async def execute_research_run(self, payload: RunWorkflowInput) -> None:
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(payload.run_id),
            name=f"temporal-heartbeat-{payload.run_id}",
        )
        try:
            await self._execute_run(
                payload.run_id,
                payload.question,
                BudgetPolicy.model_validate(payload.budget),
            )
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

    async def _heartbeat_loop(self, run_id: str) -> None:
        interval_seconds = max(1, self.settings.temporal_heartbeat_seconds)
        try:
            while True:
                activity.heartbeat({"run_id": run_id})
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            return

    def _require_client(self) -> Client:
        if self._client is None:
            raise RuntimeError("Temporal backend has not been initialized.")
        return self._client

    @staticmethod
    def workflow_id_for_run(run_id: str) -> str:
        return f"open-research-run-{run_id}"
