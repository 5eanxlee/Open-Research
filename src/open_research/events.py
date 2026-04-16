from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import orjson

from .db import ResearchStore
from .domain import RunEvent

TERMINAL_EVENT_TYPES = {"report.completed", "run.failed", "run.cancelled"}


class EventBroker:
    def __init__(self) -> None:
        self._queues: dict[str, set[asyncio.Queue[RunEvent]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def publish(self, event: RunEvent) -> None:
        async with self._lock:
            subscribers = list(self._queues[event.run_id])
        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await queue.put(event)

    @asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[asyncio.Queue[RunEvent]]:
        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._queues[run_id].add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                self._queues[run_id].discard(queue)
                if not self._queues[run_id]:
                    self._queues.pop(run_id, None)


class RunEventService:
    def __init__(self, store: ResearchStore, broker: EventBroker) -> None:
        self.store = store
        self.broker = broker

    async def publish(self, run_id: str, event_type: str, payload: dict | None = None) -> RunEvent:
        event = await self.store.append_event(run_id, event_type, payload)
        await self.broker.publish(event)
        return event

    async def replay(self, run_id: str, after_id: int = 0) -> list[RunEvent]:
        return await self.store.list_events(run_id, after_id=after_id)

    @asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[asyncio.Queue[RunEvent]]:
        async with self.broker.subscribe(run_id) as queue:
            yield queue


class EventStreamService:
    def __init__(
        self,
        *,
        events: RunEventService,
        keepalive_seconds: float = 10.0,
        mode: str = "memory",
    ) -> None:
        self.events = events
        self.keepalive_seconds = keepalive_seconds
        self.mode = mode

    async def is_ready(self) -> bool:
        return True

    async def stream(
        self,
        *,
        run_id: str,
        after_id: int = 0,
        is_disconnected,
    ) -> AsyncIterator[str]:
        current_cursor = after_id
        if self.mode == "database":
            async for chunk in self._stream_via_database(
                run_id=run_id,
                after_id=after_id,
                is_disconnected=is_disconnected,
            ):
                yield chunk
            return

        async with self.events.subscribe(run_id) as queue:
            yield format_sse_payload(
                event_type="stream.mode",
                payload={
                    "mode": "replay",
                    "run_id": run_id,
                    "cursor": current_cursor,
                },
                event_id=current_cursor or None,
            )

            historical = await self.events.replay(run_id, after_id=after_id)
            for event in historical:
                current_cursor = event.id
                yield format_sse(event)
            if historical and historical[-1].event_type in TERMINAL_EVENT_TYPES:
                yield format_sse_payload(
                    event_type="stream.mode",
                    payload={
                        "mode": "terminal",
                        "run_id": run_id,
                        "cursor": current_cursor,
                    },
                    event_id=current_cursor,
                )
                return

            yield format_sse_payload(
                event_type="stream.mode",
                payload={
                    "mode": "live",
                    "run_id": run_id,
                    "cursor": current_cursor,
                },
                event_id=current_cursor or None,
            )

            while True:
                if await is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=self.keepalive_seconds)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if event.id <= current_cursor:
                    continue
                current_cursor = event.id
                yield format_sse(event)
                if event.event_type in TERMINAL_EVENT_TYPES:
                    yield format_sse_payload(
                        event_type="stream.mode",
                        payload={
                            "mode": "terminal",
                            "run_id": run_id,
                            "cursor": current_cursor,
                        },
                        event_id=current_cursor,
                    )
                    return

    async def _stream_via_database(
        self,
        *,
        run_id: str,
        after_id: int,
        is_disconnected,
    ) -> AsyncIterator[str]:
        current_cursor = after_id
        yield format_sse_payload(
            event_type="stream.mode",
            payload={
                "mode": "replay",
                "run_id": run_id,
                "cursor": current_cursor,
            },
            event_id=current_cursor or None,
        )

        historical = await self.events.replay(run_id, after_id=after_id)
        for event in historical:
            current_cursor = event.id
            yield format_sse(event)
        if historical and historical[-1].event_type in TERMINAL_EVENT_TYPES:
            yield format_sse_payload(
                event_type="stream.mode",
                payload={
                    "mode": "terminal",
                    "run_id": run_id,
                    "cursor": current_cursor,
                },
                event_id=current_cursor,
            )
            return

        yield format_sse_payload(
            event_type="stream.mode",
            payload={
                "mode": "live",
                "run_id": run_id,
                "cursor": current_cursor,
            },
            event_id=current_cursor or None,
        )
        while True:
            if await is_disconnected():
                return
            events = await self.events.replay(run_id, after_id=current_cursor)
            if not events:
                yield ": keep-alive\n\n"
                await asyncio.sleep(self.keepalive_seconds)
                continue
            for event in events:
                if event.id <= current_cursor:
                    continue
                current_cursor = event.id
                yield format_sse(event)
                if event.event_type in TERMINAL_EVENT_TYPES:
                    yield format_sse_payload(
                        event_type="stream.mode",
                        payload={
                            "mode": "terminal",
                            "run_id": run_id,
                            "cursor": current_cursor,
                        },
                        event_id=current_cursor,
                    )
                    return


def format_sse(event: RunEvent) -> str:
    payload = {
        "id": event.id,
        "run_id": event.run_id,
        "event_type": event.event_type,
        "payload": event.payload,
        "created_at": event.created_at.isoformat(),
    }
    return f"id: {event.id}\nevent: {event.event_type}\ndata: {orjson.dumps(payload).decode()}\n\n"


def format_sse_payload(
    *,
    event_type: str,
    payload: dict,
    event_id: int | None = None,
) -> str:
    envelope = {
        "id": event_id,
        "run_id": payload.get("run_id"),
        "event_type": event_type,
        "payload": payload,
        "created_at": datetime.now(UTC).isoformat(),
    }
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event_type}")
    lines.append(f"data: {orjson.dumps(envelope).decode()}")
    return "\n".join(lines) + "\n\n"
