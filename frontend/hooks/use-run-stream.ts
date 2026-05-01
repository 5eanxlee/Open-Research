"use client";

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { getStreamUrl } from "@/lib/api";
import type { RunEvent, StreamEnvelope } from "@/lib/types";
import { useResearchStore } from "@/store/use-research-store";

const KNOWN_EVENT_TYPES = [
  "stream.mode",
  "job.created",
  "run.started",
  "prompt.profile.applied",
  "run.heartbeat",
  "run.cancel_requested",
  "run.cancellation_requested",
  "run.cancelled",
  "run.failed",
  "run.resumed",
  "run.recovered",
  "run.retried",
  "run.shutdown",
  "clarification.required",
  "clarification.answered",
  "plan.preview.created",
  "plan.preview.approved",
  "plan.preview.rejected",
  "plan.preview.changes_requested",
  "planning.preview.generated",
  "planning.execution.started",
  "planning.validation.failed",
  "planning.validation.passed",
  "planning.discovery.recorded",
  "planning.execution.completed",
  "plan.created",
  "stream.created",
  "stream.cancelled",
  "stream.failed",
  "task.started",
  "search.performed",
  "source.selection.finalized",
  "source.cache.hit",
  "source.fetch_failed",
  "source.fallback_document.created",
  "source.skipped",
  "source.fetched",
  "note.saved",
  "input_assets.ingested",
  "gap.detected",
  "replan.started",
  "passages.reranked",
  "claim.repair.started",
  "claim.repair.search_performed",
  "claim.repair.source_fetched",
  "claim.repair.completed",
  "claim.repair.skipped",
  "citation.verification.skipped",
  "citation.verified",
  "citation.contradicted",
  "citation.removed",
  "citation.audit.completed",
  "report.drafted",
  "report.sanitized",
  "report.completed",
  "provider.call.started",
  "provider.call.completed",
  "provider.retry",
  "memory.retrieved",
  "memory.compiled",
  "context.fragment.dropped",
  "context.pack.created",
  "profile.feedback.recorded",
  "conversation.message.added",
  "custom_responses.deepagent.started",
  "completion_gate.evaluated",
  "completion_gate.continuation_requested",
  "completion_gate.continuation_applied",
  "completion_gate.exhausted",
  "deepagents.tool.started",
  "deepagents.tool.completed",
  "deepagents.tool.failed",
  "tool.budget.low",
];

const SHOULD_INVALIDATE_WORKSPACE = new Set([
  "job.created",
  "run.started",
  "run.resumed",
  "run.recovered",
  "run.retried",
  "run.shutdown",
  "clarification.required",
  "clarification.answered",
  "plan.preview.created",
  "plan.preview.approved",
  "plan.preview.rejected",
  "plan.preview.changes_requested",
  "planning.preview.generated",
  "planning.execution.started",
  "planning.validation.failed",
  "planning.validation.passed",
  "planning.discovery.recorded",
  "planning.execution.completed",
  "plan.created",
  "stream.created",
  "stream.cancelled",
  "stream.failed",
  "task.started",
  "search.performed",
  "source.selection.finalized",
  "source.cache.hit",
  "source.fetch_failed",
  "source.fallback_document.created",
  "source.skipped",
  "source.fetched",
  "note.saved",
  "input_assets.ingested",
  "gap.detected",
  "replan.started",
  "passages.reranked",
  "claim.repair.started",
  "claim.repair.search_performed",
  "claim.repair.source_fetched",
  "claim.repair.completed",
  "claim.repair.skipped",
  "citation.verification.skipped",
  "citation.verified",
  "citation.contradicted",
  "citation.removed",
  "citation.audit.completed",
  "report.drafted",
  "report.sanitized",
  "memory.retrieved",
  "memory.compiled",
  "context.fragment.dropped",
  "context.pack.created",
  "profile.feedback.recorded",
  "conversation.message.added",
  "completion_gate.continuation_requested",
  "completion_gate.continuation_applied",
  "completion_gate.exhausted",
]);

interface UseRunStreamOptions {
  runId: string | null;
  apiBaseUrl: string;
}

function toRunEvent(envelope: StreamEnvelope): RunEvent | null {
  if (typeof envelope.id !== "number" || !envelope.run_id) {
    return null;
  }
  return {
    id: envelope.id,
    run_id: envelope.run_id,
    event_type: envelope.event_type,
    payload: envelope.payload,
    created_at: envelope.created_at,
  };
}

export function useRunStream({ runId, apiBaseUrl }: UseRunStreamOptions): void {
  const queryClient = useQueryClient();
  const appendEvent = useResearchStore((state) => state.appendEvent);
  const setConnectionState = useResearchStore((state) => state.setConnectionState);
  const clearRunEvents = useResearchStore((state) => state.clearRunEvents);

  useEffect(() => {
    if (!runId) {
      return;
    }

    clearRunEvents(runId);
    setConnectionState(runId, "connecting");

    const stream = new EventSource(getStreamUrl(apiBaseUrl, runId, 0));

    const invalidateRunQueries = (): void => {
      void queryClient.invalidateQueries({ queryKey: ["run-detail", apiBaseUrl, runId] });
      void queryClient.invalidateQueries({ queryKey: ["run-workspace", apiBaseUrl, runId] });
      void queryClient.invalidateQueries({ queryKey: ["run-report", apiBaseUrl, runId] });
      void queryClient.invalidateQueries({ queryKey: ["run-audit", apiBaseUrl, runId] });
      void queryClient.invalidateQueries({ queryKey: ["run-artifacts", apiBaseUrl, runId] });
      void queryClient.invalidateQueries({ queryKey: ["run-notes", apiBaseUrl, runId] });
      void queryClient.invalidateQueries({ queryKey: ["run-passages", apiBaseUrl, runId] });
      void queryClient.invalidateQueries({ queryKey: ["run-context-packs", apiBaseUrl, runId] });
      void queryClient.invalidateQueries({ queryKey: ["run-assessments", apiBaseUrl, runId] });
      void queryClient.invalidateQueries({ queryKey: ["runs", apiBaseUrl] });
    };

    const handleEnvelope = (event: MessageEvent<string>): void => {
      const envelope = JSON.parse(event.data) as StreamEnvelope;
      if (envelope.event_type === "stream.mode") {
        const mode = String(envelope.payload.mode ?? "live");
        if (mode === "replay") {
          setConnectionState(runId, "replay", mode);
        } else if (mode === "live") {
          setConnectionState(runId, "live", mode);
        } else {
          setConnectionState(runId, "terminal", mode);
          invalidateRunQueries();
        }
        return;
      }

      const runEvent = toRunEvent(envelope);
      if (!runEvent) {
        return;
      }
      appendEvent(runId, runEvent);

      if (
        runEvent.event_type === "report.completed" ||
        runEvent.event_type === "run.failed" ||
        runEvent.event_type === "run.cancelled"
      ) {
        setConnectionState(runId, "terminal");
        invalidateRunQueries();
      } else if (SHOULD_INVALIDATE_WORKSPACE.has(runEvent.event_type)) {
        invalidateRunQueries();
      }
    };

    for (const eventType of KNOWN_EVENT_TYPES) {
      stream.addEventListener(eventType, handleEnvelope as EventListener);
    }

    stream.onerror = () => {
      setConnectionState(runId, "error");
    };

    return () => {
      stream.close();
    };
  }, [
    apiBaseUrl,
    appendEvent,
    clearRunEvents,
    queryClient,
    runId,
    setConnectionState,
  ]);
}
