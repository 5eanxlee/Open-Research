"use client";

import { useMemo } from "react";

import { deriveConversationTopic } from "@/lib/report-title";
import type { RunStatus, RunSummary } from "@/lib/types";

interface RunHistoryProps {
  runs: RunSummary[];
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
}

const STATUS_LABEL: Record<RunStatus, string> = {
  queued: "Queued",
  clarifying: "Clarifying",
  awaiting_plan_approval: "Awaiting plan",
  planning: "Planning",
  researching: "Researching",
  grounding: "Grounding",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

function formatWhen(iso: string): string {
  const then = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - then.getTime();
  const diffMin = Math.round(diffMs / 60_000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.round(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.round(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  return then.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function RunHistory({ runs, selectedRunId, onSelect }: RunHistoryProps) {
  const sortedRuns = useMemo(
    () =>
      [...runs].sort(
        (a, b) =>
          new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
      ),
    [runs],
  );

  return (
    <section className="panel history-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Chats</p>
        </div>
      </div>

      <div className="history-list">
        {runs.length === 0 ? (
          <div className="empty-state compact">
            <p>No runs</p>
          </div>
        ) : null}

        {sortedRuns.map((run) => {
          const isActive = selectedRunId === run.id;
          const topic = run.conversation_topic || deriveConversationTopic(run.question);
          return (
            <button
              className={`history-item ${isActive ? "active" : ""}`}
              key={run.id}
              onClick={() => onSelect(run.id)}
              type="button"
            >
              <div className="history-item-header">
                <span className={`history-status status-${run.status}`}>
                  {STATUS_LABEL[run.status]}
                </span>
                <span
                  className="history-time"
                  title={new Date(run.updated_at).toLocaleString()}
                >
                  {formatWhen(run.updated_at)}
                </span>
              </div>
              <strong className="history-question">{topic}</strong>
            </button>
          );
        })}
      </div>
    </section>
  );
}
