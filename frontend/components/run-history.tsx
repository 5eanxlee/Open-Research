"use client";

import { useMemo } from "react";

import type { RunStatus, RunSummary } from "@/lib/types";

interface RunHistoryProps {
  runs: RunSummary[];
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
}

const LIVE_STATUSES = new Set<RunStatus>([
  "queued",
  "clarifying",
  "awaiting_plan_approval",
  "planning",
  "researching",
  "grounding",
]);

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

const GROUP_ORDER = [
  "Today",
  "Yesterday",
  "Previous 7 days",
  "Previous 30 days",
] as const;

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

function groupLabel(iso: string): string {
  const then = new Date(iso);
  const now = new Date();
  const startOfDay = (d: Date) =>
    new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const diffDays = Math.floor(
    (startOfDay(now) - startOfDay(then)) / (24 * 60 * 60 * 1000),
  );
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return "Previous 7 days";
  if (diffDays < 30) return "Previous 30 days";
  return then.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

export function RunHistory({ runs, selectedRunId, onSelect }: RunHistoryProps) {
  const liveCount = runs.filter((run) => LIVE_STATUSES.has(run.status)).length;

  const groups = useMemo(() => {
    const sorted = [...runs].sort(
      (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
    );
    const buckets = new Map<string, RunSummary[]>();
    for (const run of sorted) {
      const label = groupLabel(run.updated_at);
      const existing = buckets.get(label) ?? [];
      existing.push(run);
      buckets.set(label, existing);
    }
    const ordered: Array<[string, RunSummary[]]> = [];
    for (const label of GROUP_ORDER) {
      const entries = buckets.get(label);
      if (entries?.length) {
        ordered.push([label, entries]);
        buckets.delete(label);
      }
    }
    const rest = Array.from(buckets.entries()).sort(
      (a, b) =>
        new Date(b[1][0].updated_at).getTime() -
        new Date(a[1][0].updated_at).getTime(),
    );
    return ordered.concat(rest);
  }, [runs]);

  return (
    <section className="panel history-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Chats</p>
          <h2 className="panel-title">History</h2>
        </div>
        <div className="history-head-meta">
          {liveCount > 0 ? (
            <span className="pill status-researching">
              <span className="status-dot status-researching" />
              {liveCount} live
            </span>
          ) : null}
          <span className="pill muted">{runs.length}</span>
        </div>
      </div>

      <div className="history-list">
        {runs.length === 0 ? (
          <div className="empty-state compact">
            <p>No chats yet</p>
            <span>
              Start a question from the prompt bar below to populate the workspace.
            </span>
          </div>
        ) : null}

        {groups.map(([label, bucket]) => (
          <div className="history-group" key={label}>
            <p className="history-group-label">{label}</p>
            {bucket.map((run) => {
              const isLive = LIVE_STATUSES.has(run.status);
              const isActive = selectedRunId === run.id;
              return (
                <button
                  className={`history-item ${isActive ? "active" : ""} ${
                    isLive ? "live" : ""
                  }`}
                  key={run.id}
                  onClick={() => onSelect(run.id)}
                  type="button"
                >
                  <div className="history-item-header">
                    <span className={`status-dot status-${run.status}`} />
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
                  <strong className="history-question">{run.question}</strong>
                  <div className="history-item-meta">
                    <span className="history-backend">
                      {run.workflow_backend ?? "local"}
                    </span>
                    {run.project_id ? (
                      <span className="history-chip">project</span>
                    ) : null}
                    {run.approval_status === "pending_approval" ||
                    run.approval_status === "pending_clarification" ? (
                      <span className="history-chip warn">approval</span>
                    ) : null}
                    {run.error_message ? (
                      <span className="history-chip danger">error</span>
                    ) : null}
                  </div>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </section>
  );
}
