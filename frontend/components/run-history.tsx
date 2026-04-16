"use client";

import type { RunSummary } from "@/lib/types";

interface RunHistoryProps {
  runs: RunSummary[];
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
}

export function RunHistory({ runs, selectedRunId, onSelect }: RunHistoryProps) {
  return (
    <section className="panel history-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">History</p>
          <h2 className="panel-title">Recent runs</h2>
        </div>
        <span className="pill muted">{runs.length}</span>
      </div>

      <div className="history-list">
        {runs.length === 0 ? (
          <div className="empty-state compact">
            <p>No runs yet.</p>
            <span>Start from the composer to populate the dashboard.</span>
          </div>
        ) : null}

        {runs.map((run) => (
          <button
            className={`history-item ${selectedRunId === run.id ? "active" : ""}`}
            key={run.id}
            onClick={() => onSelect(run.id)}
            type="button"
          >
            <div className="history-item-header">
              <span className={`status-dot status-${run.status}`} />
              <span className="history-status">{run.status}</span>
              <span className="history-time">
                {new Date(run.updated_at).toLocaleString(undefined, {
                  month: "short",
                  day: "numeric",
                  hour: "numeric",
                  minute: "2-digit",
                })}
              </span>
            </div>
            <strong>{run.question}</strong>
            <div className="history-item-meta">
              <span>${run.estimated_cost_usd.toFixed(4)}</span>
              <span>{run.workflow_backend ?? "local"}</span>
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}
