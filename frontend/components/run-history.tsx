"use client";

import { SquarePen } from "lucide-react";
import { useMemo } from "react";

import { deriveConversationTopic } from "@/lib/report-title";
import type { RunSummary } from "@/lib/types";

interface RunHistoryProps {
  runs: RunSummary[];
  isLoading?: boolean;
  errorMessage?: string | null;
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
  onNewChat?: () => void;
  newChatLabel?: string;
}

export function RunHistory({
  runs,
  isLoading = false,
  errorMessage = null,
  selectedRunId,
  onSelect,
  onNewChat,
  newChatLabel = "New chat",
}: RunHistoryProps) {
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
        {onNewChat ? (
          <button
            aria-label={newChatLabel}
            className="history-new-chat-button"
            onClick={onNewChat}
            title={newChatLabel}
            type="button"
          >
            <SquarePen aria-hidden size={15} strokeWidth={1.9} />
          </button>
        ) : null}
      </div>

      <div className="history-list">
        {isLoading ? (
          <div className="history-skeleton-list" aria-label="Loading run history">
            {Array.from({ length: 6 }).map((_, index) => (
              <div className="history-skeleton" key={index}>
                <span />
                <strong />
              </div>
            ))}
          </div>
        ) : null}

        {!isLoading && errorMessage ? (
          <div className="empty-state compact" role="status">
            <p>Run history unavailable</p>
            <span>{errorMessage}</span>
          </div>
        ) : null}

        {!isLoading && !errorMessage && runs.length === 0 ? (
          <div className="empty-state compact">
            <p>No runs</p>
          </div>
        ) : null}

        {!isLoading && !errorMessage ? sortedRuns.map((run) => {
          const isActive = selectedRunId === run.id;
          const topic = run.conversation_topic || deriveConversationTopic(run.question);
          return (
            <button
              className={`history-item ${isActive ? "active" : ""}`}
              key={run.id}
              onClick={() => onSelect(run.id)}
              type="button"
            >
              <strong className="history-question">{topic}</strong>
            </button>
          );
        }) : null}
      </div>
    </section>
  );
}
