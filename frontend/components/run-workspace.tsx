"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

import type {
  RunConversationMessage,
  RunEvent,
  RunWorkspaceSnapshot,
  StreamConnectionState,
  WorkspaceCitationView,
  WorkspaceDecisionView,
  WorkspacePhaseKey,
  WorkspaceReportSectionView,
  WorkspaceSourceView,
  WorkspaceStreamView,
  WorkspaceTaskView,
} from "@/lib/types";

type WorkspaceTab =
  | "overview"
  | "plan"
  | "tasks"
  | "thinking"
  | "sources"
  | "citations"
  | "report"
  | "chat"
  | "trace";

type ThinkingSubtab = "decisions" | "agents" | "tools" | "files";

const PHASE_TO_TAB: Record<WorkspacePhaseKey, WorkspaceTab> = {
  intake: "overview",
  clarify: "plan",
  plan: "plan",
  execute: "tasks",
  ground: "report",
  audit: "citations",
  deliver: "report",
};

function formatTime(value: string | null | undefined): string {
  if (!value) return "n/a";
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDuration(start: string, end?: string | null): string {
  const from = new Date(start).getTime();
  const to = end ? new Date(end).getTime() : Date.now();
  const totalSeconds = Math.max(Math.round((to - from) / 1000), 0);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function titleCase(value: string): string {
  return value
    .replaceAll("_", " ")
    .replaceAll(".", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function EventCard({ event }: { event: RunEvent }) {
  return (
    <details className="event-card" open={event.event_type === "report.completed"}>
      <summary className="event-card-header">
        <strong>{event.event_type}</strong>
        <span>{new Date(event.created_at).toLocaleTimeString()}</span>
      </summary>
      <pre>{JSON.stringify(event.payload, null, 2)}</pre>
    </details>
  );
}

function groupBySection<T extends { section_title: string }>(entries: T[]): Array<[string, T[]]> {
  const grouped = new Map<string, T[]>();
  for (const entry of entries) {
    const current = grouped.get(entry.section_title) ?? [];
    current.push(entry);
    grouped.set(entry.section_title, current);
  }
  return Array.from(grouped.entries());
}

interface RunWorkspaceProps {
  workspace: RunWorkspaceSnapshot | undefined;
  rawEvents: RunEvent[];
  conversationMessages: RunConversationMessage[];
  connectionState: StreamConnectionState;
  connectionMode?: string | null;
  onCancel?: (runId: string) => void;
  onResume?: (runId: string) => void;
  onRetry?: (runId: string) => void;
  onAnswerClarification?: (runId: string, response: string) => void;
  onApprove?: (runId: string, note?: string) => void;
  onReject?: (runId: string, note?: string) => void;
  onRequestChanges?: (runId: string, note?: string) => void;
  onSendMessage?: (runId: string, message: string) => void;
  pending?: {
    cancel?: boolean;
    resume?: boolean;
    retry?: boolean;
    clarification?: boolean;
    approve?: boolean;
    reject?: boolean;
    requestChanges?: boolean;
    chat?: boolean;
  };
}

export function RunWorkspace({
  workspace,
  rawEvents,
  conversationMessages,
  connectionState,
  connectionMode,
  onCancel,
  onResume,
  onRetry,
  onAnswerClarification,
  onApprove,
  onReject,
  onRequestChanges,
  onSendMessage,
  pending,
}: RunWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>("overview");
  const [focusedPhase, setFocusedPhase] = useState<WorkspacePhaseKey | null>(null);
  const [selectedStreamId, setSelectedStreamId] = useState<string>("all");
  const [thinkingSubtab, setThinkingSubtab] = useState<ThinkingSubtab>("decisions");
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [selectedSectionId, setSelectedSectionId] = useState<string | null>(null);
  const [clarificationDraft, setClarificationDraft] = useState("");
  const [approvalNote, setApprovalNote] = useState("");
  const [chatDraft, setChatDraft] = useState("");

  useEffect(() => {
    if (!workspace) return;
    if (!workspace.report_sections.some((section) => section.id === selectedSectionId)) {
      setSelectedSectionId(workspace.report_sections[0]?.id ?? null);
    }
  }, [selectedSectionId, workspace]);

  useEffect(() => {
    if (!workspace) return;
    if (!workspace.sources.some((source) => source.id === selectedSourceId)) {
      setSelectedSourceId(workspace.sources[0]?.id ?? null);
    }
  }, [selectedSourceId, workspace]);

  useEffect(() => {
    if (!workspace) return;
    setActiveTab("overview");
    setFocusedPhase(null);
    setSelectedStreamId("all");
    setThinkingSubtab("decisions");
  }, [workspace?.run_id]);

  const phaseFilteredEvents = useMemo(() => {
    if (!workspace) return rawEvents;
    if (!focusedPhase) return rawEvents;
    const relevantTypes = new Set(
      workspace.phases
        .filter((phase) => phase.key === focusedPhase)
        .flatMap((phase) =>
          ({
            intake: ["run.started", "prompt.profile.applied", "job.created"],
            clarify: ["clarification.required", "clarification.answered"],
            plan: [
              "plan.preview.created",
              "plan.preview.approved",
              "plan.preview.rejected",
              "plan.preview.changes_requested",
              "plan.created",
            ],
            execute: [
              "stream.created",
              "task.started",
              "search.performed",
              "source.fetched",
              "note.saved",
              "gap.detected",
              "replan.started",
              "input_assets.ingested",
            ],
            ground: [
              "passages.reranked",
              "citation.verified",
              "claim.repair.started",
              "claim.repair.search_performed",
              "claim.repair.source_fetched",
              "claim.repair.completed",
            ],
            audit: ["citation.removed", "citation.audit.completed", "report.sanitized"],
            deliver: ["report.drafted", "report.completed"],
          })[phase.key],
        ),
    );
    return rawEvents.filter((event) => relevantTypes.has(event.event_type));
  }, [focusedPhase, rawEvents, workspace]);

  const visibleStreams = useMemo(() => {
    if (!workspace) return [];
    if (selectedStreamId === "all") return workspace.streams;
    return workspace.streams.filter((stream) => stream.id === selectedStreamId);
  }, [selectedStreamId, workspace]);

  const visibleDecisions = useMemo(() => {
    if (!workspace) return [];
    const byPhase = focusedPhase
      ? workspace.decisions.filter((decision) => {
          if (focusedPhase === "clarify") return decision.category === "clarification";
          if (focusedPhase === "plan") return decision.category === "planning";
          if (focusedPhase === "execute") {
            return decision.category === "stream_launch" || decision.category === "replan";
          }
          if (focusedPhase === "ground") {
            return (
              decision.category === "verification" || decision.category === "claim_repair"
            );
          }
          if (focusedPhase === "audit") return decision.category === "audit";
          return true;
        })
      : workspace.decisions;
    if (selectedStreamId === "all") return byPhase;
    return byPhase.filter((decision) => decision.affected_stream_id === selectedStreamId);
  }, [focusedPhase, selectedStreamId, workspace]);

  const sourceLanes = useMemo(() => {
    if (!workspace) {
      return [
        ["Project Corpus", []],
        ["Run Attachments", []],
        ["User Reference Sources", []],
        ["Web Discovered", []],
        ["Fetched", []],
        ["Retrieved for Grounding", []],
        ["Cited", []],
        ["Removed in Audit", []],
      ] as const;
    }
    return [
      ["Project Corpus", workspace.sources.filter((source) => source.origin === "project_corpus")],
      ["Run Attachments", workspace.sources.filter((source) => source.origin === "run_attachment")],
      ["User Reference Sources", workspace.sources.filter((source) => source.origin === "user_reference")],
      ["Web Discovered", workspace.sources.filter((source) => source.origin === "web_discovered")],
      ["Fetched", workspace.sources.filter((source) => source.state === "fetched")],
      ["Retrieved for Grounding", workspace.sources.filter((source) => source.state === "retrieved")],
      ["Cited", workspace.sources.filter((source) => source.state === "cited")],
      ["Removed in Audit", workspace.sources.filter((source) => source.state === "removed")],
    ] as const;
  }, [workspace]);

  if (!workspace) {
    return (
      <section className="panel workspace-panel">
        <div className="empty-state">
          <p>Select a run.</p>
          <span>The live research workspace appears here with plan, tasks, sources, citations, and report visibility.</span>
        </div>
      </section>
    );
  }

  const activePhase = workspace.phases.find((phase) => phase.key === workspace.current_phase);
  const selectedSection =
    workspace.report_sections.find((section) => section.id === selectedSectionId) ??
    workspace.report_sections[0] ??
    null;
  const selectedSource =
    workspace.sources.find((source) => source.id === selectedSourceId) ??
    workspace.sources[0] ??
    null;

  const approvalBanner =
    workspace.status === "clarifying" || workspace.status === "awaiting_plan_approval";

  return (
    <section className="panel workspace-panel">
      <div className="workspace-header">
        <div className="workspace-header-main">
          <p className="eyebrow">Run workspace</p>
          <h2 className="panel-title">{workspace.question}</h2>
          <div className="workspace-meta-row">
            <span className={`pill status-${workspace.status}`}>{workspace.status}</span>
            <span className="pill muted">{activePhase?.label ?? titleCase(workspace.current_phase)}</span>
            <span className="pill muted">{formatDuration(workspace.created_at, workspace.updated_at)}</span>
            <span className="pill muted">${workspace.estimated_cost_usd.toFixed(4)}</span>
            <span className="pill muted">{workspace.execution_mode}</span>
            <span className="pill muted">{workspace.approval_status ?? "not_required"}</span>
            <span className="pill muted">{workspace.job ? `job:${workspace.job.status}` : "direct run"}</span>
            <span className="pill muted">{connectionState}</span>
            {connectionMode ? <span className="pill muted">{connectionMode}</span> : null}
            {workspace.project_id ? <span className="pill muted">project attached</span> : null}
          </div>
          <div className="workspace-submeta">
            <span>Sources: {workspace.source_selection.length ? workspace.source_selection.join(", ") : "deployment default"}</span>
            <span>Last event: {formatTime(workspace.connection.last_event_at)}</span>
            <span>Events: {workspace.connection.event_count}</span>
          </div>
        </div>
        <div className="workspace-header-actions">
          {workspace.status !== "completed" &&
          workspace.status !== "failed" &&
          workspace.status !== "cancelled" ? (
            <button
              className="secondary-button"
              onClick={() => onCancel?.(workspace.run_id)}
              type="button"
              disabled={pending?.cancel}
            >
              Cancel
            </button>
          ) : null}
          {(workspace.status === "failed" || workspace.status === "cancelled") && onResume ? (
            <button
              className="secondary-button"
              onClick={() => onResume(workspace.run_id)}
              type="button"
              disabled={pending?.resume}
            >
              Resume
            </button>
          ) : null}
          {(workspace.status === "completed" ||
            workspace.status === "failed" ||
            workspace.status === "cancelled") &&
          onRetry ? (
            <button
              className="primary-button"
              onClick={() => onRetry(workspace.run_id)}
              type="button"
              disabled={pending?.retry}
            >
              Retry
            </button>
          ) : null}
        </div>
      </div>

      {approvalBanner ? (
        <div className="workspace-banner">
          <div>
            <strong>
              {workspace.status === "clarifying"
                ? "Clarification is blocking the plan."
                : "Plan approval is blocking execution."}
            </strong>
            <span>
              {workspace.status === "clarifying"
                ? "Answer the outstanding question so the preview can be regenerated."
                : "Approve, reject, or request changes to move the run forward."}
            </span>
          </div>
          <button
            className="secondary-button"
            onClick={() => setActiveTab("plan")}
            type="button"
          >
            Open plan tab
          </button>
        </div>
      ) : null}

      <div className="phase-rail">
        {workspace.phases.map((phase) => (
          <button
            className={`phase-node ${phase.status} ${focusedPhase === phase.key ? "active" : ""}`}
            key={phase.key}
            onClick={() => {
              setFocusedPhase((current) => (current === phase.key ? null : phase.key));
              setActiveTab(PHASE_TO_TAB[phase.key]);
            }}
            type="button"
          >
            <span className="phase-node-title">{phase.label}</span>
            <small>{phase.status}</small>
            <strong>{phase.event_count}</strong>
          </button>
        ))}
      </div>

      <div className="workspace-tabs">
        {(["overview", "plan", "tasks", "thinking", "sources", "citations", "report", "chat", "trace"] as WorkspaceTab[]).map((tab) => (
          <button
            className={`workspace-tab ${activeTab === tab ? "active" : ""}`}
            key={tab}
            onClick={() => setActiveTab(tab)}
            type="button"
          >
            {titleCase(tab)}
          </button>
        ))}
      </div>

      {activeTab === "overview" ? (
        <div className="workspace-grid overview-grid">
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Phase summary</h3>
              {focusedPhase ? <span className="pill muted">{titleCase(focusedPhase)}</span> : null}
            </div>
            <div className="workspace-phase-summary">
              {workspace.phases.map((phase) => (
                <article className={`phase-summary-card ${phase.status}`} key={phase.key}>
                  <strong>{phase.label}</strong>
                  <span>{phase.status}</span>
                  <small>{phase.blocked_reason ?? (phase.completed_at ? formatTime(phase.completed_at) : "No blockers")}</small>
                </article>
              ))}
            </div>
          </section>
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Blockers and health</h3>
            </div>
            <div className="workspace-list">
              {workspace.asset_processing_errors.map((error) => (
                <article className="workspace-inline-card danger" key={error}>
                  <strong>Asset processing issue</strong>
                  <span>{error}</span>
                </article>
              ))}
              {activePhase?.blocked_reason ? (
                <article className="workspace-inline-card warning">
                  <strong>Current blocker</strong>
                  <span>{activePhase.blocked_reason}</span>
                </article>
              ) : null}
              {workspace.asset_processing_errors.length === 0 && !activePhase?.blocked_reason ? (
                <span className="muted-text">No active blockers. The run is flowing normally.</span>
              ) : null}
            </div>
          </section>
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Active streams and tasks</h3>
            </div>
            <div className="workspace-list">
              {workspace.streams.slice(0, 6).map((stream) => (
                <article className="workspace-inline-card" key={stream.id}>
                  <strong>{stream.name}</strong>
                  <span>{stream.status} · {stream.query_count} queries · {stream.selected_source_count} sources</span>
                  <small>{stream.latest_note_summary ?? stream.objective}</small>
                </article>
              ))}
            </div>
          </section>
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Recent decisions</h3>
            </div>
            <div className="workspace-list">
              {visibleDecisions.slice(0, 6).map((decision) => (
                <DecisionCard decision={decision} key={decision.id} />
              ))}
            </div>
          </section>
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Source ingestion</h3>
            </div>
            <div className="workspace-phase-summary">
              {sourceLanes.map(([label, sources]) => (
                <article className="phase-summary-card compact" key={label}>
                  <strong>{label}</strong>
                  <span>{sources.length} sources</span>
                </article>
              ))}
            </div>
          </section>
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Report section progress</h3>
            </div>
            <div className="workspace-phase-summary">
              {workspace.report_sections.map((section) => (
                <article className="phase-summary-card compact" key={section.id}>
                  <strong>{section.title}</strong>
                  <span>{section.grounded_claim_count} grounded / {section.unsupported_claim_count} open</span>
                  <small>{section.citation_count} citations · {section.removed_citation_count} removed</small>
                </article>
              ))}
            </div>
          </section>
        </div>
      ) : null}

      {activeTab === "plan" ? (
        <div className="workspace-grid plan-grid">
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Clarification thread</h3>
            </div>
            {workspace.plan.clarification_session ? (
              <div className="workspace-thread">
                {workspace.plan.clarification_session.questions.map((question) => {
                  const answer = workspace.plan.clarification_session?.turns.find(
                    (turn) => turn.question_id === question.id,
                  );
                  return (
                    <article className="thread-item" key={question.id}>
                      <strong>{question.prompt}</strong>
                      <span className="muted-text">{question.rationale}</span>
                      <div className="thread-response">
                        {answer ? <span>{answer.response}</span> : <span className="muted-text">Awaiting response</span>}
                      </div>
                    </article>
                  );
                })}
                {workspace.status === "clarifying" && onAnswerClarification ? (
                  <div className="workspace-form">
                    <textarea
                      className="textarea-input"
                      value={clarificationDraft}
                      onChange={(event) => setClarificationDraft(event.target.value)}
                      rows={4}
                      placeholder="Describe must-cover angles, blockers, and desired outcome."
                    />
                    <button
                      className="primary-button"
                      onClick={() => onAnswerClarification(workspace.run_id, clarificationDraft.trim())}
                      type="button"
                      disabled={!clarificationDraft.trim() || pending?.clarification}
                    >
                      Submit clarification
                    </button>
                  </div>
                ) : null}
              </div>
            ) : (
              <span className="muted-text">No clarification session was needed for this run.</span>
            )}
          </section>

          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Plan preview and approved plan</h3>
            </div>
            {workspace.plan.plan_preview ? (
              <div className="workspace-stack">
                <article className="workspace-inline-card">
                  <strong>{workspace.plan.plan_preview.summary}</strong>
                  <span>{workspace.plan.plan_preview.hypothesis}</span>
                  <small>{workspace.plan.plan_preview.budget_decision_reason}</small>
                </article>
                {workspace.plan.approved_plan ? (
                  <article className="workspace-inline-card muted">
                    <strong>Approved plan</strong>
                    <span>{workspace.plan.approved_plan.summary}</span>
                    <small>{workspace.plan.approved_plan.hypothesis}</small>
                  </article>
                ) : null}
                {workspace.status === "awaiting_plan_approval" ? (
                  <div className="workspace-form">
                    <textarea
                      className="textarea-input"
                      value={approvalNote}
                      onChange={(event) => setApprovalNote(event.target.value)}
                      rows={3}
                      placeholder="Optional note or requested changes."
                    />
                    <div className="button-row">
                      <button
                        className="primary-button"
                        onClick={() => onApprove?.(workspace.run_id, approvalNote.trim() || undefined)}
                        type="button"
                        disabled={pending?.approve}
                      >
                        Approve
                      </button>
                      <button
                        className="secondary-button"
                        onClick={() => onRequestChanges?.(workspace.run_id, approvalNote.trim() || undefined)}
                        type="button"
                        disabled={pending?.requestChanges}
                      >
                        Request changes
                      </button>
                      <button
                        className="secondary-button"
                        onClick={() => onReject?.(workspace.run_id, approvalNote.trim() || undefined)}
                        type="button"
                        disabled={pending?.reject}
                      >
                        Reject
                      </button>
                    </div>
                  </div>
                ) : null}
              </div>
            ) : (
              <span className="muted-text">Plan preview will appear here when the planning phase starts.</span>
            )}
          </section>

          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Budget provenance</h3>
            </div>
            <div className="workspace-phase-summary">
              <article className="phase-summary-card compact">
                <strong>Requested</strong>
                <span>
                  {workspace.plan.requested_budget
                    ? `${workspace.plan.requested_budget.max_streams} streams / ${workspace.plan.requested_budget.max_queries_per_stream} queries`
                    : "n/a"}
                </span>
              </article>
              <article className="phase-summary-card compact">
                <strong>Recommended</strong>
                <span>
                  {workspace.plan.recommended_budget
                    ? `${workspace.plan.recommended_budget.max_streams} streams / ${workspace.plan.recommended_budget.max_queries_per_stream} queries`
                    : "n/a"}
                </span>
              </article>
              <article className="phase-summary-card compact">
                <strong>Effective</strong>
                <span>
                  {workspace.plan.effective_budget
                    ? `${workspace.plan.effective_budget.max_streams} streams / ${workspace.plan.effective_budget.max_queries_per_stream} queries`
                    : "n/a"}
                </span>
              </article>
            </div>
            <p className="muted-text">{workspace.plan.budget_decision_reason ?? "No explicit budget clamp reason."}</p>
          </section>

          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Planned streams</h3>
            </div>
            <div className="workspace-list">
              {(workspace.plan.plan_preview?.plan.streams ?? workspace.plan.approved_plan?.streams ?? []).map((stream) => (
                <article className="workspace-inline-card" key={`${stream.name}-${stream.objective}`}>
                  <strong>{stream.name}</strong>
                  <span>{stream.objective}</span>
                  <small>{stream.queries.join(" · ") || "Queries will be derived at execution time."}</small>
                </article>
              ))}
            </div>
          </section>

          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Planning assets and corpus influence</h3>
            </div>
            <div className="workspace-list">
              {workspace.plan.planning_assets.map((asset) => (
                <article className="workspace-inline-card" key={asset.id}>
                  <strong>{asset.label}</strong>
                  <span>{asset.project_id ? "Project corpus" : "Run planning asset"}</span>
                  <small>{asset.preview_excerpt ?? asset.description ?? "No preview available."}</small>
                </article>
              ))}
              {workspace.plan.project_assets.map((asset) => (
                <article className="workspace-inline-card muted" key={asset.id}>
                  <strong>{asset.label}</strong>
                  <span>Project corpus</span>
                  <small>{asset.preview_excerpt ?? asset.description ?? "Persistent corpus context."}</small>
                </article>
              ))}
            </div>
          </section>

          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Approval history</h3>
            </div>
            <div className="workspace-list">
              {workspace.plan.approval_history.map((decision, index) => (
                <article className="workspace-inline-card" key={`${decision.decision}-${decision.created_at}-${index}`}>
                  <strong>{titleCase(decision.decision)}</strong>
                  <span>{decision.note ?? "No note provided."}</span>
                  <small>{formatTime(decision.created_at)}</small>
                </article>
              ))}
              {workspace.plan.approval_history.length === 0 ? (
                <span className="muted-text">No approval actions recorded yet.</span>
              ) : null}
            </div>
          </section>
        </div>
      ) : null}

      {activeTab === "tasks" ? (
        <div className="workspace-grid tasks-grid">
          <aside className="workspace-side-rail">
            <button
              className={`ops-rail-button ${selectedStreamId === "all" ? "active" : ""}`}
              onClick={() => setSelectedStreamId("all")}
              type="button"
            >
              <span>All streams</span>
              <strong>{workspace.streams.length}</strong>
            </button>
            {workspace.streams.map((stream) => (
              <button
                className={`ops-rail-button ${selectedStreamId === stream.id ? "active" : ""}`}
                key={stream.id}
                onClick={() => setSelectedStreamId(stream.id)}
                type="button"
              >
                <div>
                  <span>{stream.name}</span>
                  <small>{stream.status}</small>
                </div>
                <strong>{stream.tasks.length}</strong>
              </button>
            ))}
          </aside>
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Stream execution board</h3>
              {focusedPhase ? <span className="pill muted">{titleCase(focusedPhase)}</span> : null}
            </div>
            <div className="workspace-stream-board">
              {visibleStreams.map((stream) => (
                <article className="stream-board-column" key={stream.id}>
                  <div className="stream-board-header">
                    <strong>{stream.name}</strong>
                    <span>{stream.status}</span>
                    <small>{stream.query_count} queries · {stream.selected_source_count} sources · {stream.note_count} notes</small>
                  </div>
                  <div className="stream-board-stack">
                    {stream.tasks.map((task) => (
                      <TaskCard key={task.id} task={task} />
                    ))}
                    {stream.tasks.length === 0 ? <span className="muted-text">No task records for this stream yet.</span> : null}
                  </div>
                </article>
              ))}
            </div>
          </section>
        </div>
      ) : null}

      {activeTab === "thinking" ? (
        <div className="workspace-grid thinking-grid">
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Structured reasoning</h3>
            </div>
            <div className="workspace-subtabs">
              {(["decisions", "agents", "tools", "files"] as ThinkingSubtab[]).map((subtab) => (
                <button
                  className={`workspace-subtab ${thinkingSubtab === subtab ? "active" : ""}`}
                  key={subtab}
                  onClick={() => setThinkingSubtab(subtab)}
                  type="button"
                >
                  {titleCase(subtab)}
                </button>
              ))}
            </div>
            {thinkingSubtab === "decisions" ? (
              <div className="workspace-list">
                {visibleDecisions.map((decision) => (
                  <DecisionCard decision={decision} key={decision.id} />
                ))}
              </div>
            ) : thinkingSubtab === "tools" ? (
              <div className="workspace-list">
                {phaseFilteredEvents
                  .filter((event) =>
                    [
                      "search.performed",
                      "source.fetched",
                      "provider.retry",
                      "claim.repair.search_performed",
                      "claim.repair.source_fetched",
                    ].includes(event.event_type),
                  )
                  .slice()
                  .reverse()
                  .map((event) => (
                    <article className="workspace-inline-card" key={event.id}>
                      <strong>{event.event_type}</strong>
                      <span>{JSON.stringify(event.payload)}</span>
                    </article>
                  ))}
              </div>
            ) : thinkingSubtab === "agents" ? (
              <div className="workspace-list">
                {workspace.streams.map((stream) => (
                  <article className="workspace-inline-card" key={stream.id}>
                    <strong>{stream.name}</strong>
                    <span>{stream.model}</span>
                    <small>{stream.objective}</small>
                  </article>
                ))}
              </div>
            ) : (
              <div className="workspace-list">
                {workspace.project_assets_available.concat(workspace.run_assets_available).map((asset) => (
                  <article className="workspace-inline-card" key={asset.id}>
                    <strong>{asset.label}</strong>
                    <span>{asset.usage.replaceAll("_", " ")}</span>
                    <small>{asset.preview_excerpt ?? asset.processing_error ?? "No asset preview."}</small>
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>
      ) : null}

      {activeTab === "sources" ? (
        <div className="workspace-grid sources-grid">
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Provenance explorer</h3>
            </div>
            <div className="source-lanes">
              {sourceLanes.map(([label, items]) => (
                <article className="source-lane" key={label}>
                  <div className="source-lane-header">
                    <strong>{label}</strong>
                    <span>{items.length}</span>
                  </div>
                  <div className="source-lane-list">
                    {items.map((source) => (
                      <button
                        className={`source-row ${selectedSource?.id === source.id ? "active" : ""}`}
                        key={source.id}
                        onClick={() => setSelectedSourceId(source.id)}
                        type="button"
                      >
                        <strong>{source.title ?? source.url}</strong>
                        <span>{source.state} · {source.trust_tier ?? "unknown trust"}</span>
                      </button>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </section>
          <section className="workspace-card source-detail-card">
            <div className="workspace-card-header">
              <h3>Source detail</h3>
            </div>
            {selectedSource ? (
              <div className="workspace-stack">
                <article className="workspace-inline-card">
                  <strong>{selectedSource.title ?? selectedSource.url}</strong>
                  <span>{selectedSource.url}</span>
                  <small>
                    {selectedSource.origin.replaceAll("_", " ")} · {selectedSource.state} · {selectedSource.provider ?? "unknown provider"}
                  </small>
                </article>
                <div className="workspace-phase-summary">
                  <article className="phase-summary-card compact">
                    <strong>Trust</strong>
                    <span>{selectedSource.trust_tier ?? "unknown"}</span>
                  </article>
                  <article className="phase-summary-card compact">
                    <strong>Passages</strong>
                    <span>{selectedSource.passages_used}</span>
                  </article>
                  <article className="phase-summary-card compact">
                    <strong>Sections</strong>
                    <span>{selectedSource.report_sections.length}</span>
                  </article>
                </div>
                <div className="workspace-list">
                  {selectedSource.note_summaries.map((summary, index) => (
                    <article className="workspace-inline-card" key={`${selectedSource.id}-summary-${index}`}>
                      <strong>Note summary</strong>
                      <span>{summary}</span>
                    </article>
                  ))}
                </div>
                {selectedSource.audit_reasons.length ? (
                  <article className="workspace-inline-card danger">
                    <strong>Audit removal</strong>
                    <span>{selectedSource.audit_reasons.join(", ")}</span>
                  </article>
                ) : null}
              </div>
            ) : (
              <span className="muted-text">Select a source lane item to inspect its provenance.</span>
            )}
          </section>
        </div>
      ) : null}

      {activeTab === "citations" ? (
        <div className="workspace-grid citations-grid">
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Citations by report section</h3>
            </div>
            <div className="workspace-list">
              {groupBySection(workspace.citations).map(([sectionTitle, sectionCitations]) => (
                <details className="workspace-detail-card" key={sectionTitle} open>
                  <summary>
                    <strong>{sectionTitle}</strong>
                    <span>{sectionCitations.length} citations</span>
                  </summary>
                  <div className="workspace-list">
                    {sectionCitations.map((citation) => (
                      <CitationCard
                        citation={citation}
                        key={citation.id}
                        onOpenSection={() => {
                          const target = workspace.report_sections.find(
                            (section) => section.title === citation.section_title,
                          );
                          if (target) {
                            setSelectedSectionId(target.id);
                            setActiveTab("report");
                          }
                        }}
                        onOpenSource={() => {
                          const target = workspace.sources.find(
                            (source) => source.source_id && source.source_id === citation.source_id,
                          );
                          if (target) {
                            setSelectedSourceId(target.id);
                            setActiveTab("sources");
                          }
                        }}
                      />
                    ))}
                  </div>
                </details>
              ))}
            </div>
          </section>
        </div>
      ) : null}

      {activeTab === "report" ? (
        <div className="workspace-grid report-grid">
          <aside className="workspace-side-rail">
            {workspace.report_sections.map((section) => (
              <button
                className={`ops-rail-button ${selectedSection?.id === section.id ? "active" : ""}`}
                key={section.id}
                onClick={() => setSelectedSectionId(section.id)}
                type="button"
              >
                <div>
                  <span>{section.title}</span>
                  <small>{section.grounded_claim_count} grounded / {section.unsupported_claim_count} open</small>
                </div>
                <strong>{section.citation_count}</strong>
              </button>
            ))}
          </aside>
          <section className="workspace-card report-body-card">
            <div className="workspace-card-header">
              <h3>{selectedSection?.title ?? "Report section"}</h3>
            </div>
            {selectedSection ? (
              <article className="markdown-body workspace-markdown">
                <ReactMarkdown>{selectedSection.body_markdown || "_No section body available._"}</ReactMarkdown>
              </article>
            ) : (
              <span className="muted-text">No report sections are available yet.</span>
            )}
          </section>
          <section className="workspace-card report-sidecar-card">
            <div className="workspace-card-header">
              <h3>Grounding sidecar</h3>
            </div>
            {selectedSection ? (
              <div className="workspace-list">
                {selectedSection.claims.map((claim) => (
                  <article className="workspace-inline-card" key={`${claim.section_title}-${claim.ordinal}`}>
                    <strong>{claim.claim}</strong>
                    <span>
                      {claim.support_label ?? "unsupported"} · {claim.citation_count} citations · {claim.removed_citation_count} removed
                    </span>
                    <small>{claim.claim_repair_ran ? "Claim repair ran" : "No claim repair"}</small>
                  </article>
                ))}
              </div>
            ) : null}
          </section>
        </div>
      ) : null}

      {activeTab === "chat" ? (
        <div className="workspace-grid chat-grid">
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Follow-up chat</h3>
            </div>
            {workspace.status !== "completed" || !workspace.final_report_markdown ? (
              <span className="muted-text">
                Follow-up chat becomes available once the research run has produced a final report.
              </span>
            ) : (
              <div className="workspace-stack">
                <div className="workspace-thread conversation-thread">
                  {conversationMessages.length === 0 ? (
                    <article className="thread-item">
                      <strong>Start a follow-up conversation.</strong>
                      <span className="muted-text">
                        Ask for clarifications, compare claims, summarize sections, or drill into citations.
                      </span>
                    </article>
                  ) : (
                    conversationMessages.map((message) => (
                      <article
                        className={`thread-item conversation-turn ${message.role}`}
                        key={message.id}
                      >
                        <strong>{message.role === "user" ? "You" : "Assistant"}</strong>
                        <div className="workspace-markdown">
                          <ReactMarkdown>{message.content}</ReactMarkdown>
                        </div>
                        <div className="thread-response">
                          <span>{formatTime(message.created_at)}</span>
                          {message.model ? <span>{message.model}</span> : null}
                          {message.references.length ? (
                            <span>Refs: {message.references.join(" · ")}</span>
                          ) : null}
                        </div>
                      </article>
                    ))
                  )}
                </div>
                <div className="workspace-form">
                  <textarea
                    className="textarea-input"
                    value={chatDraft}
                    onChange={(event) => setChatDraft(event.target.value)}
                    rows={4}
                    placeholder="Ask a follow-up about the completed research..."
                  />
                  <button
                    className="primary-button"
                    onClick={() => {
                      const trimmed = chatDraft.trim();
                      if (!trimmed) return;
                      onSendMessage?.(workspace.run_id, trimmed);
                      setChatDraft("");
                    }}
                    type="button"
                    disabled={!chatDraft.trim() || pending?.chat}
                  >
                    {pending?.chat ? "Sending..." : "Send"}
                  </button>
                </div>
              </div>
            )}
          </section>
        </div>
      ) : null}

      {activeTab === "trace" ? (
        <div className="workspace-grid trace-grid">
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Transport and replay health</h3>
            </div>
            <div className="workspace-phase-summary">
              <article className="phase-summary-card compact">
                <strong>Transport</strong>
                <span>{connectionState}</span>
              </article>
              <article className="phase-summary-card compact">
                <strong>Mode</strong>
                <span>{connectionMode ?? workspace.connection.stream_mode ?? "n/a"}</span>
              </article>
              <article className="phase-summary-card compact">
                <strong>Last event id</strong>
                <span>{workspace.connection.last_event_id}</span>
              </article>
              <article className="phase-summary-card compact">
                <strong>Workflow backend</strong>
                <span>{workspace.connection.workflow_backend ?? "local"}</span>
              </article>
            </div>
          </section>
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Raw event stream</h3>
              {focusedPhase ? <span className="pill muted">{titleCase(focusedPhase)}</span> : null}
            </div>
            <div className="event-feed workspace-trace-feed">
              {phaseFilteredEvents
                .slice()
                .reverse()
                .map((event) => <EventCard event={event} key={`${event.run_id}-${event.id}`} />)}
            </div>
          </section>
        </div>
      ) : null}
    </section>
  );
}

function TaskCard({ task }: { task: WorkspaceTaskView }) {
  return (
    <article className={`task-card task-${task.status}`}>
      <div className="task-card-header">
        <strong>{titleCase(task.task_type)}</strong>
        <span>{task.status}</span>
      </div>
      <span>{task.objective}</span>
      <div className="task-card-meta">
        <small>{task.query_count} queries</small>
        <small>{task.selected_source_count} sources</small>
        <small>{task.notes_produced} notes</small>
      </div>
      {task.latest_note_summary ? <small>{task.latest_note_summary}</small> : null}
      {task.last_tool_call ? <small>Last tool: {task.last_tool_call}</small> : null}
      {task.last_decision ? <small>Last decision: {task.last_decision}</small> : null}
      {task.blocker_reason ? <small className="danger-text">{task.blocker_reason}</small> : null}
    </article>
  );
}

function DecisionCard({ decision }: { decision: WorkspaceDecisionView }) {
  return (
    <article className="decision-card">
      <div className="decision-card-header">
        <strong>{decision.title}</strong>
        <span>{titleCase(decision.category)}</span>
      </div>
      <span>{decision.rationale}</span>
      <div className="decision-card-meta">
        {decision.affected_stream_name ? <small>{decision.affected_stream_name}</small> : null}
        {decision.affected_section ? <small>{decision.affected_section}</small> : null}
        {decision.supporting_evidence_count ? (
          <small>{decision.supporting_evidence_count} evidence items</small>
        ) : null}
        <small>{formatTime(decision.timestamp)}</small>
      </div>
    </article>
  );
}

function CitationCard({
  citation,
  onOpenSection,
  onOpenSource,
}: {
  citation: WorkspaceCitationView;
  onOpenSection: () => void;
  onOpenSource: () => void;
}) {
  return (
    <article className={`citation-card citation-${citation.status}`}>
      <div className="citation-card-header">
        <strong>{citation.claim}</strong>
        <span>{citation.status}</span>
      </div>
      <span>{citation.source_title ?? citation.source_url ?? "Unknown source"}</span>
      {citation.quote ? <small>{citation.quote}</small> : null}
      {citation.audit_reasons.length ? (
        <small className="danger-text">{citation.audit_reasons.join(", ")}</small>
      ) : null}
      <div className="button-row">
        <button className="ghost-button" onClick={onOpenSection} type="button">
          Open section
        </button>
        <button className="ghost-button" onClick={onOpenSource} type="button">
          Open source
        </button>
      </div>
    </article>
  );
}
