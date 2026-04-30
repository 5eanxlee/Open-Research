"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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

type WorkspaceDetailsTab =
  | "overview"
  | "plan"
  | "tasks"
  | "thinking"
  | "sources"
  | "citations"
  | "trace";

type WorkspacePrimaryView = "chat" | "report";

type ThinkingSubtab = "decisions" | "agents" | "tools" | "files";

const PHASE_TO_DETAILS_TAB: Record<WorkspacePhaseKey, WorkspaceDetailsTab> = {
  intake: "overview",
  clarify: "plan",
  plan: "plan",
  execute: "tasks",
  ground: "citations",
  audit: "citations",
  deliver: "overview",
};

const DETAIL_TAB_LABELS: Record<WorkspaceDetailsTab, string> = {
  overview: "Pipeline",
  plan: "Plan",
  tasks: "Agents",
  thinking: "Thinking",
  sources: "Sources",
  citations: "Report",
  trace: "Tool calls",
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

function formatElapsedMs(value: number | null | undefined): string | null {
  if (value == null || value <= 0) return null;
  const totalSeconds = Math.max(Math.round(value / 1000), 0);
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

function slugifyFilename(value: string): string {
  const normalized = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized || "research-report";
}

function normalizeInlineText(value: string | null | undefined): string {
  if (!value) return "";
  return value
    .replace(/\\[nrt]/g, " ")
    .replace(/\\+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeLookupValue(value: string | null | undefined): string | null {
  const normalized = normalizeInlineText(value);
  return normalized ? normalized.toLowerCase() : null;
}

function resolveCitationSource(
  citation: WorkspaceCitationView,
  sources: WorkspaceSourceView[],
): WorkspaceSourceView | null {
  const sourceId = normalizeLookupValue(citation.source_id);
  if (sourceId) {
    const directMatch = sources.find(
      (source) =>
        normalizeLookupValue(source.source_id) === sourceId ||
        normalizeLookupValue(source.id) === sourceId,
    );
    if (directMatch) {
      return directMatch;
    }
  }

  const sourceUrl = normalizeLookupValue(citation.source_url);
  if (sourceUrl) {
    const urlMatch = sources.find((source) => normalizeLookupValue(source.url) === sourceUrl);
    if (urlMatch) {
      return urlMatch;
    }
  }

  const sourceTitle = normalizeLookupValue(citation.source_title);
  if (sourceTitle) {
    const titleMatch = sources.find(
      (source) => normalizeLookupValue(source.title) === sourceTitle,
    );
    if (titleMatch) {
      return titleMatch;
    }
  }

  const claimTitle = normalizeLookupValue(citation.claim);
  if (claimTitle) {
    const claimMatch = sources.find(
      (source) => normalizeLookupValue(source.title) === claimTitle,
    );
    if (claimMatch) {
      return claimMatch;
    }
  }

  return null;
}

function getCitationSourceLabel(
  citation: WorkspaceCitationView,
  sources: WorkspaceSourceView[],
): string {
  const resolvedSource = resolveCitationSource(citation, sources);
  return normalizeInlineText(
    citation.source_title ??
      resolvedSource?.title ??
      citation.source_url ??
      resolvedSource?.url ??
      "Unknown source",
  );
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

function MarkdownContent({
  content,
  className,
}: {
  content: string;
  className?: string;
}) {
  return (
    <div className={`markdown-body workspace-markdown${className ? ` ${className}` : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node: _node, ...props }) => (
            <a {...props} rel="noreferrer" target="_blank" />
          ),
          table: ({ node: _node, children, ...props }) => (
            <div className="markdown-table-wrap">
              <table {...props}>{children as ReactNode}</table>
            </div>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
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
  pending?: {
    cancel?: boolean;
    resume?: boolean;
    retry?: boolean;
    clarification?: boolean;
    approve?: boolean;
    reject?: boolean;
    requestChanges?: boolean;
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
  pending,
}: RunWorkspaceProps) {
  const [detailsTab, setDetailsTab] = useState<WorkspaceDetailsTab>("overview");
  const [primaryView, setPrimaryView] = useState<WorkspacePrimaryView>("chat");
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [focusedPhase, setFocusedPhase] = useState<WorkspacePhaseKey | null>(null);
  const [selectedStreamId, setSelectedStreamId] = useState<string>("all");
  const [thinkingSubtab, setThinkingSubtab] = useState<ThinkingSubtab>("decisions");
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [selectedSectionId, setSelectedSectionId] = useState<string | null>(null);
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);
  const [clarificationDraft, setClarificationDraft] = useState("");
  const [approvalNote, setApprovalNote] = useState("");
  const [reportExpanded, setReportExpanded] = useState(false);
  const [reportActionNotice, setReportActionNotice] = useState<string | null>(null);

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
    if (!workspace.citations.some((citation) => citation.id === selectedCitationId)) {
      setSelectedCitationId(workspace.citations[0]?.id ?? null);
    }
  }, [selectedCitationId, workspace]);

  useEffect(() => {
    if (!workspace) return;
    setDetailsTab("overview");
    setPrimaryView("chat");
    setDetailsOpen(false);
    setFocusedPhase(null);
    setSelectedStreamId("all");
    setThinkingSubtab("decisions");
    setReportExpanded(false);
    setReportActionNotice(null);
  }, [workspace?.run_id]);

  useEffect(() => {
    if (!reportActionNotice) return;
    const timeout = window.setTimeout(() => setReportActionNotice(null), 2200);
    return () => window.clearTimeout(timeout);
  }, [reportActionNotice]);

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
              "source.cache.hit",
              "source.fetch_failed",
              "source.fallback_document.created",
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
              "source.cache.hit",
              "source.fetch_failed",
              "source.fallback_document.created",
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
      <section className="panel workspace-panel workspace-empty-panel">
        <div className="workspace-empty">
          <div className="workspace-empty-mark" aria-hidden>
            <span className="workspace-empty-dot status-queued" />
            <span className="workspace-empty-dot status-researching" />
            <span className="workspace-empty-dot status-completed" />
          </div>
          <p className="eyebrow">Workspace</p>
          <h2 className="workspace-empty-title">Ready to run</h2>
          <p className="workspace-empty-copy">
            Ask a question below. As the run unfolds, the workspace will show the
            plan preview, stream-by-stream execution, grounded citations, and the
            final report — all in one place.
          </p>
          <div className="workspace-empty-hints">
            <article>
              <strong>Plan before you run</strong>
              <span>
                Enable plan approval in Customize → Workflow for human-in-the-loop
                planning before any sources are touched.
              </span>
            </article>
            <article>
              <strong>Attach context</strong>
              <span>
                Link a Project to keep a session anchored to your corpus and prior
                research.
              </span>
            </article>
            <article>
              <strong>Traceable grounding</strong>
              <span>
                Every claim in the final report links back to the passage, source,
                and audit decision that supports it.
              </span>
            </article>
          </div>
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
  const selectedCitation =
    workspace.citations.find((citation) => citation.id === selectedCitationId) ??
    workspace.citations[0] ??
    null;
  const citationSummary = {
    surviving: workspace.citations.filter((citation) => citation.status === "surviving").length,
    removed: workspace.citations.filter((citation) => citation.status === "removed").length,
    uncitedSources: workspace.sources.filter(
      (source) => source.state !== "cited" && source.state !== "removed",
    ).length,
  };
  const displayConnectionState =
    (workspace.status === "completed" ||
      workspace.status === "failed" ||
      workspace.status === "cancelled") &&
    connectionState === "error"
      ? "terminal"
      : connectionState;
  const approvalBanner =
    workspace.status === "clarifying" || workspace.status === "awaiting_plan_approval";
  const reportReady = Boolean(workspace.final_report_markdown);
  const reportMarkdown = workspace.final_report_markdown ?? "";
  const reportFilename = `${slugifyFilename(workspace.question)}-report.md`;
  const plannedStreams =
    workspace.plan.plan_preview?.plan.streams ?? workspace.plan.approved_plan?.streams ?? [];
  const shellSummary = [
    {
      label: "Phase",
      value: activePhase?.label ?? titleCase(workspace.current_phase),
    },
    { label: "Elapsed", value: formatDuration(workspace.created_at, workspace.updated_at) },
    { label: "Cost", value: `$${workspace.estimated_cost_usd.toFixed(4)}` },
    { label: "Transport", value: connectionMode ?? displayConnectionState },
  ];

  const handleCopyReport = async () => {
    if (!reportMarkdown) return;
    try {
      await navigator.clipboard.writeText(reportMarkdown);
      setReportActionNotice("Report copied.");
    } catch {
      setReportActionNotice("Copy failed.");
    }
  };

  const handleDownloadReport = () => {
    if (!reportMarkdown) return;
    const blob = new Blob([reportMarkdown], { type: "text/markdown;charset=utf-8" });
    const href = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = href;
    link.download = reportFilename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(href);
    setReportActionNotice("Report downloaded.");
  };

  return (
    <section
      className={`panel workspace-panel workspace-shell-panel ${
        detailsOpen ? "activity-open" : ""
      }`}
    >
      <div className="workspace-shell-header">
        <div className="workspace-shell-title">
          <span className={`workspace-run-status status-${workspace.status}`}>
            {titleCase(workspace.status)}
          </span>
          <span className="workspace-shell-timestamp">
            Updated {formatTime(workspace.updated_at)}
          </span>
        </div>
        <div className="workspace-shell-actions">
          <div className="workspace-primary-tabs">
            <button
              className={`workspace-primary-tab ${primaryView === "chat" ? "active" : ""}`}
              onClick={() => setPrimaryView("chat")}
              type="button"
            >
              Chat
            </button>
            <button
              className={`workspace-primary-tab ${primaryView === "report" ? "active" : ""}`}
              onClick={() => setPrimaryView("report")}
              type="button"
              disabled={!reportReady}
            >
              Final report
            </button>
          </div>
          <button
            className="secondary-button"
            aria-label={detailsOpen ? "Hide research activity" : "Show research activity"}
            onClick={() => setDetailsOpen((current) => !current)}
            type="button"
          >
            {detailsOpen ? "Hide" : "Activity"}
          </button>
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

      <div className={`workspace-shell-layout ${detailsOpen ? "with-details" : ""}`}>
        <section className="workspace-primary-surface">
          {approvalBanner ? (
            <div className="workspace-banner workspace-banner-inline">
              <div>
                <strong>
                  {workspace.status === "clarifying"
                    ? "Clarification is blocking the plan."
                    : "Plan approval is blocking execution."}
                </strong>
                <span>
                  {workspace.status === "clarifying"
                    ? "Answer the outstanding question so the preview can be regenerated."
                    : "Approve, reject, or request changes to move this run forward."}
                </span>
              </div>
              <button
                className="secondary-button"
                onClick={() => {
                  setDetailsOpen(true);
                  setDetailsTab("plan");
                }}
                type="button"
              >
                Review plan
              </button>
            </div>
          ) : null}

          {primaryView === "chat" ? (
            <div className="workspace-chat-shell">
              <div className="workspace-thread conversation-thread">
                <article className="thread-item conversation-turn user prompt-turn">
                  <strong>You</strong>
                  <p>{workspace.question}</p>
                </article>

                {workspace.status === "awaiting_plan_approval" && workspace.plan.plan_preview ? (
                  <article className="thread-item conversation-turn system">
                    <strong>Plan preview</strong>
                    <div className="workspace-stack compact">
                      <span>{workspace.plan.plan_preview.summary}</span>
                      <small>{workspace.plan.plan_preview.hypothesis}</small>
                      <div className="workspace-token-list">
                        <span className="workspace-token">
                          {workspace.plan.effective_budget
                            ? `${workspace.plan.effective_budget.max_streams} streams`
                            : "Budget pending"}
                        </span>
                        <span className="workspace-token">
                          {plannedStreams.length} planned lanes
                        </span>
                      </div>
                    </div>
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
                          onClick={() =>
                            onApprove?.(workspace.run_id, approvalNote.trim() || undefined)
                          }
                          type="button"
                          disabled={pending?.approve}
                        >
                          Approve
                        </button>
                        <button
                          className="secondary-button"
                          onClick={() =>
                            onRequestChanges?.(
                              workspace.run_id,
                              approvalNote.trim() || undefined,
                            )
                          }
                          type="button"
                          disabled={pending?.requestChanges}
                        >
                          Request changes
                        </button>
                        <button
                          className="secondary-button"
                          onClick={() =>
                            onReject?.(workspace.run_id, approvalNote.trim() || undefined)
                          }
                          type="button"
                          disabled={pending?.reject}
                        >
                          Reject
                        </button>
                      </div>
                    </div>
                  </article>
                ) : null}

                {workspace.status === "clarifying" &&
                workspace.plan.clarification_session &&
                onAnswerClarification ? (
                  <article className="thread-item conversation-turn system">
                    <strong>Clarification needed</strong>
                    <div className="workspace-stack compact">
                      {workspace.plan.clarification_session.questions.map((question) => (
                        <article className="workspace-inline-card" key={question.id}>
                          <strong>{question.prompt}</strong>
                          <small>{question.rationale}</small>
                        </article>
                      ))}
                    </div>
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
                        onClick={() =>
                          onAnswerClarification(workspace.run_id, clarificationDraft.trim())
                        }
                        type="button"
                        disabled={!clarificationDraft.trim() || pending?.clarification}
                      >
                        Submit clarification
                      </button>
                    </div>
                  </article>
                ) : null}

                {reportReady ? (
                  <article className="thread-item conversation-turn assistant report-turn">
                    <div className="conversation-turn-header">
                      <strong>Assistant</strong>
                      <div className="workspace-report-actions">
                        <button
                          className="ghost-button"
                          onClick={() => setReportExpanded((current) => !current)}
                          type="button"
                        >
                          {reportExpanded ? "Collapse report" : "Read full report"}
                        </button>
                        <button
                          className="ghost-button"
                          onClick={() => setPrimaryView("report")}
                          type="button"
                        >
                          Open report view
                        </button>
                        <button
                          className="ghost-button"
                          onClick={handleCopyReport}
                          type="button"
                        >
                          Copy
                        </button>
                        <button
                          className="ghost-button"
                          onClick={handleDownloadReport}
                          type="button"
                        >
                          Download
                        </button>
                      </div>
                    </div>
                    {reportActionNotice ? (
                      <span className="workspace-inline-feedback">{reportActionNotice}</span>
                    ) : null}
                    <MarkdownContent
                      className={reportExpanded ? "workspace-report-preview expanded" : "workspace-report-preview collapsed"}
                      content={reportMarkdown}
                    />
                  </article>
                ) : workspace.status === "failed" ||
                  workspace.status === "cancelled" ? (
                  <article className="thread-item conversation-turn assistant status-turn terminal">
                    <div className="status-turn-head">
                      <span className={`status-dot status-${workspace.status}`} />
                      <strong>
                        {workspace.status === "failed"
                          ? "Run failed"
                          : "Run cancelled"}
                      </strong>
                    </div>
                    <span>
                      {activePhase?.blocked_reason ??
                        (workspace.status === "failed"
                          ? "The run stopped before a report could be grounded. Retry or resume to continue."
                          : "The run was cancelled. You can retry from the header or start a fresh question below.")}
                    </span>
                    <div className="workspace-token-list">
                      <span className="workspace-token">
                        Stopped in{" "}
                        {activePhase?.label ?? titleCase(workspace.current_phase)}
                      </span>
                      <span className="workspace-token">
                        {workspace.connection.event_count} events captured
                      </span>
                      {workspace.citations.length ? (
                        <span className="workspace-token">
                          {workspace.citations.length} citations
                        </span>
                      ) : null}
                    </div>
                  </article>
                ) : (
                  <article className="thread-item conversation-turn assistant status-turn">
                    <div className="status-turn-head">
                      <span className={`status-dot status-${workspace.status}`} />
                      <strong>Research in progress</strong>
                      <span className="status-turn-phase">
                        {activePhase?.label ?? titleCase(workspace.current_phase)}
                      </span>
                    </div>
                    <span>
                      {activePhase?.blocked_reason ??
                        (workspace.streams.length
                          ? `Running ${workspace.streams.length} stream${
                              workspace.streams.length === 1 ? "" : "s"
                            } across ${workspace.sources.length} source${
                              workspace.sources.length === 1 ? "" : "s"
                            }.`
                          : `Current phase: ${
                              activePhase?.label ?? titleCase(workspace.current_phase)
                            }.`)}
                    </span>
                    <div className="status-turn-phase-strip" aria-hidden>
                      {workspace.phases.map((phase) => (
                        <span
                          className={`phase-pip phase-${phase.status} ${
                            phase.key === workspace.current_phase ? "current" : ""
                          }`}
                          key={phase.key}
                          title={phase.label}
                        />
                      ))}
                    </div>
                    <div className="workspace-token-list">
                      <span className="workspace-token">
                        {workspace.connection.event_count} events
                      </span>
                      <span className="workspace-token">
                        {workspace.source_selection.length
                          ? `${workspace.source_selection.length} source presets`
                          : "Default sources"}
                      </span>
                      {workspace.citations.length ? (
                        <span className="workspace-token">
                          {workspace.citations.length} cites so far
                        </span>
                      ) : null}
                    </div>
                  </article>
                )}

                {conversationMessages.map((message) => (
                  <article
                    className={`thread-item conversation-turn ${message.role}`}
                    key={message.id}
                  >
                    <strong>{message.role === "user" ? "You" : "Assistant"}</strong>
                    <MarkdownContent content={message.content} />
                    <div className="thread-response">
                      <span>{formatTime(message.created_at)}</span>
                      {message.model ? <span>{message.model}</span> : null}
                      {message.references.length ? (
                        <span>Refs: {message.references.join(" · ")}</span>
                      ) : null}
                    </div>
                  </article>
                ))}
              </div>

            </div>
          ) : (
            <div className="workspace-report-shell">
              <div className="workspace-report-header">
                <div>
                  <p className="eyebrow">Final report</p>
                  <h3>{selectedSection?.title ?? "Grounded research report"}</h3>
                </div>
                <div className="workspace-report-actions">
                  <button
                    className="ghost-button"
                    onClick={() => setPrimaryView("chat")}
                    type="button"
                  >
                    Back to chat
                  </button>
                  <button
                    className="ghost-button"
                    onClick={() => {
                      setDetailsOpen(true);
                      setDetailsTab("citations");
                    }}
                    type="button"
                  >
                    View citations
                  </button>
                  <button className="ghost-button" onClick={handleCopyReport} type="button">
                    Copy
                  </button>
                  <button className="ghost-button" onClick={handleDownloadReport} type="button">
                    Download
                  </button>
                </div>
              </div>
              {reportActionNotice ? (
                <span className="workspace-inline-feedback">{reportActionNotice}</span>
              ) : null}
              {workspace.report_sections.length ? (
                <div className="workspace-report-sections">
                  {workspace.report_sections.map((section) => (
                    <button
                      className={`workspace-report-section-tab ${
                        selectedSection?.id === section.id ? "active" : ""
                      }`}
                      key={section.id}
                      onClick={() => setSelectedSectionId(section.id)}
                      type="button"
                    >
                      <span>{section.title}</span>
                      <small>{section.citation_count} cites</small>
                    </button>
                  ))}
                </div>
              ) : null}
              <article className="workspace-report-body workspace-scroll-panel">
                <MarkdownContent
                  className="workspace-report-markdown"
                  content={
                    selectedSection?.body_markdown ||
                    workspace.final_report_markdown ||
                    "_No final report available yet._"
                  }
                />
              </article>
            </div>
          )}
        </section>

        {detailsOpen ? (
          <aside className="workspace-details-panel" aria-label="Research Activity">
            <div className="workspace-details-heading">
              <div>
                <strong>Research Activity</strong>
                <span>{activePhase?.label ?? titleCase(workspace.current_phase)}</span>
              </div>
              <button
                className="ghost-button"
                onClick={() => setDetailsOpen(false)}
                type="button"
              >
                Close
              </button>
            </div>

            <div className="workspace-details-summary">
              {shellSummary.map((item) => (
                <article className="workspace-context-card" key={item.label}>
                  <span className="workspace-context-label">{item.label}</span>
                  <strong>{item.value}</strong>
                </article>
              ))}
              <article className="workspace-context-card">
                <span className="workspace-context-label">Sources</span>
                <strong>
                  {workspace.source_selection.length
                    ? workspace.source_selection.join(", ")
                    : "deployment default"}
                </strong>
              </article>
              <article className="workspace-context-card">
                <span className="workspace-context-label">Project</span>
                <strong>{workspace.project_id ? "attached" : "none"}</strong>
              </article>
            </div>

            <div className="phase-rail phase-rail-details">
              {workspace.phases.map((phase) => (
                <button
                  className={`phase-node ${phase.status} ${
                    focusedPhase === phase.key ? "active" : ""
                  }`}
                  key={phase.key}
                  onClick={() => {
                    setFocusedPhase((current) => (current === phase.key ? null : phase.key));
                    setDetailsTab(PHASE_TO_DETAILS_TAB[phase.key]);
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
              {(
                [
                  "overview",
                  "plan",
                  "tasks",
                  "thinking",
                  "sources",
                  "citations",
                  "trace",
                ] as WorkspaceDetailsTab[]
              ).map((tab) => (
                <button
                  className={`workspace-tab ${detailsTab === tab ? "active" : ""}`}
                  key={tab}
                  onClick={() => setDetailsTab(tab)}
                  type="button"
                >
                  {DETAIL_TAB_LABELS[tab]}
                </button>
              ))}
            </div>

            <div className="workspace-details-content">
      {detailsTab === "overview" ? (
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

      {detailsTab === "plan" ? (
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
                  {stream.queries.length ? (
                    <div className="workspace-token-list">
                      {stream.queries.map((query) => (
                        <span className="workspace-token" key={`${stream.name}-${query}`}>
                          {query}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <small>Queries will be derived at execution time.</small>
                  )}
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

      {detailsTab === "tasks" ? (
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

      {detailsTab === "thinking" ? (
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
                      "source.cache.hit",
                      "source.fetch_failed",
                      "source.fallback_document.created",
                      "source.fetched",
                      "provider.retry",
                      "tool.budget.low",
                      "claim.repair.search_performed",
                      "claim.repair.source_fetched",
                    ].includes(event.event_type),
                  )
                  .slice()
                  .reverse()
                  .map((event) => (
                    <article className="workspace-inline-card" key={event.id}>
                      <strong>{event.event_type}</strong>
                      <span>{normalizeInlineText(JSON.stringify(event.payload))}</span>
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

      {detailsTab === "sources" ? (
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

      {detailsTab === "citations" ? (
        <div className="workspace-grid citations-grid">
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Citations by report section</h3>
            </div>
            <div className="workspace-list">
              {groupBySection(workspace.citations).map(([sectionTitle, sectionCitations]) => (
                <details className="workspace-detail-card" key={sectionTitle} open>
                  <summary className="workspace-detail-summary">
                    <div>
                      <strong>{sectionTitle}</strong>
                      <span>{sectionCitations.length} citations</span>
                    </div>
                  </summary>
                  <div className="workspace-list">
                    {sectionCitations.map((citation) => (
                      <CitationCard
                        citation={citation}
                        key={citation.id}
                        sources={workspace.sources}
                        selected={selectedCitation?.id === citation.id}
                        onSelect={() => setSelectedCitationId(citation.id)}
                        onOpenSection={() => {
                          const target = workspace.report_sections.find(
                            (section) => section.title === citation.section_title,
                          );
                          if (target) {
                            setSelectedSectionId(target.id);
                            setPrimaryView("report");
                          }
                        }}
                        onOpenSource={() => {
                          const target = resolveCitationSource(citation, workspace.sources);
                          if (target) {
                            setSelectedSourceId(target.id);
                            setDetailsOpen(true);
                            setDetailsTab("sources");
                          }
                        }}
                      />
                    ))}
                  </div>
                </details>
              ))}
            </div>
          </section>
          <aside className="workspace-card citations-detail-card">
            <div className="workspace-card-header">
              <h3>Citation detail</h3>
            </div>
            <div className="workspace-phase-summary">
              <article className="phase-summary-card compact">
                <strong>Surviving</strong>
                <span>{citationSummary.surviving}</span>
              </article>
              <article className="phase-summary-card compact">
                <strong>Removed</strong>
                <span>{citationSummary.removed}</span>
              </article>
              <article className="phase-summary-card compact">
                <strong>Uncited sources</strong>
                <span>{citationSummary.uncitedSources}</span>
              </article>
            </div>
            {selectedCitation ? (
              <div className="workspace-stack workspace-scroll-panel">
                <article className={`workspace-inline-card ${selectedCitation.status === "removed" ? "danger" : ""}`}>
                  <strong>{normalizeInlineText(selectedCitation.claim)}</strong>
                  <span>{selectedCitation.section_title}</span>
                  <small>
                    {selectedCitation.status}
                    {selectedCitation.trust_tier ? ` · ${selectedCitation.trust_tier}` : ""}
                  </small>
                </article>
                <article className="workspace-inline-card muted">
                  <strong>Source</strong>
                  <span>{getCitationSourceLabel(selectedCitation, workspace.sources)}</span>
                </article>
                {selectedCitation.quote ? (
                  <blockquote className="citation-quote citation-quote-detail">
                    {normalizeInlineText(selectedCitation.quote)}
                  </blockquote>
                ) : null}
                {selectedCitation.audit_reasons.length ? (
                  <article className="workspace-inline-card danger">
                    <strong>Audit reason</strong>
                    <span>{selectedCitation.audit_reasons.join(", ")}</span>
                  </article>
                ) : null}
                <div className="button-row">
                  <button
                    className="ghost-button"
                    onClick={() => {
                      const target = workspace.report_sections.find(
                        (section) => section.title === selectedCitation.section_title,
                      );
                      if (target) {
                        setSelectedSectionId(target.id);
                        setPrimaryView("report");
                      }
                    }}
                    type="button"
                  >
                    Open section
                  </button>
                  <button
                    className="ghost-button"
                    onClick={() => {
                      const target = resolveCitationSource(
                        selectedCitation,
                        workspace.sources,
                      );
                      if (target) {
                        setSelectedSourceId(target.id);
                        setDetailsOpen(true);
                        setDetailsTab("sources");
                      }
                    }}
                    type="button"
                  >
                    Open source
                  </button>
                </div>
              </div>
            ) : (
              <span className="muted-text">Select a citation to inspect its claim, source, and audit state.</span>
            )}
          </aside>
        </div>
      ) : null}

      {detailsTab === "trace" ? (
        <div className="workspace-grid trace-grid">
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Transport and replay health</h3>
            </div>
            <div className="workspace-phase-summary">
              <article className="phase-summary-card compact">
                <strong>Transport</strong>
                <span>{displayConnectionState}</span>
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
            </div>
          </aside>
        ) : null}
      </div>
    </section>
  );
}

function TaskCard({ task }: { task: WorkspaceTaskView }) {
  const attemptCount =
    typeof task.metadata.attempt_count === "number" ? task.metadata.attempt_count : null;
  const elapsedLabel = formatElapsedMs(task.elapsed_ms);
  const statusLabel =
    task.status === "failed" && attemptCount && attemptCount > 1 ? "failed attempt" : task.status;

  return (
    <article className={`task-card task-${task.status}`}>
      <div className="task-card-header">
        <strong>{titleCase(task.task_type)}</strong>
        <span>{statusLabel}</span>
      </div>
      <span>{task.objective}</span>
      <div className="task-card-meta">
        <small>{task.query_count} queries</small>
        <small>{task.selected_source_count} sources</small>
        <small>{task.notes_produced} notes</small>
        {elapsedLabel ? <small>{elapsedLabel}</small> : null}
        {attemptCount ? <small>attempt {attemptCount}</small> : null}
      </div>
      {task.latest_sources.length ? (
        <small>
          Latest sources: {task.latest_sources.join(" · ")}
        </small>
      ) : null}
      {task.latest_note_summary ? <small>{task.latest_note_summary}</small> : null}
      {task.last_tool_call ? <small>Last tool: {task.last_tool_call}</small> : null}
      {task.last_decision ? <small>Last decision: {task.last_decision}</small> : null}
      {task.next_action ? <small>Next: {task.next_action}</small> : null}
      {task.blocker_reason ? <small className="danger-text">{task.blocker_reason}</small> : null}
    </article>
  );
}

function DecisionCard({ decision }: { decision: WorkspaceDecisionView }) {
  return (
    <article className="decision-card">
      <div className="decision-card-header">
        <strong>{decision.title}</strong>
        <span className="pill muted">{titleCase(decision.category)}</span>
      </div>
      <span>{normalizeInlineText(decision.rationale)}</span>
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
  sources,
  selected,
  onSelect,
  onOpenSection,
  onOpenSource,
}: {
  citation: WorkspaceCitationView;
  sources: WorkspaceSourceView[];
  selected: boolean;
  onSelect: () => void;
  onOpenSection: () => void;
  onOpenSource: () => void;
}) {
  const sourceLabel = getCitationSourceLabel(citation, sources);
  const quote = normalizeInlineText(citation.quote);

  return (
    <article
      className={`citation-card citation-${citation.status} ${selected ? "active" : ""}`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect();
        }
      }}
    >
      <div className="citation-card-header">
        <div className="citation-card-heading">
          <strong>{normalizeInlineText(citation.claim)}</strong>
          <div className="citation-card-meta">
            <span className={`pill ${citation.status === "removed" ? "status-failed" : "muted"}`}>
              {citation.status}
            </span>
            {citation.trust_tier ? <span className="pill muted">{citation.trust_tier}</span> : null}
          </div>
        </div>
      </div>
      <span className="citation-card-source">{sourceLabel}</span>
      {quote ? <blockquote className="citation-quote">{quote}</blockquote> : null}
      {citation.audit_reasons.length ? (
        <small className="danger-text">{citation.audit_reasons.join(", ")}</small>
      ) : null}
      <div className="button-row citation-card-actions">
        <button
          className="ghost-button"
          onClick={(event) => {
            event.stopPropagation();
            onOpenSection();
          }}
          type="button"
        >
          Open section
        </button>
        <button
          className="ghost-button"
          onClick={(event) => {
            event.stopPropagation();
            onOpenSource();
          }}
          type="button"
        >
          Open source
        </button>
      </div>
    </article>
  );
}
