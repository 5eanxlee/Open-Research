"use client";

import {
  cloneElement,
  isValidElement,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ComponentPropsWithoutRef,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactElement,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { deriveReportTitle, isGenericReportTitle } from "@/lib/report-title";
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
type WorkspaceReportPanelTab = WorkspaceDetailsTab | "report";

type ThinkingSubtab = "decisions" | "agents" | "tools" | "files";
type CitationTraceFilter = "referenced" | "read";
type LiveTraceStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "warning"
  | "skipped"
  | "info";

interface LiveTaskView {
  id: string;
  streamId: string | null;
  streamName: string;
  objective: string;
  status: LiveTraceStatus;
  startedAt: string | null;
  updatedAt: string | null;
  queryCount: number;
  selectedSourceCount: number;
  sourceCount: number;
  noteCount: number;
  latestSources: string[];
  latestNoteSummary: string | null;
  lastEventType: string | null;
  blockerReason: string | null;
  model: string | null;
}

interface LiveTaskBuilder extends LiveTaskView {
  eventQueryCount: number;
  eventSelectedSourceCount: number;
  eventSourceCount: number;
  eventNoteCount: number;
}

interface LiveAgentTrace {
  id: string;
  name: string;
  role: string;
  status: LiveTraceStatus;
  summary: string;
  updatedAt: string | null;
  meta: string[];
}

interface LiveToolTrace {
  id: string;
  title: string;
  status: LiveTraceStatus;
  summary: string;
  detail: string | null;
  timestamp: string;
  streamId: string | null;
  streamName: string | null;
  meta: string[];
  rawEvent: RunEvent;
}

interface LiveFileTrace {
  id: string;
  title: string;
  status: LiveTraceStatus;
  summary: string;
  meta: string[];
  timestamp: string | null;
}

interface LiveCitationTrace {
  id: string;
  kind: CitationTraceFilter;
  title: string;
  status: LiveTraceStatus;
  sourceLabel: string;
  url: string | null;
  timestamp: string | null;
  meta: string[];
  quote: string | null;
}

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
  tasks: "Tasks",
  thinking: "Thinking",
  sources: "Sources",
  citations: "Citations",
  trace: "Tool calls",
};

const REPORT_PANEL_TABS: Array<{
  key: WorkspaceReportPanelTab;
  label: string;
}> = [
  { key: "overview", label: "Pipeline" },
  { key: "plan", label: "Plan" },
  { key: "tasks", label: "Tasks" },
  { key: "thinking", label: "Thinking" },
  { key: "sources", label: "Sources" },
  { key: "citations", label: "Citations" },
  { key: "trace", label: "Tool calls" },
  { key: "report", label: "Report" },
];

const DEFAULT_REPORT_PANE_WIDTH = 72;
const MIN_REPORT_PANE_WIDTH = 56;
const MAX_REPORT_PANE_WIDTH = 82;
const MIN_CHAT_PANE_WIDTH_PX = 220;
const REPORT_PANE_GAP_PX = 18;

function clampReportPaneWidth(value: number, max = MAX_REPORT_PANE_WIDTH): number {
  return Math.min(
    max,
    Math.max(MIN_REPORT_PANE_WIDTH, Math.round(value * 10) / 10),
  );
}

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

  for (const embeddedUrl of extractUrlsFromText(citation.claim)) {
    const normalizedEmbeddedUrl = normalizeCitationUrl(embeddedUrl);
    const embeddedMatch = sources.find(
      (source) => normalizeCitationUrl(source.url) === normalizedEmbeddedUrl,
    );
    if (embeddedMatch) {
      return embeddedMatch;
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
  const embeddedUrl = extractUrlsFromText(citation.claim)[0] ?? null;
  return normalizeInlineText(
    citation.source_title ??
      resolvedSource?.title ??
      citation.source_url ??
      resolvedSource?.url ??
      embeddedUrl ??
    "Unknown source",
  );
}

function normalizeCitationUrl(value: string | null | undefined): string {
  return (value ?? "").trim().replace(/\/+$/, "").toLowerCase();
}

function extractUrlsFromText(value: string | null | undefined): string[] {
  return Array.from(normalizeInlineText(value).matchAll(/https?:\/\/[^\s)\];]+/g))
    .map((match) => match[0].replace(/[.,]+$/, ""))
    .filter(Boolean);
}

function formatCitationHost(value: string | null): string | null {
  if (!value) return null;
  try {
    return new URL(value).hostname.replace(/^www\./, "");
  } catch {
    return null;
  }
}

function payloadString(
  payload: Record<string, unknown>,
  keys: string[],
): string | null {
  for (const key of keys) {
    const value = payload[key];
    if (typeof value === "string") {
      const normalized = normalizeInlineText(value);
      if (normalized) return normalized;
    }
    if (typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }
  }
  return null;
}

function payloadNumber(
  payload: Record<string, unknown>,
  keys: string[],
): number | null {
  for (const key of keys) {
    const value = payload[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim()) {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return null;
}

function payloadStringList(
  payload: Record<string, unknown>,
  keys: string[],
): string[] {
  for (const key of keys) {
    const value = payload[key];
    if (Array.isArray(value)) {
      return value
        .map((item) => {
          if (typeof item === "string") return normalizeInlineText(item);
          if (item && typeof item === "object") {
            const record = item as Record<string, unknown>;
            return payloadString(record, ["title", "url", "name", "query"]);
          }
          return "";
        })
        .filter((item): item is string => Boolean(item));
    }
  }
  return [];
}

function payloadRecordList(
  payload: Record<string, unknown>,
  key: string,
): Record<string, unknown>[] {
  const value = payload[key];
  if (!Array.isArray(value)) return [];
  return value.filter(
    (item): item is Record<string, unknown> =>
      Boolean(item) && typeof item === "object" && !Array.isArray(item),
  );
}

function compactMeta(values: Array<string | null | undefined>): string[] {
  return values
    .map((value) => normalizeInlineText(value))
    .filter((value, index, all) => Boolean(value) && all.indexOf(value) === index);
}

function statusFromTaskStatus(value: string | null | undefined): LiveTraceStatus {
  const normalized = normalizeInlineText(value).toLowerCase();
  if (["completed", "complete", "succeeded", "success", "kept"].includes(normalized)) {
    return "completed";
  }
  if (["running", "active", "in_progress", "researching", "planning"].includes(normalized)) {
    return "running";
  }
  if (["failed", "error", "cancelled", "canceled"].includes(normalized)) {
    return "failed";
  }
  if (["skipped", "removed"].includes(normalized)) {
    return "skipped";
  }
  if (["blocked", "warning", "contradicted", "unsupported"].includes(normalized)) {
    return "warning";
  }
  if (normalized === "queued" || normalized === "pending") {
    return "queued";
  }
  return "info";
}

function statusFromEventType(eventType: string): LiveTraceStatus {
  if (
    eventType.endsWith(".failed") ||
    eventType.includes("fetch_failed") ||
    eventType === "run.failed"
  ) {
    return "failed";
  }
  if (
    eventType.endsWith(".started") ||
    eventType === "task.started" ||
    eventType === "stream.created"
  ) {
    return "running";
  }
  if (
    eventType.endsWith(".completed") ||
    eventType.endsWith(".passed") ||
    eventType === "source.fetched" ||
    eventType === "note.saved" ||
    eventType === "report.sanitized"
  ) {
    return "completed";
  }
  if (
    eventType.includes("skipped") ||
    eventType === "source.skipped" ||
    eventType === "citation.removed"
  ) {
    return "skipped";
  }
  if (
    eventType.includes("retry") ||
    eventType.includes("contradicted") ||
    eventType.endsWith(".rejected") ||
    eventType.endsWith(".changes_requested") ||
    eventType === "planning.validation.failed" ||
    eventType === "tool.budget.low"
  ) {
    return "warning";
  }
  return "info";
}

function statusClassName(status: LiveTraceStatus): string {
  return `status-${status}`;
}

function displayEventType(value: string): string {
  return titleCase(value.replace(/^deepagents\./, "").replace(/^provider\./, ""));
}

function eventStreamId(event: RunEvent): string | null {
  return payloadString(event.payload, ["stream_id", "affected_stream_id"]);
}

function eventStreamName(event: RunEvent): string | null {
  return payloadString(event.payload, ["stream_name", "name", "affected_stream_name"]);
}

function eventUrl(event: RunEvent): string | null {
  const direct = payloadString(event.payload, ["url", "source_url", "canonical_url"]);
  if (direct) return direct;
  const urls = extractUrlsFromText(JSON.stringify(event.payload));
  return urls[0] ?? null;
}

function getLatestEventTime(events: RunEvent[], predicate: (event: RunEvent) => boolean): string | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (predicate(events[index])) return events[index].created_at;
  }
  return null;
}

function buildLiveTasks(
  workspace: RunWorkspaceSnapshot,
  events: RunEvent[],
): LiveTaskView[] {
  const taskMap = new Map<string, LiveTaskBuilder>();
  const streamIdByName = new Map<string, string>();
  const hasTerminalReport = events.some((event) => event.event_type === "report.completed");

  const ensureTask = ({
    id,
    streamId,
    streamName,
    objective,
    model,
    timestamp,
  }: {
    id: string;
    streamId: string | null;
    streamName: string | null;
    objective: string | null;
    model?: string | null;
    timestamp?: string | null;
  }): LiveTaskBuilder => {
    const existing = taskMap.get(id);
    if (existing) {
      if (streamName && existing.streamName === "Research stream") existing.streamName = streamName;
      if (objective && !existing.objective) existing.objective = objective;
      if (model && !existing.model) existing.model = model;
      if (timestamp) existing.updatedAt = timestamp;
      return existing;
    }
    const created: LiveTaskBuilder = {
      id,
      streamId,
      streamName: streamName ?? "Research stream",
      objective: objective ?? "Awaiting task details.",
      status: "queued",
      startedAt: timestamp ?? null,
      updatedAt: timestamp ?? null,
      queryCount: 0,
      selectedSourceCount: 0,
      sourceCount: 0,
      noteCount: 0,
      latestSources: [],
      latestNoteSummary: null,
      lastEventType: null,
      blockerReason: null,
      model: model ?? null,
      eventQueryCount: 0,
      eventSelectedSourceCount: 0,
      eventSourceCount: 0,
      eventNoteCount: 0,
    };
    taskMap.set(id, created);
    return created;
  };

  for (const stream of workspace.streams) {
    streamIdByName.set(stream.name, stream.id);
    if (!stream.tasks.length) {
      const fallback = ensureTask({
        id: `stream:${stream.id}`,
        streamId: stream.id,
        streamName: stream.name,
        objective: stream.objective,
        model: stream.model,
      });
      fallback.status = statusFromTaskStatus(stream.status);
      fallback.queryCount = stream.query_count;
      fallback.selectedSourceCount = stream.selected_source_count;
      fallback.sourceCount = stream.selected_source_count;
      fallback.noteCount = stream.note_count;
      fallback.latestSources = stream.latest_source_titles.slice(0, 4);
      fallback.latestNoteSummary = stream.latest_note_summary;
      fallback.updatedAt = getLatestEventTime(
        events,
        (event) => eventStreamId(event) === stream.id,
      );
    }
    for (const task of stream.tasks) {
      const liveTask = ensureTask({
        id: task.id,
        streamId: task.stream_id ?? stream.id,
        streamName: task.stream_name ?? stream.name,
        objective: task.objective || stream.objective,
        model: stream.model,
      });
      liveTask.status = statusFromTaskStatus(task.status);
      liveTask.startedAt = task.started_at ?? liveTask.startedAt;
      liveTask.updatedAt = task.completed_at ?? liveTask.updatedAt;
      liveTask.queryCount = task.query_count;
      liveTask.selectedSourceCount = task.selected_source_count;
      liveTask.sourceCount = task.selected_source_count;
      liveTask.noteCount = task.notes_produced;
      liveTask.latestSources = task.latest_sources.slice(0, 4);
      liveTask.latestNoteSummary = task.latest_note_summary;
      liveTask.lastEventType = task.last_tool_call ?? task.last_decision;
      liveTask.blockerReason = task.blocker_reason;
    }
  }

  for (const event of events) {
    const streamId = eventStreamId(event);
    const streamName = eventStreamName(event);
    const key = streamId
      ? taskMap.has(`stream:${streamId}`)
        ? `stream:${streamId}`
        : Array.from(taskMap.values()).find((task) => task.streamId === streamId)?.id ??
          `stream:${streamId}`
      : streamName
        ? `stream:${streamIdByName.get(streamName) ?? streamName}`
        : null;
    if (!key) continue;

    const task = ensureTask({
      id: key,
      streamId,
      streamName,
      objective: payloadString(event.payload, ["objective", "query", "claim"]),
      model: payloadString(event.payload, ["model"]),
      timestamp: event.created_at,
    });
    task.updatedAt = event.created_at;
    task.lastEventType = event.event_type;

    if (event.event_type === "stream.created") {
      task.status = "queued";
      task.objective = payloadString(event.payload, ["objective"]) ?? task.objective;
      task.model = payloadString(event.payload, ["model"]) ?? task.model;
    } else if (event.event_type === "task.started") {
      task.status = "running";
      task.startedAt = task.startedAt ?? event.created_at;
      task.objective = payloadString(event.payload, ["objective"]) ?? task.objective;
    } else if (event.event_type === "search.performed") {
      task.status = "running";
      task.eventQueryCount += 1;
    } else if (event.event_type === "source.selection.finalized") {
      task.status = "running";
      task.eventSelectedSourceCount = Math.max(
        task.eventSelectedSourceCount,
        payloadNumber(event.payload, ["selected_count"]) ?? 0,
      );
      const selectedTitles = payloadRecordList(event.payload, "selected")
        .map((record) => payloadString(record, ["title", "url"]))
        .filter((item): item is string => Boolean(item));
      task.latestSources = [...selectedTitles, ...task.latestSources].slice(0, 4);
    } else if (
      event.event_type === "source.fetched" ||
      event.event_type === "source.cache.hit" ||
      event.event_type === "claim.repair.source_fetched"
    ) {
      task.status = task.status === "queued" ? "running" : task.status;
      task.eventSourceCount += 1;
      const sourceLabel = payloadString(event.payload, ["title", "url", "source_id"]);
      if (sourceLabel) {
        task.latestSources = [sourceLabel, ...task.latestSources]
          .filter((value, index, all) => all.indexOf(value) === index)
          .slice(0, 4);
      }
    } else if (event.event_type === "note.saved") {
      task.status = task.status === "queued" ? "running" : task.status;
      task.eventNoteCount += 1;
      task.latestNoteSummary = payloadString(event.payload, ["summary"]) ?? task.latestNoteSummary;
    } else if (event.event_type === "input_assets.ingested") {
      task.status = task.status === "queued" ? "running" : task.status;
      task.eventSourceCount += payloadNumber(event.payload, ["count"]) ?? 0;
    } else if (event.event_type === "stream.failed") {
      task.status = "failed";
      task.blockerReason = payloadString(event.payload, ["error", "reason"]);
    } else if (event.event_type === "stream.cancelled") {
      task.status = "failed";
      task.blockerReason = "Stream was cancelled.";
    } else if (event.event_type === "source.skipped" || event.event_type === "source.fetch_failed") {
      task.blockerReason = payloadString(event.payload, ["reason", "error"]) ?? task.blockerReason;
    }
  }

  const tasks = Array.from(taskMap.values()).map((task) => {
    task.queryCount = Math.max(task.queryCount, task.eventQueryCount);
    task.selectedSourceCount = Math.max(
      task.selectedSourceCount,
      task.eventSelectedSourceCount,
      task.eventSourceCount,
    );
    task.sourceCount = Math.max(task.sourceCount, task.eventSourceCount, task.selectedSourceCount);
    task.noteCount = Math.max(task.noteCount, task.eventNoteCount);
    if (
      (hasTerminalReport || workspace.status === "completed") &&
      task.status !== "failed" &&
      task.status !== "skipped"
    ) {
      task.status = "completed";
    }
    return task;
  });

  return tasks.sort((first, second) => {
    const firstTime = new Date(first.startedAt ?? first.updatedAt ?? workspace.created_at).getTime();
    const secondTime = new Date(second.startedAt ?? second.updatedAt ?? workspace.created_at).getTime();
    return firstTime - secondTime;
  });
}

function toolTracePairKey(event: RunEvent): string | null {
  if (event.event_type.startsWith("provider.call.")) {
    return compactMeta([
      "provider",
      payloadString(event.payload, ["provider"]),
      payloadString(event.payload, ["category"]),
      payloadString(event.payload, ["attempt"]),
      payloadString(event.payload, ["query", "url"]),
    ]).join(":");
  }
  if (event.event_type.startsWith("deepagents.tool.")) {
    return compactMeta([
      "deepagents",
      payloadString(event.payload, ["tool"]),
      payloadString(event.payload, ["agent_role"]),
      payloadString(event.payload, ["query_hash", "query"]),
    ]).join(":");
  }
  return null;
}

function describeToolEvent(event: RunEvent): Omit<LiveToolTrace, "id" | "timestamp" | "rawEvent"> | null {
  const payload = event.payload;
  const streamId = eventStreamId(event);
  const streamName = eventStreamName(event);
  const provider = payloadString(payload, ["provider"]);
  const query = payloadString(payload, ["query"]);
  const url = payloadString(payload, ["url", "source_url"]);
  const tool = payloadString(payload, ["tool", "tool_name"]);
  const sourceTitle = payloadString(payload, ["title", "source_title"]);
  const claim = payloadString(payload, ["claim"]);
  const reason = payloadString(payload, ["reason", "error"]);

  if (event.event_type.startsWith("provider.call.")) {
    const category = payloadString(payload, ["category"]) ?? "provider";
    return {
      title: `${titleCase(category)} provider call`,
      status: statusFromEventType(event.event_type),
      summary: query ?? url ?? `${provider ?? "Provider"} ${category} call`,
      detail: reason,
      streamId,
      streamName,
      meta: compactMeta([
        provider,
        payloadString(payload, ["attempt"]) ? `attempt ${payloadString(payload, ["attempt"])}` : null,
        payloadNumber(payload, ["result_count"]) != null
          ? `${payloadNumber(payload, ["result_count"])} results`
          : null,
        payloadNumber(payload, ["elapsed_seconds"]) != null
          ? `${payloadNumber(payload, ["elapsed_seconds"])}s`
          : null,
      ]),
    };
  }

  if (event.event_type.startsWith("deepagents.tool.")) {
    return {
      title: tool ? titleCase(tool) : displayEventType(event.event_type),
      status: statusFromEventType(event.event_type),
      summary: query ?? reason ?? payloadString(payload, ["query_hash"]) ?? "Deep agent tool call",
      detail: null,
      streamId,
      streamName,
      meta: compactMeta([
        payloadString(payload, ["agent_role"]),
        provider,
        payloadNumber(payload, ["source_count"]) != null
          ? `${payloadNumber(payload, ["source_count"])} sources`
          : null,
        payloadString(payload, ["year"]) ? `year ${payloadString(payload, ["year"])}` : null,
      ]),
    };
  }

  if (event.event_type === "search.performed" || event.event_type === "claim.repair.search_performed") {
    return {
      title: event.event_type === "search.performed" ? "Search performed" : "Repair search",
      status: "completed",
      summary: query ?? claim ?? "Search query completed.",
      detail: null,
      streamId,
      streamName,
      meta: compactMeta([
        provider,
        payloadNumber(payload, ["result_count"]) != null
          ? `${payloadNumber(payload, ["result_count"])} results`
          : null,
        payloadStringList(payload, ["result_providers"]).join(", "),
      ]),
    };
  }

  if (event.event_type === "source.selection.finalized") {
    const selected = payloadRecordList(payload, "selected");
    return {
      title: "Source selection",
      status: "completed",
      summary: `${payloadNumber(payload, ["selected_count"]) ?? selected.length} selected from ${
        payloadNumber(payload, ["candidate_count"]) ?? "candidate"
      } results`,
      detail: selected
        .map((record) => payloadString(record, ["title", "url"]))
        .filter((item): item is string => Boolean(item))
        .slice(0, 5)
        .join(" | ") || null,
      streamId,
      streamName,
      meta: compactMeta([
        payloadNumber(payload, ["per_domain_limit"]) != null
          ? `domain cap ${payloadNumber(payload, ["per_domain_limit"])}`
          : null,
      ]),
    };
  }

  if (
    [
      "source.cache.hit",
      "source.fetch_failed",
      "source.fallback_document.created",
      "source.skipped",
      "source.fetched",
      "claim.repair.source_fetched",
    ].includes(event.event_type)
  ) {
    return {
      title: displayEventType(event.event_type),
      status: statusFromEventType(event.event_type),
      summary: sourceTitle ?? url ?? payloadString(payload, ["source_id"]) ?? "Source event",
      detail: reason,
      streamId,
      streamName,
      meta: compactMeta([
        provider ?? payloadString(payload, ["search_provider"]),
        payloadString(payload, ["trust_tier"]),
        payloadString(payload, ["discovered_via"]),
        payloadString(payload, ["stage"]),
      ]),
    };
  }

  if (event.event_type === "note.saved") {
    return {
      title: "Note saved",
      status: "completed",
      summary: payloadString(payload, ["summary"]) ?? "Worker note saved.",
      detail: null,
      streamId,
      streamName,
      meta: compactMeta([
        payloadString(payload, ["source_id"]),
        payloadNumber(payload, ["confidence"]) != null
          ? `confidence ${payloadNumber(payload, ["confidence"])}`
          : null,
        payloadString(payload, ["trust_tier"]),
      ]),
    };
  }

  if (
    [
      "passages.reranked",
      "claim.repair.started",
      "claim.repair.completed",
      "claim.repair.skipped",
      "citation.verification.skipped",
      "citation.verified",
      "citation.contradicted",
      "citation.removed",
      "citation.audit.completed",
      "report.drafted",
      "report.sanitized",
      "input_assets.ingested",
      "tool.budget.low",
      "memory.retrieved",
      "memory.compiled",
      "context.fragment.dropped",
      "context.pack.created",
      "completion_gate.evaluated",
      "completion_gate.continuation_requested",
      "completion_gate.continuation_applied",
      "completion_gate.exhausted",
    ].includes(event.event_type)
  ) {
    return {
      title: displayEventType(event.event_type),
      status: statusFromEventType(event.event_type),
      summary:
        claim ??
        payloadString(payload, ["section_title", "summary", "reason", "tool_name"]) ??
        normalizeInlineText(JSON.stringify(payload)),
      detail: reason,
      streamId,
      streamName,
      meta: compactMeta([
        provider,
        payloadNumber(payload, ["candidate_count"]) != null
          ? `${payloadNumber(payload, ["candidate_count"])} candidates`
          : null,
        payloadNumber(payload, ["returned_count"]) != null
          ? `${payloadNumber(payload, ["returned_count"])} returned`
          : null,
        payloadNumber(payload, ["citation_count"]) != null
          ? `${payloadNumber(payload, ["citation_count"])} citations`
          : null,
        payloadNumber(payload, ["removed_citations"]) != null
          ? `${payloadNumber(payload, ["removed_citations"])} removed`
          : null,
      ]),
    };
  }

  return null;
}

function buildLiveToolTraces(events: RunEvent[]): LiveToolTrace[] {
  const traces = new Map<string, LiveToolTrace>();

  for (const event of events) {
    const description = describeToolEvent(event);
    if (!description) continue;
    const pairedKey = toolTracePairKey(event);
    const key = pairedKey ? `pair:${pairedKey}` : `event:${event.id}`;
    const existing = traces.get(key);
    const nextTrace: LiveToolTrace = {
      id: key,
      timestamp: event.created_at,
      rawEvent: event,
      ...description,
    };
    if (existing) {
      traces.set(key, {
        ...existing,
        ...nextTrace,
        meta: compactMeta([...existing.meta, ...nextTrace.meta]),
        detail: nextTrace.detail ?? existing.detail,
      });
    } else {
      traces.set(key, nextTrace);
    }
  }

  return Array.from(traces.values()).sort(
    (first, second) =>
      new Date(second.timestamp).getTime() - new Date(first.timestamp).getTime(),
  );
}

function buildLiveAgents(
  workspace: RunWorkspaceSnapshot,
  events: RunEvent[],
): LiveAgentTrace[] {
  const agents = new Map<string, LiveAgentTrace>();
  const hasReportCompleted = workspace.status === "completed" || events.some(
    (event) => event.event_type === "report.completed",
  );

  const upsert = (agent: LiveAgentTrace) => {
    const existing = agents.get(agent.id);
    if (!existing) {
      agents.set(agent.id, agent);
      return;
    }
    agents.set(agent.id, {
      ...existing,
      ...agent,
      meta: compactMeta([...existing.meta, ...agent.meta]),
      updatedAt: agent.updatedAt ?? existing.updatedAt,
    });
  };

  const planningEvents = events.filter(
    (event) => event.event_type.startsWith("planning.") || event.event_type.startsWith("plan."),
  );
  if (workspace.plan.plan_preview || workspace.plan.approved_plan || planningEvents.length) {
    const latest = planningEvents[planningEvents.length - 1];
    upsert({
      id: "planner",
      name: "Planner",
      role: "Planning agent",
      status:
        workspace.current_phase === "plan" && workspace.status !== "completed"
          ? "running"
          : latest?.event_type === "planning.validation.failed"
            ? "warning"
            : workspace.plan.approved_plan || latest?.event_type === "plan.created"
              ? "completed"
              : "queued",
      summary:
        workspace.plan.approved_plan?.summary ??
        workspace.plan.plan_preview?.summary ??
        "Building the research plan.",
      updatedAt: latest?.created_at ?? null,
      meta: compactMeta([
        workspace.plan.plan_preview
          ? `${workspace.plan.plan_preview.plan.streams.length} preview streams`
          : null,
        workspace.plan.approved_plan
          ? `${workspace.plan.approved_plan.streams.length} approved streams`
          : null,
        latest?.event_type,
      ]),
    });
  }

  for (const stream of workspace.streams) {
    const latest = getLatestEventTime(events, (event) => eventStreamId(event) === stream.id);
    upsert({
      id: `stream:${stream.id}`,
      name: stream.name,
      role: "Research worker",
      status: statusFromTaskStatus(stream.status),
      summary: stream.objective,
      updatedAt: latest,
      meta: compactMeta([
        stream.model,
        `${stream.query_count} queries`,
        `${stream.selected_source_count} sources`,
        `${stream.note_count} notes`,
      ]),
    });
  }

  const citationEvents = events.filter(
    (event) => event.event_type.startsWith("citation.") || event.event_type.startsWith("claim.repair."),
  );
  if (citationEvents.length || workspace.citations.length) {
    const latest = citationEvents[citationEvents.length - 1];
    upsert({
      id: "verifier",
      name: "Citation verifier",
      role: "Grounding agent",
      status:
        workspace.current_phase === "ground" && !hasReportCompleted
          ? "running"
          : workspace.citations.length || hasReportCompleted
            ? "completed"
            : "queued",
      summary: "Checks claims against retrieved passages and repairs unsupported claims.",
      updatedAt: latest?.created_at ?? null,
      meta: compactMeta([
        `${workspace.citations.length} citations`,
        latest?.event_type,
      ]),
    });
  }

  const reportEvents = events.filter((event) => event.event_type.startsWith("report."));
  if (reportEvents.length || workspace.final_report_markdown) {
    const latest = reportEvents[reportEvents.length - 1];
    upsert({
      id: "writer",
      name: "Report writer",
      role: "Synthesis agent",
      status: workspace.final_report_markdown ? "completed" : "running",
      summary: "Drafts, sanitizes, and publishes the final research report.",
      updatedAt: latest?.created_at ?? null,
      meta: compactMeta([
        workspace.report_sections.length ? `${workspace.report_sections.length} sections` : null,
        latest?.event_type,
      ]),
    });
  }

  for (const event of events) {
    const role = payloadString(event.payload, ["agent_role"]);
    if (!role) continue;
    const existing = agents.get(`agent:${role}`);
    upsert({
      id: `agent:${role}`,
      name: titleCase(role),
      role: "DeepAgents role",
      status:
        event.event_type === "deepagents.tool.failed"
          ? "failed"
          : event.event_type === "deepagents.tool.started"
            ? "running"
            : "completed",
      summary: payloadString(event.payload, ["tool", "query"]) ?? "Deep agent activity",
      updatedAt: event.created_at,
      meta: compactMeta([
        ...(existing?.meta ?? []),
        payloadString(event.payload, ["provider"]),
        payloadString(event.payload, ["tool"]),
      ]),
    });
  }

  return Array.from(agents.values()).sort((first, second) => {
    const order = ["planner", "writer", "verifier"];
    const firstOrder = order.indexOf(first.id);
    const secondOrder = order.indexOf(second.id);
    if (firstOrder !== -1 || secondOrder !== -1) {
      return (firstOrder === -1 ? 99 : firstOrder) - (secondOrder === -1 ? 99 : secondOrder);
    }
    return first.name.localeCompare(second.name);
  });
}

function buildLiveFiles(
  workspace: RunWorkspaceSnapshot,
  events: RunEvent[],
): LiveFileTrace[] {
  const assets = workspace.project_assets_available.concat(workspace.run_assets_available);
  const fileTraces: LiveFileTrace[] = assets.map((asset) => ({
    id: asset.id,
    title: asset.label,
    status: statusFromTaskStatus(asset.processing_status),
    summary: asset.preview_excerpt ?? asset.description ?? asset.processing_error ?? "Research asset",
    meta: compactMeta([
      asset.project_id ? "Project corpus" : "Run asset",
      asset.usage.replaceAll("_", " "),
      asset.extraction_method,
      asset.page_count != null ? `${asset.page_count} pages` : null,
    ]),
    timestamp: asset.updated_at,
  }));

  for (const event of events) {
    if (event.event_type === "input_assets.ingested") {
      fileTraces.unshift({
        id: `event:${event.id}`,
        title: "Input assets ingested",
        status: "completed",
        summary: `${payloadNumber(event.payload, ["count"]) ?? 0} assets added to stream context.`,
        meta: compactMeta([eventStreamId(event)]),
        timestamp: event.created_at,
      });
    }
    if (event.event_type === "context.pack.created") {
      fileTraces.unshift({
        id: `event:${event.id}`,
        title: "Context pack created",
        status: "completed",
        summary: payloadString(event.payload, ["summary", "phase"]) ?? "Context pack assembled.",
        meta: compactMeta([
          payloadString(event.payload, ["phase"]),
          payloadNumber(event.payload, ["used_tokens"]) != null
            ? `${payloadNumber(event.payload, ["used_tokens"])} tokens`
            : null,
        ]),
        timestamp: event.created_at,
      });
    }
    if (event.event_type === "context.fragment.dropped") {
      fileTraces.unshift({
        id: `event:${event.id}`,
        title: "Context fragment dropped",
        status: "skipped",
        summary: payloadString(event.payload, ["title", "reason", "dropped_reason"]) ?? "Context fragment removed.",
        meta: compactMeta([payloadString(event.payload, ["phase"])]),
        timestamp: event.created_at,
      });
    }
  }

  return fileTraces;
}

function citationTraceFromWorkspaceCitation(
  citation: WorkspaceCitationView,
  sources: WorkspaceSourceView[],
): LiveCitationTrace {
  const url = citation.source_url ?? resolveCitationSource(citation, sources)?.url ?? null;
  return {
    id: `citation:${citation.id}`,
    kind: "referenced",
    title: cleanCitationClaimText(citation.claim),
    status: statusFromTaskStatus(citation.status),
    sourceLabel: getCitationSourceLabel(citation, sources),
    url,
    timestamp: null,
    meta: compactMeta([
      citation.section_title,
      citation.support_label,
      citation.trust_tier,
      citation.citation_number ? `#${citation.citation_number}` : null,
    ]),
    quote: citation.quote,
  };
}

function buildLiveCitationTraces(
  workspace: RunWorkspaceSnapshot,
  events: RunEvent[],
): LiveCitationTrace[] {
  const traces = new Map<string, LiveCitationTrace>();

  for (const citation of workspace.citations) {
    const trace = citationTraceFromWorkspaceCitation(citation, workspace.sources);
    traces.set(trace.id, trace);
  }

  for (const source of workspace.sources) {
    const label = normalizeInlineText(source.title ?? source.url);
    if (!label && !source.url) continue;
    traces.set(`source:${source.id}`, {
      id: `source:${source.id}`,
      kind: "read",
      title: label || "Source",
      status: statusFromTaskStatus(source.state),
      sourceLabel: formatCitationHost(source.url) ?? source.provider ?? "Source",
      url: source.url,
      timestamp: null,
      meta: compactMeta([
        source.provider,
        source.origin.replaceAll("_", " "),
        source.trust_tier,
        source.stream_names.join(", "),
      ]),
      quote: source.note_summaries[0] ?? null,
    });
  }

  for (const event of events) {
    const url = eventUrl(event);
    if (
      [
        "citation.verified",
        "citation.verification.skipped",
        "citation.contradicted",
        "citation.removed",
      ].includes(event.event_type)
    ) {
      const claim = payloadString(event.payload, ["claim"]) ?? displayEventType(event.event_type);
      const sectionTitle = payloadString(event.payload, ["section_title"]);
      traces.set(`citation-event:${event.id}`, {
        id: `citation-event:${event.id}`,
        kind: "referenced",
        title: cleanCitationClaimText(claim),
        status: statusFromEventType(event.event_type),
        sourceLabel:
          payloadString(event.payload, ["source_title", "source_id"]) ??
          sectionTitle ??
          "Claim audit",
        url,
        timestamp: event.created_at,
        meta: compactMeta([
          sectionTitle,
          payloadString(event.payload, ["support_label"]),
          payloadString(event.payload, ["reason"]),
          payloadNumber(event.payload, ["repair_attempts"]) != null
            ? `${payloadNumber(event.payload, ["repair_attempts"])} repairs`
            : null,
        ]),
        quote: payloadString(event.payload, ["quote"]),
      });
    }

    if (
      [
        "source.selection.finalized",
        "source.cache.hit",
        "source.fetch_failed",
        "source.fallback_document.created",
        "source.skipped",
        "source.fetched",
        "claim.repair.source_fetched",
      ].includes(event.event_type)
    ) {
      const selected = payloadRecordList(event.payload, "selected");
      if (event.event_type === "source.selection.finalized" && selected.length) {
        selected.forEach((record, index) => {
          const selectedUrl = payloadString(record, ["url"]);
          traces.set(`selected:${event.id}:${index}`, {
            id: `selected:${event.id}:${index}`,
            kind: "read",
            title: payloadString(record, ["title", "url"]) ?? "Selected source",
            status: "queued",
            sourceLabel: formatCitationHost(selectedUrl) ?? payloadString(record, ["provider"]) ?? "Selected",
            url: selectedUrl,
            timestamp: event.created_at,
            meta: compactMeta([
              eventStreamName(event),
              payloadString(record, ["provider"]),
              payloadNumber(record, ["query_order"]) != null
                ? `query ${payloadNumber(record, ["query_order"])}`
                : null,
            ]),
            quote: null,
          });
        });
        continue;
      }
      traces.set(`source-event:${event.id}`, {
        id: `source-event:${event.id}`,
        kind: "read",
        title: payloadString(event.payload, ["title", "url", "source_id"]) ?? displayEventType(event.event_type),
        status: statusFromEventType(event.event_type),
        sourceLabel:
          formatCitationHost(url) ??
          payloadString(event.payload, ["provider", "search_provider", "source_id"]) ??
          "Source",
        url,
        timestamp: event.created_at,
        meta: compactMeta([
          eventStreamName(event),
          payloadString(event.payload, ["provider", "search_provider"]),
          payloadString(event.payload, ["trust_tier"]),
          payloadString(event.payload, ["reason", "discovered_via"]),
        ]),
        quote: payloadString(event.payload, ["error"]),
      });
    }
  }

  return Array.from(traces.values()).sort((first, second) => {
    const firstTime = first.timestamp ? new Date(first.timestamp).getTime() : 0;
    const secondTime = second.timestamp ? new Date(second.timestamp).getTime() : 0;
    return secondTime - firstTime;
  });
}

function getMarkdownTitle(content: string, fallback: string): string {
  const match = content.match(/^#\s+(.+?)\s*$/m);
  return normalizeInlineText(match?.[1] ?? fallback);
}

function stripSourceProseArtifacts(value: string): string {
  return value
    .replace(
      /\b(This (?:draft|report) uses only [^.]*?retrieved (?:arxiv\s+)?records?\/?excerpts?):\s*[^.]+\./gi,
      "This report is based on retrieved source excerpts.",
    )
    .replace(
      /\b(The retrieved record set contains[^.:;\n]*?)\s*:\s*[^;\n]+;/gi,
      "$1;",
    )
    .replace(/\b(?:at|on|from)\s+arXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?/gi, "")
    .replace(/\barXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?/gi, "")
    .replace(/\barXiv\s+records?\/excerpts?/gi, "source excerpts")
    .replace(/\barXiv\s+record\b/gi, "source record")
    .replace(/\barXiv\s+records\b/gi, "source records")
    .replace(/\s+([,.;:])/g, "$1")
    .replace(/([.;:]){2,}/g, "$1")
    .replace(/[ \t]{2,}/g, " ");
}

function prepareReportMarkdown(content: string, replacementTitle?: string): string {
  let prepared = content;
  if (replacementTitle && !isGenericReportTitle(replacementTitle)) {
    prepared = prepared.replace(/^#\s+(.+?)\s*$/m, (match, title) =>
      isGenericReportTitle(normalizeInlineText(title)) ? `# ${replacementTitle}` : match,
    );
  }
  return stripSourceProseArtifacts(prepared)
    .replace(/(?:^|\n)##\s+Citations\s*[\s\S]*$/i, "")
    .replace(/\[([^\]]+)\]\(https?:\/\/[^)]+\)/g, "$1")
    .replace(
      /\s*\bSources?:\s*(?:https?:\/\/[^\s)\]]+\s*(?:[;,]\s*)?)+(?:\s*\((partial|full)\s+support\))?/gi,
      (_match, support: string | undefined) => (support ? ` (${support} support)` : ""),
    )
    .replace(
      /\s*\bSource:\s*(?:https?:\/\/[^\s)\]]+\s*(?:[;,]\s*)?)+(?:\s*\((partial|full)\s+support\))?/gi,
      (_match, support: string | undefined) => (support ? ` (${support} support)` : ""),
    )
    .replace(/\s*https?:\/\/[^\s)\];]+[^\s)\];.,]/g, "")
    .replace(
      /\s*\bSources?:\s*(?:[;,.]\s*)+(?:\((partial|full)\s+support\))?/gi,
      (_match, support: string | undefined) => (support ? ` (${support} support)` : ""),
    )
    .replace(
      /\s*\bSource:\s*(?:[;,.]\s*)+(?:\((partial|full)\s+support\))?/gi,
      (_match, support: string | undefined) => (support ? ` (${support} support)` : ""),
    )
    .replace(/\s*\bSources?:\s*[^.\n]*?(?=\s*(?:\[\d+\]|\[(?:\d+\]\s*){2,}|\n|$))/gi, "")
    .replace(/\s*\bSource:\s*[^.\n]*?(?=\s*(?:\[\d+\]|\[(?:\d+\]\s*){2,}|\n|$))/gi, "")
    .replace(/\s+([,.;:])/g, "$1")
    .replace(/([.;:]){2,}/g, "$1")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function cleanCitationClaimText(value: string): string {
  return normalizeInlineText(prepareReportMarkdown(value));
}

function truncateSentence(value: string, maxLength = 420): string {
  if (value.length <= maxLength) return value;
  const clipped = value.slice(0, maxLength);
  const sentenceEnd = Math.max(
    clipped.lastIndexOf(". "),
    clipped.lastIndexOf("; "),
    clipped.lastIndexOf(": "),
  );
  return `${clipped.slice(0, sentenceEnd > 180 ? sentenceEnd + 1 : maxLength).trim()}...`;
}

function getReportLead(content: string, replacementTitle?: string): string {
  const prepared = prepareReportMarkdown(content, replacementTitle);
  const paragraphs = prepared
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
  const lead =
    paragraphs.find(
      (paragraph) =>
        !paragraph.startsWith("#") &&
        !paragraph.startsWith("|") &&
        !paragraph.startsWith("- ") &&
        !paragraph.startsWith("* "),
    ) ?? "";
  return truncateSentence(lead.replace(/\s+/g, " "));
}

type CitationSource = {
  id: string;
  label: string;
  url: string | null;
  trustTier: string | null;
};

type CitationLookup = Map<number, CitationSource[]>;

function buildWorkspaceCitationLookup(
  citations: WorkspaceCitationView[],
  sources: WorkspaceSourceView[],
): CitationLookup {
  const lookup: CitationLookup = new Map();
  const numberByUrl = new Map<string, number>();
  let nextNumber = 1;

  for (const citation of citations) {
    if (citation.status === "removed") continue;
    const normalizedUrl = normalizeCitationUrl(citation.source_url);
    let citationNumber = citation.citation_number;
    if (!citationNumber && normalizedUrl) {
      citationNumber = numberByUrl.get(normalizedUrl) ?? nextNumber;
      if (!numberByUrl.has(normalizedUrl)) {
        numberByUrl.set(normalizedUrl, citationNumber);
        nextNumber += 1;
      }
    }
    if (!citationNumber) continue;
    const entries = lookup.get(citationNumber) ?? [];
    const id = citation.id;
    if (!entries.some((entry) => entry.id === id)) {
      entries.push({
        id,
        label: getCitationSourceLabel(citation, sources),
        url: citation.source_url,
        trustTier: citation.trust_tier,
      });
    }
    lookup.set(citationNumber, entries);
  }

  sources.forEach((source, index) => {
    const citationNumber = index + 1;
    if (lookup.has(citationNumber)) return;
    const label = normalizeInlineText(source.title ?? source.url ?? `Source ${citationNumber}`);
    if (!label && !source.url) return;
    lookup.set(citationNumber, [
      {
        id: source.id || source.source_id || `source-${citationNumber}`,
        label,
        url: source.url,
        trustTier: source.trust_tier,
      },
    ]);
  });

  return lookup;
}

function CitationCluster({
  numbers,
  lookup,
}: {
  numbers: number[];
  lookup: CitationLookup;
}) {
  const clusterRef = useRef<HTMLButtonElement | null>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const closeTimerRef = useRef<number | null>(null);
  const [open, setOpen] = useState(false);
  const [popoverStyle, setPopoverStyle] = useState<CSSProperties>({});
  const sources = numbers.flatMap((number) => lookup.get(number) ?? []);
  const label = numbers.map((number) => `[${number}]`).join("");
  const dedupedSources = sources.filter(
    (source, index) =>
      sources.findIndex(
        (candidate) =>
          candidate.id === source.id ||
          (candidate.url && source.url && candidate.url === source.url),
      ) === index,
  );

  const clearCloseTimer = useCallback(() => {
    if (closeTimerRef.current == null) return;
    window.clearTimeout(closeTimerRef.current);
    closeTimerRef.current = null;
  }, []);

  const updatePopoverPosition = useCallback(() => {
    const element = clusterRef.current;
    if (!element) return;
    const rect = element.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const width = Math.min(430, Math.max(260, viewportWidth - 24));
    const left = Math.min(Math.max(12, rect.left), viewportWidth - width - 12);
    const estimatedHeight = Math.min(340, 92 + dedupedSources.length * 46);
    const opensBelow = rect.bottom + 10 + estimatedHeight < viewportHeight;
    const top = opensBelow
      ? rect.bottom + 10
      : Math.max(12, rect.top - estimatedHeight - 10);
    setPopoverStyle({
      left,
      maxHeight: Math.min(360, viewportHeight - 24),
      top,
      width,
    });
  }, [dedupedSources.length]);

  const openPopover = useCallback(() => {
    clearCloseTimer();
    setOpen(true);
  }, [clearCloseTimer]);

  const scheduleClose = useCallback(() => {
    clearCloseTimer();
    closeTimerRef.current = window.setTimeout(() => {
      const clusterHovered = clusterRef.current?.matches(":hover");
      const popoverHovered = popoverRef.current?.matches(":hover");
      if (!clusterHovered && !popoverHovered) {
        setOpen(false);
      }
    }, 90);
  }, [clearCloseTimer]);

  useLayoutEffect(() => {
    if (!open) return;
    updatePopoverPosition();
  }, [open, updatePopoverPosition]);

  useEffect(() => {
    if (!open) return undefined;
    window.addEventListener("resize", updatePopoverPosition);
    window.addEventListener("scroll", updatePopoverPosition, true);
    return () => {
      window.removeEventListener("resize", updatePopoverPosition);
      window.removeEventListener("scroll", updatePopoverPosition, true);
    };
  }, [open, updatePopoverPosition]);

  useEffect(() => () => clearCloseTimer(), [clearCloseTimer]);

  const popover =
    open && dedupedSources.length
      ? createPortal(
          <div
            className="citation-popover"
            onMouseEnter={openPopover}
            onMouseLeave={scheduleClose}
            ref={popoverRef}
            role="tooltip"
            style={popoverStyle}
          >
            <span className="citation-popover-heading">
              {dedupedSources.length} source{dedupedSources.length === 1 ? "" : "s"}
            </span>
            {dedupedSources.map((source) =>
              source.url ? (
                <a
                  className="citation-popover-link"
                  href={source.url}
                  key={`${source.id}-${source.url}`}
                  rel="noreferrer"
                  target="_blank"
                >
                  <strong>{source.label}</strong>
                  <small>{source.trustTier ?? formatCitationHost(source.url) ?? "Link"}</small>
                </a>
              ) : (
                <span className="citation-popover-link" key={source.id}>
                  <strong>{source.label}</strong>
                  {source.trustTier ? <small>{source.trustTier}</small> : null}
                </span>
              ),
            )}
          </div>,
          document.body,
        )
      : null;

  return (
    <button
      aria-expanded={open}
      aria-label={`Show sources ${label}`}
      className="citation-cluster"
      onBlur={scheduleClose}
      onClick={openPopover}
      onFocus={openPopover}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openPopover();
        }
        if (event.key === "Escape") {
          setOpen(false);
        }
      }}
      onMouseEnter={openPopover}
      onMouseLeave={scheduleClose}
      ref={clusterRef}
      type="button"
    >
      <span className="citation-cluster-label">{label}</span>
      {popover}
    </button>
  );
}

function replaceCitationText(value: string, lookup: CitationLookup): ReactNode[] {
  const nodes: ReactNode[] = [];
  const citationPattern = /(?:\[\d+\]\s*)+/g;
  let cursor = 0;
  for (const match of value.matchAll(citationPattern)) {
    const index = match.index ?? 0;
    if (index > cursor) nodes.push(value.slice(cursor, index));
    const numbers = Array.from(match[0].matchAll(/\[(\d+)\]/g))
      .map((numberMatch) => Number(numberMatch[1]))
      .filter((number) => Number.isFinite(number));
    nodes.push(
      <CitationCluster
        key={`citation-${index}-${numbers.join("-")}`}
        lookup={lookup}
        numbers={numbers}
      />,
    );
    cursor = index + match[0].length;
  }
  if (cursor < value.length) nodes.push(value.slice(cursor));
  return nodes;
}

function renderCitationNodes(children: ReactNode, lookup: CitationLookup): ReactNode {
  if (typeof children === "string") {
    return replaceCitationText(children, lookup);
  }
  if (Array.isArray(children)) {
    return children.map((child, index) => (
      <span key={`citation-node-${index}`}>{renderCitationNodes(child, lookup)}</span>
    ));
  }
  if (isValidElement<{ children?: ReactNode }>(children)) {
    const element = children as ReactElement<{ children?: ReactNode }>;
    if (!element.props.children) return element;
    return cloneElement(element, {
      children: renderCitationNodes(element.props.children, lookup),
    });
  }
  return children;
}

function MermaidDiagram({ chart }: { chart: string }) {
  const id = useId().replace(/:/g, "");
  const [svg, setSvg] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let mounted = true;
    setFailed(false);
    void import("mermaid")
      .then(({ default: mermaid }) => {
        mermaid.initialize({ startOnLoad: false, securityLevel: "strict" });
        return mermaid.render(`mermaid-${id}`, chart);
      })
      .then(({ svg: renderedSvg }) => {
        if (mounted) setSvg(renderedSvg);
      })
      .catch(() => {
        if (mounted) setFailed(true);
      });
    return () => {
      mounted = false;
    };
  }, [chart, id]);

  if (failed) {
    return <pre className="mermaid-fallback">{chart}</pre>;
  }
  if (!svg) {
    return <div className="mermaid-loading">Rendering diagram...</div>;
  }
  return (
    <div
      className="mermaid-diagram"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
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
  citationLookup,
  hideCitationBibliography = false,
  replacementTitle,
}: {
  content: string;
  className?: string;
  citationLookup?: CitationLookup;
  hideCitationBibliography?: boolean;
  replacementTitle?: string;
}) {
  const displayContent = hideCitationBibliography
    ? prepareReportMarkdown(content, replacementTitle)
    : content;
  const citations = citationLookup ?? new Map();
  const withCitations = (children: ReactNode) => renderCitationNodes(children, citations);

  return (
    <div className={`markdown-body workspace-markdown${className ? ` ${className}` : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a: ({ node: _node, ...props }) => (
            <a {...props} rel="noreferrer" target="_blank" />
          ),
          p: ({ node: _node, children, ...props }) => (
            <p {...props}>{withCitations(children)}</p>
          ),
          li: ({ node: _node, children, ...props }) => (
            <li {...props}>{withCitations(children)}</li>
          ),
          td: ({ node: _node, children, ...props }) => (
            <td {...props}>{withCitations(children)}</td>
          ),
          th: ({ node: _node, children, ...props }) => (
            <th {...props}>{withCitations(children)}</th>
          ),
          code: ({
            className: codeClassName,
            children,
            ...props
          }: ComponentPropsWithoutRef<"code"> & { node?: unknown }) => {
            const codeText = String(children ?? "").replace(/\n$/, "");
            if (codeClassName?.includes("language-mermaid")) {
              return <MermaidDiagram chart={codeText} />;
            }
            return (
              <code className={codeClassName} {...props}>
                {children}
              </code>
            );
          },
          table: ({ node: _node, children, ...props }) => (
            <div className="markdown-table-wrap">
              <table {...props}>{children as ReactNode}</table>
            </div>
          ),
          img: ({ node: _node, ...props }) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img {...props} className="markdown-visual" alt={props.alt ?? ""} />
          ),
        }}
      >
        {displayContent}
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
  isLoading?: boolean;
  errorMessage?: string | null;
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
  isLoading = false,
  errorMessage = null,
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
  const [reportPanelTab, setReportPanelTab] = useState<WorkspaceReportPanelTab>("report");
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [focusedPhase, setFocusedPhase] = useState<WorkspacePhaseKey | null>(null);
  const [selectedStreamId, setSelectedStreamId] = useState<string>("all");
  const [thinkingSubtab, setThinkingSubtab] = useState<ThinkingSubtab>("decisions");
  const [citationTraceFilter, setCitationTraceFilter] =
    useState<CitationTraceFilter>("referenced");
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [selectedSectionId, setSelectedSectionId] = useState<string | null>(null);
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);
  const [clarificationDraft, setClarificationDraft] = useState("");
  const [approvalNote, setApprovalNote] = useState("");
  const [reportActionNotice, setReportActionNotice] = useState<string | null>(null);
  const detailsScrollRef = useRef<HTMLDivElement | null>(null);
  const shellLayoutRef = useRef<HTMLDivElement | null>(null);
  const [reportPaneWidth, setReportPaneWidth] = useState(DEFAULT_REPORT_PANE_WIDTH);

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
    setReportPanelTab(workspace.final_report_markdown ? "report" : "tasks");
    setDetailsOpen(false);
    setFocusedPhase(null);
    setSelectedStreamId("all");
    setThinkingSubtab("decisions");
    setCitationTraceFilter("referenced");
    setReportActionNotice(null);
  }, [workspace?.run_id]);

  useEffect(() => {
    if (!reportActionNotice) return;
    const timeout = window.setTimeout(() => setReportActionNotice(null), 2200);
    return () => window.clearTimeout(timeout);
  }, [reportActionNotice]);

  useEffect(() => {
    if (!detailsOpen) return;
    detailsScrollRef.current?.scrollTo({ top: 0, behavior: "auto" });
  }, [detailsOpen, detailsTab, workspace?.run_id]);

  const workspaceShellStyle = useMemo(
    () =>
      ({
        "--report-pane-width": `${reportPaneWidth}%`,
      }) as CSSProperties,
    [reportPaneWidth],
  );

  const handleReportResizePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (primaryView !== "report") return;
      const layout = shellLayoutRef.current;
      if (!layout) return;

      event.preventDefault();

      const updateWidth = (clientX: number) => {
        const rect = layout.getBoundingClientRect();
        if (!rect.width) return;
        const dynamicMax = Math.min(
          MAX_REPORT_PANE_WIDTH,
          ((rect.width - MIN_CHAT_PANE_WIDTH_PX - REPORT_PANE_GAP_PX) / rect.width) * 100,
        );
        const nextWidth = ((rect.right - clientX) / rect.width) * 100;
        setReportPaneWidth(clampReportPaneWidth(nextWidth, dynamicMax));
      };

      updateWidth(event.clientX);
      document.body.classList.add("is-resizing-report-pane");

      const handlePointerMove = (moveEvent: PointerEvent) => {
        moveEvent.preventDefault();
        updateWidth(moveEvent.clientX);
      };

      const handlePointerUp = () => {
        document.body.classList.remove("is-resizing-report-pane");
        window.removeEventListener("pointermove", handlePointerMove);
        window.removeEventListener("pointerup", handlePointerUp);
        window.removeEventListener("pointercancel", handlePointerUp);
      };

      window.addEventListener("pointermove", handlePointerMove);
      window.addEventListener("pointerup", handlePointerUp, { once: true });
      window.addEventListener("pointercancel", handlePointerUp, { once: true });
    },
    [primaryView],
  );

  const handleReportResizeKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (primaryView !== "report") return;
      const layoutWidth = shellLayoutRef.current?.getBoundingClientRect().width ?? 0;
      const dynamicMax = layoutWidth
        ? Math.min(
            MAX_REPORT_PANE_WIDTH,
            ((layoutWidth - MIN_CHAT_PANE_WIDTH_PX - REPORT_PANE_GAP_PX) / layoutWidth) *
              100,
          )
        : MAX_REPORT_PANE_WIDTH;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setReportPaneWidth((current) => clampReportPaneWidth(current + 2, dynamicMax));
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        setReportPaneWidth((current) => clampReportPaneWidth(current - 2, dynamicMax));
      }
      if (event.key === "Home") {
        event.preventDefault();
        setReportPaneWidth(clampReportPaneWidth(MIN_REPORT_PANE_WIDTH, dynamicMax));
      }
      if (event.key === "End") {
        event.preventDefault();
        setReportPaneWidth(clampReportPaneWidth(MAX_REPORT_PANE_WIDTH, dynamicMax));
      }
    },
    [primaryView],
  );

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

  const liveTasks = useMemo(
    () => (workspace ? buildLiveTasks(workspace, rawEvents) : []),
    [rawEvents, workspace],
  );

  const liveToolTraces = useMemo(
    () => buildLiveToolTraces(phaseFilteredEvents),
    [phaseFilteredEvents],
  );

  const liveAgents = useMemo(
    () => (workspace ? buildLiveAgents(workspace, rawEvents) : []),
    [rawEvents, workspace],
  );

  const liveFiles = useMemo(
    () => (workspace ? buildLiveFiles(workspace, rawEvents) : []),
    [rawEvents, workspace],
  );

  const liveCitationTraces = useMemo(
    () => (workspace ? buildLiveCitationTraces(workspace, rawEvents) : []),
    [rawEvents, workspace],
  );

  const liveStreamFilters = useMemo(() => {
    const filters = new Map<string, { id: string; name: string; count: number }>();
    for (const stream of workspace?.streams ?? []) {
      filters.set(stream.id, { id: stream.id, name: stream.name, count: 0 });
    }
    for (const task of liveTasks) {
      const id = task.streamId ?? task.streamName;
      const name = task.streamName;
      const existing = filters.get(id) ?? { id, name, count: 0 };
      existing.count += 1;
      filters.set(id, existing);
    }
    return Array.from(filters.values());
  }, [liveTasks, workspace?.streams]);

  const visibleLiveTasks = useMemo(() => {
    if (selectedStreamId === "all") return liveTasks;
    return liveTasks.filter(
      (task) => task.streamId === selectedStreamId || task.streamName === selectedStreamId,
    );
  }, [liveTasks, selectedStreamId]);

  const liveTaskSummary = useMemo(() => {
    const completed = liveTasks.filter((task) => task.status === "completed").length;
    const running = liveTasks.filter((task) => task.status === "running").length;
    const failed = liveTasks.filter((task) => task.status === "failed").length;
    const total = liveTasks.length;
    const progress = total ? Math.round((completed / total) * 100) : 0;
    return { completed, failed, progress, running, total };
  }, [liveTasks]);

  const visibleLiveCitationTraces = useMemo(
    () => liveCitationTraces.filter((trace) => trace.kind === citationTraceFilter),
    [citationTraceFilter, liveCitationTraces],
  );

  if (!workspace && isLoading) {
    return (
      <section className="panel workspace-panel workspace-loading-panel" aria-busy="true">
        <div className="workspace-loading-grid">
          <div className="workspace-loading-main">
            <span className="skeleton skeleton-kicker" />
            <span className="skeleton skeleton-title" />
            <span className="skeleton skeleton-line" />
            <span className="skeleton skeleton-line short" />
          </div>
          <div className="workspace-loading-side">
            {Array.from({ length: 5 }).map((_, index) => (
              <span className="skeleton skeleton-card" key={index} />
            ))}
          </div>
        </div>
      </section>
    );
  }

  if (!workspace && errorMessage) {
    return (
      <section className="panel workspace-panel workspace-empty-panel">
        <div className="workspace-empty">
          <p className="eyebrow">Workspace</p>
          <h2 className="workspace-empty-title">Run unavailable</h2>
          <p className="workspace-empty-copy">{errorMessage}</p>
        </div>
      </section>
    );
  }

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
  const markdownReportTitle = getMarkdownTitle(reportMarkdown, "");
  const reportTitleCandidate = workspace.report_title ?? markdownReportTitle;
  const reportTitle = isGenericReportTitle(reportTitleCandidate)
    ? deriveReportTitle(workspace.question)
    : reportTitleCandidate;
  const displayReportMarkdown = prepareReportMarkdown(reportMarkdown, reportTitle);
  const reportLead = getReportLead(reportMarkdown, reportTitle);
  const reportFilename = `${slugifyFilename(reportTitle || workspace.question)}-report.md`;
  const reportCitationLookup = buildWorkspaceCitationLookup(workspace.citations, workspace.sources);
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
      await navigator.clipboard.writeText(displayReportMarkdown);
      setReportActionNotice("Report copied.");
    } catch {
      setReportActionNotice("Copy failed.");
    }
  };

  const handleDownloadReport = () => {
    if (!reportMarkdown) return;
    const blob = new Blob([displayReportMarkdown], { type: "text/markdown;charset=utf-8" });
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

  const renderReportPanelContent = () => {
    if (reportPanelTab === "report") {
      return (
        <MarkdownContent
          className="workspace-report-markdown"
          content={reportMarkdown || "_No final report available yet._"}
          citationLookup={reportCitationLookup}
          hideCitationBibliography
          replacementTitle={reportTitle}
        />
      );
    }

    if (reportPanelTab === "overview") {
      return (
        <div className="workspace-report-detail">
          <header className="workspace-report-detail-header">
            <span>Pipeline</span>
            <h1>Run Pipeline</h1>
            <p>{activePhase?.blocked_reason ?? "Phase status, blockers, stream activity, and report progress for this run."}</p>
          </header>
          <section className="workspace-report-detail-section">
            <div className="workspace-report-section-head">
              <h2>Phase Summary</h2>
              <span>{workspace.connection.event_count} events</span>
            </div>
            <div className="workspace-report-phase-grid">
              {workspace.phases.map((phase) => (
                <article className={`workspace-report-phase-card ${phase.status}`} key={phase.key}>
                  <strong>{phase.label}</strong>
                  <span>{phase.status.replaceAll("_", " ")}</span>
                  <small>
                    {phase.blocked_reason ??
                      (phase.completed_at ? formatTime(phase.completed_at) : "No blockers")}
                  </small>
                  <em>{phase.event_count}</em>
                </article>
              ))}
            </div>
          </section>
          <section className="workspace-report-detail-grid two">
            <article className="workspace-report-detail-section">
              <div className="workspace-report-section-head">
                <h2>Blockers And Health</h2>
              </div>
              <div className="workspace-report-row-list">
                {workspace.asset_processing_errors.map((error) => (
                  <article className="workspace-report-row danger" key={error}>
                    <strong>Asset processing issue</strong>
                    <span>{error}</span>
                  </article>
                ))}
                {activePhase?.blocked_reason ? (
                  <article className="workspace-report-row warning">
                    <strong>Current blocker</strong>
                    <span>{activePhase.blocked_reason}</span>
                  </article>
                ) : null}
                {workspace.asset_processing_errors.length === 0 && !activePhase?.blocked_reason ? (
                  <article className="workspace-report-row">
                    <strong>No active blockers</strong>
                    <span>The run is flowing normally.</span>
                  </article>
                ) : null}
              </div>
            </article>
            <article className="workspace-report-detail-section">
              <div className="workspace-report-section-head">
                <h2>Report Progress</h2>
                <span>{workspace.report_sections.length} sections</span>
              </div>
              <div className="workspace-report-row-list">
                {workspace.report_sections.map((section) => (
                  <article className="workspace-report-row" key={section.id}>
                    <strong>{section.title}</strong>
                    <span>
                      {section.grounded_claim_count} grounded / {section.unsupported_claim_count} open
                    </span>
                    <small>{section.citation_count} citations · {section.removed_citation_count} removed</small>
                  </article>
                ))}
              </div>
            </article>
          </section>
          <section className="workspace-report-detail-section">
            <div className="workspace-report-section-head">
              <h2>Recent Decisions</h2>
              <span>{visibleDecisions.length}</span>
            </div>
            <div className="workspace-report-row-list">
              {visibleDecisions.slice(0, 8).map((decision) => (
                <DecisionCard decision={decision} key={decision.id} />
              ))}
              {visibleDecisions.length === 0 ? (
                <article className="workspace-report-row">
                  <strong>No decisions recorded yet</strong>
                  <span>Decision records will appear as planning, execution, grounding, and audit steps run.</span>
                </article>
              ) : null}
            </div>
          </section>
        </div>
      );
    }

    if (reportPanelTab === "plan") {
      const streams = workspace.plan.plan_preview?.plan.streams ?? workspace.plan.approved_plan?.streams ?? [];
      const planningEvents = rawEvents.filter(
        (event) =>
          event.event_type.startsWith("planning.") ||
          event.event_type.startsWith("plan.") ||
          event.event_type.startsWith("clarification."),
      );
      return (
        <div className="workspace-report-detail">
          <header className="workspace-report-detail-header">
            <span>Plan</span>
            <h1>Research Plan</h1>
            <p>{workspace.plan.plan_preview?.summary ?? workspace.plan.approved_plan?.summary ?? "No plan preview has been generated yet."}</p>
          </header>
          <section className="workspace-report-detail-grid two">
            <article className="workspace-report-detail-section">
              <div className="workspace-report-section-head">
                <h2>Plan Preview</h2>
              </div>
              {workspace.plan.plan_preview ? (
                <div className="workspace-report-row-list">
                  <article className="workspace-report-row">
                    <strong>{workspace.plan.plan_preview.summary}</strong>
                    <span>{workspace.plan.plan_preview.hypothesis}</span>
                    <small>{workspace.plan.plan_preview.budget_decision_reason}</small>
                  </article>
                  {workspace.plan.approved_plan ? (
                    <article className="workspace-report-row muted">
                      <strong>Approved plan</strong>
                      <span>{workspace.plan.approved_plan.summary}</span>
                      <small>{workspace.plan.approved_plan.hypothesis}</small>
                    </article>
                  ) : null}
                </div>
              ) : (
                <p className="muted-text">Plan preview will appear here when planning starts.</p>
              )}
            </article>
            <article className="workspace-report-detail-section">
              <div className="workspace-report-section-head">
                <h2>Budget</h2>
              </div>
              <div className="workspace-report-metric-grid">
                <article>
                  <span>Requested</span>
                  <strong>
                    {workspace.plan.requested_budget
                      ? `${workspace.plan.requested_budget.max_streams} / ${workspace.plan.requested_budget.max_queries_per_stream}`
                      : "n/a"}
                  </strong>
                </article>
                <article>
                  <span>Recommended</span>
                  <strong>
                    {workspace.plan.recommended_budget
                      ? `${workspace.plan.recommended_budget.max_streams} / ${workspace.plan.recommended_budget.max_queries_per_stream}`
                      : "n/a"}
                  </strong>
                </article>
                <article>
                  <span>Effective</span>
                  <strong>
                    {workspace.plan.effective_budget
                      ? `${workspace.plan.effective_budget.max_streams} / ${workspace.plan.effective_budget.max_queries_per_stream}`
                      : "n/a"}
                  </strong>
                </article>
              </div>
              <p className="muted-text">{workspace.plan.budget_decision_reason ?? "No explicit budget clamp reason."}</p>
            </article>
          </section>
          <section className="workspace-report-detail-section">
            <div className="workspace-report-section-head">
              <h2>Planning Trace</h2>
              <span>{planningEvents.length}</span>
            </div>
            <div className="workspace-report-row-list">
              {planningEvents
                .slice()
                .reverse()
                .slice(0, 12)
                .map((event) => (
                  <article
                    className={`workspace-report-row ${statusClassName(statusFromEventType(event.event_type))}`}
                    key={event.id}
                  >
                    <div className="workspace-live-card-head">
                      <div>
                        <strong>{displayEventType(event.event_type)}</strong>
                        <span>{formatTime(event.created_at)}</span>
                      </div>
                      <em>{statusFromEventType(event.event_type)}</em>
                    </div>
                    <details className="workspace-live-disclosure">
                      <summary>Details</summary>
                      <p>
                        {payloadString(event.payload, ["summary", "rationale", "query", "note"]) ??
                          "Planning event payload"}
                      </p>
                      <pre>{JSON.stringify(event.payload, null, 2)}</pre>
                    </details>
                  </article>
                ))}
              {planningEvents.length === 0 ? (
                <article className="workspace-report-row">
                  <strong>No planning events yet</strong>
                  <span>Plan preview, validation, discovery, and approval events will appear here.</span>
                </article>
              ) : null}
            </div>
          </section>
          <section className="workspace-report-detail-section">
            <div className="workspace-report-section-head">
              <h2>Planned Streams</h2>
              <span>{streams.length}</span>
            </div>
            <div className="workspace-report-row-list">
              {streams.map((stream) => (
                <article className="workspace-report-row" key={`${stream.name}-${stream.objective}`}>
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
                  ) : null}
                </article>
              ))}
              {streams.length === 0 ? (
                <article className="workspace-report-row">
                  <strong>No streams yet</strong>
                  <span>Streams will be listed after planning.</span>
                </article>
              ) : null}
            </div>
          </section>
          <section className="workspace-report-detail-grid two">
            <article className="workspace-report-detail-section">
              <div className="workspace-report-section-head">
                <h2>Clarifications</h2>
              </div>
              <div className="workspace-report-row-list">
                {workspace.plan.clarification_session?.questions.map((question) => {
                  const answer = workspace.plan.clarification_session?.turns.find(
                    (turn) => turn.question_id === question.id,
                  );
                  return (
                    <article className="workspace-report-row" key={question.id}>
                      <strong>{question.prompt}</strong>
                      <span>{answer?.response ?? "Awaiting response"}</span>
                      <small>{question.rationale}</small>
                    </article>
                  );
                }) ?? (
                  <article className="workspace-report-row">
                    <strong>No clarification required</strong>
                    <span>This run proceeded without a clarification round.</span>
                  </article>
                )}
              </div>
            </article>
            <article className="workspace-report-detail-section">
              <div className="workspace-report-section-head">
                <h2>Approval History</h2>
              </div>
              <div className="workspace-report-row-list">
                {workspace.plan.approval_history.map((decision, index) => (
                  <article className="workspace-report-row" key={`${decision.decision}-${decision.created_at}-${index}`}>
                    <strong>{titleCase(decision.decision)}</strong>
                    <span>{decision.note ?? "No note provided."}</span>
                    <small>{formatTime(decision.created_at)}</small>
                  </article>
                ))}
                {workspace.plan.approval_history.length === 0 ? (
                  <article className="workspace-report-row">
                    <strong>No approval actions</strong>
                    <span>Approval history will appear after plan review actions.</span>
                  </article>
                ) : null}
              </div>
            </article>
          </section>
        </div>
      );
    }

    if (reportPanelTab === "tasks") {
      return (
        <div className="workspace-report-detail">
          <header className="workspace-report-detail-header">
            <span>Tasks</span>
            <h1>Execution Board</h1>
            <p>
              {liveTaskSummary.running} running, {liveTaskSummary.completed} complete,{" "}
              {liveTaskSummary.failed} blocked across {liveTaskSummary.total} live task records.
            </p>
          </header>
          <section className="workspace-live-summary">
            <div className="workspace-live-progress-head">
              <strong>{liveTaskSummary.progress}% complete</strong>
              <span>
                {liveTaskSummary.completed}/{liveTaskSummary.total || 0} tasks
              </span>
            </div>
            <div className="workspace-live-progress-track" aria-hidden>
              <span style={{ width: `${liveTaskSummary.progress}%` }} />
            </div>
            <div className="workspace-live-summary-grid">
              <article>
                <strong>{liveTaskSummary.running}</strong>
                <span>Running</span>
              </article>
              <article>
                <strong>{liveTaskSummary.completed}</strong>
                <span>Complete</span>
              </article>
              <article>
                <strong>{liveTaskSummary.failed}</strong>
                <span>Blocked</span>
              </article>
              <article>
                <strong>{workspace.connection.event_count}</strong>
                <span>Events</span>
              </article>
            </div>
          </section>
          <section className="workspace-report-detail-section">
            <div className="workspace-report-section-head">
              <h2>Streams</h2>
              <span>{selectedStreamId === "all" ? "All streams" : "Filtered"}</span>
            </div>
            <div className="workspace-report-filter-row">
              <button
                className={`workspace-report-filter ${selectedStreamId === "all" ? "active" : ""}`}
                onClick={() => setSelectedStreamId("all")}
                type="button"
              >
                All streams
              </button>
              {liveStreamFilters.map((stream) => (
                <button
                  className={`workspace-report-filter ${selectedStreamId === stream.id ? "active" : ""}`}
                  key={stream.id}
                  onClick={() => setSelectedStreamId(stream.id)}
                  type="button"
                >
                  {stream.name}
                </button>
              ))}
            </div>
            <div className="workspace-live-task-grid">
              {visibleLiveTasks.map((task) => (
                <LiveTaskCard key={task.id} task={task} />
              ))}
              {visibleLiveTasks.length === 0 ? (
                <article className="workspace-report-row">
                  <strong>No live tasks yet</strong>
                  <span>Stream and task events will appear here as soon as the backend emits them.</span>
                </article>
              ) : null}
            </div>
          </section>
        </div>
      );
    }

    if (reportPanelTab === "thinking") {
      return (
        <div className="workspace-report-detail">
          <header className="workspace-report-detail-header">
            <span>Thinking</span>
            <h1>Structured Reasoning</h1>
            <p>
              {liveAgents.length} agents, {liveToolTraces.length} tool traces,{" "}
              {liveFiles.length} context files and packs captured during this run.
            </p>
          </header>
          <section className="workspace-report-detail-section">
            <div className="workspace-report-section-head">
              <h2>{titleCase(thinkingSubtab)}</h2>
            </div>
            <div className="workspace-report-filter-row">
              {(["decisions", "agents", "tools", "files"] as ThinkingSubtab[]).map((subtab) => (
                <button
                  className={`workspace-report-filter ${thinkingSubtab === subtab ? "active" : ""}`}
                  key={subtab}
                  onClick={() => setThinkingSubtab(subtab)}
                  type="button"
                >
                  {titleCase(subtab)}
                </button>
              ))}
            </div>
            <div className="workspace-report-row-list">
              {thinkingSubtab === "decisions"
                ? visibleDecisions.map((decision) => (
                    <DecisionCard decision={decision} key={decision.id} />
                  ))
                : null}
              {thinkingSubtab === "tools"
                ? liveToolTraces.map((trace) => (
                    <LiveToolTraceCard key={trace.id} trace={trace} />
                  ))
                : null}
              {thinkingSubtab === "agents"
                ? liveAgents.map((agent) => (
                    <LiveAgentCard agent={agent} key={agent.id} />
                  ))
                : null}
              {thinkingSubtab === "files"
                ? liveFiles.map((file) => (
                    <LiveFileCard file={file} key={file.id} />
                  ))
                : null}
              {thinkingSubtab === "decisions" && visibleDecisions.length === 0 ? (
                <article className="workspace-report-row">
                  <strong>No decision records yet</strong>
                  <span>Planner and verifier decisions will appear after they are written to the workspace.</span>
                </article>
              ) : null}
              {thinkingSubtab === "tools" && liveToolTraces.length === 0 ? (
                <article className="workspace-report-row">
                  <strong>No tool traces yet</strong>
                  <span>Provider calls, searches, source fetches, reranks, repairs, and DeepAgents tool calls stream here.</span>
                </article>
              ) : null}
              {thinkingSubtab === "agents" && liveAgents.length === 0 ? (
                <article className="workspace-report-row">
                  <strong>No agent activity yet</strong>
                  <span>Planner, stream worker, verifier, writer, and DeepAgents roles will appear as the run starts.</span>
                </article>
              ) : null}
              {thinkingSubtab === "files" && liveFiles.length === 0 ? (
                <article className="workspace-report-row">
                  <strong>No file context yet</strong>
                  <span>Uploaded assets, context packs, and dropped fragments will appear here.</span>
                </article>
              ) : null}
            </div>
          </section>
        </div>
      );
    }

    if (reportPanelTab === "sources") {
      return (
        <div className="workspace-report-detail">
          <header className="workspace-report-detail-header">
            <span>Sources</span>
            <h1>Source Provenance</h1>
            <p>{workspace.sources.length} total sources across project corpus, run attachments, fetched web results, and cited records.</p>
          </header>
          <section className="workspace-report-detail-grid two">
            <article className="workspace-report-detail-section">
              <div className="workspace-report-section-head">
                <h2>Source Lanes</h2>
              </div>
              <div className="workspace-report-row-list">
                {sourceLanes.map(([label, items]) => (
                  <article className="workspace-report-row" key={label}>
                    <strong>{label}</strong>
                    <span>{items.length} sources</span>
                    <div className="workspace-report-source-list">
                      {items.slice(0, 8).map((source) => (
                        <button
                          className={`workspace-report-source-pill ${selectedSource?.id === source.id ? "active" : ""}`}
                          key={source.id}
                          onClick={() => setSelectedSourceId(source.id)}
                          type="button"
                        >
                          {source.title ?? source.url}
                        </button>
                      ))}
                    </div>
                  </article>
                ))}
              </div>
            </article>
            <article className="workspace-report-detail-section">
              <div className="workspace-report-section-head">
                <h2>Source Detail</h2>
              </div>
              {selectedSource ? (
                <div className="workspace-report-row-list">
                  <article className="workspace-report-row">
                    <strong>{selectedSource.title ?? selectedSource.url}</strong>
                    <span>{selectedSource.url}</span>
                    <small>
                      {selectedSource.origin.replaceAll("_", " ")} · {selectedSource.state} · {selectedSource.provider ?? "unknown provider"}
                    </small>
                  </article>
                  <div className="workspace-report-metric-grid">
                    <article>
                      <span>Trust</span>
                      <strong>{selectedSource.trust_tier ?? "unknown"}</strong>
                    </article>
                    <article>
                      <span>Passages</span>
                      <strong>{selectedSource.passages_used}</strong>
                    </article>
                    <article>
                      <span>Sections</span>
                      <strong>{selectedSource.report_sections.length}</strong>
                    </article>
                  </div>
                  {selectedSource.note_summaries.map((summary, index) => (
                    <article className="workspace-report-row" key={`${selectedSource.id}-summary-${index}`}>
                      <strong>Note summary</strong>
                      <span>{summary}</span>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="muted-text">Select a source to inspect its provenance.</p>
              )}
            </article>
          </section>
        </div>
      );
    }

    if (reportPanelTab === "citations") {
      const referencedCount = liveCitationTraces.filter((trace) => trace.kind === "referenced").length;
      const readCount = liveCitationTraces.filter((trace) => trace.kind === "read").length;
      return (
        <div className="workspace-report-detail">
          <header className="workspace-report-detail-header">
            <span>Citations</span>
            <h1>Citation Audit</h1>
            <p>
              {referencedCount} referenced claim traces and {readCount} read source traces, with{" "}
              {citationSummary.removed} removed during audit.
            </p>
          </header>
          <section className="workspace-report-detail-grid citations">
            <article className="workspace-report-detail-section">
              <div className="workspace-report-section-head">
                <h2>{citationTraceFilter === "referenced" ? "Referenced" : "Read"}</h2>
                <span>{visibleLiveCitationTraces.length}</span>
              </div>
              <div className="workspace-report-filter-row">
                <button
                  className={`workspace-report-filter ${citationTraceFilter === "referenced" ? "active" : ""}`}
                  onClick={() => setCitationTraceFilter("referenced")}
                  type="button"
                >
                  Referenced
                </button>
                <button
                  className={`workspace-report-filter ${citationTraceFilter === "read" ? "active" : ""}`}
                  onClick={() => setCitationTraceFilter("read")}
                  type="button"
                >
                  Read
                </button>
              </div>
              <div className="workspace-report-row-list">
                {visibleLiveCitationTraces.map((trace) => (
                  <LiveCitationTraceCard key={trace.id} trace={trace} />
                ))}
                {visibleLiveCitationTraces.length === 0 ? (
                  <article className="workspace-report-row">
                    <strong>No {citationTraceFilter} traces yet</strong>
                    <span>
                      {citationTraceFilter === "referenced"
                        ? "Citation verification and final report references will stream here."
                        : "Fetched, cached, skipped, and selected source records will stream here."}
                    </span>
                  </article>
                ) : null}
              </div>
            </article>
            <aside className="workspace-report-detail-section sticky">
              <div className="workspace-report-section-head">
                <h2>Final Audit Detail</h2>
              </div>
              {selectedCitation ? (
                <div className="workspace-report-row-list">
                  <article className={`workspace-report-row ${selectedCitation.status === "removed" ? "danger" : ""}`}>
                    <strong>{cleanCitationClaimText(selectedCitation.claim)}</strong>
                    <span>{selectedCitation.section_title}</span>
                    <small>
                      {selectedCitation.status}
                      {selectedCitation.trust_tier ? ` · ${selectedCitation.trust_tier}` : ""}
                    </small>
                  </article>
                  <article className="workspace-report-row muted">
                    <strong>Source</strong>
                    <span>{getCitationSourceLabel(selectedCitation, workspace.sources)}</span>
                  </article>
                  {selectedCitation.quote ? (
                    <blockquote className="citation-quote citation-quote-detail">
                      {normalizeInlineText(selectedCitation.quote)}
                    </blockquote>
                  ) : null}
                  {selectedCitation.audit_reasons.length ? (
                    <article className="workspace-report-row danger">
                      <strong>Audit reason</strong>
                      <span>{selectedCitation.audit_reasons.join(", ")}</span>
                    </article>
                  ) : null}
                </div>
              ) : (
                <p className="muted-text">Select a citation to inspect its claim, source, and audit state.</p>
              )}
            </aside>
          </section>
        </div>
      );
    }

    return (
      <div className="workspace-report-detail">
        <header className="workspace-report-detail-header">
          <span>Tool calls</span>
          <h1>Realtime Trace</h1>
          <p>Structured provider calls, searches, source reads, grounding checks, and backend trace payloads captured for this run.</p>
        </header>
        <section className="workspace-report-detail-grid two">
          <article className="workspace-report-detail-section">
            <div className="workspace-report-section-head">
              <h2>Transport</h2>
            </div>
            <div className="workspace-report-metric-grid">
              <article>
                <span>Status</span>
                <strong>{displayConnectionState}</strong>
              </article>
              <article>
                <span>Mode</span>
                <strong>{connectionMode ?? workspace.connection.stream_mode ?? "n/a"}</strong>
              </article>
              <article>
                <span>Last ID</span>
                <strong>{workspace.connection.last_event_id}</strong>
              </article>
              <article>
                <span>Backend</span>
                <strong>{workspace.connection.workflow_backend ?? "local"}</strong>
              </article>
            </div>
          </article>
          <article className="workspace-report-detail-section">
            <div className="workspace-report-section-head">
              <h2>Tool And Trace Events</h2>
              <span>{liveToolTraces.length}</span>
            </div>
            <div className="workspace-report-event-list">
              {liveToolTraces.map((trace) => (
                <LiveToolTraceCard key={trace.id} trace={trace} />
              ))}
              {liveToolTraces.length === 0 ? (
                <article className="workspace-report-row">
                  <strong>No trace events yet</strong>
                  <span>Backend tool events will appear here once execution starts.</span>
                </article>
              ) : null}
            </div>
          </article>
        </section>
      </div>
    );
  };

  return (
    <section
      className={`panel workspace-panel workspace-shell-panel ${
        detailsOpen ? "activity-open" : ""
      } ${primaryView === "report" ? "report-open" : ""}`}
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

      <div
        ref={shellLayoutRef}
        style={workspaceShellStyle}
        className={`workspace-shell-layout ${detailsOpen ? "with-details" : ""} ${
          primaryView === "report" ? "with-report" : ""
        }`}
      >
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
                        aria-label="Plan approval note"
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
                        aria-label="Clarification response"
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
                  <article className="thread-item conversation-turn assistant report-turn report-summary-turn">
                    <div className="conversation-turn-header">
                      <strong>Assistant</strong>
                      <span className="report-ready-label">Report complete</span>
                    </div>
                    <div className="report-summary-body">
                      <h2>{reportTitle}</h2>
                      {reportLead ? <p>{reportLead}</p> : null}
                      <div className="report-summary-stats" aria-label="Report summary">
                        <span>{workspace.sources.length} sources</span>
                        <span>{workspace.citations.length} citations</span>
                        <span>{workspace.connection.event_count} events</span>
                      </div>
                    </div>
                    {reportActionNotice ? (
                      <span className="workspace-inline-feedback">{reportActionNotice}</span>
                    ) : null}
                    <div className="workspace-report-actions report-actions-bottom">
                      <button
                        className="primary-button"
                        onClick={() => {
                          setDetailsOpen(false);
                          setReportPanelTab("report");
                          setPrimaryView("report");
                        }}
                        type="button"
                      >
                        View Report
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
                        Markdown
                      </button>
                    </div>
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
        </section>

        <button
          className="workspace-research-edge-tab"
          onClick={() => {
            setDetailsOpen(false);
            setReportPanelTab(reportReady ? "report" : "tasks");
            setPrimaryView("report");
          }}
          type="button"
        >
          Show Research
        </button>

        <aside
          aria-hidden={primaryView !== "report"}
          aria-label="Research report and activity"
          className="workspace-report-panel"
        >
            <div
              aria-label="Resize report panel"
              aria-orientation="vertical"
              aria-valuemax={MAX_REPORT_PANE_WIDTH}
              aria-valuemin={MIN_REPORT_PANE_WIDTH}
              aria-valuenow={Math.round(reportPaneWidth)}
              className="workspace-report-resize-handle"
              onKeyDown={handleReportResizeKeyDown}
              onPointerDown={handleReportResizePointerDown}
              role="separator"
              tabIndex={primaryView === "report" ? 0 : -1}
              title="Drag to resize report"
            />
            <div className="workspace-report-panel-topbar">
              <div className="workspace-report-panel-tabs" role="tablist" aria-label="Report views">
                {REPORT_PANEL_TABS.map((tab) => (
                  <button
                    aria-selected={reportPanelTab === tab.key}
                    className={`workspace-report-panel-tab ${
                      reportPanelTab === tab.key ? "active" : ""
                    }`}
                    key={tab.key}
                    onClick={() => {
                      setReportPanelTab(tab.key);
                      if (tab.key !== "report") {
                        setDetailsOpen(false);
                        setDetailsTab(tab.key);
                      }
                    }}
                    role="tab"
                    tabIndex={primaryView === "report" ? 0 : -1}
                    type="button"
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
              <button
                aria-label="Stop researching"
                className="workspace-report-stop"
                disabled={
                  workspace.status === "completed" ||
                  workspace.status === "failed" ||
                  workspace.status === "cancelled"
                }
                onClick={() => onCancel?.(workspace.run_id)}
                title="Stop researching"
                tabIndex={primaryView === "report" ? 0 : -1}
                type="button"
              >
                Stop Researching
              </button>
              <button
                aria-label="Close report"
                className="workspace-report-close"
                onClick={() => setPrimaryView("chat")}
                tabIndex={primaryView === "report" ? 0 : -1}
                type="button"
              >
                <span aria-hidden />
              </button>
            </div>
            {reportActionNotice ? (
              <span className="workspace-inline-feedback">{reportActionNotice}</span>
            ) : null}
            <article className="workspace-report-canvas">
              {renderReportPanelContent()}
            </article>
            {reportPanelTab === "report" ? (
              <div className="workspace-report-footer-actions">
                <button
                  className="ghost-button"
                  disabled={!reportMarkdown}
                  onClick={handleDownloadReport}
                  tabIndex={primaryView === "report" ? 0 : -1}
                  type="button"
                >
                  Markdown
                </button>
                <button
                  className="ghost-button"
                  disabled={!reportMarkdown}
                  onClick={handleCopyReport}
                  tabIndex={primaryView === "report" ? 0 : -1}
                  type="button"
                >
                  Copy Memo
                </button>
              </div>
            ) : null}
          </aside>

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

            <div className="workspace-details-scroll" ref={detailsScrollRef}>
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

              <div className="workspace-tabs" role="tablist" aria-label="Research activity views">
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
                    aria-selected={detailsTab === tab}
                    className={`workspace-tab ${detailsTab === tab ? "active" : ""}`}
                    key={tab}
                    onClick={() => setDetailsTab(tab)}
                    role="tab"
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
              <strong>{liveTasks.length}</strong>
            </button>
            {liveStreamFilters.map((stream) => (
              <button
                className={`ops-rail-button ${selectedStreamId === stream.id ? "active" : ""}`}
                key={stream.id}
                onClick={() => setSelectedStreamId(stream.id)}
                type="button"
              >
                <div>
                  <span>{stream.name}</span>
                  <small>{stream.count} tasks</small>
                </div>
                <strong>{stream.count}</strong>
              </button>
            ))}
          </aside>
          <section className="workspace-card">
            <div className="workspace-card-header">
              <h3>Live execution board</h3>
              {focusedPhase ? <span className="pill muted">{titleCase(focusedPhase)}</span> : null}
            </div>
            <div className="workspace-live-task-grid">
              {visibleLiveTasks.map((task) => (
                <LiveTaskCard key={task.id} task={task} />
              ))}
              {visibleLiveTasks.length === 0 ? (
                <span className="muted-text">No live task records for this stream yet.</span>
              ) : null}
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
                {liveToolTraces.map((trace) => (
                  <LiveToolTraceCard key={trace.id} trace={trace} />
                ))}
                {liveToolTraces.length === 0 ? (
                  <span className="muted-text">No tool traces emitted yet.</span>
                ) : null}
              </div>
            ) : thinkingSubtab === "agents" ? (
              <div className="workspace-list">
                {liveAgents.map((agent) => (
                  <LiveAgentCard agent={agent} key={agent.id} />
                ))}
                {liveAgents.length === 0 ? (
                  <span className="muted-text">No agent traces emitted yet.</span>
                ) : null}
              </div>
            ) : (
              <div className="workspace-list">
                {liveFiles.map((file) => (
                  <LiveFileCard file={file} key={file.id} />
                ))}
                {liveFiles.length === 0 ? (
                  <span className="muted-text">No file context traces emitted yet.</span>
                ) : null}
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
                            setReportPanelTab("report");
                            setPrimaryView("report");
                          }
                        }}
                        onOpenSource={() => {
                          const target = resolveCitationSource(citation, workspace.sources);
                          if (target) {
                            setSelectedSourceId(target.id);
                            setReportPanelTab("sources");
                            setPrimaryView("report");
                            setDetailsOpen(false);
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
                  <strong>{cleanCitationClaimText(selectedCitation.claim)}</strong>
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
                        setReportPanelTab("report");
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
                        setReportPanelTab("sources");
                        setPrimaryView("report");
                        setDetailsOpen(false);
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
            </div>
          </aside>
        ) : null}
      </div>
    </section>
  );
}

function LiveTaskCard({ task }: { task: LiveTaskView }) {
  const hasDetails =
    Boolean(task.objective) ||
    task.latestSources.length > 0 ||
    Boolean(task.latestNoteSummary) ||
    Boolean(task.blockerReason) ||
    Boolean(task.lastEventType);

  return (
    <article className={`workspace-live-task-card ${statusClassName(task.status)}`}>
      <div className="workspace-live-card-head">
        <div>
          <strong>{task.streamName}</strong>
          <span>{task.model ?? "Research worker"}</span>
        </div>
        <em>{task.status}</em>
      </div>
      <div className="workspace-live-metrics">
        <span>{task.queryCount} queries</span>
        <span>{task.selectedSourceCount || task.sourceCount} sources</span>
        <span>{task.noteCount} notes</span>
        {task.updatedAt ? <span>{formatTime(task.updatedAt)}</span> : null}
      </div>
      {hasDetails ? (
        <details className="workspace-live-disclosure">
          <summary>Details</summary>
          {task.objective ? <p>{task.objective}</p> : null}
          {task.latestSources.length ? (
            <div className="workspace-live-source-strip">
              {task.latestSources.map((source) => (
                <span key={`${task.id}-${source}`}>{source}</span>
              ))}
            </div>
          ) : null}
          {task.latestNoteSummary ? <small>{task.latestNoteSummary}</small> : null}
          {task.blockerReason ? <small className="danger-text">{task.blockerReason}</small> : null}
          {task.lastEventType ? <small>Latest event: {task.lastEventType}</small> : null}
        </details>
      ) : null}
    </article>
  );
}

function LiveAgentCard({ agent }: { agent: LiveAgentTrace }) {
  return (
    <article className={`workspace-live-agent-card ${statusClassName(agent.status)}`}>
      <div className="workspace-live-card-head">
        <div>
          <strong>{agent.name}</strong>
          <span>{agent.role}</span>
        </div>
        <em>{agent.status}</em>
      </div>
      {agent.meta.length ? (
        <div className="workspace-live-metrics">
          {agent.meta.slice(0, 3).map((item) => (
            <span key={`${agent.id}-${item}`}>{item}</span>
          ))}
        </div>
      ) : null}
      <details className="workspace-live-disclosure">
        <summary>{agent.updatedAt ? formatTime(agent.updatedAt) : "Details"}</summary>
        <p>{agent.summary}</p>
        {agent.meta.length > 3 ? (
          <div className="workspace-live-metrics">
            {agent.meta.slice(3).map((item) => (
              <span key={`${agent.id}-detail-${item}`}>{item}</span>
            ))}
          </div>
        ) : null}
      </details>
    </article>
  );
}

function LiveToolTraceCard({ trace }: { trace: LiveToolTrace }) {
  return (
    <article className={`workspace-live-trace-card ${statusClassName(trace.status)}`}>
      <div className="workspace-live-card-head">
        <div>
          <strong>{trace.title}</strong>
          <span>{trace.streamName ?? trace.streamId ?? "Run trace"}</span>
        </div>
        <em>{trace.status}</em>
      </div>
      {trace.meta.length ? (
        <div className="workspace-live-metrics">
          {trace.meta.slice(0, 3).map((item) => (
            <span key={`${trace.id}-${item}`}>{item}</span>
          ))}
        </div>
      ) : null}
      <details className="workspace-live-disclosure">
        <summary>{formatTime(trace.timestamp)}</summary>
        <p>{trace.summary}</p>
        {trace.detail ? <small className="danger-text">{trace.detail}</small> : null}
        {trace.meta.length > 3 ? (
          <div className="workspace-live-metrics">
            {trace.meta.slice(3).map((item) => (
              <span key={`${trace.id}-detail-${item}`}>{item}</span>
            ))}
          </div>
        ) : null}
        <pre>{JSON.stringify(trace.rawEvent.payload, null, 2)}</pre>
      </details>
    </article>
  );
}

function LiveFileCard({ file }: { file: LiveFileTrace }) {
  return (
    <article className={`workspace-live-trace-card ${statusClassName(file.status)}`}>
      <div className="workspace-live-card-head">
        <div>
          <strong>{file.title}</strong>
          <span>{file.timestamp ? formatTime(file.timestamp) : "Available context"}</span>
        </div>
        <em>{file.status}</em>
      </div>
      {file.meta.length ? (
        <div className="workspace-live-metrics">
          {file.meta.slice(0, 3).map((item) => (
            <span key={`${file.id}-${item}`}>{item}</span>
          ))}
        </div>
      ) : null}
      <details className="workspace-live-disclosure">
        <summary>Details</summary>
        <p>{file.summary}</p>
        {file.meta.length > 3 ? (
          <div className="workspace-live-metrics">
            {file.meta.slice(3).map((item) => (
              <span key={`${file.id}-detail-${item}`}>{item}</span>
            ))}
          </div>
        ) : null}
      </details>
    </article>
  );
}

function LiveCitationTraceCard({ trace }: { trace: LiveCitationTrace }) {
  const host = trace.url ? formatCitationHost(trace.url) : null;
  const hasDetails = Boolean(trace.quote) || trace.meta.length > 3;

  return (
    <article className={`workspace-live-citation-card ${statusClassName(trace.status)}`}>
      <div className="workspace-live-card-head">
        <div>
          <strong>{trace.title}</strong>
          <span>{trace.sourceLabel}</span>
        </div>
        <em>{trace.status}</em>
      </div>
      {trace.meta.length ? (
        <div className="workspace-live-metrics">
          {trace.meta.slice(0, 3).map((item) => (
            <span key={`${trace.id}-${item}`}>{item}</span>
          ))}
        </div>
      ) : null}
      <div className="workspace-live-card-foot">
        {trace.timestamp ? <span>{formatTime(trace.timestamp)}</span> : <span>Workspace snapshot</span>}
        {trace.url ? (
          <a href={trace.url} rel="noreferrer" target="_blank">
            {host ?? "Open source"}
          </a>
        ) : null}
      </div>
      {hasDetails ? (
        <details className="workspace-live-disclosure">
          <summary>Details</summary>
          {trace.quote ? <blockquote className="citation-quote">{normalizeInlineText(trace.quote)}</blockquote> : null}
          {trace.meta.length > 3 ? (
            <div className="workspace-live-metrics">
              {trace.meta.slice(3).map((item) => (
                <span key={`${trace.id}-detail-${item}`}>{item}</span>
              ))}
            </div>
          ) : null}
        </details>
      ) : null}
    </article>
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
  const claim = cleanCitationClaimText(citation.claim);
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
          <strong>{claim}</strong>
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
