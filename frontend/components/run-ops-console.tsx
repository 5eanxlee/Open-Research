"use client";

import { useEffect, useMemo, useState } from "react";

import type {
  BehaviorAssessment,
  CitationRecord,
  ContextPack,
  FinalReport,
  PassageInspectionRecord,
  ResearchStreamView,
  RunDetail,
  RunEvent,
  RunNoteRecord,
} from "@/lib/types";

type ConsoleTab = "streams" | "notes" | "sources" | "tools" | "citations" | "thinking";

interface RunOpsConsoleProps {
  detail: RunDetail | undefined;
  notes: RunNoteRecord[] | undefined;
  passages: PassageInspectionRecord[] | undefined;
  contextPacks: ContextPack[] | undefined;
  assessments: BehaviorAssessment[] | undefined;
  events: RunEvent[];
  report: FinalReport | null;
}

const TOOL_EVENT_TYPES = new Set([
  "search.performed",
  "source.fetched",
  "provider.retry",
  "claim.repair.started",
  "claim.repair.search_performed",
  "claim.repair.source_fetched",
  "claim.repair.completed",
  "memory.retrieved",
  "memory.compiled",
  "context.fragment.dropped",
  "context.pack.created",
]);

const CITATION_EVENT_TYPES = new Set([
  "citation.verified",
  "citation.removed",
  "citation.audit.completed",
  "report.sanitized",
  "report.completed",
]);

function formatTime(value: string): string {
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function truncate(value: string, limit = 220): string {
  return value.length > limit ? `${value.slice(0, limit)}…` : value;
}

function eventStreamId(event: RunEvent): string | null {
  const candidate = event.payload.stream_id;
  return typeof candidate === "string" ? candidate : null;
}

function payloadText(payload: Record<string, unknown>): string {
  return JSON.stringify(payload, null, 2);
}

function streamMatches(
  streamId: string | null,
  activeStreamId: string,
  streamName: string | null,
  streamObjective: string | null,
  query: string,
): boolean {
  if (activeStreamId !== "all" && streamId !== activeStreamId) {
    return false;
  }
  if (!query) {
    return true;
  }
  const haystack = `${streamName ?? ""} ${streamObjective ?? ""}`.toLowerCase();
  return haystack.includes(query);
}

export function RunOpsConsole({
  detail,
  notes,
  passages,
  contextPacks,
  assessments,
  events,
  report,
}: RunOpsConsoleProps) {
  const [activeTab, setActiveTab] = useState<ConsoleTab>("streams");
  const [activeStreamId, setActiveStreamId] = useState<string>("all");
  const [filterText, setFilterText] = useState("");
  const normalizedFilter = filterText.trim().toLowerCase();

  const notesByStream = useMemo(() => {
    const map = new Map<string, RunNoteRecord[]>();
    for (const note of notes ?? []) {
      const current = map.get(note.stream_id) ?? [];
      current.push(note);
      map.set(note.stream_id, current);
    }
    return map;
  }, [notes]);

  const streamSourceEvents = useMemo(() => {
    const map = new Map<string, RunEvent[]>();
    for (const event of events) {
      if (event.event_type !== "source.fetched") {
        continue;
      }
      const streamId = eventStreamId(event);
      if (!streamId) {
        continue;
      }
      const current = map.get(streamId) ?? [];
      current.push(event);
      map.set(streamId, current);
    }
    return map;
  }, [events]);

  const streams = detail?.streams ?? [];
  useEffect(() => {
    if (activeStreamId === "all") {
      return;
    }
    if (!streams.some((stream) => stream.id === activeStreamId)) {
      setActiveStreamId("all");
    }
  }, [activeStreamId, streams]);

  const visibleStreams = useMemo(() => {
    return streams.filter((stream) => {
      if (activeStreamId !== "all" && stream.id !== activeStreamId) {
        return false;
      }
      if (!normalizedFilter) {
        return true;
      }
      return `${stream.name} ${stream.objective}`.toLowerCase().includes(normalizedFilter);
    });
  }, [activeStreamId, normalizedFilter, streams]);

  const visibleNotes = useMemo(() => {
    return (notes ?? []).filter((note) =>
      streamMatches(
        note.stream_id,
        activeStreamId,
        note.stream_name,
        note.stream_objective,
        normalizedFilter,
      ),
    );
  }, [activeStreamId, normalizedFilter, notes]);

  const visiblePassages = useMemo(() => {
    return (passages ?? []).filter((passage) => {
      const sourceText = `${passage.source_title} ${passage.text}`.toLowerCase();
      if (normalizedFilter && !sourceText.includes(normalizedFilter)) {
        return false;
      }
      if (activeStreamId === "all") {
        return true;
      }
      const relatedStream = streams.find((stream) =>
        (notesByStream.get(stream.id) ?? []).some((note) => note.source_id === passage.source_id),
      );
      return relatedStream?.id === activeStreamId;
    });
  }, [activeStreamId, normalizedFilter, notesByStream, passages, streams]);

  const visibleToolEvents = useMemo(() => {
    return events.filter((event) => {
      if (!TOOL_EVENT_TYPES.has(event.event_type)) {
        return false;
      }
      const streamId = eventStreamId(event);
      if (activeStreamId !== "all" && streamId && streamId !== activeStreamId) {
        return false;
      }
      if (!normalizedFilter) {
        return true;
      }
      return `${event.event_type} ${payloadText(event.payload)}`.toLowerCase().includes(normalizedFilter);
    });
  }, [activeStreamId, events, normalizedFilter]);

  const visibleCitationEvents = useMemo(() => {
    return events.filter((event) => {
      if (!CITATION_EVENT_TYPES.has(event.event_type)) {
        return false;
      }
      if (!normalizedFilter) {
        return true;
      }
      return `${event.event_type} ${payloadText(event.payload)}`.toLowerCase().includes(normalizedFilter);
    });
  }, [events, normalizedFilter]);

  const visibleContextPacks = useMemo(() => {
    return (contextPacks ?? []).filter((pack) => {
      if (!normalizedFilter) {
        return true;
      }
      const haystack = `${pack.phase} ${pack.summary} ${pack.fragments
        .map((fragment) => `${fragment.title} ${fragment.content}`)
        .join(" ")}`.toLowerCase();
      return haystack.includes(normalizedFilter);
    });
  }, [contextPacks, normalizedFilter]);

  const visibleAssessments = useMemo(() => {
    return (assessments ?? []).filter((assessment) => {
      if (!normalizedFilter) {
        return true;
      }
      return `${assessment.kind} ${assessment.rationale}`.toLowerCase().includes(normalizedFilter);
    });
  }, [assessments, normalizedFilter]);

  const streamRail = (
    <aside className="ops-rail">
      <button
        className={`ops-rail-button ${activeStreamId === "all" ? "active" : ""}`}
        onClick={() => setActiveStreamId("all")}
        type="button"
      >
        <span>All streams</span>
        <strong>{streams.length}</strong>
      </button>
      {streams.map((stream) => (
        <button
          className={`ops-rail-button ${activeStreamId === stream.id ? "active" : ""}`}
          key={stream.id}
          onClick={() => setActiveStreamId(stream.id)}
          type="button"
        >
          <div>
            <span>{stream.name}</span>
            <small>{stream.status}</small>
          </div>
          <strong>{notesByStream.get(stream.id)?.length ?? 0}</strong>
        </button>
      ))}
    </aside>
  );

  const renderStreams = () => (
    <div className="ops-list">
      {visibleStreams.map((stream) => {
        const streamNotes = notesByStream.get(stream.id) ?? [];
        const sourceEvents = streamSourceEvents.get(stream.id) ?? [];
        return (
          <details className="ops-card" key={stream.id} open>
            <summary>
              <div className="ops-card-heading">
                <div>
                  <strong>{stream.name}</strong>
                  <span>{stream.objective}</span>
                </div>
                <div className="ops-badges">
                  <span className={`pill status-${stream.status}`}>{stream.status}</span>
                  <span className="pill muted">{streamNotes.length} notes</span>
                  <span className="pill muted">{sourceEvents.length} sources</span>
                </div>
              </div>
            </summary>
            <div className="ops-card-body">
              <div className="ops-stat-grid">
                <div>
                  <span>Sources examined</span>
                  <strong>{stream.sources_examined}</strong>
                </div>
                <div>
                  <span>Elapsed</span>
                  <strong>{Math.round(stream.elapsed_ms / 1000)}s</strong>
                </div>
                <div>
                  <span>Cost</span>
                  <strong>${stream.cost_so_far.toFixed(4)}</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{stream.confidence?.toFixed(2) ?? "n/a"}</strong>
                </div>
              </div>
              <div className="ops-subgrid">
                <section>
                  <h4>Latest notes</h4>
                  {streamNotes.slice(-4).map((note) => (
                    <article className="ops-inline-card" key={note.id}>
                      <strong>{note.summary}</strong>
                      <span>{note.source_title ?? "Source pending"}</span>
                    </article>
                  ))}
                  {!streamNotes.length ? <span className="muted-text">No notes yet.</span> : null}
                </section>
                <section>
                  <h4>Fetched sources</h4>
                  {sourceEvents.slice(-4).map((event) => (
                    <article className="ops-inline-card" key={`${event.id}`}>
                      <strong>{String(event.payload.title ?? "Untitled source")}</strong>
                      <span>{String(event.payload.trust_tier ?? "unknown trust")}</span>
                    </article>
                  ))}
                  {!sourceEvents.length ? <span className="muted-text">No fetched sources yet.</span> : null}
                </section>
              </div>
            </div>
          </details>
        );
      })}
      {!visibleStreams.length ? <div className="empty-state compact"><p>No streams match the current filter.</p></div> : null}
    </div>
  );

  const renderNotes = () => (
    <div className="ops-list">
      {visibleNotes.map((note) => (
        <details className="ops-card" key={note.id}>
          <summary>
            <div className="ops-card-heading">
              <div>
                <strong>{note.summary}</strong>
                <span>{note.stream_name} · {note.source_title ?? "Source pending"}</span>
              </div>
              <div className="ops-badges">
                <span className="pill muted">{Math.round(note.confidence * 100)} conf</span>
                {note.trust_tier ? <span className="pill muted">{note.trust_tier}</span> : null}
              </div>
            </div>
          </summary>
          <div className="ops-card-body">
            <section>
              <h4>Key facts</h4>
              <ul className="ops-bullet-list">
                {note.key_facts.map((fact, index) => (
                  <li key={`${note.id}-fact-${index}`}>{fact}</li>
                ))}
              </ul>
            </section>
            {note.open_questions.length ? (
              <section>
                <h4>Open questions</h4>
                <ul className="ops-bullet-list">
                  {note.open_questions.map((question, index) => (
                    <li key={`${note.id}-question-${index}`}>{question}</li>
                  ))}
                </ul>
              </section>
            ) : null}
            {note.trust_rationale ? (
              <section>
                <h4>Trust rationale</h4>
                <p className="ops-long-text">{note.trust_rationale}</p>
              </section>
            ) : null}
          </div>
        </details>
      ))}
      {!visibleNotes.length ? <div className="empty-state compact"><p>No notes yet.</p></div> : null}
    </div>
  );

  const renderSources = () => (
    <div className="ops-list">
      {visiblePassages.map((passage) => (
        <details className="ops-card" key={`${passage.source_id}-${passage.passage_index}`}>
          <summary>
            <div className="ops-card-heading">
              <div>
                <strong>{passage.source_title}</strong>
                <span>Passage {passage.passage_index} · {passage.token_count} tokens</span>
              </div>
              <div className="ops-badges">
                {passage.trust_tier ? <span className="pill muted">{passage.trust_tier}</span> : null}
                {passage.retrieval_method ? <span className="pill muted">{passage.retrieval_method}</span> : null}
              </div>
            </div>
          </summary>
          <div className="ops-card-body">
            <a className="ops-link" href={passage.source_url} rel="noreferrer" target="_blank">
              {passage.source_url}
            </a>
            <p className="ops-long-text">{passage.text}</p>
            {passage.trust_rationale ? (
              <p className="muted-text">Trust rationale: {passage.trust_rationale}</p>
            ) : null}
          </div>
        </details>
      ))}
      {!visiblePassages.length ? <div className="empty-state compact"><p>No passages yet.</p></div> : null}
    </div>
  );

  const renderTools = () => (
    <div className="ops-list">
      {visibleToolEvents.map((event) => (
        <details className="ops-card" key={event.id}>
          <summary>
            <div className="ops-card-heading">
              <div>
                <strong>{event.event_type}</strong>
                <span>{formatTime(event.created_at)}</span>
              </div>
              <div className="ops-badges">
                {eventStreamId(event) ? <span className="pill muted">{eventStreamId(event)}</span> : null}
              </div>
            </div>
          </summary>
          <div className="ops-card-body">
            <pre className="ops-code">{payloadText(event.payload)}</pre>
          </div>
        </details>
      ))}
      {!visibleToolEvents.length ? <div className="empty-state compact"><p>No tool activity yet.</p></div> : null}
    </div>
  );

  const renderCitations = () => (
    <div className="ops-list">
      {(report?.citations ?? []).map((citation: CitationRecord, index) => (
        <details className="ops-card" key={`${citation.source_id}-${index}`}>
          <summary>
            <div className="ops-card-heading">
              <div>
                <strong>{citation.claim}</strong>
                <span>{citation.source_title}</span>
              </div>
              <div className="ops-badges">
                <span className="pill muted">{citation.support_label}</span>
                <span className="pill muted">{Math.round(citation.confidence * 100)} conf</span>
              </div>
            </div>
          </summary>
          <div className="ops-card-body">
            <a className="ops-link" href={citation.source_url} rel="noreferrer" target="_blank">
              {citation.source_url}
            </a>
            <blockquote className="ops-quote">{citation.quote}</blockquote>
          </div>
        </details>
      ))}
      {visibleCitationEvents.map((event) => (
        <details className="ops-card" key={`event-${event.id}`}>
          <summary>
            <div className="ops-card-heading">
              <div>
                <strong>{event.event_type}</strong>
                <span>{formatTime(event.created_at)}</span>
              </div>
            </div>
          </summary>
          <div className="ops-card-body">
            <pre className="ops-code">{payloadText(event.payload)}</pre>
          </div>
        </details>
      ))}
      {!(report?.citations?.length || visibleCitationEvents.length) ? (
        <div className="empty-state compact"><p>No citation activity yet.</p></div>
      ) : null}
    </div>
  );

  const renderThinking = () => (
    <div className="ops-list">
      <article className="ops-banner">
        <strong>Reasoning visibility</strong>
        <span>
          Private chain-of-thought is not exposed. This console shows model-visible context, emitted
          notes, tool activity, verification steps, and grounded evidence instead.
        </span>
      </article>
      {visibleContextPacks.map((pack) => (
        <details className="ops-card" key={pack.id} open>
          <summary>
            <div className="ops-card-heading">
              <div>
                <strong>{pack.phase}</strong>
                <span>{pack.summary}</span>
              </div>
              <div className="ops-badges">
                <span className="pill muted">{pack.used_tokens}/{pack.token_budget} tokens</span>
                <span className="pill muted">{pack.fragments.length} fragments</span>
              </div>
            </div>
          </summary>
          <div className="ops-card-body">
            <section>
              <h4>Selected fragments</h4>
              {pack.fragments.map((fragment) => (
                <details className="ops-inline-expandable" key={fragment.id}>
                  <summary>
                    <strong>{fragment.title}</strong>
                    <span>{fragment.kind} · {fragment.token_estimate} tokens</span>
                  </summary>
                  <p className="ops-long-text">{fragment.content}</p>
                  <p className="muted-text">Why selected: {fragment.selected_reason}</p>
                </details>
              ))}
            </section>
            {pack.dropped_fragments.length ? (
              <section>
                <h4>Dropped fragments</h4>
                {pack.dropped_fragments.map((fragment) => (
                  <article className="ops-inline-card" key={fragment.id}>
                    <strong>{fragment.title}</strong>
                    <span>{fragment.dropped_reason ?? "dropped"}</span>
                  </article>
                ))}
              </section>
            ) : null}
          </div>
        </details>
      ))}
      {visibleAssessments.map((assessment) => (
        <article className="ops-card" key={assessment.id}>
          <div className="ops-card-body">
            <div className="ops-card-heading">
              <div>
                <strong>{assessment.kind}</strong>
                <span>{assessment.source}</span>
              </div>
              <div className="ops-badges">
                <span className="pill muted">{Math.round(assessment.score * 100)} / 100</span>
              </div>
            </div>
            <p className="ops-long-text">{assessment.rationale}</p>
          </div>
        </article>
      ))}
    </div>
  );

  const renderTab = () => {
    switch (activeTab) {
      case "streams":
        return renderStreams();
      case "notes":
        return renderNotes();
      case "sources":
        return renderSources();
      case "tools":
        return renderTools();
      case "citations":
        return renderCitations();
      case "thinking":
        return renderThinking();
      default:
        return null;
    }
  };

  return (
    <section className="panel terminal-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Operator Console</p>
          <h2 className="panel-title">Research terminal</h2>
        </div>
        <div className="ops-badges">
          <span className="pill muted">{events.length} events</span>
          <span className="pill muted">{notes?.length ?? 0} notes</span>
          <span className="pill muted">{passages?.length ?? 0} passages</span>
        </div>
      </div>
      {!detail ? (
        <div className="empty-state compact">
          <p>No run selected.</p>
          <span>Launch a run to inspect streams, notes, passages, tools, citations, and context.</span>
        </div>
      ) : (
        <>
          <div className="ops-toolbar">
            {(["streams", "notes", "sources", "tools", "citations", "thinking"] as ConsoleTab[]).map((tab) => (
              <button
                className={`ops-tab ${activeTab === tab ? "active" : ""}`}
                key={tab}
                onClick={() => setActiveTab(tab)}
                type="button"
              >
                {tab}
              </button>
            ))}
            <div className="ops-toolbar-spacer" />
            <input
              className="ops-filter-input"
              onChange={(event) => setFilterText(event.target.value)}
              placeholder="Filter visible content"
              value={filterText}
            />
          </div>
          <div className="ops-layout">
            {streamRail}
            <div className="ops-content">{renderTab()}</div>
          </div>
        </>
      )}
    </section>
  );
}
