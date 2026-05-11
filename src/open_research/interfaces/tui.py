from __future__ import annotations

import shlex
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, ClassVar

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Input, RichLog, Static

from open_research.core.domain import (
    AgentConfig,
    AnswerStyle,
    BehaviorAssessment,
    BudgetPolicy,
    CitationAuditRecord,
    CitationDiscipline,
    ClaimGranularity,
    ContextPack,
    PassageInspectionRecord,
    PublicRuntimeConfig,
    RecencyPolicy,
    ResearchProfile,
    RunDetail,
    RunEvent,
    RunNoteRecord,
    RunStatus,
    RunSummary,
    SourceTrustTier,
)

from .terminal_client import ResearchTerminalClient, StreamEnvelope, TerminalClientError


def _cycle_enum[T: StrEnum](options: Sequence[T], current: T) -> T:
    if current not in options:
        return options[0]
    index = options.index(current)
    return options[(index + 1) % len(options)]


def _row_key_value(row_key: Any) -> str:
    value = getattr(row_key, "value", row_key)
    return str(value)


def _short_run_id(run_id: str | None) -> str:
    if not run_id:
        return "none"
    return run_id[:8]


class OpenResearchTerminalApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: #08111f;
        color: #e5edf6;
    }

    #statusbar {
        height: auto;
        padding: 0 1;
        background: #0f1a2d;
        color: #dbe7f5;
    }

    #body {
        height: 1fr;
    }

    #transcript-pane {
        width: 1fr;
        padding: 1 1 0 1;
    }

    #transcript {
        height: 1fr;
        border: round #29516f 35%;
        background: #060d18;
        padding: 0 1;
    }

    #inspector {
        width: 42;
        min-width: 36;
        padding: 1 1 0 0;
    }

    #session-summary {
        height: auto;
        margin-bottom: 1;
    }

    #config-summary {
        height: auto;
        margin-bottom: 1;
    }

    #selected-run {
        height: auto;
        margin-bottom: 1;
    }

    #runs-table {
        height: 1fr;
        border: round #29516f 35%;
        background: #060d18;
    }

    #composer-row {
        height: auto;
        padding: 0 1 1 1;
    }

    #prompt-input {
        width: 1fr;
        margin-right: 1;
    }

    #send-prompt {
        width: 12;
        margin-right: 1;
    }

    #refresh-all {
        width: 12;
    }

    Input {
        border: round #4f91c2 45%;
        background: #071526;
        color: #f3f7fb;
    }

    DataTable {
        border: round #29516f 35%;
    }

    RichLog {
        scrollbar-background: #0f1a2d;
        scrollbar-color: #4f91c2;
    }

    Button {
        min-width: 10;
    }
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("ctrl+r", "start_run", "Send"),
        ("ctrl+l", "refresh_all", "Refresh"),
        ("ctrl+k", "cancel_selected", "Cancel"),
        ("ctrl+e", "resume_selected", "Resume"),
        ("ctrl+t", "retry_selected", "Retry"),
        ("ctrl+w", "show_selected_report", "Report"),
        ("ctrl+p", "cycle_profile", "Profile"),
        ("ctrl+y", "cycle_recency", "Recency"),
        ("ctrl+a", "cycle_answer_style", "Style"),
        ("ctrl+d", "cycle_citation_discipline", "Citations"),
        ("ctrl+g", "cycle_claim_granularity", "Claims"),
        ("ctrl+f", "cycle_trust_floor", "Trust"),
        ("ctrl+u", "toggle_counterevidence", "Counter"),
        ("ctrl+j", "focus_prompt", "Prompt"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, *, api_base_url: str, initial_prompt: str | None = None) -> None:
        super().__init__()
        self.api_base_url = api_base_url.rstrip("/")
        self.initial_prompt = initial_prompt
        self.client: ResearchTerminalClient | None = None
        self.public_config: PublicRuntimeConfig | None = None
        self.selected_run_id: str | None = None
        self.selected_detail: RunDetail | None = None
        self.current_budget = BudgetPolicy()
        self.current_agent_config = AgentConfig()
        self._runs_by_id: dict[str, RunSummary] = {}
        self._config_initialized = False
        self._session_message = "Connecting to API…"
        self._session_kind = "info"

    def compose(self) -> ComposeResult:
        yield Static(id="statusbar")
        with Horizontal(id="body"):
            with Vertical(id="transcript-pane"):
                yield RichLog(id="transcript", wrap=True, markup=False, auto_scroll=True)
            with Vertical(id="inspector"):
                yield Static(id="session-summary")
                yield Static(id="config-summary")
                yield Static(id="selected-run")
                yield DataTable(id="runs-table")
        with Horizontal(id="composer-row"):
            yield Input(
                placeholder="Ask a research question or type /help",
                id="prompt-input",
            )
            yield Button("Send", id="send-prompt", variant="primary")
            yield Button("Refresh", id="refresh-all")
        yield Footer()

    async def on_mount(self) -> None:
        self._configure_tables()
        self._render_statusbar()
        self._render_session_summary()
        self._render_config_summary()
        self._render_selected_run_summary()
        self._append_welcome()
        self.load_public_config()
        self.load_runs()
        prompt = self.query_one("#prompt-input", Input)
        prompt.focus()
        if self.initial_prompt:
            prompt.value = self.initial_prompt
            self.start_run(self.initial_prompt)

    async def on_unmount(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def _configure_tables(self) -> None:
        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.cursor_type = "row"
        runs_table.zebra_stripes = True
        runs_table.add_columns("Status", "Updated", "Question")

    def _get_client(self) -> ResearchTerminalClient:
        if self.client is None or self.client.base_url != self.api_base_url:
            if self.client is not None:
                self.run_worker(self.client.aclose(), exclusive=False)
            self.client = ResearchTerminalClient(base_url=self.api_base_url)
        return self.client

    def _set_session(self, message: str, *, kind: str) -> None:
        self._session_message = message
        self._session_kind = kind
        self._render_statusbar()
        self._render_session_summary()

    def _render_statusbar(self) -> None:
        config = self.public_config
        detail = self.selected_detail
        workflow = (
            detail.workflow_backend
            if detail and detail.workflow_backend
            else config.backends.get("workflow", "connecting")
            if config
            else "connecting"
        )
        lead_model = config.models.lead_model if config else "loading"
        status = detail.status.value if detail else "idle"
        statusbar = self.query_one("#statusbar", Static)
        line = Text()
        line.append(" Open Research ", style="bold white on #194a6a")
        line.append(f" endpoint {self.api_base_url} ", style="bold #8ed1fc on #102238")
        line.append(f" workflow {workflow} ", style="bold #c4f1be on #122b1d")
        line.append(f" lead {lead_model} ", style="bold #ffe8a3 on #31260c")
        line.append(
            f" selected {_short_run_id(self.selected_run_id)} [{status}] ",
            style="bold #f4d8ff on #32163d",
        )
        statusbar.update(line)

    def _render_session_summary(self) -> None:
        palette = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        summary = self.query_one("#session-summary", Static)
        config = self.public_config
        body = Group(
            Text("Session", style="bold white"),
            Text(self._session_message, style=palette.get(self._session_kind, "white")),
            Text(f"Endpoint: {self.api_base_url}", style="dim"),
            Text(
                f"Models: lead={config.models.lead_model} · worker={config.models.worker_model}"
                if config
                else "Models: loading…",
                style="dim",
            ),
            Text(
                f"Selected run: {_short_run_id(self.selected_run_id)}",
                style="dim",
            ),
        )
        summary.update(
            Panel(body, title="Connection", border_style=palette.get(self._session_kind, "white"))
        )

    def _render_config_summary(self) -> None:
        config_widget = self.query_one("#config-summary", Static)
        agent = self.current_agent_config
        budget = self.current_budget
        body = Group(
            Text("Composer", style="bold white"),
            Text(f"Profile: {agent.research_profile.value}", style="cyan"),
            Text(f"Recency: {agent.recency_policy.value}", style="cyan"),
            Text(f"Style: {agent.answer_style.value}", style="cyan"),
            Text(f"Citations: {agent.citation_discipline.value}", style="cyan"),
            Text(f"Claims: {agent.claim_granularity.value}", style="cyan"),
            Text(f"Trust floor: {agent.source_trust_floor.value}", style="cyan"),
            Text(
                f"Counterevidence: {'on' if agent.include_counterevidence else 'off'}",
                style="cyan",
            ),
            Text("Budget", style="bold white"),
            Text(
                f"{budget.max_streams} streams · {budget.max_queries_per_stream} queries/stream",
                style="dim",
            ),
            Text(
                f"{budget.max_sources_per_stream} sources/stream · "
                f"{budget.max_results_per_query} results/query",
                style="dim",
            ),
            Text(
                "Use /help for slash commands. Hotkeys: ctrl+r send, ctrl+w report, "
                "ctrl+k cancel, ctrl+t retry.",
                style="dim",
            ),
        )
        config_widget.update(Panel(body, title="Policy", border_style="blue"))

    def _render_selected_run_summary(self) -> None:
        widget = self.query_one("#selected-run", Static)
        detail = self.selected_detail
        if detail is None:
            widget.update(
                Panel(
                    Group(
                        Text("Run inspector", style="bold white"),
                        Text("No run selected.", style="dim"),
                        Text("Select a recent run or ask a new question.", style="dim"),
                    ),
                    title="Selected Run",
                    border_style="white",
                )
            )
            return

        border_style = {
            RunStatus.COMPLETED: "green",
            RunStatus.FAILED: "red",
            RunStatus.CANCELLED: "yellow",
        }.get(detail.status, "cyan")
        body = Group(
            Text(f"{detail.id}", style="bold white"),
            Text(f"Status: {detail.status.value}", style=border_style),
            Text(detail.question[:180], style="white"),
            Text(
                f"Streams: {len(detail.streams)} · Cost: ${detail.estimated_cost_usd:.4f}",
                style="dim",
            ),
            Text(
                f"Workflow: {detail.workflow_backend or 'local'} · "
                f"Worker: {detail.worker_id or 'n/a'}",
                style="dim",
            ),
            Text(
                f"Heartbeat: {detail.last_heartbeat_at.isoformat(timespec='seconds')}"
                if detail.last_heartbeat_at is not None
                else "Heartbeat: n/a",
                style="dim",
            ),
            Text(
                f"Terminal reason: {detail.terminal_reason}"
                if detail.terminal_reason
                else "Terminal reason: n/a",
                style="dim",
            ),
        )
        widget.update(Panel(body, title="Selected Run", border_style=border_style))

    def _append_transcript(self, renderable: RenderableType) -> None:
        self.query_one("#transcript", RichLog).write(renderable)

    def _append_welcome(self) -> None:
        welcome = Group(
            Text("Prompt-first research shell", style="bold white"),
            Text(
                "Ask a question directly, or use slash commands like /help, /runs, /report, "
                "/config, /profile, or /budget.",
                style="white",
            ),
            Text(
                "Examples: /profile official_first · /trust primary · "
                "/budget max-streams 6 · /show <run-id>",
                style="dim",
            ),
        )
        self._append_transcript(Panel(welcome, title="Open Research", border_style="cyan"))

    def _append_user_prompt(self, question: str) -> None:
        self._append_transcript(
            Panel(Text(question, style="white"), title="You", border_style="green")
        )

    def _append_system_message(self, message: str, *, border_style: str = "cyan") -> None:
        self._append_transcript(
            Panel(Text(message, style="white"), title="Open Research", border_style=border_style)
        )

    def _append_help(self) -> None:
        help_markdown = Markdown(
            "\n".join(
                [
                    "# Commands",
                    "",
                    "- `/help` show this help",
                    "- `/runs` refresh and print recent runs",
                    "- `/show [run-id]` load a run into the inspector",
                    "- `/report [run-id]` print the grounded report",
                    "- `/notes [run-id]` list extracted notes",
                    "- `/passages [run-id]` list stored passages",
                    "- `/context [run-id]` show context packs",
                    "- `/assessments [run-id]` show behavior assessments",
                    "- `/events [run-id]` show persisted events",
                    "- `/artifacts [run-id]` list stored artifacts",
                    "- `/audit [run-id]` list citation audit decisions",
                    "- `/config` print active runtime and client settings",
                    "- `/cancel [run-id]`, `/resume [run-id]`, `/retry [run-id]` control runs",
                    "- `/profile <balanced|official_first|wide_net>`",
                    "- `/recency <auto|recent_first|evergreen>`",
                    "- `/style <analyst|executive|technical>`",
                    "- `/citations <strict|balanced>`",
                    "- `/claims <atomic|balanced>`",
                    "- `/trust <low|standard|high|primary>`",
                    "- `/counter <on|off>`",
                    "- `/budget <field> <value>` where field is one of "
                    "`max-streams`, `max-replans`, `max-queries-per-stream`, "
                    "`max-results-per-query`, `max-sources-per-stream`, `per-domain-limit`",
                    "- `/clear` clear the transcript",
                    "",
                    "Type a normal sentence and press Enter to start a run.",
                ]
            )
        )
        self._append_transcript(Panel(help_markdown, title="Help", border_style="blue"))

    def _append_config_snapshot(self) -> None:
        table = Table(show_header=False)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Endpoint", self.api_base_url)
        if self.public_config is not None:
            table.add_row("Prompt profile", self.public_config.prompt_profile_version)
            table.add_row("Trust policy", self.public_config.source_trust_policy_version)
            table.add_row(
                "Backends",
                ", ".join(f"{name}={value}" for name, value in self.public_config.backends.items()),
            )
            table.add_row(
                "Models",
                ", ".join(
                    f"{name}={value}"
                    for name, value in self.public_config.models.model_dump(mode="json").items()
                ),
            )
        table.add_row(
            "Agent config",
            ", ".join(
                f"{key}={value}"
                for key, value in self.current_agent_config.model_dump(mode="json").items()
            ),
        )
        table.add_row(
            "Budget",
            ", ".join(
                f"{key}={value}"
                for key, value in self.current_budget.model_dump(mode="json").items()
            ),
        )
        self._append_transcript(Panel(table, title="Configuration", border_style="blue"))

    def _append_runs_snapshot(self, runs: Sequence[RunSummary]) -> None:
        table = Table()
        table.add_column("Run", style="cyan")
        table.add_column("Status")
        table.add_column("Updated")
        table.add_column("Question", max_width=48)
        for run in runs:
            table.add_row(
                _short_run_id(run.id),
                run.status.value,
                run.updated_at.strftime("%m-%d %H:%M"),
                run.question[:48],
            )
        self._append_transcript(Panel(table, title="Recent Runs", border_style="cyan"))

    def _append_run_snapshot(self, detail: RunDetail) -> None:
        table = Table(show_header=False)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Run", detail.id)
        table.add_row("Status", detail.status.value)
        table.add_row("Question", detail.question)
        table.add_row("Cost", f"${detail.estimated_cost_usd:.4f}")
        table.add_row("Workflow", detail.workflow_backend or "local")
        table.add_row("Streams", str(len(detail.streams)))
        if detail.error_message:
            table.add_row("Error", detail.error_message)
        self._append_transcript(Panel(table, title="Run Snapshot", border_style="cyan"))

    def _append_artifacts_snapshot(self, run_id: str, artifacts: Sequence[dict[str, Any]]) -> None:
        table = Table()
        table.add_column("Kind", style="cyan")
        table.add_column("Type")
        table.add_column("Size")
        table.add_column("URI", max_width=48)
        for artifact in artifacts:
            table.add_row(
                str(artifact["kind"]),
                str(artifact["content_type"]),
                str(artifact["size_bytes"]),
                str(artifact["uri"])[:48],
            )
        self._append_transcript(
            Panel(
                table if artifacts else Text("No artifacts found.", style="dim"),
                title=f"Artifacts · {_short_run_id(run_id)}",
                border_style="blue",
            )
        )

    def _append_notes_snapshot(self, run_id: str, notes: Sequence[RunNoteRecord]) -> None:
        table = Table()
        table.add_column("Stream", style="cyan")
        table.add_column("Source")
        table.add_column("Conf")
        table.add_column("Summary", max_width=48)
        for note in notes:
            table.add_row(
                note.stream_name,
                note.source_title or "n/a",
                f"{note.confidence:.2f}",
                note.summary,
            )
        self._append_transcript(
            Panel(
                table if notes else Text("No notes found.", style="dim"),
                title=f"Notes · {_short_run_id(run_id)}",
                border_style="blue",
            )
        )

    def _append_passages_snapshot(
        self,
        run_id: str,
        passages: Sequence[PassageInspectionRecord],
    ) -> None:
        table = Table()
        table.add_column("Source", style="cyan")
        table.add_column("Index")
        table.add_column("Trust")
        table.add_column("Excerpt", max_width=48)
        for passage in passages:
            table.add_row(
                passage.source_title,
                str(passage.passage_index),
                passage.trust_tier.value if passage.trust_tier is not None else "n/a",
                passage.text[:160],
            )
        self._append_transcript(
            Panel(
                table if passages else Text("No passages found.", style="dim"),
                title=f"Passages · {_short_run_id(run_id)}",
                border_style="blue",
            )
        )

    def _append_context_snapshot(self, run_id: str, packs: Sequence[ContextPack]) -> None:
        table = Table()
        table.add_column("Phase", style="cyan")
        table.add_column("Tokens")
        table.add_column("Fragments")
        table.add_column("Summary", max_width=42)
        for pack in packs:
            table.add_row(
                pack.phase.value,
                f"{pack.used_tokens}/{pack.token_budget}",
                f"{len(pack.fragments)} / {len(pack.dropped_fragments)}",
                pack.summary,
            )
        self._append_transcript(
            Panel(
                table if packs else Text("No context packs found.", style="dim"),
                title=f"Context Packs · {_short_run_id(run_id)}",
                border_style="blue",
            )
        )

    def _append_assessments_snapshot(
        self,
        run_id: str,
        assessments: Sequence[BehaviorAssessment],
    ) -> None:
        table = Table()
        table.add_column("Kind", style="cyan")
        table.add_column("Source")
        table.add_column("Score")
        table.add_column("Rationale", max_width=44)
        for assessment in assessments:
            table.add_row(
                assessment.kind.value,
                assessment.source.value,
                f"{assessment.score:.2f}",
                assessment.rationale,
            )
        self._append_transcript(
            Panel(
                table if assessments else Text("No assessments found.", style="dim"),
                title=f"Assessments · {_short_run_id(run_id)}",
                border_style="blue",
            )
        )

    def _append_events_snapshot(self, run_id: str, events: Sequence[RunEvent]) -> None:
        table = Table()
        table.add_column("Time", style="cyan")
        table.add_column("Type")
        table.add_column("Payload", max_width=48)
        for event in events:
            table.add_row(
                event.created_at.isoformat(timespec="seconds"),
                event.event_type,
                str(event.payload),
            )
        self._append_transcript(
            Panel(
                table if events else Text("No events found.", style="dim"),
                title=f"Events · {_short_run_id(run_id)}",
                border_style="blue",
            )
        )

    def _append_audit_snapshot(
        self,
        run_id: str,
        audits: Sequence[CitationAuditRecord],
    ) -> None:
        table = Table()
        table.add_column("Decision", style="cyan")
        table.add_column("Section")
        table.add_column("Reasons", max_width=36)
        for audit in audits:
            table.add_row(
                audit.decision.value,
                audit.section_title,
                ", ".join(reason.value for reason in audit.reasons) or "-",
            )
        self._append_transcript(
            Panel(
                table if audits else Text("No citation audits found.", style="dim"),
                title=f"Citation Audit · {_short_run_id(run_id)}",
                border_style="blue",
            )
        )

    def _append_report(self, detail: RunDetail) -> None:
        if detail.final_report is not None:
            markdown = detail.final_report.markdown
        elif detail.final_report_markdown:
            markdown = detail.final_report_markdown
        elif detail.error_message:
            markdown = f"# Run Error\n\n{detail.error_message}"
        else:
            markdown = "# Report pending\n\nThe run has not produced a final grounded report yet."
        self._append_transcript(
            Panel(
                Markdown(markdown),
                title=f"Report · {_short_run_id(detail.id)}",
                border_style="green",
            )
        )

    def _append_stream_event(self, envelope: StreamEnvelope) -> None:
        if envelope.event_type == "run.heartbeat":
            return

        message = self._summarize_event(envelope)
        if message is None:
            return
        border_style = {
            "report.completed": "green",
            "run.failed": "red",
            "run.cancelled": "yellow",
            "citation.removed": "yellow",
            "provider.retry": "yellow",
        }.get(envelope.event_type, "cyan")
        self._append_transcript(
            Panel(
                Text(message, style="white"),
                title=envelope.event_type,
                border_style=border_style,
            )
        )

    def _summarize_event(self, envelope: StreamEnvelope) -> str | None:
        payload = envelope.payload
        if envelope.event_type == "stream.mode":
            return f"Streaming mode is {payload.get('mode', 'unknown')}."
        if envelope.event_type == "run.started":
            return f"Run started on {payload.get('workflow_backend', 'local')}."
        if envelope.event_type == "plan.created":
            return f"Planner created {payload.get('stream_count', '?')} research streams."
        if envelope.event_type == "stream.created":
            return f"Created stream {payload.get('name', 'unknown stream')}."
        if envelope.event_type == "search.performed":
            return (
                f"Searched for {payload.get('query', 'unknown query')} and received "
                f"{payload.get('result_count', '?')} results."
            )
        if envelope.event_type == "source.fetched":
            return (
                f"Fetched {payload.get('title', 'untitled source')} "
                f"[trust={payload.get('trust_tier', 'unknown')}]."
            )
        if envelope.event_type == "note.saved":
            return f"Saved note: {payload.get('summary', '')[:160]}"
        if envelope.event_type == "provider.retry":
            return (
                f"Retrying provider {payload.get('provider', 'unknown')} "
                f"({payload.get('attempt', '?')}/{payload.get('max_attempts', '?')})."
            )
        if envelope.event_type == "report.drafted":
            return (
                f"Drafted report with {payload.get('section_count', 0)} sections and "
                f"{payload.get('open_questions', 0)} open questions."
            )
        if envelope.event_type == "memory.retrieved":
            return (
                f"Retrieved {payload.get('memory_kind', 'unknown')} memory: "
                f"{payload.get('title', 'untitled')}."
            )
        if envelope.event_type == "memory.compiled":
            return (
                f"Compiled {payload.get('memory_kind', 'unknown')} memory: "
                f"{payload.get('summary', 'untitled')}."
            )
        if envelope.event_type == "context.pack.created":
            return (
                f"Built {payload.get('phase', 'unknown')} context pack with "
                f"{payload.get('fragment_count', 0)} fragments."
            )
        if envelope.event_type == "context.fragment.dropped":
            return (
                f"Dropped context fragment {payload.get('title', 'untitled')} because "
                f"{payload.get('reason', 'unknown')}."
            )
        if envelope.event_type == "citation.verified":
            return (
                f"Claim verification result: {payload.get('support_label', 'unknown')} "
                f"(repairs={payload.get('repair_attempts', 0)})."
            )
        if envelope.event_type == "citation.removed":
            reasons = ", ".join(payload.get("reasons", [])) or "unknown"
            return f"Removed citation during audit: {reasons}."
        if envelope.event_type == "citation.audit.completed":
            return (
                f"Citation audit completed: kept={payload.get('kept_count', '?')} "
                f"removed={payload.get('removed_count', '?')}."
            )
        if envelope.event_type == "report.completed":
            return f"Grounded report completed with {payload.get('citation_count', 0)} citations."
        if envelope.event_type == "run.cancellation_requested":
            return "Cancellation was requested for the active run."
        if envelope.event_type == "run.cancelled":
            return f"Run cancelled: {payload.get('reason', 'unknown reason')}."
        if envelope.event_type == "run.failed":
            return f"Run failed: {payload.get('error', 'unknown error')}."
        if envelope.event_type == "run.resumed":
            return f"Run resumed from {payload.get('prior_status', 'unknown')}."
        if envelope.event_type == "run.recovered":
            return f"Recovered run from {payload.get('prior_status', 'unknown')}."
        if envelope.event_type == "run.shutdown":
            return f"Run shutdown: {payload.get('reason', 'runtime shutdown')}."
        return None

    def _update_agent_config(self, **updates: Any) -> None:
        payload = self.current_agent_config.model_dump(mode="json")
        payload.update(updates)
        self.current_agent_config = AgentConfig.model_validate(payload)
        self._render_config_summary()

    def _update_budget(self, **updates: Any) -> None:
        payload = self.current_budget.model_dump(mode="json")
        payload.update(updates)
        self.current_budget = BudgetPolicy.model_validate(payload)
        self._render_config_summary()

    def _prompt_text(self) -> str:
        return self.query_one("#prompt-input", Input).value.strip()

    def _clear_prompt(self) -> None:
        self.query_one("#prompt-input", Input).value = ""

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    def action_cycle_profile(self) -> None:
        self._update_agent_config(
            research_profile=_cycle_enum(
                list(ResearchProfile),
                self.current_agent_config.research_profile,
            ).value
        )

    def action_cycle_recency(self) -> None:
        self._update_agent_config(
            recency_policy=_cycle_enum(
                list(RecencyPolicy),
                self.current_agent_config.recency_policy,
            ).value
        )

    def action_cycle_answer_style(self) -> None:
        self._update_agent_config(
            answer_style=_cycle_enum(
                list(AnswerStyle),
                self.current_agent_config.answer_style,
            ).value
        )

    def action_cycle_citation_discipline(self) -> None:
        self._update_agent_config(
            citation_discipline=_cycle_enum(
                list(CitationDiscipline),
                self.current_agent_config.citation_discipline,
            ).value
        )

    def action_cycle_claim_granularity(self) -> None:
        self._update_agent_config(
            claim_granularity=_cycle_enum(
                list(ClaimGranularity),
                self.current_agent_config.claim_granularity,
            ).value
        )

    def action_cycle_trust_floor(self) -> None:
        self._update_agent_config(
            source_trust_floor=_cycle_enum(
                list(SourceTrustTier),
                self.current_agent_config.source_trust_floor,
            ).value
        )

    def action_toggle_counterevidence(self) -> None:
        self._update_agent_config(
            include_counterevidence=not self.current_agent_config.include_counterevidence
        )

    def action_refresh_all(self) -> None:
        self.load_public_config()
        self.load_runs()
        if self.selected_run_id is not None:
            self.load_selected_run(self.selected_run_id)

    def action_start_run(self) -> None:
        prompt = self._prompt_text()
        if not prompt:
            self.notify("Enter a question or slash command.", severity="warning")
            return
        self._clear_prompt()
        if prompt.startswith("/"):
            self._dispatch_command(prompt)
            return
        if len(prompt) < 12:
            self.notify("Enter a question with at least 12 characters.", severity="warning")
            return
        self.start_run(prompt)

    def action_cancel_selected(self) -> None:
        if self.selected_run_id is None:
            self.notify("Select a run first.", severity="warning")
            return
        self.cancel_selected(self.selected_run_id)

    def action_resume_selected(self) -> None:
        if self.selected_run_id is None:
            self.notify("Select a run first.", severity="warning")
            return
        self.resume_selected(self.selected_run_id)

    def action_retry_selected(self) -> None:
        if self.selected_run_id is None:
            self.notify("Select a run first.", severity="warning")
            return
        self.retry_selected(self.selected_run_id)

    def action_show_selected_report(self) -> None:
        if self.selected_run_id is None:
            self.notify("Select a run first.", severity="warning")
            return
        self.show_report(self.selected_run_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-prompt":
            self.action_start_run()
        elif event.button.id == "refresh-all":
            self.action_refresh_all()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "prompt-input":
            self.action_start_run()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "runs-table":
            return
        run_id = _row_key_value(event.row_key)
        self.selected_run_id = run_id
        self._render_statusbar()
        self.load_selected_run(run_id)

    def _dispatch_command(self, raw_text: str) -> None:
        try:
            tokens = shlex.split(raw_text[1:])
        except ValueError as exc:
            self._append_system_message(f"Invalid command: {exc}", border_style="red")
            return
        if not tokens:
            self._append_system_message("Enter a command after /.", border_style="yellow")
            return

        command, *args = tokens
        try:
            if command == "help":
                self._append_help()
            elif command == "clear":
                self.query_one("#transcript", RichLog).clear()
            elif command == "runs":
                self.load_runs(announce=True)
            elif command == "show":
                self.load_selected_run(self._resolve_run_id(args), announce=True)
            elif command == "report":
                self.show_report(self._resolve_run_id(args))
            elif command == "notes":
                self.show_notes(self._resolve_run_id(args))
            elif command == "passages":
                self.show_passages(self._resolve_run_id(args))
            elif command == "context":
                self.show_context(self._resolve_run_id(args))
            elif command == "assessments":
                self.show_assessments(self._resolve_run_id(args))
            elif command == "events":
                self.show_events(self._resolve_run_id(args))
            elif command == "artifacts":
                self.show_artifacts(self._resolve_run_id(args))
            elif command == "audit":
                self.show_audit(self._resolve_run_id(args))
            elif command == "config":
                self._append_config_snapshot()
            elif command == "cancel":
                self.cancel_selected(self._resolve_run_id(args))
            elif command == "resume":
                self.resume_selected(self._resolve_run_id(args))
            elif command == "retry":
                self.retry_selected(self._resolve_run_id(args))
            elif command == "profile":
                self._set_config_enum(
                    field="research_profile",
                    raw_value=self._expect_single_value(command, args),
                    enum_type=ResearchProfile,
                )
            elif command == "recency":
                self._set_config_enum(
                    field="recency_policy",
                    raw_value=self._expect_single_value(command, args),
                    enum_type=RecencyPolicy,
                )
            elif command == "style":
                self._set_config_enum(
                    field="answer_style",
                    raw_value=self._expect_single_value(command, args),
                    enum_type=AnswerStyle,
                )
            elif command == "citations":
                self._set_config_enum(
                    field="citation_discipline",
                    raw_value=self._expect_single_value(command, args),
                    enum_type=CitationDiscipline,
                )
            elif command == "claims":
                self._set_config_enum(
                    field="claim_granularity",
                    raw_value=self._expect_single_value(command, args),
                    enum_type=ClaimGranularity,
                )
            elif command == "trust":
                self._set_config_enum(
                    field="source_trust_floor",
                    raw_value=self._expect_single_value(command, args),
                    enum_type=SourceTrustTier,
                )
            elif command == "counter":
                self._set_counterevidence(self._expect_single_value(command, args))
            elif command == "budget":
                self._handle_budget_command(args)
            else:
                self._append_system_message(
                    f"Unknown command '/{command}'. Use /help for supported commands.",
                    border_style="red",
                )
                return
        except ValueError as exc:
            self._append_system_message(str(exc), border_style="yellow")

    def _resolve_run_id(self, args: Sequence[str]) -> str:
        if args:
            return args[0]
        if self.selected_run_id is not None:
            return self.selected_run_id
        raise ValueError("No run selected. Provide a run id explicitly.")

    def _expect_single_value(self, command: str, args: Sequence[str]) -> str:
        if len(args) != 1:
            raise ValueError(f"/{command} expects exactly one value.")
        return args[0]

    def _set_config_enum(self, *, field: str, raw_value: str, enum_type: type[StrEnum]) -> None:
        try:
            value = enum_type(raw_value)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in enum_type)
            raise ValueError(f"Invalid value '{raw_value}'. Choose from: {allowed}.") from exc
        self._update_agent_config(**{field: value.value})
        self._append_system_message(f"Updated {field.replace('_', ' ')} to {value.value}.")

    def _set_counterevidence(self, raw_value: str) -> None:
        normalized = raw_value.lower()
        if normalized not in {"on", "off"}:
            raise ValueError("/counter expects 'on' or 'off'.")
        enabled = normalized == "on"
        self._update_agent_config(include_counterevidence=enabled)
        self._append_system_message(
            f"Counterevidence is now {'enabled' if enabled else 'disabled'}."
        )

    def _handle_budget_command(self, args: Sequence[str]) -> None:
        if not args:
            self._append_config_snapshot()
            return
        if len(args) != 2:
            raise ValueError("/budget expects '<field> <value>'.")
        field_aliases = {
            "max-streams": "max_streams",
            "max-replans": "max_replans",
            "max-queries-per-stream": "max_queries_per_stream",
            "max-results-per-query": "max_results_per_query",
            "max-sources-per-stream": "max_sources_per_stream",
            "per-domain-limit": "per_domain_limit",
        }
        raw_field, raw_value = args
        if raw_field not in field_aliases:
            allowed = ", ".join(field_aliases)
            raise ValueError(f"Unknown budget field '{raw_field}'. Choose from: {allowed}.")
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError("Budget values must be integers.") from exc
        field_name = field_aliases[raw_field]
        self._update_budget(**{field_name: value})
        self._append_system_message(f"Updated {raw_field} to {value}.")

    @work(exclusive=True)
    async def load_public_config(self) -> None:
        try:
            config = await self._get_client().public_config()
        except Exception as exc:
            self._set_session(f"Failed to load config: {exc}", kind="error")
            return
        self.public_config = config
        if not self._config_initialized:
            self.current_budget = config.default_budget
            self.current_agent_config = config.default_agent_config
            self._config_initialized = True
        self._render_config_summary()
        self._render_statusbar()
        self._set_session(
            f"Connected. Lead model {config.models.lead_model} on {config.backends['workflow']}.",
            kind="success",
        )

    @work(exclusive=True)
    async def load_runs(self, announce: bool = False) -> None:
        try:
            runs = await self._get_client().list_runs(limit=50)
        except Exception as exc:
            self.notify(f"Failed to load runs: {exc}", severity="error")
            return
        self._runs_by_id = {run.id: run for run in runs}
        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.clear(columns=False)
        for run in runs:
            runs_table.add_row(
                run.status.value,
                run.updated_at.strftime("%m-%d %H:%M"),
                run.question[:64],
                key=run.id,
            )
        if self.selected_run_id is None and runs:
            self.selected_run_id = runs[0].id
            self.load_selected_run(runs[0].id)
        if announce:
            self._append_runs_snapshot(runs)
        self._render_statusbar()

    @work(exclusive=True)
    async def load_selected_run(self, run_id: str, announce: bool = False) -> None:
        try:
            detail = await self._get_client().get_run_detail(run_id)
        except Exception as exc:
            self.notify(f"Failed to load run {run_id}: {exc}", severity="error")
            return
        self.selected_run_id = detail.id
        self.selected_detail = detail
        self._render_selected_run_summary()
        self._render_statusbar()
        if announce:
            self._append_run_snapshot(detail)

    @work(exclusive=True)
    async def start_run(self, question: str) -> None:
        self._append_user_prompt(question)
        try:
            run = await self._get_client().create_run(
                question=question,
                budget=self.current_budget,
                agent_config=self.current_agent_config,
                metadata={"client": {"surface": "textual-shell"}},
            )
        except Exception as exc:
            self._set_session(f"Failed to start run: {exc}", kind="error")
            self._append_system_message(f"Failed to start run: {exc}", border_style="red")
            return
        self.selected_run_id = run.id
        self.selected_detail = None
        self._render_selected_run_summary()
        self._render_statusbar()
        self._set_session("Run started. Streaming progress…", kind="success")
        self._append_system_message(f"Started run {run.id}.")
        self.load_runs()
        self.load_selected_run(run.id)
        self.stream_selected_run(run.id)

    @work(exclusive=True)
    async def cancel_selected(self, run_id: str) -> None:
        try:
            await self._get_client().cancel_run(run_id)
        except Exception as exc:
            self._append_system_message(f"Failed to cancel run: {exc}", border_style="red")
            return
        self._set_session("Cancellation requested.", kind="warning")
        self._append_system_message(
            f"Cancellation requested for {_short_run_id(run_id)}.",
            border_style="yellow",
        )
        self.load_selected_run(run_id)
        self.load_runs()

    @work(exclusive=True)
    async def resume_selected(self, run_id: str) -> None:
        try:
            await self._get_client().resume_run(run_id)
        except Exception as exc:
            self._append_system_message(f"Failed to resume run: {exc}", border_style="red")
            return
        self.selected_run_id = run_id
        self._set_session("Run resumed.", kind="success")
        self._append_system_message(f"Resumed run {_short_run_id(run_id)}.")
        self.load_selected_run(run_id)
        self.load_runs()
        self.stream_selected_run(run_id)

    @work(exclusive=True)
    async def retry_selected(self, run_id: str) -> None:
        try:
            new_run = await self._get_client().retry_run(run_id)
        except Exception as exc:
            self._append_system_message(f"Failed to retry run: {exc}", border_style="red")
            return
        self.selected_run_id = new_run.id
        self.selected_detail = None
        self._render_selected_run_summary()
        self._render_statusbar()
        self._set_session(f"Retry created as {_short_run_id(new_run.id)}.", kind="success")
        self._append_system_message(
            f"Created retry run {_short_run_id(new_run.id)} from {_short_run_id(run_id)}."
        )
        self.load_selected_run(new_run.id)
        self.load_runs()
        self.stream_selected_run(new_run.id)

    @work(exclusive=True)
    async def show_report(self, run_id: str) -> None:
        try:
            detail = await self._get_client().get_run_detail(run_id)
        except Exception as exc:
            self._append_system_message(f"Failed to load report: {exc}", border_style="red")
            return
        self.selected_run_id = detail.id
        self.selected_detail = detail
        self._render_selected_run_summary()
        self._render_statusbar()
        self._append_report(detail)

    @work(exclusive=True)
    async def show_artifacts(self, run_id: str) -> None:
        try:
            artifacts = await self._get_client().get_artifacts(run_id)
        except Exception as exc:
            self._append_system_message(f"Failed to load artifacts: {exc}", border_style="red")
            return
        self._append_artifacts_snapshot(
            run_id,
            [artifact.model_dump(mode="json") for artifact in artifacts],
        )

    @work(exclusive=True)
    async def show_notes(self, run_id: str) -> None:
        try:
            notes = await self._get_client().get_notes(run_id)
        except Exception as exc:
            self._append_system_message(f"Failed to load notes: {exc}", border_style="red")
            return
        self._append_notes_snapshot(run_id, notes)

    @work(exclusive=True)
    async def show_passages(self, run_id: str) -> None:
        try:
            passages = await self._get_client().get_passages(run_id)
        except Exception as exc:
            self._append_system_message(f"Failed to load passages: {exc}", border_style="red")
            return
        self._append_passages_snapshot(run_id, passages)

    @work(exclusive=True)
    async def show_context(self, run_id: str) -> None:
        try:
            packs = await self._get_client().get_context_packs(run_id)
        except Exception as exc:
            self._append_system_message(f"Failed to load context packs: {exc}", border_style="red")
            return
        self._append_context_snapshot(run_id, packs)

    @work(exclusive=True)
    async def show_assessments(self, run_id: str) -> None:
        try:
            assessments = await self._get_client().get_assessments(run_id)
        except Exception as exc:
            self._append_system_message(f"Failed to load assessments: {exc}", border_style="red")
            return
        self._append_assessments_snapshot(run_id, assessments)

    @work(exclusive=True)
    async def show_events(self, run_id: str) -> None:
        try:
            events = await self._get_client().list_events(run_id)
        except Exception as exc:
            self._append_system_message(f"Failed to load events: {exc}", border_style="red")
            return
        self._append_events_snapshot(run_id, events[-50:])

    @work(exclusive=True)
    async def show_audit(self, run_id: str) -> None:
        try:
            audits = await self._get_client().get_audit(run_id)
        except Exception as exc:
            self._append_system_message(
                f"Failed to load citation audit: {exc}",
                border_style="red",
            )
            return
        self._append_audit_snapshot(run_id, audits)

    @work(exclusive=True)
    async def stream_selected_run(self, run_id: str) -> None:
        try:
            async for envelope in self._get_client().stream_run_events(run_id):
                if self.selected_run_id != run_id:
                    return
                self._append_stream_event(envelope)
                if envelope.event_type in {
                    "plan.created",
                    "stream.created",
                    "source.fetched",
                    "note.saved",
                    "citation.audit.completed",
                    "run.cancelled",
                    "run.failed",
                    "report.completed",
                }:
                    self.load_selected_run(run_id)
                    self.load_runs()
                if envelope.is_terminal:
                    detail = await self._get_client().get_run_detail(run_id)
                    self.selected_detail = detail
                    self._render_selected_run_summary()
                    self._render_statusbar()
                    if detail.status == RunStatus.COMPLETED:
                        self._append_report(detail)
                        self._set_session(
                            f"Run {_short_run_id(run_id)} completed.",
                            kind="success",
                        )
                    elif detail.status == RunStatus.CANCELLED:
                        self._set_session(
                            f"Run {_short_run_id(run_id)} cancelled.",
                            kind="warning",
                        )
                    else:
                        self._set_session(
                            f"Run {_short_run_id(run_id)} failed.",
                            kind="error",
                        )
                    return
        except TerminalClientError as exc:
            self._append_system_message(f"Stream error: {exc}", border_style="red")
