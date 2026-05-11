from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from open_research.core.domain import (
    AgentConfig,
    AsyncJob,
    BehaviorAssessment,
    BudgetPolicy,
    ContextPack,
    ExecutionMode,
    ModelConfigOverride,
    PassageInspectionRecord,
    PublicRuntimeConfig,
    RunDetail,
    RunEvent,
    RunNoteRecord,
    RunStatus,
)

from .terminal_client import (
    ResearchTerminalClient,
    StreamEnvelope,
    TerminalClientError,
    build_request,
)
from .tui import OpenResearchTerminalApp

DEFAULT_API_BASE_URL = os.environ.get("OPEN_RESEARCH_API_BASE_URL", "http://127.0.0.1:8010")
DEFAULT_API_HOST = os.environ.get("OPEN_RESEARCH_API_HOST", "127.0.0.1")
DEFAULT_API_PORT = int(os.environ.get("OPEN_RESEARCH_API_PORT", "8010"))
DEFAULT_FRONTEND_PORT = int(os.environ.get("OPEN_RESEARCH_FRONTEND_PORT", "3010"))
KNOWN_COMMANDS = {
    "tui",
    "ask",
    "runs",
    "show",
    "config",
    "notes",
    "passages",
    "context",
    "assessments",
    "events",
    "clarify",
    "approve",
    "reject",
    "request-changes",
    "submit-job",
    "job",
    "serve",
    "dev",
}
console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    direct_prompt_args = _parse_direct_prompt_args(argv_list)
    if direct_prompt_args is not None:
        return direct_prompt_args

    parser = argparse.ArgumentParser(
        description=(
            "Open Research terminal client. Run with no subcommand to open the shell UI, "
            "or pass a question directly to execute a prompt immediately."
        ),
        epilog=(
            "Examples:\n"
            "  open-research\n"
            '  open-research "Compare Temporal and LangGraph for durable agents"\n'
            "  open-research serve\n"
            "  open-research dev --install-frontend\n"
            "  open-research runs\n"
            "  open-research config"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="Base URL for the Open Research API.",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("tui", help="Launch the interactive terminal shell UI.")

    ask = subparsers.add_parser("ask", help="Start a run and stream progress in the terminal.")
    ask.add_argument("question", nargs="?", help="Research question.")
    _add_request_options(ask)

    runs = subparsers.add_parser("runs", help="List recent runs.")
    runs.add_argument("--limit", type=int, default=20)
    runs.add_argument(
        "--status",
        choices=[status.value for status in RunStatus],
        default=None,
    )

    show = subparsers.add_parser("show", help="Show a run detail snapshot.")
    show.add_argument("run_id")

    notes = subparsers.add_parser("notes", help="Show extracted notes for a run.")
    notes.add_argument("run_id")

    passages = subparsers.add_parser("passages", help="Show stored passages for a run.")
    passages.add_argument("run_id")

    context = subparsers.add_parser("context", help="Show context packs for a run.")
    context.add_argument("run_id")

    assessments = subparsers.add_parser(
        "assessments",
        help="Show behavior assessments for a run.",
    )
    assessments.add_argument("run_id")

    events = subparsers.add_parser("events", help="Show persisted events for a run.")
    events.add_argument("run_id")
    events.add_argument("--limit", type=int, default=50)

    clarify = subparsers.add_parser("clarify", help="Answer a pending clarification question.")
    clarify.add_argument("run_id")
    clarify.add_argument("response")

    approve = subparsers.add_parser("approve", help="Approve a pending plan preview.")
    approve.add_argument("run_id")
    approve.add_argument("--note")

    reject = subparsers.add_parser("reject", help="Reject a pending plan preview.")
    reject.add_argument("run_id")
    reject.add_argument("--note")

    request_changes = subparsers.add_parser(
        "request-changes",
        help="Request changes on a pending plan preview.",
    )
    request_changes.add_argument("run_id")
    request_changes.add_argument("--note")

    submit_job = subparsers.add_parser("submit-job", help="Create an async job wrapper for a run.")
    submit_job.add_argument("question", nargs="?", help="Research question.")
    _add_request_options(submit_job)

    job = subparsers.add_parser("job", help="Show async job state.")
    job.add_argument("job_id")

    serve = subparsers.add_parser("serve", help="Run the backend API server.")
    serve.add_argument("--host", default=DEFAULT_API_HOST, help="Host for the backend server.")
    serve.add_argument(
        "--port",
        type=int,
        default=DEFAULT_API_PORT,
        help="Port for the backend server.",
    )
    serve.set_defaults(reload=True)
    serve.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        help="Enable backend auto-reload (default).",
    )
    serve.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable backend auto-reload.",
    )

    dev = subparsers.add_parser(
        "dev",
        help="Run the backend API and frontend dashboard together from a repo checkout.",
    )
    dev.add_argument("--host", default=DEFAULT_API_HOST, help="Host for the backend server.")
    dev.add_argument(
        "--port",
        type=int,
        default=DEFAULT_API_PORT,
        help="Port for the backend server.",
    )
    dev.add_argument(
        "--frontend-port",
        type=int,
        default=DEFAULT_FRONTEND_PORT,
        help="Port for the frontend development server.",
    )
    dev.add_argument(
        "--install-frontend",
        action="store_true",
        help="Run `npm install` in the frontend workspace before starting the dashboard.",
    )
    dev.set_defaults(reload=True)
    dev.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        help="Enable backend auto-reload (default).",
    )
    dev.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable backend auto-reload.",
    )

    config = subparsers.add_parser("config", help="Show the public runtime config.")
    config.add_argument("--json", action="store_true", help="Print JSON instead of a rich table.")

    parser.set_defaults(command="tui")
    return parser.parse_args(argv_list)


def _parse_direct_prompt_args(argv_list: list[str]) -> argparse.Namespace | None:
    root_parser = argparse.ArgumentParser(add_help=False)
    root_parser.add_argument("--api-base-url")
    _, remainder = root_parser.parse_known_args(argv_list)
    first_non_option = next((token for token in remainder if not token.startswith("-")), None)
    if first_non_option is None or first_non_option in KNOWN_COMMANDS:
        return None

    prompt_parser = argparse.ArgumentParser(add_help=False)
    prompt_parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
    )
    _add_request_options(prompt_parser)
    prompt_parser.add_argument("question_parts", nargs="+")
    args = prompt_parser.parse_args(argv_list)
    args.command = "ask"
    args.question = " ".join(args.question_parts).strip()
    delattr(args, "question_parts")
    return args


def _add_request_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile-id", default="default")
    parser.add_argument(
        "--execution-mode",
        choices=[mode.value for mode in ExecutionMode],
        default=ExecutionMode.DEEP.value,
    )
    parser.add_argument(
        "--require-plan-approval",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-plan-approval",
        dest="require_plan_approval",
        action="store_false",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source/provider ids to enable for the run.",
    )
    parser.add_argument("--max-streams", type=int)
    parser.add_argument("--max-replans", type=int)
    parser.add_argument("--max-queries-per-stream", type=int)
    parser.add_argument("--max-results-per-query", type=int)
    parser.add_argument("--max-sources-per-stream", type=int)
    parser.add_argument("--per-domain-limit", type=int)
    parser.add_argument(
        "--research-profile",
        choices=["balanced", "official_first", "wide_net"],
    )
    parser.add_argument(
        "--recency-policy",
        choices=["auto", "recent_first", "evergreen"],
    )
    parser.add_argument(
        "--answer-style",
        choices=["analyst", "executive", "technical"],
    )
    parser.add_argument(
        "--citation-discipline",
        choices=["strict", "balanced"],
    )
    parser.add_argument(
        "--claim-granularity",
        choices=["atomic", "balanced"],
    )
    parser.add_argument(
        "--source-trust-floor",
        choices=["standard", "high", "primary", "low"],
    )
    parser.add_argument("--lead-model", help="Override the lead/orchestrator model for this run.")
    parser.add_argument("--planner-model", help="Override the planner model for this run.")
    parser.add_argument("--worker-model", help="Override the worker/researcher model for this run.")
    parser.add_argument("--verifier-model", help="Override the verifier model for this run.")
    parser.add_argument(
        "--include-counterevidence",
        action="store_true",
        default=None,
        help="Force counterevidence gathering on.",
    )
    parser.add_argument(
        "--no-counterevidence",
        dest="include_counterevidence",
        action="store_false",
        help="Disable counterevidence gathering.",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _frontend_dir() -> Path:
    return _repo_root() / "frontend"


def _display_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _origin(host: str, port: int) -> str:
    return f"http://{_display_host(host)}:{port}"


def _require_command(command: str, *, install_hint: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise TerminalClientError(f"`{command}` was not found on PATH. {install_hint}")
    return resolved


def _run_backend_server(*, host: str, port: int, reload: bool) -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency should exist in normal installs
        raise TerminalClientError(
            "uvicorn is not installed in this environment. Install project dependencies first."
        ) from exc

    console.print(
        Panel.fit(
            f"Backend API: {_origin(host, port)}\nReload: {'on' if reload else 'off'}",
            title="Open Research Backend",
        )
    )
    uvicorn.run(
        "open_research.server.main:app",
        host=host,
        port=port,
        reload=reload,
    )
    return 0


def _run_dev_server(args: argparse.Namespace) -> int:
    frontend_dir = _frontend_dir()
    if not frontend_dir.exists():
        raise TerminalClientError(
            "The frontend workspace was not found. "
            "`ore dev` must be run from a repository checkout."
        )

    npm = _require_command(
        "npm",
        install_hint="Install Node.js and npm to use the frontend dashboard.",
    )
    if args.install_frontend:
        console.print("[cyan]Installing frontend dependencies...[/cyan]")
        subprocess.run([npm, "install"], cwd=frontend_dir, check=True)
    elif not (frontend_dir / "node_modules").exists():
        raise TerminalClientError(
            "Frontend dependencies are not installed. Run `ore dev --install-frontend` or "
            "`npm install` in the frontend directory first."
        )

    api_origin = _origin(args.host, args.port)
    frontend_origin = _origin("127.0.0.1", args.frontend_port)
    frontend_env = os.environ.copy()
    frontend_env["NEXT_PUBLIC_API_BASE_URL"] = api_origin

    console.print(
        Panel.fit(
            f"Backend API: {api_origin}\n"
            f"Frontend: {frontend_origin}\n"
            f"Backend reload: {'on' if args.reload else 'off'}",
            title="Open Research Dev",
        )
    )
    frontend_process = subprocess.Popen(
        [npm, "exec", "--", "next", "dev", "--port", str(args.frontend_port)],
        cwd=frontend_dir,
        env=frontend_env,
    )
    try:
        return _run_backend_server(host=args.host, port=args.port, reload=args.reload)
    finally:
        if frontend_process.poll() is None:
            frontend_process.terminate()
            with suppress(subprocess.TimeoutExpired):
                frontend_process.wait(timeout=5)
            if frontend_process.poll() is None:
                frontend_process.kill()
                frontend_process.wait()


async def _run_async(args: argparse.Namespace) -> int:
    client = ResearchTerminalClient(base_url=args.api_base_url)
    try:
        if args.command == "config":
            return await _run_config(client, as_json=args.json)
        if args.command == "runs":
            status = RunStatus(args.status) if args.status else None
            return await _run_list_runs(client, limit=args.limit, status=status)
        if args.command == "show":
            return await _run_show(client, args.run_id)
        if args.command == "notes":
            return await _run_notes(client, args.run_id)
        if args.command == "passages":
            return await _run_passages(client, args.run_id)
        if args.command == "context":
            return await _run_context(client, args.run_id)
        if args.command == "assessments":
            return await _run_assessments(client, args.run_id)
        if args.command == "events":
            return await _run_events(client, args.run_id, limit=args.limit)
        if args.command == "clarify":
            detail = await client.answer_clarification(args.run_id, args.response)
            return _print_run_detail(detail)
        if args.command == "approve":
            detail = await client.approve_plan(args.run_id, note=args.note)
            return _print_run_detail(detail)
        if args.command == "reject":
            detail = await client.reject_plan(args.run_id, note=args.note)
            return _print_run_detail(detail)
        if args.command == "request-changes":
            detail = await client.request_plan_changes(args.run_id, note=args.note)
            return _print_run_detail(detail)
        if args.command == "submit-job":
            question = _resolve_question(args.question)
            return await _run_submit_job(client, question=question, args=args)
        if args.command == "job":
            return await _run_job(client, args.job_id)
        if args.command == "ask":
            question = _resolve_question(args.question)
            return await _run_ask(client, question=question, args=args)
        raise TerminalClientError(f"Unsupported command: {args.command}")
    except TerminalClientError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1
    finally:
        await client.aclose()


async def _run_config(client: ResearchTerminalClient, *, as_json: bool) -> int:
    config = await client.public_config()
    if as_json:
        console.print_json(data=config.model_dump(mode="json"))
        return 0

    table = Table(title="Open Research Runtime Config")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("App", config.app_name)
    table.add_row("Environment", config.environment)
    table.add_row("Prompt profile", config.prompt_profile_version)
    table.add_row("Trust policy", config.source_trust_policy_version)
    table.add_row(
        "Backends",
        ", ".join(f"{key}={value}" for key, value in config.backends.items()),
    )
    table.add_row(
        "Models",
        ", ".join(f"{key}={value}" for key, value in config.models.model_dump(mode="json").items()),
    )
    table.add_row(
        "Default agent config",
        ", ".join(
            f"{key}={value}"
            for key, value in config.default_agent_config.model_dump(mode="json").items()
        ),
    )
    console.print(table)
    return 0


def _print_run_detail(detail: RunDetail) -> int:
    console.print_json(data=detail.model_dump(mode="json"))
    return 0


async def _run_list_runs(
    client: ResearchTerminalClient,
    *,
    limit: int,
    status: RunStatus | None,
) -> int:
    runs = await client.list_runs(limit=limit, status=status)
    table = Table(title="Open Research Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status")
    table.add_column("Updated")
    table.add_column("Cost")
    table.add_column("Question", max_width=60)
    for run in runs:
        table.add_row(
            run.id,
            run.status.value,
            run.updated_at.isoformat(timespec="seconds"),
            f"${run.estimated_cost_usd:.4f}",
            run.question,
        )
    console.print(table)
    return 0


async def _run_show(client: ResearchTerminalClient, run_id: str) -> int:
    detail = await client.get_run_detail(run_id)
    _render_run_detail(detail)
    if detail.final_report is not None:
        console.print(Markdown(detail.final_report.markdown))
    return 0


async def _run_notes(client: ResearchTerminalClient, run_id: str) -> int:
    notes = await client.get_notes(run_id)
    _render_notes(notes, run_id=run_id)
    return 0


async def _run_passages(client: ResearchTerminalClient, run_id: str) -> int:
    passages = await client.get_passages(run_id)
    _render_passages(passages, run_id=run_id)
    return 0


async def _run_context(client: ResearchTerminalClient, run_id: str) -> int:
    packs = await client.get_context_packs(run_id)
    _render_context_packs(packs, run_id=run_id)
    return 0


async def _run_assessments(client: ResearchTerminalClient, run_id: str) -> int:
    assessments = await client.get_assessments(run_id)
    _render_assessments(assessments, run_id=run_id)
    return 0


async def _run_events(client: ResearchTerminalClient, run_id: str, *, limit: int) -> int:
    events = await client.list_events(run_id)
    _render_events(events[-limit:], run_id=run_id)
    return 0


async def _run_ask(
    client: ResearchTerminalClient,
    *,
    question: str,
    args: argparse.Namespace,
) -> int:
    config = await client.public_config()
    budget = _budget_from_args(args, config)
    agent_config = _agent_config_from_args(args, config)
    source_selection = _source_selection_from_args(args)

    run = await client.create_run(
        question=question,
        budget=budget,
        agent_config=agent_config,
        profile_id=args.profile_id,
        execution_mode=ExecutionMode(args.execution_mode),
        require_plan_approval=args.require_plan_approval,
        source_selection=source_selection,
        model_config_override=_model_config_override_from_args(args),
        metadata={"client": {"surface": "terminal-cli"}},
    )
    console.print(
        Panel.fit(
            f"Started run [bold]{run.id}[/bold]\n"
            f"Status: {run.status.value}\n"
            f"Prompt profile: {config.prompt_profile_version}",
            title="Open Research",
        )
    )

    last_status_line = ""
    async for envelope in client.stream_run_events(run.id):
        summary = _summarize_event(envelope)
        if summary == last_status_line:
            continue
        console.print(summary)
        last_status_line = summary

    detail = await client.get_run_detail(run.id)
    if detail.status != RunStatus.COMPLETED:
        _render_run_detail(detail)
        return 1
    report = await client.get_report(run.id)
    _render_run_detail(detail)
    console.print(Markdown(report.markdown))
    return 0


async def _run_submit_job(
    client: ResearchTerminalClient,
    *,
    question: str,
    args: argparse.Namespace,
) -> int:
    config = await client.public_config()
    request = build_request(
        question=question,
        budget=_budget_from_args(args, config),
        agent_config=_agent_config_from_args(args, config),
        profile_id=args.profile_id,
        execution_mode=ExecutionMode(args.execution_mode),
        require_plan_approval=args.require_plan_approval,
        source_selection=_source_selection_from_args(args),
        model_config_override=_model_config_override_from_args(args),
        metadata={"client": {"surface": "terminal-cli", "mode": "submit-job"}},
    )
    job = await client.create_job(request=request)
    return _render_job(job)


async def _run_job(client: ResearchTerminalClient, job_id: str) -> int:
    job = await client.get_job(job_id)
    return _render_job(job)


def _resolve_question(question: str | None) -> str:
    if question and question.strip():
        return question.strip()
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            return stdin_text
    raise TerminalClientError("Provide a research question as an argument or via stdin.")


def _source_selection_from_args(args: argparse.Namespace) -> list[str] | None:
    if not args.sources:
        return None
    values = [value.strip() for value in args.sources.split(",")]
    selected = [value for value in values if value]
    return selected or None


def _budget_from_args(args: argparse.Namespace, config: PublicRuntimeConfig) -> BudgetPolicy:
    defaults = config.default_budget
    return defaults.model_copy(
        update={
            key: value
            for key, value in {
                "max_streams": args.max_streams,
                "max_replans": args.max_replans,
                "max_queries_per_stream": args.max_queries_per_stream,
                "max_results_per_query": args.max_results_per_query,
                "max_sources_per_stream": args.max_sources_per_stream,
                "per_domain_limit": args.per_domain_limit,
            }.items()
            if value is not None
        }
    )


def _agent_config_from_args(args: argparse.Namespace, config: PublicRuntimeConfig) -> AgentConfig:
    defaults = config.default_agent_config
    return defaults.model_copy(
        update={
            key: value
            for key, value in {
                "research_profile": args.research_profile,
                "recency_policy": args.recency_policy,
                "answer_style": args.answer_style,
                "citation_discipline": args.citation_discipline,
                "claim_granularity": args.claim_granularity,
                "source_trust_floor": args.source_trust_floor,
                "include_counterevidence": args.include_counterevidence,
            }.items()
            if value is not None
        }
    )


def _model_config_override_from_args(args: argparse.Namespace) -> ModelConfigOverride | None:
    values = {
        "lead_model": args.lead_model,
        "planner_model": args.planner_model,
        "worker_model": args.worker_model,
        "verifier_model": args.verifier_model,
    }
    if not any(values.values()):
        return None
    return ModelConfigOverride(**values)


def _render_run_detail(detail: RunDetail) -> None:
    table = Table(title=f"Run {detail.id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Status", detail.status.value)
    table.add_row("Question", detail.question)
    table.add_row("Cost", f"${detail.estimated_cost_usd:.4f}")
    table.add_row("Workflow", detail.workflow_backend or "local")
    table.add_row("Execution mode", detail.execution_mode.value)
    table.add_row("Approval", detail.approval_status.value)
    table.add_row(
        "Prompt profile",
        str(detail.metadata.get("prompt_profile_version", "unknown")),
    )
    if detail.requested_budget is not None:
        table.add_row(
            "Requested budget",
            (
                f"{detail.requested_budget.max_streams} streams / "
                f"{detail.requested_budget.max_queries_per_stream} queries / "
                f"{detail.requested_budget.max_sources_per_stream} sources"
            ),
        )
    if detail.recommended_budget is not None:
        table.add_row(
            "Recommended budget",
            (
                f"{detail.recommended_budget.max_streams} streams / "
                f"{detail.recommended_budget.max_queries_per_stream} queries / "
                f"{detail.recommended_budget.max_sources_per_stream} sources"
            ),
        )
    if detail.effective_budget is not None:
        table.add_row(
            "Effective budget",
            (
                f"{detail.effective_budget.max_streams} streams / "
                f"{detail.effective_budget.max_queries_per_stream} queries / "
                f"{detail.effective_budget.max_sources_per_stream} sources"
            ),
        )
    if detail.source_selection:
        table.add_row("Sources", ", ".join(detail.source_selection))
    if detail.job is not None:
        table.add_row("Job", f"{detail.job.job_id} · {detail.job.status}")
    if detail.agent_config is not None:
        table.add_row(
            "Agent config",
            ", ".join(
                f"{key}={value}"
                for key, value in detail.agent_config.model_dump(mode="json").items()
            ),
        )
    if detail.error_message:
        table.add_row("Error", detail.error_message)
    console.print(table)

    if detail.plan_preview is not None:
        preview = detail.plan_preview
        preview_table = Table(title="Plan Preview")
        preview_table.add_column("Field", style="cyan")
        preview_table.add_column("Value")
        preview_table.add_row("Status", preview.approval_status.value)
        preview_table.add_row("Summary", preview.summary)
        preview_table.add_row("Hypothesis", preview.hypothesis)
        preview_table.add_row("Streams", str(len(preview.streams)))
        preview_table.add_row("Recommended mode", preview.recommended_execution_mode.value)
        preview_table.add_row(
            "Recommended budget",
            (
                f"{preview.recommended_budget.max_streams} streams / "
                f"{preview.recommended_budget.max_queries_per_stream} queries / "
                f"{preview.recommended_budget.max_sources_per_stream} sources"
            ),
        )
        preview_table.add_row(
            "Effective budget",
            (
                f"{preview.effective_budget.max_streams} streams / "
                f"{preview.effective_budget.max_queries_per_stream} queries / "
                f"{preview.effective_budget.max_sources_per_stream} sources"
            ),
        )
        console.print(preview_table)

    if detail.clarification_session is not None:
        clarification = Table(title="Clarification")
        clarification.add_column("Question", style="cyan")
        clarification.add_column("Status")
        clarification.add_column("Answer", max_width=50)
        for question in detail.clarification_session.questions:
            answer = next(
                (
                    turn.response
                    for turn in detail.clarification_session.turns
                    if turn.question_id == question.id
                ),
                "",
            )
            clarification.add_row(question.prompt, question.status.value, answer or "pending")
        console.print(clarification)

    streams = Table(title="Streams")
    streams.add_column("Name", style="cyan")
    streams.add_column("Status")
    streams.add_column("Sources")
    streams.add_column("Elapsed")
    streams.add_column("Cost")
    for stream in detail.streams:
        streams.add_row(
            stream.name,
            stream.status.value,
            str(stream.sources_examined),
            f"{stream.elapsed_ms} ms",
            f"${stream.cost_so_far:.4f}",
        )
    console.print(streams)


def _render_job(job: AsyncJob) -> int:
    table = Table(title=f"Job {job.job_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Run", job.run_id)
    table.add_row("Status", job.status)
    table.add_row("Submission mode", job.submission_mode)
    table.add_row("Submitted", job.submitted_at.isoformat(timespec="seconds"))
    if job.started_at is not None:
        table.add_row("Started", job.started_at.isoformat(timespec="seconds"))
    if job.ended_at is not None:
        table.add_row("Ended", job.ended_at.isoformat(timespec="seconds"))
    if job.last_heartbeat_at is not None:
        table.add_row("Heartbeat", job.last_heartbeat_at.isoformat(timespec="seconds"))
    console.print(table)
    return 0


def _render_notes(notes: Sequence[RunNoteRecord], *, run_id: str) -> None:
    table = Table(title=f"Notes · {run_id}")
    table.add_column("Stream", style="cyan")
    table.add_column("Source", max_width=28)
    table.add_column("Confidence")
    table.add_column("Summary", max_width=60)
    for note in notes:
        table.add_row(
            note.stream_name,
            note.source_title or "n/a",
            f"{note.confidence:.2f}",
            note.summary,
        )
    console.print(table)


def _render_passages(passages: Sequence[PassageInspectionRecord], *, run_id: str) -> None:
    table = Table(title=f"Passages · {run_id}")
    table.add_column("Source", style="cyan", max_width=28)
    table.add_column("Index")
    table.add_column("Trust")
    table.add_column("Excerpt", max_width=72)
    for passage in passages:
        table.add_row(
            passage.source_title,
            str(passage.passage_index),
            passage.trust_tier.value if passage.trust_tier is not None else "n/a",
            passage.text[:220],
        )
    console.print(table)


def _render_context_packs(packs: Sequence[ContextPack], *, run_id: str) -> None:
    table = Table(title=f"Context Packs · {run_id}")
    table.add_column("Phase", style="cyan")
    table.add_column("Tokens")
    table.add_column("Fragments")
    table.add_column("Summary", max_width=56)
    for pack in packs:
        table.add_row(
            pack.phase.value,
            f"{pack.used_tokens}/{pack.token_budget}",
            f"{len(pack.fragments)} selected · {len(pack.dropped_fragments)} dropped",
            pack.summary,
        )
    console.print(table)


def _render_assessments(assessments: Sequence[BehaviorAssessment], *, run_id: str) -> None:
    table = Table(title=f"Assessments · {run_id}")
    table.add_column("Kind", style="cyan")
    table.add_column("Source")
    table.add_column("Score")
    table.add_column("Rationale", max_width=68)
    for assessment in assessments:
        table.add_row(
            assessment.kind.value,
            assessment.source.value,
            f"{assessment.score:.2f}",
            assessment.rationale,
        )
    console.print(table)


def _render_events(events: Sequence[RunEvent], *, run_id: str) -> None:
    table = Table(title=f"Events · {run_id}")
    table.add_column("Time", style="cyan")
    table.add_column("Type")
    table.add_column("Payload", max_width=80)
    for event in events:
        table.add_row(
            event.created_at.isoformat(timespec="seconds"),
            event.event_type,
            str(event.payload),
        )
    console.print(table)


def _summarize_event(envelope: StreamEnvelope) -> str:
    if envelope.event_type == "stream.mode":
        return f"[dim]stream mode -> {envelope.payload.get('mode', 'unknown')}[/dim]"
    if envelope.event_type == "plan.created":
        return f"[cyan]plan created[/cyan] ({envelope.payload.get('stream_count', '?')} streams)"
    if envelope.event_type == "search.performed":
        return (
            f"[blue]search[/blue] {envelope.payload.get('query', '')} "
            f"({envelope.payload.get('result_count', '?')} results)"
        )
    if envelope.event_type == "stream.created":
        return f"[cyan]stream[/cyan] {envelope.payload.get('name', 'unknown')} created"
    if envelope.event_type == "source.fetched":
        return (
            f"[magenta]source[/magenta] {envelope.payload.get('title', 'unknown')} "
            f"[dim]{envelope.payload.get('trust_tier', 'unknown')}[/dim]"
        )
    if envelope.event_type == "note.saved":
        return f"[green]note[/green] {envelope.payload.get('summary', '')[:120]}"
    if envelope.event_type == "citation.verified":
        return (
            f"[yellow]citation[/yellow] {envelope.payload.get('support_label', 'unknown')} "
            f"· repairs {envelope.payload.get('repair_attempts', 0)}"
        )
    if envelope.event_type == "report.drafted":
        return (
            f"[green]draft[/green] {envelope.payload.get('section_count', 0)} sections · "
            f"{envelope.payload.get('open_questions', 0)} open questions"
        )
    if envelope.event_type == "memory.retrieved":
        return (
            f"[cyan]memory[/cyan] {envelope.payload.get('memory_kind', 'unknown')} · "
            f"{envelope.payload.get('title', 'untitled')}"
        )
    if envelope.event_type == "memory.compiled":
        return (
            f"[green]memory compiled[/green] {envelope.payload.get('memory_kind', 'unknown')} · "
            f"{envelope.payload.get('summary', 'untitled')}"
        )
    if envelope.event_type == "context.pack.created":
        return (
            f"[blue]context pack[/blue] {envelope.payload.get('phase', 'unknown')} "
            f"({envelope.payload.get('fragment_count', 0)} fragments)"
        )
    if envelope.event_type == "context.fragment.dropped":
        return (
            f"[yellow]context drop[/yellow] {envelope.payload.get('title', 'untitled')} · "
            f"{envelope.payload.get('reason', 'unknown')}"
        )
    if envelope.event_type == "citation.removed":
        return f"[red]citation removed[/red] {', '.join(envelope.payload.get('reasons', []))}"
    if envelope.event_type == "report.completed":
        return (
            f"[bold green]report completed[/bold green] "
            f"({envelope.payload.get('citation_count', 0)} citations)"
        )
    if envelope.event_type in {"run.failed", "run.cancelled"}:
        return f"[bold red]{envelope.event_type}[/bold red] {envelope.payload}"
    return f"[dim]{envelope.event_type}[/dim] {envelope.payload}"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.command == "tui":
            OpenResearchTerminalApp(api_base_url=args.api_base_url).run()
            return
        if args.command == "serve":
            raise SystemExit(
                _run_backend_server(host=args.host, port=args.port, reload=args.reload)
            )
        if args.command == "dev":
            raise SystemExit(_run_dev_server(args))
        raise SystemExit(asyncio.run(_run_async(args)))
    except TerminalClientError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
