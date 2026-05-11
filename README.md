# Open Research

Open Research is a full-stack deep research agent for public-web research. It runs research jobs asynchronously, streams progress to a web dashboard and terminal client, stores the evidence trail, and grounds final reports against retrieved sources before returning them.

The project is designed as an application runtime, not a notebook. The backend handles planning, provider selection, source fetching, citation checks, persistence, recovery, and optional worker execution. The frontend gives operators a focused console for starting runs, watching progress, reviewing sources, and continuing from previous reports.

## What It Does

Open Research can run without external API keys by using deterministic local fallbacks, which is useful for trying the app and exercising the workflow. With provider credentials, it can use OpenAI or OpenAI-compatible model servers, broker web search across supported providers, fetch and normalize source pages, embed passages for grounding, and store source artifacts locally or in S3.

The runtime keeps a durable record of each run: plan state, tasks, notes, source registry entries, retrieved passages, citation audits, events, artifacts, and final report data. Reports are streamed as work progresses, and the final answer is checked against the source registry so unsafe or unsupported citations can be removed before the user sees them.

## Quick Start

Install dependencies and start the API:

```bash
uv venv --python 3.13
uv sync --extra dev
uv run ore serve
```

The API runs at `http://127.0.0.1:8010`. Its interactive docs are available at `http://127.0.0.1:8010/docs`.

Start the dashboard in another shell:

```bash
cd frontend
npm install
npm run dev
```

The dashboard runs at `http://127.0.0.1:3010` and talks to the local API by default.

You can also run the backend and frontend together from the repository root:

```bash
uv run ore dev --install-frontend
```

## CLI

The `open-research` command can start a run directly, open an interactive terminal UI, list recent runs, inspect a run, or print active configuration.

```bash
uv run open-research
uv run open-research "Compare current approaches to citation-grounded research agents"
uv run open-research runs
uv run open-research show <run-id>
uv run open-research config
```

For a shell-wide install:

```bash
uv tool install -e .
open-research
```

The shorter `ore` command is also available:

```bash
ore serve
ore dev --install-frontend
```

## Configuration

Configuration is read from environment variables. A local `.env` file is supported and intentionally ignored by git.

The app accepts provider-native variables such as `OPENAI_API_KEY`, plus repo-scoped variables such as `OPEN_RESEARCH_DATABASE_URL`. With no provider keys, the app uses local fallback behavior. For real research runs, configure at least one LLM backend, one search backend, and one fetch backend.

Common settings:

```bash
export OPENAI_API_KEY=...
export OPEN_RESEARCH_LLM_BACKEND=openai
export OPEN_RESEARCH_SEARCH_BACKEND=openai
export OPEN_RESEARCH_FETCH_BACKEND=firecrawl
export FIRECRAWL_API_KEY=...
```

For an OpenAI-compatible model server:

```bash
export OPEN_RESEARCH_LLM_BACKEND=openai_compatible
export OPEN_RESEARCH_LLM_BASE_URL=http://127.0.0.1:8000/v1
export OPEN_RESEARCH_LLM_API_KEY=<provider-api-key-or-placeholder>
export OPEN_RESEARCH_LEAD_MODEL=<model-name>
export OPEN_RESEARCH_WORKER_MODEL=<model-name>
```

SQLite is the default database for local development. For Postgres, install the `postgres` extra and set `OPEN_RESEARCH_DATABASE_URL` to a PostgreSQL URL. Optional extras are available for browser fetching, Temporal workers, local reranking, and S3 artifact storage:

```bash
uv sync --extra dev --extra browser --extra postgres --extra temporal --extra grounding --extra storage
```

## Docker

Build the API and worker image:

```bash
docker build -t open-research .
```

Run the bundled stack:

```bash
docker compose up --build
```

The Compose stack includes the API, worker, frontend, Postgres, Temporal, Temporal UI, and OpenTelemetry collector. Prometheus is available behind the optional `observability` profile.

## Project Layout

The backend lives under `src/open_research`:

```text
core/          domain models, settings, citation checks, telemetry, shared utilities
storage/       SQLAlchemy persistence and artifact storage
runtime/       planning, orchestration, grounding, events, memory, prompts, workers
integrations/  provider clients and uploaded asset ingestion
server/        FastAPI application entry point
interfaces/    CLI, TUI, terminal client, and worker process entry points
tools/         reusable tool implementations
prompts/       prompt templates used by the runtime
```

The Next.js dashboard lives under `frontend`.

## Verification

```bash
uv run ruff check src
uv run python -m compileall -q src/open_research
```

