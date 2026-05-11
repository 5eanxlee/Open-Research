# Open Research

This repository implements a production-shaped public-web research agent. The current codebase focuses on the backend runtime that matters first:

- FastAPI API with asynchronous run execution
- replayable run event log plus SSE streaming with explicit replay/live cursor semantics
- explicit LangGraph orchestrator with planning, research, assessment, synthesis, and grounding phases
- persisted scratchpad for runs, plans, streams, tasks, notes, sources, passages, claims, citations, and budget events
- persisted artifacts, source registry entries, citation audits, and worker heartbeats
- durable local recovery plus an optional Temporal workflow backend
- native OpenAI structured-output parsing with usage-based cost accounting
- generic OpenAI-compatible LLM support for open-source model servers such as GLM-5.1, Qwen3.5, and DeepSeek
- claim repair during grounding when the first citation pass cannot support a statement
- brokered search and fetch providers for Brave, Exa, Tavily, Firecrawl, Browserbase Fetch, Browserbase browser sessions, and Playwright
- persisted source artifacts for normalized text, rendered HTML, and screenshots on local disk or S3
- hybrid claim grounding with passage embeddings plus optional reranking
- deterministic post-grounding citation sanitization against a persisted source registry
- OpenTelemetry-compatible tracing, Prometheus-style metrics, and JSON structured logs with redaction
- Postgres-ready schema management with Alembic and `pgvector`/`tsvector`-shaped storage
- deterministic mock fallbacks so the system can run locally without external API keys
- worker/API split entrypoints plus a portable Docker Compose stack

## What is implemented

- `POST /runs` starts a research run asynchronously.
- `GET /runs` lists recent runs and supports status filtering.
- `GET /runs/{run_id}` returns the latest run state, streams, plan snapshot, and recent events.
- `GET /runs/{run_id}/report` returns the final report payload once grounding completes.
- `GET /runs/{run_id}/artifacts` returns persisted artifact metadata for the run.
- `GET /runs/{run_id}/audit` returns persisted citation audit decisions.
- `POST /runs/{run_id}/cancel` requests cancellation and halts the run cleanly.
- `POST /runs/{run_id}/resume` resumes a cancelled or failed run from persisted state.
- `POST /runs/{run_id}/retry` starts a fresh run and tags it with `retry_of_run_id`.
- `GET /runs/{run_id}/events` replays persisted run events.
- `GET /runs/{run_id}/stream` streams the same events over SSE.
- `GET /runs/{run_id}/stream/{last_event_id}` resumes SSE from an explicit cursor.
- `GET /healthz` returns a simple liveness response.
- `GET /readyz` verifies database, workflow, and stream readiness.
- `GET /metrics` exposes Prometheus-style metrics.
- `GET /config/public` exposes safe runtime defaults and capability flags for the frontend.

The runtime uses:

- LangGraph for the orchestrator graph
- FastAPI for the API surface
- SQLAlchemy for the persistence layer
- Alembic for production bootstrap when running on Postgres
- Temporal as an optional durable execution layer around the orchestrator
- a claim-level citation pass that retrieves passages and verifies support before finalizing the report
- a deterministic citation audit layer that strips unsafe, shortened, truncated, or unregistered URLs after grounding
- a reconciler that requeues stale or orphaned runs using persisted heartbeats and workflow state

## Local run

```bash
uv venv --python 3.13
uv sync --extra dev
uv run ore serve
```

To enable all production-oriented integrations locally:

```bash
uv sync --extra dev --extra browser --extra postgres --extra temporal --extra grounding --extra storage
```

Then open the API at `http://127.0.0.1:8010/docs` or start a run directly:

```bash
curl -X POST http://127.0.0.1:8010/runs \
  -H 'content-type: application/json' \
  -d '{
    "question": "How should a research agent combine search, extraction, and citation grounding?"
  }'
```

Stream progress:

```bash
curl -N http://127.0.0.1:8010/runs/<run-id>/stream
```

Replay from a cursor:

```bash
curl -N http://127.0.0.1:8010/runs/<run-id>/stream/42
```

Start a dedicated worker process:

```bash
uv run python -m open_research.worker_main
```

Run the backend and frontend together from the repo:

```bash
uv run ore dev --install-frontend
```

Use the terminal client:

```bash
uv run open-research
uv run open-research "How should a research agent ground claims with strict citations?"
uv run open-research ask "How should a research agent expose runtime hardening in a CLI?"
uv run open-research runs
uv run open-research show <run-id>
uv run open-research config
```

The terminal client is prompt-first by default. Running `open-research` opens an interactive
shell-style Textual UI, and passing a plain question directly runs that prompt immediately without
requiring the `ask` subcommand. The terminal surface is backed by the same API and SSE stream as
the web dashboard, so run history, live progress, report rendering, and citation/audit inspection
stay consistent across surfaces. Set `OPEN_RESEARCH_API_BASE_URL` only if the CLI/TUI should talk
to an API endpoint other than `http://127.0.0.1:8010`.

For a Helix-style install-once workflow, install the CLI into your shell path and launch it with
one command:

```bash
uv tool install -e .
open-research
```

There is also a short alias:

```bash
ore
ore "Summarize the latest robust design patterns for research agents"
ore serve
ore dev --install-frontend
```

The TUI exposes client-side research controls for:

- research profile
- recency policy
- answer style
- citation discipline
- claim granularity
- source trust floor
- counterevidence inclusion

Default key bindings:

- `ctrl+r` start a run
- `ctrl+l` refresh runs and details
- `ctrl+k` cancel the selected run
- `ctrl+e` resume the selected run
- `ctrl+t` retry the selected run
- `ctrl+p` cycle research profile
- `ctrl+y` cycle recency policy
- `ctrl+a` cycle answer style
- `ctrl+d` cycle citation discipline
- `ctrl+g` cycle claim granularity
- `ctrl+f` cycle trust floor
- `ctrl+u` toggle counterevidence
- `ctrl+j` focus the question composer
- `ctrl+q` quit

Start the frontend dashboard:

```bash
cd frontend
npm install
npm run dev
```

The dashboard runs on `http://127.0.0.1:3010` and defaults to API endpoint `http://127.0.0.1:8010`,
but the API endpoint, budget controls, and
typed agent policy settings can all be changed client-side and are persisted locally in the
browser.

## Provider modes

Without API keys, the app runs in deterministic local fallback mode:

- heuristic planner, note writer, report writer, and verifier
- mock search results
- mock fetcher output

With credentials configured, the runtime can use:

- OpenAI Responses API for planning, notes, synthesis, and verification
- OpenAI-compatible chat-completions backends for open-source model servers such as vLLM and SGLang
- OpenAI embeddings for hybrid passage retrieval
- OpenAI-compatible embeddings endpoints for self-hosted embedding models
- a search broker that can prioritize Brave, Exa, or Tavily depending on the query
- a fetch ladder that tries Firecrawl, then Browserbase Fetch, then Browserbase browser sessions, then optional Playwright
- local or S3 artifact storage for normalized source text, rendered HTML, and screenshots
- an optional sentence-transformers reranker for higher-confidence grounding
- an optional Temporal workflow engine that runs the orchestrator inside durable activities

The code defaults to SQLite for local work and tests. For Postgres deployments, set `OPEN_RESEARCH_DATABASE_BOOTSTRAP_MODE=alembic` and install the `postgres` extra so the runtime boots through Alembic migrations.

## Configuration

The app accepts both provider-native env vars like `OPENAI_API_KEY` and repo-scoped env vars such
as `OPEN_RESEARCH_DATABASE_URL`.

Set environment variables directly or through a local `.env` file that is not committed. Common
server-side defaults include:

- model defaults
- stream/query budget defaults
- planner discovery thresholds
- upload / OCR limits
- retry and timeout tuning
- Temporal / workflow settings
- S3 artifact storage
- observability and tracing

The strict deep-research path now supports both backends as long as they provide:

- a real LLM endpoint
- a real embedding endpoint
- OpenAI Web Search, Exa, Brave, or Tavily for search
- Firecrawl, Browserbase, Browserbase Session, or Playwright for fetch
- `sentence-transformers` reranking

The local OpenAI-compatible profile does not fall back to heuristic planning or synthesis. It uses the same deep path as hosted OpenAI, but latency and output quality depend on the local model server you choose.

Key runtime switches:

- `OPEN_RESEARCH_LLM_BACKEND=auto|heuristic|openai|openai_compatible`
- `OPEN_RESEARCH_LLM_API_STYLE=auto|responses|chat_completions`
- `OPEN_RESEARCH_LLM_STRUCTURED_OUTPUT_MODE=auto|parse|json_schema|prompted`
- `OPEN_RESEARCH_LLM_MODEL_FAMILY=auto|generic|openai|glm|qwen|deepseek`
- `OPEN_RESEARCH_LLM_REASONING_EFFORT=minimal|low|medium|high`
- `OPEN_RESEARCH_SEARCH_BACKEND=auto|mock|openai|brave|exa|tavily`
- `OPEN_RESEARCH_FETCH_BACKEND=auto|mock|firecrawl|browserbase|browserbase_session|playwright`
- `OPEN_RESEARCH_CUSTOM_RESPONSES_RUNTIME_BACKEND=auto|pipeline|deepagents`
- `OPEN_RESEARCH_TOOL_REGISTRY_ENABLED=true|false`
- `OPEN_RESEARCH_MAX_SEARCH_TOOL_CALLS_PER_RUN`
- `OPEN_RESEARCH_MAX_FETCH_TOOL_CALLS_PER_RUN`
- `OPEN_RESEARCH_WORKFLOW_BACKEND=auto|local|temporal`
- `OPEN_RESEARCH_PROCESS_ROLE=all|api|worker`
- `OPEN_RESEARCH_DATABASE_BOOTSTRAP_MODE=auto|create_all|alembic`
- `OPEN_RESEARCH_ARTIFACT_STORE_BACKEND=auto|disabled|local|s3`
- `OPEN_RESEARCH_EMBEDDING_BACKEND=auto|disabled|mock|openai|openai_compatible`
- `OPEN_RESEARCH_RERANKER_BACKEND=auto|disabled|heuristic|sentence_transformers`
- `OPEN_RESEARCH_PLANNER_MODEL`
- `OPEN_RESEARCH_RUN_HEARTBEAT_SECONDS`
- `OPEN_RESEARCH_STALE_RUN_TIMEOUT_SECONDS`
- `OPEN_RESEARCH_RECONCILER_INTERVAL_SECONDS`
- `OPEN_RESEARCH_PROVIDER_RETRY_ATTEMPTS`
- `OPEN_RESEARCH_PROVIDER_COOLDOWN_SECONDS`
- `OPEN_RESEARCH_OTLP_ENDPOINT`
- `OPEN_RESEARCH_METRICS_ENABLED`
- `OPEN_RESEARCH_API_BASE_URL` only for the terminal client / TUI default endpoint

When `OPEN_RESEARCH_WORKFLOW_BACKEND=temporal`, the API process can optionally run a co-located Temporal worker by leaving `OPEN_RESEARCH_TEMPORAL_START_WORKER=true`.

When `OPEN_RESEARCH_FETCH_BACKEND=browserbase_session`, configure `BROWSERBASE_API_KEY` and optionally `BROWSERBASE_PROJECT_ID`. When `OPEN_RESEARCH_RERANKER_BACKEND=sentence_transformers`, install the `grounding` extra.

For open-source model servers, point the runtime at an OpenAI-compatible endpoint. For example, a local GLM-5.1 or Qwen3.5 deployment behind vLLM or SGLang can be configured with:

```bash
export OPEN_RESEARCH_LLM_BACKEND=openai_compatible
export OPEN_RESEARCH_LLM_BASE_URL=http://127.0.0.1:8000/v1
export OPEN_RESEARCH_LLM_API_KEY=<provider-api-key-or-placeholder>
export OPEN_RESEARCH_LEAD_MODEL=<model-name>
export OPEN_RESEARCH_WORKER_MODEL=<model-name>
export OPEN_RESEARCH_VERIFIER_MODEL=deepseek-ai/<model-name>
export OPEN_RESEARCH_LLM_MODEL_FAMILY=glm
```

If your embedding model is also served through an OpenAI-compatible endpoint, configure:

```bash
export OPEN_RESEARCH_EMBEDDING_BACKEND=openai_compatible
export OPEN_RESEARCH_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export OPEN_RESEARCH_EMBEDDING_API_KEY=EMPTY
```

## Deployment

Build the shared API/worker image:

```bash
docker build -t open-research .
```

Run the portable stack:

```bash
docker compose up --build
```

The Compose stack includes `api`, `worker`, `postgres`, `temporal`, `temporal-ui`, and
`otel-collector`, plus an optional `prometheus` profile. The frontend is a separate Next.js app
under [`frontend`](frontend) and can be run locally with `npm run dev` or containerized separately.

## Verification

```bash
uv run ruff check src tests
uv run pytest
```
