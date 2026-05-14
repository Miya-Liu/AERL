# AERL

Agent / evaluation **request logging** layer: OpenAI-compatible HTTP surface with minimal operational endpoints.

Normative design: [docs/specs/2026-05-13-aerl-minimal-core-design.md](docs/specs/2026-05-13-aerl-minimal-core-design.md). Task checklist: [docs/plans/2026-05-13-aerl-minimal-core-implementation.md](docs/plans/).

## Run

Requires Python 3.11+.

```bash
export UPSTREAM_OPENAI_BASE_URL=https://api.openai.com/v1   # or your OpenAI-compatible base (no trailing slash except for bare root)
export AERL_DATA_DIR=/tmp/aerl-data
uv sync
uv run aerl
# or: python -m aerl
```

Listens on `AERL_LISTEN_HOST` / `AERL_LISTEN_PORT` (defaults `0.0.0.0:8765`). Traces append to `{AERL_DATA_DIR}/traces.jsonl`.

### Example Docker

From the repo root:

```bash
docker compose -f examples/docker-compose.yml up --build
```

Set `UPSTREAM_OPENAI_BASE_URL` in your shell or a `.env` file next to the compose file if you do not want the default.

## Environment

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `UPSTREAM_OPENAI_BASE_URL` | yes | — | Upstream OpenAI-compatible API base (e.g. `https://api.openai.com/v1`). |
| `AERL_DATA_DIR` | yes | — | Writable directory for `traces.jsonl` and related data. |
| `AERL_LISTEN_HOST` | no | `0.0.0.0` | Bind address for Uvicorn. |
| `AERL_LISTEN_PORT` | no | `8765` | Bind port. |
| `AERL_MAX_BODY_BYTES` | no | `4194304` (4 MiB) | Max stored bytes per logged request/response body (and aggregated SSE text); larger payloads are truncated in logs. |
| `AERL_MAX_BUFFERED_REQUEST_BYTES` | no | `33554432` (32 MiB) | Max client body AERL buffers for logging and forwarding; above this, bodies are streamed without being retained for logs (`request_body_omitted`). |
| `AERL_UPSTREAM_TIMEOUT` | no | `120.0` | Seconds for proxied `/v1/*` upstream HTTP calls. |
| `AERL_READY_CHECK_UPSTREAM` | no | off | If `true` / `1` / `yes`, `GET /ready` probes the upstream. |
| `AERL_READY_PROBE_PATH` | no | `models` | Path segment joined after `UPSTREAM_OPENAI_BASE_URL` for the readiness probe (e.g. `…/v1/models`). |
| `AERL_READY_AUTH` | no | — | Optional `Authorization` value for the readiness probe only. |
| `AERL_JOB_WEBHOOK_URL` | no | — | If set, accepted jobs are POSTed here as JSON. |
| `AERL_JOB_WEBHOOK_AUTH` | no | — | Optional `Authorization` header for the job webhook. |
| `AERL_JOB_WEBHOOK_TIMEOUT` | no | `30.0` | Seconds for the job webhook HTTP client. |
| `AERL_MAX_JOB_BYTES` | no | `1048576` (1 MiB) | Max JSON body size for `POST /aerl/v1/jobs`. |
| `AERL_PRICING_JSON` | no | — | Path to a JSON file for **estimated** USD cost (see below). If unset, traces omit `cost_usd_estimated`. |

## Endpoints

- `GET /health` — liveness JSON with version.
- `GET /ready` — readiness; optional upstream probe when `AERL_READY_CHECK_UPSTREAM` is enabled.
- `POST /aerl/v1/jobs` — opaque JSON job envelope (max `AERL_MAX_JOB_BYTES`); optional webhook forwarding; returns `{ "job_id", "status" }`.
- `GET/POST/PUT/PATCH/DELETE/OPTIONS /v1/{path}` — reverse proxy to `{UPSTREAM_OPENAI_BASE_URL}/{path}`; responses pass through unchanged; `X-AERL-Request-Id` is added.

### Proxy logging (v1)

JSONL records include redacted request headers (see spec §7). Non-stream responses log truncated bodies when over the byte cap. **SSE** (`Content-Type: text/event-stream`): the same SSE bytes are returned to the client; traces store **`stream: true`**, **`aggregated_text`** (concatenated `choices[0].delta.content` from `data:` JSON lines), and **`aggregated_text_truncated`** when over the cap — raw SSE lines are not persisted in full.

**Service metrics (every proxied `/v1/*` record):**

- **`latency_ms_total`** — wall timing from handler entry until the full upstream response is available (milliseconds, `perf_counter`).
- **`latency_ms_upstream`** — time spent on the upstream HTTP call only (send through response complete).
- **`stream`** — `true` for SSE (`text/event-stream` on HTTP 200), otherwise `false`.
- **`openai_user`** — when the JSON request body includes OpenAI’s optional `user` string, it is copied for tenancy / stable-id analytics.
- **`caller_label`** — first non-empty of `X-AERL-User`, `X-User-Id`, or `X-Request-User` (orchestrator identity; not the bearer token).
- **`usage`** — when the upstream JSON (non-stream) or SSE `data:` lines include a `usage` object with token counts, AERL logs normalized `prompt_tokens`, `completion_tokens`, `total_tokens` plus the raw **`upstream`** usage dict.
- **`cost_usd_estimated`** — only when `AERL_PRICING_JSON` is set **and** both prompt and completion token counts are present: `(prompt_tokens / 1e6) * input_per_million_usd + (completion_tokens / 1e6) * output_per_million_usd` using rates from the pricing file. This is an **estimate** (your provider invoice is authoritative).

**`AERL_PRICING_JSON` format** (see `examples/aerl-pricing.example.json`):

```json
{
  "default": { "input_per_million_usd": 5.0, "output_per_million_usd": 15.0 },
  "per_model": {
    "gpt-4o-mini": { "input_per_million_usd": 0.15, "output_per_million_usd": 0.60 }
  }
}
```

Per-model rates override `default`; if neither matches the request `model`, `default` is used when present.

## Development

```bash
uv sync --extra dev
uv run pytest -q
```

(Or `python -m venv .venv && pip install -e '.[dev]'` then `pytest`.)

## Integration

For self-coaching or similar stacks that expect an OpenAI-compatible base URL, point clients at AERL (e.g. `http://localhost:8765/v1`) and configure the upstream with `UPSTREAM_OPENAI_BASE_URL`. See [self-coaching](https://github.com/Miya-Liu/self-coaching) for how your runner may set `service.url` / `OPENAI_BASE_URL`.

## Local mock upstream (optional)

```bash
uv run uvicorn examples.mock_upstream:app --host 127.0.0.1 --port 9999
UPSTREAM_OPENAI_BASE_URL=http://127.0.0.1:9999/v1 AERL_DATA_DIR=/tmp/aerl-data uv run aerl
```
