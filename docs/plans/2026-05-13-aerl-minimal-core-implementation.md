# AERL minimal core (v1) implementation plan

> **Ship status:** Minimal core v1 is implemented on `main` (see git history through 2026-05-13). Step checkboxes below are marked complete for archival traceability; use git for the source of truth.

> **For agentic workers:** REQUIRED SUB-SKILL: Use @superpowers/subagent-driven-development (recommended) or @superpowers/executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a single-process ASGI service that proxies OpenAI-compatible `/v1/*` to an upstream while appending structured JSONL traces, exposes `/health` and `/ready`, and accepts `POST /aerl/v1/jobs` with optional webhook forwarding.

**Architecture:** **Starlette** ASGI app routes `/health`, `/ready`, `/aerl/v1/jobs`, and a catch-all for `/v1/{path:path}`. **httpx.AsyncClient** performs upstream HTTP calls. **Append-only JSONL** under `AERL_DATA_DIR` (one line per completed request/response or job event). Request correlation via `X-AERL-Request-Id` (UUID). Upstream HTTP responses are returned unchanged (status + headers + body); AERL-only failures return JSON `{ "error": { "code", "message", "request_id" } }`.

**Architecture (wiring):** `create_app()` loads `Settings` once from the environment (see Task 2) and passes `settings` into route closures / `State` so tests and production share the same construction path.

**Request correlation:** `RequestIdMiddleware` runs on **every** request (including `/health`, `/ready`, `/aerl/v1/jobs`, `/v1/*`), sets `request.state.request_id` (UUID), and adds `X-AERL-Request-Id` on **outgoing** responses. `aerl_error_response` and job validation errors **must** use this same id (spec §3 AERL errors).

**Scope note:** Spec §7 optional `/v1` path allowlist is **explicitly out of scope for v1** (YAGNI).

**Tech stack:** Python 3.11+, Starlette, Uvicorn, httpx, pytest, pytest-asyncio, respx (httpx route mocking in tests). Optional: pydantic-settings for typed env (recommended one small dependency).

**Spec:** `/home/liumy26/AERL/docs/specs/2026-05-13-aerl-minimal-core-design.md`

---

## File map (create)

| Path | Responsibility |
|------|------------------|
| `pyproject.toml` | Package `aerl`, runtime deps, pytest config |
| `README.md` | Env table, curl examples, link to spec + self-coaching |
| `src/aerl/__init__.py` | `__version__ = "0.1.0"` |
| `src/aerl/settings.py` | Load/normalize env per spec §4 |
| `src/aerl/redact.py` | Redact `Authorization` for logs |
| `src/aerl/trace_store.py` | Thread-safe JSONL append + truncation helpers |
| `src/aerl/errors.py` | Build `JSONResponse` for AERL-native errors |
| `src/aerl/middleware.py` | `RequestIdMiddleware` (optional: fold into `app.py` if very small) |
| `src/aerl/upstream_probe.py` | Build ready-check URL from `Settings.upstream_openai_base_url` + `ready_probe_path` (same join rule as proxy) |
| `src/aerl/proxy.py` | `/v1` forward + logging (non-stream + SSE aggregate) |
| `src/aerl/jobs.py` | `POST /aerl/v1/jobs` + optional webhook |
| `src/aerl/app.py` | Assemble `Starlette` routes + middleware for `request_id` |
| `src/aerl/main.py` | `main()` calling `uvicorn.run` with `AERL_LISTEN_*` |
| `src/aerl/__main__.py` | `python -m aerl` → invokes `main()` |
| `tests/conftest.py` | Shared fixtures: tmp `AERL_DATA_DIR`, `TestClient`/`httpx.AsyncClient` against ASGI app |
| `tests/test_health_ready.py` | `/health`, `/ready` |
| `tests/test_jobs.py` | Jobs + webhook success/failure |
| `tests/test_proxy.py` | Non-stream proxy + upstream error passthrough |
| `tests/test_proxy_stream.py` | SSE stream passthrough + log linkage |
| `tests/test_truncation.py` | Log truncation flags |
| `examples/docker-compose.yml` | AERL + tiny mock upstream (optional Task 10) |

*The table lists primary runtime modules and core test files; additional tests are named inside each task.*

---

### Task 1: Project scaffold and test harness

**Files:**
- Create: `pyproject.toml`
- Create: `src/aerl/__init__.py`
- Create: `tests/conftest.py`

- [x] **Step 1: Write failing test** — `tests/test_health_ready.py` imports app (will fail).

```python
import pytest


@pytest.mark.asyncio
async def test_health_returns_version():
    from aerl.app import create_app
    from starlette.testclient import TestClient

    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "version" in r.json()
```

- [x] **Step 2: Run test — expect failure**

Run: `cd /home/liumy26/AERL && uv run pytest tests/test_health_ready.py::test_health_returns_version -v`  
(if `uv` unavailable: `python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]' && pytest ...`)  
Expected: `ImportError` or `ModuleNotFoundError` for `aerl.app`.

- [x] **Step 3: Minimal `pyproject.toml` + package + stub `create_app`**

After first edit, run `cd /home/liumy26/AERL && uv sync` (or `pip install -e '.[dev]'`) and fix Hatchling `src` layout (`[tool.hatch.build.targets.wheel] packages = ["src/aerl"]` may need adjustment per [Hatchling packages](https://hatch.pypa.io/latest/config/build/#packages)) until imports work.

`pyproject.toml` (representative):

```toml
[project]
name = "aerl"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "starlette>=0.37",
  "uvicorn[standard]>=0.27",
  "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "respx>=0.21"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/aerl"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

`src/aerl/app.py` stub:

```python
from starlette.applications import Starlette
from starlette.routing import Route


async def health(request):
    from aerl import __version__
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "version": __version__})


def create_app():
    return Starlette(routes=[Route("/health", health, methods=["GET"])])
```

`src/aerl/__init__.py`: `__version__ = "0.1.0"`

- [x] **Step 4: Run test — expect pass**

Run: `uv run pytest tests/test_health_ready.py::test_health_returns_version -v`  
Expected: PASS

- [x] **Step 5: Commit**

```bash
cd /home/liumy26/AERL && git add pyproject.toml src/aerl tests && git commit -m "chore: scaffold aerl package and health route"
```

---

### Task 2: Typed settings

**Files:**
- Create: `src/aerl/settings.py`
- Create: `tests/test_settings.py`

Decisions locked here: `AERL_MAX_BODY_BYTES` default **4194304** (4 MiB). **`AERL_MAX_BUFFERED_REQUEST_BYTES` default `33554432` (32 MiB)** — max request body size AERL will buffer for logging+forward before switching to stream-only logging. `AERL_LISTEN_HOST` default **0.0.0.0**, `AERL_LISTEN_PORT` default **8765**. **`AERL_READY_PROBE_PATH` default `models`** — joined to `UPSTREAM_OPENAI_BASE_URL` after normalization as `{normalized_upstream}/{probe_path}` (so a base `https://api.openai.com/v1` becomes `https://api.openai.com/v1/models`). `AERL_JOB_WEBHOOK_TIMEOUT` default **30.0** seconds. **`AERL_UPSTREAM_TIMEOUT` default `120.0`** seconds for proxied `/v1/*` calls (prevents indefinite hang; document in README).

- [x] **Step 1: Failing tests** — `tests/test_settings.py`: (a) missing `UPSTREAM_OPENAI_BASE_URL` raises clear error; (b) missing `AERL_DATA_DIR` raises clear error (spec §4 both required); (c) given `UPSTREAM_OPENAI_BASE_URL=https://x/v1/` and `AERL_DATA_DIR` set, `load_settings()` returns a normalized base suitable for joining `v1/chat/completions` without `//` artifacts (exact assertion defined in test).

- [x] **Step 2: Run pytest — FAIL**

- [x] **Step 3: Implement** `load_settings()` returning a frozen `dataclass` or `pydantic_settings.BaseSettings` with all §4 fields **plus** optional `job_webhook_auth` sourced from `AERL_JOB_WEBHOOK_AUTH`. Normalize `UPSTREAM_OPENAI_BASE_URL` (strip trailing slash except root; document join rules with `/v1/...` paths).

- [x] **Step 4: Update `tests/conftest.py`** — `autouse=True` pytest fixture (or `pytest_configure`) sets `UPSTREAM_OPENAI_BASE_URL` to a dummy `https://upstream.test/v1` and `AERL_DATA_DIR` to a session-scoped temporary directory so **all** tests run in clean CI once `create_app()` loads settings (Tasks 5+).

- [x] **Step 5: PASS**

- [x] **Step 6: Wire `load_settings()` into `create_app()`** — `create_app()` calls `load_settings()` once and stores on `app.state.settings` (or closure); update the Task 1 stub `create_app()` accordingly.

- [x] **Step 7: PASS** full settings + health tests.

- [x] **Step 8: Commit** `feat: add settings loader and test env defaults`

---

### Task 3: Trace store + redaction

**Files:**
- Create: `src/aerl/redact.py`
- Create: `src/aerl/trace_store.py`
- Create: `tests/test_trace_store.py`

- [x] **Step 1: Test** — given a dict with `Authorization: "Bearer sk-verysecret"`, redacted value must not contain `verysecret` in full; allow `Bearer ****cret` or hash-only per spec §7.

- [x] **Step 2: FAIL**

- [x] **Step 3: Implement** `redact_headers(headers: dict) -> dict` and `TraceStore.append(record: dict)` writing one JSON line to `{AERL_DATA_DIR}/traces.jsonl` under `threading.Lock`. **`TraceStore` must `mkdir(parents=True, exist_ok=True)` for `AERL_DATA_DIR` on init** before opening the file.

- [x] **Step 4: Test** — append two records, read lines back, assert valid JSON.

- [x] **Step 5: Test (log contract baseline)** — append a minimal **proxy** record dict and assert required keys exist per spec §5: `request_id`, `ts_request_received`, `ts_upstream_sent`, `ts_response_complete` (ISO8601 strings), `method`, `path`, `upstream_status` (or `status_code`), optional `model`, `request_body_truncated` / `response_body_truncated` booleans when applicable. (Implementers define exact key names once in `trace_store` or `proxy` module docstring and keep tests aligned.)

- [x] **Step 6: Commit** `feat: add JSONL trace store and header redaction`

---

### Task 4: Request ID middleware + AERL-native errors

**Files:**
- Create: `src/aerl/errors.py`
- Create: `src/aerl/middleware.py` (or inline in `app.py` if tiny)
- Modify: `src/aerl/app.py`
- Create: `tests/test_errors.py`

- [x] **Step 1: Test** — `aerl_error_response(request, code, message)` returns JSONResponse with nested `error.code`, `error.message`, and `error.request_id` equal to `request.state.request_id`.

- [x] **Step 2: Test** — `GET /health` returns header `X-AERL-Request-Id` matching UUID format.

- [x] **Step 3: Implement** `RequestIdMiddleware` + `aerl_error_response` helper; wire middleware first in `create_app`.

- [x] **Step 4: PASS** full `tests/test_errors.py` and health header test (extend `test_health_ready.py` or `test_errors.py`).

- [x] **Step 5: Commit** `feat: add request id middleware and AERL error JSON helper`

---

### Task 5: `/ready` and upstream probe

**Files:**
- Create: `src/aerl/upstream_probe.py`
- Modify: `src/aerl/app.py`
- Modify: `tests/test_health_ready.py`

Behavior: if `AERL_READY_CHECK_UPSTREAM` is false/unset, `/ready` mirrors `/health` (200). If true, **`GET`** the URL `join_upstream(settings.upstream_openai_base_url, settings.ready_probe_path)` using the **same path-join helper** as the `/v1` proxy (no duplicate `/v1` segments). Use `Authorization` from env `AERL_READY_AUTH` if set, else no auth — document: for OpenAI-style upstream set `AERL_READY_AUTH` to `Bearer sk-...` for probe only. 503 if probe not 2xx. When probe runs and returns 2xx, add **`"upstream_ok": true`** to the JSON body (spec §3 optional field); on 503 omit or set `upstream_ok: false` per README contract.

- [x] **Step 1: Tests** with `respx` — mock upstream reachable at `settings`-compatible URL.

- [x] **Step 2: Implement** — probe disabled: `/ready` JSON equals `/health` fields for `status`/`version`.

- [x] **Step 3: Implement** — probe enabled + 2xx: `upstream_ok: true`.

- [x] **Step 4: Implement** — probe enabled + non-2xx: HTTP 503, JSON includes **`"upstream_ok": false`** (required on 503), plus `status`/`version` like `/health`.

- [x] **Step 5: Implement** — when `AERL_READY_AUTH` set, assert mock received `Authorization` header.

- [x] **Step 6: PASS** + **Step 7: Commit** `feat: add /ready with optional upstream probe`

---

### Task 6: `POST /aerl/v1/jobs`

**Files:**
- Create: `src/aerl/jobs.py`
- Modify: `src/aerl/app.py`
- Create: `tests/test_jobs.py`

Rules: max JSON body **1 MiB** for jobs (configurable `AERL_MAX_JOB_BYTES` default 1048576). Webhook call (TLS verify on, httpx default):

```python
import httpx

webhook_headers: dict[str, str] = {}
if settings.job_webhook_auth:
    webhook_headers["Authorization"] = settings.job_webhook_auth

async with httpx.AsyncClient() as client:
    wh_resp = await client.post(
        settings.job_webhook_url,
        json=body,
        headers=webhook_headers,
        timeout=settings.job_webhook_timeout,
    )
```

Map webhook 2xx → `forwarded`, else `failed`. No webhook URL configured → `accepted`. Always append a **job** trace record: include `event_type: "job"`, `request_id` (from `request.state`), `job_id`, `status`, same §5 timestamp fields as proxy where applicable, and `webhook_http_status` when a webhook was attempted.

**Response shape (spec §3, mandatory):** JSON body must be exactly `{ "job_id": "<string>", "status": "<accepted|forwarded|failed>" }` on **HTTP 200** for successful job **acceptance** (including webhook transport failure → `status: failed` — still 200 so clients can read JSON without parsing HTML errors). **Validation failures** (oversized body, invalid JSON) use **AERL-native error JSON** and non-2xx status as in spec §3.

**`job_id` selection:** If client JSON contains string `job_id` that is **non-empty after strip** and **length ≤ 128**, echo it; otherwise generate **UUIDv4**.

- [x] **Step 1: Tests** — (a) no `job_id` in body → response `job_id` is UUID format; (b) valid echo `job_id` in body → same returned; **(b2)** overlong / invalid `job_id` string → server generates UUID; (c) webhook off → `accepted`; (d) webhook 200 → `forwarded`; (e) webhook 500 → `failed`; (f) invalid JSON → AERL error JSON, `request_id` matches `X-AERL-Request-Id`; (g) webhook transport failure → HTTP **200**, `status: failed`, trace has `webhook_error`; (h) body larger than `AERL_MAX_JOB_BYTES` → **413** AERL-native error, no webhook call.

- [x] **Step 2–5: TDD cycle + commit** `feat: add /aerl/v1/jobs with optional webhook`

---

### Task 7: Proxy non-streaming `/v1/*`

**Files:**
- Create: `src/aerl/proxy.py`
- Modify: `src/aerl/app.py` — **Route order (Starlette):** register **`GET /health`**, **`GET /ready`**, **`POST /aerl/v1/jobs`** first; register **`/v1/{path:path}` last** with explicit methods **`GET, POST, PUT, PATCH, DELETE, OPTIONS`** (spec §3 `POST/GET/... /v1/*`); do **not** rely on Route default (GET-only).

**Normative note vs spec prose:** Spec §3 “Errors” mentions webhook failures alongside AERL-native errors; **this plan is normative for jobs:** webhook HTTP non-2xx and transport errors return **HTTP 200** with `{ "job_id", "status": "failed" }` as in the jobs table; only **request validation** failures use AERL `{ "error": ... }` JSON. Document this prominently in README to avoid spec/plan drift during review.

Implementation sketch:

1. Read **`request.state.request_id`** (set by `RequestIdMiddleware`); never mint a second id in the proxy.
2. If `Content-Length` is absent or ≤ **`settings.max_buffered_request_bytes`** (default **32 MiB**, overridable via `AERL_MAX_BUFFERED_REQUEST_BYTES`), buffer the full body for forwarding and logging (stored fields still truncated per `AERL_MAX_BODY_BYTES`); if larger, **stream** client→upstream without retaining the body for logs and set `request_body_omitted: true`.
3. Build upstream URL from `settings.upstream_openai_base_url` + request path under `/v1/...` using one shared `join_urls()` helper (same as ready probe); normalize double slashes.
4. Forward headers: copy whitelist (`authorization`, `content-type`, `user-agent`, `openai-*` case-insensitive); drop `host`.
5. `httpx.request(method, url, content=body, headers=..., timeout=settings.upstream_timeout)`
6. Log record with redacted headers, truncated bodies per `AERL_MAX_BODY_BYTES`, upstream status.
7. Return `Response(content=resp.content, status_code=resp.status_code, headers=filtered_response_headers)` — filter hop-by-hop headers; add `X-AERL-Request-Id`.

- [x] **Step 1: Test** with `respx` — mock `POST https://upstream.example/v1/chat/completions` returns 200 JSON; client hits AERL; assert response body matches upstream; assert one JSONL line with `request_id` and model field parsed.

- [x] **Step 2: Test** upstream returns 401 — assert client receives 401 identical body (passthrough).

- [x] **Step 3: Test (proxy log contract)** — on successful non-stream completion, parsed JSONL record MUST include the §5 timestamp trio, method `POST`, path containing `chat/completions`, upstream status `200`, and `X-AERL-Request-Id` header on HTTP response matching record `request_id`.

- [x] **Step 4: Test (upstream unreachable)** — when upstream raises `httpx.ConnectError` or times out (use `respx`/`MockTransport` to force), client receives **AERL-native** JSON error (spec §3) with appropriate HTTP status (e.g. **502**), `request_id` present, and **no** partial upstream body.

- [x] **Step 5–7: Implement + commits** (split `test_proxy.py` implementation commit if large).

---

### Task 8: SSE streaming `/v1/*`

**Files:**
- Modify: `src/aerl/proxy.py`
- Create: `tests/test_proxy_stream.py`

Decision (spec §5): **aggregate** assistant text from `data: {json}` lines where `choices[0].delta.content` exists; single final log record with `stream: true`, `aggregated_text`, `truncated` flag if over cap. Passthrough: `StreamingResponse` iterator wrapping upstream bytes.

- [x] **Step 1: Test** — mock upstream returns chunked SSE fixture (minimal two chunks + `data: [DONE]`).

- [x] **Step 2: Test (stream log contract)** — JSONL record has `stream: true`, same timestamp trio and `request_id` as response header, and non-empty `aggregated_text` (or documented field name) matching concatenated deltas.

- [x] **Step 3–5: Implement + PASS**

- [x] **Step 6: Commit** `feat: support SSE streaming with aggregated trace`

---

### Task 9: Truncation flags

**Files:**
- Modify: `src/aerl/trace_store.py` if needed
- Modify: `tests/test_truncation.py`

- [x] **Step 1: Test** — send JSON body > `AERL_MAX_BODY_BYTES` with `Content-Length` small enough to buffer; log must include `"request_body_truncated": true`.

- [x] **Step 2–5: Implement + commit** `test: cover log truncation flags`

---

### Task 10: CLI entry + README + example compose

**Files:**
- Create: `src/aerl/main.py`
- Create: `src/aerl/__main__.py` with:

```python
from aerl.main import main

if __name__ == "__main__":
    main()
```

- Modify: `pyproject.toml` — `[project.scripts] aerl = "aerl.main:main"`
- Modify: `README.md`
- Create: `examples/docker-compose.yml` (optional: `examples/mock_upstream.py` minimal Starlette returning fixed completion)

- [x] **Step 1: Manual smoke** — `UPSTREAM_OPENAI_BASE_URL=http://localhost:9999/v1 AERL_DATA_DIR=/tmp/aerl-data uv run aerl` (after mock starts).

- [x] **Step 2: Document** env vars in README table mirroring spec §4 + new `AERL_READY_AUTH`, `AERL_MAX_JOB_BYTES`, `AERL_READY_PROBE_PATH`, `AERL_JOB_WEBHOOK_TIMEOUT`, `AERL_MAX_BUFFERED_REQUEST_BYTES`, `AERL_JOB_WEBHOOK_AUTH`; add **Integration** subsection linking [self-coaching](https://github.com/Miya-Liu/self-coaching) (`service.url`, `OPENAI_BASE_URL`) per spec §6.

- [x] **Step 3: Commit** `docs: add README and run entrypoint`

---

## Verification (after all tasks)

Run: `cd /home/liumy26/AERL && uv run pytest -q`  
Expected: all tests pass.

Document resolved spec §10 choices (**defaults**, **aggregated SSE trace**, framework picks) **in `README.md`** only for v1; revise the spec file only if you intentionally re-baseline the normative document.

---

## Handoff

After this plan is reviewed (below), implementers follow tasks in order; each task ends with a **green pytest** commit.
