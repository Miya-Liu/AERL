# AERL — Minimal shared core (v1) design

**Status:** Draft → Approved for planning (human sign-off in thread)  
**Date:** 2026-05-13  
**Context:** AERL mirrors the *AReaL-style* idea of an OpenAI-compatible HTTP boundary so agents and tooling can point at a single `base_url`, but **does not host the training implementation**. Training is triggered and run elsewhere (e.g. [self-coaching](https://github.com/Miya-Liu/self-coaching) via `scripts/run-pipeline.sh` and `training/pipelines/`). This document locks **option 3**: a **minimal shared core** API; other behavior is documented as negotiable until AERL and consumer repos stabilize.

---

## 1. Goals

1. **OpenAI-compatible proxy** under `/v1/*` that forwards to a configurable upstream and **records** requests and responses for downstream training or analysis.
2. **Operational endpoints** so orchestrators can preflight a deployment (`/health`, optional `/ready`).
3. **Opaque job hook** under `POST /aerl/v1/jobs` so HTTP-based pipelines can signal “run something external” without embedding a trainer in AERL.
4. **Single deployable** (one process / one container image) for v1 to keep `service.url` in consumer configs simple.

## 2. Non-goals (v1)

- RL-specific protocols, weight sync, or GPU training inside AERL.
- Durable distributed job queues (no required Redis/Kafka contract in v1).
- Hosted dataset storage or model registry.
- Guaranteeing byte-for-byte compatibility with every OpenAI edge case; v1 targets **common** chat/completions usage used by typical agent stacks.

## 3. HTTP surface

| Method & path | Behavior |
|---------------|----------|
| `GET /health` | Returns `200` JSON: `{"status":"ok","version":"<semver>"}`. No upstream call. |
| `GET /ready` | Returns `200` if optional readiness checks pass; `503` if upstream probe fails (when enabled). If upstream probe is disabled, same as health. |
| `POST/GET/... /v1/*` | Reverse-proxy to `UPSTREAM_OPENAI_BASE_URL` (path preserved). Log structured record per logical request/response. Support non-streaming JSON; support **SSE streaming** with chunk-indexed or aggregated logging (see §5). |
| `POST /aerl/v1/jobs` | Accept `application/json` body (opaque object, max size configurable). Log request. If `AERL_JOB_WEBHOOK_URL` is set, `POST` the same JSON there and include webhook status in response. Return JSON: `{ "job_id": "<string>", "status": "<accepted|forwarded|failed>" }` where `job_id` is server-generated UUID unless client supplies `job_id` in body (echo if present and valid string). |

**Namespace:** AERL-specific routes live under `/aerl/v1/` to avoid colliding with future OpenAI-style paths.

**Errors:** Use JSON bodies where practical; document stable `error` object shape `{ "code", "message", "request_id" }` for AERL-generated errors.

## 4. Configuration (environment)

| Variable | Required | Description |
|----------|----------|-------------|
| `UPSTREAM_OPENAI_BASE_URL` | Yes | Base URL of the real LLM API (e.g. `https://api.openai.com/v1` — trailing slash normalized). |
| `AERL_DATA_DIR` | Yes | Directory for append-only logs / SQLite (implementation choice documented in plan). |
| `AERL_JOB_WEBHOOK_URL` | No | If set, job endpoint forwards JSON to this URL (server-side only; not exposed to clients). |
| `AERL_MAX_BODY_BYTES` | No | Cap for logged body size per direction (default TBD in plan, e.g. 4 MiB). |
| `AERL_READY_CHECK_UPSTREAM` | No | If `true`, `/ready` performs lightweight upstream probe. |
| `AERL_LISTEN_HOST` / `AERL_LISTEN_PORT` | No | Bind address (defaults in plan). |

Secrets: upstream API keys arrive on **client requests** to AERL; AERL forwards `Authorization` upstream and **must not** persist full bearer tokens (see §7).

## 5. Logging model

Each proxied request generates at least one **record** containing:

- `request_id` (UUID, also returned to client in a response header, e.g. `X-AERL-Request-Id`)
- ISO timestamps (request received, upstream request sent, response complete)
- HTTP method, path, status code
- Parsed `model` field when present in JSON body
- Request/response payloads **subject to** `AERL_MAX_BODY_BYTES` truncation (mark truncated)
- For streaming: either **aggregated** final assistant text in one record **or** append-only chunk records sharing `request_id` and `chunk_index` (implementation picks one; document in README)

Storage backend v1: **append-only JSONL** *or* **SQLite** single file under `AERL_DATA_DIR` — pick one in implementation plan for simplicity (JSONL easiest for pipeline ingestion).

## 6. Integration notes (self-coaching and peers)

Consumers may set:

- `OPENAI_BASE_URL` (or equivalent) to AERL’s origin + `/v1` so all LLM traffic is logged.
- `service.url` (or pipeline registry base) to the same origin for health checks and `POST /aerl/v1/jobs`.

**Contract stability:** v1 documents the above paths and env vars; **breaking changes** require semver bump and changelog entry.

## 7. Security

- Redact `Authorization` in stored logs (e.g. keep prefix `Bearer ` + last 4 chars or hash only).
- Optional: allowlist of path prefixes under `/v1` (default allow `/v1/`).
- Run behind TLS and network policy in production; AERL assumes a **trusted** orchestrator network unless auth middleware is added later.

## 8. Testing (v1 acceptance)

- Health returns 200 and version.
- Non-stream chat completion: request logged, upstream called, response returned correctly.
- Streaming completion: client receives valid SSE; at least one log record exists with expected linkage to `request_id`.
- Job endpoint: without webhook — `accepted` and log line; with mock webhook — `forwarded` or `failed` per webhook outcome.
- Truncation: oversized body is truncated and flagged in log record.

## 9. Repository layout (target)

```
AERL/
  docs/specs/2026-05-13-aerl-minimal-core-design.md
  src/aerl/          # application package
  examples/          # e.g. docker-compose: AERL + mock upstream
  README.md          # env table, curl examples, self-coaching pointer
  pyproject.toml     # or equivalent; TBD in implementation plan
```

## 10. Open items (for implementation plan, not blockers for this spec)

- Exact default for `AERL_MAX_BODY_BYTES` and streaming aggregation vs chunk records.
- Choice of ASGI framework (e.g. Starlette/FastAPI) and HTTP client (httpx).
- Whether to expose `GET /aerl/v1/jobs/{job_id}` in v1 (**out of scope** unless added in a later revision).

---

## Revision history

| Date | Author | Change |
|------|--------|--------|
| 2026-05-13 | Design session | Initial minimal-core spec (option 3 approved). |
