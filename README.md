# AERL

Agent / evaluation **request logging** layer: OpenAI-compatible HTTP surface with minimal operational endpoints.

See [docs/specs/2026-05-13-aerl-minimal-core-design.md](docs/specs/2026-05-13-aerl-minimal-core-design.md) and the implementation plan under `docs/plans/`.

## Development

```bash
uv sync --extra dev
uv run pytest -q
```

(Or `python -m venv .venv && pip install -e '.[dev]'` then `pytest`.)

## Readiness probe

Optional upstream check for `GET /ready` when `AERL_READY_CHECK_UPSTREAM=true`: probes `GET {UPSTREAM_OPENAI_BASE_URL}/{AERL_READY_PROBE_PATH}` (default path segment `models`). Set `AERL_READY_AUTH` to a `Bearer …` value if the upstream requires auth for that probe.
