# AERL

Agent / evaluation **request logging** layer: OpenAI-compatible HTTP surface with minimal operational endpoints.

See [docs/specs/2026-05-13-aerl-minimal-core-design.md](docs/specs/2026-05-13-aerl-minimal-core-design.md) and the implementation plan under `docs/plans/`.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```
