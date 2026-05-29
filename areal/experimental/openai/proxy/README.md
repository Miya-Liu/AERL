# Internal / legacy OpenAI proxy (AReaL)

This package is **not** the supported OpenAI-compatible boundary for AERL deployments.

Use **AERL Component 1** instead:

- Package: `aerl.proxy`
- CLI: `uv run aerl` (default port **8765**)
- Spec: [docs/specs/2026-05-13-aerl-minimal-core-design.md](../../../../docs/specs/2026-05-13-aerl-minimal-core-design.md)

Code here remains for **RL rollout integration** inside the vendored AReaL infrastructure
(`proxy_rollout_server`, workflow hooks). Do not add new product features in this tree;
extend `src/aerl/proxy/` for HTTP proxy behavior.
