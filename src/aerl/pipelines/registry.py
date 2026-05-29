from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_MANIFESTS_DIR = Path(__file__).resolve().parent / "manifests"
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    id: str
    description: str
    requires_gpu: bool
    requires_inference: bool
    timeout_seconds: int
    entrypoint: str | None = None
    default_config: str | None = None
    runner: str | None = None

    def resolve_entrypoint(self) -> Path | None:
        if not self.entrypoint:
            return None
        return (_REPO_ROOT / self.entrypoint).resolve()

    def resolve_default_config(self) -> Path | None:
        if not self.default_config:
            return None
        return (_REPO_ROOT / self.default_config).resolve()


def _load_manifest(path: Path) -> PipelineSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw.get("id"):
        raise ValueError(f"Invalid pipeline manifest: {path}")
    return PipelineSpec(
        id=str(raw["id"]),
        description=str(raw.get("description", "")),
        requires_gpu=bool(raw.get("requires_gpu", True)),
        requires_inference=bool(raw.get("requires_inference", True)),
        timeout_seconds=int(raw.get("timeout_seconds", 86400)),
        entrypoint=raw.get("entrypoint"),
        default_config=raw.get("default_config"),
        runner=raw.get("runner"),
    )


def list_pipelines() -> list[PipelineSpec]:
    specs: list[PipelineSpec] = []
    for path in sorted(_MANIFESTS_DIR.glob("*.yaml")):
        specs.append(_load_manifest(path))
    return specs


def get_pipeline(pipeline_id: str) -> PipelineSpec:
    for spec in list_pipelines():
        if spec.id == pipeline_id:
            return spec
    raise KeyError(f"Unknown pipeline: {pipeline_id!r}")


def load_runner(spec: PipelineSpec) -> Any:
    if not spec.runner:
        raise ValueError(f"Pipeline {spec.id!r} has no in-process runner")
    module_name, _, attr = spec.runner.partition(":")
    if not attr:
        raise ValueError(f"Invalid runner spec: {spec.runner!r}")
    mod = importlib.import_module(module_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"Runner {spec.runner!r} is not callable")
    return fn
