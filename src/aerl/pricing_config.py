from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aerl.llm_trace import PricingTable


def load_pricing_table(path: str | None) -> PricingTable | None:
    if not path or not path.strip():
        return None
    p = Path(path.strip())
    if not p.is_file():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    default = _parse_rate_pair(raw.get("default"))
    per_model: dict[str, tuple[float, float]] = {}
    pm = raw.get("per_model")
    if isinstance(pm, dict):
        for name, val in pm.items():
            if not isinstance(name, str) or not name.strip():
                continue
            pair = _parse_rate_pair(val)
            if pair is not None:
                per_model[name.strip()] = pair
    if default is None and not per_model:
        return None
    return PricingTable(default=default, per_model=per_model)


def _parse_rate_pair(obj: Any) -> tuple[float, float] | None:
    if not isinstance(obj, dict):
        return None
    inp = obj.get("input_per_million_usd")
    out = obj.get("output_per_million_usd")
    if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
        return (float(inp), float(out))
    return None
