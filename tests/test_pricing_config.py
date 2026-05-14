from __future__ import annotations

import json
from pathlib import Path

from aerl.pricing_config import load_pricing_table


def test_load_pricing_table_missing_returns_none():
    assert load_pricing_table(None) is None
    assert load_pricing_table("") is None


def test_load_pricing_table_reads_file(tmp_path):
    p = tmp_path / "pricing.json"
    p.write_text(
        json.dumps(
            {
                "default": {"input_per_million_usd": 1.0, "output_per_million_usd": 2.0},
                "per_model": {"m-1": {"input_per_million_usd": 3.0, "output_per_million_usd": 4.0}},
            }
        ),
        encoding="utf-8",
    )
    t = load_pricing_table(str(p))
    assert t is not None
    assert t.default == (1.0, 2.0)
    assert t.per_model["m-1"] == (3.0, 4.0)
    assert t.rates_for_model("unknown") == (1.0, 2.0)
    assert t.rates_for_model("m-1") == (3.0, 4.0)


def test_load_pricing_nonexistent_path(tmp_path):
    assert load_pricing_table(str(tmp_path / "nope.json")) is None
