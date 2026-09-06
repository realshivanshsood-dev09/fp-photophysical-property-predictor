from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def fixture_records() -> list[dict]:
    fixture = Path(__file__).parent / "fixtures" / "fpbase_small.json"
    return json.loads(fixture.read_text(encoding="utf-8"))["records"]


def training_frame() -> pd.DataFrame:
    """Fifteen independent synthetic rows for pipeline mechanics, not biology."""
    rows = []
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for index in range(15):
        sequence = "M" + "A" * 30 + alphabet[index] + "G" * 20 + alphabet[(index + 3) % len(alphabet)]
        rows.append({
            "record_id": f"synthetic-{index}", "name": f"synthetic-{index}", "sequence": sequence,
            "brightness": float(index + 1), "group_id": f"group-{index}",
        })
    return pd.DataFrame(rows)
