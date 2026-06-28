import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


def disk_mb(path: str | Path) -> float:
    """File size in MB; NaN if file doesn't exist."""
    p = Path(path)
    return p.stat().st_size / (1024 ** 2) if p.exists() else float("nan")


def build_comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Build a sorted comparison DataFrame (by precision then descending top-1)."""
    return (
        pd.DataFrame(rows)
        .sort_values(["precision", "top1_%"], ascending=[True, False])
        .reset_index(drop=True)
    )


def create_results_summary(
    results: dict[str, Any],
    config,
    output_path: str | Path,
) -> None:
    """
    Save experiment results + config to JSON.

    `config` can be a dataclass (uses asdict), a dict, or any object with
    a to_dict() method.
    """
    if hasattr(config, "__dataclass_fields__"):
        cfg_dict = asdict(config)
    elif hasattr(config, "to_dict"):
        cfg_dict = config.to_dict()
    else:
        cfg_dict = dict(config)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({"config": cfg_dict, **results}, f, indent=2, default=str)
