"""Aggregate per-model summary JSONs from a `scripts.cluster submit-sweep` run into one CSV.

Each sweep job runs a single model and writes its own results/{model}_summary.json under
{runtime.root}/{experiment}/{model}/results/ (see scripts/train.py's _make_model_runs). Reading those
back — rather than relying on scripts/train.py's own {experiment}_comparison.csv, which each concurrent
job overwrites — avoids the write race between sibling jobs sharing one experiment name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from configs.loader import load_config
from ml import expand_path


def _load_runtime_config(runtime: str) -> dict:
    candidate = Path(runtime)
    if candidate.exists():
        import yaml
        return yaml.safe_load(candidate.read_text()) or {}
    return load_config(f"runtime/{runtime}.yaml")


def aggregate(experiment: str, runtime: str) -> Path:
    runtime_cfg = _load_runtime_config(runtime)
    root = expand_path(runtime_cfg.get("root"), default="outputs/pcad") or Path("outputs/pcad")
    summary_paths = sorted((root / experiment).glob("*/results/*_summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No summary JSONs found under {root / experiment}/*/results/")

    rows = [json.loads(p.read_text()) for p in summary_paths]
    output_path = root / "results_aggregate" / f"{experiment}_final_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate per-model sweep summaries into one comparison CSV.")
    parser.add_argument("--experiment", default="large_scale")
    parser.add_argument("--runtime", default="pcad")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = aggregate(args.experiment, args.runtime)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
