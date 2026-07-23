"""Phase 7 cross-phase results analysis.

Joins detection/segmentation results to Phase 3 classification results
and tests hypotheses H1-H4.

Usage:
    python scripts/phase7_analysis.py --phase7-dir runs/phase7 --results-dir results
"""
import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PhaseResult:
    arch: str
    phase: int
    task: str  # 'classification', 'detection', 'segmentation'
    metric_name: str
    metric_value: float
    fp32_value: float
    int8_value: float = None
    int8_drop: float = None


def load_phase3_results(results_dir: Path) -> pd.DataFrame:
    """Load Phase 3 classification results."""
    csv_path = results_dir / "results_aggregate" / "model_details_cross_phase.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Skipping Phase 3 data.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Keep only the three backbones we care about
    backbones = ["AlexNetBottleneck", "AlexNetFire", "AlexNetTV"]
    if "model_name" in df.columns:
        df = df[df["model_name"].isin(backbones)]
    elif "model" in df.columns:
        df = df[df["model"].isin(backbones)]

    return df


def load_phase7_results(phase7_dir: Path) -> pd.DataFrame:
    """Load Phase 7 detection/segmentation results."""
    results = []

    for run_dir in phase7_dir.glob("ssd_*"):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.json"

        if not metrics_path.exists():
            continue

        # Extract model name from run_dir
        # e.g., ssd_alexnet_bottleneck_fp32 → alexnet_bottleneck
        parts = run_dir.name.split("_")
        if parts[0] == "ssd" and len(parts) >= 2:
            model = "_".join(parts[1:-1]) if parts[-1] in ["fp32", "qat", "int8"] else "_".join(parts[1:])
        else:
            model = "unknown"

        with open(metrics_path) as f:
            metrics = json.load(f)

        if "val_mAP" in metrics:
            results.append({
                "arch": model,
                "task": "detection",
                "metric": "mAP",
                "val_mAP": np.mean(metrics["val_mAP"][-5:]) if metrics["val_mAP"] else 0,
            })

    return pd.DataFrame(results)


def test_hypothesis_h1(phase3_df: pd.DataFrame, phase7_df: pd.DataFrame):
    """H1: Small-kernel compensation transfers to dense prediction."""
    print("\n=== H1: Small-kernel compensation transfers ===")

    if phase3_df.empty or phase7_df.empty:
        print("Insufficient data")
        return

    # Extract accuracy per backbone from Phase 3
    arch_map = {
        "AlexNetBottleneck": "alexnet_bottleneck",
        "AlexNetFire": "alexnet_fire",
        "AlexNetTV": "alexnet_tv",
    }

    for old_name, new_name in arch_map.items():
        # Find phase3 row (try both column names)
        if "model_name" in phase3_df.columns:
            phase3_row = phase3_df[phase3_df["model_name"] == old_name]
        else:
            phase3_row = phase3_df[phase3_df["model"] == old_name]

        phase7_row = phase7_df[phase7_df["arch"] == new_name]

        if not phase3_row.empty and not phase7_row.empty:
            top1 = phase3_row.iloc[0].get("fp32_top1", 0)
            mAP = phase7_row.iloc[0].get("val_mAP", 0)
            print(f"  {new_name:20s}: top1={top1:.1f}% → mAP={mAP:.4f}")


def test_hypothesis_h2(phase3_df: pd.DataFrame, phase7_df: pd.DataFrame):
    """H2: Quantization robustness ranking transfers to heads."""
    print("\n=== H2: Quantization robustness transfers ===")
    print("(Requires INT8 runs — deferred until Phase 7 completes)")


def test_hypothesis_h3(phase3_df: pd.DataFrame, phase7_df: pd.DataFrame):
    """H3: Dense prediction is more receptive-field sensitive."""
    print("\n=== H3: Dense prediction is RF-sensitive ===")
    print("(Requires Phase 3 vs Phase 7 gap analysis — TODO)")


def test_hypothesis_h4(phase3_df: pd.DataFrame, phase7_df: pd.DataFrame):
    """H4: Head architecture reintroduces Winograd-irrelevant structure."""
    print("\n=== H4: Head latency dominance ===")
    print("(Requires Phase 6 profiling data — TODO)")


def main():
    parser = argparse.ArgumentParser(description="Phase 7 cross-phase analysis")
    parser.add_argument("--phase7-dir", default="runs/phase7", help="Phase 7 results directory")
    parser.add_argument("--results-dir", default="results", help="Previous phases results directory")

    args = parser.parse_args()

    phase3_df = load_phase3_results(Path(args.results_dir))
    phase7_df = load_phase7_results(Path(args.phase7_dir))

    print("\n" + "="*60)
    print("PHASE 7 CROSS-PHASE ANALYSIS")
    print("="*60)

    print(f"\nPhase 3 models loaded: {len(phase3_df)}")
    print(f"Phase 7 runs loaded: {len(phase7_df)}")

    test_hypothesis_h1(phase3_df, phase7_df)
    test_hypothesis_h2(phase3_df, phase7_df)
    test_hypothesis_h3(phase3_df, phase7_df)
    test_hypothesis_h4(phase3_df, phase7_df)

    print("\n" + "="*60)
    print("Analysis complete. See runs/phase7/ for detailed results.")
    print("="*60)


if __name__ == "__main__":
    main()
