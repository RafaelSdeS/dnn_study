"""Recomputes every statistic cited in the report's Eixo 3 (Winograd hardware validation)
directly from the tracked Phase 6 CSVs, so those numbers stop being hand-copied from stale
notebook output.

Usage:
    python -m scripts.phase6_eixo3_stats

Reads:
    results/phase_6_hardware_profiling_analysis/layer_summary.csv
    results/phase_6_hardware_profiling_analysis/model_summary.csv
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, wilcoxon

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase_6_hardware_profiling_analysis"

DENSE_MODELS = ["vgg_style", "alexnet_bottleneck", "alexnet_final_bottleneck_residual", "alexnet_final_fire_residual"]
DEPTHWISE_MODELS = ["alexnet_depthwisesep", "mobilenetv2"]


def per_flop_wilcoxon(layer_df: pd.DataFrame, groups_mode: str) -> None:
    label = "dense (H1)" if groups_mode == "dense" else "depthwise (H2)"
    dense = layer_df[(layer_df["precision"] == "fp32") & (layer_df["groups_mode"] == groups_mode)]
    wide = dense.pivot_table(
        index=["in_ch", "out_ch", "batch_size", "input_resolution"],
        columns="kernel_size", values="latency_median_ms",
    ).reset_index()
    wide["norm_3"] = wide[3] / 3**2
    wide["norm_5"] = wide[5] / 5**2

    print(f"\n--- {label}: per-FLOP latency, k=3 vs k=5 (Wilcoxon signed-rank) ---")
    for regime, batches in [("compute-bound batch=64 (PRIMARY)", [64]), ("overhead-bound batch=1/8 (secondary)", [1, 8])]:
        subset = wide[wide["batch_size"].isin(batches)]
        stat, p = wilcoxon(subset["norm_3"], subset["norm_5"])
        m3, m5 = subset["norm_3"].median(), subset["norm_5"].median()
        lower = m3 < m5
        pct = (m5 - m3) / m5 * 100
        print(f"  [{regime}] n={len(subset)}, p={p:.3f}, stat={stat:.1f}, "
              f"median k=3 {'lower' if lower else 'NOT lower'} than k=5 "
              f"({'-' if lower else '+'}{abs(pct):.1f}%)")


def cudnn_kernel_family(layer_df: pd.DataFrame) -> None:
    print("\n--- Dense k=3, fp32: cuDNN kernel family by batch size (regime-switch check) ---")

    def classify(name: str) -> str:
        name = name.lower()
        if "winograd" in name:
            return "Winograd"
        if "xmma" in name or "cutlass" in name or "tf32" in name:
            return "TF32 (tensor core)"
        return "Direct/other"

    k3 = layer_df[(layer_df["precision"] == "fp32") & (layer_df["groups_mode"] == "dense") & (layer_df["kernel_size"] == 3)]
    k3 = k3.assign(family=k3["cudnn_kernel_name"].map(classify))
    print(k3.groupby("batch_size")["family"].value_counts().to_string())


def gflops_medians(model_df: pd.DataFrame) -> None:
    print("\n--- H2 model-level compute efficiency (fp32, GFLOP/s) ---")
    fp32 = model_df[model_df["precision"] == "fp32"]
    dense = fp32[fp32["model"].isin(DENSE_MODELS)].set_index("model")["compute_efficiency_gflops_s"]
    depthwise = fp32[fp32["model"].isin(DEPTHWISE_MODELS)].set_index("model")["compute_efficiency_gflops_s"]
    print(f"  dense models:      {dense.round(1).to_dict()} -> median={dense.median():.1f}")
    print(f"  depthwise models:  {depthwise.round(1).to_dict()} -> median={depthwise.median():.1f}")


def h4_spearman(layer_df: pd.DataFrame) -> None:
    print("\n--- H4 layer-level Spearman rho: FP32(CPU) vs INT8 latency rank (same hardware) ---")
    h4 = layer_df[layer_df["precision"].isin(["fp32_cpu", "int8"])]
    wide = h4.pivot_table(
        index=["kernel_size", "in_ch", "out_ch", "groups_mode"],
        columns="precision", values="latency_median_ms",
    ).reset_index()
    for groups_mode in ["dense", "depthwise"]:
        subset = wide[wide["groups_mode"] == groups_mode]
        rho, p = spearmanr(subset["fp32_cpu"], subset["int8"])
        print(f"  {groups_mode}: n={len(subset)}, rho={rho:.4f}, p={p:.2e}")


def main() -> None:
    layer_df = pd.read_csv(RESULTS_DIR / "layer_summary.csv")
    model_df = pd.read_csv(RESULTS_DIR / "model_summary.csv")

    per_flop_wilcoxon(layer_df, "dense")
    per_flop_wilcoxon(layer_df, "depthwise")
    cudnn_kernel_family(layer_df)
    gflops_medians(model_df)
    h4_spearman(layer_df)


if __name__ == "__main__":
    main()
