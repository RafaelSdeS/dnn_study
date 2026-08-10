#!/usr/bin/env python3
"""
Generate the report-only figure variants for report/ic_report.tex.

These three PNGs (kernel_restriction_cost, quant_stability_bar, extreme_compression_methods)
started as copies of presentation/make_figures.py's slide figures, hand-edited ad hoc for print
(white background, portrait bar chart) with no committed source. This script is that source now,
so the report figures can be regenerated instead of hand-patched again.

A fourth figure (accuracy_vs_latency_pareto) is a from-scratch reproduction of the Phase 6
notebook's cell 24 plot, minus the "H3:" hypothesis-numbering title -- the report shouldn't carry
that internal label, but rerunning the whole notebook (recomputes Wilcoxon tests etc.) just to
retitle one chart isn't worth it. It reads the notebook's own pre-computed
results/phase_6_hardware_profiling_analysis/h3_latency_pareto.csv, so the numbers are identical.

A fifth figure (phase6_latency_vs_kernel_size) is the same approach applied to the notebook's
cell 15 plot: reproduced here (with the report's own color palette, instead of running the whole
notebook) from its pre-computed results/phase_6_hardware_profiling_analysis/h1_winograd_per_kernel_size.csv.

Data source: results/results_aggregate/results_cross_phase.csv +
results/phase_4_compression_and_final_architecture_training/final_comparison.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

OUTPUT_DIR = Path("report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Vivid, saturated categorical palette -- the earlier muted/desaturated version read as flat
# and washed out on screen, so this trades "sober IEEE" for clearly distinguishable, punchy colors.
BLUE = "#1565c0"
RED = "#dc2626"
GREEN = "#16a34a"
PURPLE = "#9333ea"
AMBER = "#d97706"

TEXT_PRIMARY = "#1a1a1a"
TEXT_SECONDARY = "#4d4d4d"
GRID = "#d9d9d9"

df = pd.read_csv("results/results_aggregate/results_cross_phase.csv")
_p4 = pd.read_csv("results/phase_4_compression_and_final_architecture_training/final_comparison.csv")
_fire_residual_row = pd.DataFrame([{
    "base_model": "alexnet_final_fire_residual",
    "top1_%_FP32": _p4.loc[_p4["model"] == "alexnet_final_fire_residual", "top1_%"].iloc[0],
    "top1_%_INT8": _p4.loc[_p4["model"] == "alexnet_final_fire_residual_INT8", "top1_%"].iloc[0],
    "size_MB_FP32": _p4.loc[_p4["model"] == "alexnet_final_fire_residual", "size_MB"].iloc[0],
    "size_MB_INT8": _p4.loc[_p4["model"] == "alexnet_final_fire_residual_INT8", "size_MB"].iloc[0],
}])
df = pd.concat([df, _fire_residual_row], ignore_index=True)


def _style_axes(ax):
    ax.grid(axis="x", color=GRID, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=GRID, labelcolor=TEXT_SECONDARY)


# ====== Figure 1: Kernel Restriction Cost & Recovery ======
# Horizontal bars (portrait, single IEEEtran column), 4 colors grouped by narrative role
# instead of the original 5 (one per model family) -- Mixed/SmallKernel/Residual are all
# exploratory compensation attempts not detailed individually in the report text, so they
# share one color.
models_fig1 = [
    "mobilenetv2", "vgg_style", "resnet18_tv",
    "alexnet_3x3_fc", "alexnet_2x2_fc",
    "alexnet_mixed", "alexnet_small_kernel", "alexnet_residual",
    "alexnet_final_fire_residual",
]
labels_fig1 = [
    "MobileNetV2 (unrestricted)", "VGG-Style (unrestricted)", "ResNet18 (unrestricted)",
    "3×3 only (naive)", "2×2 only (naive)",
    "Mixed (3×3+2×2)", "SmallKernel (exploratory)", "Residual (exploratory)",
    "Fire-Residual (final hybrid)",
]
group_fig1 = [
    "Baselines irrestritos", "Baselines irrestritos", "Baselines irrestritos",
    "Restrição ingênua", "Restrição ingênua",
    "Tentativas de compensação", "Tentativas de compensação", "Tentativas de compensação",
    "Arquitetura híbrida final",
]
GROUP_COLORS = {
    "Baselines irrestritos": BLUE,
    "Restrição ingênua": RED,
    "Tentativas de compensação": GREEN,
    "Arquitetura híbrida final": PURPLE,
}
colors_fig1 = [GROUP_COLORS[g] for g in group_fig1]

rows = df[df["base_model"].isin(models_fig1)]
model_to_acc = dict(zip(rows["base_model"], rows["top1_%_FP32"]))
model_to_size = dict(zip(rows["base_model"], rows["size_MB_FP32"]))
accuracies = [model_to_acc[m] for m in models_fig1]
sizes = [model_to_size[m] for m in models_fig1]

fig, ax = plt.subplots(figsize=(5.4, 7), facecolor="white")
ax.set_facecolor("white")

y_pos = range(len(labels_fig1))
bars = ax.barh(y_pos, accuracies, color=colors_fig1, height=0.6, edgecolor="none")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels_fig1, fontsize=9, color=TEXT_SECONDARY)
ax.invert_yaxis()

for i, (acc, size) in enumerate(zip(accuracies, sizes)):
    ax.text(acc + 1.0, i, f"{acc:.1f}% ({size:.1f} MB)", ha="left", va="center",
            fontsize=8.5, color=TEXT_PRIMARY)

ax.set_xlabel("FP32 Top-1 Accuracy (%)", fontsize=10, color=TEXT_PRIMARY)
ax.set_xlim(0, 78)
_style_axes(ax)

legend_handles = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
          ncol=2, frameon=False, fontsize=8.5)

plt.title("Custo e recuperação da restrição de kernel", fontsize=12, fontweight="600",
          color=TEXT_PRIMARY, pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kernel_restriction_cost.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ kernel_restriction_cost.png")
plt.close()


# ====== Figure: Quantization Stability by Architecture ======
models_fig3 = ["alexnet_small_kernel", "vgg_style", "alexnet_bottleneck", "alexnet_fire"]
labels_fig3 = ["SmallKernel\n(fragile)", "VGG-Style\n(baseline)", "Bottleneck\n(robust)", "Fire\n(robust)"]

rows3 = []
for model in models_fig3:
    if model == "alexnet_small_kernel":
        rows3.append(df[(df["base_model"] == model) & (df["phase"] == "Phase 2 — AlexNet Variants")].iloc[0])
    else:
        rows3.append(df[df["base_model"] == model].sort_values("phase", ascending=False).iloc[0])
rows3 = pd.DataFrame(rows3)
quant_drops = rows3["qat_top1_drop_%"].values

bar_colors = [RED if d > 0.5 else (GREEN if d < -0.5 else AMBER) for d in quant_drops]

fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
ax.set_facecolor("white")

bars = ax.bar(labels_fig3, quant_drops, color=bar_colors, width=0.6, edgecolor="none")
ax.axhline(0, color=GRID, linestyle="-", linewidth=1, zorder=0)
for i, drop in enumerate(quant_drops):
    va = "bottom" if drop >= 0 else "top"
    y_offset = 0.2 if drop >= 0 else -0.2
    ax.text(i, drop + y_offset, f"{drop:.2f}pp", ha="center", va=va, fontsize=10, color=TEXT_PRIMARY)

ax.set_ylabel("QAT Drop (FP32 → INT8, pp)", fontsize=11, color=TEXT_PRIMARY)
ax.set_ylim(-2.5, 12)
ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=GRID, labelcolor=TEXT_SECONDARY)

plt.title("Quantization Stability by Architecture", fontsize=13, fontweight="600", color=TEXT_PRIMARY, pad=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "quant_stability_bar.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ quant_stability_bar.png")
plt.close()


# ====== Figure: Extreme Compression: Model & Quantization Trade-offs ======
MODEL_COLORS = {"MobileNetV2": BLUE, "Fire-Residual": PURPLE, "Bottleneck": GREEN, "Fire / DepthwiseSep": RED}
_model_color = {"MobileNetV2": BLUE, "Fire-Residual": PURPLE, "Bottleneck": GREEN, "Fire": RED, "DepthwiseSep": RED}

scenarios = [
    ("MobileNetV2", "FP32", 28.8, 57.99), ("Fire-Residual", "FP32", 8.09, 49.79),
    ("Bottleneck", "FP32", 4.49, 44.62), ("Fire", "FP32", 5.99, 43.98),
    ("DepthwiseSep", "FP32", 3.65, 44.39),
    ("MobileNetV2", "INT8", 7.19, 57.99), ("Fire-Residual", "INT8", 2.02, 49.20),
    ("Bottleneck", "INT8", 1.12, 44.54), ("Fire", "INT8", 1.50, 44.30),
    ("DepthwiseSep", "INT8", 0.91, 41.47),
    ("Fire-Residual", "Ternary", 0.25, 37.0), ("Bottleneck", "Ternary", 0.14, 32.0),
    ("Fire", "Ternary", 0.19, 34.5), ("DepthwiseSep", "Ternary", 0.11, 30.0),
    ("Fire-Residual", "Int4", 1.02, 42.0), ("Bottleneck", "Int4", 0.56, 38.0),
    ("Fire", "Int4", 0.75, 40.5), ("DepthwiseSep", "Int4", 0.45, 37.0),
    ("Fire-Residual", "Binary", 0.05, 15.0), ("Bottleneck", "Binary", 0.03, 12.0),
]
marker_map = {"FP32": "o", "INT8": "s", "Ternary": "^", "Int4": "D", "Binary": "v"}

fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")
ax.set_facecolor("white")

for method in ["FP32", "INT8", "Ternary", "Int4", "Binary"]:
    pts = [s for s in scenarios if s[1] == method]
    sizes = [p[2] for p in pts]
    accs = [p[3] for p in pts]
    colors = [_model_color[p[0]] for p in pts]
    ax.scatter(sizes, accs, s=200, c=colors, marker=marker_map[method],
               edgecolors=(0.1, 0.1, 0.1, 0.3), linewidth=1.5, zorder=3, label=method)

ax.set_xlabel("Model Size (MB, log scale)", fontsize=11, color=TEXT_PRIMARY)
ax.set_ylabel("Accuracy (%)", fontsize=11, color=TEXT_PRIMARY)
ax.set_xscale("log")
ax.set_xlim(0.02, 60)
ax.set_ylim(10, 65)
ax.grid(axis="both", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=GRID, labelcolor=TEXT_SECONDARY)

quant_legend = [Line2D([0], [0], marker=m, color="w", markerfacecolor=TEXT_SECONDARY, markersize=8, label=name)
                for name, m in marker_map.items()]
leg1 = ax.legend(handles=quant_legend, loc="upper left", title="Quantization", frameon=False, fontsize=9, title_fontsize=10)

model_legend = [Patch(facecolor=c, label=name) for name, c in MODEL_COLORS.items()]
ax.legend(handles=model_legend, loc="lower left", title="Model", frameon=False, fontsize=9, title_fontsize=10)
ax.add_artist(leg1)

plt.title("Extreme Compression: Model & Quantization Trade-offs", fontsize=13, fontweight="600",
          color=TEXT_PRIMARY, pad=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "extreme_compression_methods.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ extreme_compression_methods.png")
plt.close()


# ====== Figure: Accuracy vs. Latency — Pareto Frontier ======
# Reproduction of the Phase 6 notebook's pareto plot (cell 24), from its own precomputed CSV,
# with the "H3:" hypothesis-numbering title dropped and the sober palette applied.
pareto_df = pd.read_csv("results/phase_6_hardware_profiling_analysis/h3_latency_pareto.csv")
pareto_df = pareto_df.dropna(subset=["accuracy"])
PRECISION_COLORS = {"fp32": BLUE, "int8": RED}
PRECISION_MARKERS = {"fp32": "o", "int8": "s"}


def _label_points(ax, xs, ys, labels, fontsize=7.5):
    """Offset-search collision avoidance for point labels (same approach as the source notebook)."""
    candidates = [(7, 5), (7, -13), (-7, 5), (-7, -13), (7, 15), (-7, 15), (7, -23), (-7, -23)]
    placed = []
    min_dist_px = 16
    for x, y, label in zip(xs, ys, labels):
        anchor = np.array(ax.transData.transform((x, y)))
        chosen = candidates[0]
        for dx, dy in candidates:
            cand = anchor + np.array([dx, dy])
            if all(np.hypot(*(cand - p)) > min_dist_px for p in placed):
                chosen = (dx, dy)
                break
        placed.append(anchor + np.array(chosen))
        ax.annotate(label, (x, y), fontsize=fontsize, xytext=chosen, textcoords="offset points")


fig, ax = plt.subplots(figsize=(9, 6.5), facecolor="white")

for model_name, grp in pareto_df.groupby("model"):
    if set(grp["precision"]) >= {"fp32", "int8"}:
        p32 = grp[grp["precision"] == "fp32"].iloc[0]
        p8 = grp[grp["precision"] == "int8"].iloc[0]
        ax.plot([p32["latency_ms"], p8["latency_ms"]], [p32["accuracy"], p8["accuracy"]],
                color="0.75", linewidth=1, zorder=1)

for prec in ["fp32", "int8"]:
    subset = pareto_df[pareto_df["precision"] == prec]
    ax.scatter(subset["latency_ms"], subset["accuracy"], marker=PRECISION_MARKERS[prec],
               color=PRECISION_COLORS[prec], s=90, label=prec, alpha=0.9, zorder=3,
               edgecolors="white", linewidths=0.5)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
acc_pad = (pareto_df["accuracy"].max() - pareto_df["accuracy"].min()) * 0.18
ax.set_ylim(pareto_df["accuracy"].min() - acc_pad, pareto_df["accuracy"].max() + acc_pad)

_label_points(ax, pareto_df["latency_ms"].to_numpy(), pareto_df["accuracy"].to_numpy(),
              pareto_df["model"].to_numpy())

# Pareto frontier: points where no other point is both faster and more accurate.
pareto_mask = np.ones(len(pareto_df), dtype=bool)
rows = pareto_df.reset_index(drop=True)
for i in range(len(rows)):
    for j in range(len(rows)):
        if i != j:
            if (rows.iloc[j]["latency_ms"] <= rows.iloc[i]["latency_ms"] and
                rows.iloc[j]["accuracy"] >= rows.iloc[i]["accuracy"] and
                (rows.iloc[j]["latency_ms"] < rows.iloc[i]["latency_ms"] or
                 rows.iloc[j]["accuracy"] > rows.iloc[i]["accuracy"])):
                pareto_mask[i] = False
                break
pareto_points = rows[pareto_mask].sort_values("latency_ms")
if len(pareto_points) > 1:
    ax.plot(pareto_points["latency_ms"], pareto_points["accuracy"], linestyle="--", color="0.35",
            linewidth=2, label="Pareto frontier", zorder=2)

baseline = rows[rows["model"] == "alexnet_tv"].set_index("precision")["efficiency"]
ax.text(0.02, 0.02,
        f"Baseline (alexnet_tv) efficiency:\nFP32: {baseline['fp32']:.1f} acc/ms\nINT8: {baseline['int8']:.1f} acc/ms",
        transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="0.95", edgecolor="0.7", alpha=0.9))

ax.set_xlabel("Latency (ms, log scale)", fontsize=11, color=TEXT_PRIMARY)
ax.set_ylabel("Top-1 accuracy (%)", fontsize=11, color=TEXT_PRIMARY)
ax.set_title("RTX 4060 Laptop (local)", fontsize=11)
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("Accuracy vs. latency: Pareto frontier & baseline reference\n"
             "(fp32 = GPU inference, int8 = CPU inference; per-precision deployment target)",
             fontsize=12)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "phase6_accuracy_vs_latency_pareto.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ phase6_accuracy_vs_latency_pareto.png")
plt.close()


# ====== Figure: Latency per FLOP vs. Kernel Size (3 panels) ======
# Reproduction of the Phase 6 notebook's cell 15 plot, from its own precomputed CSV, with this
# report's color palette instead of the notebook's.
kernel_df = pd.read_csv("results/phase_6_hardware_profiling_analysis/h1_winograd_per_kernel_size.csv")
kernel_df = kernel_df[(kernel_df["groups_mode"] == "dense") & (kernel_df["input_resolution"] == 64)]
KSIZES = [2, 3, 5, 7, 9, 11]
IN_CH_COLORS = {16: BLUE, 32: RED, 64: GREEN, 128: PURPLE}
IN_CH_MARKERS = {16: "o", 32: "s", 64: "^", 128: "D"}
PANELS = [
    ("fp32", 1, "FP32 GPU, batch=1 (overhead-bound)"),
    ("fp32", 64, "FP32 GPU, batch=64 (compute-bound, PRIMARY)"),
    ("int8", 1, "INT8 CPU, batch=1"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")

for ax, (prec, batch, subtitle) in zip(axes, PANELS):
    ax.set_facecolor("white")
    subset = kernel_df[(kernel_df["precision"] == prec) & (kernel_df["batch_size"] == batch)]
    for in_ch, color in IN_CH_COLORS.items():
        row = subset[subset["in_ch"] == in_ch]
        if row.empty:
            continue
        row = row.iloc[0]
        latencies = [row[str(float(k))] / k**2 for k in KSIZES]
        ax.plot(KSIZES, latencies, marker=IN_CH_MARKERS[in_ch], color=color,
                label=f"in_ch={in_ch}", markersize=7, linewidth=2)
    ax.set_yscale("log")
    ax.set_xticks(KSIZES)
    ax.axvline(3, color="0.6", linestyle=":", linewidth=1)
    ax.set_xlabel("kernel_size", color=TEXT_PRIMARY)
    ax.set_title(subtitle, fontsize=10.5, color=TEXT_PRIMARY)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(colors=GRID, labelcolor=TEXT_SECONDARY)
axes[0].set_ylabel("latency / $k^2$ (ms per unit FLOP)", color=TEXT_PRIMARY)

fig.suptitle("RTX 4060 Laptop (local): layer latency per FLOP vs. kernel size (dense/groups=1, res=64)",
             fontsize=12, color=TEXT_PRIMARY)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "phase6_latency_vs_kernel_size.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ phase6_latency_vs_kernel_size.png")
plt.close()
