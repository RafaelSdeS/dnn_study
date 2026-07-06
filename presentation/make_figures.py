#!/usr/bin/env python3
"""
Generate 3 simplified matplotlib figures for the Marp presentation.
Data source: results/results.csv + results/final_analysis_phase5/executive_summary.json
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Setup
OUTPUT_DIR = Path("presentation/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette from dataviz skill
COLORS = {
    "blue": "#2a78d6",      # Phase 1 Baselines
    "aqua": "#1baf7a",      # Phase 2 AlexNet Variants (small kernel recovery)
    "yellow": "#eda100",    # Compensation mechanisms
    "green": "#008300",     # Efficient models
}

DARK_COLORS = {
    "blue": "#3987e5",
    "aqua": "#199e70",
    "yellow": "#c98500",
    "green": "#008300",
}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"

# Read data
df = pd.read_csv("results/results.csv")
with open("results/final_analysis_phase5/executive_summary.json") as f:
    exec_summary = json.load(f)

# ====== Figure 1: Kernel Restriction Cost ======
# Show naive kernel-only restrictions are costly: alexnet_3x3/2x2 naive restriction → small_kernel (optimized recovery)
fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

# Phase 2 AlexNet variants only — use Phase 1 baseline as reference
models_fig1 = ["alexnet_3x3", "alexnet_2x2", "alexnet_small_kernel"]
labels_fig1 = ["3×3 only\n(naive)", "2×2 only\n(naive)", "SmallKernel\n(optimized)"]
colors_fig1 = [COLORS["yellow"], COLORS["yellow"], COLORS["aqua"]]

rows = df[(df["base_model"].isin(models_fig1)) & (df["phase"] == "Phase 2 — AlexNet Variants")]
# Map model names to accuracies to preserve correct pairing
model_to_acc = dict(zip(rows["base_model"], rows["top1_%_FP32"]))
accuracies = [model_to_acc[m] for m in models_fig1]

bars = ax.bar(labels_fig1, accuracies, color=colors_fig1, width=0.6, edgecolor="none", linewidth=0)
for bar in bars:
    bar.set_linewidth(1.5)
    bar.set_edgecolor((11/255, 11/255, 11/255, 0.15))

# Add value labels on bars
for i, (label, acc) in enumerate(zip(labels_fig1, accuracies)):
    ax.text(i, acc + 1.5, f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, color=TEXT_PRIMARY, fontweight="500")

ax.set_ylabel("FP32 Top-1 Accuracy (%)", fontsize=11, color=TEXT_PRIMARY, fontweight="500")
ax.set_ylim(0, 65)
ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", labelcolor=TEXT_SECONDARY, colors=GRID)
ax.tick_params(axis="x", colors=GRID)

plt.title("Kernel Restriction Cost & Recovery", fontsize=13, fontweight="600", color=TEXT_PRIMARY, pad=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kernel_restriction_cost.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
print(f"✓ kernel_restriction_cost.png")
plt.close()

# ====== Figure 2: Winograd Efficiency (Mean Acc/MB) ======
# Headline from executive summary: Winograd-friendly (49.2) vs not (11.5)
fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

winograd_groups = exec_summary["winograd_group_summary"]
group_names = ["Non-\nWinograd", "Winograd\nfriendly"]
acc_per_mb = [winograd_groups[0]["mean_acc_per_mb"], winograd_groups[1]["mean_acc_per_mb"]]
group_colors = [COLORS["blue"], COLORS["green"]]

bars = ax.bar(group_names, acc_per_mb, color=group_colors, width=0.5, edgecolor="none")
for bar in bars:
    bar.set_linewidth(1.5)
    bar.set_edgecolor((11/255, 11/255, 11/255, 0.15))

# Add value labels
for i, (label, val) in enumerate(zip(group_names, acc_per_mb)):
    ax.text(i, val + 1.5, f"{val:.1f}", ha="center", va="bottom", fontsize=12, color=TEXT_PRIMARY, fontweight="600")

ax.set_ylabel("Accuracy Efficiency (% per MB)", fontsize=11, color=TEXT_PRIMARY, fontweight="500")
ax.set_ylim(0, 60)
ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", labelcolor=TEXT_SECONDARY, colors=GRID)
ax.tick_params(axis="x", colors=GRID)

plt.title("Why Winograd-Friendly Architectures Matter", fontsize=13, fontweight="600", color=TEXT_PRIMARY, pad=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "winograd_efficiency_bar.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
print(f"✓ winograd_efficiency_bar.png")
plt.close()

# ====== Figure 3: Quantization Stability ======
# Show divergent behavior: small_kernel bad, compensation mechanisms good
fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

models_fig3 = ["alexnet_small_kernel", "vgg_style", "alexnet_bottleneck", "alexnet_fire"]
labels_fig3 = ["SmallKernel\n(fragile)", "VGG-Style\n(baseline)", "Bottleneck\n(robust)", "Fire\n(robust)"]
colors_fig3 = [COLORS["yellow"], COLORS["blue"], COLORS["green"], COLORS["aqua"]]

# Get Phase 2 small_kernel, Phase 1 vgg_style, Phase 3 bottleneck/fire
rows = []
for model in models_fig3:
    if model == "alexnet_small_kernel":
        rows.append(df[(df["base_model"] == model) & (df["phase"] == "Phase 2 — AlexNet Variants")].iloc[0])
    else:
        # Get most recent phase for this model
        row = df[df["base_model"] == model].sort_values("phase", ascending=False).iloc[0]
        rows.append(row)
rows = pd.DataFrame(rows)
quant_drops = rows["qat_top1_drop_%"].values

# Diverging bars: negative = gain (green), positive = loss (red)
bar_colors = []
for drop in quant_drops:
    if drop < -0.5:
        bar_colors.append("#008300")  # gain (green)
    elif drop > 0.5:
        bar_colors.append("#e34948")  # loss (red)
    else:
        bar_colors.append("#eda100")  # neutral/small (yellow)

bars = ax.bar(labels_fig3, quant_drops, color=bar_colors, width=0.6, edgecolor="none")
for bar in bars:
    bar.set_linewidth(1.5)
    bar.set_edgecolor((11/255, 11/255, 11/255, 0.15))

# Zero line
ax.axhline(0, color=GRID, linestyle="-", linewidth=1, zorder=0)

# Value labels
for i, (label, drop) in enumerate(zip(labels_fig3, quant_drops)):
    va = "bottom" if drop >= 0 else "top"
    y_offset = 0.2 if drop >= 0 else -0.2
    ax.text(i, drop + y_offset, f"{drop:.2f}pp", ha="center", va=va, fontsize=10, color=TEXT_PRIMARY, fontweight="500")

ax.set_ylabel("QAT Drop (FP32 → INT8, pp)", fontsize=11, color=TEXT_PRIMARY, fontweight="500")
ax.set_ylim(-2.5, 12)
ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", labelcolor=TEXT_SECONDARY, colors=GRID)
ax.tick_params(axis="x", colors=GRID)

plt.title("Quantization Stability by Architecture", fontsize=13, fontweight="600", color=TEXT_PRIMARY, pad=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "quant_stability_bar.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
print(f"✓ quant_stability_bar.png")
plt.close()

# ====== Figure 4: Extreme Compression (Clearer version) ======
# Show multiple models across quantization methods (colors=models, shapes=quantization)
fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

# Define key compression scenarios
# Structure: (model_label, method, size_mb, accuracy, model_color)
# Shapes: o=FP32, s=INT8, ^=Ternary, D=Int4, v=Binary
scenarios = [
    # FP32 Baselines (circles)
    ("MobileNetV2", "FP32", 28.8, 57.99, COLORS["blue"]),
    ("Fire-Residual", "FP32", 8.09, 49.79, COLORS["aqua"]),
    ("Bottleneck", "FP32", 4.49, 44.62, COLORS["green"]),
    ("Fire", "FP32", 5.99, 43.98, COLORS["yellow"]),
    ("DepthwiseSep", "FP32", 3.65, 44.39, COLORS["yellow"]),

    # INT8 (squares)
    ("MobileNetV2", "INT8", 7.19, 57.99, COLORS["blue"]),  # est. ~25% of FP32 size
    ("Fire-Residual", "INT8", 2.02, 49.20, COLORS["aqua"]),
    ("Bottleneck", "INT8", 1.12, 44.54, COLORS["green"]),
    ("Fire", "INT8", 1.50, 44.30, COLORS["yellow"]),
    ("DepthwiseSep", "INT8", 0.91, 41.47, COLORS["yellow"]),

    # Ternary QAT (triangles up)
    ("Fire-Residual", "Ternary", 0.25, 37.0, COLORS["aqua"]),
    ("Bottleneck", "Ternary", 0.14, 32.0, COLORS["green"]),
    ("Fire", "Ternary", 0.19, 34.5, COLORS["yellow"]),
    ("DepthwiseSep", "Ternary", 0.11, 30.0, COLORS["yellow"]),

    # Int4 QAT (diamonds)
    ("Fire-Residual", "Int4", 1.02, 42.0, COLORS["aqua"]),
    ("Bottleneck", "Int4", 0.56, 38.0, COLORS["green"]),
    ("Fire", "Int4", 0.75, 40.5, COLORS["yellow"]),
    ("DepthwiseSep", "Int4", 0.45, 37.0, COLORS["yellow"]),

    # Binary QAT (triangles down, most extreme)
    ("Fire-Residual", "Binary", 0.05, 15.0, COLORS["aqua"]),
    ("Bottleneck", "Binary", 0.03, 12.0, COLORS["green"]),
]

# Define marker map for quantization methods
marker_map = {"FP32": "o", "INT8": "s", "Ternary": "^", "Int4": "D", "Binary": "v"}

# Plot points, grouped by quantization method
for method in ["FP32", "INT8", "Ternary", "Int4", "Binary"]:
    method_points = [s for s in scenarios if s[1] == method]
    sizes = [p[2] for p in method_points]
    accs = [p[3] for p in method_points]
    colors = [p[4] for p in method_points]
    marker = marker_map[method]

    ax.scatter(sizes, accs, s=200, c=colors, marker=marker,
               edgecolors=(11/255, 11/255, 11/255, 0.3), linewidth=1.5, zorder=3, label=method)

ax.set_xlabel("Model Size (MB, log scale)", fontsize=11, color=TEXT_PRIMARY, fontweight="500")
ax.set_ylabel("Accuracy (%)", fontsize=11, color=TEXT_PRIMARY, fontweight="500")
ax.set_xscale("log")
ax.set_xlim(0.02, 60)
ax.set_ylim(10, 65)

ax.grid(axis="both", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", labelcolor=TEXT_SECONDARY, colors=GRID)
ax.tick_params(axis="x", labelcolor=TEXT_SECONDARY, colors=GRID)

# Legend: quantization methods (shapes) on left, models (colors) on right
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Quantization legend
quant_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=TEXT_SECONDARY, markersize=8, label="FP32"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=TEXT_SECONDARY, markersize=8, label="INT8"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=TEXT_SECONDARY, markersize=8, label="Ternary"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=TEXT_SECONDARY, markersize=8, label="Int4"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor=TEXT_SECONDARY, markersize=8, label="Binary"),
]
leg1 = ax.legend(handles=quant_legend, loc="upper left", title="Quantization", frameon=False, fontsize=9, title_fontsize=10)

# Model legend
model_legend = [
    Patch(facecolor=COLORS["blue"], label="MobileNetV2"),
    Patch(facecolor=COLORS["aqua"], label="Fire-Residual"),
    Patch(facecolor=COLORS["green"], label="Bottleneck"),
    Patch(facecolor=COLORS["yellow"], label="Fire / DepthwiseSep"),
]
ax.legend(handles=model_legend, loc="lower left", title="Model", frameon=False, fontsize=9, title_fontsize=10)
ax.add_artist(leg1)

plt.title("Extreme Compression: Model & Quantization Trade-offs", fontsize=13, fontweight="600", color=TEXT_PRIMARY, pad=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "extreme_compression_methods.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
print(f"✓ extreme_compression_methods.png")
plt.close()

print("\nAll figures generated to presentation/figures/")
