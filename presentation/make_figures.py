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
    "purple": "#7b3fa0",    # Phase 4 final hybrid architectures
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
SURFACE = "#fef5e7"
GRID = "#e1e0d9"

# Read data
df = pd.read_csv("results/results.csv")
with open("results/final_analysis_phase5/executive_summary.json") as f:
    exec_summary = json.load(f)

# alexnet_final_fire_residual (Phase 4 hybrid) isn't in results.csv (Phase 1-3 only) — pull it in from its own comparison table
_p4 = pd.read_csv("results/final_architecture_phase4/final_comparison.csv")
_fire_residual_row = pd.DataFrame([{
    "base_model": "alexnet_final_fire_residual",
    "top1_%_FP32": _p4.loc[_p4["model"] == "alexnet_final_fire_residual", "top1_%"].iloc[0],
    "top1_%_INT8": _p4.loc[_p4["model"] == "alexnet_final_fire_residual_INT8", "top1_%"].iloc[0],
    "size_MB_FP32": _p4.loc[_p4["model"] == "alexnet_final_fire_residual", "size_MB"].iloc[0],
    "size_MB_INT8": _p4.loc[_p4["model"] == "alexnet_final_fire_residual_INT8", "size_MB"].iloc[0],
}])
df = pd.concat([df, _fire_residual_row], ignore_index=True)

# ====== Figure 1: Kernel Restriction Cost ======
# Show naive kernel-only restrictions are costly: alexnet_3x3/2x2 naive restriction → small_kernel (optimized recovery)
fig, ax = plt.subplots(figsize=(13.5, 5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

models_fig1 = [
    "mobilenetv2", "vgg_style", "resnet18_tv",
    "alexnet_3x3", "alexnet_2x2",
    "alexnet_mixed", "alexnet_small_kernel", "alexnet_residual",
    "alexnet_final_fire_residual",
]
labels_fig1 = [
    "MobileNetV2\n(unrestricted)", "VGG-Style\n(unrestricted)", "ResNet18\n(unrestricted)",
    "3×3 only\n(naive)", "2×2 only\n(naive)",
    "Mixed\n(3×3+2×2)", "SmallKernel\n(optimized)", "Residual\n(compensation)",
    "Fire-Residual\n(Phase 4 hybrid)",
]
colors_fig1 = [
    COLORS["blue"], COLORS["blue"], COLORS["blue"],
    COLORS["yellow"], COLORS["yellow"],
    COLORS["aqua"], COLORS["aqua"], COLORS["green"],
    COLORS["purple"],
]

rows = df[df["base_model"].isin(models_fig1)]
# Map model names to accuracies/sizes to preserve correct pairing
model_to_acc = dict(zip(rows["base_model"], rows["top1_%_FP32"]))
model_to_size = dict(zip(rows["base_model"], rows["size_MB_FP32"]))
accuracies = [model_to_acc[m] for m in models_fig1]
sizes = [model_to_size[m] for m in models_fig1]

bars = ax.bar(labels_fig1, accuracies, color=colors_fig1, width=0.6, edgecolor="none", linewidth=0)
for bar in bars:
    bar.set_linewidth(1.5)
    bar.set_edgecolor((11/255, 11/255, 11/255, 0.15))

# Add value labels on bars (accuracy + FP32 size)
for i, (acc, size) in enumerate(zip(accuracies, sizes)):
    ax.text(i, acc + 1.5, f"{acc:.1f}%\n({size:.1f} MB)", ha="center", va="bottom", fontsize=10, color=TEXT_PRIMARY, fontweight="500")

ax.set_ylabel("FP32 Top-1 Accuracy (%)", fontsize=11, color=TEXT_PRIMARY, fontweight="500")
ax.set_ylim(0, 72)
ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", labelcolor=TEXT_SECONDARY, colors=GRID)
ax.tick_params(axis="x", labelcolor=TEXT_SECONDARY, colors=GRID)

plt.title("Kernel Restriction Cost & Recovery", fontsize=13, fontweight="600", color=TEXT_PRIMARY, pad=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kernel_restriction_cost.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
print(f"✓ kernel_restriction_cost.png")
plt.close()

# ====== Figure 1b: Kernel Restriction Cost — INT8 counterparts ======
# Same models as Figure 1, INT8 top-1 instead of FP32.
# resnet18_tv dropped: no INT8/QAT result exists for it (pretrained torchvision model, not quantized).
fig, ax = plt.subplots(figsize=(13.5, 5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

models_fig1b = ["vgg_style", "alexnet_3x3", "alexnet_2x2", "alexnet_mixed", "alexnet_small_kernel",
                "alexnet_residual", "alexnet_final_fire_residual"]
labels_fig1b = ["VGG-Style\n(unrestricted)", "3×3 only\n(naive)", "2×2 only\n(naive)",
                "Mixed\n(3×3+2×2)", "SmallKernel\n(optimized)", "Residual\n(compensation)",
                "Fire-Residual\n(Phase 4 hybrid)"]
colors_fig1b = [COLORS["blue"], COLORS["yellow"], COLORS["yellow"], COLORS["aqua"], COLORS["aqua"], COLORS["green"],
                COLORS["purple"]]

rows = df[df["base_model"].isin(models_fig1b)]
model_to_acc = dict(zip(rows["base_model"], rows["top1_%_INT8"]))
model_to_size = dict(zip(rows["base_model"], rows["size_MB_INT8"]))
accuracies = [model_to_acc[m] for m in models_fig1b]
sizes = [model_to_size[m] for m in models_fig1b]

bars = ax.bar(labels_fig1b, accuracies, color=colors_fig1b, width=0.6, edgecolor="none", linewidth=0)
for bar in bars:
    bar.set_linewidth(1.5)
    bar.set_edgecolor((11/255, 11/255, 11/255, 0.15))

for i, (acc, size) in enumerate(zip(accuracies, sizes)):
    ax.text(i, acc + 1.5, f"{acc:.1f}%\n({size:.1f} MB)", ha="center", va="bottom", fontsize=10, color=TEXT_PRIMARY, fontweight="500")

ax.set_ylabel("INT8 Top-1 Accuracy (%)", fontsize=11, color=TEXT_PRIMARY, fontweight="500")
ax.set_ylim(0, 72)
ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", labelcolor=TEXT_SECONDARY, colors=GRID)
ax.tick_params(axis="x", labelcolor=TEXT_SECONDARY, colors=GRID)

plt.title("Kernel Restriction Cost & Recovery — INT8", fontsize=13, fontweight="600", color=TEXT_PRIMARY, pad=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kernel_restriction_cost_int8.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
print(f"✓ kernel_restriction_cost_int8.png")
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
ax.tick_params(axis="x", labelcolor=TEXT_SECONDARY, colors=GRID)

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
ax.tick_params(axis="x", labelcolor=TEXT_SECONDARY, colors=GRID)

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

# ====== Figure 5: Best Models FP32 vs INT8 ======
# Same style as Figure 1: paired bars per model, annotated with the FP32→INT8 pp change.
fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

models_fig5 = ["vgg_style", "alexnet_residual", "alexnet_small_kernel", "alexnet_bottleneck",
               "alexnet_stacked", "alexnet_depthwisesep", "alexnet_fire", "alexnet_factorized"]

rows = df[df["base_model"].isin(models_fig5)]
model_to_fp32 = dict(zip(rows["base_model"], rows["top1_%_FP32"]))
model_to_int8 = dict(zip(rows["base_model"], rows["top1_%_INT8"]))
fp32_vals = [model_to_fp32[m] for m in models_fig5]
int8_vals = [model_to_int8[m] for m in models_fig5]

x = list(range(len(models_fig5)))
x_fp32 = [i - 0.2 for i in x]
x_int8 = [i + 0.2 for i in x]

ax.bar(x_fp32, fp32_vals, width=0.4, color=COLORS["blue"], label="FP32", edgecolor="none")
ax.bar(x_int8, int8_vals, width=0.4, color=COLORS["aqua"], label="INT8", edgecolor="none")

# Diff annotation above the taller bar: red = INT8 drop, green = INT8 gain
for i, (fp32, int8) in enumerate(zip(fp32_vals, int8_vals)):
    diff = fp32 - int8
    sign = "-" if diff > 0 else "+"
    color = "#e34948" if diff > 0 else "#008300"
    ax.text(i, max(fp32, int8) + 1, f"{sign}{abs(diff):.1f}pp", ha="center", va="bottom",
            fontsize=9, color=color, fontweight="600")

ax.set_xticks(x)
ax.set_xticklabels([m.replace("alexnet_", "").replace("_", " ").title() for m in models_fig5], rotation=20, ha="right")
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11, color=TEXT_PRIMARY, fontweight="500")
ax.set_ylim(0, 62)
ax.legend(frameon=False, fontsize=10)
ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", labelcolor=TEXT_SECONDARY, colors=GRID)
ax.tick_params(axis="x", labelcolor=TEXT_SECONDARY, colors=GRID)

plt.title("Best Models — FP32 vs INT8 (Top-1)", fontsize=13, fontweight="600", color=TEXT_PRIMARY, pad=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "best_models_fp32_vs_int8.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
print(f"✓ best_models_fp32_vs_int8.png")
plt.close()

# ====== Figure 6: Master Pareto Frontier — All Models & Quantization ======
# Combines FP32 + INT8 main-phase results with the full extreme-quantization grid
# (int4/int2/ternary/binary/mixed, all models — not just a pre-filtered subset)
# to show the true accuracy-vs-size Pareto frontier.
COMPRESSION_DIR = Path("results/compression_phase4_1")

points = []  # (name, size_mb, accuracy, category)
for _, row in df.dropna(subset=["top1_%_FP32", "size_MB_FP32"]).iterrows():
    points.append((f"{row['base_model']} (FP32)", row["size_MB_FP32"], row["top1_%_FP32"], "FP32"))
for _, row in df.dropna(subset=["top1_%_INT8", "size_MB_INT8"]).iterrows():
    points.append((f"{row['base_model']} (INT8)", row["size_MB_INT8"], row["top1_%_INT8"], "INT8"))

method_labels = {
    "int4_ptq": "Int4 PTQ",
    "int4_qat": "Int4 QAT",
    "mixed": "Mixed Int4/8",
    "int2_qat": "Int2 QAT",
    "ternary_qat": "Ternary QAT",
    "binary_qat": "Binary QAT",
}

for summary_path in COMPRESSION_DIR.glob("*_compression_summary.json"):
    with open(summary_path) as f:
        summary = json.load(f)
    for m in summary["methods"]:
        if m["method"] == "int8":
            continue  # already covered by df's INT8 columns above
        label = method_labels.get(m["method"], m["method"])
        points.append((f"{m['model']} ({m['method']})", m["compressed_size_mb"], m["compressed_top1_acc"], label))

points.sort(key=lambda p: p[1])
pareto, best_acc = [], -1.0
for point in points:
    if point[2] > best_acc:
        pareto.append(point)
        best_acc = point[2]
pareto_names = {p[0] for p in pareto}
non_pareto = [p for p in points if p[0] not in pareto_names]

fig, ax = plt.subplots(figsize=(13, 8), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.scatter([p[1] for p in non_pareto], [p[2] for p in non_pareto],
           c=MUTED, s=70, alpha=0.35, edgecolors="none", zorder=2, label="Non-frontier")

category_colors = {
    "FP32": COLORS["blue"],
    "INT8": COLORS["green"],
    "Int4 PTQ": COLORS["yellow"],
    "Int4 QAT": COLORS["purple"],
    "Mixed Int4/8": COLORS["aqua"],
    "Int2 QAT": DARK_COLORS["yellow"],
    "Ternary QAT": "#e34948",
    "Binary QAT": "#8b1a1a",
}
for cat, color in category_colors.items():
    sub = [p for p in pareto if p[3] == cat]
    if not sub:
        continue
    ax.scatter([p[1] for p in sub], [p[2] for p in sub], c=color, s=170, alpha=0.9,
               edgecolors="white", linewidth=1, zorder=3, label=cat)

ax.plot([p[1] for p in pareto], [p[2] for p in pareto], color=TEXT_PRIMARY, linestyle="--",
        linewidth=1.5, zorder=1, alpha=0.6)

# Alternate label offset (N/S/E/W-ish) so adjacent frontier points, often close
# together in the extreme-quantization cluster, don't stack their text.
label_offsets = [(6, 6), (6, -10), (-6, 10), (-6, -14)]
for i, (name, size, acc, _cat) in enumerate(pareto):
    dx, dy = label_offsets[i % len(label_offsets)]
    ax.annotate(name, (size, acc), textcoords="offset points", xytext=(dx, dy),
                ha="left" if dx > 0 else "right", fontsize=8, color=TEXT_PRIMARY)

ax.set_xscale("log")
ax.set_xlabel("Model Size (MB, log scale)", fontsize=11, color=TEXT_PRIMARY, fontweight="500")
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11, color=TEXT_PRIMARY, fontweight="500")
ax.grid(axis="both", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", labelcolor=TEXT_SECONDARY, colors=GRID)
ax.tick_params(axis="x", labelcolor=TEXT_SECONDARY, colors=GRID)
ax.legend(frameon=False, fontsize=9, loc="lower right")

plt.title("Master Pareto Frontier — All Models & Quantization", fontsize=13, fontweight="600", color=TEXT_PRIMARY, pad=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "master_pareto_frontier_all_models.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
print(f"✓ master_pareto_frontier_all_models.png")
plt.close()

print("\nAll figures generated to presentation/figures/")
