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

# Shared model->group map, reused by every other figure below that plots individual models
# (Figures 3 and 6) so the color coding stays identical to this one across the whole report.
# Two spellings of ResNet18 appear because the source CSVs disagree (resnet18_tv vs resnet18tv).
MODEL_GROUP = {
    "alexnet_tv": "Baselines irrestritos", "mobilenetv2": "Baselines irrestritos",
    "resnet18_tv": "Baselines irrestritos", "resnet18tv": "Baselines irrestritos",
    "vgg_style": "Baselines irrestritos",
    "alexnet_3x3_fc": "Restrição ingênua", "alexnet_2x2_fc": "Restrição ingênua",
    "alexnet_3x3_gap": "Restrição ingênua",
    "alexnet_mixed": "Tentativas de compensação", "alexnet_small_kernel": "Tentativas de compensação",
    "alexnet_residual": "Tentativas de compensação", "alexnet_bottleneck": "Tentativas de compensação",
    "alexnet_fire": "Tentativas de compensação", "alexnet_depthwisesep": "Tentativas de compensação",
    "alexnet_dilated_gap": "Tentativas de compensação",
    "alexnet_final_fire_residual": "Arquitetura híbrida final",
    "alexnet_final_bottleneck_residual": "Arquitetura híbrida final",
    "alexnet_final_bottleneck_fire": "Arquitetura híbrida final",
    "alexnet_final_depthwise_fire": "Arquitetura híbrida final",
    "alexnet_fire_bypass": "Arquitetura híbrida final",
}

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


# ====== Figure: Compression Methods — real Phase 4.1 measurements ======
# Same horizontal-bar language as Figure 1 (kernel_restriction_cost), but for the 6 quantization
# methods from compression_phase4_1.ipynb, averaged across their target models. Previously this
# figure used hand-typed placeholder numbers (a `scenarios` list of guessed values) that didn't
# match the real notebook results and only covered 5 of the 7 methods named in the report caption
# -- replaced with the actual results/.../compression_by_method.csv aggregates.
comp_df = pd.read_csv("results/phase_4_compression_and_final_architecture_training/compression_by_method.csv")
COMP_METHOD_ORDER = ["int8", "int4_ptq", "int4_qat", "mixed", "ternary_qat", "int2_qat", "binary_qat"]
COMP_LABELS = {
    "int8": "INT8 (âncora)", "int4_ptq": "INT4 PTQ (sem retreino)",
    "int4_qat": "INT4 QAT (com retreino)", "mixed": "Mista INT4/INT8 (QAT)",
    "ternary_qat": "Ternary QAT", "int2_qat": "INT2 QAT", "binary_qat": "Binary QAT",
}
COMP_GROUP = {
    "int8": "INT8 (âncora)", "int4_ptq": "INT4 PTQ (sem retreino)",
    "int4_qat": "INT4 QAT (com retreino)", "mixed": "Mista INT4/INT8 (QAT)",
    "ternary_qat": "Sub-INT4 agressivo (QAT)", "int2_qat": "Sub-INT4 agressivo (QAT)",
    "binary_qat": "Sub-INT4 agressivo (QAT)",
}
COMP_GROUP_COLORS = {
    "INT8 (âncora)": BLUE, "INT4 PTQ (sem retreino)": RED,
    "INT4 QAT (com retreino)": GREEN, "Mista INT4/INT8 (QAT)": PURPLE,
    "Sub-INT4 agressivo (QAT)": AMBER,
}
comp_df = comp_df.set_index("method").loc[COMP_METHOD_ORDER].reset_index()
comp_df["label"] = comp_df["method"].map(COMP_LABELS)
comp_df["color"] = comp_df["method"].map(COMP_GROUP).map(COMP_GROUP_COLORS)

fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
ax.set_facecolor("white")

y_pos = range(len(comp_df))
ax.barh(y_pos, comp_df["mean_top1"], color=comp_df["color"], height=0.6, edgecolor="none")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(comp_df["label"], fontsize=13, color=TEXT_SECONDARY)
ax.invert_yaxis()

for i, row in comp_df.reset_index(drop=True).iterrows():
    ax.text(row["mean_top1"] + 1.0, i,
             f"{row['mean_top1']:.1f}% ({row['mean_ratio']:.1f}$\\times$ menor que FP32, n={row['n']})",
            ha="left", va="center", fontsize=12, color=TEXT_PRIMARY)

ax.set_xlabel("Top-1 accuracy média (%)", fontsize=13, color=TEXT_PRIMARY)
ax.tick_params(axis="x", labelsize=11)
ax.set_xlim(0, 82)
_style_axes(ax)

legend_handles = [Patch(facecolor=c, label=g) for g, c in COMP_GROUP_COLORS.items()]
# bbox_to_anchor is in axes-fraction coords; plt.tight_layout() (removed below) fights this by
# shrinking the axes to fit the legend inside the original figsize, which paradoxically pushes the
# per-bar annotation text further into the legend's column -- bbox_inches="tight" on savefig alone
# already crops the canvas to content, so tight_layout() isn't needed here.
ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
          ncol=1, frameon=False, fontsize=12)

fig.suptitle("Métodos de compressão além de INT8\n"
             "(taxa = tamanho do checkpoint FP32 ÷ tamanho do checkpoint comprimido)",
             fontsize=12, color=TEXT_PRIMARY)
plt.savefig(OUTPUT_DIR / "extreme_compression_methods.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ extreme_compression_methods.png")
plt.close()


# ====== Figure: Accuracy vs. Latency — Pareto Frontier ======
# Reproduction of the Phase 6 notebook's pareto plot (cell 24), from its own precomputed CSV,
# with the "H3:" hypothesis-numbering title dropped and the sober palette applied.
pareto_df = pd.read_csv("results/phase_6_hardware_profiling_analysis/h3_latency_pareto.csv")
pareto_df = pareto_df.dropna(subset=["accuracy"])
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
    point_colors = subset["model"].map(MODEL_GROUP).map(GROUP_COLORS)
    ax.scatter(subset["latency_ms"], subset["accuracy"], marker=PRECISION_MARKERS[prec],
               c=point_colors, s=90, alpha=0.9, zorder=3, edgecolors="white", linewidths=0.5)

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

shape_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markersize=8, label="fp32"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="0.4", markersize=8, label="int8"),
    Line2D([0], [0], color="0.35", linestyle="--", linewidth=2, label="Pareto frontier"),
]
leg1 = ax.legend(handles=shape_legend, loc="upper left", fontsize=8, frameon=False)
group_legend = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
ax.legend(handles=group_legend, loc="lower right", fontsize=8, frameon=False)
ax.add_artist(leg1)
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


# ====== Figure: Accuracy vs. Size — every model discussed in this report, FP32 & INT8 ======
# Same visual language as the accuracy-vs-latency Pareto figure above (gray FP32->INT8 connector,
# BLUE/RED circle/square markers, log-x, offset-search labels) but x-axis is model size instead of
# latency. Phase 4/9 hybrids aren't in results_cross_phase.csv (three different CSV schemas across
# phases -- same situation the phase5 notebook already flagged), so those rows are pulled from their
# own source CSVs and concatenated, same approach as the alexnet_final_fire_residual splice above.
size_acc_models = [
    "alexnet_tv", "alexnet_3x3_fc", "alexnet_3x3_gap", "alexnet_2x2_fc", "alexnet_small_kernel",
    "alexnet_bottleneck", "alexnet_fire", "mobilenetv2", "resnet18_tv", "vgg_style",
]
# alexnet_tv has two independent training runs in this CSV (79 vs. 67 epochs, see report
# Limitations); the report standardizes on the 79-epoch "Phase 1 — Baselines" run everywhere,
# so drop the other one here too or it'd plot as a second, contradictory AlexNet-irrestrita point.
_size_acc_source = df[~((df["base_model"] == "alexnet_tv") & (df["phase"] != "Phase 1 — Baselines"))]
size_acc_rows = _size_acc_source[_size_acc_source["base_model"].isin(size_acc_models)].rename(columns={
    "base_model": "model", "top1_%_FP32": "fp32_top1", "size_MB_FP32": "fp32_size_mb",
    "top1_%_INT8": "int8_top1", "size_MB_INT8": "int8_size_mb",
})[["model", "fp32_top1", "fp32_size_mb", "int8_top1", "int8_size_mb"]]

_p4_pairs = []
for base in ["alexnet_final_fire_residual", "alexnet_final_bottleneck_fire",
             "alexnet_final_bottleneck_residual", "alexnet_final_depthwise_fire"]:
    fp32_row = _p4.loc[_p4["model"] == base].iloc[0]
    int8_row = _p4.loc[_p4["model"] == f"{base}_INT8"].iloc[0]
    _p4_pairs.append({"model": base, "fp32_top1": fp32_row["top1_%"], "fp32_size_mb": fp32_row["size_MB"],
                       "int8_top1": int8_row["top1_%"], "int8_size_mb": int8_row["size_MB"]})

_p9 = pd.read_csv("outputs/pcad/results_aggregate/results_phase_9_fire_bypass_large_scale.csv").iloc[0]
_p4_pairs.append({"model": "alexnet_fire_bypass", "fp32_top1": _p9["fp32_top1"],
                   "fp32_size_mb": _p9["fp32_size_mb"], "int8_top1": _p9["int8_top1"],
                   "int8_size_mb": _p9["int8_size_mb"]})

size_acc_df = pd.concat([size_acc_rows, pd.DataFrame(_p4_pairs)], ignore_index=True)

DISPLAY_NAME = {
    "alexnet_tv": "AlexNet (irrestrita)", "alexnet_3x3_fc": "AlexNet3x3-FC",
    "alexnet_3x3_gap": "AlexNet3x3-GAP", "alexnet_2x2_fc": "AlexNet2x2-FC",
    "alexnet_small_kernel": "AlexNetSmallKernel", "alexnet_bottleneck": "AlexNetBottleneck",
    "alexnet_fire": "AlexNetFire", "mobilenetv2": "MobileNetV2", "resnet18_tv": "ResNet18",
    "vgg_style": "VGG-Style", "alexnet_final_fire_residual": "FinalFireResidual",
    "alexnet_final_bottleneck_fire": "FinalBottleneckFire",
    "alexnet_final_bottleneck_residual": "FinalBottleneckResidual",
    "alexnet_final_depthwise_fire": "FinalDepthwiseFire", "alexnet_fire_bypass": "FireBypass",
}

# Same 4-group color coding as Figure 1 (MODEL_GROUP/GROUP_COLORS, defined above) -- shape
# (circle/square) encodes precision, color encodes group, matching the dual-legend pattern
# already used in the compression-methods figure elsewhere in this repo.
point_colors = size_acc_df["model"].map(MODEL_GROUP).map(GROUP_COLORS)

fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")

for _, row in size_acc_df.iterrows():
    ax.plot([row["fp32_size_mb"], row["int8_size_mb"]], [row["fp32_top1"], row["int8_top1"]],
            color="0.85", linewidth=1, zorder=1)

ax.scatter(size_acc_df["fp32_size_mb"], size_acc_df["fp32_top1"], marker="o", c=point_colors,
           s=100, alpha=0.9, zorder=3, edgecolors="white", linewidths=0.7)
ax.scatter(size_acc_df["int8_size_mb"], size_acc_df["int8_top1"], marker="s", c=point_colors,
           s=100, alpha=0.9, zorder=3, edgecolors="white", linewidths=0.7)


def _pareto_frontier(xs, ys):
    """Skyline: sorted by x ascending, keep points whose y beats every prior kept point."""
    pts = sorted(zip(xs, ys), key=lambda p: p[0])
    frontier, best_y = [], -1.0
    for x, y in pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


fp32_frontier = _pareto_frontier(size_acc_df["fp32_size_mb"], size_acc_df["fp32_top1"])
int8_frontier = _pareto_frontier(size_acc_df["int8_size_mb"], size_acc_df["int8_top1"])
ax.plot([p[0] for p in fp32_frontier], [p[1] for p in fp32_frontier], linestyle="--",
        color="0.3", linewidth=1.8, zorder=2)
ax.plot([p[0] for p in int8_frontier], [p[1] for p in int8_frontier], linestyle=":",
        color="0.3", linewidth=1.8, zorder=2)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

all_xs = pd.concat([size_acc_df["fp32_size_mb"], size_acc_df["int8_size_mb"]]).to_numpy()
all_ys = pd.concat([size_acc_df["fp32_top1"], size_acc_df["int8_top1"]]).to_numpy()
# fp32/int8 blocks (matches the pd.concat order above), not interleaved per model.
labels = [DISPLAY_NAME[m] for m in size_acc_df["model"]] * 2
_label_points(ax, all_xs, all_ys, labels, fontsize=7)

ax.set_xlabel("Model Size (MB, log scale)", fontsize=11, color=TEXT_PRIMARY)
ax.set_ylabel("Top-1 accuracy (%)", fontsize=11, color=TEXT_PRIMARY)

shape_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markersize=9, label="FP32"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="0.4", markersize=9, label="INT8"),
    Line2D([0], [0], color="0.3", linestyle="--", linewidth=1.8, label="Fronteira Pareto (FP32)"),
    Line2D([0], [0], color="0.3", linestyle=":", linewidth=1.8, label="Fronteira Pareto (INT8)"),
]
leg1 = ax.legend(handles=shape_legend, loc="upper left", fontsize=8.5, frameon=False)
group_legend = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
ax.legend(handles=group_legend, loc="center right", bbox_to_anchor=(1.0, 0.62),
          fontsize=8.5, frameon=False)
ax.add_artist(leg1)

ax.grid(alpha=0.3)

fig.suptitle("Acurácia vs. tamanho: todos os modelos discutidos neste relatório (FP32 e INT8)",
             fontsize=12, color=TEXT_PRIMARY)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_vs_size_all_models.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ accuracy_vs_size_all_models.png")
plt.close()
