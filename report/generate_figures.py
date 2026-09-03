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

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

OUTPUT_DIR = Path("report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette shared with generate_architecture_figures.py -- see report/palette.py.
from palette import BLUE, RED, GREEN, PURPLE, AMBER, TEXT_PRIMARY, TEXT_SECONDARY, GRID

# One legend fontsize for every figure in this script -- previously each legend picked its own
# size ad hoc (8 to 12), so the same "elegível a Winograd" / group-color legend looked a different
# size depending on which figure it appeared in.
LEGEND_FONTSIZE = 11

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
    "MobileNetV2 (irrestrita)", "VGG-Style (irrestrita)", "ResNet18 (irrestrita)",
    "Apenas 3×3 (ingênua)", "Apenas 2×2 (ingênua)",
    "Mista (3×3+2×2)", "SmallKernel (exploratória)", "Residual (exploratória)",
    "Fire-Residual (híbrida final)",
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
    "Atenção local": AMBER,
}
colors_fig1 = [GROUP_COLORS[g] for g in group_fig1]


def _group_legend(groups):
    """Legend patches for only the groups actually plotted, in GROUP_COLORS order.

    Not every figure plots every group (Figure 1 has no attention models, for instance), and a
    legend built straight from GROUP_COLORS would show entries with no points under them.
    """
    present = set(groups)
    return [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items() if g in present]

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
    "alexnet_dilated_gap": "Tentativas de compensação", "alexnet_stacked": "Tentativas de compensação",
    "alexnet_groupconv": "Tentativas de compensação", "alexnet_factorized": "Tentativas de compensação",
    "alexnet_se": "Tentativas de compensação",
    # Legacy duplicate result files (same architecture/params/MACs as their non-suffixed twin from
    # a later superseded run, see WINOGRAD_ELIGIBLE's comment below) -- same group as their twin.
    "alexnet_2x2": "Restrição ingênua", "alexnet_2x2_gap": "Restrição ingênua",
    "alexnet_3x3": "Restrição ingênua", "alexnet_gap": "Restrição ingênua",
    "alexnet_final_fire_residual": "Arquitetura híbrida final",
    "alexnet_final_bottleneck_residual": "Arquitetura híbrida final",
    "alexnet_final_bottleneck_fire": "Arquitetura híbrida final",
    "alexnet_final_depthwise_fire": "Arquitetura híbrida final",
    "alexnet_fire_bypass": "Arquitetura híbrida final",
    "swin_pico_w2": "Atenção local", "swin_pico_w4": "Atenção local",
    "swin_pico_w8": "Atenção local", "swin_pico_poolmixer": "Atenção local",
    "hybrid_bottleneck_swin": "Atenção local", "vit_tiny": "Atenção local",
    "deit_tiny": "Atenção local",
    "vit_tiny_convstem": "Atenção local", "swin_pico_convstem": "Atenção local",
}

# Architectural Winograd eligibility (has >=1 dense, stride-1, 3x3 Conv2d -- the only structure
# that triggers Winograd F(2x2,3x3), ic_report.tex Eixo 3/7) for every model plotted in the two
# "all models" figures below. NOT the same as measured winograd_trace_detected (Eixo 3's cuDNN
# kernel-name trace, only run on ~10 of these models) -- e.g. alexnet_bottleneck is eligible here
# but measured to NOT trigger in practice. Computed once by instantiating every ml/registry.py
# model and inspecting its Conv2d layers (kernel_size==(3,3), stride==(1,1), groups==1); the three
# legacy duplicate result files (alexnet_2x2/alexnet_3x3/alexnet_gap, same params/MACs/size as
# alexnet_2x2_gap/alexnet_3x3_fc/alexnet_3x3_gap from an earlier superseded run) share their
# twin's architecture and therefore its eligibility.
WINOGRAD_ELIGIBLE = {
    "alexnet_3x3", "alexnet_3x3_fc", "alexnet_3x3_gap", "alexnet_gap",
    "alexnet_bottleneck", "alexnet_fire", "alexnet_fire_bypass", "alexnet_residual",
    "alexnet_mixed", "alexnet_stacked", "alexnet_small_kernel", "alexnet_smallkernel",
    "alexnet_tv", "resnet18_tv", "resnet18tv", "vgg_style",
    "alexnet_final_bottleneck_fire", "alexnet_final_bottleneck_residual",
    "alexnet_final_depthwise_fire", "alexnet_final_fire_residual",
    "vit_tiny_convstem", "swin_pico_convstem",
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

ax.set_xlabel("Acurácia top-1 FP32 (%)", fontsize=10, color=TEXT_PRIMARY)
ax.set_xlim(0, 78)
_style_axes(ax)

legend_handles = _group_legend(group_fig1)
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
          ncol=2, frameon=False, fontsize=LEGEND_FONTSIZE)

plt.title("Custo e recuperação da restrição de kernel", fontsize=12, fontweight="normal",
          color=TEXT_PRIMARY, pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kernel_restriction_cost.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ kernel_restriction_cost.png")
plt.close()


# ====== Figure: Quantization Stability by Architecture ======
models_fig3 = ["alexnet_small_kernel", "vgg_style", "alexnet_bottleneck", "alexnet_fire"]
labels_fig3 = ["SmallKernel\n(frágil)", "VGG-Style\n(referência)", "Bottleneck\n(robusta)", "Fire\n(robusta)"]

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

ax.set_ylabel("Queda de quantização (FP32 → INT8, p.p.)", fontsize=11, color=TEXT_PRIMARY)
ax.set_ylim(-2.5, 12)
ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=GRID, labelcolor=TEXT_SECONDARY)

plt.title("Estabilidade de quantização por arquitetura", fontsize=13, fontweight="normal", color=TEXT_PRIMARY, pad=16)
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

ax.set_xlabel("Acurácia top-1 média (%)", fontsize=13, color=TEXT_PRIMARY)
ax.tick_params(axis="x", labelsize=11)
ax.set_xlim(0, 82)
_style_axes(ax)

legend_handles = [Patch(facecolor=c, label=g) for g, c in COMP_GROUP_COLORS.items()]
# bbox_to_anchor is in axes-fraction coords; plt.tight_layout() (removed below) fights this by
# shrinking the axes to fit the legend inside the original figsize, which paradoxically pushes the
# per-bar annotation text further into the legend's column -- bbox_inches="tight" on savefig alone
# already crops the canvas to content, so tight_layout() isn't needed here.
ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
          ncol=1, frameon=False, fontsize=LEGEND_FONTSIZE)

fig.suptitle("Métodos de compressão além de INT8\n"
             "(taxa = tamanho teórico dos pesos em FP32 ÷ tamanho teórico após compressão)",
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
# Human-readable name for this CSV's "model" values -- point labels/annotations should read the
# architecture name, not the internal registry key.
P6_DISPLAY_NAME = {
    "alexnet_tv": "AlexNetTV", "alexnet_bottleneck": "AlexNetBottleneck",
    "alexnet_fire": "AlexNetFire", "alexnet_depthwisesep": "AlexNetDepthwiseSep",
    "alexnet_dilated_gap": "AlexNetDilatedGAP",
    "alexnet_final_fire_residual": "AlexNetFinalFireResidual",
    "alexnet_final_bottleneck_residual": "AlexNetFinalBottleneckResidual",
    "mobilenetv2": "MobileNetV2", "resnet18tv": "ResNet18", "vgg_style": "VGG-Style",
}


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
    elig_mask = subset["model"].isin(WINOGRAD_ELIGIBLE).to_numpy()
    # Two scatter calls, not one: PathCollection.set_hatch() applies to the whole collection, so
    # per-point hatching needs the eligible/ineligible points split apart (same approach as the
    # accuracy_vs_size/macs_vs_accuracy figures below).
    ax.scatter(subset["latency_ms"][~elig_mask], subset["accuracy"][~elig_mask],
               marker=PRECISION_MARKERS[prec], c=np.array(point_colors)[~elig_mask],
               s=90, alpha=0.9, zorder=3, edgecolors="white", linewidths=0.5)
    ax.scatter(subset["latency_ms"][elig_mask], subset["accuracy"][elig_mask],
               marker=PRECISION_MARKERS[prec], c=np.array(point_colors)[elig_mask],
               s=90, hatch="////", alpha=0.9, zorder=3, edgecolors="black", linewidths=0.7)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
acc_pad = (pareto_df["accuracy"].max() - pareto_df["accuracy"].min()) * 0.18
ax.set_ylim(pareto_df["accuracy"].min() - acc_pad, pareto_df["accuracy"].max() + acc_pad)

_label_points(ax, pareto_df["latency_ms"].to_numpy(), pareto_df["accuracy"].to_numpy(),
              pareto_df["model"].map(P6_DISPLAY_NAME).to_numpy())

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
            linewidth=2, label="Fronteira de Pareto", zorder=2)

ax.set_xlabel("Latência (ms, escala log)", fontsize=11, color=TEXT_PRIMARY)
ax.set_ylabel("Acurácia top-1 (%)", fontsize=11, color=TEXT_PRIMARY)
ax.set_title("RTX 4060 Laptop (local)", fontsize=11)

shape_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markersize=8, label="FP32"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="0.4", markersize=8, label="INT8"),
    Line2D([0], [0], color="0.35", linestyle="--", linewidth=2, label="Fronteira de Pareto"),
    Patch(facecolor="0.85", edgecolor="black", hatch="////", label="Elegível a Winograd"),
]
leg1 = ax.legend(handles=shape_legend, loc="upper left", fontsize=LEGEND_FONTSIZE, frameon=False)
group_legend = _group_legend(pareto_df["model"].map(MODEL_GROUP))
# Smaller than the other legends in this script -- at LEGEND_FONTSIZE it covered the
# alexnet_tv INT8 point and label sitting right behind it in the lower-right corner.
ax.legend(handles=group_legend, loc="lower right", fontsize=8, frameon=False)
ax.add_artist(leg1)
ax.grid(alpha=0.3)

fig.suptitle("Acurácia vs. latência: fronteira de Pareto e referência", fontsize=12)
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
    ("fp32", 1, "FP32 GPU, lote=1"),
    ("fp32", 64, "FP32 GPU, lote=64"),
    ("int8", 1, "INT8 CPU, lote=1"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")

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
                label=f"canais de entrada = {in_ch}", markersize=7, linewidth=2)
    ax.set_yscale("log")
    ax.set_xticks(KSIZES)
    ax.axvline(3, color="0.6", linestyle=":", linewidth=1)
    ax.set_xlabel("tamanho de kernel ($k$)", color=TEXT_PRIMARY)
    ax.set_title(subtitle, fontsize=10.5, color=TEXT_PRIMARY)
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(colors=GRID, labelcolor=TEXT_SECONDARY)
axes[0].set_ylabel("latência / $k^2$ (ms por unidade de FLOP)", color=TEXT_PRIMARY)

# One shared legend below all 3 panels instead of one per panel -- a per-panel legend sat right
# on top of the "canais de entrada = 128" curve, which stays near the top of every panel.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4,
           frameon=False, fontsize=LEGEND_FONTSIZE)

fig.suptitle("RTX 4060 Laptop (local): latência de camada por FLOP vs. tamanho de kernel "
             "(densa/groups=1, resolução 64)",
             fontsize=12, color=TEXT_PRIMARY)
fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(OUTPUT_DIR / "phase6_latency_vs_kernel_size.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ phase6_latency_vs_kernel_size.png")
plt.close()


# ====== Figure: Accuracy vs. Size — 15-model subset, data source for the MACs companion below ======
# NOT the report's "accuracy_vs_size_all_models.png" (that's the real 33-model/4-panel figure from
# notebooks/phase_10_final_summary/final_summary.ipynb, cell 12 -- copied in as a file below; this
# script previously duplicated the filename with a simplified single-panel/15-model reproduction,
# silently overwriting the real one, while the report's caption/prose kept describing the 33-model
# version. This block now only exists to build size_acc_df/DISPLAY_NAME for the MACs figure).
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

# The aggregate CSV this used to read (outputs/pcad/results_aggregate/results_phase_9_fire_bypass_
# large_scale.csv) was a git-ignored artifact removed by a later repo cleanup; the run's own
# per-model summary JSON has the same numbers and is still tracked.
_p9 = json.load(open(
    "outputs/pcad/phase_9_bypass_ablation/fire_bypass_large_scale/alexnet_fire_bypass/"
    "results/alexnet_fire_bypass_summary.json"))
_p4_pairs.append({"model": "alexnet_fire_bypass", "fp32_top1": _p9["fp32_top1"],
                   "fp32_size_mb": _p9["fp32_size_mb"], "int8_top1": _p9["int8_top1"],
                   "int8_size_mb": _p9["int8_size_mb"]})

# The seven Phase 8 attention models live in yet another CSV schema (fp32_top1/int8_top1 instead of
# top1_%_FP32/top1_%_INT8), which is why they were absent from these two figures originally. Their
# accuracy/size columns are complete there; only vit_tiny/deit_tiny lack MACs, filled in below from
# the notebook-driven run's own CSV -- same main-source-plus-fill-in pattern as the Phase 4/9 rows.
_p8 = pd.read_csv(
    "results/phase_8_efficient_vit_hybrid_attention_analysis/phase8_comparison.csv"
).rename(columns={"model_name": "model"})
_p8_rows = _p8[["model", "fp32_top1", "fp32_size_mb", "int8_top1", "int8_size_mb", "macs"]].copy()

# H5 follow-up (Winograd-eligible conv-stem counterfactual, ic_report.tex Eixo 7): two more
# attention models, trained separately (configs/experiments/phase8_convstem.yaml) and never
# folded into phase8_comparison.csv -- read straight from their own summary JSONs, same
# own-summary-JSON splice pattern as the alexnet_fire_bypass row above.
_p8_convstem = pd.DataFrame([
    json.load(open(f"outputs/local/phase8_convstem/{name}/results/{name}_summary.json"))
    for name in ["vit_tiny_convstem", "swin_pico_convstem"]
]).rename(columns={"model_name": "model"})[
    ["model", "fp32_top1", "fp32_size_mb", "int8_top1", "int8_size_mb", "macs"]
]

size_acc_df = pd.concat(
    [size_acc_rows, pd.DataFrame(_p4_pairs), _p8_rows.drop(columns="macs"),
     _p8_convstem.drop(columns="macs")],
    ignore_index=True,
)

# MACs for the same models, reused below for the MACs-vs-accuracy companion figure. The 10
# base models are in model_details_cross_phase.csv; the Phase 4/9 hybrids aren't (same schema gap
# as the accuracy/size data above), so their MACs come from their own per-model summary JSONs.
_macs_by_model = pd.read_csv("results/results_aggregate/model_details_cross_phase.csv") \
    .set_index("model_name")["macs"].to_dict()
for _base in ["alexnet_final_fire_residual", "alexnet_final_bottleneck_fire",
              "alexnet_final_bottleneck_residual", "alexnet_final_depthwise_fire"]:
    _macs_by_model[_base] = json.load(open(
        f"results/phase_4_compression_and_final_architecture_training/{_base}_summary.json"))["macs"]
_macs_by_model["alexnet_fire_bypass"] = _p9["macs"]
_macs_by_model.update(_p8_rows.dropna(subset=["macs"]).set_index("model")["macs"].to_dict())
_macs_by_model.update(_p8_convstem.set_index("model")["macs"].to_dict())
_macs_by_model.update(
    pd.read_csv("results/phase_8_efficient_vit_hybrid_attention_training/"
                "phase8_vit_deit_comparison.csv")
    .query("precision == 'FP32'").set_index("model")["macs"].to_dict()
)
size_acc_df["macs_m"] = size_acc_df["model"].map(_macs_by_model) / 1e6
assert size_acc_df["macs_m"].notna().all(), \
    f"MACs ausentes: {size_acc_df.loc[size_acc_df['macs_m'].isna(), 'model'].tolist()}"

DISPLAY_NAME = {
    "alexnet_tv": "AlexNet (irrestrita)", "alexnet_3x3_fc": "AlexNet3x3-FC",
    "alexnet_3x3_gap": "AlexNet3x3-GAP", "alexnet_2x2_fc": "AlexNet2x2-FC",
    "alexnet_small_kernel": "AlexNetSmallKernel", "alexnet_bottleneck": "AlexNetBottleneck",
    "alexnet_fire": "AlexNetFire", "mobilenetv2": "MobileNetV2", "resnet18_tv": "ResNet18",
    "vgg_style": "VGG-Style", "alexnet_final_fire_residual": "FinalFireResidual",
    "alexnet_final_bottleneck_fire": "FinalBottleneckFire",
    "alexnet_final_bottleneck_residual": "FinalBottleneckResidual",
    "alexnet_final_depthwise_fire": "FinalDepthwiseFire", "alexnet_fire_bypass": "FireBypass",
    "swin_pico_w2": "swin-w2", "swin_pico_w4": "swin-w4", "swin_pico_w8": "swin-w8",
    "swin_pico_poolmixer": "swin-poolmixer", "hybrid_bottleneck_swin": "hybrid-bottleneck-swin",
    "vit_tiny": "ViT-Tiny", "deit_tiny": "DeiT-Tiny",
    "vit_tiny_convstem": "ViT-Tiny-ConvStem", "swin_pico_convstem": "swin-convstem",
    "alexnet_depthwisesep": "AlexNetDepthwiseSep",
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

ax.set_xlabel("Tamanho do modelo (MB, escala log)", fontsize=11, color=TEXT_PRIMARY)
ax.set_ylabel("Acurácia top-1 (%)", fontsize=11, color=TEXT_PRIMARY)

shape_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markersize=9, label="FP32"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="0.4", markersize=9, label="INT8"),
    Line2D([0], [0], color="0.3", linestyle="--", linewidth=1.8, label="Fronteira Pareto (FP32)"),
    Line2D([0], [0], color="0.3", linestyle=":", linewidth=1.8, label="Fronteira Pareto (INT8)"),
]
leg1 = ax.legend(handles=shape_legend, loc="upper left", fontsize=LEGEND_FONTSIZE, frameon=False)
group_legend = _group_legend(size_acc_df["model"].map(MODEL_GROUP))
ax.legend(handles=group_legend, loc="center right", bbox_to_anchor=(1.0, 0.62),
          fontsize=LEGEND_FONTSIZE, frameon=False)
ax.add_artist(leg1)

ax.grid(alpha=0.3)

fig.suptitle("Acurácia vs. tamanho: subconjunto de 15 modelos, fonte da figura de MACs",
             fontsize=12, color=TEXT_PRIMARY)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_vs_size_15model_subset.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ accuracy_vs_size_15model_subset.png (not embedded in the report -- data source only)")
plt.close()

# ====== Figure: Accuracy vs. Size — real 33-model, 4-panel figure (the report's actual
# "accuracy_vs_size_all_models.png") ======
# Reproduced here from notebooks/phase_10_final_summary/final_summary.ipynb, cell 12, instead of
# copying that cell's PNG output in as a file: the notebook labels each of the 6 groups "Phase N —
# ..." (English, and "Phase" doesn't appear anywhere else in this Portuguese-language report) and
# colors them with its own tab10 colormap, which disagreed with the MODEL_GROUP/GROUP_COLORS
# scheme every other figure in this script uses for the same phases (e.g. baselines were blue in
# Figure 1 but red here). Both are fixed below: GROUP_DIRS' labels are Portuguese purely for
# readability (they no longer drive color -- point color comes from MODEL_GROUP/GROUP_COLORS via
# model_name, same as every other figure), and the legend reuses _group_legend so the two figures
# are colored identically. The underlying data assembly (GROUP_DIRS, SHARED_COLS, cross-frame
# dedup, pareto_front_mask) is otherwise unchanged from the notebook, so the two 8/33 and 5/30
# Pareto-front counts in the report's caption still hold.
def _load_summary_jsons(root: Path, group_label: str) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()
    rows = []
    for path in sorted(root.rglob("*_summary.json")):
        if path.name == "experiment_summary.json" or path.name.endswith("_compression_summary.json"):
            continue
        rows.append(json.loads(path.read_text()))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["phase"] = group_label
    if "model_name" in df.columns and "epochs" in df.columns:
        df = df.sort_values("epochs").drop_duplicates(subset="model_name", keep="last").reset_index(drop=True)
    return df


GROUP_DIRS = {
    "Baselines externos": Path("results/phase_1_baseline_training"),
    "Variantes AlexNet (kernel restrito)": Path("results/phase_2_kernel_restriction_training"),
    "Compensação estrutural": Path("results/phase_3_compensation_and_hybrids_training"),
    "Arquitetura final": Path("results/phase_4_compression_and_final_architecture_training"),
    "Ablação do atalho (bypass)": Path("outputs/pcad/phase_9_bypass_ablation"),
    "Atenção local": Path("outputs/local/phase8_convstem"),
}
_group8_csv = Path("results/phase_8_efficient_vit_hybrid_attention_analysis/phase8_comparison.csv")

_frames = [_load_summary_jsons(d, label) for label, d in GROUP_DIRS.items()]
if _group8_csv.exists():
    _df8 = pd.read_csv(_group8_csv)
    _df8["phase"] = "Atenção local"
    _frames.append(_df8)

_shared_cols = [
    "phase", "model_name", "fp32_top1", "fp32_top5", "int8_top1", "int8_top5",
    "fp32_size_mb", "int8_size_mb",
]
_frames = [f for f in _frames if not f.empty]
df_all_classification = pd.concat(
    [f[[c for c in _shared_cols if c in f.columns]] for f in _frames], ignore_index=True,
) if _frames else pd.DataFrame(columns=_shared_cols)
df_all_classification = df_all_classification.drop_duplicates(subset="model_name", keep="last").reset_index(drop=True)
# alexnet_se's training collapsed (~0.5% top-1, ic_report.tex Eixo 4) -- not a real
# accuracy/size trade-off point, just a training failure, so it's dropped from this figure
# entirely rather than plotted as a bottom-corner outlier.
df_all_classification = df_all_classification[df_all_classification["model_name"] != "alexnet_se"]


def _pareto_front_mask(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    dominated = np.zeros(len(xs), dtype=bool)
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i != j and xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                dominated[i] = True
                break
    return ~dominated


# ponytail: top-5 panels dropped -- the report's prose only ever discusses the top-1 frontiers,
# so a 2x2 (top-1+top-5 x FP32+INT8) grid was paying clutter for a metric nobody reads. Re-add a
# top-5 row here if the discussion in ic_report.tex grows to actually use it.
_panels = [
    ("fp32_size_mb", "fp32_top1", "FP32", "Top-1"),
    ("int8_size_mb", "int8_top1", "INT8", "Top-1"),
]
with plt.style.context("seaborn-v0_8-whitegrid"):
    # Already full double-column width (figure* in ic_report.tex) -- printed width is capped at
    # \linewidth regardless of figsize, so making this "bigger" means taller, not wider: more
    # absolute vertical room for the crowded 33-point scatter and its labels, at the same
    # print-scale factor (unaffected by height, only by width) as before.
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), facecolor="white")
    for ax, (size_col, acc_col, precision, metric) in zip(axes.flat, _panels):
        df_size = df_all_classification[
            df_all_classification[acc_col].notna() & df_all_classification[size_col].notna()
        ].copy()
        mask = _pareto_front_mask(df_size[size_col].values, df_size[acc_col].values)
        pf = df_size[mask].sort_values(size_col)
        point_colors = df_size["model_name"].map(MODEL_GROUP).map(GROUP_COLORS)
        elig_mask = df_size["model_name"].isin(WINOGRAD_ELIGIBLE).to_numpy()
        # Two scatter calls, not one: PathCollection.set_hatch() applies to the whole
        # collection, so per-point hatching needs the eligible/ineligible points split apart.
        ax.scatter(df_size[size_col][~elig_mask], df_size[acc_col][~elig_mask],
                   c=np.array(point_colors)[~elig_mask], s=110,
                   edgecolors="white", lw=0.5, alpha=0.9, zorder=3)
        ax.scatter(df_size[size_col][elig_mask], df_size[acc_col][elig_mask],
                   c=np.array(point_colors)[elig_mask], s=110, hatch="////",
                   edgecolors="black", lw=0.7, alpha=0.9, zorder=3)
        ax.step(pf[size_col], pf[acc_col], where="post", color="black", lw=1.2, ls="--", alpha=0.6, zorder=2)
        # ponytail: labels alternate above/below the point instead of a fixed offset, since the
        # Pareto front is a monotonic up-right staircase and same-direction offsets stack labels
        # from adjacent points on top of each other. Swap for adjustText if this stops being enough.
        for i, (_, row) in enumerate(pf.iterrows()):
            dy = 6 if i % 2 == 0 else -14
            ax.annotate(DISPLAY_NAME[row["model_name"]], (row[size_col], row[acc_col]), xytext=(6, dy),
                       textcoords="offset points", fontsize=9, fontweight="bold",
                       va="bottom" if dy > 0 else "top")
        ax.set_xscale("log")
        ax.set_xlabel(f"Tamanho {precision} do modelo (MB, escala log)")
        ax.set_ylabel(f"Acurácia {precision} {metric} (%)")
        ax.set_title(f"{precision} {metric} (fronteira de Pareto: {len(pf)}/{len(df_size)})")
        # Headroom above the topmost point -- its offset label (dy=6 points, above the point)
        # otherwise landed right on the axes' top border for whichever point happens to be highest.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.08)

    handles = _group_legend(df_all_classification["model_name"].map(MODEL_GROUP))
    labels = [h.get_label() for h in handles]
    handles.append(Patch(facecolor="0.85", edgecolor="black", hatch="////"))
    labels.append("Elegível a Winograd")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.11), ncol=3,
              fontsize=LEGEND_FONTSIZE + 2, title="Grupo de modelos", title_fontsize=LEGEND_FONTSIZE + 4,
              handlelength=1.4, handleheight=1.4, markerscale=1.1)
    fig.suptitle("Acurácia vs. Tamanho — Fronteiras de Pareto (todos os modelos de classificação)",
                fontsize=13)
    # rect[1] (the axes' bottom edge) and the legend's bbox_to_anchor y are independent knobs --
    # the legend sits entirely below y=0 (outside the rect-reserved margin), so rect[1] only needs
    # to clear the x-axis tick labels below the panels, not the legend itself. Previous values
    # (rect[1] up to 0.19 against anchor -0.14) reserved a margin far bigger than the x-labels
    # need, which is exactly the dead blank band between the panels and the legend.
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "accuracy_vs_size_all_models.png", bbox_inches="tight", facecolor="white")
print("✓ accuracy_vs_size_all_models.png")
plt.close()


# ====== Figure: MACs vs. Accuracy — same 15 models, same visual language ======
# MACs don't change between FP32 and INT8 (same architecture, same multiply-accumulate count), so
# each model is a single x position with a vertical FP32->INT8 connector instead of the diagonal
# one above.
point_colors = size_acc_df["model"].map(MODEL_GROUP).map(GROUP_COLORS)

fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")

for _, row in size_acc_df.iterrows():
    ax.plot([row["macs_m"], row["macs_m"]], [row["fp32_top1"], row["int8_top1"]],
            color="0.85", linewidth=1, zorder=1)

_elig = size_acc_df["model"].isin(WINOGRAD_ELIGIBLE).to_numpy()
# Two calls per marker shape, not one: PathCollection.set_hatch() applies to the whole
# collection, so per-point hatching needs the eligible/ineligible points split apart.
ax.scatter(size_acc_df["macs_m"][~_elig], size_acc_df["fp32_top1"][~_elig], marker="o",
           c=np.array(point_colors)[~_elig], s=100, alpha=0.9, zorder=3,
           edgecolors="white", linewidths=0.7)
ax.scatter(size_acc_df["macs_m"][~_elig], size_acc_df["int8_top1"][~_elig], marker="s",
           c=np.array(point_colors)[~_elig], s=100, alpha=0.9, zorder=3,
           edgecolors="white", linewidths=0.7)
ax.scatter(size_acc_df["macs_m"][_elig], size_acc_df["fp32_top1"][_elig], marker="o",
           c=np.array(point_colors)[_elig], s=100, hatch="////", alpha=0.9, zorder=3,
           edgecolors="black", linewidths=0.7)
ax.scatter(size_acc_df["macs_m"][_elig], size_acc_df["int8_top1"][_elig], marker="s",
           c=np.array(point_colors)[_elig], s=100, hatch="////", alpha=0.9, zorder=3,
           edgecolors="black", linewidths=0.7)

fp32_frontier = _pareto_frontier(size_acc_df["macs_m"], size_acc_df["fp32_top1"])
int8_frontier = _pareto_frontier(size_acc_df["macs_m"], size_acc_df["int8_top1"])
ax.plot([p[0] for p in fp32_frontier], [p[1] for p in fp32_frontier], linestyle="--",
        color="0.3", linewidth=1.8, zorder=2)
ax.plot([p[0] for p in int8_frontier], [p[1] for p in int8_frontier], linestyle=":",
        color="0.3", linewidth=1.8, zorder=2)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

# One label per model, not one per point: FP32/INT8 share an x position here, so labelling both
# just prints every name twice on top of itself. Anchor on whichever of the pair is higher so the
# label sits above the vertical connector rather than between the two markers.
_label_points(ax, size_acc_df["macs_m"].to_numpy(),
              size_acc_df[["fp32_top1", "int8_top1"]].max(axis=1).to_numpy(),
              [DISPLAY_NAME[m] for m in size_acc_df["model"]], fontsize=9)

ax.set_xlabel("MACs (M, escala log)", fontsize=11, color=TEXT_PRIMARY)
ax.set_ylabel("Acurácia top-1 (%)", fontsize=11, color=TEXT_PRIMARY)

# Both legends sit fully outside the axes (bbox_to_anchor x > 1) -- with 22 models now spread
# across the whole plot area, there's no corner left that a legend can sit in without covering a
# point or its label.
group_legend = _group_legend(size_acc_df["model"].map(MODEL_GROUP))
leg1 = ax.legend(handles=group_legend, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=LEGEND_FONTSIZE, frameon=False)
_shape_legend_hatched = shape_legend + [
    Patch(facecolor="0.85", edgecolor="black", hatch="////",
          label="Elegível a Winograd")
]
ax.legend(handles=_shape_legend_hatched, loc="lower left", bbox_to_anchor=(1.01, 0.0),
          fontsize=LEGEND_FONTSIZE, frameon=False)
ax.add_artist(leg1)

ax.grid(alpha=0.3)

fig.suptitle("Acurácia vs. MACs: todos os modelos discutidos neste relatório (FP32 e INT8)",
             fontsize=12, color=TEXT_PRIMARY)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "macs_vs_accuracy_all_models.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ macs_vs_accuracy_all_models.png")
plt.close()


# ====== Figure: Eixo 6 detection accuracy vs. true model size (bigger legend) ======
# Reproduced from notebooks/phase_7_detection_segmentation_analysis/phase7_results_analysis.ipynb,
# cell 19 -- same 9 post-anchor-fix, non-pretrained SSD detection runs (3 backbones x FP32/QAT/INT8),
# same true_size_mb-with-model_size_mb-fallback logic, same MODEL_COLORS (PALETTE[2]/[1]/[0] for
# bottleneck/fire/tv) and FP32/QAT/INT8 marker shapes.
_p7_dir = Path("outputs/detection_segmentation/phase7")
_p7_models = ["alexnet_bottleneck", "alexnet_fire", "alexnet_tv"]
# ml/plotting.py's PALETTE[2]/[1]/[0] -- same colors the source notebook used.
_p7_colors = {"alexnet_bottleneck": "#1baf7a", "alexnet_fire": "#eb6834", "alexnet_tv": "#2a78d6"}
_p7_names = {"alexnet_bottleneck": "AlexNetBottleneck", "alexnet_fire": "AlexNetFire",
             "alexnet_tv": "AlexNetTV"}
_p7_markers = {"fp32": "o", "qat": "s", "int8": "^"}

_p7_rows = []
for _model in _p7_models:
    for _stage in ("fp32", "qat", "int8"):
        _run = _p7_dir / f"ssd_{_model}_{_stage}_phase7_detection" / "metrics.json"
        _d = json.loads(_run.read_text())
        _mAP = _d["best_val_mAP"] if "best_val_mAP" in _d else _d["val_mAP"][0]
        _summary = _d["summary"]
        _true = _summary.get("true_size_mb")
        _size_mb = _true if _true is not None else _summary.get("model_size_mb")
        _p7_rows.append({"model": _model, "stage": _stage, "mAP": _mAP, "size_mb": _size_mb,
                         "is_true": _true is not None})
p7_df = pd.DataFrame(_p7_rows)

fig, ax = plt.subplots(figsize=(7.5, 5), facecolor="white")
for _model in _p7_models:
    sub = p7_df[p7_df.model == _model].set_index("stage")
    pts = [(sub.loc[s, "size_mb"], sub.loc[s, "mAP"]) for s in ("fp32", "qat", "int8") if s in sub.index]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="0.8", linewidth=1, zorder=1)

any_estimated = False
for _, row in p7_df.iterrows():
    any_estimated = any_estimated or not row["is_true"]
    ax.scatter(row["size_mb"], row["mAP"], s=100, marker=_p7_markers[row["stage"]],
              facecolor=_p7_colors[row["model"]] if row["is_true"] else "none",
              edgecolor=_p7_colors[row["model"]], linewidth=0.7 if row["is_true"] else 1.6, zorder=3)

ax.set_xscale("log")
ax.set_xlabel("Tamanho do modelo (MB, escala log)")
ax.set_ylabel("mAP de detecção")
_title = "Acurácia vs. tamanho real do modelo (FP32/QAT/INT8"
_title += "; vazado = tamanho não corrigido/estimado)" if any_estimated else ")"
ax.set_title(_title)

stage_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markeredgecolor="black", markersize=9, label="FP32"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="0.4", markeredgecolor="black", markersize=9, label="QAT"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="0.4", markeredgecolor="black", markersize=9, label="INT8"),
]
if any_estimated:
    stage_legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                               markeredgecolor="0.4", markersize=9, label="tamanho não corrigido/estimado"))
model_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=_p7_names[m])
               for m, c in _p7_colors.items()]
# This figure prints in a single IEEEtran column (~3.5in) like Figures 1 and 8, but at 7.5in wide
# it's shrunk to fit that column more than either of those (~0.65x for Fig.1's 5.4in, ~0.58x for
# Fig.8's 6in, ~0.47x here) -- so LEGEND_FONTSIZE alone still prints visibly smaller. Scaled up by
# roughly that same ratio (7.5/5.4) to come out the same size on the page as Figure 1's legend.
_FIG7_LEGEND_FONTSIZE = round(LEGEND_FONTSIZE * 7.5 / 5.4)
leg1 = ax.legend(handles=stage_legend, loc="upper left", fontsize=_FIG7_LEGEND_FONTSIZE, frameon=False)
# Below the axes, not "lower right" -- at this legend size that corner sat on top of the
# alexnet_fire QAT/INT8 points (the two lowest-mAP, ~0.06-0.065, near x=3.5-6.6 MB).
ax.legend(handles=model_legend, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
          fontsize=_FIG7_LEGEND_FONTSIZE, frameon=False)
ax.add_artist(leg1)
ax.grid(alpha=0.3)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "phase7_accuracy_vs_size.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ phase7_accuracy_vs_size.png")
plt.close()


# ====== Figure: Eixo 7 window-size sweep (H1) ======
# Reproduced from notebooks/phase_8_efficient_vit/phase8_results_analysis.ipynb, cell 10, from the
# same phase8_comparison.csv (_p8, loaded above) -- same reason as the Phase 6 figures above: the
# notebook's version carries English labels and an "H1 —" hypothesis-numbering title, and rerunning
# the whole notebook (which live-profiles the GPU for H5) to retitle one chart isn't worth it.
WINDOW_SIZES = {"swin_pico_w2": 2, "swin_pico_w4": 4, "swin_pico_w8": 8}
h1_df = _p8.set_index("model").loc[list(WINDOW_SIZES), ["fp32_top1", "int8_top1"]]
h1_df["window_size"] = [WINDOW_SIZES[m] for m in h1_df.index]

fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="white")
ax.set_facecolor("white")
ax.plot(h1_df["window_size"], h1_df["fp32_top1"], "o-", label="FP32", color=BLUE, linewidth=2)
ax.plot(h1_df["window_size"], h1_df["int8_top1"], "s--", label="INT8", color=RED, linewidth=2)
ax.set_xticks(h1_df["window_size"])
ax.set_xlabel("Tamanho da janela de atenção", fontsize=11, color=TEXT_PRIMARY)
ax.set_ylabel("Acurácia top-1 (%)", fontsize=11, color=TEXT_PRIMARY)
ax.set_title("Acurácia vs. tamanho da janela de atenção local", fontsize=12, color=TEXT_PRIMARY)
ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
ax.grid(alpha=0.3)
ax.tick_params(colors=GRID, labelcolor=TEXT_SECONDARY)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "phase8_h1_window_size_sweep.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ phase8_h1_window_size_sweep.png")
plt.close()


# ====== Figure: conv-stem before/after ======
# Paired before/after: vit_tiny/swin_pico_w4 vs. their _convstem counterparts (_p8_convstem,
# loaded above), which add a Winograd-eligible 3x3 stride-1 stem. Two panels (FP32 | INT8), each
# with grouped "sem stem" / "com stem" bars showing the actual accuracy, plus the delta annotated
# between them -- a bare delta-only bar (this figure's first version) hid what accuracy range the
# jump happens in; showing both ends makes the comparison self-contained without the table.
H5_PAIRS = {"ViT-Tiny": ("vit_tiny", "vit_tiny_convstem"), "Swin-pico (janela 4)": ("swin_pico_w4", "swin_pico_convstem")}
without_idx = _p8.set_index("model")
with_idx = _p8_convstem.set_index("model")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")
x = np.arange(len(H5_PAIRS))
width = 0.32

for ax, col, precision, color in zip(axes, ["fp32_top1", "int8_top1"], ["FP32", "INT8"], [BLUE, RED]):
    ax.set_facecolor("white")
    before = [without_idx.loc[without_m, col] for without_m, _ in H5_PAIRS.values()]
    after = [with_idx.loc[with_m, col] for _, with_m in H5_PAIRS.values()]

    bars_before = ax.bar(x - width / 2, before, width, label="Sem stem 3×3", color=color, alpha=0.35, edgecolor="none")
    bars_after = ax.bar(x + width / 2, after, width, label="Com stem 3×3", color=color, edgecolor="none")
    for bars in (bars_before, bars_after):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color=TEXT_PRIMARY)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.annotate(f"+{a - b:.2f}pp", xy=(i, max(b, a) + 3.4), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=TEXT_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(H5_PAIRS.keys(), fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylabel("Acurácia top-1 (%)", fontsize=10.5, color=TEXT_PRIMARY)
    # Headroom above 60 (the tallest bar+annotation tops out around 53) so the "+X pp" delta
    # annotations, which sit above the bars, clear the upper-left legend instead of overlapping it.
    ax.set_ylim(0, 75)
    ax.set_title(precision, fontsize=11.5, color=TEXT_PRIMARY, pad=8)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=GRID, labelcolor=TEXT_SECONDARY)

fig.suptitle("Acurácia antes/depois de acrescentar um stem 3×3 Winograd-elegível",
             fontsize=12.5, fontweight="normal", color=TEXT_PRIMARY, y=1.0)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "phase8_h5_convstem_gain.png", dpi=150, bbox_inches="tight", facecolor="white")
print("✓ phase8_h5_convstem_gain.png")
plt.close()
