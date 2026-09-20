"""Phase 11 figure — mixed vs uniform vs original kernels, AlexNet (GAP/FC) and VGG16.

Pulls fp32/int8 top-1 + size directly from the three curated Phase 11 result trees
(phase_11_kernel_size_comparison, phase_11_mixed_kernel_comparison,
phase_11_head_bn_ablation) rather than a models: list, since the mixed-kernel models
split across two head styles (GAP vs the AlexNetTV FC classifier) that differ in size
by ~30x -- plotted as two panels sharing a y-axis instead of one crowded log-log plot.

    python -m scripts.phase11.plot_kernel_comparison
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from ml.plotting import BLUE, RED, AMBER, apply_report_style

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "results/figures_generated/phase_11_kernel_size_comparison"

# kernel pattern -> color, shared across both panels
PATTERN_COLOR = {"uniforme": BLUE, "original": AMBER, "misto": RED}

# model_name -> (label, pattern); left = GAP head (compact), right = FC head (AlexNetTV/VGG16)
GAP_MODELS = {
    "alexnet_2x2_gap": ("2-2-2-2-2", "uniforme"),
    "alexnet_3x3_gap": ("3-3-3-3-3", "uniforme"),
    "alexnet_mixed_preheinit": ("alternado\n(Phase 2)", "misto"),
    "alexnet_tv_mixed_alt_gap": ("2-3-2-3-2", "misto"),
    "alexnet_tv_mixed_early3_gap": ("3-3-3-2-2", "misto"),
    "alexnet_tv_mixed_early2_gap": ("2-2-2-3-3", "misto"),
}
FC_MODELS = {
    "alexnet_tv_scratch": ("AlexNet 11-5-3-3-3\n(original)", "original"),
    "alexnet_tv_3x3": ("AlexNet 3-3-3-3-3", "uniforme"),
    "alexnet_tv_2x2": ("AlexNet 2-2-2-2-2", "uniforme"),
    "alexnet_tv_mixed_early3": ("AlexNet 3-3-3-2-2", "misto"),
    "alexnet_tv_mixed_alt": ("AlexNet 2-3-2-3-2", "misto"),
    "alexnet_tv_mixed_early2": ("AlexNet 2-2-2-3-3", "misto"),
    "vgg16": ("VGG16 3x3\n(nativo)", "original"),
    "vgg16_2x2": ("VGG16 2x2", "uniforme"),
}


def load_rows() -> dict:
    # Per-model summary.json (always current) takes priority; the curated final_comparison.csv
    # is a fallback for models whose json isn't synced from PCAD yet.
    rows = {}
    for csv_path in [
        ROOT / "results/phase_11_kernel_size_comparison/phase_11_kernel_size_comparison_final_comparison.csv",
        ROOT / "results/phase_11_mixed_kernel_comparison/phase_11_mixed_kernel_comparison_final_comparison.csv",
    ]:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.setdefault(r["model_name"], r)
    for experiment in ["phase_11_kernel_size_comparison", "phase_11_mixed_kernel_comparison", "phase_11_head_bn_ablation"]:
        for summary_path in (ROOT / "outputs/pcad" / experiment).glob("*/results/*_summary.json"):
            model_dir = summary_path.parents[1].name  # unique per run, unlike json's internal model_name
            rows[model_dir] = json.loads(summary_path.read_text())
    return rows


def plot_scatter_panel(ax, rows: dict, models: dict, title: str):
    """Size varies meaningfully within this group -- accuracy-vs-size scatter."""
    for model, (label, pattern) in models.items():
        r = rows[model]
        color = PATTERN_COLOR[pattern]
        fp32_size, fp32_top1 = float(r["fp32_size_mb"]), float(r["fp32_top1"])
        int8_size, int8_top1 = float(r["int8_size_mb"]), float(r["int8_top1"])
        ax.plot([fp32_size, int8_size], [fp32_top1, int8_top1], color="gray", lw=1, alpha=0.4, zorder=1)
        ax.scatter(fp32_size, fp32_top1, color=color, marker="o", s=130, edgecolors="white", lw=0.6, zorder=3)
        ax.scatter(int8_size, int8_top1, color=color, marker="s", s=110, edgecolors="white", lw=0.6, zorder=3)
        ax.annotate(label, (fp32_size, fp32_top1), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=color)
    ax.set_xlabel("Tamanho do modelo (MB)")
    ax.set_title(title)


def plot_bar_panel(ax, rows: dict, models: dict, title: str):
    """All ~same size within this group (FC head dominates) -- size doesn't discriminate,
    so accuracy alone as grouped bars reads far cleaner than a crowded scatter."""
    labels = [lbl.replace("\n", " ") for lbl, _ in models.values()]
    patterns = [pat for _, pat in models.values()]
    fp32 = [float(rows[m]["fp32_top1"]) for m in models]
    int8 = [float(rows[m]["int8_top1"]) for m in models]

    x = range(len(models))
    width = 0.36
    colors = [PATTERN_COLOR[p] for p in patterns]
    ax.bar([i - width / 2 for i in x], fp32, width, color=colors, edgecolor="white", lw=0.6, label="FP32")
    ax.bar([i + width / 2 for i in x], int8, width, color=colors, alpha=0.45, edgecolor="white", lw=0.6, label="INT8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_title(title)
    fp32_patch = plt.Rectangle((0, 0), 1, 1, facecolor="gray", label="FP32")
    int8_patch = plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.45, label="INT8")
    ax.legend(handles=[fp32_patch, int8_patch], loc="upper right", fontsize=9)


def main():
    apply_report_style(figsize=(13, 6.5))
    rows = load_rows()

    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1, 1.15])
    fig.subplots_adjust(top=0.72, wspace=0.25)
    plot_scatter_panel(ax1, rows, GAP_MODELS, "Head GAP compacto (AlexNet)\ntamanho varia 1-9 MB")
    plot_bar_panel(ax2, rows, FC_MODELS, "Head FC grande (AlexNetTV + VGG16)\ntamanho ~215-515 MB, quase fixo por família")
    ax1.set_ylabel("Top-1 (%)")
    ax2.set_ylabel("Top-1 (%)")

    pattern_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                                   markeredgecolor="white", markersize=10, label=p)
                        for p, c in PATTERN_COLOR.items()]
    stage_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
                   markeredgecolor="white", markersize=9, label="FP32 (círculo, esq.)"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="gray",
                   markeredgecolor="white", markersize=9, label="INT8 (quadrado, esq.)"),
    ]
    fig.legend(handles=pattern_handles, title="Padrão de kernel", loc="upper center",
               bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=9)
    fig.legend(handles=stage_handles, title="Estágio (painel esquerdo)", loc="upper center",
               bbox_to_anchor=(0.5, 0.86), ncol=2, fontsize=9)

    fig.suptitle("Phase 11 — kernel misto 3x3/2x2 vs uniforme vs original (AlexNet, AlexNetTV, VGG16)",
                 y=1.04, fontsize=13)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "kernel_pattern_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
