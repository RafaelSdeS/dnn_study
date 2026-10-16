"""Phase 11 figures 01-03 -- the kernel pattern, one network/layout per figure. Cells are picked by their factors
(ml/model_registrations.py:CELL_FACTORS, via design_figures.frame); within one panel only the kernel changes.

    01_kernel_original_layout.png  AlexNet, torchvision's layout: FC + Dropout (= AlexNet) | GAP
    02_kernel_64px_layout.png      AlexNet, 64px layout: {GAP, FC + Dropout} x {no BN, BN}, 2 x 2 (dots = seeds, where replicated)

Bars are seed 42, as in every other Phase 11 figure; a replicated cell's seeds are the dots beside them.
    03_kernel_vgg16.png            VGG16 + BN at its own geometry: GAP | FC | FC + Dropout (= VGG16)

The old figure 01 (accuracy x size) is retired: figure 13 (scripts/phase11/design_figures.py) plots every run against MACs and
INT8 size, with the Pareto front. "layout original" / "layout 64px": scripts/phase11/analyze_geometry.py's docstring.

    python -m scripts.phase11.plot_kernel_comparison
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ml.plotting import NEUTRAL, apply_report_style
from scripts.phase11.analyze_geometry import (BASE_KEY, BASE_LABEL, VGG_BASE, early_handle, int8_bar, precision_handles,
                                              reference_lines, savefig)
from scripts.phase11.design_figures import K4, KCOLOR, KLABEL, frame

ORIGINAL = dict(family="alexnet", stride=4, pooling="3pool3x3", bn=False)
PX64 = dict(family="alexnet", stride=2, pooling="2pool2x2")
VGG = dict(family="vgg16")
FIGURES = {  # file: (title, [(panel title, factor filter)], baseline key, baseline label, panels per row)
    "01_kernel_original_layout.png": (
        "Kernel no AlexNet de layout original (o do torchvision: conv1 stride 4 + 3 max-pools 3×3/2 → mapa final 1×1 em 64×64)\n"
        "em cada painel só o kernel muda; treino do zero, sem BN, seed 42",
        [("FC + Dropout 0,5 (a cabeça do AlexNet original)", {**ORIGINAL, "head": "fc", "dropout": True}),
         ("GAP (média global + 1 linear)", {**ORIGINAL, "head": "gap"})],
        BASE_KEY, BASE_LABEL, 2),
    "02_kernel_64px_layout.png": (
        "Kernel no AlexNet de layout 64px (adaptado a 64×64: conv1 stride 2 + 2 max-pools 2×2 → mapa final 8×8)\n"
        "em cada painel só o kernel muda; treino do zero, barras = seed 42 (pontos = cada seed, onde replicada)",
        [(f"{'GAP' if h == 'gap' else 'FC + Dropout'} · {'com' if bn else 'sem'} BatchNorm",
          {**PX64, "head": h, "bn": bn, "dropout": h == "fc"}) for h in ("gap", "fc") for bn in (False, True)],
        BASE_KEY, BASE_LABEL, 2),  # 2 x 2: four panels in a row were ~4300 px wide, the text too small on a page
    "03_kernel_vgg16.png": (
        "Kernel na VGG16 (13 convs + BatchNorm, stride 1, 5 max-pools 2×2 → mapa 2×2), treino do zero, seed 42\n"
        "em cada painel só o kernel das 13 convs muda",
        [("GAP (média global + 1 linear)", {**VGG, "head": "gap"}), ("FC sem Dropout", {**VGG, "head": "fc", "dropout": False}),
         ("FC + Dropout 0,5 (a cabeça da VGG16 original)", {**VGG, "head": "fc", "dropout": True})],
        *VGG_BASE, 3),
}


def select(df, filt):
    """The from-scratch runs (every seed) whose factors match filt."""
    return df[np.logical_and.reduce([df[c] == v for c, v in filt.items()]) & ~df.pretrained & df.kernels.isin(K4)]


def fig_kernel(df, filename, title, panels, base_key, base_label, ncols):
    nrows = math.ceil(len(panels) / ncols)
    sizes = [max(1, select(df, f).kernels.nunique()) for _, f in panels]
    widths = [max(sizes[c::ncols]) for c in range(ncols)]  # a column as wide as its widest panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * sum(widths) + 3, 5.6 * nrows + 0.6), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": widths})
    shown, seeded, flagged = set(), False, False
    for n, (ax, (panel, filt)) in enumerate(zip(axes.flat, panels)):
        sel = select(df, filt)
        kernels = [k for k in K4 if k in set(sel[sel.seed == 42].kernels)]
        if not kernels:
            ax.text(0.5, 0.5, "pendente", transform=ax.transAxes, ha="center", color=NEUTRAL, fontsize=12)
        ticks = []
        for i, k in enumerate(kernels):
            runs, color = sel[sel.kernels == k], KCOLOR[k]
            r42 = runs[runs.seed == 42].iloc[0]  # the bar; every seed is a dot
            shown.add(k)
            fp32, int8 = r42.fp32, r42.int8
            ax.bar(i - 0.19, fp32, 0.38, color=color, edgecolor="white")
            early = bool(r42.early_best)
            flagged |= early
            int8_bar(ax, i + 0.19, int8, 0.38, color, early=early)
            value = dict(ha="center", va="bottom", fontsize=8, zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
            ax.text(i - 0.19, runs.fp32.max() + 0.5, f"{fp32:.1f}", **value)
            if np.isfinite(int8):
                ax.text(i + 0.19, runs.int8.max() + 0.5, f"{int8:.1f}", **value)
            if len(runs) > 1:
                seeded = True
                ax.scatter([i - 0.19] * len(runs), runs.fp32, color="k", s=12, zorder=5)
                ax.scatter([i + 0.19] * len(runs), runs.int8, color="k", s=12, zorder=5)
            ticks.append(f"{KLABEL[k]}\n{r42.macs_eff_m:.0f}M MACs")
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, fontsize=9)
        ax.set_title(panel, fontsize=10)
        ax.margins(y=0.08)
        last_in_row = n % ncols == ncols - 1 or n == len(panels) - 1
        reference_lines(ax, df, base_key, base_label, text="right" if last_in_row else None)  # sharey: one label per row
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-1 (%)")
    rest = precision_handles() + ([early_handle("hatch")] if flagged else [])
    if seeded:
        rest.append(Line2D([], [], marker="o", color="k", ls="", ms=4, label="uma seed (42/43/44)"))
    fig.legend(handles=[Patch(color=KCOLOR[k], label=KLABEL[k]) for k in K4 if k in shown], loc="upper right",
               bbox_to_anchor=(0.49, 0.0), fontsize=9, title="cor = kernel (da 1ª à última conv)", title_fontsize=9)
    fig.legend(handles=rest, loc="upper left", bbox_to_anchor=(0.51, 0.0), fontsize=9, title="preenchimento / linhas", title_fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    savefig(fig, filename)


def main():
    apply_report_style(figsize=(9, 6))
    df = frame()
    if df.empty:
        print("no Phase 11 run has a summary yet")
        return
    for filename, (title, panels, base_key, base_label, ncols) in FIGURES.items():
        fig_kernel(df, filename, title, panels, base_key, base_label, ncols)


if __name__ == "__main__":
    main()
