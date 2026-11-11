"""Phase 11 figures 01-03, 27, 28 -- the kernel pattern, one network/layout per figure. Cells are picked by their factors
(ml/model_registrations.py:CELL_FACTORS, via design_figures.frame); within one panel only the kernel changes.

    01_kernel_original_layout.png  AlexNet, torchvision's layout: FC + Dropout (= AlexNet) | GAP
    02_kernel_64px_layout.png      AlexNet, 64px layout: {GAP, FC + Dropout} x {no BN, BN}, 2 x 2 (dots = seeds, where replicated)
    03_kernel_vgg16.png            VGG16 + BN at its own geometry: GAP | FC | FC + Dropout (= VGG16)
    27_large_stem.png              AlexNet 64px + BN, GAP | FC + Dropout: 11-5-3-3-3 against conv1 11x11 + a 3x3 / 2x2 body
                                   (phase_11_large_stem) and the all-3x3 / all-2x2 nets -- is the 11-5-3-3-3 lead conv1's?
    28_resnet_kernel_stem.png      ResNet-18 + BN + GAP: its 7-3 against the stem / block kernels (phase_11_resnet_kernel_stem)

Bars are seed 42, as in every other Phase 11 figure; a replicated cell's seeds are the dots beside them. A kernel the
design plans for a panel (factor_effects.design_cells) that has not run yet is a "pendente" tick, so a figure shows its
whole block while the queue runs. 27 / 28 hold what figure 11 leaves out since 2026-10-24 (contrasts of 1-2 pairs).
The old figure 01 (accuracy x size) is retired: figure 13 (scripts/phase11/design_figures.py) plots every run against MACs and
INT8 size, with the Pareto front. "layout original" / "layout 64px": scripts/phase11/analyze_geometry.py's docstring.

    python -m scripts.phase11.plot_kernel_comparison
"""
import math
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ml.plotting import NEUTRAL, apply_report_style
from scripts.phase11.analyze_geometry import (BASE_KEY, BASE_LABEL, RESNET_BASE, VGG_BASE, early_handle, early_ring, int8_bar,
                                              precision_handles, reference_handles, reference_lines, savefig)
from scripts.phase11.design_figures import K4, KCOLOR, KLABEL, RESNET_K, frame
from scripts.phase11.factor_effects import design_cells

ORIGINAL = dict(family="alexnet", stride=4, pooling="3pool3x3", bn=False)
PX64 = dict(family="alexnet", stride=2, pooling="2pool2x2")
VGG = dict(family="vgg16")
SEEDS_NOTE = "em cada painel só o kernel muda; treino do zero, barras = seed 42 (pontos = cada seed, onde replicada)"
FIGURES = {  # file: (title, [(panel title, factor filter)], baseline key, baseline label, panels per row[, kernels in bar order])
    "01_kernel_original_layout.png": (
        "Kernel no AlexNet de layout original (o da AlexNet original: conv1 stride 4 + 3 max-pools 3×3/2 → mapa final 1×1 em 64×64)\n"
        "em cada painel só o kernel muda; treino do zero, sem BN, seed 42",
        [("FC + Dropout 0,5 (o classificador do AlexNet original)", {**ORIGINAL, "head": "fc", "dropout": True}),
         ("GAP (média global + 1 linear)", {**ORIGINAL, "head": "gap"})],
        BASE_KEY, BASE_LABEL, 2),
    "02_kernel_64px_layout.png": (
        "Kernel no AlexNet de layout 64px (adaptado a 64×64: conv1 stride 2 + 2 max-pools 2×2 → mapa final 8×8)\n"
        "em cada painel só o kernel muda; treino do zero, barras = seed 42 (pontos = cada seed, onde replicada)",
        [(f"{'GAP' if h == 'gap' else 'FC + Dropout'} · {'com' if bn else 'sem'} BatchNorm",
          {**PX64, "head": h, "bn": bn, "dropout": h == "fc"}) for h in ("gap", "fc") for bn in (False, True)],
        BASE_KEY, BASE_LABEL, 2),  # 2 x 2: four panels in a row were ~4300 px wide, the text too small on a page
    "03_kernel_vgg16.png": (
        "Kernel na VGG16 (13 convs + BatchNorm, stride 1, 5 max-pools 2×2 → mapa 2×2)\n"
        "em cada painel só o kernel das 13 convs muda; treino do zero, barras = seed 42 (pontos = cada seed, onde replicada)",
        [("GAP (média global\n+ 1 linear)", {**VGG, "head": "gap"}), ("FC sem Dropout", {**VGG, "head": "fc", "dropout": False}),
         ("FC + Dropout 0,5\n(como na VGG16 original)", {**VGG, "head": "fc", "dropout": True})],
        *VGG_BASE, 3),
    "27_large_stem.png": (
        "Conv1 grande + corpo de kernel pequeno (AlexNet de layout 64px, com BatchNorm): a vantagem do 11-5-3-3-3 vem da conv1 11×11?\n"
        + SEEDS_NOTE,
        [(f"{'GAP' if h == 'gap' else 'FC + Dropout'} · com BN", {**PX64, "head": h, "bn": True, "dropout": h == "fc"})
         for h in ("gap", "fc")],
        BASE_KEY, BASE_LABEL, 2, ["k11-5-3", "k11-3", "k3x3", "k11-2", "k2x2"]),
    "28_resnet_kernel_stem.png": (
        "Kernel na ResNet-18 (BatchNorm, classificador GAP, conv1 stride 2 + 1 max-pool 3×3 → mapa final 2×2)\n"
        "só o kernel da conv1 e/ou dos blocos muda; conv1 profunda = três convs pequenas no lugar da conv1 (ResNet-C)\n"
        "treino do zero, barras = seed 42 (pontos = cada seed, onde replicada)",
        [("GAP + BatchNorm (o classificador da ResNet-18 original)", {"family": "resnet18"})],
        *RESNET_BASE, 1, RESNET_K),
}


def select(df, filt, kernels=K4):
    """The from-scratch runs (every seed) -- or design cells -- whose factors match filt."""
    return df[np.logical_and.reduce([df[c] == v for c, v in filt.items()]) & ~df.pretrained & df.kernels.isin(kernels)]


def fig_kernel(df, filename, title, panels, base_key, base_label, ncols, kernels=K4):
    nrows = math.ceil(len(panels) / ncols)
    plan = design_cells()
    plan = plan[plan.input_size == 64]  # the 224 px twins are figure 23's
    # per panel: every kernel it has a seed-42 run of or the design still plans (shown as "pendente"), in the figure's order
    ks = [[k for k in kernels if k in set(select(df[df.seed == 42], f, kernels).kernels) | set(select(plan, f, kernels).kernels)]
          for _, f in panels]
    sizes = [max(1, len(k)) for k in ks]
    widths = [max(sizes[c::ncols]) for c in range(ncols)]  # a column as wide as its widest panel
    # ~0.8 in a bar pair: 03 / 27 print at 0.86 / 1 \textwidth in the article, drawn near that size
    fig, axes = plt.subplots(nrows, ncols, figsize=(0.8 * sum(widths) + 0.6, 3.4 * nrows + 0.6), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": widths}, layout="constrained")
    shown, seeded, flagged, ringed = set(), False, False, False
    for n, (ax, (panel, filt), panel_ks) in enumerate(zip(axes.flat, panels, ks)):
        sel = select(df, filt, kernels)
        if not panel_ks:
            ax.text(0.5, 0.5, "pendente", transform=ax.transAxes, ha="center", color=NEUTRAL, fontsize=12)
        ticks = []
        for i, k in enumerate(panel_ks):
            runs, color = sel[sel.kernels == k], KCOLOR[k]
            if not (runs.seed == 42).any():
                ticks.append(f"{fill(KLABEL[k], 10)}\npendente")
                continue
            r42 = runs[runs.seed == 42].iloc[0]  # the bar; every seed is a dot
            shown.add(k)
            fp32, int8 = r42.fp32, r42.int8
            ax.bar(i - 0.19, fp32, 0.38, color=color, edgecolor="white")
            early = bool(r42.early_best)
            flagged |= early
            int8_bar(ax, i + 0.19, int8, 0.38, color, early=early)
            # one label over the pair, FP32 above INT8 (two side by side touched at the printed size); above the bars and
            # the non-early dots: over an early-best ring it read as the ring's value
            top = np.nanmax([runs.fp32.max(), int8, runs.int8.where(~runs.early_best.astype(bool)).max()])
            ax.text(i, top + 0.5, f"{fp32:.1f}" + (f"\n{int8:.1f}" if np.isfinite(int8) else ""), ha="center", va="bottom",
                    fontsize=9, zorder=6, linespacing=1.0, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
            if len(runs) > 1:
                seeded = True
                ax.scatter([i - 0.19] * len(runs), runs.fp32, color="k", s=12, zorder=5)
                ax.scatter([i + 0.19] * len(runs), runs.int8, color="k", s=12, zorder=5)
                for e in runs[runs.early_best].itertuples():  # an early-best INT8 is marked wherever drawn, a seed's dot too
                    early_ring(ax, i + 0.19, e.int8, s=60)
                ringed |= bool(runs.early_best.any())
            ticks.append(f"{fill(KLABEL[k], 10)}\n{r42.macs_eff_m:.0f}M")
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, fontsize=9)
        ax.set_xlabel("kernel · MACs por imagem", fontsize=9)
        ax.set_title(panel, fontsize=9)
        ax.margins(y=0.2)  # room for the two-line value labels under the panel title
        reference_lines(ax, df, base_key, base_label, text=None)  # named in the legend: text beside the axes cost width
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-1 (%)")
    rest = (precision_handles() + ([early_handle("hatch")] if flagged else []) + ([early_handle("ring")] if ringed else [])
            + reference_handles(df, base_key, base_label))
    if seeded:
        rest.append(Line2D([], [], marker="o", color="k", ls="", ms=4, label="uma seed (42/43/44)"))
    fig.legend(handles=[Patch(color=KCOLOR[k], label=KLABEL[k]) for k in kernels if k in shown], loc="upper right",
               bbox_to_anchor=(0.49, 0.0), fontsize=9, title="cor = kernel (da 1ª à última conv)", title_fontsize=9)
    fig.legend(handles=rest, loc="upper left", bbox_to_anchor=(0.51, 0.0), fontsize=9, title="preenchimento / linhas", title_fontsize=9)
    fig.suptitle(title, fontsize=11)
    savefig(fig, filename)


def main():
    apply_report_style(figsize=(9, 6))
    df = frame()
    if df.empty:
        print("no Phase 11 run has a summary yet")
        return
    for filename, spec in FIGURES.items():
        fig_kernel(df, filename, *spec)


if __name__ == "__main__":
    main()
