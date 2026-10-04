"""Phase 11 figures 02-04 -- the kernel pattern, one network/layout per figure. Cells are picked by their factors
(ml/model_registrations.py:CELL_FACTORS, via design_figures.frame); within one panel only the kernel changes.

    02_kernel_original_layout.png  AlexNet, torchvision's layout: FC + Dropout (= AlexNet) | FC | GAP
    03_kernel_64px_layout.png      AlexNet, 64px layout: {GAP, FC} x {no BN, BN} (dots = seeds, where replicated)
    04_kernel_vgg16.png            VGG16 + BN at its own geometry: GAP | FC | FC + Dropout (= VGG16)

Figure 01 (accuracy x size) is retired: figure 16 (scripts/phase11/design_figures.py) plots every run against MACs and
INT8 size, with the Pareto front. "layout original" / "layout 64px": scripts/phase11/analyze_geometry.py's docstring.

    python -m scripts.phase11.plot_kernel_comparison
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ml.plotting import NEUTRAL, apply_report_style
from scripts.phase11.analyze_geometry import BASE_KEY, BASE_LABEL, int8_bar, precision_handles, reference_lines, savefig
from scripts.phase11.design_figures import K4, KCOLOR, KLABEL, frame

ORIGINAL = dict(family="alexnet", stride=4, pooling="3pool3x3", bn=False)
PX64 = dict(family="alexnet", stride=2, pooling="2pool2x2", dropout=False)
VGG = dict(family="vgg16")
FIGURES = {  # file: (title, [(panel title, factor filter)], baseline key, baseline label)
    "02_kernel_original_layout.png": (
        "Kernel no AlexNet de layout original (o do torchvision: conv1 stride 4 + 3 max-pools 3×3/2 → mapa final 1×1 em 64×64)\n"
        "em cada painel só o kernel muda; treino do zero, sem BN, seed 42",
        [("FC + Dropout 0,5 (a cabeça do AlexNet original)", {**ORIGINAL, "head": "fc", "dropout": True}),
         ("FC sem Dropout", {**ORIGINAL, "head": "fc", "dropout": False}), ("GAP (média global + 1 linear)", {**ORIGINAL, "head": "gap"})],
        BASE_KEY, BASE_LABEL),
    "03_kernel_64px_layout.png": (
        "Kernel no AlexNet de layout 64px (adaptado a 64×64: conv1 stride 2 + 2 max-pools 2×2 → mapa final 8×8, sem Dropout)\n"
        "em cada painel só o kernel muda; treino do zero, seed 42 (pontos = seeds, onde replicada)",
        [(f"{'GAP' if h == 'gap' else 'FC'} · {'com' if bn else 'sem'} BatchNorm", {**PX64, "head": h, "bn": bn})
         for h in ("gap", "fc") for bn in (False, True)],
        BASE_KEY, BASE_LABEL),
    "04_kernel_vgg16.png": (
        "Kernel na VGG16 (13 convs + BatchNorm, stride 1, 5 max-pools 2×2 → mapa 2×2), treino do zero, seed 42\n"
        "em cada painel só o kernel das 13 convs muda",
        [("GAP (média global + 1 linear)", {**VGG, "head": "gap"}), ("FC sem Dropout", {**VGG, "head": "fc", "dropout": False}),
         ("FC + Dropout 0,5 (a cabeça da VGG16 original)", {**VGG, "head": "fc", "dropout": True})],
        "vgg16_k3x3_stride1_5pool2x2_map2_fcdrop_bn", "VGG16 original (3×3)"),
}


def select(df, filt):
    """The from-scratch runs (every seed) whose factors match filt."""
    return df[np.logical_and.reduce([df[c] == v for c, v in filt.items()]) & ~df.pretrained & df.kernels.isin(K4)]


def fig_kernel(df, filename, title, panels, base_key, base_label):
    sizes = [max(1, select(df, f).kernels.nunique()) for _, f in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.6 * sum(sizes) + 3, 6.2), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": sizes})
    shown, seeded, ref = set(), False, []
    for ax, (panel, filt) in zip(axes[0], panels):
        sel = select(df, filt)
        kernels = [k for k in K4 if k in set(sel.kernels)]
        if not kernels:
            ax.text(0.5, 0.5, "pendente", transform=ax.transAxes, ha="center", color=NEUTRAL, fontsize=12)
        ticks = []
        for i, k in enumerate(kernels):
            runs, color = sel[sel.kernels == k], KCOLOR[k]
            shown.add(k)
            fp32, int8 = runs.fp32.mean(), runs.int8.mean()
            ax.bar(i - 0.19, fp32, 0.38, color=color, edgecolor="white")
            int8_bar(ax, i + 0.19, int8, 0.38, color)
            value = dict(ha="center", va="bottom", fontsize=8, zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
            ax.text(i - 0.19, runs.fp32.max() + 0.5, f"{fp32:.1f}", **value)
            if np.isfinite(int8):
                ax.text(i + 0.19, runs.int8.max() + 0.5, f"{int8:.1f}", **value)
            if len(runs) > 1:
                seeded = True
                ax.scatter([i - 0.19] * len(runs), runs.fp32, color="k", s=12, zorder=5)
                ax.scatter([i + 0.19] * len(runs), runs.int8, color="k", s=12, zorder=5)
            ticks.append(f"{KLABEL[k]}\n{runs.macs_m.iloc[0]:.0f}M MACs" + (f"\nmédia de {len(runs)} seeds" if len(runs) > 1 else ""))
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, fontsize=9)
        ax.set_title(panel, fontsize=10)
        ax.margins(y=0.08)
        ref = reference_lines(ax, df, base_key, base_label) or ref
    axes[0][0].set_ylabel("Top-1 (%)")
    rest = precision_handles() + ref
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
    for filename, (title, panels, base_key, base_label) in FIGURES.items():
        fig_kernel(df, filename, title, panels, base_key, base_label)


if __name__ == "__main__":
    main()
