"""Phase 11 figures 01-04 — kernel pattern, one network/layout per figure, plus the accuracy-vs-size overview.

    01_overview_accuracy_vs_size.png   every kernel run below (the one figure that mixes factors on purpose)
    02_kernel_original_layout.png      AlexNet, layout original, head FC | GAP
    03_kernel_64px_layout.png          AlexNet, layout 64px, head FC | GAP (dots = seeds 42/43/44)
    04_kernel_vgg16.png                VGG16, 3x3 (its original) vs 2x2

Figures 05-14 come from scripts/phase11/factor_effects.py, 15 from scripts/phase11/analyze_geometry.py, whose module
docstring defines "layout original" / "layout 64px". Data comes from analyze_geometry.load(), so a pre-fix INT8 is only
ever drawn faded. Within one panel only the kernel changes, except for the init: † = He init, while the original-kernel
runs use PyTorch's default (the original AlexNet's He-init retry never trained -- docs/logs/PHASE11_LOG.md). Across
layouts the gap is geometry, not kernel ("Geometry confound" in the log).

    python -m scripts.phase11.plot_kernel_comparison
"""
from statistics import geometric_mean

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ml.plotting import AMBER, BLUE, GREEN, RED, apply_report_style
from scripts.phase11.analyze_geometry import (BASE_KEY, BASE_LABEL, LAYOUTS, OLD_INT8, int8_bar, load, precision_handles,
                                              reference_lines, savefig)

KERNEL_COLOR = {"11-5-3-3-3": AMBER, "3×3": GREEN, "2×2": BLUE}  # any other label is a mixed 2x2/3x3 pattern -> RED
KERNEL_LEGEND = [Patch(color=AMBER, label="11-5-3-3-3 (kernels originais do AlexNet)"),
                 Patch(color=GREEN, label="3×3 em todas as convs"), Patch(color=BLUE, label="2×2 em todas as convs"),
                 Patch(color=RED, label="misto 2×2/3×3 (rótulo = kernel de cada conv, da 1ª à 5ª)")]

# (layout, head) -> [(run key, kernel label)]: within one list only the kernel changes (and the init, marked †).
KERNEL_SETS = {
    ("original", "FC"): [(BASE_KEY, "11-5-3-3-3"), ("alexnet_tv_3x3", "3×3†"), ("alexnet_tv_2x2", "2×2†"),
                         ("alexnet_tv_mixed_early3", "3-3-3-2-2†"), ("alexnet_tv_mixed_alt", "2-3-2-3-2†"),
                         ("alexnet_tv_mixed_early2", "2-2-2-3-3†")],
    ("original", "GAP"): [("alexnet_geo_s4_p3_gap", "11-5-3-3-3"), ("alexnet_tv_mixed_early3_gap", "3-3-3-2-2†"),
                          ("alexnet_tv_mixed_alt_gap", "2-3-2-3-2†"), ("alexnet_tv_mixed_early2_gap", "2-2-2-3-3†")],
    ("64px", "FC"): [("alexnet_adapted_orig_fc", "11-5-3-3-3"), ("alexnet_3x3_fc", "3×3"), ("alexnet_adapted_2x2_fc", "2×2"),
                     ("alexnet_mixed_fc", "3-2-3-2-3†")],
    ("64px", "GAP"): [("alexnet_adapted_orig_gap", "11-5-3-3-3"), ("alexnet_3x3_gap", "3×3"),
                      ("alexnet_adapted_2x2_gap", "2×2"), ("alexnet_mixed", "3-2-3-2-3†")],
    ("VGG16", "FC"): [("vgg16", "3×3 (original)"), ("vgg16_2x2", "2×2")],
}
HEAD = {"FC": "cabeça FC (3 camadas densas, 4096 neurônios)", "GAP": "cabeça GAP (média global + 1 camada linear)"}
GROUP = {("original", "FC"): "AlexNet · layout original · FC", ("original", "GAP"): "AlexNet · layout original · GAP",
         ("64px", "FC"): "AlexNet · layout 64px · FC", ("64px", "GAP"): "AlexNet · layout 64px · GAP",
         ("VGG16", "FC"): "VGG16 (13 convs) · FC"}


def kernel_color(label):
    return KERNEL_COLOR.get(label.split(" ")[0].rstrip("†"), RED)


def runs_of(df, key):
    runs = df[df.key == key]
    assert len(runs), f"no summary for {key}"
    return runs


def fig_kernel(df, layout, filename, title, note, base_key=BASE_KEY, base_label=BASE_LABEL):
    heads = [h for (lay, h) in KERNEL_SETS if lay == layout]
    sizes = [len(KERNEL_SETS[(layout, h)]) for h in heads]
    fig, axes = plt.subplots(1, len(heads), figsize=(1.5 * sum(sizes) + 3, 6.2), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": sizes})
    colors, seeded = set(), False
    for ax, head in zip(axes[0], heads):
        ticks = []
        for i, (key, label) in enumerate(KERNEL_SETS[(layout, head)]):
            runs, color = runs_of(df, key), kernel_color(label)
            colors.add(color)
            fp32, int8 = runs.fp32.mean(), runs.int8.mean()
            ax.bar(i - 0.19, fp32, 0.38, color=color, edgecolor="white")
            int8_bar(ax, i + 0.19, int8, runs.int8_raw.mean(), 0.38, color)
            value = dict(ha="center", va="bottom", fontsize=8, zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
            ax.text(i - 0.19, max(fp32, runs.fp32.max()) + 0.5, f"{fp32:.1f}", **value)
            if pd.notna(int8):
                ax.text(i + 0.19, max(int8, runs.int8.max()) + 0.5, f"{int8:.1f}", **value)
            if len(runs) > 1:
                seeded = True
                valid = runs.int8.notna().all()
                ax.scatter([i - 0.19] * len(runs), runs.fp32, color="k", s=12, zorder=5)
                ax.scatter([i + 0.19] * len(runs), runs.int8 if valid else runs.int8_raw, color="k", s=12, zorder=5,
                           alpha=1 if valid else 0.35)
            ticks.append(f"{label}\n{runs.macs_m.iloc[0]:.0f}M MACs" + (f"\nmédia de {len(runs)} seeds" if len(runs) > 1 else ""))
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, fontsize=9)
        ax.set_title(HEAD[head], fontsize=10)
        ax.margins(y=0.08)
        ref = reference_lines(ax, df, base_key, base_label)
    axes[0][0].set_ylabel("Top-1 (%)")
    kernels = [h for h in KERNEL_LEGEND if h.get_facecolor()[:3] in {Patch(color=c).get_facecolor()[:3] for c in colors}]
    rest = precision_handles() + ref
    if seeded:
        rest.append(Line2D([], [], marker="o", color="k", ls="", ms=4, label="uma seed (42/43/44)"))
    fig.legend(handles=kernels, loc="upper right", bbox_to_anchor=(0.49, 0.0), fontsize=9, title="cor = kernel", title_fontsize=9)
    fig.legend(handles=rest, loc="upper left", bbox_to_anchor=(0.51, 0.0), fontsize=9, title="preenchimento / linhas", title_fontsize=9)
    fig.suptitle(f"{title}\n{note}", fontsize=11)
    fig.tight_layout()
    savefig(fig, filename)


def fig_overview(df):
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for group, entries in KERNEL_SETS.items():
        pts = []
        for key, label in entries:
            runs, color = runs_of(df, key), kernel_color(label)
            fp32, int8, int8_raw = runs.fp32.mean(), runs.int8.mean(), runs.int8_raw.mean()
            x32, x8 = runs.fp32_mb.iloc[0], runs.int8_mb.iloc[0]
            ax.plot([x32, x8], [fp32, int8 if pd.notna(int8) else int8_raw], color="gray", lw=1, alpha=0.3, zorder=1)
            ax.scatter(x32, fp32, color=color, marker="o", s=110, edgecolors="white", lw=0.6, zorder=3)
            if pd.notna(int8):
                ax.scatter(x8, int8, color=color, marker="s", s=90, edgecolors="white", lw=0.6, zorder=3)
            else:
                ax.scatter(x8, int8_raw, facecolors="none", edgecolors=color, marker="s", s=80, alpha=0.6, zorder=3)
            if key == BASE_KEY:
                ax.scatter(x32, fp32, s=380, facecolors="none", edgecolors="k", lw=1.6, zorder=4)
            pts.append((x32, fp32))
        x, y = geometric_mean([p[0] for p in pts]), max(p[1] for p in pts)  # centred above its cluster (log x)
        ax.annotate(GROUP[group], (x, y), xytext=(0, 14), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")
    ax.set_xscale("log")
    ax.margins(x=0.12, y=0.1)
    ax.set_xlabel("Tamanho do modelo (MB, escala log)")
    ax.set_ylabel("Top-1 (%)")
    ax.set_title("Visão geral: acurácia × tamanho de toda rede de kernel da Fase 11 (a figura que mistura fatores;\n"
                 "um fator por vez nas figuras 02–14; ponto ligado ao seu INT8 pela linha cinza)\n" + LAYOUTS, fontsize=10.5)
    marker_handles = [
        Line2D([], [], marker="o", color="gray", ls="", ms=9, label="FP32"),
        Line2D([], [], marker="s", color="gray", ls="", ms=8, label="INT8"),
        Line2D([], [], marker="s", mfc="none", mec="gray", ls="", ms=8, label=OLD_INT8),
        Line2D([], [], marker="o", mfc="none", mec="k", mew=1.6, ls="", ms=16, label=BASE_LABEL)]
    ax.legend(handles=KERNEL_LEGEND + marker_handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
              title="cor = kernel · forma = precisão", title_fontsize=9, labelspacing=1.0)
    savefig(fig, "01_overview_accuracy_vs_size.png")


def main():
    apply_report_style(figsize=(9, 6))
    df = load()
    # phase_11_reuse_old_init/alexnet_tv_3x3 is the default-init twin of the He-init alexnet_tv_3x3 used here
    df = df[df.exp != "reuse_old_init"]
    fig_overview(df)
    fig_kernel(df, "original", "02_kernel_original_layout.png",
               "Kernel no AlexNet de layout original (o do torchvision: conv1 stride 4 + 3 max-pools 3×3/2 → mapa final 1×1 "
               "em 64×64)\nem cada painel só o kernel muda; treino do zero, seed 42",
               "† = inicialização He; sem † = inicialização padrão do PyTorch (a versão He da original não treinou, PHASE11_LOG)")
    fig_kernel(df, "64px", "03_kernel_64px_layout.png",
               "Kernel no AlexNet de layout 64px (adaptado a 64×64: conv1 stride 2 + 2 max-pools 2×2 → mapa final 8×8, "
               "sem Dropout)\nem cada painel só o kernel muda; treino do zero",
               "† = AlexNetMixed: inicialização He e convs 2×2 sem padding (os mapas encolhem 1 px a cada 2×2)")
    fig_kernel(df, "VGG16", "04_kernel_vgg16.png",
               "Kernel na VGG16 (torchvision, 13 convs + BatchNorm), treinada do zero, seed 42\n"
               "mesma rede, só o kernel das 13 convs muda", "baseline = a própria VGG16 com seu kernel original 3×3",
               base_key="vgg16", base_label="VGG16 original (3×3)")


if __name__ == "__main__":
    main()
