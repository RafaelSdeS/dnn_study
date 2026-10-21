"""Phase 12 figures -- robustness of the Phase 11 conclusions (docs/logs/PHASE11_LOG.md "Controls and robustness runs"):

    01_lr_sensitivity.png       the three AlexNet GAP + BN kernels at half / the / double the recipe's lr
                                (phase_12_lr_half / lr_double, and phase_11's lr 0.01 runs)
    02_finetune_pretrained.png  each pretrained twin: from scratch, pretrained on the one recipe (lr 0.01, which erases the
                                pretraining), fine-tuned at lr 1e-3 and 1e-4 (the one with the better *validation* top-1 starred)
    03_cifar100.png             the kernel contrast on CIFAR-100: AlexNet's 4 kernel sets, VGG16's 3, three seeds

Phase 11's conventions: bars / lines = seed 42, dots = every seed, INT8 = the light bar, test top-1. A panel without its runs
yet is skipped. Table: results/phase_12_robustness/all_runs.csv.

    python -m scripts.phase12.robustness
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ml.model_registrations import CELL_FACTORS
from ml.plotting import AMBER, BLUE, GREEN, NEUTRAL, TEXT_SECONDARY, apply_report_style
from scripts.phase11.analyze_geometry import ROOT, RUNS, int8_bar, load, noise_band, precision_handles, savefig
from scripts.phase11.design_figures import KCOLOR, KLABEL

FIGS = ROOT / "results/figures_generated/phase_12_robustness"
TABLES = ROOT / "results/phase_12_robustness"
LR = {"lr_half": 0.005, "lr_double": 0.02}  # phase_12_<exp>.yaml: the FP32 lr of the lr-sensitivity runs
FINETUNE = {"finetune_pretrained_lr1e3": 1e-3, "finetune_pretrained_in224_lr1e3": 1e-3,
            "finetune_pretrained_lr1e4": 1e-4, "finetune_pretrained_in224_lr1e4": 1e-4}
LR_KERNELS = ["k11-5-3", "k3x3", "k2x2"]
LR_KEY = {k: f"alexnet_{k}_stride2_2pool2x2_map8_gap_bn" for k in LR_KERNELS}


def with_factors(df):
    """kernels / family of every run whose model is a registered cell (mobilenetv2 and the like: None)."""
    f = [CELL_FACTORS.get(k, {}) for k in df.key]
    return df.assign(kernels=[x.get("kernels") for x in f], family=[x.get("family") for x in f])


def fig_lr(p11, p12, figs):
    runs = pd.concat([p11.assign(lr=0.01), p12[p12.exp.isin(LR)].assign(lr=p12.exp.map(LR))])
    runs = runs[runs.key.isin(LR_KEY.values())]
    if not (runs.lr != 0.01).any():
        print("01: no lr-sensitivity run yet")
        return
    lrs = sorted(set(runs.lr))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, stage, title in zip(axes, ("fp32", "int8"), ("FP32", "INT8")):
        for k in LR_KERNELS:
            r = runs[runs.key == LR_KEY[k]]
            ax.scatter(r.lr, r[stage], color=KCOLOR[k], s=14, alpha=0.6, zorder=3)  # every seed
            s42 = r[r.seed == 42].sort_values("lr")
            ax.plot(s42.lr, s42[stage], color=KCOLOR[k], marker="o", lw=2, label=KLABEL[k], zorder=4)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(lrs)
        ax.set_xticklabels([f"{x:g}" for x in lrs])
        ax.set_xlabel("lr do FP32 (a receita da Phase 11 usa 0,01)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Top-1 (%)")
    axes[0].legend(title="kernel (todas as convs)", fontsize=9)
    fig.suptitle("O ranking dos kernels depende do lr? AlexNet, layout 64px, GAP + BN\n"
                 "linha = seed 42, pontos = cada seed (as 3 em 0,01) · top-1 no test set", fontsize=11)
    savefig(fig, "01_lr_sensitivity.png", figs)


def fig_finetune(p11, p12, figs):
    ft = p12[p12.exp.isin(FINETUNE)].assign(lr=p12.exp.map(FINETUNE))
    if ft.empty:
        print("02: no fine-tune run yet")
        return
    keys = sorted(set(ft.key))
    w = 0.115
    fig, ax = plt.subplots(figsize=(max(9.0, 2.6 * len(keys) + 1.5), 5.6))
    conditions = [("do zero, 500 épocas", NEUTRAL), ("pré-treino, receita única (lr 0,01)", AMBER),
                  ("fine-tune, lr 1e-3", BLUE), ("fine-tune, lr 1e-4", GREEN)]
    for i, key in enumerate(keys):
        twin = key.removesuffix("_pretrained") if key.endswith("_pretrained") else f"{key}_scratch"
        sets = [p11[p11.key == twin], p11[p11.key == key], ft[(ft.key == key) & (ft.lr == 1e-3)], ft[(ft.key == key) & (ft.lr == 1e-4)]]
        both = ft[ft.key == key]
        chosen = both.loc[both.fp32_val.idxmax(), "lr"]  # the lr read off the validation split, never the test's
        for j, ((_, color), r) in enumerate(zip(conditions, sets)):
            r = r[r.seed == 42]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.bar(i + (2 * j - 3.5) * w, r.fp32, w, color=color, edgecolor="white")
            int8_bar(ax, i + (2 * j - 2.5) * w, r.int8, w, color, early=bool(r.early_best))
            ax.text(i + (2 * j - 3.5) * w, r.fp32 + 0.5, f"{r.fp32:.1f}", ha="center", fontsize=7.5)
            if j >= 2 and r.lr == chosen:
                ax.text(i + (2 * j - 3) * w, max(r.fp32, r.int8) + 3, "★", ha="center", fontsize=11, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([textwrap.fill(k.removesuffix("_pretrained").replace("_", " "), 24) for k in keys], fontsize=8.5)
    ax.set_ylabel("Top-1 (%)")
    ax.margins(y=0.1)
    handles = [Patch(color=c, label=n) for n, c in conditions] + precision_handles() + \
              [Line2D([], [], ls="", label="★ = lr com melhor top-1 na validação")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)
    fig.suptitle("Os gêmeos pré-treinados com fine-tuning de verdade: lr 1e-3 e 1e-4, 100 épocas, QAT de 10 (seed 42)\n"
                 "a receita única (lr 0,01) apaga o pré-treino; aqui ele é mantido · top-1 no test set", fontsize=11)
    savefig(fig, "02_finetune_pretrained.png", figs)


def fig_cifar(p12, figs):
    c = p12[p12.exp.str.startswith("cifar100")]
    if c.empty:
        print("03: no CIFAR-100 run yet")
        return
    panels = [("AlexNet (layout 64px, GAP + BN)", "alexnet", ["k11-5-3", "k3x3", "k2x2", "kalt3-2"]),
              ("VGG16 (GAP + BN)", "vgg16", ["k3x3", "k2x2", "kalt3-2"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [4, 3]})
    for ax, (title, family, kernels) in zip(axes, panels):
        shown = []
        for i, k in enumerate(kernels):
            r = c[(c.family == family) & (c.kernels == k)]
            if (r.seed == 42).any():
                r42 = r[r.seed == 42].iloc[0]
                ax.bar(i - 0.19, r42.fp32, 0.38, color=KCOLOR[k], edgecolor="white")
                int8_bar(ax, i + 0.19, r42.int8, 0.38, KCOLOR[k], early=bool(r42.early_best))
                ax.text(i - 0.19, r.fp32.max() + 0.5, f"{r42.fp32:.1f}", ha="center", fontsize=8.5)
                if len(r) > 1:
                    ax.scatter([i - 0.19] * len(r), r.fp32, color="k", s=12, zorder=5)
                    ax.scatter([i + 0.19] * len(r), r.int8, color="k", s=12, zorder=5)
                shown.append(k)
        ax.set_xticks(range(len(kernels)))
        ax.set_xticklabels([KLABEL[k] if k in shown else f"{KLABEL[k]}\npendente" for k in kernels], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.12)
    axes[0].set_ylabel("Top-1 (%)")
    band = noise_band(c, set(c.key))
    note = (f" · ruído entre seeds (2√2·SD): ±{band['fp32']:.1f} pp FP32, ±{band['int8']:.1f} pp INT8" if np.isfinite(band["fp32"]) else "")
    fig.legend(handles=precision_handles() + [Line2D([], [], marker="o", color="k", ls="", ms=4, label="uma seed (42/43/44)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)
    fig.suptitle("O contraste de kernels no CIFAR-100 (32×32, 100 classes), as mesmas redes e a mesma receita\n"
                 "barra = seed 42, pontos = cada seed · top-1 nas 10 000 imagens oficiais de teste" + note, fontsize=11)
    savefig(fig, "03_cifar100.png", figs)


def main():
    apply_report_style()
    p12 = load(*RUNS, prefix="phase_12_")
    if p12.empty:
        print("no Phase 12 run has a summary yet")
        return
    p11, p12 = with_factors(load(*RUNS)), with_factors(p12)
    TABLES.mkdir(parents=True, exist_ok=True)
    p12.drop(columns="results").sort_values(["exp", "key", "seed"]).to_csv(TABLES / "all_runs.csv", index=False)
    fig_lr(p11, p12, FIGS)
    fig_finetune(p11, p12, FIGS)
    fig_cifar(p12, FIGS)


if __name__ == "__main__":
    main()
