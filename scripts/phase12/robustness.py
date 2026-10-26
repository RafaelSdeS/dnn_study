"""Phase 12 figures -- robustness of the Phase 11 conclusions (docs/logs/PHASE11_LOG.md "Controls and robustness runs"):

    01_lr_sensitivity.png       the three AlexNet GAP + BN kernels at half / the / double the recipe's lr
                                (phase_12_lr_half / lr_double, and phase_11's lr 0.01 runs)
    02_finetune_pretrained.png  each pretrained twin: from scratch, pretrained on the one recipe (lr 0.01, which erases the
                                pretraining), fine-tuned at lr 1e-3 and 1e-4 (the one with the better *validation* top-1 starred)
    03_cifar100.png             the kernel contrast on CIFAR-100: AlexNet's 4 kernel sets, VGG16's 3, every seed a dot
                                (Phase 11's figure 26 form, 2026-10-26: bars from 0 hid the ~1 pp, seed 42 alone misled)
    04_kernel_effect_everywhere.png  the story in one figure: 3x3 -> 2x2 and original -> 3x3 (B - A per seed) in every
                                scenario both kernels trained -- Tiny ImageNet AlexNet at 3 lrs, VGG16, ResNet-18,
                                CIFAR-100 -- and 3x3 -> 2x2 at equal MACs (Phase 11's width controls)
    05_dataset_transfer.png     Tiny ImageNet against CIFAR-100 top-1 of the same nets, one panel per family, Kendall's tau

Phase 11's conventions: bars / lines = seed 42, dots = every seed, INT8 = the light bar or the hollow marker, test top-1.
A panel without its runs yet is skipped. Table: results/phase_12_robustness/all_runs.csv.

    python -m scripts.phase12.robustness
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ml.model_registrations import CELL_FACTORS
from ml.plotting import AMBER, BLUE, GREEN, NEUTRAL, RED, TEXT_SECONDARY, apply_report_style
from scripts.phase11.analyze_geometry import (EARLY_GRAY, EARLY_LABEL, ROOT, RUNS, early_handle, int8_bar, load, noise_band,
                                              precision_handles, savefig)
from scripts.phase11.design_figures import FMARK, KCOLOR, KLABEL, SEED_HANDLES, seed_dots

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
    handles = axes[0].get_legend_handles_labels()[0] + [Line2D([], [], marker="o", color=TEXT_SECONDARY, ls="", ms=4, alpha=0.6,
                                                               label="uma seed (as 3 em 0,01)")]
    fig.legend(handles=handles, title="kernel (todas as convs)", loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)
    fig.suptitle("O ranking dos kernels depende do lr? AlexNet, layout 64px, GAP + BN\n"
                 "linha = seed 42, pontos = cada seed (as 3 em 0,01) · top-1 no test set", fontsize=11, y=1.04)
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
    ringed = False
    for ax, (title, family, kernels) in zip(axes, panels):  # each panel its own zoomed y: AlexNet ~63 %, VGG16 ~71 %
        shown = []
        for i, k in enumerate(kernels):
            r = c[(c.family == family) & (c.kernels == k)]
            if len(r):
                ringed |= seed_dots(ax, i, r, KCOLOR[k])
                shown.append(k)
        ax.set_xticks(range(len(kernels)))
        ax.set_xticklabels([KLABEL[k] if k in shown else f"{KLABEL[k]}\npendente" for k in kernels], fontsize=9)
        ax.set_xlim(-0.5, len(kernels) - 0.5)
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.15)
    axes[0].set_ylabel("Top-1 (%)")
    band = noise_band(c, set(c.key))
    note = (f"\nruído entre seeds (2√2·SD): ±{band['fp32']:.1f} pp FP32, ±{band['int8']:.1f} pp INT8" if np.isfinite(band["fp32"]) else "")
    fig.legend(handles=SEED_HANDLES + ([early_handle("ring")] if ringed else []), loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, fontsize=9)
    fig.suptitle("O contraste de kernels no CIFAR-100 (32×32, 100 classes), as mesmas redes e a mesma receita\n"
                 "cada ponto = uma seed (42/43/44) · top-1 nas 10 000 imagens oficiais de teste" + note, fontsize=11, y=1.06)
    savefig(fig, "03_cifar100.png", figs)


AK = "alexnet_{}_stride2_2pool2x2_map8_gap_bn".format  # Tiny ImageNet, 64px layout, GAP + BN: the main factorial's best column
VK = "vgg16_{}_stride1_5pool2x2_map2_gap_bn".format
RK = "resnet18_{}_stride2_1pool3x3_map2_gap_bn".format
CAK = "alexnet_{}_stride2_2pool2x2_map4_gap_bn_in32".format  # the same nets on CIFAR-100
CVK = "vgg16_{}_stride1_5pool2x2_map1_gap_bn_in32".format
SMALL, ORIG, SAME = "3×3 → 2×2", "kernel original → 3×3 (AlexNet 11-5-3-3-3, ResNet-18 stem 7×7)", "3×3 → 2×2 com os mesmos MACs"


def everywhere_rows(p11, p12):
    """[(group, label, A runs, B runs)] of figure 04, every seed; only the scenarios where both nets trained."""
    tiny = lambda k: p11[p11.key == k]  # noqa: E731
    at = lambda exp: lambda k: p12[p12.exp.str.startswith(exp) & (p12.key == k)]  # noqa: E731
    rows = []
    for a, b, group in [("k3x3", "k2x2", SMALL), ("k11-5-3", "k3x3", ORIG)]:
        for lr, get in [("0,01 (a receita)", tiny), ("0,005", at("lr_half")), ("0,02", at("lr_double"))]:
            rows.append((group, f"AlexNet · Tiny ImageNet · lr {lr}", get(AK(a)), get(AK(b))))
        if group == SMALL:
            rows.append((group, "VGG16 · Tiny ImageNet", tiny(VK(a)), tiny(VK(b))))
        rows.append((group, "ResNet-18 · Tiny ImageNet", tiny(RK("k7-3" if a == "k11-5-3" else a)), tiny(RK(b))))
        rows.append((group, "AlexNet · CIFAR-100", at("cifar100")(CAK(a)), at("cifar100")(CAK(b))))
        if group == SMALL:
            rows.append((group, "VGG16 · CIFAR-100", at("cifar100")(CVK(a)), at("cifar100")(CVK(b))))
    rows += [(SAME, "AlexNet · no custo do 3×3 (2×2 com canais ×1,5)", tiny(AK("k3x3")), tiny(AK("k2x2") + "_w150")),
             (SAME, "AlexNet · no custo do 2×2 (3×3 com canais ×0,67)", tiny(AK("k3x3") + "_w67"), tiny(AK("k2x2")))]
    return [r for r in rows if len(r[2]) and len(r[3])]


def fig_everywhere(p11, p12, figs):
    """Figure 04, figure 11's form (Phase 11) on the scenarios instead of the factors: each dot one seed's B - A (the same
    seed on both sides), hollow inside the scenario's own seed band (both cells' seeds pooled; none with one seed), grey
    INT8 where a side's best FP32 epoch came early, the diamond the mean over seeds."""
    rows = everywhere_rows(p11, p12)
    if not rows:
        print("04: no scenario with both kernels yet")
        return
    ticks, y = [], {}  # top to bottom: (text, bold) -- a group header, then its scenarios
    for group, label, a, b in rows:
        if (group, True) not in ticks:
            ticks.append((group, True))
        n = len(a[["seed"]].merge(b[["seed"]], on="seed"))
        y[(group, label)] = len(ticks)
        ticks.append((f"{label} ({n} {'seed' if n == 1 else 'seeds'})", False))
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(ticks) + 2.2), sharey=True)
    for ax, m in zip(axes, ["fp32", "int8"]):
        ax.axvline(0, color="k", lw=0.8)
        for group, label, a, b in rows:
            p = a[["seed", m, "early_best"]].merge(b[["seed", m, "early_best"]], on="seed", suffixes=("_a", "_b"))
            if p.empty:
                continue
            d = (p[f"{m}_b"] - p[f"{m}_a"]).to_numpy()
            band = noise_band(pd.concat([a, b]), {*a.key, *b.key})[m]
            inside = np.abs(d) < band if np.isfinite(band) else np.zeros(len(d), bool)
            c = np.where((p.early_best_a | p.early_best_b).to_numpy() & (m == "int8"), EARLY_GRAY, BLUE)
            yy = np.full(len(d), -y[(group, label)]) + np.linspace(-0.12, 0.12, len(d)) * (len(d) > 1)
            ax.scatter(d[~inside], yy[~inside], s=22, color=list(c[~inside]), zorder=3)
            ax.scatter(d[inside], yy[inside], s=18, facecolors="none", edgecolors=list(c[inside]), lw=1, zorder=3)
            ax.scatter(d.mean(), -y[(group, label)], marker="D", s=55, color=RED, zorder=4)
        ax.set_title(f"Top-1 {m.upper()}: B − A (pp)", fontsize=10)
        ax.set_xlabel("← A melhor   ·   B melhor →", fontsize=9)
    axes[0].set_yticks([-i for i in range(len(ticks))])
    axes[0].set_yticklabels([text for text, _ in ticks], fontsize=9)
    for t, (_, bold) in zip(axes[0].get_yticklabels(), ticks):
        t.set_weight("bold" if bold else "normal")
    handles = [Line2D([], [], marker="o", color=BLUE, ls="", ms=5, label="uma seed: B − A com a mesma seed nos dois lados"),
               Line2D([], [], marker="D", color=RED, ls="", ms=7, label="média das seeds"),
               Line2D([], [], marker="o", ls="", ms=5, mfc="none", mec=BLUE, label="vazado = dentro do ruído entre seeds das duas redes")]
    if any(a.early_best.any() or b.early_best.any() for *_, a, b in rows):
        handles.append(Line2D([], [], marker="o", color=EARLY_GRAY, ls="", ms=5, label="cinza (INT8) = " + EARLY_LABEL))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)
    fig.suptitle("O custo do kernel pequeno é o mesmo em todo lugar? Duas redes iguais exceto pelo kernel, em cada cenário\n"
                 "datasets, lr e famílias diferentes; e o 3×3 → 2×2 com o custo de cálculo igualado · top-1 no test set de cada dataset",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    savefig(fig, "04_kernel_effect_everywhere.png", figs)


def fig_transfer(p11, p12, figs):
    """Figure 05: does a net's place on Tiny ImageNet carry to CIFAR-100? Seed means (bars = 1 SD where replicated), one
    panel per family -- pooled, the families' ~8 pp gap alone would make any correlation look high."""
    from scipy.stats import kendalltau

    c = p12[p12.exp.str.startswith("cifar100")]
    panels = [("AlexNet (layout 64px, GAP + BN)", "alexnet", ["k11-5-3", "k3x3", "k2x2", "kalt3-2"], AK, CAK),
              ("VGG16 (GAP + BN)", "vgg16", ["k3x3", "k2x2", "kalt3-2"], VK, CVK)]
    if c.empty:
        print("05: no CIFAR-100 run yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (title, family, kernels, tiny, cifar) in zip(axes, panels):
        pts = []
        for k in kernels:
            t, s = p11[p11.key == tiny(k)].fp32, c[c.key == cifar(k)].fp32
            if len(t) and len(s):
                pts.append((t.mean(), s.mean()))
                ax.errorbar(t.mean(), s.mean(), xerr=t.std() if len(t) > 1 else None, yerr=s.std() if len(s) > 1 else None,
                            marker=FMARK[family], ms=9, color=KCOLOR[k], ecolor=KCOLOR[k], elinewidth=1, capsize=3, ls="")
                ax.annotate(f"{KLABEL[k]} ({len(t)} / {len(s)})", (t.mean(), s.mean()), xytext=(7, 4), textcoords="offset points",
                            fontsize=9, color=KCOLOR[k])
        tau = kendalltau(*zip(*pts)).statistic if len(pts) > 2 else np.nan
        ax.set_title(f"{title}\nτ de Kendall = {tau:+.2f} ({len(pts)} redes)" if np.isfinite(tau) else title, fontsize=10)
        ax.set_xlabel("Top-1 FP32 no Tiny ImageNet (%)")
        ax.margins(x=0.35, y=0.25)  # room for the names right of the points
    axes[0].set_ylabel("Top-1 FP32 no CIFAR-100 (%)")
    fig.suptitle("A ordem dos kernels no Tiny ImageNet se repete no CIFAR-100? Média das seeds; barras = 1 desvio-padrão entre seeds\n"
                 "τ = +1: mesma ordem nos dois datasets; entre parênteses, quantas seeds no Tiny ImageNet / no CIFAR-100", fontsize=11, y=1.04)
    fig.tight_layout()
    savefig(fig, "05_dataset_transfer.png", figs)


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
    fig_everywhere(p11, p12, FIGS)
    fig_transfer(p11, p12, FIGS)


if __name__ == "__main__":
    main()
