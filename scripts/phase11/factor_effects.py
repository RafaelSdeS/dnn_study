"""Phase 11 — isolate one factor at a time: paired contrasts (A -> B, everything else identical).

Each contrast compares two runs that differ in exactly one factor (kernel, head, conv1 stride, pooling, Dropout,
pretraining, BN, compensation block). Where seeds 42/43/44 exist for both runs the delta is per seed (paired);
otherwise it is a single seed-42 pair. The grey band is the noise floor: 2*sqrt(2)*pooled seed-to-seed SD of
the 4 replicated models (a single-seed delta inside it is not distinguishable from re-drawing the seed).
"layout original" / "layout 64px" are defined in scripts/phase11/analyze_geometry.py's docstring.

    05..11_*.png                  one figure per factor (absolute top-1 of A and B, baseline line)
    12_factor_effects_forest.png  every contrast's delta at once
    13_interactions.png           conv1 stride x pooling, kernel x head
    14_factorial_matched_pairs.png every matched pair of the factorial cells (CELL_FACTORS)

    python -m scripts.phase11.factor_effects
"""
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ml.plotting import AMBER, BLUE, GREEN, NEUTRAL, RED, apply_report_style
from ml.reporting import holm, mcnemar_p, wilson_ci
from scripts.phase11.analyze_geometry import (LAYOUTS, OLD_INT8, TABLES, int8_bar, load, noise_band, precision_handles,
                                              reference_lines, savefig)

PURPLE = "#7b3fa0"
FACTOR_COLOR = {"Kernel": GREEN, "Cabeça (FC → GAP)": RED, "Stride da conv1 (4 → 2)": BLUE, "Pooling": AMBER,
                "Dropout (0 → 0,5)": "#8d6e63", "Pré-treino ImageNet (não → sim)": PURPLE, "BatchNorm (sem → com)": "#0d8b8b",
                "Bloco de compensação (sobre 3×3 + BN + GAP)": "#c2185b"}
FACTOR_FIG = {"Cabeça (FC → GAP)": "05_head_fc_vs_gap.png", "Stride da conv1 (4 → 2)": "06_conv1_stride.png",
              "Pooling": "07_pooling.png", "Dropout (0 → 0,5)": "08_dropout.png",
              "Pré-treino ImageNet (não → sim)": "09_pretraining.png", "BatchNorm (sem → com)": "10_batchnorm.png",
              "Bloco de compensação (sobre 3×3 + BN + GAP)": "11_compensation_block.png"}  # Kernel: figures 02-04

# (factor, label, A, B, note). † in label = a second variable changes too (see note). "64px"/"original" = layout.
C = [
    ("Kernel", "64px · FC: 11-5-3-3-3 → 3×3", "alexnet_adapted_orig_fc", "alexnet_3x3_fc", ""),
    ("Kernel", "64px · GAP: 11-5-3-3-3 → 3×3", "alexnet_adapted_orig_gap", "alexnet_3x3_gap", ""),
    ("Kernel", "64px · FC: 3×3 → 2×2", "alexnet_3x3_fc", "alexnet_adapted_2x2_fc", ""),
    ("Kernel", "64px · GAP: 3×3 → 2×2", "alexnet_3x3_gap", "alexnet_adapted_2x2_gap", ""),
    ("Kernel", "original sem Dropout · FC: 11-5-3-3-3 → 3×3", "alexnet_geo_s4_p3_fc", "alexnet_geo_s4_p3_fc_k3", ""),
    ("Cabeça (FC → GAP)", "64px · 3×3", "alexnet_3x3_fc", "alexnet_3x3_gap", ""),
    ("Cabeça (FC → GAP)", "64px · 11-5-3-3-3", "alexnet_adapted_orig_fc", "alexnet_adapted_orig_gap", ""),
    ("Cabeça (FC → GAP)", "64px · 2×2", "alexnet_adapted_2x2_fc", "alexnet_adapted_2x2_gap", ""),
    ("Cabeça (FC → GAP)", "64px · misto 3-2-3-2-3 · sem BN", "alexnet_mixed_fc", "alexnet_mixed", ""),
    ("Cabeça (FC → GAP)", "64px · misto 3-2-3-2-3 · com BN", "alexnet_mixed_fc_bn", "alexnet_mixed_bn", ""),
    ("Cabeça (FC → GAP)", "64px · 3×3 empilhado (2 convs/estágio) · com BN", "alexnet_stacked", "alexnet_stacked_gap", ""),
    ("Cabeça (FC → GAP)", "3×3 estreito, conv1 stride 1", "alexnet_smallkernel_fc", "alexnet_smallkernel",
     "AlexNetSmallKernel (canais 64→256)"),
    ("Cabeça (FC → GAP)", "original sem Dropout · 11-5-3-3-3", "alexnet_geo_s4_p3_fc", "alexnet_geo_s4_p3_gap", ""),
    ("Cabeça (FC → GAP)", "original · misto 3-3-3-2-2†", "alexnet_tv_mixed_early3", "alexnet_tv_mixed_early3_gap",
     "a FC do torchvision tem Dropout, que sai junto"),
    ("Cabeça (FC → GAP)", "original · misto 2-3-2-3-2†", "alexnet_tv_mixed_alt", "alexnet_tv_mixed_alt_gap",
     "a FC do torchvision tem Dropout, que sai junto"),
    ("Cabeça (FC → GAP)", "original · misto 2-2-2-3-3†", "alexnet_tv_mixed_early2", "alexnet_tv_mixed_early2_gap",
     "a FC do torchvision tem Dropout, que sai junto"),
    ("Stride da conv1 (4 → 2)", "com 3 max-pools 3×3/2 (os do layout original)", "alexnet_geo_s4_p3_fc", "alexnet_geo_s2_p3_fc", ""),
    ("Stride da conv1 (4 → 2)", "com 2 max-pools 2×2 (os do layout 64px)", "alexnet_geo_s4_p2_fc", "alexnet_adapted_orig_fc", ""),
    ("Pooling", "conv1 stride 4: 3 pools 3×3 → 2 pools 2×2", "alexnet_geo_s4_p3_fc", "alexnet_geo_s4_p2_fc", ""),
    ("Pooling", "conv1 stride 2: 3 pools 3×3 → 2 pools 2×2", "alexnet_geo_s2_p3_fc", "alexnet_adapted_orig_fc", ""),
    ("Pooling", "conv1 stride 2, 3 pools: janela 3×3 → 2×2", "alexnet_geo_s2_p3_fc", "alexnet_geo_s2_pk2n3_fc", ""),
    ("Pooling", "conv1 stride 2, janela 2×2: 3 → 2 pools", "alexnet_geo_s2_pk2n3_fc", "alexnet_adapted_orig_fc", ""),
    ("Pooling", "conv1 stride 2, janela 3×3: 3 → 2 pools", "alexnet_geo_s2_p3_fc", "alexnet_geo_s2_pk3n2_fc", ""),
    ("Dropout (0 → 0,5)", "64px · 11-5-3-3-3 · FC", "alexnet_adapted_orig_fc", "alexnet_geo_s2_p2_drop_fc", ""),
    ("Dropout (0 → 0,5)", "original · 11-5-3-3-3 · FC†", "alexnet_geo_s4_p3_fc", "alexnet_tv_scratch",
     "B é a AlexNet original do torchvision (com Dropout)"),
    ("Pré-treino ImageNet (não → sim)", "original · 11-5-3-3-3 · FC + Dropout†", "alexnet_tv_scratch", "alexnet_tv",
     "A é a AlexNet original do torchvision (com Dropout)"),
    ("Pré-treino ImageNet (não → sim)", "64px · 11-5-3-3-3 · FC†", "alexnet_adapted_orig_fc", "alexnet_adapted_orig_fc_pt",
     ""),
    ("BatchNorm (sem → com)", "64px · 3×3 · GAP", "alexnet_3x3_gap", "alexnet_3x3_gap_bn", ""),
    ("BatchNorm (sem → com)", "64px · misto 3-2-3-2-3 · GAP", "alexnet_mixed", "alexnet_mixed_bn", ""),
    ("BatchNorm (sem → com)", "64px · misto 3-2-3-2-3 · FC", "alexnet_mixed_fc", "alexnet_mixed_fc_bn", ""),
    ("BatchNorm (sem → com)", "64px · 3×3 empilhado (2 convs/estágio) · GAP", "alexnet_stacked_gap_nobn", "alexnet_stacked_gap", ""),
    ("Bloco de compensação (sobre 3×3 + BN + GAP)", "64px · 3×3 + BN + GAP → Bottleneck", "alexnet_3x3_gap_bn",
     "alexnet_bottleneck", ""),
    ("Bloco de compensação (sobre 3×3 + BN + GAP)", "64px · 3×3 + BN + GAP → Fire†", "alexnet_3x3_gap_bn", "alexnet_fire",
     "o Fire usa conv1 stride 1 (mapas 16×16)"),
]


def _test_hits(results, key, stage):
    """Per-image top-1 hits on the held-out test set, or None before the run's current-protocol rerun wrote them."""
    p = results / f"{key}_{stage}_test_logits.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["logits"].argmax(1) == z["labels"]


def _pair_stats(idx, a, b):
    """Seed-42 McNemar p-value (Dietterich 1998) of A vs B on the same test images, FP32 and INT8, plus each side's
    95% Wilson interval -- the test-set sampling error; the seed band covers the training noise."""
    out = {}
    for stage in ("fp32", "int8"):
        ha, hb = (_test_hits(idx.loc[(k, 42), "results"], k, stage) for k in (a, b))
        if ha is None or hb is None:
            continue
        out[f"p_{stage}"] = mcnemar_p(ha, hb)
        for side, h in (("A", ha), ("B", hb)):
            lo, hi = wilson_ci(int(h.sum()), len(h))
            out[f"{stage}_{side}_ci95"] = f"{100 * lo:.1f}–{100 * hi:.1f}"
    return out


def contrasts(df):
    # the default-init alexnet_tv_3x3 re-QAT (phase_11_reuse_old_init) shares key+seed with the he_init run
    idx = df[df.exp != "reuse_old_init"].set_index(["key", "seed"])
    assert idx.index.is_unique, idx.index[idx.index.duplicated()]
    rows = []
    for factor, label, a, b, note in C:
        seeds = sorted({s for k, s in idx.index if k == a} & {s for k, s in idx.index if k == b})
        assert seeds, (a, b)
        va, vb = ({c: np.array([idx.loc[(k, s), c] for s in seeds]) for c in ["fp32", "int8", "int8_raw", "macs_m", "params_m"]}
                  for k in (a, b))
        d = {c: vb[c] - va[c] for c in va}
        rows.append(dict(
            factor=factor, contrast=label, A=a, B=b, n_seeds=len(seeds), **_pair_stats(idx, a, b),
            d_fp32=d["fp32"].mean(), d_fp32_seeds=" / ".join(f"{x:+.2f}" for x in d["fp32"]),
            d_int8=d["int8"].mean(), d_int8_seeds=" / ".join(f"{x:+.2f}" for x in d["int8"]), d_int8_raw=d["int8_raw"].mean(),
            **{f"{c}_{side}": v[c].mean() for side, v in [("A", va), ("B", vb)] for c in ["fp32", "int8", "int8_raw"]},
            macs_change_pct=100 * (vb["macs_m"][0] / va["macs_m"][0] - 1),
            params_change_pct=100 * (vb["params_m"][0] / va["params_m"][0] - 1), note=note,
            pts_fp32=list(d["fp32"]), pts_int8=list(d["int8"]), pts_int8_raw=list(d["int8_raw"])))
    t = pd.DataFrame(rows)
    for stage in ("fp32", "int8"):  # ~35 tests in one table: family-wise correction (Holm 1979)
        if f"p_{stage}" in t:
            t[f"p_{stage}_holm"] = holm(t[f"p_{stage}"].tolist())
    return t


def fig_factor(t, df, factor, per_row=6):
    """Absolute top-1 of A and B for every contrast of one factor: A grey, B in the factor's colour, + baseline."""
    g, color = t[t.factor == factor], FACTOR_COLOR[factor]
    chunks = [g.iloc[k:k + per_row] for k in range(0, len(g), per_row)]
    width = min(len(g), per_row)
    fig, axes = plt.subplots(len(chunks), 1, figsize=(max(11.0, 2.5 * width + 1.5), 5.2 * len(chunks) + 1.2), squeeze=False,
                             sharey=True, layout="constrained")
    w = 0.19
    for ax, chunk in zip(axes[:, 0], chunks):
        for i, r in enumerate(chunk.itertuples()):
            for j, (side, c) in enumerate([("A", NEUTRAL), ("B", color)]):
                fp32 = getattr(r, f"fp32_{side}")
                ax.bar(i + (2 * j - 1.5) * w, fp32, w, color=c, edgecolor="white")
                int8_bar(ax, i + (2 * j - 0.5) * w, getattr(r, f"int8_{side}"), getattr(r, f"int8_raw_{side}"), w, c)
                ax.text(i + (2 * j - 1.5) * w, fp32 + 0.4, f"{fp32:.1f}", ha="center", va="bottom", fontsize=7.5, zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
        ax.set_xticks(range(len(chunk)))
        ax.set_xticklabels([textwrap.fill(r.contrast, 22) + f"\nparâm. {r.params_change_pct:+.0f}% · MACs {r.macs_change_pct:+.0f}%"
                            + (f"\nmédia de {r.n_seeds} seeds" if r.n_seeds > 1 else "") for r in chunk.itertuples()], fontsize=8.5)
        ax.set_xlim(-0.6, width - 0.4)  # same bar width in every row
        ax.set_ylabel("Top-1 (%)")
        ax.margins(y=0.08)
        ref = reference_lines(ax, df)
    notes = {}
    for r in g.itertuples():
        if r.note:
            notes.setdefault(r.note, []).append(r.contrast)
    fig.suptitle(f"{factor}: cada grupo compara duas redes idênticas exceto por este fator (seed 42)\n{LAYOUTS}"
                 + "".join(f"\nnota — {'; '.join(cs)}: {note}" for note, cs in notes.items()), fontsize=10)
    fig.legend(handles=[Patch(color=NEUTRAL, label="A = antes (1º nível do fator)"), Patch(color=color, label="B = depois (2º nível)")]
               + precision_handles() + ref, loc="outside lower center", ncol=2, fontsize=8.5)
    savefig(fig, FACTOR_FIG[factor])


def fig_forest(t, band):
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.33 * len(t) + 4), sharey=True)
    ypos, headers, cur = [], [], 0.0  # one blank header row above each factor group, rows run top -> bottom
    for f, g in t.groupby("factor", sort=False):
        headers.append((f, cur)); cur += 1
        for _ in range(len(g)):
            ypos.append(cur); cur += 1
        cur += 0.4
    y = [cur - p for p in ypos]
    for ax, col, title in [(axes[0], "fp32", "Δ top-1 FP32 (pp)"), (axes[1], "int8", "Δ top-1 INT8 real (pp)")]:
        if np.isfinite(band[col]):
            ax.axvspan(-band[col], band[col], color="#d0d0d0", alpha=0.6, zorder=0)
        ax.axvline(0, color="k", lw=0.8)
        for yi, r in zip(y, t.itertuples()):
            c, d, pts = FACTOR_COLOR[r.factor], getattr(r, f"d_{col}"), getattr(r, f"pts_{col}")
            old = col == "int8" and np.isnan(d)
            if old:  # only the pre-fix INT8 exists: hollow + faded, never drawn like a valid delta
                d, pts = r.d_int8_raw, r.pts_int8_raw
            style = dict(facecolors="none", edgecolors=c, alpha=0.5) if old else dict(color=c, edgecolors="white")
            if len(pts) > 1:
                ax.scatter(pts, [yi] * len(pts), s=16, zorder=3, **(style if old else dict(color=c, alpha=0.6)))
            ax.scatter(d, yi, s=90, zorder=4, **style)
            ax.annotate(f"{d:+.1f}", (max(pts + [d]) if d >= 0 else min(pts + [d]), yi), xytext=(8 if d >= 0 else -8, 0),
                        textcoords="offset points", va="center", ha="left" if d >= 0 else "right", fontsize=7.5, color=c,
                        alpha=0.5 if old else 1)
        for f, hy in headers:
            ax.text(0.005, cur - hy, f, transform=ax.get_yaxis_transform(), fontsize=9.5, fontweight="bold", color=FACTOR_COLOR[f], va="center")
        ax.set_xlabel(title); ax.grid(axis="y", alpha=0); ax.set_ylim(-0.6, cur); ax.margins(x=0.12)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r.contrast}  (FP32 {r.fp32_A:.1f} → {r.fp32_B:.1f}%, n={r.n_seeds})" for r in t.itertuples()],
                            fontsize=8.5)
    fig.legend(handles=[Line2D([], [], marker="o", color=NEUTRAL, ls="", ms=9, label="Δ médio do par (cor = fator)"),
                        Line2D([], [], marker="o", color=NEUTRAL, ls="", ms=4, alpha=0.6, label="Δ de uma seed (42/43/44)"),
                        Line2D([], [], marker="o", mfc="none", mec=NEUTRAL, ls="", ms=9, label=f"Δ calculado com o {OLD_INT8}"),
                        Patch(color="#d0d0d0", label=f"ruído entre seeds (±{band['fp32']:.1f}pp FP32"
                              + (f", ±{band['int8']:.1f}pp INT8)" if np.isfinite(band["int8"]) else "; INT8 após o rerun)"))],
               loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize=9)
    fig.suptitle("Efeito de cada fator isolado: cada linha compara duas redes idênticas exceto pelo fator do grupo (A → B)\n"
                 "† = uma segunda variável muda junto (nota nas figuras 05–11)   ·   64px / original = layout:\n" + LAYOUTS, fontsize=10.5)
    fig.tight_layout(rect=(0.0, 0, 1, 0.95))
    savefig(fig, "12_factor_effects_forest.png")


def fig_interaction(df):
    v = df[(df.seed == 42) & (df.exp != "reuse_old_init")].set_index("key").fp32
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    for lab, ys, c in [("3 max-pools 3×3/2 (os do layout original)", [v.alexnet_geo_s4_p3_fc, v.alexnet_geo_s2_p3_fc], AMBER),
                       ("2 max-pools 2×2 (os do layout 64px)", [v.alexnet_geo_s4_p2_fc, v.alexnet_adapted_orig_fc], BLUE)]:
        ax.plot([0, 1], ys, "-o", color=c, lw=2.2, ms=8, label=lab)
        for x, yv in zip([0, 1], ys):
            ax.annotate(f"{yv:.1f}", (x, yv), xytext=(-30 if x == 0 else 6, (6 if c == BLUE else -14) if x == 0 else -12), textcoords="offset points", fontsize=9, color=c)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["conv1 stride 4", "conv1 stride 2"]); ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("Top-1 FP32 (%)")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + reference_lines(ax, df, int8=False), fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Interação stride da conv1 × pooling\n(cabeça FC sem Dropout, kernels 11-5-3-3-3, seed 42)\n"
                 "linhas não paralelas = os efeitos não se somam", fontsize=10)
    ax = axes[1]
    for lab, ks, c in [("cabeça FC", ["alexnet_3x3_fc", "alexnet_adapted_orig_fc", "alexnet_adapted_2x2_fc"], "#555555"),
                       ("cabeça GAP", ["alexnet_3x3_gap", "alexnet_adapted_orig_gap", "alexnet_adapted_2x2_gap"], RED)]:
        ax.plot(range(3), [v[k] for k in ks], "-o", color=c, lw=2.2, ms=8, label=lab)
        for i, k in enumerate(ks):
            ax.annotate(f"{v[k]:.1f}", (i, v[k]), xytext=(6, -12), textcoords="offset points", fontsize=9, color=c)
    ax.set_xticks(range(3)); ax.set_xticklabels(["3×3", "11-5-3-3-3", "2×2"]); ax.set_xlim(-0.3, 2.3)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + reference_lines(ax, df, int8=False), fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Interação kernel × cabeça\n(layout 64px, sem BN, seed 42)\n"
                 "linhas ≈ paralelas: kernel e cabeça são quase aditivos", fontsize=10)
    fig.tight_layout()
    savefig(fig, "13_interactions.png")


# ── Factorial cells (ml/model_registrations.py:CELL_FACTORS): every matched pair, not one hand-picked pair per factor ──
FACTORS = ["family", "kernels", "stride", "pool_kernel", "pool_count", "head", "bn", "dropout", "pretrained"]  # map_side
# follows from kernels + geometry, so it is not a pairing key
METRICS = ["fp32", "int8", "dqat"]  # dqat = INT8 - FP32, the report's ΔQAT (negative = loses under INT8)
# (factor, level a, level b, label): Δ = b - a over every pair of cells equal in all other factors
PAIRS = [("kernels", "k11-5-3", "k3x3", "Kernel 11-5-3-3-3 → 3×3"), ("kernels", "k3x3", "k2x2", "Kernel 3×3 → 2×2"),
         ("kernels", "k11-5-3", "k2x2", "Kernel 11-5-3-3-3 → 2×2"), ("kernels", "k3x3", "kalt3-2", "Kernel 3×3 → alternado 3-2"),
         ("kernels", "k3x3", "k3x3stacked", "3×3 → dois 3×3 empilhados por estágio"),
         ("head", "fc", "gap", "Cabeça FC → GAP"), ("bn", False, True, "BatchNorm não → sim"),
         ("stride", 4, 2, "Stride da conv1 4 → 2"), ("pool_kernel", 3, 2, "Janela do max-pool 3×3 → 2×2"),
         ("pool_count", 3, 2, "Nº de max-pools 3 → 2"), ("dropout", False, True, "Dropout 0 → 0,5 (só cabeça FC)"),
         ("pretrained", False, True, "Pré-treino ImageNet")]


def factorial_cells(df):
    """Seed-42 table, one row per trained cell of CELL_FACTORS (AlexNet and VGG grids + stacked/narrow)."""
    from ml.model_registrations import CELL_FACTORS

    runs = df[(df.seed == 42) & df.key.isin(CELL_FACTORS)].set_index("key")
    assert runs.index.is_unique, runs.index[runs.index.duplicated()]
    cells = pd.DataFrame([{"cell": k, **CELL_FACTORS[k], **r[["fp32", "fp32_val", "int8", "int8_raw", "params_m", "macs_m"]].to_dict()}
                          for k, r in runs.iterrows()])
    cells["dqat"] = cells.int8 - cells.fp32
    cells["dqat_raw"] = cells.int8_raw - cells.fp32_val
    return cells


def matched_pairs(cells):
    out = []
    for f, a, b, label in PAIRS:
        others = [x for x in FACTORS if x != f]
        j = cells[cells[f] == a].set_index(others).join(cells[cells[f] == b].set_index(others), lsuffix="_a", rsuffix="_b", how="inner")
        for _, r in j.iterrows():
            out.append(dict(factor=f, contrast=label, cell_a=r.cell_a, cell_b=r.cell_b,
                            **{m: r[f"{m}_b"] - r[f"{m}_a"] for m in METRICS + ["int8_raw", "dqat_raw"]},
                            macs_pct=100 * (r.macs_m_b / r.macs_m_a - 1), params_pct=100 * (r.params_m_b / r.params_m_a - 1)))
    return pd.DataFrame(out)


def bootstrap_median_ci(v, n_boot=10_000):
    """95% percentile-bootstrap interval of the median (Efron & Tibshirani 1993), fixed RNG so reruns match. Each
    matched pair holds every other factor fixed, so the spread over pairs is the effect's spread over the rest of the
    design -- the replication an unreplicated factorial has."""
    if len(v) < 2:
        return np.nan, np.nan
    meds = np.median(np.random.default_rng(0).choice(np.asarray(v), (n_boot, len(v))), axis=1)
    return tuple(np.percentile(meds, [2.5, 97.5]))


def summarize_pairs(pairs):
    """Per contrast: n pairs, median/IQR of each Δ with a bootstrap 95% CI of the median, and in how many pairs b beat a
    (the 'in k of n configurations' line)."""
    rows = []
    for (f, label), g in pairs.groupby(["factor", "contrast"], sort=False):
        row = dict(factor=f, contrast=label, n_pairs=len(g), macs_pct=g.macs_pct.median(), params_pct=g.params_pct.median())
        for m in METRICS:
            v = g[m].dropna()
            lo, hi = bootstrap_median_ci(v)
            row.update({f"{m}_n": len(v), f"{m}_median": v.median(), f"{m}_q25": v.quantile(0.25), f"{m}_q75": v.quantile(0.75),
                        f"{m}_ci_lo": lo, f"{m}_ci_hi": hi, f"{m}_n_positive": int((v > 0).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def fig_matched_pairs(pairs, summary, n_cells):
    titles = {"fp32": "Δ top-1 FP32 (pp)", "int8": "Δ top-1 INT8 (pp)", "dqat": "Δ ΔQAT (pp; >0 = mais robusto)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.55 * len(summary) + 3), sharey=True)
    y = {c: i for i, c in enumerate(summary.contrast[::-1])}
    for ax, m in zip(axes, METRICS):
        ax.axvline(0, color="k", lw=0.8)
        drawn = False
        for s in summary.itertuples():
            g = pairs[pairs.contrast == s.contrast]
            v, old = g[m].dropna(), False
            if v.empty and m != "fp32":  # only pre-fix INT8 so far: hollow + faded
                v, old = g[f"{m}_raw"].dropna(), True
            if v.empty:
                continue
            drawn = True
            yy = y[s.contrast] + np.random.default_rng(0).uniform(-0.18, 0.18, len(v))
            ax.scatter(v, yy, s=10 if not old else 16, alpha=0.5, **(dict(facecolors="none", edgecolors=BLUE) if old else dict(color=BLUE)))
            ax.scatter(v.median(), y[s.contrast], marker="D", s=50, zorder=3,
                       **(dict(facecolors="none", edgecolors=RED, alpha=0.6) if old else dict(color=RED)))
            if not old:
                ax.hlines(y[s.contrast], *bootstrap_median_ci(v), color=RED, lw=2, zorder=2)
            ax.text(1.01, y[s.contrast], f"{int((v > 0).sum())}/{len(v)}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=8, alpha=0.5 if old else 1)
        if not drawn:
            ax.text(0.5, 0.5, "pendente (rerun na fila)", transform=ax.transAxes, ha="center", color=NEUTRAL)
        ax.set_title(titles[m], fontsize=10)
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels(list(y.keys()), fontsize=9)
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", ms=4, alpha=0.5, label="um par de células iguais em todo o resto"),
                        Line2D([], [], marker="D", color=RED, ls="-", ms=7, label="mediana dos pares (barra: IC 95% bootstrap)"),
                        Line2D([], [], marker="o", mfc="none", mec=BLUE, ls="", ms=5, label=f"par com {OLD_INT8}"),
                        Line2D([], [], ls="", label="k/n à direita = pares em que o 2º nível ganha")],
               loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize=9)
    fig.suptitle(f"Células do desenho da Fase 11 (seed 42; {n_cells} treinadas até agora): cada ponto compara "
                 "duas células que só diferem no fator da linha", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "14_factorial_matched_pairs.png")


def main():
    apply_report_style()
    df = load()
    t = contrasts(df)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.drop(columns=[c for c in t if c.startswith("pts_")]).to_csv(TABLES / "factor_effects.csv", index=False)
    band = noise_band(df)
    print("noise band (pp):", {k: round(v, 2) for k, v in band.items()})
    for factor in FACTOR_FIG:
        fig_factor(t, df, factor)
    fig_forest(t, band)
    fig_interaction(df)
    pd.set_option("display.width", 250, "display.max_colwidth", 45)
    print(t[["factor", "contrast", "n_seeds", "d_fp32", "d_int8", "macs_change_pct", "params_change_pct"]].round(2).to_string(index=False))

    cells = factorial_cells(df)
    pairs = matched_pairs(cells)
    summary = summarize_pairs(pairs)
    cells.to_csv(TABLES / "factorial_cells.csv", index=False)
    pairs.to_csv(TABLES / "factorial_pairs.csv", index=False)
    summary.to_csv(TABLES / "factorial_pair_summary.csv", index=False)
    fig_matched_pairs(pairs, summary, len(cells))
    print(f"\n{len(cells)} factorial cells trained (seed 42)")
    print(summary[["contrast", "n_pairs", "fp32_median", "fp32_n_positive", "int8_n", "int8_median", "dqat_median", "dqat_n_positive",
                   "macs_pct"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
