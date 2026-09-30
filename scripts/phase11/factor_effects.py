"""Phase 11 — isolate one factor at a time: paired contrasts (A -> B, everything else identical).

Each contrast compares two runs that differ in exactly one factor (kernel, head, stem stride, pooling, Dropout,
pretraining, BN, compensation block). Where seeds 42/43/44 exist for both runs the delta is per seed (paired);
otherwise it is a single seed-42 pair. The grey band is the noise floor: 2*sqrt(2)*pooled seed-to-seed SD of
the 4 replicated models (a single-seed delta inside it is not distinguishable from re-drawing the seed).

    python -m scripts.phase11.factor_effects
"""
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml.plotting import AMBER, BLUE, GREEN, RED, apply_report_style
from scripts.phase11.analyze_geometry import FIGS, TABLES, load, savefig

PURPLE = "#7b3fa0"
FACTOR_COLOR = {"Kernel": GREEN, "Cabeça (FC→GAP)": RED, "Stride do stem (4→2)": BLUE, "Pooling": AMBER,
                "Dropout": "#4d4d4d", "Pré-treino": PURPLE, "BatchNorm": "#0d8b8b", "Bloco (sobre BN+GAP)": "#c2185b"}

# (factor, label, A, B, note). † in note = a second variable changes too (see note).
C = [
    ("Kernel", "FC adaptado: 11-5-3-3-3 → 3x3", "alexnet_adapted_orig_fc", "alexnet_3x3_fc", ""),
    ("Kernel", "GAP adaptado: 11-5-3-3-3 → 3x3", "alexnet_adapted_orig_gap", "alexnet_3x3_gap", ""),
    ("Kernel", "FC adaptado: 3x3 → 2x2", "alexnet_3x3_fc", "alexnet_adapted_2x2_fc", ""),
    ("Kernel", "GAP adaptado: 3x3 → 2x2", "alexnet_3x3_gap", "alexnet_adapted_2x2_gap", ""),
    ("Kernel", "FC torchvision: 11-5-3-3-3 → 3x3", "alexnet_geo_s4_p3_fc", "alexnet_geo_s4_p3_fc_k3", ""),
    ("Cabeça (FC→GAP)", "adaptado 3x3", "alexnet_3x3_fc", "alexnet_3x3_gap", ""),
    ("Cabeça (FC→GAP)", "adaptado 11-5-3-3-3", "alexnet_adapted_orig_fc", "alexnet_adapted_orig_gap", ""),
    ("Cabeça (FC→GAP)", "adaptado 2x2", "alexnet_adapted_2x2_fc", "alexnet_adapted_2x2_gap", ""),
    ("Cabeça (FC→GAP)", "torchvision 11-5-3-3-3", "alexnet_geo_s4_p3_fc", "alexnet_geo_s4_p3_gap", ""),
    ("Cabeça (FC→GAP)", "misto 3-2-3-2-3, sem BN", "alexnet_mixed_fc", "alexnet_mixed", "he_init"),
    ("Cabeça (FC→GAP)", "misto 3-2-3-2-3, com BN", "alexnet_mixed_fc_bn", "alexnet_mixed_bn", "he_init"),
    ("Stride do stem (4→2)", "com 3 pools 3×3/2", "alexnet_geo_s4_p3_fc", "alexnet_geo_s2_p3_fc", ""),
    ("Stride do stem (4→2)", "com 2 pools 2×2", "alexnet_geo_s4_p2_fc", "alexnet_adapted_orig_fc", ""),
    ("Pooling", "stride 4: 3 pools 3×3 → 2 pools 2×2", "alexnet_geo_s4_p3_fc", "alexnet_geo_s4_p2_fc", ""),
    ("Pooling", "stride 2: 3 pools 3×3 → 2 pools 2×2", "alexnet_geo_s2_p3_fc", "alexnet_adapted_orig_fc", ""),
    ("Pooling", "stride 2, 3 pools: kernel 3 → 2", "alexnet_geo_s2_p3_fc", "alexnet_geo_s2_pk2n3_fc", ""),
    ("Pooling", "stride 2, kernel 2: 3 → 2 pools", "alexnet_geo_s2_pk2n3_fc", "alexnet_adapted_orig_fc", ""),
    ("Pooling", "stride 2, kernel 3: 3 → 2 pools", "alexnet_geo_s2_p3_fc", "alexnet_geo_s2_pk3n2_fc", ""),
    ("Dropout", "adaptado (+2×0,5)", "alexnet_adapted_orig_fc", "alexnet_geo_s2_p2_drop_fc", ""),
    ("Dropout", "torchvision (+2×0,5)†", "alexnet_geo_s4_p3_fc", "alexnet_tv_scratch", "run antigo (job 821243), init padrão nos dois"),
    ("Pré-treino", "torchvision†", "alexnet_tv_scratch", "alexnet_tv", "scratch = run antigo"),
    ("Pré-treino", "adaptado†", "alexnet_adapted_orig_fc", "alexnet_adapted_orig_fc_pt", "melhor epoch=7, depois o QAT (lr 1e-5) sobe +5,7pp"),
    ("BatchNorm", "3x3 GAP adaptado", "alexnet_3x3_gap", "alexnet_3x3_gap_bn", ""),
    ("BatchNorm", "misto GAP", "alexnet_mixed", "alexnet_mixed_bn", "he_init"),
    ("BatchNorm", "misto FC", "alexnet_mixed_fc", "alexnet_mixed_fc_bn", "he_init"),
    ("BatchNorm", "stacked GAP", "alexnet_stacked_gap_nobn", "alexnet_stacked_gap", "he_init"),
    ("Bloco (sobre BN+GAP)", "3x3+BN → Bottleneck", "alexnet_3x3_gap_bn", "alexnet_bottleneck", ""),
    ("Bloco (sobre BN+GAP)", "3x3+BN → Fire†", "alexnet_3x3_gap_bn", "alexnet_fire", "stem stride 1 (mapas 16×16)"),
]


def contrasts(df):
    # the default-init alexnet_tv_3x3 re-QAT (phase_11_reuse_old_init) shares key+seed with the he_init run
    idx = df[df.exp != "reuse_old_init"].set_index(["key", "seed"])
    assert idx.index.is_unique, idx.index[idx.index.duplicated()]
    rows = []
    for factor, label, a, b, note in C:
        seeds = sorted({s for k, s in idx.index if k == a} & {s for k, s in idx.index if k == b})
        assert seeds, (a, b)
        d = {c: [idx.loc[(b, s), c] - idx.loc[(a, s), c] for s in seeds] for c in ["fp32", "int8", "macs_m", "params_m"]}
        ra, rb = idx.loc[(a, seeds[0])], idx.loc[(b, seeds[0])]
        rows.append(dict(
            factor=factor, contrast=label, A=a, B=b, n_seeds=len(seeds),
            d_fp32=np.mean(d["fp32"]), d_fp32_seeds=" / ".join(f"{x:+.2f}" for x in d["fp32"]),
            d_int8=np.mean(d["int8"]), d_int8_seeds=" / ".join(f"{x:+.2f}" for x in d["int8"]),
            fp32_A=ra.fp32, fp32_B=rb.fp32, macs_change_pct=100 * (rb.macs_m / ra.macs_m - 1),
            params_change_pct=100 * (rb.params_m / ra.params_m - 1), note=note,
            pts_fp32=d["fp32"], pts_int8=d["int8"]))
    return pd.DataFrame(rows)


def noise_band(df):
    """2*sqrt(2)*pooled SD across seeds, for the 4 models replicated at seeds 42-44."""
    rep = df[df.key.isin(["alexnet_3x3_fc", "alexnet_adapted_orig_fc", "alexnet_3x3_gap", "alexnet_adapted_orig_gap"])]
    g = rep.groupby("key")
    return {c: 2 * np.sqrt(2) * np.sqrt((g[c].var(ddof=1)).mean()) for c in ["fp32", "int8"]}


def fig_forest(t, band):
    fig, axes = plt.subplots(1, 2, figsize=(13, 10.5), sharey=True)
    ypos, headers, cur = [], [], 0.0  # one blank header row above each factor group, rows run top -> bottom
    for f, g in t.groupby("factor", sort=False):
        headers.append((f, cur)); cur += 1
        for _ in range(len(g)):
            ypos.append(cur); cur += 1
        cur += 0.4
    y = [cur - p for p in ypos]
    for ax, col, title in [(axes[0], "fp32", "Δ top-1 FP32 (pp)"), (axes[1], "int8", "Δ top-1 INT8 real (pp)")]:
        ax.axvspan(-band[col], band[col], color="#d0d0d0", alpha=0.6, zorder=0, label=f"ruído entre seeds (±{band[col]:.1f}pp)")
        ax.axvline(0, color="k", lw=0.8)
        for yi, r in zip(y, t.itertuples()):
            c, d, pts = FACTOR_COLOR[r.factor], getattr(r, f"d_{col}"), getattr(r, f"pts_{col}")
            if len(pts) > 1:
                ax.scatter(pts, [yi] * len(pts), color=c, s=16, alpha=0.6, zorder=3)
            ax.scatter(d, yi, color=c, s=90, edgecolors="white", zorder=4)
            ax.text(d + (0.7 if d >= 0 else -0.7), yi, f"{d:+.1f}", va="center", ha="left" if d >= 0 else "right", fontsize=7.5, color=c)
        for f, hy in headers:
            ax.text(0.005, cur - hy, f, transform=ax.get_yaxis_transform(), fontsize=9.5, fontweight="bold", color=FACTOR_COLOR[f], va="center")
        ax.set_xlabel(title); ax.grid(axis="y", alpha=0); ax.set_ylim(-0.6, cur)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r.contrast}  (n={r.n_seeds})" for r in t.itertuples()], fontsize=8.5)
    axes[0].set_xlim(-8, 14); axes[1].set_xlim(-10, 32)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Efeito de cada fator isolado (par A→B, só um fator muda) — ponto grande = média, pontos pequenos = seeds 42/43/44\n"
                 "† = uma segunda variável muda junto (ver tabela). INT8 dos modelos GAP sem BN inclui a perda de conversão (fig. 13)",
                 fontsize=11)
    fig.tight_layout(rect=(0.0, 0, 1, 0.96))
    savefig(fig, "15_factor_effects_forest.png")


def fig_interaction(df):
    v = df[df.seed == 42].set_index("key").fp32
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for lab, ys, c in [("3 pools 3×3/2 (torchvision)", [v.alexnet_geo_s4_p3_fc, v.alexnet_geo_s2_p3_fc], AMBER),
                       ("2 pools 2×2 (adaptado)", [v.alexnet_geo_s4_p2_fc, v.alexnet_adapted_orig_fc], BLUE)]:
        ax.plot([0, 1], ys, "-o", color=c, lw=2.2, ms=8, label=lab)
        for x, yv in zip([0, 1], ys):
            ax.annotate(f"{yv:.1f}", (x, yv), xytext=(-30 if x == 0 else 6, (6 if c == BLUE else -14) if x == 0 else -12), textcoords="offset points", fontsize=9, color=c)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["stem stride 4", "stem stride 2"]); ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("Top-1 FP32 (%)"); ax.legend(fontsize=9)
    ax.set_title("Interação stride × pooling (FC, sem Dropout, 11-5-3-3-3)\nlinhas não paralelas = os efeitos não se somam", fontsize=10)
    ax = axes[1]
    x = ["3x3", "11-5-3-3-3", "2x2"]
    for lab, ks, c in [("FC", ["alexnet_3x3_fc", "alexnet_adapted_orig_fc", "alexnet_adapted_2x2_fc"], "#555555"),
                       ("GAP", ["alexnet_3x3_gap", "alexnet_adapted_orig_gap", "alexnet_adapted_2x2_gap"], RED)]:
        ax.plot(range(3), [v[k] for k in ks], "-o", color=c, lw=2.2, ms=8, label=f"{lab} (seed 42)")
        for i, k in enumerate(ks):
            ax.annotate(f"{v[k]:.1f}", (i, v[k]), xytext=(6, -12), textcoords="offset points", fontsize=9, color=c)
    ax.set_xticks(range(3)); ax.set_xticklabels(x); ax.set_xlim(-0.3, 2.3); ax.legend(fontsize=9)
    ax.set_title("Interação kernel × cabeça (geometria adaptada, sem BN)\nlinhas ≈ paralelas: kernel e cabeça são quase aditivos", fontsize=10)
    savefig(fig, "16_interactions_stride_pool_and_kernel_head.png")


# ── Full factorial (alexnet_fx_*): every matched pair, not one hand-picked pair per factor ──
FX_RE = re.compile(r"alexnet_fx_(?P<kernel>orig|k3|k2|mix)_s(?P<stride>[24])_pk(?P<pool_k>[23])n(?P<pool_n>[23])_"
                   r"(?P<head>fc|gap)(?P<bn>_bn)?(?P<drop>_d)?(?P<pt>_pt)?$")
FACTORS = ["kernel", "stride", "pool_k", "pool_n", "head", "bn", "drop", "pt"]
METRICS = ["fp32", "int8", "dqat"]  # dqat = INT8 - FP32, the report's ΔQAT (negative = loses under INT8)
# (factor, level a, level b, label): Δ = b - a over every pair of cells equal in all other factors
PAIRS = [("kernel", "orig", "k3", "Kernel 11-5-3-3-3 → 3×3"), ("kernel", "k3", "k2", "Kernel 3×3 → 2×2"),
         ("kernel", "orig", "k2", "Kernel 11-5-3-3-3 → 2×2"), ("kernel", "k3", "mix", "Kernel 3×3 → misto 3-2-3-2-3"),
         ("head", "fc", "gap", "Cabeça FC → GAP"), ("bn", False, True, "BatchNorm não → sim"),
         ("stride", "4", "2", "Stride do stem 4 → 2"), ("pool_k", "3", "2", "Kernel do pool 3 → 2"),
         ("pool_n", "3", "2", "Nº de pools 3 → 2"), ("drop", False, True, "Dropout 0 → 0,5 (FC)"),
         ("pt", False, True, "Pré-treino ImageNet (11-5-3, FC, sem BN)")]


def factorial_cells(df):
    """Seed-42 factorial table, one row per trained cell; FX_EXISTING cells come from the run they alias."""
    from ml.model_registrations import FX_EXISTING
    from ml.registry import MODEL_REGISTRY

    runs = df[df.seed == 42].set_index(["exp", "key"])
    assert runs.index.is_unique, runs.index[runs.index.duplicated()]
    rows = []
    for cell in (n for n in MODEL_REGISTRY if n.startswith("alexnet_fx_")):
        if cell in FX_EXISTING:
            exp, key = FX_EXISTING[cell].split("/")
            hits = runs.loc[[(exp.removeprefix("phase_11_"), key)]] if (exp.removeprefix("phase_11_"), key) in runs.index else runs.iloc[:0]
        else:
            hits = runs[runs.index.get_level_values("key") == cell]
        if hits.empty:
            continue
        levels = {k: (v is not None) if k in ("bn", "drop", "pt") else v for k, v in FX_RE.match(cell).groupdict().items()}
        rows.append({"cell": cell, **levels, **hits.iloc[0][["fp32", "int8", "params_m", "macs_m"]].to_dict()})
    cells = pd.DataFrame(rows)
    cells["dqat"] = cells.int8 - cells.fp32
    return cells


def matched_pairs(cells):
    out = []
    for f, a, b, label in PAIRS:
        others = [x for x in FACTORS if x != f]
        j = cells[cells[f] == a].set_index(others).join(cells[cells[f] == b].set_index(others), lsuffix="_a", rsuffix="_b", how="inner")
        for _, r in j.iterrows():
            out.append(dict(factor=f, contrast=label, cell_a=r.cell_a, cell_b=r.cell_b,
                            **{m: r[f"{m}_b"] - r[f"{m}_a"] for m in METRICS},
                            macs_pct=100 * (r.macs_m_b / r.macs_m_a - 1), params_pct=100 * (r.params_m_b / r.params_m_a - 1)))
    return pd.DataFrame(out)


def summarize_pairs(pairs):
    """Per contrast: n pairs, median/IQR of each Δ, and in how many pairs b beat a (the 'in k of n configurations' line)."""
    rows = []
    for (f, label), g in pairs.groupby(["factor", "contrast"], sort=False):
        row = dict(factor=f, contrast=label, n_pairs=len(g), macs_pct=g.macs_pct.median(), params_pct=g.params_pct.median())
        for m in METRICS:
            v = g[m].dropna()
            row.update({f"{m}_n": len(v), f"{m}_median": v.median(), f"{m}_q25": v.quantile(0.25), f"{m}_q75": v.quantile(0.75),
                        f"{m}_n_positive": int((v > 0).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def fig_matched_pairs(pairs, summary):
    titles = {"fp32": "Δ top-1 FP32 (pp)", "int8": "Δ top-1 INT8 (pp)", "dqat": "Δ ΔQAT (pp; >0 = mais robusto)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.55 * len(summary) + 2), sharey=True)
    y = {c: i for i, c in enumerate(summary.contrast[::-1])}
    for ax, m in zip(axes, METRICS):
        ax.axvline(0, color="k", lw=0.8)
        for s in summary.itertuples():
            v = pairs[pairs.contrast == s.contrast][m].dropna()
            if v.empty:
                continue
            yy = y[s.contrast] + np.random.default_rng(0).uniform(-0.18, 0.18, len(v))
            ax.scatter(v, yy, s=10, alpha=0.5, color=BLUE)
            ax.scatter(v.median(), y[s.contrast], marker="D", s=50, color=RED, zorder=3)
            ax.text(1.01, y[s.contrast], f"{int((v > 0).sum())}/{len(v)}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        ax.set_title(titles[m], fontsize=10)
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels(list(y.keys()), fontsize=9)
    fig.suptitle("Fatorial completo (AlexNetAdapted, seed 42): cada ponto é um par de células iguais em todo o resto; "
                 "losango = mediana; k/n = pares em que o 2º nível ganha", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "17_factorial_matched_pairs.png")


def main():
    apply_report_style()
    df = load()
    t = contrasts(df)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.drop(columns=[c for c in t if c.startswith("pts_")]).to_csv(TABLES / "factor_effects.csv", index=False)
    band = noise_band(df)
    print("noise band (pp):", {k: round(v, 2) for k, v in band.items()})
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
    fig_matched_pairs(pairs, summary)
    print(f"\n{len(cells)}/208 factorial cells trained")
    print(summary[["contrast", "n_pairs", "fp32_median", "fp32_n_positive", "int8_n", "int8_median", "dqat_median", "dqat_n_positive",
                   "macs_pct"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
