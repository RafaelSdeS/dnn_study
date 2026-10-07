"""Phase 11 -- isolate one factor at a time: paired contrasts (A -> B, everything else identical).

The pairs are every matched pair of the design's cells (ml/model_registrations.py:CELL_FACTORS, the phase_11_*.yaml
models): two cells equal in all factors but one -- plus the few family contrasts the cell grid can't express
(FAMILY_CONTRASTS). Where seeds 42/43/44 exist for both runs the delta is per seed (paired); otherwise it is a single
seed-42 pair. The grey band is the noise floor: 2*sqrt(2)*pooled seed-to-seed SD of the replicated cells (a single-seed
delta inside it is not distinguishable from re-drawing the seed). "layout original" / "layout 64px" are defined in
scripts/phase11/analyze_geometry.py's docstring.

    05..11_*.png                    one figure per factor (absolute top-1 of A and B, baseline line)
    14_factorial_matched_pairs.png  every matched pair, per contrast: the deltas, their median and its bootstrap CI

Figures 12 (one row per pair: ~100 rows with every matched pair) and 13 (interactions) are retired: 14 summarizes every
pair with the noise band, 17/18 (scripts/phase11/design_figures.py) show kernel x head x BN and kernel x geometry.

    python -m scripts.phase11.factor_effects
"""
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from configs.loader import load_config
from ml.plotting import AMBER, BLUE, GREEN, NEUTRAL, RED, apply_report_style
from ml.reporting import holm, mcnemar_p, wilson_ci
from scripts.phase11.analyze_geometry import (LAYOUTS, TABLES, int8_bar, noise_band, precision_handles, reference_lines,
                                              savefig)
from scripts.phase11.design_figures import cell_label, frame

PURPLE = "#7b3fa0"
COMP = "Compensação (profundidade e blocos)"
PRE = "Pré-treino ImageNet (não → sim)"
FACTOR_COLOR = {"Kernel": GREEN, "Cabeça (FC → GAP)": RED, "Stride da conv1 (4 → 2)": BLUE, "Pooling": AMBER,
                "Dropout (0 → 0,5)": "#8d6e63", PRE: PURPLE, "BatchNorm (sem → com)": "#0d8b8b", COMP: "#c2185b"}
FACTOR_FIG = {"Cabeça (FC → GAP)": "05_head_fc_vs_gap.png", "Stride da conv1 (4 → 2)": "06_conv1_stride.png",
              "Pooling": "07_pooling.png", "Dropout (0 → 0,5)": "08_dropout.png", PRE: "09_pretraining.png",
              "BatchNorm (sem → com)": "10_batchnorm.png", COMP: "11_compensation_block.png"}  # Kernel: figures 02-04

# The factors a matched pair holds fixed. pooling = '<n>pool<k>x<k>' (count and window always change together here);
# map_side follows from kernels + geometry, so it is not a pairing key.
FACTORS = ["family", "kernels", "stride", "pooling", "head", "bn", "dropout", "pretrained"]
# (factor, level a, level b, label): delta = b - a over every pair of cells equal in all other factors. A level may be a
# {column: value} dict when the factor moves two columns: the head contrast is GAP vs the reference nets' FC head, which
# has Dropout (2026-10-07) -- a GAP head has none, so head and dropout change together
FC_HEAD, GAP_HEAD = {"head": "fc", "dropout": True}, {"head": "gap", "dropout": False}
PAIRS = [("kernels", "k11-5-3", "k3x3", "Kernel 11-5-3-3-3 → 3×3"), ("kernels", "k3x3", "k2x2", "Kernel 3×3 → 2×2"),
         ("kernels", "k11-5-3", "k2x2", "Kernel 11-5-3-3-3 → 2×2"), ("kernels", "k3x3", "kalt3-2", "Kernel 3×3 → alternado 3-2"),
         ("kernels", "k3x3", "k3x3stacked", "3×3 → dois 3×3 empilhados por estágio"),
         ("kernels", "k2x2", "k2x2stacked", "2×2 → dois 2×2 empilhados por estágio"),
         ("head", FC_HEAD, GAP_HEAD, "Cabeça FC (+ Dropout) → GAP"), ("bn", False, True, "BatchNorm não → sim"),
         ("stride", 4, 2, "Stride da conv1 4 → 2"), ("pooling", "3pool3x3", "2pool2x2", "3 max-pools 3×3 → 2 max-pools 2×2"),
         ("dropout", False, True, "Dropout 0 → 0,5 (só cabeça FC)"), ("pretrained", False, True, "Pré-treino ImageNet")]
FIGURE_OF = {"head": "Cabeça (FC → GAP)", "stride": "Stride da conv1 (4 → 2)", "pooling": "Pooling", "dropout": "Dropout (0 → 0,5)",
             "pretrained": PRE, "bn": "BatchNorm (sem → com)"}
K3_GAP_BN = "alexnet_k3x3_stride2_2pool2x2_map8_gap_bn"
FAMILY_CONTRASTS = [  # (factor, label, A, B, note): what the cell grid can't express. † = a second variable changes too
    (COMP, "64px · 3×3 + BN + GAP → Bottleneck", K3_GAP_BN, "alexnet_bottleneck", ""),
    (COMP, "64px · 3×3 + BN + GAP → Fire†", K3_GAP_BN, "alexnet_fire", "o Fire usa conv1 stride 1 (mapas 16×16)"),
    (PRE, "MobileNetV2 (torchvision)", "mobilenetv2_scratch", "mobilenetv2", ""),
    (PRE, "ResNet-18 (torchvision)", "resnet18tv_scratch", "resnet18tv", ""),
]


def with_pooling(cells):
    return cells.assign(pooling=cells.pool_count.astype(int).astype(str) + "pool" + cells.pool_kernel.astype(int).astype(str)
                        + "x" + cells.pool_kernel.astype(int).astype(str))


def design_cells():
    """Every cell the phase_11_*.yaml files train (trained yet or not), with its factors."""
    from pathlib import Path
    from ml.model_registrations import CELL_FACTORS

    models = {m for p in (Path(__file__).resolve().parents[2] / "configs/experiments").glob("phase_11_*.yaml")
              for m in load_config(f"experiments/{p.name}")["models"]}
    return with_pooling(pd.DataFrame([{"cell": k, **f} for k, f in CELL_FACTORS.items() if k in models]))


def pairs_of(cells):
    """Every (a, b) of cells equal in all FACTORS but the one(s) a PAIRS entry varies; the other columns of a and b come
    along with _a / _b suffixes, the shared factors as plain columns."""
    out = []
    for f, a, b, label in PAIRS:
        a, b = (lv if isinstance(lv, dict) else {f: lv} for lv in (a, b))
        others = [x for x in FACTORS if x not in a]
        side = lambda lv: cells[np.logical_and.reduce([cells[c] == v for c, v in lv.items()])].set_index(others)  # noqa: E731
        j = side(a).join(side(b), lsuffix="_a", rsuffix="_b", how="inner")
        out.append(j.reset_index().assign(factor=f, contrast=label))
    return pd.concat(out, ignore_index=True)


def contrast_list():
    """(factor, label, A, B, note) for every matched pair of the design, then the family contrasts."""
    from ml.model_registrations import CELL_FACTORS

    rows = []
    for r in pairs_of(design_cells()).itertuples():
        context = cell_label(CELL_FACTORS[r.cell_a], skip=(r.factor,) + (("dropout",) if r.factor == "head" else ()))
        if r.factor == "kernels":
            factor = COMP if "stacked" in r.cell_b else "Kernel"
            label = f"{r.contrast.removeprefix('Kernel ')} · {context}"
        else:
            factor, label = FIGURE_OF[r.factor], context
        rows.append((factor, label, r.cell_a, r.cell_b, ""))
    return rows + FAMILY_CONTRASTS


def _test_hits(results, key, stage):
    """Per-image top-1 hits on the held-out test set, or None if the run did not write its test logits."""
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
    """One row per contrast both of whose runs have finished (a pending pair is left out until then)."""
    idx = df.set_index(["key", "seed"])
    assert idx.index.is_unique, idx.index[idx.index.duplicated()]
    rows = []
    for factor, label, a, b, note in contrast_list():
        seeds = sorted({s for k, s in idx.index if k == a} & {s for k, s in idx.index if k == b})
        if 42 not in seeds:
            continue
        va, vb = ({c: np.array([idx.loc[(k, s), c] for s in seeds]) for c in ["fp32", "int8", "macs_eff_m", "params_eff_m"]} for k in (a, b))
        d = {c: vb[c] - va[c] for c in va}
        rows.append(dict(
            factor=factor, contrast=label, A=a, B=b, n_seeds=len(seeds), **_pair_stats(idx, a, b),
            d_fp32=d["fp32"].mean(), d_fp32_seeds=" / ".join(f"{x:+.2f}" for x in d["fp32"]),
            d_int8=d["int8"].mean(), d_int8_seeds=" / ".join(f"{x:+.2f}" for x in d["int8"]),
            **{f"{c}_{side}": v[c].mean() for side, v in [("A", va), ("B", vb)] for c in ["fp32", "int8"]},
            macs_change_pct=100 * (vb["macs_eff_m"][0] / va["macs_eff_m"][0] - 1),  # effective cost (analyze_geometry.load)
            params_change_pct=100 * (vb["params_eff_m"][0] / va["params_eff_m"][0] - 1), note=note))
    t = pd.DataFrame(rows)
    for stage in ("fp32", "int8"):  # many tests in one table: family-wise correction (Holm 1979)
        if f"p_{stage}" in t:
            t[f"p_{stage}_holm"] = holm(t[f"p_{stage}"].tolist())
    return t


def fig_factor(t, df, factor, per_row=6):
    """Absolute top-1 of A and B for every contrast of one factor: A grey, B in the factor's colour, + baseline."""
    g, color = t[t.factor == factor], FACTOR_COLOR[factor]
    if g.empty:
        print(f"{FACTOR_FIG[factor]}: no finished pair yet")
        return
    chunks = [g.iloc[k:k + per_row] for k in range(0, len(g), per_row)]
    width = min(len(g), per_row)
    fig, axes = plt.subplots(len(chunks), 1, figsize=(max(11.0, 2.5 * width + 1.5), 5.2 * len(chunks) + 1.2), squeeze=False,
                             sharey=True, layout="constrained")
    w = 0.19
    ref = []
    for ax, chunk in zip(axes[:, 0], chunks):
        for i, r in enumerate(chunk.itertuples()):
            for j, (side, c) in enumerate([("A", NEUTRAL), ("B", color)]):
                fp32 = getattr(r, f"fp32_{side}")
                ax.bar(i + (2 * j - 1.5) * w, fp32, w, color=c, edgecolor="white")
                int8_bar(ax, i + (2 * j - 0.5) * w, getattr(r, f"int8_{side}"), w, c)
                ax.text(i + (2 * j - 1.5) * w, fp32 + 0.4, f"{fp32:.1f}", ha="center", va="bottom", fontsize=7.5, zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
        ax.set_xticks(range(len(chunk)))
        ax.set_xticklabels([textwrap.fill(r.contrast, 22) + f"\nparâm. {r.params_change_pct:+.0f}% · MACs {r.macs_change_pct:+.0f}%"
                            + (f"\nmédia de {r.n_seeds} seeds" if r.n_seeds > 1 else "") for r in chunk.itertuples()], fontsize=8.5)
        ax.set_xlim(-0.6, width - 0.4)  # same bar width in every row
        ax.set_ylabel("Top-1 (%)")
        ax.margins(y=0.08)
        ref = reference_lines(ax, df) or ref
    notes = {}
    for r in g.itertuples():
        if r.note:
            notes.setdefault(r.note, []).append(r.contrast)
    fig.suptitle(f"{factor}: cada grupo compara duas redes idênticas exceto por este fator (seed 42)\n{LAYOUTS}"
                 + "".join(f"\nnota — {'; '.join(cs)}: {note}" for note, cs in notes.items()), fontsize=10)
    fig.legend(handles=[Patch(color=NEUTRAL, label="A = antes (1º nível do fator)"), Patch(color=color, label="B = depois (2º nível)")]
               + precision_handles() + ref, loc="outside lower center", ncol=2, fontsize=8.5)
    savefig(fig, FACTOR_FIG[factor])


METRICS = ["fp32", "int8", "dqat"]  # dqat = INT8 - FP32, the report's ΔQAT (negative = loses under INT8)


def factorial_cells(df):
    """Seed-42 table, one row per trained cell of CELL_FACTORS (AlexNet and VGG grids + stacked)."""
    from ml.model_registrations import CELL_FACTORS

    runs = df[(df.seed == 42) & df.key.isin(CELL_FACTORS)]
    assert runs.key.is_unique, runs.key[runs.key.duplicated()]
    if runs.empty:
        return pd.DataFrame()
    cells = with_pooling(pd.DataFrame([{"cell": r.key, **CELL_FACTORS[r.key], "fp32": r.fp32, "int8": r.int8,
                                        "params_m": r.params_eff_m, "macs_m": r.macs_eff_m} for r in runs.itertuples()]))  # effective cost
    cells["dqat"] = cells.int8 - cells.fp32
    return cells


def matched_pairs(cells):
    j = pairs_of(cells)
    return pd.DataFrame({"factor": j.factor, "contrast": j.contrast, "cell_a": j.cell_a, "cell_b": j.cell_b,
                         **{m: j[f"{m}_b"] - j[f"{m}_a"] for m in METRICS},
                         "macs_pct": 100 * (j.macs_m_b / j.macs_m_a - 1), "params_pct": 100 * (j.params_m_b / j.params_m_a - 1)})


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


def fig_matched_pairs(pairs, summary, n_cells, band):
    titles = {"fp32": "Δ top-1 FP32 (pp)", "int8": "Δ top-1 INT8 (pp)", "dqat": "Δ ΔQAT (pp; >0 = mais robusto)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.55 * len(summary) + 3), sharey=True)
    y = {c: i for i, c in enumerate(summary.contrast[::-1])}
    for ax, m in zip(axes, METRICS):
        if np.isfinite(band.get(m, np.nan)):
            ax.axvspan(-band[m], band[m], color="#d0d0d0", alpha=0.6, zorder=0)
        ax.axvline(0, color="k", lw=0.8)
        for s in summary.itertuples():
            v = pairs[pairs.contrast == s.contrast][m].dropna()
            if v.empty:
                continue
            yy = y[s.contrast] + np.random.default_rng(0).uniform(-0.18, 0.18, len(v))
            ax.scatter(v, yy, s=10, alpha=0.5, color=BLUE)
            ax.scatter(v.median(), y[s.contrast], marker="D", s=50, zorder=3, color=RED)
            ax.hlines(y[s.contrast], *bootstrap_median_ci(v), color=RED, lw=2, zorder=2)
            ax.text(1.01, y[s.contrast], f"{int((v > 0).sum())}/{len(v)}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        ax.set_title(titles[m], fontsize=10)
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels(list(y.keys()), fontsize=9)
    handles = [Line2D([], [], marker="o", color=BLUE, ls="", ms=4, alpha=0.5, label="um par de células iguais em todo o resto"),
               Line2D([], [], marker="D", color=RED, ls="-", ms=7, label="mediana dos pares (barra: IC 95% bootstrap)"),
               Line2D([], [], ls="", label="k/n à direita = pares em que o 2º nível ganha")]
    if np.isfinite(band.get("fp32", np.nan)):
        handles.append(Patch(color="#d0d0d0", label=f"ruído entre seeds (±{band['fp32']:.1f} pp FP32"
                             + (f", ±{band['int8']:.1f} pp INT8)" if np.isfinite(band.get("int8", np.nan)) else ")")))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize=9)
    fig.suptitle(f"Células do desenho da Fase 11 (seed 42; {n_cells} treinadas até agora): cada ponto compara "
                 "duas células que só diferem no fator da linha", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "14_factorial_matched_pairs.png")


def main():
    apply_report_style()
    df = frame()
    if df.empty:
        print("no Phase 11 run has a summary yet")
        return
    t = contrasts(df)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "factor_effects.csv", index=False)
    band = noise_band(df)
    band["dqat"] = float("nan")  # no seed band of its own for the INT8 - FP32 difference
    print("noise band (pp):", {k: round(v, 2) for k, v in band.items()})
    if not t.empty:
        for factor in FACTOR_FIG:
            fig_factor(t, df, factor)
        pd.set_option("display.width", 250, "display.max_colwidth", 45)
        print(t[["factor", "contrast", "n_seeds", "d_fp32", "d_int8", "macs_change_pct", "params_change_pct"]].round(2).to_string(index=False))

    cells = factorial_cells(df)
    pairs = matched_pairs(cells) if len(cells) else pd.DataFrame()
    if pairs.empty:
        print("14: no finished matched pair yet")
        return
    summary = summarize_pairs(pairs)
    cells.to_csv(TABLES / "factorial_cells.csv", index=False)
    pairs.to_csv(TABLES / "factorial_pairs.csv", index=False)
    summary.to_csv(TABLES / "factorial_pair_summary.csv", index=False)
    fig_matched_pairs(pairs, summary, len(cells), band)
    print(f"\n{len(cells)} factorial cells trained (seed 42)")
    print(summary[["contrast", "n_pairs", "fp32_median", "fp32_n_positive", "int8_n", "int8_median", "dqat_median", "dqat_n_positive",
                   "macs_pct"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
