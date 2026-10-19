"""Phase 11 -- isolate one factor at a time: paired contrasts (A -> B, everything else identical).

The pairs are every matched pair of the design's cells (ml/model_registrations.py:CELL_FACTORS, the phase_11_*.yaml
models): two cells equal in all factors but one -- plus the few family contrasts the cell grid can't express
(FAMILY_CONTRASTS). Where seeds 42/43/44 exist for both runs the delta is per seed (paired); otherwise it is a single
seed-42 pair. The grey band is the noise floor: 2*sqrt(2)*pooled seed-to-seed SD of the replicated cells (a single-seed
delta inside it is not distinguishable from re-drawing the seed). "layout original" / "layout 64px" are defined in
scripts/phase11/analyze_geometry.py's docstring.

    04..10_*.png                    one figure per factor (absolute top-1 of A and B, baseline line)
    11_factorial_matched_pairs.png  every matched pair, per contrast: the deltas, their median and its bootstrap CI
    21_kernel_rank_critical_difference.png  the 4 kernels' mean rank over the AlexNet configurations that have all four,
                                            beside the table of positions per configuration it averages

The old figures 12 (one row per pair: ~100 rows with every matched pair) and 13 (interactions) are retired: 11 summarizes
every pair with the noise band, 14/15 (scripts/phase11/design_figures.py) show kernel x head x BN and kernel x geometry.
Numbered 01-19 without gaps since 2026-10-14 (docs/logs/PHASE11_LOG.md "Figures renumbered").

    python -m scripts.phase11.factor_effects
"""
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from configs.loader import load_config
from ml.plotting import AMBER, BLUE, GRID, NEUTRAL, RED, TEXT_PRIMARY, apply_report_style
from ml.reporting import holm, mcnemar_p, wilson_ci
from scripts.phase11.analyze_geometry import (BASE_KEY, BASE_LABEL, LAYOUTS, REGIME_LABEL, RESNET_BASE, TABLES, VGG_BASE, band_for,
                                              early_handle, early_ring, int8_bar, noise_bands, pair_band, precision_handles,
                                              reference_lines, savefig)
from scripts.phase11.design_figures import K4, KCOLOR, KLABEL, cell_label, frame

COMP = "Compensação (profundidade e blocos)"
PRE = "Pré-treino ImageNet (não → sim)"
POOL = "Pooling (3 max-pools 3×3 → 2 max-pools 2×2)"
A_COLOR, B_COLOR = BLUE, AMBER  # A and B in every factor figure (report palette); the legend names each, never a kernel
# the factor figures' rows: one family per row, each with its own reference net's line (the families compare to AlexNet)
FAMILY_ROWS = [("alexnet", (BASE_KEY, BASE_LABEL)), ("família", (BASE_KEY, BASE_LABEL)), ("vgg16", VGG_BASE),
               ("resnet18", RESNET_BASE)]
FACTOR_FIG = {"Cabeça (FC + Dropout → GAP)": "04_head_fc_vs_gap.png", "Stride da conv1 (4 → 2)": "05_conv1_stride.png",
              POOL: "06_pooling.png", "Dropout (0 → 0,5)": "07_dropout.png", PRE: "08_pretraining.png",
              "BatchNorm (sem → com)": "09_batchnorm.png", COMP: "10_compensation_block.png"}  # Kernel: figures 01-03

# The factors a matched pair holds fixed. pooling = '<n>pool<k>x<k>' (count and window always change together here);
# map_side follows from kernels + geometry + input_size, so it is not a pairing key; input_size (64, or 224 for the
# phase_11_input224 cells) keeps every other contrast within one input size.
FACTORS = ["family", "kernels", "stride", "pooling", "head", "bn", "dropout", "pretrained", "input_size"]
# (factor, level a, level b, label): delta = b - a over every pair of cells equal in all other factors. A level may be a
# {column: value} dict when the factor moves two columns: the head contrast is GAP vs the reference nets' FC head, which
# has Dropout (2026-10-07) -- a GAP head has none, so head and dropout change together
FC_HEAD, GAP_HEAD = {"head": "fc", "dropout": True}, {"head": "gap", "dropout": False}
PAIRS = [("kernels", "k11-5-3", "k3x3", "Kernel 11-5-3-3-3 → 3×3"), ("kernels", "k3x3", "k2x2", "Kernel 3×3 → 2×2"),
         ("kernels", "k11-5-3", "k2x2", "Kernel 11-5-3-3-3 → 2×2"), ("kernels", "k3x3", "kalt3-2", "Kernel 3×3 → alternado 3-2"),
         ("kernels", "k3x3", "k3x3stacked", "3×3 → dois 3×3 empilhados por estágio"),
         ("kernels", "k2x2", "k2x2stacked", "2×2 → dois 2×2 empilhados por estágio"),
         # the large-stem block (phase_11_large_stem.yaml): AlexNet's 11x11 conv1 kept, a 3x3 / 2x2 body
         ("kernels", "k3x3", "k11-3", "Conv1 3×3 → 11×11, corpo 3×3"), ("kernels", "k2x2", "k11-2", "Conv1 2×2 → 11×11, corpo 2×2"),
         ("kernels", "k11-3", "k11-5-3", "Conv2 3×3 → 5×5, conv1 11×11"),
         ("kernels", "k11-3", "k11-2", "Corpo 3×3 → 2×2, conv1 11×11 mantida"),
         ("head", FC_HEAD, GAP_HEAD, "Cabeça FC (+ Dropout) → GAP"), ("bn", False, True, "BatchNorm não → sim"),
         ("stride", 4, 2, "Stride da conv1 4 → 2"), ("pooling", "3pool3x3", "2pool2x2", "3 max-pools 3×3 → 2 max-pools 2×2"),
         ("dropout", False, True, "Dropout 0 → 0,5 (só cabeça FC)"), ("pretrained", False, True, "Pré-treino ImageNet"),
         # the input-size block (phase_11_input224.yaml): the same net fed Tiny ImageNet enlarged to 224x224
         ("input_size", 64, 224, "Entrada 64 → 224 px (ampliada)"),
         # ResNet-18 (phase_11_resnet_kernel_stem.yaml): its own kernels are k7-3 (7x7 stem, 3x3 blocks); stem and blocks
         # apart through k7-2; the deep stems (ResNet-C) are a depth compensation, like stacked
         ("kernels", "k7-3", "k3x3", "Kernel 7-3 (ResNet) → 3×3"), ("kernels", "k7-3", "k2x2", "Kernel 7-3 (ResNet) → 2×2"),
         ("kernels", "k7-3", "k7-2", "Kernel blocos 3×3 → 2×2, stem 7×7 mantido"),
         ("kernels", "k7-2", "k2x2", "Kernel stem 7×7 → 2×2, blocos 2×2"),
         ("kernels", "k3x3deepstem", "k2x2deepstem", "Kernel 3×3 → 2×2, stem profundo"),
         ("kernels", "k3x3", "k3x3deepstem", "Stem 3×3 → três 3×3 (ResNet-C)"),
         ("kernels", "k2x2", "k2x2deepstem", "Stem 2×2 → três 2×2")]
FIGURE_OF = {"head": "Cabeça (FC + Dropout → GAP)", "stride": "Stride da conv1 (4 → 2)", "pooling": POOL, "dropout": "Dropout (0 → 0,5)",
             "pretrained": PRE, "bn": "BatchNorm (sem → com)",
             "input_size": "Entrada (64 → 224 px)"}  # ponytail: no figure file yet -- the 224 runs are outside core()
K3_GAP_BN = "alexnet_k3x3_stride2_2pool2x2_map8_gap_bn"
FAMILY_CONTRASTS = [  # (factor, label, A, B, note): what the cell grid can't express. † = a second variable changes too
    (COMP, "64px · 3×3 + BN + GAP → Bottleneck", K3_GAP_BN, "alexnet_bottleneck", ""),
    (COMP, "64px · 3×3 + BN + GAP → Fire†", K3_GAP_BN, "alexnet_fire", "o Fire usa conv1 stride 1 (mapas 16×16)"),
    (PRE, "MobileNetV2 (torchvision)", "mobilenetv2_scratch", "mobilenetv2", ""),
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
    # the VGG kernel blocks (phase_11_vgg_kernel_blocks.yaml): VGG16's 3x3 -> each one-kernel-per-block pattern
    blocks = [("kernels", "k3x3", k, f"Kernel 3×3 → blocos {k.removeprefix('kblocks')}")
              for k in sorted(set(cells.kernels.dropna())) if k.startswith("kblocks")]
    out = []
    for f, a, b, label in PAIRS + blocks:
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
            depth = "stacked" in r.cell_b or ("deepstem" in r.cell_b and "deepstem" not in r.cell_a)  # one conv -> several
            factor = COMP if depth else "Kernel"
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


def contrasts(df, bands=None):
    """One row per contrast both of whose runs have finished (a pending pair is left out until then); with bands
    (analyze_geometry.noise_bands), the noise band its single-seed difference is judged by (band_fp32 / band_int8)."""
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
            factor=factor, contrast=label, A=a, B=b, family=idx.loc[(a, 42), "family"], n_seeds=len(seeds), **_pair_stats(idx, a, b),
            **{f"early_{side}": any(idx.loc[(k, s), "early_best"] for s in seeds) for side, k in (("A", a), ("B", b))},
            d_fp32=d["fp32"].mean(), d_fp32_seeds=" / ".join(f"{x:+.2f}" for x in d["fp32"]),
            d_int8=d["int8"].mean(), d_int8_seeds=" / ".join(f"{x:+.2f}" for x in d["int8"]),
            **{f"{c}_{side}": v[c].mean() for side, v in [("A", va), ("B", vb)] for c in ["fp32", "int8"]},
            macs_change_pct=100 * (vb["macs_eff_m"][0] / va["macs_eff_m"][0] - 1),  # effective cost (analyze_geometry.load)
            params_change_pct=100 * (vb["params_eff_m"][0] / va["params_eff_m"][0] - 1), note=note,
            **({f"band_{c}": v for c, v in pair_band(*(band_for(bands, *idx.loc[(k, 42), ["family", "bn"]]) for k in (a, b))).items()}
               if bands is not None else {})))
    t = pd.DataFrame(rows)
    for stage in ("fp32", "int8"):  # many tests in one table: family-wise correction (Holm 1979)
        if f"p_{stage}" in t:
            t[f"p_{stage}_holm"] = holm(t[f"p_{stage}"].tolist())
    return t


def wrap_dots(s, width=24):
    """Break a ' · '-joined label only between its terms (textwrap cut 'alternado 3-2 · 2 | pools 2×2'), each line
    ending in ' ·' when another follows."""
    lines = []
    for term in s.split(" · "):
        if lines and len(lines[-1]) + 3 + len(term) <= width:
            lines[-1] += " · " + term
        else:
            lines.append(term)
    return " ·\n".join(lines)


def family_rows(g, per_row=6):
    """(rows of contrasts, reference net) per family in FAMILY_ROWS order; a family's contrasts split into balanced rows
    (8 -> 4 + 4, not 6 + 2)."""
    out = []
    for fam, base in FAMILY_ROWS:
        f = g[g.family == fam]
        if len(f):
            k = math.ceil(len(f) / math.ceil(len(f) / per_row))
            out += [(f.iloc[i:i + k], base) for i in range(0, len(f), k)]
    return out


def fig_factor(t, df, factor):
    """Absolute top-1 of A and B for every contrast of one factor (A_COLOR, B_COLOR), one family per row with its own
    reference net's line; each group's label states B - A. The legend names A and B from the factor's '(a → b)'."""
    g, color = t[t.factor == factor], B_COLOR
    if g.empty:
        print(f"{FACTOR_FIG[factor]}: no finished pair yet")
        return
    rows = family_rows(g)
    width = max(len(chunk) for chunk, _ in rows)
    fig, axes = plt.subplots(len(rows), 1, figsize=(max(13.0, 2.5 * width + 1.5), 5.2 * len(rows) + 1.2), squeeze=False,
                             sharey=True, layout="constrained")
    w = 0.19
    value = dict(ha="center", va="bottom", fontsize=7.5, zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    for ax, (chunk, base) in zip(axes[:, 0], rows):
        for i, r in enumerate(chunk.itertuples()):
            for j, (side, c) in enumerate([("A", A_COLOR), ("B", color)]):
                fp32, int8 = getattr(r, f"fp32_{side}"), getattr(r, f"int8_{side}")
                ax.bar(i + (2 * j - 1.5) * w, fp32, w, color=c, edgecolor="white")
                int8_bar(ax, i + (2 * j - 0.5) * w, int8, w, c, early=getattr(r, f"early_{side}"))
                ax.text(i + (2 * j - 1.5) * w, fp32 + 0.4, f"{fp32:.1f}", **value)
                if pd.notna(int8):  # INT8 labelled too, as in figures 01-03
                    ax.text(i + (2 * j - 0.5) * w, int8 + 0.4, f"{int8:.1f}", **value)
        ax.set_xticks(range(len(chunk)))
        ax.set_xticklabels([wrap_dots(r.contrast)
                            + f"\nB − A: FP32 {r.d_fp32:+.1f}" + (f" · INT8 {r.d_int8:+.1f}" if pd.notna(r.d_int8) else "") + " pp"
                            + f"\nparâm. {r.params_change_pct:+.0f}% · MACs {r.macs_change_pct:+.0f}%"
                            + (f"\nmédia de {r.n_seeds} seeds" if r.n_seeds > 1 else "") for r in chunk.itertuples()], fontsize=8.5)
        ax.set_xlim(-0.6, width - 0.4)  # same bar width in every row
        ax.set_ylabel("Top-1 (%)")
        ax.margins(y=0.08)
        reference_lines(ax, df, *base)
    notes = {}
    for r in g.itertuples():
        if r.note:
            notes.setdefault(r.note, []).append(r.contrast)
    fig.suptitle(f"{factor}: cada grupo compara duas redes idênticas exceto por este fator (seed 42)\n{LAYOUTS}"
                 + "".join(f"\nnota — {'; '.join(cs)}: {note}" for note, cs in notes.items()), fontsize=11)
    m = re.fullmatch(r"(.+) \((.+) → (.+)\)", factor)
    a, b = (f"{m[1]}: {m[2]}", f"{m[1]}: {m[3]}") if m else ("antes", "depois")
    fig.legend(handles=[Patch(color=A_COLOR, label=f"A = {a}"), Patch(color=color, label=f"B = {b}")]
               + precision_handles() + ([early_handle("hatch")] if (g.early_A | g.early_B).any() else []),
               loc="outside lower center", ncol=2, fontsize=9)
    savefig(fig, FACTOR_FIG[factor])


METRICS = ["fp32", "int8", "dqat"]  # dqat = INT8 - FP32, the report's ΔQAT (negative = loses under INT8)


def factorial_cells(df):
    """Seed-42 table, one row per trained cell of CELL_FACTORS (AlexNet and VGG grids + stacked)."""
    from ml.model_registrations import CELL_FACTORS

    runs = df[(df.seed == 42) & df.key.isin(CELL_FACTORS)]
    assert runs.key.is_unique, runs.key[runs.key.duplicated()]
    if runs.empty:
        return pd.DataFrame()
    cells = with_pooling(pd.DataFrame([{"cell": r.key, **CELL_FACTORS[r.key], "fp32": r.fp32, "int8": r.int8, "early_best": r.early_best,
                                        "params_m": r.params_eff_m, "macs_m": r.macs_eff_m} for r in runs.itertuples()]))  # effective cost
    cells["dqat"] = cells.int8 - cells.fp32
    return cells


def matched_pairs(cells):
    j = pairs_of(cells)
    bn_a, bn_b = ((j[f"bn_{s}"].fillna(j["bn"]) if "bn" in j else j[f"bn_{s}"]) if f"bn_{s}" in j else j["bn"] for s in "ab")
    return pd.DataFrame({"factor": j.factor, "contrast": j.contrast, "cell_a": j.cell_a, "cell_b": j.cell_b,
                         "family": j.family, "bn_a": bn_a, "bn_b": bn_b,
                         **{m: j[f"{m}_b"] - j[f"{m}_a"] for m in METRICS}, "early": j.early_best_a.astype(bool) | j.early_best_b.astype(bool),
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


def fig_matched_pairs(pairs, summary, n_cells, bands):
    titles = {"fp32": "Top-1 FP32: B − A (pp)", "int8": "Top-1 INT8: B − A (pp)", "dqat": "Perda no INT8 (INT8 − FP32): B − A (pp)"}
    xlabels = {"fp32": "← A melhor   ·   B melhor →", "int8": "← A melhor   ·   B melhor →",
               "dqat": "← B perde mais no INT8   ·   B perde menos →"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.55 * len(summary) + 3), sharey=True)
    y = {c: i for i, c in enumerate(summary.contrast[::-1])}
    # one grey span per regime floor, widest (lightest) first; each pair is judged by its own cells' band (band_<m>)
    floors = sorted(((r, b) for r, b in bands.items() if np.isfinite(b["fp32"])), key=lambda rb: -rb[1]["fp32"])
    for ax, m in zip(axes, METRICS):
        for n, (_, b) in enumerate(floors):
            if m in b and np.isfinite(b[m]):
                ax.axvspan(-b[m], b[m], color="#d0d0d0", alpha=0.35 + 0.25 * n / max(len(floors) - 1, 1), zorder=0)
        ax.axvline(0, color="k", lw=0.8)
        for s in summary.itertuples():
            v = pairs[pairs.contrast == s.contrast][m].dropna()
            if v.empty:
                continue
            yy = y[s.contrast] + np.random.default_rng(0).uniform(-0.18, 0.18, len(v))
            inside = (v.abs() < pairs.loc[v.index, f"band_{m}"]).to_numpy() if f"band_{m}" in pairs else np.zeros(len(v), bool)
            ax.scatter(v[~inside], yy[~inside], s=14, alpha=0.8, color=BLUE)
            ax.scatter(v[inside], yy[inside], s=10, alpha=0.7, facecolors="none", edgecolors=BLUE, lw=0.8)
            e = pairs.loc[v.index, "early"].to_numpy()
            if m != "fp32" and e.any():  # INT8 and ΔQAT of a pair with an early-best cell
                early_ring(ax, v[e], yy[e], s=45)
            ax.scatter(v.median(), y[s.contrast], marker="D", s=50, zorder=3, color=RED)
            ax.hlines(y[s.contrast], *bootstrap_median_ci(v), color=RED, lw=2, zorder=2)
            ax.text(1.01, y[s.contrast], f"{int((v > 0).sum())}/{len(v)}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        ax.set_title(titles[m], fontsize=10)
        ax.set_xlabel(xlabels[m], fontsize=9)
    axes[0].text(0, 1.0, "fator trocado: A → B", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=9, weight="bold")
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels(list(y.keys()), fontsize=9)
    handles = [Line2D([], [], marker="o", color=BLUE, ls="", ms=4, alpha=0.8, label="um par de redes iguais em tudo menos no fator (valor = B − A)"),
               Line2D([], [], marker="D", color=RED, ls="-", ms=7, label="mediana dos pares (barra: IC 95% bootstrap)"),
               Line2D([], [], color="k", lw=0.8, label="linha 0 = A e B iguais (sem efeito)"),
               Line2D([], [], ls="", label="k/n à direita = em quantos pares B ficou acima de A")]
    for r, b in floors:
        handles.append(Patch(color="#d0d0d0", label=f"ruído entre seeds, {REGIME_LABEL.get(r, r)}: ±{b['fp32']:.1f} pp FP32"
                             + (f", ±{b['int8']:.1f} pp INT8" if np.isfinite(b["int8"]) else "")))
    if floors:
        handles.append(Line2D([], [], marker="o", ls="", ms=4, mfc="none", color=BLUE,
                              label="par dentro do ruído das suas duas redes (vazado)"))
    if pairs.early.any():
        handles.append(early_handle("ring"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize=9)
    fig.suptitle("Efeito de trocar um fator de cada vez (A → B, todo o resto igual): cada ponto azul é um par de redes e "
                 f"o valor é B − A; 0 = sem efeito\nseed 42 · {n_cells} redes do desenho treinadas até agora", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "11_factorial_matched_pairs.png")


def critical_difference(cells, metric):
    """Demšar (2006): the 4 kernels' mean rank (1 = most accurate) over the AlexNet configurations that trained all four
    (each configuration a block, the other factors fixed), Friedman's p and Nemenyi's critical difference at alpha 0.05.
    None while fewer than 2 configurations are complete."""
    from scipy.stats import friedmanchisquare, rankdata, studentized_range

    if cells.empty:
        return None
    c = cells[(cells.family == "alexnet") & cells.kernels.isin(K4)]
    t = c.pivot_table(index=[f for f in FACTORS if f not in ("family", "kernels")], columns="kernels", values=metric)
    t = t.reindex(columns=K4).dropna()
    if len(t) < 2:
        return None
    k, n = len(K4), len(t)
    ranks = np.array([rankdata(-row) for row in t.to_numpy()]).mean(0)
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    return dict(zip(K4, ranks)), friedmanchisquare(*t.to_numpy().T).pvalue, cd, t


def rank_table(ax, t, rank):
    """What one diagram averages: each kernel's position (top-1 %) per configuration, the 1st place filled with its
    kernel's colour, and the mean position (= the diagram's dot) as the last row."""
    pos = t.rank(axis=1, ascending=False)
    rows = []
    for i in t.index:
        f = dict(zip(t.index.names, i))
        rows.append(cell_label({**f, "family": "alexnet", "pool_count": f["pooling"][0], "pool_kernel": f["pooling"][-1]},
                               skip=("kernels",)))
    text = [[f"{pos.at[i, k]:g}º ({t.at[i, k]:.1f})" for k in K4] for i in t.index] + [[f"{rank[k]:.2f}" for k in K4]]
    fill = [[to_rgba(KCOLOR[k], 0.35 if pos.at[i, k] == 1 else 0) for k in K4] for i in t.index] + [["white"] * len(K4)]
    tb = ax.table(cellText=text, cellColours=fill, rowLabels=rows + ["posição média (= a bolinha)"],
                  colLabels=[KLABEL[k] for k in K4], colWidths=[0.13] * len(K4), rowLoc="left", cellLoc="center",
                  bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0 and c >= 0:
            cell.get_text().set(color=KCOLOR[K4[c]], weight="bold")
        if r == len(t) + 1:
            cell.get_text().set_weight("bold")
    ax.set_title("de onde vem a média: posição de cada kernel (top-1 %) em cada configuração\n"
                 "célula colorida = 1º lugar naquela configuração", fontsize=10)


def fig_critical_difference(cells):
    """Critical-difference diagrams (Demšar 2006), FP32 and INT8: kernels on a mean-rank axis, a black bar joining the
    ones Nemenyi cannot tell apart; beside each, the table of positions it averages (rank_table)."""
    fig, axes = plt.subplots(2, 2, figsize=(19, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    for (ax, tab), m in zip(axes, ["fp32", "int8"]):
        ax.axis("off")
        tab.axis("off")
        ax.set(xlim=(0.6, len(K4) + 0.4), ylim=(-1.45, 0.95))
        res = critical_difference(cells, m)
        if res is None:
            ax.text(0.5, 0.5, f"{m.upper()}: pendente", transform=ax.transAxes, ha="center", color=NEUTRAL, fontsize=12)
            continue
        rank, p, cd, t = res
        ax.hlines(0, 1, len(K4), color=TEXT_PRIMARY, lw=1.2)
        for i in range(1, len(K4) + 1):
            ax.vlines(i, 0, 0.08, color=TEXT_PRIMARY, lw=1.2)
            ax.text(i, 0.13, str(i), ha="center", va="bottom", fontsize=9)
        ax.text(0.95, 0.13, "melhor", ha="right", va="bottom", fontsize=9, color=NEUTRAL)
        ax.hlines(0.6, 1, 1 + cd, color=TEXT_PRIMARY, lw=2)  # the CD as a length on the rank axis
        ax.text(1 + cd / 2, 0.66, f"diferença crítica = {cd:.2f}", ha="center", va="bottom", fontsize=9)
        order = sorted(K4, key=rank.get)
        cliques = []  # maximal runs of rank-ordered kernels spanning less than the CD
        for i in range(len(order)):
            j = max(jj for jj in range(i, len(order)) if rank[order[jj]] - rank[order[i]] < cd)
            if j > i and not any(a <= i and j <= b for a, b in cliques):
                cliques.append((i, j))
        for h, (i, j) in enumerate(cliques):
            ax.hlines(-0.18 - 0.14 * h, rank[order[i]] - 0.03, rank[order[j]] + 0.03, color=TEXT_PRIMARY, lw=4)
        for n, kk in enumerate(order):
            x, y = rank[kk], -0.85 - 0.35 * (n % 2)  # alternate rows: close ranks don't print over each other
            ax.scatter(x, 0, s=90, color=KCOLOR[kk], zorder=3)
            ax.plot([x, x], [0, y + 0.12], color=KCOLOR[kk], lw=1)
            ax.text(x, y, f"{KLABEL[kk]} ({x:.2f})", ha="center", va="top", fontsize=10, color=KCOLOR[kk], weight="bold", zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))  # over the other kernels' leader lines
        ax.set_title(f"{m.upper()}: posição média em {len(t)} configurações · Friedman "
                     + ("p < 0.001" if p < 0.001 else f"p = {p:.3f}"), fontsize=10)
        rank_table(tab, t, rank)
        print(f"21 {m}: mean ranks {({KLABEL[k]: round(v, 2) for k, v in rank.items()})}, CD {cd:.2f}, Friedman p {p:.4f}, "
              f"configurations:\n{t.round(2).to_string()}")
    fig.suptitle("Qual kernel é melhor no geral? Posição média de cada kernel (1 = mais acurado) nas configurações do AlexNet\n"
                 "em que os quatro foram treinados; kernels ligados por uma barra preta não se distinguem (Friedman + Nemenyi,\n"
                 "α = 0,05; Demšar 2006) · seed 42 · top-1 no test set", fontsize=11)
    fig.tight_layout()
    savefig(fig, "21_kernel_rank_critical_difference.png")


def main():
    apply_report_style()
    df = frame()
    if df.empty:
        print("no Phase 11 run has a summary yet")
        return
    bands = noise_bands(df)  # one floor per regime; no seed band of its own for the INT8 - FP32 difference (dqat)
    t = contrasts(df, bands)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "factor_effects.csv", index=False)
    print("noise bands (pp):", {REGIME_LABEL.get(r, r): {k: round(v, 2) for k, v in b.items()} for r, b in bands.items()})
    if not t.empty:
        for factor in FACTOR_FIG:
            fig_factor(t, df, factor)
        pd.set_option("display.width", 250, "display.max_colwidth", 45)
        print(t[["factor", "contrast", "n_seeds", "d_fp32", "d_int8", "macs_change_pct", "params_change_pct"]].round(2).to_string(index=False))

    cells = factorial_cells(df)
    fig_critical_difference(cells)
    pairs = matched_pairs(cells) if len(cells) else pd.DataFrame()
    if pairs.empty:
        print("14: no finished matched pair yet")
        return
    for m in ("fp32", "int8"):
        pairs[f"band_{m}"] = [pair_band(band_for(bands, f, a), band_for(bands, f, b))[m]
                              for f, a, b in zip(pairs.family, pairs.bn_a, pairs.bn_b)]
    summary = summarize_pairs(pairs)
    cells.to_csv(TABLES / "factorial_cells.csv", index=False)
    pairs.to_csv(TABLES / "factorial_pairs.csv", index=False)
    summary.to_csv(TABLES / "factorial_pair_summary.csv", index=False)
    fig_matched_pairs(pairs, summary, len(cells), bands)
    print(f"\n{len(cells)} factorial cells trained (seed 42)")
    print(summary[["contrast", "n_pairs", "fp32_median", "fp32_n_positive", "int8_n", "int8_median", "dqat_median", "dqat_n_positive",
                   "macs_pct"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
