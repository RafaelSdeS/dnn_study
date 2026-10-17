"""Phase 11 figures 13-20 -- the design read through its factors (ml/model_registrations.py:CELL_FACTORS), not run names.
Selecting by factor is what lets one file draw the live runs and, with --archive, the superseded AdamW-recipe runs
(LEGACY maps their names onto the same factors) -- a preview of each figure's form, not its numbers: validation split,
pre-fix INT8, latencies that still timed the DataLoader, and some only-approximate twins (marked †).

    13_pareto_accuracy_cost.png     top-1 INT8 vs MACs and vs INT8 size; Pareto front labelled
    14_main_factorial_grid.png      kernel x {GAP, FC} x {no BN, BN} at the 64px layout, and the delta vs 11-5-3-3-3
    15_kernel_by_geometry.png       one line per kernel across the 4 geometries (FC + Dropout, no BN)
    16_quantization_robustness.png  INT8 - FP32 per kernel / family, and vs each net's worst layer-input max/p99.9
    17_latency_vs_macs.png          batch-1 latency vs MACs on one machine, against the AlexNet original's ms per MAC
    18_training_curves.png          val/train top-1 per epoch at the 64px layout, {GAP, FC} x {no BN, BN}
    19_qat_curves.png               the 50 QAT epochs of the same nets: val top-1 with fake-quant, from the FP32 best to INT8
    20_int8_prediction_flips.png    % of test images right -> wrong and wrong -> right under INT8 (from the test logits)

18 and 19 read the per-epoch logs, which only PCAD keeps -- fetch them first (analyze_geometry.convergence's rsync; add
archive_adamw_recipe/ to both paths for --archive).

    python -m scripts.phase11.design_figures [--archive]
"""
import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from configs.loader import load_config
from ml.plotting import AMBER, BLUE, GREEN, NEUTRAL, PURPLE, RED, TEXT_PRIMARY, TEXT_SECONDARY, apply_report_style, shade
from scripts.phase11.analyze_geometry import (ARCHIVE, BASE_KEY, BASE_LABEL, EARLY_LABEL, FIGS, LOG_EPOCH, REFERENCE_REGIME, ROOT,
                                              RUNS, band_for, early_handle, early_ring, load, noise_band, noise_bands,
                                              reference_lines, savefig)

PREVIEW = ROOT / "results/archive_adamw_recipe/figures_generated/phase_11_preview"
PREVIEW_NOTE = ("PRÉVIA — runs antigas: receita AdamW, split de validação, INT8 antes das correções de 30/09–03/10; "
                "† = só aproximadamente a célula nova")
LIVE_NOTE = "seed 42 · top-1 no test set (val oficial do Tiny ImageNet)"

K4 = ["k11-5-3", "k3x3", "k2x2", "kalt3-2"]
KLABEL = {"k11-5-3": "11-5-3-3-3", "k3x3": "3×3", "k2x2": "2×2", "kalt3-2": "alternado 3-2", "k3x3stacked": "3×3 empilhado",
          "k2x2stacked": "2×2 empilhado", "k11-3": "conv1 11×11 + 3×3", "k11-2": "conv1 11×11 + 2×2", "k7-3": "7×7 + 3×3 (original)", "k7-2": "stem 7×7 + 2×2",
          "k3x3deepstem": "3×3, stem profundo", "k2x2deepstem": "2×2, stem profundo"}
# every Phase 11 colour comes from ml/plotting.py's report palette; a deeper variant of a kernel = a darker shade of its hue
KCOLOR = {"k11-5-3": AMBER, "k3x3": GREEN, "k2x2": BLUE, "kalt3-2": RED, "k3x3stacked": shade(GREEN, 0.45), "k2x2stacked": shade(BLUE, 0.45),
          "k11-3": PURPLE, "k11-2": shade(PURPLE, 0.45),  # purple = a large conv1 / stem before a small-kernel body (as k7-2); darker = 2x2 body
          "k7-3": AMBER, "k7-2": PURPLE, "k3x3deepstem": shade(GREEN, 0.45), "k2x2deepstem": shade(BLUE, 0.45)}  # deep stem = depth, as stacked
RESNET_K = ["k7-3", "k3x3", "k2x2", "kalt3-2", "k7-2", "k3x3deepstem", "k2x2deepstem"]
FAMILY_TAG = {"vgg16": "VGG16", "resnet18": "ResNet-18"}  # nets kept at their own geometry, always with BN
HMARK = {"gap": "o", "fc": "s", "fcdrop": "^", "família": "X"}
HLABEL = {"gap": "cabeça GAP", "fc": "cabeça FC", "fcdrop": "FC + Dropout", "família": "rede de compensação / família"}
GEOMS = {(2, 2, 2): "layout 64px\nconv1 s2, 2 pools 2×2", (4, 2, 2): "conv1 s4,\n2 pools 2×2",
         (2, 3, 3): "conv1 s2,\n3 pools 3×3", (4, 3, 3): "layout original\nconv1 s4, 3 pools 3×3"}


def _f(kernels, family="alexnet", stride=2, pool_kernel=2, pool_count=2, head="fc", bn=False, dropout=False,
       pretrained=False, approx=False):
    return dict(family=family, kernels=kernels, stride=stride, pool_kernel=pool_kernel, pool_count=pool_count, head=head,
                bn=bn, dropout=dropout, pretrained=pretrained, approx=approx)


S4P3 = dict(stride=4, pool_kernel=3, pool_count=3)  # torchvision's AlexNet layout
LEGACY = {  # superseded run -> the factors of the cell it previews; approx = not layer-for-layer that cell
    "alexnet_adapted_orig_fc": _f("k11-5-3"), "alexnet_adapted_orig_gap": _f("k11-5-3", head="gap"),
    "alexnet_3x3_fc": _f("k3x3"), "alexnet_3x3_gap": _f("k3x3", head="gap"), "alexnet_3x3_gap_bn": _f("k3x3", head="gap", bn=True),
    "alexnet_adapted_2x2_fc": _f("k2x2", approx=True), "alexnet_adapted_2x2_gap": _f("k2x2", head="gap", approx=True),
    "alexnet_mixed_fc": _f("kalt3-2", approx=True), "alexnet_mixed": _f("kalt3-2", head="gap", approx=True),
    "alexnet_mixed_fc_bn": _f("kalt3-2", bn=True, approx=True), "alexnet_mixed_bn": _f("kalt3-2", head="gap", bn=True, approx=True),
    "alexnet_geo_s4_p3_fc": _f("k11-5-3", **S4P3), "alexnet_geo_s4_p3_gap": _f("k11-5-3", head="gap", **S4P3),
    "alexnet_geo_s4_p3_fc_k3": _f("k3x3", **S4P3), "alexnet_geo_s4_p2_fc": _f("k11-5-3", stride=4),
    "alexnet_geo_s2_p3_fc": _f("k11-5-3", pool_kernel=3, pool_count=3), "alexnet_geo_s2_p2_drop_fc": _f("k11-5-3", dropout=True),
    "alexnet_tv_scratch": _f("k11-5-3", dropout=True, **S4P3), "alexnet_tv": _f("k11-5-3", dropout=True, pretrained=True, **S4P3),
    "alexnet_tv_3x3": _f("k3x3", dropout=True, **S4P3), "alexnet_tv_2x2": _f("k2x2", dropout=True, approx=True, **S4P3),
    "alexnet_adapted_orig_fc_pt": _f("k11-5-3", pretrained=True),
    "alexnet_stacked": _f("k3x3stacked", bn=True), "alexnet_stacked_gap": _f("k3x3stacked", head="gap", bn=True),
    "vgg16": _f("k3x3", "vgg16", stride=1, pool_count=5, bn=True, dropout=True),
    "vgg16_2x2": _f("k2x2", "vgg16", stride=1, pool_count=5, bn=True, dropout=True, approx=True),
}
LEGACY_NOISE = ["alexnet_3x3_fc", "alexnet_adapted_orig_fc", "alexnet_3x3_gap", "alexnet_adapted_orig_gap"]  # seeds 42-44


def core(df: pd.DataFrame) -> pd.DataFrame:
    """Figures 01-22's 64 px design: without the input-size block (phase_11_input224, Tiny ImageNet enlarged to 224) and
    the VGG kernel blocks (phase_11_vgg_kernel_blocks), whose questions get figures of their own."""
    return df[(df.input_size.fillna(64) == 64) & ~df.kernels.fillna("").str.startswith("kblocks")]


def frame(archive: bool = False, everything: bool = False) -> pd.DataFrame:
    """load() + each run's factors; the families (phase_11_families.yaml) get family 'família' and no grid factors.
    pooling = '<n>pool<k>x<k>', the one pooling factor of the design (pool count and window always change together).
    Only the core() runs unless everything (the all-runs tables)."""
    if archive:
        df = load(ARCHIVE)
        # the old runs only have the validation split; no early-best mark: it is about the current recipe's cosine-to-0
        # SGD, and the previews stay as they were
        df = df.assign(fp32=df.fp32_val, qat=df.qat_raw, int8=df.int8_raw, early_best=False)
        facts = LEGACY
    else:
        from ml.model_registrations import CELL_FACTORS
        df, facts = load(*RUNS), CELL_FACTORS
    if df.empty:
        return df
    families = load_config("experiments/phase_11_families.yaml")["models"]
    df = df[df.key.isin(facts) | df.key.isin(families)]
    assert not df.duplicated(["key", "seed"]).any(), df[df.duplicated(["key", "seed"], keep=False)][["exp", "key", "seed"]]
    f = pd.DataFrame([{"key": k, **v} for k, v in facts.items()])
    df = df.merge(f, on="key", how="left")
    df["family"] = df.family.fillna("família")
    for c in ("bn", "dropout", "pretrained", "approx"):  # bool, so ~ / query("not bn") negate (object dtype would not)
        df[c] = df[c].fillna(False).astype(bool) if c in df else False
    df["hk"] = np.where(df.family == "família", "família", np.where(df.dropout, "fcdrop", df["head"]))
    df["pooling"] = [f"{n:.0f}pool{k:.0f}x{k:.0f}" if pd.notna(k) else None for n, k in zip(df.pool_count, df.pool_kernel)]
    return df if archive or everything else core(df)


def pareto_front(cost, acc):
    """Mask of the points no other point beats on both axes (cost no higher and accuracy higher)."""
    cost, acc = np.asarray(cost, float), np.asarray(acc, float)
    mask, best = np.zeros(len(cost), bool), -np.inf
    for i in np.lexsort((-acc, cost)):  # by cost; at equal cost the most accurate first
        if acc[i] > best:
            mask[i], best = True, acc[i]
    return mask


def plain_log(ax, *axes):
    """Log scale with plain tick numbers (1, 2, 5 per decade) instead of 2x10^0."""
    for a in axes:
        getattr(ax, f"set_{a}scale")("log")
        axis = getattr(ax, f"{a}axis")
        axis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(mticker.NullFormatter())


def color(r):
    return KCOLOR.get(r.kernels, NEUTRAL) if r.family != "família" else NEUTRAL


GEO_LABEL = {(2, 2, 2): "64px", (4, 3, 3): "layout original"}


def cell_label(f, skip=()):
    """kernel · geometry · head · Dropout · BN · pretraining of one cell (a factor dict, row or Series), leaving out the
    factors in skip -- with skip = the factor a matched pair varies, what its two cells share. A family net: its name."""
    g = f.get if hasattr(f, "get") else (lambda k, d=None: getattr(f, k, d))
    if g("family") == "família":
        return g("key").removeprefix("alexnet_")
    s, pk, pn = (int(g(k)) for k in ("stride", "pool_kernel", "pool_count"))
    tag = FAMILY_TAG.get(g("family"), "")
    geo = ("" if tag else f"{pn} pools {pk}×{pk}" if "stride" in skip else f"conv1 stride {s}" if "pooling" in skip
           else GEO_LABEL.get((s, pk, pn), f"conv1 s{s} · {pn} pools {pk}×{pk}"))
    tags = [tag + ("" if "kernels" in skip else " " * bool(tag) + KLABEL[g("kernels")] + "†" * bool(g("approx", False))), geo,
            "" if "head" in skip else {"gap": "GAP", "fc": "FC"}[g("head")], "Dropout" * (bool(g("dropout")) and "dropout" not in skip),
            "" if "bn" in skip or tag else ("BN" if g("bn") else "sem BN"), "pré-treino" * (bool(g("pretrained")) and "pretrained" not in skip)]
    return " · ".join(t for t in tags if t)


short = cell_label


def seed42(df):
    return df[df.seed == 42]


def grid_cells(df, family="alexnet", geom=(2, 2, 2)):
    """The from-scratch cells of one geometry with each head as the reference nets have it -- GAP, or FC with AlexNet's
    Dropout 0.5 (since 2026-10-07; the FC cells without Dropout are only the Dropout contrast): the main factorial's slice."""
    g = seed42(df)
    return g[(g.family == family) & (g.stride == geom[0]) & (g.pool_kernel == geom[1]) & (g.pool_count == geom[2])
             & ((g["head"] == "gap") | g.dropout.astype(bool)) & ~g.pretrained.astype(bool)]


def baseline(df):
    """AlexNet trained from scratch, torchvision's layout (with Dropout): the reference line of every figure."""
    g = seed42(df)
    b = g[(g.family == "alexnet") & (g.kernels == "k11-5-3") & (g.stride == 4) & (g.pool_kernel == 3) & g.dropout.astype(bool)
          & ~g.pretrained.astype(bool) & ~g.bn.astype(bool)]
    return b.iloc[0] if len(b) else None


def legend_kernels(kernels):
    return [Patch(color=KCOLOR[k], label=KLABEL[k]) for k in KCOLOR if k in set(kernels)]


def base_ring(ax, x, y):
    """The big black ring that marks the AlexNet original (baseline) wherever it is a point."""
    ax.scatter(x, y, s=320, facecolors="none", edgecolors="k", lw=1.5, zorder=4)


BASE_HANDLE = Line2D([], [], marker="o", mfc="none", mec="k", mew=1.5, ls="", ms=15, label=BASE_LABEL)
VGG_HANDLE = Line2D([], [], marker="o", mfc="white", mec="k", mew=1.2, ls="", ms=8, label="contorno preto = VGG16")


def vgg_edge(r):
    """scatter() kwargs of figures 13/17: a VGG16 point gets a black outline, the AlexNet ones a thin white one."""
    return dict(edgecolors="k", lw=1.2) if r.family == "vgg16" else dict(edgecolors="white", lw=0.5)


def fig_pareto(df, figs, note):
    d = seed42(df).dropna(subset=["int8"])
    costs = [("macs_eff_m", "MACs por imagem (milhões, escala log)"), ("int8_eff_mb", "Tamanho do modelo INT8 (MB, escala log)")]
    # the front's nets get a number (both panels share it) and a key beside the plot: names on the points overlapped
    on_front = np.logical_or.reduce([pareto_front(d[x], d.int8) for x, _ in costs])
    num = {i: n + 1 for n, i in enumerate(d[on_front].sort_values("int8").index)}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, (x, xlabel) in zip(axes, costs):
        for r in d.itertuples():
            ax.scatter(getattr(r, x), r.fp32, marker=HMARK[r.hk], s=28, color=color(r), alpha=0.22, edgecolors="none", zorder=2)
            ax.scatter(getattr(r, x), r.int8, marker=HMARK[r.hk], s=75, color=color(r), zorder=3, **vgg_edge(r))
            if r.early_best:
                early_ring(ax, getattr(r, x), r.int8)
        front = d[pareto_front(d[x], d.int8)].sort_values(x)
        ax.step(front[x], front.int8, where="post", color=TEXT_PRIMARY, lw=1, ls="--", zorder=1)
        for r in front.itertuples():
            n = num[r.Index]  # odd left, even right: neighbours on the front don't print over each other
            ax.annotate(str(n), (getattr(r, x), r.int8), xytext=(-5, 4) if n % 2 else (5, 4), ha="right" if n % 2 else "left",
                        textcoords="offset points", fontsize=8, weight="bold", zorder=5)
        b = baseline(df)
        if b is not None:
            base_ring(ax, b[x], b.int8)
        plain_log(ax, "x")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Top-1 INT8 (%)")
    handles = (legend_kernels(d.kernels)
               + [Line2D([], [], marker=m, color=TEXT_SECONDARY, ls="", ms=8, label=HLABEL[h]) for h, m in HMARK.items() if h in set(d.hk)]
               + [Line2D([], [], marker="o", color=TEXT_SECONDARY, alpha=0.22, ls="", ms=6, label="marcador claro = FP32 da mesma rede"),
                  VGG_HANDLE,
                  Line2D([], [], color=TEXT_PRIMARY, ls="--", label="fronteira de Pareto (nenhuma rede é mais barata e melhor)"),
                  BASE_HANDLE]
               + ([early_handle("ring")] if d.early_best.any() else []))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)
    fig.legend(handles=[Line2D([], [], ls="", label=f"{n}  {short(d.loc[i])}") for i, n in num.items()], loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=8.5, handlelength=0, title="redes na fronteira (número no gráfico)", title_fontsize=9)
    fig.suptitle("Acurácia × custo de todas as redes da Fase 11: as redes da fronteira são as que valem a pena\n" + note, fontsize=11)
    fig.tight_layout()
    savefig(fig, "13_pareto_accuracy_cost.png", figs)


def fig_grid(df, figs, note, bands):
    d = grid_cells(df)
    cols = [("gap", False), ("gap", True), ("fc", False), ("fc", True)]
    get = lambda k, h, bn, m: d[(d.kernels == k) & (d["head"] == h) & (d.bn.astype(bool) == bn)][m].mean()  # noqa: E731
    fp32 = np.array([[get(k, h, bn, "fp32") for h, bn in cols] for k in K4])
    int8 = np.array([[get(k, h, bn, "int8") for h, bn in cols] for k in K4])
    early = np.array([[get(k, h, bn, "early_best") for h, bn in cols] for k in K4]) > 0  # NaN (no run) -> False
    star = early | early[0]  # a Δ against an early-best reference is marked too
    dfp, d8 = fp32 - fp32[0], int8 - int8[0]
    # each column against its own regime's floor (no BN / BN); NaN while a floor's replicates run -> nothing greyed
    braw = np.array([band_for(bands, "alexnet", bn)["fp32"] for _, bn in cols])
    bcol = np.nan_to_num(braw)
    btxt = lambda b: f"±{b:.1f} pp" if np.isfinite(b) else "pendente"  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    lim = np.nanmax(np.abs(dfp[1:])) if np.isfinite(dfp[1:]).any() else 1
    shown = np.where(np.abs(dfp) < bcol, np.nan, dfp)  # inside the seed band: no colour, it is not a difference
    shown[0] = np.nan  # the reference row: no colour in every column (it was white where a floor was still pending)
    seq, div = (LinearSegmentedColormap.from_list(n, c) for n, c in [("acc", ["white", GREEN]), ("delta", [RED, "white", BLUE])])
    norms = [Normalize(np.nanmin(fp32), np.nanmax(fp32)) if np.isfinite(fp32).any() else Normalize(), Normalize(-lim, lim)]
    for ax, m, cmap, norm in [(axes[0], fp32, seq, norms[0]), (axes[1], shown, div, norms[1])]:
        ax.imshow(np.ma.masked_invalid(m), cmap=cmap, norm=norm, aspect="auto")
        ax.set_facecolor("#f0f0f0")
        # white text where the cell itself is dark (its luminance), not past a fixed share of the scale
        on_dark = np.vectorize(lambda v: np.isfinite(v) and np.dot(cmap(norm(v))[:3], [0.2126, 0.7152, 0.0722]) < 0.5)(m)
        for i in range(len(K4)):
            for j in range(len(cols)):
                if np.isnan(fp32[i, j]):
                    txt, c = "pendente", NEUTRAL
                elif ax is axes[0]:
                    txt, c = f"FP32 {fp32[i, j]:.1f}\nINT8 {int8[i, j]:.1f}" + ("*" if early[i, j] else ""), "k"
                elif i == 0:
                    txt, c = "referência", TEXT_SECONDARY
                elif np.isnan(dfp[i, j]):
                    txt, c = "sem referência", NEUTRAL
                else:
                    small = abs(dfp[i, j]) < bcol[j]
                    txt = (f"FP32 {dfp[i, j]:+.1f} pp" + (" ≈ ruído" if small else "")
                           + (f"\nINT8 {d8[i, j]:+.1f} pp" + ("*" if star[i, j] else "") if np.isfinite(d8[i, j]) else ""))
                    c = NEUTRAL if small else "k"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if c == "k" and on_dark[i, j] else c)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{'GAP' if h == 'gap' else 'FC + Dropout'}\n{'com BN' if bn else 'sem BN'}" for h, bn in cols])
        approx = {k for k in K4 if d[d.kernels == k].approx.any()}
        ax.set_yticks(range(len(K4)))
        ax.set_yticklabels([KLABEL[k] + "†" * (k in approx) for k in K4])
        ax.grid(False)
    axes[0].set_title("Top-1 (%) de cada célula", fontsize=10)
    axes[1].set_title(f"Diferença para o 11-5-3-3-3 da mesma coluna: azul = melhor, vermelho = pior\n"
                      f"cinza = FP32 dentro do ruído entre seeds ({btxt(braw[1])} com BN, {btxt(braw[0])} sem BN)", fontsize=10)
    b = baseline(df)
    fig.suptitle("Fatorial principal: kernel × cabeça × BatchNorm no layout 64px (conv1 stride 2, 2 max-pools 2×2, mapa 8×8)\n" + note
                 + (f"\npara comparar, o AlexNet original do zero (layout original, FC + Dropout, sem BN): FP32 {b.fp32:.1f} · INT8 {b.int8:.1f}"
                    if b is not None else "")
                 + ("\n* = " + EARLY_LABEL if early.any() else ""), fontsize=11)
    fig.tight_layout()
    savefig(fig, "14_main_factorial_grid.png", figs)


def fig_geometry(df, figs, note, pending=True):
    """FC + Dropout only: GAP was trained at the two layouts alone (figures 01/02 have it). pending: name the design's
    cells of this figure that have not run yet (off for --archive, whose cells are not the design's)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    xs = list(GEOMS)
    b = baseline(df)
    ringed = False
    for ax, m in zip(axes, ["fp32", "int8"]):
        for k in K4:
            runs = [grid_cells(df, geom=g).query("kernels == @k and head == 'fc' and ~bn") for g in xs]
            pts = [(i, r[m].mean()) for i, r in enumerate(runs) if np.isfinite(r[m].mean())]
            if not pts:
                continue
            # ponytail: no value labels -- the left column clusters them; the numbers are in figure 14 and the tables
            ax.plot(*zip(*pts), marker="s", color=KCOLOR[k], lw=2.2, ms=7)
            for i, v in pts:
                if m == "int8" and runs[i].early_best.any():
                    early_ring(ax, i, v)
                    ringed = True
        reference_lines(ax, df, stage=m, text="left")
        if b is not None and np.isfinite(b[m]):
            base_ring(ax, xs.index((4, 3, 3)), b[m])  # the baseline is this figure's 11-5-3-3-3 at the original layout
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([GEOMS[g] for g in xs], fontsize=9)
        ax.set_title({"fp32": "FP32", "int8": "INT8"}[m], fontsize=10)
    axes[0].set_ylabel("Top-1 (%)")
    missing = {}
    if pending:
        from scripts.phase11.factor_effects import design_cells  # imported here: factor_effects imports this module
        c = design_cells()
        c = c[(c.family == "alexnet") & (c["head"] == "fc") & c.dropout & ~c.bn & ~c.pretrained & c.kernels.isin(K4)]
        for r in c[~c.cell.isin(seed42(df).key)].itertuples():
            missing.setdefault(GEOMS[(r.stride, r.pool_kernel, r.pool_count)].replace("\n", " "), []).append(KLABEL[r.kernels])
    handles = legend_kernels(K4) + ([BASE_HANDLE] if b is not None else []) + ([early_handle("ring")] if ringed else [])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)
    fig.suptitle("O custo do kernel depende da geometria? Mesmas redes (AlexNet sem BN, cabeça FC + Dropout), só muda como a rede reduz o mapa\n"
                 "da esquerda (layout 64px) para a direita (layout do AlexNet original); linhas não paralelas = interação kernel × geometria\n"
                 + note + "".join(f"\nainda na fila: {' e '.join(ks)} em {g}" for g, ks in missing.items()), fontsize=11)
    fig.tight_layout()
    savefig(fig, "15_kernel_by_geometry.png", figs)


def worst_input_ratio(r):
    """max / p99.9 of each Conv/Linear input after the first (the image's range is fixed): how far a rare peak stretches
    the per-tensor INT8 scale past the bulk of the values. The worst layer, from the run's *_layer_stats.json."""
    p = r.results / f"{r.key}_layer_stats.json"
    if not p.exists():
        return np.nan
    acts = [v["activation_in"] for v in list(json.loads(p.read_text()).values())[1:]]
    return max((a["max"] / a["p999"] for a in acts if a.get("p999")), default=np.nan)


def kernel_groups(d):
    """[((family, kernel), runs)] of figures 16/20: AlexNet kernels in KCOLOR order, VGG16, ResNet-18, the families last;
    only the groups that have a run."""
    order = [("alexnet", k) for k in KCOLOR] + [("vgg16", k) for k in K4] + [("resnet18", k) for k in RESNET_K] + [("família", None)]
    groups = [(g, d[(d.family == g[0]) & ((d.kernels == g[1]) if g[1] else True)]) for g in order]
    return [(g, runs) for g, runs in groups if len(runs)]


def group_ticks(ax, groups):
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([(FAMILY_TAG[fam] + "\n" if fam in FAMILY_TAG else "") + (KLABEL[k] if k else "famílias") for (fam, k), _ in groups],
                       fontsize=9)


def run_point(ax, x, y, r):
    """One run as figures 16/20 draw it: kernel colour, head marker, filled = BN, rings for early best and baseline."""
    c = color(r)
    ax.scatter(x, y, marker=HMARK[r.hk], s=60, zorder=3, **(dict(color=c, edgecolors="white") if r.bn or r.family == "família"
                                                         else dict(facecolors="none", edgecolors=c, lw=1.3)))
    if r.early_best:
        early_ring(ax, x, y)
    if r.key == BASE_KEY:
        base_ring(ax, x, y)


def run_handles(d):
    """The legend of run_point's encoding, for the runs in d."""
    return (legend_kernels(d.kernels)
            + [Line2D([], [], marker=m, color=TEXT_SECONDARY, ls="", ms=8, label=HLABEL[h]) for h, m in HMARK.items() if h in set(d.hk)]
            + [Line2D([], [], marker="o", color=TEXT_SECONDARY, ls="", ms=8, label="cheio = com BN"),
               Line2D([], [], marker="o", mfc="none", mec=TEXT_SECONDARY, ls="", ms=8, label="vazado = sem BN")]
            + ([BASE_HANDLE] if (d.key == BASE_KEY).any() else []) + ([early_handle("ring")] if d.early_best.any() else []))


def fig_quant(df, figs, note):
    d = seed42(df).dropna(subset=["fp32", "int8"]).copy()
    d["dq"] = d.int8 - d.fp32
    d["ratio"] = [worst_input_ratio(r) for r in d.itertuples()]
    groups = kernel_groups(d)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    rng = np.random.default_rng(0)
    for i, (_, g) in enumerate(groups):
        for r in g.itertuples():
            for ax, x in [(axes[0], i + rng.uniform(-0.2, 0.2)), (axes[1], r.ratio)]:
                run_point(ax, x, r.dq, r)
    group_ticks(axes[0], groups)
    # without the early-best runs: their INT8 - FP32 is QAT's extra training (analyze_geometry.EARLY_FRAC), not the
    # quantization this panel asks about -- they are still drawn, ringed
    q = d[~d.early_best & d.ratio.notna()]
    rho = q[["ratio", "dq"]].corr(method="spearman").iloc[0, 1]
    plain_log(axes[1], "x")
    axes[1].set_xlabel("pior razão max / p99,9 entre as entradas das camadas (escala log)")
    axes[1].set_title(f"Ativações com picos raros perdem mais? Spearman ρ = {rho:+.2f}, n = {len(q)}\n"
                      "(sem as redes de anel cinza: o ganho delas no INT8 não é quantização)", fontsize=10)
    axes[0].set_title("INT8 − FP32 por kernel / família (negativo = perde no INT8)", fontsize=10)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("INT8 − FP32 (pp; negativo = perde no INT8)")
    fig.legend(handles=run_handles(d), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=9)
    fig.suptitle("Robustez à quantização: quanto cada rede perde do FP32 ao INT8 real, e um porquê possível\n" + note, fontsize=11)
    fig.tight_layout()
    savefig(fig, "16_quantization_robustness.png", figs)


def flips(r):
    """(% of the test images right in FP32 and wrong in INT8, % wrong -> right) of one run, from its test logits."""
    from scripts.phase11.factor_effects import _test_hits  # imported here: factor_effects imports this module
    a, b = (_test_hits(r.results, r.key, s) for s in ("fp32", "int8"))
    return (100 * np.mean(a & ~b), 100 * np.mean(~a & b)) if a is not None and b is not None else (np.nan, np.nan)


def fig_flips(df, figs, note):
    """Hooker et al. (2019): a compressed net can match its FP32 accuracy while changing its answer on many images. Left:
    the % of test images whose top-1 changed sides (right <-> wrong) under INT8, per kernel; right: the two directions,
    the diagonal = no net change (INT8 - FP32 = 0)."""
    d = seed42(df).copy()
    d[["r2w", "w2r"]] = [flips(r) for r in d.itertuples()] if len(d) else np.empty((0, 2))
    d = d.dropna(subset=["r2w"])
    groups = kernel_groups(d)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    rng = np.random.default_rng(0)
    for i, (_, g) in enumerate(groups):
        for r in g.itertuples():
            run_point(axes[0], i + rng.uniform(-0.2, 0.2), r.r2w + r.w2r, r)
            run_point(axes[1], r.r2w, r.w2r, r)
    group_ticks(axes[0], groups)
    axes[0].set_ylabel("% das imagens de teste que mudam de lado no INT8")
    axes[0].set_title("Quantas imagens o INT8 troca (certo ↔ errado), por kernel", fontsize=10)
    if len(d):
        lim = 1.05 * max(d.r2w.max(), d.w2r.max())
        axes[1].plot([0, lim], [0, lim], ls="--", color=TEXT_PRIMARY, lw=1, zorder=1)
        axes[1].set(xlim=(0, lim), ylim=(0, lim), aspect="equal")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "pendente", transform=ax.transAxes, ha="center", color=NEUTRAL, fontsize=12)
    axes[1].set_xlabel("% certo no FP32 → errado no INT8")
    axes[1].set_ylabel("% errado no FP32 → certo no INT8")
    axes[1].set_title("As duas direções; na diagonal tracejada o saldo é zero\n(acima = o INT8 ganha no saldo)", fontsize=10)
    fig.legend(handles=run_handles(d), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=9)
    fig.suptitle("A mesma acurácia não quer dizer as mesmas respostas: quantas imagens de teste mudam de certo para errado (e vice-versa)\n"
                 "do FP32 para o INT8 real (Hooker et al. 2019)\n" + note, fontsize=11)
    fig.tight_layout()
    savefig(fig, "20_int8_prediction_flips.png", figs)


def fig_latency(df, figs, note):
    """One machine only (GPU + CPU: batch-1 latency on a GPU is launch-bound, so the host CPU moves all three panels):
    latency is compared within a machine (CLAUDE.md), so the runs measured elsewhere wait for the one-machine
    re-measurement. The dashed line is the AlexNet original's ms per MAC: above it = slower per MAC than the baseline."""
    d = seed42(df)
    ref = d.machine.mode().iloc[0] if len(d) else None
    out = d[d.machine != ref]
    if len(out):
        print("17: " + str(ref) + "; left out: " + "; ".join(f"{m} ({n})" for m, n in out.machine.value_counts().items()))
    d = d[d.machine == ref]
    gpu = next(iter(d.gpu.dropna()), "?")
    b = baseline(df)
    b = b if b is not None and b.machine == ref else None
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    axes[2].sharey(axes[1])  # FP32 and INT8 on the same CPU, one scale
    for ax, m, title in [(axes[0], "lat_gpu", f"FP32 na GPU ({gpu}), batch 1"), (axes[1], "lat_cpu", "FP32 na CPU, batch 1"),
                         (axes[2], "lat_int8", "INT8 na CPU, batch 1")]:
        for r in d.dropna(subset=[m]).itertuples():
            ax.scatter(r.macs_eff_m, getattr(r, m), marker=HMARK[r.hk], s=55, zorder=3, color=color(r), **vgg_edge(r))
        plain_log(ax, "x", "y")
        if b is not None and pd.notna(b[m]):
            base_ring(ax, b.macs_eff_m, b[m])
        ax.set_xlabel("MACs por imagem (milhões, log)")
        ax.set_title(title, fontsize=10)
    for ax, m in zip(axes, ["lat_gpu", "lat_cpu", "lat_int8"]):  # after every point: the shared CPU scale must not grow to fit the line
        if b is not None and pd.notna(b[m]):
            ax.autoscale(False)
            x = np.array(ax.get_xlim())
            ax.plot(x, b[m] * x / b.macs_eff_m, ls="--", color=TEXT_PRIMARY, lw=1.2, zorder=2)
    axes[0].set_ylabel("Latência por imagem (ms, log)")
    handles = (legend_kernels(d.kernels)
               + [Line2D([], [], marker=m, color=TEXT_SECONDARY, ls="", ms=8, label=HLABEL[h]) for h, m in HMARK.items() if h in set(d.hk)]
               + ([VGG_HANDLE] if (d.family == "vgg16").any() else [])
               + ([BASE_HANDLE, Line2D([], [], ls="--", color=TEXT_PRIMARY, lw=1.2,
                                       label="tempo se crescesse junto com os MACs, a partir do AlexNet original")] if b is not None else []))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)
    fig.suptitle("A latência acompanha os MACs? Cada ponto é uma rede; acima da linha tracejada = mais lenta por MAC que o AlexNet "
                 "original, abaixo = mais rápida\n" + note + f" · {len(d)} redes medidas em {ref}"
                 + (f"\n{len(out)} redes medidas em outras máquinas ficam de fora: latência só se compara na mesma máquina"
                    if len(out) else ""), fontsize=11)
    fig.tight_layout()
    savefig(fig, "17_latency_vs_macs.png", figs)


def curve(r, stage="fp32"):
    """(epoch, train top-1, val top-1) of one stage (fp32 or qat), from the log Trainer writes next to the run's results/."""
    p = r.results.parent / "logs" / f"{'qat_' * (stage == 'qat')}{r.key}.log"
    if not p.exists():
        return None
    ep = {int(m[1]): (float(m[3]), float(m[5])) for m in LOG_EPOCH.finditer(p.read_text(errors="ignore"))}  # resumed: last wins
    return (sorted(ep), [ep[e][0] for e in sorted(ep)], [ep[e][1] for e in sorted(ep)]) if ep else None


CURVE_PANELS = [("gap", False), ("gap", True), ("fc", False), ("fc", True)]


def fig_curves(df, figs, note):
    d = grid_cells(df)
    b = baseline(df)
    cb = curve(b) if b is not None else None  # the baseline's validation curve, in black in every panel
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    shown = set()
    for ax, (head, bn) in zip(axes.flat, CURVE_PANELS):
        drawn = False
        for k in K4:
            for r in d[(d.kernels == k) & (d["head"] == head) & (d.bn.astype(bool) == bn)].itertuples():
                c = curve(r)
                if c is None:
                    continue
                drawn = True
                shown.add(k)
                ax.plot(c[0], c[2], color=KCOLOR[k], lw=1.6)
                ax.plot(c[0], c[1], color=KCOLOR[k], lw=0.8, ls="--", alpha=0.6)
        if cb is not None:
            ax.plot(cb[0], cb[2], color="k", lw=1.2)
        if not drawn:
            ax.text(0.5, 0.5, "pendente", transform=ax.transAxes, ha="center", color=NEUTRAL, fontsize=12)
        # grid_cells' FC level is the reference nets' head, FC + Dropout (as figure 19 says)
        ax.set_title(f"{'GAP' if head == 'gap' else 'FC + Dropout'} · {'com BN' if bn else 'sem BN'}", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Época (FP32)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-1 (%)")
    # one legend for the four panels (it was repeated in each)
    fig.legend(handles=legend_kernels(shown)
               + ([Line2D([], [], color="k", lw=1.2, label=f"{BASE_LABEL}, layout original (validação)")] if cb is not None else [])
               + [Line2D([], [], color=TEXT_SECONDARY, lw=1.6, label="validação (split 90/10)"),
                  Line2D([], [], color=TEXT_SECONDARY, lw=0.8, ls="--", label="treino (com augmentation)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)
    fig.suptitle("Curvas de treino do fatorial principal (layout 64px): convergiu? quanto o treino se afasta da validação?\n" + note,
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "18_training_curves.png", figs)


def fig_qat_curves(df, figs, note):
    """The QAT stage of figure 18's nets: val top-1 with fake-quant per epoch, the FP32 best it starts from (dotted) and
    the real INT8 (diamond) at the QAT epoch it is converted from -- QAT keeps its best val epoch, not its last -- all
    three on the validation split. No train curve: at ~20 pp above
    val it would flatten the 0.1-3 pp the QAT moves; each panel zooms on its own nets (no sharey)."""
    d = grid_cells(df)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    ringed = False
    for ax, (head, bn) in zip(axes.flat, CURVE_PANELS):
        drawn = False
        for k in K4:
            for r in d[(d.kernels == k) & (d["head"] == head) & (d.bn.astype(bool) == bn)].itertuples():
                c = curve(r, "qat")
                if c is None:
                    continue
                drawn, x8 = True, r.qat_best_ep + 1 if pd.notna(r.qat_best_ep) else c[0][-1]  # log epochs are 1-based
                ax.plot(c[0], c[2], color=KCOLOR[k], lw=1.6, label=KLABEL[k] + "†" * r.approx)
                ax.axhline(r.fp32_val, color=KCOLOR[k], ls=":", lw=1.2)
                ax.scatter(x8, r.int8_raw, marker="D", s=45, color=KCOLOR[k], edgecolors="k", lw=0.8, zorder=5)
                if r.early_best:
                    early_ring(ax, x8, r.int8_raw)
                    ringed = True
        if not drawn:
            ax.text(0.5, 0.5, "pendente", transform=ax.transAxes, ha="center", color=NEUTRAL, fontsize=12)
        ax.set_title(f"{'GAP' if head == 'gap' else 'FC + Dropout'} · {'com BN' if bn else 'sem BN'}", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Época do QAT")
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-1 na validação (%)")
    fig.legend(handles=legend_kernels(K4) + [Line2D([], [], color=TEXT_SECONDARY, lw=1.6, label="validação durante o QAT (com fake-quant)"),
                        Line2D([], [], color=TEXT_SECONDARY, ls=":", lw=1.2, label="melhor FP32: o checkpoint de onde o QAT parte"),
                        Line2D([], [], marker="D", color=TEXT_SECONDARY, ls="", ms=7, label="INT8 real, convertido da melhor época do QAT (desenhado nela)")]
               + ([early_handle("ring")] if ringed else []), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)
    fig.suptitle("O que o QAT faz com cada rede do fatorial principal (layout 64px): parte do melhor FP32, treina 50 épocas\n"
                 "simulando o INT8; a melhor época vira o INT8 real (losango); acima do pontilhado = o QAT ganhou acurácia\n" + note,
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "19_qat_curves.png", figs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", action="store_true", help="draw the superseded AdamW-recipe runs into the preview dir")
    archive = ap.parse_args(argv).archive
    apply_report_style()
    df = frame(archive)
    if df.empty:
        print("no Phase 11 run has a summary yet")
        return
    figs, note = (PREVIEW, PREVIEW_NOTE) if archive else (FIGS, LIVE_NOTE)
    bands = {REFERENCE_REGIME: noise_band(df, LEGACY_NOISE)} if archive else noise_bands(df)
    print(f"{len(df)} runs, noise bands (pp): {bands}")
    fig_pareto(df, figs, note)
    fig_grid(df, figs, note, bands)
    fig_geometry(df, figs, note, pending=not archive)
    fig_quant(df, figs, note)
    fig_flips(df, figs, note)
    fig_latency(df, figs, note if archive else "seed 42 · latência por imagem, batch 1")
    curves_note = note if archive else "seed 42 · curvas no split de validação (10% do train), não no test set"
    fig_curves(df, figs, curves_note)
    fig_qat_curves(df, figs, curves_note)


if __name__ == "__main__":
    main()
