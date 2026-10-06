"""Phase 11 figures 16-21 -- the design read through its factors (ml/model_registrations.py:CELL_FACTORS), not run names.
Selecting by factor is what lets one file draw the live runs and, with --archive, the superseded AdamW-recipe runs
(LEGACY maps their names onto the same factors) -- a preview of each figure's form, not its numbers: validation split,
pre-fix INT8, latencies that still timed the DataLoader, and some only-approximate twins (marked †).

    16_pareto_accuracy_cost.png     top-1 INT8 vs MACs and vs INT8 size; Pareto front labelled
    17_main_factorial_grid.png      kernel x {GAP, FC} x {no BN, BN} at the 64px layout, and the delta vs 11-5-3-3-3
    18_kernel_by_geometry.png       one line per kernel across the 4 geometries (FC; GAP at the two layouts)
    19_quantization_robustness.png  INT8 - FP32 per kernel / family, and vs each net's worst layer-input max/p99.9
    20_latency_vs_macs.png          batch-1 latency vs MACs: FP32 GPU, FP32 CPU, INT8 CPU
    21_training_curves.png          val/train top-1 per epoch at the 64px layout, {GAP, FC} x {no BN, BN}

21 reads the per-epoch logs, which only PCAD keeps -- fetch them first (analyze_geometry.convergence's rsync; add
archive_adamw_recipe/ to both paths for --archive).

    python -m scripts.phase11.design_figures [--archive]
"""
import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from configs.loader import load_config
from ml.plotting import AMBER, BLUE, GREEN, NEUTRAL, RED, TEXT_SECONDARY, apply_report_style
from scripts.phase11.analyze_geometry import ARCHIVE, FIGS, LOG_EPOCH, NOISE_KEYS, ROOT, RUNS, load, noise_band, savefig

PREVIEW = ROOT / "results/archive_adamw_recipe/figures_generated/phase_11_preview"
PREVIEW_NOTE = ("PRÉVIA — runs antigas: receita AdamW, split de validação, INT8 antes das correções de 30/09–03/10; "
                "† = só aproximadamente a célula nova")
LIVE_NOTE = "seed 42 · top-1 no test set (val oficial do Tiny ImageNet)"

K4 = ["k11-5-3", "k3x3", "k2x2", "kalt3-2"]
KLABEL = {"k11-5-3": "11-5-3-3-3", "k3x3": "3×3", "k2x2": "2×2", "kalt3-2": "alternado 3-2", "k3x3stacked": "3×3 empilhado",
          "k2x2stacked": "2×2 empilhado"}
KCOLOR = {"k11-5-3": AMBER, "k3x3": GREEN, "k2x2": BLUE, "kalt3-2": RED, "k3x3stacked": "#0b6b2e", "k2x2stacked": "#0b3d7a"}
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


def frame(archive: bool = False) -> pd.DataFrame:
    """load() + each run's factors; the families (phase_11_families.yaml) get family 'família' and no grid factors.
    pooling = '<n>pool<k>x<k>', the one pooling factor of the design (pool count and window always change together)."""
    if archive:
        df = load(ARCHIVE)
        df = df.assign(fp32=df.fp32_val, qat=df.qat_raw, int8=df.int8_raw)  # the old runs only have the validation split
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
    return df


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
    vgg = g("family") == "vgg16"
    geo = ("" if vgg else f"{pn} pools {pk}×{pk}" if "stride" in skip else f"conv1 stride {s}" if "pooling" in skip
           else GEO_LABEL.get((s, pk, pn), f"conv1 s{s} · {pn} pools {pk}×{pk}"))
    tags = ["VGG16" * vgg + ("" if "kernels" in skip else " " * vgg + KLABEL[g("kernels")] + "†" * bool(g("approx", False))), geo,
            "" if "head" in skip else {"gap": "GAP", "fc": "FC"}[g("head")], "Dropout" * (bool(g("dropout")) and "dropout" not in skip),
            "" if "bn" in skip or vgg else ("BN" if g("bn") else "sem BN"), "pré-treino" * (bool(g("pretrained")) and "pretrained" not in skip)]
    return " · ".join(t for t in tags if t)


short = cell_label


def seed42(df):
    return df[df.seed == 42]


def grid_cells(df, family="alexnet", geom=(2, 2, 2)):
    """The from-scratch, no-Dropout cells of one geometry (the main factorial's slice)."""
    g = seed42(df)
    return g[(g.family == family) & (g.stride == geom[0]) & (g.pool_kernel == geom[1]) & (g.pool_count == geom[2])
             & ~g.dropout.astype(bool) & ~g.pretrained.astype(bool)]


def baseline(df):
    """AlexNet trained from scratch, torchvision's layout (with Dropout): the reference line of every figure."""
    g = seed42(df)
    b = g[(g.family == "alexnet") & (g.kernels == "k11-5-3") & (g.stride == 4) & (g.pool_kernel == 3) & g.dropout.astype(bool)
          & ~g.pretrained.astype(bool) & ~g.bn.astype(bool)]
    return b.iloc[0] if len(b) else None


def legend_kernels(kernels):
    return [Patch(color=KCOLOR[k], label=KLABEL[k]) for k in KCOLOR if k in set(kernels)]


def fig_pareto(df, figs, note):
    d = seed42(df).dropna(subset=["int8"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, x, xlabel in [(axes[0], "macs_m", "MACs por imagem (milhões, escala log)"), (axes[1], "int8_mb", "Tamanho do modelo INT8 (MB, escala log)")]:
        for r in d.itertuples():
            ax.scatter(getattr(r, x), r.fp32, marker=HMARK[r.hk], s=28, color=color(r), alpha=0.22, edgecolors="none", zorder=2)
            ax.scatter(getattr(r, x), r.int8, marker=HMARK[r.hk], s=75, color=color(r), zorder=3,
                       edgecolors="k" if r.family == "vgg16" else "white", lw=1.2 if r.family == "vgg16" else 0.5)
        front = d[pareto_front(d[x], d.int8)].sort_values(x)
        ax.step(front[x], front.int8, where="post", color="#333333", lw=1, ls="--", zorder=1)
        for k, r in enumerate(front.itertuples()):
            ax.annotate(short(r), (getattr(r, x), r.int8), xytext=(-6, 8) if k % 2 else (6, 6), ha="right" if k % 2 else "left",
                        textcoords="offset points", fontsize=7.5,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5), zorder=5)
        b = baseline(df)
        if b is not None:
            ax.scatter(b[x], b.int8, s=320, facecolors="none", edgecolors="k", lw=1.5, zorder=4)
        plain_log(ax, "x")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Top-1 INT8 (%)   ·   transparente = o FP32 da mesma rede")
    handles = (legend_kernels(d.kernels)
               + [Line2D([], [], marker=m, color=TEXT_SECONDARY, ls="", ms=8, label=HLABEL[h]) for h, m in HMARK.items() if h in set(d.hk)]
               + [Line2D([], [], marker="o", mfc="white", mec="k", mew=1.2, ls="", ms=8, label="contorno preto = VGG16"),
                  Line2D([], [], color="#333333", ls="--", label="fronteira de Pareto (nenhuma rede é mais barata e melhor)"),
                  Line2D([], [], marker="o", mfc="none", mec="k", mew=1.5, ls="", ms=15, label="AlexNet original do zero")])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)
    fig.suptitle("Acurácia × custo de todas as redes da Fase 11: as redes da fronteira são as que valem a pena\n" + note, fontsize=11)
    fig.tight_layout()
    savefig(fig, "16_pareto_accuracy_cost.png", figs)


def fig_grid(df, figs, note, band):
    d = grid_cells(df)
    cols = [("gap", False), ("gap", True), ("fc", False), ("fc", True)]
    get = lambda k, h, bn, m: d[(d.kernels == k) & (d["head"] == h) & (d.bn.astype(bool) == bn)][m].mean()  # noqa: E731
    fp32 = np.array([[get(k, h, bn, "fp32") for h, bn in cols] for k in K4])
    int8 = np.array([[get(k, h, bn, "int8") for h, bn in cols] for k in K4])
    dfp, d8 = fp32 - fp32[0], int8 - int8[0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    lim = np.nanmax(np.abs(dfp[1:])) if np.isfinite(dfp[1:]).any() else 1
    shown = np.where(np.abs(dfp) < band["fp32"], np.nan, dfp)  # inside the seed band: no colour, it is not a difference
    lo, hi = np.nanmin(fp32), np.nanmax(fp32)
    dark = [(fp32 - lo) / (hi - lo + 1e-9) > 0.6, np.abs(shown) / lim > 0.6]  # white text on the darkest cells
    for ax, m, cmap, kw in [(axes[0], fp32, "YlGn", {}), (axes[1], shown, "RdBu", dict(vmin=-lim, vmax=lim))]:
        ax.imshow(np.ma.masked_invalid(m), cmap=cmap, aspect="auto", **kw)
        ax.set_facecolor("#f0f0f0")
        on_dark = dark[ax is axes[1]]
        for i in range(len(K4)):
            for j in range(len(cols)):
                if np.isnan(fp32[i, j]):
                    txt, c = "pendente", NEUTRAL
                elif ax is axes[0]:
                    txt, c = f"FP32 {fp32[i, j]:.1f}\nINT8 {int8[i, j]:.1f}", "k"
                elif i == 0:
                    txt, c = "referência", TEXT_SECONDARY
                elif np.isnan(dfp[i, j]):
                    txt, c = "sem referência", NEUTRAL
                else:
                    small = abs(dfp[i, j]) < band["fp32"]
                    txt = f"{dfp[i, j]:+.1f} pp" + (" ≈ ruído" if small else "") + (f"\nINT8 {d8[i, j]:+.1f}" if np.isfinite(d8[i, j]) else "")
                    c = NEUTRAL if small else "k"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if c == "k" and on_dark[i, j] else c)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{'GAP' if h == 'gap' else 'FC'}\n{'com BN' if bn else 'sem BN'}" for h, bn in cols])
        approx = {k for k in K4 if d[d.kernels == k].approx.any()}
        ax.set_yticks(range(len(K4)))
        ax.set_yticklabels([KLABEL[k] + "†" * (k in approx) for k in K4])
        ax.grid(False)
    axes[0].set_title("Top-1 (%) de cada célula", fontsize=10)
    axes[1].set_title(f"Δ FP32 vs 11-5-3-3-3 na mesma coluna (pp); cinza = dentro do ruído entre seeds (±{band['fp32']:.1f} pp)", fontsize=10)
    fig.suptitle("Fatorial principal: kernel × cabeça × BatchNorm no layout 64px (conv1 stride 2, 2 max-pools 2×2, mapa 8×8)\n" + note,
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "17_main_factorial_grid.png", figs)


def fig_geometry(df, figs, note):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    xs = list(GEOMS)
    b = baseline(df)
    for ax, m in zip(axes, ["fp32", "int8"]):
        for k in K4:
            for head, ls, lw in [("fc", "-", 2.2), ("gap", "--", 1.4)]:
                y = [grid_cells(df, geom=g).query("kernels == @k and head == @head and ~bn")[m].mean() for g in xs]
                if np.isfinite(y).sum() == 0:
                    continue
                pts = [(i, v) for i, v in enumerate(y) if np.isfinite(v)]
                # ponytail: no value labels -- the left column clusters them; the numbers are in figure 17 and the tables
                ax.plot(*zip(*pts), ls=ls, marker="o" if head == "gap" else "s", color=KCOLOR[k], lw=lw, ms=7, alpha=1 if head == "fc" else 0.7)
        if b is not None and np.isfinite(b[m]):
            ax.axhline(b[m], color="#333333", ls=":", lw=1.3, label="AlexNet original do zero (com Dropout)")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([GEOMS[g] for g in xs], fontsize=9)
        ax.set_title({"fp32": "FP32", "int8": "INT8"}[m], fontsize=10)
    axes[0].set_ylabel("Top-1 (%)")
    handles = legend_kernels(K4) + [Line2D([], [], color=TEXT_SECONDARY, marker="s", lw=2.2, label="cabeça FC (linha cheia)"),
                                    Line2D([], [], color=TEXT_SECONDARY, marker="o", ls="--", lw=1.4, label="cabeça GAP (tracejada, só nos 2 layouts)"),
                                    Line2D([], [], color="#333333", ls=":", lw=1.3, label="AlexNet original do zero (com Dropout)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)
    fig.suptitle("O custo do kernel depende da geometria? Mesmas redes (sem BN, sem Dropout), só muda como a rede reduz o mapa\n"
                 "da esquerda (layout 64px) para a direita (layout do AlexNet original); linhas não paralelas = interação kernel × geometria\n"
                 + note, fontsize=11)
    fig.tight_layout()
    savefig(fig, "18_kernel_by_geometry.png", figs)


def worst_input_ratio(r):
    """max / p99.9 of each Conv/Linear input after the first (the image's range is fixed): how far a rare peak stretches
    the per-tensor INT8 scale past the bulk of the values. The worst layer, from the run's *_layer_stats.json."""
    p = r.results / f"{r.key}_layer_stats.json"
    if not p.exists():
        return np.nan
    acts = [v["activation_in"] for v in list(json.loads(p.read_text()).values())[1:]]
    return max((a["max"] / a["p999"] for a in acts if a.get("p999")), default=np.nan)


def fig_quant(df, figs, note):
    d = seed42(df).dropna(subset=["fp32", "int8"]).copy()
    d["dq"] = d.int8 - d.fp32
    d["ratio"] = [worst_input_ratio(r) for r in d.itertuples()]
    order = [("alexnet", k) for k in KCOLOR] + [("vgg16", k) for k in K4] + [("família", None)]
    groups = [g for g in order if len(d[(d.family == g[0]) & ((d.kernels == g[1]) if g[1] else True)])]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    rng = np.random.default_rng(0)
    for i, (fam, k) in enumerate(groups):
        g = d[(d.family == fam) & ((d.kernels == k) if k else True)]
        for r in g.itertuples():
            filled = bool(r.bn) if fam != "família" else True
            c = color(r)
            for ax, x in [(axes[0], i + rng.uniform(-0.2, 0.2)), (axes[1], r.ratio)]:
                ax.scatter(x, r.dq, marker=HMARK[r.hk], s=60, zorder=3, **(dict(color=c, edgecolors="white") if filled
                                                                            else dict(facecolors="none", edgecolors=c, lw=1.3)))
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels([("VGG16\n" if fam == "vgg16" else "") + (KLABEL[k] if k else "famílias") for fam, k in groups], fontsize=9)
    rho = d[["ratio", "dq"]].corr(method="spearman").iloc[0, 1]
    plain_log(axes[1], "x")
    axes[1].set_xlabel("pior razão max / p99,9 entre as entradas das camadas (escala log)")
    axes[1].set_title(f"Ativações com picos raros perdem mais? (Spearman ρ = {rho:+.2f}, n = {d.ratio.notna().sum()})", fontsize=10)
    axes[0].set_title("ΔINT8 por kernel / família", fontsize=10)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("INT8 − FP32 (pp; negativo = perde no INT8)")
    handles = (legend_kernels(d.kernels)
               + [Line2D([], [], marker=m, color=TEXT_SECONDARY, ls="", ms=8, label=HLABEL[h]) for h, m in HMARK.items() if h in set(d.hk)]
               + [Line2D([], [], marker="o", color=TEXT_SECONDARY, ls="", ms=8, label="cheio = com BN"),
                  Line2D([], [], marker="o", mfc="none", mec=TEXT_SECONDARY, ls="", ms=8, label="vazado = sem BN")])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=9)
    fig.suptitle("Robustez à quantização: quanto cada rede perde do FP32 ao INT8 real, e um porquê possível\n" + note, fontsize=11)
    fig.tight_layout()
    savefig(fig, "19_quantization_robustness.png", figs)


def fig_latency(df, figs, note):
    d = seed42(df)
    # one machine = GPU + CPU: batch-1 latency on a GPU is launch-bound, so the host CPU moves all three panels
    ref = d.machine.mode().iloc[0] if len(d) else None
    gpu = next(iter(d[d.machine == ref].gpu.dropna()), "?")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    for ax, m, title in [(axes[0], "lat_gpu", f"FP32 na GPU ({gpu}), batch 1"), (axes[1], "lat_cpu", "FP32 na CPU, batch 1"),
                         (axes[2], "lat_int8", "INT8 na CPU, batch 1")]:
        for r in d.dropna(subset=[m]).itertuples():
            same = r.machine == ref
            ax.scatter(r.macs_m, getattr(r, m), marker=HMARK[r.hk], s=55, zorder=3,
                       **(dict(color=color(r), edgecolors="white") if same else dict(facecolors="none", edgecolors=color(r), lw=1.2)))
        plain_log(ax, "x", "y")
        ax.set_xlabel("MACs por imagem (milhões, log)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Latência por imagem (ms, log)")
    others = sorted(set(d.machine) - {ref})
    handles = (legend_kernels(d.kernels)
               + [Line2D([], [], marker=m, color=TEXT_SECONDARY, ls="", ms=8, label=HLABEL[h]) for h, m in HMARK.items() if h in set(d.hk)]
               + ([Line2D([], [], marker="o", mfc="none", mec=TEXT_SECONDARY, ls="", ms=8, label=f"vazado = medida em outra máquina ({'; '.join(others)})")]
                  if others else []))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=9)
    fig.suptitle("O MAC prevê o tempo real? Mesma conta, kernels diferentes: pontos acima da nuvem = kernel lento por MAC\n" + note,
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "20_latency_vs_macs.png", figs)


def curve(r):
    """(epoch, train top-1, val top-1) of the FP32 stage, from the log Trainer writes next to the run's results/."""
    p = r.results.parent / "logs" / f"{r.key}.log"
    if not p.exists():
        return None
    ep = {int(m[1]): (float(m[3]), float(m[5])) for m in LOG_EPOCH.finditer(p.read_text(errors="ignore"))}  # resumed: last wins
    return (sorted(ep), [ep[e][0] for e in sorted(ep)], [ep[e][1] for e in sorted(ep)]) if ep else None


def fig_curves(df, figs, note):
    d = grid_cells(df)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    for ax, (head, bn) in zip(axes.flat, [("gap", False), ("gap", True), ("fc", False), ("fc", True)]):
        drawn = False
        for k in K4:
            for r in d[(d.kernels == k) & (d["head"] == head) & (d.bn.astype(bool) == bn)].itertuples():
                c = curve(r)
                if c is None:
                    continue
                drawn = True
                ax.plot(c[0], c[2], color=KCOLOR[k], lw=1.6, label=KLABEL[k] + "†" * r.approx)
                ax.plot(c[0], c[1], color=KCOLOR[k], lw=0.8, ls="--", alpha=0.6)
        if not drawn:
            ax.text(0.5, 0.5, "pendente", transform=ax.transAxes, ha="center", color=NEUTRAL, fontsize=12)
        ax.set_title(f"{'GAP' if head == 'gap' else 'FC'} · {'com BN' if bn else 'sem BN'}", fontsize=10)
        if drawn:
            ax.legend(fontsize=8.5, loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("Época (FP32)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-1 (%)")
    fig.legend(handles=[Line2D([], [], color=TEXT_SECONDARY, lw=1.6, label="validação (split 90/10)"),
                        Line2D([], [], color=TEXT_SECONDARY, lw=0.8, ls="--", label="treino (com augmentation)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)
    fig.suptitle("Curvas de treino do fatorial principal (layout 64px): convergiu? quanto o treino se afasta da validação?\n" + note,
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "21_training_curves.png", figs)


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
    band = noise_band(df, LEGACY_NOISE if archive else NOISE_KEYS)
    band = {k: v if np.isfinite(v) else 0.0 for k, v in band.items()}
    print(f"{len(df)} runs, noise band (pp): {band}")
    fig_pareto(df, figs, note)
    fig_grid(df, figs, note, band)
    fig_geometry(df, figs, note)
    fig_quant(df, figs, note)
    fig_latency(df, figs, note)
    fig_curves(df, figs, note)


if __name__ == "__main__":
    main()
