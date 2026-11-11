"""Phase 13 -- the hybrids (docs/logs/PHASE13_LOG.md), an extra study outside Phase 11: do the techniques that worked there
add up? A factorial on the 64px GAP + BN AlexNet: body (5 convs / two per stage / two per stage as a residual block) x conv1
(the body's kernel / 11x11) x body kernel (3x3 / 2x2) x conv1 stride (2 / 1) = 24 cells. The 6 trained in Phase 11 (stride 2,
5 convs or two per stage) are read from its runs as references, never retrained; the 18 others are phase_13_hybrids. Phase
11's figures and tables never read these runs.

    01_hybrids.png   one panel per conv1 stride; per conv1 the three bodies, 3x3 and 2x2 side by side, FP32 / INT8, seed 42;
                     a cell's other seeds (43 / 44) as small FP32 dots beside it
    02_caminho.png   the article's path in one figure: from the unadapted AlexNet to the best hybrid, one factor per step
                     (PATH), FP32 / INT8 averaged over every seed a cell has, MACs under each step
    contrasts.csv    every matched pair of the 24 cells (one factor changed, B - A), seed 42
    contrasts_by_seed.csv  the same pairs at every seed where both cells ran (a `seed` column; 42 = contrasts.csv)
    all_runs.csv     every run of the 24 cells (the Phase 11 references with their replicates), VGG16's reference and the PATH cells

Phase 11's conventions: test top-1, INT8 hollow, early-best ring.

    python -m scripts.phase13.hybrids
"""
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from ml.plotting import BLUE, NEUTRAL, TEXT_SECONDARY, apply_report_style
from scripts.phase11.analyze_geometry import ROOT, RUNS, band_for, early_handle, early_ring, load, noise_bands, savefig
from scripts.phase11.design_figures import KCOLOR, LIVE_NOTE, frame, legend_kernels

FIGS = ROOT / "results/figures_generated/phase_13_hybrids"
TABLES = ROOT / "results/phase_13_hybrids"
BODIES = [("", "corpo\nde 5"), ("stacked", "2 por\nestágio"), ("residual", "2 por\nestágio\n+ atalho")]  # short: \textwidth, 12 ticks
VGG_REF = "vgg16_k3x3_stride1_5pool2x2_map2_gap_bn"  # the reference line: VGG16 3x3 + BN + GAP, Phase 11
# (cell, what changed to reach it): each step changes one factor of the one before
PATH = [("alexnet_k11-5-3_stride4_3pool3x3_map1_fcdrop_nobn", "AlexNet original\nsem adaptar"),
        ("alexnet_k11-5-3_stride2_2pool2x2_map8_fcdrop_nobn", "layout\n64 px"),
        ("alexnet_k11-5-3_stride2_2pool2x2_map8_fcdrop_bn", "+ BN"),
        ("alexnet_k11-5-3_stride2_2pool2x2_map8_gap_bn", "FC → GAP"),
        ("alexnet_k2x2_stride2_2pool2x2_map8_gap_bn", "kernels\n→ 2×2"),
        ("alexnet_k2x2stacked_stride2_2pool2x2_map8_gap_bn", "dois por\nestágio"),
        ("alexnet_k2x2stacked_stride1_2pool2x2_map16_gap_bn", "stride da\nconv1 → 1")]


def cell(k, big, body, s):
    """The registered name (ml/model_registrations.py) of the grid cell: body kernel k, 11x11 conv1 if big, conv1 stride s."""
    return f"alexnet_{f'k11-{k}' if big else f'k{k}x{k}'}{body}_stride{s}_2pool2x2_map{16 if s == 1 else 8}_gap_bn"


GRID = {(k, big, body, s): cell(k, big, body, s) for k, big, (body, _), s in product((3, 2), (False, True), BODIES, (2, 1))}
# (label, position in the GRID key, level A, level B): every matched pair = two cells equal but in that position
CHANGES = [("corpo: 5 convs → 2 por estágio", 2, "", "stacked"), ("corpo: 2 por estágio → + atalho residual", 2, "stacked", "residual"),
           ("conv1: kernel do corpo → 11×11", 1, False, True), ("kernel do corpo: 3×3 → 2×2", 0, 3, 2),
           ("stride da conv1: 2 → 1", 3, 2, 1)]


def runs():
    """Every run of the 24 cells -- the phase_13_hybrids runs and the Phase 11 references -- VGG16's reference and the PATH
    cells."""
    df = pd.concat([load(*RUNS), load(*RUNS, prefix="phase_13_")], ignore_index=True)
    return df[df.key.isin([*GRID.values(), VGG_REF, *(k for k, _ in PATH)])] if len(df) else df


def contrasts(s42):
    v = s42.set_index("key")
    rows = []
    for label, i, a, b in CHANGES:
        for f, ka in GRID.items():
            kb = GRID[f[:i] + (b,) + f[i + 1:]]
            if f[i] == a and ka in v.index and kb in v.index:
                A, B = v.loc[ka], v.loc[kb]
                rows.append(dict(factor=label, A=ka, B=kb, fp32_A=A.fp32, fp32_B=B.fp32, d_fp32=B.fp32 - A.fp32, int8_A=A.int8,
                                 int8_B=B.int8, d_int8=B.int8 - A.int8, macs_change_pct=100 * (B.macs_eff_m / A.macs_eff_m - 1)))
    return pd.DataFrame(rows)


def contrasts_by_seed(df):
    """`contrasts` at each seed: a pair appears for a seed only if both of its cells ran at that seed."""
    parts = [contrasts(g).assign(seed=s) for s, g in df.groupby("seed")]
    return pd.concat([p for p in parts if "factor" in p], ignore_index=True)


def fig_hybrids(s42, bands, figs, reps=None):
    """One panel per conv1 stride; per conv1 (the body's kernel / 11x11) the three bodies left to right, 3x3 and 2x2 side by
    side (FP32 filled, INT8 hollow), a line through each kernel's FP32 across the bodies, VGG16 3x3 GAP + BN dashed. `reps`: the
    runs at other seeds, drawn as small FP32 dots beside their cell."""
    v = s42.set_index("key")
    extra = {k: g.fp32.tolist() for k, g in reps.groupby("key")} if reps is not None and len(reps) else {}
    kernels = ((3, -0.14), (2, 0.14))
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 4.2), sharey=True, layout="constrained")  # printed at \textwidth
    queued, ringed = 0, False
    for ax, s in zip(axes, (2, 1)):
        ticks = []
        for gi, big in enumerate((False, True)):
            for bi, (body, blabel) in enumerate(BODIES):
                x, macs = 3.6 * gi + bi, []
                for k, dx in kernels:
                    key = GRID[(k, big, body, s)]
                    if key not in v.index:
                        queued += 1
                        continue
                    r, c = v.loc[key], KCOLOR[f"k{k}x{k}"]
                    ax.scatter(x + dx, r.fp32, s=60, color=c, zorder=3)
                    ax.scatter(x + dx, r.int8, s=60, marker="D", facecolors="white", edgecolors=c, lw=1.6, zorder=3)
                    for y in extra.get(key, []):
                        ax.scatter(x + dx, y, s=18, color=c, alpha=0.5, zorder=2)
                    if r.early_best:
                        early_ring(ax, x + dx, r.int8)
                        ringed = True
                    macs.append(f"{r.macs_eff_m:.0f}")
                ticks.append((x, f"{blabel}\n" + ("\n".join(f"{m}M" for m in macs) if macs else "pendente")))
            for k, dx in kernels:  # each kernel's FP32 across the three bodies
                pts = [(3.6 * gi + bi + dx, v.loc[GRID[(k, big, body, s)], "fp32"]) for bi, (body, _) in enumerate(BODIES)
                       if GRID[(k, big, body, s)] in v.index]
                if len(pts) > 1:
                    ax.plot(*zip(*pts), color=KCOLOR[f"k{k}x{k}"], lw=1, alpha=0.6, zorder=2)
            ax.text(3.6 * gi + 1, 1.01, "conv1 11×11" if big else "conv1 = kernel do corpo", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, weight="bold")
        if VGG_REF in v.index:
            ax.axhline(v.loc[VGG_REF, "fp32"], color=NEUTRAL, ls="--", lw=1, zorder=1)
        ax.set_xticks([t for t, _ in ticks])
        ax.set_xticklabels([t for _, t in ticks], fontsize=9)
        ax.set_title(f"conv1 stride {s} (mapa final {8 if s == 2 else 16}×{8 if s == 2 else 16})", fontsize=9, pad=18)
        ax.margins(x=0.06, y=0.12)
    axes[0].set_ylabel("Top-1 (%)")
    handles = legend_kernels(["k3x3", "k2x2"]) + [
        Line2D([], [], marker="o", color=TEXT_SECONDARY, ls="", ms=8, label="FP32"),
        Line2D([], [], marker="D", mfc="white", mec=TEXT_SECONDARY, ls="", ms=8, label="INT8"),
        Line2D([], [], color=NEUTRAL, ls="--", label="VGG16 3×3 (GAP + BN)")]
    if extra:
        handles.append(Line2D([], [], marker="o", color=TEXT_SECONDARY, alpha=0.5, ls="", ms=4, label="seeds 43 e 44 (FP32)"))
    fig.legend(handles=handles + ([early_handle("ring")] if ringed else []), loc="outside lower center", ncol=3, fontsize=9)
    band = band_for(bands, "alexnet", True)["fp32"]
    fig.suptitle("Híbridos: profundidade, atalho residual, conv1 11×11 e stride 1 somam? (AlexNet, 2 max-pools 2×2, GAP + BN)\n"
                 "sob cada corpo: MACs por imagem do 3×3 e, abaixo, do 2×2\n" + LIVE_NOTE
                 + (f" · ruído entre seeds (GAP + BN): ±{band:.1f} p.p. FP32" if pd.notna(band) else "")
                 + (f"\nainda na fila: {queued} de 24 células" if queued else ""), fontsize=11)
    savefig(fig, "01_hybrids.png", figs)


def fig_path(df, figs):
    """PATH as a staircase: each cell's FP32 (filled) and INT8 (hollow), the mean over its seeds, the step's gain above it,
    MACs under it; VGG16 3x3 GAP + BN dashed. A missing cell ends the path there."""
    m = df.groupby("key").agg(fp32=("fp32", "mean"), int8=("int8", "mean"), n=("seed", "nunique"), macs=("macs_eff_m", "first"),
                              early=("early_best", "any"))
    steps = [(k, label) for k, label in PATH if k in m.index]
    if len(steps) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.7, 3.7), layout="constrained")  # printed at 0.86 \textwidth
    xs = range(len(steps))
    fp32 = [m.loc[k, "fp32"] for k, _ in steps]
    ax.step(xs, fp32, where="mid", color=BLUE, lw=1.2, alpha=0.5, zorder=2)
    ax.scatter(xs, fp32, s=70, color=BLUE, zorder=3)
    ax.scatter(xs, [m.loc[k, "int8"] for k, _ in steps], s=60, marker="D", facecolors="white", edgecolors=BLUE, lw=1.6, zorder=3)
    for x, (k, _) in enumerate(steps):
        r = m.loc[k]
        if r.early:
            early_ring(ax, x, r.int8)
        ax.text(x, max(r.fp32, r.int8) + 0.9, f"{r.fp32:.1f}%" + ("" if x == 0 else f"\n({r.fp32 - fp32[x - 1]:+.1f})"),
                ha="center", va="bottom", fontsize=9)
    if VGG_REF in m.index:
        ax.axhline(m.loc[VGG_REF, "fp32"], color=NEUTRAL, ls="--", lw=1, zorder=1)
        ax.text(0, m.loc[VGG_REF, "fp32"] + 0.3, f"VGG16 3×3 (GAP + BN): {m.loc[VGG_REF, 'fp32']:.1f}%", fontsize=9, color=TEXT_SECONDARY,
                va="bottom")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{label}\n{m.loc[k, 'macs']:.0f}M" + ("" if m.loc[k, "n"] > 1 else "\n(1 seed)") for k, label in steps],
                       fontsize=9)
    ax.set_ylabel("Top-1 no teste (%)")
    ax.set_xlabel("passo · MACs por imagem (custo efetivo)", fontsize=9)
    ax.set_ylim(min(fp32) - 4, max(fp32) + 5)
    handles = [Line2D([], [], marker="o", color=BLUE, ls="", ms=8, label="FP32"),
               Line2D([], [], marker="D", mfc="white", mec=BLUE, ls="", ms=8, label="INT8")]
    if m.loc[[k for k, _ in steps], "early"].any():
        handles.append(early_handle("ring"))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_title("Da AlexNet original à melhor rede, um fator por vez\n(média das seeds; entre parênteses, o ganho do passo em p.p.)",
                 fontsize=10)
    savefig(fig, "02_caminho.png", figs)


def main():
    apply_report_style()
    df = runs()
    if df.empty or not (df.exp == "hybrids").any():  # load() strips the phase_13_ prefix
        print("no phase_13_hybrids run yet")
        return
    TABLES.mkdir(parents=True, exist_ok=True)
    df.drop(columns="results").sort_values(["exp", "key", "seed"]).to_csv(TABLES / "all_runs.csv", index=False)
    s42 = df[df.seed == 42]
    c = contrasts(s42)
    c.to_csv(TABLES / "contrasts.csv", index=False)
    contrasts_by_seed(df).to_csv(TABLES / "contrasts_by_seed.csv", index=False)
    fig_hybrids(s42, noise_bands(frame()), FIGS, df[df.seed != 42])
    fig_path(df, FIGS)
    if len(c):
        print(c.groupby("factor", sort=False)[["d_fp32", "d_int8"]].agg(["median", "count"]).round(2).to_string())
    n = df.groupby("key").seed.nunique()
    if (n > 1).any():  # cells with replicates: mean and SD over their seeds
        print(df[df.key.isin(n[n > 1].index)].groupby("key")[["fp32", "int8"]].agg(["mean", "std", "count"]).round(2).to_string())


if __name__ == "__main__":
    main()
