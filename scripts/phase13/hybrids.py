"""Phase 13 -- the hybrids (docs/logs/PHASE13_LOG.md), an extra study outside Phase 11: do the techniques that worked there
add up? A factorial on the 64px GAP + BN AlexNet: body (5 convs / two per stage / two per stage as a residual block) x conv1
(the body's kernel / 11x11) x body kernel (3x3 / 2x2) x conv1 stride (2 / 1) = 24 cells. The 6 trained in Phase 11 (stride 2,
5 convs or two per stage) are read from its runs as references, never retrained; the 18 others are phase_13_hybrids. Phase
11's figures and tables never read these runs.

    01_hybrids.png   one panel per conv1 stride; per conv1 the three bodies, 3x3 and 2x2 side by side, FP32 / INT8, seed 42
    contrasts.csv    every matched pair of the 24 cells (one factor changed, B - A), seed 42
    all_runs.csv     every run of the 24 cells (the Phase 11 references with their replicates) and VGG16's reference

Phase 11's conventions: test top-1, INT8 hollow, early-best ring.

    python -m scripts.phase13.hybrids
"""
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from ml.plotting import NEUTRAL, TEXT_SECONDARY, apply_report_style
from scripts.phase11.analyze_geometry import ROOT, RUNS, band_for, early_handle, early_ring, load, noise_bands, savefig
from scripts.phase11.design_figures import KCOLOR, LIVE_NOTE, frame, legend_kernels

FIGS = ROOT / "results/figures_generated/phase_13_hybrids"
TABLES = ROOT / "results/phase_13_hybrids"
BODIES = [("", "5 convs"), ("stacked", "2 por estágio"), ("residual", "2 por estágio\n+ skip residual")]
VGG_REF = "vgg16_k3x3_stride1_5pool2x2_map2_gap_bn"  # the reference line: VGG16 3x3 + BN + GAP, Phase 11


def cell(k, big, body, s):
    """The registered name (ml/model_registrations.py) of the grid cell: body kernel k, 11x11 conv1 if big, conv1 stride s."""
    return f"alexnet_{f'k11-{k}' if big else f'k{k}x{k}'}{body}_stride{s}_2pool2x2_map{16 if s == 1 else 8}_gap_bn"


GRID = {(k, big, body, s): cell(k, big, body, s) for k, big, (body, _), s in product((3, 2), (False, True), BODIES, (2, 1))}
# (label, position in the GRID key, level A, level B): every matched pair = two cells equal but in that position
CHANGES = [("corpo: 5 convs → 2 por estágio", 2, "", "stacked"), ("corpo: 2 por estágio → + skip residual", 2, "stacked", "residual"),
           ("conv1: kernel do corpo → 11×11", 1, False, True), ("kernel do corpo: 3×3 → 2×2", 0, 3, 2),
           ("stride da conv1: 2 → 1", 3, 2, 1)]


def runs():
    """Every run of the 24 cells -- the phase_13_hybrids runs and the Phase 11 references -- and VGG16's reference."""
    df = pd.concat([load(*RUNS), load(*RUNS, prefix="phase_13_")], ignore_index=True)
    return df[df.key.isin([*GRID.values(), VGG_REF])] if len(df) else df


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


def fig_hybrids(s42, bands, figs):
    """One panel per conv1 stride; per conv1 (the body's kernel / 11x11) the three bodies left to right, 3x3 and 2x2 side by
    side (FP32 filled, INT8 hollow), a line through each kernel's FP32 across the bodies, VGG16 3x3 GAP + BN dashed."""
    v = s42.set_index("key")
    kernels = ((3, -0.14), (2, 0.14))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
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
                    if r.early_best:
                        early_ring(ax, x + dx, r.int8)
                        ringed = True
                    macs.append(f"{r.macs_eff_m:.0f}")
                ticks.append((x, f"{blabel}\n" + (f"{' · '.join(macs)}M" if macs else "pendente")))
            for k, dx in kernels:  # each kernel's FP32 across the three bodies
                pts = [(3.6 * gi + bi + dx, v.loc[GRID[(k, big, body, s)], "fp32"]) for bi, (body, _) in enumerate(BODIES)
                       if GRID[(k, big, body, s)] in v.index]
                if len(pts) > 1:
                    ax.plot(*zip(*pts), color=KCOLOR[f"k{k}x{k}"], lw=1, alpha=0.6, zorder=2)
            ax.text(3.6 * gi + 1, 1.01, "conv1 11×11" if big else "conv1 = kernel do corpo", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9.5, weight="bold")
        if VGG_REF in v.index:
            ax.axhline(v.loc[VGG_REF, "fp32"], color=NEUTRAL, ls="--", lw=1, zorder=1)
        ax.set_xticks([t for t, _ in ticks])
        ax.set_xticklabels([t for _, t in ticks], fontsize=8.5)
        ax.set_title(f"conv1 stride {s} (mapa final {8 if s == 2 else 16}×{8 if s == 2 else 16})", fontsize=10, pad=22)
        ax.margins(x=0.06, y=0.12)
    axes[0].set_ylabel("Top-1 (%)")
    handles = legend_kernels(["k3x3", "k2x2"]) + [
        Line2D([], [], marker="o", color=TEXT_SECONDARY, ls="", ms=8, label="FP32"),
        Line2D([], [], marker="D", mfc="white", mec=TEXT_SECONDARY, ls="", ms=8, label="INT8"),
        Line2D([], [], color=NEUTRAL, ls="--", label="VGG16 3×3 (GAP + BN)")]
    fig.legend(handles=handles + ([early_handle("ring")] if ringed else []), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=9)
    band = band_for(bands, "alexnet", True)["fp32"]
    fig.suptitle("Híbridos: profundidade, skip residual, conv1 11×11 e stride 1 somam? (AlexNet, 2 max-pools 2×2, GAP + BN)\n"
                 "sob cada corpo: MACs por imagem do 3×3 · do 2×2\n" + LIVE_NOTE
                 + (f" · ruído entre seeds (GAP + BN): ±{band:.1f} pp FP32" if pd.notna(band) else "")
                 + (f"\nainda na fila: {queued} de 24 células" if queued else ""), fontsize=11)
    fig.tight_layout()
    savefig(fig, "01_hybrids.png", figs)


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
    fig_hybrids(s42, noise_bands(frame()), FIGS)
    if len(c):
        print(c.groupby("factor", sort=False)[["d_fp32", "d_int8"]].agg(["median", "count"]).round(2).to_string())


if __name__ == "__main__":
    main()
