"""Phase 13, Stage B -- Bottleneck and Fire units on the stride-1 bodies (docs/logs/PHASE13_LOG.md, "Stage B"), an extra
study outside Phase 11: does replacing every conv of the GAP + BN AlexNet body by a Bottleneck (1x1 -> kxk -> 1x1) or a Fire
unit (1x1 squeeze -> 1x1 || kxk expands) cut the cost without losing accuracy? 12 cells at seed 42: unit (conv / Bottleneck /
Fire) x units per stage (one: beside the 5-conv cell / two: beside `k<k>stacked`) x kernel (3x3 / 2x2), all conv1 stride 1.
The 4 plain bodies are Phase 13's hybrids (or Phase 11's), `k3x3fire` is `alexnet_fire` of `phase_11_families` (same net, layer
for layer); the 7 others are phase_13_bottleneck_fire. Phase 11's figures and tables never read these runs.

    01_bottleneck_fire.png   one panel per units-per-stage; conv / Bottleneck / Fire left to right, 3x3 and 2x2 side by side
                             (FP32 filled, INT8 hollow, early-best ring), a line through each kernel's FP32 across the units
    contrasts.csv            every matched pair (one unit swapped, same kernel and units per stage): B - A in FP32 / INT8,
                             params and MACs change in %
    contrasts_by_seed.csv    the same pairs at every seed where both cells ran (a `seed` column; 42 = contrasts.csv)
    all_runs.csv             every run of the cells found (+ winograd_pct: % of the MACs a Winograd engine runs)

Phase 11's conventions: test top-1, INT8 hollow, early-best ring.

    python -m scripts.phase13.bottleneck_fire
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from ml.plotting import NEUTRAL, TEXT_SECONDARY, apply_report_style
from scripts.phase11.analyze_geometry import ROOT, RUNS, band_for, early_handle, early_ring, load, noise_bands, savefig
from scripts.phase11.design_figures import KCOLOR, LIVE_NOTE, frame, legend_kernels, winograd_share
from scripts.phase13.hybrids import GRID, contrasts_by_seed

FIGS = ROOT / "results/figures_generated/phase_13_bottleneck_fire"
TABLES = ROOT / "results/phase_13_bottleneck_fire"
UNITS = [("", "conv comum"), ("bottleneck", "Bottleneck"), ("fire", "Fire")]
PER_STAGE = [(1, "uma unidade por estágio (corpo de 5 convs)"), (2, "duas unidades por estágio")]
FIRE_3X3 = "alexnet_fire"  # = k3x3fire, one unit per stage: read from phase_11_families, never retrained


def cell(k, unit, n):
    """The registered name (ml/model_registrations.py) of the cell: body kernel k, unit ("" = plain conv), n units per stage."""
    if not unit:
        return GRID[(k, False, "stacked" if n == 2 else "", 1)]
    if (k, unit, n) == (3, "fire", 1):
        return FIRE_3X3
    return f"alexnet_k{k}x{k}{unit}{'stacked' if n == 2 else ''}_stride1_2pool2x2_map16_gap_bn"


CELLS = {(k, u, n): cell(k, u, n) for k in (3, 2) for u, _ in UNITS for n, _ in PER_STAGE}
# (label, unit A, unit B): the swap that makes a matched pair
CHANGES = [("conv → Bottleneck", "", "bottleneck"), ("conv → Fire", "", "fire"), ("Bottleneck → Fire", "bottleneck", "fire")]


def runs():
    """Every run of the 12 cells: phase_13_bottleneck_fire, the plain bodies and alexnet_fire."""
    df = pd.concat([load(*RUNS), load(*RUNS, prefix="phase_13_")], ignore_index=True)
    return df[df.key.isin(CELLS.values())] if len(df) else df


def contrasts(s42):
    v = s42.set_index("key")
    rows = []
    for label, ua, ub in CHANGES:
        for (k, u, n), ka in CELLS.items():
            kb = CELLS[(k, ub, n)]
            if u == ua and ka in v.index and kb in v.index:
                A, B = v.loc[ka], v.loc[kb]
                rows.append(dict(factor=label, kernel=f"{k}x{k}", per_stage=n, A=ka, B=kb, fp32_A=A.fp32, fp32_B=B.fp32,
                                 d_fp32=B.fp32 - A.fp32, int8_A=A.int8, int8_B=B.int8, d_int8=B.int8 - A.int8,
                                 params_change_pct=100 * (B.params_eff_m / A.params_eff_m - 1),
                                 macs_change_pct=100 * (B.macs_eff_m / A.macs_eff_m - 1)))
    return pd.DataFrame(rows)


def fig_bottleneck_fire(s42, bands, figs):
    v = s42.set_index("key")
    kernels = ((3, -0.14), (2, 0.14))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    queued, ringed = 0, False
    for ax, (n, nlabel) in zip(axes, PER_STAGE):
        ticks = []
        for ui, (unit, ulabel) in enumerate(UNITS):
            costs = []
            for k, dx in kernels:
                key = CELLS[(k, unit, n)]
                if key not in v.index:
                    queued += 1
                    continue
                r, c = v.loc[key], KCOLOR[f"k{k}x{k}"]
                ax.scatter(ui + dx, r.fp32, s=60, color=c, zorder=3)
                ax.scatter(ui + dx, r.int8, s=60, marker="D", facecolors="white", edgecolors=c, lw=1.6, zorder=3)
                if r.early_best:
                    early_ring(ax, ui + dx, r.int8)
                    ringed = True
                costs.append(f"{r.params_eff_m:.2f}")
            ticks.append((ui, f"{ulabel}\n" + (f"{' · '.join(costs)} M params" if costs else "pendente")))
        for k, dx in kernels:  # each kernel's FP32 across the three units
            pts = [(ui + dx, v.loc[CELLS[(k, u, n)], "fp32"]) for ui, (u, _) in enumerate(UNITS) if CELLS[(k, u, n)] in v.index]
            if len(pts) > 1:
                ax.plot(*zip(*pts), color=KCOLOR[f"k{k}x{k}"], lw=1, alpha=0.6, zorder=2)
        ax.set_xticks([t for t, _ in ticks])
        ax.set_xticklabels([t for _, t in ticks], fontsize=9)
        ax.set_title(nlabel, fontsize=10)
        ax.margins(x=0.1, y=0.12)
    axes[0].set_ylabel("Top-1 (%)")
    handles = legend_kernels(["k3x3", "k2x2"]) + [
        Line2D([], [], marker="o", color=TEXT_SECONDARY, ls="", ms=8, label="FP32"),
        Line2D([], [], marker="D", mfc="white", mec=TEXT_SECONDARY, ls="", ms=8, label="INT8")]
    fig.legend(handles=handles + ([early_handle("ring")] if ringed else []), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=9)
    band = band_for(bands, "alexnet", True)["fp32"]
    fig.suptitle("Bottleneck e Fire: custo menor sem perder acurácia? (AlexNet, conv1 stride 1, 2 max-pools 2×2, GAP + BN)\n"
                 "sob cada unidade: milhões de parâmetros do 3×3 · do 2×2\n" + LIVE_NOTE
                 + (f" · ruído entre seeds (GAP + BN): ±{band:.1f} p.p. FP32" if pd.notna(band) else "")
                 + (f"\nainda na fila: {queued} de 24 pontos" if queued else ""), fontsize=11)
    fig.tight_layout()
    savefig(fig, "01_bottleneck_fire.png", figs)


def main():
    apply_report_style()
    df = runs()
    if df.empty or not (df.exp == "bottleneck_fire").any():  # load() strips the phase_13_ prefix
        print("no phase_13_bottleneck_fire run yet")
        return
    TABLES.mkdir(parents=True, exist_ok=True)
    df["winograd_pct"] = df.apply(winograd_share, axis=1)  # the article's tab:winograd rule: stride-1 dense 2x2 / 3x3 MACs
    df.drop(columns="results").sort_values(["key", "seed"]).to_csv(TABLES / "all_runs.csv", index=False)
    s42 = df[df.seed == 42]
    c = contrasts(s42)
    c.to_csv(TABLES / "contrasts.csv", index=False)
    contrasts_by_seed(df, contrasts).to_csv(TABLES / "contrasts_by_seed.csv", index=False)
    fig_bottleneck_fire(s42, noise_bands(frame()), FIGS)
    if len(c):
        print(c.groupby("factor", sort=False)[["d_fp32", "d_int8", "params_change_pct", "macs_change_pct"]].agg(["median", "count"]).round(2).to_string())


if __name__ == "__main__":
    main()
