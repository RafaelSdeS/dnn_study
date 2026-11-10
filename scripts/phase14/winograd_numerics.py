"""Phase 14 -- Winograd-FPGA numerics on Phase 11's recipe (docs/logs/PHASE14_LOG.md), an extra study outside Phase 11: for each
net and each packed-Winograd variant (F23 / F43 / F63), the test top-1 of its `qat_wino` arm minus the net's FP32 top-1 on the same
10k test images (Tiny ImageNet's official val split).

    runs.csv     one row per (net, variant) with a finished arm: fp32, arm (test top-1, %), d = arm - fp32 (pp), the top-1
                 agreement of the arm with the FP32 net on the test set
    medians.csv  per variant: the number of nets (n) and the median / min / max of d
    results/figures_generated/phase_14_winograd_numerics/01_winograd_numerics.png
                 d per net, one panel per variant, the median a line; a ring on a net whose FP32 best epoch came early
                 (scripts/phase11/analyze_geometry.py:EARLY_FRAC), as every Phase 11 figure marks it

The FP32 number is recomputed from `<net>_fp32_test_logits.npz` (copied into every arm dir, the same source for all 16 nets, the two
Phase 11 cells included); it can differ from a summary's own number by one image (31.53 vs 31.52 on the canary cell).
A net whose arm has not finished is simply absent, so `n` says how many nets a median rests on.

    python -m scripts.phase14.winograd_numerics
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ROOT / "outputs/pcad"  # holds phase_14_winograd_f{23,43,63}_pack/<net>/results/
DEFAULT_OUT = ROOT / "results/phase_14_winograd_numerics"
DEFAULT_FIGS = ROOT / "results/figures_generated/phase_14_winograd_numerics"
VARIANTS = ("f23", "f43", "f63")
TITLES = {"f23": "F(2,3)", "f43": "F(4,3)", "f63": "F(6,3)"}
# Plain names for the figure; a net missing here keeps its key
NAMES = {
    "alexnet_k3x3_stride4_3pool3x3_map1_fcdrop_nobn": "AlexNet 3×3, layout original",
    "alexnet_3x3_fc_fpga": "AlexNet 3×3 FC", "alexnet_stacked_fpga": "AlexNet 3×3 em pares",
    "alexnet_bottleneck_fpga": "AlexNet Bottleneck", "alexnet_fire_fpga": "AlexNet Fire",
    "alexnet_fire_bypass_fpga": "AlexNet Fire + atalho", "alexnet_final_fire_residual_fpga": "AlexNet Fire + residual",
    "alexnet_final_bottleneck_residual_fpga": "AlexNet Bottleneck + residual", "vgg_style_fpga": "VGG-Style",
    "vgg13_fpga": "VGG-13", "vgg16_k3x3_stride1_5pool2x2_map2_fcdrop_bn": "VGG16", "resnet18_fpga": "ResNet-18",
    "repvgg_a0_fpga": "RepVGG-A0", "wrn_16_4_fpga": "WRN-16-4", "wrn_28_2_fpga": "WRN-28-2", "googlenet_fpga": "GoogLeNet"}


def fp32_top1(logits_npz):
    """Top-1 (%) of the saved logits against the saved labels."""
    z = np.load(logits_npz)
    return 100 * float((z["logits"].argmax(1) == z["labels"]).mean())


def collect(runs):
    """One row per finished arm: every `<net>_qat_wino_summary.json` under the three arm experiments."""
    rows = []
    for v in VARIANTS:
        for summary in sorted((runs / f"phase_14_winograd_{v}_pack").glob("*/results/*_qat_wino_summary.json")):
            net = summary.name.removesuffix("_qat_wino_summary.json")
            s = json.loads(summary.read_text())
            fp32, arm = fp32_top1(summary.parent / f"{net}_fp32_test_logits.npz"), s["test_eval"]["top1"]
            rows.append(dict(net=net, variant=v, fp32=fp32, arm=arm, d=arm - fp32, agreement=s["agreement_fp32_qat_wino_test"]))
    return pd.DataFrame(rows, columns=["net", "variant", "fp32", "arm", "d", "agreement"])


def medians(df):
    g = df.groupby("variant").d
    return pd.DataFrame({"n": g.size(), "median_d": g.median(), "min_d": g.min(), "max_d": g.max()}).reindex(VARIANTS).dropna().astype({"n": int}).reset_index()


def early_best(runs, net):
    """True when the net's seed-42 FP32 run (phase_14_winograd_fp32, or the Phase 11 cell it reuses) picked its best epoch
    before EARLY_FRAC of its budget; False when that summary is not under `runs`."""
    from scripts.phase11.analyze_geometry import EARLY_FRAC

    for p in sorted(runs.glob(f"phase_1[14]_*/{net}/results/{net}_summary.json")):
        d = json.loads(p.read_text())
        if d.get("epochs_budget") and d["config"]["experiment"]["seed"] == 42:
            return d["epochs"] + 1 < EARLY_FRAC * d["epochs_budget"]  # epochs is 0-based
    return False


def fig_numerics(df, figs):
    """d per net, one panel per variant (own x scale: F(6,3) reaches -60), nets in the order of their F(6,3) loss."""
    import matplotlib.pyplot as plt

    from ml.plotting import BLUE, RED, TEXT_SECONDARY, apply_report_style
    from scripts.phase11.analyze_geometry import early_handle, early_ring, savefig

    apply_report_style()
    order = df[df.variant == VARIANTS[-1]].sort_values("d").net.tolist()
    order += sorted(set(df.net) - set(order))
    y = {n: i for i, n in enumerate(order)}
    variants = [v for v in VARIANTS if (df.variant == v).any()]
    fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 0.32 * len(order) + 1.6), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], variants):
        s = df[df.variant == v]
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.axvline(s.d.median(), color=RED, lw=1.6, ls="--")
        ax.scatter(s.d, s.net.map(y), color=BLUE, s=30, zorder=5)
        for _, r in s[s.early].iterrows():
            early_ring(ax, r.d, y[r.net], s=120)
        ax.set_title(f"{TITLES[v]} · mediana {s.d.median():+.1f} p.p.", fontsize=11)
        ax.set_xlabel("simulação − FP32 (p.p.)")
        ax.grid(axis="x", alpha=0.3)
    axes[0][0].set_yticks(range(len(order)), [NAMES.get(n, n) for n in order])
    axes[0][0].invert_yaxis()
    if df.early.any():
        ring = early_handle("ring")
        ring.set_label("simulação inflada pelo QAT (melhor época FP32 antes de 80% do treino)")
        fig.legend(handles=[ring], loc="lower center", bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout()
    savefig(fig, "01_winograd_numerics.png", Path(figs))


def main(runs=DEFAULT_RUNS, out=DEFAULT_OUT, figs=DEFAULT_FIGS):
    df = collect(Path(runs))
    if df.empty:
        print("no finished qat_wino arm yet")
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values(["variant", "net"]).round(4).to_csv(out / "runs.csv", index=False)
    m = medians(df)
    m.round(4).to_csv(out / "medians.csv", index=False)
    print(m.round(2).to_string(index=False))
    df["early"] = [early_best(Path(runs), n) for n in df.net]
    fig_numerics(df, figs)


if __name__ == "__main__":
    main()
