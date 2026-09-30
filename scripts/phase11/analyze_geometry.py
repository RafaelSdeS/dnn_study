"""Phase 11 geometry/kernel/BN analysis — tables + figures 11-14, from the raw per-run summaries.

Reads every outputs/pcad/phase_11_*/<model>/results/*_summary.json (seed 42 in phase_11_geometry_controls /
_factorial and the earlier Phase 11 experiments, seeds 43/44 in phase_11_geometry_seeds_s43/_s44) and writes
  results/phase_11_geometry_analysis/{all_runs,kernel_seeds,geometry_factorial,bn_block}.csv
  results/figures_generated/phase_11_kernel_size_comparison/{11..14}_*.png

    python -m scripts.phase11.analyze_geometry
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ml.plotting import AMBER, BLUE, GREEN, RED, apply_report_style

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results/phase_11_geometry_analysis"
FIGS = ROOT / "results/figures_generated/phase_11_kernel_size_comparison"
GRAY = "#8a8a8a"


def load() -> pd.DataFrame:
    """One row per run. qat/int8 are NaN for a run whose QAT predates the fusion fix and whose model's fuse map
    needed it (registry fuse_root_attr): those numbers measured an unfused QAT graph (docs/logs/PHASE11_LOG.md,
    "QAT fusion bug"). A run made with the fix records git_dirty_files in its provenance (same commit)."""
    from ml import model_registrations  # noqa: F401 -- populates the registry
    from ml.registry import MODEL_REGISTRY

    rows = []
    for p in sorted((ROOT / "outputs/pcad").glob("phase_11_*/*/results/*_summary.json")):
        d = json.loads(p.read_text())
        prov, key = d["config"]["provenance"], p.parents[1].name
        post_fix = "git_dirty_files" in prov
        fused = post_fix or (key in MODEL_REGISTRY and not MODEL_REGISTRY[key].get("fuse_root_attr"))
        nan = float("nan")
        rows.append(dict(
            exp=p.parents[2].name.removeprefix("phase_11_"), key=key, seed=d["config"]["experiment"]["seed"],
            fp32=d["fp32_top1"], qat=d["qat_top1"] if fused else nan, int8=d["int8_top1"] if fused else nan,
            params_m=d["params_m"], macs_m=d["macs"] / 1e6, int8_mb=d["int8_size_mb"], best_ep=d["epochs"],
            ece=d.get("fp32_ece"), qat_fused=fused, post_fix=post_fix, git_hash=prov.get("git_hash", "")[:7],
            git_dirty=prov.get("git_dirty")))
    df = pd.DataFrame(rows)
    assert not df[df.post_fix & df.git_dirty].shape[0], df[df.post_fix & df.git_dirty][["exp", "key", "git_hash"]]
    df["drop_qat"] = df.fp32 - df.qat      # FP32 -> fake-quant (what QAT costs)
    df["drop_convert"] = df.qat - df.int8  # fake-quant -> real INT8 (what convert_to_int8 costs)
    df["drop_total"] = df.fp32 - df.int8
    return df


def savefig(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / name}")


def kernel_seeds(df):
    """3x3 vs 11-5-3-3-3 vs 2x2 at the adapted 8x8-map geometry, mean/std over seeds 42-44 where they exist."""
    keys = {"alexnet_adapted_orig_fc": ("FC", "11-5-3-3-3"), "alexnet_3x3_fc": ("FC", "3x3"), "alexnet_adapted_2x2_fc": ("FC", "2x2"),
            "alexnet_adapted_orig_gap": ("GAP", "11-5-3-3-3"), "alexnet_3x3_gap": ("GAP", "3x3"), "alexnet_adapted_2x2_gap": ("GAP", "2x2")}
    d = df[df.key.isin(keys) & ~df.exp.isin(["kernel_size_comparison", "head_bn_ablation"])].copy()
    d["head"], d["kernel"] = zip(*d.key.map(keys))
    agg = d.groupby(["head", "kernel"]).agg(
        n=("seed", "count"), fp32=("fp32", "mean"), fp32_sd=("fp32", "std"), int8=("int8", "mean"), int8_sd=("int8", "std"),
        drop_qat=("drop_qat", "mean"), drop_convert=("drop_convert", "mean"), params_m=("params_m", "first"), macs_m=("macs_m", "first"))
    return d, agg.reset_index()


def fig_kernel(d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    colors = {"11-5-3-3-3": AMBER, "3x3": GREEN, "2x2": BLUE}
    for ax, head in zip(axes, ["FC", "GAP"]):
        for i, kern in enumerate(["11-5-3-3-3", "3x3", "2x2"]):
            s = d[(d["head"] == head) & (d.kernel == kern)]
            for j, (col, off) in enumerate([("fp32", -0.18), ("int8", 0.18)]):
                ax.bar(i + off, s[col].mean(), 0.34, color=colors[kern], alpha=1 if col == "fp32" else 0.5,
                       label=None, edgecolor="white")
                ax.scatter([i + off] * len(s), s[col], color="k", s=14, zorder=3)
            ax.text(i, -0.26, f"{s.macs_m.iloc[0]:.0f}M MACs · n={len(s)}", ha="center", fontsize=8, color="#4d4d4d", transform=ax.get_xaxis_transform())
        ax.set_xticks(range(3)); ax.set_xticklabels(["11-5-3-3-3\n(original)", "3x3", "2x2"]); ax.tick_params(axis="x", pad=14)
        ax.set_title(f"Classificador {head}"); ax.set_ylabel("Top-1 (%)")
    axes[0].set_ylim(0, 42); axes[1].set_ylim(0, 52)
    fig.suptitle("Efeito do kernel na geometria adaptada (mapa 8×8, sem BN) — barra cheia FP32, clara INT8, pontos = seeds 42/43/44", fontsize=11)
    savefig(fig, "11_kernel_effect_adapted_geometry_seeds.png")


FACTORIAL = {  # key: (stride, pooling, final map, label)
    "alexnet_geo_s4_p3_fc": ("s4", "3×pool(3,2)", "1×1", "s4 / pool 3×3 (torchvision)"),
    "alexnet_geo_s4_p2_fc": ("s4", "2×pool(2)", "3×3", "s4 / pool 2×2"),
    "alexnet_geo_s2_p3_fc": ("s2", "3×pool(3,2)", "3×3", "s2 / pool 3×3"),
    "alexnet_geo_s2_pk3n2_fc": ("s2", "2×pool(3,2)", "7×7", "s2 / 2 pools k3"),
    "alexnet_geo_s2_pk2n3_fc": ("s2", "3×pool(2)", "4×4", "s2 / 3 pools k2"),
    "alexnet_adapted_orig_fc": ("s2", "2×pool(2)", "8×8", "s2 / pool 2×2 (adaptado)"),
    "alexnet_geo_s2_p2_drop_fc": ("s2", "2×pool(2)+Dropout", "8×8", "adaptado + Dropout"),
    "alexnet_geo_s4_p3_gap": ("s4", "3×pool(3,2)", "1×1", "s4 / pool 3×3, GAP"),
    "alexnet_geo_s4_p3_fc_k3": ("s4", "3×pool(3,2), k=3", "1×1", "s4 / pool 3×3, kernels 3×3"),
    "alexnet_tv_scratch": ("s4", "3×pool(3,2)+Dropout", "1×1", "AlexNetTV do zero (+Dropout)"),
    "alexnet_tv": ("s4", "3×pool(3,2)+Dropout, pré-treino", "1×1", "AlexNetTV pré-treinado"),
    "alexnet_adapted_orig_fc_pt": ("s2", "2×pool(2), pré-treino", "8×8", "adaptado pré-treinado"),
}


def geometry_table(df):
    d = df[df.key.isin(FACTORIAL) & (df.seed == 42)].drop_duplicates("key").set_index("key").loc[list(FACTORIAL)]
    d["stem"], d["pooling"], d["map"], d["label"] = zip(*[FACTORIAL[k] for k in d.index])
    return d.reset_index()[["key", "label", "stem", "pooling", "map", "fp32", "qat", "int8", "best_ep", "params_m", "macs_m"]]


def fig_geometry(g):
    g = g.sort_values("fp32")
    color = [RED if "GAP" in r.label else (BLUE if "pré" in r.label else (GREEN if r.map == "8×8" else GRAY)) for r in g.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(g))
    ax.barh(y, g.fp32, color=color, alpha=0.9)
    ax.scatter(g.int8, y, color="k", marker="s", s=22, zorder=3, label="INT8")
    for i, r in enumerate(g.itertuples()):
        ax.text(max(r.fp32, r.int8) + 0.6, i, f"{r.fp32:.1f}  (mapa {r.map})", va="center", fontsize=8)
    ax.set_yticks(list(y)); ax.set_yticklabels(g.label, fontsize=9)
    ax.set_xlabel("Top-1 FP32 (%)"); ax.set_xlim(0, 52); ax.legend(loc="lower right")
    ax.set_title("Fatorial de geometria a 64×64 (kernels 11-5-3-3-3, sem BN, seed 42; ruído entre seeds ≈ ±0,3–0,9pp)\n"
                 "cinza = geometria torchvision · verde = adaptada · azul = pré-treinado · vermelho = GAP", fontsize=10)
    savefig(fig, "12_geometry_factorial.png")


def fig_convert(df):
    keys = ["alexnet_3x3_fc", "alexnet_adapted_orig_fc", "alexnet_adapted_2x2_fc", "alexnet_3x3_gap", "alexnet_adapted_orig_gap",
            "alexnet_adapted_2x2_gap", "alexnet_2x2_gap", "alexnet_mixed", "alexnet_smallkernel", "alexnet_stacked_gap_nobn",
            "alexnet_3x3_gap_bn", "alexnet_mixed_bn", "alexnet_stacked_gap", "alexnet_bottleneck", "alexnet_fire"]
    d = df[df.key.isin(keys) & (df.seed == 42)].drop_duplicates("key").set_index("key").loc[keys]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(d))
    ax.bar(x, d.drop_qat, color=BLUE, label="FP32 → QAT (fake-quant)")
    ax.bar(x, d.drop_convert, bottom=d.drop_qat.clip(lower=0), color=RED, label="QAT → INT8 real (convert_to_int8)")
    ax.set_xticks(list(x)); ax.set_xticklabels([k.replace("alexnet_", "") for k in d.index], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Perda de top-1 (pp)")
    ax.set_title("Onde o INT8 perde acurácia: no QAT ou na conversão? (seed 42)\n"
                 "GAP sem BN perde 5–25pp só ao converter; com BN (3x3_gap_bn, mixed_bn, stacked_gap, bottleneck, fire) a conversão é ≈ 0", fontsize=10)
    ax.legend()
    savefig(fig, "13_quantization_drop_where.png")


def bn_block(df):
    keys = ["alexnet_2x2_gap", "alexnet_adapted_2x2_gap", "alexnet_adapted_orig_gap", "alexnet_3x3_gap", "alexnet_3x3_gap_bn",
            "alexnet_bottleneck", "alexnet_fire"]
    d = df[df.key.isin(keys) & (df.seed == 42)].drop_duplicates("key").set_index("key").loc[keys].reset_index()
    return d[["key", "fp32", "qat", "int8", "params_m", "macs_m", "int8_mb"]]


def fig_bn_block(b):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in b.itertuples():
        col = GREEN if r.key in ("alexnet_bottleneck", "alexnet_fire") else (AMBER if "bn" in r.key else GRAY)
        ax.scatter(r.macs_m, r.fp32, color=col, s=110, edgecolors="white", zorder=3)
        ax.scatter(r.macs_m, r.int8, color=col, marker="s", s=80, edgecolors="white", zorder=3)
        ax.plot([r.macs_m] * 2, [r.fp32, r.int8], color=col, alpha=0.4)
        ax.annotate(f"{r.key.replace('alexnet_', '')} ({r.params_m:.2f}M par.)", (r.macs_m, r.fp32), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("MACs (M, 64×64)"); ax.set_ylabel("Top-1 (%)"); ax.margins(x=0.12)
    ax.set_title("BN e blocos de compensação no layout adaptado, GAP (○ FP32  □ INT8, seed 42)\n"
                 "cinza = sem BN · laranja = 3x3 + BN · verde = Bottleneck/Fire (com BN)", fontsize=10)
    savefig(fig, "14_bn_and_compensation_blocks.png")


def main():
    apply_report_style()
    df = load()
    TABLES.mkdir(parents=True, exist_ok=True)
    df.sort_values(["exp", "key", "seed"]).to_csv(TABLES / "all_runs.csv", index=False)
    d, agg = kernel_seeds(df)
    agg.to_csv(TABLES / "kernel_seeds.csv", index=False)
    g = geometry_table(df); g.to_csv(TABLES / "geometry_factorial.csv", index=False)
    b = bn_block(df); b.to_csv(TABLES / "bn_block.csv", index=False)
    fig_kernel(d); fig_geometry(g); fig_convert(df); fig_bn_block(b)
    pd.set_option("display.width", 200)
    for name, t in [("kernel_seeds", agg), ("geometry_factorial", g), ("bn_block", b)]:
        print(f"\n== {name}\n{t.round(2).to_string(index=False)}")


if __name__ == "__main__":
    main()
