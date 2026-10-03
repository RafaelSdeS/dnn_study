"""Phase 11 geometry/kernel/BN analysis — tables + figure 15, from the raw per-run summaries; also the shared
loader and plot helpers of scripts/phase11/{plot_kernel_comparison,factor_effects}.py.

Reads every outputs/pcad/phase_11_*/<model>/results/*_summary.json (seed 42 in phase_11_geometry_controls /
_factorial and the earlier Phase 11 experiments, seeds 43/44 in phase_11_geometry_seeds_s43/_s44) and writes
  results/phase_11_geometry_analysis/{all_runs,kernel_seeds,geometry_factorial,bn_block}.csv
  results/figures_generated/phase_11_kernel_size_comparison/15_quantization_drop_where.png

Figure vocabulary (every Phase 11 figure): "layout original" = torchvision's AlexNet geometry (conv1 stride 4, three
3x3/2 max-pools -> 1x1 map before the classifier at 64x64); "layout 64px" = the adapted one (conv1 stride 2, two 2x2
max-pools -> 8x8 map, no Dropout). Baseline = alexnet_tv_scratch, the original AlexNet trained from scratch.

    python -m scripts.phase11.analyze_geometry
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Patch

from ml.plotting import BLUE, RED, TEXT_SECONDARY, apply_report_style

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results/phase_11_geometry_analysis"
FIGS = ROOT / "results/figures_generated/phase_11_kernel_size_comparison"

BASE_KEY = "alexnet_tv_scratch"  # the original AlexNet (11-5-3-3-3, layout original, FC + Dropout), from scratch
BASE_LABEL = "AlexNet original do zero (baseline)"
OLD_INT8 = "INT8 antigo: QAT antes das correções de 30/09–02/10, inválido (rerun na fila)"
LAYOUTS = ("layout original = o do AlexNet torchvision: conv1 stride 4 + 3 max-pools 3×3/2 → mapa final 1×1 em 64×64\n"
           "layout 64px = adaptado a 64×64: conv1 stride 2 + 2 max-pools 2×2 → mapa final 8×8, sem Dropout")


def load() -> pd.DataFrame:
    """One row per run, accuracies on the held-out test set (Tiny ImageNet's official val split; the 90/10 split of
    train/ only picked each run's best epoch). fp32/qat/int8 are NaN for a run without the current quant_protocol
    (ml/quantization.py:QUANT_PROTOCOL) -- every Phase 11 QAT is redone on that one definition, and the test set
    arrived with it, so an older run is shown only faded, from its validation-split qat_raw/int8_raw. qat_fused
    (unfused QAT before 09-30) stays as a column for the tables. `results` is the run's results dir, where the
    per-image *_test_logits.npz live (factor_effects' McNemar tests)."""
    from ml import model_registrations  # noqa: F401 -- populates the registry
    from ml.quantization import QUANT_PROTOCOL
    from ml.registry import MODEL_REGISTRY

    rows = []
    for p in sorted((ROOT / "outputs/pcad").glob("phase_11_*/*/results/*_summary.json")):
        d = json.loads(p.read_text())
        prov, key = d["config"]["provenance"], p.parents[1].name
        post_fix = d.get("quant_protocol") == QUANT_PROTOCOL
        fused = post_fix or (key in MODEL_REGISTRY and not MODEL_REGISTRY[key].get("fuse_root_attr"))
        nan = float("nan")
        rows.append(dict(
            exp=p.parents[2].name.removeprefix("phase_11_"), key=key, seed=d["config"]["experiment"]["seed"],
            fp32=d["test_fp32_top1"] if post_fix else nan, qat=d["test_qat_top1"] if post_fix else nan,
            int8=d["test_int8_top1"] if post_fix else nan, fp32_val=d["fp32_top1"], results=p.parent,
            qat_raw=d["qat_top1"], int8_raw=d["int8_top1"],
            params_m=d["params_m"], macs_m=d["macs"] / 1e6, fp32_mb=d["fp32_size_mb"], int8_mb=d["int8_size_mb"], best_ep=d["epochs"],
            ece=d.get("test_fp32_ece") if post_fix else nan, qat_fused=fused, post_fix=post_fix, git_hash=prov.get("git_hash", "")[:7],
            git_dirty=prov.get("git_dirty")))
    df = pd.DataFrame(rows)
    assert not df[df.post_fix & df.git_dirty].shape[0], df[df.post_fix & df.git_dirty][["exp", "key", "git_hash"]]
    df["drop_qat"] = df.fp32 - df.qat      # FP32 -> fake-quant (what QAT costs)
    df["drop_convert"] = df.qat - df.int8  # fake-quant -> real INT8 (what convert_to_int8 costs)
    df["drop_total"] = df.fp32 - df.int8
    df["drop_qat_raw"] = df.fp32_val - df.qat_raw  # the superseded numbers only exist on the validation split
    df["drop_convert_raw"] = df.qat_raw - df.int8_raw
    return df


def savefig(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / name}")


def int8_bar(ax, x, valid, raw, width, color):
    """INT8 bar beside its FP32 one: light fill when valid, faded hatched outline when only the pre-fix value exists."""
    if pd.notna(valid):
        ax.bar(x, valid, width, color=color, alpha=0.45, edgecolor="white")
    elif pd.notna(raw):
        ax.bar(x, raw, width, facecolor="none", edgecolor=color, hatch="///", alpha=0.6, lw=0.8)


def precision_handles(color=TEXT_SECONDARY):
    return [Patch(facecolor=color, label="barra cheia = FP32"), Patch(facecolor=color, alpha=0.45, label="barra clara ao lado = INT8"),
            Patch(facecolor="none", edgecolor=color, hatch="///", alpha=0.6, label=f"barra hachurada = {OLD_INT8}")]


def reference_lines(ax, df, key=BASE_KEY, label=BASE_LABEL, int8=True):
    """Dashed (FP32) / dotted (INT8) lines at the baseline run's top-1; returns them as legend handles."""
    r = df[(df.key == key) & (df.seed == 42)].iloc[0]
    lines = [ax.axhline(r.fp32, color="#333333", ls="--", lw=1.3, zorder=4, label=f"{label}: FP32 {r.fp32:.1f}%")]
    if int8:
        v, old = (r.int8, "") if pd.notna(r.int8) else (r.int8_raw, " antigo")
        lines.append(ax.axhline(v, color="#333333", ls=":", lw=1.3, zorder=4, alpha=0.6 if old else 1,
                                label=f"{label}: INT8{old} {v:.1f}%"))
    return lines


def kernel_seeds(df):
    """3x3 vs 11-5-3-3-3 vs 2x2 at the adapted 8x8-map geometry, mean/std over seeds 42-44 where they exist."""
    keys = {"alexnet_adapted_orig_fc": ("FC", "11-5-3-3-3"), "alexnet_3x3_fc": ("FC", "3x3"), "alexnet_adapted_2x2_fc": ("FC", "2x2"),
            "alexnet_adapted_orig_gap": ("GAP", "11-5-3-3-3"), "alexnet_3x3_gap": ("GAP", "3x3"), "alexnet_adapted_2x2_gap": ("GAP", "2x2")}
    d = df[df.key.isin(keys) & ~df.exp.isin(["kernel_size_comparison", "head_bn_ablation"])].copy()
    d["head"], d["kernel"] = zip(*d.key.map(keys))
    agg = d.groupby(["head", "kernel"]).agg(
        n=("seed", "count"), fp32=("fp32", "mean"), fp32_sd=("fp32", "std"), int8=("int8", "mean"), int8_sd=("int8", "std"),
        drop_qat=("drop_qat", "mean"), drop_convert=("drop_convert", "mean"), params_m=("params_m", "first"), macs_m=("macs_m", "first"))
    return agg.reset_index()


FACTORIAL = {  # key: (conv1 stride, pooling, final map, label)
    "alexnet_geo_s4_p3_fc": ("s4", "3×pool(3,2)", "1×1", "layout original (s4, 3 pools 3×3), sem Dropout"),
    "alexnet_geo_s4_p2_fc": ("s4", "2×pool(2)", "3×3", "s4, 2 pools 2×2"),
    "alexnet_geo_s2_p3_fc": ("s2", "3×pool(3,2)", "3×3", "s2, 3 pools 3×3"),
    "alexnet_geo_s2_pk3n2_fc": ("s2", "2×pool(3,2)", "7×7", "s2, 2 pools 3×3"),
    "alexnet_geo_s2_pk2n3_fc": ("s2", "3×pool(2)", "4×4", "s2, 3 pools 2×2"),
    "alexnet_adapted_orig_fc": ("s2", "2×pool(2)", "8×8", "layout 64px (s2, 2 pools 2×2)"),
    "alexnet_geo_s2_p2_drop_fc": ("s2", "2×pool(2)+Dropout", "8×8", "layout 64px + Dropout"),
    "alexnet_geo_s4_p3_gap": ("s4", "3×pool(3,2)", "1×1", "layout original, cabeça GAP"),
    "alexnet_geo_s4_p3_fc_k3": ("s4", "3×pool(3,2), k=3", "1×1", "layout original, kernels 3×3, sem Dropout"),
    "alexnet_tv_scratch": ("s4", "3×pool(3,2)+Dropout", "1×1", "AlexNet original do zero (+Dropout)"),
    "alexnet_tv": ("s4", "3×pool(3,2)+Dropout, pré-treino", "1×1", "AlexNet original pré-treinada"),
    "alexnet_adapted_orig_fc_pt": ("s2", "2×pool(2), pré-treino", "8×8", "layout 64px pré-treinado"),
}


def geometry_table(df):
    d = df[df.key.isin(FACTORIAL) & (df.seed == 42)].drop_duplicates("key").set_index("key").loc[list(FACTORIAL)]
    d["stem"], d["pooling"], d["map"], d["label"] = zip(*[FACTORIAL[k] for k in d.index])
    return d.reset_index()[["key", "label", "stem", "pooling", "map", "fp32", "qat", "int8", "best_ep", "params_m", "macs_m"]]


# "64px"/"original" = layout (module docstring); s1 = conv1 stride 1 (neither layout)
CONVERT_KEYS = {
    BASE_KEY: "AlexNet original\n(baseline)", "alexnet_3x3_fc": "64px · 3×3 · FC",
    "alexnet_adapted_orig_fc": "64px · 11-5-3-3-3 · FC", "alexnet_adapted_2x2_fc": "64px · 2×2 · FC",
    "alexnet_3x3_gap": "64px · 3×3 · GAP", "alexnet_adapted_orig_gap": "64px · 11-5-3-3-3 · GAP",
    "alexnet_adapted_2x2_gap": "64px · 2×2 · GAP", "alexnet_2x2_gap": "64px · 2×2 sem padding · GAP",
    "alexnet_mixed": "64px · misto 3-2-3-2-3 · GAP", "alexnet_smallkernel": "3×3 estreito, conv1 s1 · GAP",
    "alexnet_stacked_gap_nobn": "64px · 3×3 empilhado · GAP",
    "alexnet_3x3_gap_bn": "64px · 3×3 · GAP", "alexnet_mixed_bn": "64px · misto 3-2-3-2-3 · GAP",
    "alexnet_stacked_gap": "64px · 3×3 empilhado · GAP", "alexnet_bottleneck": "Bottleneck · GAP", "alexnet_fire": "Fire · GAP",
}


def fig_convert(df):
    d = df[df.key.isin(CONVERT_KEYS) & (df.seed == 42)].drop_duplicates("key").set_index("key").loc[list(CONVERT_KEYS)]
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, r in enumerate(d.itertuples()):
        valid = pd.notna(r.int8)
        dq, dc = (r.drop_qat, r.drop_convert) if valid else (r.drop_qat_raw, r.drop_convert_raw)
        for val, bottom, color in [(dq, 0, BLUE), (dc, max(dq, 0), RED)]:
            kw = dict(color=color) if valid else dict(facecolor="none", edgecolor=color, hatch="///", alpha=0.6, lw=0.8)
            ax.bar(i, val, 0.7, bottom=bottom, **kw)
    first_bn = list(CONVERT_KEYS).index("alexnet_3x3_gap_bn")
    ax.axvline(first_bn - 0.5, color=TEXT_SECONDARY, lw=1, ls="--")
    for x, txt, ha in [(first_bn - 0.6, "sem BatchNorm ←", "right"), (first_bn - 0.4, "→ com BatchNorm", "left")]:
        ax.text(x, 0.97, txt, transform=ax.get_xaxis_transform(), ha=ha, va="top", fontsize=9, color=TEXT_SECONDARY)
    ax.axhline(0, color="k", lw=0.8)
    ax.margins(y=0.1)  # headroom for the BN labels above the tallest bar
    ax.set_xticks(range(len(d))); ax.set_xticklabels(list(CONVERT_KEYS.values()), rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Perda de top-1 (pp)")
    ax.set_title("Onde o INT8 perde acurácia: no treino QAT ou na conversão para INT8 real? (seed 42)\n"
                 "64px = layout 64px (conv1 stride 2, 2 max-pools 2×2)\nhachurado = QAT antigo, antes das correções de fusão e de "
                 "pooling INT8 de 30/09 — a perda grande na conversão vinha desses bugs (PHASE11_LOG)", fontsize=10)
    ax.legend(handles=[Patch(color=BLUE, label="FP32 → QAT (perda no treino com fake-quant)"),
                       Patch(color=RED, label="QAT → INT8 real (perda na conversão)"),
                       Patch(facecolor="none", edgecolor=TEXT_SECONDARY, hatch="///", alpha=0.6, label=OLD_INT8)],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    savefig(fig, "15_quantization_drop_where.png")


def bn_block(df):
    keys = ["alexnet_2x2_gap", "alexnet_adapted_2x2_gap", "alexnet_adapted_orig_gap", "alexnet_3x3_gap", "alexnet_3x3_gap_bn",
            "alexnet_bottleneck", "alexnet_fire"]
    d = df[df.key.isin(keys) & (df.seed == 42)].drop_duplicates("key").set_index("key").loc[keys].reset_index()
    return d[["key", "fp32", "qat", "int8", "params_m", "macs_m", "int8_mb"]]


def noise_band(df):
    """2*sqrt(2)*pooled SD across seeds, for the 4 models replicated at seeds 42-44: the training-noise floor a
    single-seed difference has to clear (Bouthillier et al., MLSys 2021: seed variance is a first-order source of
    variance in benchmark comparisons)."""
    rep = df[df.key.isin(["alexnet_3x3_fc", "alexnet_adapted_orig_fc", "alexnet_3x3_gap", "alexnet_adapted_orig_gap"])]
    g = rep.groupby("key")
    return {c: 2 * np.sqrt(2) * np.sqrt((g[c].var(ddof=1)).mean()) for c in ["fp32", "int8"]}


LOG_EPOCH = re.compile(r"Epoch\s+(\d+)/(\d+) \| train_loss=[\d.]+ train_acc=([\d.]+)% \| val_loss=([\d.]+) val_acc=([\d.]+)%")
CONVERGED_PP = 0.5  # fallback threshold until the seed replicates carry test numbers (then noise_band's FP32 value)


def convergence(df, band_pp: float) -> pd.DataFrame:
    """Per run and stage (FP32, QAT), from the per-epoch lines Trainer logs. A run counts as converged when its last
    20% of epochs raised the best validation top-1 by less than band_pp: the cosine schedule has annealed to ~0 there
    (a fixed budget with the LR decayed to zero, Li, Yumer & Ramanan, ICLR 2020), so a longer one would act on a
    plateau. Logs are gitignored and PCAD holds the only copy -- fetch them first:
        rsync -a --prune-empty-dirs --include='*/' --include='phase_11_*/*/logs/*.log' --exclude='*' \\
            rsdsouza@gppd-hpc.inf.ufrgs.br:dnn_study/outputs/pcad/ outputs/pcad/"""
    rows = []
    for r in df.itertuples():
        for stage, log in (("fp32", r.results.parent / "logs" / f"{r.key}.log"),
                           ("qat", r.results.parent / "logs" / f"qat_{r.key}.log")):
            if not log.exists():
                continue
            ep = {int(m[1]): (int(m[2]), float(m[3]), float(m[4]), float(m[5]))  # a resumed epoch keeps its last line
                  for m in LOG_EPOCH.finditer(log.read_text(errors="ignore"))}
            if not ep:
                continue
            budget, last = ep[max(ep)][0], max(ep)
            va = [ep[e][3] for e in sorted(ep)]
            best, cut = max(va), int(0.8 * len(va))
            rows.append(dict(
                exp=r.exp, key=r.key, seed=r.seed, stage=stage, budget=budget, complete=last == budget,
                best_val=best, best_epoch=va.index(best) + 1,
                epoch_within_1pp=next(i + 1 for i, v in enumerate(va) if v >= best - 1),
                epoch_within_0p5pp=next(i + 1 for i, v in enumerate(va) if v >= best - 0.5),
                gain_last20pct=best - max(va[:cut]) if cut else float("nan"),
                min_val_loss_epoch=min(sorted(ep), key=lambda e: ep[e][2]), final_train_acc=ep[last][1]))
    c = pd.DataFrame(rows)
    if not c.empty:
        c["converged"] = c.complete & (c.gain_last20pct < band_pp)
    return c


def main():
    apply_report_style()
    df = load()
    TABLES.mkdir(parents=True, exist_ok=True)
    df.drop(columns="results").sort_values(["exp", "key", "seed"]).to_csv(TABLES / "all_runs.csv", index=False)
    band = noise_band(df)["fp32"]
    band = band if np.isfinite(band) else CONVERGED_PP
    conv = convergence(df, band)
    conv.to_csv(TABLES / "convergence.csv", index=False)
    if not conv.empty:
        done = conv[conv.complete]
        print(f"\n== convergence (last 20% of epochs < {band:.2f}pp): {int(done.converged.sum())}/{len(done)} complete runs; "
              f"median best epoch {done.best_epoch.median():.0f}, median last-20% gain {done.gain_last20pct.median():.2f}pp")
        print(done[~done.converged][["exp", "key", "seed", "stage", "best_epoch", "gain_last20pct"]].to_string(index=False))
    agg = kernel_seeds(df)
    agg.to_csv(TABLES / "kernel_seeds.csv", index=False)
    g = geometry_table(df); g.to_csv(TABLES / "geometry_factorial.csv", index=False)
    b = bn_block(df); b.to_csv(TABLES / "bn_block.csv", index=False)
    fig_convert(df)
    pd.set_option("display.width", 200)
    for name, t in [("kernel_seeds", agg), ("geometry_factorial", g), ("bn_block", b)]:
        print(f"\n== {name}\n{t.round(2).to_string(index=False)}")


if __name__ == "__main__":
    main()
