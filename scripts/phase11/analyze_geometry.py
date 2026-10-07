"""Phase 11 tables + figure 15, from the raw per-run summaries; also the shared loader and plot helpers of
scripts/phase11/{plot_kernel_comparison,factor_effects,design_figures}.py.

Reads every outputs/pcad/phase_11_*/<model>/results/*_summary.json -- the live experiments of
configs/experiments/phase_11_*.yaml (seeds 43/44 in the *_seed43/_seed44 ones), never outputs/pcad/archive_* -- and writes
  results/phase_11_geometry_analysis/{all_runs,main_grid,kernel_geometry,convergence}.csv
  results/figures_generated/phase_11_kernel_size_comparison/15_quantization_drop_where.png
Runs are picked by their factors (ml/model_registrations.py:CELL_FACTORS, via design_figures.frame), never by name.

Figure vocabulary (every Phase 11 figure): "layout original" = torchvision's AlexNet geometry (conv1 stride 4, three
3x3/2 max-pools -> 1x1 map before the classifier at 64x64); "layout 64px" = the adapted one (conv1 stride 2, two 2x2
max-pools -> 8x8 map). Baseline = BASE_KEY, the original AlexNet trained from scratch.

    python -m scripts.phase11.analyze_geometry
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from matplotlib.patches import Patch

from ml.plotting import BLUE, RED, TEXT_SECONDARY, apply_report_style

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results/phase_11_geometry_analysis"
FIGS = ROOT / "results/figures_generated/phase_11_kernel_size_comparison"
RUNS = (ROOT / "outputs/pcad", ROOT / "outputs/local")  # every runtime a run can come from (CLAUDE.md "Layout")
ARCHIVE = RUNS[0] / "archive_adamw_recipe"  # the superseded AdamW-recipe runs, same phase_11_*/<model>/results layout

BASE_KEY = "alexnet_k11-5-3_stride4_3pool3x3_map1_fcdrop_nobn"  # the original AlexNet (torchvision layout), from scratch
BASE_LABEL = "AlexNet original do zero (baseline)"
LAYOUTS = ("layout original = o do AlexNet torchvision: conv1 stride 4 + 3 max-pools 3×3/2 → mapa final 1×1 em 64×64\n"
           "layout 64px = adaptado a 64×64: conv1 stride 2 + 2 max-pools 2×2 → mapa final 8×8")


def load(*roots: Path) -> pd.DataFrame:
    """One row per run, accuracies on the held-out test set (Tiny ImageNet's official val split; the 90/10 split of
    train/ only picked each run's best epoch). fp32/qat/int8 are NaN for a run without the current quant_protocol
    (ml/quantization.py:QUANT_PROTOCOL); qat_raw/int8_raw/fp32_val are its validation-split numbers, all the superseded
    runs have (design_figures --archive draws those). `results` is the run's results dir, where the per-image
    *_test_logits.npz live (factor_effects' McNemar tests). roots default to RUNS (every runtime: a cell may have run on
    PCAD or the laptop, never on both); ARCHIVE reads the superseded runs.
    lat_* = forward latency per image at batch 1 (ms): FP32 on the training GPU and on CPU, INT8 on CPU.
    params_eff_m/macs_eff_m/int8_eff_mb = the same costs without the FC-head weights that only multiply AdaptiveAvgPool
    copies (replicated_weights): the cost axes of the figures; the as-built ones stay for the tables."""
    from ml.quantization import QUANT_PROTOCOL

    rows = []
    for p in sorted(q for r in (roots or RUNS) for q in r.glob("phase_11_*/*/results/*_summary.json")):
        d = json.loads(p.read_text())
        prov, key = d["config"]["provenance"], p.parents[1].name
        post_fix = d.get("quant_protocol") == QUANT_PROTOCOL
        nan = float("nan")
        rows.append(dict(
            exp=p.parents[2].name.removeprefix("phase_11_"), key=key, seed=d["config"]["experiment"]["seed"],
            fp32=d["test_fp32_top1"] if post_fix else nan, qat=d["test_qat_top1"] if post_fix else nan,
            int8=d["test_int8_top1"] if post_fix else nan, fp32_val=d["fp32_top1"], results=p.parent,
            qat_raw=d["qat_top1"], int8_raw=d["int8_top1"],
            params_m=d["params_m"], macs_m=d["macs"] / 1e6, fp32_mb=d["fp32_size_mb"], int8_mb=d["int8_size_mb"], best_ep=d["epochs"],
            ece=d.get("test_fp32_ece") if post_fix else nan, post_fix=post_fix, git_hash=prov.get("git_hash", "")[:7],
            git_dirty=prov.get("git_dirty"), gpu=prov.get("gpu_name"),
            # where the latencies were measured: same GPU name, different CPUs on tupi1/2 vs tupi3-6 (no cpu_model
            # before 2026-10-06 -> hostname)
            machine=f"{prov.get('gpu_name')} / {prov.get('cpu_model') or prov.get('hostname')}",
            lat_gpu=d.get("fp32_bs1_latency_ms_per_image"),
            lat_cpu=d.get("fp32_cpu_bs1_latency_ms_per_image"), lat_int8=d.get("int8_bs1_latency_ms_per_image")))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    assert not df[df.post_fix & df.git_dirty].shape[0], df[df.post_fix & df.git_dirty][["exp", "key", "git_hash"]]
    twice = df[df.duplicated(["exp", "key"], keep=False)]  # one canonical result per cell (CLAUDE.md): archive the other
    assert twice.empty, twice[["exp", "key", "results"]]
    rep = df.key.map(replicated_weights) / 1e6  # one int8 byte per weight; int8_mb is MiB (ml.reporting.disk_mb)
    df["params_eff_m"], df["macs_eff_m"], df["int8_eff_mb"] = df.params_m - rep, df.macs_m - rep, df.int8_mb - rep * 1e6 / 2**20
    df["drop_qat"] = df.fp32 - df.qat      # FP32 -> fake-quant (what QAT costs)
    df["drop_convert"] = df.qat - df.int8  # fake-quant -> real INT8 (what convert_to_int8 costs)
    df["drop_total"] = df.fp32 - df.int8
    return df


@lru_cache(maxsize=None)
def replicated_weights(key: str) -> int:
    """ml.reporting.replicated_fc_weights of a run's architecture, built on the meta device: shapes only, and a
    pretrained net is measured on its from-scratch twin's identical architecture, so no weights are downloaded."""
    import torch
    import ml.model_registrations  # noqa: F401
    from ml.registry import MODEL_REGISTRY
    from ml.reporting import replicated_fc_weights

    arch = next((k for k in (key.removesuffix("_pretrained"), f"{key}_scratch", key) if k in MODEL_REGISTRY), None)
    if arch is None:  # ponytail: an archived run whose name left the registry -- as built, only the --archive preview
        return 0
    with torch.device("meta"):
        return replicated_fc_weights(MODEL_REGISTRY[arch]["ctor"]())


def savefig(fig, name, figs=None):
    figs = figs or FIGS
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figs / name}")


def int8_bar(ax, x, value, width, color):
    """INT8 bar beside its FP32 one, in a light fill (nothing while the run's INT8 is still missing)."""
    if pd.notna(value):
        ax.bar(x, value, width, color=color, alpha=0.45, edgecolor="white")


def precision_handles(color=TEXT_SECONDARY):
    return [Patch(facecolor=color, label="barra cheia = FP32"), Patch(facecolor=color, alpha=0.45, label="barra clara ao lado = INT8")]


def reference_lines(ax, df, key=BASE_KEY, label=BASE_LABEL, int8=True):
    """Dashed (FP32) / dotted (INT8) lines at the baseline run's top-1; returns them as legend handles (none until
    the baseline has run)."""
    r = df[(df.key == key) & (df.seed == 42)]
    if r.empty:
        return []
    r = r.iloc[0]
    lines = [ax.axhline(r.fp32, color="#333333", ls="--", lw=1.3, zorder=4, label=f"{label}: FP32 {r.fp32:.1f}%")]
    if int8 and pd.notna(r.int8):
        lines.append(ax.axhline(r.int8, color="#333333", ls=":", lw=1.3, zorder=4, label=f"{label}: INT8 {r.int8:.1f}%"))
    return lines


def main_grid_table(df):
    """The main factorial (kernel x head x BN, 64px layout): mean/SD over the seeds a cell has, and where INT8 loses."""
    from scripts.phase11.design_figures import grid_cells

    d = df[df.key.isin(grid_cells(df).key)]  # grid_cells is seed 42 only; keep every seed of those cells
    return d.groupby(["kernels", "head", "bn"]).agg(
        n_seeds=("seed", "count"), fp32=("fp32", "mean"), fp32_sd=("fp32", "std"), int8=("int8", "mean"), int8_sd=("int8", "std"),
        drop_qat=("drop_qat", "mean"), drop_convert=("drop_convert", "mean"), params_m=("params_m", "first"),
        macs_m=("macs_m", "first")).reset_index()


def kernel_geometry_table(df):
    """Top-1 of every AlexNet cell without BN or pretraining (FC = AlexNet's head, with Dropout), by kernel x geometry x
    head (seed 42)."""
    from scripts.phase11.design_figures import GEOMS, grid_cells

    d = pd.concat([grid_cells(df, geom=g).assign(geometry=GEOMS[g].replace("\n", " ")) for g in GEOMS])
    d = d[~d.bn]
    return d[["kernels", "geometry", "head", "fp32", "int8", "drop_total", "macs_m", "params_m", "key"]]


def fig_convert(df):
    """Where INT8 loses its accuracy, per main-factorial cell (+ the baseline): in QAT training or in the conversion."""
    from scripts.phase11.design_figures import K4, baseline, cell_label, grid_cells

    cells = grid_cells(df)
    cells = cells[cells.kernels.isin(K4)].assign(k=lambda c: c.kernels.map(K4.index)).sort_values(["bn", "head", "k"])
    b = baseline(df)
    d = pd.concat([b.to_frame().T, cells]) if b is not None else cells
    if d.empty:
        print("15: no runs yet")
        return
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, r in enumerate(d.itertuples()):
        ax.bar(i, r.drop_qat, 0.7, color=BLUE)
        ax.bar(i, r.drop_convert, 0.7, bottom=max(r.drop_qat, 0), color=RED)
    if cells.bn.any():
        first_bn = list(d.key).index(cells[cells.bn].key.iloc[0])
        ax.axvline(first_bn - 0.5, color=TEXT_SECONDARY, lw=1, ls="--")
        for x, txt, ha in [(first_bn - 0.6, "sem BatchNorm ←", "right"), (first_bn - 0.4, "→ com BatchNorm", "left")]:
            ax.text(x, 0.97, txt, transform=ax.get_xaxis_transform(), ha=ha, va="top", fontsize=9, color=TEXT_SECONDARY)
    ax.axhline(0, color="k", lw=0.8)
    ax.margins(y=0.1)  # headroom for the BN labels above the tallest bar
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(["AlexNet original\n(baseline)" if r.key == BASE_KEY else cell_label(r, skip=("bn",)) for r in d.itertuples()],
                       rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Perda de top-1 (pp)")
    ax.set_title("Onde o INT8 perde acurácia: no treino QAT ou na conversão para INT8 real? (seed 42)\n"
                 "fatorial principal, layout 64px (conv1 stride 2, 2 max-pools 2×2)", fontsize=10)
    ax.legend(handles=[Patch(color=BLUE, label="FP32 → QAT (perda no treino com fake-quant)"),
                       Patch(color=RED, label="QAT → INT8 real (perda na conversão)")],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    savefig(fig, "15_quantization_drop_where.png")


# the cells replicated at seeds 43/44: read off their yaml, so the noise floor is whatever the design reruns
NOISE_KEYS = yaml.safe_load((ROOT / "configs/experiments/phase_11_kernel_head_bn_seed43.yaml").read_text())["models"]


def noise_band(df, keys=NOISE_KEYS):
    """2*sqrt(2)*pooled SD across seeds, for the cells replicated at seeds 42-44 (phase_11_kernel_head_bn_seed43/44):
    the training-noise floor a single-seed difference has to clear (Bouthillier et al., MLSys 2021: seed variance is a
    first-order source of variance in benchmark comparisons). NaN until the replicates have run."""
    rep = df[df.key.isin(keys)] if len(df) else df
    if rep.empty:
        return {"fp32": float("nan"), "int8": float("nan")}
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
    from scripts.phase11.design_figures import frame

    apply_report_style()
    df = frame(archive=False)
    if df.empty:
        print("no Phase 11 run has a summary yet")
        return
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
    tables = [("main_grid", main_grid_table(df)), ("kernel_geometry", kernel_geometry_table(df))]
    for name, t in tables:
        t.to_csv(TABLES / f"{name}.csv", index=False)
    fig_convert(df)
    pd.set_option("display.width", 200)
    for name, t in tables:
        print(f"\n== {name}\n{t.round(2).to_string(index=False)}")


if __name__ == "__main__":
    main()
