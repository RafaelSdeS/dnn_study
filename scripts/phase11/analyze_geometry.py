"""Phase 11 tables + figure 12, from the raw per-run summaries; also the shared loader and plot helpers of
scripts/phase11/{plot_kernel_comparison,factor_effects,design_figures}.py.

Reads every outputs/pcad/phase_11_*/<model>/results/*_summary.json -- the live experiments of
configs/experiments/phase_11_*.yaml (seeds 43/44 in the *_seed43/_seed44 ones), never outputs/pcad/archive_* -- and writes
  results/phase_11_geometry_analysis/{all_runs,main_grid,kernel_geometry,convergence}.csv
  results/figures_generated/phase_11_kernel_size_comparison/12_quantization_drop_where.png
Runs are picked by their factors (ml/model_registrations.py:CELL_FACTORS, via design_figures.frame), never by name.

Figure vocabulary (every Phase 11 figure): "layout original" = torchvision's AlexNet geometry (conv1 stride 4, three
3x3/2 max-pools -> 1x1 map before the classifier at 64x64); "layout 64px" = the adapted one (conv1 stride 2, two 2x2
max-pools -> 8x8 map). Baseline = BASE_KEY, torchvision's AlexNet unadapted (1x1 map) trained from scratch; beside it
HEALTHY_KEY, the same net at a 6x6 map (reference_lines).

    python -m scripts.phase11.analyze_geometry
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.table import Table
from matplotlib.text import Text

from ml.plotting import BLUE, RED, TEXT_PRIMARY, TEXT_SECONDARY, apply_report_style

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results/phase_11_geometry_analysis"
FIGS = ROOT / "results/figures_generated/phase_11_kernel_size_comparison"
RUNS = (ROOT / "outputs/pcad", ROOT / "outputs/local")  # every runtime a run can come from (CLAUDE.md "Layout")
ARCHIVE = RUNS[0] / "archive_adamw_recipe"  # the superseded AdamW-recipe runs, same phase_11_*/<model>/results layout

BASE_KEY = "alexnet_k11-5-3_stride4_3pool3x3_map1_fcdrop_nobn"  # torchvision's AlexNet, unadapted (1x1 map at 64 px), from scratch
BASE_LABEL = "AlexNet torchvision sem adaptar (mapa 1×1)"
# The same net at the 6x6 map it was designed for (Tiny ImageNet enlarged to 224, phase_11_input224): the AlexNet reference
# at a healthy map, drawn beside BASE_KEY wherever that one is a line -- BASE_KEY's 35.5% is the 1x1 map's, not AlexNet's
# (docs/logs/PHASE11_LOG.md "Progress 2026-10-19").
HEALTHY_KEY = "alexnet_k11-5-3_stride4_3pool3x3_map6_fcdrop_nobn_in224"
HEALTHY_LABEL = "AlexNet torchvision com entrada\nampliada a 224 px (mapa 6×6)"
# the other families' own reference nets, for their rows of the factor figures (figure 03 draws VGG_BASE too)
VGG_BASE = ("vgg16_k3x3_stride1_5pool2x2_map2_fcdrop_bn", "VGG16 original (3×3)")
RESNET_BASE = ("resnet18_k7-3_stride2_1pool3x3_map2_gap_bn", "ResNet-18 original (do zero)")
LAYOUTS = ("layout original = o do AlexNet torchvision: conv1 stride 4 + 3 max-pools 3×3/2 → mapa final 1×1 em 64×64\n"
           "layout 64px = adaptado a 64×64: conv1 stride 2 + 2 max-pools 2×2 → mapa final 8×8")

# A run whose best FP32 epoch falls before EARLY_FRAC of its budget was picked while the cosine LR was still high: QAT
# starts from that un-annealed checkpoint and its own LR decay adds 2-4 pp, so its FP32 -> INT8 is not a quantization
# effect (docs/logs/PHASE11_LOG.md "Early best FP32 epoch", 2026-10-13). The figures mark these runs; none is dropped.
EARLY_FRAC = 0.8
EARLY_GRAY = "#6b6b6b"
EARLY_LABEL = "melhor época FP32 antes de 80% do treino: o ganho no INT8 é treino extra do QAT, não quantização"


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
            qat_best_ep=d.get("qat_best_epoch"),  # 0-based like best_ep: the QAT epoch the INT8 model is converted from
            early_best=bool(d.get("epochs_budget")) and d["epochs"] + 1 < EARLY_FRAC * d["epochs_budget"],  # epochs is 0-based
            ece=d.get("test_fp32_ece") if post_fix else nan, post_fix=post_fix, git_hash=prov.get("git_hash", "")[:7],
            git_dirty=prov.get("git_dirty"), gpu=prov.get("gpu_name"),
            # where the latencies were measured: same GPU name, different CPUs on tupi1/2 vs tupi3-6 (no cpu_model
            # before 2026-10-06 -> hostname)
            machine=f"{prov.get('gpu_name')} / {prov.get('cpu_model') or prov.get('hostname')}",
            host=prov.get("hostname"),  # nyx and hefesto share one GPU / CPU model
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
    from ml.model_registrations import CELL_FACTORS
    from ml.registry import MODEL_REGISTRY
    from ml.reporting import replicated_fc_weights

    arch = next((k for k in (key.removesuffix("_pretrained"), f"{key}_scratch", key) if k in MODEL_REGISTRY), None)
    if arch is None:  # ponytail: an archived run whose name left the registry -- as built, only the --archive preview
        return 0
    side = CELL_FACTORS.get(arch, {}).get("input_size", 64)  # at 224 the FC head sees torchvision's 6x6: nothing copied
    with torch.device("meta"):
        return replicated_fc_weights(MODEL_REGISTRY[arch]["ctor"](), input_size=(1, 3, side, side))


DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")  # only digit.digit: 'et al.' and the like stay


class _Comma(mticker.Formatter):
    """Any tick formatter, with a decimal comma: delegates to the one it wraps, then swaps the point."""

    def __init__(self, inner):
        self.inner = inner

    def set_locs(self, locs):
        self.inner.set_locs(locs)

    def __call__(self, x, pos=None):
        return DECIMAL.sub(",", self.inner(x, pos))

    def format_ticks(self, values):
        return [DECIMAL.sub(",", s) for s in self.inner.format_ticks(values)]

    def get_offset(self):
        return DECIMAL.sub(",", self.inner.get_offset())


def comma_decimals(fig):
    """Every number of a figure with a decimal comma, as report/ic_report.tex writes them (pt-BR): its texts (titles,
    labels, legends, table cells) now, its ticks when they are drawn -- one pass here instead of a format per f-string."""
    texts = fig.findobj(Text) + [c.get_text() for t in fig.findobj(Table) for c in t.get_celld().values()]
    for t in texts:
        t.set_text(DECIMAL.sub(",", t.get_text()))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):  # shared axes share one formatter: wrapped once
            if not isinstance(axis.get_major_formatter(), _Comma):
                axis.set_major_formatter(_Comma(axis.get_major_formatter()))
            if not isinstance(axis.get_minor_formatter(), _Comma):
                axis.set_minor_formatter(_Comma(axis.get_minor_formatter()))


def savefig(fig, name, figs=None):
    figs = figs or FIGS
    figs.mkdir(parents=True, exist_ok=True)
    comma_decimals(fig)
    fig.savefig(figs / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figs / name}")


def early_hatch(early):
    """bar() kwargs that hatch an early-best run's bar; none otherwise, so an unflagged bar is drawn exactly as before."""
    return dict(hatch="///", edgecolor=EARLY_GRAY) if early else {}


def early_ring(ax, x, y, s=170):
    """Grey ring around an early-best run's point."""
    ax.scatter(x, y, s=s, facecolors="none", edgecolors=EARLY_GRAY, lw=1.4, zorder=6)


def early_handle(kind):
    """Legend entry for the early-best mark: 'ring' (scatter figures) or 'hatch' (bar figures)."""
    if kind == "ring":
        return Line2D([], [], marker="o", mfc="none", mec=EARLY_GRAY, mew=1.4, ms=12, ls="", label=EARLY_LABEL)
    return Patch(facecolor="white", hatch="///", edgecolor=EARLY_GRAY, label=EARLY_LABEL)


def int8_bar(ax, x, value, width, color, early=False):
    """INT8 bar beside its FP32 one, in a light fill (nothing while the run's INT8 is still missing); hatched for an
    early-best run."""
    if pd.notna(value):
        ax.bar(x, value, width, color=color, alpha=0.45, **{"edgecolor": "white", **early_hatch(early)})


def precision_handles(color=TEXT_SECONDARY):
    return [Patch(facecolor=color, label="barra cheia = FP32"), Patch(facecolor=color, alpha=0.45, label="barra clara ao lado = INT8")]


def healthy_run():
    """HEALTHY_KEY's FP32 / INT8 test top-1 (seed 42, current quant protocol), None until that run has finished. It is
    outside core() (input_size 224), so its summary is read directly."""
    from ml.quantization import QUANT_PROTOCOL

    for root in RUNS:
        p = root / "phase_11_input224" / HEALTHY_KEY / "results" / f"{HEALTHY_KEY}_summary.json"
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("quant_protocol") == QUANT_PROTOCOL:
                return {"fp32": d["test_fp32_top1"], "int8": d["test_int8_top1"]}


def reference_lines(ax, df, key=BASE_KEY, label=BASE_LABEL, stage="fp32", text="right"):
    """One dashed line at the baseline run's top-1 (stage), named on the plot with both its numbers -- its FP32 and
    INT8 sit ~0.2 pp apart, two lines overlapped. For BASE_KEY a dotted line beside it at HEALTHY_KEY's (the same net at
    a healthy map), once that run has finished. text: "right" = outside the axes (bar figures: bars cover the inside),
    "left" = inside, above the line, None = line only (a middle panel). Nothing until the baseline has run."""
    r = df[(df.key == key) & (df.seed == 42)]
    if r.empty or pd.isna(r.iloc[0][stage]):
        return
    lines = [(r.iloc[0], label, "--")]
    healthy = healthy_run() if key == BASE_KEY else None
    if healthy:
        lines.append((healthy, HEALTHY_LABEL, ":"))
    for r, name, ls in lines:
        ax.axhline(r[stage], color=TEXT_PRIMARY, ls=ls, lw=1.3, zorder=4)
        if text:
            out = text == "right"
            ax.text(1.01 if out else 0.01, r[stage], f"{name}\n{r['fp32']:.1f}% FP32 · {r['int8']:.1f}% INT8",
                    transform=ax.get_yaxis_transform(), ha="left", va="center" if out else "bottom", fontsize=8.5, clip_on=False,
                    zorder=6, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5))


def main_grid_table(df):
    """The main factorial (kernel x head x BN, 64px layout): mean/SD over the seeds a cell has, and where INT8 loses."""
    from scripts.phase11.design_figures import grid_cells

    d = df[df.key.isin(grid_cells(df).key)]  # grid_cells is seed 42 only; keep every seed of those cells
    return d.groupby(["kernels", "head", "bn"]).agg(
        n_seeds=("seed", "count"), fp32=("fp32", "mean"), fp32_sd=("fp32", "std"), int8=("int8", "mean"), int8_sd=("int8", "std"),
        drop_qat=("drop_qat", "mean"), drop_convert=("drop_convert", "mean"), params_m=("params_m", "first"),
        macs_m=("macs_m", "first"), early_best=("early_best", "any")).reset_index()


def kernel_geometry_table(df):
    """Top-1 of every AlexNet cell without BN or pretraining (FC = AlexNet's head, with Dropout), by kernel x geometry x
    head (seed 42)."""
    from scripts.phase11.design_figures import GEOMS, grid_cells

    d = pd.concat([grid_cells(df, geom=g).assign(geometry=GEOMS[g].replace("\n", " ")) for g in GEOMS])
    d = d[~d.bn]
    return d[["kernels", "geometry", "head", "fp32", "int8", "drop_total", "early_best", "macs_m", "params_m", "key"]]


def fig_convert(df):
    """Where INT8 - FP32 comes from, per main-factorial cell (+ the baseline): QAT training (QAT - FP32), then the
    conversion (INT8 - QAT) stacked on it -- the bars end at INT8 - FP32, negative = loses (figures 11 and 16's sign)."""
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
        ax.bar(i, -r.drop_qat, 0.7, color=BLUE, **early_hatch(r.early_best))
        ax.bar(i, -r.drop_convert, 0.7, bottom=-r.drop_qat, color=RED)
    if cells.bn.any():
        first_bn = list(d.key).index(cells[cells.bn].key.iloc[0])
        ax.axvline(first_bn - 0.5, color=TEXT_SECONDARY, lw=1, ls="--")
        for x, txt, ha in [(first_bn - 0.6, "sem BatchNorm ←", "right"), (first_bn - 0.4, "→ com BatchNorm", "left")]:
            ax.text(x, 0.97, txt, transform=ax.get_xaxis_transform(), ha=ha, va="top", fontsize=9, color=TEXT_SECONDARY)
    ax.axhline(0, color="k", lw=0.8)
    if b is not None:
        ax.axvspan(-0.5, 0.5, color="#e6e6e6", zorder=0)  # the baseline's bar, to compare the others against
    ax.margins(y=0.1)  # headroom for the BN labels above the tallest bar
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(["AlexNet sem adaptar\n(baseline)" if r.key == BASE_KEY else cell_label(r, skip=("bn",)) for r in d.itertuples()],
                       rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("INT8 − FP32 (pp; negativo = perde no INT8)")
    ax.set_title("De onde vem a diferença INT8 − FP32: do treino QAT ou da conversão para INT8 real? (seed 42)\n"
                 "fatorial principal, layout 64px (conv1 stride 2, 2 max-pools 2×2); a barra termina em INT8 − FP32", fontsize=11)
    fig.tight_layout()
    fig.legend(handles=[Patch(color=BLUE, label="QAT − FP32 (treino com fake-quant)"),
                        Patch(color=RED, label="INT8 − QAT (conversão para INT8 real)")]
               + ([early_handle("hatch")] if d.early_best.any() else []),
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)
    savefig(fig, "12_quantization_drop_where.png")


# the cells replicated at seeds 43/44: read off every *_seed43 yaml, so the noise floors are whatever the design reruns
NOISE_KEYS = sorted({m for p in (ROOT / "configs/experiments").glob("phase_11_*_seed43.yaml")
                     for m in yaml.safe_load(p.read_text())["models"]})
# One floor per regime (family, BN), since 2026-10-13 (PHASE11_LOG "Noise floors per regime and a ResNet block"): AlexNet
# GAP + BN (the reference: every cell whose regime has no floor of its own -- families, ResNet, all GAP/BN-like), AlexNet
# FC + Dropout without BN (every no-BN cell: the early best epochs move a run 2-4 pp), VGG16 FC + Dropout.
REFERENCE_REGIME = ("alexnet", True)
REGIME_LABEL = {("alexnet", True): "AlexNet com BN", ("alexnet", False): "AlexNet sem BN", ("vgg16", True): "VGG16"}


def noise_bands(df) -> dict:
    """{(family, bn): noise_band} for every regime the seed yamls replicate."""
    rep = df[df.key.isin(NOISE_KEYS)] if len(df) else df
    return {(fam, bool(bn)): noise_band(g, set(g.key)) for (fam, bn), g in rep.groupby(["family", "bn"])} if len(rep) else {}


def band_for(bands, family, bn) -> dict:
    """A cell's band: its regime's floor (NaN while that floor's replicates run), else the reference regime's."""
    nan = {"fp32": float("nan"), "int8": float("nan")}
    return bands.get((family, bool(bn) if pd.notna(bn) else True), bands.get(REFERENCE_REGIME, nan))


def pair_band(a: dict, b: dict) -> dict:
    """Band of a single-seed difference between two cells with floors a and b: 2*sqrt(sd_a^2 + sd_b^2), i.e. the RMS of
    their 2*sqrt(2)*sd bands -- the plain band when both share a regime."""
    return {c: float(np.sqrt((a[c] ** 2 + b[c] ** 2) / 2)) for c in ("fp32", "int8")}


def noise_band(df, keys):
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


def design_status(df) -> pd.DataFrame:
    """One row per planned run -- each experiment x model x seed of the phase_11_*.yaml files -- with the machine it
    ran on and its test top-1, blank while it has not: what ran where (PCAD, nyx) and what is left, in one table."""
    from configs.loader import load_config

    plan = pd.DataFrame([dict(exp=p.stem.removeprefix("phase_11_"), key=m, seed=cfg["seed"])
                         for p in sorted((ROOT / "configs/experiments").glob("phase_11_*.yaml"))
                         for cfg in [load_config(f"experiments/{p.name}")] for m in cfg["models"]])
    status = plan.merge(df[["exp", "key", "seed", "host", "fp32", "qat", "int8"]], on=["exp", "key", "seed"], how="left",
                        indicator=True)
    return status.assign(done=status.pop("_merge").eq("both"))


def main():
    from scripts.phase11.design_figures import core, frame

    apply_report_style()
    everything = frame(archive=False, everything=True)
    if everything.empty:
        print("no Phase 11 run has a summary yet")
        return
    TABLES.mkdir(parents=True, exist_ok=True)
    everything.drop(columns="results").sort_values(["exp", "key", "seed"]).to_csv(TABLES / "all_runs.csv", index=False)
    status = design_status(everything)
    status.to_csv(TABLES / "design_status.csv", index=False)
    print(f"== design status: {status.done.sum()}/{len(status)} planned runs done "
          f"({status.host.value_counts().to_dict()}); {TABLES / 'design_status.csv'}")
    df = core(everything)
    band = band_for(noise_bands(df), *REFERENCE_REGIME)["fp32"]  # the strictest floor
    band = band if np.isfinite(band) else CONVERGED_PP
    conv = convergence(df, band)
    conv.to_csv(TABLES / "convergence.csv", index=False)
    if not conv.empty:
        done = conv[conv.complete]
        print(f"\n== convergence (last 20% of epochs < {band:.2f}pp): {int(done.converged.sum())}/{len(done)} complete runs; "
              f"median best epoch {done.best_epoch.median():.0f}, median last-20% gain {done.gain_last20pct.median():.2f}pp")
        print(done[~done.converged][["exp", "key", "seed", "stage", "best_epoch", "gain_last20pct"]].to_string(index=False))
    early = df[df.early_best]
    if len(early):
        print(f"\n== best FP32 epoch (0-based) before {EARLY_FRAC:.0%} of the budget, marked in the figures: {len(early)} runs")
        print(early[["exp", "key", "seed", "best_ep", "fp32", "qat", "int8"]].to_string(index=False))
    tables = [("main_grid", main_grid_table(df)), ("kernel_geometry", kernel_geometry_table(df))]
    for name, t in tables:
        t.to_csv(TABLES / f"{name}.csv", index=False)
    fig_convert(df)
    pd.set_option("display.width", 200)
    for name, t in tables:
        print(f"\n== {name}\n{t.round(2).to_string(index=False)}")


if __name__ == "__main__":
    main()
