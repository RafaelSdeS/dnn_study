"""Every Phase 11 figure and table on a fake run tree: each model of the phase_11_*.yaml files (or only the 5 pilot
cells) gets a summary, layer stats and a log with made-up numbers. The scripts pick runs by their factors, so before the
real runs land this is the only check that they still find the design's cells -- and survive a half-finished queue."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from configs.loader import load_config
from ml.quantization import QUANT_PROTOCOL

matplotlib.use("Agg")
EXPERIMENTS = sorted(p.stem for p in (Path(__file__).resolve().parents[1] / "configs/experiments").glob("phase_11_*.yaml"))
PILOT = {"alexnet_k11-5-3_stride2_2pool2x2_map8_fc_nobn", "alexnet_k2x2_stride2_2pool2x2_map8_fc_nobn",
         "alexnet_k3x3_stride2_2pool2x2_map8_gap_bn", "vgg16_k3x3_stride1_5pool2x2_map2_fcdrop_bn",
         "alexnet_k11-5-3_stride4_3pool3x3_map1_fcdrop_nobn"}
FIGURES = {f"{n}.png" for n in ["01_kernel_original_layout", "02_kernel_64px_layout", "03_kernel_vgg16", "04_head_fc_vs_gap",
                                "05_conv1_stride", "06_pooling", "07_dropout", "08_pretraining", "09_batchnorm",
                                "10_compensation_block", "11_factorial_matched_pairs", "12_quantization_drop_where",
                                "13_pareto_accuracy_cost", "14_main_factorial_grid", "15_kernel_by_geometry",
                                "16_quantization_robustness", "17_latency_vs_macs", "18_training_curves", "19_qat_curves", "20_int8_prediction_flips",
                                "21_kernel_rank_critical_difference", "22_effective_receptive_field"]}


def _early(m):  # the fake runs whose best FP32 epoch comes early, so every figure draws its early-best mark
    return m.endswith("_fcdrop_nobn")


def _fake_runs(root, keep):
    rng = np.random.default_rng(0)
    for exp in EXPERIMENTS:
        cfg = load_config(f"experiments/{exp}.yaml")
        for m in cfg["models"]:
            if not keep(exp, m):
                continue
            run = root / exp / m
            (run / "results").mkdir(parents=True)
            (run / "logs").mkdir()
            fp32 = 30 + 20 * rng.random()
            summary = {"config": {"provenance": {"git_hash": "abc1234", "git_dirty": False, "gpu_name": "NVIDIA GeForce RTX 4090"},
                                  "experiment": {"seed": cfg["seed"]}},
                       "quant_protocol": QUANT_PROTOCOL, "test_fp32_top1": fp32, "test_qat_top1": fp32 - 0.5,
                       "test_int8_top1": fp32 - 1, "fp32_top1": fp32, "qat_top1": fp32 - 0.5, "int8_top1": fp32 - 1,
                       "params_m": 1 + rng.random(), "macs": 1e8 * (1 + rng.random()), "fp32_size_mb": 10.0, "int8_size_mb": 2.5,
                       "epochs": 50 if _early(m) else 400, "epochs_budget": 500, "test_fp32_ece": 0.05, "fp32_bs1_latency_ms_per_image": 1.0,
                       "fp32_cpu_bs1_latency_ms_per_image": 5.0, "int8_bs1_latency_ms_per_image": 3.0}
            (run / "results" / f"{m}_summary.json").write_text(json.dumps(summary))
            stats = {"features.0": {"activation_in": {"max": 2.6, "p999": 2.6}},
                     "features.3": {"activation_in": {"max": 2 + 10 * rng.random(), "p999": 2.0}}}
            (run / "results" / f"{m}_layer_stats.json").write_text(json.dumps(stats))
            labels = rng.integers(0, 200, 50)  # test logits: figure 20's flips and the McNemar tests read them
            for stage in ("fp32", "int8"):
                np.savez(run / "results" / f"{m}_{stage}_test_logits.npz", logits=rng.random((50, 200)).astype(np.float16), labels=labels)
            (run / "logs" / f"{m}.log").write_text("".join(
                f"Epoch {e:3d}/500 | train_loss=1.0 train_acc={2 * e:.2f}% | val_loss=2.0 val_acc={e:.2f}% val_top5=50.00%\n"
                for e in range(1, 4)))


@pytest.mark.parametrize("subset", ["design", "pilot"])
def test_every_phase_11_figure_renders(tmp_path, monkeypatch, subset):
    from scripts.phase11 import analyze_geometry, design_figures, factor_effects, plot_kernel_comparison, receptive_field

    runs, figs, tables = tmp_path / "runs", tmp_path / "figs", tmp_path / "tables"
    _fake_runs(runs, (lambda e, m: True) if subset == "design" else (lambda e, m: m in PILOT and "_seed" not in e))
    for mod, name, value in [(analyze_geometry, "RUNS", (runs,)), (design_figures, "RUNS", (runs,)), (analyze_geometry, "FIGS", figs),
                             (design_figures, "FIGS", figs), (analyze_geometry, "TABLES", tables), (factor_effects, "TABLES", tables)]:
        monkeypatch.setattr(mod, name, value)
    analyze_geometry.main()
    plot_kernel_comparison.main()
    factor_effects.main()
    design_figures.main([])
    receptive_field.main()  # no checkpoints in the fake tree: every panel says so, nothing is loaded
    written = {p.name for p in figs.glob("*.png")}
    df = design_figures.frame()
    assert set(df[df.early_best].key) == {k for k in df.key if _early(k)}  # epoch 51 of 500 is flagged, 401 is not
    if subset == "design":
        assert written == FIGURES, (FIGURES - written, written - FIGURES)
        assert {p.name for p in tables.glob("*.csv")} >= {"all_runs.csv", "main_grid.csv", "kernel_geometry.csv", "factor_effects.csv"}
    else:  # a half-finished queue: whatever has data renders, the rest is skipped, nothing crashes
        assert {"01_kernel_original_layout.png", "13_pareto_accuracy_cost.png", "14_main_factorial_grid.png"} <= written


def test_figure_numbers_get_a_decimal_comma(tmp_path):
    """savefig writes numbers as report/ic_report.tex does (pt-BR): texts, table cells and ticks; only digit.digit."""
    import matplotlib.pyplot as plt
    from scripts.phase11.analyze_geometry import savefig

    fig, ax = plt.subplots()
    ax.plot([0, 0.5], [0, 2.5])
    ax.set_title("35.5% FP32 · p99.9 · Hooker et al. 2019")
    tab = ax.table([["1.44"]])
    savefig(fig, "x.png", tmp_path)
    assert ax.get_title() == "35,5% FP32 · p99,9 · Hooker et al. 2019"
    assert tab[0, 0].get_text().get_text() == "1,44"
    ticks = ax.xaxis.get_major_formatter().format_ticks(ax.get_xticks())
    assert "0,5" in ticks and not any("." in t for t in ticks), ticks


# The design, contrast by contrast (docs/logs/PHASE11_LOG.md "Minimal design" + "Protocol audit", 2026-10-07): how many
# matched pairs the yamls give each one. A yaml edit that drops or breaks a contrast fails here instead of in a figure
# months later -- as the 2026-10-07 FC -> FC + Dropout swap did (the head contrast fell from 17 pairs to 6 until
# factor_effects paired GAP with the real FC head). Change these numbers only with the design.
DESIGN_PAIRS = {"Kernel 11-5-3-3-3 → 3×3": 9, "Kernel 3×3 → 2×2": 13, "Kernel 11-5-3-3-3 → 2×2": 9,
                "Kernel 3×3 → alternado 3-2": 12, "3×3 → dois 3×3 empilhados por estágio": 2,
                "2×2 → dois 2×2 empilhados por estágio": 2, "Cabeça FC (+ Dropout) → GAP": 19, "BatchNorm não → sim": 8,
                "Stride da conv1 4 → 2": 8, "3 max-pools 3×3 → 2 max-pools 2×2": 8, "Dropout 0 → 0,5 (só cabeça FC)": 6,
                "Pré-treino ImageNet": 4,
                # the ResNet-18 block (2026-10-13): +1 each to 3x3 -> 2x2 / alternating / pretraining above, and its own
                "Kernel 7-3 (ResNet) → 3×3": 1, "Kernel 7-3 (ResNet) → 2×2": 1, "Kernel blocos 3×3 → 2×2, stem 7×7 mantido": 1,
                "Kernel stem 7×7 → 2×2, blocos 2×2": 1, "Kernel 3×3 → 2×2, stem profundo": 1,
                "Stem 3×3 → três 3×3 (ResNet-C)": 1, "Stem 2×2 → três 2×2": 1,
                # the large-stem block (2026-10-14): +2 to the head contrast above, and its own, at GAP and FC + Dropout
                "Conv1 3×3 → 11×11, corpo 3×3": 2, "Conv1 2×2 → 11×11, corpo 2×2": 2, "Conv2 3×3 → 5×5, conv1 11×11": 2,
                "Corpo 3×3 → 2×2, conv1 11×11 mantida": 2}


def test_the_design_has_every_contrast_and_no_orphan_cell():
    from scripts.phase11.factor_effects import design_cells, pairs_of

    cells, pairs = design_cells(), pairs_of(design_cells())
    assert pairs.groupby("contrast").size().to_dict() == DESIGN_PAIRS
    assert set(cells.cell) == set(pairs.cell_a) | set(pairs.cell_b), "a design cell no contrast reads: compute for nothing"
    # the noise floors, one per regime (2026-10-13): AlexNet GAP + BN and FC + Dropout without BN, 4 kernels x 3 seeds
    # each; VGG16's own head (FC + Dropout), 3x3 and 2x2 x 3 seeds
    seeds = [set(load_config(f"experiments/phase_11_kernel_head_bn_seed{s}.yaml")["models"]) for s in (43, 44)]
    k64 = cells[(cells.family == "alexnet") & (cells.stride == 2) & (cells.pooling == "2pool2x2")
                & cells.kernels.isin(["k11-5-3", "k3x3", "k2x2", "kalt3-2"]) & ~cells.pretrained]
    assert seeds[0] == seeds[1] == set(k64[(k64["head"] == "gap") & k64.bn].cell) | set(k64[k64.dropout & ~k64.bn].cell)
    vgg = [set(load_config(f"experiments/phase_11_vgg_kernel_head_seed{s}.yaml")["models"]) for s in (43, 44)]
    assert vgg[0] == vgg[1] == set(cells[(cells.family == "vgg16") & cells.dropout & ~cells.pretrained
                                         & cells.kernels.isin(["k3x3", "k2x2"])].cell)
