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
FIGURES = {f"{n}.png" for n in ["02_kernel_original_layout", "03_kernel_64px_layout", "04_kernel_vgg16", "05_head_fc_vs_gap",
                                "06_conv1_stride", "07_pooling", "08_dropout", "09_pretraining", "10_batchnorm",
                                "11_compensation_block", "14_factorial_matched_pairs", "15_quantization_drop_where",
                                "16_pareto_accuracy_cost", "17_main_factorial_grid", "18_kernel_by_geometry",
                                "19_quantization_robustness", "20_latency_vs_macs", "21_training_curves"]}


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
                       "epochs": 400, "test_fp32_ece": 0.05, "fp32_bs1_latency_ms_per_image": 1.0,
                       "fp32_cpu_bs1_latency_ms_per_image": 5.0, "int8_bs1_latency_ms_per_image": 3.0}
            (run / "results" / f"{m}_summary.json").write_text(json.dumps(summary))
            stats = {"features.0": {"activation_in": {"max": 2.6, "p999": 2.6}},
                     "features.3": {"activation_in": {"max": 2 + 10 * rng.random(), "p999": 2.0}}}
            (run / "results" / f"{m}_layer_stats.json").write_text(json.dumps(stats))
            (run / "logs" / f"{m}.log").write_text("".join(
                f"Epoch {e:3d}/500 | train_loss=1.0 train_acc={2 * e:.2f}% | val_loss=2.0 val_acc={e:.2f}% val_top5=50.00%\n"
                for e in range(1, 4)))


@pytest.mark.parametrize("subset", ["design", "pilot"])
def test_every_phase_11_figure_renders(tmp_path, monkeypatch, subset):
    from scripts.phase11 import analyze_geometry, design_figures, factor_effects, plot_kernel_comparison

    runs, figs, tables = tmp_path / "runs", tmp_path / "figs", tmp_path / "tables"
    _fake_runs(runs, (lambda e, m: True) if subset == "design" else (lambda e, m: m in PILOT and "_seed" not in e))
    for mod, name, value in [(analyze_geometry, "RUNS", (runs,)), (design_figures, "RUNS", (runs,)), (analyze_geometry, "FIGS", figs),
                             (design_figures, "FIGS", figs), (analyze_geometry, "TABLES", tables), (factor_effects, "TABLES", tables)]:
        monkeypatch.setattr(mod, name, value)
    analyze_geometry.main()
    plot_kernel_comparison.main()
    factor_effects.main()
    design_figures.main([])
    written = {p.name for p in figs.glob("*.png")}
    if subset == "design":
        assert written == FIGURES, (FIGURES - written, written - FIGURES)
        assert {p.name for p in tables.glob("*.csv")} >= {"all_runs.csv", "main_grid.csv", "kernel_geometry.csv", "factor_effects.csv"}
    else:  # a half-finished queue: whatever has data renders, the rest is skipped, nothing crashes
        assert {"02_kernel_original_layout.png", "16_pareto_accuracy_cost.png", "17_main_factorial_grid.png"} <= written
