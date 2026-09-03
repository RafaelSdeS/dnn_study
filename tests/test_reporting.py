"""make_run_summary/build_comparison_table coverage - previously untested despite feeding
every results table (final_comparison.csv, experiment_summary.json)."""
import pytest
from ml.reporting import build_comparison_table, make_run_summary


def _fit_results():
    return {
        "best_epoch": 3,
        "best_val_top1": 55.0,
        "best_val_top5": 80.0,
        "final_val_top1": 54.0,
        "final_val_top5": 79.0,
        "best_val_loss": 1.2,
        "total_training_time_s": 100.0,
        "history": {
            "train_loss": [2.0, 1.5, 1.0],
            "epoch_time_s": [10.0, 10.0, 10.0],
            "peak_gpu_mem_mb": [512, 600, 550],
            "images_per_sec": [100.0, 110.0],
            "avg_batch_time_s": [0.1, 0.09],
            "cpu_percent": [50.0, 55.0],
            "ram_used_mb": [2000.0, 2100.0],
            "gpu_power_avg_w": [None, None],
            "gpu_utilization_pct": [None, None],
            "gpu_temp_avg_c": [None, None],
            "gpu_energy_wh": [None, None],
        },
    }


def test_make_run_summary_derives_ratios_and_gaps_from_raw_inputs():
    summary = make_run_summary(
        name="alexnet_fire", mode="fp32_qat_int8", fit_results=_fit_results(),
        fp32_eval={"top1": 55.0, "top5": 80.0, "loss": 1.2},
        params_m=2.0, fp32_size_mb=8.0, int8_size_mb=2.0,
        fp32_benchmark={"latency_ms_per_image": 5.0, "throughput_img_per_s": 200.0},
        flops_results={"macs": 1_000_000, "flops": 2_000_000},
        int8_eval={"top1": 50.0, "top5": 78.0, "loss": 1.4},
        int8_benchmark={"latency_ms_per_image": 2.0, "throughput_img_per_s": 500.0},
        fp32_gzip_mb=4.0, int8_gzip_mb=1.0,
    )

    assert summary["model_name"] == "alexnet_fire"
    assert summary["compression_ratio"] == 4.0  # 8 / 2
    assert summary["fp32_gzip_ratio"] == 2.0  # 8 / 4
    assert summary["int8_gzip_ratio"] == 2.0  # 2 / 1
    assert summary["fp32_top1_top5_gap"] == 25.0  # 80 - 55
    assert summary["int8_top1_top5_gap"] == 28.0  # 78 - 50
    assert summary["quantization_drop_top1"] == 5.0  # 55 - 50
    assert summary["param_efficiency_top1_per_m"] == 27.5  # best_val_top1(55) / params_m(2)
    assert summary["avg_epoch_time_s"] == 10.0
    assert summary["peak_gpu_mem_mb"] == 600
    assert summary["avg_images_per_sec"] == 105.0  # mean(100, 110), skips nothing
    # avg_gpu_power_w's history is all-None -> _avg must degrade to None, not crash on empty mean
    assert summary["avg_gpu_power_w"] is None
    assert summary["total_gpu_energy_wh"] is None
    assert summary["macs"] == 1_000_000
    assert summary["int8_latency_ms_per_image"] == 2.0


def test_make_run_summary_handles_missing_int8_results():
    summary = make_run_summary(
        name="alexnet_bottleneck", mode="fp32_only", fit_results=_fit_results(),
        fp32_eval={"top1": 45.0, "top5": 70.0, "loss": 2.0},
        params_m=1.5, fp32_size_mb=6.0, int8_size_mb=0.0,
        fp32_benchmark={"latency_ms_per_image": 4.0, "throughput_img_per_s": 250.0},
        flops_results={"macs": 500_000, "flops": 1_000_000},
    )

    assert summary["int8_top1"] is None
    assert summary["quantization_drop_top1"] is None
    assert summary["compression_ratio"] is None  # int8_size_mb=0 must not raise ZeroDivisionError


def test_make_run_summary_avg_epoch_time_ignores_suspend_outlier():
    fit_results = _fit_results()
    # e.g. one epoch's wall-clock timer spans a laptop suspend (~9000s) between two normal ~10s epochs.
    fit_results["history"]["epoch_time_s"] = [10.0, 10.0, 9013.3, 10.0, 10.0]
    summary = make_run_summary(
        name="swin_pico_convstem", mode="fp32_only", fit_results=fit_results,
        fp32_eval={"top1": 45.0, "top5": 70.0, "loss": 2.0},
        params_m=1.5, fp32_size_mb=6.0, int8_size_mb=0.0,
        fp32_benchmark={"latency_ms_per_image": 4.0, "throughput_img_per_s": 250.0},
        flops_results={"macs": 500_000, "flops": 1_000_000},
    )

    assert summary["avg_epoch_time_s"] == 10.0


def test_build_comparison_table_sorts_by_precision_then_descending_top1_when_present():
    rows = [
        {"precision": "int8", "top1_%": 40.0},
        {"precision": "fp32", "top1_%": 50.0},
        {"precision": "fp32", "top1_%": 60.0},
    ]

    df = build_comparison_table(rows)

    assert list(df["precision"]) == ["fp32", "fp32", "int8"]
    assert list(df["top1_%"]) == [60.0, 50.0, 40.0]


def test_build_comparison_table_returns_input_order_when_sort_columns_absent():
    # make_run_summary never emits "precision"/"top1_%" (see fields above), so on its
    # output build_comparison_table has no sort columns and silently skips sorting.
    rows = [{"model_name": "b", "fp32_top1": 1.0}, {"model_name": "a", "fp32_top1": 2.0}]

    df = build_comparison_table(rows)

    assert list(df["model_name"]) == ["b", "a"]


def test_disk_mb_excludes_optimizer_state(tmp_path):
    """disk_mb must measure the weights, not the training checkpoint wrapped around them.

    save_checkpoint() stores AdamW state (2 momentum buffers/param) next to the weights, so
    measuring the file raw reported ~3x the model size on the FP32 side while the INT8 side
    (a bare torch.save(model, ...)) was measured model-only -- inflating every recorded
    compression ratio by ~3x.
    """
    import torch
    import torch.nn as nn
    from ml.checkpoint import save_checkpoint
    from ml.reporting import disk_mb

    model = nn.Linear(512, 512)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.randn(4, 512)).sum().backward()
    opt.step()  # materialize the momentum buffers

    ckpt = tmp_path / "m_best.pth"
    save_checkpoint(ckpt, model, opt, None, 0, {"val_acc": 1.0})
    weights_only = tmp_path / "m.pth"
    torch.save(model.state_dict(), weights_only)

    raw_file_mb = ckpt.stat().st_size / (1024 ** 2)
    assert raw_file_mb > 2.5 * disk_mb(ckpt), "checkpoint should be ~3x its weights"
    assert disk_mb(ckpt) == pytest.approx(disk_mb(weights_only), rel=0.02)
