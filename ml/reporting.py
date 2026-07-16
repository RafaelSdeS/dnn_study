import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


def disk_mb(path: str | Path) -> float | None:
    """File size in MB; None if file doesn't exist."""
    p = Path(path)
    return p.stat().st_size / (1024 ** 2) if p.exists() else None


def compute_flops(model, input_size: tuple = (1, 3, 64, 64)) -> dict:
    """MACs and FLOPs via fvcore. Returns {"macs": int, "flops": int}."""
    from fvcore.nn import FlopCountAnalysis
    import torch
    inp = torch.zeros(input_size)
    # move to same device as model params if possible
    try:
        device = next(model.parameters()).device
        inp = inp.to(device)
    except StopIteration:
        pass
    analysis = FlopCountAnalysis(model, inp)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    macs = analysis.total()
    return {"macs": macs, "flops": macs * 2}


def _avg(values: list) -> float | None:
    """Mean of the non-None entries in values, or None if there are none."""
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _total(values: list) -> float | None:
    """Sum of the non-None entries in values, or None if there are none."""
    clean = [v for v in values if v is not None]
    return sum(clean) if clean else None


def make_run_summary(
    name: str,
    mode: str,
    fit_results: dict,
    fp32_eval: dict,
    params_m: float,
    fp32_size_mb: float,
    int8_size_mb: float,
    fp32_benchmark: dict,
    flops_results: dict,
    int8_eval: dict | None = None,
    int8_benchmark: dict | None = None,
) -> dict:
    """Assemble the full per-model run summary with FP32 and INT8 metrics."""
    history = fit_results.get("history", {})
    final_train_loss = history.get("train_loss", [None])[-1]
    epoch_times = history.get("epoch_time_s", [])
    avg_epoch_time_s = sum(epoch_times) / len(epoch_times) if epoch_times else None
    peak_gpu_mem_mb = max(history.get("peak_gpu_mem_mb", [0]) or [0])
    avg_images_per_sec = _avg(history.get("images_per_sec", []))
    avg_batch_time_s = _avg(history.get("avg_batch_time_s", []))
    avg_cpu_percent = _avg(history.get("cpu_percent", []))
    avg_ram_used_mb = _avg(history.get("ram_used_mb", []))
    avg_gpu_power_w = _avg(history.get("gpu_power_avg_w", []))
    avg_gpu_utilization_pct = _avg(history.get("gpu_utilization_pct", []))
    avg_gpu_temp_c = _avg(history.get("gpu_temp_avg_c", []))
    total_gpu_energy_wh = _total(history.get("gpu_energy_wh", []))

    compression_ratio = (
        fp32_size_mb / int8_size_mb
        if int8_size_mb and int8_size_mb > 0
        else None
    )
    best_top1 = fit_results.get("best_val_top1", fp32_eval.get("top1"))
    param_efficiency = best_top1 / params_m if (best_top1 and params_m) else None
    top1_top5_gap = (
        fp32_eval.get("top5", 0) - fp32_eval.get("top1", 0)
        if "top1" in fp32_eval else None
    )
    int8_top1_top5_gap = (
        int8_eval.get("top5", 0) - int8_eval.get("top1", 0)
        if int8_eval and "top1" in int8_eval else None
    )

    return {
        "model_name": name,
        "mode": mode,
        "epochs": fit_results.get("best_epoch"),
        # FP32 training metrics
        "best_val_top1": fit_results.get("best_val_top1"),
        "best_val_top5": fit_results.get("best_val_top5"),
        "final_val_top1": fit_results.get("final_val_top1"),
        "final_val_top5": fit_results.get("final_val_top5"),
        "best_val_loss": fit_results.get("best_val_loss"),
        "final_train_loss": final_train_loss,
        # FP32 standalone eval
        "fp32_top1": fp32_eval.get("top1"),
        "fp32_top5": fp32_eval.get("top5"),
        "fp32_loss": fp32_eval.get("loss"),
        "fp32_top1_top5_gap": top1_top5_gap,
        # INT8 eval
        "int8_top1": int8_eval.get("top1") if int8_eval else None,
        "int8_top5": int8_eval.get("top5") if int8_eval else None,
        "int8_loss": int8_eval.get("loss") if int8_eval else None,
        "int8_top1_top5_gap": int8_top1_top5_gap,
        "quantization_drop_top1": (
            fp32_eval.get("top1", 0) - int8_eval.get("top1", 0)
            if int8_eval and "top1" in int8_eval else None
        ),
        # Model sizes & efficiency
        "fp32_size_mb": fp32_size_mb,
        "int8_size_mb": int8_size_mb,
        "params_m": params_m,
        "compression_ratio": compression_ratio,
        "param_efficiency_top1_per_m": param_efficiency,
        # FLOPs
        "macs": flops_results.get("macs"),
        "flops": flops_results.get("flops"),
        # Inference benchmarks
        "fp32_latency_ms_per_image": fp32_benchmark.get("latency_ms_per_image"),
        "fp32_throughput_img_per_s": fp32_benchmark.get("throughput_img_per_s"),
        "int8_latency_ms_per_image": int8_benchmark.get("latency_ms_per_image") if int8_benchmark else None,
        "int8_throughput_img_per_s": int8_benchmark.get("throughput_img_per_s") if int8_benchmark else None,
        # Training efficiency
        "avg_epoch_time_s": avg_epoch_time_s,
        "total_training_time_s": fit_results.get("total_training_time_s"),
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "avg_images_per_sec": avg_images_per_sec,
        "avg_batch_time_s": avg_batch_time_s,
        # Hardware utilization (averaged/summed across training epochs)
        "avg_cpu_percent": avg_cpu_percent,
        "avg_ram_used_mb": avg_ram_used_mb,
        "avg_gpu_power_w": avg_gpu_power_w,
        "avg_gpu_utilization_pct": avg_gpu_utilization_pct,
        "avg_gpu_temp_c": avg_gpu_temp_c,
        "total_gpu_energy_wh": total_gpu_energy_wh,
    }


def build_comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Build a sorted comparison DataFrame (by precision then descending top-1)."""
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["precision", "top1_%"] if c in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    ascending = [True] * len(sort_cols)
    if len(sort_cols) > 1 and "top1_%" in sort_cols:
        ascending[-1] = False
    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def create_results_summary(
    results: dict[str, Any],
    config,
    output_path: str | Path,
) -> None:
    """Save experiment results + config to JSON."""
    cfg_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"config": cfg_dict, **results}, f, indent=2, default=str)
