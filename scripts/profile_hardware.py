"""
Phase 6 Hardware Profiling CLI.

Usage:
    python -m scripts.profile_hardware --experiment phase6 --runtime local [--resume] [--dry-run]
    python -m scripts.profile_hardware --experiment phase6 --runtime pcad [--resume] [--dry-run]

Outputs JSON to: outputs/{runtime}/phase6/{device_tag}_profile.json
Logs to: outputs/{runtime}/phase6/{device_tag}.log
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.profiling import (
    profile_layer_latency_per_batch_resolution,
    profile_layer_latency_int8,
    profile_model_with_efficiency_metrics,
    profile_kernel_trace,
    profile_layer_conv_fft,
    profile_layer_conv_winograd,
)
from ml.quantization import build_qat_from_model, convert_to_int8
from ml.reporting import compute_flops
from ml.registry import MODEL_REGISTRY
from ml.runtime import set_global_seed
import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY for the model sweep


def sanitize_device_name(device_name: str) -> str:
    """Sanitize device name to snake_case for file names."""
    return device_name.lower().replace(" ", "_").replace("-", "_")


def get_device_tag(device: torch.device) -> str:
    """Get device tag with optional hostname to avoid collisions on multi-GPU nodes."""
    try:
        if device.type == "cuda":
            base_tag = sanitize_device_name(torch.cuda.get_device_name(device))
        else:
            base_tag = "cpu"
    except RuntimeError:
        base_tag = "cuda_unavailable"

    # Append hostname if available (prevents collisions when multiple jobs on the
    # same node/GPU model write to the same output file). Works in SLURM, local, etc.
    try:
        import socket
        hostname = socket.gethostname()
        if hostname and hostname != "localhost":
            base_tag = f"{base_tag}_{sanitize_device_name(hostname)}"
    except Exception:
        pass

    return base_tag


def get_device_metadata(device: torch.device) -> dict:
    """Capture environment metadata for reproducibility."""
    metadata = {
        "kind": "metadata",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "seed": 42,
        "timestamp": datetime.now().isoformat(),
        "config_file": "configs/profiling.yaml",
    }

    if device.type == "cuda":
        metadata["device_name"] = torch.cuda.get_device_name(device)
        metadata["device_capability"] = list(torch.cuda.get_device_capability(device))
    else:
        metadata["device_name"] = "CPU"
        metadata["device_capability"] = None

    return metadata


def load_profiling_config(config_path: str | Path) -> dict:
    """Load profiling.yaml config."""
    import yaml
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_completed_configs(output_path: Path) -> set:
    """Load set of completed config keys from output JSON."""
    completed = set()
    if not output_path.exists():
        return completed

    try:
        with open(output_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record["kind"] == "metadata":
                        continue
                    elif record["kind"] == "layer":
                        key = (
                            "layer",
                            record["kernel_size"],
                            record["in_ch"],
                            record["out_ch"],
                            record.get("batch_size", 1),
                            record.get("input_resolution", 64),
                            record["precision"],
                            record.get("groups_mode", "dense"),
                        )
                    elif record["kind"] == "model":
                        key = ("model", record["model"], record["precision"])
                    elif record["kind"] == "fft":
                        key = (
                            "fft",
                            record["kernel_size"],
                            record["in_ch"],
                            record["out_ch"],
                        )
                    elif record["kind"] in ("winograd_custom", "winograd_compiled"):
                        key = (
                            record["kind"],
                            record["kernel_size"],
                            record["in_ch"],
                            record["out_ch"],
                        )
                    else:
                        continue
                    completed.add(key)
                except (json.JSONDecodeError, KeyError):
                    pass
    except FileNotFoundError:
        pass

    return completed


def append_result_atomic(output_path: Path, record: dict) -> None:
    """Atomically append result to JSON output."""
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup dual file + console logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("phase6_profiling")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(fh)

    # Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(sh)

    return logger


def profile_layer_sweep(
    config: dict,
    device: torch.device,
    output_path: Path,
    completed: set,
    logger: logging.Logger,
    dry_run: bool = False,
) -> None:
    """Run layer profiling sweep."""
    kernel_sizes = config.get("kernel_sizes", [2, 3, 5, 7, 9, 11])
    layer_channels = config.get("layer_channels", [16, 32, 64, 128])
    batch_sizes = config.get("batch_sizes", [1])
    input_resolutions = config.get("input_resolutions", [64])
    precisions = config.get("precisions", ["fp32"])
    groups_modes = config.get("groups_modes", ["dense"])
    warmup = config.get("warmup", 50)
    iters = config.get("iters", 200)
    fft_min_kernel_size = config.get("fft_min_kernel_size", 5)

    device_tag = get_device_tag(device)

    # int8 only runs at the base shape (batch=1, res=64), each paired with one fp32_cpu
    # companion row -- so it doesn't scale with the full batch/resolution grid like fp32 does.
    non_int8_precisions = [p for p in precisions if p != "int8"]
    layer_configs = (
        len(kernel_sizes) * len(layer_channels) * len(batch_sizes) * len(input_resolutions)
        * len(non_int8_precisions) * len(groups_modes)
    )
    if "int8" in precisions and 1 in batch_sizes and 64 in input_resolutions:
        layer_configs += 2 * len(kernel_sizes) * len(layer_channels) * len(groups_modes)

    total_configs = (
        layer_configs
        + len([k for k in kernel_sizes if k >= fft_min_kernel_size]) * len(layer_channels)
        + len(layer_channels)  # custom Winograd sweep (kernel_size=3 only)
        + len(layer_channels)  # torch.compile'd Winograd sweep (kernel_size=3 only)
    )

    i = 0
    logger.info(f"Starting layer profiling sweep ({total_configs} configs)")

    for kernel_size in kernel_sizes:
        for in_ch in layer_channels:
            for batch_size in batch_sizes:
                for input_resolution in input_resolutions:
                    for precision in precisions:
                        for groups_mode in groups_modes:
                            i += 1

                            groups = in_ch if groups_mode == "depthwise" else 1

                            key = (
                                "layer",
                                kernel_size,
                                in_ch,
                                in_ch,
                                batch_size,
                                input_resolution,
                                precision,
                                groups_mode,
                            )

                            if key in completed:
                                logger.info(f"[{i}/{total_configs}] Skipping layer config {key}")
                                continue

                            # Eager-mode INT8 has no CUDA kernel (profile_layer_latency_int8 is
                            # CPU-only, like the model sweep's INT8 path) -- only run it at the
                            # base shape rather than mislabeling an FP32 GPU timing as INT8.
                            if precision == "int8" and (batch_size, input_resolution) != (1, 64):
                                logger.info(f"[{i}/{total_configs}] Skipping int8 layer config {key} "
                                           f"(int8 layer sweep only runs at batch=1,res=64)")
                                continue

                            try:
                                input_shape = (batch_size, in_ch, input_resolution, input_resolution)

                                if precision == "int8":
                                    latency_ms = profile_layer_latency_int8(
                                        kernel_size, in_ch, in_ch, input_shape,
                                        warmup=warmup, iters=iters, groups=groups,
                                    )
                                    cudnn_kernel_name = None  # CPU INT8 has no cuDNN kernel
                                else:
                                    latency_ms = profile_layer_latency_per_batch_resolution(
                                        kernel_size,
                                        in_ch,
                                        in_ch,
                                        batch_size,
                                        input_resolution,
                                        device,
                                        warmup=warmup,
                                        iters=iters,
                                        groups=groups,
                                    )
                                    # Ground truth for which cuDNN algorithm actually ran (implicit
                                    # GEMM / FFT / Winograd), replacing the old 3x3-vs-5x5
                                    # speedup-ratio inference -- only meaningful on the GPU cuDNN
                                    # dispatches on, not the CPU fp32_cpu companion row below.
                                    cudnn_kernel_name = None
                                    if device.type == "cuda":
                                        trace_conv = nn.Conv2d(
                                            in_ch, in_ch, kernel_size, stride=1,
                                            padding=(kernel_size - 1) // 2, bias=False, groups=groups,
                                        )
                                        trace_input = torch.randn(input_shape, device=device)
                                        trace_result = profile_kernel_trace(trace_conv, trace_input, device)
                                        cudnn_kernel_name = trace_result["top_cuda_kernel"]

                                result = {
                                    "kind": "layer",
                                    "kernel_size": kernel_size,
                                    "in_ch": in_ch,
                                    "out_ch": in_ch,
                                    "batch_size": batch_size,
                                    "input_resolution": input_resolution,
                                    "precision": precision,
                                    "groups": groups,
                                    "groups_mode": groups_mode,
                                    "latency_ms": latency_ms,
                                    "cudnn_kernel_name": cudnn_kernel_name,
                                    "device": device_tag,
                                }

                                if not dry_run:
                                    append_result_atomic(output_path, result)

                                logger.info(f"[{i}/{total_configs}] layer k={kernel_size} ch={in_ch} "
                                           f"b={batch_size} res={input_resolution} {precision} "
                                           f"groups={groups_mode}: {latency_ms:.3f}ms")

                                # H4 needs a same-hardware baseline: pair each real INT8 conv with
                                # an FP32 conv timed on the same CPU (not the FP32 GPU row), so
                                # int8-vs-fp32 latency ranking isn't confounded with GPU-vs-CPU.
                                if precision == "int8":
                                    fp32_cpu_key = ("layer", kernel_size, in_ch, in_ch, batch_size,
                                                    input_resolution, "fp32_cpu", groups_mode)
                                    if fp32_cpu_key not in completed:
                                        fp32_cpu_latency = profile_layer_latency_per_batch_resolution(
                                            kernel_size, in_ch, in_ch, batch_size, input_resolution,
                                            torch.device("cpu"), warmup=warmup, iters=iters, groups=groups,
                                        )
                                        fp32_cpu_result = {**result, "precision": "fp32_cpu",
                                                            "latency_ms": fp32_cpu_latency,
                                                            "cudnn_kernel_name": None}
                                        if not dry_run:
                                            append_result_atomic(output_path, fp32_cpu_result)
                                        logger.info(f"[{i}/{total_configs}] layer k={kernel_size} ch={in_ch} "
                                                   f"b={batch_size} res={input_resolution} fp32_cpu "
                                                   f"groups={groups_mode}: {fp32_cpu_latency:.3f}ms")

                            except Exception as e:
                                logger.error(f"[{i}/{total_configs}] Error profiling layer config {key}: {e}")

    # FFT convolution sweep (FP32 only, kernel_size >= fft_min_kernel_size)
    logger.info("Starting FFT convolution sweep")
    for kernel_size in [k for k in kernel_sizes if k >= fft_min_kernel_size]:
        for in_ch in layer_channels:
            i += 1

            key = ("fft", kernel_size, in_ch, in_ch)

            if key in completed:
                logger.info(f"[{i}/{total_configs}] Skipping FFT config {key}")
                continue

            try:
                input_shape = (1, in_ch, 64, 64)
                fft_result = profile_layer_conv_fft(
                    kernel_size, in_ch, in_ch, input_shape, device, warmup=warmup, iters=iters
                )

                result = {
                    "kind": "fft",
                    "kernel_size": kernel_size,
                    "in_ch": in_ch,
                    "out_ch": in_ch,
                    "latency_ms": fft_result.get("latency_ms"),
                    "note": fft_result.get("note"),
                    "device": device_tag,
                }

                if not dry_run:
                    append_result_atomic(output_path, result)

                logger.info(f"[{i}/{total_configs}] FFT k={kernel_size} ch={in_ch}: {fft_result['latency_ms'] or 'N/A'}")

            except Exception as e:
                logger.error(f"[{i}/{total_configs}] Error profiling FFT config {key}: {e}")

    # Custom Winograd F(2x2,3x3) sweep (real transform, kernel_size=3 only, dense groups=1) --
    # gives a direct kind:layer (cuDNN's own algorithm choice) vs. kind:winograd_custom
    # (explicit reduced-multiply algorithm) comparison at the same channel widths.
    logger.info("Starting custom Winograd F(2x2,3x3) sweep")
    for in_ch in layer_channels:
        i += 1

        key = ("winograd_custom", 3, in_ch, in_ch)

        if key in completed:
            logger.info(f"[{i}/{total_configs}] Skipping Winograd config {key}")
            continue

        try:
            input_shape = (1, in_ch, 64, 64)
            winograd_result = profile_layer_conv_winograd(
                3, in_ch, in_ch, input_shape, device, warmup=warmup, iters=iters
            )

            result = {
                "kind": "winograd_custom",
                "kernel_size": 3,
                "in_ch": in_ch,
                "out_ch": in_ch,
                "latency_ms": winograd_result.get("latency_ms"),
                "note": winograd_result.get("note"),
                "device": device_tag,
            }

            if not dry_run:
                append_result_atomic(output_path, result)

            logger.info(f"[{i}/{total_configs}] Winograd k=3 ch={in_ch}: {winograd_result['latency_ms'] or 'N/A'}")

        except Exception as e:
            logger.error(f"[{i}/{total_configs}] Error profiling Winograd config {key}: {e}")

    # torch.compile'd Winograd F(2x2,3x3) sweep -- same shapes as winograd_custom above, isolates
    # how much of the eager-vs-cuDNN gap (see notebook cell 12) is Python/dispatch overhead.
    logger.info("Starting torch.compile'd Winograd F(2x2,3x3) sweep")
    for in_ch in layer_channels:
        i += 1

        key = ("winograd_compiled", 3, in_ch, in_ch)

        if key in completed:
            logger.info(f"[{i}/{total_configs}] Skipping compiled Winograd config {key}")
            continue

        try:
            input_shape = (1, in_ch, 64, 64)
            winograd_compiled_result = profile_layer_conv_winograd(
                3, in_ch, in_ch, input_shape, device, warmup=warmup, iters=iters, compile=True
            )

            result = {
                "kind": "winograd_compiled",
                "kernel_size": 3,
                "in_ch": in_ch,
                "out_ch": in_ch,
                "latency_ms": winograd_compiled_result.get("latency_ms"),
                "note": winograd_compiled_result.get("note"),
                "device": device_tag,
            }

            if not dry_run:
                append_result_atomic(output_path, result)

            logger.info(f"[{i}/{total_configs}] Compiled Winograd k=3 ch={in_ch}: "
                       f"{winograd_compiled_result['latency_ms'] or 'N/A'}")

        except Exception as e:
            logger.error(f"[{i}/{total_configs}] Error profiling compiled Winograd config {key}: {e}")


def profile_model_sweep(
    config: dict,
    device: torch.device,
    output_path: Path,
    completed: set,
    logger: logging.Logger,
    dry_run: bool = False,
) -> None:
    """Run model profiling sweep."""
    models = config.get("models", [])
    precisions = config.get("precisions", ["fp32"])
    batch_size = config.get("batch_sizes", [1])[0]  # Use smallest batch size for model profiling
    input_resolution = config.get("input_resolutions", [64])[0]

    device_tag = get_device_tag(device)

    total_configs = len(models) * len(precisions)
    i = 0

    logger.info(f"Starting model profiling sweep ({total_configs} configs)")

    for model_name in models:
        for precision in precisions:
            i += 1

            key = ("model", model_name, precision)

            if key in completed:
                logger.info(f"[{i}/{total_configs}] Skipping model {model_name} {precision}")
                continue

            try:
                # Build model from registry (no checkpoint)
                spec = MODEL_REGISTRY.get(model_name)
                if spec is None:
                    logger.error(f"Model {model_name} not in MODEL_REGISTRY")
                    continue

                model = spec["ctor"]()
                input_size = (batch_size, 3, input_resolution, input_resolution)
                # FLOPs depend on the op graph, not precision — compute from the FP32
                # model before any INT8 conversion (fvcore's tracer can't trace a
                # quantized graph, so this must happen here, not inside the profiler).
                total_flops = compute_flops(model, input_size=input_size).get("flops", 0)

                # Convert to INT8 if needed
                if precision == "int8":
                    # get_default_qat_qconfig("fbgemm")'s activation observer bakes in
                    # reduce_range=True, which warns (at observer construction, i.e.
                    # inside prepare_qat) that it's deprecated in favor of explicit
                    # quant_min/quant_max -- an upstream default in
                    # ml/quantization.py's prepare_qat_model (shared with the real QAT
                    # training pipeline), not something to change just for this sweep.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message=".*reduce_range will be deprecated.*", category=UserWarning,
                        )
                        model_qat = build_qat_from_model(model, model_name, device)
                    # Calibration pass for the QAT observers -- without this, convert()
                    # quantizes every activation with default (never-observed) qparams
                    # and PyTorch warns "must run observer before calling
                    # calculate_qparams" on every layer. Random input matches this
                    # sweep's scope: profiling runs freshly constructed (untrained)
                    # models (see configs/profiling.yaml), so there's no real
                    # checkpoint's activation distribution to calibrate against anyway
                    # -- same synthetic-calibration caveat as profile_layer_latency_int8.
                    calib_input = torch.randn(input_size, device=device)
                    with torch.no_grad():
                        for _ in range(10):
                            model_qat(calib_input)
                    model = convert_to_int8(model_qat)
                    model = model.to("cpu")  # INT8 inference on CPU
                    profile_device = torch.device("cpu")
                else:
                    model = model.to(device)
                    profile_device = device

                # Profile with efficiency metrics
                metrics = profile_model_with_efficiency_metrics(
                    model, input_size, profile_device, total_flops, warmup=50, iters=200
                )

                # Kernel trace for Winograd detection (FP32 only)
                winograd_trace_detected = False
                if precision == "fp32":
                    input_tensor = torch.randn(input_size, device=profile_device)
                    trace_result = profile_kernel_trace(model, input_tensor, profile_device)
                    winograd_trace_detected = trace_result["winograd_trace_detected"]

                result = {
                    "kind": "model",
                    "model": model_name,
                    "precision": precision,
                    "latency_ms": metrics["latency_ms"],
                    "throughput_img_s": metrics["throughput_img_s"],
                    "power_draw_avg_w": metrics["power_draw_avg_w"],
                    "power_draw_std_w": metrics["power_draw_std_w"],
                    "gpu_utilization_pct": metrics["gpu_utilization_pct"],
                    "gpu_memory_peak_mb": metrics["gpu_memory_peak_mb"],
                    "compute_efficiency_gflops_s": metrics["compute_efficiency_gflops_s"],
                    "winograd_trace_detected": winograd_trace_detected if precision == "fp32" else None,
                    "device": device_tag,
                }

                if not dry_run:
                    append_result_atomic(output_path, result)

                logger.info(
                    f"[{i}/{total_configs}] {model_name} {precision}: "
                    f"{metrics['latency_ms']:.2f}ms, "
                    f"{metrics['throughput_img_s']:.1f} img/s"
                )

            except Exception as e:
                logger.error(f"[{i}/{total_configs}] Error profiling model {model_name} {precision}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase 6 Hardware Profiling")
    parser.add_argument("--experiment", default="phase6", help="Experiment name")
    parser.add_argument("--runtime", choices=["local", "pcad"], required=True, help="Runtime profile")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed config")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (don't write output)")
    parser.add_argument("--model-split", type=str, help="Split model sweep: 'N:M' runs part N of M (e.g. '1:2' runs first half)")
    parser.add_argument("--skip-layer-sweep", action="store_true", help="Skip layer/FFT sweep, run model sweep only")
    args = parser.parse_args()

    # Set up paths
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger_name = f"phase6_{args.runtime}"

    output_dir = Path("outputs") / args.runtime / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    device_tag = get_device_tag(device)

    output_path = output_dir / f"{device_tag}_profile.json"
    log_path = output_dir / f"{device_tag}.log"

    logger = setup_logging(log_path)
    logger.info(f"Starting Phase 6 profiling run on {device}")
    logger.info(f"Output: {output_path}")

    # Load config
    config_path = Path("configs/profiling.yaml")
    config = load_profiling_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Apply model split if specified
    if args.model_split:
        part, total = map(int, args.model_split.split(":"))
        models = config.get("models", [])
        start = (part - 1) * len(models) // total
        end = part * len(models) // total
        config["models"] = models[start:end]
        logger.info(f"Model split {part}:{total} → running models {start}-{end-1}: {config['models']}")

    # Load completed configs if resuming
    completed = load_completed_configs(output_path) if args.resume else set()
    logger.info(f"Resuming: {len(completed)} configs already completed")

    # Write metadata (if not resuming or output is empty)
    if not output_path.exists():
        metadata = get_device_metadata(device)
        if not args.dry_run:
            append_result_atomic(output_path, metadata)
        logger.info(f"Metadata: {metadata}")

    # Run profiling sweeps
    try:
        if not args.skip_layer_sweep:
            profile_layer_sweep(config, device, output_path, completed, logger, args.dry_run)
        profile_model_sweep(config, device, output_path, completed, logger, args.dry_run)
        logger.info("Profiling complete")
    except KeyboardInterrupt:
        logger.warning("Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
