import time
import json
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.profiler

from .reporting import compute_flops


def profile_layer_latency(
    kernel_size: int,
    in_ch: int,
    out_ch: int,
    input_shape: tuple,
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
) -> float:
    """
    Profile latency of a bare Conv2d layer.

    Args:
        kernel_size: Conv kernel size.
        in_ch, out_ch: Input and output channels.
        input_shape: (batch, channels, height, width) tuple.
        device: torch.device for profiling.
        warmup: Warmup iterations (not timed).
        iters: Timed iterations.

    Returns:
        Latency in milliseconds (per iteration).
    """
    conv = nn.Conv2d(
        in_ch, out_ch, kernel_size, stride=1,
        padding=(kernel_size - 1) // 2, bias=False
    )
    conv = conv.to(device).eval()
    torch.set_grad_enabled(False)

    input_tensor = torch.randn(input_shape, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            _ = conv(input_tensor)

    torch.cuda.synchronize(device)
    start_time = time.time()

    for _ in range(iters):
        with torch.no_grad():
            _ = conv(input_tensor)

    torch.cuda.synchronize(device)
    elapsed_ms = (time.time() - start_time) * 1000 / iters

    return elapsed_ms


def profile_layer_latency_per_batch_resolution(
    kernel_size: int,
    in_ch: int,
    out_ch: int,
    batch_size: int,
    input_resolution: int,
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
) -> float:
    """
    Profile latency with specified batch size and input resolution.

    Args:
        batch_size: Batch size for inference.
        input_resolution: Spatial size (assumes square input_resolution x input_resolution).

    Returns:
        Latency in milliseconds per iteration.
    """
    input_shape = (batch_size, in_ch, input_resolution, input_resolution)
    return profile_layer_latency(
        kernel_size, in_ch, out_ch, input_shape, device, warmup, iters
    )


def detect_winograd_via_speedup(
    in_ch: int,
    out_ch: int,
    input_shape: tuple,
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
    speedup_threshold: float = 1.8,
) -> dict:
    """
    Empirical Winograd detection: compare 3x3 vs. 5x5 latency.

    Winograd only accelerates 3x3, not 5x5. If 3x3 is significantly faster
    than 5x5, Winograd is likely active.

    Returns:
        {
            "speedup_ratio": float (5x5_latency / 3x3_latency),
            "winograd_inferred": bool (ratio > threshold),
        }
    """
    latency_3x3 = profile_layer_latency(3, in_ch, out_ch, input_shape, device, warmup, iters)
    latency_5x5 = profile_layer_latency(5, in_ch, out_ch, input_shape, device, warmup, iters)

    ratio = latency_5x5 / latency_3x3 if latency_3x3 > 0 else 0
    winograd_inferred = ratio > speedup_threshold

    return {
        "speedup_ratio": ratio,
        "winograd_inferred": winograd_inferred,
    }


def profile_model_latency(
    model: nn.Module,
    input_size: tuple,
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
) -> float:
    """
    Profile full-model latency (no efficiency metrics).

    Args:
        model: PyTorch model.
        input_size: (batch, channels, height, width) tuple.
        device: torch.device.
        warmup: Warmup iterations.
        iters: Timed iterations.

    Returns:
        Latency in milliseconds per iteration.
    """
    model = model.to(device).eval()
    torch.set_grad_enabled(False)

    input_tensor = torch.randn(input_size, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)

    torch.cuda.synchronize(device)
    start_time = time.time()

    for _ in range(iters):
        with torch.no_grad():
            _ = model(input_tensor)

    torch.cuda.synchronize(device)
    elapsed_ms = (time.time() - start_time) * 1000 / iters

    return elapsed_ms


def profile_model_with_efficiency_metrics(
    model: nn.Module,
    input_size: tuple,
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    """
    Profile model latency + power + utilization + memory + compute efficiency.

    Args:
        model: PyTorch model.
        input_size: (batch, channels, height, width) tuple.
        device: torch.device.
        warmup: Warmup iterations.
        iters: Timed iterations.

    Returns:
        {
            "latency_ms": float,
            "throughput_img_s": float,
            "power_draw_avg_w": float or None,
            "power_draw_std_w": float or None,
            "gpu_utilization_pct": float or None,
            "gpu_memory_peak_mb": float or None,
            "compute_efficiency_gflops_s": float or None,
        }
    """
    model = model.to(device).eval()
    torch.set_grad_enabled(False)

    # Compute FLOPs for efficiency calculation (one-time, no per-iteration cost)
    flops_dict = compute_flops(model, input_size=input_size)
    total_flops = flops_dict.get("flops", 0)

    # Background power and utilization sampling
    power_samples = []
    gpu_util_samples = []
    stop_sampling = threading.Event()

    def sample_gpu_metrics():
        try:
            proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
                 "--loop-ms=100", "-l", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in proc.stdout:
                if stop_sampling.is_set():
                    proc.terminate()
                    break
                try:
                    parts = line.split()
                    power_w = float(parts[0])
                    gpu_util_pct = float(parts[1])
                    power_samples.append(power_w)
                    gpu_util_samples.append(gpu_util_pct)
                except (ValueError, IndexError):
                    pass
        except FileNotFoundError:
            pass  # nvidia-smi not available; continue without metrics.

    sampler = threading.Thread(target=sample_gpu_metrics, daemon=True)
    sampler.start()

    # Memory measurement
    torch.cuda.reset_peak_memory_stats(device)

    # Warmup + timed forward passes
    input_tensor = torch.randn(input_size, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)

    torch.cuda.synchronize(device)
    start_time = time.time()

    for _ in range(iters):
        with torch.no_grad():
            _ = model(input_tensor)

    torch.cuda.synchronize(device)
    elapsed_ms = (time.time() - start_time) * 1000 / iters

    stop_sampling.set()
    sampler.join(timeout=2.0)

    # Aggregate metrics
    latency_ms = elapsed_ms
    batch_size = input_size[0]
    throughput_img_s = (batch_size * 1000.0) / latency_ms
    power_avg = np.mean(power_samples) if power_samples else None
    power_std = np.std(power_samples) if power_samples else None
    gpu_util_avg = np.mean(gpu_util_samples) if gpu_util_samples else None
    memory_peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    # Compute efficiency: actual GFLOPs / second = (total_flops / latency_ms) / 1e9
    compute_efficiency = (total_flops / latency_ms) / 1e9 if latency_ms > 0 else None

    return {
        "latency_ms": latency_ms,
        "throughput_img_s": throughput_img_s,
        "power_draw_avg_w": power_avg,
        "power_draw_std_w": power_std,
        "gpu_utilization_pct": gpu_util_avg,
        "gpu_memory_peak_mb": memory_peak_mb,
        "compute_efficiency_gflops_s": compute_efficiency,
    }


def profile_kernel_trace(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Profile model using torch.profiler and extract per-op durations.

    Flags whether any op name contains 'winograd'. Best-effort only:
    cuDNN's Winograd kernel naming isn't stable across versions.

    Args:
        model: PyTorch model.
        input_tensor: Single input tensor for profiling.
        device: torch.device.

    Returns:
        {
            "winograd_trace_detected": bool,
            "op_timings": dict (op_name -> duration_us),
        }
    """
    model = model.to(device).eval()
    torch.set_grad_enabled(False)

    op_timings = {}
    winograd_detected = False

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            _ = model(input_tensor.to(device))

    for evt in prof.key_averages():
        if evt.device_type == "cuda":
            op_name = evt.key
            duration_us = evt.cuda_time_total
            op_timings[op_name] = duration_us

            if "winograd" in op_name.lower():
                winograd_detected = True

    return {
        "winograd_trace_detected": winograd_detected,
        "op_timings": op_timings,
    }


def profile_layer_conv_fft(
    kernel_size: int,
    in_ch: int,
    out_ch: int,
    input_shape: tuple,
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    """
    Profile FFT-based convolution (FP32 only, no stride support).

    FFT-based conv: zero-pad input and kernel to (input_size + kernel_size - 1),
    apply rfft2, multiply, sum over input channels, irfft2, crop.

    Only meaningful for kernel_size >= 5 (FFT overhead is a loss for 2x2/3x3).

    Args:
        kernel_size: Conv kernel size (should be >= 5 for FFT to be worthwhile).
        in_ch, out_ch: Input and output channels.
        input_shape: (batch, channels, height, width).
        device: torch.device.
        warmup, iters: Profiling parameters.

    Returns:
        {
            "latency_ms": float (FFT-based conv latency),
            "note": str (warning if kernel_size < 5 or if stride > 1),
        }
    """
    batch, _, height, width = input_shape

    if kernel_size < 5:
        return {
            "latency_ms": None,
            "note": f"Skipped: kernel_size={kernel_size} < 5; FFT overhead too high",
        }

    # Pad to linear convolution size
    pad_h = height + kernel_size - 1
    pad_w = width + kernel_size - 1

    input_tensor = torch.randn(input_shape, device=device, dtype=torch.float32)
    kernels = torch.randn(out_ch, in_ch, kernel_size, kernel_size, device=device, dtype=torch.float32)

    torch.cuda.synchronize(device) if device.type == "cuda" else None

    for _ in range(warmup):
        # Pad input and kernels
        input_padded = torch.nn.functional.pad(
            input_tensor,
            (0, pad_w - width, 0, pad_h - height),
            mode="constant",
            value=0,
        )
        kernel_padded = torch.nn.functional.pad(
            kernels,
            (0, pad_w - kernel_size, 0, pad_h - kernel_size),
            mode="constant",
            value=0,
        )

        # FFT-based convolution
        input_fft = torch.fft.rfft2(input_padded, dim=(-2, -1))
        kernel_fft = torch.fft.rfft2(kernel_padded, dim=(-2, -1))

        # Multiply and sum over input channels
        # input_fft: (batch, in_ch, pad_h, pad_w//2+1)
        # kernel_fft: (out_ch, in_ch, pad_h, pad_w//2+1)
        output_fft = torch.einsum("bchw,ochw->bohw", input_fft, kernel_fft)

        # IRFFT
        output = torch.fft.irfft2(output_fft, s=(pad_h, pad_w), dim=(-2, -1))

        # Crop to valid output size
        output = output[..., :height, :width]

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start_time = time.time()

    for _ in range(iters):
        input_padded = torch.nn.functional.pad(
            input_tensor,
            (0, pad_w - width, 0, pad_h - height),
            mode="constant",
            value=0,
        )
        kernel_padded = torch.nn.functional.pad(
            kernels,
            (0, pad_w - kernel_size, 0, pad_h - kernel_size),
            mode="constant",
            value=0,
        )

        input_fft = torch.fft.rfft2(input_padded, dim=(-2, -1))
        kernel_fft = torch.fft.rfft2(kernel_padded, dim=(-2, -1))
        output_fft = torch.einsum("bchw,ochw->bohw", input_fft, kernel_fft)
        output = torch.fft.irfft2(output_fft, s=(pad_h, pad_w), dim=(-2, -1))
        output = output[..., :height, :width]

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    elapsed_ms = (time.time() - start_time) * 1000 / iters

    return {
        "latency_ms": elapsed_ms,
        "note": "FFT-based convolution; stride=1 only",
    }
