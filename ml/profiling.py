import time
import json
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.profiler
from torch.autograd import DeviceType


class GpuSampler:
    """Background nvidia-smi sampler for power/utilization/temperature/memory.

    Use as a context manager around any block of work; call summary() after exit.
    Degrades to all-None fields when nvidia-smi is unavailable (e.g. CPU-only boxes) —
    same graceful-degradation behavior this sampling logic has always had.
    """

    _QUERY = "power.draw,utilization.gpu,temperature.gpu,memory.used"

    def __init__(self, interval_ms: int = 200):
        self._interval_ms = interval_ms
        self._samples: list[tuple[float, float, float, float]] = []
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._elapsed_s = 0.0

    def __enter__(self) -> "GpuSampler":
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._elapsed_s = time.time() - self._t0

    def _run(self) -> None:
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self._QUERY}",
                 "--format=csv,noheader,nounits", f"--loop-ms={self._interval_ms}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                try:
                    power, util, temp, mem = (float(x) for x in line.split(","))
                    self._samples.append((power, util, temp, mem))
                except (ValueError, IndexError):
                    pass
        except FileNotFoundError:
            pass  # nvidia-smi not available; continue without metrics.

    def summary(self) -> dict:
        """Return averaged power/util/temp/memory and total energy (Wh) over the sampled window."""
        if not self._samples:
            return {
                "gpu_power_avg_w": None, "gpu_power_std_w": None,
                "gpu_utilization_pct": None, "gpu_temp_avg_c": None,
                "gpu_memory_used_avg_mb": None, "gpu_energy_wh": None,
            }
        arr = np.array(self._samples)
        power, util, temp, mem = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        energy_wh = float(power.mean()) * (self._elapsed_s / 3600.0) if self._elapsed_s else None
        return {
            "gpu_power_avg_w": float(power.mean()),
            "gpu_power_std_w": float(power.std()),
            "gpu_utilization_pct": float(util.mean()),
            "gpu_temp_avg_c": float(temp.mean()),
            "gpu_memory_used_avg_mb": float(mem.mean()),
            "gpu_energy_wh": energy_wh,
        }


def profile_layer_latency(
    kernel_size: int,
    in_ch: int,
    out_ch: int,
    input_shape: tuple,
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
    groups: int = 1,
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
        groups: Conv2d groups (1 = dense, in_ch = depthwise).

    Returns:
        Latency in milliseconds (per iteration).
    """
    conv = nn.Conv2d(
        in_ch, out_ch, kernel_size, stride=1,
        padding=(kernel_size - 1) // 2, bias=False, groups=groups
    )
    conv = conv.to(device).eval()

    input_tensor = torch.randn(input_shape, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            _ = conv(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.time()

    for _ in range(iters):
        with torch.no_grad():
            _ = conv(input_tensor)

    if device.type == "cuda":
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
    groups: int = 1,
) -> float:
    """
    Profile latency with specified batch size and input resolution.

    Args:
        batch_size: Batch size for inference.
        input_resolution: Spatial size (assumes square input_resolution x input_resolution).
        groups: Conv2d groups (1 = dense, in_ch = depthwise).

    Returns:
        Latency in milliseconds per iteration.
    """
    input_shape = (batch_size, in_ch, input_resolution, input_resolution)
    return profile_layer_latency(
        kernel_size, in_ch, out_ch, input_shape, device, warmup, iters, groups=groups
    )


def profile_layer_latency_int8(
    kernel_size: int,
    in_ch: int,
    out_ch: int,
    input_shape: tuple,
    warmup: int = 50,
    iters: int = 200,
    groups: int = 1,
) -> float:
    """
    Profile latency of a real statically-quantized Conv2d.

    CPU-only: eager-mode PyTorch INT8 has no CUDA kernel (same reason
    profile_model_sweep converts and profiles INT8 models on CPU), so there is
    no GPU equivalent of this measurement.

    Args:
        kernel_size, in_ch, out_ch, groups: as profile_layer_latency.
        input_shape: (batch, channels, height, width) tuple.
        warmup: Warmup iterations (not timed).
        iters: Timed iterations.

    Returns:
        Latency in milliseconds (per iteration).
    """
    conv = nn.Conv2d(
        in_ch, out_ch, kernel_size, stride=1,
        padding=(kernel_size - 1) // 2, bias=False, groups=groups
    )
    model = nn.Sequential(tq.QuantStub(), conv, tq.DeQuantStub()).eval()
    model.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(model, inplace=True)

    input_tensor = torch.randn(input_shape)  # CPU tensor -- quantized conv is CPU-only

    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)  # calibration pass for the activation observers

    tq.convert(model, inplace=True)

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)

    start_time = time.time()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(input_tensor)
    elapsed_ms = (time.time() - start_time) * 1000 / iters

    return elapsed_ms


def conv_multiplies_per_flop(kernel_size: int, algorithm: str, input_resolution: int = 64) -> float:
    """
    Analytic multiplies-per-output-pixel-per-channel-pair for direct / Winograd / FFT conv.

    Hardware-independent op count -- the actual "does the algorithm save work" claim, as
    opposed to inferring it from measured latency (which also carries memory-traffic,
    launch-overhead, and cuDNN-engine-selection noise; see profile_kernel_trace for the
    ground-truth version of that).

    Args:
        kernel_size: Conv kernel size.
        algorithm: "direct", "winograd_f2" (F(2x2,3x3), kernel_size must be 3),
            "winograd_f4" (F(4x4,3x3), kernel_size must be 3), or "fft".
        input_resolution: Spatial size (square), only used by "fft" to size the padded
            transform.

    Returns:
        Multiplies per output pixel per (in_ch, out_ch) pair.
    """
    if algorithm == "direct":
        return float(kernel_size ** 2)
    if algorithm in ("winograd_f2", "winograd_f4"):
        if kernel_size != 3:
            raise ValueError("Winograd F(m,3) op count only applies to kernel_size=3")
        m = 2 if algorithm == "winograd_f2" else 4
        return ((m + kernel_size - 1) / m) ** 2
    if algorithm == "fft":
        pad = input_resolution + kernel_size - 1
        bins = pad * (pad // 2 + 1)
        return 4 * bins / (input_resolution ** 2)
    raise ValueError(f"Unknown algorithm: {algorithm}")


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
    total_flops: float,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    """
    Profile model latency + power + utilization + memory + compute efficiency.

    Args:
        model: PyTorch model.
        input_size: (batch, channels, height, width) tuple.
        device: torch.device.
        total_flops: Precomputed FLOPs (from the FP32 model — fvcore's JIT tracer
            can't trace quantized graphs, and FLOPs are precision-independent since
            INT8 conversion doesn't change the op graph, only weight/activation dtype).
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

    # Memory measurement (CUDA only)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    input_tensor = torch.randn(input_size, device=device)

    with GpuSampler(interval_ms=100) as sampler:
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(input_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.time()

        for _ in range(iters):
            with torch.no_grad():
                _ = model(input_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.time() - start_time) * 1000 / iters

    gpu_metrics = sampler.summary()

    # Aggregate metrics
    latency_ms = elapsed_ms
    batch_size = input_size[0]
    throughput_img_s = (batch_size * 1000.0) / latency_ms
    memory_peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else None

    # Compute efficiency: actual GFLOPs / second. latency_ms is milliseconds, so convert to
    # seconds (/1000) before dividing -- equivalent to total_flops / latency_ms / 1e6.
    compute_efficiency = (total_flops / (latency_ms / 1000)) / 1e9 if latency_ms > 0 else None

    return {
        "latency_ms": latency_ms,
        "throughput_img_s": throughput_img_s,
        "power_draw_avg_w": gpu_metrics["gpu_power_avg_w"],
        "power_draw_std_w": gpu_metrics["gpu_power_std_w"],
        "gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
        "gpu_memory_peak_mb": memory_peak_mb,
        "gpu_temp_avg_c": gpu_metrics["gpu_temp_avg_c"],
        "gpu_memory_used_avg_mb": gpu_metrics["gpu_memory_used_avg_mb"],
        "gpu_energy_wh": gpu_metrics["gpu_energy_wh"],
        "compute_efficiency_gflops_s": compute_efficiency,
    }


def profile_kernel_trace(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Profile model using torch.profiler and extract per-op durations, including the name
    of the actual CUDA kernel cuDNN selected -- ground truth for which algorithm (implicit
    GEMM / FFT / Winograd) ran, instead of inferring it from a 3x3-vs-5x5 speedup ratio.

    Args:
        model: PyTorch model.
        input_tensor: Single input tensor for profiling.
        device: torch.device.

    Returns:
        {
            "winograd_trace_detected": bool (best-effort: cuDNN's Winograd kernel naming
                isn't stable across versions, so a False here doesn't rule it out),
            "top_cuda_kernel": str or None -- the device-side kernel with the highest
                self device time. For a bare single-layer forward pass (the layer
                sweep's use case) this is the convolution kernel cuDNN selected.
            "op_timings": dict (op_name -> device_time_total_us, wrapper aten ops
                included -- inclusive times, for context only).
        }
    """
    model = model.to(device).eval()

    op_timings = {}

    # CPU activity must be recorded too, or CUDA kernels launched from Python ops don't
    # get attributed in key_averages().
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            _ = model(input_tensor.to(device))

    for evt in prof.key_averages():
        if evt.device_time_total > 0:
            op_timings[evt.key] = evt.device_time_total

    # top_cuda_kernel must come from the device-side kernel entries only (device_type ==
    # DeviceType.CUDA -- an enum, not the string "cuda"; that string comparison was an
    # earlier bug here). Ranking op_timings instead would always surface the outermost
    # CPU-side dispatcher (aten::conv2d), whose device_time_total is inclusive of every
    # child kernel and therefore >= the real kernel's own time -- which is how a run of
    # 288 layer configs came back with cudnn_kernel_name == "aten::conv2d" on all of them.
    # self_device_time_total on these leaf kernel entries is their own execution time.
    kernel_timings = {
        evt.key: evt.self_device_time_total
        for evt in prof.key_averages()
        if evt.device_type == DeviceType.CUDA and evt.self_device_time_total > 0
    }
    winograd_detected = any("winograd" in name.lower() for name in kernel_timings)
    top_cuda_kernel = max(kernel_timings, key=kernel_timings.get) if kernel_timings else None

    return {
        "winograd_trace_detected": winograd_detected,
        "top_cuda_kernel": top_cuda_kernel,
        "op_timings": op_timings,
    }


def _winograd_kernel_transform(weight: torch.Tensor) -> torch.Tensor:
    """G-side transform of the kernel into the Winograd domain: (out_ch, in_ch, 3, 3) ->
    (out_ch, in_ch, 4, 4). Shared by winograd_conv2d_f23 and the split-transform path in
    profile_layer_conv_winograd, which precomputes this once outside the timed loop --
    real inference does the same, since weights (unlike the input) are static."""
    G = torch.tensor([[1.0, 0.0, 0.0],
                       [0.5, 0.5, 0.5],
                       [0.5, -0.5, 0.5],
                       [0.0, 0.0, 1.0]], device=weight.device)
    return torch.einsum("xr,ocrs,ys->ocxy", G, weight, G)


def winograd_conv2d_f23(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Real Winograd F(2x2,3x3) convolution: stride=1, groups=1, 'same' padding.

    Fixed transform matrices (Lavin & Gray, "Fast Algorithms for Convolutional
    Neural Networks"): 4x3 G (kernel transform), 4x4 B^T (input transform), 2x4
    A^T (output transform). Input is tiled into overlapping 4x4 blocks (stride 2,
    so each tile maps to one 2x2 output block); both input tiles and the kernel are
    transformed into the Winograd domain, elementwise-multiplied (16 multiplies per
    tile vs. direct GEMM's 36), then inverse-transformed back to 2x2 output tiles.

    Args:
        x: (batch, in_ch, height, width).
        weight: (out_ch, in_ch, 3, 3).

    Returns:
        (batch, out_ch, height, width) -- same shape as F.conv2d(x, weight, padding=1).
    """
    U = _winograd_kernel_transform(weight)
    return winograd_conv2d_f23_transformed(x, U)


def winograd_conv2d_f23_transformed(x: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Same computation as winograd_conv2d_f23, but takes an already Winograd-domain
    kernel U (out_ch, in_ch, 4, 4) from _winograd_kernel_transform, instead of a raw
    (out_ch, in_ch, 3, 3) weight -- only the input-side transform runs per call.

    Args:
        x: (batch, in_ch, height, width).
        U: (out_ch, in_ch, 4, 4), from _winograd_kernel_transform(weight).

    Returns:
        (batch, out_ch, height, width) -- same shape as F.conv2d(x, weight, padding=1).
    """
    device = x.device
    batch, in_ch, height, width = x.shape
    out_ch = U.shape[0]

    BT = torch.tensor([[1.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [0.0, -1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, -1.0]], device=device)
    AT = torch.tensor([[1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, -1.0, -1.0]], device=device)

    # padding=1 on the near side (matches conv's 'same' padding); the far side gets
    # an extra +1 whenever height/width is odd, so the output rounds up to an even
    # size and tiles evenly into 2x2 blocks (cropped back to the true size below).
    pad_bottom = 1 + (height % 2)
    pad_right = 1 + (width % 2)
    x_pad = torch.nn.functional.pad(x, (1, pad_right, 1, pad_bottom))
    tiles = x_pad.unfold(2, 4, 2).unfold(3, 4, 2)  # (b, c, n_tile_h, n_tile_w, 4, 4)

    V = torch.einsum("xr,bcHWrs,ys->bcHWxy", BT, tiles, BT)
    M = torch.einsum("ocxy,bcHWxy->boHWxy", U, V)
    Y = torch.einsum("px,boHWxy,qy->boHWpq", AT, M, AT)

    n_tile_h, n_tile_w = Y.shape[2], Y.shape[3]
    out = Y.permute(0, 1, 2, 4, 3, 5).reshape(batch, out_ch, n_tile_h * 2, n_tile_w * 2)
    return out[..., :height, :width]


# torch.compile artifacts, built once and reused across calls -- compiling per-call would
# time compilation itself instead of the kernel it produces.
_compiled_winograd_f23 = None
_compiled_winograd_f23_transformed = None


def _get_compiled_winograd_f23():
    global _compiled_winograd_f23
    if _compiled_winograd_f23 is None:
        _compiled_winograd_f23 = torch.compile(winograd_conv2d_f23, mode="max-autotune")
    return _compiled_winograd_f23


def _get_compiled_winograd_f23_transformed():
    global _compiled_winograd_f23_transformed
    if _compiled_winograd_f23_transformed is None:
        _compiled_winograd_f23_transformed = torch.compile(winograd_conv2d_f23_transformed, mode="max-autotune")
    return _compiled_winograd_f23_transformed


def profile_layer_conv_winograd(
    kernel_size: int,
    in_ch: int,
    out_ch: int,
    input_shape: tuple,
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
    compile: bool = False,
) -> dict:
    """
    Profile the hand-rolled Winograd F(2x2,3x3) convolution above -- a real transform,
    not cuDNN's internal kernel selection. Only meaningful for kernel_size=3, stride=1,
    groups=1 (F(2,3)'s domain); mirrors profile_layer_conv_fft's kernel_size>=5
    restriction on the other side. The kernel transform U is precomputed once before the
    timed loop (weights are static in real inference, unlike the input) -- only the
    input-side transform, elementwise multiply, and output-side transform are timed.

    Args:
        kernel_size: must be 3; anything else is skipped (see FFT's kernel_size<5 note).
        in_ch, out_ch: channel counts.
        input_shape: (batch, channels, height, width).
        device: torch.device.
        warmup, iters: profiling parameters.
        compile: if True, times `torch.compile(winograd_conv2d_f23_transformed,
            mode="max-autotune")` instead of the eager function -- isolates how much of
            the eager-vs-cuDNN gap is Python/dispatch overhead vs. the algorithm itself.
            Compilation happens once (module-level cache) and is absorbed into `warmup`,
            not the timed loop.

    Returns:
        {"latency_ms": float or None, "note": str}
    """
    if kernel_size != 3:
        return {
            "latency_ms": None,
            "note": f"Skipped: kernel_size={kernel_size} != 3; F(2x2,3x3) only applies to 3x3 kernels",
        }

    conv_fn = _get_compiled_winograd_f23_transformed() if compile else winograd_conv2d_f23_transformed

    input_tensor = torch.randn(input_shape, device=device)
    weight = torch.randn(out_ch, in_ch, 3, 3, device=device)

    # winograd_conv2d_f23's einsums lower to matmul, which defaults to full-FP32 precision
    # (torch.backends.cuda.matmul.allow_tf32=False) -- unlike cuDNN's conv path, which already
    # defaults to TF32 tensor cores. Without this, "vs cuDNN" isn't a fair comparison; applies
    # to both eager and compiled so compiled-vs-eager isn't confounded by a precision change.
    prior_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        U = _winograd_kernel_transform(weight)

        for _ in range(warmup):
            with torch.no_grad():
                _ = conv_fn(input_tensor, U)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.time()

        for _ in range(iters):
            with torch.no_grad():
                _ = conv_fn(input_tensor, U)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.time() - start_time) * 1000 / iters
    finally:
        torch.set_float32_matmul_precision(prior_precision)

    return {
        "latency_ms": elapsed_ms,
        "note": ("torch.compile'd Winograd F(2x2,3x3), kernel transform precomputed (max-autotune, TF32); "
                  "stride=1, groups=1 only"
                  if compile else
                  "Hand-rolled Winograd F(2x2,3x3), kernel transform precomputed (TF32); stride=1, groups=1 only"),
    }


def _next_smooth5(n: int) -> int:
    """Smallest integer >= n whose only prime factors are 2, 3, 5.

    cuFFT is fast on these "smooth" sizes and falls back to slow Bluestein convolution on
    sizes with large prime factors. Padding to exactly the minimum linear-convolution size
    (input_size + kernel_size - 1) can land on one of those (e.g. at input=64: k=5 gives
    68=2^2*17, k=11 gives 74=2*37) and tank performance for no algorithmic reason -- a few
    extra zero-padded rows/columns of amortized-filter FFT are cheaper than that penalty.
    """
    candidate = n
    while True:
        m = candidate
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return candidate
        candidate += 1


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

    FFT-based conv: zero-pad input and kernel to a 5-smooth size >= (input_size +
    kernel_size - 1), apply rfft2, multiply, sum over input channels, irfft2, crop. The
    filter FFT is precomputed once (real inference does the same, since weights are
    static); only the input-side FFT, multiply, and inverse FFT are timed per iteration.

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

    pad_h = _next_smooth5(height + kernel_size - 1)
    pad_w = _next_smooth5(width + kernel_size - 1)

    input_tensor = torch.randn(input_shape, device=device, dtype=torch.float32)
    kernels = torch.randn(out_ch, in_ch, kernel_size, kernel_size, device=device, dtype=torch.float32)

    # Filter transform is static per weight -- precompute once. Redoing it every iteration
    # (the old behavior) FFTs out_ch*in_ch filters per call (16384 FFTs of ~74x74 at
    # C=128, ~368MB allocated) vs. in_ch FFTs for the input, and dominated the measurement.
    kernel_padded = torch.nn.functional.pad(
        kernels, (0, pad_w - kernel_size, 0, pad_h - kernel_size), mode="constant", value=0,
    )
    kernel_fft = torch.fft.rfft2(kernel_padded, dim=(-2, -1))

    torch.cuda.synchronize(device) if device.type == "cuda" else None

    for _ in range(warmup):
        input_padded = torch.nn.functional.pad(
            input_tensor,
            (0, pad_w - width, 0, pad_h - height),
            mode="constant",
            value=0,
        )
        input_fft = torch.fft.rfft2(input_padded, dim=(-2, -1))

        # Multiply and sum over input channels
        # input_fft: (batch, in_ch, pad_h, pad_w//2+1)
        # kernel_fft: (out_ch, in_ch, pad_h, pad_w//2+1)
        output_fft = torch.einsum("bchw,ochw->bohw", input_fft, kernel_fft)

        # IRFFT, cropped to valid output size
        output = torch.fft.irfft2(output_fft, s=(pad_h, pad_w), dim=(-2, -1))
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
        input_fft = torch.fft.rfft2(input_padded, dim=(-2, -1))
        output_fft = torch.einsum("bchw,ochw->bohw", input_fft, kernel_fft)
        output = torch.fft.irfft2(output_fft, s=(pad_h, pad_w), dim=(-2, -1))
        output = output[..., :height, :width]

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    elapsed_ms = (time.time() - start_time) * 1000 / iters

    return {
        "latency_ms": elapsed_ms,
        "note": "FFT-based convolution, filter FFT precomputed, padded to a 5-smooth size; stride=1 only",
    }
