"""GpuSampler must degrade to all-None fields when nvidia-smi is unavailable."""
import time

import pytest
import torch
import torch.nn.functional as F

import torch.nn as nn

from ml.profiling import (
    GpuSampler,
    profile_layer_latency,
    profile_layer_conv_winograd,
    profile_kernel_trace,
    conv_multiplies_per_flop,
    winograd_conv2d_f23,
    _get_compiled_winograd_f23,
)


def test_winograd_f23_matches_direct_conv():
    # The real test of a hand-rolled Winograd transform: does it compute the same
    # thing as direct conv? A sign/transpose error in B/G/A gives a plausible-looking
    # but wrong number, not a crash. Covers both even (8) and odd (7) spatial sizes,
    # since odd sizes exercise the extra-padding branch in winograd_conv2d_f23.
    torch.manual_seed(0)
    for size in (8, 7):
        x = torch.randn(2, 3, size, size)
        weight = torch.randn(4, 3, 3, 3)
        expected = F.conv2d(x, weight, padding=1)
        actual = winograd_conv2d_f23(x, weight)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-4), f"mismatch at size={size}"


def test_profile_layer_conv_winograd_skips_non_3x3_kernels():
    result = profile_layer_conv_winograd(5, 8, 8, (1, 8, 8, 8), torch.device("cpu"), warmup=0, iters=1)
    assert result["latency_ms"] is None


def test_profile_layer_conv_winograd_restores_float32_matmul_precision():
    # profile_layer_conv_winograd flips torch's global TF32 matmul precision for its timed
    # region (see comment at the call site) -- a botched try/finally would leak that setting
    # into every other test/profiling call that runs after it.
    prior = torch.get_float32_matmul_precision()
    profile_layer_conv_winograd(3, 4, 4, (1, 4, 8, 8), torch.device("cpu"), warmup=0, iters=1)
    assert torch.get_float32_matmul_precision() == prior


def test_profile_layer_conv_winograd_runs_on_3x3():
    result = profile_layer_conv_winograd(3, 8, 8, (1, 8, 8, 8), torch.device("cpu"), warmup=0, iters=1)
    assert result["latency_ms"] > 0


def test_profile_layer_conv_winograd_compiled_matches_direct_conv():
    # torch.compile is supposed to preserve output parity with the eager function
    # (test_winograd_f23_matches_direct_conv already covers that function's own
    # correctness) -- this confirms compile=True actually reaches the compiled path and
    # that path still agrees with F.conv2d. Runs on CUDA only: mode="max-autotune" is the
    # real target (matches the production profiling sweep); on CPU it needs a C++
    # toolchain this box may not have, and isn't what gets profiled anyway.
    if not torch.cuda.is_available():
        pytest.skip("compiled Winograd path targets CUDA, matching the production sweep")
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8, device="cuda")
    weight = torch.randn(4, 3, 3, 3, device="cuda")
    expected = F.conv2d(x, weight, padding=1)

    # Matches profile_layer_conv_winograd's own precision setting (see the docstring there):
    # without it, torch.compile emits a UserWarning that TF32 tensor cores are available but
    # unused, and this direct call wouldn't actually exercise the precision mode the real
    # profiling sweep runs under.
    prior_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        actual = _get_compiled_winograd_f23()(x, weight)
    finally:
        torch.set_float32_matmul_precision(prior_precision)
    # TF32 (~10-bit mantissa) is genuinely lossy relative to F.conv2d's full-precision
    # reference -- the old atol=1e-3 was calibrated for this call running at full FP32
    # (before precision="high" was added above), where it happened to match tightly. Under
    # TF32, both sides round independently and no longer agree to 1e-3; rtol=1e-2 reflects
    # TF32's real precision instead of re-hiding it behind an unset precision mode.
    assert torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)

    result = profile_layer_conv_winograd(
        3, 3, 4, x.shape, torch.device("cuda"), warmup=0, iters=1, compile=True
    )
    assert result["latency_ms"] > 0
    assert "compile" in result["note"].lower()


def test_conv_multiplies_per_flop():
    # direct: raw k^2, no acceleration claim.
    assert conv_multiplies_per_flop(3, "direct") == 9.0
    assert conv_multiplies_per_flop(5, "direct") == 25.0
    # Winograd F(2x2,3x3)=4 mults/tile-output vs. direct's 9; F(4x4,3x3)=2.25 -- both only
    # defined for kernel_size=3 (F(m,3)'s domain).
    assert conv_multiplies_per_flop(3, "winograd_f2") == 4.0
    assert conv_multiplies_per_flop(3, "winograd_f4") == 2.25
    with pytest.raises(ValueError):
        conv_multiplies_per_flop(5, "winograd_f2")
    # FFT's amortized-filter cost is near-flat in kernel_size (the whole point of FFT conv),
    # unlike direct's k^2 growth -- k=11 should cost barely more than k=5.
    fft_5 = conv_multiplies_per_flop(5, "fft", input_resolution=64)
    fft_11 = conv_multiplies_per_flop(11, "fft", input_resolution=64)
    assert fft_5 < conv_multiplies_per_flop(5, "direct")
    assert fft_11 / fft_5 < 1.5


def test_profile_kernel_trace_populates_op_timings_on_cuda():
    # Regression test for two successive bugs here: (1) evt.device_type is a
    # torch.autograd.DeviceType enum, never the string "cuda", so the old
    # `evt.device_type == "cuda"` guard silently zeroed op_timings on every call;
    # (2) top_cuda_kernel was then picked by inclusive device_time_total, which always
    # surfaces the outermost CPU-side dispatcher (aten::conv2d) instead of the actual
    # device kernel -- an entire 288-config sweep recorded cudnn_kernel_name ==
    # "aten::conv2d" before this was caught.
    if not torch.cuda.is_available():
        pytest.skip("kernel trace only runs real device kernels on CUDA")
    conv = nn.Conv2d(4, 4, 3, padding=1, bias=False).to("cuda").eval()
    input_tensor = torch.randn(1, 4, 8, 8, device="cuda")
    result = profile_kernel_trace(conv, input_tensor, torch.device("cuda"))
    assert result["op_timings"], "op_timings must be populated for a real CUDA conv forward pass"
    assert result["top_cuda_kernel"] is not None
    assert not result["top_cuda_kernel"].startswith("aten::"), (
        "top_cuda_kernel must be the device-side kernel cuDNN selected, not the "
        f"CPU-side aten wrapper: got {result['top_cuda_kernel']}"
    )


def test_profile_layer_latency_groups_runs_dense_and_depthwise():
    # Smoke-tests that the `groups` kwarg reaches nn.Conv2d without shape errors:
    # groups=1 (dense) and groups=in_ch (depthwise) must both build and run.
    device = torch.device("cpu")
    dense_ms = profile_layer_latency(3, 16, 16, (1, 16, 8, 8), device, warmup=0, iters=1, groups=1)
    depthwise_ms = profile_layer_latency(3, 16, 16, (1, 16, 8, 8), device, warmup=0, iters=1, groups=16)
    assert dense_ms > 0 and depthwise_ms > 0


def test_gpu_sampler_degrades_gracefully_without_nvidia_smi(monkeypatch):
    # A dev box with a real GPU has nvidia-smi, so relying on the ambient environment lacking
    # it (as this test used to) only passes by accident. Force the actual failure mode instead --
    # Popen raising FileNotFoundError, exactly what happens when nvidia-smi isn't on PATH -- so this
    # test exercises the `except FileNotFoundError` branch in GpuSampler._run regardless of machine.
    def fake_popen(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr("ml.profiling.subprocess.Popen", fake_popen)

    with GpuSampler(interval_ms=50) as sampler:
        time.sleep(0.15)
    summary = sampler.summary()

    expected_keys = {
        "gpu_power_avg_w", "gpu_power_std_w", "gpu_utilization_pct",
        "gpu_temp_avg_c", "gpu_memory_used_avg_mb", "gpu_energy_wh",
    }
    assert set(summary.keys()) == expected_keys
    for key, value in summary.items():
        assert value is None, f"{key} unexpectedly populated without nvidia-smi: {value}"


def test_gpu_sampler_summary_without_entering_context():
    sampler = GpuSampler()
    summary = sampler.summary()
    assert all(v is None for v in summary.values())
