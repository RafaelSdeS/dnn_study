"""GpuSampler must degrade to all-None fields when nvidia-smi is unavailable."""
import time

import torch
import torch.nn.functional as F

from ml.profiling import (
    GpuSampler,
    profile_layer_latency,
    profile_layer_conv_winograd,
    winograd_conv2d_f23,
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


def test_profile_layer_conv_winograd_runs_on_3x3():
    result = profile_layer_conv_winograd(3, 8, 8, (1, 8, 8, 8), torch.device("cpu"), warmup=0, iters=1)
    assert result["latency_ms"] > 0


def test_profile_layer_latency_groups_runs_dense_and_depthwise():
    # Smoke-tests that the `groups` kwarg reaches nn.Conv2d without shape errors:
    # groups=1 (dense) and groups=in_ch (depthwise) must both build and run.
    device = torch.device("cpu")
    dense_ms = profile_layer_latency(3, 16, 16, (1, 16, 8, 8), device, warmup=0, iters=1, groups=1)
    depthwise_ms = profile_layer_latency(3, 16, 16, (1, 16, 8, 8), device, warmup=0, iters=1, groups=16)
    assert dense_ms > 0 and depthwise_ms > 0


def test_gpu_sampler_degrades_gracefully_without_nvidia_smi():
    with GpuSampler(interval_ms=50) as sampler:
        time.sleep(0.15)
    summary = sampler.summary()

    expected_keys = {
        "gpu_power_avg_w", "gpu_power_std_w", "gpu_utilization_pct",
        "gpu_temp_avg_c", "gpu_memory_used_avg_mb", "gpu_energy_wh",
    }
    assert set(summary.keys()) == expected_keys
    # This dev box has no nvidia-smi, so every field should be None rather than raising.
    for key, value in summary.items():
        assert value is None, f"{key} unexpectedly populated without nvidia-smi: {value}"


def test_gpu_sampler_summary_without_entering_context():
    sampler = GpuSampler()
    summary = sampler.summary()
    assert all(v is None for v in summary.values())
