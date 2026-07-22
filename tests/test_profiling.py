"""GpuSampler must degrade to all-None fields when nvidia-smi is unavailable."""
import time

import torch

from ml.profiling import GpuSampler, profile_layer_latency


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
