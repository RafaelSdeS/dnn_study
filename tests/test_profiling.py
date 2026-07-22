"""GpuSampler must degrade to all-None fields when nvidia-smi is unavailable."""
import time

from ml.profiling import GpuSampler


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
