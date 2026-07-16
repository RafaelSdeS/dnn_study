"""End-to-end smoke test: Trainer.fit() on a tiny synthetic CPU dataset for 1 epoch.

Verifies the full loop (train, validate, checkpoint, history/metrics collection) runs without
crashing and that every new hardware/throughput metric field is present and well-formed.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import TrainerConfig
from ml.trainer import Trainer

NEW_HISTORY_KEYS = {
    "images_per_sec", "avg_batch_time_s", "cpu_percent", "ram_used_mb",
    "gpu_power_avg_w", "gpu_utilization_pct", "gpu_temp_avg_c",
    "gpu_memory_used_avg_mb", "gpu_energy_wh",
}


def _tiny_loader():
    x = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 5, (8,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def _tiny_model():
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5)
    )


def test_fit_one_epoch_populates_all_history_fields(tmp_path):
    loader = _tiny_loader()
    cfg = TrainerConfig(epochs=1, use_amp=False, early_stopping_patience=None)
    trainer = Trainer(
        _tiny_model(), loader, loader, cfg, torch.device("cpu"),
        tmp_path, "smoke", num_classes=5,
    )

    results = trainer.fit()
    history = results["history"]

    assert NEW_HISTORY_KEYS <= history.keys()
    for key in NEW_HISTORY_KEYS:
        assert len(history[key]) == 1, key

    # No GPU on this box: hardware fields degrade to None rather than crashing.
    assert history["gpu_power_avg_w"][0] is None
    # CPU/RAM/throughput are always measurable, even without a GPU.
    assert history["cpu_percent"][0] is not None
    assert history["ram_used_mb"][0] > 0
    assert history["images_per_sec"][0] > 0

    assert (tmp_path / "smoke_best.pth").exists()
    assert (tmp_path / "smoke_resume.pth").exists()
