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
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 3, 8, 8, generator=g)
    y = torch.randint(0, 5, (8,), generator=g)
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

    # GpuSampler polls nvidia-smi in the background regardless of self.device (it measures
    # system-wide GPU telemetry, not just this run's usage) and degrades to None if no sample
    # lands before the tiny 2-batch epoch ends -- on a box with a real GPU this is a race, not
    # a guarantee, so assert well-formedness rather than hardcoding "no GPU on this box".
    assert history["gpu_power_avg_w"][0] is None or history["gpu_power_avg_w"][0] >= 0
    # CPU/RAM/throughput are always measurable, even without a GPU.
    assert history["cpu_percent"][0] is not None
    assert history["ram_used_mb"][0] > 0
    assert history["images_per_sec"][0] > 0

    assert (tmp_path / "smoke_best.pth").exists()
    assert (tmp_path / "smoke_resume.pth").exists()


class _AlwaysWrongModel(nn.Module):
    """Ignores the input and always predicts class 0 -> a hard, deterministic 0% val accuracy."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.dummy = nn.Linear(3 * 8 * 8, num_classes)  # keeps a param in the backward graph

    def forward(self, x):
        batch = x.shape[0]
        logits = torch.full((batch, self.dummy.out_features), -10.0, device=x.device)
        logits[:, 0] = 10.0
        return logits + 0.0 * self.dummy(x.flatten(1))


def test_zero_accuracy_run_still_saves_a_best_checkpoint(tmp_path):
    """Regression: best_val_acc used to be seeded at 0.0, so `val_acc > best_val_acc` was
    never true for a model stuck at exactly 0% accuracy, and no checkpoint was ever saved."""
    x = torch.randn(8, 3, 8, 8)
    y = torch.ones(8, dtype=torch.long)  # never class 0 -> the always-0 model always misses
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    cfg = TrainerConfig(epochs=1, use_amp=False, early_stopping_patience=None)
    trainer = Trainer(
        _AlwaysWrongModel(), loader, loader, cfg, torch.device("cpu"),
        tmp_path, "zero_acc", num_classes=5,
    )

    results = trainer.fit()

    assert results["history"]["val_acc"][0] == 0.0
    assert (tmp_path / "zero_acc_best.pth").exists()


class _DecayingValTrainer(Trainer):
    """Scripts val accuracy to peak on epoch 1 and decay, so best epoch != last epoch
    deterministically (real training on 8 random samples wouldn't guarantee that)."""

    SCRIPTED_ACC = [50.0, 40.0, 30.0]

    def _validate(self, model, criterion):
        model.eval()
        acc = self.SCRIPTED_ACC[len(self._seen)]
        self._seen.append(acc)
        return 1.0, acc, acc


def test_fit_restores_best_checkpoint_not_last_epoch(tmp_path):
    """Regression: fit() used to leave the LAST epoch's weights in self.model, so a caller
    evaluating right after fit() (scripts/train.py) measured a different model than the one
    saved as {name}_best.pth -- the one build_qat()/load_best_model() reloads for QAT."""
    loader = _tiny_loader()
    cfg = TrainerConfig(epochs=3, lr=0.5, use_amp=False, early_stopping_patience=None)
    trainer = _DecayingValTrainer(
        _tiny_model(), loader, loader, cfg, torch.device("cpu"),
        tmp_path, "restore", num_classes=5,
    )
    trainer._seen = []

    results = trainer.fit()
    assert results["best_epoch"] == 0  # epoch 1 scored best

    best = torch.load(tmp_path / "restore_best.pth", weights_only=False)["model_state_dict"]
    last = torch.load(tmp_path / "restore_resume.pth", weights_only=False)["model_state_dict"]
    # Guard against a vacuous assertion: the lr has to have actually moved the weights.
    assert any(not torch.equal(last[k], best[k]) for k in best), "weights never changed"

    live = trainer.model.state_dict()
    for key, tensor in best.items():
        assert torch.equal(live[key], tensor), f"{key} kept the last epoch's weights"
