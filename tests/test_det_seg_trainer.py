"""SegmentationTrainer checkpoint regression + TinyLoader coverage - previously untested.

Mirrors ml/trainer.py's best-checkpoint bug (best_val_acc seeded at 0.0, so a model stuck at
exactly 0% never got a checkpoint saved) which also existed in both det_seg_trainer.py loops.
"""
import torch
import torch.nn as nn

from ml.config import TrainerConfig
from ml.det_seg_trainer import SegmentationTrainer, TinyLoader


def test_tiny_loader_exposes_batch_size_and_iterates_its_batches():
    batches = [("a", "b"), ("c", "d")]
    loader = TinyLoader(batches, batch_size=4)

    assert loader.batch_size == 4
    assert len(loader) == 2
    assert list(loader) == batches


class _AlwaysClassZeroSegModel(nn.Module):
    """Ignores the input, always predicts class 0 for every pixel -> a hard, deterministic 0% mIoU."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.dummy = nn.Conv2d(3, num_classes, 1)  # keeps a param in the backward graph

    def forward(self, x):
        b, _, h, w = x.shape
        bias = torch.zeros(1, self.num_classes, 1, 1, device=x.device)
        bias[:, 0] = 10.0
        return bias.expand(b, -1, h, w) + 0.0 * self.dummy(x)


def test_zero_miou_run_still_saves_a_best_checkpoint(tmp_path):
    """Regression: best_val_mIoU used to be seeded at 0.0, so `val_mIoU > best_val_mIoU` was
    never true for a model stuck at exactly 0% mIoU, and no checkpoint was ever saved."""
    images = torch.randn(2, 3, 8, 8)
    masks = torch.ones(2, 8, 8, dtype=torch.long)  # never class 0 -> always-0 model always misses
    loader = TinyLoader([(images, masks)], batch_size=2)
    cfg = TrainerConfig(epochs=1, use_amp=False, early_stopping_patience=None)
    trainer = SegmentationTrainer(
        _AlwaysClassZeroSegModel(), loader, loader, cfg, torch.device("cpu"),
        tmp_path, "zero_miou", num_classes=2,
    )

    history = trainer.fit()

    assert history["val_mIoU"][0] == 0.0
    assert (tmp_path / "zero_miou_best.pth").exists()
