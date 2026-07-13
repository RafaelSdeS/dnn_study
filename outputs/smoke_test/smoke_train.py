"""Throwaway PCAD pipeline smoke test — NOT part of the repo, NOT committed.
Registers one small model by hand (production scripts/train.py has no
registration step yet — see conversation) and runs 1 epoch on a 64-image
subset to prove CUDA + data path + train loop + checkpoint + log all work
end-to-end on a compute node.
"""
import random
import sys
from pathlib import Path

REPO_ROOT = Path("/home/users/rsdsouza/dnn_study")
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from ml import DataConfig, TrainerConfig, MODEL_REGISTRY, register_model, Trainer, create_imagenet_loaders, set_global_seed
from models.alexnet_variants import AlexNet3x3GAP

register_model("smoke_alexnet_3x3_gap", AlexNet3x3GAP, fuse_map=[], lr=1e-3)
set_global_seed(42)

dataset_root = Path.home() / ".cache/kagglehub/datasets/akash2sharma/tiny-imagenet/versions/1/tiny-imagenet-200/train"
assert dataset_root.exists(), f"dataset not found at {dataset_root}"

data_cfg = DataConfig(dataset_path=str(dataset_root), batch_size=16, num_workers=2, pin_memory=True, train_val_split=0.9, seed=42)
train_ds, val_ds, _, _ = create_imagenet_loaders(data_cfg, persistent_workers=False)

# shrink to a tiny slice so this finishes in seconds, not minutes
rng = random.Random(42)
train_small = Subset(train_ds, rng.sample(range(len(train_ds)), 48))
val_small = Subset(val_ds, rng.sample(range(len(val_ds)), 16))
train_loader = DataLoader(train_small, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_small, batch_size=16, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device} cuda_available={torch.cuda.is_available()}", flush=True)

spec = MODEL_REGISTRY["smoke_alexnet_3x3_gap"]
model = spec["ctor"]()
trainer_cfg = TrainerConfig(epochs=1, lr=spec.get("lr", 1e-3), use_amp=True, early_stopping_patience=None)

smoke_root = REPO_ROOT / "outputs" / "smoke_test"
checkpoints_dir = smoke_root / "checkpoints"
logs_dir = smoke_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
trainer = Trainer(
    model, train_loader, val_loader, cfg=trainer_cfg, device=device,
    save_dir=checkpoints_dir, run_name="smoke_alexnet_3x3_gap", num_classes=200,
    log_file=logs_dir / "smoke_alexnet_3x3_gap.log",
)
result = trainer.fit()
print("SMOKE TEST RESULT:", result["best_val_top1"], result["total_training_time_s"], flush=True)
print("SMOKE TEST OK", flush=True)
