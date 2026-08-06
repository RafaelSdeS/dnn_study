import gzip
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    scaler=None,
    best_val_acc: float = 0.0,
    best_val_top5: float = 0.0,
    history: Optional[dict] = None,
    wandb_run_id: Optional[str] = None,
    patience_counter: int = 0,
    elapsed_time_s: float = 0.0,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # ponytail: atomic write via tmp → rename avoids corrupt checkpoint on crash
    tmp = Path(path).with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "metrics": metrics,
        "best_val_acc": best_val_acc,
        "best_val_top5": best_val_top5,
        "history": history or {},
        "wandb_run_id": wandb_run_id,
        "patience_counter": patience_counter,
        "elapsed_time_s": elapsed_time_s,
    }, tmp)
    tmp.rename(path)


def load_resume_state(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler=None,
    device: str = "cpu",
) -> dict:
    """Load full training state from resume checkpoint. Returns dict with epoch, best_val_acc, history, wandb_run_id, patience_counter."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    history = ckpt.get("history", {})
    elapsed_time_s = ckpt.get("elapsed_time_s")
    if elapsed_time_s is None:
        elapsed_time_s = sum(history.get("epoch_time_s", []))
    return {
        "epoch": ckpt.get("epoch", 0),
        "best_val_acc": ckpt.get("best_val_acc", 0.0),
        "best_val_top5": ckpt.get("best_val_top5", 0.0),
        "history": history,
        "wandb_run_id": ckpt.get("wandb_run_id"),
        "patience_counter": ckpt.get("patience_counter", 0),
        "elapsed_time_s": elapsed_time_s,
    }


def auto_resume_path(save_dir: Path | str, run_name: str) -> Optional[Path]:
    """Return path to resume checkpoint if it exists, else None. Enables notebook auto-detection."""
    p = Path(save_dir) / f"{run_name}_resume.pth"
    return p if p.exists() else None


def compress_checkpoint(path: Path | str) -> Optional[Path]:
    """Gzip a finished checkpoint file to <path>.gz — a real compressed artifact, not just a
    size estimate. Call once, after a checkpoint is done being written (final FP32 best model,
    converted INT8 model) — not from the per-epoch resume-checkpoint path, which is rewritten
    too often for re-gzipping every epoch to be worthwhile. Returns the .gz path, or None if
    the source file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    gz_path = p.with_suffix(p.suffix + ".gz")
    with open(p, "rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return gz_path
