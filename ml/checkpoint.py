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
    return {
        "epoch": ckpt.get("epoch", 0),
        "best_val_acc": ckpt.get("best_val_acc", 0.0),
        "best_val_top5": ckpt.get("best_val_top5", 0.0),
        "history": ckpt.get("history", {}),
        "wandb_run_id": ckpt.get("wandb_run_id"),
        "patience_counter": ckpt.get("patience_counter", 0),
    }


def auto_resume_path(save_dir: Path | str, run_name: str) -> Optional[Path]:
    """Return path to resume checkpoint if it exists, else None. Enables notebook auto-detection."""
    p = Path(save_dir) / f"{run_name}_resume.pth"
    return p if p.exists() else None
