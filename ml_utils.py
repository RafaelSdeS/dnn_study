"""
Reusable utilities for ML research notebooks.
Provides reproducibility, training loops, evaluation, checkpointing, and logging utilities.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import time
import random
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Callable, List, Iterable, Sequence
from collections import defaultdict
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugmentPolicy
import torch.ao.quantization as tq
import torch.optim as optim
from tqdm.auto import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================================================
# CONSTANTS
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Global registry of architectures available for training/QAT/comparison.
MODEL_REGISTRY = {}

EPOCHS_FP32 = 100
EPOCHS_INT8 = 20
EARLY_STOPPING_PATIENCE = 5

# =============================================================================
# REPRODUCIBILITY & SYSTEM UTILITIES
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_system_info() -> Dict[str, Any]:
    """
    Capture system and hardware information for experiment reproducibility.

    Returns:
        Dictionary containing CPU, GPU, and PyTorch version info
    """
    info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info

def setup_experiment(seed: int, cudnn_benchmark: bool = True) -> torch.device:
    """
    Sets reproducibility + selects device + configures cuDNN.
    """
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determinism vs speed tradeoff (explicit control)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    return device

# =============================================================================
# DATA LOADING
# =============================================================================

def create_imagenet_loaders(
    dataset_path: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    train_split: float,
    seed: int,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    """
    Build train/val ImageFolder datasets (with separate transforms) and
    DataLoaders from a single ImageNet-style directory, using a seeded
    deterministic split.
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.7, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=15,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_full = datasets.ImageFolder(dataset_path, transform=transform_train)
    val_full = datasets.ImageFolder(dataset_path, transform=transform_val)

    assert train_full.classes == val_full.classes

    n_total = len(train_full)
    n_train = int(train_split * n_total)

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen).tolist()

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_ds, val_ds, train_loader, val_loader

# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: str
) -> None:
    """
    Save training checkpoint with full reproducibility info.

    Args:
        model: Neural network model
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state (optional)
        epoch: Current epoch number
        metrics: Dictionary of training metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu"
) -> Tuple[int, Dict[str, float]]:
    """
    Load checkpoint and restore training state.

    Args:
        checkpoint_path: Path to checkpoint
        model: Neural network model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load checkpoint to

    Returns:
        Tuple of (epoch, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})

def save_model(model: nn.Module, model_path: str) -> None:
    """Save model weights only."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

def load_model(model: nn.Module, model_path: str, device: str = "cpu") -> None:
    """Load model weights."""
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

# =============================================================================
# MODEL ANALYSIS
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model: nn.Module, dtype_bits: int = 32) -> float:
    """
    Estimate model size in MB.

    Args:
        model: Neural network model
        dtype_bits: Bits per parameter (default 32 for FP32)

    Returns:
        Model size in MB
    """
    param_count = sum(p.numel() for p in model.parameters())
    bytes_per_param = dtype_bits // 8
    return (param_count * bytes_per_param) / (1024 ** 2)

def model_summary(model: nn.Module, device: str = "cpu") -> Dict[str, Any]:
    """
    Get model complexity summary.

    Args:
        model: Neural network model
        device: Device model is on

    Returns:
        Dictionary with parameter count, size, etc.
    """
    model = model.to(device)
    params = count_parameters(model)
    size_fp32 = get_model_size_mb(model, dtype_bits=32)
    size_int8 = get_model_size_mb(model, dtype_bits=8)

    return {
        "total_parameters": params,
        "parameters_millions": params / 1e6,
        "size_fp32_mb": size_fp32,
        "size_int8_mb": size_int8,
    }

def disk_mb(path: str) -> float:
    """Size of a file on disk, in MB (NaN if it doesn't exist)."""
    if not os.path.exists(path):
        return float("nan")
    return os.path.getsize(path) / (1024 ** 2)

# =============================================================================
# TRAINING LOOP PRIMITIVES
# =============================================================================

class TrainingMetrics:
    """Track and accumulate training metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_epoch = defaultdict(float)

    def reset_epoch(self) -> None:
        """Reset current epoch metrics."""
        self.current_epoch.clear()

    def update(self, key: str, value: float, count: int = 1) -> None:
        """Update running metric."""
        if key not in self.current_epoch:
            self.current_epoch[key] = 0.0
        self.current_epoch[key] += value * count

    def compute_epoch(self, count: int = 1) -> Dict[str, float]:
        """Compute averaged metrics for epoch."""
        return {k: v / count for k, v in self.current_epoch.items()}

    def log_epoch(self, epoch_metrics: Dict[str, float]) -> None:
        """Log epoch metrics to history."""
        for key, value in epoch_metrics.items():
            self.metrics[key].append(value)

    def get_history(self) -> Dict[str, List[float]]:
        """Get full training history."""
        return dict(self.metrics)

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    max_batches: Optional[int] = None,
    grad_clip_norm: Optional[float] = None,
    # ema: Optional[EMA] = None,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        scaler: AMP gradient scaler (optional)
        use_amp: Whether to use automatic mixed precision
        max_batches: Limit training batches per epoch (for debugging)
        grad_clip_norm: If set, clip gradient norm to this value before
            optimizer.step() (cheap insurance against rare AMP loss spikes,
            mainly useful for from-scratch models)
        ema: Optional EMA tracker; updated once per step right after
            optimizer.step()

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch_idx, (data, target) in enumerate(progress_bar):
        if max_batches and batch_idx >= max_batches:
            break

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if use_amp and scaler:
            with torch.amp.autocast("cuda"):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        # if ema is not None:
        #     ema.update(model)

        total_loss += loss.item() * target.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy

@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Evaluate model on validation set.

    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        max_batches: Limit evaluation batches (for debugging)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(val_loader, desc="Evaluation")

    for batch_idx, (data, target) in enumerate(progress_bar):
        if max_batches and batch_idx >= max_batches:
            break

        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * target.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        progress_bar.set_postfix({"acc": f"{100 * correct / total:.2f}%"})

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy

@torch.no_grad()
def evaluate_topk(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    topk=(1, 5),
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on validation set, reporting Top-k accuracy for each k
    in `topk` (computed in a single pass, so the model only runs once).
    """
    model.eval()

    total_loss = 0.0
    total = 0
    counts = {k: 0.0 for k in topk}

    for batch_idx, (data, target) in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break

        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * target.size(0)

        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in topk:
            counts[k] += correct[:k].reshape(-1).float().sum().item()

        total += target.size(0)

    result = {
        "loss": total_loss / total,
    }

    for k in topk:
        result[f"top{k}"] = 100.0 * counts[k] / total

    return result

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    save_dir: str = "./checkpoints",
    model_name: str = "model",
    use_amp: bool = False,
    early_stopping_patience: Optional[int] = None,
    wandb_log: bool = True,
    max_batches: Optional[int] = None,
    grad_clip_norm: Optional[float] = None,
    # ema: Optional[EMA] = None,
    epoch_callback: Optional[Callable[[int, nn.Module], None]] = None,
) -> Dict[str, Any]:
    """
    Complete training loop with checkpointing and optional wandb logging.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Number of training epochs
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        save_dir: Directory to save checkpoints
        model_name: Name for saved models
        use_amp: Whether to use automatic mixed precision
        early_stopping_patience: Early stopping patience (optional)
        wandb_log: Whether to log to wandb
        max_batches: Limit batches per epoch (for debugging)
        grad_clip_norm: If set, clip gradient norm during training (see
            `train_epoch`)
        ema: Optional EMA tracker (see `EMA` class). If provided, EMA
            weights are saved alongside the best raw checkpoint as
            "{model_name}_ema.pth" once training finishes.
        epoch_callback: Optional `fn(epoch, model)` called at the START of
            each epoch (0-indexed). Used e.g. to freeze QAT observers/BN
            partway through QAT fine-tuning via `freeze_qat_observers`.

    Returns:
        Dictionary with training history and best metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    metrics = TrainingMetrics()
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        if epoch_callback is not None:
            epoch_callback(epoch, model)

        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, use_amp=use_amp, max_batches=max_batches,
            grad_clip_norm=grad_clip_norm, 
            # ema=ema,
        )

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, max_batches=max_batches)

        # Learning rate scheduling
        if scheduler:
            scheduler.step()

        # Record metrics
        epoch_metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": time.time() - epoch_start,
        }
        metrics.log_epoch(epoch_metrics)

        # Log to wandb
        if wandb_log and HAS_WANDB:
            wandb.log(epoch_metrics, step=epoch + 1)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
            save_model(model, best_model_path)

            if wandb_log and HAS_WANDB:
                wandb.log({"best_val_acc": best_val_acc})
        else:
            patience_counter += 1

        # Early stopping
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_metrics['epoch_time']:.1f}s")

    # if ema is not None:
    #     ema_model = copy.deepcopy(model).cpu()
    #     ema.copy_to(ema_model)
    #     save_model(ema_model, os.path.join(save_dir, f"{model_name}_ema.pth"))

    return {
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "history": metrics.get_history(),
    }

# =============================================================================
# HIGH-LEVEL TRAIN/EVAL ORCHESTRATION
# =============================================================================

def train_and_evaluate(
    model,
    model_name,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    device,
    save_dir,
    scheduler=None,
    use_amp=False,
    early_stopping_patience=None,
    model_ctor=None,
    config_dict=None,
):
    """Train `model`, then reload its best checkpoint and report top-1/top-5.

    `model_ctor` is a zero-arg callable that returns a fresh CPU instance of
    the same architecture; needed to load the best weights cleanly for FP32
    models (for QAT we just keep the in-place model).
    """
    current_lr = optimizer.param_groups[0]["lr"]

    print("=" * 72)
    print(f"Training: {model_name} (lr={current_lr}, epochs={epochs})")
    print("=" * 72)

    init_wandb(
        project="tiny-imagenet",
        experiment_name=model_name,
        config=config_dict or {},
        offline=True,
    )

    try:
        results = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            num_epochs=epochs,
            device=device,
            scheduler=scheduler,
            save_dir=save_dir,
            model_name=model_name,
            use_amp=use_amp,
            early_stopping_patience=early_stopping_patience,
        )

        # Reload best checkpoint and report Top-1/Top-5
        best_path = os.path.join(save_dir, f"{model_name}_best.pth")

        if HAS_WANDB and os.path.exists(best_path):
            artifact = wandb.Artifact(
                name=f"{model_name}-best",
                type="model"
            )
            artifact.add_file(best_path)
            wandb.log_artifact(artifact)

        if model_ctor is not None and os.path.exists(best_path):
            eval_model = model_ctor()
            load_model(eval_model, best_path, device=device)
            eval_model.to(device)
        else:
            eval_model = model

        metrics = evaluate_topk(
            eval_model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"[best] loss={metrics['loss']:.4f}  "
            f"top1={metrics['top1']:.2f}%  "
            f"top5={metrics['top5']:.2f}%"
        )

        if HAS_WANDB:
            wandb.log({
                "final_loss": metrics["loss"],
                "final_top1": metrics["top1"],
                "final_top5": metrics["top5"],
            })

        results["final_metrics"] = metrics

        return results

    except KeyboardInterrupt:
        print(
            f"\n[!] Training interrupted by user. "
            f"Best checkpoint so far is on disk as "
            f"{model_name}_best.pth."
        )

        return {
            "best_val_accuracy": None,
            "best_epoch": None,
            "history": {},
        }

    finally:
        finish_wandb()

def run_fp32_training(
    model_registry: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config,
    device: torch.device,
    skip_names: Optional[Sequence[str]] = None,
    resume_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, nn.Module], Dict[str, Any]]:
    """
    Run FP32 training for every architecture in `model_registry`.

    skip_names: architectures already fully trained in a previous run —
        their best checkpoint is reloaded from disk instead of retraining.
    resume_names: architectures that were interrupted mid-training — their
        best checkpoint (if any) is loaded before continuing training,
        instead of starting from random initialization again.
    """
    skip_names = set(skip_names or [])
    resume_names = set(resume_names or [])

    fp32_models = {}
    fp32_training_results = {}

    for name, spec in model_registry.items():
        ckpt_path = os.path.join(config["save_dir"], f"{name}_best.pth")

        if name in skip_names:
            print(f"Skipping {name} (already trained) — reloading checkpoint.")
            model = spec["ctor"]()
            load_model(model, ckpt_path, device=str(device))
            fp32_models[name] = model.cpu()
            del model
            torch.cuda.empty_cache()
            continue

        model = spec["ctor"]().to(device)

        if name in resume_names and os.path.exists(ckpt_path):
            print(f"Resuming {name} from {ckpt_path}")
            load_model(model, ckpt_path, device=str(device))
            model.to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=spec["lr"],
            weight_decay=config["weight_decay"],
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs_fp32"],
        )

        results = train_and_evaluate(
            model=model,
            model_name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=config["epochs_fp32"],
            scheduler=scheduler,
            device=device,
            save_dir=config["save_dir"],
            use_amp=config["use_amp"],
            early_stopping_patience=config["early_stopping_patience"],
            model_ctor=spec["ctor"],
        )

        fp32_training_results[name] = results
        fp32_models[name] = model.cpu()

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    return fp32_models, fp32_training_results
def run_qat_training(
    model_registry: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config,
    device: torch.device,
    skip_names: Optional[Sequence[str]] = None,
    resume_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, nn.Module], Dict[str, Any]]:
    """
    Run QAT fine-tuning for every architecture in `model_registry`.

    skip_names: architectures already fully trained in a previous run —
        their best checkpoint is reloaded from disk instead of retraining.
    resume_names: architectures that were interrupted mid-training — their
        best QAT checkpoint (if any) is loaded into the fresh QAT-prepared
        model before continuing training, instead of starting from the
        FP32 weights again.
    """
    skip_names = set(skip_names or [])
    resume_names = set(resume_names or [])

    qat_models = {}
    qat_training_results = {}

    for name in model_registry:
        ckpt_path = os.path.join(config["save_dir"], f"qat_{name}_best.pth")

        if name in skip_names:
            print(f"Skipping {name} (already trained) — reloading checkpoint.")
            qat_model = build_qat(name, save_dir=config["save_dir"], device=device)
            load_model(qat_model, ckpt_path, device=str(device))
            qat_models[name] = qat_model.cpu()
            del qat_model
            torch.cuda.empty_cache()
            continue

        qat_model = build_qat(name, save_dir=config["save_dir"], device=device)

        if name in resume_names and os.path.exists(ckpt_path):
            print(f"Resuming {name} from {ckpt_path}")
            load_model(qat_model, ckpt_path, device=str(device))
            qat_model.to(device)

        optimizer = optim.AdamW(
            qat_model.parameters(),
            lr=config["lr_qat"],
            weight_decay=config["weight_decay"],
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs_qat"],
        )

        results = train_and_evaluate(
            model=qat_model,
            model_name=f"qat_{name}",
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=config["epochs_qat"],
            device=device,
            save_dir=config["save_dir"],
            scheduler=scheduler,
            use_amp=False,
            early_stopping_patience=config["early_stopping_patience"],
        )

        qat_training_results[name] = results
        qat_models[name] = qat_model.cpu()

        del qat_model, optimizer, scheduler
        torch.cuda.empty_cache()

    return qat_models, qat_training_results
# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
class ExperimentConfig:
    """Centralized experiment configuration."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from dictionary."""
        self.config = config_dict

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.config)

    def save(self, path: str) -> None:
        """Save config to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def __repr__(self) -> str:
        return json.dumps(self.config, indent=2)

def build_default_config(seed: int, device: torch.device, save_dir: str) -> ExperimentConfig:
    """
    Centralized experiment configuration for reproducibility.
    """

    config = ExperimentConfig({
        "seed": seed,
        "device": str(device),

        "dataset": "tiny-imagenet-200",
        "img_size": 64,
        "num_classes": 200,
        "train_val_split": 0.9,

        "batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,

        "epochs_fp32": EPOCHS_FP32,
        "epochs_qat": EPOCHS_INT8,

        "lr_pretrained": 1e-4,
        "lr_from_scratch": 3e-4,
        "qat_freeze_bn_epoch": 3,
        "qat_disable_observer_epoch":5,
        "lr_qat":1e-5,

        "label_smoothing": 0.1,
        "weight_decay": 5e-4,
        "use_amp": True,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,

        "save_dir": save_dir,
    })

    os.makedirs(save_dir, exist_ok=True)
    config.save(os.path.join(save_dir, "config.json"))

    return config

# =============================================================================
# RESULTS & REPORTING
# =============================================================================

def create_results_summary(
    results: Dict[str, Any],
    config: Dict[str, Any],
    output_path: str,
    system_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Create comprehensive experiment summary.

    Args:
        results: Dictionary containing experiment results
        config: Experiment configuration
        output_path: Path to save summary
        system_info: System information (optional, auto-detected if None)
    """
    summary = {
        "config": config,
        "system_info": system_info or get_system_info(),
        **results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

def build_comparison_table(rows):
    """Build a sorted comparison DataFrame (by precision, then descending top-1)."""
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["precision", "top1_%"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

# =============================================================================
# WANDB INTEGRATION
# =============================================================================

def init_wandb(
    project: str,
    experiment_name: str,
    config: Dict[str, Any],
    tags: Optional[List[str]] = None,
    offline: bool = True,
) -> None:
    """
    Initialize Weights & Biases experiment tracking.

    Args:
        project: wandb project name
        experiment_name: Experiment/run name
        config: Experiment configuration dictionary
        tags: Optional list of tags for the run
        offline: Run wandb in offline mode (no network calls)
    """
    if not HAS_WANDB:
        print("Warning: wandb not installed. Skipping wandb initialization.")
        return

    wandb.init(
        project=project,
        name=experiment_name,
        config=config,
        tags=tags or [],
        mode="offline" if offline else "online",
    )

    wandb.config.update(get_system_info())

def finish_wandb() -> None:
    if HAS_WANDB:
        wandb.finish()

def log_artifact(file_path: str, artifact_type: str = "model") -> None:
    """Log artifact to wandb."""
    if not HAS_WANDB:
        return

    artifact = wandb.Artifact(
        name=os.path.basename(file_path),
        type=artifact_type
    )
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)

# =============================================================================
# MODEL REGISTRY
# =============================================================================

def register_model(
    name: str,
    ctor,
    fuse_map=None,
    **metadata,
):
    """Register an architecture (ctor + QAT fuse map + metadata) by name."""
    MODEL_REGISTRY[name] = {
        "ctor": ctor,
        "fuse_map": fuse_map or [],
        **metadata,
    }

# =============================================================================
# QAT HELPERS
# =============================================================================

def prepare_qat_model(
    model: nn.Module,
    fuse_pairs: list,
    fuse_root: nn.Module | None = None,
    qengine: str = "fbgemm",
) -> nn.Module:
    """
    Prepare a model for Quantization-Aware Training (QAT).

    This function:
    1. Creates a deep copy of the model to avoid modifying the original.
    2. Sets the model to training mode.
    3. Assigns a QAT quantization configuration for the selected backend.
    4. Fuses specified module pairs (e.g., Conv + ReLU).
    5. Wraps the model with QAT fake-quantization observers.
    """

    # Work on a copy to avoid mutating the original FP32 model
    model = copy.deepcopy(model)

    # QAT requires training mode to simulate inference-time statistics
    model.train()

    # Assign backend-specific QAT configuration
    model.qconfig = tq.get_default_qat_qconfig(qengine)

    # Select fusion root (full model or submodule)
    root = model if fuse_root is None else fuse_root

    # Fuse adjacent layers to match quantized inference kernels
    if fuse_pairs:
        tq.fuse_modules_qat(root, fuse_pairs, inplace=True)

    # Insert fake quantization modules for QAT simulation
    return tq.prepare_qat(model, inplace=False)

def load_best_model(
    arch_name: str,
    ctor,
    save_dir: str,
    device: torch.device,
    eval_mode: bool = True,
) -> nn.Module:
    model = ctor()
    path = os.path.join(save_dir, f"{arch_name}_best.pth")

    load_model(model, path, device=str(device))
    model = model.to(device)

    if eval_mode:
        model.eval()

    return model

def build_qat_from_model(model: nn.Module, arch_name: str, device: torch.device) -> nn.Module:
    spec = MODEL_REGISTRY[arch_name]
    root_attr = spec.get("fuse_root_attr")
    fuse_root = getattr(model, root_attr) if root_attr else model
    qat_model = prepare_qat_model(
        model,
        spec["fuse_map"],
        fuse_root=fuse_root,
    )
    return qat_model.to(device)

def convert_to_int8(qat_model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Convert a trained QAT model into a real INT8 quantized model.
    """

    qat_model = qat_model.to("cpu")
    qat_model.eval()
    return torch.ao.quantization.convert(qat_model, inplace=inplace)

def build_qat(arch_name: str, save_dir: str, device: torch.device) -> nn.Module:
    """
    Load FP32 best model → convert to QAT model.
    """
    spec = MODEL_REGISTRY[arch_name]

    # 1. Load pretrained FP32 model
    model = load_best_model(
        arch_name=arch_name,
        ctor=spec["ctor"],
        save_dir=save_dir,
        device=device,
        eval_mode=False,  # IMPORTANT: QAT must start in train mode
    )

    # 2. Convert to QAT
    qat_model = build_qat_from_model(
        model,
        arch_name,
        device,
    )

    return qat_model.to(device)

# =============================================================================
# SANITY CHECK
# =============================================================================

def sanity_check_models(
    model_ctors,
    img_size: int,
    num_classes: int,
    device: str | torch.device = "cpu",
) -> None:
    """
    Smoke-test one or more model constructors by verifying the output shape.

    Args:
        model_ctors: Iterable of zero-argument model constructors.
        img_size: Input image size.
        num_classes: Expected number of output classes.
        device: Device on which to run the test.
    """
    x = torch.randn(2, 3, img_size, img_size, device=device)

    for ctor in model_ctors:
        model = ctor().to(device)
        model.eval()

        with torch.no_grad():
            y = model(x)

        assert y.shape == (2, num_classes), (
            f"{ctor.__name__}: expected {(2, num_classes)}, got {tuple(y.shape)}"
        )

        print(
            f"{ctor.__name__:25s} OK -> {tuple(y.shape)}, "
            f"params={count_parameters(model)/1e6:.2f}M"
        )