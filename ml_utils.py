"""
Reusable utilities for ML research notebooks.
Provides reproducibility, training loops, evaluation, checkpointing, and logging utilities.
"""

import os
import time
import random
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Callable, List
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================================
# REPRODUCIBILITY & SYSTEM UTILITIES
# ============================================================================

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


# ============================================================================
# CHECKPOINTING & MODEL MANAGEMENT
# ============================================================================

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


# ============================================================================
# MODEL ANALYSIS
# ============================================================================

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


# ============================================================================
# TRAINING LOOP
# ============================================================================

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
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
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
    wandb_log: bool = False,
    max_batches: Optional[int] = None,
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
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, use_amp=use_amp, max_batches=max_batches
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
    
    return {
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "history": metrics.get_history(),
    }


# ============================================================================
# EXPERIMENT CONFIGURATION & LOGGING
# ============================================================================

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


# ============================================================================
# VISUALIZATION & REPORTING
# ============================================================================

def create_results_summary(
    model: nn.Module,
    results: Dict[str, Any],
    config: Dict[str, Any],
    system_info: Dict[str, Any],
    output_path: str = "experiment_summary.json"
) -> None:
    """
    Create comprehensive experiment summary.
    
    Args:
        model: Trained model
        results: Training results dictionary
        config: Experiment configuration
        system_info: System information
        output_path: Path to save summary
    """
    summary = {
        "config": config,
        "system_info": system_info,
        "model_info": model_summary(model),
        "results": results,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)


# ============================================================================
# WANDB INTEGRATION
# ============================================================================

def init_wandb(
    project: str,
    experiment_name: str,
    config: Dict[str, Any],
    tags: Optional[List[str]] = None,
) -> None:
    """
    Initialize Weights & Biases experiment tracking.
    
    Args:
        project: wandb project name
        experiment_name: Experiment/run name
        config: Experiment configuration dictionary
        tags: Optional list of tags for the run
    """
    if not HAS_WANDB:
        print("Warning: wandb not installed. Skipping wandb initialization.")
        return
    
    wandb.init(
        project=project,
        name=experiment_name,
        config=config,
        tags=tags or [],
    )


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
