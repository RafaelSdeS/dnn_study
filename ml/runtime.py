from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from configs.loader import load_config


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    logs: Path
    aggregates: Path


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def expand_path(value: str | None, default: str | Path | None = None) -> Path | None:
    raw = value if value not in {None, ""} else default
    if raw in {None, ""}:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve()


def build_runtime_paths(root: str | Path) -> RuntimePaths:
    base = Path(root).resolve()
    # ponytail: only these two are read; per-model checkpoints/tensorboard dirs
    # live under <root>/<experiment>/<model>/ and are made by make_model_runs below
    paths = RuntimePaths(
        root=base,
        logs=base / "logs",
        aggregates=base / "aggregates",
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_runtime_root(runtime_name: str) -> Path:
    """Resolve a runtime profile's output root (configs/runtime/<name>.yaml's `root` field)."""
    cfg = load_config(f"runtime/{runtime_name}.yaml")
    return expand_path(cfg.get("root"), default=f"outputs/{runtime_name}")


def capture_provenance() -> dict[str, Any]:
    """Git hash, dirty flag, hostname, UTC timestamp, and the training environment (torch/
    torchvision/CUDA/cuDNN versions, GPU name, Python, CPU count, SLURM job id) -- so a run is
    still reproducible months later without re-deriving what it ran on."""
    import torchvision

    try:
        git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_hash = "unknown"
    try:
        git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        git_dirty = False
    return {
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def resolve_dataset_train_path(dataset_root: str | Path | None) -> Path | None:
    if dataset_root in {None, ""}:
        return None
    root = Path(os.path.expandvars(os.path.expanduser(str(dataset_root)))).resolve()
    if not root.exists():
        return None
    if (root / "tiny-imagenet-200" / "train").exists():
        return root / "tiny-imagenet-200" / "train"
    if (root / "train").exists():
        return root / "train"
    return root


def load_profile(name_or_path: str, subdir: str) -> dict[str, Any]:
    """configs/<subdir>/<name>.yaml (via load_config, so `extends:` works), or the file
    itself when name_or_path is an existing path."""
    path = Path(name_or_path)
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return load_config(f"{subdir}/{name_or_path}.yaml")


def ensure_dataset_path(runtime_cfg: dict[str, Any]) -> Path:
    """Tiny ImageNet's train/ dir from the runtime profile's dataset_root, else a KaggleHub download."""
    dataset_root = expand_path(runtime_cfg.get("dataset_root"))
    dataset_train = resolve_dataset_train_path(dataset_root) if dataset_root else None
    if dataset_train is not None:
        return dataset_train

    if not runtime_cfg.get("use_kagglehub_fallback", True):
        raise FileNotFoundError("dataset_root is unset and KaggleHub fallback is disabled")

    import kagglehub

    dataset_path = kagglehub.dataset_download(runtime_cfg.get("kaggle_dataset", "akash2sharma/tiny-imagenet"))
    return Path(dataset_path) / "tiny-imagenet-200" / "train"


def make_model_runs(root: Path, experiment_name: str, model_name: str) -> tuple[Path, Path, Path, Path, Path]:
    """Create <root>/<experiment>/<model>/{checkpoints,logs,tensorboard,results}; returns (run_root, *those)."""
    run_root = root / experiment_name / model_name
    checkpoints = run_root / "checkpoints"
    logs = run_root / "logs"
    tensorboard = run_root / "tensorboard"
    results = run_root / "results"
    for path in (checkpoints, logs, tensorboard, results):
        path.mkdir(parents=True, exist_ok=True)
    return run_root, checkpoints, logs, tensorboard, results


def save_resolved_config(run_root: Path, config: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "resolved_config.json").write_text(json.dumps(config, indent=2, default=str))