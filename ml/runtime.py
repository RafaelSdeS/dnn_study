from __future__ import annotations

import os
import random
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    # live under <root>/<experiment>/<model>/ and are made by scripts.train._make_model_runs
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
    """Git hash, dirty flag, hostname, UTC timestamp, torch version — for run provenance records."""
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