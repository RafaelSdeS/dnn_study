from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


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