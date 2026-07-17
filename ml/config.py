from dataclasses import dataclass
from typing import Optional

DEFAULT_EPOCHS = 100
DEFAULT_QAT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 4e-4
DEFAULT_EARLY_STOPPING_PATIENCE = 5


@dataclass
class DataConfig:
    dataset_path: str = ""       # set at runtime via kagglehub
    img_size: int = 64
    num_classes: int = 200
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    train_val_split: float = 0.9
    seed: int = 42


@dataclass
class TrainerConfig:
    # ── core ─────────────────────────────────────────────────────────
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LEARNING_RATE            # override per-model via register_model(lr=...)
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    # ── regularisation ───────────────────────────────────────────────
    label_smoothing: float = 0.1
    grad_clip_norm: Optional[float] = None
    # ── training options ─────────────────────────────────────────────
    use_amp: bool = True         # set False for QAT fine-tuning
    early_stopping_patience: Optional[int] = DEFAULT_EARLY_STOPPING_PATIENCE


@dataclass
class QATConfig:
    # ── fine-tuning schedule ─────────────────────────────────────────
    epochs: int = DEFAULT_QAT_EPOCHS
    lr: float = 1e-5
    weight_decay: float = 5e-4
    # ── observer / BN freeze schedule ────────────────────────────────
    freeze_bn_epoch: int = 3     # freeze BN running stats after this epoch
    disable_observer_epoch: int = 5  # disable fake-quant observers after this


@dataclass
class DetSegDataConfig:
    img_size: int = 256
    voc_root: str = ""  # set at runtime, e.g. ~/.cache/torchvision/datasets/voc
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
