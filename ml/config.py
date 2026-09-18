from dataclasses import dataclass
from typing import Optional

DEFAULT_EPOCHS = 100
DEFAULT_QAT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 5e-4  # must match configs/training.yaml, which always overrides this
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
    warmup_epochs: int = 0       # linear LR warmup before cosine decay; 0 disables
    reset_scheduler_on_resume: bool = False  # skip the checkpoint's stale scheduler state
                                              # (wrong T_max) when resuming into a bigger epochs budget


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
class QATWinoConfig(QATConfig):
    """QATConfig + o que descreve CONTRA QUAL acelerador se treina.

    O estágio `qat_wino` não é "QAT genérico": ele troca as convs 3x3 pela
    numérica exata de uma linha de hardware do repo irmão Winograd-FPGA. Qual
    linha (`variant`) e se ela roda com 2 multiplicações por DSP (`pack`) mudam
    o resultado, então são CONFIG — não constante escondida na ponte.

    Até 2026-09-18 o bridge chamava `convert(model)` sem argumento nenhum, o que
    fixava F(4,3) sem packing; por isso as 14 medições de acurácia existentes são
    todas dessa combinação, e saem marcadas `≠HW` nas tabelas do artigo (o
    bitstream de deploy roda `--pack 1`).

    Os defaults abaixo reproduzem aquelas corridas. Mude-os pelo bloco
    `qat_wino:` do YAML do experimento, que é o mecanismo que o resto do repo já
    usa.
    """
    variant: str = "f43"      # f23 | f43 | f63
    pack: bool = False        # 2 mult/DSP, com requantização por posição
    u_w: int = 9              # largura do campo de U dentro do DSP
    v_w: int = 8              # largura do campo de V
    k_dsp: int = 2            # acumulações dentro do DSP (não muda a aritmética)


@dataclass
class DetSegDataConfig:
    img_size: int = 256
    voc_root: str = ""  # set at runtime, e.g. ~/.cache/torchvision/datasets/voc
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
