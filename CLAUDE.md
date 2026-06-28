# CLAUDE.md — alexnet_rafael

## Default workflow

For any non-trivial coding request:

1. Inspect the relevant files.
2. Produce a short implementation plan.
3. Wait for approval before modifying code.

Do not immediately edit files.

## Project Overview

This is a deep-learning research project investigating how CNN kernel size affects accuracy, efficiency, and quantization performance on **Tiny ImageNet-200** (64×64 RGB images, 200 classes). The core goal is to compare classical (AlexNet-style) and modern efficient architectures (MobileNet-style) under FP32 training and **Quantization-Aware Training (QAT)** → INT8 conversion for edge deployment.

Research questions:
- How do 3×3 vs. larger kernels affect accuracy and quantization?
- Can small, efficient CNNs match pretrained models at a fraction of the size?
- What is the FP32 → INT8 accuracy drop for different architecture families?

---

## Repository Structure

```
alexnet_rafael/
├── ml/                      # Core ML package — import everything from here
│   ├── __init__.py          # Public re-exports
│   ├── config.py            # DataConfig, TrainerConfig, QATConfig dataclasses (all hyperparameters with defaults)
│   ├── data.py              # create_imagenet_loaders(cfg: DataConfig)
│   ├── checkpoint.py        # save_checkpoint / load_checkpoint
│   ├── registry.py          # MODEL_REGISTRY dict + register_model()
│   ├── trainer.py           # Trainer class: fit(), evaluate(), _train_one_epoch(), _validate()
│   ├── quantization.py      # find_fuse_groups, prepare_qat_model, build_qat,
│   │                        #   convert_to_int8, load_best_model, make_qat_callback
│   └── reporting.py         # build_comparison_table(), create_results_summary(), disk_mb()
├── models/                  # Model architecture definitions (five experimental phases)
│   ├── __init__.py          # Re-exports all public constructors
│   ├── baselines.py         # Stage 1: AlexNetTV, StrongCNN, VGGStyleCNN, ResNet18TV,
│   │                        #   FireMobileResidual, TinyHybridNet
│   ├── alexnet_variants.py  # Stage 2: AlexNet3x3, AlexNetSmallKernel, AlexNetStacked,
│   │                        #   AlexNetFactorized, AlexNetBottleneck, AlexNetResidual
│   ├── efficient_cnns.py    # Stage 3: InvertedResidual, TinyMobileNetV2, MobileNetV2TV,
│   │                        #   ShuffleNetV2TV, EfficientNetB0TV, ConvNeXtTinyTV
│   └── hybrids.py           # H1–H10: ResidualVGGHybrid, AlexNetInceptionHybrid,
│                            #   MobileNetResidualHybrid, FireResidualHybrid, FactorizedAlexNet,
│                            #   LargeSmallKernelHybrid, ShuffleResidualHybrid,
│                            #   ConvNeXtLiteHybrid, KernelConstraintNetwork,
│                            #   AttentionAugmentedCNN
├── configs/                 # Hyperparameter YAML files — loaded via configs/loader.py
│   ├── loader.py            # load_config(filename) → dict (searches configs/ dir)
│   ├── data.yaml            # DataConfig defaults
│   ├── training.yaml        # TrainerConfig defaults
│   ├── qat.yaml             # QATConfig defaults
│   └── models/              # Per-model lr overrides (loaded and applied in notebooks)
│       ├── alexnet_fp32.yaml
│       ├── alexnet_3x3.yaml
│       ├── alexnet_small_kernel.yaml
│       ├── tinyhybridnet.yaml
│       └── tinymobilenetv2.yaml
├── notebooks/               # Experiment notebooks
│   ├── alexnet_qat.ipynb                    # AlexNet family FP32 + QAT
│   ├── tinyhybridnet_qat.ipynb              # TinyHybridNet & TinyMobileNetV2 FP32 + QAT
│   └── alexnet_qat_wandb_results_analysis.ipynb  # W&B offline sync + plotting
├── results/                 # Output CSVs and JSON summaries (git-ignored)
├── README.md
├── AGENTS.md
├── TODO.md
└── .venv/                   # Python 3.12.3 virtual environment (pip, CUDA 12.1)
```

Checkpoint artifacts (git-ignored, created at runtime):
- `checkpoints/` — per-notebook checkpoint storage (`{arch}_best.pth`, `qat_{arch}_best.pth`, `{arch}.pth` for INT8)

---

## Key Technologies

| Tool | Version | Role |
|------|---------|------|
| Python | 3.12.3 | Runtime |
| PyTorch | 2.5.1+cu121 | Deep learning framework |
| torchvision | 0.20.1+cu121 | Datasets, pretrained models, transforms |
| torchmetrics | 1.9.0 | Top-k accuracy in `Trainer.evaluate()` |
| torchinfo | — | Model summary (`torchinfo.summary()`) |
| W&B (wandb) | 0.28.0 | Experiment tracking (offline-first) |
| kagglehub | — | Tiny ImageNet download |
| JupyterLab | 4.5.7 | Notebook environment |
| numpy / pandas | — | Numerics and results tables |
| matplotlib / seaborn / plotly | — | Visualization |
| optuna | 4.8.0 | Hyperparameter optimization (not yet wired up) |
| CUDA | 12.1 | GPU compute |
| Hardware | RTX 4060 Laptop (8.2 GB) | Training target |

Quantization backend: **fbgemm** (CPU-optimized, set via `torch.backends.quantized.engine = "fbgemm"`).

---

## Architectural Patterns & Design Decisions

### 1. `ml/` is the Authority

All notebooks import from the `ml/` package. `ml_utils.py` has been deleted. When extending training logic, edit the relevant module in `ml/`.

### 2. Hyperparameter Visibility

All hyperparameters live in typed `@dataclass` fields with explicit defaults. Configs are loaded from `configs/*.yaml` at notebook startup:

```python
from configs.loader import load_config
data_cfg  = DataConfig(**load_config("data.yaml"))
fp32_cfg  = TrainerConfig(**load_config("training.yaml"))
qat_cfg   = QATConfig(**load_config("qat.yaml"))
data_cfg.seed = SEED  # tie split seed to notebook-level constant
```

Quick one-off override (e.g., to test with 2 epochs):
```python
fp32_cfg = replace(fp32_cfg, epochs=2)
```

Per-model lr overrides live in `configs/models/{arch}.yaml` and are applied in the notebook loop via `replace(fp32_cfg, lr=spec["lr"])`. Use `dataclasses.replace()` to override any field without mutating the base config.

### 3. QAT Workflow

```
FP32 train (Trainer.fit() → saves {arch}_best.pth)
    ↓
build_qat(arch_name, save_dir, device)
    → load_best_model() → deep copy → fuse Conv-BN(-ReLU) → prepare_qat()
    ↓
Trainer.fit(epoch_callback=make_qat_callback(freeze_bn_epoch, disable_observer_epoch))
    - epoch == freeze_bn_epoch  → freeze_bn_stats
    - epoch == disable_observer_epoch → disable_observer
    → saves qat_{arch}_best.pth
    ↓
convert_to_int8(qat_model)  — CPU-only, eval() mode required
    ↓
Trainer.evaluate(topk=(1,5))  on CPU val loader
```

### 4. Model Registry

```python
from ml import register_model, MODEL_REGISTRY

register_model("alexnet_fp32", build_alexnet, fuse_map=[...], fuse_root_attr="features", lr=1e-4)
# MODEL_REGISTRY[name] = {"ctor": ..., "fuse_map": ..., "fuse_root_attr": ..., "lr": ...}
```

`fuse_map` is a list of dotted-path lists used to fuse Conv-BN-ReLU. For flat architectures, supply it manually. For nested blocks (TinyHybridNet, TinyMobileNetV2), use `find_fuse_groups(model())` to auto-detect.

### 5. Trainer Usage

```python
from ml import Trainer
from dataclasses import replace

# FP32 — override lr per model from registry
trainer = Trainer(model, train_loader, val_loader,
                  cfg=replace(fp32_cfg, lr=spec["lr"]),
                  device=device, save_dir=SAVE_DIR, run_name=name,
                  num_classes=200)
results = trainer.fit()           # → {"best_val_accuracy": ..., "best_epoch": ..., "history": {...}}
metrics = trainer.evaluate(topk=(1, 5))  # → {"top1": ..., "top5": ..., "loss": ...}

# QAT — use_amp=False, pass epoch_callback
qat_cfg = replace(fp32_cfg, epochs=20, lr=1e-5, use_amp=False)
cb = make_qat_callback(freeze_bn_epoch=3, disable_observer_epoch=5)
trainer = Trainer(qat_model, ..., cfg=qat_cfg, epoch_callback=cb)
trainer.fit()
```

Resume interrupted training with `trainer.fit(resume_from=SAVE_DIR / f"{name}_best.pth")`.

Skip/resume logic lives in the notebook loop — not hidden inside a wrapper function.

### 6. QAT-Compatible Architecture Rules

- All ReLU must be `inplace=False` — fake-quant hooks cannot observe in-place ops.
- Residual skip-adds must use `torch.nn.quantized.FloatFunctional()` not `+` or `torch.add`.
- BN layers that are fused must appear immediately after Conv in the module tree.
- INT8 conversion and INT8 inference must run on **CPU** (fbgemm limitation).

### 7. Module Fusion Patterns (by arch family)

- **AlexNet (flat sequential):** Hand-written index-based fuse maps, e.g., `[["0","1"],["3","4"],...]`
- **TinyHybridNet / TinyMobileNetV2 (nested blocks):** `find_fuse_groups(model())` walks the tree recursively and returns dotted-path lists automatically.
- **New architectures:** Call `find_fuse_groups` or write a fuse map and pass it to `register_model`.

### 8. Data Pipeline

- Dataset: Tiny ImageNet-200, loaded via `torchvision.datasets.ImageFolder`
- Deterministic 90/10 train/val split using `torch.Generator` seeded at `seed=42`
- Normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (ImageNet stats)
- Training augmentation: `RandomResizedCrop(scale=0.7-1.0)`, `RandomHorizontalFlip`, `RandomRotation(15°)`, `AutoAugment(ImageNet policy)`
- Validation: `Resize(img_size) → CenterCrop(img_size)`

---

## Coding Conventions

- **Reproducibility first:** Seed everything at the top of each notebook (`random`, `numpy`, `torch`, `torch.cuda`); set `cudnn.benchmark = True`.
- **Config pattern:** Instantiate `DataConfig`, `TrainerConfig`, `QATConfig` directly in the notebook — all defaults are explicit. Use `dataclasses.replace()` to override per-model.
- **Checkpointing:** `save_checkpoint` / `load_checkpoint` in `ml/checkpoint.py` store full training state `{model, optimizer, scheduler, epoch, metrics}`.
- **W&B logging:** Call `wandb.init(project=..., config=dataclasses.asdict(cfg), mode="offline")` directly. No wrapper. Sync later with `wandb sync --sync-all`.
- **No in-place ReLU** in any QAT model (`inplace=False` everywhere).
- **Quantization engine** must be set before any QAT operation: `torch.backends.quantized.engine = "fbgemm"`.
- **Model summary:** Use `torchinfo.summary(model, input_size=(1,3,64,64))` instead of manual `count_parameters`.

---

## Module Interactions

```
Notebook
  ├─ from configs.loader import load_config
  ├─ from ml import DataConfig, TrainerConfig, QATConfig
  │                 create_imagenet_loaders
  │                 MODEL_REGISTRY, register_model
  │                 Trainer, make_qat_callback
  │                 build_qat, load_best_model, convert_to_int8
  │                 build_comparison_table, create_results_summary, disk_mb
  ├─ from models import AlexNetTV, AlexNet3x3, ...  (pick from baselines/alexnet_variants/efficient_cnns/hybrids)
  ├─ register_model() → MODEL_REGISTRY
  ├─ for name, spec in MODEL_REGISTRY.items():
  │     Trainer(..., cfg=replace(fp32_cfg, lr=spec["lr"])).fit()
  ├─ for name in MODEL_REGISTRY:
  │     build_qat(name, SAVE_DIR, device) → Trainer(..., epoch_callback=cb).fit()
  │     convert_to_int8(qat_model) → Trainer.evaluate(topk=(1,5))
  └─ build_comparison_table(rows) → results/alexnet_qat/final_comparison.csv
     create_results_summary(results, config, output_path) → results/alexnet_qat/experiment_summary.json
```

---

## Model Inventory

Five experimental phases (see `TODO.md`), all in `models/`:

| Phase | File | Models |
|-------|------|--------|
| 1 — Reference Architectures | `baselines.py` | AlexNetTV (pretrained), StrongCNN, VGGStyleCNN, ResNet18TV (pretrained), MobileNetV2TV (pretrained), TinyHybridNet |
| 2 — Kernel Restriction Study | `alexnet_variants.py` | AlexNet3x3, AlexNetSmallKernel, AlexNetStacked, AlexNetFactorized, AlexNetBottleneck, AlexNetResidual |
| 3 — Compensation Mechanisms | `efficient_cnns.py` + `hybrids.py` | TinyMobileNetV2, ShuffleNetV2TV (pretrained), EfficientNetB0TV (pretrained), ConvNeXtTinyTV (pretrained); H1–H10: ResidualVGGHybrid, AlexNetInceptionHybrid, MobileNetResidualHybrid, FireResidualHybrid, FactorizedAlexNet, LargeSmallKernelHybrid, ShuffleResidualHybrid, ConvNeXtLiteHybrid, KernelConstraintNetwork, AttentionAugmentedCNN |
| 4 — Final Architecture | TBD | Design pending — combines best ideas from phases 1–3 |
| 5 — Final Analysis | — | FP32/INT8 rankings, latency, model size, Winograd, conclusions |

---

## Experimental Results (Reference)

AlexNet family (from `notebooks/alexnet_qat.ipynb`, 10 FP32 epochs + 2 QAT epochs):

| Model | Params | FP32 Top-1 | INT8 Top-1 | FP32 Top-5 | INT8 Top-5 | FP32 Size | INT8 Size |
|-------|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| AlexNet (pretrained) | 57.82M | 27.30% | 30.22% | 53.65% | 56.78% | 661.75 MB | 55.33 MB |
| AlexNet3x3 | 57.61M | 6.82% | 7.93% | 21.20% | 23.48% | 659.26 MB | 55.12 MB |
| AlexNetSmallKernel | 1.60M | 8.68% | 9.53% | 25.36% | 27.48% | 18.35 MB | 1.56 MB |

Key findings:
- Residual connections are essential for from-scratch 3×3 CNNs.
- INT8 for pretrained AlexNet improves top-1 (+2.9%), suggesting mild FP32 overfit / quantization regularization.
- AlexNetSmallKernel: 1.6M params, 1.56 MB INT8 — best efficiency in the AlexNet family.
- TinyHybridNet achieves <1% INT8 accuracy drop at only 0.31 MB — best quantization efficiency overall.

---

## Running the Project

**Environment activation:**
```bash
source .venv/bin/activate
```

**Launch notebooks:**
```bash
jupyter lab
```

**Dataset:** Tiny ImageNet-200 is downloaded via `kagglehub` on first run. Cached at `~/.cache/kagglehub/` (git-ignored).

**Checkpoint locations:** Each notebook sets its own `SAVE_DIR`. Best FP32 model: `{SAVE_DIR}/{arch}_best.pth`. Best QAT model: `{SAVE_DIR}/qat_{arch}_best.pth`. INT8 state dict: `{SAVE_DIR}/{arch}.pth`.

**Sync W&B offline runs:**
```bash
wandb sync --sync-all
```

**INT8 inference note:** Always call `model.eval()` and move model to CPU before `torch.ao.quantization.convert()` and before INT8 forward passes.
