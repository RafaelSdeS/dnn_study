# CNN Kernel Research: Winograd Acceleration & Efficiency

This repository investigates how convolutional kernel-size restrictions affect deep neural network accuracy, efficiency, and quantization robustness. The motivation is **Winograd-accelerated convolution**, which achieves high computational efficiency for small kernels (2×2, 3×3) but faces scalability limitations for larger filters.

**Current scope:** Classification on Tiny ImageNet-200 (64×64 RGB, 200 classes) with FP32 training, Quantization-Aware Training (QAT), and INT8 conversion for edge deployment.

---

## Quick Start

**Activate environment:**
```bash
source .venv/bin/activate
```

**Launch notebooks:**
```bash
jupyter lab
```

**Notebooks (in order of scope):**
1. `notebooks/baselines_qat.ipynb` — Phase 1: Reference pretrained models (ResNet18, MobileNetV2, AlexNet, VGG-style)
2. `notebooks/alexnet_qat.ipynb` — Phase 2: AlexNet kernel-restriction variants (3×3, 2×2, stacked, mixed, small-kernel)
3. `notebooks/compensation_qat.ipynb` — Phase 3a: Compensation mechanisms (bottleneck, residual, factorized, etc.)
4. `notebooks/tinyhybridnet_qat.ipynb` — Phase 3b: Efficient hybrids (TinyHybridNet, TinyMobileNetV2)
5. `notebooks/results_analysis.ipynb` — Cross-phase results analysis and figure generation

---

## Architecture & Design

### `ml/` — Core ML Package (Authority)

All notebooks import from `ml/`:

- **`config.py`** — `DataConfig`, `TrainerConfig`, `QATConfig` dataclasses with explicit defaults
- **`data.py`** — `create_imagenet_loaders()` with deterministic train/val split
- **`trainer.py`** — `Trainer` class: `fit()`, `evaluate()`, checkpoint management
- **`quantization.py`** — `build_qat()`, `prepare_qat_model()`, `convert_to_int8()`, `find_fuse_groups()`
- **`registry.py`** — `MODEL_REGISTRY`, `register_model()`
- **`checkpoint.py`** — `save_checkpoint()`, `load_checkpoint()`
- **`reporting.py`** — `build_comparison_table()`, `create_results_summary()`, result formatting

### `models/` — Experimental Phases

| Phase | File | Purpose |
|-------|------|---------|
| 1 — Reference | `baselines.py` | Pretrained models, baseline architectures |
| 2 — Kernel Restriction | `alexnet_variants.py` | AlexNet variants with kernel constraints (3×3, 2×2, stacked, mixed, small-kernel) |
| 3a — Compensation | `compensation.py` | Architectural compensations for small kernels (bottleneck, residual, SE, etc.) |
| 3b — Efficient Hybrids | `tinyhybridnet.py` | TinyHybridNet, TinyMobileNetV2 (MobileNet-style efficient CNNs) |
| 4+ — Future | — | See `TODO.md` |

### `configs/` — Hyperparameter Defaults

YAML-based configuration loading via `configs/loader.py`:
- `data.yaml` — Dataset, augmentation, normalization
- `training.yaml` — Epochs, batch size, learning rate
- `qat.yaml` — QAT-specific settings
- `models/` — Per-architecture learning rate overrides

---

## QAT Workflow

```
1. FP32 Train     → Trainer.fit() → saves {arch}_best.pth
2. Build QAT      → load_best_model() + fuse Conv-BN-ReLU + prepare_qat()
3. QAT Train      → Trainer.fit(epoch_callback=make_qat_callback(...)) → saves qat_{arch}_best.pth
4. Convert INT8   → convert_to_int8(qat_model) on CPU
5. Evaluate       → Trainer.evaluate(topk=(1,5)) on INT8 model
```

**Key requirements:**
- All ReLU: `inplace=False` (fake-quant hooks cannot observe in-place ops)
- Skip-adds: `torch.nn.quantized.FloatFunctional()` (not bare `+`)
- INT8 inference: CPU-only (fbgemm limitation)

---

## Key Findings (Phase 1 & 2)

**Phase 1 — Reference baselines** (FP32):

| Model | Params | FP32 Top-1 | FP32 Top-5 |
|-------|--------|-----------|-----------|
| ResNet18 (pretrained) | 11.28M | 46.95% | 73.50% |
| MobileNetV2 (pretrained) | 2.48M | 45.81% | 73.00% |
| AlexNet (pretrained) | 57.82M | 19.63% | 45.43% |
| VGGStyleCNN | 2.41M | 10.64% | 30.42% |

**Phase 2 — AlexNet kernel-restriction variants** (FP32 + INT8, 57 epochs):

| Model | Params | FP32 Top-1 | INT8 Top-1 | INT8 Size |
|-------|--------|-----------|-----------|-----------|
| AlexNetSmallKernel | 1.60M | 45.84% | 35.95% | 1.56 MB |
| AlexNetStacked | 60.48M | 44.56% | 42.79% | 57.94 MB |
| AlexNetMixed | 1.75M | 38.74% | 38.00% | 1.71 MB |
| AlexNet3x3 | 57.61M | 35.79% | 36.19% | 55.12 MB |
| AlexNet2x2 | 1.05M | 30.02% | 30.89% | 1.04 MB |

**Key insights:**
- AlexNetSmallKernel matches MobileNetV2 accuracy at 1.6M params and 1.56 MB INT8 — strong Winograd candidate
- Mixed kernels offer the best accuracy/size trade-off among compact models (38.74% FP32 at 1.71 MB INT8)
- Quantization drop is high for AlexNetSmallKernel (–9.9pp) but acceptable for stacked/mixed/3×3 variants

---

## Running Experiments

**See `CLAUDE.md` for:**
- Reproducibility patterns (seed, cudnn settings)
- Trainer API examples
- Model registry usage
- Checkpoint & config workflows
- INT8 inference requirements
