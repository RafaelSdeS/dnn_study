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

**Notebooks (in order of execution):**
1. `notebooks/baselines_qat.ipynb` — Phase 1: Reference pretrained models (ResNet18, MobileNetV2, AlexNet, VGG-style)
2. `notebooks/alexnet_qat.ipynb` — Phase 2: AlexNet kernel-restriction variants (3×3, 2×2, stacked, mixed, small-kernel)
3. `notebooks/compensation_qat.ipynb` — Phase 3a: Compensation mechanisms (bottleneck, residual, factorized, etc.)
4. `notebooks/tinyhybridnet_qat.ipynb` — Phase 3b: Efficient hybrids (TinyHybridNet, TinyMobileNetV2)
5. `notebooks/results_analysis.ipynb` — Cross-phase results analysis and figure generation (5 visualizations: FP32 accuracy, FP32 vs INT8, accuracy vs size, accuracy vs MACs, **efficiency per MB**, plus training curves)

---

## Results & Recommendations

**See `ideas/BEST_MODELS.md`** for:
- Overall accuracy rankings (FP32 + INT8)
- Analysis by dimension (accuracy-first, efficiency-first, quantization stability, kernel impact, compensation mechanisms)
- Tiered model recommendations (Tier 1 production-ready, Tier 2 high-accuracy, Tier 3 exploratory)
- Next steps for Phase 3.5 (deployment fine-tuning, Winograd benchmarking)

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

## Key Findings (Phases 1–3)

**See `ideas/BEST_MODELS.md` for comprehensive ranking, Pareto analysis, and deployment recommendations.**

### Phase 1 — Reference Baselines (FP32, ~80 epochs):

| Model | Params | FP32 Top-1 | FP32 Top-5 |
|-------|--------|-----------|-----------|
| **MobileNetV2** | 2.48M | **57.99%** | 81.51% |
| **ResNet18** | 11.28M | **53.91%** | 77.80% |
| VGGStyle | 2.41M | 51.81% | 75.88% |
| AlexNetTV | 57.82M | 32.89% | 58.22% |

### Phase 2 — AlexNet Kernel Restriction (FP32 + QAT INT8, 57+20 epochs):

| Model | Params | FP32 Top-1 | INT8 Top-1 | QAT Drop | INT8 Size |
|-------|--------|-----------|-----------|----------|-----------|
| **AlexNetSmallKernel** | 1.60M | 45.84% | 35.95% | –9.89pp ⚠️ | 1.56 MB |
| **AlexNetStacked** | 60.48M | 44.56% | 42.79% | –1.77pp | 57.94 MB |
| AlexNetMixed | 1.75M | 38.74% | 37.99% | –0.75pp | 1.71 MB |
| AlexNet3x3 | 57.61M | 35.79% | 36.19% | +0.40pp ✓ | 55.12 MB |
| AlexNet2x2 | 1.05M | 30.02% | 30.89% | +0.87pp ✓ | 1.04 MB |

### Phase 3 — Compensation Mechanisms (FP32 + QAT INT8, 41–95+20 epochs):

| Model | Params | FP32 Top-1 | INT8 Top-1 | QAT Drop | Efficiency | Tier |
|-------|--------|-----------|-----------|----------|------------|------|
| **AlexNetBottleneck** | 0.39M | 44.62% | 44.54% | –0.08pp ✓✓ | 9.93 Acc/MB | **1** |
| **AlexNetFire** | 0.52M | 43.98% | 44.30% | +0.33pp ✓✓ | 7.34 Acc/MB | **1** |
| **AlexNetResidual** | 60.67M | 48.01% | 47.27% | –0.74pp | — | **2** |
| AlexNetDepthwiseSep | 0.31M | 44.39% | 41.47% | –2.92pp ⚠️ | 12.15 Acc/MB | 3 |
| AlexNetFactorized | 57.07M | 42.89% | 42.60% | –0.29pp | — | 3 |

**Key insights:**
- **Baselines dominate pure accuracy:** MobileNetV2 (57.99%), ResNet18 (53.91%) are hard to beat
- **Tier 1 models (Bottleneck, Fire) are Pareto-optimal:** Tiny (4–6 MB), competitive accuracy (43–44%), quantization-stable
- **Small kernels face QAT challenges:** AlexNetSmallKernel's –9.89pp drop suggests aggressive quantization sensitivity
- **Compensation mechanisms work:** Bottleneck & Fire achieve high efficiency (7–10 Acc/MB) while maintaining accuracy
- **Model efficiency summary:** Results in `results/results.csv` and per-model details in `results/model_details.csv`
- **Visualizations:** Cross-phase figures in `results/figures/` (6 PNG plots covering accuracy, efficiency, latency, training curves)

---

## Running Experiments

**See `CLAUDE.md` for:**
- Reproducibility patterns (seed, cudnn settings)
- Trainer API examples
- Model registry usage
- Checkpoint & config workflows
- INT8 inference requirements
