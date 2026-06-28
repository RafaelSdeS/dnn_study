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
1. `notebooks/alexnet_qat.ipynb` — AlexNet family (classical, 3×3, small-kernel variants) under FP32 and QAT
2. `notebooks/tinyhybridnet_qat.ipynb` — Hybrid efficient architectures (TinyHybridNet, TinyMobileNetV2) under FP32 and QAT
3. `notebooks/baselines_qat.ipynb` — Reference pretrained models (ResNet18, MobileNetV2, EfficientNetB0, ConvNeXt)

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

### `models/` — Five Experimental Phases

| Phase | File | Purpose |
|-------|------|---------|
| 1 — Reference | `baselines.py` | Pretrained models, baseline architectures |
| 2 — Classical Restriction | `alexnet_variants.py` | AlexNet variants with kernel constraints |
| 3 — Modern Efficient | `efficient_cnns.py` + `hybrids.py` | MobileNet, ShuffleNet, EfficientNet, ConvNeXt, H1–H10 hybrids |
| 4 — Synthesis | — | Final optimized architecture (TBD) |
| 5 — Analysis | — | Ranking, latency, architecture recommendations |

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

## Key Findings (Reference)

| Model | Params | FP32 Top-1 | INT8 Top-1 | FP32 Size | INT8 Size |
|-------|--------|-----------|-----------|-----------|-----------|
| AlexNet (pretrained) | 57.82M | 27.30% | 30.22% | 661.75 MB | 55.33 MB |
| AlexNetSmallKernel | 1.60M | 8.68% | 9.53% | 18.35 MB | 1.56 MB |
| TinyHybridNet | 0.21M | 51.15% | 50.45% | 2.46 MB | 0.31 MB |

**Key insights:**
- Small kernels + residuals achieve reasonable accuracy at 1.6M params (1.56 MB INT8)
- Hybrid architectures (MobileNet-style + residuals) best balance efficiency and quantization robustness
- TinyHybridNet: <1% INT8 accuracy drop at 0.31 MB — best quantization efficiency

---

## Running Experiments

**See `CLAUDE.md` for:**
- Reproducibility patterns (seed, cudnn settings)
- Trainer API examples
- Model registry usage
- Checkpoint & config workflows
- INT8 inference requirements
