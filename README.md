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
1. `notebooks/phase_1_baseline_training/baselines_qat.ipynb` — Phase 1: Reference pretrained models (ResNet18, MobileNetV2, AlexNet, VGG-style)
2. `notebooks/phase_2_kernel_restriction_training/alexnet_qat.ipynb` — Phase 2: AlexNet kernel-restriction variants (3×3, 2×2, stacked, mixed, small-kernel)
3. `notebooks/phase_3_compensation_and_hybrids_training/compensation_qat.ipynb` — Phase 3a: Compensation mechanisms (bottleneck, residual, factorized, etc.)
4. `notebooks/phase_3_compensation_and_hybrids_training/tinyhybridnet_qat.ipynb` — Phase 3b: Efficient hybrids (TinyHybridNet, TinyMobileNetV2)
5. `notebooks/phase_4_compression_and_final_architecture_training/compression_phase4_1.ipynb` — Phase 4.1: Aggressive compression (INT4/INT2/ternary/binary) of the best Phase 1–3 models
6. `notebooks/phase_4_compression_and_final_architecture_training/final_architecture_qat.ipynb` — Phase 4: Combines Phase 3's best mechanisms into final hybrid architectures
7. `notebooks/phase_5_cross_phase_results_analysis/final_analysis_phase5.ipynb` — Phase 5: Cross-phase results analysis and figure generation
8. `notebooks/phase_6_hardware_profiling_analysis/hardware_profiling_phase6.ipynb` — Phase 6: Hardware profiling & Winograd efficiency validation (RTX 4090)
9. `notebooks/phase_9_pcad_bypass_ablation_analysis/pcad_results_analysis.ipynb` — Phase 9: Cross-phase PCAD results (bypass ablation, large-scale runs)

Phase 7 (detection/segmentation) has no notebook — it's CLI-only via `scripts/train_det_seg.py`; see `docs/PHASE7_QUICKSTART.md`. Phase 8 is planned only (`ideas/PHASE8_PLAN.md`), no code yet. Full model/phase inventory: `CLAUDE.md`.

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
| 4 — Final Architecture | `final_architecture.py` | Combines Phase 3's best mechanisms into final hybrid models |

Phases 6–9 (hardware profiling, detection/segmentation, bypass ablation) reuse these model files rather than adding new ones. Full model inventory + per-phase file map: `CLAUDE.md`.

### `configs/` — Hyperparameter Defaults

YAML-based configuration loading via `configs/loader.py`. Full list in `CLAUDE.md`; the core ones:
- `data.yaml` — Dataset, augmentation, normalization
- `training.yaml` — Epochs, batch size, learning rate
- `qat.yaml` — QAT-specific settings
- `experiments/` — Per-run overrides (one YAML per experiment, `default.yaml` as the template)
- `runtime/` — `local.yaml` / `pcad.yaml` — dataset root, conda env per runtime
- `slurm/` — Partition/GPU/wall-time per cluster profile

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
| AlexNet3x3GAP | 57.61M | 38.74% | 37.60% | –1.14pp | 2.23 MB |
| AlexNet3x3FC | 57.61M | 35.79% | 36.19% | +0.40pp ✓ | 55.12 MB |
| AlexNet2x2GAP | 1.05M | 30.02% | 30.89% | +0.87pp ✓ | 1.04 MB |

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
- **Model efficiency summary:** Results in `results/results_aggregate/results_cross_phase.csv` and per-model details in `results/results_aggregate/model_details_cross_phase.csv`
- **Visualizations:** Cross-phase figures in `results/figures_generated/phase_1_2_3_baseline_and_kernels/` and `phase_5_cross_phase_analysis/`

Phases 4–9 have since completed — see `CLAUDE.md`'s Model Inventory and `ideas/BEST_MODELS.md` (currently scoped to Phases 1–3; not yet updated with later-phase rankings).

---

## Running Experiments

**See `CLAUDE.md` for:**
- Reproducibility patterns (seed, cudnn settings)
- Trainer API examples
- Model registry usage
- Checkpoint & config workflows
- INT8 inference requirements

## PCAD Training Infrastructure

The repo now includes a cluster-friendly runner that reuses the existing `ml/` training code and keeps the same checkpoint/resume flow on both local machines and PCAD.

**Environment setup:**
```bash
conda env create -f environment.yml
conda activate alexnet_rafael
```

**Local run:**
```bash
python -m scripts.train --experiment default --runtime local
```

**PCAD submission:**
```bash
python -m scripts.cluster submit --experiment default --runtime pcad --slurm single_gpu
```

**Monitor, cancel, resume:**
```bash
python -m scripts.cluster status <job_id>
python -m scripts.cluster cancel <job_id>
python -m scripts.cluster resume outputs/pcad/default/<model_name>
```

**Runtime layout:**
- `outputs/<runtime>/<experiment>/<model>/checkpoints/` for `*_best.pth`, `*_resume.pth`, and INT8 artifacts
- `outputs/<runtime>/<experiment>/<model>/logs/` for per-run logs
- `outputs/<runtime>/<experiment>/<model>/tensorboard/` for TensorBoard event files
- `outputs/<runtime>/<experiment>/<model>/results/` for per-run JSON summaries
- `outputs/<runtime>/results_aggregate/` for aggregated comparison CSVs (via `scripts/aggregate_results.py`)

Detection/segmentation (Phase 7) uses a separate layout: `outputs/detection_segmentation/phase7/<run>/` — see `docs/PHASE7_QUICKSTART.md`.

**PCAD-specific settings:**
- Edit [configs/runtime/pcad.yaml](configs/runtime/pcad.yaml) for dataset root, conda env, and runtime toggles
- Edit [configs/slurm/single_gpu.yaml](configs/slurm/single_gpu.yaml) for partition, GPU, CPU, memory, and wall-time settings
- Set `PCAD_DATASET_ROOT` on the cluster when Tiny ImageNet is staged on shared storage; otherwise the runner falls back to KaggleHub

The default experiment config lives in [configs/experiments/default.yaml](configs/experiments/default.yaml). Duplicate it to create a new reproducible run without changing training code.
