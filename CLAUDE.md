# CLAUDE.md — alexnet_rafael

## Default workflow

For any non-trivial coding request:

1. Inspect the relevant files.
2. Produce a short implementation plan.
3. Wait for approval before modifying code.

Do not immediately edit files.

## Project Overview

This is a deep-learning research project investigating the impact of **convolutional kernel size restriction** on deep neural networks across classification, object detection, and semantic segmentation tasks. The central motivation is **Winograd-accelerated convolution**: these specialized hardware accelerators achieve high computational efficiency for small kernels (2×2, 3×3) but face scalability limitations for larger filters. This project explores the trade-off between **computational efficiency** (via small kernels and Winograd acceleration) and **prediction quality**, aiming to identify when kernel restrictions preserve model performance and when they incur significant losses.

**Current implementation scope:** Classification on **Tiny ImageNet-200** (64×64 RGB images, 200 classes) under FP32 training and **Quantization-Aware Training (QAT)** → INT8 conversion. Stages 1–3 compare classical (AlexNet-style) and modern efficient architectures (MobileNet-style) to inform architecture design for Winograd-compatible CNNs.

Research questions:
- How do kernel-size restrictions (e.g., 2×2, 3×3 vs. mixed/larger kernels) affect accuracy, efficiency, and quantization robustness?
- Can small-kernel CNNs match pretrained models while remaining Winograd-accelerator-friendly?
- What is the FP32 → INT8 accuracy drop across architecture families under kernel constraints?
- Which kernel sizes and architectural patterns preserve prediction quality sufficiently for Winograd deployment?

---

## Expected Results

At project completion, expect:

1. **Quantitative & Qualitative Analysis:** Measure kernel-restriction impact on accuracy, model size, latency, and INT8 robustness across classification, detection, and segmentation tasks (initial focus: classification on Tiny ImageNet-200).

2. **Scenario Map:** Identify which scenarios tolerate small kernels (2×2, 3×3) without significant accuracy loss vs. which cases require mixed or larger kernels. Compare impact by architecture family (AlexNet, MobileNet, hybrid) and task.

3. **Architecture Recommendations:** Propose CNN designs optimized for Winograd-accelerated deployment, balancing kernel-size restrictions with prediction quality, and provide guidelines for practitioners choosing kernels for specialized accelerators.

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
│   ├── trainer.py           # Trainer class: fit(), evaluate(), benchmark(), _train_one_epoch(), _validate()
│   ├── quantization.py      # find_fuse_groups, prepare_qat_model, build_qat,
│   │                        #   convert_to_int8, load_best_model, make_qat_callback
│   └── reporting.py         # build_comparison_table(), create_results_summary(), disk_mb(),
│                            #   compute_flops(), make_run_summary()
├── models/                  # Model architecture definitions (experimental phases)
│   ├── __init__.py          # Re-exports all public constructors
│   ├── baselines.py         # Phase 1 — Reference: AlexNetTV, VGGStyleCNN, ResNet18TV,
│   │                        #   MobileNetV2TV
│   ├── alexnet_variants.py  # Phase 2 — Kernel Restriction: AlexNet3x3, AlexNet2x2,
│   │                        #   AlexNetStacked, AlexNetMixed, AlexNetSmallKernel
│   ├── compensation.py      # Phase 3a — Compensation Mechanisms: AlexNetBottleneck,
│   │                        #   AlexNetFactorized, AlexNetGroupConv, AlexNetDepthwiseSep,
│   │                        #   AlexNetResidual, AlexNetFire, AlexNetGAP, AlexNetSE
│   └── tinyhybridnet.py     # Phase 3b — Efficient Hybrids: TinyHybridNet, TinyMobileNetV2,
│                            #   FireMobileResidual, InvertedResidual,
│                            #   build_tinyhybridnet, build_tinymobilenetv2
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
│   ├── alexnet_qat.ipynb                    # Phase 2: AlexNet variants (3×3, small-kernel, mixed) FP32 + QAT
│   ├── baselines_qat.ipynb                  # Phase 1: Reference architectures (ResNet18, MobileNetV2, VGG-style) FP32 + QAT
│   ├── compensation_qat.ipynb               # Phase 3: Compensation mechanisms (bottleneck, residual, factorized, etc.) FP32 + QAT
│   ├── tinyhybridnet_qat.ipynb              # Phase 3b: TinyHybridNet, TinyMobileNetV2 FP32 + QAT
│   ├── alexnet_qat_results_analysis.ipynb   # Phase 2 W&B offline sync + plotting (legacy)
│   └── results_analysis.ipynb               # Cross-phase results analysis and figures
├── results/                 # Output CSVs, JSON summaries, and figures (git-ignored)
│   ├── baselines_qat_phase1/  # Phase 1 per-model JSONs, final_comparison.csv, experiment_summary.json
│   ├── alexnet_qat_phase2/    # Phase 2 per-model JSONs, final_comparison.csv, experiment_summary.json
│   └── figures/               # Generated charts (accuracy_vs_macs.png, fp32_accuracy_bar.png, etc.)
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
| fvcore | — | FLOPs/MACs counting in `compute_flops()` |
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

`fuse_map` is a list of dotted-path lists used to fuse Conv-BN-ReLU. For flat architectures, supply it manually. For nested block architectures, use `find_fuse_groups(model())` to auto-detect.

### 5. Trainer Usage

```python
from ml import Trainer, build_qat, build_qat_from_model
from dataclasses import replace
from pathlib import Path

# FP32 — override lr per model from registry, optionally log to file
trainer = Trainer(model, train_loader, val_loader,
                  cfg=replace(fp32_cfg, lr=spec["lr"]),
                  device=device, save_dir=SAVE_DIR, run_name=name,
                  num_classes=200,
                  log_file=SAVE_DIR / f"{name}.log")  # optional; logs to file + stdout
results = trainer.fit()
# → {"best_val_accuracy": ..., "best_epoch": ..., "total_training_time_s": ...,
#    "history": {"train_loss": [...], "lr": [...], "epoch_time_s": [...], "peak_gpu_mem_mb": [...]}, ...}

metrics = trainer.evaluate(topk=(1, 5))  # → {"top1": ..., "top5": ..., "loss": ...}

# Benchmark inference (FP32 on GPU, INT8 on CPU)
benchmark_results = trainer.benchmark()  # → {"latency_ms_per_image": ..., "throughput_img_per_s": ..., "device": ...}

# QAT — two patterns: build_qat() by arch name, or build_qat_from_model(model)
qat_cfg = replace(fp32_cfg, epochs=20, lr=1e-5, use_amp=False)
cb = make_qat_callback(freeze_bn_epoch=3, disable_observer_epoch=5)

# Option 1: by registry name
qat_model = build_qat("alexnet_small_kernel", SAVE_DIR, device)
# Option 2: from loaded model directly
# qat_model = build_qat_from_model(trainer.model, freeze_maps=[...], fuse_root="features")

trainer = Trainer(qat_model, ..., cfg=qat_cfg, epoch_callback=cb,
                  log_file=SAVE_DIR / f"qat_{name}.log")
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

- **Flat sequential architectures (AlexNet style):** Hand-written index-based fuse maps, e.g., `[["0","1"],["3","4"],...]`
- **Nested block architectures:** `find_fuse_groups(model())` walks the tree recursively and returns dotted-path lists automatically.
- **New architectures:** Call `find_fuse_groups` or write a fuse map and pass it to `register_model`.

### 8. Data Pipeline

- Dataset: Tiny ImageNet-200, loaded via `torchvision.datasets.ImageFolder`
- Deterministic 90/10 train/val split using `torch.Generator` seeded at `seed=42`
- Worker RNG seeded via `worker_init_fn` (`random.seed(cfg.seed + worker_id)`) for reproducible PIL-based augmentation across DataLoader workers
- Normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (ImageNet stats)
- Training augmentation: `RandomResizedCrop(scale=0.7-1.0)`, `RandomHorizontalFlip`, `RandomRotation(15°)`, `AutoAugment(ImageNet policy)`
- Validation: `Resize(img_size) → CenterCrop(img_size)`

### 9. Logging & Reporting

**File-based logging:** Pass `log_file=Path(...)` to `Trainer.__init__()`. Creates a timestamped `.log` file with all epoch-level metrics. All `print()` calls → `self.logger.info(...)`. Logs to both console and file.

**W&B logging:** Includes per-epoch: `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_top5`, `lr`, `epoch_time_s`, `peak_gpu_mem_mb`, `grad_norm` (only when `grad_clip_norm` is set). Run init must include `tags`, `num_classes`, `img_size`, `dataset` in config for full traceability.

**Inference benchmarking:** `Trainer.benchmark(loader=None, warmup=100)` measures latency and throughput on the given loader (or val_loader). Warm-up skips, then times a full pass with `torch.no_grad()` and (if CUDA) `torch.cuda.synchronize()`.

**Per-model summaries:** `make_run_summary(name, mode, fit_results, fp32_eval, ...)` assembles 30+ fields: training metrics, FP32/INT8 eval, model sizes, MACs/FLOPs, latency, throughput, compression ratio, accuracy drops, parameter efficiency. Returns a dict; save to JSON per model for crash-safety.

**FLOPs calculation:** `compute_flops(model, input_size=(1,3,64,64))` uses fvcore's FlopCountAnalysis. Returns `{"macs": int, "flops": int}`.

---

## Coding Conventions

- **Reproducibility first:** Seed everything at the top of each notebook (`random`, `numpy`, `torch`, `torch.cuda`); set `cudnn.deterministic = True`. Do **not** set `cudnn.benchmark = True` — it conflicts with determinism and provides no benefit for fixed 64×64 inputs.
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
  │                 build_qat, build_qat_from_model, load_best_model, convert_to_int8
  │                 build_comparison_table, create_results_summary, disk_mb,
  │                 compute_flops, make_run_summary
  ├─ from models import AlexNetTV, AlexNet3x3, AlexNetSmallKernel, AlexNetResidual, ...
  │                     TinyHybridNet, TinyMobileNetV2, ...
  │                     (pick from baselines.py/alexnet_variants.py/compensation.py/tinyhybridnet.py)
  ├─ register_model() → MODEL_REGISTRY
  │
  ├─ FP32 training loop:
  │   ├─ for name, spec in MODEL_REGISTRY.items():
  │   │     wandb.init(..., tags=[], config={...num_classes, img_size, dataset...})
  │   │     Trainer(..., log_file=SAVE_DIR / f"{name}.log", wandb_run=run).fit()
  │   │     wandb.finish()
  │
  ├─ QAT training loop:
  │   ├─ for name in MODEL_REGISTRY:
  │   │     wandb.init(..., tags=[], config={...})
  │   │     build_qat(name, SAVE_DIR, device)
  │   │     Trainer(..., log_file=..., epoch_callback=cb).fit()
  │   │     wandb.finish()
  │
  ├─ INT8 conversion & evaluation:
  │   ├─ convert_to_int8(qat_model) → evaluate(topk=(1,5)) on CPU
  │
  ├─ Benchmarking & per-model summaries (NEW):
  │   ├─ fp32_benchmarks[name] = Trainer(...).benchmark()  # GPU
  │   ├─ int8_benchmarks[name] = Trainer(...).benchmark()  # CPU, warmup=50
  │   ├─ fp32_flops[name] = compute_flops(model)
  │   ├─ for each model:
  │   │     make_run_summary(...) → RESULTS_DIR / f"{name}_summary.json"
  │
  ├─ Comparison table & aggregate summary:
  │   ├─ build_comparison_table(rows) → results/.../final_comparison.csv
  │   └─ create_results_summary(results, config, output_path) → results/.../experiment_summary.json
```

---

## Model Inventory

Four implemented model groups, plus future phases defined in `TODO.md`:

| Phase | File | Models |
|-------|------|--------|
| 1 — Reference Architectures | `baselines.py` | AlexNetTV (pretrained), VGGStyleCNN, ResNet18TV (pretrained), MobileNetV2TV (pretrained) |
| 2 — Kernel Restriction Study | `alexnet_variants.py` | AlexNet3x3, AlexNet2x2, AlexNetStacked, AlexNetMixed, AlexNetSmallKernel |
| 3a — Compensation Mechanisms | `compensation.py` | AlexNetBottleneck, AlexNetFactorized, AlexNetGroupConv, AlexNetDepthwiseSep, AlexNetResidual, AlexNetFire, AlexNetGAP, AlexNetSE |
| 3b — Efficient Hybrids | `tinyhybridnet.py` | TinyHybridNet, TinyMobileNetV2 |
| 4+ (Future) | — | See `TODO.md` for phases 4–8: Final Architecture, Final Analysis, Hardware Profiling, ViT/Attention Hybrids, Architecture Search |

---

## Experimental Results (Reference)

**Phase 1 — Reference Architectures** (`results/baselines_qat_phase1/`; FP32 only):

| Model | Params | FP32 Top-1 | FP32 Top-5 |
|-------|--------|-----------|-----------|
| ResNet18 (pretrained) | 11.28M | 46.95% | 73.50% |
| MobileNetV2 (pretrained) | 2.48M | 45.81% | 73.00% |
| AlexNet (pretrained) | 57.82M | 19.63% | 45.43% |
| VGGStyleCNN | 2.41M | 10.64% | 30.42% |

**Phase 2 — Kernel Restriction Study** (`results/alexnet_qat_phase2/`; FP32 + INT8, 57 epochs):

| Model | Params | FP32 Top-1 | INT8 Top-1 | FP32 Size | INT8 Size |
|-------|--------|-----------|-----------|-----------|-----------|
| AlexNetSmallKernel | 1.60M | 45.84% | 35.95% | 18.35 MB | 1.56 MB |
| AlexNetStacked | 60.48M | 44.56% | 42.79% | 692.25 MB | 57.94 MB |
| AlexNetMixed | 1.75M | 38.74% | 38.00% | 20.04 MB | 1.71 MB |
| AlexNet3x3 | 57.61M | 35.79% | 36.19% | 659.26 MB | 55.12 MB |
| AlexNet2x2 | 1.05M | 30.02% | 30.89% | 12.06 MB | 1.04 MB |
| AlexNet (pretrained, retrained) | 57.82M | 24.29% | 26.28% | 661.75 MB | 55.33 MB |

Key findings:
- AlexNetSmallKernel reaches 45.84% FP32 Top-1 at 1.6M params (1.56 MB INT8) — competitive with MobileNetV2 and ResNet18 at a fraction of the size.
- AlexNetStacked is the most quantization-stable (–1.77pp INT8 drop) but at 58× more parameters than AlexNetSmallKernel.
- 3×3 and mixed kernels recover well post-quantization; 2×2 also recovers (+0.87pp INT8 gain), suggesting quantization acts as regularization at this scale.
- Per-model summaries: `{RESULTS_DIR}/{arch}_summary.json` (30+ fields: FP32/INT8 accuracy, latency, throughput, MACs/FLOPs, quantization drop). See `make_run_summary()` for full schema.
- Cross-phase figures in `results/figures/` (accuracy_vs_macs.png, fp32_vs_int8_bar.png, fp32_training_curves.png, etc.).

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

**Checkpoint & log locations:** Each notebook sets its own `SAVE_DIR`. 
- Best FP32 model: `{SAVE_DIR}/{arch}_best.pth`. 
- Best QAT model: `{SAVE_DIR}/qat_{arch}_best.pth`. 
- INT8 state dict: `{SAVE_DIR}/{arch}.pth`.
- Training logs: `{SAVE_DIR}/{arch}.log` and `{SAVE_DIR}/qat_{arch}.log` (timestamped per epoch).

**Results & summaries:** Each notebook writes to a phase-specific `RESULTS_DIR` (e.g., `results/baselines_qat_phase1/`, `results/alexnet_qat_phase2/`):
- Per-model summaries: `{RESULTS_DIR}/{arch}_summary.json` (30+ metrics, crash-safe).
- Comparison table: `{RESULTS_DIR}/final_comparison.csv` (all models, FP32 vs INT8).
- Aggregate summary: `{RESULTS_DIR}/experiment_summary.json` (all metrics combined).
- Cross-phase figures: `results/figures/` (generated by `notebooks/results_analysis.ipynb`).

**Sync W&B offline runs:**
```bash
wandb sync --sync-all
```

**INT8 inference note:** Always call `model.eval()` and move model to CPU before `torch.ao.quantization.convert()` and before INT8 forward passes.

**Install dependencies** (if needed):
```bash
pip install fvcore
```
