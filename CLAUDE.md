# CLAUDE.md — alexnet_rafael

## Workflow

For non-trivial changes: inspect the relevant files, give a short plan, wait for approval. Don't edit immediately. Never run notebooks/training or commit unless explicitly asked.

## What this is

Deep-learning research on how **convolutional kernel-size restriction** affects CNNs. Motivation: **Winograd accelerators** are efficient for small kernels (2×2, 3×3) but scale poorly for large filters. We measure the accuracy/efficiency trade-off to recommend Winograd-friendly architectures.

**Scope:** Classification on **Tiny ImageNet-200** (64×64 RGB, 200 classes), FP32 training → **QAT → INT8**. Phases 1–3 compare AlexNet-style vs. efficient (MobileNet-style) architectures; Phase 4 builds and compresses final hybrid designs. Detection/segmentation are future work (`TODO.md`).

Questions: how kernel size affects accuracy/efficiency/quantization robustness; whether small-kernel CNNs match pretrained models; the FP32→INT8 drop per architecture family.

---

## Layout

```
ml/                       # Core package — notebooks import everything from here
  config.py               # DataConfig, TrainerConfig, QATConfig dataclasses (defaults explicit)
  data.py                 # create_imagenet_loaders(cfg)
  checkpoint.py           # save/load_checkpoint, load_resume_state, auto_resume_path
  registry.py             # MODEL_REGISTRY + register_model()
  trainer.py              # Trainer: fit(), evaluate(), benchmark()
  quantization.py         # find_fuse_groups, build_qat, convert_to_int8, load_best_model, make_qat_callback
  quantization_advanced.py# Mixed-precision / sub-INT8 PTQ: make_qconfig, prepare_sim, calibrate,
                          #   compute_layer_sensitivity, assign_mixed_precision, apply_weight_ptq, theoretical_size_mb
  reporting.py            # build_comparison_table, create_results_summary, disk_mb, compute_flops, make_run_summary
models/                   # Architectures by phase (see Model Inventory)
  baselines.py alexnet_variants.py compensation.py tinyhybridnet.py final_architecture.py
configs/                  # YAML hyperparameters, loaded via configs/loader.py → load_config(name)
  data.yaml training.yaml qat.yaml experiments/
  models/alexnet_fp32.yaml   # per-model lr overrides (only this one remains)
notebooks/                # One per phase: baselines_qat, alexnet_qat, compensation_qat,
                          #   tinyhybridnet_qat, compression_phase4_1, final_architecture_qat,
                          #   *_results / results_analysis (figures)
results/                  # git-ignored CSVs/JSON/figures, one dir per phase + results.csv, model_details.csv
docs/                     # Documentation (flat): PHASE7_QUICKSTART.md, PHASE7_MULTINODE.md, PHASE7_LOG.md
ideas/                    # Research notes (flat):
  BEST_MODELS.md          #   cross-phase rankings & recommendations
  MODELS.md               #   architecture notes & design rationale
  PHASE6_PLAN.md PHASE7_PLAN.md PHASE8_PLAN.md  # research & execution plans
outputs/                  # Training artifacts & logs
  pcad/
    logs/large_scale/     # 26 SLURM job output files (detection/segmentation runs)
    large_scale/          # Training outputs from large-scale PCAD runs
    alexnet_3x3_gap/      # Model checkpoints, results, tensorboard
    phase6_backfill/      # GPU profiling backfill data
    etc/                  # Other model runs
  local/                  # Local machine runs
```

Runtime artifacts (git-ignored): `checkpoints/{arch}_best.pth`, `qat_{arch}_best.pth`, `{arch}.pth` (INT8); logs `{arch}.log`, `qat_{arch}.log`.

---

## Recent Changes (2026-07-22)

**Reorganization commits:** `214c20a` (structure), `3ce1ffe` (path fix)

- Moved all 26 SLURM output files from root to `outputs/pcad/logs/large_scale/`
- Flattened `ideas/` and `docs/` directories (removed subdirectories, all files at root level)
  - `docs/`: PHASE7_QUICKSTART.md, PHASE7_MULTINODE.md, PHASE7_LOG.md
  - `ideas/`: BEST_MODELS.md, MODELS.md, PHASE6/7/8_PLAN.md, PHASE7_LOG.md (moved from ideas/)
- Fixed nested `outputs/pcad/alexnet_3x3_gap/alexnet_3x3_gap/` → flattened to `outputs/pcad/alexnet_3x3_gap/`
- Verified all path references in notebooks and scripts are correct (no hardcoded breakage)

---

## Stack

Python 3.12 · PyTorch 2.5.1+cu121 · torchvision 0.20.1 · torchmetrics · torchinfo · fvcore (FLOPs) · wandb (offline-first) · kagglehub · optuna (not yet wired) · CUDA 12.1 on RTX 4060 Laptop (8.2 GB).

Quantization backend: **fbgemm** — set `torch.backends.quantized.engine = "fbgemm"` before any QAT op. INT8 convert + inference are **CPU-only**.

---

## Key patterns

**Config** — instantiate dataclasses from YAML, override with `dataclasses.replace()`:
```python
data_cfg = DataConfig(**load_config("data.yaml"))
fp32_cfg = TrainerConfig(**load_config("training.yaml"))
qat_cfg  = QATConfig(**load_config("qat.yaml"))
data_cfg.seed = SEED
trainer_cfg = replace(fp32_cfg, lr=spec["lr"], epochs=2)
```

**Registry:**
```python
register_model("alexnet_fp32", build_alexnet, fuse_map=[...], fuse_root_attr="features", lr=1e-4)
```
`fuse_map` = list of dotted-path lists for Conv-BN(-ReLU) fusion. Flat/AlexNet-style: hand-write index maps like `[["0","1"],["3","4"]]`. Nested blocks: `find_fuse_groups(model())` auto-detects.

**Trainer:**
```python
trainer = Trainer(model, train_loader, val_loader, cfg=..., device=device,
                  save_dir=SAVE_DIR, run_name=name, num_classes=200,
                  log_file=SAVE_DIR/f"{name}.log")   # optional file+stdout logging
trainer.fit()                       # → best_val_accuracy, best_epoch, history{...}; saves {name}_best.pth
trainer.fit(resume_from=SAVE_DIR/f"{name}_best.pth")
trainer.evaluate(topk=(1,5))        # → {top1, top5, loss}
trainer.benchmark(warmup=100)       # latency/throughput; FP32 on GPU, INT8 on CPU
```
Skip/resume logic lives in the notebook loop, not in a wrapper.

**QAT flow:**
```
FP32 fit → saves {arch}_best.pth
build_qat(name, save_dir, device)   # load_best_model → copy → fuse → prepare_qat
fit(epoch_callback=make_qat_callback(freeze_bn_epoch, disable_observer_epoch))  # → qat_{arch}_best.pth
convert_to_int8(qat_model)          # eval() + CPU
evaluate(topk=(1,5))                # CPU val loader
```
QAT cfg is typically `replace(fp32_cfg, epochs=20, lr=1e-5, use_amp=False)`.

**QAT architecture rules:** all ReLU `inplace=False`; residual adds via `nn.quantized.FloatFunctional()` (not `+`); BN must sit immediately after its Conv.

**Data:** ImageFolder, deterministic 90/10 split (`torch.Generator`, seed 42), workers seeded via `worker_init_fn`. ImageNet normalization. Train aug: `RandomResizedCrop(0.7–1.0)`, hflip, `RandomRotation(15)`, `AutoAugment(ImageNet)`. Val: `Resize → CenterCrop`.

**Reproducibility:** seed `random`/`numpy`/`torch`/`cuda` at notebook top; `cudnn.deterministic=True`; do **not** set `cudnn.benchmark`.

**Reporting:** `make_run_summary(...)` builds a 30+ field dict per model → save one JSON each (crash-safe). `build_comparison_table` → `final_comparison.csv`; `create_results_summary` → `experiment_summary.json`. `compute_flops(model, input_size=(1,3,64,64))` → `{macs, flops}`. W&B: `wandb.init(project=..., config=asdict(cfg), mode="offline")`, sync later with `wandb sync --sync-all`; no auto-sync.

---

## Model Inventory

| Phase | File | Models |
|-------|------|--------|
| 1 — Reference | `baselines.py` | AlexNetTV, VGGStyleCNN, ResNet18TV, MobileNetV2TV (pretrained) |
| 2 — Kernel restriction | `alexnet_variants.py` | AlexNet3x3FC, AlexNet3x3GAP, AlexNet2x2GAP, AlexNet2x2FC, AlexNetStacked, AlexNetMixed, AlexNetSmallKernel |
| 3a — Compensation | `compensation.py` | AlexNet{Bottleneck, Factorized, GroupConv, DepthwiseSep, Residual, Fire, SE} |
| 3b — Efficient hybrids | `tinyhybridnet.py` | TinyHybridNet, TinyMobileNetV2, FireMobileResidual, InvertedResidual |
| 4 — Final architectures | `final_architecture.py` | AlexNetFinal{BottleneckFire, FireResidual, BottleneckResidual, DepthwiseFire} |

**Results & rankings:** see `ideas/BEST_MODELS.md` (Pareto tiers, recommendations) and `results/results.csv` / `results/model_details.csv`. Headlines: MobileNetV2 best overall (~58% top-1); AlexNetBottleneck/AlexNetFire are Pareto-optimal (43–44%, 4–6 MB, quantization-stable). Known issues: AlexNetSmallKernel severe QAT drop (~–10pp), AlexNetSE training failure.

---

## Running

```bash
source .venv/bin/activate
jupyter lab
```
Tiny ImageNet-200 downloads via `kagglehub` on first run (cached in `~/.cache/kagglehub/`). Before INT8 convert/inference: `model.eval()` and move to CPU.
