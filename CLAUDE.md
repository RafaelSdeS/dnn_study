# CLAUDE.md — alexnet_rafael

## Workflow

For non-trivial changes: inspect the relevant files, give a short plan, wait for approval. Don't edit immediately. Never run notebooks/training or commit unless explicitly asked.

## What this is

Deep-learning research on how **convolutional kernel-size restriction** affects CNNs. Motivation: **Winograd accelerators** are efficient for small kernels (2×2, 3×3) but scale poorly for large filters. We measure the accuracy/efficiency trade-off to recommend Winograd-friendly architectures.

**Scope:** Classification on **Tiny ImageNet-200** (64×64 RGB, 200 classes), FP32 training → **QAT → INT8**. Phases 1–3 compare AlexNet-style vs. efficient (MobileNet-style) architectures; Phase 4 builds and compresses final hybrid designs; Phase 5 is cross-phase results analysis; Phase 6 profiles hardware (latency/power/GPU utilization) for the best models; **Phase 7 (done/ongoing)** tests whether classification's compensation findings transfer to dense prediction — SSD detection + segmentation on PASCAL VOC (`ml/det_seg_data.py`, `det_seg_models.py`, `det_seg_trainer.py`, driven by `scripts/train_det_seg.py`, results under `outputs/detection_segmentation/phase7/`). Phase 8 (efficient ViT / hybrid-attention) is planned only — see `ideas/PHASE8_PLAN.md` — no code yet.

Questions: how kernel size affects accuracy/efficiency/quantization robustness; whether small-kernel CNNs match pretrained models; the FP32→INT8 drop per architecture family; whether classification-derived compensation mechanisms (bottleneck, Fire, depthwise) transfer to detection/segmentation.

---

## Layout

```
ml/                       # Core package — notebooks and scripts import everything from here
  config.py               # DataConfig, TrainerConfig, QATConfig dataclasses (defaults explicit)
  data.py                 # create_imagenet_loaders(cfg)
  det_seg_data.py         # Phase 7: create_voc_detection_loaders / create_voc_segmentation_loaders
  det_seg_models.py       # Phase 7: build_ssd_detector, build_qat_ssd_detector, convert_ssd_to_int8, compute_anchor_recall
  det_seg_trainer.py      # Phase 7: DetectionTrainer (subclasses the Trainer loop for box/mask losses)
  checkpoint.py           # save/load_checkpoint, load_resume_state, auto_resume_path, compress_checkpoint (.pth.gz)
  registry.py             # MODEL_REGISTRY + register_model()
  model_registrations.py  # Populates MODEL_REGISTRY for standalone scripts (mirrors notebook registrations — keep in sync)
  trainer.py              # Trainer: fit(), evaluate(), benchmark()
  quantization.py         # find_fuse_groups, build_qat, convert_to_int8, load_best_model, make_qat_callback
  quantization_advanced.py# Mixed-precision / sub-INT8 PTQ: make_qconfig, prepare_sim, calibrate,
                          #   compute_layer_sensitivity, assign_mixed_precision, apply_weight_ptq, theoretical_size_mb
  profiling.py            # Phase 6: GpuSampler (nvidia-smi power/util/temp/mem sampling), latency/throughput profiling
  pruning.py              # Phase 9: prune_model_channels — structured (whole-channel) pruning, stays Winograd-dense
  runtime.py              # RuntimePaths, set_global_seed — shared by scripts/train.py and scripts/train_det_seg.py
  reporting.py            # build_comparison_table, create_results_summary, disk_mb, compute_flops, make_run_summary
models/                   # Architectures by phase (see Model Inventory)
  baselines.py alexnet_variants.py compensation.py tinyhybridnet.py final_architecture.py
configs/                  # YAML hyperparameters, loaded via configs/loader.py → load_config(name)
  data.yaml training.yaml qat.yaml profiling.yaml compression.yaml detection.yaml
  runtime/                # local.yaml, pcad.yaml — dataset root, conda env, per-runtime toggles
  slurm/                  # single_gpu.yaml, tupi_4090.yaml, beagle.yaml — partition/GPU/CPU/wall-time
  experiments/            # default.yaml + per-run overrides (alexnet_3x3_gap, phase7_detection, phase7_diag_*, phase7_smoke, large_scale, ...)
  models/alexnet_fp32.yaml   # per-model lr overrides (only this one remains)
scripts/                  # CLI entry points (used instead of notebooks for PCAD/cluster runs)
  train.py                # `python -m scripts.train --experiment ... --runtime local|pcad` — classification FP32→QAT→INT8
  cluster.py               # `python -m scripts.cluster submit|status|cancel|resume` — submits slurm/train.sbatch or profile.sbatch
  train_det_seg.py         # Phase 7 detection/segmentation CLI, mirrors train.py
  profile_hardware.py      # Phase 6 hardware profiling CLI
  phase7_analysis.py       # Joins detection/segmentation results to Phase 3 classification results, tests H1-H4
  aggregate_results.py     # Aggregates per-model summary JSONs from a cluster submit-sweep into one CSV
  winograd_quant_error.py  # Phase 6 extension: INT8 quantization error from Winograd F(2x2,3x3) transforms
  measure_compression.py   # Phase 9 Task 3: entropy/k-means weight-compression headroom above plain gzip
  prune_channels.py        # Phase 9 Task 2: structured (channel) pruning CLI, mechanics-only (no fine-tuning)
  check_anchor_recall.py / diag_stage7.py / backfill_gzip.py  # one-off Phase 7 diagnostics / backfill tools
  migrate_pcad_gitignored.sh  # merges gitignored artifacts (*.pth, *.log) left in pre-reorg folder names after a pull
  submit_phase7_simple.sh / submit_phase7_multinode.sh  # PCAD multi-node Phase 7 submission (simple vs FP32→QAT→INT8 chaining) — see docs/PHASE7_MULTINODE.md
  slurm/*.sbatch           # sbatch templates — train.sbatch/profile.sbatch submitted by cluster.py, det_seg.sbatch by the submit_phase7_*.sh scripts, others called directly
tests/                    # pytest: test_registry, test_checkpoint, test_config, test_trainer_smoke,
                          #   test_quantization, test_profiling, test_train_cli
notebooks/                # Organized by phase + purpose
  phase_1_baseline_training/                          # baselines_qat
  phase_2_kernel_restriction_training/                # alexnet_qat
  phase_3_compensation_and_hybrids_training/          # compensation_qat, tinyhybridnet_qat
  phase_4_compression_and_final_architecture_training/ # compression_phase4_1, final_architecture_qat
  phase_5_cross_phase_results_analysis/               # final_analysis_phase5, final_architecture_results, _archived_results_analysis
  phase_6_hardware_profiling_analysis/                # hardware_profiling_phase6
  phase_9_pcad_bypass_ablation_analysis/              # pcad_results_analysis
results/                  # git-ignored CSVs/JSON/figures, one dir per phase + results.csv, model_details.csv
presentation/             # slides.md/slides.pdf + figures/ (generated by presentation/make_figures.py)
docs/                     # Documentation (flat): PHASE7_QUICKSTART.md, PHASE7_MULTINODE.md, PHASE7_LOG.md
ideas/                    # Research notes (flat):
  BEST_MODELS.md          #   cross-phase rankings & recommendations
  MODELS.md               #   architecture notes & design rationale
  PHASE6_PLAN.md PHASE7_PLAN.md PHASE8_PLAN.md  # research & execution plans (6/7 executed, 8 planned only)
outputs/                  # Training artifacts & logs
  detection_segmentation/ # Phase 7 detection/segmentation run outputs (SSD checkpoints/logs per model+config)
    phase7/
  pcad/
    phase_6_hardware_profiling/ # GPU profiling data (runs/, backfill/)
    phase_9_bypass_ablation/    # Phase 9 runs (fire_bypass/, fire_bypass_large_scale/)
    archive_legacy_phases/      # Phase 2, 4, 5 runs (phase_2_kernel_restriction/, phase_4_5_large_scale/)
    logs_by_phase/              # SLURM job output files (all_slurm_jobs/)
    figures_generated/          # Generated visualizations
    results_aggregate/          # Cross-phase result CSVs
  local/                  # Local machine runs
  smoke_test/             # Quick validation runs
```

Runtime artifacts (git-ignored): `checkpoints/{arch}_best.pth`, `qat_{arch}_best.pth`, `{arch}.pth` (INT8, plus `.pth.gz` compressed copy via `ml/checkpoint.py`); logs `{arch}.log`, `qat_{arch}.log`.

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
| 3a — Compensation | `compensation.py` | AlexNet{Bottleneck, Factorized, GroupConv, DepthwiseSep, Residual, Fire, SE, SmallKernelWithBN, DilatedFC, DilatedGAP} |
| 3b — Efficient hybrids | `tinyhybridnet.py` | TinyHybridNet, TinyMobileNetV2, FireMobileResidual, InvertedResidual |
| 4 — Final architectures | `final_architecture.py` | AlexNetFinal{BottleneckFire, FireResidual, BottleneckResidual, DepthwiseFire} |
| 6 — Hardware profiling | (reuses Phase 1–4 models) | `ml/profiling.py` + `scripts/profile_hardware.py`; dilated variants added to test whether dilated 3×3 retains Winograd acceleration |
| 7 — Detection/segmentation | `ml/det_seg_models.py` | Bottleneck/Fire/AlexNetTV backbones + SSD head on PASCAL VOC, via `scripts/train_det_seg.py` |

**Results & rankings:** see `ideas/BEST_MODELS.md` (Pareto tiers, recommendations) and `results/results.csv` / `results/model_details.csv`. Headlines: MobileNetV2 best overall (~58% top-1); AlexNetBottleneck/AlexNetFire are Pareto-optimal (43–44%, 4–6 MB, quantization-stable). Known issues: AlexNetSmallKernel severe QAT drop (~–10pp), AlexNetSE training failure. Phase 7 hypotheses (H1–H4, does compensation transfer to dense prediction) and progress: `ideas/PHASE7_PLAN.md`, `docs/PHASE7_LOG.md`.

---

## Running

**Notebooks** (Phases 1–5, exploratory):
```bash
source .venv/bin/activate
jupyter lab
```
Tiny ImageNet-200 downloads via `kagglehub` on first run (cached in `~/.cache/kagglehub/`). Before INT8 convert/inference: `model.eval()` and move to CPU.

**CLI / cluster runs** (Phases 6–7, reproducible local or PCAD SLURM runs):
```bash
conda env create -f environment.yml && conda activate alexnet_rafael
python -m scripts.train --experiment default --runtime local        # classification, local
python -m scripts.cluster submit --experiment default --runtime pcad --slurm single_gpu
python -m scripts.cluster status <job_id>   # / cancel / resume
python -m scripts.train_det_seg detection --model alexnet_bottleneck --dry-run   # Phase 7
python -m scripts.profile_hardware --experiment phase6 --runtime local           # Phase 6
```
Edit `configs/runtime/pcad.yaml` (dataset root, conda env) and `configs/slurm/single_gpu.yaml` (partition/GPU/wall-time) for cluster settings; duplicate `configs/experiments/default.yaml` for a new reproducible run.

**Tests:** `pytest tests/`
