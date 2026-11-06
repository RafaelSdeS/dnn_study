# CNN Kernel Research: Winograd Acceleration & Efficiency

How convolutional **kernel-size restriction** affects CNN accuracy, efficiency and quantization robustness. The motivation is **Winograd-accelerated convolution**: efficient for small kernels (2×2, 3×3), poorly scalable to large filters. The study measures the accuracy / efficiency trade-off to recommend Winograd-friendly architectures.

**Scope:** classification on Tiny ImageNet-200 (64×64 RGB, 200 classes), FP32 → quantization-aware training (QAT) → INT8. Phase 7 tests detection / segmentation on PASCAL VOC; Phase 8 tests windowed attention (Swin / ViT) against small-kernel CNNs.

| Phases | What | Where to read |
|---|---|---|
| 1–4 | AlexNet-style vs efficient nets, kernel restriction, compensation (Fire, bottleneck, depthwise), final hybrids | `docs/plans/BEST_MODELS.md`, `notebooks/` |
| 5, 10 | cross-phase and cross-project roll-ups | `notebooks/phase_5_*`, `phase_10_*` |
| 6 | hardware profiling (latency, power, GPU use) | `docs/plans/PHASE6_PLAN.md` |
| 7, 8, 9 | detection / segmentation, efficient ViT, bypass ablation + pruning | `docs/logs/PHASE7_LOG.md`, `PHASE8_LOG.md`, `docs/plans/PHASE9_PLAN.md` |
| **11–13** | **the main study**: kernel size on one recipe (152 runs), its robustness (39), hybrids (18 + seeds) | `docs/plans/BEST_MODELS.md` ("Cross-phase synthesis"), `docs/logs/PHASE11_LOG.md`, `PHASE13_LOG.md`, `docs/plans/PROTOCOL.pdf` |
| 14 | Winograd-FPGA accelerator numerics (F23 / F43 / F63) on the same recipe — running | `docs/logs/PHASE14_LOG.md` |

Results and rankings: **`docs/plans/BEST_MODELS.md`**; open work: `TODO.md`; machine-readable tables: `results/` (`results_aggregate/*.csv`, `runs_index.csv`).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # first time
source .venv/bin/activate
python -m scripts.train --experiment default --runtime local          # FP32 → QAT → INT8, one config
python -m scripts.cluster submit --experiment default --runtime pcad --slurm single_gpu
python -m scripts.cluster status <job_id>                             # / cancel / resume
pytest tests/test_config.py -q                                        # run tests by file
```

Phases 1–5 and 7–10 have analysis notebooks under `notebooks/<phase slug>/`; Phases 11–14 are CLI-driven (`scripts/train.py`, `scripts/cluster.py`) and analysed by `scripts/phase11/`, `phase12/`, `phase13/`. Tiny ImageNet downloads through `kagglehub` on first use (set `PCAD_DATASET_ROOT` on the cluster to use staged data). Detection / segmentation: `docs/logs/PHASE7_QUICKSTART.md`.

## Layout

```
ml/          core package: data, trainer, quantization, reporting, registry, profiling
models/      architectures, one file per phase
configs/     YAML: data / training / qat + runtime/ + slurm/ + experiments/ (+ _protocols/)
scripts/     CLI entry points; per-phase tooling in phase6/ … phase13/, winograd_fpga/, pcad/, lab/
notebooks/   one dir per phase slug
outputs/     RAW per-run artifacts: local/ pcad/ notebooks/ (by where the run happened)
results/     CURATED tracked CSVs / JSON / figures, feeds report/ and presentation/
docs/        plans/ (research notes) + logs/ (dated execution history) + REFERENCE.md
report/      LaTeX writeup      presentation/  slides
```

Two conventions hold it together: **`outputs/` is raw, `results/` is curated**; and **one slug per phase** (`phase_N_description`), identical in `notebooks/`, `results/`, `results/figures_generated/`, `outputs/*/` and `configs/experiments/` (the experiment name *is* the output directory name).

## QAT in one paragraph

FP32 `Trainer.fit()` → `build_qat()` (load best, fuse Conv-BN-ReLU, `prepare_qat`) → QAT `fit` → `convert_to_int8()` on CPU → evaluate. Requirements: ReLU `inplace=False`; residual adds through `FloatFunctional`; BN right after its Conv. INT8 is CPU-only; accuracy is measured on the qnnpack engine (exact int32 accumulation on any CPU). Details: `CLAUDE.md` "Key patterns", `docs/REFERENCE.md`.

## Cluster (PCAD)

`scripts/cluster.py` submits `scripts/slurm/*.sbatch` and keeps the same checkpoint / resume flow as a local run. Edit `configs/runtime/pcad.yaml` (dataset root) and `configs/slurm/*.yaml` (partition, GPU, wall time). Per-run layout: `outputs/<runtime>/<experiment>/<model>/{checkpoints,logs,tensorboard,results}/`; per-experiment comparison CSVs in `outputs/<runtime>/aggregates/`. Operating notes (queues, nodes, pulling on PCAD): `docs/REFERENCE.md` "Running".
