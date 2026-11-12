# CLAUDE.md — alexnet_rafael

## Workflow

For non-trivial changes: inspect the relevant files, give a short plan, wait for approval. Don't edit immediately.
Never run notebooks/training or commit unless explicitly asked.
Per-file notes, fix history, the full model inventory and the PCAD / beagle / nyx operating notes are in **`docs/REFERENCE.md`**; dated decisions in `docs/logs/PHASE*_LOG.md`; results and rankings in `docs/plans/BEST_MODELS.md`; open work in `TODO.md`.

## What this is

Research on how **convolutional kernel-size restriction** affects CNNs. Motivation: Winograd accelerators are efficient for small kernels (2×2, 3×3) and scale poorly for large ones; we measure the accuracy / efficiency / quantization trade-off to recommend Winograd-friendly architectures.

Scope: classification on **Tiny ImageNet-200** (64×64, 200 classes), FP32 → QAT → INT8. Questions: how kernel size affects accuracy, efficiency and quantization robustness; whether small-kernel CNNs match pretrained ones; the FP32 → INT8 drop per family; whether compensation mechanisms (bottleneck, Fire, depthwise) transfer to detection / segmentation (Phase 7) and to attention (Phase 8).

## Phases (slug = `phase_N_description`, identical in `notebooks/`, `results/`, `outputs/*/`, `configs/experiments/`)

| Phase | Slug | What |
|---|---|---|
| 1–3 | `phase_1_baseline`, `phase_2_kernel_restriction`, `phase_3_compensation_and_hybrids` | AlexNet-style vs efficient nets; kernel restriction; compensation (Fire, bottleneck, depthwise…) |
| 4 | `phase_4_compression_and_final_architecture` | final hybrids + compression |
| 5, 10 | `phase_5_cross_phase_analysis`, `phase_10_final_summary` | roll-ups (Phase 10 here ≠ TODO's future NAS "Phase 10") |
| 6 | `phase_6_hardware_profiling` | latency / power / GPU utilization |
| 7 | `phase_7_detection_segmentation` | SSD + segmentation on VOC (`ml/det_seg_*`, `scripts/train_det_seg.py`) |
| 8 | `phase_8_efficient_vit` (+`_convstem`) | windowed Swin / ViT vs small-kernel CNNs (`models/vit_variants.py`, H1–H5) |
| 9 | `phase_9_bypass_ablation` (+`_large_scale`, `_pruning`, `_compression`) | SqueezeNet-style bypass, pruning, compression |
| 11 | `phase_11_kernel_size_comparison` — experiments `phase_11_*` | **the main study**: 152 runs on ONE recipe (done 2026-10-30) |
| 12 | `phase_12_robustness` | Phase 11's conclusions under another lr, fine-tuning, CIFAR-100 (39 runs, done) |
| 13 | `phase_13_hybrids` | do two convs per stage / skip / 11×11 conv1 / conv1 stride 1 add up? 18 of 18 + seeds 43/44, all done (2026-11-07); Stage B `phase_13_bottleneck_fire` (Bottleneck / Fire units, 7 runs): 6 of 7 synced 2026-11-12 |
| 14 | `phase_14_winograd_numerics` | packed Winograd-FPGA `qat_wino` numerics (F23/F43/F63) on Phase 11's recipe; 14/14 FP32 and 48/48 arms done and synced 2026-11-09; medians arm − FP32: F23 −0.16, F43 −6.30, F63 −40.09 pp |

Phases 13 and 14 are **extra studies kept apart from Phase 11**: nothing under Phase 11 (yamls, figures, tests, log) reads their runs. A new experiment gets its own slug.
Phase 11's one recipe (`configs/experiments/_protocols/no_patience.yaml`): 500 ep SGD m0.9 lr 0.01 wd 5e-4 bs128, cosine, pad4-crop-flip + AutoAugment, He init, no early stopping; QAT 50 ep lr 1e-4 → 1e-6, BN intact, EMA observers. Every value is cited in the yaml and `docs/logs/PHASE11_LOG.md`.

## Artifacts

`outputs/` = raw per-run output; `results/` = curated tracked tree feeding `report/` and `presentation/`. Nothing else at top level holds artifacts.

- `outputs/{local,pcad,notebooks}/` splits by where the run happened; the tree is per experiment: `outputs/<runtime>/<experiment>/<model>/{checkpoints,logs,tensorboard,results}/`. Roll-up CSVs: `outputs/<runtime>/aggregates/`.
- Git-ignored: `*.pth`, logs, `*_best.pth.gz`. Tracked (analysis reads them): `metrics.json`, `config.yaml`, `resolved_config.json`, `*_summary.json`, `*_meta.json`, `*_logits.npz`, and the small `qat_*.pth.gz` INT8 artifact (VGG16 FC-head ones are over GitHub's 100 MB and ignored by pattern).
- **The laptop keeps no gitignored artifacts.** PCAD (`~/dnn_study`) holds every `*.pth` and log at the same relative path: `rsync -avP rsdsouza@gppd-hpc.inf.ufrgs.br:dnn_study/<relpath> <relpath>`. Compare by md5 before deleting a checkpoint. `archive_adamw_recipe/` is hidden here by git sparse-checkout (`git sparse-checkout disable` restores it) — a regenerated `results/runs_index.csv` therefore loses its 54 rows, merge them back from git. The laptop's clone is **partial** (`blob:limit=1m`, 2026-11-10): blobs ≥ 1 MB not in the working tree come from GitHub on demand (needs network; a sweep like `git ls-tree -l` over `outputs/` downloads every blob it touches). Reflogs never expire (`gc.reflogExpire=never`). A full copy needs a normal re-clone; PCAD's clone is full. Writing a tree (any commit, stash or rebase) prefetches every missing blob of the checked-out files (git's cache-tree prefetch; `GIT_NO_LAZY_FETCH=1 git write-tree` fails without them), so the working tree's 1582 blobs ≥ 1 MB stay local (one 8.6 GB pack since 2026-11-12, `.git` ~9 GB): deleting that pack only re-downloads it at the next commit.
- **One canonical result per model + phase + protocol** (more completed epochs wins for the same protocol; a different protocol is a separate experiment).
- **Superseded runs are archived, never left beside current ones**: `outputs/<runtime>/archive_<what>/` + `results/archive_<what>/`, each with a `SUPERSEDED.md`. New experiments get new names, so no current run shares a path with an archived one. `scripts/train.py` refuses to resume a run dir with another protocol; resubmitting a finished run only re-evaluates it.
- Syncing finished runs from PCAD: `rsync -avr --relative` of the run dirs, excluding `*.pth`, `*_best.pth.gz`, `*.log`, `tensorboard/`, `logs/`; sync only runs whose summary is complete. Then `python -m scripts.phase13.hybrids` (and `scripts.phase13.bottleneck_fire` for Stage B), `scripts.build_runs_index`, `scripts.build_cross_phase_results`.
- `git pull --ff-only` on PCAD stops on run files committed from the laptop that sit there untracked: move exactly those paths to `~/dnn_study_prepull_untracked_<date>/`, pull, `cmp` each against the checked-out copy, delete the copies. An aggregate CSV PCAD modified: copy aside, `git checkout --` it, pull, compare rows. Leave PCAD's `archive_adamw_recipe/` modifications and appended `aggregates/*_comparison.csv` rows alone.

## Layout

```
ml/          core package; notebooks and scripts import everything from here
             config, data, trainer, quantization, reporting, registry, model_registrations (keep in sync with notebooks),
             checkpoint, runtime, profiling, pruning, det_seg_*, distillation_trainer, winograd_bridge, yolov2, plotting
models/      architectures by phase: baselines, alexnet_variants, compensation, tinyhybridnet, final_architecture, vit_variants
configs/     YAML via configs/loader.py:load_config(name); runtime/ (local, pcad), slurm/ (partitions), experiments/ (+ _protocols/ extends-only fragments)
scripts/     CLI entry points: train.py, cluster.py, train_det_seg.py, profile_hardware.py, aggregate_results.py,
             build_runs_index.py, build_cross_phase_results.py; per-phase dirs phase6/ phase7/ phase9/ phase11/ phase12/ phase13/ winograd_fpga/ lab/ pcad/ slurm/ oneoff/
tests/       pytest (by file; see Running)
notebooks/   by phase slug (Phases 1–5, 7–10)
results/     curated tracked CSV/JSON/figures; figures_generated/phase_*/; results_aggregate/ (regenerated, never hand-edited)
outputs/     raw per-run artifacts
presentation/ report/   slides and LaTeX writeup (figures generated by their own scripts)
docs/        logs/ (dated, per phase), plans/ (BEST_MODELS, MODELS, PROTOCOL.tex/.pdf, per-phase plans), REFERENCE.md
```

## Stack

Python 3.12 · PyTorch 2.5.1+cu121 · torchvision 0.20.1 · torchmetrics · torchinfo · fvcore · wandb (offline-first, no auto-sync) · kagglehub. `.venv` from `requirements.txt` (the source of truth; neither laptop nor PCAD uses conda).
INT8 **accuracy** runs on **qnnpack** (`ml.quantization.ACCURACY_ENGINE`, exact int32 accumulation on any CPU); INT8 **latency** on onednn (`QUANT_ENGINE`). INT8 convert and inference are CPU-only.

## Key patterns

**Config:** dataclasses from YAML, override with `dataclasses.replace()`:
```python
data_cfg = DataConfig(**load_config("data.yaml"))
trainer_cfg = replace(TrainerConfig(**load_config("training.yaml")), lr=spec["lr"], epochs=2)
```
**Registry:** `register_model("alexnet_fp32", build_alexnet, fuse_map=[...], fuse_root_attr="features", lr=1e-4)`. `fuse_map` = dotted-path lists for Conv-BN(-ReLU) fusion (hand-written for flat nets, `find_fuse_groups(model())` for nested blocks). An unknown name in an experiment's `models:` fails `scripts/train.py`.
**Trainer:** `Trainer(model, train_loader, val_loader, cfg=..., device=..., save_dir=..., run_name=..., num_classes=200)`; `fit()` saves `{name}_best.pth` and reloads it before returning; `evaluate(topk=(1,5), save_logits=path)`; `benchmark()`. Skip / resume logic lives in the caller.
**QAT flow:** FP32 `fit` → `build_qat(name, save_dir, device)` (load best → copy → fuse → prepare_qat) → `fit(epoch_callback=make_qat_callback(...))` → `convert_to_int8(qat_model)` (eval, CPU) → `evaluate` on a CPU loader.
**QAT architecture rules:** ReLU `inplace=False`; residual adds via `nn.quantized.FloatFunctional()`; BN immediately after its Conv; the logits Linear keeps INT8 input/weights and an FP32 output.
**Data:** ImageFolder, deterministic 90/10 train/val split (seeded; the val split only picks the best epoch). **Test = Tiny ImageNet's official val split (10k)** — every stage reports `test_{fp32,qat,int8}_*` and writes `{model}_{stage}_test_logits.npz`; that is the number the papers use. `num_workers: 4` kept for every run on purpose.
**Reproducibility:** seed `random`/`numpy`/`torch`/`cuda`; `cudnn.deterministic=True`; never `cudnn.benchmark`. `scripts/train.py` turns TF32 off, checks the INT8 kernels against exact int32 before every INT8 evaluation, and fails a job whose `agreement_qat_int8` < 0.90. A number must not depend on the machine; summaries record host / GPU / CPU, and latency is compared only within a machine.
**Reporting:** `make_run_summary(..., extra=dict)` → one JSON per model; `build_comparison_table` → CSV; `compute_flops(model, input_size=(1,3,64,64))`. Every expensive run emits every metric (logits, layer stats, latencies, ECE) so a later question never needs a rerun.

## Running

```bash
source .venv/bin/activate
python -m scripts.train --experiment default --runtime local                 # classification FP32→QAT→INT8 (--smoke: 1 epoch/stage, output discarded)
python -m scripts.cluster submit --experiment <exp> --runtime pcad --slurm tupi_4090 --model <name>   # [--smoke] [--dry-run]
python -m scripts.cluster submit-sweep --experiment <exp> --runtime pcad --slurm tupi_4090             # one job per model
python -m scripts.cluster status <job_id>                                    # / cancel / resume
python -m scripts.train_det_seg detection --model alexnet_bottleneck --dry-run   # Phase 7
python -m scripts.profile_hardware --experiment phase_6_hardware_profiling --runtime local
pytest tests/test_config.py -q       # by file, never `pytest tests` whole (it crashed the laptop 2026-10-21); test_quantization in -k slices
```

Cluster: PCAD (`ssh rsdsouza@gppd-hpc.inf.ufrgs.br`, `~/dnn_study`) is Slurm only — inspection and submission, no compute on the login node. The `normal` QOS caps submitted jobs at 10. `scripts.cluster` refuses a real submit while `ml/models/scripts/configs` are dirty: commit, push and `git pull` on PCAD first. `tupi_4090` / `tupi_beagle` carry `exclude: tupi4` (kept 2026-11-06: signal deaths by node, `PHASE13_LOG` "2026-11-06"); a run killed by a signal on an i9 node is rerun from scratch, not resumed. Beagle (2× 1080 Ti, exclusive, ~2–3× slower) is the overflow: `--slurm tupi_beagle`. nyx (lab box) is off the network since 2026-10-31. Details, queues and feeders: `docs/REFERENCE.md` "Running".
Winograd-FPGA runs need `WINOGRAD_FPGA_ROOT` exported in the same shell (PCAD: `~/winograd_bridge_m7/avaliacao_redes`); `ml/winograd_bridge.py` is the only import path into that sibling repo.
