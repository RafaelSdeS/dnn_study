# CLAUDE.md — alexnet_rafael

## Workflow

For non-trivial changes: inspect the relevant files, give a short plan, wait for approval. Don't edit immediately. Never run notebooks/training or commit unless explicitly asked.

## What this is

Deep-learning research on how **convolutional kernel-size restriction** affects CNNs. Motivation: **Winograd accelerators** are efficient for small kernels (2×2, 3×3) but scale poorly for large filters. We measure the accuracy/efficiency trade-off to recommend Winograd-friendly architectures.

**Scope:** Classification on **Tiny ImageNet-200** (64×64 RGB, 200 classes), FP32 training → **QAT → INT8**. Phases 1–3 compare AlexNet-style vs. efficient (MobileNet-style) architectures; Phase 4 builds and compresses final hybrid designs; Phase 5 is cross-phase results analysis; Phase 6 profiles hardware (latency/power/GPU utilization) for the best models; **Phase 7 (done/ongoing)** tests whether classification's compensation findings transfer to dense prediction — SSD detection + segmentation on PASCAL VOC (`ml/det_seg_data.py`, `det_seg_models.py`, `det_seg_trainer.py`, driven by `scripts/train_det_seg.py`, results under `outputs/pcad/phase_7_detection_segmentation/`). **Phase 8 (done)** asks whether local self-attention (windowed Swin / ViT) matches small-kernel CNNs' accuracy/efficiency/quantization profile — 7 models (`models/vit_variants.py`) covering H1 (window-size sweep), H2 (CNN-stem + attention hybrid), H3 (quantization robustness), H4 (DeiT distillation), H5 (Winograd-eligibility); see `docs/plans/PHASE8_PLAN.md` and `docs/logs/PHASE8_LOG.md` for the plan and build history, including the D6 QAT-for-attention revision found during implementation.

Questions: how kernel size affects accuracy/efficiency/quantization robustness; whether small-kernel CNNs match pretrained models; the FP32→INT8 drop per architecture family; whether classification-derived compensation mechanisms (bottleneck, Fire, depthwise) transfer to detection/segmentation.

---

## Layout

**One rule for artifacts:** `outputs/` is raw per-run output, `results/` is the curated
tracked tree that feeds `report/`/`presentation/`. Nothing else at the top level holds
artifacts — the old top-level `checkpoints/` is now `outputs/notebooks/`.

`outputs/{local,pcad,notebooks}/` splits by *where the run happened* (large checkpoints move
between machines by hand via scp/rsync, not git). Inside it, weights (`*.pth`) and logs are
gitignored; provenance records (`metrics.json`, `config.yaml`, `git_hash.txt`,
`*_summary.json`, `*_meta.json`, and the small `*.pth.gz` INT8 artifacts) are tracked, because
the analysis notebooks read them. The word "results" never appears at the top of an `outputs/`
runtime — that roll-up dir is `outputs/<runtime>/aggregates/`.

**One slug per phase**, `phase_N_description`, used identically in `notebooks/`, `results/`,
`results/figures_generated/`, `outputs/*/` **and `configs/experiments/`**. The experiment name
*is* the output directory name (`outputs/<runtime>/<experiment>/<model>/`), so keeping the
config filenames on the same slug is what stops the naming drift from coming back — rename the
experiment, not the folder it produced.

**One canonical result per model+phase+protocol.** Two runs of the same model under the *same*
protocol: keep the one with more completed epochs before early stopping. A run under a
*different* protocol (different phase, different epoch budget/patience) is not a duplicate to
epoch-compare — it is a separate experiment, labeled with its own phase slug, never filed into
another phase's slot. `scripts/build_runs_index.py` reads a backfilled run's `source`, not its
directory, for exactly this reason.

| Phase | slug |
|-------|------|
| 1 | `phase_1_baseline` |
| 2 | `phase_2_kernel_restriction` |
| 3 | `phase_3_compensation_and_hybrids` |
| 4 | `phase_4_compression_and_final_architecture` |
| 5 | `phase_5_cross_phase_analysis` |
| 6 | `phase_6_hardware_profiling` |
| 7 | `phase_7_detection_segmentation` (experiments `phase_7_detection` / `_segmentation` / `_smoke`) |
| 8 | `phase_8_efficient_vit` (+ `phase_8_efficient_vit_convstem`) |
| 9 | `phase_9_bypass_ablation` (+ `_large_scale`, `phase_9_pruning`, `phase_9_compression`) |
| 10 | `phase_10_final_summary` |
| 11 | `phase_11_kernel_size_comparison` |

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
  trainer.py              # Trainer: fit(), evaluate(save_logits=path) -> adds ece + writes {model}_{stage}_val_logits.npz
                          #   (float16 logits + labels) so cross-stage/agreement/calibration questions never need a
                          #   rerun, benchmark(device=...) -> override self.device for one call (e.g. CPU latency
                          #   alongside GPU), both added 2026-09-13 for Phase 11's metrics-per-run requirement
  distillation_trainer.py # Phase 8 H4: DistillationTrainer (hard-label KD from a frozen teacher, deit_tiny only)
  quantization.py         # find_fuse_groups, build_qat, convert_to_int8, load_best_model, make_qat_callback;
                          #   Phase 8 D6: exclude_attention_from_qat (LayerNorm/ShiftedWindowAttention/MultiheadAttention
                          #   -> qconfig=None), swap_quantizable_mha (kept for correctness, not on the QAT path — see D6)
  quantization_advanced.py# Mixed-precision / sub-INT8 PTQ: make_qconfig, prepare_sim, calibrate,
                          #   compute_layer_sensitivity, assign_mixed_precision, apply_weight_ptq, theoretical_size_mb
  profiling.py            # Phase 6: GpuSampler (nvidia-smi power/util/temp/mem sampling), latency/throughput profiling
  pruning.py              # Phase 9: prune_model_channels — structured (whole-channel) pruning, stays Winograd-dense
  runtime.py              # Shared CLI plumbing: set_global_seed, capture_provenance (also records torchvision/CUDA/
                          #   cuDNN/Python versions, GPU name, cpu_count, SLURM_JOB_ID since 2026-09-13), load_profile
                          #   (experiment/runtime yaml by name or path), ensure_dataset_path, make_model_runs
                          #   (<root>/<exp>/<model>/...), save_resolved_config — scripts import these from `ml`, not
                          #   from scripts/train.py's privates
  winograd_bridge.py      # The ONLY import path into the sibling Winograd-FPGA repo ($WINOGRAD_FPGA_ROOT): study-model
                          #   ctors (custom_model/torchvision_model — geometry owned there, so checkpoints load 1:1 in
                          #   its Fase 2.5), the qat_wino stage (load_qat_wino_model), bridge_provenance (its commit)
  reporting.py            # build_comparison_table, create_results_summary, disk_mb, compute_flops, make_run_summary
                          #   (extra=dict merged in last, so callers add fields without inflating the signature);
                          #   expected_calibration_error, prediction_agreement(logits_a, logits_b) -> top-1 agreement
                          #   fraction from two save_logits .npz files, layer_stats(model, loader, device) -> per
                          #   Conv/Linear geometry+MACs+weight range+activation percentiles (the calibration data
                          #   Winograd-FPGA export needs), all added 2026-09-13
  plotting.py             # Figure style for report/ + notebooks/: palette, GROUP_COLORS/MODEL_GROUP, apply_report_style()
                          #   (presentation/make_figures.py keeps its own slide palette on purpose)
models/                   # Architectures by phase (see Model Inventory)
  baselines.py alexnet_variants.py compensation.py tinyhybridnet.py final_architecture.py vit_variants.py
configs/                  # YAML hyperparameters, loaded via configs/loader.py → load_config(name)
  data.yaml training.yaml qat.yaml qat_wino.yaml profiling.yaml compression.yaml detection.yaml segmentation.yaml
  runtime/                # local.yaml, pcad.yaml — dataset root, conda env, per-runtime toggles
  slurm/                  # single_gpu.yaml, tupi_4090.yaml, beagle.yaml — partition/GPU/CPU/wall-time
  experiments/            # default.yaml + per-run overrides (alexnet_3x3_gap, phase_7_detection, large_scale, phase8, ...);
                          #   budget_unico.yaml = Winograd-FPGA study Fase 2 (14 *_fpga models, stages fp32+qat_wino,
                          #   uniform_hparams, via extends: _protocols/winograd_fpga); an unknown name in any
                          #   `models:` list now fails scripts/train.py (and tests/test_registry.py) instead of
                          #   being silently dropped. tests/test_config.py also fails any `models:`-style
                          #   experiment file that doesn't `extends:` a _protocols/*.yaml fragment — phase_7_*.yaml
                          #   below are the deliberate exception (different schema, different script)
                          #   `--smoke` on scripts/train.py and scripts/train_det_seg.py caps every stage (fp32/qat/
                          #   qat_wino) to 1 epoch for a fast local pipeline check, superseding the old per-phase
                          #   smoke config files; on train_det_seg.py it runs the whole fp32->qat->int8 chain.
                          #   Smoke output (checkpoints/logs/tensorboard/resolved_config.json/aggregates CSV) is
                          #   written to a temp dir and discarded on exit -- never touches outputs/, wandb disabled
                          #   `extends: _protocols/<name>` (load_config, one level, dict-valued keys
                          #   merge field-by-field) lets a file inherit a shared protocol instead of
                          #   repeating it — e.g. large_scale.yaml/alexnet_dilated_gap.yaml/
                          #   phase_9_bypass_ablation_large_scale.yaml/large_scale_fire_residual_resume.yaml
                          #   all extend _protocols/large_scale.yaml (1000ep/patience 50/QAT 100ep);
                          #   phase_8_efficient_vit(_convstem).yaml extend _protocols/phase_8_vit.yaml;
                          #   alexnet_3x3_fc/alexnet_3x3_gap/default/phase_9_bypass_ablation.yaml extend
                          #   _protocols/standard.yaml (just seed: 42 + stages: [fp32, qat, int8]);
                          #   budget_unico.yaml extends _protocols/winograd_fpga.yaml;
                          #   phase_11_kernel_size_comparison.yaml extends _protocols/no_patience.yaml (seed 42,
                          #   uniform_hparams, 500ep FP32/100ep QAT, early_stopping_patience: null — the QAT stage
                          #   inherits null too, since scripts/train.py builds its cfg via replace() off the same base)
                          #   phase_7_detection.yaml/phase_7_segmentation.yaml are deliberately NOT this
                          #   schema (no models:/extends:) — consumed by scripts/train_det_seg.py, which reads
                          #   data.num_workers/trainer.epochs directly; different task, different loader
    _protocols/            # extends-only fragments (no models:/name: — not runnable, excluded from
                          #   _experiment_names()'s non-recursive glob): large_scale.yaml, phase_8_vit.yaml,
                          #   standard.yaml, winograd_fpga.yaml, no_patience.yaml
scripts/                  # CLI entry points (used instead of notebooks for PCAD/cluster runs)
  train.py                # `python -m scripts.train --experiment ... --runtime local|pcad` — classification FP32→QAT→INT8
  cluster.py               # `python -m scripts.cluster submit|submit-sweep|status|cancel|resume` — submits
                           #   slurm/train.sbatch or profile.sbatch; `submit --smoke` (2026-09-13) caps epochs to 1
                           #   and discards output, for a fast real-cluster pipeline check before a full submission
                           #   (catches env/bridge issues --smoke on scripts/train.py alone can't, since that only
                           #   runs locally)
  train_det_seg.py         # Phase 7 detection/segmentation CLI, mirrors train.py; one run() for both tasks,
                           #   task-specific pieces (builders, loaders, trainer, int8 metrics) in its TASKS table
  profile_hardware.py      # Phase 6 hardware profiling CLI
  aggregate_results.py     # Aggregates per-model summary JSONs from a cluster submit-sweep into one CSV,
                           #   written to the curated results/<experiment>/ tree
  build_runs_index.py      # `python -m scripts.build_runs_index` — scans every run layout under outputs/
                           #   (train.py, train_det_seg.py, notebooks, profile_hardware.py) into one row-per-run
                           #   results/runs_index.csv, without unifying the four writer layouts themselves.
                           #   Trusts a backfilled *_meta.json's own `source` field over the directory it was
                           #   filed under, so a run imported from a different phase/protocol can't be indexed
                           #   as that directory's phase (see CLAUDE.md's "one canonical result" rule above)
  build_cross_phase_results.py # `python -m scripts.build_cross_phase_results` — rolls every curated
                           #   results/phase_*/*_summary.json (+ Phase 8's phase8_comparison.csv) into
                           #   results/results_aggregate/{results,model_details}_cross_phase.csv; idempotent,
                           #   replaces hand-maintaining those two files
  # --- everything below is grouped, so `ls scripts/` shows the 7 entry points above ---
  # A `[one-off]` prefix on a script's docstring means it was a historical fixup, already applied,
  # kept for provenance — not part of the reproducible pipeline. Untagged = pipeline, re-runnable.
  # `grep -rl '"""\[one-off\]' scripts/`
  phase6/                  # `python -m scripts.phase6.<name>`
    winograd_quant_error.py    # Phase 6 extension: INT8 quantization error from Winograd F(2x2,3x3) transforms
    phase6_eixo3_stats.py      # Eixo 3 statistics over the profiling runs
  phase7/            # Phase 7 one-off diagnostics / backfill tools (`python -m scripts.phase7.<name>`)
    check_anchor_recall.py / backfill_gzip.py / backfill_int8_size.py / backfill_int8_size_segmentation.py
    diagnose_segmentation_quality.py / measure_true_model_size_detection.py / measure_true_model_size_segmentation.py
  phase9/                  # Phase 9 CLIs (`python -m scripts.phase9.<name>`)
    measure_compression.py # Task 3: entropy/k-means weight-compression headroom above plain gzip;
                           #   --evaluate actually clusters the weights and measures real accuracy vs. FP32
    prune_channels.py      # Task 2: structured (channel) pruning CLI;
                           #   --finetune-epochs fine-tunes the pruned model then runs it through QAT->INT8
  oneoff/                  # Retired one-shot fixups, kept for provenance (`python -m scripts.oneoff.<name>`)
    backfill_model_size.py / backfill_best_epoch_eval.py / dilated_gap_local.py
  pcad/                    # PCAD submission wrappers
    migrate_pcad_gitignored.sh  # merges gitignored artifacts (*.pth, *.log) left in pre-reorg folder names after a pull
    submit_phase_7_simple.sh / submit_phase_7_multinode.sh  # PCAD Phase 7 detection + segmentation submission
                                #   (simple vs FP32→QAT→INT8 chaining; TASK=segmentation env var / positional
                                #   arg selects the task; --pretrained-ckpt is detection-only) — see docs/logs/PHASE7_MULTINODE.md
    preflight_budget_unico.py   # `python -m scripts.pcad.preflight_budget_unico` — sanity check for
                                #   configs/experiments/budget_unico.yaml before burning a PCAD allocation (run it on
                                #   PCAD too): bridge importable, every model builds, bridge commit recorded + clean.
                                #   On PCAD the bridge is the tarball from Winograd-FPGA's
                                #   scripts/package_avaliacao_bridge_for_pcad.sh (writes BRIDGE_COMMIT.json). It was
                                #   never synced there until 2026-09-13 (preflight caught all 14 *_fpga models
                                #   failing to construct) — package + rsync it straight to PCAD with
                                #   `scripts/package_avaliacao_bridge_for_pcad.sh user@host:winograd_bridge`, then
                                #   `export WINOGRAD_FPGA_ROOT=~/winograd_bridge/avaliacao_redes` in the SAME shell
                                #   you run `scripts.cluster submit(-sweep)` from (its `--export=ALL` is what
                                #   propagates the var into the job) — re-sync whenever Winograd-FPGA's bridge files change.
  winograd_fpga/            # `python -m scripts.winograd_fpga.<name>` (2026-09-13)
    dump_layer_configs.py      # Emits scripts/avaliacao_redes/layer_configs/*.json in the sibling Winograd-FPGA
                               #   repo, in its layer_configs.LAYER_CONFIGS schema, for every budget_unico *_fpga
                               #   model + phase_11's alexnet_tv_3x3/vgg16, across f23/f43/f63 (via net_manifest.
                               #   to_manifest + eligibility_wino.audit_net — geometry only, RTL sim always uses
                               #   synthetic weights) — lets Winograd-FPGA's run_sim*.py throughput-sim ANY network
                               #   trained here, not just its bundled VGG16. Asserts net_manifest.vgg16() still
                               #   matches LAYER_CONFIGS before writing anything. See run_vu9p_redes.sh there
                               #   (gates on reproducing the published VU9P GOPS before touching the 16 networks).
  slurm/*.sbatch           # sbatch templates — train.sbatch/profile.sbatch submitted by cluster.py, det_seg.sbatch by the pcad/submit_phase_7_*.sh scripts, others called directly.
                          # train.sbatch fixed 2026-09-13: conda never activates in a non-interactive Slurm batch
                          #   shell (conda init lives in ~/.bashrc, which such shells don't source) — every real
                          #   submission died in ~1s as "python: command not found" (jobs 821240/821241, on both
                          #   beagle and tupi). Switched to `cd "$(git rev-parse --show-toplevel)" && source
                          #   .venv/bin/activate`, the pattern det_seg.sbatch/prune_channels.sbatch already needed
                          #   after hitting the identical failure. profile.sbatch/measure_compression.sbatch/
                          #   notebook.sbatch still have the same latent bug, not yet fixed.
tests/                    # pytest: test_registry, test_checkpoint, test_config, test_trainer_smoke,
                          #   test_quantization, test_profiling, test_train_cli, test_train_det_seg_cli,
                          #   test_train_pipeline (run_experiment end to end; a stop signal ends the run
                          #   instead of rolling into the next stage on a truncated model)
notebooks/                # Organized by phase + purpose
  phase_1_baseline/                          # baselines_qat
  phase_2_kernel_restriction/                # alexnet_qat
  phase_3_compensation_and_hybrids/          # compensation_qat, efficient_hybrids_qat
  phase_4_compression_and_final_architecture/ # compression_phase4_1, final_architecture_qat, final_architecture_results
  phase_5_cross_phase_analysis/               # final_analysis_phase5
  phase_6_hardware_profiling/                # hardware_profiling_phase6
  phase_7_detection_segmentation/            # phase7_results_analysis
  phase_8_efficient_vit/                              # vit_qat_phase8 (vit_tiny/deit_tiny FP32→distill/QAT→INT8 —
                                                       #   the 2 of 7 Phase 8 models scripts/train.py can't drive, see D6),
                                                       #   phase8_results_analysis (Task 7 cross-phase analysis)
  phase_9_bypass_ablation/              # phase9_ablation_analysis
  phase_10_final_summary/                             # final_summary, training_dynamics — cross-project rollup of
                                                       #   Phases 1-4/8/9 (classification+detection+segmentation);
                                                       #   NOT the same "Phase 10" as TODO.md's future NAS section
results/                  # tracked CSVs/JSON/figures, one dir per phase slug + results_aggregate/
                          #   figures_generated/phase_*/ — every analysis notebook's FIGURES_DIR now points
                          #   straight at its own phase subdir, so a rerun lands in the right place with no
                          #   hand-sorting (this used to be manual, and the phase dirs went stale as a result)
                          #   results_aggregate/results_cross_phase.csv and model_details_cross_phase.csv
                          #   are regenerated by `python -m scripts.build_cross_phase_results` from the
                          #   curated per-model summary JSONs — don't hand-edit; rerun it
presentation/             # slides.md/slides.pdf + figures/ (generated by presentation/make_figures.py)
report/                   # LaTeX writeup: ic_report.tex/.pdf, figures/ (generated by generate_figures.py,
                          #   generate_architecture_figures.py)
docs/                     # Research docs, split by kind (was research/ until 712a63b — old commits/notes may say research/)
  logs/                   # Documentation (flat): PHASE7_QUICKSTART.md, PHASE7_MULTINODE.md, PHASE7_LOG.md, PHASE8_LOG.md
  plans/                  # Research notes (flat):
    BEST_MODELS.md        #   cross-phase rankings & recommendations
    MODELS.md             #   architecture notes & design rationale
    EFFICIENCY_IDEAS.md   #   candidate efficiency techniques not yet executed
    WINOGRAD_FPGA_BITWIDTH_PLAN.md  # satellite study for a separate repo (Winograd-FPGA), not a dnn_study phase
    PHASE6_PLAN.md PHASE7_PLAN.md PHASE8_PLAN.md PHASE9_PLAN.md  # research & execution plans (6/7/8/9 executed)
outputs/                  # Raw per-run artifacts & logs — exactly three children, by where the run happened
  pcad/                   # PCAD/SLURM cluster runs
    phase_6_hardware_profiling/ # GPU profiling data (runs/ — flat, host-named files; backfill/)
    phase_7_detection_segmentation/ # SSD/segmentation runs, one dir per model+stage+config (+ logs/)
    phase_8_efficient_vit/      # Phase 8 CLI-driven runs (swin_pico_*, hybrid_bottleneck_swin)
    phase_9_bypass_ablation/    # Phase 9 runs (fire_bypass/, fire_bypass_large_scale/)
    archive_legacy_phases/      # Phase 2, 4, 5 runs (phase_2_kernel_restriction/, phase_4_5_large_scale/); only *_best.pth kept, no *_resume.pth
    logs/<experiment>/          # SLURM job stdout/stderr, one dir per experiment name (matches scripts/cluster.py's log_dir)
    aggregates/                 # Per-experiment comparison CSVs written by scripts/train.py
  local/                  # Local machine runs (same shape as pcad/)
  notebooks/              # Notebook-era runs, Phases 1-4/8 (was the top-level checkpoints/):
                          #   flat {arch}_best.pth + {arch}_meta.json per phase slug
```

Per-run layout under `outputs/<runtime>/<experiment>/<model>/`: `checkpoints/`, `logs/`,
`tensorboard/`, `results/` (that innermost `results/` is the raw per-model summary JSON — the
curated tree is the top-level `results/`; `scripts/aggregate_results.py` reads the former and
writes the latter).

Runtime artifacts (git-ignored): `{arch}_best.pth`, `qat_{arch}_best.pth`, `{arch}.pth` (INT8 —
the `.pth.gz` compressed copy from `ml/checkpoint.py` *is* tracked); logs `{arch}.log`, `qat_{arch}.log`.

After a `git pull` on a machine that still has artifacts under old folder names, run
`scripts/pcad/migrate_pcad_gitignored.sh` — it is idempotent and covers every rename to date.

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
                                     #   and reloads it into self.model before returning
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

**Reporting:** `make_run_summary(..., extra=dict)` builds a 30+ field dict per model → save one JSON each (crash-safe); `extra` merges in per-run additions without inflating the signature. `build_comparison_table` → `final_comparison.csv`; `create_results_summary` → `experiment_summary.json`. `compute_flops(model, input_size=(1,3,64,64))` → `{macs, flops}`. W&B: `wandb.init(project=..., config=asdict(cfg), mode="offline")`, sync later with `wandb sync --sync-all`; no auto-sync.

**Metrics-per-run (2026-09-13, `scripts/train.py`):** every stage (FP32/QAT/INT8/`qat_wino`) saves `{model}_{stage}_val_logits.npz` via `Trainer.evaluate(save_logits=...)` and reports `ece`; QAT gets its own fake-quant accuracy on a **deepcopy** with observers/BN forced off (`tq.disable_observer` + `torch.nn.intrinsic.qat.freeze_bn_stats`) — evaluating the live QAT model directly would recalibrate its observers from val data and change what `convert_to_int8` produces next. FP32/INT8 also get bs1 and (when training was on GPU) CPU latency via `Trainer.benchmark(device=...)`. `ml.reporting.layer_stats(model, loader, device)` dumps per-Conv/Linear geometry/MACs/weight range/activation percentiles to `{model}_layer_stats.json` — the calibration data Winograd-FPGA's export needs, captured once from the saved checkpoint. `prediction_agreement` cross-stage. The point: an expensive run (PCAD, hours) should never need a rerun to answer a later accuracy/calibration question.

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
| 8 — Efficient ViT / hybrid-attention | `models/vit_variants.py` | vit_tiny, deit_tiny (H4 distillation), swin_pico_{w2,w4,w8} (H1 window sweep), swin_pico_poolmixer (H5 cross-check), hybrid_bottleneck_swin (H2) — 5 of 7 train via `scripts/train.py --experiment phase_8_efficient_vit`; vit_tiny/deit_tiny need `notebooks/phase_8_efficient_vit/vit_qat_phase8.ipynb` (deit_tiny's `DistillationTrainer` stage; see D6 for why their QAT stage no longer needs anything special) |
| Winograd-FPGA study (`budget_unico`) | none here — `ml/winograd_bridge.py` builds them from the sibling repo | 15 `*_fpga` models registered in `ml/model_registrations.py`: vgg_style, alexnet_{3x3_fc, stacked, fire, fire_bypass, bottleneck, final_fire_residual, final_bottleneck_residual}, repvgg_a0 (trained raw), wrn_{16_4, 28_2}, googlenet, resnet18, vgg13 — plus squeezenet1_1, registered but out of budget_unico (qat_wino breaks on its 15×15 maps). Add a study model in the sibling repo, then one `register_model(..., custom_model/torchvision_model(...))` line here |
| 11 — Kernel size comparison | `models/baselines.py` | `AlexNetTV(kernel_size=None\|3\|2)` (original 11×11/5×5/3×3, 3×3, 2×2, no BN) and `VGG16(kernel_size=3\|2)` (torchvision cfgs["D"] + BatchNorm -- plain (no-BN) VGG16 from scratch measured stuck at ln(200) loss for 22 epochs on PCAD, 2026-09-13; kernel_size=3 is VGG's own native design) — all 5 trained from scratch, no early stopping, via `configs/experiments/phase_11_kernel_size_comparison.yaml` (`_protocols/no_patience.yaml`: 500ep FP32 / 100ep QAT) |

**Results & rankings:** see `docs/plans/BEST_MODELS.md` (Pareto tiers, recommendations, now covering Phases 1–4/6/7/8/9) and `results/results_aggregate/results_cross_phase.csv` / `results/results_aggregate/model_details_cross_phase.csv`. Headlines: MobileNetV2 best overall (~58% top-1) among Phase 1–3 models, though Phase 4's AlexNetFinalFireResidual (49.79%) and Phase 9's AlexNetFireBypass (50.57%) close most of the gap — the latter now *exceeds* the full hybrid's FP32 gain outright (+6.59pp vs. +5.81pp over AlexNetFire) — at a fraction of the size; AlexNetBottleneck/AlexNetFire remain Pareto-optimal on efficiency (43–44%, 1.5–2 MB, quantization-stable). A size-reporting bug (fixed 2026-09-02, `ml/reporting.py`) had `disk_mb()`/`gzip_mb()` measuring the raw `{model}_best.pth`, which carries AdamW optimizer state (~3× the weights), while the INT8 artifact was already weights-only — so every FP32 size and FP32-vs-INT8 compression ratio was inflated ~3× (~11.9× recorded vs. the true ~4×). Both sides now measure `model_state_dict`; summaries and CSVs backfilled via `scripts/oneoff/backfill_model_size.py`. Accuracies, params, MACs and all rankings are unaffected; the analysis notebooks were re-run on 2026-09-12, so only `phase9_ablation_analysis.ipynb` (its PCAD summary inputs are gone) and the training notebooks' output cells still show pre-fix sizes. Known issues: AlexNetSmallKernel severe QAT drop (~–10pp), AlexNetSE training failure. A `Trainer.fit()` bug (fixed 2026-08-29, `ml/trainer.py`) returned the last epoch's model instead of reloading the best checkpoint, so FP32 was evaluated on different weights than INT8 — spurious INT8 "gains" of up to +6.5pp for runs with a long post-peak tail; backfilled via `scripts/oneoff/backfill_best_epoch_eval.py` for the 5 CLI-trained Phase 8 models plus AlexNetFireBypass (FP32 corrected for all 6; INT8 only rebuilt where a full-precision QAT-best checkpoint survived — FireBypass and `vit_tiny`/`deit_tiny` were otherwise unaffected). See `report/ic_report.tex` Eixo 4/7 for the corrected findings. Phase 7 detection: anchor-recall root cause fixed and A4 retrain complete on PCAD — all 3 backbones (bottleneck/fire/tv) × FP32/QAT/INT8 × plain/pretrained now have valid mAP (see `docs/plans/BEST_MODELS.md`). Phase 7 segmentation: PCAD runs now complete for all 3 backbones × FP32/QAT/INT8 (`outputs/pcad/phase_7_detection_segmentation/seg_*`); not yet folded into the H1–H4 analysis notebook. Phase 7 hypotheses (H1–H4, does compensation transfer to dense prediction) and progress: `docs/plans/PHASE7_PLAN.md`, `docs/logs/PHASE7_LOG.md`. Phase 8: all 7 models trained on PCAD, results in — H1 (window-size sweep) and H4 (DeiT distillation) confirmed, H3 (quantization robustness) inverted (5 of 7 models gain accuracy under INT8), H5 (Winograd-eligibility) confirmed but not attention-specific (no model has a stride-1 3×3 conv). D6's QAT-for-attention revision (swap_quantizable_mha can't drive this codebase's eager-mode `prepare_qat()`, so attention stays FP32-excluded like Swin's fallback) and full H1–H5 detail are in `docs/plans/PHASE8_PLAN.md` and `docs/logs/PHASE8_LOG.md`.

---

## Running

**Notebooks** (Phases 1–5, exploratory):
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # first time / resync
source .venv/bin/activate
jupyter lab
```
`requirements.txt` (pip freeze, tightly pinned) is the source of truth for `.venv` — it has drifted before (`docs/logs/PHASE7_LOG.md`); re-run the `pip install -r requirements.txt` line if notebook imports start failing. `environment.yml` predates the switch to `.venv` below; `configs/runtime/*.yaml`'s `conda_env` field and `cluster.py`'s `CONDA_ENV_NAME` export are similarly vestigial for `train.sbatch` (fixed 2026-09-13) but still read by `profile.sbatch`/`measure_compression.sbatch`/`notebook.sbatch`, which still have the old (broken) conda-activation block.
Tiny ImageNet-200 downloads via `kagglehub` on first run (cached in `~/.cache/kagglehub/`). Before INT8 convert/inference: `model.eval()` and move to CPU.

**CLI / cluster runs** (Phases 6–8, reproducible local or PCAD SLURM runs). Despite `environment.yml`'s
name, actual practice — here and on PCAD — is a plain `.venv` (`python3 -m venv .venv && source
.venv/bin/activate && pip install -r requirements.txt`); neither machine has `conda` installed, and
`scripts/slurm/train.sbatch` was fixed 2026-09-13 to activate `.venv` directly instead of trying `conda`
(see its file entry above):
```bash
python -m scripts.train --experiment default --runtime local        # classification, local
python -m scripts.cluster submit --experiment default --runtime pcad --slurm single_gpu
python -m scripts.cluster submit --experiment phase_11_kernel_size_comparison --runtime pcad --slurm tupi_4090 --model vgg16 --smoke   # fast real-cluster check, one model, output discarded
python -m scripts.cluster submit-sweep --experiment phase_11_kernel_size_comparison --runtime pcad --slurm tupi_4090   # one job per model, real budget (no --smoke)
python -m scripts.cluster submit-sweep --experiment phase_8_efficient_vit --runtime pcad   # one job per model, Phase 8's 5 CLI-drivable models
python -m scripts.cluster submit --experiment budget_unico --runtime pcad --slurm tupi_4090 --model wrn_16_4_fpga   # one model only -- needs WINOGRAD_FPGA_ROOT exported first, see the preflight_budget_unico.py entry above
python -m scripts.cluster submit-sweep --experiment budget_unico --runtime pcad --dry-run   # print the sbatch commands, submit nothing
python -m scripts.cluster status <job_id>   # / cancel / resume
python -m scripts.train_det_seg detection --model alexnet_bottleneck --dry-run   # Phase 7
python -m scripts.profile_hardware --experiment phase_6_hardware_profiling --runtime local           # Phase 6
python -m scripts.winograd_fpga.dump_layer_configs   # emit layer_configs JSONs for the sibling repo's VU9P throughput sim
```
Edit `configs/runtime/pcad.yaml` (dataset root) and `configs/slurm/single_gpu.yaml` (partition/GPU/wall-time) for cluster settings; duplicate `configs/experiments/default.yaml` for a new reproducible run.

**Tests:** `pytest tests/`
