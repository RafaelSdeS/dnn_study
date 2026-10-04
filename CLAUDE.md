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

**Superseded runs are archived, never left beside current ones (2026-10-03).** A protocol change moves the old runs to
`outputs/<runtime>/archive_<what>/` (and their curated tables/figures to `results/archive_<what>/`), each with a
`SUPERSEDED.md` saying what replaced them -- the same path on the laptop (git) and PCAD. `build_runs_index` gives runs
under such a dir `superseded=<dir>` and no phase; `build_cross_phase_results` and `scripts/phase11/*` only read the live
`phase_*` dirs (`design_figures --archive` excepted: a preview of figures 16-21 on the archive, written under it); `scripts/train.py:refuse_foreign_run_dir` stops instead of resuming a run dir whose
`resolved_config.json` has another protocol (data/training/qat/seed). New experiments and models get new names, so no
current run ever shares a path with an archived one. Phase 11's old recipe: `outputs/pcad/archive_adamw_recipe/`.

**The laptop keeps no gitignored artifacts (since 2026-09-16).** PCAD is the only copy of every
`*.pth`/log from all three runtimes, at the same relative path under `~/dnn_study`. Fetch one when
needed: `rsync -avP rsdsouza@gppd-hpc.inf.ufrgs.br:dnn_study/<relpath> <relpath>`. Before deleting
a checkpoint on either side, compare by md5, not size — same-model checkpoints have identical sizes.

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
  config.py               # DataConfig, TrainerConfig, QATConfig dataclasses (defaults explicit);
                          #   QATWinoConfig(QATConfig) (2026-09-19) adds variant/pack/u_w/v_w/k_dsp --
                          #   WHICH Winograd-FPGA accelerator line the qat_wino stage trains against.
                          #   Until 2026-09-18 that stage always trained F(4,3) with no packing, so
                          #   the 14 budget_unico accuracy runs are that one combination (≠HW: the
                          #   deploy bitstream packs) -- see the M7 note on the Winograd-FPGA study row
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
                          #   alongside GPU), both added 2026-09-13 for Phase 11's metrics-per-run requirement;
                          #   frozen_observers (2026-09-30): _validate/evaluate no longer calibrate QAT observers on val data;
                          #   evaluate's top1/top5 are micro (standard) since 2026-09-30 -- macro before, up to ~0.2pp off;
                          #   benchmark (2026-10-03) times only model(x) on one batch already on the device
                          #   (torch.utils.benchmark median + IQR, fixed threads) -- before, it timed the DataLoader too
  distillation_trainer.py # Phase 8 H4: DistillationTrainer (hard-label KD from a frozen teacher, deit_tiny only)
  quantization.py         # find_fuse_groups, build_qat, convert_to_int8, load_best_model, make_qat_callback;
                          #   prepare_qat_model re-locates fuse_root inside its deep copy (fixed 2026-09-30: it used to fuse
                          #   the caller's original, leaving every fuse_root_attr model's QAT unfused), and wraps every avg
                          #   pool as DeQuantStub->pool->QuantStub (requantize_avg_pools, 2026-09-30: eager INT8 pooling kept
                          #   its input's coarse scale, costing GAP models 2-3pp that QAT never simulated); load_int8_model
                          #   rebuilds the INT8 state_dict train.py saves (the pickled module saved before 2026-09-30 can't load);
                          #   INT8_QAT_QCONFIG (2026-10-03) is the ONE INT8 definition, literature-standard: 8-bit per-tensor
                          #   affine activations (Jacob 2018; fbgemm's default reduce_range made them 7-bit before) +
                          #   per-channel symmetric weights in [-127,127] (Wu 2020); QUANT_ENGINE="onednn" runs it, set by
                          #   convert_to_int8 itself (no runtime yaml key any more). fuse_sequential_relus fuses every
                          #   Conv/Linear-ReLU left in a Sequential (FC heads were unfused, except vgg16's), residual blocks use
                          #   FloatFunctional.add_relu. The logits Linear (keep_logits_float/_FloatLogits) has INT8 input and
                          #   weights and an FP32 output (Wu 2020: no quantized layer reads it; an 8-bit logits grid tied 14-31%
                          #   of top-1). Summaries carry quant_protocol=QUANT_PROTOCOL; analysis treats any other as superseded
                          #   Phase 8 D6: exclude_attention_from_qat (LayerNorm/ShiftedWindowAttention/MultiheadAttention
                          #   -> qconfig=None), swap_quantizable_mha (kept for correctness, not on the QAT path — see D6)
  quantization_advanced.py# Mixed-precision / sub-INT8 PTQ: make_qconfig, prepare_sim, calibrate,
                          #   compute_layer_sensitivity, assign_mixed_precision, apply_weight_ptq, theoretical_size_mb
  profiling.py            # Phase 6: GpuSampler (nvidia-smi power/util/temp/mem sampling), latency/throughput profiling
  pruning.py              # Phase 9: prune_model_channels — structured (whole-channel) pruning, stays Winograd-dense
  runtime.py              # Shared CLI plumbing: set_global_seed, capture_provenance (git_dirty = code_changes() over
                          #   ml/models/scripts/configs only, + git_dirty_files, since 2026-09-30; also records torchvision/CUDA/
                          #   cuDNN/Python versions, GPU name, cpu_count, SLURM_JOB_ID since 2026-09-13), load_profile
                          #   (experiment/runtime yaml by name or path), ensure_dataset_path, make_model_runs
                          #   (<root>/<exp>/<model>/...), save_resolved_config — scripts import these from `ml`, not
                          #   from scripts/train.py's privates
  winograd_bridge.py      # The ONLY import path into the sibling Winograd-FPGA repo ($WINOGRAD_FPGA_ROOT): study-model
                          #   ctors (custom_model/torchvision_model — geometry owned there, so checkpoints load 1:1 in
                          #   its Fase 2.5), the qat_wino stage (load_qat_wino_model(..., cfg=QATWinoConfig)
                          #   passes variant/pack/u_w/v_w/k_dsp through to qat_wino.convert(); cfg=None keeps the
                          #   old F(4,3)-no-pack behavior), bridge_provenance (its commit)
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
                          #   every phase_11_*.yaml extends _protocols/no_patience.yaml (seed 42 unless *_seed<N>,
                          #   uniform_hparams, 500ep FP32 on AlexNet's recipe since 2026-10-03 (SGD m0.9 lr 0.01 wd 5e-4 bs128,
                          #   cosine to 0, pad4-crop-flip + AutoAugment, He init everywhere -- AdamW 3e-4 / RRC+rotation before,
                          #   every Phase 11 run retrained; each value's reference in the yaml and docs/logs/PHASE11_LOG.md "One
                          #   recipe"), QAT 50ep SGD lr 1e-4 cosine to 1e-6 (Wu et al. 2020 App. A.2; 100ep at 1e-5 until
                          #   2026-10-03), early_stopping_patience: null — the QAT stage
                          #   inherits null too, since scripts/train.py builds its cfg via replace() off the same base)
                          #   wino_f23_pack.yaml/wino_f43_pack.yaml/wino_f63_pack.yaml (2026-09-19, M7) extend
                          #   _protocols/winograd_fpga.yaml, stages: [qat_wino] only (FP32 reused from budget_unico's
                          #   checkpoint), one `qat_wino:` override each (variant: f23|f43|f63, pack: true,
                          #   u_w=9/v_w=8/k_dsp=2 -- the iso-DSP-budget point, u_w+2*v_w<=25 for K=2 on a DSP48E2) --
                          #   measures the accuracy cost of matching the deploy bitstream's packing + each transform;
                          #   results (alexnet_fire_bypass_fpga, vgg_style_fpga): F23 ~unchanged (+0.09/-0.14pp),
                          #   F43 moderate loss (-3.93/-8.28pp), F63 severe (-15.65/-30.55pp) vs FP32
                          #   2026-09-24: extended to the 14 other networks (12 budget_unico + phase_11's alexnet_tv_3x3/vgg16) from a
                          #   SEPARATE PCAD checkout ~/dnn_study_m7 (main @ c9750c0: 1 commit NOT on origin + uncommitted 14-model
                          #   `models:` lists) -- median vs FP32: F23 +0.05, F43 -5.1, F63 -18.9pp; budget_unico's no-packing F43
                          #   numbers overstate the bitstream by 3.9-10.4pp (docs/logs/PHASE11_LOG.md "M7 extension"). The yaml files
                          #   HERE still list only alexnet_fire_bypass_fpga/vgg_style_fpga and scripts/train.py lacks that fix
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
                           #   runs locally). Refuses a real submit while ml/models/scripts/configs have uncommitted
                           #   changes (2026-09-30; `--allow-dirty` overrides) -- commit, push, `git pull` on PCAD first
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
  phase11/                 # `python -m scripts.phase11.<name>`
    # The 18 PNGs in results/figures_generated/phase_11_kernel_size_comparison/ (02-11, 14-21; rewritten 2026-10-04 for the
    # 76-run design): runs are picked by their factors (CELL_FACTORS, via design_figures.frame), never by name, each
    # figure with the original AlexNet trained from scratch as a baseline. Names say "layout original" (torchvision's
    # stride-4/three 3x3 pools, 1x1 map at 64px) vs "layout 64px" (stride 2, two 2x2 pools, 8x8 map). 01/12/13 retired
    # (16, 14 and 17/18 cover them). tests/test_phase11_figures.py renders all of them on a fake run tree (whole design
    # and the 5 pilot cells) -- run the scripts again whenever runs land; a pair or panel without data is skipped.
    plot_kernel_comparison.py  # 02-04 kernel per layout/head/BN (AlexNet original, 64px; VGG16)
    factor_effects.py          # 05-11 one figure per factor over every matched pair of the design (+ FAMILY_CONTRASTS),
                               #   14 every matched pair per contrast with the seed noise band
    analyze_geometry.py        # 15 QAT-vs-conversion INT8 loss + the shared loader/plot helpers + main_grid/kernel_geometry tables
    design_figures.py          # 16 Pareto accuracy x cost, 17 kernel x head x BN grid, 18 kernel x geometry, 19 INT8
                               #   robustness, 20 latency vs MACs, 21 training curves; --archive previews them on the
                               #   superseded runs (results/archive_adamw_recipe/figures_generated/phase_11_preview/)
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
                               #   --input-size 224 (2026-09-19) dumps a second, separate layer_configs_224/ at the
                               #   resolution the literature compares at -- eligibility changes with resolution
                               #   (the accelerator steps NUM_CORES*m px in X, so on <=8px output F(4,3) wastes
                               #   most of the tile and "loses" to F(2,3) on resolution, not on the transform),
                               #   so never mix its cells with the 64px dump's in one table.
  slurm/*.sbatch           # sbatch templates — train.sbatch/profile.sbatch submitted by cluster.py, det_seg.sbatch by the pcad/submit_phase_7_*.sh scripts, others called directly.
                          # train.sbatch fixed 2026-09-13: conda never activates in a non-interactive Slurm batch
                          #   shell (conda init lives in ~/.bashrc, which such shells don't source) — every real
                          #   submission died in ~1s as "python: command not found" (jobs 821240/821241, on both
                          #   beagle and tupi). Switched to `cd "$(git rev-parse --show-toplevel)" && source
                          #   .venv/bin/activate`, the pattern det_seg.sbatch/prune_channels.sbatch already needed
                          #   after hitting the identical failure. profile.sbatch/measure_compression.sbatch/
                          #   notebook.sbatch got the same fix 2026-09-18 -- no sbatch script still has the bug.
tests/                    # pytest: test_registry, test_checkpoint, test_config, test_trainer_smoke,
                          #   test_quantization, test_profiling, test_train_cli, test_train_det_seg_cli,
                          #   test_train_pipeline (run_experiment end to end; a stop signal ends the run
                          #   instead of rolling into the next stage on a truncated model), test_stats
                          #   (Wilson/McNemar/Holm/Pareto), test_phase11_figures (every Phase 11 figure on a fake run tree)
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
  logs/                   # Documentation (flat): PHASE7_QUICKSTART.md, PHASE7_MULTINODE.md, PHASE7_LOG.md, PHASE8_LOG.md,
                          #   PHASE11_LOG.md (every Phase 11 decision, dated: recipe, citations, design, figures)
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
`*_best.pth.gz` (2026-09-19) is now git-ignored too — it carries AdamW optimizer state (~3× the
weights) and can run hundreds of MB; only the small `qat_*.pth.gz` INT8 artifact stays tracked.
VGG16 FC-head INT8 artifacts are over GitHub's 100 MB hard limit, so `.gitignore` names them: the archived
`outputs/pcad/archive_adamw_recipe/phase_11_kernel_size_comparison/vgg16/checkpoints/qat_vgg16.pth.gz` (105.78 MB) and,
by pattern, every `phase_11_vgg_kernel_head*/vgg16_*_fc*` cell's `qat_*.pth.gz`.

After a `git pull` on a machine that still has artifacts under old folder names, run
`scripts/pcad/migrate_pcad_gitignored.sh` — it is idempotent and covers every rename to date.

---

## Stack

Python 3.12 · PyTorch 2.5.1+cu121 · torchvision 0.20.1 · torchmetrics · torchinfo · fvcore (FLOPs) · wandb (offline-first) · kagglehub · optuna (not yet wired) · CUDA 12.1 on RTX 4060 Laptop (8.2 GB).

Quantization backend: **onednn** (`ml.quantization.QUANT_ENGINE`, set by `convert_to_int8` itself) — the kernels that run the literature's full 8-bit activations; fbgemm needs `reduce_range` (7-bit) on CPUs without AVX-512 VNNI, which every run used until 2026-10-03. INT8 convert + inference are **CPU-only**.

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

**Data:** ImageFolder, deterministic 90/10 split (`torch.Generator`, the run's seed; the train-set hold-out of He et al. 2016 Sec. 4.2's 45k/5k), workers seeded via `worker_init_fn`. ImageNet normalization. Train aug (`DataConfig.train_aug`): `legacy` = `RandomResizedCrop(0.7–1.0)`, hflip, `RandomRotation(15)`, `AutoAugment(ImageNet)` (Phases 1–10); `crop_flip_autoaug` = pad-4 random crop + hflip (He 2016) + `AutoAugment(ImageNet)` (Cubuk 2019), Phase 11 since 2026-10-03. Val: `Resize → CenterCrop`. **Test** (2026-10-03, `create_test_loader`): Tiny ImageNet's official val split (10k, 50/class) — the 90/10 split only picks the best epoch; `scripts/train.py` reports every stage on test too (`test_{fp32,qat,int8}_*`, `{model}_{stage}_test_logits.npz`), and that is the number the paper uses (same images for every seed; the 90/10 split moves with the seed). `num_workers: 4` (`configs/data.yaml`) for every run: the FP32 stage is loader-bound (4–40% GPU on the 4090s), but more workers would change each worker's augmentation random stream against the runs already trained -- kept on purpose (2026-10-02).

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
| Winograd-FPGA study (`budget_unico`) | none here — `ml/winograd_bridge.py` builds them from the sibling repo | 15 `*_fpga` models registered in `ml/model_registrations.py`: vgg_style, alexnet_{3x3_fc, stacked, fire, fire_bypass, bottleneck, final_fire_residual, final_bottleneck_residual}, repvgg_a0 (trained raw), wrn_{16_4, 28_2}, googlenet, resnet18, vgg13 — plus squeezenet1_1, registered but out of budget_unico (qat_wino breaks on its 15×15 maps). Add a study model in the sibling repo, then one `register_model(..., custom_model/torchvision_model(...))` line here. **M7 (2026-09-19):** the 14-model budget_unico accuracy sweep trains F(4,3) with no packing; the deploy bitstream packs, so those numbers are marked `≠HW`. `wino_f{23,43,63}_pack.yaml` re-run `qat_wino` at the iso-DSP-budget packed point on 2 models (alexnet_fire_bypass_fpga, vgg_style_fpga) to measure the real cost: F23 ~unchanged, F43 moderate loss, F63 severe — see the `configs/experiments/` entry above and the sibling repo's `achados_varredura.md §6` for the full write-up |
| 11 — Kernel size comparison | `models/baselines.py` | **Current design (2026-10-04, `docs/logs/PHASE11_LOG.md` "Minimal design, symmetric 2x2 padding, scratch twins"): 76 runs, every one retrained on the one cited recipe, seed 42, in 6 blocks = `phase_11_kernel_head_bn` (the main question; `_seed43/_seed44` rerun 2 of its cells for the noise floor), `phase_11_kernel_geometry`, `phase_11_vgg_kernel_head`, `phase_11_dropout_pretraining`, `phase_11_stacked`, `phase_11_families` (incl. `mobilenetv2_scratch`/`resnet18tv_scratch`: every pretrained net has a from-scratch twin, tests/test_config.py). Every stride-1 2x2 conv uses `models/baselines.py:SymmetricPad2d` (C2sp, Wu et al. NeurIPS 2019) -- one-sided padding until 2026-10-04; `k2x2stacked` = `AlexNetStacked(kernel_size=2)`. Cells are named by what they are -- `alexnet_<kernels>_stride<s>_<n>pool<k>x<k>_map<m>_<gap|fc|fcdrop>_<bn|nobn>[_pretrained]` / `vgg16_...` (`ml/model_registrations.py:CELL_FACTORS`, every name checked against the built net in tests/test_registry.py); the yamls/names below (alexnet_fx_*, vgg_fx_*, geometry_controls, ...) are history, their runs archived in `outputs/pcad/archive_adamw_recipe/`.** History: `AlexNetTV(kernel_size=None\|3\|2)` (original 11×11/5×5/3×3, 3×3, 2×2, no BN) and `VGG16(kernel_size=3\|2)` (torchvision cfgs["D"] + BatchNorm -- plain (no-BN) VGG16 from scratch measured stuck at ln(200) loss for 22 epochs on PCAD, 2026-09-13; kernel_size=3 is VGG's own native design) — all 5 trained from scratch, no early stopping, via `configs/experiments/phase_11_kernel_size_comparison.yaml` (`_protocols/no_patience.yaml`: 500ep FP32 / 50ep QAT since 2026-10-03, Wu et al. 2020). **Mixed-kernel + head/BN ablation extension:** `alexnet_variants.py`'s `AlexNetMixed`/`AlexNetStacked`/`AlexNetSmallKernel` and `AlexNetTV(kernel_size="mixed_alt"\|"mixed_early2"\|"mixed_early3")` cross kernel pattern × GAP/FC head × BN on/off, via `phase_11_mixed_kernel_comparison.yaml` and `phase_11_head_bn_ablation.yaml`. `models/baselines.py:he_init` (2026-09-17, `docs/logs/PHASE11_LOG.md`) fixes a from-scratch dead-ReLU plateau that killed several no-BN cells (`alexnet_stacked_fc_nobn` still doesn't train — BN turns out load-bearing for that depth+FC-head combination, treated as a finding not a bug). `alexnet_tv_mixed_early2_gap` (added 2026-09-19) closes the last FC/GAP pairing gap, 26.51% FP32 on PCAD — not yet folded into the curated `results/phase_11_head_bn_ablation_final_comparison.csv`. `scripts/phase11/{plot_kernel_comparison,factor_effects,analyze_geometry,design_figures}.py` render the 18 figures (see the scripts/phase11/ entry above). **Geometry caveat (verified 2026-09-24, `docs/logs/PHASE11_LOG.md` "Geometry confound"):** `AlexNetTV` keeps the 224×224 stride/pool layout, which collapses to a 1×1 map before the classifier at 64×64 (and with `kernel_size=3/2` its stride-4 conv1 reads only 56%/25% of the pixels), while `AlexNet3x3FC/GAP`/`Mixed`/`Stacked` and the compensation family use an adapted layout (stem s2, 2× MaxPool(2), no Dropout) — ~17pp better at matched protocol (`alexnet_mixed` 45.28% vs. `alexnet_tv_mixed_alt_gap` 27.90%). So "AlexNet compacto" vs. `alexnet_tv_*`, and `AlexNet3x3-FC` vs. the pretrained AlexNetTV baseline, are NOT pure kernel-size comparisons (Dropout, init, protocol, pretraining also differ); never describe either as a single-variable kernel ablation, and don't extrapolate the 64×64 absolutes to 224×224 AlexNets. Controls for that confound: `models/alexnet_variants.py:AlexNetAdapted` (`alexnet_adapted_orig_{fc,gap}` = 11-5-3-3-3 kernels, `alexnet_adapted_2x2_{fc,gap}` = 2×2 at the same 8×8 maps, `alexnet_3x3_gap_bn` = BN control) run via `phase_11_geometry_controls.yaml` (`_protocols/no_patience.yaml`: 500 ep / QAT 50 ep, seed 42, one job per model); submitted 2026-09-24, results pending (`docs/logs/PHASE11_LOG.md`). **Geometry factorial + seed replicates** (2026-09-25): `AlexNetAdapted` also takes `stem_stride`/`stem_padding`/`pool_kernel`/`pool_count`/`dropout`/`pretrained` (defaults = adapted layout), registered as `alexnet_geo_<stem>_<pool>_<head>[_drop|_k3]` + `alexnet_adapted_orig_fc_pt`; run via `phase_11_geometry_factorial.yaml` (10 models incl. pretrained `alexnet_tv`) and `phase_11_geometry_seeds_s43/_s44.yaml` (4 kernel-pair models each) — same `_protocols/no_patience.yaml` protocol, enforced by `tests/test_config.py`; results in. **QAT fusion bug + full factorial (2026-09-30, `docs/logs/PHASE11_LOG.md`):** `prepare_qat_model` fused the caller's original instead of the QAT copy, so every `fuse_root_attr` model (43, incl. all hand-mapped AlexNets/VGGs) trained QAT unfused -- their INT8/ΔQAT are invalid until rerun (`scripts/pcad/rerun_qat_fused.sh`, old artifacts -> `outputs/pcad/archive_unfused_qat/`); FP32 is unaffected. **Float logits layer (2026-10-02):** the 15 QATs finished on the 09-30 code are rerun too (old artifacts -> `outputs/pcad/archive_quantized_logits_qat/`, gate `alexnet_3x3_gap` + `alexnet_smallkernel_fc` first), so every Phase 11 QAT/INT8 comes from one commit. The report's classification CNNs all move to this protocol: `phase_11_families.yaml` + the 208-cell `alexnet_fx_*` AlexNetAdapted factorial (`phase_11_factorial_core/_ext.yaml`, 19 cells reused via `FX_EXISTING`); analysis `scripts/phase11/factor_effects.py`. **VGG factorial (2026-10-02):** `models/baselines.py:VGGAdapted` (VGG16+BN, per-conv 3/2 kernels with ZeroPad'd 2×2, stem stride 1/2, MaxPool 2 or 3/s2/p1, 5/4 pools, GAP/FC/FC+Dropout, `vgg16_bn` pretraining) -> 168 `vgg_fx_*` cells registered, `vgg16` reused as `vgg_fx_k3_s1_pk2n5_fc_d`; wave 1 = `phase_11_vgg_factorial.yaml` (32 runs). PCAD queue order: QAT reruns -> the 21 AlexNet wave-1 cells (kernel×stride×pool×head, no BN/Dropout/pt) -> VGG wave 1 -> families -> rest of core/ext (`docs/logs/PHASE11_LOG.md` "VGG factorial"). QAT coverage audited 2026-10-02: the 49 Phase 11 runs with a usable FP32 = the 49 pending reruns, incl. all 20 reused factorial cells; only `alexnet_stacked_fc_nobn` (dead FP32) is left out on purpose |

**Results & rankings:** see `docs/plans/BEST_MODELS.md` (Pareto tiers, recommendations, now covering Phases 1–4/6/7/8/9) and `results/results_aggregate/results_cross_phase.csv` / `results/results_aggregate/model_details_cross_phase.csv`. Headlines: MobileNetV2 best overall (~58% top-1) among Phase 1–3 models, though Phase 4's AlexNetFinalFireResidual (49.79%) and Phase 9's AlexNetFireBypass (50.57%) close most of the gap — the latter now *exceeds* the full hybrid's FP32 gain outright (+6.59pp vs. +5.81pp over AlexNetFire) — at a fraction of the size; AlexNetBottleneck/AlexNetFire remain Pareto-optimal on efficiency (43–44%, 1.5–2 MB, quantization-stable). A size-reporting bug (fixed 2026-09-02, `ml/reporting.py`) had `disk_mb()`/`gzip_mb()` measuring the raw `{model}_best.pth`, which carries AdamW optimizer state (~3× the weights), while the INT8 artifact was already weights-only — so every FP32 size and FP32-vs-INT8 compression ratio was inflated ~3× (~11.9× recorded vs. the true ~4×). Both sides now measure `model_state_dict`; summaries and CSVs backfilled via `scripts/oneoff/backfill_model_size.py`. Accuracies, params, MACs and all rankings are unaffected; the analysis notebooks were re-run on 2026-09-12, so only `phase9_ablation_analysis.ipynb` (its PCAD summary inputs are gone) and the training notebooks' output cells still show pre-fix sizes. Known issues: AlexNetSmallKernel severe QAT drop (~–10pp), AlexNetSE training failure. A `Trainer.fit()` bug (fixed 2026-08-29, `ml/trainer.py`) returned the last epoch's model instead of reloading the best checkpoint, so FP32 was evaluated on different weights than INT8 — spurious INT8 "gains" of up to +6.5pp for runs with a long post-peak tail; backfilled via `scripts/oneoff/backfill_best_epoch_eval.py` for the 5 CLI-trained Phase 8 models plus AlexNetFireBypass (FP32 corrected for all 6; INT8 only rebuilt where a full-precision QAT-best checkpoint survived — FireBypass and `vit_tiny`/`deit_tiny` were otherwise unaffected). See `report/ic_report.tex` Eixo 4/7 for the corrected findings. Phase 7 detection: anchor-recall root cause fixed and A4 retrain complete on PCAD — all 3 backbones (bottleneck/fire/tv) × FP32/QAT/INT8 × plain/pretrained now have valid mAP (see `docs/plans/BEST_MODELS.md`). Phase 7 segmentation: PCAD runs now complete for all 3 backbones × FP32/QAT/INT8 (`outputs/pcad/phase_7_detection_segmentation/seg_*`); not yet folded into the H1–H4 analysis notebook. Phase 7 hypotheses (H1–H4, does compensation transfer to dense prediction) and progress: `docs/plans/PHASE7_PLAN.md`, `docs/logs/PHASE7_LOG.md`. Phase 8: all 7 models trained on PCAD, results in — H1 (window-size sweep) and H4 (DeiT distillation) confirmed, H3 (quantization robustness) inverted (5 of 7 models gain accuracy under INT8), H5 (Winograd-eligibility) confirmed but not attention-specific (no model has a stride-1 3×3 conv). D6's QAT-for-attention revision (swap_quantizable_mha can't drive this codebase's eager-mode `prepare_qat()`, so attention stays FP32-excluded like Swin's fallback) and full H1–H5 detail are in `docs/plans/PHASE8_PLAN.md` and `docs/logs/PHASE8_LOG.md`.

---

## Running

**Notebooks** (Phases 1–5, exploratory):
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # first time / resync
source .venv/bin/activate
jupyter lab
```
`requirements.txt` (pip freeze, tightly pinned) is the source of truth for `.venv` — it has drifted before (`docs/logs/PHASE7_LOG.md`); re-run the `pip install -r requirements.txt` line if notebook imports start failing. `environment.yml` predates the switch to `.venv` below; `configs/runtime/*.yaml`'s `conda_env` field and `cluster.py`'s `CONDA_ENV_NAME` export are now fully vestigial — every `scripts/slurm/*.sbatch` activates `.venv` directly (fixed 2026-09-13 for `train.sbatch`, 2026-09-18 for `profile.sbatch`/`measure_compression.sbatch`/`notebook.sbatch`), none of them read `CONDA_ENV_NAME`/`conda_env` anymore.
Tiny ImageNet-200 downloads via `kagglehub` on first run (cached in `~/.cache/kagglehub/`). Before INT8 convert/inference: `model.eval()` and move to CPU.

**CLI / cluster runs** (Phases 6–8, reproducible local or PCAD SLURM runs). Despite `environment.yml`'s
name, actual practice — here and on PCAD — is a plain `.venv` (`python3 -m venv .venv && source
.venv/bin/activate && pip install -r requirements.txt`); neither machine has `conda` installed, and
`scripts/slurm/train.sbatch` was fixed 2026-09-13 to activate `.venv` directly instead of trying `conda`
(see its file entry above):
```bash
python -m scripts.train --experiment default --runtime local        # classification, local
python -m scripts.cluster submit --experiment default --runtime pcad --slurm single_gpu
python -m scripts.cluster submit --experiment phase_11_vgg_kernel_head --runtime pcad --slurm tupi_4090 --model vgg16_k3x3_stride1_5pool2x2_map2_fcdrop_bn --smoke   # fast real-cluster check, one model, output discarded
python -m scripts.cluster submit-sweep --experiment phase_11_kernel_head_bn --runtime pcad --slurm tupi_4090   # one job per model, real budget (no --smoke)
python -m scripts.cluster submit-sweep --experiment phase_8_efficient_vit --runtime pcad   # one job per model, Phase 8's 5 CLI-drivable models
python -m scripts.cluster submit --experiment budget_unico --runtime pcad --slurm tupi_4090 --model wrn_16_4_fpga   # one model only -- needs WINOGRAD_FPGA_ROOT exported first, see the preflight_budget_unico.py entry above
python -m scripts.cluster submit-sweep --experiment budget_unico --runtime pcad --dry-run   # print the sbatch commands, submit nothing
python -m scripts.cluster status <job_id>   # / cancel / resume
python -m scripts.train_det_seg detection --model alexnet_bottleneck --dry-run   # Phase 7
python -m scripts.profile_hardware --experiment phase_6_hardware_profiling --runtime local           # Phase 6
python -m scripts.winograd_fpga.dump_layer_configs   # emit layer_configs JSONs for the sibling repo's VU9P throughput sim
```
Edit `configs/runtime/pcad.yaml` (dataset root) and `configs/slurm/single_gpu.yaml` (partition/GPU/wall-time) for cluster settings; duplicate `configs/experiments/default.yaml` for a new reproducible run.
When tupi's six 4090s are full, a pending job can overflow to `beagle` (2× GTX 1080 Ti, but node-exclusive: one job at a time; 32086 MB RAM, so a tupi job's 32G never fits): `scontrol update JobId=<id> MinMemoryNode=16384` (MB, "16G" is rejected), then `Partition=tupi,beagle`. Its GPU/CPU latency and INT8 kernel path then come from a 1080 Ti / Sandy Bridge without AVX2 (~2-3× slower per epoch); the accuracy protocol is the same.

**Tests:** `pytest tests/`
