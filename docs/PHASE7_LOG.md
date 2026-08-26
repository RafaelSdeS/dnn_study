# Phase 7 Implementation Log

Decision continuity across `/compact` boundaries. Append per stage.

Note: "Stage N" below numbers Phase 7's own internal build-out steps (data
pipeline, model heads, trainer, CLI integration, ...) — unrelated to the
project's Phase 1-9 numbering in `CLAUDE.md`. Every stage here belongs to
Phase 7.

---

## Stage 1 — VOC Detection Data Pipeline ✓

**Commit:** `f5dadab` feat: VOC detection data pipeline

- Implemented `ml/det_seg_data.py` with VOCDetectionDataset + VOCSegmentationDataset wrappers
- Detection: VOC 2007+2012 trainval (train) + VOC 2007 test (eval); ~16.5K + ~4.9K images
- Segmentation: VOC 2012 train/val splits; ~1.5K + ~1.5K images
- Tensor format validation: images float32 [0,1], boxes float32 [0,256] in [xmin,ymin,xmax,ymax], labels int64 [1,20]
- Smoke check: data loads, shapes validated, no silent alignment bugs

---

## Stage 2 — Backbone + SSD Assembly ✓

**Commit:** `77df528` feat: SSD detector assembly for alexnet_bottleneck

- Implemented `ml/det_seg_models.py` with:
  - `DetSegBackbone`: taps Phase 3 backbones at feature indices, appends 2 extra SSDLite blocks
  - `build_ssd_detector()`: assembles full SSD with DefaultBoxGenerator + SSDLiteHead
  - `BACKBONE_FEATURE_CONFIG`: tap indices + channel counts for all 3 backbones
- Backbones working:
  - alexnet_bottleneck, alexnet_fire: tap at [3, 6] (stride ~8)
  - alexnet_tv: tap at [2, 12] (stride 8 + 32)
- Smoke check: forward pass (train + eval modes) validated, no NaN losses

---

## Stage 3 — Detection Trainer + mAP ✓

**Commit:** `0ab3035` feat: detection trainer + mAP evaluation

- Implemented `ml/det_seg_trainer.py` with `DetectionTrainer` class
- Early stopping on mAP (not accuracy), patience=10
- Metrics: mAP (all IoU thresholds), mAP@.5
- Checkpoint best-mAP model; save resume state per epoch
- Smoke check: 10-image overfit test (loss decreases, mAP increases)
- **Next: User runs first FP32 training via `python scripts/train_det_seg.py detection --model alexnet_bottleneck`**

---

## Stage 4 — Expand to 3 Backbones ✓

**Commit:** (no separate commit; backbones already in config)

- All 3 backbones (bottleneck, fire, tv) pre-populated in `BACKBONE_FEATURE_CONFIG`
- Verified forward pass works for all 3
- Anchor recall validation ready (deferred: full threshold sweep after FP32 results)

---

## Stage 5 — QAT/INT8 Support ✓

**Commit:** (integrated in Stage 3 trainer)

- Added `build_qat_detector()` placeholder in trainer
- Full QAT flow deferred until FP32 baseline results available (per blueprint's Blocking Issue #2 priority)
- Note: Box regression quantization is highest-uncertainty; fallback to FP32 head if needed

---

## Stage 6 — Segmentation Infrastructure ✓

**Commit:** `308b0ef` feat: segmentation data + trainer placeholders (Stage 6)

- `create_voc_segmentation_loaders()` implemented (VOC 2012 only)
- `build_deeplabv3_segmenter()` placeholder (TODO: custom backbone integration)
- `SegmentationTrainer` stub (follows detection trainer pattern)
- Deferred full implementation until detection Phase completes (reduces scope per user feedback)

---

## Stage 7 — CLI + Cluster Integration ✓

**Commit:** `4f013d4` feat: det/seg CLI + cluster integration (Stage 7)

- `scripts/train_det_seg.py`: mirrors `train.py`, supports `--model`, `--experiment`, `--dry-run`, `--runtime {local|pcad}`
- `configs/detection.yaml` + `configs/experiments/phase7_detection.yaml`
- `scripts/slurm/det_seg.sbatch`: SLURM template for PCAD cluster
- Tested dry-run; CLI ready for user to run training

---

## Stage 8 — Cross-Phase Analysis ✓

**Commit:** `43335e3` feat: cross-phase analysis (Stage 8)

- `scripts/phase7_analysis.py`: joins Phase 7 detection/segmentation to Phase 3 classification
- Tests hypotheses H1-H4 (small-kernel transfer, quantization robustness, RF sensitivity, head latency)
- Ready to run once Phase 7 FP32 results available
- **Superseded:** replaced by `notebooks/phase_7_detection_segmentation_analysis/phase7_results_analysis.ipynb`
  (see A5 below) — the script's H2-H4 were unimplemented stubs and its run-dir parser had drifted
  from the current `ssd_<model>_<stage>[_pretrained]_phase7_detection` naming; deleted.

---

## Stage 9 — Anchor-Recall Diagnosis & Fix (A1–A3) ✓

**Commit:** (uncommitted this session)

- **A1 (diagnose):** Ran `scripts/phase7_tools/check_anchor_recall.py` to completion for all 3 backbones at both
  256px and 512px (previously never finished — earlier attempts were killed by SIGKILL/OOM on the
  dev laptop, unrelated to the anchor logic itself). Confirmed recall well below the 95% bar at
  every combination (0.76–0.80) and resolution-independent — ruling out "just use bigger images"
  and confirming the anchor-generator config itself was the root cause of the 0.4–7.1% mAP below.
- **A2 (fix):** Root-caused two independent bugs:
  1. Tap-index bug — `alexnet_bottleneck`/`alexnet_fire` tapped `[3,6]`, producing two pyramid
     levels at the *same* spatial resolution instead of 4 genuinely distinct ones. `alexnet_tv`'s
     `[2,12]` had the opposite problem: its deepest level collapsed to a near-degenerate 2×2 grid
     once extra SSDLite blocks stacked on top of an already-deep tap.
  2. Anchor-scale bug — `DefaultBoxGenerator`'s `min_ratio`/`max_ratio` linear interpolation left a
     large scale gap (0.1→0.383) exactly where most VOC objects live (median GT box side-ratio
     0.265).

  Fixed: tap indices (`bottleneck`/`fire` → `[2,6]`; `alexnet_tv` → `[2,5]` + `out_channels`
  `[64,192]`), `num_extra_blocks` 2→3, and replaced `min_ratio`/`max_ratio` with explicit
  percentile-matched `scales=[0.03,0.08,0.16,0.3,0.55,1.0]` + wider `aspect_ratios=[1.5,2,3,4]`.

  Result @512px: anchor recall 0.797/0.799/0.797 → **0.991/0.991/0.932**
  (bottleneck/fire/alexnet_tv). Bottleneck/fire comfortably clear 95%; `alexnet_tv`'s ~93% is an
  accepted residual gap from its structurally coarser native pyramid, not chased further
  (diminishing returns — closing it would mean yet more asymmetric backbone treatment). Note: the
  undocumented `_minratio02` retry below had already tried tuning `min_ratio` and failed (0.96% vs
  1.17% mAP) — but was never validated against the anchor-recall check itself, only against
  expensive final mAP. This fix is empirically confirmed at the recall-check level first.
- **A3 (re-enable gate):** Uncommented the anchor-recall pre-flight abort in
  `scripts/train_det_seg.py`. The original "hangs 120+s" complaint that got it disabled was
  actually the unbounded-`max_samples` bug, separately fixed earlier (`82ed115`) — confirmed the
  gate now runs end-to-end in ~56s and correctly aborts for `alexnet_tv` (recall 0.925 < 0.95)
  before training starts. `alexnet_tv` needs `--skip-anchor-check` passed explicitly for A4 given
  the accepted tradeoff above.
- **Budget bump:** `configs/experiments/phase7_detection.yaml` epochs 30→1000,
  `early_stopping_patience` 10→50 (all 3 models); `scripts/train_det_seg.py`'s QAT stage epochs
  15→100 — matching the actual PCAD budget planned for A4.
- **Environment note:** this diagnostic work ran on the local laptop (not PCAD) as a workaround
  while PCAD's frontend node had CPU problems. `.venv` had drifted from `requirements.txt` (missing
  `sympy` + several other pinned packages) — resynced via `pip install -r requirements.txt`.
  `scripts/phase7_tools/check_anchor_recall.py` gained a `--num-workers` override (yaml default is 4, but the
  laptop needed 0 to avoid OOM at 512px — a milder version of the same `num_workers`-at-512px
  fragility already noted in `configs/experiments/phase7_diag_512.yaml`'s comment about job
  805529's segfault).
- **A4 attempt 1 (failed, PCAD, jobs 809066-809074):** First post-fix retrain submission
  (`bash scripts/pcad/submit_phase7_multinode.sh qat int8`) crashed all 3 FP32 jobs immediately after
  the anchor-recall check (which passed) — `RuntimeError` on `load_state_dict` inside
  `trainer.fit(resume_from=...)`. Root cause: `ml/det_seg_trainer.py`'s `fit()` unconditionally
  auto-resumes from `<run_dir>/<run_id>_resume.pth` if the path exists (`ml/checkpoint.py`'s
  `auto_resume_path` pattern), and the 3 target run dirs
  (`ssd_alexnet_{bottleneck,fire,tv}_fp32_phase7_detection/`) still held `_resume.pth`/`_best.pth`
  from **pre-Stage-9** runs — old anchor config, so old checkpoints had 126/24 anchors per location
  vs. the new 210/40, a head-shape mismatch. Downstream QAT/INT8 jobs sat `PENDING` with SLURM
  reason `DependencyNeverSatisfied` (permanently stuck, not queue congestion — distinct from the
  normal `(Priority)`/`(Resources)` reasons other cluster users' jobs show).
  Fix: `scancel`'d the 6 stuck QAT/INT8 jobs, deleted the stale `_resume.pth`/`_best.pth`/`.log` in
  the 3 run dirs (gitignored; `config.yaml`/`metrics.json` are tracked and get overwritten by the
  next run regardless, so left alone), resubmitted the same chain fresh.
- **A4 attempt 2 (in progress, PCAD, jobs 809701-809709):** FP32 809701-703 → QAT 809704-706 → INT8
  809707-709, queued cleanly (no `DependencyNeverSatisfied`). Not yet confirmed past first epoch as
  of this log entry — cluster (`tupi` partition) had other users' jobs ahead in queue.
- **A4 progress since:** `alexnet_bottleneck` FP32/QAT (`_phase7_detection`) completed post-fix
  with valid mAP (~0.21 both stages); `alexnet_tv` QAT also completed post-fix (mAP 0.147). A
  second submission with backbone pretraining (`_pretrained` suffix, from Phase 3 Tiny-ImageNet
  checkpoints) is in flight: `bottleneck` FP32/QAT complete, `fire` FP32 running (job 812165), `tv`
  FP32 queued (job 812166). No INT8 detection run has produced metrics — the one attempt (job
  811101) crashed on uncalibrated observers (`min tensor(inf) should be less than max
  tensor(-inf)`), unfixed. Segmentation (Part B) untouched.
- **A4 complete:** all 3 backbones × FP32/QAT/INT8 × scratch/pretrained now have valid metrics on
  disk. The INT8 crash is fixed (see Implementation Status below for the fix and the real numbers).

---

## Stage 10 — Cross-Phase Analysis Notebook ✓

- Replaced `scripts/phase7_analysis.py` (H2-H4 unimplemented stubs; run-dir parser broken for the
  current `_pretrained` naming) with
  `notebooks/phase_7_detection_segmentation_analysis/phase7_results_analysis.ipynb`, matching the
  notebook-based analysis convention every other executed phase (5/6/9) already uses.
- Parses every run directory directly off disk (regex on `ssd_<model>_<stage>[_pretrained]_<exp>`),
  computes per-run provenance via `git merge-base --is-ancestor 686b419 <hash>`, and only allows
  H1/H3 to plot runs that are both post-anchor-fix and complete — so a future stale/superseded run
  can't silently corrupt a hypothesis test the way the pre-Stage-9 numbers did.
- H2 (quantization robustness) is explicitly reported as blocked (no INT8 metrics exist yet) rather
  than silently skipped.
- H4 defaults to the recorded per-run `latency_ms_per_image` (already measured during training) and
  only builds/profiles a live model if `RUN_PROFILING=True` is set by hand — never runs on the PCAD
  front-end.
- Phase 8 notebook intentionally **not** created: `ideas/PHASE8_PLAN.md` Tasks 1-6 (models,
  registry, `configs/experiments/phase8.yaml`) don't exist yet, so a notebook now would be an empty
  shell. Build it once at least FP32 results exist for the seven Phase 8 models.

---

## Implementation Status

**Infrastructure:** All 9 training-side stages complete and smoke-tested; Stage 10 (analysis
notebook) also complete.

**Superseded:** The FP32/QAT/INT8 runs below (`phase7_detection`'s original fire/tv FP32 runs,
`_minratio02` anchor-config retry, `_diag_256`, `_diag_512`, `_early_30ep`) all trained against the
broken anchor config fixed in Stage 9. Validation mAP was 0.4–7.1% across every one of them — far
below a working SSD's expected 40–70%+ on VOC, consistent with the anchor-recall check never having
been run to completion before those runs (see Stage 9 for the root cause). **Do not cite these
numbers; they are invalid, not just low.**

**A4 — complete.** All 3 backbones (bottleneck/fire/tv) × FP32/QAT/INT8 × scratch/pretrained-backbone
now have valid metrics on disk (`outputs/detection_segmentation/phase7/ssd_*_phase7_detection*`).
The INT8 observer-calibration crash (job 811101) is resolved — `scripts/train_det_seg.py`'s `int8`
branch now saves a checkpoint and computes a size/params summary (backfilled onto older runs via
`scripts/phase7_tools/backfill_int8_size.py`), and the Fire backbone's INT8 concat-quantization mismatch is
fixed (`models/compensation.py`, commit `565fef4`). Real numbers and a first read: `ideas/BEST_MODELS.md`
Phase 7 section.

**A5 — done.** `notebooks/phase_7_detection_segmentation_analysis/phase7_results_analysis.ipynb`
has been re-run with H1/H4 plots and real INT8 model sizes against the full A4 result set. H2
(quantization robustness) is now testable since INT8 metrics exist for all backbones.

**Next:**
1. Segmentation (Part B) — code is complete: `build_deeplabv3_segmenter()`, `SegmentationTrainer`,
   and the CLI's `run_segmentation()` are fully implemented (no longer placeholders/stubs), and
   PCAD runs are now complete for all 3 backbones × FP32/QAT/INT8
   (`outputs/detection_segmentation/phase7/seg_*`). What's left: extending the H1–H4 analysis
   notebook to segmentation now that results are on disk.

**Ground Rules Applied:**
- ✓ Context hygiene: all decisions logged here for `/compact` recovery
- ✓ Reproducibility: `outputs/detection_segmentation/phase7/<exp>/config.yaml` + `git_hash.txt` per run
- ✓ Failure triage framework in place (distinguish bugs, limitations, real findings)
- ✓ Budgeting: FP32 max 1000 epochs, patience 50; QAT 100 epochs (Stage 9 bump)
- ✓ No abstraction creep: reused existing trainer/data patterns
- ✓ Baseline mandatory: all three backbones wired + tested

---
