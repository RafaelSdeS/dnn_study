# Phase 7 — Detection/Segmentation Implementation ✓

**Status:** Infrastructure complete. Early FP32/QAT/INT8 runs trained against a broken anchor
config (mAP 0.4–7.1%, invalid — not just low) — root-caused and fixed in `docs/PHASE7_LOG.md`
Stage 9 (recall now 0.991/0.991/0.932 for bottleneck/fire/alexnet_tv at 512px). Post-fix retraining
(A4) is in progress on PCAD: `alexnet_bottleneck` FP32/QAT are done and valid; `fire`/`tv` are
still in-flight and INT8 detection still crashes on uncalibrated observers (unfixed). See
`ideas/BEST_MODELS.md`'s Phase 7 section and `docs/PHASE7_LOG.md`'s Implementation Status for the
current, disk-sourced state before trusting any mAP number.

## What's Been Built (8 Stages)

| Stage | Component | Status | Files |
|-------|-----------|--------|-------|
| 1 | VOC data pipeline | ✓ | `ml/det_seg_data.py`, `ml/config.py` |
| 2 | SSD backbone+head | ✓ | `ml/det_seg_models.py` |
| 3 | Detection trainer | ✓ | `ml/det_seg_trainer.py` |
| 4 | 3 backbones (bottleneck, fire, tv) | ✓ | BACKBONE_FEATURE_CONFIG |
| 5 | QAT/INT8 support | ✓ | `build_qat_ssd_detector()` ready |
| 6 | Segmentation stubs | ✓ | Loaders + trainer placeholders |
| 7 | CLI + SLURM | ✓ | `scripts/train_det_seg.py` + YAML configs |
| 8 | Cross-phase analysis | ✓ | `notebooks/phase_7_detection_segmentation_analysis/phase7_results_analysis.ipynb` |

## Quick Start — Run FP32 Detection Training

### Local (test/debug):
```bash
python scripts/train_det_seg.py detection --model alexnet_bottleneck --dry-run
# Shows config, exits. Then:
python scripts/train_det_seg.py detection --model alexnet_bottleneck
```

### PCAD Cluster:
```bash
python scripts/train_det_seg.py detection --model alexnet_bottleneck --runtime pcad
# Or manually:
sbatch scripts/slurm/det_seg.sbatch detection alexnet_bottleneck phase7_detection
```

## Configuration

**Data & trainer defaults** → `configs/detection.yaml`:
- Batch size: 16
- Learning rate: 1.0e-4
- Epochs: 30 (early stop patience 10)
- VOC 07+12 trainval (train), 07 test (eval)

**Per-backbone overrides** → `configs/experiments/phase7_detection.yaml`:
- (Currently identical across all 3; customize as needed)

## Output Structure

Results saved to `outputs/detection_segmentation/phase7/ssd_<model>_<stage>_<experiment>/` (e.g.
`ssd_alexnet_bottleneck_fp32_phase7_detection/`):
```
├── config.yaml         # Full config (reproducible)
├── git_hash.txt        # Commit hash at run start
├── metrics.json        # Full history + best mAP
├── <model>_best.pth    # Checkpoint (best validation mAP)
└── <model>.log         # Training logs
```

## Key Metrics to Track

**For each backbone:**
- mAP@[.5:.95] (primary metric)
- mAP@.5
- Loss convergence
- Training time / inference latency
- Comparison to Phase 3 classification accuracy

## Next Steps

1. **A4** — finish the in-flight post-fix `_pretrained` FP32/QAT/INT8 chain for `fire`/`tv` on
   PCAD, and fix the INT8 observer-calibration crash (job 811101) so H2 becomes testable.
2. **A5** — re-run the cross-phase analysis notebook (Stage 10, below) once more results land to
   actually test H1–H4 across all three backbones.
3. **Extend to segmentation** (reuse same trainer pattern) — `SegmentationTrainer.fit()` is still a
   one-line stub and the CLI's `run_segmentation()` just prints "not yet implemented"; no training
   run yet.

## Ground Rules Enforced

✓ **Reproducibility:** git hash + full config saved per run  
✓ **Early stopping:** patience=10 on mAP (not epochs)  
✓ **Baselines:** all 3 backbones tested, same anchor config  
✓ **No ablation creep:** only backbone varies, heads identical  
✓ **Failure triage:** log distinguishes bugs from real results  

## Troubleshooting

**"VOC download slow?"**  
→ VOC datasets cache in `~/.cache/torchvision/`. First run downloads ~3 GB; subsequent runs use cache.

**"mAP is suspiciously low (e.g., <10%)?"**  
→ Before reacting, check:
  1. Anchor recall >95% (run `compute_anchor_recall()`)
  2. Feature map resolution (run demo, check shapes)
  3. Tensor format (boxes are [xmin,ymin,xmax,ymax], not normalized)

**"QAT crashes or mAP drops >10pp?"**  
→ Expected (Blocking Issue #2). Fallback: keep box regression FP32 (see `build_qat_ssd_detector()`).

## Reference

- **Blueprint:** `ideas/PHASE7_PLAN.md` (detailed research hypotheses H1-H4)
- **Log:** `docs/PHASE7_LOG.md` (per-stage decisions)
- **CLAUDE.md:** project conventions (reproducibility, trainer patterns)

---

**Phase 7 infrastructure is ready and the anchor-recall blocker is fixed — A4 (post-fix retraining of `fire`/`tv`) is the remaining in-flight work before H1–H4 can be tested.**
