# Phase 7 — Detection/Segmentation Implementation ✓

**Status:** Infrastructure complete, training has run (FP32/QAT/INT8, all 3 backbones, several
configs) — but results are currently unreliable. Validation mAP came back at 0.4–7.1% across every
run, consistent with the anchor-recall check below never being run to confirmed completion before
training. See `ideas/BEST_MODELS.md`'s Phase 7 section and `docs/PHASE7_LOG.md`'s Implementation
Status before launching more training or trusting any existing mAP number.

## What's Been Built (8 Stages)

| Stage | Component | Status | Files |
|-------|-----------|--------|-------|
| 1 | VOC data pipeline | ✓ | `ml/det_seg_data.py`, `ml/config.py` |
| 2 | SSD backbone+head | ✓ | `ml/det_seg_models.py` |
| 3 | Detection trainer | ✓ | `ml/det_seg_trainer.py` |
| 4 | 3 backbones (bottleneck, fire, tv) | ✓ | BACKBONE_FEATURE_CONFIG |
| 5 | QAT/INT8 support | ✓ | `build_qat_detector()` ready |
| 6 | Segmentation stubs | ✓ | Loaders + trainer placeholders |
| 7 | CLI + SLURM | ✓ | `scripts/train_det_seg.py` + YAML configs |
| 8 | Cross-phase analysis | ✓ | `scripts/phase7_analysis.py` |

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

Results saved to `outputs/detection_segmentation/phase7/ssd_<model>_fp32/`:
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

1. **Fix the anchor-recall blocker** — run `compute_anchor_recall()` / `scripts/check_anchor_recall.py`
   to completion, confirm >95% for all 3 backbones (currently unconfirmed — the likely root cause
   of the mAP results below 10%).
2. **Retrain FP32 detection** with the corrected anchor config.
3. **Compare mAP across backbones** (H1: does compensation transfer?)
4. **Re-run QAT/INT8** (H2: is quantization robustness stable?) — already run once, but on top of
   the broken FP32 checkpoints, so not yet diagnostic.
5. **Extend to segmentation** (reuse same trainer pattern) — no training run yet, infrastructure only.
6. **Cross-phase analysis** (`python scripts/phase7_analysis.py`) — currently prints H1–H4 against
   the unreliable mAP data; rerun once FP32 is fixed.

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
→ Expected (Blocking Issue #2). Fallback: keep box regression FP32 (see `build_qat_detector()`).

## Reference

- **Blueprint:** `ideas/PHASE7_PLAN.md` (detailed research hypotheses H1-H4)
- **Log:** `docs/PHASE7_LOG.md` (per-stage decisions)
- **CLAUDE.md:** project conventions (reproducibility, trainer patterns)

---

**Phase 7 infrastructure is ready and has been used — fix the anchor-recall blocker before launching more training.**
