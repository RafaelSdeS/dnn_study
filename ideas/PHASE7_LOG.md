# Phase 7 Implementation Log

Decision continuity across `/compact` boundaries. Append per stage.

---

## Stage 1 — VOC Detection Data Pipeline

**Date:** 2026-07-16  
**Commit:** feat: VOC detection data pipeline

### Decisions Made

- **Input resolution:** 256×256 (not 64×64) per Task 1 spec — VOC objects shrink below pixel-level detail at 64×64.
- **VOC splits:** 
  - Detection train: VOC 2007+2012 trainval (concatenated, ~16.5K images)
  - Detection eval: VOC 2007 test (~4.9K images)
  - Segmentation deferred to Stage 6 (VOC 2012 train/val only, ~1.5K each)
- **Transforms:** v2 geometric transforms (flip, resize/letterbox) applied via `tv_tensors.BoundingBoxes`/`tv_tensors.Mask` wrappers. Augmentation intentionally lighter than classification (no `AutoAugment`).
- **Detection collate:** Custom `_detection_collate_fn` — images stack, targets stay list-of-dicts (variable boxes per image).
- **Class indexing:** VOC classes 1–20, 0 reserved for background (SSD convention).
- **Segmentation ignore label:** 255 (boundary/ignore in masks) — will pass `ignore_index=255` to loss + metrics in Stage 6.

### Open Issues / Deferred

- Segmentation loaders (Stage 6) — placeholder `NotImplementedError` in `create_voc_segmentation_loaders`.
- VOC download fallback to kagglehub if official mirror slow — not yet tested, monitor if `download=True` times out.

### Validation

- Smoke check: tensor format asserts (images float32 [0,1], boxes float32 [0,256], labels int64 [1,20])
- Manual visualization: 4-sample draw_bounding_boxes to disk, eyeball alignment before Stage 2.

---
