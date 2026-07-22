#!/usr/bin/env python
"""One-off diagnostics while the Stage 7 sweep runs:
  1. Smoke-test compute_detection_summary() end-to-end (never exercised yet).
  2. Anchor-scale vs. VOC GT-box-size distribution — quantify the 64.2% recall cap.

Usage: python scripts/diag_stage7.py
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from ml import DetSegDataConfig, build_ssd_detector, create_voc_detection_loaders
from ml.reporting import compute_detection_summary
from ml.runtime import expand_path

with open("configs/detection.yaml") as f:
    base_data_cfg = DetSegDataConfig(**yaml.safe_load(f).get("data", {}))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- 1. smoke test
print("=" * 60)
print("1. compute_detection_summary() smoke test")
print("=" * 60)

data_cfg = replace(base_data_cfg, img_size=256, batch_size=4, num_workers=2)
data_cfg.voc_root = expand_path(data_cfg.voc_root)
_, _, _, val_loader = create_voc_detection_loaders(data_cfg)

model = build_ssd_detector("alexnet_bottleneck", num_classes=21, image_size=256)
summary = compute_detection_summary(model, 256, val_loader, device, checkpoint_path=None)
for k, v in summary.items():
    print(f"  {k}: {v}")
missing = [k for k, v in summary.items() if v is None and k != "model_size_mb"]
if missing:
    print(f"  ⚠ fields returned None (silently failed): {missing}")
else:
    print("  ✓ all fields populated")

# ---------------------------------------------------------------- 2. anchor coverage
print("\n" + "=" * 60)
print("2. Anchor scale vs. GT box size distribution")
print("=" * 60)

for arch in ["alexnet_bottleneck", "alexnet_fire", "alexnet_tv"]:
    m = build_ssd_detector(arch, num_classes=21, image_size=512)
    scales = m.anchor_generator.scales
    print(f"\n{arch}: anchor scales (fraction of image) = {[round(s, 3) for s in scales]}")
    print(f"  -> smallest anchor covers objects >= {scales[0]:.1%} of image side")

# GT box size distribution from VOC07 test (val_ds), independent of backbone
data_cfg_gt = replace(base_data_cfg, img_size=512, num_workers=0)
data_cfg_gt.voc_root = expand_path(data_cfg_gt.voc_root)
_, val_ds, _, _ = create_voc_detection_loaders(data_cfg_gt)

ratios = []
for i in range(len(val_ds)):
    _, target = val_ds[i]
    boxes = target["boxes"]
    if len(boxes) == 0:
        continue
    w = (boxes[:, 2] - boxes[:, 0]) / 512.0
    h = (boxes[:, 3] - boxes[:, 1]) / 512.0
    side = torch.sqrt(w * h)  # geometric-mean side, comparable to anchor "scale"
    ratios.append(side)
ratios = torch.cat(ratios)

print(f"\nVOC07-test GT boxes: n={len(ratios)}")
for pct in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  p{pct}: {ratios.quantile(pct / 100):.3f}")

min_scale = 0.2  # current min_ratio
below = (ratios < min_scale).float().mean().item()
print(f"\nFraction of GT boxes with side-ratio < min_ratio={min_scale}: {below:.1%}")
print("(these boxes are structurally too small for ANY anchor to reach IoU 0.5 — "
      "recall on them is 0% regardless of training or image resolution)")
