#!/usr/bin/env python
"""Quick anchor/GT-box overlap check — bounded sample, no training.

Usage:
    python scripts/check_anchor_recall.py --model alexnet_bottleneck --img-size 256 512
"""
import argparse

import torch

from ml import DetSegDataConfig, build_ssd_detector, compute_anchor_recall, create_voc_detection_loaders
from ml.runtime import expand_path

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="alexnet_bottleneck")
parser.add_argument("--img-size", type=int, nargs="+", default=[256, 512])
parser.add_argument("--max-samples", type=int, default=200)
args = parser.parse_args()

for img_size in args.img_size:
    data_cfg = DetSegDataConfig(img_size=img_size)
    data_cfg.voc_root = expand_path(data_cfg.voc_root)
    _, _, _, val_loader = create_voc_detection_loaders(data_cfg)

    model = build_ssd_detector(args.model, num_classes=21, image_size=img_size)
    recall = compute_anchor_recall(model, val_loader, iou_threshold=0.5, max_samples=args.max_samples)
    print(f"img_size={img_size}: anchor recall @IoU 0.5 = {recall:.3f}  ({args.max_samples} images)")
