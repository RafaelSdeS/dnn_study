"""Backfill real INT8 model sizes for completed Phase 7 segmentation runs.

Segmentation counterpart to scripts/backfill_int8_size.py (same rationale: the `int8`
stage in scripts/train_det_seg.py's run_segmentation() didn't save a checkpoint or call
compute_segmentation_summary() until the fix in 58d290b, so existing INT8 runs'
metrics.json has accuracy but no `summary.model_size_mb`). Rebuilds each INT8 model from
its QAT checkpoint, measures its real serialized size, and writes a `summary` block into
that run's metrics.json.

No VOC dataset needed: compute_segmentation_summary()'s latency/throughput benchmark is
skipped by passing an empty val_loader.

Usage:
    python -m scripts.backfill_int8_size_segmentation
    python -m scripts.backfill_int8_size_segmentation --phase7-dir outputs/detection_segmentation/phase7
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml

from ml.det_seg_models import (
    build_deeplabv3_segmenter, build_qat_deeplabv3_segmenter, convert_deeplabv3_to_int8,
)
from ml.reporting import compute_segmentation_summary

RUN_NAME_RE = re.compile(
    r"^seg_(?P<model>alexnet_(?:bottleneck|fire|tv))_qat(?P<pretrained>_pretrained)?_(?P<exp>.+)$"
)


def backfill(phase7_dir: Path) -> None:
    device = torch.device("cpu")  # INT8 convert/inference is CPU-only, project convention

    for qat_dir in sorted(phase7_dir.glob("seg_*_qat*")):
        m = RUN_NAME_RE.match(qat_dir.name)
        if not m:
            continue
        model_name, pretrained, exp = m.group("model"), m.group("pretrained") or "", m.group("exp")
        qat_ckpt = qat_dir / f"{qat_dir.name}_best.pth"
        if not qat_ckpt.exists():
            print(f"[skip] {qat_dir.name}: no _best.pth checkpoint locally")
            continue

        int8_dir = phase7_dir / f"seg_{model_name}_int8{pretrained}_{exp}"
        int8_metrics_path = int8_dir / "metrics.json"
        if not int8_metrics_path.exists():
            print(f"[skip] {qat_dir.name}: matching INT8 run {int8_dir.name} has no metrics.json yet")
            continue

        metrics = json.loads(int8_metrics_path.read_text())
        if "summary" in metrics:
            print(f"[skip] {int8_dir.name}: already has a summary block")
            continue

        config = yaml.safe_load((qat_dir / "config.yaml").read_text())
        img_size = config["data"]["img_size"]

        print(f"[measure] {int8_dir.name} (from {qat_dir.name}, img_size={img_size})")
        model_fp32_skeleton = build_deeplabv3_segmenter(model_name, num_classes=21, image_size=img_size)
        # True architecture param count, from the untouched FP32 skeleton -- quantized
        # modules pack weights as torch.qint8 buffers, not nn.Parameter, so counting on
        # the converted INT8 model itself silently undercounts.
        true_params_m = sum(p.numel() for p in model_fp32_skeleton.parameters()) / 1e6

        model_qat_skeleton = build_qat_deeplabv3_segmenter(model_fp32_skeleton, device)
        state = torch.load(qat_ckpt, map_location=device, weights_only=False)
        try:
            model_qat_skeleton.load_state_dict(state)
        except RuntimeError as e:
            print(f"  [skip] checkpoint architecture doesn't match current QAT prep code "
                  f"(stale local .pth predating a later fix -- needs a fresh copy from PCAD): {e}")
            continue
        model_int8 = convert_deeplabv3_to_int8(model_qat_skeleton)

        int8_ckpt_path = int8_dir / f"{int8_dir.name}_best.pth"
        torch.save(model_int8.state_dict(), int8_ckpt_path)

        summary = compute_segmentation_summary(
            model_int8, img_size, val_loader=[], device=device, checkpoint_path=int8_ckpt_path,
        )
        summary["params_m"] = true_params_m
        metrics["summary"] = summary
        int8_metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"  -> model_size_mb={summary['model_size_mb']:.4f}  params_m={summary['params_m']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-dir", default="outputs/detection_segmentation/phase7")
    args = parser.parse_args()
    backfill(Path(args.phase7_dir))
