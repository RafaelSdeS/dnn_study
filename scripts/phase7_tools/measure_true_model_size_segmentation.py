"""Measure the *true* deployable size of Phase 7 segmentation checkpoints.

The on-disk `*_best.pth` size (what the notebook's `ckpt_size_mb` reads) and the existing
`summary.model_size_mb` (from compute_segmentation_summary / scripts/phase7_tools/backfill_int8_size_segmentation.py)
both still include backbone_full's dead weight -- see ml.det_seg_models.trim_dead_backbone_weights
for why: DetSegBackbone carries the *entire* classification network for pretrained-checkpoint
loading, but forward() only uses a prefix of its `.features` and never touches `.classifier`.
For alexnet_tv that dead weight is ~57M of ~58M "backbone" params (its stock torchvision
classifier), so the raw checkpoint size barely reflects the deployed model and barely shrinks
across FP32/QAT/INT8. This script rebuilds each stage's model from its checkpoint, drops that
dead weight, and writes the honest size to metrics.json as `summary.true_size_mb` (existing
fields are left alone for comparison).

Usage:
    python -m scripts.phase7_tools.measure_true_model_size_segmentation
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import torch
import yaml

from ml.det_seg_models import (
    build_deeplabv3_segmenter, build_qat_deeplabv3_segmenter, convert_deeplabv3_to_int8,
    trim_dead_backbone_weights,
)

RUN_NAME_RE = re.compile(
    r"^seg_(?P<model>alexnet_(?:bottleneck|fire|tv))_(?P<stage>fp32|qat|int8)"
    r"(?P<pretrained>_pretrained)?_(?P<exp>.+)$"
)


def serialized_size_mb(state_dict) -> float:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.tell() / 1024 ** 2


def write_true_size(run_dir: Path, model: torch.nn.Module) -> None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"  [skip] {run_dir.name}: no metrics.json")
        return
    trim_dead_backbone_weights(model)
    true_size_mb = serialized_size_mb(model.state_dict())
    metrics = json.loads(metrics_path.read_text())
    metrics.setdefault("summary", {})["true_size_mb"] = true_size_mb
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  {run_dir.name}: true_size_mb={true_size_mb:.4f}")


def measure(phase7_dir: Path) -> None:
    device = torch.device("cpu")

    for run_dir in sorted(phase7_dir.glob("seg_*")):
        if not run_dir.is_dir():
            continue
        m = RUN_NAME_RE.match(run_dir.name)
        if not m or m.group("stage") not in ("fp32", "qat"):
            continue
        model_name = m.group("model")
        ckpt_path = run_dir / f"{run_dir.name}_best.pth"
        if not ckpt_path.exists():
            print(f"[skip] {run_dir.name}: no _best.pth checkpoint locally")
            continue

        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        img_size = config["data"]["img_size"]
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

        fp32_skeleton = build_deeplabv3_segmenter(model_name, num_classes=21, image_size=img_size)
        if m.group("stage") == "fp32":
            model = fp32_skeleton
            model.load_state_dict(state)
            print(f"[measure] {run_dir.name} (fp32)")
            write_true_size(run_dir, model)
        else:
            model_qat = build_qat_deeplabv3_segmenter(fp32_skeleton, device)
            try:
                model_qat.load_state_dict(state)
            except RuntimeError as e:
                print(f"  [skip] {run_dir.name}: checkpoint doesn't match current QAT prep code: {e}")
                continue
            print(f"[measure] {run_dir.name} (qat)")
            write_true_size(run_dir, model_qat)

            # Derive the matching INT8 run's true size too, same conversion the training
            # script does at eval time -- see scripts/phase7_tools/backfill_int8_size_segmentation.py.
            int8_dir = run_dir.parent / run_dir.name.replace("_qat_", "_int8_")
            if not int8_dir.is_dir():
                continue
            model_qat2 = build_qat_deeplabv3_segmenter(
                build_deeplabv3_segmenter(model_name, num_classes=21, image_size=img_size), device
            )
            model_qat2.load_state_dict(state)
            model_int8 = convert_deeplabv3_to_int8(model_qat2)
            print(f"[measure] {int8_dir.name} (int8, derived from {run_dir.name})")
            write_true_size(int8_dir, model_int8)


if __name__ == "__main__":
    measure(Path("outputs/detection_segmentation/phase7"))
