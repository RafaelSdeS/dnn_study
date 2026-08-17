"""Diagnose whether Phase 7 segmentation models learned real structure or collapsed to a
near-constant (mostly-background) prediction -- the low absolute mIoU (~0.16-0.18) and the
fact that FP32/QAT/INT8 mIoU are all nearly identical are each individually consistent with
either "quantization is robust" or "the model barely learned anything, so nothing moves it."
This script tells them apart: per-class IoU (a degenerate model gets ~0 IoU on every
foreground class), predicted-vs-ground-truth pixel-class distribution (a degenerate model's
prediction histogram is background-only), and a qualitative image/GT/pred grid.

Runs FP32 checkpoints only (best-trained stage, most informative for this question).

Usage:
    python -m scripts.diagnose_segmentation_quality
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.classification import MulticlassJaccardIndex

from ml import DetSegDataConfig, create_voc_segmentation_loaders
from ml.det_seg_data import VOC_CLASSES
from ml.det_seg_models import build_deeplabv3_segmenter
from ml.runtime import expand_path

ARCHES = ["alexnet_bottleneck", "alexnet_fire", "alexnet_tv"]
PHASE7_DIR = Path("outputs/detection_segmentation/phase7")
FIGURES_DIR = Path("results/figures_generated/phase_7_detection_segmentation")
CLASS_NAMES = ["background"] + VOC_CLASSES  # index 0..20
N_SAMPLES = 4  # qualitative grid: same val images across all 3 archs


def load_model(arch: str, device: torch.device) -> torch.nn.Module:
    run_dir = PHASE7_DIR / f"seg_{arch}_fp32_phase7_segmentation"
    ckpt_path = run_dir / f"{run_dir.name}_best.pth"
    model = build_deeplabv3_segmenter(arch, num_classes=21, image_size=256)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model.to(device).eval()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = DetSegDataConfig(
        img_size=256, voc_root=expand_path("~/.cache/torchvision/datasets"),
        batch_size=16, num_workers=4,
    )
    _, val_ds, _, val_loader = create_voc_segmentation_loaders(data_cfg)
    print(f"Val set: {len(val_ds)} images\n")

    per_class_iou = {}
    pixel_frac = {}
    sample_preds = {}  # arch -> list of (image, gt, pred) for the first N_SAMPLES

    gt_pixel_counts = np.zeros(21, dtype=np.int64)
    gt_counted = False

    for arch in ARCHES:
        print(f"[eval] {arch} (FP32)")
        model = load_model(arch, device)
        iou_metric = MulticlassJaccardIndex(num_classes=21, ignore_index=255, average=None)
        pred_pixel_counts = np.zeros(21, dtype=np.int64)
        samples = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)

                iou_metric.update(preds.cpu(), masks)
                valid = masks != 255
                pred_pixel_counts += np.bincount(preds.cpu()[valid].numpy(), minlength=21)
                if not gt_counted:
                    gt_pixel_counts_batch = np.bincount(masks[valid].numpy(), minlength=21)
                    gt_pixel_counts[: len(gt_pixel_counts_batch)] += gt_pixel_counts_batch

                if len(samples) < N_SAMPLES:
                    for i in range(min(N_SAMPLES - len(samples), images.shape[0])):
                        samples.append((
                            images[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy(),
                            masks[i].numpy(),
                            preds[i].cpu().numpy(),
                        ))

        gt_counted = True
        per_class_iou[arch] = iou_metric.compute().numpy()
        pixel_frac[arch] = pred_pixel_counts / pred_pixel_counts.sum()
        sample_preds[arch] = samples

    gt_frac = gt_pixel_counts / gt_pixel_counts.sum()

    # --- Per-class IoU table ---
    print("\nPer-class IoU (FP32):")
    header = f"{'class':14s}" + "".join(f"{a:>18s}" for a in ARCHES)
    print(header)
    for c in range(21):
        row = f"{CLASS_NAMES[c]:14s}" + "".join(f"{per_class_iou[a][c]:18.4f}" for a in ARCHES)
        print(row)

    # --- Pixel-class distribution: GT vs predicted (collapse check) ---
    print("\nPixel-class fraction, ground truth vs. predicted (top 6 GT classes):")
    top_classes = np.argsort(-gt_frac)[:6]
    header = f"{'class':14s}{'gt_frac':>12s}" + "".join(f"{a + '_pred':>18s}" for a in ARCHES)
    print(header)
    for c in top_classes:
        row = f"{CLASS_NAMES[c]:14s}{gt_frac[c]:12.4f}" + "".join(f"{pixel_frac[a][c]:18.4f}" for a in ARCHES)
        print(row)

    # --- Qualitative grid: rows = sample images, cols = [input, GT, bottleneck, fire, tv] ---
    cmap = plt.get_cmap("tab20", 21)
    fig, axes = plt.subplots(N_SAMPLES, 2 + len(ARCHES), figsize=(3.2 * (2 + len(ARCHES)), 3.2 * N_SAMPLES))
    for row in range(N_SAMPLES):
        img, gt, _ = sample_preds[ARCHES[0]][row]
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("image" if row == 0 else "")
        axes[row, 1].imshow(gt, cmap=cmap, vmin=0, vmax=20)
        axes[row, 1].set_title("ground truth" if row == 0 else "")
        for col, arch in enumerate(ARCHES):
            _, _, pred = sample_preds[arch][row]
            ax = axes[row, 2 + col]
            ax.imshow(pred, cmap=cmap, vmin=0, vmax=20)
            ax.set_title(arch if row == 0 else "")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "phase7_seg_quality_diagnostic.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved qualitative grid: {out_path}")


if __name__ == "__main__":
    main()
