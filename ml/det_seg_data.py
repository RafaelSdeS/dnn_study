"""
Detection/segmentation data pipelines for Phase 7.
VOC 2007+2012 for detection, VOC 2012 for segmentation.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import VOCDetection, VOCSegmentation
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage
from torchvision.utils import draw_bounding_boxes

from .config import DetSegDataConfig


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_NAME_TO_IDX = {name: idx + 1 for idx, name in enumerate(VOC_CLASSES)}  # 1-20, 0 = background

# Standard SSD-training augmentations (torchvision's own recipe). Order matters:
# photometric distort first (pixel-only), then geometry ops, sanitize last to
# drop boxes degenerated/cropped away by RandomIoUCrop.
_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomPhotometricDistort(p=0.5),
    transforms.RandomZoomOut(fill=0, side_range=(1.0, 3.0), p=0.5),
    transforms.RandomIoUCrop(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.SanitizeBoundingBoxes(),
])


class VOCDetectionDataset(Dataset):
    """Minimal VOC detection wrapper that parses XML to tv_tensors."""

    def __init__(self, voc_dataset: VOCDetection, img_size: int = 256, augment: bool = False):
        self.voc_dataset = voc_dataset
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx: int):
        image, target = self.voc_dataset[idx]

        # target is a dict with 'annotation' key which is also a dict
        boxes = []
        labels = []
        anno = target.get("annotation")
        if anno is not None:
            for obj in anno.get("object", []):
                name = obj.get("name")
                bbox = obj.get("bndbox")
                if name in CLASS_NAME_TO_IDX and bbox is not None:
                    xmin = float(bbox["xmin"])
                    ymin = float(bbox["ymin"])
                    xmax = float(bbox["xmax"])
                    ymax = float(bbox["ymax"])
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(CLASS_NAME_TO_IDX[name])

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Convert PIL to tensor
        img_w, img_h = image.size
        image_np = np.array(image)
        image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        if self.augment:
            image = TVImage(image)
            boxes_tvt = BoundingBoxes(boxes, format="xyxy", canvas_size=(img_h, img_w))
            image, aug_target = _TRAIN_TRANSFORMS(image, {"boxes": boxes_tvt, "labels": labels})
            boxes, labels = aug_target["boxes"], aug_target["labels"]

        # Resize image and scale boxes
        image, boxes, labels = _resize_image_and_boxes(image, boxes, labels, self.img_size)

        # Wrap boxes in BoundingBoxes (for consistency, though not strictly needed if not using v2 transforms)
        boxes = BoundingBoxes(boxes, format="xyxy", canvas_size=(self.img_size, self.img_size))

        return image, {"boxes": boxes, "labels": labels}


def _detection_collate_fn(batch):
    """Custom collate: images stack, targets stay as list (variable boxes per image)."""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def _resize_image_and_boxes(image, boxes, labels, img_size: int):
    """Resize image and scale boxes proportionally."""
    # image is tensor float32 (C, H, W), boxes are float32 (N, 4) in [xmin, ymin, xmax, ymax]
    old_h, old_w = image.shape[1], image.shape[2]

    # Resize image using torch functional
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)

    # Scale boxes
    if len(boxes) > 0:
        scale_x = img_size / old_w
        scale_y = img_size / old_h
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= scale_x  # xmin, xmax
        boxes[:, [1, 3]] *= scale_y  # ymin, ymax

    return image, boxes, labels


def create_voc_detection_loaders(cfg: DetSegDataConfig) -> Tuple:
    """Create VOC detection train/val loaders (07+12 trainval for train, 07 test for eval).

    Args:
        cfg: DetSegDataConfig with img_size, voc_root, batch_size, num_workers, seed

    Returns:
        (train_ds, val_ds, train_loader, val_loader)
    """
    torch.manual_seed(cfg.seed)

    # VOC 07+12 trainval for training
    voc07_raw = VOCDetection(
        root=cfg.voc_root, year="2007", image_set="trainval", download=True
    )
    voc12_raw = VOCDetection(
        root=cfg.voc_root, year="2012", image_set="trainval", download=True
    )
    voc07_train_ds = VOCDetectionDataset(voc07_raw, img_size=cfg.img_size, augment=True)
    voc12_train_ds = VOCDetectionDataset(voc12_raw, img_size=cfg.img_size, augment=True)
    train_ds = ConcatDataset([voc07_train_ds, voc12_train_ds])

    # VOC 07 test for evaluation
    voc07_test_raw = VOCDetection(
        root=cfg.voc_root, year="2007", image_set="test", download=True
    )
    val_ds = VOCDetectionDataset(voc07_test_raw, img_size=cfg.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_detection_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_detection_collate_fn,
    )

    return train_ds, val_ds, train_loader, val_loader


class VOCSegmentationDataset(Dataset):
    """Minimal VOC segmentation wrapper."""

    def __init__(self, voc_dataset: VOCSegmentation, img_size: int = 256):
        self.voc_dataset = voc_dataset
        self.img_size = img_size

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx: int):
        image, target = self.voc_dataset[idx]

        # Convert PIL to tensors
        img_w, img_h = image.size
        image_np = np.array(image)
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        # Mask (already numpy array)
        mask_t = torch.from_numpy(np.array(target)).long()

        # Resize image and mask
        image_t = torch.nn.functional.interpolate(image_t.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
        mask_t = torch.nn.functional.interpolate(mask_t.unsqueeze(0).unsqueeze(0).float(), size=(self.img_size, self.img_size), mode='nearest').squeeze(0).squeeze(0).long()

        return image_t, mask_t


def create_voc_segmentation_loaders(cfg: DetSegDataConfig) -> Tuple:
    """Create VOC segmentation train/val loaders (2012 only)."""
    torch.manual_seed(cfg.seed)

    # VOC 2012 train/val
    voc12_train_raw = VOCSegmentation(
        root=cfg.voc_root, year="2012", image_set="train", download=True
    )
    voc12_val_raw = VOCSegmentation(
        root=cfg.voc_root, year="2012", image_set="val", download=True
    )

    train_ds = VOCSegmentationDataset(voc12_train_raw, img_size=cfg.img_size)
    val_ds = VOCSegmentationDataset(voc12_val_raw, img_size=cfg.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_ds, val_ds, train_loader, val_loader


def demo(cfg: Optional[DetSegDataConfig] = None, num_samples: int = 4):
    """Load one batch, validate tensor format, visualize.

    Checks:
    - Images: float32, shape (B, 3, 256, 256), values in [0, 1]
    - Boxes: float32, format [xmin, ymin, xmax, ymax], values in [0, 256]
    - Labels: int64, values in [1, 20]
    """
    if cfg is None:
        cfg = DetSegDataConfig(
            img_size=256,
            voc_root=os.path.expanduser("~/.cache/torchvision/datasets"),
            batch_size=4,
            num_workers=0,
        )

    print("Loading VOC detection data...")
    _, _, train_loader, _ = create_voc_detection_loaders(cfg)

    images, targets = next(iter(train_loader))

    # Tensor format checks
    print(f"\n=== TENSOR FORMAT VALIDATION ===")
    print(f"Images shape: {images.shape}, dtype: {images.dtype}")
    assert images.shape[1:] == (3, cfg.img_size, cfg.img_size), f"Expected (B, 3, {cfg.img_size}, {cfg.img_size})"
    assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
    assert images.min() >= 0 and images.max() <= 1, f"Values out of [0, 1]: [{images.min()}, {images.max()}]"
    print("✓ Images OK")

    for i, target in enumerate(targets):
        boxes = target["boxes"]
        labels = target["labels"]

        print(f"\nSample {i}: {len(labels)} objects")
        print(f"  Boxes shape: {boxes.shape}, dtype: {boxes.dtype}")
        print(f"  Boxes range: [{boxes.min():.1f}, {boxes.max():.1f}]")
        print(f"  Labels: {labels.tolist()}")

        assert boxes.dtype == torch.float32, f"Expected float32, got {boxes.dtype}"
        assert (boxes >= 0).all() and (boxes <= cfg.img_size).all(), \
            f"Boxes out of [0, {cfg.img_size}]: [{boxes.min()}, {boxes.max()}]"
        assert labels.dtype == torch.int64, f"Expected int64, got {labels.dtype}"
        assert (labels >= 1).all() and (labels <= 20).all(), \
            f"Labels out of [1, 20]: {labels.unique().tolist()}"

    print("\n✓ All tensor format checks passed")

    # Visualization
    print(f"\n=== VISUALIZATION ===")
    save_dir = Path("./phase7_demo_vis")
    save_dir.mkdir(exist_ok=True)

    for i in range(min(num_samples, len(images))):
        img = (images[i] * 255).byte()
        boxes = targets[i]["boxes"]
        labels = targets[i]["labels"]

        if len(boxes) > 0:
            labels_str = [f"{VOC_CLASSES[int(l)-1]}:{l}" for l in labels]
            img_vis = draw_bounding_boxes(img, boxes, labels=labels_str, width=2, colors="red")
        else:
            img_vis = img

        save_path = save_dir / f"sample_{i}.png"
        transforms.functional.to_pil_image(img_vis).save(str(save_path))
        print(f"Saved: {save_path}")

    print(f"\n✓ Visualizations saved to {save_dir}/")
    print("\n=== SMOKE CHECK PASSED ===")


if __name__ == "__main__":
    demo()
