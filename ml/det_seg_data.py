"""
Detection/segmentation data pipelines for Phase 7.
VOC 2007+2012 for detection, VOC 2012 for segmentation.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import VOCDetection, VOCSegmentation
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from .config import DetSegDataConfig


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_NAME_TO_IDX = {name: idx + 1 for idx, name in enumerate(VOC_CLASSES)}  # 1-20, 0 = background


def _parse_voc_detection_target(xml_path: str) -> Tuple[list, list]:
    """Parse VOC XML annotation into (boxes, labels).

    Returns:
        boxes: list of [xmin, ymin, xmax, ymax] (absolute pixels)
        labels: list of class indices (1-20, 0 reserved for background)
    """
    root = ET.parse(xml_path).getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_NAME_TO_IDX[name])
    return boxes, labels


class VOCDetectionWrapper(VOCDetection):
    """Wrapper around torchvision.datasets.VOCDetection to return tv_tensors."""

    def __getitem__(self, idx: int):
        image, target = super().__getitem__(idx)

        # Parse XML annotation
        image_id = self.ids[idx]
        anno_path = self.annotations.format(*image_id)
        boxes, labels = _parse_voc_detection_target(anno_path)

        # Convert to torch tensors and wrap in tv_tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        boxes = BoundingBoxes(boxes, format="xyxy", canvas_size=image.size[::-1])

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, {"boxes": boxes, "labels": labels}


def _detection_collate_fn(batch):
    """Custom collate: images stack, targets stay as list (variable boxes per image)."""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def _build_detection_transforms(img_size: int, train: bool):
    """Build v2 transform pipeline for detection."""
    if train:
        return transforms.Compose([
            transforms.ToImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToDtype(torch.float32, scale=True),
        ])
    else:
        return transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToDtype(torch.float32, scale=True),
        ])


def create_voc_detection_loaders(cfg: DetSegDataConfig) -> Tuple:
    """Create VOC detection train/val loaders (07+12 trainval for train, 07 test for eval).

    Args:
        cfg: DetSegDataConfig with img_size, voc_root, batch_size, num_workers, seed

    Returns:
        (train_ds, val_ds, train_loader, val_loader)
    """
    torch.manual_seed(cfg.seed)

    train_transform = _build_detection_transforms(cfg.img_size, train=True)
    val_transform = _build_detection_transforms(cfg.img_size, train=False)

    # VOC 07+12 trainval for training
    voc07_train = VOCDetectionWrapper(
        root=cfg.voc_root, year="2007", image_set="trainval", download=True, transforms=train_transform
    )
    voc12_train = VOCDetectionWrapper(
        root=cfg.voc_root, year="2012", image_set="trainval", download=True, transforms=train_transform
    )
    train_ds = ConcatDataset([voc07_train, voc12_train])

    # VOC 07 test for evaluation
    val_ds = VOCDetectionWrapper(
        root=cfg.voc_root, year="2007", image_set="test", download=True, transforms=val_transform
    )

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


def create_voc_segmentation_loaders(cfg: DetSegDataConfig) -> Tuple:
    """Create VOC segmentation train/val loaders (2012 only).

    Deferred to Stage 6; placeholder here for API consistency.
    """
    raise NotImplementedError("Segmentation loaders deferred to Stage 6")


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
        transforms.functional.to_image(img_vis).save(str(save_path))
        print(f"Saved: {save_path}")

    print(f"\n✓ Visualizations saved to {save_dir}/")
    print("\n=== SMOKE CHECK PASSED ===")


if __name__ == "__main__":
    demo()
