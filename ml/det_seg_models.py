"""
Detection/segmentation model assembly for Phase 7.
Backbone feature extraction + SSD/DeepLab heads.
"""
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import DefaultBoxGenerator, SSDLiteHead
from torchvision.ops import nms

from models.compensation import AlexNetBottleneck, AlexNetFire
from models.baselines import AlexNetTV


BACKBONE_FEATURE_CONFIG = {
    "alexnet_bottleneck": {
        "feature_indices": [3, 6],
        "out_channels": [192, 256],
        "ctor": AlexNetBottleneck,
    },
    "alexnet_fire": {
        "feature_indices": [3, 6],
        "out_channels": [192, 256],
        "ctor": AlexNetFire,
    },
    "alexnet_tv": {
        "feature_indices": [2, 12],
        "out_channels": [64, 256],
        "ctor": AlexNetTV,
    },
}


class DetSegBackbone(nn.Module):
    """Feature extractor: taps intermediate layers of classification backbone."""

    def __init__(self, arch_name: str, num_classes: int = 200, num_extra_blocks: int = 2):
        super().__init__()
        if arch_name not in BACKBONE_FEATURE_CONFIG:
            raise ValueError(f"Unknown arch: {arch_name}. Must be in {list(BACKBONE_FEATURE_CONFIG.keys())}")

        config = BACKBONE_FEATURE_CONFIG[arch_name]
        ctor = config["ctor"]
        feature_indices = config["feature_indices"]
        out_channels = config["out_channels"]

        # Load backbone (strip classifier head later)
        self.backbone_full = ctor(num_classes=num_classes)

        # Tap intermediate features
        self.feature_indices = feature_indices
        self.out_channels = out_channels

        # Build extra blocks (SSDLite-style depthwise separable)
        self.extra_blocks = self._build_extra_blocks(out_channels[-1], num_extra_blocks)

    def _build_extra_blocks(self, in_channels: int, num_blocks: int) -> nn.ModuleList:
        """Build SSDLite extra blocks (depthwise separable + stride-2 downsampling)."""
        blocks = nn.ModuleList()
        for i in range(num_blocks):
            out_channels = in_channels * 2 if i == 0 else in_channels
            block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=True),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, groups=in_channels, bias=True),
                nn.ReLU(inplace=False),
            )
            blocks.append(block)
            in_channels = out_channels
            self.out_channels.append(out_channels)
        return blocks

    def forward(self, x: torch.Tensor) -> OrderedDict:
        """Extract feature maps at tapped indices + extra blocks."""
        features = OrderedDict()
        level = 0

        # Tap from backbone
        for i, layer in enumerate(self.backbone_full.features):
            x = layer(x)
            if i in self.feature_indices:
                features[str(level)] = x
                level += 1

        # Extra blocks (stride-2 downsampling on deepest feature)
        for block in self.extra_blocks:
            x = block(x)
            features[str(level)] = x
            level += 1

        return features


def build_deeplabv3_segmenter(
    arch_name: str,
    num_classes: int = 21,
    image_size: int = 256,
) -> nn.Module:
    """Build DeepLabV3 segmenter with custom backbone.

    For now, uses torchvision's DeepLabV3 pretrained on COCO.
    TODO: Replace backbone with Phase 7 models for proper ablation.
    """
    from torchvision.models.segmentation import deeplabv3_resnet50

    # TODO: Implement custom DeepLabV3 with alexnet backbones
    # For now, return placeholder
    model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes, aux_loss=False)
    return model


def build_ssd_detector(
    arch_name: str,
    num_classes: int = 21,
    image_size: int = 256,
    confidence_threshold: float = 0.05,
) -> SSD:
    """Assemble SSD detector: backbone + anchor generator + head.

    Args:
        arch_name: One of ["alexnet_bottleneck", "alexnet_fire", "alexnet_tv"]
        num_classes: 21 for VOC (20 + background)
        image_size: Input resolution (default 256 for Phase 7)
        confidence_threshold: NMS confidence threshold

    Returns:
        SSD model ready for training/eval
    """
    if arch_name not in BACKBONE_FEATURE_CONFIG:
        raise ValueError(f"Unknown arch: {arch_name}")

    config = BACKBONE_FEATURE_CONFIG[arch_name]
    out_channels = config["out_channels"] + config["out_channels"][-1:] * 2  # +2 extra blocks

    # Backbone
    backbone = DetSegBackbone(arch_name, num_classes=200, num_extra_blocks=2)

    # Anchor generator
    aspect_ratios = [[2, 3]] * len(backbone.out_channels)
    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=aspect_ratios,
        min_ratio=0.2,
        max_ratio=0.95,
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]

    # Head
    head = SSDLiteHead(
        in_channels=backbone.out_channels,
        num_anchors=anchor_generator.num_anchors_per_location(),
        num_classes=num_classes,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.03),
    )

    # Full SSD model
    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(image_size, image_size),
        num_classes=num_classes,
        head=head,
    )

    # Override score_thresh for NMS
    model.score_thresh = confidence_threshold

    return model


def compute_anchor_recall(
    model: SSD,
    dataloader,
    iou_threshold: float = 0.5,
    max_samples: int = 1000,
) -> float:
    """Compute what fraction of ground-truth boxes overlap any default box at IoU > threshold.

    This is a cheap sanity check: if anchor recall is <95%, mAP will be capped regardless of
    how good the backbone/head are.
    """
    from torchvision.ops import box_iou

    total_boxes = 0
    matched_boxes = 0

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            if total_boxes >= max_samples * 100:  # ~100 boxes per sample
                break

            # Get default boxes (anchors) for this batch
            anchors = model.anchor_generator(images, model.backbone(images))

            for img_anchors, target in zip(anchors, targets):
                gt_boxes = target["boxes"]
                if len(gt_boxes) == 0:
                    continue

                # Compute IoU between GT boxes and anchors
                iou_matrix = box_iou(gt_boxes, img_anchors)  # (num_gt, num_anchors)
                max_iou_per_gt = iou_matrix.max(dim=1)[0]  # (num_gt,)

                # Count how many GT boxes have at least one anchor with IoU > threshold
                matched = (max_iou_per_gt > iou_threshold).sum().item()
                matched_boxes += matched
                total_boxes += len(gt_boxes)

    recall = matched_boxes / max(total_boxes, 1)
    return recall


def demo(arch_name: str = "alexnet_bottleneck"):
    """Smoke check: forward pass through detector, validate output shapes."""
    print(f"Loading {arch_name}...")
    model = build_ssd_detector(arch_name, num_classes=21, image_size=256)
    model.eval()

    print(f"Testing forward pass...")
    dummy = torch.randn(2, 3, 256, 256)

    # Train mode (returns loss dict)
    model.train()
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[50.0, 50.0, 200.0, 200.0]]), "labels": torch.tensor([5])},
    ]
    loss_dict = model(dummy, targets)
    assert isinstance(loss_dict, dict)
    assert "bbox_regression" in loss_dict and "classification" in loss_dict
    assert not torch.isnan(loss_dict["bbox_regression"]).any()
    assert not torch.isnan(loss_dict["classification"]).any()
    print(f"  ✓ Train mode loss dict: {loss_dict}")

    # Eval mode (returns detections)
    model.eval()
    with torch.no_grad():
        detections = model(dummy)
    assert len(detections) == 2
    for det in detections:
        assert "boxes" in det and "scores" in det and "labels" in det
        assert det["boxes"].shape[1] == 4
    print(f"  ✓ Eval mode detections: {len(detections)} images")

    print(f"\n✓✓✓ DEMO PASSED ✓✓✓")


if __name__ == "__main__":
    demo()
