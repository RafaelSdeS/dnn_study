"""
Detection/segmentation model assembly for Phase 7.
Backbone feature extraction + SSD/DeepLab heads.
"""
import copy
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.ao.quantization as tq
import torch.nn as nn
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import DefaultBoxGenerator, SSDLiteHead
from torchvision.ops import nms

from models.compensation import AlexNetBottleneck, AlexNetFire
from models.baselines import AlexNetTV
from .quantization import find_fuse_groups


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

        # Load backbone (strip classifier head later).
        # alexnet_tv only: force random init for a fair cross-backbone comparison
        # (bottleneck/fire have no pretrained checkpoints available either).
        if arch_name == "alexnet_tv":
            self.backbone_full = ctor(num_classes=num_classes, pretrained=False)
        else:
            self.backbone_full = ctor(num_classes=num_classes)

        # Tap intermediate features
        self.feature_indices = feature_indices
        self.out_channels = list(out_channels)  # copy: _build_extra_blocks appends in place

        # Build extra blocks (SSDLite-style depthwise separable)
        self.extra_blocks = self._build_extra_blocks(out_channels[-1], num_extra_blocks)

        # QAT stubs: identity in FP32; become real quantize/dequantize ops after
        # prepare_qat()+convert(). Placed here (not on backbone_full) since forward()
        # taps backbone_full.features[i] directly, bypassing backbone_full's own
        # quant/dequant stubs (those exist for backbone_full's standalone classification
        # QAT pipeline and are unused/inert in this detection path).
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

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

        x = self.quant(x)

        # Tap from backbone
        for i, layer in enumerate(self.backbone_full.features):
            x = layer(x)
            if i in self.feature_indices:
                # Dequantize a copy for the (FP32) SSD head; keep x itself in the
                # (fake-)quantized domain so later taps aren't re-quantized redundantly.
                features[str(level)] = self.dequant(x)
                level += 1

        # Extra blocks (stride-2 downsampling on deepest feature)
        for block in self.extra_blocks:
            x = block(x)
            features[str(level)] = self.dequant(x)
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


def build_qat_ssd_detector(model_fp32: SSD, device: torch.device) -> SSD:
    """Prepare a trained FP32 SSD detector for QAT fine-tuning.

    Design decision: only the backbone (Conv-BN-ReLU stack) is quantized. SSDLiteHead's
    classification/regression convs stay FP32 — per PHASE7_PLAN.md's own fallback for
    INT8 instability on continuous box-regression outputs (Blocking Issue #2). This is
    the safe first cut: the backbone's Conv-BN-ReLU pattern is already QAT-validated in
    Phase 3/6 (same fuse_groups machinery), while quantizing a box-regression head has
    no precedent in this codebase. If FP32-vs-QAT mAP holds up, quantizing the head too
    is a natural follow-up, not a blocker for this first pass.

    Args:
        model_fp32: SSD model with a DetSegBackbone (from build_ssd_detector), already
            trained to its FP32 best checkpoint.
        device: CUDA device for QAT fine-tuning (fake-quant training still runs on GPU;
            only the final convert() step is CPU-only).

    Returns:
        SSD model with backbone fused + fake-quant-observer-wrapped; head untouched.
    """
    model = copy.deepcopy(model_fp32)
    backbone = model.backbone
    if not isinstance(backbone, DetSegBackbone):
        raise TypeError(f"Expected DetSegBackbone, got {type(backbone)}")

    backbone.train()
    backbone.qconfig = tq.get_default_qat_qconfig("fbgemm")

    # Fuse Conv-BN(-ReLU) triples inside backbone_full (bottleneck/fire only —
    # alexnet_tv's stock torchvision features have no BatchNorm, so find_fuse_groups
    # returns [] there and prepare_qat still inserts fake-quant on bare Conv2d/ReLU).
    fuse_groups = find_fuse_groups(backbone.backbone_full)
    if fuse_groups:
        prefixed = [[f"backbone_full.{step}" for step in group] for group in fuse_groups]
        tq.fuse_modules_qat(backbone, prefixed, inplace=True)

    prepared_backbone = tq.prepare_qat(backbone, inplace=False)
    model.backbone = prepared_backbone

    return model.to(device)


def convert_ssd_to_int8(qat_model: SSD) -> SSD:
    """Convert a QAT-trained SSD's backbone to real INT8 ops. CPU-only (project convention);
    head stays FP32 to match build_qat_ssd_detector's design."""
    qat_model = qat_model.to("cpu").eval()
    qat_model.backbone = tq.convert(qat_model.backbone, inplace=False)
    return qat_model


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
    from torchvision.models.detection.image_list import ImageList
    from torchvision.ops import box_iou

    total_boxes = 0
    matched_boxes = 0

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            if total_boxes >= max_samples * 100:  # ~100 boxes per sample
                break

            # Get default boxes (anchors) for this batch
            # backbone returns OrderedDict; anchor_generator expects list of tensors
            # and images wrapped in ImageList (not a raw batched Tensor)
            features = model.backbone(images)
            feature_list = [features[str(i)] for i in range(len(features))]
            image_sizes = [(images.shape[-2], images.shape[-1])] * images.shape[0]
            anchors = model.anchor_generator(ImageList(images, image_sizes), feature_list)

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
