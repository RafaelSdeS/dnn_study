"""
Detection/segmentation model assembly for Phase 7.
Backbone feature extraction + SSD/DeepLab heads.
"""
import copy
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.ao.quantization as tq
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import DefaultBoxGenerator, SSDLiteHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.ops import nms

from models.compensation import AlexNetBottleneck, AlexNetFire
from models.baselines import AlexNetTV
from .quantization import find_fuse_groups


BACKBONE_FEATURE_CONFIG = {
    "alexnet_bottleneck": {
        "feature_indices": [2, 6],
        "out_channels": [192, 256],
        "ctor": AlexNetBottleneck,
    },
    "alexnet_fire": {
        "feature_indices": [2, 6],
        "out_channels": [192, 256],
        "ctor": AlexNetFire,
    },
    "alexnet_tv": {
        # [2, 12] (native stride 8/32) left the deepest pyramid level nearly
        # degenerate (2x2 @ 256px) once 2-3 extra SSDLite blocks stacked on top of
        # an already-coarse tap; [2, 5] (stride 8/17) keeps the post-extra-block
        # pyramid comparable in coarseness to the other two backbones.
        "feature_indices": [2, 5],
        "out_channels": [64, 192],
        "ctor": AlexNetTV,
    },
}


class DetSegBackbone(nn.Module):
    """Feature extractor: taps intermediate layers of classification backbone."""

    def __init__(
        self,
        arch_name: str,
        num_classes: int = 200,
        num_extra_blocks: int = 2,
        pretrained_ckpt: Optional[Path] = None,
    ):
        super().__init__()
        if arch_name not in BACKBONE_FEATURE_CONFIG:
            raise ValueError(f"Unknown arch: {arch_name}. Must be in {list(BACKBONE_FEATURE_CONFIG.keys())}")

        config = BACKBONE_FEATURE_CONFIG[arch_name]
        ctor = config["ctor"]
        feature_indices = config["feature_indices"]
        out_channels = config["out_channels"]

        # Load backbone (strip classifier head later).
        # alexnet_tv only: force random init for a fair cross-backbone comparison
        # (bottleneck/fire have no pretrained checkpoints available either). Moot
        # when pretrained_ckpt is given below — that overwrites all of backbone_full's
        # weights regardless of this initial construction.
        if arch_name == "alexnet_tv":
            self.backbone_full = ctor(num_classes=num_classes, pretrained=False)
        else:
            self.backbone_full = ctor(num_classes=num_classes)

        # Init from a Tiny-ImageNet classification checkpoint instead of random
        # weights — tests whether classification-learned features (not just
        # architecture) transfer to detection. Strict load: same ctor, same
        # num_classes, so this also doubles as an architecture-match check.
        if pretrained_ckpt is not None:
            ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
            self.backbone_full.load_state_dict(ckpt.get("model_state_dict", ckpt))

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

        # Tap from backbone. Stop after the last tapped index — some backbones
        # (bottleneck/fire) end their .features Sequential in a classification-only
        # AdaptiveAvgPool2d(1); running past the last tap would collapse the feature
        # map to 1x1 before extra_blocks ever see it.
        last_tap = max(self.feature_indices)
        for i, layer in enumerate(self.backbone_full.features):
            if i > last_tap:
                break
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


class DeepLabV3Segmenter(nn.Module):
    """DeepLabV3-style segmenter: DetSegBackbone (deepest tap only) + DeepLabHead + upsample."""

    def __init__(self, arch_name: str, num_classes: int = 21, pretrained_ckpt: Optional[Path] = None):
        super().__init__()
        self.backbone = DetSegBackbone(
            arch_name, num_classes=200, num_extra_blocks=0, pretrained_ckpt=pretrained_ckpt
        )
        self.head = DeepLabHead(self.backbone.out_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        deepest = features[str(len(self.backbone.feature_indices) - 1)]
        logits = self.head(deepest)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_deeplabv3_segmenter(
    arch_name: str,
    num_classes: int = 21,
    image_size: int = 256,
    pretrained_ckpt: Optional[Path] = None,
) -> DeepLabV3Segmenter:
    """Assemble DeepLabV3-style segmenter: backbone (deepest tap) + DeepLabHead.

    Args:
        arch_name: One of ["alexnet_bottleneck", "alexnet_fire", "alexnet_tv"]
        num_classes: 21 for VOC (20 + background), same convention as build_ssd_detector
        image_size: kept for call-site parity with build_ssd_detector; forward() upsamples
            to the actual input tensor's spatial size, not this constant.
        pretrained_ckpt: Optional Tiny-ImageNet classification checkpoint to init the
            backbone from, instead of random init (same convention as build_ssd_detector).
    """
    if arch_name not in BACKBONE_FEATURE_CONFIG:
        raise ValueError(f"Unknown arch: {arch_name}")
    return DeepLabV3Segmenter(arch_name, num_classes=num_classes, pretrained_ckpt=pretrained_ckpt)


def build_qat_deeplabv3_segmenter(model_fp32: DeepLabV3Segmenter, device: torch.device) -> DeepLabV3Segmenter:
    """Prepare a trained FP32 DeepLabV3 segmenter for QAT fine-tuning.

    Design decision: only the backbone is quantized, mirroring build_qat_ssd_detector.
    DeepLabHead (ASPP + classifier) stays FP32 — its atrous/dilated convs plus ASPPPooling's
    global-context branch (AdaptiveAvgPool2d + dynamic interpolate) have no quantization
    precedent in this codebase, same category of decision as leaving SSDLiteHead FP32 for
    detection. If FP32-vs-QAT mIoU holds up, quantizing the head is a natural follow-up,
    not a blocker for this first pass.
    """
    model = copy.deepcopy(model_fp32)
    backbone = model.backbone
    if not isinstance(backbone, DetSegBackbone):
        raise TypeError(f"Expected DetSegBackbone, got {type(backbone)}")

    backbone.train()
    backbone.qconfig = tq.get_default_qat_qconfig("fbgemm")

    # Same opt-out as build_qat_ssd_detector: backbone_full's classifier and its own
    # quant/dequant stubs are never invoked by DetSegBackbone.forward (it taps .features
    # directly and uses its own quant/dequant), and .features layers past the deepest tap
    # are unreachable too. Left qconfig'd, prepare_qat wraps them with observers that never
    # see data and tq.convert() hard-asserts on their inf min/max.
    backbone_full = backbone.backbone_full
    backbone_full.classifier.qconfig = None
    backbone_full.quant.qconfig = None
    backbone_full.dequant.qconfig = None
    last_tap = max(backbone.feature_indices)
    for i, layer in enumerate(backbone_full.features):
        if i > last_tap:
            layer.qconfig = None

    fuse_groups = find_fuse_groups(backbone.backbone_full)
    if fuse_groups:
        prefixed = [[f"backbone_full.{step}" for step in group] for group in fuse_groups]
        tq.fuse_modules_qat(backbone, prefixed, inplace=True)

    prepared_backbone = tq.prepare_qat(backbone, inplace=False)
    model.backbone = prepared_backbone
    return model.to(device)


def convert_deeplabv3_to_int8(qat_model: DeepLabV3Segmenter) -> DeepLabV3Segmenter:
    """Convert a QAT-trained DeepLabV3 segmenter's backbone to real INT8 ops. CPU-only
    (project convention); head stays FP32 to match build_qat_deeplabv3_segmenter's design."""
    qat_model = qat_model.to("cpu").eval()
    qat_model.backbone = tq.convert(qat_model.backbone, inplace=False)
    return qat_model


def build_ssd_detector(
    arch_name: str,
    num_classes: int = 21,
    image_size: int = 256,
    confidence_threshold: float = 0.05,
    pretrained_ckpt: Optional[Path] = None,
) -> SSD:
    """Assemble SSD detector: backbone + anchor generator + head.

    Args:
        arch_name: One of ["alexnet_bottleneck", "alexnet_fire", "alexnet_tv"]
        num_classes: 21 for VOC (20 + background)
        image_size: Input resolution (default 256 for Phase 7)
        confidence_threshold: NMS confidence threshold
        pretrained_ckpt: Optional Tiny-ImageNet classification checkpoint to init
            the backbone from, instead of random init

    Returns:
        SSD model ready for training/eval
    """
    if arch_name not in BACKBONE_FEATURE_CONFIG:
        raise ValueError(f"Unknown arch: {arch_name}")

    # Backbone
    backbone = DetSegBackbone(
        arch_name, num_classes=200, num_extra_blocks=3, pretrained_ckpt=pretrained_ckpt
    )

    # Anchor generator — explicit scales (not min/max_ratio) placed at VOC's actual
    # GT-box-size percentiles. DefaultBoxGenerator's min/max_ratio interpolates scales
    # linearly, which left a large gap (0.1 -> 0.383 under the old min=0.1/max=0.95)
    # exactly where most VOC boxes live (median GT box side-ratio 0.265) — anchor
    # recall was ~0.78-0.80 regardless of image resolution. See check_anchor_recall.py.
    aspect_ratios = [[1.5, 2, 3, 4]] * len(backbone.out_channels)
    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=aspect_ratios,
        scales=[0.03, 0.08, 0.16, 0.3, 0.55, 1.0],
    )

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

    # backbone_full's own classifier and its standalone quant/dequant stubs are never
    # invoked by DetSegBackbone.forward (which taps .features directly and uses its own
    # quant/dequant — see the comment on those stubs above); .features layers past the
    # deepest tap are unreachable too. qconfig otherwise propagates to all of these, so
    # prepare_qat wraps them with observers that never see data, and tq.convert() later
    # hard-asserts on their inf min/max (confirmed via the fp32_pretrained/qat checkpoints
    # that hit this in production — e.g. jobs 811687/811101/811132). Opt them out.
    backbone_full = backbone.backbone_full
    backbone_full.classifier.qconfig = None
    backbone_full.quant.qconfig = None
    backbone_full.dequant.qconfig = None
    last_tap = max(backbone.feature_indices)
    for i, layer in enumerate(backbone_full.features):
        if i > last_tap:
            layer.qconfig = None

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


def trim_dead_backbone_weights(model: nn.Module) -> None:
    """Drop backbone_full submodules DetSegBackbone.forward never reaches (in place).

    DetSegBackbone keeps the whole classification network (backbone_full) around only to
    support loading a pretrained Tiny-ImageNet checkpoint, but forward() walks backbone_full
    .features only up to the deepest tap and never touches backbone_full.classifier (or any
    .features layers past that tap) -- see DetSegBackbone.forward's "break" past last_tap. Those
    layers still get saved in every training/QAT/INT8 checkpoint at full precision: for
    alexnet_tv that's ~57M of ~58M "backbone" params (its stock torchvision classifier), which is
    also never quantized (qconfig=None), so raw checkpoint size barely reflects the deployed
    model or the QAT->INT8 size drop. Call after loading a trained checkpoint, before measuring
    or saving a "true" deployable size -- safe because nothing downstream of this call touches
    the dropped submodules.
    """
    backbone = model.backbone
    last_tap = max(backbone.feature_indices)
    backbone_full = backbone.backbone_full
    backbone_full.features = backbone_full.features[: last_tap + 1]
    if hasattr(backbone_full, "classifier"):
        del backbone_full.classifier
    if hasattr(backbone_full, "avgpool"):
        del backbone_full.avgpool


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
    images_seen = 0

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            if images_seen >= max_samples:
                break
            images_seen += images.shape[0]

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
