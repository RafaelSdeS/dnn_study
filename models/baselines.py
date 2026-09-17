"""Phase 1 — Reference Architectures for Tiny ImageNet-200."""

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torchvision.models import alexnet, mobilenet_v2
from torchvision.models.quantization import mobilenet_v2 as mobilenet_v2_qat
from torchvision.models.quantization import resnet18 as resnet18_qat
from torchvision.models.vgg import VGG, cfgs as VGG_CFGS


def _fix_relu_inplace(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            _fix_relu_inplace(child)


# ─── AlexNetTV ────────────────────────────────────────────────────────────────

# features[0, 3, 6, 8, 10] -> (kernel, stride, padding), for kernel_size={3,2} below. Chosen so
# every pool's output size matches the original 11x11/5x5/3x3 network's exactly (verified at
# 64x64: pool outputs 7, 3, 1 for both variants, same as the original).
_ALEXNET_SPEC_3 = [(3, 4, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)]
_ALEXNET_SPEC_2 = [(2, 4, 0), (2, 1, 1), (2, 1, 1), (2, 1, 0), (2, 1, 1)]
# Per-layer 3x3/2x2 mixes for the Phase 11 mixed-kernel comparison: each entry picks, per conv
# position, the matching tuple from _ALEXNET_SPEC_3/_ALEXNET_SPEC_2 above (no new padding/stride
# geometry). Only 26/30 non-uniform 3/2 combos survive the fixed MaxPool2d(3, stride=2) layers at
# 64x64 input without collapsing to a non-positive spatial size -- these 3 were verified to.
_ALEXNET_KERNEL_SPECS = {
    3: _ALEXNET_SPEC_3,
    2: _ALEXNET_SPEC_2,
    "mixed_alt": [_ALEXNET_SPEC_2[0], _ALEXNET_SPEC_3[1], _ALEXNET_SPEC_2[2], _ALEXNET_SPEC_3[3], _ALEXNET_SPEC_2[4]],
    "mixed_early3": [_ALEXNET_SPEC_3[0], _ALEXNET_SPEC_3[1], _ALEXNET_SPEC_3[2], _ALEXNET_SPEC_2[3], _ALEXNET_SPEC_2[4]],
    "mixed_early2": [_ALEXNET_SPEC_2[0], _ALEXNET_SPEC_2[1], _ALEXNET_SPEC_2[2], _ALEXNET_SPEC_3[3], _ALEXNET_SPEC_3[4]],
}
_ALEXNET_CONV_INDICES = [0, 3, 6, 8, 10]


class AlexNetTV(nn.Module):
    """Torchvision AlexNet, fine-tuned for 200 classes.

    Architecture: 5 conv stages with large kernels (11×11, 5×5, then 3×3), FC head.
    Expected top-1: ~27-30% (pretrained weights give strong feature initialization).
    Size: ~220 MB FP32 / ~55 MB INT8.
    Training speed: medium (large FC head dominates memory).
    QAT: full — flat Sequential features, easy Conv-BN-ReLU fusion via fuse_map.
    Trade-off: large kernel sizes vs accuracy; classical vs modern architecture.
    kernel_size=3 or 2 replaces all 5 convs with that kernel; kernel_size="mixed_alt"/
    "mixed_early3"/"mixed_early2" replaces them with a per-layer 3x3/2x2 mix (see
    _ALEXNET_KERNEL_SPECS), keeping channels/pool structure -- for the kernel-restriction
    comparison (Phase 11).
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True, kernel_size: int | str | None = None):
        super().__init__()
        base = alexnet(weights="IMAGENET1K_V1" if pretrained else None)
        base.classifier[6] = nn.Linear(4096, num_classes)
        for name, module in base.features.named_children():
            if isinstance(module, nn.ReLU):
                setattr(base.features, name, nn.ReLU(inplace=False))

        if kernel_size is not None:
            for idx, (k, s, p) in zip(_ALEXNET_CONV_INDICES, _ALEXNET_KERNEL_SPECS[kernel_size]):
                old_conv = base.features[idx]
                base.features[idx] = nn.Conv2d(old_conv.in_channels, old_conv.out_channels, k, stride=s, padding=p)

        self.quant = tq.QuantStub()
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


# ─── VGGStyleCNN ──────────────────────────────────────────────────────────────

class VGGStyleCNN(nn.Module):
    """VGG-style deep stacked 3×3 convolutions — small-kernel depth baseline.

    Architecture: 5 stages each with 2× stacked 3×3 conv + BN + ReLU, MaxPool between
    stages, global average pooling head. No pretrained weights.
    Expected top-1: ~35-45% (depth compensates for width reduction vs larger models).
    Size: ~5 MB FP32 / ~1.3 MB INT8.
    Training speed: fast (narrow channels, GAP head).
    QAT: full — flat Sequential, Conv-BN-ReLU fuseable throughout.
    Trade-off: depth via stacked 3×3 vs large single kernels; classic VGG-style design.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=False), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=False), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False), nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


# ─── VGG16 ────────────────────────────────────────────────────────────────────

def _vgg16_stages(cfg: list) -> list[list[int]]:
    """Split torchvision's VGG cfg (e.g. cfgs["D"]) into per-stage channel lists, one list per
    'M' (maxpool) marker, with the markers themselves dropped."""
    stages, stage = [], []
    for v in cfg:
        if v == "M":
            stages.append(stage)
            stage = []
        else:
            stage.append(v)
    return stages


def _vgg16_features(kernel_size: int) -> nn.Sequential:
    """VGG16 features (torchvision cfgs["D"]), from scratch, with BatchNorm -- a 13-layer plain
    conv stack with no normalization does not train (measured on PCAD, jobs 821246/821247:
    loss stuck at ln(200)=5.30 for 22 epochs). kernel_size=3 is the original architecture (all
    convs 3x3/s1/pad1). kernel_size=2 restarts a 1/0 padding alternation at every stage, which
    keeps every stage's pre-pool spatial size -- and so every pooled size -- identical to the
    original (verified at 64x64: pools output 32,16,8,4,2 either way).
    """
    layers: list[nn.Module] = []
    in_ch = 3
    for stage in _vgg16_stages(VGG_CFGS["D"]):
        for i, out_ch in enumerate(stage):
            padding = 1 if kernel_size == 3 or i % 2 == 0 else 0
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False)]
            in_ch = out_ch
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """VGG16 (torchvision cfgs["D"]), from scratch, with BatchNorm -- no pretrained weights.

    Architecture: 13 conv layers in 5 stages, FC head -- the original VGG16, which is already
    all-3x3 by design. kernel_size=2 keeps every pooled spatial size identical (see
    _vgg16_features) so both variants share one classifier head; only the kernel changes.
    QAT: full — flat Sequential features, Conv-BN-ReLU fuseable throughout.
    Trade-off: kernel-size restriction below VGG's own native size (Phase 11).
    """

    def __init__(self, num_classes: int = 200, kernel_size: int = 3):
        super().__init__()
        base = VGG(_vgg16_features(kernel_size), num_classes=num_classes)
        _fix_relu_inplace(base)  # base.classifier's ReLUs default to inplace=True

        self.quant = tq.QuantStub()
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


# ─── ResNet18TV ───────────────────────────────────────────────────────────────

class ResNet18TV(nn.Module):
    """Torchvision ResNet-18 pretrained on ImageNet, fine-tuned for 200 classes.

    Architecture: 4 residual layer groups (2 BasicBlock each), 7×7 stem, GAP head.
    Expected top-1: ~55-65% (pretrained + residual learning significantly helps).
    Size: ~44 MB FP32 / ~11 MB INT8.
    Training speed: fast (lightweight residual blocks).
    QAT: full — uses torchvision's quantizable ResNet18 (QuantizableBasicBlock), whose
    residual add already goes through FloatFunctional.add_relu, so both weights and
    activations are fake-quantized correctly through the skip connection. QuantStub/
    DeQuantStub are already built into QuantizableResNet.forward() (same as
    MobileNetV2TV below) — no external quant/dequant wrapping needed or wanted here.
    Trade-off: residual connections vs no residuals; modern vs classical design.
    Note: 7×7 stem stride=2 + MaxPool reduces 64×64 → 8×8 early, may lose fine detail.
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = resnet18_qat(weights=weights, quantize=False)
        base.fc = nn.Linear(512, num_classes)
        _fix_relu_inplace(base)

        self.base = base

    def forward(self, x):
        return self.base(x)


# ─── MobileNetV2TV ────────────────────────────────────────────────────────────

class MobileNetV2TV(nn.Module):
    """Torchvision MobileNetV2 pretrained on ImageNet, fine-tuned for 200 classes.

    Architecture: inverted residual blocks with depthwise separable convolutions,
    linear bottlenecks, width multiplier 1.0. Uses quantization-aware version with
    proper quantized residual additions (FloatFunctional). Replaces final Linear(1280, 1000)
    with Linear(1280, 200).
    Expected top-1: ~55-65% (pretrained weights; efficient for inference).
    Size: ~14 MB FP32 / ~3.5 MB INT8.
    Training speed: fast (depthwise separable convolutions reduce FLOPs ~8-9×).
    QAT: full — quantization-aware model handles residual adds correctly.
    Trade-off: efficiency via depthwise separable convolutions vs accuracy.
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        base = mobilenet_v2_qat(weights=weights, quantize=False)
        base.classifier[1] = nn.Linear(1280, num_classes)
        _fix_relu_inplace(base)

        self.base = base

    def forward(self, x):
        return self.base(x)


