"""Phase 1 — Reference Architectures for Tiny ImageNet-200."""

import torch.nn as nn
import torch.ao.quantization as tq
from torchvision.models import alexnet, resnet18, mobilenet_v2


def _fix_relu_inplace(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            _fix_relu_inplace(child)


# ─── AlexNetTV ────────────────────────────────────────────────────────────────

class AlexNetTV(nn.Module):
    """Torchvision AlexNet pretrained on ImageNet, fine-tuned for 200 classes.

    Architecture: 5 conv stages with large kernels (11×11, 5×5, then 3×3), FC head.
    Expected top-1: ~27-30% (pretrained weights give strong feature initialization).
    Size: ~220 MB FP32 / ~55 MB INT8.
    Training speed: medium (large FC head dominates memory).
    QAT: full — flat Sequential features, easy Conv-BN-ReLU fusion via fuse_map.
    Trade-off: large kernel sizes vs accuracy; classical vs modern architecture.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        base = alexnet(weights="IMAGENET1K_V1")
        base.classifier[6] = nn.Linear(4096, num_classes)
        for name, module in base.features.named_children():
            if isinstance(module, nn.ReLU):
                setattr(base.features, name, nn.ReLU(inplace=False))

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


def build_alexnet(num_classes: int = 200) -> nn.Module:
    return AlexNetTV(num_classes)


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


def build_vggstylecnn(num_classes: int = 200) -> nn.Module:
    return VGGStyleCNN(num_classes)


# ─── ResNet18TV ───────────────────────────────────────────────────────────────

class ResNet18TV(nn.Module):
    """Torchvision ResNet-18 pretrained on ImageNet, fine-tuned for 200 classes.

    Architecture: 4 residual layer groups (2 BasicBlock each), 7×7 stem, GAP head.
    Expected top-1: ~55-65% (pretrained + residual learning significantly helps).
    Size: ~44 MB FP32 / ~11 MB INT8.
    Training speed: fast (lightweight residual blocks).
    QAT: partial — QuantStub/DeQuantStub added, inplace ReLU fixed. For full INT8
    with correct residual quantization, use torchvision.models.quantization.resnet18().
    Trade-off: residual connections vs no residuals; modern vs classical design.
    Note: 7×7 stem stride=2 + MaxPool reduces 64×64 → 8×8 early, may lose fine detail.
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = resnet18(weights=weights)
        base.fc = nn.Linear(512, num_classes)
        _fix_relu_inplace(base)

        self.quant = tq.QuantStub()
        self.base = base
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.base(x)
        x = self.dequant(x)
        return x


def build_resnet18(num_classes: int = 200) -> nn.Module:
    return ResNet18TV(num_classes)


# ─── MobileNetV2TV ────────────────────────────────────────────────────────────

class MobileNetV2TV(nn.Module):
    """Torchvision MobileNetV2 pretrained on ImageNet, fine-tuned for 200 classes.

    Architecture: inverted residual blocks with depthwise separable convolutions,
    linear bottlenecks, width multiplier 1.0. Replaces final Linear(1280, 1000) with
    Linear(1280, 200).
    Expected top-1: ~55-65% (pretrained weights; efficient for inference).
    Size: ~14 MB FP32 / ~3.5 MB INT8.
    Training speed: fast (depthwise separable convolutions reduce FLOPs ~8-9×).
    QAT: partial — QuantStub/DeQuantStub + inplace ReLU fixed. Full INT8 correctness
    for residual adds requires torchvision.models.quantization.mobilenet_v2().
    Trade-off: efficiency via depthwise separable convolutions vs accuracy.
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        base = mobilenet_v2(weights=weights)
        base.classifier[1] = nn.Linear(1280, num_classes)
        _fix_relu_inplace(base)

        self.quant = tq.QuantStub()
        self.base = base
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.base(x)
        x = self.dequant(x)
        return x


def build_mobilenetv2(num_classes: int = 200) -> nn.Module:
    return MobileNetV2TV(num_classes)
