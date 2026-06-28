"""Stage 1 baseline architectures for the Tiny ImageNet-200 ablation study."""

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torchvision.models import alexnet, resnet18


def _float_functional():
    try:
        return torch.nn.quantized.FloatFunctional()
    except AttributeError:
        return tq.FloatFunctional()


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


# ─── StrongCNN ────────────────────────────────────────────────────────────────

class StrongCNN(nn.Module):
    """High-capacity conventional CNN — upper reference baseline for classical architectures.

    Architecture: 5 stages (64→128→256→512→512 channels), 3×3 conv + BN + ReLU + MaxPool,
    followed by a 2-layer FC head with Dropout. No pretrained weights.
    Expected top-1: ~40-50% (trained from scratch with strong capacity).
    Size: ~15 MB FP32 / ~4 MB INT8.
    Training speed: medium-slow (wide channels + FC head).
    QAT: full — flat Sequential, Conv-BN-ReLU fuseable throughout.
    Trade-off: raw capacity vs efficiency; tests how much width alone can achieve.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4, 1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_strongcnn(num_classes: int = 200) -> nn.Module:
    return StrongCNN(num_classes)


# ─── VGGStyleCNN ──────────────────────────────────────────────────────────────

class VGGStyleCNN(nn.Module):
    """VGG-style deep stacked 3×3 convolutions — small-kernel depth baseline.

    Architecture: 5 stages each with 2× stacked 3×3 conv + BN + ReLU, MaxPool between
    stages, global average pooling head. No pretrained weights.
    Expected top-1: ~35-45% (depth compensates for width reduction vs StrongCNN).
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


# ─── TinyHybridNet (moved from tinyhybridnet.py) ──────────────────────────────

class FireMobileResidual(nn.Module):
    """Fire-inspired mobile residual block (squeeze → depthwise → expand + skip).

    Architecture: 1×1 squeeze → depthwise 3×3 → 1×1 expand, with identity/projected skip.
    Combines SqueezeNet channel compression with MobileNet depthwise efficiency.
    QAT: full — FloatFunctional skip-add, inplace=False ReLU, BN after every Conv.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, squeeze_ratio: float = 0.25):
        super().__init__()
        squeeze_ch = max(16, int(out_ch * squeeze_ratio))

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, squeeze_ch, 1, bias=False),
            nn.BatchNorm2d(squeeze_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(squeeze_ch, squeeze_ch, 3, stride=stride, padding=1, groups=squeeze_ch, bias=False),
            nn.BatchNorm2d(squeeze_ch),
            nn.ReLU(inplace=False),
            # no ReLU here — applied after the residual add
            nn.Conv2d(squeeze_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.skip_add = _float_functional()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.skip_add.add(self.block(x), self.shortcut(x)))


class TinyHybridNet(nn.Module):
    """Lightweight hybrid CNN for 64×64 images (200 classes), QAT-ready.

    Architecture: stem → 6× FireMobileResidual blocks → GAP → Linear.
    Expected top-1: ~32-36%. Size: ~0.73 MB FP32 / ~0.31 MB INT8.
    Training speed: very fast (~0.18M params).
    QAT: full — best quantization efficiency in project (<1% INT8 drop observed).
    Trade-off: extreme compactness vs accuracy; validates that tiny models quantize cleanly.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        )
        self.features = nn.Sequential(
            FireMobileResidual(32, 64),
            FireMobileResidual(64, 64),
            FireMobileResidual(64, 128, stride=2),
            FireMobileResidual(128, 128),
            FireMobileResidual(128, 256, stride=2),
            FireMobileResidual(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x


def build_tinyhybridnet(num_classes: int = 200) -> nn.Module:
    return TinyHybridNet(num_classes)
