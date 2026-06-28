"""Stage 3 efficient CNN architectures: torchvision wrappers and lightweight custom models."""

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0, efficientnet_b0, convnext_tiny


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


# ─── TinyMobileNetV2 (moved from tinyhybridnet.py) ───────────────────────────

class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block (expand → depthwise → project).

    Inverts the standard bottleneck: expansion first (width), then depthwise, then
    pointwise projection. Linear bottleneck (no ReLU on output projection preserves
    information in the low-dimensional manifold).
    QAT: full — FloatFunctional skip-add, inplace=False ReLU, BN after every Conv.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: int = 6):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=False),
            ]
        layers += [
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=False),
            # linear bottleneck — no ReLU on projection
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)
        self.skip_add = _float_functional()

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = self.skip_add.add(out, x)
        return out


class TinyMobileNetV2(nn.Module):
    """MobileNetV2-style tiny CNN for 64×64 images, QAT-ready.

    Architecture: stem → 6× InvertedResidual blocks → GAP → Linear.
    Expected top-1: ~46-52%. Size: ~5.85 MB FP32 / ~1.76 MB INT8.
    Training speed: fast (~1.51M params, depthwise convolutions).
    QAT: full — inverted residuals with FloatFunctional, BN fuse-able via find_fuse_groups.
    Trade-off: MobileNetV2 inverted bottleneck (expand first) vs TinyHybridNet Fire blocks.
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
            InvertedResidual(32, 64, stride=1, expand_ratio=1),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 128, stride=2, expand_ratio=6),
            InvertedResidual(128, 128, stride=1, expand_ratio=6),
            InvertedResidual(128, 256, stride=2, expand_ratio=6),
            InvertedResidual(256, 256, stride=1, expand_ratio=6),
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


def build_tinymobilenetv2(num_classes: int = 200) -> nn.Module:
    return TinyMobileNetV2(num_classes)


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


# ─── ShuffleNetV2TV ───────────────────────────────────────────────────────────

class ShuffleNetV2TV(nn.Module):
    """Torchvision ShuffleNetV2 ×1.0 pretrained on ImageNet, fine-tuned for 200 classes.

    Architecture: hardware-aware efficient CNN using channel split and shuffle instead of
    group convolutions. 1024 output channels. Replaces final fc with Linear(1024, 200).
    Expected top-1: ~55-62% (pretrained; channel shuffle enables cross-group information mixing).
    Size: ~9 MB FP32 / ~2.3 MB INT8.
    Training speed: very fast (channel split avoids grouped conv overhead).
    QAT: partial — QuantStub/DeQuantStub added, inplace ReLU fixed. Channel shuffle and
    residual concatenation need special handling for full INT8.
    Trade-off: channel shuffle (hardware-friendly) vs group convolutions.
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = shufflenet_v2_x1_0(weights=weights)
        base.fc = nn.Linear(1024, num_classes)
        _fix_relu_inplace(base)

        self.quant = tq.QuantStub()
        self.base = base
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.base(x)
        x = self.dequant(x)
        return x


def build_shufflenetv2(num_classes: int = 200) -> nn.Module:
    return ShuffleNetV2TV(num_classes)


# ─── EfficientNetB0TV ─────────────────────────────────────────────────────────

class EfficientNetB0TV(nn.Module):
    """Torchvision EfficientNet-B0 pretrained on ImageNet, fine-tuned for 200 classes.

    Architecture: NAS-designed with compound scaling (depth × width × resolution), MBConv
    blocks with SE attention. Replaces final Linear(1280, 1000) with Linear(1280, 200).
    Expected top-1: ~70-78% (strongest pretrained baseline; SE blocks + compound scaling).
    Size: ~21 MB FP32 / ~5 MB INT8.
    Training speed: medium (MBConv + SE adds overhead vs plain MobileNet).
    QAT: partial — QuantStub/DeQuantStub added, inplace ReLU fixed. SE Sigmoid and
    SiLU activations are not fbgemm-compatible; full INT8 needs custom quantization config.
    Trade-off: accuracy ceiling via compound scaling vs simplicity of design.
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = efficientnet_b0(weights=weights)
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


def build_efficientnetb0(num_classes: int = 200) -> nn.Module:
    return EfficientNetB0TV(num_classes)


# ─── ConvNeXtTinyTV ───────────────────────────────────────────────────────────

class ConvNeXtTinyTV(nn.Module):
    """Torchvision ConvNeXt-Tiny pretrained on ImageNet, fine-tuned for 200 classes.

    Architecture: transformer-era CNN design — depthwise 7×7 conv, inverted bottleneck
    (4× expand), GELU, LayerNorm, no BN. Replaces final Linear(768, 1000) with
    Linear(768, 200). No inplace ReLU to fix (uses GELU throughout).
    Expected top-1: ~78-85% (best in class; modern design with transformer-inspired blocks).
    Size: ~112 MB FP32 / ~28 MB INT8.
    Training speed: medium (GELU + LayerNorm + large depthwise kernels).
    QAT: partial — QuantStub/DeQuantStub added. GELU and LayerNorm are not fbgemm-
    compatible; full INT8 requires approximating GELU and replacing LayerNorm with BN.
    Trade-off: transformer-era design principles in a pure CNN vs classical CNN ideas.
    """

    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = convnext_tiny(weights=weights)
        base.classifier[2] = nn.Linear(768, num_classes)
        # ConvNeXt uses GELU, not ReLU — no inplace activation fix needed

        self.quant = tq.QuantStub()
        self.base = base
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.base(x)
        x = self.dequant(x)
        return x


def build_convnexttiny(num_classes: int = 200) -> nn.Module:
    return ConvNeXtTinyTV(num_classes)
