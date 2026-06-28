"""Stage 2: AlexNet-derived variants isolating single architectural choices."""

import torch
import torch.nn as nn
import torch.ao.quantization as tq


def _float_functional():
    try:
        return torch.nn.quantized.FloatFunctional()
    except AttributeError:
        return tq.FloatFunctional()


# ─── AlexNet3x3 (moved from alexnet3x3.py) ───────────────────────────────────

class AlexNet3x3(nn.Module):
    """All-3x3 AlexNet — tests impact of enforcing small kernels across all layers.

    Architecture: 5 conv stages (all 3x3, same channels as AlexNet), AdaptiveAvgPool(6x6),
    3-layer FC head. Identical structure to AlexNetTV but with 3x3 kernels everywhere.
    Expected top-1: ~8-12% (from scratch; 3x3 alone insufficient without residuals).
    Size: ~220 MB FP32 / ~55 MB INT8 (large FC head dominates).
    Training speed: slow (same large FC head as original AlexNet).
    QAT: full — flat Sequential, hand-written fuse_map for Conv-ReLU pairs.
    Trade-off: kernel size is the single variable; proves large kernels matter for AlexNet.
    Note: no BatchNorm (matches original AlexNet design for clean kernel-size comparison).
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),    # 0
            nn.ReLU(inplace=False),                        # 1
            nn.MaxPool2d(2),                               # 2
            nn.Conv2d(64, 192, 3, padding=1),              # 3
            nn.ReLU(inplace=False),                        # 4
            nn.MaxPool2d(2),                               # 5
            nn.Conv2d(192, 384, 3, padding=1),             # 6
            nn.ReLU(inplace=False),                        # 7
            nn.Conv2d(384, 256, 3, padding=1),             # 8
            nn.ReLU(inplace=False),                        # 9
            nn.Conv2d(256, 256, 3, padding=1),             # 10
            nn.ReLU(inplace=False),                        # 11
            nn.AdaptiveAvgPool2d((6, 6)),                  # 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


# ─── AlexNetSmallKernel (moved from alexnet_smallkernel.py) ──────────────────

class AlexNetSmallKernel(nn.Module):
    """All-3×3 CNN with global-pool head — kernel-restricted, lightweight variant.

    Architecture: 5 conv stages (3×3, narrower channels), AdaptiveAvgPool(1×1),
    single Linear classifier (~250× smaller than AlexNet3x3's FC head).
    Expected top-1: ~8-10%. Size: ~6 MB FP32 / ~1.6 MB INT8.
    Training speed: fast (GAP head eliminates most parameters).
    QAT: full — flat Sequential, Conv-ReLU pairs fuseable.
    Trade-off: GAP vs large FC head — measures how much the head contributes to accuracy.
    Note: no BatchNorm (matches AlexNet3x3 for controlled comparison).
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),     # 0
            nn.ReLU(inplace=False),                         # 1
            nn.MaxPool2d(2),                                # 2
            nn.Conv2d(64, 128, 3, padding=1),               # 3
            nn.ReLU(inplace=False),                         # 4
            nn.MaxPool2d(2),                                # 5
            nn.Conv2d(128, 256, 3, padding=1),              # 6
            nn.ReLU(inplace=False),                         # 7
            nn.Conv2d(256, 256, 3, padding=1),              # 8
            nn.ReLU(inplace=False),                         # 9
            nn.Conv2d(256, 256, 3, padding=1),              # 10
            nn.ReLU(inplace=False),                         # 11
            nn.AdaptiveAvgPool2d((1, 1)),                   # 12
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


# ─── AlexNetStacked ───────────────────────────────────────────────────────────

class AlexNetStacked(nn.Module):
    """2× stacked 3×3 per stage — tests whether depth recovers receptive field.

    Architecture: 5 stages with 2 consecutive 3×3 conv + BN + ReLU each (10 conv total),
    same channel widths as AlexNet3x3, same 3-layer FC head. Two stacked 3×3 convs have
    the same receptive field as one 5×5 conv, approximating VGG-style depth.
    Expected top-1: ~12-18% (depth helps but FC head still limits generalization).
    Size: ~220 MB FP32 / ~55 MB INT8.
    Training speed: slow (double convolutions + large FC head).
    QAT: full — Conv-BN-ReLU triples fuseable throughout.
    Trade-off: depth (stacking) vs width; compares receptive field recovery strategies.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            # Stage 1 — 2× 3×3 stacked
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Stage 2
            nn.Conv2d(64, 192, 3, padding=1, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=False),
            nn.Conv2d(192, 192, 3, padding=1, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Stage 3
            nn.Conv2d(192, 384, 3, padding=1, bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=False),
            nn.Conv2d(384, 384, 3, padding=1, bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=False),
            # Stage 4
            nn.Conv2d(384, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            # Stage 5
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 36, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_alexnet_stacked(num_classes: int = 200) -> nn.Module:
    return AlexNetStacked(num_classes)


# ─── AlexNetFactorized ────────────────────────────────────────────────────────

class AlexNetFactorized(nn.Module):
    """1×3 + 3×1 asymmetric convolution pairs — tests spatial factorization (Inception-style).

    Architecture: each stage replaces a single 3×3 conv with a 1×3 followed by a 3×1 conv.
    Same channel progression as AlexNet3x3. Factorized convolutions have the same receptive
    field as 3×3 but ~33% fewer parameters and FLOPs per stage.
    Expected top-1: ~10-15% (factorization adds asymmetry; may hurt for small images).
    Size: ~200 MB FP32 / ~50 MB INT8.
    Training speed: similar to AlexNet3x3 (slightly fewer FLOPs per stage).
    QAT: full — Conv-BN-ReLU fuseable, asymmetric kernels supported by fbgemm.
    Trade-off: parameter efficiency via factorization vs representational completeness.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            # Stage 1: 1×3 → 3×1 pair (two MaxPools to compensate for no stride=2 conv)
            nn.Conv2d(3, 64, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 64→32
            nn.MaxPool2d(2),  # 32→16
            # Stage 2
            nn.Conv2d(64, 192, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=False),
            nn.Conv2d(192, 192, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 16→8
            # Stage 3
            nn.Conv2d(192, 384, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=False),
            nn.Conv2d(384, 384, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=False),
            # Stage 4
            nn.Conv2d(384, 256, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            # Stage 5
            nn.Conv2d(256, 256, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 36, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_alexnet_factorized(num_classes: int = 200) -> nn.Module:
    return AlexNetFactorized(num_classes)


# ─── AlexNetBottleneck ────────────────────────────────────────────────────────

class _AlexBottleneck(nn.Module):
    """1×1 squeeze → 3×3 → 1×1 expand bottleneck block, no residual."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 4):
        super().__init__()
        mid_ch = max(out_ch // reduction, 32)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)


class AlexNetBottleneck(nn.Module):
    """Bottleneck blocks + global average pooling — tests parameter reduction with 3×3 core.

    Architecture: 5 stages of 1×1→3×3→1×1 bottleneck blocks (reduction=4), GAP head.
    Bottleneck squeezes channels to 1/4 before the 3×3, reducing FLOPs and params.
    Expected top-1: ~12-18% (bottleneck + GAP cuts params dramatically; may underfit).
    Size: ~50 MB FP32 / ~13 MB INT8.
    Training speed: medium (bottleneck reduces per-stage cost; GAP removes large FC).
    QAT: full — Conv-BN-ReLU triples fuseable throughout nested blocks.
    Trade-off: parameter efficiency via bottleneck + GAP vs raw FC capacity.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            _AlexBottleneck(3, 64, stride=2),
            nn.MaxPool2d(2),
            _AlexBottleneck(64, 192),
            nn.MaxPool2d(2),
            _AlexBottleneck(192, 384),
            _AlexBottleneck(384, 256),
            _AlexBottleneck(256, 256),
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


def build_alexnet_bottleneck(num_classes: int = 200) -> nn.Module:
    return AlexNetBottleneck(num_classes)


# ─── AlexNetResidual ──────────────────────────────────────────────────────────

class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Sigmoid-based, not QAT-friendly).

    Note: SE uses Sigmoid which is unsupported by fbgemm. Disable for QAT (use_se=False).
    """

    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(inplace=False),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.pool(x).flatten(1)
        return x * self.fc(s).view(x.size(0), -1, 1, 1)


class _ResBlock(nn.Module):
    """Two 3×3 convolutions with BatchNorm and a residual skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.se = _SEBlock(out_ch) if use_se else None
        self.skip_add = _float_functional()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.block(x)
        if self.se is not None:
            out = self.se(out)
        return self.relu(self.skip_add.add(out, self.shortcut(x)))


class AlexNetResidual(nn.Module):
    """AlexNet3x3 stages + residual skip connections + optional SE attention.

    Architecture: 5 residual blocks (pair of 3×3 conv + BN + ReLU + FloatFunctional add),
    same channel widths as AlexNet3x3, AdaptiveAvgPool(6×6), 3-layer FC head.
    Optional SE blocks add channel attention after each residual pair.
    Expected top-1: ~20-30% (residuals are the single most impactful upgrade for small CNNs).
    Size: ~220 MB FP32 / ~55 MB INT8 (same FC head; backbone params similar).
    Training speed: slow (large FC head; SE adds overhead when enabled).
    QAT: full without SE (use_se=False, default). SE uses Sigmoid — disable for INT8.
    Trade-off: residual optimization stability vs plain feedforward; most research-relevant.
    """

    def __init__(self, num_classes: int = 200, use_se: bool = False):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            _ResBlock(3, 64, stride=2, use_se=use_se),
            nn.MaxPool2d(2),
            _ResBlock(64, 192, use_se=use_se),
            nn.MaxPool2d(2),
            _ResBlock(192, 384, use_se=use_se),
            _ResBlock(384, 256, use_se=use_se),
            _ResBlock(256, 256, use_se=use_se),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 36, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_alexnet_residual(num_classes: int = 200) -> nn.Module:
    return AlexNetResidual(num_classes)
