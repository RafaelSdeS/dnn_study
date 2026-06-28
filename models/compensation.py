"""Phase 3 — Compensation Mechanisms: architectural modifications to kernel-restricted AlexNet.

Each model takes AlexNet3x3 as its base and adds exactly one compensation mechanism,
isolating its effect on accuracy, efficiency, and quantization.
"""

import torch
import torch.nn as nn
import torch.ao.quantization as tq


def _float_functional():
    try:
        return torch.nn.quantized.FloatFunctional()
    except AttributeError:
        return tq.FloatFunctional()


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
    """Bottleneck blocks — 1×1→3×3→1×1 per stage, reduces params with same 3×3 core.

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


# ─── AlexNetFactorized ────────────────────────────────────────────────────────

class AlexNetFactorized(nn.Module):
    """1×3 + 3×1 asymmetric pairs — Inception-style spatial factorization.

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


# ─── AlexNetGroupConv ─────────────────────────────────────────────────────────

class AlexNetGroupConv(nn.Module):
    """Grouped convolutions (groups=4) — reduces cross-channel mixing cost by 4×.

    Architecture: 5 stages of 3×3 grouped conv (groups=4 for stages 2-5, groups=1 for
    stage 1 since 3 channels are not divisible by 4), BatchNorm, same channel widths as
    AlexNet3x3 (all divisible by 4), AdaptiveAvgPool(6×6), 3-layer FC head.
    Grouped convolutions partition channels into 4 independent groups, reducing parameter
    count and FLOPs by ~4× at the cost of cross-group information mixing.
    Expected top-1: ~10-18% (grouped conv saves params; cross-group isolation hurts).
    Size: ~60 MB FP32 / ~15 MB INT8.
    Training speed: medium (grouped convolutions reduce FLOPs but FC head still expensive).
    QAT: full — flat Sequential, Conv-BN-ReLU fuseable; fbgemm supports grouped convolutions.
    Trade-off: parameter efficiency via grouping vs full cross-channel information mixing.
    Note: channel counts 64, 192, 384, 256 are all divisible by groups=4.
    """

    def __init__(self, num_classes: int = 200, groups: int = 4):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            # Stage 1: groups=1 (3 input channels not divisible by 4)
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Stage 2
            nn.Conv2d(64, 192, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Stage 3
            nn.Conv2d(192, 384, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=False),
            # Stage 4
            nn.Conv2d(384, 256, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            # Stage 5
            nn.Conv2d(256, 256, 3, padding=1, groups=groups, bias=False),
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


def build_alexnet_groupconv(num_classes: int = 200) -> nn.Module:
    return AlexNetGroupConv(num_classes)


# ─── AlexNetDepthwiseSep ──────────────────────────────────────────────────────

class AlexNetDepthwiseSep(nn.Module):
    """Depthwise separable convolutions — MobileNet-style spatial factorization.

    Architecture: 5 stages of DW 3×3 + PW 1×1 (replacing standard 3×3), BatchNorm after
    each, same channel widths as AlexNet3x3, AdaptiveAvgPool(1×1), single Linear head.
    DW separable: ~8-9× fewer FLOPs and params than standard 3×3 per stage. GAP head avoids
    the large FC since parameter savings from DW already justify this change.
    Expected top-1: ~10-15% (DW reduces cross-channel mixing; GAP head is lightweight).
    Size: ~8 MB FP32 / ~2 MB INT8.
    Training speed: fast (DW separable dramatically reduces computation).
    QAT: full — flat Sequential, DW+PW Conv-BN-ReLU pairs fuseable; fbgemm supports depthwise.
    Trade-off: efficiency via DW factorization vs accuracy from full cross-channel conv.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            # Stage 1: DW on 3 channels (groups=3), then PW to 64
            nn.Conv2d(3, 3, 3, stride=2, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(3), nn.ReLU(inplace=False),
            nn.Conv2d(3, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Stage 2
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Conv2d(64, 192, 1, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Stage 3
            nn.Conv2d(192, 192, 3, padding=1, groups=192, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=False),
            nn.Conv2d(192, 384, 1, bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=False),
            # Stage 4
            nn.Conv2d(384, 384, 3, padding=1, groups=384, bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            # Stage 5
            nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 1, bias=False),
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


def build_alexnet_depthwisesep(num_classes: int = 200) -> nn.Module:
    return AlexNetDepthwiseSep(num_classes)


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
    """Residual skip connections — most impactful single upgrade for small 3×3 CNNs.

    Architecture: 5 residual blocks (pair of 3×3 conv + BN + ReLU + FloatFunctional add),
    same channel widths as AlexNet3x3, AdaptiveAvgPool(6×6), 3-layer FC head.
    Optional SE blocks add channel attention after each residual pair (use_se=True);
    disable for QAT since Sigmoid is not fbgemm-compatible.
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


# ─── AlexNetFire ──────────────────────────────────────────────────────────────

class _FireModule(nn.Module):
    """SqueezeNet Fire module: 1×1 squeeze → parallel 1×1 + 3×3 expand, concatenated."""

    def __init__(self, in_ch: int, squeeze_ch: int, expand_ch: int):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_ch, squeeze_ch, 1, bias=False),
            nn.BatchNorm2d(squeeze_ch), nn.ReLU(inplace=False),
        )
        self.expand1 = nn.Sequential(
            nn.Conv2d(squeeze_ch, expand_ch, 1, bias=False),
            nn.BatchNorm2d(expand_ch), nn.ReLU(inplace=False),
        )
        self.expand3 = nn.Sequential(
            nn.Conv2d(squeeze_ch, expand_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(expand_ch), nn.ReLU(inplace=False),
        )

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1(s), self.expand3(s)], dim=1)


class AlexNetFire(nn.Module):
    """Fire modules (SqueezeNet-style) — aggressive channel compression with multi-scale expand.

    Architecture: 5 Fire stages (1×1 squeeze → parallel 1×1 + 3×3 expand, concatenated),
    matching AlexNet3x3's channel progression (out: 64→192→384→256→256), GAP head.
    Fire modules compress the squeeze path to 1/4 of output channels, then expand with two
    parallel paths — pure channel mixing (1×1) and spatial mixing (3×3) — concatenated.
    Expected top-1: ~10-16% (compression reduces overfitting; parallel expand adds expressivity).
    Size: ~5 MB FP32 / ~1.5 MB INT8.
    Training speed: fast (aggressive squeeze reduces FLOPs; GAP head).
    QAT: full — torch.cat is QAT-compatible; Conv-BN-ReLU fuseable in each branch.
    Trade-off: SqueezeNet-style compression + multi-scale features vs plain 3×3.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        # Each FireModule outputs squeeze_ch → expand_ch × 2 channels
        # Target: 3→64, 64→192, 192→384, 384→256, 256→256
        self.features = nn.Sequential(
            _FireModule(3,   16,  32),   # out: 64
            nn.MaxPool2d(2),
            _FireModule(64,  48,  96),   # out: 192
            nn.MaxPool2d(2),
            _FireModule(192, 96, 192),   # out: 384
            _FireModule(384, 64, 128),   # out: 256
            _FireModule(256, 64, 128),   # out: 256
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


def build_alexnet_fire(num_classes: int = 200) -> nn.Module:
    return AlexNetFire(num_classes)


# ─── AlexNetGAP ───────────────────────────────────────────────────────────────

class AlexNetGAP(nn.Module):
    """AlexNet3x3 with Global Average Pooling head — tests head type as the variable.

    Architecture: identical backbone to AlexNet3x3 (5× 3×3 conv, same channels, no BN),
    AdaptiveAvgPool(1×1) + single Linear, replacing the 94M-parameter FC head.
    GAP enforces spatial invariance and eliminates the large FC head entirely, reducing
    overfitting risk while the backbone is held constant.
    Expected top-1: ~8-12% (GAP removes FC capacity; may underfit vs AlexNet3x3 on 200 classes).
    Size: ~3 MB FP32 / ~1 MB INT8 (vs ~220 MB for AlexNet3x3 with large FC).
    Training speed: fast (same backbone cost as AlexNet3x3; GAP removes most parameters).
    QAT: full — flat Sequential, Conv-ReLU pairs fuseable (identical fuse_map to AlexNet3x3).
    Trade-off: GAP spatial invariance + regularization vs FC arbitrary mapping capacity.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=False),
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


def build_alexnet_gap(num_classes: int = 200) -> nn.Module:
    return AlexNetGAP(num_classes)


# ─── AlexNetSE ────────────────────────────────────────────────────────────────

class AlexNetSE(nn.Module):
    """Squeeze-and-Excitation blocks — global channel recalibration per stage.

    Architecture: identical backbone to AlexNet3x3 (5× 3×3 conv, same channels, no BN),
    with a _SEBlock (global avg pool → FC squeeze → ReLU → FC excitation → Sigmoid scale)
    applied after each conv stage. Same AdaptiveAvgPool(6×6) + 3-layer FC head as AlexNet3x3.
    SE recalibrates channel responses by learning per-channel weights from global context,
    compensating for the local-only receptive field of 3×3 kernels.
    Expected top-1: ~15-22% (SE provides global context; modest but consistent improvement).
    Size: ~220 MB FP32 / N/A INT8 (Sigmoid not fbgemm-supported — QAT partial).
    Training speed: slow (5 SE blocks add overhead; large FC head).
    QAT: partial — Sigmoid in SE is not fbgemm-compatible. Disable SE or replace with
    hard-sigmoid approximation (ReLU6(x+3)/6) for full INT8.
    Trade-off: global context via channel attention vs local-only 3×3 receptive field.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(inplace=False))
        self.se1 = _SEBlock(64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(inplace=False))
        self.se2 = _SEBlock(192)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=False))
        self.se3 = _SEBlock(384)

        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=False))
        self.se4 = _SEBlock(256)

        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=False))
        self.se5 = _SEBlock(256)

        self.pool_final = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 36, 4096), nn.ReLU(inplace=False),
            nn.Linear(4096, 4096), nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.se1(self.conv1(x))
        x = self.pool1(x)
        x = self.se2(self.conv2(x))
        x = self.pool2(x)
        x = self.se3(self.conv3(x))
        x = self.se4(self.conv4(x))
        x = self.se5(self.conv5(x))
        x = self.pool_final(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_alexnet_se(num_classes: int = 200) -> nn.Module:
    return AlexNetSE(num_classes)
