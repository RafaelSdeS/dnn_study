"""Hybrid CNN proposals H1–H10: combining architectural principles from classic and modern CNNs."""

import torch
import torch.nn as nn
import torch.ao.quantization as tq


def _float_functional():
    try:
        return torch.nn.quantized.FloatFunctional()
    except AttributeError:
        return tq.FloatFunctional()


# ─── H1: ResidualVGGHybrid ────────────────────────────────────────────────────

class _VGGResBlock(nn.Module):
    """Two stacked 3×3 convolutions with a residual skip connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.skip_add = _float_functional()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.skip_add.add(self.block(x), self.shortcut(x)))


class ResidualVGGHybrid(nn.Module):
    """H1: VGG-style stacked 3×3 convolutions + residual skip connections + GAP head.

    Architecture: 5 stages of _VGGResBlock (2× 3×3 conv + skip), MaxPool between stages,
    global average pooling head. Merges VGG depth philosophy with ResNet optimization stability.
    Expected top-1: ~35-45% (residuals + depth compensate for small kernels substantially).
    Size: ~5 MB FP32 / ~1.3 MB INT8 (GAP eliminates FC overhead).
    Training speed: fast (narrow channels + GAP head + efficient residual blocks).
    QAT: full — FloatFunctional skip-adds, Conv-BN-ReLU fuseable, inplace=False throughout.
    Trade-off: does adding residuals to a VGG-style backbone close the gap to modern models?
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            _VGGResBlock(3, 64), nn.MaxPool2d(2),
            _VGGResBlock(64, 128), nn.MaxPool2d(2),
            _VGGResBlock(128, 256), nn.MaxPool2d(2),
            _VGGResBlock(256, 512), nn.MaxPool2d(2),
            _VGGResBlock(512, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_residual_vgg(num_classes: int = 200) -> nn.Module:
    return ResidualVGGHybrid(num_classes)


# ─── H2: AlexNetInceptionHybrid ──────────────────────────────────────────────

class _InceptionBlock(nn.Module):
    """3-branch Inception block: 1×1 / 3×3 / stacked-3×3, outputs concatenated."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        per_branch = out_ch // 3
        branch3_ch = out_ch - 2 * per_branch  # remainder to 3×3 branch

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_ch, per_branch, 1, bias=False),
            nn.BatchNorm2d(per_branch), nn.ReLU(inplace=False),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_ch, branch3_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch3_ch), nn.ReLU(inplace=False),
        )
        self.branch3x3_stack = nn.Sequential(
            nn.Conv2d(in_ch, per_branch, 3, padding=1, bias=False),
            nn.BatchNorm2d(per_branch), nn.ReLU(inplace=False),
            nn.Conv2d(per_branch, per_branch, 3, padding=1, bias=False),
            nn.BatchNorm2d(per_branch), nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return torch.cat([self.branch1x1(x), self.branch3x3(x), self.branch3x3_stack(x)], dim=1)


class AlexNetInceptionHybrid(nn.Module):
    """H2: AlexNet backbone with Inception-style multi-branch blocks.

    Architecture: stride-2 stem → 4× _InceptionBlock stages (1×1 / 3×3 / stacked-3×3
    branches concatenated) → GAP → Linear. Multi-scale feature extraction replaces
    the single-kernel-size convolution in each AlexNet stage.
    Expected top-1: ~25-35% (multi-scale features help; lighter than AlexNet3x3).
    Size: ~8 MB FP32 / ~2 MB INT8.
    Training speed: medium (parallel branches add computation but reduce per-branch channels).
    QAT: full — Conv-BN-ReLU fuseable in each branch, torch.cat is QAT-compatible.
    Trade-off: multi-scale features vs fixed kernel size; Inception vs AlexNet philosophy.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.features = nn.Sequential(
            _InceptionBlock(64, 192),
            nn.MaxPool2d(2),
            _InceptionBlock(192, 384),
            _InceptionBlock(384, 384),
            _InceptionBlock(384, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_alexnet_inception(num_classes: int = 200) -> nn.Module:
    return AlexNetInceptionHybrid(num_classes)


# ─── H3: MobileNetResidualHybrid ─────────────────────────────────────────────

class _DWResBlock(nn.Module):
    """Depthwise-separable convolution with a residual skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
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


class MobileNetResidualHybrid(nn.Module):
    """H3: Depthwise-separable convolutions + residual connections every 2 blocks.

    Architecture: stem → 8× _DWResBlock (DW 3×3 + PW 1×1 + skip) → GAP → Linear.
    Combines MobileNet's depthwise separable efficiency with ResNet's gradient flow.
    Expected top-1: ~45-55% (residuals significantly improve depthwise-only architectures).
    Size: ~3-5 MB FP32 / ~1 MB INT8.
    Training speed: very fast (depthwise convolutions ~8-9× cheaper than standard).
    QAT: full — FloatFunctional skip-adds, Conv-BN-ReLU fuseable, inplace=False throughout.
    Trade-off: does residual learning improve optimization in lightweight depthwise CNNs?
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=False),
        )
        self.features = nn.Sequential(
            _DWResBlock(32, 64),
            _DWResBlock(64, 64),
            _DWResBlock(64, 128, stride=2),
            _DWResBlock(128, 128),
            _DWResBlock(128, 256, stride=2),
            _DWResBlock(256, 256),
            _DWResBlock(256, 512, stride=2),
            _DWResBlock(512, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_mobilenet_residual(num_classes: int = 200) -> nn.Module:
    return MobileNetResidualHybrid(num_classes)


# ─── H4: FireResidualHybrid ───────────────────────────────────────────────────

class _FireResidual(nn.Module):
    """SqueezeNet Fire module (squeeze + expand) with a residual skip connection."""

    def __init__(self, in_ch: int, squeeze_ch: int, expand1_ch: int, expand3_ch: int):
        super().__init__()
        out_ch = expand1_ch + expand3_ch
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_ch, squeeze_ch, 1, bias=False),
            nn.BatchNorm2d(squeeze_ch), nn.ReLU(inplace=False),
        )
        self.expand1 = nn.Sequential(
            nn.Conv2d(squeeze_ch, expand1_ch, 1, bias=False),
            nn.BatchNorm2d(expand1_ch), nn.ReLU(inplace=False),
        )
        self.expand3 = nn.Sequential(
            nn.Conv2d(squeeze_ch, expand3_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(expand3_ch), nn.ReLU(inplace=False),
        )
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.skip_add = _float_functional()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        s = self.squeeze(x)
        out = torch.cat([self.expand1(s), self.expand3(s)], dim=1)
        return self.relu(self.skip_add.add(out, self.shortcut(x)))


class FireResidualHybrid(nn.Module):
    """H4: SqueezeNet Fire modules + residual skip connections.

    Architecture: stem → 6× _FireResidual blocks (1×1 squeeze, parallel 1×1+3×3 expand,
    skip add) → GAP → Linear. Tests whether skip connections improve the highly compressed
    Fire bottleneck without adding significant parameters.
    Expected top-1: ~30-40% (Fire compression works well with residuals for gradient flow).
    Size: ~2-3 MB FP32 / ~0.6 MB INT8.
    Training speed: fast (aggressive channel compression via squeeze).
    QAT: full — FloatFunctional skip-adds, torch.cat is QAT-compatible, Conv-BN-ReLU fuseable.
    Trade-off: Fire module compression efficiency + residual stability vs parameter count.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.features = nn.Sequential(
            _FireResidual(64, 16, 64, 64),     # 64 → 128
            _FireResidual(128, 16, 64, 64),    # 128 → 128
            nn.MaxPool2d(2),
            _FireResidual(128, 32, 128, 128),  # 128 → 256
            _FireResidual(256, 32, 128, 128),  # 256 → 256
            nn.MaxPool2d(2),
            _FireResidual(256, 48, 192, 192),  # 256 → 384
            _FireResidual(384, 48, 192, 192),  # 384 → 384
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(384, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_fire_residual(num_classes: int = 200) -> nn.Module:
    return FireResidualHybrid(num_classes)


# ─── H5: FactorizedAlexNet ────────────────────────────────────────────────────

class _FactBlock(nn.Module):
    """1×k + k×1 factorized conv pair, with optional 1×1 bottleneck and residual."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3,
                 use_bottleneck: bool = True, use_residual: bool = True):
        super().__init__()
        mid_ch = max(out_ch // 2, 16) if use_bottleneck else in_ch

        layers = []
        if use_bottleneck and in_ch != mid_ch:
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                       nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False)]
        layers += [
            nn.Conv2d(mid_ch, mid_ch, (1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False),
            nn.Conv2d(mid_ch, out_ch, (k, 1), padding=(k // 2, 0), bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
        ]
        self.block = nn.Sequential(*layers)

        self.use_residual = use_residual
        if use_residual:
            self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
            )
            self.skip_add = _float_functional()

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = self.skip_add.add(out, self.shortcut(x))
        return out


class FactorizedAlexNet(nn.Module):
    """H5: Systematic 1×k + k×1 factorization + 1×1 bottleneck + optional residuals.

    Architecture: 5 _FactBlock stages (optional 1×1 bottleneck → 1×3 → 3×1 factorized pair,
    optional residual skip), GAP head. Parametric: use_bottleneck and use_residual flags
    control which components are active. More systematic than AlexNetFactorized.
    Expected top-1: ~15-22% (bottleneck + factorization + residuals approach AlexNetResidual).
    Size: ~30-100 MB FP32 depending on flags.
    Training speed: medium.
    QAT: full when use_residual=True (FloatFunctional), asymmetric kernels fbgemm-supported.
    Trade-off: pure impact of kernel factorization as a parameter-efficiency technique.
    """

    def __init__(self, num_classes: int = 200, use_bottleneck: bool = True,
                 use_residual: bool = True):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            # First block: no bottleneck/residual since in_ch=3 is too small
            _FactBlock(3, 64, k=3, use_bottleneck=False, use_residual=False),
            nn.MaxPool2d(2),
            _FactBlock(64, 192, k=3, use_bottleneck=use_bottleneck, use_residual=use_residual),
            nn.MaxPool2d(2),
            _FactBlock(192, 384, k=3, use_bottleneck=use_bottleneck, use_residual=use_residual),
            _FactBlock(384, 256, k=3, use_bottleneck=use_bottleneck, use_residual=use_residual),
            _FactBlock(256, 256, k=3, use_bottleneck=use_bottleneck, use_residual=use_residual),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_factorized_alexnet(num_classes: int = 200) -> nn.Module:
    return FactorizedAlexNet(num_classes)


# ─── H6: LargeSmallKernelHybrid ──────────────────────────────────────────────

class _LargeSmallBlock(nn.Module):
    """Parallel stacked-3×3 path and 7×7 path, fused by element-wise sum."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.small_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
        )
        self.large_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, padding=3, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
        )
        self.fuse_add = _float_functional()

    def forward(self, x):
        return self.fuse_add.add(self.small_path(x), self.large_path(x))


class LargeSmallKernelHybrid(nn.Module):
    """H6: Parallel stacked-3×3 and depthwise-7×7 paths per stage, fused by sum.

    Architecture: 4 stages of _LargeSmallBlock (two parallel paths — stacked 3×3 for local
    features and 7×7 for wider receptive field — added together), GAP head.
    Tests whether mixing receptive field scales within a stage improves over single-kernel.
    Expected top-1: ~30-40% (dual paths increase model capacity and receptive field diversity).
    Size: ~12 MB FP32 / ~3 MB INT8.
    Training speed: medium (two parallel forward passes per stage).
    QAT: full — FloatFunctional fuse-add, Conv-BN-ReLU fuseable in each path.
    Trade-off: large vs small kernel receptive fields; inspired by VGG and RepLKNet.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            _LargeSmallBlock(3, 64), nn.MaxPool2d(2),
            _LargeSmallBlock(64, 128), nn.MaxPool2d(2),
            _LargeSmallBlock(128, 256), nn.MaxPool2d(2),
            _LargeSmallBlock(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_large_small_kernel(num_classes: int = 200) -> nn.Module:
    return LargeSmallKernelHybrid(num_classes)


# ─── H7: ShuffleResidualHybrid ───────────────────────────────────────────────

def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    B, C, H, W = x.shape
    channels_per_group = C // groups
    x = x.view(B, groups, channels_per_group, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class _ShuffleResBlock(nn.Module):
    """Grouped 1×1 + depthwise 3×3 + grouped 1×1, channel shuffle, residual skip."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 4):
        super().__init__()
        self.groups = groups
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),  # DW
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
        )
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.skip_add = _float_functional()

    def forward(self, x):
        out = _channel_shuffle(self.block(x), self.groups)
        return self.skip_add.add(out, self.shortcut(x))


class ShuffleResidualHybrid(nn.Module):
    """H7: Grouped convolutions + channel shuffle + residual connections.

    Architecture: stem → 6× _ShuffleResBlock (grouped 1×1 → DW 3×3 → grouped 1×1,
    channel shuffle, residual skip) → GAP → Linear.
    Channel shuffle ensures cross-group information mixing without dense convolutions.
    Expected top-1: ~40-50% (grouped convolutions + shuffle approaches full-channel efficiency).
    Size: ~8 MB FP32 / ~2 MB INT8.
    Training speed: fast (grouped convolutions reduce FLOPs, DW for spatial mixing).
    QAT: full — FloatFunctional skip-adds, channel shuffle is a reshape (no arithmetic).
    Note: all channel counts must be divisible by groups=4.
    Trade-off: channel mixing strategy — shuffle vs full convolution; tests ShuffleNet insight.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.features = nn.Sequential(
            _ShuffleResBlock(64, 128),
            _ShuffleResBlock(128, 128),
            nn.MaxPool2d(2),
            _ShuffleResBlock(128, 256),
            _ShuffleResBlock(256, 256),
            nn.MaxPool2d(2),
            _ShuffleResBlock(256, 512),
            _ShuffleResBlock(512, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_shuffle_residual(num_classes: int = 200) -> nn.Module:
    return ShuffleResidualHybrid(num_classes)


# ─── H8: ConvNeXtLiteHybrid ──────────────────────────────────────────────────

class _ConvNeXtLiteBlock(nn.Module):
    """ConvNeXt-inspired block: DW 7×7 → BN → 1×1 expand (4×) → GELU → 1×1 project + residual.

    Uses BatchNorm instead of LayerNorm for QAT compatibility with fbgemm.
    Note: GELU is not natively supported by fbgemm; replace with ReLU for full INT8.
    """

    def __init__(self, ch: int, expand_ratio: int = 4):
        super().__init__()
        mid_ch = ch * expand_ratio
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 7, padding=3, groups=ch, bias=False),  # DW 7×7
            nn.BatchNorm2d(ch),
            nn.Conv2d(ch, mid_ch, 1, bias=False),                    # expand
            nn.GELU(),
            nn.Conv2d(mid_ch, ch, 1, bias=False),                    # project
            nn.BatchNorm2d(ch),
        )
        self.skip_add = _float_functional()

    def forward(self, x):
        return self.skip_add.add(self.block(x), x)


class ConvNeXtLiteHybrid(nn.Module):
    """H8: Depthwise 7×7 + inverted bottleneck (4× expand) + GELU + residual (ConvNeXt-style).

    Architecture: Conv2d(4,4) patchify stem → 3 stages of 2× _ConvNeXtLiteBlock +
    strided downsampler → GAP → Linear. Uses BN instead of LN for fbgemm compatibility.
    Expected top-1: ~45-55% (transformer-era design in a tiny model; GELU helps expressivity).
    Size: ~4 MB FP32 / ~1 MB INT8.
    Training speed: medium (DW 7×7 + inverted bottleneck; GELU slightly slower than ReLU).
    QAT: partial — BN fuseable, FloatFunctional skip-add; GELU not fbgemm-supported
    (replace with ReLU in the block for full INT8 compatibility).
    Trade-off: modern transformer-era CNN principles in a constrained parameter budget.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=4, bias=False),  # 64×64 → 16×16
            nn.BatchNorm2d(64),
        )
        self.stages = nn.Sequential(
            _ConvNeXtLiteBlock(64), _ConvNeXtLiteBlock(64),
            nn.Conv2d(64, 128, 2, stride=2, bias=False), nn.BatchNorm2d(128),   # 16→8
            _ConvNeXtLiteBlock(128), _ConvNeXtLiteBlock(128),
            nn.Conv2d(128, 256, 2, stride=2, bias=False), nn.BatchNorm2d(256),  # 8→4
            _ConvNeXtLiteBlock(256), _ConvNeXtLiteBlock(256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.stages(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_convnext_lite(num_classes: int = 200) -> nn.Module:
    return ConvNeXtLiteHybrid(num_classes)


# ─── H9: KernelConstraintNetwork ─────────────────────────────────────────────

class _KCNBlock(nn.Module):
    """Kernel-constrained block: two Inception-lite branches (1×1→3×3 and 1×1→3×3→3×3),
    outputs concatenated, residual skip added."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch_ch = out_ch // 2

        self.branch_a = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=False),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=False),
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=False),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=False),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=False),
        )
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.skip_add = _float_functional()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat([self.branch_a(x), self.branch_b(x)], dim=1)
        return self.relu(self.skip_add.add(out, self.shortcut(x)))


class KernelConstraintNetwork(nn.Module):
    """H9: Only 1×1 and 3×3 kernels, compensated by depth + residuals + multi-branch. Core model.

    Architecture: stem (3×3) → 12× _KCNBlock stages (two Inception-lite branches: 1×1→3×3
    and 1×1→3×3→3×3, concatenated + residual) → GAP → Linear. All convolutions are
    strictly 1×1 or 3×3, maximizing Winograd compatibility.
    Expected top-1: ~40-55% (depth + residuals + multi-branch compensate for kernel constraint;
    this is the primary research question of the project).
    Size: ~20-30 MB FP32 / ~6 MB INT8.
    Training speed: medium-slow (12 stages with dual branches).
    QAT: full — 1×1 and 3×3 kernels are the most Winograd-compatible and fbgemm-efficient;
    FloatFunctional skip-adds; Conv-BN-ReLU fuseable in each branch.
    Trade-off: strict kernel constraint vs accuracy; tests the central research hypothesis.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.features = nn.Sequential(
            _KCNBlock(64, 128),   # stages 1-2
            _KCNBlock(128, 128),
            nn.MaxPool2d(2),
            _KCNBlock(128, 256),  # stages 3-4
            _KCNBlock(256, 256),
            nn.MaxPool2d(2),
            _KCNBlock(256, 384),  # stages 5-6
            _KCNBlock(384, 384),
            _KCNBlock(384, 512),  # stages 7-8
            _KCNBlock(512, 512),
            _KCNBlock(512, 512),  # stages 9-10
            _KCNBlock(512, 512),  # stages 11-12
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_kernel_constraint(num_classes: int = 200) -> nn.Module:
    return KernelConstraintNetwork(num_classes)


# ─── H10: AttentionAugmentedCNN ──────────────────────────────────────────────

class _ChannelSEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention module.

    Note: Sigmoid is not fbgemm-compatible; disable SE or replace Sigmoid with ReLU
    clamp for full INT8 quantization.
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


class AttentionAugmentedCNN(nn.Module):
    """H10: AlexNet3x3 backbone augmented with Squeeze-and-Excitation channel attention.

    Architecture: 5 AlexNet3x3-style stages (3×3 conv + ReLU + optional MaxPool) each
    followed by a _ChannelSEBlock (global avg pool → FC squeeze → FC excitation → Sigmoid
    channel re-weighting). Same FC head as AlexNet3x3.
    Expected top-1: ~15-25% (SE attention modestly helps small CNNs; same FC bottleneck).
    Size: ~220 MB FP32 / ~55 MB INT8 (large FC head dominates).
    Training speed: slow (5 SE blocks add forward + backward overhead; large FC head).
    QAT: partial — Sigmoid in SE blocks is not fbgemm-supported. Run with SE disabled
    or replace Sigmoid with hard-sigmoid approximation for INT8.
    Trade-off: channel/spatial attention vs raw capacity; does SE compensate for kernel limits?
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=False), nn.MaxPool2d(2),
            _ChannelSEBlock(64),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(inplace=False), nn.MaxPool2d(2),
            _ChannelSEBlock(192),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=False),
            _ChannelSEBlock(384),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=False),
            _ChannelSEBlock(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=False),
            _ChannelSEBlock(256),
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


def build_attention_augmented(num_classes: int = 200) -> nn.Module:
    return AttentionAugmentedCNN(num_classes)
