"""Phase 4 — Final Architecture: combinations of Phase 1-3's most successful mechanisms.

Each model pairs two Tier-1/notable compensation mechanisms from Phase 3
(Bottleneck, Fire, Residual, Depthwise-Separable) to test whether combining
them compounds their individual gains. All use GAP + single Linear head
(the pattern shared by Phase 3's smallest, most quantization-stable models).
"""

import torch.nn as nn
import torch.ao.quantization as tq

from .compensation import _AlexBottleneck, _FireModule, _float_functional


class _FireResBlock(nn.Module):
    """Fire module (squeeze -> parallel 1x1+3x3 expand) with a residual shortcut."""

    def __init__(self, in_ch: int, squeeze_ch: int, expand_ch: int):
        super().__init__()
        out_ch = expand_ch * 2
        self.fire = _FireModule(in_ch, squeeze_ch, expand_ch)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.skip_add = _float_functional()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.skip_add.add(self.fire(x), self.shortcut(x)))


class _BottleneckResBlock(nn.Module):
    """1x1->3x3->1x1 bottleneck with a residual shortcut (Bottleneck + Residual)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 4):
        super().__init__()
        mid_ch = max(out_ch // reduction, 32)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
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


# ─── AlexNetFinalBottleneckFire ────────────────────────────────────────────────

class AlexNetFinalBottleneckFire(nn.Module):
    """Bottleneck stem (params-efficient downsampling) + Fire body (multi-scale expand).

    Architecture: 2 bottleneck stages (stride-2 downsampling) then 3 Fire stages
    (squeeze-expand), GAP head. Combines Phase 3's two smallest, quantization-stable
    mechanisms: bottleneck for early spatial reduction, Fire for the channel-expansion
    body. Channel progression 3->64->192->384->256->256 matches AlexNetBottleneck/Fire.
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
            _FireModule(192, 96, 192),   # -> 384
            _FireModule(384, 64, 128),   # -> 256
            _FireModule(256, 64, 128),   # -> 256
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


# ─── AlexNetFinalFireResidual ──────────────────────────────────────────────────

class AlexNetFinalFireResidual(nn.Module):
    """Fire modules with residual shortcuts — adds gradient-flow to Phase 3's Fire.

    Architecture: stem conv -> 3 _FireResBlock stages (Fire path + 1x1-projected
    shortcut, FloatFunctional add), GAP head. Tests whether residual connections
    (Phase 3's single most impactful accuracy mechanism) compound with Fire's
    channel compression.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
        )
        self.features = nn.Sequential(
            nn.MaxPool2d(2),
            _FireResBlock(64, 48, 96),    # -> 192
            nn.MaxPool2d(2),
            _FireResBlock(192, 96, 192),  # -> 384
            _FireResBlock(384, 64, 128),  # -> 256
            _FireResBlock(256, 64, 128),  # -> 256 (true residual, in_ch == out_ch)
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


# ─── AlexNetFinalBottleneckResidual ────────────────────────────────────────────

class AlexNetFinalBottleneckResidual(nn.Module):
    """Bottleneck blocks with residual shortcuts — Phase 3's two best mechanisms merged.

    Architecture: 5 _BottleneckResBlock stages (1x1->3x3->1x1 + FloatFunctional
    skip-add), GAP head. AlexNetBottleneck had no residual; AlexNetResidual used
    plain 3x3 pairs. This tests the bottleneck-efficiency + residual-gradient-flow
    combination directly.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            _BottleneckResBlock(3, 64, stride=2),
            nn.MaxPool2d(2),
            _BottleneckResBlock(64, 192),
            nn.MaxPool2d(2),
            _BottleneckResBlock(192, 384),
            _BottleneckResBlock(384, 256),
            _BottleneckResBlock(256, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


# ─── AlexNetFinalDepthwiseFire ─────────────────────────────────────────────────

class AlexNetFinalDepthwiseFire(nn.Module):
    """Depthwise-separable stem (max FLOPs reduction) + Fire body (multi-scale expand).

    Architecture: 2 DW+PW stages for early downsampling, 3 Fire stages for the
    channel-expansion body, GAP head. Pairs Phase 3's most FLOPs-efficient
    mechanism (depthwise-separable, though QAT-unstable at -2.92pp alone) with
    Fire's quantization-stable, multi-scale expand — testing whether Fire's
    stability compensates for depthwise-separable's QAT weakness.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.features = nn.Sequential(
            # Stage 1: DW on 3 channels, then PW to 64
            nn.Conv2d(3, 3, 3, stride=2, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(3), nn.ReLU(inplace=False),
            nn.Conv2d(3, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Stage 2: DW+PW to 192
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Conv2d(64, 192, 1, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Stages 3-5: Fire body
            _FireModule(192, 96, 192),   # -> 384
            _FireModule(384, 64, 128),   # -> 256
            _FireModule(256, 64, 128),   # -> 256
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x
