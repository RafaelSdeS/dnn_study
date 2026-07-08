"""Phase 2 — Kernel Restriction Study: AlexNet with enforced kernel sizes."""

import torch.nn as nn
import torch.ao.quantization as tq


# ─── AlexNet3x3FC ─────────────────────────────────────────────────────────────

class AlexNet3x3FC(nn.Module):
    """All-3×3 AlexNet, FC head — uniform small kernel baseline for the restriction study.

    Architecture: 5 conv stages (all 3×3, same channels as AlexNetTV), AdaptiveAvgPool(6×6),
    3-layer FC head. Identical structure to AlexNetTV but with 3×3 kernels everywhere.
    Paired with AlexNet3x3GAP (same backbone, GAP head) to isolate head type as a variable.
    Expected top-1: ~8-12% (from scratch; 3×3 alone insufficient without residuals).
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


# ─── AlexNet2x2GAP ────────────────────────────────────────────────────────────

class AlexNet2x2GAP(nn.Module):
    """All-2×2 AlexNet, GAP head — tests the impact of even-size kernels (no Winograd, no center pixel).

    Architecture: 5 conv stages (all 2×2, no padding, same channels as AlexNet3x3FC),
    AdaptiveAvgPool(1×1), single Linear head. 2×2 kernels reduce spatial dims by 1 per
    layer; GAP absorbs the resulting non-standard spatial sizes.
    Paired with AlexNet2x2FC (same backbone, FC head) to isolate head type as a variable.
    2×2 kernels: smallest even kernel, no natural center pixel, incompatible with Winograd,
    receptive field smaller than 3×3 per layer.
    Expected top-1: ~5-9% (smallest receptive field in Phase 2; hard to compensate).
    Size: ~3 MB FP32 / ~1 MB INT8 (GAP head vs large FC; fewer params per layer).
    Training speed: fast (GAP head, slightly fewer params per conv than 3×3).
    QAT: full — flat Sequential, Conv-ReLU pairs fuseable; fbgemm supports 2×2 kernels.
    Trade-off: 2×2 vs 3×3 receptive field; even vs odd kernel; Winograd compatibility.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        # Spatial dims: 64 → 32 → 16 → 15 → 7 → 6 → 5 → 4 → AdaptiveAvgPool(1)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 2, stride=2),         # 64→32
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),                        # 32→16
            nn.Conv2d(64, 192, 2),                  # 16→15
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),                        # 15→7 (floor)
            nn.Conv2d(192, 384, 2),                 # 7→6
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, 2),                 # 6→5
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 2),                 # 5→4
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


# ─── AlexNet2x2FC ─────────────────────────────────────────────────────────────

class AlexNet2x2FC(nn.Module):
    """All-2×2 AlexNet, FC head — pairs with AlexNet2x2GAP to isolate head type as a variable.

    Architecture: identical 2×2 backbone to AlexNet2x2GAP (5 conv stages, no padding,
    same channels), but keeps the native 4×4 spatial output instead of collapsing it with
    AdaptiveAvgPool2d(1) — followed by a 3-layer FC head sized to that 4×4×256 output.
    Expected top-1: similar or slightly above AlexNet2x2GAP (FC head adds capacity, but
    the 2×2 backbone's receptive field remains the bottleneck).
    Size: much larger than AlexNet2x2GAP (FC head dominates: 4096*4096 + 4*4*256*4096 params).
    Training speed: slower than AlexNet2x2GAP (large FC head), same conv cost.
    QAT: full — flat Sequential, Conv-ReLU pairs fuseable; identical fuse_map to AlexNet2x2GAP.
    Trade-off: FC vs GAP head on the same 2×2 backbone.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        # Spatial dims: 64 → 32 → 16 → 15 → 7 → 6 → 5 → 4 (no final GAP — kept for FC head)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 2, stride=2),         # 64→32
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),                        # 32→16
            nn.Conv2d(64, 192, 2),                  # 16→15
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),                        # 15→7 (floor)
            nn.Conv2d(192, 384, 2),                 # 7→6
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, 2),                 # 6→5
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 2),                 # 5→4
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 4096),
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


# ─── AlexNet3x3GAP ────────────────────────────────────────────────────────────

class AlexNet3x3GAP(nn.Module):
    """All-3×3 AlexNet, GAP head — moved from compensation.py (formerly AlexNetGAP).

    Architecture: identical backbone to AlexNet3x3FC (5× 3×3 conv, same channels, no BN),
    AdaptiveAvgPool(1×1) + single Linear, replacing the 94M-parameter FC head. Pairs with
    AlexNet3x3FC to isolate head type as a variable, holding the 3×3 backbone constant.
    Previously trained as `AlexNetGAP` in Phase 3 (results.csv row `alexnet_gap`,
    38.74% top-1) — that historical record is unaffected by this move.
    Size: ~3 MB FP32 / ~1 MB INT8 (vs ~220 MB for AlexNet3x3FC with large FC).
    Training speed: fast (same backbone cost as AlexNet3x3FC; GAP removes most parameters).
    QAT: full — flat Sequential, Conv-ReLU pairs fuseable (identical fuse_map to AlexNet3x3FC).
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


# ─── AlexNetStacked ───────────────────────────────────────────────────────────

class AlexNetStacked(nn.Module):
    """2× stacked 3×3 per stage — tests whether depth recovers receptive field.

    Architecture: 5 stages with 2 consecutive 3×3 conv + BN + ReLU each (10 conv total),
    same channel widths as AlexNet3x3FC, same 3-layer FC head. Two stacked 3×3 convs have
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


# ─── AlexNetMixed ─────────────────────────────────────────────────────────────

class AlexNetMixed(nn.Module):
    """AlexNet with alternating 3×3 and 2×2 kernels — tests heterogeneous kernel restriction.

    Architecture: 5 stages alternating 3×3 and 2×2 kernels (3×3 → 2×2 → 3×3 → 2×2 → 3×3),
    same channel widths as AlexNet3x3FC, AdaptiveAvgPool(1×1), single Linear head. GAP handles
    the non-uniform spatial sizes introduced by the 2×2 stages (no padding, -1 per dim).
    Tests whether alternating kernel sizes outperforms either uniform 3×3 or uniform 2×2,
    combining their respective receptive field scales across stages.
    Expected top-1: ~7-12% (mixed strategy between AlexNet3x3FC and AlexNet2x2GAP baselines).
    Size: ~3 MB FP32 / ~1 MB INT8 (GAP head like AlexNet2x2GAP).
    Training speed: fast (GAP head; alternating kernel sizes have similar cost to 3×3).
    QAT: full — flat Sequential, Conv-ReLU pairs fuseable; both 2×2 and 3×3 fbgemm-supported.
    Trade-off: kernel diversity within a single model vs the uniform-restriction baselines.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        # Spatial dims: 64 → 32 → 16 → 15 → 7 → 7 → 6 → 6 → AdaptiveAvgPool(1)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),     # 3×3  64→32
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),                               # 32→16
            nn.Conv2d(64, 192, 2),                         # 2×2  16→15
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),                               # 15→7 (floor)
            nn.Conv2d(192, 384, 3, padding=1),             # 3×3  7→7
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, 2),                        # 2×2  7→6
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1),             # 3×3  6→6
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


# ─── AlexNetSmallKernel ───────────────────────────────────────────────────────

class AlexNetSmallKernel(nn.Module):
    """Kernel-restricted AlexNet — all-3×3, narrow channels, GAP head (minimal design).

    Architecture: 5 conv stages (3×3, narrower channels: 64→128→256→256→256),
    AdaptiveAvgPool(1×1), single Linear classifier. ~36× fewer parameters than AlexNet3x3FC
    (narrow channels + GAP head vs large FC). The most parameter-efficient Phase 2 model.
    Training speed: fast (GAP head eliminates most parameters).
    QAT: full — flat Sequential, Conv-ReLU pairs fuseable.
    Trade-off: narrow channels + GAP vs wide channels + FC; measures head and width contributions.
    Note: no BatchNorm (matches AlexNet3x3FC for controlled comparison).
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


