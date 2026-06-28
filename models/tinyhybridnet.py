import torch
import torch.nn as nn
import torch.ao.quantization as tq


def _float_functional():
    try:
        return torch.nn.quantized.FloatFunctional()
    except AttributeError:
        return tq.FloatFunctional()


class FireMobileResidual(nn.Module):
    """Fire-inspired mobile residual block (squeeze → depthwise → expand + skip)."""

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
    """Efficient hybrid CNN for 64×64 images (200 classes), QAT-ready."""

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


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block (expand → depthwise → project)."""

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
    """MobileNetV2-style CNN for 64×64 images, QAT-ready."""

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
