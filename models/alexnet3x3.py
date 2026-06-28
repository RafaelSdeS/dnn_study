import torch.nn as nn
import torch.ao.quantization as tq


class AlexNet3x3(nn.Module):
    """All-3×3 AlexNet variant, QAT-compatible (inplace=False ReLU throughout)."""

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
