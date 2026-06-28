import torch.nn as nn
import torch.ao.quantization as tq
from torchvision.models import alexnet


class AlexNetTV(nn.Module):
    """Torchvision AlexNet with pretrained weights, adapted for Tiny-ImageNet (200 classes).

    Final FC replaced, all ReLU set to inplace=False for QAT compatibility.
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
