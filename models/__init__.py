from .baselines import AlexNetTV, VGGStyleCNN, ResNet18TV, MobileNetV2TV
from .alexnet_variants import (
    AlexNet3x3FC, AlexNet3x3GAP, AlexNet2x2GAP, AlexNet2x2FC,
    AlexNetStacked, AlexNetMixed, AlexNetSmallKernel,
)
from .compensation import (
    AlexNetBottleneck, AlexNetFactorized, AlexNetGroupConv, AlexNetDepthwiseSep,
    AlexNetResidual, AlexNetFire, AlexNetSE, AlexNetSmallKernelWithBN,
    AlexNetDilatedFC, AlexNetDilatedGAP,
)
from .tinyhybridnet import (
    FireMobileResidual, TinyHybridNet,
    InvertedResidual, TinyMobileNetV2,
)
from .final_architecture import (
    AlexNetFinalBottleneckFire, AlexNetFinalFireResidual,
    AlexNetFinalBottleneckResidual, AlexNetFinalDepthwiseFire,
)
