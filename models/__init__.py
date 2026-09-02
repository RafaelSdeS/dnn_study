from .baselines import AlexNetTV, VGGStyleCNN, ResNet18TV, MobileNetV2TV
from .alexnet_variants import (
    AlexNet3x3FC, AlexNet3x3GAP, AlexNet2x2GAP, AlexNet2x2FC,
    AlexNetStacked, AlexNetMixed, AlexNetSmallKernel,
)
from .compensation import (
    AlexNetBottleneck, AlexNetFactorized, AlexNetGroupConv, AlexNetDepthwiseSep,
    AlexNetResidual, AlexNetFire, AlexNetFireBypass, AlexNetSE, AlexNetSmallKernelWithBN,
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
from .vit_variants import (
    vit_tiny, deit_tiny, vit_tiny_convstem, swin_pico_w2, swin_pico_w4, swin_pico_w8,
    swin_pico_poolmixer, swin_pico_convstem, hybrid_bottleneck_swin,
)
