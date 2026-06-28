from .baselines import (
    AlexNetTV, build_alexnet,
    VGGStyleCNN, build_vggstylecnn,
    ResNet18TV, build_resnet18,
    MobileNetV2TV, build_mobilenetv2,
)
from .alexnet_variants import (
    AlexNet3x3, build_alexnet_3x3,
    AlexNet2x2, build_alexnet_2x2,
    AlexNetStacked, build_alexnet_stacked,
    AlexNetMixed, build_alexnet_mixed,
    AlexNetSmallKernel, build_alexnet_smallkernel,
)
from .compensation import (
    AlexNetBottleneck, build_alexnet_bottleneck,
    AlexNetFactorized, build_alexnet_factorized,
    AlexNetGroupConv, build_alexnet_groupconv,
    AlexNetDepthwiseSep, build_alexnet_depthwisesep,
    AlexNetResidual, build_alexnet_residual,
    AlexNetFire, build_alexnet_fire,
    AlexNetGAP, build_alexnet_gap,
    AlexNetSE, build_alexnet_se,
)
