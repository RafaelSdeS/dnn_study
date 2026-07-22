"""Populates MODEL_REGISTRY for standalone scripts (e.g. scripts/profile_hardware.py).

Training notebooks register models inline in their own cells and never import
this module. This exists only for entry points that need MODEL_REGISTRY
populated without running a notebook first. Values are copied from the
notebook that owns each model (see CLAUDE.md's Model Inventory table) — keep
them in sync if a notebook's fuse_map or lr changes.
"""

from ml.quantization import find_fuse_groups
from ml.registry import register_model
from models import (
    AlexNetTV,
    VGGStyleCNN,
    MobileNetV2TV,
    ResNet18TV,
    AlexNetBottleneck,
    AlexNetDepthwiseSep,
    AlexNetFire,
    AlexNetSmallKernel,
    AlexNetFinalBottleneckResidual,
    AlexNetFinalFireResidual,
    AlexNetFinalBottleneckFire,
    AlexNetFinalDepthwiseFire,
    AlexNetDilatedFC,
    AlexNetDilatedGAP,
)

# notebooks/training/baselines_qat.ipynb
FUSE_MAP_ALEXNET_TV = [["0", "1"], ["3", "4"], ["6", "7"], ["8", "9"], ["10", "11"]]
FUSE_MAP_VGG = [
    ["0", "1", "2"], ["3", "4", "5"],
    ["7", "8", "9"], ["10", "11", "12"],
    ["14", "15", "16"], ["17", "18", "19"],
    ["21", "22", "23"], ["24", "25", "26"],
    ["28", "29", "30"], ["31", "32", "33"],
]
register_model("alexnet_tv", AlexNetTV, fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("vgg_style", VGGStyleCNN, fuse_map=FUSE_MAP_VGG, fuse_root_attr="features", lr=1e-3)
register_model("mobilenetv2", MobileNetV2TV, fuse_map=[], lr=1e-4)
register_model("resnet18tv", ResNet18TV, fuse_map=[], lr=1e-4)

# large-scale sweep (see configs/experiments/large_scale.yaml)
FUSE_MAP_ALEXNET_SMALLKERNEL = [["0", "1"], ["3", "4"], ["6", "7"], ["8", "9"], ["10", "11"]]
register_model(
    "alexnet_smallkernel",
    AlexNetSmallKernel,
    fuse_map=FUSE_MAP_ALEXNET_SMALLKERNEL,
    fuse_root_attr="features",
    lr=3e-4,
)

# notebooks/training/compensation_qat.ipynb
FUSE_DEPTHWISESEP = [
    ["0", "1", "2"], ["3", "4", "5"],
    ["7", "8", "9"], ["10", "11", "12"],
    ["14", "15", "16"], ["17", "18", "19"],
    ["20", "21", "22"], ["23", "24", "25"],
    ["26", "27", "28"], ["29", "30", "31"],
]
FUSE_MAP_DILATED = [
    ["0", "1", "2"], ["4", "5", "6"],
    ["8", "9", "10"], ["11", "12", "13"],
    ["14", "15", "16"],
]
register_model("alexnet_bottleneck", AlexNetBottleneck, fuse_map=find_fuse_groups(AlexNetBottleneck()), lr=1e-3)
register_model("alexnet_depthwisesep", AlexNetDepthwiseSep, fuse_map=FUSE_DEPTHWISESEP, fuse_root_attr="features", lr=1e-3)
register_model("alexnet_fire", AlexNetFire, fuse_map=find_fuse_groups(AlexNetFire()), lr=1e-3)
register_model("alexnet_dilated_fc", AlexNetDilatedFC, fuse_map=FUSE_MAP_DILATED, fuse_root_attr="features", lr=1e-3)
register_model("alexnet_dilated_gap", AlexNetDilatedGAP, fuse_map=FUSE_MAP_DILATED, fuse_root_attr="features", lr=1e-3)

# notebooks/training/final_architecture_qat.ipynb
register_model(
    "alexnet_final_bottleneck_residual",
    AlexNetFinalBottleneckResidual,
    fuse_map=find_fuse_groups(AlexNetFinalBottleneckResidual()),
    lr=1e-3,
)
register_model(
    "alexnet_final_fire_residual",
    AlexNetFinalFireResidual,
    fuse_map=find_fuse_groups(AlexNetFinalFireResidual()),
    lr=1e-3,
)
register_model(
    "alexnet_final_bottleneck_fire",
    AlexNetFinalBottleneckFire,
    fuse_map=find_fuse_groups(AlexNetFinalBottleneckFire()),
    lr=1e-3,
)
register_model(
    "alexnet_final_depthwise_fire",
    AlexNetFinalDepthwiseFire,
    fuse_map=find_fuse_groups(AlexNetFinalDepthwiseFire()),
    lr=1e-3,
)
