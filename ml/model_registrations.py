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
    AlexNet3x3FC,
    AlexNet3x3GAP,
    AlexNet2x2GAP,
    AlexNet2x2FC,
    AlexNetStacked,
    AlexNetMixed,
    AlexNetBottleneck,
    AlexNetFactorized,
    AlexNetGroupConv,
    AlexNetDepthwiseSep,
    AlexNetResidual,
    AlexNetFire,
    AlexNetFireBypass,
    AlexNetSE,
    AlexNetSmallKernel,
    AlexNetFinalBottleneckResidual,
    AlexNetFinalFireResidual,
    AlexNetFinalBottleneckFire,
    AlexNetFinalDepthwiseFire,
    AlexNetDilatedFC,
    AlexNetDilatedGAP,
    TinyHybridNet,
    TinyMobileNetV2,
    vit_tiny,
    deit_tiny,
    swin_pico_w2,
    swin_pico_w4,
    swin_pico_w8,
    swin_pico_poolmixer,
    hybrid_bottleneck_swin,
)

# notebooks/phase_1_baseline_training/baselines_qat.ipynb
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

# notebooks/phase_2_kernel_restriction_training/alexnet_qat.ipynb
# FUSE_MAP_ALEXNET_TV is the same Conv-ReLU (no BN) pattern the notebook calls FUSE_CONV_RELU,
# shared by AlexNetTV, 3x3FC/GAP, 2x2GAP/FC, and Mixed.
FUSE_MAP_STACKED = [
    ["0", "1", "2"], ["3", "4", "5"],
    ["7", "8", "9"], ["10", "11", "12"],
    ["14", "15", "16"], ["17", "18", "19"],
    ["20", "21", "22"], ["23", "24", "25"],
    ["26", "27", "28"], ["29", "30", "31"],
]
register_model("alexnet_3x3_fc", AlexNet3x3FC, fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_3x3_gap", AlexNet3x3GAP, fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_2x2_gap", AlexNet2x2GAP, fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_2x2_fc", AlexNet2x2FC, fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_stacked", AlexNetStacked, fuse_map=FUSE_MAP_STACKED, fuse_root_attr="features", lr=1e-3)
register_model("alexnet_mixed", AlexNetMixed, fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)

# large-scale sweep (see configs/experiments/large_scale.yaml)
FUSE_MAP_ALEXNET_SMALLKERNEL = [["0", "1"], ["3", "4"], ["6", "7"], ["8", "9"], ["10", "11"]]
register_model(
    "alexnet_smallkernel",
    AlexNetSmallKernel,
    fuse_map=FUSE_MAP_ALEXNET_SMALLKERNEL,
    fuse_root_attr="features",
    lr=3e-4,
)

# notebooks/phase_3_compensation_and_hybrids_training/compensation_qat.ipynb
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
FUSE_FACTORIZED = [
    ["0", "1", "2"], ["3", "4", "5"],
    ["8", "9", "10"], ["11", "12", "13"],
    ["15", "16", "17"], ["18", "19", "20"],
    ["21", "22", "23"], ["24", "25", "26"],
    ["27", "28", "29"], ["30", "31", "32"],
]
FUSE_GROUPCONV = [
    ["0", "1", "2"],
    ["4", "5", "6"],
    ["8", "9", "10"],
    ["11", "12", "13"],
    ["14", "15", "16"],
]
register_model("alexnet_bottleneck", AlexNetBottleneck, fuse_map=find_fuse_groups(AlexNetBottleneck()), lr=1e-3)
register_model("alexnet_factorized", AlexNetFactorized, fuse_map=FUSE_FACTORIZED, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_groupconv", AlexNetGroupConv, fuse_map=FUSE_GROUPCONV, fuse_root_attr="features", lr=1e-3)
register_model("alexnet_depthwisesep", AlexNetDepthwiseSep, fuse_map=FUSE_DEPTHWISESEP, fuse_root_attr="features", lr=1e-3)
register_model("alexnet_residual", AlexNetResidual, fuse_map=find_fuse_groups(AlexNetResidual()), lr=3e-4)
register_model("alexnet_fire", AlexNetFire, fuse_map=find_fuse_groups(AlexNetFire()), lr=1e-3)
register_model("alexnet_fire_bypass", AlexNetFireBypass, fuse_map=find_fuse_groups(AlexNetFireBypass()), lr=1e-3)
# QAT skipped — Sigmoid isn't fbgemm-fusable, so this trains FP32-only (matches the notebook).
register_model("alexnet_se", AlexNetSE, fuse_map=[], lr=3e-4)
register_model("alexnet_dilated_fc", AlexNetDilatedFC, fuse_map=FUSE_MAP_DILATED, fuse_root_attr="features", lr=1e-3)
register_model("alexnet_dilated_gap", AlexNetDilatedGAP, fuse_map=FUSE_MAP_DILATED, fuse_root_attr="features", lr=1e-3)

# notebooks/phase_3_compensation_and_hybrids_training/efficient_hybrids_qat.ipynb
register_model("tinyhybridnet", TinyHybridNet, fuse_map=find_fuse_groups(TinyHybridNet()), lr=3e-4)
register_model("tinymobilenetv2", TinyMobileNetV2, fuse_map=find_fuse_groups(TinyMobileNetV2()), lr=3e-4)

# notebooks/phase_4_compression_and_final_architecture_training/final_architecture_qat.ipynb
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

# ideas/PHASE8_PLAN.md Task 2. patch-embedding Conv2d in ViT/Swin has no adjacent BN
# (uses LayerNorm), so fuse_map=[] for the pure-attention models -- only
# hybrid_bottleneck_swin's _AlexBottleneck stem has fusable Conv-BN-ReLU triples.
# lr/weight_decay follow DeiT's recipe (Touvron et al. 2021), not this project's
# CNN-tuned defaults (Task 4 Pitfalls / Blocking Issue #5); warmup_epochs is set at
# the experiment-config level (configs/experiments/phase8.yaml) since it isn't a
# per-model registry field.
register_model("vit_tiny", vit_tiny, fuse_map=[], lr=5e-4, weight_decay=0.05)
register_model("deit_tiny", deit_tiny, fuse_map=[], lr=5e-4, weight_decay=0.05)
register_model("swin_pico_w2", swin_pico_w2, fuse_map=[], lr=5e-4, weight_decay=0.05)
register_model("swin_pico_w4", swin_pico_w4, fuse_map=[], lr=5e-4, weight_decay=0.05)
register_model("swin_pico_w8", swin_pico_w8, fuse_map=[], lr=5e-4, weight_decay=0.05)
register_model("swin_pico_poolmixer", swin_pico_poolmixer, fuse_map=[], lr=5e-4, weight_decay=0.05)
register_model(
    "hybrid_bottleneck_swin",
    hybrid_bottleneck_swin,
    fuse_map=find_fuse_groups(hybrid_bottleneck_swin()),
    lr=5e-4,
    weight_decay=0.05,
)
