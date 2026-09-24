"""Populates MODEL_REGISTRY for standalone scripts (e.g. scripts/profile_hardware.py).

Training notebooks register models inline in their own cells and never import
this module. This exists only for entry points that need MODEL_REGISTRY
populated without running a notebook first. Values are copied from the
notebook that owns each model (see CLAUDE.md's Model Inventory table) — keep
them in sync if a notebook's fuse_map or lr changes.
"""

from functools import partial

from torchvision.models.vgg import cfgs as VGG_CFGS

from ml.quantization import find_fuse_groups
from ml.registry import register_model
from ml.winograd_bridge import custom_model, torchvision_model
from models import (
    AlexNetTV,
    VGG16,
    VGGStyleCNN,
    MobileNetV2TV,
    ResNet18TV,
    AlexNet3x3FC,
    AlexNet3x3GAP,
    AlexNet2x2GAP,
    AlexNet2x2FC,
    AlexNetStacked,
    AlexNetMixed,
    AlexNetAdapted,
    AlexNetBottleneck,
    AlexNetFactorized,
    AlexNetGroupConv,
    AlexNetDepthwiseSep,
    AlexNetResidual,
    AlexNetFire,
    AlexNetFireBypass,
    AlexNetSE,
    AlexNetSmallKernel,
    AlexNetSmallKernelWithBN,
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
    vit_tiny_convstem,
    swin_pico_w2,
    swin_pico_w4,
    swin_pico_w8,
    swin_pico_poolmixer,
    swin_pico_convstem,
    hybrid_bottleneck_swin,
)

# notebooks/phase_1_baseline/baselines_qat.ipynb
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

# notebooks/phase_2_kernel_restriction/alexnet_qat.ipynb
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

# Phase 11 (configs/experiments/phase_11_kernel_size_comparison.yaml): original torchvision
# AlexNet/VGG16, from scratch, only kernel_size changes -- see models/baselines.py.
def _fuse_map_vgg16(cfg=VGG_CFGS["D"]):
    """[[conv_idx, bn_idx, relu_idx], ...] for the Sequential models.baselines._vgg16_features
    builds, computed from cfgs["D"] alone -- avoids instantiating VGG16 (134M params) just for
    this."""
    triples, idx = [], 0
    for v in cfg:
        if v == "M":
            idx += 1
        else:
            triples.append([str(idx), str(idx + 1), str(idx + 2)])
            idx += 3
    return triples


FUSE_MAP_VGG16 = _fuse_map_vgg16()
register_model("alexnet_tv_scratch", partial(AlexNetTV, pretrained=False),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_tv_3x3", partial(AlexNetTV, pretrained=False, kernel_size=3),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_tv_2x2", partial(AlexNetTV, pretrained=False, kernel_size=2),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
# Mixed 3x3/2x2 per-layer kernels (configs/experiments/phase_11_mixed_kernel_comparison.yaml) --
# same family/fuse_map as alexnet_tv_3x3/2x2 above, see _ALEXNET_KERNEL_SPECS for the patterns.
register_model("alexnet_tv_mixed_alt", partial(AlexNetTV, pretrained=False, kernel_size="mixed_alt"),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_tv_mixed_early3", partial(AlexNetTV, pretrained=False, kernel_size="mixed_early3"),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_tv_mixed_early2", partial(AlexNetTV, pretrained=False, kernel_size="mixed_early2"),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)

# Phase 11 head/BN ablation (configs/experiments/phase_11_head_bn_ablation.yaml): does GAP vs FC
# head, or BatchNorm vs none, explain the gap between the mixed-kernel/stacked-3x3 models above? A
# head or BN swap only changes classifier/features cosmetically -- `features`' Conv-ReLU indices
# are unchanged for a head-only swap, so those reuse the base model's own fuse_map; a BN swap
# changes indices, so those use find_fuse_groups on a throwaway instance instead of hand-counting.
register_model("alexnet_tv_mixed_alt_gap", partial(AlexNetTV, pretrained=False, kernel_size="mixed_alt", head="gap"),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_tv_mixed_early3_gap", partial(AlexNetTV, pretrained=False, kernel_size="mixed_early3", head="gap"),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_tv_mixed_early2_gap", partial(AlexNetTV, pretrained=False, kernel_size="mixed_early2", head="gap"),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_mixed_fc", partial(AlexNetMixed, head="fc"),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_mixed_bn", partial(AlexNetMixed, batch_norm=True),
               fuse_map=find_fuse_groups(AlexNetMixed(batch_norm=True)), lr=3e-4)
register_model("alexnet_mixed_fc_bn", partial(AlexNetMixed, head="fc", batch_norm=True),
               fuse_map=find_fuse_groups(AlexNetMixed(head="fc", batch_norm=True)), lr=3e-4)

# Phase 11 geometry/BN controls (configs/experiments/phase_11_geometry_controls_s*.yaml): the
# original 11/5/3/3/3 kernels at AlexNet3x3FC/GAP's adapted geometry (Conv-ReLU only -> same
# hand-written fuse_map as 3x3FC/GAP), and 3x3-GAP + BN as the BN control for Bottleneck/Fire.
register_model("alexnet_adapted_orig_fc", partial(AlexNetAdapted, kernels=(11, 5, 3, 3, 3), head="fc"),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_adapted_orig_gap", partial(AlexNetAdapted, kernels=(11, 5, 3, 3, 3), head="gap"),
               fuse_map=FUSE_MAP_ALEXNET_TV, fuse_root_attr="features", lr=3e-4)
register_model("alexnet_3x3_gap_bn", partial(AlexNetAdapted, kernels=(3,) * 5, head="gap", batch_norm=True),
               fuse_map=find_fuse_groups(AlexNetAdapted(kernels=(3,) * 5, head="gap", batch_norm=True)), lr=3e-4)
register_model("alexnet_stacked_gap", partial(AlexNetStacked, head="gap"),
               fuse_map=FUSE_MAP_STACKED, fuse_root_attr="features", lr=1e-3)
# No BN -> features compresses to plain Conv-ReLU pairs (BN entries drop out, shifting every
# index after them); hand-counted once here rather than reusing find_fuse_groups, which only
# detects Conv-(BN-)ReLU and would return [] with batch_norm=False.
FUSE_MAP_STACKED_NOBN = [
    ["0", "1"], ["2", "3"],
    ["5", "6"], ["7", "8"],
    ["10", "11"], ["12", "13"],
    ["14", "15"], ["16", "17"],
    ["18", "19"], ["20", "21"],
]
register_model("alexnet_stacked_fc_nobn", partial(AlexNetStacked, batch_norm=False),
               fuse_map=FUSE_MAP_STACKED_NOBN, fuse_root_attr="features", lr=1e-3)
register_model("alexnet_stacked_gap_nobn", partial(AlexNetStacked, head="gap", batch_norm=False),
               fuse_map=FUSE_MAP_STACKED_NOBN, fuse_root_attr="features", lr=1e-3)
# torchvision's VGG classifier: Linear(0)-ReLU(1)-Dropout(2)-Linear(3)-ReLU(4)-Dropout(5)-Linear(6,
# logits). Fuse the two Linear-ReLU pairs (see prepare_qat_model's classifier_fuse_pairs docstring
# for why vgg16 needs this); classifier.6 has no ReLU after it and stays a standalone quantized Linear.
CLASSIFIER_FUSE_MAP_VGG16 = [["0", "1"], ["3", "4"]]
# qat_disable_observer_epoch=None: freezing observers collapsed vgg16's QAT to ln(200) in 3/3 runs
# (docs/logs/PHASE11_LOG.md, "Revisit 2"), so its fake-quant ranges keep adapting for all of QAT.
register_model("vgg16", partial(VGG16, kernel_size=3),
               fuse_map=FUSE_MAP_VGG16, fuse_root_attr="features", lr=1e-3,
               classifier_fuse_map=CLASSIFIER_FUSE_MAP_VGG16, qat_disable_observer_epoch=None)
register_model("vgg16_2x2", partial(VGG16, kernel_size=2),
               fuse_map=FUSE_MAP_VGG16, fuse_root_attr="features", lr=1e-3,
               classifier_fuse_map=CLASSIFIER_FUSE_MAP_VGG16)

# large-scale sweep (see configs/experiments/large_scale.yaml)
FUSE_MAP_ALEXNET_SMALLKERNEL = [["0", "1"], ["3", "4"], ["6", "7"], ["8", "9"], ["10", "11"]]
register_model(
    "alexnet_smallkernel",
    AlexNetSmallKernel,
    fuse_map=FUSE_MAP_ALEXNET_SMALLKERNEL,
    fuse_root_attr="features",
    lr=3e-4,
)
# Phase 11 head/BN ablation FC twin (configs/experiments/phase_11_head_bn_ablation.yaml) --
# features indices are unchanged by the head swap, so this reuses the GAP variant's fuse_map.
register_model("alexnet_smallkernel_fc", partial(AlexNetSmallKernel, head="fc"),
               fuse_map=FUSE_MAP_ALEXNET_SMALLKERNEL, fuse_root_attr="features", lr=3e-4)
# models/compensation.py — exists since the Phase 2 QAT-drop investigation, never
# trained (Winograd-FPGA plano_avaliacao_redes_winograd.md Fase 1). No `features`
# Sequential (named conv/bn/relu attrs instead), so find_fuse_groups like the
# compensation.py models below rather than an explicit index-based fuse_map.
register_model(
    "alexnet_small_kernel_with_bn",
    AlexNetSmallKernelWithBN,
    fuse_map=find_fuse_groups(AlexNetSmallKernelWithBN()),
    lr=3e-4,
)

# notebooks/phase_3_compensation_and_hybrids/compensation_qat.ipynb
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

# notebooks/phase_3_compensation_and_hybrids/efficient_hybrids_qat.ipynb
register_model("tinyhybridnet", TinyHybridNet, fuse_map=find_fuse_groups(TinyHybridNet()), lr=3e-4)
register_model("tinymobilenetv2", TinyMobileNetV2, fuse_map=find_fuse_groups(TinyMobileNetV2()), lr=3e-4)

# notebooks/phase_4_compression_and_final_architecture/final_architecture_qat.ipynb
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

# docs/plans/PHASE8_PLAN.md Task 2. patch-embedding Conv2d in ViT/Swin has no adjacent BN
# (uses LayerNorm), so fuse_map=[] for the pure-attention models -- only
# hybrid_bottleneck_swin's _AlexBottleneck stem has fusable Conv-BN-ReLU triples.
# lr/weight_decay follow DeiT's recipe (Touvron et al. 2021), not this project's
# CNN-tuned defaults (Task 4 Pitfalls / Blocking Issue #5); warmup_epochs is set at
# the experiment-config level (configs/experiments/phase_8_efficient_vit.yaml) since it isn't a
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

# 3x3-restricted conv stem in front of attention (H5 follow-up) -- unlike the models
# above, these DO have fusable Conv-BN-ReLU triples in their stem (find_fuse_groups
# recurses into ViTTinyConvStem's conv_stem_configs Sequential / SwinPico's replaced
# features[0][0]), see models/vit_variants.py for why.
register_model(
    "vit_tiny_convstem",
    vit_tiny_convstem,
    fuse_map=find_fuse_groups(vit_tiny_convstem()),
    lr=5e-4,
    weight_decay=0.05,
)
register_model(
    "swin_pico_convstem",
    swin_pico_convstem,
    fuse_map=find_fuse_groups(swin_pico_convstem()),
    lr=5e-4,
    weight_decay=0.05,
)

# Winograd-FPGA study (configs/experiments/budget_unico.yaml). Geometry comes from the
# sibling repo's own builders via ml/winograd_bridge.py -- see its docstring for why the
# native classes above can't be reused (same shapes, different state_dict keys). Network
# set follows that repo's docs/plano_avaliacao_redes_winograd.md §2.3/§2.4 as of 2026-09-10.
# No fuse_map (no fbgemm qat/int8 stage) and no lr: budget_unico's uniform_hparams gives
# every model the same one.
# NOTE: ctor() needs WINOGRAD_FPGA_ROOT to resolve -- unlike every ctor above it can raise
# at model-construction time; scripts/pcad/preflight_budget_unico.py builds each one first.
register_model("vgg_style_fpga", custom_model("vgg_style"))
register_model("alexnet_3x3_fc_fpga", custom_model("alexnet_3x3_fc"))
register_model("alexnet_stacked_fpga", custom_model("alexnet_stacked"))
register_model("alexnet_fire_fpga", custom_model("alexnet_fire"))
register_model("alexnet_fire_bypass_fpga", custom_model("alexnet_fire_bypass"))
register_model("alexnet_bottleneck_fpga", custom_model("alexnet_bottleneck"))
register_model("alexnet_final_fire_residual_fpga", custom_model("alexnet_final_fire_residual"))
register_model("alexnet_final_bottleneck_residual_fpga", custom_model("alexnet_final_bottleneck_residual"))
register_model("repvgg_a0_fpga", custom_model("repvgg_a0", convert=False))  # trains raw -- see custom_model's caveat
register_model("wrn_16_4_fpga", custom_model("wrn_16_4"))
register_model("wrn_28_2_fpga", custom_model("wrn_28_2"))
register_model("googlenet_fpga", torchvision_model("googlenet"))
register_model("resnet18_fpga", torchvision_model("resnet18"))
register_model("vgg13_fpga", torchvision_model("vgg13"))
register_model("squeezenet1_1_fpga", torchvision_model("squeezenet1_1"))  # out of budget_unico: qat_wino breaks on it
