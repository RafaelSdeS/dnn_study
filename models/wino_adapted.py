"""Architectures adapted to the Winograd F(4,3) accelerator's eligibility contract
(Winograd-FPGA repo, docs/plano_avaliacao_redes_winograd.md §1.2/§4).

Two kinds of adaptation live here:

1. **Torchvision topologies from scratch** (`googlenet_fpga`, `resnet50_fpga`,
   `squeezenet1_1_fpga`) -- Fase 1 of the plan. Their 7x7/s2 stem becomes three
   3x3/s1 convs + maxpool (equivalent receptive field), their 3x3/s2 convs
   become 3x3/s1 + maxpool 2x2 (free on this hardware -- disjoint quadrants of
   the 4x4 output tile), and any pool that *reduces* size becomes 2x2/s2 (a
   3x3/s2 pool crosses tile and would need a line buffer this accelerator
   doesn't have). No pretrained weights (Fase 2: everything trains from zero,
   single budget -- there'd be nothing to transfer from anyway once the stem
   and strides change).

2. **Existing alexnet_rafael architectures that have exactly one 3x3/s2 conv**
   (`alexnet_stacked_fpga`, `alexnet_bottleneck_fpga`,
   `alexnet_final_fire_residual_fpga`, `alexnet_final_bottleneck_residual_fpga`,
   `alexnet_3x3_fc_fpga`) -- same stride-2->stride-1+maxpool conversion, applied
   right after construction so FP32 training (Fase 2's from-scratch, single
   budget) already fits the eligible architecture end to end, rather than
   training the native stride-2 version and patching just that one layer during
   the much shorter qat_wino fine-tune. These are SEPARATE registry entries
   with their own checkpoints -- the native alexnet_stacked/alexnet_bottleneck/
   etc. registrations (used by other experiments: phase3, phase7 det/seg,
   phase9 compression) are untouched.

The actual conversion logic (stem_3x3, convert_stride2_and_pools) is NOT
duplicated here -- it's imported from the sibling Winograd-FPGA repo's
scripts/avaliacao_redes/models_wino.py. That's a deliberate exception to this
project's "don't cross-import model definitions" rule (see
scripts/avaliacao_redes/README.md over there): models_tier.py's Tier A/B
networks are from-scratch designs whose exact geometry is the content that
must not silently drift, so those ARE duplicated on purpose. stem_3x3/
convert_stride2_and_pools are a generic torchvision+conversion recipe with no
unique design content -- importing it once is less likely to drift than
copy-pasting the same recipe into two repos.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _import_models_wino():
    default_root = Path.home() / "Documents" / "Winograd-FPGA" / "scripts" / "avaliacao_redes"
    root = Path(os.environ.get("WINOGRAD_FPGA_ROOT", default_root)).expanduser()
    if not (root / "models_wino.py").exists():
        raise FileNotFoundError(
            f"models_wino.py not found at {root} -- set WINOGRAD_FPGA_ROOT to the "
            "Winograd-FPGA repo's scripts/avaliacao_redes directory"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import models_wino
    return models_wino


def googlenet_fpga(num_classes: int = 200):
    return _import_models_wino().googlenet_wino(num_classes)


def resnet50_fpga(num_classes: int = 200):
    return _import_models_wino().resnet50_wino(num_classes)


def squeezenet1_1_fpga(num_classes: int = 200):
    return _import_models_wino().squeezenet1_1_wino(num_classes)


def _stride1_variant(ctor):
    """Build `ctor(num_classes)`, then apply the plan's §4 rule 3 conversion
    in place (fresh random weights for the surgically-replaced layer -- this
    runs before any training, so there's nothing to preserve there)."""
    def build(num_classes: int = 200):
        model = ctor(num_classes)
        _import_models_wino().convert_stride2_and_pools(model)
        return model
    return build


def alexnet_stacked_fpga(num_classes: int = 200):
    from models import AlexNetStacked
    return _stride1_variant(AlexNetStacked)(num_classes)


def alexnet_bottleneck_fpga(num_classes: int = 200):
    from models import AlexNetBottleneck
    return _stride1_variant(AlexNetBottleneck)(num_classes)


def alexnet_final_fire_residual_fpga(num_classes: int = 200):
    from models import AlexNetFinalFireResidual
    return _stride1_variant(AlexNetFinalFireResidual)(num_classes)


def alexnet_final_bottleneck_residual_fpga(num_classes: int = 200):
    from models import AlexNetFinalBottleneckResidual
    return _stride1_variant(AlexNetFinalBottleneckResidual)(num_classes)


def alexnet_3x3_fc_fpga(num_classes: int = 200):
    """The closest thing in this repo to "the original AlexNet, adapted to the
    FPGA": AlexNet3x3FC already keeps the classic 64->192->384->256->256 channel
    progression and the original's large 3-layer FC head (4096-4096-num_classes)
    -- unlike AlexNet3x3GAP, which swaps that head for GAP+Linear as a deliberate
    ablation variable. Its one 3x3/s2 stem conv gets the same treatment as above.
    """
    from models import AlexNet3x3FC
    return _stride1_variant(AlexNet3x3FC)(num_classes)
