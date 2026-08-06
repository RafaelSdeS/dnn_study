"""Structured channel pruning (Phase 9 Task 2) coverage - previously untested."""
import torch
import torch.nn as nn

from models.compensation import _AlexBottleneck
from ml.pruning import bottleneck_prune_plan, prune_model_channels


def test_bottleneck_prune_plan_reports_kept_channels_without_mutating():
    block = _AlexBottleneck(in_ch=8, out_ch=256, reduction=4)  # mid_ch = max(256//4, 32) = 64
    before_mid = block.block[0].out_channels

    plan = bottleneck_prune_plan(nn.Sequential(block), ratio=0.5)

    assert plan == [("0", 64, 32)]
    assert block.block[0].out_channels == before_mid, "plan must not mutate the model"


def test_prune_model_channels_shrinks_mid_conv_and_keeps_forward_shape():
    model = nn.Sequential(_AlexBottleneck(in_ch=8, out_ch=256, reduction=4))
    x = torch.randn(2, 8, 16, 16)

    prune_model_channels(model, ratio=0.5)

    block = model[0]
    assert block.block[0].out_channels == 32  # mid_ch halved: 64 -> 32
    assert block.block[3].in_channels == 32 and block.block[3].out_channels == 32
    assert block.block[6].in_channels == 32 and block.block[6].out_channels == 256  # out_ch untouched

    y = model.eval()(x)
    assert y.shape == (2, 256, 16, 16)
    assert all(m.groups == 1 for m in model.modules() if isinstance(m, nn.Conv2d)), "must stay Winograd-dense"
