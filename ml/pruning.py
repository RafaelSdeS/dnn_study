"""Structured (channel) pruning — Phase 9 Task 2.

Removes whole output channels, never individual weights: every surviving `nn.Conv2d`
stays dense (`groups == 1`), just narrower — unlike unstructured/masked sparsity, which
keeps the tensor's shape but zeros entries irregularly and gains nothing on Winograd
hardware. See `ideas/PHASE9_PLAN.md` H2/D3.

Scoped to `_AlexBottleneck`'s internal squeeze width (the `mid_ch` between its 1x1 and
3x3 convs): that width is private to the block (never consumed outside it), so pruning
it needs no cross-module channel propagation — the block's public in_ch/out_ch, and
everything downstream, is untouched. See D3/Task 2's blocking-issue note on why
`_FireModule`'s squeeze-feeds-two-branches case is out of scope for this pass.
"""

import torch
import torch.nn as nn

from models.compensation import _AlexBottleneck


def _l1_keep_indices(conv: nn.Conv2d, keep_ratio: float) -> torch.Tensor:
    """Indices (ascending) of the top `keep_ratio` fraction of output channels by L1 weight norm."""
    scores = conv.weight.detach().abs().sum(dim=(1, 2, 3))
    n_keep = max(1, round(conv.out_channels * keep_ratio))
    keep = torch.topk(scores, n_keep).indices
    return torch.sort(keep).values


def _slim_conv(conv: nn.Conv2d, out_idx: torch.Tensor | None = None, in_idx: torch.Tensor | None = None) -> nn.Conv2d:
    """A smaller dense Conv2d built from `conv`'s weights at the given channel indices."""
    out_ch = len(out_idx) if out_idx is not None else conv.out_channels
    in_ch = len(in_idx) if in_idx is not None else conv.in_channels
    new_conv = nn.Conv2d(
        in_ch, out_ch, conv.kernel_size, stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None,
    )
    w = conv.weight.detach()
    if out_idx is not None:
        w = w[out_idx]
    if in_idx is not None:
        w = w[:, in_idx]
    new_conv.weight.data.copy_(w)
    if conv.bias is not None:
        b = conv.bias.detach()
        if out_idx is not None:
            b = b[out_idx]
        new_conv.bias.data.copy_(b)
    return new_conv


def _slim_bn(bn: nn.BatchNorm2d, idx: torch.Tensor) -> nn.BatchNorm2d:
    new_bn = nn.BatchNorm2d(len(idx), eps=bn.eps, momentum=bn.momentum)
    new_bn.weight.data.copy_(bn.weight.detach()[idx])
    new_bn.bias.data.copy_(bn.bias.detach()[idx])
    new_bn.running_mean.copy_(bn.running_mean[idx])
    new_bn.running_var.copy_(bn.running_var[idx])
    return new_bn


def prune_bottleneck_block(block: _AlexBottleneck, ratio: float) -> None:
    """Prune a `_AlexBottleneck`'s internal mid_ch width in place (L1-norm channel ranking)."""
    conv1, bn1, _relu1, conv2, bn2, _relu2, conv3, _bn3, _relu3 = block.block
    keep = _l1_keep_indices(conv1, 1.0 - ratio)

    block.block[0] = _slim_conv(conv1, out_idx=keep)
    block.block[1] = _slim_bn(bn1, keep)
    block.block[3] = _slim_conv(conv2, out_idx=keep, in_idx=keep)
    block.block[4] = _slim_bn(bn2, keep)
    block.block[6] = _slim_conv(conv3, in_idx=keep)
    # block.block[7] (bn3) sits on out_ch, which this pass never changes.


def prune_model_channels(model: nn.Module, ratio: float) -> nn.Module:
    """Prune every `_AlexBottleneck` block found in `model`, in place. Returns `model`."""
    for module in model.modules():
        if isinstance(module, _AlexBottleneck):
            prune_bottleneck_block(module, ratio)
    return model


def bottleneck_prune_plan(model: nn.Module, ratio: float) -> list[tuple[str, int, int]]:
    """[(block_name, mid_ch_before, mid_ch_after), ...] without mutating `model`."""
    plan = []
    for name, module in model.named_modules():
        if isinstance(module, _AlexBottleneck):
            mid = module.block[0].out_channels
            kept = max(1, round(mid * (1.0 - ratio)))
            plan.append((name, mid, kept))
    return plan


if __name__ == "__main__":
    from models.compensation import AlexNetBottleneck

    m = AlexNetBottleneck()
    before = sum(p.numel() for p in m.parameters())
    for name, mid, kept in bottleneck_prune_plan(m, 0.4):
        print(f"{name:20s} mid_ch {mid:4d} -> {kept:4d}")

    prune_model_channels(m, 0.4)
    after = sum(p.numel() for p in m.parameters())
    print(f"params: {before:,} -> {after:,}  ({after / before:.1%})")

    x = torch.randn(2, 3, 64, 64)
    m.eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 200), y.shape
    assert all(mod.groups == 1 for mod in m.modules() if isinstance(mod, nn.Conv2d))
    print("self-check passed: forward OK, every Conv2d still dense (groups=1)")
