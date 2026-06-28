import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.ao.quantization as tq

from .registry import MODEL_REGISTRY
from .checkpoint import load_checkpoint


def find_fuse_groups(module: nn.Module, prefix: str = "") -> list:
    """Walk the module tree and collect fusable Conv-BN(-ReLU) groups.

    Works for arbitrarily nested modules (e.g. FireMobileResidual.block).
    Returns a list of dotted-path lists suitable for tq.fuse_modules_qat,
    e.g. ["stem.0", "stem.1", "stem.2"].
    """
    groups = []
    children = list(module.named_children())
    i = 0
    while i < len(children):
        name, child = children[i]
        path = f"{prefix}{name}"

        if isinstance(child, nn.Conv2d) and i + 1 < len(children) and isinstance(children[i + 1][1], nn.BatchNorm2d):
            bpath = f"{prefix}{children[i + 1][0]}"
            if i + 2 < len(children) and isinstance(children[i + 2][1], nn.ReLU):
                groups.append([path, bpath, f"{prefix}{children[i + 2][0]}"])
                i += 3
                continue
            groups.append([path, bpath])
            i += 2
            continue

        if len(list(child.children())) > 0:
            groups.extend(find_fuse_groups(child, prefix=f"{path}."))
        i += 1

    return groups


def prepare_qat_model(
    model: nn.Module,
    fuse_pairs: list,
    fuse_root: nn.Module | None = None,
    qengine: str = "fbgemm",
) -> nn.Module:
    """Deep-copy model, fuse Conv-BN(-ReLU) pairs, insert fake-quant observers."""
    model = copy.deepcopy(model)
    model.train()
    model.qconfig = tq.get_default_qat_qconfig(qengine)
    root = model if fuse_root is None else fuse_root
    if fuse_pairs:
        tq.fuse_modules_qat(root, fuse_pairs, inplace=True)
    return tq.prepare_qat(model, inplace=False)


def build_qat_from_model(model: nn.Module, arch_name: str, device: torch.device) -> nn.Module:
    """Apply QAT preparation to a pre-loaded FP32 model."""
    spec = MODEL_REGISTRY[arch_name]
    root_attr = spec.get("fuse_root_attr")
    fuse_root = getattr(model, root_attr) if root_attr else None
    return prepare_qat_model(model, spec["fuse_map"], fuse_root=fuse_root).to(device)


def load_best_model(
    arch_name: str,
    ctor,
    save_dir: str | Path,
    device: torch.device,
    eval_mode: bool = True,
) -> nn.Module:
    """Reload the best FP32 checkpoint for an architecture."""
    model = ctor()
    path = Path(save_dir) / f"{arch_name}_best.pth"
    # weights_only=False needed here: checkpoint may contain full training state
    ckpt = torch.load(path, map_location=str(device), weights_only=False)
    # support both full checkpoint dicts and bare state dicts
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def build_qat(arch_name: str, save_dir: str | Path, device: torch.device) -> nn.Module:
    """Load best FP32 checkpoint → prepare QAT model."""
    spec = MODEL_REGISTRY[arch_name]
    model = load_best_model(arch_name, spec["ctor"], save_dir, device, eval_mode=False)
    return build_qat_from_model(model, arch_name, device)


def convert_to_int8(qat_model: nn.Module, inplace: bool = False) -> nn.Module:
    """Convert a trained QAT model to real INT8 ops (CPU-only)."""
    qat_model = qat_model.to("cpu").eval()
    return torch.ao.quantization.convert(qat_model, inplace=inplace)


def make_qat_callback(freeze_bn_epoch: int = 3, disable_observer_epoch: int = 5):
    """Return an epoch_callback that freezes BN stats then disables observers."""
    def cb(epoch: int, model: nn.Module) -> None:
        if epoch == freeze_bn_epoch:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        if epoch == disable_observer_epoch:
            model.apply(torch.ao.quantization.disable_observer)
    return cb
