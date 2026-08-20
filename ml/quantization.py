import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.ao.quantization as tq

from .registry import MODEL_REGISTRY


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


def exclude_attention_from_qat(model: nn.Module) -> nn.Module:
    """Set qconfig=None on LayerNorm and ShiftedWindowAttention (Phase 8, D6).

    fbgemm QAT has no fused quantized LayerNorm, and ShiftedWindowAttention's
    windowing/softmax math is functional (not decomposed into observable submodules
    the way torch.ao.nn.quantizable.MultiheadAttention is) -- excluding the whole
    subtree is D6's documented, coarser fallback for it. A no-op for every pre-Phase-8
    model (none contain LayerNorm or ShiftedWindowAttention), so this is safe to run
    unconditionally inside prepare_qat_model() below rather than needing a per-model
    call site.
    """
    from torchvision.models.swin_transformer import ShiftedWindowAttention
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, ShiftedWindowAttention)):
            module.qconfig = None
    return model


def swap_quantizable_mha(model: nn.Module) -> nn.Module:
    """Replace nn.MultiheadAttention with torch.ao.nn.quantizable.MultiheadAttention
    (Phase 8, D6 -- ViT/DeiT path only; Swin's ShiftedWindowAttention has no quantizable
    counterpart and uses exclude_attention_from_qat instead).

    Splits the stock module's fused in_proj_weight/bias into the quantizable module's
    separate linear_Q/K/V, mirroring the weight-splitting torch.ao.nn.quantizable's own
    from_float() classmethod does. Does NOT call from_float()/prepare() directly:
    those force PyTorch's static-PTQ observer-insertion flow, which is incompatible
    with this project's QAT flow (build_qat_from_model -> tq.prepare_qat()). The
    swapped module is left as plain float so prepare_qat_model() below processes its
    Linears/QuantStubs the same generic way as any other module in the tree.

    Call this BEFORE build_qat_from_model() on ViT/DeiT models, then verify with
    torch.allclose() on a pre/post-swap forward pass before spending any QAT training
    time (ideas/PHASE8_PLAN.md Task 3 Blocking Issue #1) -- a naive
    load_state_dict(strict=False) transfer verified to silently succeed while leaving
    linear_Q/K/V at random init, so this hand-split is the actual fix, not that.
    """
    from torch.ao.nn.quantizable.modules.activation import MultiheadAttention as QuantizableMHA

    for name, child in model.named_children():
        if isinstance(child, nn.MultiheadAttention):
            assert child._qkv_same_embed_dim, "separate q/k/v-dim MHA not handled"
            e = child.embed_dim
            qmha = QuantizableMHA(
                e, child.num_heads, dropout=child.dropout,
                bias=child.in_proj_bias is not None, batch_first=child.batch_first,
            )
            qmha.linear_Q.weight = nn.Parameter(child.in_proj_weight[0:e, :].clone())
            qmha.linear_K.weight = nn.Parameter(child.in_proj_weight[e:2 * e, :].clone())
            qmha.linear_V.weight = nn.Parameter(child.in_proj_weight[2 * e:, :].clone())
            if child.in_proj_bias is not None:
                qmha.linear_Q.bias = nn.Parameter(child.in_proj_bias[0:e].clone())
                qmha.linear_K.bias = nn.Parameter(child.in_proj_bias[e:2 * e].clone())
                qmha.linear_V.bias = nn.Parameter(child.in_proj_bias[2 * e:].clone())
            qmha.out_proj.weight = nn.Parameter(child.out_proj.weight.clone())
            if child.out_proj.bias is not None:
                qmha.out_proj.bias = nn.Parameter(child.out_proj.bias.clone())
            setattr(model, name, qmha)
        else:
            swap_quantizable_mha(child)
    return model


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
    exclude_attention_from_qat(model)
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
    spec = MODEL_REGISTRY.get(arch_name)
    if spec is None:
        raise KeyError(f"{arch_name!r} not in MODEL_REGISTRY. Registered: {list(MODEL_REGISTRY)}")
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


def demo() -> None:
    """Assert-based self-check for swap_quantizable_mha's weight transfer (Phase 8 Task
    3 Blocking Issue #1 -- mandatory before any QAT time is spent). Not run automatically
    -- invoke directly (`python -m ml.quantization`)."""
    embed_dim, num_heads, seq_len, batch = 32, 4, 5, 2
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).eval()
    x = torch.randn(batch, seq_len, embed_dim)
    with torch.no_grad():
        expected, _ = mha(x, x, x, need_weights=False)

    holder = nn.Module()
    holder.add_module("self_attention", mha)
    swap_quantizable_mha(holder)
    with torch.no_grad():
        actual, _ = holder.self_attention(x, x, x, need_weights=False)

    assert torch.allclose(expected, actual, atol=1e-5), (
        "swap_quantizable_mha output diverged from the original nn.MultiheadAttention -- "
        "weight transfer is broken, do not proceed to QAT training"
    )
    print("swap_quantizable_mha: OK, pre/post-swap outputs match within atol=1e-5")

    ln = nn.LayerNorm(8)
    swa_model = nn.Module()
    from torchvision.models.swin_transformer import ShiftedWindowAttention
    swa = ShiftedWindowAttention(dim=8, window_size=[2, 2], shift_size=[0, 0], num_heads=2)
    swa_model.add_module("norm", ln)
    swa_model.add_module("attn", swa)
    exclude_attention_from_qat(swa_model)
    assert ln.qconfig is None and swa.qconfig is None, "exclude_attention_from_qat did not set qconfig=None"
    print("exclude_attention_from_qat: OK, LayerNorm/ShiftedWindowAttention excluded")


if __name__ == "__main__":
    demo()
