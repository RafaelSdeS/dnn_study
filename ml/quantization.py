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
    """Set qconfig=None on LayerNorm, ShiftedWindowAttention, and MultiheadAttention
    (Phase 8, D6 -- revised).

    fbgemm QAT has no fused quantized LayerNorm, and ShiftedWindowAttention's
    windowing/softmax math is functional (not decomposed into observable submodules) --
    excluding the whole subtree is D6's documented, coarser fallback for it.

    D6 originally planned a "strict improvement" path for nn.MultiheadAttention
    (ViT/DeiT): swap_quantizable_mha() below replaces it with
    torch.ao.nn.quantizable.MultiheadAttention so its internal Linears become
    individually quantizable. VERIFIED BROKEN with this codebase's QAT pipeline:
    torch.ao.nn.quantizable.MultiheadAttention is registered in PyTorch's default
    `observed_to_quantized_custom_module_class` mapping, and tq.prepare_qat()'s first
    internal step (`convert()`, which runs BEFORE `prepare()` attaches any observers)
    matches on it and calls its `.from_observed()` classmethod immediately --
    AttributeError: 'Linear' object has no attribute 'activation_post_process',
    confirmed via direct testing, not a guess. torch.ao.nn.quantizable.MultiheadAttention
    is only usable via the static-PTQ prepare()->calibrate->convert() flow or FX
    graph-mode QAT, neither of which this codebase's eager-mode prepare_qat()-based
    pipeline uses. Fallback (same as ShiftedWindowAttention): exclude the whole
    nn.MultiheadAttention subtree via qconfig=None instead. swap_quantizable_mha() is
    kept below (its weight-transfer math is independently correct, verified by
    demo()) but is NOT called from the QAT path for this reason -- do not wire it into
    prepare_qat_model()/build_qat_from_model() without first solving the
    custom-module/prepare_qat ordering problem above.

    A no-op for every pre-Phase-8 model (none contain LayerNorm, ShiftedWindowAttention,
    or MultiheadAttention), so this is safe to run unconditionally inside
    prepare_qat_model() below rather than needing a per-model call site.

    KNOWN UNVERIFIED RISK (found while wiring this up, not in ideas/PHASE8_PLAN.md's
    own D6 analysis): SwinTransformerBlock.forward() does
    `x = x + self.attn(self.norm1(x))` / `x = x + self.mlp(self.norm2(x))` with a bare
    Python `+`, not nn.quantized.FloatFunctional().add() (this project's own
    convention for every other residual add, per CLAUDE.md's QAT rules). norm2's MLP
    (fc1/fc2) is NOT excluded here -- D6 deliberately wants it quantized -- so after
    convert(), that `+` mixes an INT8 tensor (mlp output, exited through a real
    nn.quantized.Linear) with an FP32 tensor (the skip path, since norm2 is excluded).
    Quantized and float tensors cannot be added via bare `+` in eager-mode PyTorch.
    This is torchvision's own forward(), not something this project can patch without
    subclassing SwinTransformerBlock with explicit QuantStub/DeQuantStub boundaries
    around the residual -- untested here since nothing in this pass was executed.
    Run a convert()+forward() smoke test on a Swin-derived model before trusting any
    Phase 8 QAT/INT8 numbers; this may be the real blocker, not the weight-transfer
    risk Blocking Issue #1 already covers.
    """
    from torchvision.models.swin_transformer import ShiftedWindowAttention
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, ShiftedWindowAttention, nn.MultiheadAttention)):
            module.qconfig = None
    return model


class _BatchFirstMHAWrapper(nn.Module):
    """torch.ao.nn.quantizable.MultiheadAttention's batch_first=True path has a broken
    final reshape in torch==2.5.1 (verified directly: constructed with batch_first=True
    it diverges from an equal-weight nn.MultiheadAttention by ~0.7 max-abs-diff on random
    input; the identical module built with batch_first=False matches to 0.0). Always run
    the wrapped module seq-first and transpose at this module's boundary instead of
    trusting its own batch_first flag.
    """

    def __init__(self, qmha: nn.Module):
        super().__init__()
        self.qmha = qmha

    def forward(self, query, key, value, **kwargs):
        query, key, value = (t.transpose(0, 1) for t in (query, key, value))
        attn_output, attn_weights = self.qmha(query, key, value, **kwargs)
        return attn_output.transpose(0, 1), attn_weights


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

    Always constructs the quantizable module batch_first=False (see
    _BatchFirstMHAWrapper) and wraps it when the source module was batch_first=True
    (torchvision's ViT/DeiT EncoderBlock always is), rather than passing batch_first
    through directly.

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
                bias=child.in_proj_bias is not None, batch_first=False,
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
            replacement = _BatchFirstMHAWrapper(qmha) if child.batch_first else qmha
            setattr(model, name, replacement)
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
    """Assert-based self-checks for Phase 8's QAT-for-attention helpers. Not run
    automatically -- invoke directly (`python -m ml.quantization`)."""
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
        "weight transfer is broken"
    )
    print("swap_quantizable_mha: OK, pre/post-swap outputs match within atol=1e-5 "
          "(NOTE: correct in isolation, but not called from the QAT path -- see "
          "exclude_attention_from_qat's docstring for why)")

    ln = nn.LayerNorm(8)
    swa_model = nn.Module()
    from torchvision.models.swin_transformer import ShiftedWindowAttention
    swa = ShiftedWindowAttention(dim=8, window_size=[2, 2], shift_size=[0, 0], num_heads=2)
    mha2 = nn.MultiheadAttention(8, 2, batch_first=True)
    swa_model.add_module("norm", ln)
    swa_model.add_module("attn", swa)
    swa_model.add_module("self_attention", mha2)
    exclude_attention_from_qat(swa_model)
    assert ln.qconfig is None and swa.qconfig is None and mha2.qconfig is None, (
        "exclude_attention_from_qat did not set qconfig=None on every excluded type"
    )
    print("exclude_attention_from_qat: OK, LayerNorm/ShiftedWindowAttention/MultiheadAttention excluded")

    # Regression test for the custom-module/prepare_qat ordering bug this docstring
    # describes: build a tiny model containing a bare nn.MultiheadAttention and run it
    # through the real prepare_qat_model() -> tq.prepare_qat() path end-to-end. This
    # crashed (AttributeError: 'Linear' object has no attribute
    # 'activation_post_process') before exclude_attention_from_qat covered
    # nn.MultiheadAttention -- catch any regression here instead of discovering it
    # mid-training-run.
    class _TinyAttnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.fc = nn.Linear(embed_dim, 10)

        def forward(self, x):
            out, _ = self.attn(x, x, x, need_weights=False)
            return self.fc(out.mean(dim=1))

    tiny = _TinyAttnModel()
    qat_tiny = prepare_qat_model(tiny, fuse_pairs=[])
    qat_tiny.train()
    y = qat_tiny(torch.randn(2, seq_len, embed_dim))
    assert y.shape == (2, 10), f"unexpected output shape: {y.shape}"
    print("prepare_qat_model on an nn.MultiheadAttention-containing model: OK, "
          "forward pass succeeds (regression test for the custom-module/prepare_qat "
          "ordering bug)")


if __name__ == "__main__":
    demo()
