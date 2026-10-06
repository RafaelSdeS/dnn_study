import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq

from .registry import MODEL_REGISTRY

# Tiny ImageNet input; keep_logits_float only runs it through the model to see which layer executes last
LOGITS_PROBE_SHAPE = (1, 3, 64, 64)

# INT8 as the literature defines it -- 8-bit per-tensor affine activations with EMA min/max ranges (Jacob et al.,
# CVPR 2018; affine costs activations nothing over scale quantization, Wu et al. 2020 Sec. 3.3) and 8-bit per-channel symmetric weights in [-127, 127] (Wu et al. 2020, Sec. 6 / LiteRT int8 spec).
# PyTorch's onednn QAT qconfig is exactly the first part; fbgemm's default instead sets reduce_range=True, i.e. 7-bit
# activations (0..127), to dodge int16 saturation on CPUs without VNNI -- what every run used until 2026-10-03.
_ONEDNN_QAT = tq.get_default_qat_qconfig("onednn")
INT8_QAT_QCONFIG = tq.QConfig(activation=_ONEDNN_QAT.activation, weight=_ONEDNN_QAT.weight.with_args(quant_min=-127))
# Kernels every INT8 *accuracy* comes from: QNNPACK accumulates u8*s8 products exactly in int32 on any CPU (Jacob et al.
# 2018's arithmetic). oneDNN does that only with VNNI; without it (beagle, tupi1/2, the laptop) VPMADDUBSW sums u8*s8
# pairs in int16 and saturates -- the user's job to avoid, per oneDNN's dev guide ("Nuances of int8 computations";
# pytorch/pytorch#103646) -- which left the first conv (input zero point ~114) wrong in 7-13% of its outputs and INT8
# 1-5pp under QAT (docs/logs/PHASE11_LOG.md, "INT8 accuracy engine").
ACCURACY_ENGINE = "qnnpack"
# x86-optimized kernels for INT8 *latency* only: the saturation above changes their results, not their speed.
QUANT_ENGINE = "onednn"
# Written into every run summary; analysis treats a QAT/INT8 number without the current value as superseded.
# 2026-10-03: the INT8 definition above, fused Linear/add + ReLU, W8 logits layer, Wu et al. 2020's QAT schedule --
# and, for Phase 11, the new FP32 recipe that landed the same day, so analyze_geometry.load drops FP32 without it too.
# 2026-10-06: INT8 accuracy on ACCURACY_ENGINE; nothing retrained, a 2026-10-03 run is re-evaluated by resubmitting it.
QUANT_PROTOCOL = "2026-10-06"


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

    KNOWN UNVERIFIED RISK (found while wiring this up, not in docs/plans/PHASE8_PLAN.md's
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
    time (docs/plans/PHASE8_PLAN.md Task 3 Blocking Issue #1) -- a naive
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
    float_logits: bool = False,
) -> nn.Module:
    """Deep-copy model, fuse Conv-BN(-ReLU) pairs, insert fake-quant observers (INT8_QAT_QCONFIG).

    float_logits gives the logits Linear an FP32 output (keep_logits_float); build_qat_from_model, the path every real
    run takes, always sets it. Off by default only so toy modules without a logits layer still prepare.

    After the registered fuse_pairs, every Conv/Linear still directly followed by its ReLU inside an nn.Sequential
    is fused too (fuse_sequential_relus) -- the activation is folded into the layer that feeds it, as Jacob et al.
    2018 and every INT8 runtime do. That covers the FC heads (Linear-ReLU), which only vgg16 had fused until
    2026-10-03: unfused, the observer sat on the pre-ReLU output, spending half the 8-bit range on negatives that
    ReLU then drops (for vgg16 a p999 ~27k tail that collapsed QAT outright, docs/logs/PHASE11_LOG.md).

    fuse_root is a submodule of the *input* model; it is re-located inside the deep copy by name.
    Until 2026-09-30 the fusion ran on the caller's fuse_root itself, i.e. on the original model,
    so every registry entry with fuse_root_attr trained QAT with no Conv-(BN-)ReLU fusion at all
    (observer before the ReLU, BN left unfolded) -- see docs/logs/PHASE11_LOG.md, "QAT fusion bug".

    Every AvgPool2d/AdaptiveAvgPool2d is wrapped as DeQuantStub -> pool -> QuantStub (see
    requantize_avg_pools).
    """
    root_name = "" if fuse_root is None else next(n for n, m in model.named_modules() if m is fuse_root)
    model = copy.deepcopy(model)
    model.train()
    model.qconfig = INT8_QAT_QCONFIG
    exclude_attention_from_qat(model)
    root = model.get_submodule(root_name)
    if fuse_pairs:
        tq.fuse_modules_qat(root, fuse_pairs, inplace=True)
    fuse_sequential_relus(model)
    requantize_avg_pools(model)
    if float_logits:
        keep_logits_float(model, torch.zeros(LOGITS_PROBE_SHAPE, device=next(model.parameters()).device))
    return tq.prepare_qat(model, inplace=False)


def fuse_sequential_relus(model: nn.Module) -> nn.Module:
    """Fuse every run of adjacent [Conv2d, BatchNorm2d(, ReLU)] or [Conv2d|Linear, ReLU] children of a plain
    nn.Sequential, in place -- whatever a registry fuse_map left unfused. Only plain Sequentials, where child order is
    execution order; fused intrinsic modules subclass Sequential, hence the exact type check."""
    for seq in [m for m in model.modules() if type(m) is nn.Sequential]:
        names, mods = list(seq._modules), list(seq._modules.values())
        groups, i = [], 0
        while i < len(mods):
            nxt = mods[i + 1] if i + 1 < len(mods) else None
            if isinstance(mods[i], nn.Conv2d) and isinstance(nxt, nn.BatchNorm2d):
                n = 3 if i + 2 < len(mods) and isinstance(mods[i + 2], nn.ReLU) else 2
            elif isinstance(mods[i], (nn.Conv2d, nn.Linear)) and isinstance(nxt, nn.ReLU):
                n = 2
            else:
                i += 1
                continue
            groups.append(names[i:i + n])
            i += n
        if groups:
            tq.fuse_modules_qat(seq, groups, inplace=True)
    return model


def requantize_avg_pools(model: nn.Module) -> nn.Module:
    """Replace every AvgPool2d/AdaptiveAvgPool2d with DeQuantStub -> pool -> QuantStub, in place.

    Eager-mode quantized avg pooling keeps its INPUT's scale, and QAT puts no observer after the pool, so
    the fake-quant model never sees that rounding. A GAP averages a ReLU map whose per-tensor scale is set
    by rare peaks (alexnet_3x3_gap: scale 1.13 from a max of ~144), so most channel means fall below one
    step and round to 0 -- the gate's fused alexnet_3x3_gap lost 2.4pp fake-quant -> INT8 (46.55 -> 44.21)
    to this alone; float pool + a fitted requant scale recovered 46.45 (docs/logs/PHASE11_LOG.md, "Quantized
    GAP"). The pool now runs in float and its output gets its own observer, i.e. INT8 hardware's int32
    accumulate + requantize: the quantizer sits at the input of the next compute layer, Wu et al. 2020's placement (Sec. 4)
    (ONNX Runtime's QLinear(Global)AveragePool likewise takes its own y_scale). Functional pooling (torchvision's
    quantizable MobileNetV2) is not covered.
    """
    for parent in list(model.modules()):
        for name, child in parent.named_children():
            if isinstance(child, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                setattr(parent, name, _RequantizedPool(child))
    return model


class _RequantizedPool(nn.Module):
    """DeQuantStub -> pool -> QuantStub. After convert, a pool that receives a float tensor sits in a float region
    (Phase 8's attention-excluded heads, which re-quantize with their own stub) and stays a plain float pool."""

    def __init__(self, pool: nn.Module):
        super().__init__()
        self.dequant, self.pool, self.quant = tq.DeQuantStub(), pool, tq.QuantStub()

    def forward(self, x):
        if x.is_quantized or isinstance(self.quant, tq.QuantStub):  # INT8 quantized region, or QAT/float
            return self.quant(self.pool(self.dequant(x)))
        return self.pool(x)


def keep_logits_float(model: nn.Module, probe: torch.Tensor) -> nn.Module:
    """Replace the logits Linear with _FloatLogits (INT8 input and weights, FP32 output), in place.

    Its output used to get the same 8-bit fake-quant as every activation, so QAT/INT8 logits sat on a grid of 8-31
    distinct values per image: 14-31% of val images tied for top-1, top-5 depended on the tie-break, and FC heads lost
    ~0.4pp top-1 to the grid alone (docs/logs/PHASE11_LOG.md, "Float logits layer"). Wu et al. 2020 (Sec. 4) quantize a layer's
    inputs and weights and requantize its int32 accumulator only where another quantized layer reads it -- the logits
    have no such reader. The logits layer is the last Conv/Linear to run on `probe` -- execution order, not
    registration order (quantizable ResNet18 registers its input QuantStub after fc). Any other head shape fails here.
    """
    order = []
    hooks = [m.register_forward_hook(lambda mod, i, o: order.append(mod)) for m in model.modules() if not list(m.children())]
    was_training = model.training
    with torch.no_grad():
        model.eval()(probe)
    model.train(was_training)
    for h in hooks:
        h.remove()
    i = max(k for k, m in enumerate(order) if isinstance(m, (nn.Linear, nn.Conv2d)))
    tail = [type(m).__name__ for m in order[i + 1:]]
    assert isinstance(order[i], nn.Linear) and all(isinstance(m, tq.DeQuantStub) for m in order[i + 1:]), \
        f"unsupported logits layer: {type(order[i]).__name__} followed by {tail}"
    name = next(n for n, m in model.named_modules() if m is order[i])
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent), child, _FloatLogits(order[i]))
    return model


class _FloatLogits(nn.Module):
    """DeQuantStub -> the logits Linear with INT8 weights and an FP32 output. The Linear has qconfig=None, so
    prepare_qat/convert leave the module itself alone; its input arrives dequantized from the previous layer's 8-bit
    grid, and its weight goes through INT8_QAT_QCONFIG's per-channel fake-quant (frozen with every other observer).
    freeze(), run by convert_to_int8, keeps that weight as int8 codes + the fake-quant's scales, so the INT8 model
    holds no FP32 weight copy and computes exactly what the fake-quant did."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        linear.qconfig = None
        self.dequant, self.linear = tq.DeQuantStub(), linear
        self.weight_fake_quant = INT8_QAT_QCONFIG.weight().to(linear.weight.device)  # prepare_qat needs one device
        self.weight_fake_quant.qconfig = None
        self.register_buffer("qweight", None)

    def forward(self, x):
        if self.qweight is None:
            w = self.weight_fake_quant(self.linear.weight)
        else:  # symmetric: zero_point is 0
            w = self.qweight.float() * self.weight_fake_quant.scale[:, None]
        return F.linear(self.dequant(x), w, self.linear.bias)

    def freeze(self) -> None:
        """int8 codes on the fake-quant's frozen grid. Calibrates it first only if it never ran: load_int8_model's
        freshly built model, whose codes and scales the loaded state_dict then overwrites."""
        fq, w = self.weight_fake_quant, self.linear.weight.detach()
        if fq.scale.numel() != w.size(0):
            fq.enable_observer()
            fq(w)
        q = torch.fake_quantize_per_channel_affine(w, fq.scale, fq.zero_point, fq.ch_axis, fq.quant_min, fq.quant_max)
        self.qweight = torch.round(q / fq.scale[:, None]).to(torch.int8)
        self.linear.weight = None


def build_qat_from_model(model: nn.Module, arch_name: str, device: torch.device) -> nn.Module:
    """Apply QAT preparation to a pre-loaded FP32 model."""
    spec = MODEL_REGISTRY[arch_name]
    root_attr = spec.get("fuse_root_attr")
    fuse_root = getattr(model, root_attr) if root_attr else None
    return prepare_qat_model(model, spec["fuse_map"], fuse_root=fuse_root, float_logits=True).to(device)


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


def convert_to_int8(qat_model: nn.Module, inplace: bool = False, engine: str = ACCURACY_ENGINE) -> nn.Module:
    """Convert a trained QAT model to real INT8 ops (CPU-only), on `engine`'s kernels: ACCURACY_ENGINE (exact) unless
    only latency is measured (QUANT_ENGINE). The engine is process-global for some ops (add, pooling), so run the model
    before converting another on a different engine. The logits layer is frozen first: convert() strips every
    fake-quant's observer, which an uncalibrated one still needs."""
    torch.backends.quantized.engine = engine
    model = (qat_model if inplace else copy.deepcopy(qat_model)).to("cpu").eval()
    for m in model.modules():
        if isinstance(m, _FloatLogits):
            m.freeze()
    return torch.ao.quantization.convert(model, inplace=True)


def load_int8_model(arch_name: str, save_dir: str | Path, engine: str = ACCURACY_ENGINE) -> nn.Module:
    """Rebuild the INT8 model scripts/train.py saved as a state_dict (qat_<arch>.pth): same QAT graph -> convert ->
    load, on any engine (the state_dict holds unpacked int8 weights). Until 2026-09-30 it saved the pickled module
    instead, which can't be loaded back at all -- quantized convs don't unpickle their nn.Module internals."""
    model = convert_to_int8(build_qat_from_model(MODEL_REGISTRY[arch_name]["ctor"](), arch_name, torch.device("cpu")),
                            engine=engine)
    model.load_state_dict(torch.load(Path(save_dir) / f"qat_{arch_name}.pth", map_location="cpu", weights_only=True))
    return model


def make_qat_callback(freeze_bn_epoch: int = 3, disable_observer_epoch: int | None = 5):
    """Return an epoch_callback that freezes BN stats then disables observers (never, if
    disable_observer_epoch is None)."""
    # >= (idempotent), not ==: a run resumed past either epoch must re-apply it -- freeze_bn is a
    # plain module attribute, not state_dict, so the resumed model would train with BN unfrozen
    def cb(epoch: int, model: nn.Module) -> None:
        if epoch >= freeze_bn_epoch:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        if disable_observer_epoch is not None and epoch >= disable_observer_epoch:
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
