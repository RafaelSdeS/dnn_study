"""Phase 8 — Efficient Vision Transformers & Hybrid Attention Architectures.

Thin wrappers around torchvision's VisionTransformer/SwinTransformer, custom-sized for
64x64 Tiny ImageNet-200 (~100K images) per ideas/PHASE8_PLAN.md D2/D3 — torchvision's
stock presets target 224x224 ImageNet-1k and would dwarf every Phase 1-4 model.
See ml/quantization.py's exclude_attention_from_qat/swap_quantizable_mha (D6) for how
these get QAT-prepared; LayerNorm and attention math can't go through this codebase's
fbgemm QAT path the way pure Conv-BN-ReLU models can.
"""

from collections import OrderedDict
from functools import partial

import torch
import torch.ao.quantization as tq
import torch.nn as nn
from torchvision.models.swin_transformer import (
    PatchMerging,
    ShiftedWindowAttention,
    SwinTransformer,
    SwinTransformerBlock,
    _patch_merging_pad,
)
from torchvision.models.vision_transformer import ConvStemConfig, Encoder, EncoderBlock, VisionTransformer

from .compensation import _AlexBottleneck, _float_functional


# ─── QAT-safe Swin block (D6 follow-up) ────────────────────────────────────────
#
# torchvision's SwinTransformerBlock.forward() does `x = x + self.attn(self.norm1(x))`
# / `x = x + self.mlp(self.norm2(x))` with a bare Python `+`. Under this project's QAT
# scheme, norm1/attn are excluded (qconfig=None, D6) so their branch stays FP32, but
# mlp's Linears are NOT excluded (D6 wants them quantized) -- after convert(), the mlp
# branch's output would be a real INT8 tensor while the skip path stays FP32. Bare `+`
# (and this project's own FloatFunctional convention, CLAUDE.md's QAT rules) both
# require same-dtype operands; eager-mode PyTorch has no implicit float<->quantized
# coercion. torchvision's MLP also sandwiches nn.GELU (no quantized-eager kernel)
# between fc1/fc2, so simply bracketing the whole mlp call isn't enough either --
# each Linear needs its own Quant/Dequant boundary, GELU runs in float between them.
# This mirrors exactly how torch.ao.nn.quantizable.MultiheadAttention brackets its own
# FP32-only softmax core with quant_attn_output/dequant_q/k/v (see
# ml/quantization.py's swap_quantizable_mha).

class _QuantizableMLP(nn.Module):
    """Linear-GELU-Linear with each Linear individually Quant/DeQuant-bracketed."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, in_dim)
        self.drop2 = nn.Dropout(dropout)
        self.quant1 = tq.QuantStub()
        self.dequant1 = tq.DeQuantStub()
        self.quant2 = tq.QuantStub()
        self.dequant2 = tq.DeQuantStub()

        for m in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = self.quant1(x)
        x = self.fc1(x)
        x = self.dequant1(x)
        x = self.drop1(self.act(x))
        x = self.quant2(x)
        x = self.fc2(x)
        x = self.dequant2(x)
        return self.drop2(x)


class _QuantizableSwinBlock(SwinTransformerBlock):
    """SwinTransformerBlock with FloatFunctional residual adds and a
    Quant/Dequant-bracketed MLP (see module docstring above).

    Both operands of both residual adds are FP32 by construction here: norm1/attn are
    excluded (D6), and the mlp branch re-dequantizes to FP32 internally via
    _QuantizableMLP -- so skip_add_attn/skip_add_mlp are excluded too (qconfig=None),
    keeping them plain float adds after convert() rather than QFunctional's real-int8
    add, which requires both inputs already quantized.
    """

    def __init__(
        self, dim, num_heads, window_size, shift_size, mlp_ratio: float = 4.0,
        dropout: float = 0.0, attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0, norm_layer=nn.LayerNorm,
        attn_layer=ShiftedWindowAttention,
    ):
        super().__init__(
            dim, num_heads, window_size, shift_size, mlp_ratio=mlp_ratio,
            dropout=dropout, attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob, norm_layer=norm_layer,
            attn_layer=attn_layer,
        )
        self.mlp = _QuantizableMLP(dim, int(dim * mlp_ratio), dropout)
        self.skip_add_attn = _float_functional()
        self.skip_add_mlp = _float_functional()
        self.skip_add_attn.qconfig = None
        self.skip_add_mlp.qconfig = None

    def forward(self, x):
        x = self.skip_add_attn.add(x, self.stochastic_depth(self.attn(self.norm1(x))))
        x = self.skip_add_mlp.add(x, self.stochastic_depth(self.mlp(self.norm2(x))))
        return x


class _QuantizablePatchMerging(PatchMerging):
    """PatchMerging with the norm->reduction boundary bracketed: norm stays excluded
    (LayerNorm, D6) but reduction is a plain Linear that stays quantized -- same root
    cause/fix as _QuantizableMLP above."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__(dim, norm_layer=norm_layer)
        self.quant_reduction = tq.QuantStub()
        self.dequant_reduction = tq.DeQuantStub()

    def forward(self, x):
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.quant_reduction(x)
        x = self.reduction(x)
        return self.dequant_reduction(x)


class _QuantizableSwinTransformer(SwinTransformer):
    """SwinTransformer with its two remaining boundary crossings bracketed: patch
    embedding's quantized Conv2d feeding directly into the excluded LayerNorm right
    after it, and the excluded final norm feeding directly into the quantized head
    Linear. PatchMerging's own crossing is handled by defaulting downsample_layer to
    _QuantizablePatchMerging above."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("downsample_layer", _QuantizablePatchMerging)
        super().__init__(*args, **kwargs)
        conv, permute, norm = self.features[0]
        self.features[0] = nn.Sequential(conv, permute, tq.DeQuantStub(), norm)
        self.quant_head = tq.QuantStub()

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.quant_head(x)
        return self.head(x)


# ─── QAT-safe ViT encoder (D6 follow-up, same root cause as the Swin fix above) ─
#
# EncoderBlock.forward() does `x = x + input` / `x = x_after_mlp + x`, and
# Encoder.forward() does `input = input + self.pos_embedding` -- all bare `+`.
# ln_1/ln_2 are excluded (LayerNorm, D6); self_attention's Linears (once swapped to
# torch.ao.nn.quantizable.MultiheadAttention via ml/quantization.py's
# swap_quantizable_mha, D6/Task 3) and mlp's Linears stay quantized -- the exact
# same FP32-vs-INT8 residual mismatch as Swin's blocks, plus VisionTransformer.
# forward()'s `torch.cat([class_token, conv_proj_output])` mixes the always-FP32
# class_token parameter with conv_proj's quantized output. Neither Encoder nor
# VisionTransformer exposes a pluggable block class the way SwinTransformer does
# (no `block=` constructor arg), so this reimplements their __init__/forward
# instead of subclassing lightly -- safe since Phase 8 trains from scratch (D3),
# so there's no pretrained state dict this fresh reconstruction could lose.

class _QuantizableEncoderBlock(EncoderBlock):
    """EncoderBlock with FloatFunctional residual adds and a quantized `mlp` branch.

    self_attention stays plain nn.MultiheadAttention, entirely FP32 (D6, revised):
    torch.ao.nn.quantizable.MultiheadAttention's linear_Q/K/V/out_proj can only become
    real nn.quantized.Linear via the static-PTQ prepare()->calibrate->convert() flow
    (its from_float()/from_observed() classmethods) -- this codebase's single-call
    tq.prepare_qat() cannot drive that (verified directly: PyTorch registers it in
    `observed_to_quantized_custom_module_class`, so prepare_qat()'s first-phase
    convert() calls its `.from_observed()` before any observer exists ->
    AttributeError). Excluding it via qconfig=None (exclude_attention_from_qat) avoids
    that crash but then also skips recursing into its own children during
    prepare/convert, so its internal Linears never become quantized either --
    swapping in QuantizableMHA bought nothing under this constraint, so this block no
    longer brackets self_attention with Quant/DeQuant stubs (that bracketing assumed
    a quantized-input-capable MHA, which stock nn.MultiheadAttention is not -- feeding
    it a real quantized tensor after convert() raises NotImplementedError:
    aten::mm.out on 'QuantizedCPU', confirmed directly). self_attention is treated
    like ln_1/ln_2: a plain excluded FP32 submodule, no stub needed either side.
    """

    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
                 norm_layer=None):
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        super().__init__(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
        self.mlp = _QuantizableMLP(hidden_dim, mlp_dim, dropout)
        self.skip_add_attn = _float_functional()
        self.skip_add_mlp = _float_functional()
        self.skip_add_attn.qconfig = None
        self.skip_add_mlp.qconfig = None

    def forward(self, input):
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = self.skip_add_attn.add(x, input)

        y = self.mlp(self.ln_2(x))
        return self.skip_add_mlp.add(x, y)


class _QuantizableEncoder(Encoder):
    """Encoder with a FloatFunctional pos_embedding add (both operands FP32 by
    construction -- see _QuantizableVisionTransformer's dequant_patches below) and
    _QuantizableEncoderBlock layers instead of stock EncoderBlock. Bypasses
    Encoder.__init__ (it hardcodes EncoderBlock) rather than calling super().__init__().
    """

    def __init__(self, seq_length, num_layers, num_heads, hidden_dim, mlp_dim,
                 dropout, attention_dropout, norm_layer=None):
        nn.Module.__init__(self)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = _QuantizableEncoderBlock(
                num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)
        self.skip_add_pos = _float_functional()
        self.skip_add_pos.qconfig = None

    def forward(self, input):
        input = self.skip_add_pos.add(input, self.pos_embedding)
        return self.ln(self.layers(self.dropout(input)))


class _QuantizableVisionTransformer(VisionTransformer):
    """VisionTransformer with the class_token concatenation boundary bracketed
    (conv_proj's quantized output meeting the always-FP32 class_token parameter)
    and _QuantizableEncoder swapped in for the stock Encoder."""

    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim,
                 mlp_dim, dropout: float = 0.0, attention_dropout: float = 0.0,
                 num_classes: int = 1000, representation_size=None, norm_layer=None,
                 conv_stem_configs=None):
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        super().__init__(
            image_size=image_size, patch_size=patch_size, num_layers=num_layers,
            num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim,
            dropout=dropout, attention_dropout=attention_dropout,
            num_classes=num_classes, representation_size=representation_size,
            norm_layer=norm_layer, conv_stem_configs=conv_stem_configs,
        )
        self.dequant_patches = tq.DeQuantStub()
        self.quant_head = tq.QuantStub()
        self.encoder = _QuantizableEncoder(
            self.seq_length, num_layers, num_heads, hidden_dim, mlp_dim,
            dropout, attention_dropout, norm_layer,
        )

    def forward(self, x):
        x = self._process_input(x)
        x = self.dequant_patches(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = x[:, 0]
        x = self.quant_head(x)
        return self.heads(x)


# ─── ViT-Tiny (H1 baseline / DeiT-Ti base) ─────────────────────────────────────

class ViTTiny(nn.Module):
    """Global-attention ViT sized for 64x64 (D3): patch_size=8 -> 8x8=64 patches + cls
    token = 65-token sequence. num_layers=6, half DeiT-Ti's 12 -- ViT overfits rather
    than underfits at this data scale, so depth is the first lever to cut (D3)."""

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.vit = _QuantizableVisionTransformer(
            image_size=64, patch_size=8, num_layers=6, num_heads=3,
            hidden_dim=192, mlp_dim=768, num_classes=num_classes,
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.vit(x)
        x = self.dequant(x)
        return x


def vit_tiny(num_classes: int = 200) -> nn.Module:
    return ViTTiny(num_classes=num_classes)


# deit_tiny (H4) shares vit_tiny's architecture exactly; only the training loop
# (DistillationTrainer, Task 4) differs, so it reuses this constructor rather than a
# duplicate class -- the two registry entries just point at the same ctor.
deit_tiny = vit_tiny


# ─── ViT with a Winograd-eligible 3x3 conv stem ────────────────────────────────
#
# Investigation finding: no attention model in this repo has a real Winograd target.
# ViTTiny/SwinPico patchify with a single big-kernel conv (8x8 / 4x4, any stride is
# Winograd-irrelevant for a non-overlapping patchify anyway); hybrid_bottleneck_swin's
# _AlexBottleneck stem has 3x3 convs but both blocks run stride=2 -- F(2x2,3x3) only
# triggers on stride=1, groups=1 convs (ml/profiling.py's winograd_conv2d_f23), so even
# that "3x3" stem is Winograd-ineligible. This is the first one with a genuine dense
# stride=1 3x3 conv in the pipeline.

def _dense_relu(inplace: bool = True) -> nn.Module:
    """torchvision's Conv2dNormActivation always calls activation_layer(inplace=True);
    this project's QAT rule (CLAUDE.md) requires every ReLU inplace=False."""
    return nn.ReLU(inplace=False)


# Xiao et al. 2021 ("Early Convolutions Help Transformers See Better") stem, restricted
# to 3x3 kernels. Stride-2-first-then-dense mirrors AlexNet3x3FC's own stem
# (models/alexnet_variants.py): a stride=2 3x3 downsamples first, then stride=1 3x3s
# run dense -- only the middle layer here (32x32, stride=1) is Winograd-eligible; the
# other three are downsampling (stride=2) and, like the patchify conv they replace,
# outside Winograd's applicability. Reaches the same 8x8 grid as ViTTiny's patch_size=8.
_CONV_STEM_3X3 = [
    ConvStemConfig(out_channels=32, kernel_size=3, stride=2, activation_layer=_dense_relu),   # 64->32
    ConvStemConfig(out_channels=64, kernel_size=3, stride=1, activation_layer=_dense_relu),   # 32->32, dense: Winograd-eligible
    ConvStemConfig(out_channels=64, kernel_size=3, stride=2, activation_layer=_dense_relu),   # 32->16
    ConvStemConfig(out_channels=192, kernel_size=3, stride=2, activation_layer=_dense_relu),  # 16->8
]


class ViTTinyConvStem(nn.Module):
    """ViTTiny with its 8x8 non-overlapping patchify replaced by _CONV_STEM_3X3. Same
    encoder as ViTTiny (num_layers=6, num_heads=3, hidden_dim=192, mlp_dim=768) and the
    same 8x8 token grid -- patchify method is the only variable that changes."""

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.vit = _QuantizableVisionTransformer(
            image_size=64, patch_size=8, num_layers=6, num_heads=3,
            hidden_dim=192, mlp_dim=768, num_classes=num_classes,
            conv_stem_configs=_CONV_STEM_3X3,
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.vit(x)
        x = self.dequant(x)
        return x


def vit_tiny_convstem(num_classes: int = 200) -> nn.Module:
    return ViTTinyConvStem(num_classes=num_classes)


# ─── Swin-Pico (H1 window-size sweep) ──────────────────────────────────────────

class SwinPico(nn.Module):
    """2-stage windowed-attention Swin sized for 64x64 (D3). window_size in {2,4,8}
    sweeps H1 -- patch_size=4 -> 16x16 tokens -> one PatchMerging stage -> 8x8.
    window_size=8 is full/global attention at the 8x8 stage, H1's "unrestricted"
    sweep endpoint. attn_layer lets D5's pool-mixer variant reuse this class."""

    def __init__(self, num_classes: int = 200, window_size: int = 4, attn_layer=None, conv_stem: bool = False):
        super().__init__()
        # D4: window_size must divide the 16x16 first-stage grid, else
        # shifted_window_attention silently pads instead of failing loudly.
        assert 16 % window_size == 0, f"window_size={window_size} must divide the 16x16 patch grid"
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        block = partial(_QuantizableSwinBlock, attn_layer=attn_layer) if attn_layer else _QuantizableSwinBlock
        self.swin = _QuantizableSwinTransformer(
            patch_size=[4, 4], embed_dim=48, depths=[2, 2], num_heads=[2, 4],
            window_size=[window_size, window_size], num_classes=num_classes,
            block=block,
        )
        if conv_stem:
            # SwinTransformer has no conv_stem_configs hook (unlike VisionTransformer) --
            # replace the single 4x4-kernel patchify conv in-place with a 3x3-restricted
            # stack, same stride-2-first-then-dense pattern as _CONV_STEM_3X3 above.
            # Output stays (B, embed_dim, 16, 16), so permute/DeQuantStub/norm right
            # after it (see _QuantizableSwinTransformer.__init__) need no changes.
            embed_dim = 48
            self.swin.features[0][0] = nn.Sequential(
                nn.Conv2d(3, embed_dim, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=False),          # 64->32
                nn.Conv2d(embed_dim, embed_dim, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=False),          # 32->32, dense: Winograd-eligible
                nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=False),          # 32->16
            )

    def forward(self, x):
        x = self.quant(x)
        x = self.swin(x)
        x = self.dequant(x)
        return x


def swin_pico(num_classes: int = 200, window_size: int = 4) -> nn.Module:
    return SwinPico(num_classes=num_classes, window_size=window_size)


def swin_pico_w2(num_classes: int = 200) -> nn.Module:
    return swin_pico(num_classes=num_classes, window_size=2)


def swin_pico_w4(num_classes: int = 200) -> nn.Module:
    return swin_pico(num_classes=num_classes, window_size=4)


def swin_pico_w8(num_classes: int = 200) -> nn.Module:
    return swin_pico(num_classes=num_classes, window_size=8)


def swin_pico_convstem(num_classes: int = 200, window_size: int = 4) -> nn.Module:
    return SwinPico(num_classes=num_classes, window_size=window_size, conv_stem=True)


# ─── Lightweight-attention Swin-Pico (D5 / H5 cross-check) ────────────────────

class _PoolMixer(nn.Module):
    """Parameter-free 3x3 average-pool token-mixer, MetaFormer/PoolFormer-style (D5).

    Drop-in replacement for ShiftedWindowAttention at SwinTransformerBlock's attn_layer
    slot -- same call signature (dim, window_size, shift_size, num_heads, plus
    attention_dropout/dropout kwargs, per SwinTransformerBlock.__init__'s
    `self.attn = attn_layer(dim, window_size, shift_size, num_heads, ...)` call) -- but
    ignores every attention-specific arg since pooling has no windows or heads. Uses
    the exact op category (depthwise/parameter-free spatial mixing) Phase 6's H2 already
    profiled as non-Winograd-eligible, making H5's prediction for this variant a direct
    extension of Phase 6 data rather than an unanchored claim (D5).
    """

    def __init__(self, dim, window_size, shift_size, num_heads, attention_dropout=0.0, dropout=0.0):
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        # x: (B, H, W, C) -- SwinTransformerBlock's channel-last convention
        x = x.permute(0, 3, 1, 2)
        x = self.pool(x)
        return x.permute(0, 2, 3, 1)


def swin_pico_poolmixer(num_classes: int = 200, window_size: int = 4) -> nn.Module:
    return SwinPico(num_classes=num_classes, window_size=window_size, attn_layer=_PoolMixer)


# ─── Hybrid: Bottleneck CNN stem + windowed-attention stages (H2) ──────────────

class HybridBottleneckSwin(nn.Module):
    """_AlexBottleneck stem (Phase 3) -> 8x8 grid -> 2 windowed-attention stages (H2).

    Combines convolution's cheap local-pattern extraction (early downsampling) with
    attention's long-range mixing (late stages). The CNN stem already spatially
    downsamples to 8x8, so each spatial location becomes one token directly (channel
    dim = hidden_dim) -- no separate patch embedding needed. window_size=4 is the H1
    sweep's presumed sweet spot; provisional until H1 completes (Task 1 pitfalls).
    """

    def __init__(self, num_classes: int = 200, hidden_dim: int = 96, window_size: int = 4):
        super().__init__()
        assert 8 % window_size == 0, f"window_size={window_size} must divide the 8x8 stem output grid"
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            _AlexBottleneck(3, 32, stride=2),           # 64 -> 32
            _AlexBottleneck(32, hidden_dim, stride=2),  # 32 -> 16
            nn.MaxPool2d(2),                            # 16 -> 8
        )  # 64x64 -> 8x8, hidden_dim channels (2 stride-2 blocks + 1 maxpool = 8x reduction)
        # stem is quantized (plain Conv-BN-ReLU) but self.blocks[0]'s norm1 is excluded
        # (D6) and needs FP32 input -- same boundary-crossing problem as
        # _QuantizableSwinBlock's MLP, bracketed the same way: dequant right after the
        # last quantized op, quant right before the next one (quant_head, below).
        self.dequant_stem = tq.DeQuantStub()
        shift = window_size // 2
        self.blocks = nn.Sequential(
            _QuantizableSwinBlock(hidden_dim, num_heads=4, window_size=[window_size, window_size], shift_size=[0, 0]),
            _QuantizableSwinBlock(hidden_dim, num_heads=4, window_size=[window_size, window_size], shift_size=[shift, shift]),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # norm is excluded (LayerNorm, D6) but head is a plain quantized Linear -- again
        # the same boundary; self.dequant (already model-final) handles the far side.
        self.quant_head = tq.QuantStub()
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)             # (B, C, 8, 8), quantized
        x = self.dequant_stem(x)     # -> FP32 for the excluded norm1 ahead
        x = x.permute(0, 2, 3, 1)    # (B, 8, 8, C) -- SwinTransformerBlock's channel-last convention
        x = self.blocks(x)           # FP32 throughout (norm1/norm2 excluded, mlp/attn re-bracket internally)
        x = self.norm(x)             # FP32 (excluded)
        x = x.mean(dim=(1, 2))       # FP32, global average pool over tokens
        x = self.quant_head(x)       # -> quantized for the classifier head
        x = self.head(x)
        x = self.dequant(x)
        return x


def hybrid_bottleneck_swin(num_classes: int = 200) -> nn.Module:
    return HybridBottleneckSwin(num_classes=num_classes)


def demo() -> None:
    """Assert-based self-check (Task 1 validation): shape + sane param-count bounds
    for every constructor, incl. the window-size sweep. Not run automatically --
    invoke directly (`python -m models.vit_variants`) since this repo's Trainer/QAT
    pipeline is CPU-heavy and this file's job is architecture wiring, not execution."""
    ctors = {
        "vit_tiny": vit_tiny,
        "deit_tiny": deit_tiny,
        "vit_tiny_convstem": vit_tiny_convstem,
        "swin_pico_w2": swin_pico_w2,
        "swin_pico_w4": swin_pico_w4,
        "swin_pico_w8": swin_pico_w8,
        "swin_pico_poolmixer": swin_pico_poolmixer,
        "swin_pico_convstem": swin_pico_convstem,
        "hybrid_bottleneck_swin": hybrid_bottleneck_swin,
    }
    x = torch.randn(2, 3, 64, 64)
    for name, ctor in ctors.items():
        model = ctor(num_classes=200).eval()
        out = model(x)
        assert out.shape == (2, 200), f"{name}: expected (2, 200), got {tuple(out.shape)}"
        n_params = sum(p.numel() for p in model.parameters())
        # ponytail: 0.2M floor, not 0.5M -- swin_pico_w2/w4/w8 (~0.32M) and
        # swin_pico_poolmixer (~0.23M, parameter-free pooling has no QKV/proj weights)
        # are legitimately this small; bound only needs to catch a gross
        # hidden_dim/num_layers typo, not gatekeep an already-validated model.
        assert 0.2e6 <= n_params <= 15e6, f"{name}: param count {n_params} outside sane 0.2M-15M bound"
        print(f"{name}: OK, {n_params / 1e6:.2f}M params")


if __name__ == "__main__":
    demo()
