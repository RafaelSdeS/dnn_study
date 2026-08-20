"""Phase 8 — Efficient Vision Transformers & Hybrid Attention Architectures.

Thin wrappers around torchvision's VisionTransformer/SwinTransformer, custom-sized for
64x64 Tiny ImageNet-200 (~100K images) per ideas/PHASE8_PLAN.md D2/D3 — torchvision's
stock presets target 224x224 ImageNet-1k and would dwarf every Phase 1-4 model.
See ml/quantization.py's exclude_attention_from_qat/swap_quantizable_mha (D6) for how
these get QAT-prepared; LayerNorm and attention math can't go through this codebase's
fbgemm QAT path the way pure Conv-BN-ReLU models can.
"""

from functools import partial

import torch.ao.quantization as tq
import torch.nn as nn
from torchvision.models.swin_transformer import ShiftedWindowAttention, SwinTransformer, SwinTransformerBlock
from torchvision.models.vision_transformer import VisionTransformer

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


# ─── ViT-Tiny (H1 baseline / DeiT-Ti base) ─────────────────────────────────────

class ViTTiny(nn.Module):
    """Global-attention ViT sized for 64x64 (D3): patch_size=8 -> 8x8=64 patches + cls
    token = 65-token sequence. num_layers=6, half DeiT-Ti's 12 -- ViT overfits rather
    than underfits at this data scale, so depth is the first lever to cut (D3)."""

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.vit = VisionTransformer(
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


# ─── Swin-Pico (H1 window-size sweep) ──────────────────────────────────────────

class SwinPico(nn.Module):
    """2-stage windowed-attention Swin sized for 64x64 (D3). window_size in {2,4,8}
    sweeps H1 -- patch_size=4 -> 16x16 tokens -> one PatchMerging stage -> 8x8.
    window_size=8 is full/global attention at the 8x8 stage, H1's "unrestricted"
    sweep endpoint. attn_layer lets D5's pool-mixer variant reuse this class."""

    def __init__(self, num_classes: int = 200, window_size: int = 4, attn_layer=None):
        super().__init__()
        # D4: window_size must divide the 16x16 first-stage grid, else
        # shifted_window_attention silently pads instead of failing loudly.
        assert 16 % window_size == 0, f"window_size={window_size} must divide the 16x16 patch grid"
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        block = partial(_QuantizableSwinBlock, attn_layer=attn_layer) if attn_layer else _QuantizableSwinBlock
        self.swin = SwinTransformer(
            patch_size=[4, 4], embed_dim=48, depths=[2, 2], num_heads=[2, 4],
            window_size=[window_size, window_size], num_classes=num_classes,
            block=block,
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
    import torch

    ctors = {
        "vit_tiny": vit_tiny,
        "deit_tiny": deit_tiny,
        "swin_pico_w2": swin_pico_w2,
        "swin_pico_w4": swin_pico_w4,
        "swin_pico_w8": swin_pico_w8,
        "swin_pico_poolmixer": swin_pico_poolmixer,
        "hybrid_bottleneck_swin": hybrid_bottleneck_swin,
    }
    x = torch.randn(2, 3, 64, 64)
    for name, ctor in ctors.items():
        model = ctor(num_classes=200).eval()
        out = model(x)
        assert out.shape == (2, 200), f"{name}: expected (2, 200), got {tuple(out.shape)}"
        n_params = sum(p.numel() for p in model.parameters())
        assert 0.5e6 <= n_params <= 15e6, f"{name}: param count {n_params} outside sane 0.5M-15M bound"
        print(f"{name}: OK, {n_params / 1e6:.2f}M params")


if __name__ == "__main__":
    demo()
