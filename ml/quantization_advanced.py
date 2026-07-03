"""Phase 4.1 — aggressive low-bit quantization (fake-quant simulation).

fbgemm has no true INT4 kernel, so INT4/mixed-precision here are simulated with
FakeQuantize modules: weights/activations are fake-quantized in the forward pass
(range-restricted to the target bit-width). Accuracy is faithful; reported model
size is the *theoretical* packed size (params × bits / 8), computed by
`theoretical_size_mb`. Evaluation runs on GPU — no CPU-only convert step.

Public API:
    make_qconfig(w_bits, a_bits, per_channel)      → QConfig with bit-restricted FakeQuantize
    prepare_sim(model, fuse_map, qconfig, device)  → fused + fake-quant model (train mode)
    calibrate(model, loader, device, n_samples)    → PTQ: freeze observers after calibration
    compute_layer_sensitivity(model, loader, ...)  → per-layer INT4 sensitivity scores
    assign_mixed_precision(model, sensitivities, int8_ratio) → per-module qconfig assignment
    apply_weight_ptq(model, scheme)                → weight-only binary/ternary PTQ (BWN/TWN)
    theoretical_size_mb(model, w_bits, bits_map)   → packed size in MB
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import QConfig, FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)

_QUANTIZABLE = (nn.Conv2d, nn.Linear)


def _act_fakequant(bits: int):
    """Unsigned per-tensor activation FakeQuantize restricted to `bits`."""
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=(1 << bits) - 1,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )


def _weight_fakequant(bits: int, per_channel: bool = True):
    """Signed symmetric weight FakeQuantize restricted to `bits`."""
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if per_channel:
        return FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=qmin,
            quant_max=qmax,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            reduce_range=False,
        )
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=qmin,
        quant_max=qmax,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )


def make_qconfig(w_bits: int = 8, a_bits: int = 8, per_channel: bool = True) -> QConfig:
    """QConfig with weight/activation FakeQuantize restricted to the given bit-widths."""
    return QConfig(
        activation=_act_fakequant(a_bits),
        weight=_weight_fakequant(w_bits, per_channel),
    )


def prepare_sim(
    model: nn.Module,
    fuse_map: list,
    qconfig: QConfig,
    device: torch.device,
    fuse_root: Optional[nn.Module] = None,
) -> nn.Module:
    """Deep-copy, fuse Conv-BN(-ReLU), attach qconfig, insert FakeQuantize (train mode).

    Fusion is best-effort: if a fuse_map entry is incompatible (e.g. ReLU6 in
    MobileNetV2), we retry unfused rather than crash the whole run.
    """
    model = copy.deepcopy(model).to(device)
    model.train()
    root = model if fuse_root is None else fuse_root
    if fuse_map:
        try:
            tq.fuse_modules_qat(root, fuse_map, inplace=True)
        except Exception as e:  # noqa: BLE001 — fall back to unfused simulation
            print(f"  [prepare_sim] fusion skipped ({type(e).__name__}: {e}); continuing unfused")
    model.qconfig = qconfig
    tq.prepare_qat(model, inplace=True)
    return model


@torch.no_grad()
def calibrate(model: nn.Module, loader, device: torch.device, n_samples: int = 512) -> nn.Module:
    """PTQ: run forward passes to collect ranges, then freeze observers + BN stats."""
    model.eval()
    seen = 0
    for data, _ in loader:
        model(data.to(device))
        seen += data.size(0)
        if seen >= n_samples:
            break
    model.apply(tq.disable_observer)
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    return model


@torch.no_grad()
def compute_layer_sensitivity(model: nn.Module, loader, device: torch.device, w_bits: int = 4) -> dict:
    """Per-layer INT4 sensitivity: logit MSE when one layer's weights are quantized.

    Higher score → more sensitive → keep at higher precision. Uses a single
    calibration batch for speed.
    """
    model = model.eval().to(device)
    data, _ = next(iter(loader))
    data = data.to(device)
    baseline = model(data).detach()

    qmin, qmax = -(1 << (w_bits - 1)), (1 << (w_bits - 1)) - 1
    sensitivities: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, _QUANTIZABLE):
            continue
        w = module.weight.data
        # per-output-channel symmetric fake-quant of this layer's weights
        flat = w.reshape(w.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp(min=1e-8) / qmax
        scale = scale.reshape([-1] + [1] * (w.dim() - 1))
        w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
        module.weight.data = w_q
        out = model(data).detach()
        module.weight.data = w  # restore
        sensitivities[name] = torch.mean((out - baseline) ** 2).item()
    return sensitivities


def assign_mixed_precision(
    model: nn.Module,
    sensitivities: dict,
    int8_ratio: float = 0.3,
    per_channel: bool = True,
) -> dict:
    """Assign per-module qconfigs: top sensitive layers INT8, the rest INT4.

    Sets `module.qconfig` in place and returns {layer_name: bits} for reporting.
    """
    ranked = sorted(sensitivities.items(), key=lambda kv: kv[1], reverse=True)
    n_int8 = max(1, round(len(ranked) * int8_ratio))
    int8_layers = {name for name, _ in ranked[:n_int8]}

    qconfig_int8 = make_qconfig(w_bits=8, a_bits=8, per_channel=per_channel)
    qconfig_int4 = make_qconfig(w_bits=4, a_bits=8, per_channel=per_channel)

    bits_map: dict[str, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, _QUANTIZABLE):
            continue
        bits = 8 if name in int8_layers else 4
        module.qconfig = qconfig_int8 if bits == 8 else qconfig_int4
        bits_map[name] = bits
    return bits_map


def apply_weight_ptq(model: nn.Module, scheme: str) -> nn.Module:
    """Weight-only PTQ at 1-2 bits, per output channel. Returns a deep copy.

    scheme="binary"  → {-α, +α}     (XNOR-Net BWN: α = mean|w| per channel)
    scheme="ternary" → {-α, 0, +α}  (TWN: Δ = 0.7·mean|w|, α = mean|w| over |w|>Δ)

    Activations stay FP32 — the weight-only setup those papers report. No retraining,
    so at 1-2 bits accuracy typically collapses without QAT; that is the finding.
    """
    model = copy.deepcopy(model)
    for module in model.modules():
        if not isinstance(module, _QUANTIZABLE):
            continue
        w = module.weight.data
        flat = w.reshape(w.shape[0], -1)  # [out_ch, fan_in]
        if scheme == "binary":
            alpha = flat.abs().mean(dim=1, keepdim=True)
            q = torch.sign(flat) * alpha
        elif scheme == "ternary":
            delta = 0.7 * flat.abs().mean(dim=1, keepdim=True)
            mask = (flat.abs() > delta).float()
            alpha = (flat.abs() * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            q = torch.sign(flat) * mask * alpha
        else:
            raise ValueError(f"unknown scheme: {scheme!r} (expected 'binary' or 'ternary')")
        module.weight.data = q.reshape(w.shape).to(w.dtype)
    return model


def theoretical_size_mb(
    model: nn.Module,
    w_bits: int = 8,
    bits_map: Optional[dict] = None,
) -> float:
    """Packed model size in MB.

    Weight tensors (ndim >= 2) count at `w_bits` (or per-layer `bits_map`);
    biases / BN params (ndim < 2) count at 8 bits (small, kept higher precision).
    """
    total_bits = 0
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            b = w_bits
            if bits_map is not None:
                owner = name.rsplit(".", 1)[0]  # strip ".weight"
                b = bits_map.get(owner, w_bits)
            total_bits += p.numel() * b
        else:
            total_bits += p.numel() * 8
    return total_bits / 8 / (1024 ** 2)


if __name__ == "__main__":
    # ponytail: self-check — INT4 sim runs and shrinks theoretical size vs FP32
    net = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10),
    )
    dev = torch.device("cpu")
    loader = [(torch.randn(4, 3, 16, 16), torch.randint(0, 10, (4,))) for _ in range(3)]

    fp32_mb = theoretical_size_mb(net, w_bits=32)
    int4_mb = theoretical_size_mb(net, w_bits=4)
    assert int4_mb < fp32_mb, (int4_mb, fp32_mb)

    m = prepare_sim(net, fuse_map=[["0", "1", "2"]], qconfig=make_qconfig(4, 8), device=dev)
    m = calibrate(m, loader, dev, n_samples=8)
    with torch.no_grad():
        y = m(loader[0][0])
    assert y.shape == (4, 10), y.shape

    sens = compute_layer_sensitivity(net, loader, dev, w_bits=4)
    assert len(sens) == 2, sens  # one conv, one linear
    bits = assign_mixed_precision(copy.deepcopy(net), sens, int8_ratio=0.5)
    assert set(bits.values()) <= {4, 8}
    print(f"OK — fp32={fp32_mb:.4f}MB int4={int4_mb:.4f}MB sens={sens} bits={bits}")
