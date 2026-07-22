"""Compression measurement — Phase 9 Task 3 (ideas/PHASE9_PLAN.md D5/D6).

Measurement only: no changes to ml/checkpoint.py. Answers whether there's headroom
above `compress_checkpoint`'s plain gzip by comparing, on the same population of
Conv2d/Linear weight tensors:
  1. Nominal INT8 (8 bits/weight, no entropy coding).
  2. Shannon entropy of the actual INT8-quantized weight distribution (bits/weight
     really used — INT8 ranges are rarely uniformly occupied).
  3. k-means weight-clustering on the pre-quantization FP32 weights, at 16/32/64
     clusters (Deep Compression's 4/5/6-bit weight-sharing sweep): codebook +
     Huffman-coded-index-stream accounting, not an actual bitstream.

These three are weights-only and directly comparable to each other. The real on-disk
gzip size of the actual INT8 checkpoint file is reported separately as context — that
file includes biases/BN params/quantization metadata gzip also compresses, so its
ratio isn't directly comparable to the weights-only numbers above it.

    python -m scripts.measure_compression --model alexnet_fire --runtime local
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.vq import kmeans2

import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY
from ml import MODEL_REGISTRY, build_runtime_paths, disk_mb, gzip_mb, load_best_model
from scripts.train import _load_profile, _make_model_runs


def _weight_tensors(model: nn.Module) -> list[torch.Tensor]:
    """Every Conv2d/Linear weight — the tensors real INT8 quantization actually touches."""
    return [m.weight.detach() for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def _quantize_per_channel_int8(w: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel INT8 codes, matching fbgemm's weight qconfig (-127..127)."""
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    codes = torch.round(flat / scale.unsqueeze(1)).clamp(-127, 127)
    return codes.to(torch.int8)


def entropy_bits_per_weight(model: nn.Module) -> float:
    codes = torch.cat([_quantize_per_channel_int8(w).flatten() for w in _weight_tensors(model)])
    _, counts = torch.unique(codes, return_counts=True)
    probs = counts.float() / counts.sum()
    return float(-(probs * torch.log2(probs)).sum())


def kmeans_codebook_size_mb(model: nn.Module, n_clusters: int) -> tuple[float, float]:
    """(codebook_mb, index_stream_mb) for k-means weight-sharing on the FP32 weights."""
    all_w = torch.cat([w.flatten() for w in _weight_tensors(model)]).numpy().astype(np.float64)
    n_clusters = min(n_clusters, len(np.unique(all_w)))
    _, labels = kmeans2(all_w, n_clusters, minit="++", seed=42)
    codebook_mb = n_clusters * 4 / (1024 ** 2)  # fp32 centroids
    bits_per_index = max(1, math.ceil(math.log2(n_clusters)))
    index_stream_mb = len(labels) * bits_per_index / 8 / (1024 ** 2)
    return codebook_mb, index_stream_mb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare entropy/k-means weight-sharing against plain gzip.")
    parser.add_argument("--model", default="alexnet_fire", help="Registered model name")
    parser.add_argument("--runtime", default="local")
    parser.add_argument("--experiment", default="default")
    parser.add_argument("--fp32-checkpoint", default=None, help="Explicit FP32 checkpoint override")
    parser.add_argument("--int8-checkpoint", default=None, help="Explicit INT8 checkpoint file override (gzip context line)")
    parser.add_argument("--clusters", type=int, nargs="+", default=[16, 32, 64])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}; registered: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[args.model]

    if args.fp32_checkpoint:
        fp32_path = Path(args.fp32_checkpoint)
        model = spec["ctor"]()
        state = torch.load(fp32_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
    else:
        runtime_cfg = _load_profile(args.runtime, "runtime")
        runtime_paths = build_runtime_paths(runtime_cfg.get("root", "outputs/local"))
        _, checkpoints_dir, _, _, _ = _make_model_runs(runtime_paths.root, args.experiment, args.model)
        fp32_path = checkpoints_dir / f"{args.model}_best.pth"
        model = load_best_model(args.model, spec["ctor"], checkpoints_dir, torch.device("cpu"))

    int8_path = Path(args.int8_checkpoint) if args.int8_checkpoint else fp32_path.parent / f"qat_{args.model}.pth"

    n_weights = sum(w.numel() for w in _weight_tensors(model))
    entropy = entropy_bits_per_weight(model)
    nominal_mb = n_weights * 8 / 8 / (1024 ** 2)
    entropy_mb = n_weights * entropy / 8 / (1024 ** 2)

    print(f"Model: {args.model}  ({n_weights:,} Conv2d/Linear weights, from {fp32_path.name})")
    print()
    print("Weights-only estimators (self-consistent, directly comparable to each other):")
    print(f"  {'method':26s} {'bits/weight':>12s} {'size_MB':>10s}")
    print(f"  {'INT8 nominal':26s} {8.0:12.2f} {nominal_mb:10.4f}")
    print(f"  {'INT8 entropy (actual)':26s} {entropy:12.2f} {entropy_mb:10.4f}")
    assert entropy <= 8.0, f"entropy {entropy:.2f} bits/weight exceeds the INT8 range — sanity check failed"

    for k in args.clusters:
        codebook_mb, index_mb = kmeans_codebook_size_mb(model, k)
        total_mb = codebook_mb + index_mb
        bits = max(1, math.ceil(math.log2(k)))
        print(f"  {f'k-means k={k} ({bits}-bit)':26s} {bits:12d} {total_mb:10.4f}")

    print()
    print("On-disk context (whole checkpoint file — includes biases/BN/quant metadata,")
    print("not directly comparable to the weights-only rows above):")
    raw_mb = disk_mb(int8_path)
    gz_mb = gzip_mb(int8_path)
    if raw_mb is None:
        print(f"  No INT8 checkpoint file found at {int8_path}.")
    else:
        print(f"  {int8_path}")
        print(f"  raw: {raw_mb:.4f} MB  ->  gzip: {gz_mb:.4f} MB  (ratio {raw_mb / gz_mb:.2f}x)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
