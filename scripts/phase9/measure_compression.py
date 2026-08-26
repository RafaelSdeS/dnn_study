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

    python -m scripts.phase9.measure_compression --model alexnet_fire --runtime local
    python -m scripts.phase9.measure_compression --model alexnet_fire --runtime local --evaluate

--evaluate closes the gap left open by the measurement-only pass: it actually snaps
each Conv2d/Linear weight to its nearest k-means centroid (not just sizes the result)
and runs Trainer.evaluate() on the clustered model, so the "does clustering preserve
accuracy" question the size-only numbers can't answer gets a real number.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.cluster.vq import kmeans2

import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY
from ml import (
    MODEL_REGISTRY,
    DataConfig,
    TrainerConfig,
    Trainer,
    build_runtime_paths,
    create_imagenet_loaders,
    disk_mb,
    gzip_mb,
    load_best_model,
)
from configs.loader import load_config
from scripts.train import _ensure_dataset_path, _load_profile, _make_model_runs


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


def cluster_weights(model: nn.Module, n_clusters: int, seed: int = 42) -> tuple[nn.Module, float, float]:
    """Deep-copies `model`, snaps every Conv2d/Linear weight to its nearest of `n_clusters`
    global k-means centroids (Deep Compression's weight-sharing step, applied for real rather
    than just sized). Returns (clustered_model, codebook_mb, index_stream_mb); `model` itself is
    untouched."""
    clustered = copy.deepcopy(model)
    weight_modules = _weight_tensors_owners(clustered)
    all_w = torch.cat([m.weight.detach().flatten() for m in weight_modules]).numpy().astype(np.float64)
    k = min(n_clusters, len(np.unique(all_w)))
    centroids, labels = kmeans2(all_w, k, minit="++", seed=seed)
    labels_t = torch.from_numpy(labels.astype(np.int64))
    centroids_t = torch.from_numpy(centroids.astype(np.float32))

    offset = 0
    for m in weight_modules:
        n = m.weight.numel()
        m.weight.data = centroids_t[labels_t[offset:offset + n]].reshape(m.weight.shape).to(m.weight.dtype)
        offset += n

    codebook_mb = k * 4 / (1024 ** 2)  # fp32 centroids
    bits_per_index = max(1, math.ceil(math.log2(k)))
    index_stream_mb = labels_t.numel() * bits_per_index / 8 / (1024 ** 2)
    return clustered, codebook_mb, index_stream_mb


def _weight_tensors_owners(model: nn.Module) -> list[nn.Module]:
    """The Conv2d/Linear modules themselves (not just their weight tensors) — needed to write
    clustered weights back in place."""
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare entropy/k-means weight-sharing against plain gzip.")
    parser.add_argument("--model", default="alexnet_fire", help="Registered model name")
    parser.add_argument("--runtime", default="local")
    parser.add_argument("--experiment", default="default")
    parser.add_argument("--fp32-checkpoint", default=None, help="Explicit FP32 checkpoint override")
    parser.add_argument("--int8-checkpoint", default=None, help="Explicit INT8 checkpoint file override (gzip context line)")
    parser.add_argument("--clusters", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--evaluate", action="store_true",
                         help="Actually cluster the weights and run Trainer.evaluate() per cluster count, "
                              "against an unclustered FP32 baseline (needs the dataset)")
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

    cluster_sizes = {}
    for k in args.clusters:
        _, codebook_mb, index_mb = cluster_weights(model, k)
        cluster_sizes[k] = codebook_mb + index_mb
        bits = max(1, math.ceil(math.log2(k)))
        print(f"  {f'k-means k={k} ({bits}-bit)':26s} {bits:12d} {cluster_sizes[k]:10.4f}")

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

    if args.evaluate:
        _evaluate_clustered_accuracy(args, model, cluster_sizes)

    return 0


def _evaluate_clustered_accuracy(args: argparse.Namespace, model: nn.Module, cluster_sizes: dict[int, float]) -> None:
    """Real Top-1/Top-5 after clustering, vs. the unclustered FP32 baseline — the accuracy half
    of H3's "does this preserve accuracy" question the size-only numbers above can't answer."""
    runtime_cfg = _load_profile(args.runtime, "runtime")
    runtime_paths = build_runtime_paths(runtime_cfg.get("root", "outputs/local"))
    data_cfg = DataConfig(**load_config("data.yaml"))
    data_cfg.dataset_path = str(_ensure_dataset_path(runtime_cfg))
    _, _, _, val_loader = create_imagenet_loaders(data_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer_cfg = TrainerConfig(**load_config("training.yaml"))
    eval_dir = runtime_paths.root / "phase9_compression"

    def _eval(m: nn.Module, name: str) -> dict:
        trainer = Trainer(m.to(device), val_loader, val_loader, trainer_cfg, device, eval_dir, name, num_classes=data_cfg.num_classes)
        return trainer.evaluate(topk=(1, 5))

    baseline = _eval(copy.deepcopy(model), f"{args.model}_baseline")
    print(f"\nFP32 baseline (unclustered): top1={baseline['top1']:.2f}% top5={baseline['top5']:.2f}%")

    rows = [{"cluster_k": None, "bits_per_weight": 32.0, "size_mb": None,
             "top1": baseline["top1"], "top5": baseline["top5"], "top1_drop_pp": 0.0}]
    for k in args.clusters:
        clustered_model, codebook_mb, index_mb = cluster_weights(model, k)
        metrics = _eval(clustered_model, f"{args.model}_k{k}")
        bits = max(1, math.ceil(math.log2(k)))
        drop = baseline["top1"] - metrics["top1"]
        print(f"k={k:3d} ({bits}-bit): top1={metrics['top1']:.2f}% (Δ{drop:+.2f}pp) top5={metrics['top5']:.2f}% "
              f"size={cluster_sizes[k]:.4f} MB")
        rows.append({"cluster_k": k, "bits_per_weight": bits, "size_mb": codebook_mb + index_mb,
                      "top1": metrics["top1"], "top5": metrics["top5"], "top1_drop_pp": drop})

    out_csv = runtime_paths.results / f"{args.model}_compression_accuracy.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    raise SystemExit(main())
