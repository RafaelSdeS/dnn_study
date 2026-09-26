"""Phase 6 — one conv config, sized for Nsight Compute.

Usage (see scripts/slurm/ncu.sbatch):
    ncu --set full --profile-from-start off python -m scripts.phase6.ncu_targets {conv2,conv3,conv5,winograd3}

Only the single call between cudaProfilerStart/Stop is profiled, so warmup and cuDNN init stay
out of the report. Layer mirrors ml/profiling.py's layer sweep: dense, stride 1,
padding=(k-1)//2, bias-free; `winograd3` is the hand-rolled eager F(2x2,3x3) with the kernel
transform precomputed and TF32 matmuls, like profile_layer_conv_winograd.
"""

import argparse

import torch
import torch.nn as nn

from ml.profiling import _winograd_kernel_transform, winograd_conv2d_f23_transformed
from ml.runtime import set_global_seed

TARGETS = ("conv2", "conv3", "conv5", "winograd3")
DEFAULT_CHANNELS = 64
DEFAULT_BATCH = 64       # the batch-64 point of configs/profiling.yaml, the one that leaves launch-bound
DEFAULT_RESOLUTION = 64
DEFAULT_WARMUP = 10


def build_layer(target: str, channels: int):
    if target == "winograd3":
        U = _winograd_kernel_transform(torch.randn(channels, channels, 3, 3, device="cuda"))
        return lambda x: winograd_conv2d_f23_transformed(x, U)
    k = int(target.removeprefix("conv"))
    return nn.Conv2d(channels, channels, k, stride=1, padding=(k - 1) // 2, bias=False).cuda().eval()


def main():
    parser = argparse.ArgumentParser(description="Phase 6 single-conv target for Nsight Compute")
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    args = parser.parse_args()

    set_global_seed(42)
    torch.set_float32_matmul_precision("high")  # cuDNN already runs TF32; without this Winograd's einsums wouldn't
    layer = build_layer(args.target, args.channels)
    x = torch.randn(args.batch, args.channels, args.resolution, args.resolution, device="cuda")

    with torch.no_grad():
        for _ in range(args.warmup):
            layer(x)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        layer(x)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    print(f"{args.target}: profiled 1 call, batch={args.batch} ch={args.channels} res={args.resolution}")


if __name__ == "__main__":
    main()
