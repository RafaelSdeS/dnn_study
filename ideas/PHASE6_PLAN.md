# Phase 6 — Hardware Profiling & Winograd Efficiency Validation (Implementation Plan)

Profiling is **inference-only** — no FP32/QAT/INT8 training happens in this phase. Latency and
memory-bandwidth timings depend on layer shapes and dtypes, not on trained weight values, so models
are built directly from `MODEL_REGISTRY` (random init). No checkpoint files need to exist or be
synced between the RTX 4060 laptop and PCAD.

---

## Selected models

Computed the actual Pareto frontier (FP32 top-1 accuracy vs. size-MB, from `results/model_details.csv`
+ Phase 4's `final_architecture_phase4/final_comparison.csv` — a model is on the frontier if no other
model has both higher accuracy *and* smaller-or-equal size) instead of eyeballing it from
`ideas/BEST_MODELS.md`. Every frontier model is included, plus two off-frontier models kept for a
reason stated next to them. Full retraining of every Phase 1–4 model is not needed — profiling is
inference-only (see top of file) and this set already brackets the interesting cases.

Winograd eligibility was checked directly against each model's `Conv2d` calls in `models/*.py`
(kernel size, stride, `groups`) — the classic cuDNN Winograd F(2×2,3×3) transform only accelerates
**dense (`groups=1`) 3×3 convs at stride 1**. It does not help 1×1 convs (no spatial reuse), strided
convs, kernels other than 3×3, or depthwise/grouped convs (no cross-channel reduction to batch into a
GEMM, and they're already memory-bound, so the extra transform traffic makes them slower, not faster).

| # | Model (registry name) | Role | FP32 Top-1 | INT8 Top-1 | Quant Δ | Size (MB) | Winograd-eligible | Architectural efficiency technique |
|---|---|---|---|---|---|---|---|---|
| 1 | `alexnet_tv` | **Off-frontier, kept deliberately**: large-kernel control (11×11→5×5→3×3) — the baseline everything else is measured against | 32.89% | 31.90% | −0.98pp | 661.75 (FP32) | ⚠️ Only the trailing 3 conv layers (3×3, s1); the 11×11 s4 and 5×5 layers that dominate compute are not | **None** — classic dense conv stack, no compression trick; this is the un-optimized baseline |
| 2 | `alexnet_depthwisesep` | Pareto frontier: smallest model, best accuracy at that size | 44.39% | 41.47% | −2.92pp ⚠️ | 3.65 | ❌ None — every 3×3 is `groups=in_channels` (true depthwise) | **Depthwise separable conv** (depthwise 3×3 + pointwise 1×1) — factorizes a full conv into per-channel spatial filtering + channel mixing, cutting FLOPs ~k² fold |
| 3 | `alexnet_bottleneck` | Pareto frontier | 44.62% | 44.54% | −0.08pp | 4.49 | ✅ The one dense 3×3 per block (1×1s don't count) | **Bottleneck** (1×1 reduce → 3×3 → 1×1 expand, ResNet-style) — shrinks channel count before the expensive 3×3, restores it after |
| 4 | `alexnet_final_bottleneck_residual` | Pareto frontier, and a quantization *gain* outlier | 45.10% | 45.98% | +0.88pp | 6.65 | ✅ Same dense 3×3 bottleneck branch | **Bottleneck + residual connection** — adds a skip connection (via `FloatFunctional`) around the bottleneck to recover accuracy lost to channel compression |
| 5 | `alexnet_final_fire_residual` | Pareto frontier: best hybrid accuracy, quantization-stable | 49.79% | 49.20% | −0.60pp | 8.09 | ✅ Fire's dense 3×3 expand branch | **Fire module + residual** (SqueezeNet-style squeeze 1×1 → parallel 1×1/3×3 expand) with a skip connection added |
| 6 | `vgg_style` | Pareto frontier | 51.81% | 51.19% | −0.63pp | 27.58 | ✅ **Fully** — every conv is dense 3×3, stride 1 | **None** — plain stacked dense 3×3 (VGG-style depth), no channel-reduction trick; relies on small kernel size alone |
| 7 | `mobilenetv2` | Pareto frontier: best overall accuracy | 57.99% | — (not INT8 converted) | — | 28.75 | ❌ None — every 3×3 is depthwise inside the inverted-residual blocks; the rest are 1×1 | **Inverted residual + depthwise separable** (1×1 expand → depthwise 3×3 → 1×1 project, skip connection between the thin ends) |
| 8 | `alexnet_fire` | **Off-frontier, kept deliberately**: dominated by `alexnet_bottleneck` on accuracy/size, but included for its quantization *gain* (+0.33pp) — a different axis than this frontier measures | 43.98% | 44.30% | +0.33pp | 5.99 | ✅ Fire's dense 3×3 expand branch | **Fire module** (SqueezeNet-style squeeze 1×1 → parallel 1×1/3×3 expand), no residual |

Two different efficiency categories are stacked in this table and worth keeping distinct in the
write-up: **architectural techniques** (bottleneck/fire/depthwise-separable/inverted-residual) reduce
the *FLOP/param count* of the model itself, independent of hardware; **Winograd/FFT** reduce the *cost
of executing a given fixed convolution* on specific hardware. A model can win on one axis and lose on
the other — `alexnet_depthwisesep` is the smallest model here (architectural win) but has zero
Winograd-eligible layers (hardware-acceleration loss).

**Counter-intuitive result worth calling out in the final write-up:** two of the Pareto-frontier
"small-kernel" champions — `alexnet_depthwisesep` and `mobilenetv2` — are built almost entirely from
ops Winograd doesn't accelerate, despite using 3×3 kernels. "Small kernel" and "Winograd-compatible"
are not the same claim. `profile_kernel_trace()` exists specifically to confirm this empirically
(does cuDNN ever emit a Winograd kernel for these two, vs. `vgg_style`/the bottleneck-fire hybrids
where it plausibly should) rather than resting on this static analysis alone.

Excluded (dominated on accuracy/size, and not needed on another axis either): `resnet18_tv` (dominated
by `vgg_style`/`mobilenetv2`), `alexnet_residual`/`alexnet_stacked`/`alexnet_factorized`/`alexnet_groupconv`/`alexnet_se`
(all 400–700 MB with lower accuracy than the frontier at 1/100th the size), all Phase 2 pure
kernel-restricted variants (superseded by Phase 3/4 compensated designs).

---

## New files

**`ml/profiling.py`**
- `profile_layer_latency(kernel_size, in_ch, out_ch, input_size, device, warmup=50, iters=200)` — bare `nn.Conv2d`, timed forward passes (same warmup/sync pattern as `Trainer.benchmark()`).
- `profile_model_latency(model, input_size, device, warmup=50, iters=200)` — full-model forward pass, random input tensor, no dataloader.
- `profile_kernel_trace(model, input, device)` — wraps `torch.profiler.profile()`; returns per-op durations and flags whether any op name contains `"winograd"`. Best-effort only: cuDNN's Winograd kernel naming isn't a stable public contract, varies by cuDNN version, and may not appear at all if cuDNN picks an unnamed fused kernel — treat a `False` here as "not confirmed," not "definitely not used."
- `profile_layer_conv_fft(kernel_size, in_ch, out_ch, input_size, device, warmup=50, iters=200)` — FFT-based conv2d via `torch.fft.rfft2`/`irfft2` (already in installed torch 2.5.1, no new dependency): zero-pad input and kernel to `input_size + kernel_size - 1` per spatial dim (linear, not circular convolution), `rfft2` both, multiply, sum over input channels, `irfft2`, crop to the valid output region. Only meaningful for `kernel_size >= 5` — skip 2×2/3×3, FFT's fixed transform overhead is a guaranteed loss there. FP32 only — there's no standard quantized/complex-FFT INT8 path, so this ignores the `precisions` list entirely. Has no native stride support: a strided layer is profiled by computing the full stride-1 output and discarding samples, which is itself a real (and worth reporting) strike against FFT for a layer like `alexnet_tv`'s stride-4 first conv.

**`configs/profiling.yaml`**
```yaml
name: phase6
kernel_sizes: [2, 3, 5, 7, 9, 11] # single-layer sweep; 7/9/11 added to find the FFT-vs-direct/Winograd crossover point (matches alexnet_tv's real 5×5 and 11×11 layers)
fft_min_kernel_size: 5            # profile_layer_conv_fft only runs for kernel_size >= this
layer_channels: [16, 32, 64, 128] # each value used as both in_ch and out_ch (4 layer configs, not a 16-way cross product)
input_size: [1, 3, 64, 64]
models:                          # full-model sweep, built from MODEL_REGISTRY, no checkpoint needed
  - alexnet_tv
  - alexnet_depthwisesep
  - alexnet_bottleneck
  - alexnet_final_bottleneck_residual
  - alexnet_final_fire_residual
  - vgg_style
  - mobilenetv2
  - alexnet_fire
precisions: [fp32, int8]         # int8 = build_qat_from_model(spec["ctor"]().to(device), arch_name, device) + convert_to_int8 — NOT build_qat(), which calls load_best_model() and hard-requires an existing {arch_name}_best.pth checkpoint (none exist locally; see top of file). build_qat_from_model takes an already-constructed model and does zero disk I/O. Uncalibrated observers are fine here — structure determines latency, not calibration accuracy.
warmup: 50
iters: 200
```

**`scripts/profile_hardware.py`** — mirrors `scripts/train.py`'s CLI shape (`--experiment`/`--runtime`/`--dry-run`), loads `profiling.yaml` + a `runtime` profile, runs the layer sweep and model sweep, writes JSON to `outputs/{runtime}/phase6/{device_tag}_profile.json`, where `device_tag = torch.cuda.get_device_name(0)` (sanitized to snake_case) — the two output files self-identify as "RTX 4060" vs "RTX 4090" without a separate machine registry.

Progress: one `logging.info("[%d/%d] %s", i, total, label)` line per config as each finishes (~80 configs total: 48 direct-layer [6 kernel sizes × 4 channel widths × 2 precisions] + 16 FFT-layer [4 kernel sizes ≥5 × 4 channel widths, fp32 only] + 16 model [8 models × 2 precisions]). No tqdm — each config is a few seconds, not an epoch loop, so a per-item log line is enough to confirm it's alive.

Logging setup mirrors `Trainer(log_file=...)`: a `logging.FileHandler` writing to `outputs/{runtime}/phase6/{device_tag}.log` *plus* a `StreamHandler` to stdout — same dual console+file pattern, so the run history survives independent of whether stdout was a live terminal or a SLURM `%x-%j.out` capture. `tail -f` either the `.log` file or the sbatch `.out` file to watch it live.

Skipped by default: a *per-config* W&B log (no time series to plot here). Still optional and cheap to turn on — reuse `scripts/train.py`'s existing `_maybe_init_wandb()` and the `wandb` flag already in `runtime.yaml`; at sweep end, one `wandb.log({"phase6_results": wandb.Table(dataframe=df)})` call uploads the whole results table for cross-device comparison in the W&B UI. No new wandb code needed, just the one call gated by the existing flag.

Output JSON shape (flat, one record per run — matches what a `pandas.read_json(..., lines=True)` or a simple list-comprehension in the notebook expects):
```json
{"kind": "layer", "kernel_size": 3, "in_ch": 32, "out_ch": 32, "precision": "fp32", "latency_ms": 0.42, "device": "rtx_4090"}
{"kind": "model",  "model": "alexnet_fire", "precision": "int8", "latency_ms": 3.1, "throughput_img_s": 322.0, "device": "rtx_4090"}
```

`kind: model` rows carry latency only — accuracy is **not** re-measured here (these are untrained/random-init forward passes; see top of file). Kernel-size-vs-accuracy was already answered in Phases 2–4 and lives in `results/model_details.csv` (`fp32_top1`, `int8_top1`) and `results/final_architecture_phase4/final_comparison.csv`. The analysis notebook joins Phase 6's `model` rows to those existing accuracy columns on `model_name` — no retraining, just a merge — so the final plot is accuracy vs. latency per model, not latency alone.

**`configs/slurm/tupi_4090.yaml`** — `partition: tupi`, `gres: gpu:1`, to pin PCAD submissions to an RTX 4090 node instead of whatever `single_gpu.yaml` lands on.

**`scripts/slurm/profile.sbatch`** — copy of `train.sbatch` calling `python -m scripts.profile_hardware "$@"`.

**`scripts/cluster.py`** — add a `profile-submit` subcommand reusing `_build_sbatch_command`, parameterized with the new sbatch script path instead of the hardcoded `train.sbatch`.

**`notebooks/analysis/hardware_profiling_phase6.ipynb`** — loads the RTX 4060 (local) and RTX 4090 (PCAD tupi) JSON outputs side by side; joins `kind: model` rows to `results/model_details.csv` accuracy columns on model name; builds the TODO.md outputs: latency heatmap (kernel size × layer depth), speedup ratio (5×5 / 3×3 time) per GPU, Winograd feasibility threshold, CPU INT8 latency ranking, cross-GPU comparison, **plus an accuracy-vs-latency scatter (the actual efficiency/quality trade-off this whole project is measuring)**.

**Reused as-is:** `configs/runtime/local.yaml`, `configs/runtime/pcad.yaml`, `ml/runtime.py` (`set_global_seed`, `expand_path`), `ml/reporting.py` (`disk_mb`, `compute_flops`), `MODEL_REGISTRY`, `ml/quantization.py` (`build_qat_from_model`, `convert_to_int8` — specifically **not** `build_qat`, see checkpoint note above).

**Manual/out of scope for tooling:**
- Nsight Compute (`ncu --set full python -m scripts.profile_hardware ...`) — external CLI, run directly on the node, not wrapped in Python.
- GPU power draw — sample `nvidia-smi --query-gpu=power.draw --loop-ms=100` in the background during a profiling run; only pursue if the PCAD node actually exposes it.
