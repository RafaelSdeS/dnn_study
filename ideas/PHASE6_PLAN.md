# Phase 6 — Hardware Profiling & Winograd Efficiency Validation (Implementation Plan)

Profiling is **inference-only** — no FP32/QAT/INT8 training happens in this phase. Latency and
memory-bandwidth timings depend on layer shapes and dtypes, not on trained weight values, so models
are built directly from `MODEL_REGISTRY` (random init). No checkpoint files need to exist or be
synced between the RTX 4060 laptop and PCAD.

---

## Research Hypotheses

The following hypotheses drive Phase 6 experiments. Each is testable via measurements in the output JSON.

### H1: Winograd Acceleration on Dense 3×3 Convolutions (RTX 4090)
**Claim:** Dense 3×3 convolutions at stride 1 (e.g., in bottleneck, fire, vgg_style models) trigger Winograd acceleration on RTX 4090 and show measurable speedup vs. direct GEMM.

**Expected Outcome:** 
- Winograd kernel detected in trace for vgg_style, bottleneck-fire hybrids (>80% of model latency from Winograd ops).
- Empirical speedup ratio (5×5 direct latency / 3×3 latency) > 1.8, confirming hardware preference for 3×3.
- Measured latency improvement: 1.5–2.5× faster for 3×3 dense convs vs. 5×5 direct on RTX 4090.

**Evidence to Collect:**
- `winograd_trace_detected` (kernel-name heuristic).
- `winograd_speedup_info.speedup_ratio` (empirical 3×3 vs. 5×5 comparison).
- Full-model latency comparison: vgg_style (all 3×3) vs. alexnet_tv (mixed sizes).

**Acceptance Criterion:** Both trace and empirical speedup signals agree (both positive) for vgg_style on 4090. Supporting evidence: high `gpu_utilization_pct` (>85%) and high `compute_efficiency_gflops_s` (>400 on RTX 4090) for dense-conv models, indicating compute-bound execution (Winograd's target).

---

### H2: Depthwise Convolutions Do NOT Benefit from Winograd
**Claim:** Models built from depthwise separable convolutions (alexnet_depthwisesep, mobilenetv2) use 3×3 kernels but do NOT trigger Winograd, because Winograd only accelerates *dense* (groups=1) 3×3 convs.

**Expected Outcome:**
- Winograd kernel use <10% of total model latency for alexnet_depthwisesep and mobilenetv2.
- Empirical speedup detector does NOT flag 3×3 as Winograd-accelerated (speedup ratio <1.8 or no clear separation).
- Despite lack of hardware acceleration, alexnet_depthwisesep beats vgg_style on latency due to architectural efficiency (lower FLOPs).
- Latency comparison: alexnet_depthwisesep (3.65 MB, low FLOPs, no Winograd) should be <50% of vgg_style (27.58 MB, more FLOPs, Winograd-accelerated) on both 4060 and 4090.

**Evidence to Collect:**
- Per-layer type classification: mark each layer as "dense 3×3 (Winograd-eligible)" vs. "depthwise 3×3 (not eligible)".
- `winograd_trace_detected` and `winograd_speedup_info` for each model (all-negative for depthwise models).
- Full-model latency: alexnet_depthwisesep vs. vgg_style.

**Acceptance Criterion:** Depthwise models show <10% Winograd kernel use; dense models show >60%. Supporting evidence: depthwise-sep models show lower `compute_efficiency_gflops_s` (<100) and lower `gpu_utilization_pct` (<70%), indicating memory-bound execution where depthwise convs' low FLOPs are the bottleneck, not hardware acceleration.

---

### H3: Efficiency Frontier Maintains Accuracy-per-Latency Advantage
**Claim:** The Pareto frontier models identified in Phase 4 (optimized for accuracy vs. size) also maintain efficiency on latency-per-unit-accuracy vs. the unoptimized baseline (alexnet_tv).

**Expected Outcome:**
- Frontier models (alexnet_bottleneck, alexnet_fire, vgg_style, mobilenetv2) show better accuracy/latency efficiency than alexnet_tv.
- Efficiency metric: top-1-accuracy / latency_ms (or equivalently, latency_ms / top-1-accuracy).
- Frontier models achieve target accuracy (45–58%) with <5 ms latency (FP32 on 4090), vs. alexnet_tv (~33% accuracy, 10–15 ms).
- The trade-off curve (accuracy vs. latency) shows frontier models in the upper-right quadrant (high accuracy, low latency) relative to baseline.

**Evidence to Collect:**
- Full-model latency (FP32 and INT8) for all 8 selected models on both 4060 and 4090.
- Accuracy numbers from Phase 4 results (`model_details.csv`, `final_comparison.csv`).
- Compute efficiency metric: accuracy / latency_ms for each model.

**Acceptance Criterion:** All frontier models show higher accuracy/latency ratio than alexnet_tv on at least one precision (FP32 or INT8).

---

### H4: INT8 Quantization Does Not Significantly Impact Layer Latency Structure (Structure, Not Accuracy)
**Claim:** INT8 inference latency may differ from FP32, but the relative ranking of layers and models should remain consistent (same fast/slow layers in both precisions).

**Expected Outcome:**
- INT8 and FP32 latency rankings are correlated (Spearman ρ > 0.9).
- Layers that are fast in FP32 remain fast in INT8; slow layers remain slow.
- Quantization may introduce 5–20% latency overhead (due to dequantization, potential fusions not applied) but does not reverse the order.

**Evidence to Collect:**
- Latency for all layer configs in both FP32 and INT8.
- Per-model latency for both precisions.
- Correlation of rankings.

**Acceptance Criterion:** Spearman rank correlation between FP32 and INT8 latencies > 0.85 for layers and models.

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

**`configs/profiling.yaml`** (Committed to git for reproducibility)
```yaml
name: phase6
kernel_sizes: [2, 3, 5, 7, 9, 11] # single-layer sweep; 7/9/11 added to find the FFT-vs-direct/Winograd crossover point (matches alexnet_tv's real 5×5 and 11×11 layers)
fft_min_kernel_size: 5            # profile_layer_conv_fft only runs for kernel_size >= this
layer_channels: [16, 32, 64, 128] # each value used as both in_ch and out_ch (4 layer configs, not a 16-way cross product)
batch_sizes: [1, 8]               # Batch 1 (single image, latency-critical) + batch 8 (realistic inference). Latency scales differently for memory-bound vs. compute-bound layers.
input_resolutions: [64, 128]      # Tiny ImageNet (64×64) + intermediate. Winograd gains may increase with spatial resolution.
# Layer sweep now expands: 6 kernel sizes × 4 channels × 2 batch sizes × 2 resolutions × 2 precisions = 192 configs (vs. 48 without sweeps).

models:                          # full-model sweep, built from MODEL_REGISTRY, no checkpoint needed
  - alexnet_tv
  - alexnet_depthwisesep
  - alexnet_bottleneck
  - alexnet_final_bottleneck_residual
  - alexnet_final_fire_residual
  - vgg_style
  - mobilenetv2
  - alexnet_fire
precisions: [fp32, int8]         # int8 = build_qat_from_model(spec["ctor"]().to(device), arch_name, device) + convert_to_int8 — NOT build_qat(), which calls load_best_model() and hard-requires an existing {arch_name}_best.pth checkpoint (none exist locally; see top of file). build_qat_from_model takes an already-constructed model and does zero disk I/O. Uncalibrated observers are fine here — structure determines latency, not calibration accuracy (see PHASE6_CALIBRATION.md for rationale and limitations).
warmup: 50
iters: 200
```

**`scripts/profile_hardware.py`** — mirrors `scripts/train.py`'s CLI shape (`--experiment`/`--runtime`/`--dry-run`), loads `profiling.yaml` + a `runtime` profile, runs the layer sweep and model sweep, writes JSON to `outputs/{runtime}/phase6/{device_tag}_profile.json`, where `device_tag = torch.cuda.get_device_name(0)` (sanitized to snake_case) — the two output files self-identify as "RTX 4060" vs "RTX 4090" without a separate machine registry.

Progress: one `logging.info("[%d/%d] %s", i, total, label)` line per config as each finishes (~80 configs total: 48 direct-layer [6 kernel sizes × 4 channel widths × 2 precisions] + 16 FFT-layer [4 kernel sizes ≥5 × 4 channel widths, fp32 only] + 16 model [8 models × 2 precisions]). No tqdm — each config is a few seconds, not an epoch loop, so a per-item log line is enough to confirm it's alive.

Logging setup mirrors `Trainer(log_file=...)`: a `logging.FileHandler` writing to `outputs/{runtime}/phase6/{device_tag}.log` *plus* a `StreamHandler` to stdout — same dual console+file pattern, so the run history survives independent of whether stdout was a live terminal or a SLURM `%x-%j.out` capture. `tail -f` either the `.log` file or the sbatch `.out` file to watch it live.

Skipped by default: a *per-config* W&B log (no time series to plot here). Still optional and cheap to turn on — reuse `scripts/train.py`'s existing `_maybe_init_wandb()` and the `wandb` flag already in `runtime.yaml`; at sweep end, one `wandb.log({"phase6_results": wandb.Table(dataframe=df)})` call uploads the whole results table for cross-device comparison in the W&B UI. No new wandb code needed, just the one call gated by the existing flag.

Output JSON shape (flat, one record per run — matches what a `pandas.read_json(..., lines=True)` or a simple list-comprehension in the notebook expects):
```json
{"kind": "metadata", "pytorch_version": "2.5.1", "cuda_version": "12.1", "cudnn_version": 90100, "device_name": "NVIDIA RTX 4090", "device_capability": [8, 9], "seed": 42, "timestamp": "2026-01-15T14:23:45", "config_file": "configs/profiling.yaml"}
{"kind": "layer", "kernel_size": 3, "in_ch": 32, "out_ch": 32, "batch_size": 1, "input_resolution": 64, "precision": "fp32", "latency_ms": 0.42, "winograd_trace_detected": true, "winograd_speedup_info": {"speedup_ratio": 2.1, "winograd_inferred": true}, "device": "rtx_4090"}
{"kind": "model", "model": "alexnet_fire", "precision": "int8", "latency_ms": 3.1, "throughput_img_s": 322.0, "power_draw_avg_w": 145.2, "power_draw_std_w": 3.4, "gpu_utilization_pct": 87.5, "gpu_memory_peak_mb": 2456, "compute_efficiency_gflops_s": 245.0, "device": "rtx_4090"}
```

**First record** (`kind: metadata`) captures environment: PyTorch/cuDNN/CUDA versions, GPU capability, seed, timestamp, config file path. Enables reproduction months later.

**Layer records** now include:
- `batch_size` and `input_resolution` to track parameter sweep (vs. fixed 64×64, batch-1).
- `winograd_trace_detected` (kernel-name heuristic) + `winograd_speedup_info` (empirical 3×3 vs. 5×5 ratio). Report both; infer Winograd use only when both agree.

**Model records** include efficiency metrics sampled during inference:
- Power consumption: `power_draw_avg_w`, `power_draw_std_w` (sampled via nvidia-smi).
- GPU utilization: `gpu_utilization_pct` (% device active during inference; differentiates compute-bound from memory-bound layers).
- Memory: `gpu_memory_peak_mb` (peak allocated during forward pass; FFT and Winograd use scratch buffers).
- Compute efficiency: `compute_efficiency_gflops_s` (theoretical FLOPs / latency; indicates hardware utilization vs. peak).

`kind: model` rows carry latency only — accuracy is **not** re-measured here (these are untrained/random-init forward passes; see top of file). Kernel-size-vs-accuracy was already answered in Phases 2–4 and lives in `results/model_details.csv` (`fp32_top1`, `int8_top1`) and `results/final_architecture_phase4/final_comparison.csv`. The analysis notebook joins Phase 6's `model` rows to those existing accuracy columns on `model_name` — no retraining, just a merge — so the final plot is accuracy vs. latency-per-accuracy-point and **energy-per-accuracy-point** (latency × power) per model, not latency alone.

**`configs/slurm/tupi_4090.yaml`** — `partition: tupi`, `gres: gpu:1`, to pin PCAD submissions to an RTX 4090 node instead of whatever `single_gpu.yaml` lands on.

**`scripts/slurm/profile.sbatch`** — copy of `train.sbatch` calling `python -m scripts.profile_hardware "$@"`.

**`scripts/cluster.py`** — add a `profile-submit` subcommand reusing `_build_sbatch_command`, parameterized with the new sbatch script path instead of the hardcoded `train.sbatch`.

**`notebooks/analysis/hardware_profiling_phase6.ipynb`** — loads the RTX 4060 (local) and RTX 4090 (PCAD tupi) JSON outputs side by side; joins `kind: model` rows to `results/model_details.csv` accuracy columns on model name; builds the TODO.md outputs: latency heatmap (kernel size × layer depth), speedup ratio (5×5 / 3×3 time) per GPU, Winograd feasibility threshold, CPU INT8 latency ranking, cross-GPU comparison, **plus an accuracy-vs-latency scatter (the actual efficiency/quality trade-off this whole project is measuring)**.

**Reused as-is:** `configs/runtime/local.yaml`, `configs/runtime/pcad.yaml`, `ml/runtime.py` (`set_global_seed`, `expand_path`), `ml/reporting.py` (`disk_mb`, `compute_flops`), `MODEL_REGISTRY`, `ml/quantization.py` (`build_qat_from_model`, `convert_to_int8` — specifically **not** `build_qat`, see checkpoint note above).

**Efficiency Metrics Measurement (Full-Model Inference)**

GPU power draw, utilization, memory, and compute efficiency are sampled during each model profiling run via background `nvidia-smi` polling and `torch.cuda` introspection:

```python
def profile_model_with_efficiency_metrics(model, input_size, device, warmup=50, iters=200):
    """
    Profile latency + power + utilization + memory + compute efficiency.
    
    Returns: {
        "latency_ms": float,
        "throughput_img_s": float,
        "power_draw_avg_w": float or None,
        "power_draw_std_w": float or None,
        "gpu_utilization_pct": float or None,
        "gpu_memory_peak_mb": float or None,
        "compute_efficiency_gflops_s": float or None,
    }
    """
    import subprocess
    import threading
    import torch
    from ml.reporting import compute_flops
    
    model = model.to(device).eval()
    torch.set_grad_enabled(False)
    
    # Compute FLOPs for efficiency calculation (one-time, no per-iteration cost)
    flops_dict = compute_flops(model, input_size=(1, 3, input_size[-2], input_size[-1]))
    total_flops = flops_dict.get('flops', 0)
    
    # Background power and utilization sampling
    power_samples = []
    gpu_util_samples = []
    stop_sampling = threading.Event()
    
    def sample_gpu_metrics():
        try:
            proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu", 
                 "--loop-ms=100", "-l", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            for line in proc.stdout:
                if stop_sampling.is_set():
                    proc.terminate()
                    break
                try:
                    parts = line.split()
                    power_w = float(parts[0])
                    gpu_util_pct = float(parts[1])
                    power_samples.append(power_w)
                    gpu_util_samples.append(gpu_util_pct)
                except (ValueError, IndexError):
                    pass
        except FileNotFoundError:
            pass  # nvidia-smi not available; continue without metrics.
    
    sampler = threading.Thread(target=sample_gpu_metrics, daemon=True)
    sampler.start()
    
    # Memory measurement
    torch.cuda.reset_peak_memory_stats(device)
    
    # Warmup + timed forward passes
    input_tensor = torch.randn(input_size, device=device)
    
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize(device)
    start_time = time.time()
    
    for _ in range(iters):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize(device)
    elapsed_ms = (time.time() - start_time) * 1000 / iters
    
    stop_sampling.set()
    sampler.join(timeout=2.0)
    
    # Aggregate metrics
    latency_ms = elapsed_ms
    throughput_img_s = 1000.0 / latency_ms  # Images per second (batch=1)
    power_avg = np.mean(power_samples) if power_samples else None
    power_std = np.std(power_samples) if power_samples else None
    gpu_util_avg = np.mean(gpu_util_samples) if gpu_util_samples else None
    memory_peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    
    # Compute efficiency: actual GFLOPs / second = (total_flops / latency_ms) / 1e9
    compute_efficiency = (total_flops / latency_ms) / 1e9 if latency_ms > 0 else None
    
    return {
        "latency_ms": latency_ms,
        "throughput_img_s": throughput_img_s,
        "power_draw_avg_w": power_avg,
        "power_draw_std_w": power_std,
        "gpu_utilization_pct": gpu_util_avg,
        "gpu_memory_peak_mb": memory_peak_mb,
        "compute_efficiency_gflops_s": compute_efficiency,
    }
```

**Metric Interpretation:**

| Metric | Interpretation |
|--------|-----------------|
| `power_draw_avg_w` | Average GPU power during inference. High power + low efficiency suggests memory-bound layer (not fully utilizing compute). |
| `gpu_utilization_pct` | % time GPU is active. <50% suggests the kernel is latency-bound or stalled on memory. >90% suggests good hardware utilization. |
| `gpu_memory_peak_mb` | Peak GPU memory allocated. FFT and Winograd use scratch buffers; may be higher than FP32 even if final output is smaller (INT8). |
| `compute_efficiency_gflops_s` | Actual GFLOPs/sec achieved. Compare to GPU's theoretical peak (e.g., RTX 4090 = 1456 FP32 GFLOPs/sec). Ratio of achieved/peak is utilization efficiency. |

**Availability:**
- `power_draw_avg_w`, `gpu_utilization_pct` depend on nvidia-smi access. Gracefully skipped if unavailable (`null` in JSON).
- `gpu_memory_peak_mb` and `compute_efficiency_gflops_s` always available (no external dependencies).
- Only collected for `kind: model` runs. Layer profiling skips these (per-layer metrics too noisy; full-model metrics reliable).

---

## INT8 Calibration Strategy

Phase 6 profiles latency on untrained (random-init) models; INT8 conversion uses **uncalibrated observers**. This section documents the choice and trade-offs.

### Current Approach: Uncalibrated Observers
**Definition:** Observers (quantization parameter calculators) use default quantization ranges (e.g., [-127, 127] per layer) without calibration on real data.

**Rationale:**
- **Weight values do NOT affect latency.** Latency depends on layer shape (in_ch, out_ch, kernel_size, stride), dtype (FP32 vs. INT8), and hardware (GPU model, cuDNN version), not on trained weight values.
- **Structure determines algorithm choice.** cuDNN's algorithm selection (GEMM, Winograd, FFT) depends on layer geometry, not on quantization accuracy. Uncalibrated quantization ranges do not prevent algorithm enumeration.
- **Speed and simplicity.** Avoids checkpoint dependency: models are built fresh from `MODEL_REGISTRY`, no disk I/O for best-checkpoint loading. Profiling can run on any hardware without syncing trained weights.

### Known Limitation: ±5–10% Latency Bias

Uncalibrated quantization ranges may differ from calibration on real data. This can affect:

1. **Layer fusion patterns.** Some Conv-BatchNorm fusions are only applied if quantization parameters are within expected ranges. Arbitrary ranges may prevent fusions, adding extra dequantize ops, increasing latency.
2. **Operator selection.** Some GEMM kernels are selected based on layer type and quantization scheme (per-channel vs. symmetric vs. asymmetric). Uncalibrated layers may not use the best kernel.
3. **Memory layout.** INT8 kernels may choose different memory layouts (channels-last vs. channels-first) based on quantization. Arbitrary ranges could affect data layout, impacting cache behavior.

**Magnitude:** Typically 5–10% latency variance, but can be higher (up to 20%) for layers with extreme value distributions (uncalibrated range may be much wider than real data range).

**Impact on Results:**
- Relative ranking of models may change by 1–2 positions if latency bias is asymmetric across architectures.
- Example: If depthwise-conv layers suffer larger fusion penalties than dense convs under uncalibrated quantization, alexnet_depthwisesep's latency advantage could be underestimated.

### Mitigation Strategy

**Phase 6 Core (Execution):** Use uncalibrated observers as-is.
- **Rationale:** Phase 6's goal is to measure hardware-architecture interactions (Winograd, FFT, layer efficiency). Absolute INT8 latency is secondary; relative rankings matter.
- **Mitigation:** Flag the ±5–10% bias in the final report's Limitations section.

**Phase 6 Verification (If Results Are Inconclusive):**
If accuracy-vs-latency trade-off is ambiguous (two models within 10% latency), re-run INT8 profiling with calibrated observers:
```python
def build_calibrated_qat_model(model_name, train_loader, device):
    """Load FP32 checkpoint, convert to QAT, calibrate on training set."""
    checkpoint_path = SAVE_DIR / f"{model_name}_best.pth"
    model = MODEL_REGISTRY[model_name]()
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Existing quantization flow (from ml/quantization.py):
    model_qat = build_qat_from_model(model, model_name, device)
    
    # Calibration on training set:
    for images, _ in train_loader:
        images = images.to(device)
        model_qat(images)
    
    model_int8 = convert_to_int8(model_qat)
    return model_int8
```

Then re-profile latency on calibrated models and compare to uncalibrated results.

### Decision Record

| Aspect | Uncalibrated | Calibrated | Chosen |
|--------|--------------|-----------|--------|
| Speed | Fast (no calibration loop) | Slow (iterate over training set) | **Uncalibrated** |
| Checkpoint dependency | None | Requires {model}_best.pth | **Uncalibrated** |
| Latency accuracy | ±5–10% bias (unknown direction) | Ground truth | Uncalibrated (acceptable for ranking) |
| Effort | Minimal | 30–60 min re-profiling if needed | **Uncalibrated** |
| Risk | Ranking instability if bias is asymmetric | Low | **Uncalibrated** (acceptable with limitations flag) |

**Chosen: Uncalibrated.** Accept ±5–10% bias. Revisit only if Phase 6 results are inconclusive or contradictory to Phase 4 accuracy rankings.

---

## BLOCKING ISSUES & REQUIRED FIXES

The following must be addressed before executing profiling runs. See `ideas/PHASE6_HYPOTHESES.md` and implementation checklist (TBD) for details.

### 1. Explicit Research Hypotheses (BLOCKING)
**Issue:** Plan lists experiments but no testable hypotheses or expected outcomes.
**Fix:** Create `ideas/PHASE6_HYPOTHESES.md` with three hypotheses:
- H1: Dense 3×3 models (bottleneck, fire, vgg_style) trigger Winograd on RTX 4090; expect 1.5–2.5× speedup vs. direct GEMM.
- H2: Depthwise convs (alexnet_depthwisesep, mobilenetv2) do NOT trigger Winograd; expect <10% Winograd kernel use vs. >60% for dense 3×3.
- H3: Pareto frontier models beat alexnet_tv baseline on accuracy/latency efficiency.

### 2. Environment Metadata Capture (BLOCKING)
**Issue:** cuDNN/PyTorch/CUDA versions not recorded; runs become unreproducible after software updates.
**Fix:** First JSON record in output appends metadata (see Output JSON shape above):
- PyTorch version, CUDA version, cuDNN version, GPU device name + capability.
- Seed, timestamp, config file path.
- Append as `{"kind": "metadata", ...}` before all layer/model configs.

### 3. Winograd Detection Robustness (BLOCKING)
**Issue:** `profile_kernel_trace()` is best-effort; False = "not confirmed," not "not used." This is the core claim of Phase 6.
**Fix:** Dual-signal Winograd detection:
- Signal A: Kernel-name trace heuristic (existing `profile_kernel_trace()`).
- Signal B: Empirical speedup via 3×3 vs. 5×5 latency ratio. If ratio > 1.8, Winograd likely accelerates 3×3.
- **Infer Winograd use only when both signals agree.** Report three columns: `winograd_trace_detected`, `winograd_speedup_info`, `winograd_inferred`.

### 4. Resumability & Checkpointing (BLOCKING)
**Issue:** If 4090 profiling run crashes at 80% (65/80 configs), no way to resume without recomputing from 0.
**Fix:** In `scripts/profile_hardware.py`:
- Load previously completed configs from output JSON.
- Skip already-completed configs.
- Atomic writes (flush + fsync) to prevent corruption on crash.
- Add `--resume` flag to sbatch: `python -m scripts.profile_hardware --resume`.

### 5. Model State Specification (BLOCKING)
**Issue:** Plan doesn't explicitly state whether models are `.eval()` or `.train()`. Batch norm / dropout mode affects latency.
**Fix:** All models set to `.eval()` before profiling. Explicitly add to every profiling function:
```python
model = model.to(device)
model.eval()
torch.set_grad_enabled(False)
```

---

## MEDIUM-PRIORITY IMPROVEMENTS (High ROI)

Execute before analysis phase; modest effort, high signal.

### 6. Batch-Size & Resolution Sweeps
**Motivation:** Single batch-size (1) and resolution (64×64) may underestimate Winograd gains. Latency scales differently for memory-bound (depthwise) vs. compute-bound (dense 3×3) layers.

**Change:** Modify `configs/profiling.yaml`:
```yaml
batch_sizes: [1, 8]                    # Single-image + realistic batch
input_resolutions: [64, 128]           # Tiny ImageNet (64) + intermediate
# Expands layer sweep from 48 configs to 192 (6 kernels × 4 channels × 2 batch × 2 resolution × 2 precision).
```

Output JSON includes `batch_size` and `input_resolution` per layer config.

### 7. Per-Layer Latency Breakdown (Full Models)
**Motivation:** Identify which layers are bottlenecks. MobileNetV2 is fast overall, but is it bottlenecked on depthwise or pointwise?

**Change:** For each `kind: model` run, register forward hooks on Conv2d layers and profile each separately:
```python
def profile_model_per_layer(model, input_size, device, ...):
    """Use torch.profiler with record_shapes=True to extract per-layer timings."""
    # Output per-layer CSV: {model_name, layer_name, layer_type, latency_ms, in_shape, out_shape, precision}.
```

Append to existing `{model_name}_layer_breakdown.csv` for analysis.

### 8. INT8 Calibration Strategy Documentation
**Motivation:** Plan says "uncalibrated observers are fine," but this introduces ~5–10% latency bias due to arbitrary quantization ranges.

**Change:** Document the choice explicitly in `ml/profiling.py` docstring or new `ideas/PHASE6_CALIBRATION.md`:
- **Current:** Uncalibrated observers, random init.
  - ✅ Pros: Fast, no checkpoint dependency.
  - ❌ Cons: Quantization ranges are arbitrary; layer schedules may not match deployed models.
  - **Risk:** ±5–10% latency bias.
- **Alternative (if needed):** Calibrate on training set using existing checkpoints. Revisit if Phase 6 results are inconclusive (models A/B within margin of error).

**Decision:** Accept uncalibrated bias for Phase 6 core. Flag as limitation in final report. If accuracy-vs-latency scatter shows surprising results, re-run with calibrated models.

---

## REPRODUCIBILITY & VERIFICATION CHECKLIST

Before submitting profiling runs:

- [ ] `ideas/PHASE6_HYPOTHESES.md` created with H1, H2, H3 and acceptance criteria.
- [ ] `scripts/profile_hardware.py` captures metadata (PyTorch, cuDNN, CUDA, seed, timestamp, config path).
- [ ] All models explicitly set to `.eval()` in profiling functions.
- [ ] Winograd detection includes dual signals (trace + empirical speedup); output includes `winograd_inferred` flag.
- [ ] Output JSON includes first `kind: metadata` record.
- [ ] Resumability implemented: `--resume` flag, checkpoint skipping, atomic writes.
- [ ] `configs/profiling.yaml` updated with batch sizes [1, 8] and resolutions [64, 128].
- [ ] Per-layer latency breakdown collected for full-model runs (via torch.profiler hooks).
- [ ] Efficiency metrics collected for model runs: power (`power_draw_avg_w`, `power_draw_std_w`), GPU utilization (`gpu_utilization_pct`), peak memory (`gpu_memory_peak_mb`), compute efficiency (`compute_efficiency_gflops_s`).
- [ ] nvidia-smi background sampling thread implemented (gracefully skips if unavailable).
- [ ] INT8 calibration strategy documented (uncalibrated with known ±5–10% bias, or calibrated on training set).
- [ ] `configs/profiling.yaml` committed to git so config version is tied to output metadata.
- [ ] Test run on local RTX 4060 with first 2–3 configs to verify no crashes before PCAD submission.

---

## SCOPE & EFFORT

**Blocking fixes:** ~75 minutes (metadata, Winograd detection, resumability, model state).
**Medium-priority improvements:** ~70 minutes (batch/resolution sweep, per-layer breakdown, power + utilization + memory + efficiency sampling, documentation).
**Total pre-execution effort:** ~2.5–3 hours.

**Payoff:** Ensures reproducibility, verifiability, and resumability — the difference between a credible hardware benchmark and one that's hard to defend in a paper. Efficiency metrics (utilization, compute GFLOPs/s) distinguish compute-bound (Winograd-accelerated) from memory-bound layers, directly answering the core research question.

**Manual/out of scope for tooling:**
- Nsight Compute (`ncu --set full python -m scripts.profile_hardware ...`) — external CLI, run directly on the node, not wrapped in Python; use for deep kernel analysis post-Phase6 if needed.
- GPU power draw variance (transient spikes) — sampled via nvidia-smi loop; only reliable for inference over many iterations (not layer-level).
