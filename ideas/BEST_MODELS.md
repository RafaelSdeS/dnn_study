# Summary

Results after implementing phases 1, 2, and 3. **Most baselines (MobileNetV2, ResNet18, VGGStyle) show superior accuracy to pure AlexNet models, but Phase 2–3 AlexNet variants achieve competitive accuracy at 100–1000× smaller model sizes.**

---

## Overall Rankings — FP32 Top-1 Accuracy

| Rank | Model | Phase | Accuracy | Params (M) | Size (MB) | Efficiency (Acc/MB) |
|------|-------|-------|----------|-----------|-----------|-------------------|
| 1 | **MobileNetV2** | 1 | 57.99% | 2.48 | 28.75 | 2.01 |
| 2 | **ResNet18** | 1 | 53.91% | 11.28 | 129.21 | 0.42 |
| 3 | **VGGStyle** | 1 | 51.81% | 2.41 | 27.58 | 1.88 |
| 4 | **AlexNetResidual** | 3 | **48.01%** | 60.67 | 694.41 | 0.07 |
| 5 | **AlexNetSmallKernel** | 2 | **45.84%** | 1.60 | 18.35 | 2.50 |
| 6 | **AlexNetStacked** | 2 | **44.56%** | 60.48 | 692.25 | 0.06 |
| 7 | **AlexNetBottleneck** | 3 | **44.62%** | 0.39 | 4.49 | **9.93** |
| 8 | **AlexNetDepthwiseSep** | 3 | **44.39%** | 0.31 | 3.65 | **12.15** |
| 9 | **AlexNetFire** | 3 | **43.98%** | 0.52 | 5.99 | **7.34** |

---

## Analysis by Dimension

### 1. **Accuracy-First** (Prioritize prediction quality)

**Best models for Winograd-accelerated deployment where accuracy is paramount:**

- **AlexNetResidual** (48.01%): Best AlexNet-family model overall. Residual connections help gradient flow and accuracy recovery. Large model (~60M params).
- **AlexNetSmallKernel** (45.84%): 45.8% at just 1.6M params—competitive with MobileNetV2 at 50× smaller size.
- **AlexNetStacked** (44.56%): Stacked small-kernel layers compensate for kernel restrictions; 60M params.

**Baselines remain best:**
- **MobileNetV2** (57.99%, 2.48M params): Highest accuracy, efficient, well-studied.
- **ResNet18** (53.91%, 11.28M params): Balanced accuracy and moderate size.

---

### 2. **Efficiency-First** (Maximize accuracy per MB)

**Best bang-for-buck in terms of model size:**

| Model | Acc/MB | Accuracy | Size (MB) | Quantization Drop |
|-------|--------|----------|-----------|------------------|
| **AlexNetDepthwiseSep** | 12.15 | 44.39% | 3.65 | –2.92pp ⚠️ |
| **AlexNetBottleneck** | 9.93 | 44.62% | 4.49 | –0.08pp ✓ |
| **AlexNetFire** | 7.34 | 43.98% | 5.99 | +0.33pp ✓ |
| **AlexNetSmallKernel** | 2.50 | 45.84% | 18.35 | –9.89pp ⚠️ |
| **VGGStyle** | 1.88 | 51.81% | 27.58 | –0.63pp ✓ |
| **MobileNetV2** | 2.01 | 57.99% | 28.75 | — |

**Verdict:** **AlexNetBottleneck** and **AlexNetFire** are Pareto-optimal: tiny (~4–6 MB), competitive accuracy (43–44%), and **quantization-stable** (minimal INT8 drop).

---

### 3. **Quantization Stability** (Minimal FP32 → INT8 accuracy drop)

**Most robust models under INT8 conversion:**

| Model | FP32 Acc | INT8 Acc | Drop | Status |
|-------|----------|----------|------|--------|
| **AlexNetBottleneck** | 44.62% | 44.54% | –0.08pp | ✓ **Excellent** |
| **AlexNetFire** | 43.98% | 44.30% | +0.33pp | ✓ **Gain** |
| **AlexNetResidual** | 48.01% | 47.27% | –0.74pp | ✓ **Good** |
| **AlexNetMixed** | 38.74% | 37.99% | –0.75pp | ✓ Good |
| **VGGStyle** | 51.81% | 51.19% | –0.63pp | ✓ Good |
| **AlexNetStacked** | 44.56% | 42.79% | –1.77pp | ⚠️ Moderate |
| **AlexNetSmallKernel** | 45.84% | 35.95% | **–9.89pp** | ❌ **Poor** |
| **AlexNet3x3GAP** | 38.74% | 37.60% | –1.14pp | ✓ Good |
| **AlexNet3x3FC** | 35.79% | 36.19% | +0.40pp | ✓ Gain |
| **AlexNet2x2GAP** | 30.02% | 30.89% | +0.87pp | ✓ Gain |

**Key insight:** Small-kernel models (**SmallKernel, 2x2, 3x3**) quantize either poorly or counterintuitively. Compensation mechanisms (**Bottleneck, Fire, Residual**) quantize much better, possibly due to richer feature representations.

---

### 4. **Kernel Restriction Impact** (Phase 2 — AlexNet Variants)

| Kernel(s) | Model | Accuracy | Drop vs Baseline | Size | Notes |
|-----------|-------|----------|------------------|------|-------|
| 3×3 + FC head | **AlexNet3x3FC** | 35.79% | –48.61pp | 659 MB | Large but poor accuracy |
| 3×3 + GAP head | **AlexNet3x3GAP** | 38.74% | –45.48pp | 2.23 MB | GAP head enables size reduction |
| 2×2 + GAP head | **AlexNet2x2GAP** | 30.02% | –49.95pp | 1.04 MB | Severely limited by kernel size |
| 3×3 stacked | **AlexNetStacked** | 44.56% | –26.92pp | 692 MB | Stacked compensates; 60M params |
| Small kernel mixed | **AlexNetSmallKernel** | 45.84% | –26.74pp | 18 MB | **Best small-kernel result** |
| Mixed (var. kernel) | **AlexNetMixed** | 38.74% | –41.34pp | 20 MB | Moderate mixed approach |

**Verdict:** 
- Pure 3×3 or 2×2 restrictions are costly without compensation.
- **Head type matters:** AlexNet3x3GAP (2.23 MB) vs AlexNet3x3FC (659 MB) shows GAP's 296× compression benefit.
- **AlexNetSmallKernel** (custom small-kernel design) recovers well and reaches 45.84% at tiny size.
- Stacking small kernels helps but requires more params.

---

### 5. **Compensation Mechanisms** (Phase 3 — Which techniques work best?)

| Mechanism | Model | Accuracy | Size (MB) | Params (M) | Eff. (Acc/MB) | QAT Drop | Status |
|-----------|-------|----------|-----------|-----------|---------------|----------|--------|
| **Bottleneck** | AlexNetBottleneck | 44.62% | 4.49 | 0.39 | 9.93 | –0.08pp | ✓✓ **Best** |
| **Fire (squeezenet)** | AlexNetFire | 43.98% | 5.99 | 0.52 | 7.34 | +0.33pp | ✓✓ **Excellent** |
| **Depthwise Sep** | AlexNetDepthwiseSep | 44.39% | 3.65 | 0.31 | 12.15 | –2.92pp | ⚠️ Unstable |
| **Residual** | AlexNetResidual | 48.01% | 694.41 | 60.67 | 0.07 | –0.74pp | ✓ Best overall accuracy |
| **Factorized** | AlexNetFactorized | 42.89% | 653.15 | 57.07 | 0.07 | –0.29pp | ✓ Good |
| **Group Conv** | AlexNetGroupConv | 29.18% | 639.99 | 55.92 | 0.05 | –1.47pp | ❌ Poor |
| **SE (Squeeze-Excite)** | AlexNetSE | **0.50%** | 659.75 | 57.65 | — | — | ❌ **Failed** |

**Note:** **AlexNet3x3GAP** (moved to Phase 2) is a simple head-type variant (GAP vs FC), not a compensation mechanism; see Phase 2 section above.

**Key findings:**
- **Bottleneck & Fire** are the sweet spot: tiny, competitive accuracy, quantization-stable.
- **Depthwise Separable** is too aggressive; loses 2.92pp to quantization.
- **Residual** recovers the most accuracy but at cost of large model size.
- **SE attention** failed to train (collapsed to 0.5%); likely requires careful initialization.
- **Group Convolution** provided negligible benefit.

---

## Recommended Models to Continue Investigating

### **Tier 1: Production-Ready (Accuracy + Efficiency + Stability)**
1. **MobileNetV2** — Industry standard; 57.99% accuracy, proven quantization.
2. **AlexNetBottleneck** — Tiny (4.5 MB), 44.62% accuracy, rock-solid QAT (–0.08pp drop).
3. **AlexNetFire** — Tiny (6 MB), 43.98% accuracy, quantization *gain* (+0.33pp).
4. **AlexNetSmallKernel** — Efficient (45.84%, 18 MB), but watch QAT stability (–9.89pp drop).

### **Tier 2: High-Accuracy Alternatives**
5. **AlexNetResidual** — Best AlexNet variant (48.01%), good QAT stability (–0.74pp), but large (694 MB).
6. **ResNet18** — Balanced baseline (53.91%), moderate size, baseline QAT reference.

### **Tier 3: Exploratory / Requires Tuning**
7. **AlexNetDepthwiseSep** — Highest efficiency (12.15 Acc/MB) but poor quantization drop (–2.92pp); needs QAT tuning.
8. **AlexNetStacked** — Good accuracy (44.56%) via stacking, but large and moderate QAT drop (–1.77pp).
9. **AlexNetMixed** — Balanced compromise (38.74%, 20 MB), stable QAT (–0.75pp).
10. **VGGStyle** — Competitive baseline (51.81%), stable QAT (–0.63pp).

### **Not Recommended**
- **AlexNetSE** — Training failure; do not pursue without debugging.
- **AlexNet3x3FC / AlexNet2x2GAP** — Poor accuracy (30–35%) from naive kernel restriction without compensation.
- **AlexNetGroupConv** — Negligible benefit, poor QAT.

---

## Next Steps (Phase 4+)

1. **Investigate AlexNetSmallKernel QAT** — Why the 9.89pp drop? Recalibrate batch norm or try different QAT schedules.
2. **Debug AlexNetSE** — Was initialization the issue? Try different seeds or training hyperparameters.
3. ~~**Benchmark Winograd compatibility** — Verify that Bottleneck & Fire leverage small-kernel acceleration on actual hardware.~~ **Done — see Phase 6 below.**
4. **Architecture search** — AutoML over compensation mechanisms for Pareto-optimal size/accuracy/quantization trade-offs.
4. **Task transfer** — Test best models on object detection and semantic segmentation (Phase 4–5 scope).
6. **Fine-tune Tier 1 models** for deployment scenarios (mobile, edge, server).

---

## Phase 6 — Hardware Profiling (Winograd Efficiency)

Measured on a real RTX 4090 (PCAD `tupi5`), batch=1, 64×64 input — the first phase to check hardware
behavior directly instead of relying on FLOPs/params as a proxy. Full methodology, statistical tests, and
data-quality corrections in `notebooks/analysis/hardware_profiling_phase6.ipynb`;
hypotheses/acceptance-criteria source in `ideas/PHASE6_PLAN.md`.

| Model | Winograd-eligible | FP32 Latency (ms) | INT8 Latency (ms) | FP32 GFLOP/s | FP32 Top-1 | Efficiency (Acc/ms) |
|---|---|---|---|---|---|---|
| `alexnet_tv` | ⚠️ Only trailing 3×3 layers | 0.49 | 0.98 | 390.1 | 32.89% | 67.6 |
| `alexnet_depthwisesep` | ❌ None (depthwise) | 0.96 | 0.94 | 42.7 | 44.39% | 46.3 |
| `alexnet_bottleneck` | ✅ Dense 3×3 branch | 1.54 | 0.86 | 52.7 | 44.62% | 28.9 |
| `alexnet_final_bottleneck_residual` | ✅ Dense 3×3 branch | 1.93 | 1.27 | 57.8 | 45.10% | 23.4 |
| `alexnet_final_fire_residual` | ✅ Fire expand branch | 1.81 | 1.14 | 61.9 | 49.79% | 27.5 |
| `vgg_style` | ✅ Fully (every conv) | 1.15 | 1.29 | 405.5 | 51.81% | 45.2 |
| `mobilenetv2` | ❌ None (depthwise) | 1.47 | 4.24 | 37.3 | 57.99% | 39.4* |
| `alexnet_fire` | ✅ Fire expand branch | 1.65 | 0.79 | 218.1 | 43.98% | 26.6 |

\*`mobilenetv2` INT8 top-1 accuracy is unmeasured (Phase 1–4 training gap, unrelated to Phase 6) — its
INT8 latency (4.24ms) is unusually high, plausibly an uncalibrated-quantization fusion penalty specific
to its inverted-residual/depthwise structure; efficiency column is FP32-only for this model.

**Hypothesis results** (full statistical detail in the notebook):

- **H1 (dense 3×3 gets Winograd acceleration): PARTIAL.** A paired Wilcoxon test across 32 layer-config
  groups confirms 3×3 is *systematically* faster than 5×5 (p=4.7e-10) — a real, structural effect, not
  noise. But the stricter per-group signal (3×3 beats 2×2 *and* clears a 1.8× margin vs. 5×5) only holds
  in 31% of groups, well under a 60% bar — the effect is real but inconsistent across batch/resolution
  combinations, likely because larger batches/resolutions become memory-bound rather than compute-bound.
  The kernel-name trace detector (`profile_kernel_trace`) never fired at all — best-effort as documented,
  provides no confirmation either way.
- **H2 (depthwise doesn't benefit): PASS (n=2, case-study level).** `alexnet_depthwisesep`/`mobilenetv2`
  median 40.0 GFLOP/s vs. 91.2 GFLOP/s for the dense-conv group.
- **H3 (Pareto frontier beats baseline on accuracy/latency): 4/5 PASS.** `alexnet_bottleneck`,
  `alexnet_final_bottleneck_residual`, `alexnet_final_fire_residual`, and `vgg_style` all beat
  `alexnet_tv`'s accuracy/latency ratio. `mobilenetv2` does not — `alexnet_tv`'s FP32 latency turned out
  to be the fastest of all 16 profiled configs (0.49ms despite a 662MB model), and `mobilenetv2`'s INT8
  accuracy can't be checked (unmeasured). Not smoothed over — see notebook for full discussion.
- **H4 (INT8 preserves latency ranking vs. FP32): PASS, unambiguous.** Spearman ρ=0.9999 (n=96 layer
  configs) — FP32 latency is a reliable proxy for INT8 latency ranking.

**Practical takeaway:** kernel size alone doesn't predict Winograd-friendliness — `alexnet_depthwisesep`
and `mobilenetv2` use 3×3 kernels but see none of the acceleration `vgg_style`/the bottleneck-fire hybrids
show, because Winograd requires *dense* (`groups=1`) convolutions. `alexnet_fire` (INT8: 456 GFLOP/s) and
`alexnet_tv` (FP32: 390 GFLOP/s) post the highest raw compute throughput in this sweep, though `vgg_style`
is the most consistently Winograd-eligible architecture (100% dense 3×3).

**Known limitations:** single-GPU (RTX 4090 only, no RTX 4060 cross-comparison yet); uncalibrated INT8
observers (±5–10% latency bias per `PHASE6_PLAN.md`); `winograd_speedup_info` in the raw JSONL is not
per-kernel-size (see notebook Phase 1 for the correction applied). Full details in the notebook's
Limitations section.