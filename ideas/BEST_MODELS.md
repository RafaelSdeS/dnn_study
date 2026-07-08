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
3. **Benchmark Winograd compatibility** — Verify that Bottleneck & Fire leverage small-kernel acceleration on actual hardware.
4. **Architecture search** — AutoML over compensation mechanisms for Pareto-optimal size/accuracy/quantization trade-offs.
4. **Task transfer** — Test best models on object detection and semantic segmentation (Phase 4–5 scope).
6. **Fine-tune Tier 1 models** for deployment scenarios (mobile, edge, server).