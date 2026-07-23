# Summary

Results after implementing phases 1–4, 6, and 9. **Most baselines (MobileNetV2, ResNet18, VGGStyle) show superior accuracy to pure AlexNet models, but Phase 2–4 AlexNet variants achieve competitive accuracy at 100–1000× smaller model sizes.** Phase 4's final hybrid architectures push AlexNet-family accuracy past 49% for the first time — within 3pp of VGGStyle — while Phase 9 shows a single residual bypass, with zero added parameters, closes most of that gap on its own. Phase 7 (detection) is trained end-to-end but its results are not yet trustworthy — see the Phase 7 section below. Phase 5 is this document plus `results/phase_5_cross_phase_results_analysis/`; Phase 8 is planned only, no results yet.

---

## Overall Rankings — FP32 Top-1 Accuracy

| Rank | Model | Phase | Accuracy | Params (M) | Size (MB) | Efficiency (Acc/MB) |
|------|-------|-------|----------|-----------|-----------|-------------------|
| 1 | **MobileNetV2** | 1 | 57.99% | 2.48 | 28.75 | 2.01 |
| 2 | **ResNet18** | 1 | 53.91% | 11.28 | 129.21 | 0.42 |
| 3 | **VGGStyle** | 1 | 51.81% | 2.41 | 27.58 | 1.88 |
| 4 | **AlexNetFinalFireResidual** | 4 | **49.79%** | 0.70 | 8.09 | 6.15 |
| 5 | **AlexNetFireBypass** | 9 | **49.03%** | 0.52 | 5.99 | **8.18** |
| 6 | **AlexNetResidual** | 3 | **48.01%** | 60.67 | 694.41 | 0.07 |
| 7 | **AlexNetSmallKernel** | 2 | **45.84%** | 1.60 | 18.35 | 2.50 |
| 8 | **AlexNetFinalBottleneckResidual** | 4 | **45.10%** | 0.57 | 6.65 | 6.78 |
| 9 | **AlexNetBottleneck** | 3 | **44.62%** | 0.39 | 4.49 | **9.93** |
| 10 | **AlexNetStacked** | 2 | **44.56%** | 60.48 | 692.25 | 0.06 |
| 11 | **AlexNetDepthwiseSep** | 3 | **44.39%** | 0.31 | 3.65 | **12.15** |
| 12 | **AlexNetFire** | 3 | **43.98%** | 0.52 | 5.99 | 7.34 |
| 13 | **AlexNetFinalDepthwiseFire** | 4 | **43.46%** | 0.47 | 5.51 | 7.88 |
| 14 | **AlexNetFinalBottleneckFire** | 4 | **42.29%** | 0.51 | 5.88 | 7.19 |

All 9 original Phase 1–3 models are still here, just renumbered — rows 4, 5, 8, 13, and 14 are the
new Phase 4/9 additions. **The "Analysis by Dimension" section below still discusses the
Phase 1–3-only picture** (written before Phase 4/9 existed) — read it with the new rows above in
mind; Phase 4 and 9 get their own dedicated analysis further down instead.

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

## Next Steps

1. **Investigate AlexNetSmallKernel QAT** — Why the 9.89pp drop? Recalibrate batch norm or try different QAT schedules. Still open.
2. **Debug AlexNetSE** — Was initialization the issue? Try different seeds or training hyperparameters. Still open.
3. ~~**Benchmark Winograd compatibility** — Verify that Bottleneck & Fire leverage small-kernel acceleration on actual hardware.~~ **Done — see Phase 6.**
4. **Architecture search** — AutoML over compensation mechanisms for Pareto-optimal size/accuracy/quantization trade-offs. Not started (see "Phase 10" in `TODO.md`, contingent on Phase 8).
5. ~~**Task transfer** — Test best models on object detection and semantic segmentation.~~ **In progress — see Phase 7. Detection trains end-to-end but results are blocked on an unresolved anchor-recall issue; segmentation has no training run yet.**
6. **Fine-tune Tier 1 models** for deployment scenarios (mobile, edge, server). Not started.
7. **Fix Phase 7's anchor-recall blocker** and rerun detection before trusting any backbone comparison there.

---

## Phase 6 — Hardware Profiling (Winograd Efficiency)

Measured on a real RTX 4090 (PCAD `tupi5`), batch=1, 64×64 input — the first phase to check hardware
behavior directly instead of relying on FLOPs/params as a proxy. Full methodology, statistical tests, and
data-quality corrections in `notebooks/phase_6_hardware_profiling_analysis/hardware_profiling_phase6.ipynb`;
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

---

## Phase 4 — Final Architecture & Compression

Combines Phase 3's best mechanisms (Bottleneck, Fire, Residual, Depthwise-Separable) into four
hybrid architectures, then separately tests how far each can be compressed below plain INT8. Full
data: `results/phase_4_compression_and_final_architecture_training/`; notebooks:
`notebooks/phase_4_compression_and_final_architecture_training/`.

### Final hybrid architectures — FP32 vs INT8

| Model | Mechanisms combined | FP32 Top-1 | INT8 Top-1 | QAT Δ | Size (MB) | Acc/MB |
|---|---|---|---|---|---|---|
| **AlexNetFinalFireResidual** | Fire + residual shortcuts | 49.79% | 49.20% | –0.59pp | 8.09 | 6.15 |
| AlexNetFinalBottleneckResidual | Bottleneck + residual shortcuts | 45.10% | 45.98% | +0.88pp ✓ | 6.65 | 6.78 |
| AlexNetFinalDepthwiseFire | Depthwise-separable stem + Fire body | 43.46% | 42.79% | –0.67pp | 5.51 | 7.88 |
| AlexNetFinalBottleneckFire | Bottleneck stem + Fire body | 42.29% | 44.00% | +1.71pp ✓✓ | 5.88 | 7.19 |

### Ablation — does combining mechanisms help, or does one alone explain the gain?

Each hybrid compared against its two single-mechanism Phase 3 parents (no retraining, direct lookup):

| Hybrid | Hybrid FP32 | Best single-mechanism parent | Parent FP32 | Δ vs best parent |
|---|---|---|---|---|
| **AlexNetFinalFireResidual** | 49.79% | AlexNetResidual | 48.01% | **+1.78pp — only combo that beats both parents** |
| AlexNetFinalBottleneckFire | 42.29% | AlexNetBottleneck | 44.62% | –2.34pp — combining hurts |
| AlexNetFinalBottleneckResidual | 45.10% | AlexNetResidual | 48.01% | –2.91pp — combining hurts |
| AlexNetFinalDepthwiseFire | 43.46% | AlexNetDepthwiseSep | 44.39% | –0.93pp — combining hurts |

**Only Fire+Residual beats its best single-mechanism parent; the other three combinations
underperform their own ancestor.** Combining mechanisms isn't automatically additive, and can
actively hurt accuracy (Bottleneck+Residual: –2.91pp; Bottleneck+Fire: –2.34pp) — see Phase 9 below
for how much of Fire+Residual's win is attributable to the bypass alone vs. its added stem.

### Compression (Phase 4.1) — how far below plain INT8 can these go?

5 models (`mobilenetv2`, `alexnet_bottleneck`, `alexnet_fire`, `alexnet_depthwisesep`,
`alexnet_final_fire_residual`) pushed through INT8/INT4/INT2/ternary/binary, both PTQ and QAT, plus
per-layer mixed precision. Mean across all 5 models per method:

| Method | Mean Top-1 | Mean Drop (pp) | Mean Ratio | Mean Acc/MB |
|---|---|---|---|---|
| INT8 (anchor) | 47.40% | 0.76pp | 3.97× | 90.8 |
| **Mixed INT4/8 (per-layer)** | 44.52% | **–0.19pp (gain)** | 7.05× | 215.3 |
| INT4 QAT | 43.31% | 1.02pp | 7.86× | 230.2 |
| INT4 PTQ | 25.31% | 22.85pp | 7.86× | 106.2 |
| Ternary QAT | 26.06% | 22.10pp | 15.42× | 205.4 |
| INT2 QAT | 17.40% | 30.75pp | 15.42× | 145.4 |
| Binary QAT | 10.73% | 37.43pp | 29.70× | 164.8 |

**Verdict:** Mixed-precision INT4/8 is the actual Pareto winner among the aggressive options — it
beats plain INT8 on *both* accuracy (slight gain) and size (7.05× vs. 3.97×) by keeping the
sensitive ~30% of layers at INT8 and pushing the rest to INT4. Pure INT4 QAT is close behind and
simpler (uniform bit-width, no sensitivity analysis needed). Below INT4, retraining recovers
meaningfully more than PTQ at the same bit-width (ternary QAT 26.1% vs. INT4 PTQ 25.3%, despite
ternary being *more* aggressive) — but ternary/binary still cost 22–37pp and aren't viable without
further work. Full per-model breakdown: `.../pareto_frontier.csv`.

---

## Phase 7 — Detection & Segmentation: results NOT yet reliable, blocked on anchor recall

**Status: do not cite these numbers as a backbone comparison.** Detection trains end-to-end
(FP32 → QAT → INT8) for 3 backbones on PASCAL VOC via `scripts/train_det_seg.py`, producing real
loss curves and checkpoints — but validation mAP is 0.4–7.1% across every configuration tried
(`phase7_detection`, the `_minratio02` anchor-config retry, `_diag_256`, `_diag_512`):

| Model | Config | Best val mAP | Best epoch |
|---|---|---|---|
| alexnet_tv | phase7_detection | 7.14% | 27 |
| alexnet_bottleneck | phase7_detection | 1.17% | 24 |
| alexnet_bottleneck | phase7_detection_minratio02 | 0.96% | 28 |
| alexnet_fire | phase7_detection | 0.50% | 25 |

A working SSD on VOC typically scores 40–70%+ mAP — these numbers indicate a training/configuration
problem, not a real backbone comparison. `ideas/PHASE7_PLAN.md` (line 869) flags the anchor-recall
sanity check ("for every ground-truth box, does *some* default anchor achieve IoU > 0.5?") as
**"Blocking #1"** and **the single most likely cause** of exactly this failure mode — and that
checkbox is still unchecked. The `_minratio02` run was one anchor-config fix attempt; it didn't
resolve the issue (0.96% vs. 1.17% baseline). Segmentation has data-loading + trainer scaffolding
built (`docs/PHASE7_LOG.md` Stage 6) but no training run at all yet.

**Before trusting any Phase 7 ranking:** run `scripts/check_anchor_recall.py` to completion and
confirm >95% recall for all 3 backbones — `ideas/PHASE7_PLAN.md`'s own acceptance criterion — then
retrain.

---

## Phase 9 — Bypass Ablation: does Phase 4's whole architecture change matter, or just the bypass?

Phase 4's `AlexNetFinalFireResidual` (49.79% FP32) beats Phase 3's `AlexNetFire` (43.98% FP32) by
+5.81pp — but it changes two things at once versus Fire: it adds a 3×3 stride-2 stem *and* wraps
every Fire stage in a residual shortcut. `AlexNetFireBypass` isolates the shortcut alone (Fire's
exact stem, one identity bypass added, zero extra parameters) to find out which change is doing the
work. Full plan: `ideas/PHASE9_PLAN.md`.

| Run | Epochs | FP32 Top-1 | INT8 Top-1 | Quant. Δ |
|---|---|---|---|---|
| Initial (PCAD job 806654) | 66 | 47.05% | 47.16% | +0.11pp (gain) |
| **Large-scale (final)** | **152** | **49.03%** | **49.86%** | **+0.84pp (gain)** |

**The bypass-alone fraction of Phase 4's total gain grew substantially with more training.** The
original ~55% estimate (git commit `924d553`) came from the 66-epoch initial run; the 152-epoch
large-scale run shows bypass alone closing **87% of the FP32 gap** to `AlexNetFinalFireResidual`
(5.05pp of 5.82pp) and **fully exceeding it on INT8** (49.86% vs. 49.20% — the bypass-only model,
despite fewer total architectural changes, edges out the full Phase 4 hybrid once quantized). Zero
added parameters (0.516M, identical to `AlexNetFire`), and it keeps Fire's quantization-gain
property (INT8 improves over FP32) that the full Phase 4 hybrid loses (see Phase 4 table above:
FireResidual drops –0.59pp under QAT).

**Practical implication:** the stem change in `AlexNetFinalFireResidual` contributes comparatively
little once training is run to completion — a plain Fire backbone plus one residual shortcut gets
nearly all of the benefit at Fire's exact parameter count, and quantizes better doing it. Structured
pruning (`scripts/prune_channels.py`) and weight-compression headroom measurement
(`scripts/measure_compression.py`) tooling also exists for this phase but has no saved results yet
(Tasks 2–3, `ideas/PHASE9_PLAN.md`).