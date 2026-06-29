# Experimental Phases

Each experiment follows the same pipeline:

1. Train the model in **FP32**.
2. Fine-tune using **Quantization-Aware Training (QAT)**.
3. Convert to **INT8**.
4. Evaluate both FP32 and INT8 models.
5. Compare accuracy, efficiency, model size, and quantization impact.

---

# High Priority

## Notebook Output

* ✅ Reduce notebook output volume.

  * ✅ Remove or greatly reduce per-batch `print()` statements, but keep it trackable so I know it is running.
  * ✅ Use `tqdm` progress bars instead of creating new output lines.

* ✅ Separate logs from notebook output.

  * ✅ Save training logs to a `.log` file — `Trainer.__init__(log_file=...)` writes timestamped logs.
  * ✅ Continue logging metrics to Weights & Biases — now includes LR, epoch time, peak GPU memory, `grad_norm` (when `grad_clip_norm` is set).
  * ✅ Save the final run summary to a JSON or CSV file — per-model summary JSON via `make_run_summary()`.

* ✅ Keep notebooks lightweight.

  * ✅ Avoid storing thousands of lines of training logs in the notebook.

* ✅ Improve experiment reporting.

  * ✅ Save a concise `run_summary` (Top-1, Top-5, loss, training time, etc.) as a standalone file.
  * ✅ Aggregate all experiment summaries into a single CSV for easy comparison — comparison CSV still exists; individual summaries are JSON for crash-safety.

* ✅ Verify notebook execution.

  * ✅ Ensure execution errors are written to the log file and are easy to locate — `log_file` handler captures all logging.
  * ✅ Ensure the final run summary is always printed and saved, even if notebook output is collapsed — logged to file + console + W&B.

* ✅ Weights & Biases (W&B)

  * ✅ Always initialize W&B runs with clear, descriptive names and complete metadata (project, group, tags, config with num_classes/img_size/dataset, model, params).
  * ✅ Use clear, descriptive, and consistent names for all W&B artifacts so they are easy to identify and distinguish.
---

## Metrics to Add

### Efficiency

* ✅ Report FP32 model size (MB) — `disk_mb(SAVE_DIR / f"{name}_best.pth")`.
* ✅ Report INT8 model size (MB) — `disk_mb(SAVE_DIR / f"{name}.pth")`.
* ✅ Measure inference latency (ms/image) — `Trainer.benchmark()` returns `latency_ms_per_image`.
* ✅ Measure throughput (images/s) — `Trainer.benchmark()` returns `throughput_img_per_s`.
* ✅ Measure memory usage (GPU) — peak GPU memory per epoch tracked in `history["peak_gpu_mem_mb"]`, logged to W&B.
* ✅ Report MACs/FLOPs — `compute_flops(model)` via fvcore, includes in `make_run_summary()`.
* ✅ Report parameter count — via `torchinfo.summary()`, included in per-model summaries.

### Architecture Comparison

* [ ] Estimate receptive field — requires manual calculation per architecture, not yet automated.
* ✅ Compute parameter efficiency (Top-1 accuracy per million parameters) — `param_efficiency_top1_per_m` in `make_run_summary()`.
* ✅ Compute accuracy drop from FP32 → INT8 (Top-1 and Top-5) — `quantization_drop_top1`, `int8_top1_top5_gap` in summary.
* ✅ Report Top-1 / Top-5 gap — `fp32_top1_top5_gap`, `int8_top1_top5_gap` in summary.

---

## Run Summary

✅ Per-model summaries implemented via `make_run_summary()`. Includes all fields below, saved to `{RESULTS_DIR}/{name}_summary.json`:

* ✅ Model name
* ✅ Training mode (FP32 / QAT / INT8 evaluation)
* ✅ Number of epochs
* ✅ Best validation Top-1
* ✅ Best validation Top-5
* ✅ Final validation Top-1
* ✅ Final validation Top-5
* ✅ Best validation loss
* ✅ Final training loss
* ✅ FP32 model size (MB)
* ✅ INT8 model size (MB)
* ✅ Number of parameters (M)
* ✅ Compression ratio (FP32 size / INT8 size)
* ✅ MACs/FLOPs (via fvcore)
* ✅ Inference latency (ms/image, FP32 GPU + INT8 CPU)
* ✅ Throughput (images/s, FP32 GPU + INT8 CPU)
* ✅ Epoch time (average per epoch, seconds)
* ✅ Total training time (seconds)
* ✅ Peak GPU memory (MB)
* ✅ FP32 vs INT8 Top-1 accuracy drop
* ✅ Top-1 / Top-5 gap (both FP32 and INT8)
* ✅ Parameter efficiency (Top-1 per M params)

---

## Analysis

* Generate a single CSV summarizing every experiment.
* Compare all architectures (AlexNet, AlexNet 3×3, Small Kernel, VGG, etc.).
* Compare FP32 vs INT8 side by side.
* Rank models by:

  * Top-1 accuracy
  * Top-5 accuracy
  * Model size
  * Inference latency
  * Parameter efficiency
  * Accuracy loss after quantization
  * Overall efficiency (accuracy vs compute).

---

## Phase 1 — Reference Architectures

Establish baseline performance using representative CNN families.

Models:

* [ ] AlexNet (fine-tuned)
* [ ] VGG-style CNN
* [ ] ResNet-18
* [ ] MobileNetV2

For each model:

* [ ] FP32 training
* [ ] QAT fine-tuning (when supported)
* [ ] INT8 conversion
* [ ] FP32 vs INT8 evaluation
* [ ] Record latency, model size, and quantization accuracy drop

---

## Phase 2 — Kernel Restriction Study

Investigate the impact of restricting convolution kernels while keeping the architecture as controlled as possible.

Models:

* [ ] AlexNet (3×3)
* [ ] AlexNet (2×2)
* [ ] AlexNet (stacked 3×3)
* [ ] AlexNet (mixed kernels)
* [ ] Kernel-restricted AlexNet

For each variant:

* [ ] FP32 training
* [ ] QAT fine-tuning
* [ ] INT8 conversion
* [ ] FP32 vs INT8 comparison
* [ ] Compare against the original AlexNet

---

## Phase 3 — Compensation Mechanisms

Evaluate architectural modifications that compensate for the reduced receptive field caused by small kernels.

Candidate variants:

* [ ] Bottleneck
* [ ] Factorized convolutions
* [ ] Group convolutions
* [ ] Depthwise separable convolutions
* [ ] Residual connections
* [ ] Fire modules
* [ ] Global Average Pooling
* [ ] Squeeze-and-Excitation (SE)

For each variant:

* [ ] FP32 training
* [ ] QAT fine-tuning
* [ ] INT8 conversion
* [ ] FP32 vs INT8 comparison
* [ ] Compare with the corresponding kernel-restricted baseline

---

## Phase 4 — Final Architecture

Combine the most successful architectural ideas from previous phases into a final AlexNet-inspired network.

Tasks:

* [ ] Design the final architecture
* [ ] FP32 training
* [ ] QAT fine-tuning
* [ ] INT8 conversion
* [ ] Compare with all previous models
* [ ] Perform complete ablation study

---

## Phase 5 — Final Analysis

Summarize the complete experimental study.

Produce:

* [ ] FP32 ranking
* [ ] INT8 ranking
* [ ] Quantization accuracy-drop analysis
* [ ] Model size comparison
* [ ] Latency comparison
* [ ] Winograd compatibility discussion
* [ ] Final conclusions and recommendations

---

## Phase 6 — Hardware Profiling & Winograd Efficiency Validation

Measure actual latency, memory bandwidth, and power consumption on RTX 4060 to empirically validate Winograd acceleration claims. Compare theoretical vs real-world efficiency gains across kernel sizes.

Benchmarks:

* [ ] Profile single-layer latency for Conv(k=2×2), Conv(k=3×3), Conv(k=5×5) on RTX 4060
* [ ] Measure memory bandwidth utilization (NVIDIA Nsight Compute or similar)
* [ ] Profile full forward pass for Phase 1–5 reference architectures (AlexNet, MobileNetV2, TinyHybridNet, etc.)
* [ ] Profile INT8 vs FP32 inference latency on CPU (fbgemm backend)
* [ ] Identify which layers contribute most to latency (are Winograd gains concentrated or distributed?)
* [ ] Measure Winograd reordering overhead (input/output packing costs)
* [ ] Profile power consumption under sustained inference (if HW supports NVIDIA RAPL or equivalent)

Outputs:

* [ ] Latency heatmap: kernel size vs layer depth
* [ ] Speedup ratio: (Conv 5×5 time) / (Conv 3×3 time)
* [ ] Winograd feasibility threshold: break-even point for reordering overhead
* [ ] CPU INT8 latency ranking (identify bottleneck layers)
* [ ] Revised efficiency recommendations based on empirical data

---

## Phase 7 — Efficient Vision Transformers & Hybrid Attention Architectures

Explore whether attention-based models can match or exceed CNN efficiency within Winograd constraints. Investigate local-attention Vision Transformers as an alternative paradigm to small-kernel CNNs.

Models:

* [ ] Vision Transformer (ViT-Tiny) with local attention windows (e.g., 3×3 or 5×5 patches, local self-attention)
* [ ] DeiT-Tiny (Knowledge-distilled ViT)
* [ ] Hybrid CNN-Transformer (small-kernel CNN stem + local transformer blocks)
* [ ] Lightweight attention variant (linear attention or depthwise attention)

For each variant:

* [ ] FP32 training
* [ ] QAT fine-tuning
* [ ] INT8 conversion
* [ ] FP32 vs INT8 evaluation
* [ ] Compare accuracy, latency, model size, and quantization robustness vs Phase 5–6 CNNs
* [ ] Assess Winograd deployment feasibility (attention ops, memory layout)
* [ ] Determine whether attention can substitute for large receptive fields in resource-constrained inference

---

## Phase 8 — Extended Architecture Search (Future)

If Phase 7 yields promising hybrid results, consider automated architecture search (NAS or evolutionary search) to discover optimal kernel-size, depth, width, and attention-ratio combinations under Winograd constraints.
