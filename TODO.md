# Experimental Phases

Each experiment follows the same pipeline:

1. Train the model in **FP32**.
2. Fine-tune using **Quantization-Aware Training (QAT)**.
3. Convert to **INT8**.
4. Evaluate both FP32 and INT8 models.
5. Compare accuracy, efficiency, model size, and quantization impact.

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
