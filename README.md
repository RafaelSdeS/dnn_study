# QAT Research Framework (PyTorch)

This repository provides a modular framework for experimenting with:

- FP32 training
- Quantization-Aware Training (QAT)
- INT8 model conversion and inference
- Architecture comparison under quantization constraints

All core logic is centralized in `ml_utils.py`, ensuring reproducibility and avoiding duplicated training code across notebooks.

---

# Project Structure

## Core library

### `ml_utils.py`

This file contains the entire ML pipeline:

- Data loading (ImageNet-style datasets)
- Training and evaluation loops
- Checkpointing utilities
- Model registry system
- QAT preparation and fusion
- INT8 conversion utilities
- Experiment configuration system
- Weights & Biases (optional) logging
- Model analysis utilities

All experiments should import and use this module exclusively.

---

## Notebooks

### `alexnet_qat.ipynb`

Focused on AlexNet-family architectures:

- Pretrained AlexNet (fine-tuned)
- 3×3 kernel AlexNet variant
- Small-kernel lightweight AlexNet
- QAT versions of all above
- INT8 converted models

**Goal:**
Evaluate how architectural changes in classical CNNs affect:

- Accuracy
- QAT stability
- INT8 degradation

---

### `tinyhybrid_qat.ipynb`

Focused on modern efficient architectures:

- Fire modules (SqueezeNet-style blocks)
- MobileNet-style depthwise separable convolutions
- Hybrid CNN architectures
- Custom fusion strategies for QAT

**Goal:**
Study:

- Efficiency vs accuracy trade-offs
- Impact of QAT fusion strategies
- Performance of modern lightweight CNNs under INT8

---
